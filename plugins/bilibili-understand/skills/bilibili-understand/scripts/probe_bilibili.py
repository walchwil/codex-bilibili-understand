"""Probe one Bilibili URL with yt-dlp and cache sanitized metadata."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit, urlunsplit


ALLOWED_HOSTS = ("bilibili.com", "b23.tv")
DEFAULT_OUTPUT_ROOT = Path("outputs") / "bilibili-understand"


class InputError(ValueError):
    """Raised when the supplied URL is not a supported public Bilibili URL."""


def normalize_url(raw_url: str) -> str:
    url = raw_url.strip()
    parsed = urlsplit(url)
    host = (parsed.hostname or "").lower().rstrip(".")
    if parsed.scheme not in {"http", "https"} or not any(
        host == allowed or host.endswith(f".{allowed}") for allowed in ALLOWED_HOSTS
    ):
        raise InputError("expected an http(s) URL on bilibili.com or b23.tv")
    if parsed.username or parsed.password:
        raise InputError("credentials are not allowed in the URL")

    netloc = host
    if parsed.port:
        netloc = f"{host}:{parsed.port}"

    # Keep only the multipart selector. Shared Bilibili links often carry tracking
    # parameters such as vd_source; they are unnecessary and should not be cached.
    page = parse_qs(parsed.query).get("p", [None])[0]
    query = f"p={page}" if page and page.isdigit() else ""
    return urlunsplit((parsed.scheme.lower(), netloc, parsed.path or "/", query, ""))


def video_key(url: str) -> str:
    parsed = urlsplit(url)
    match = re.search(r"/video/(BV[0-9A-Za-z]+)", parsed.path, flags=re.IGNORECASE)
    if match:
        key = match.group(1)
        page = parse_qs(parsed.query).get("p", [None])[0]
        return f"{key}-p{page}" if page and page.isdigit() else key
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
    return f"url-{digest}"


def find_yt_dlp() -> list[str] | None:
    executable = shutil.which("yt-dlp")
    if executable:
        return [executable]
    if importlib.util.find_spec("yt_dlp") is not None:
        return [sys.executable, "-m", "yt_dlp"]
    return None


def tool_report() -> dict[str, Any]:
    yt_dlp = find_yt_dlp()
    faster_whisper = importlib.util.find_spec("faster_whisper") is not None
    missing = [
        name
        for name, available in (("yt-dlp", bool(yt_dlp)), ("faster-whisper", faster_whisper))
        if not available
    ]
    return {
        "status": "ok" if not missing else "tool_missing",
        "missing": missing,
        "tools": {
            "python": sys.executable,
            "yt_dlp": " ".join(yt_dlp) if yt_dlp else None,
            "faster_whisper": faster_whisper,
            "ffmpeg": shutil.which("ffmpeg"),
            "ffprobe": shutil.which("ffprobe"),
        },
        "required": ["yt-dlp", "faster-whisper"],
        "optional": ["ffmpeg", "ffprobe"],
    }


def summarize_tracks(tracks: Any) -> list[dict[str, Any]]:
    if not isinstance(tracks, dict):
        return []

    summary = []
    for language, formats in sorted(tracks.items()):
        safe_formats = []
        if isinstance(formats, list):
            for item in formats:
                if isinstance(item, dict):
                    safe_formats.append(
                        {
                            key: item[key]
                            for key in ("ext", "name", "protocol")
                            if item.get(key) is not None
                        }
                    )
        summary.append({"language": language, "formats": safe_formats})
    return summary


def sanitize_metadata(info: dict[str, Any], requested_url: str) -> dict[str, Any]:
    allowed_fields = (
        "id",
        "display_id",
        "title",
        "duration",
        "timestamp",
        "release_timestamp",
        "uploader",
        "uploader_id",
        "channel",
        "channel_id",
        "webpage_url",
        "extractor",
        "extractor_key",
    )
    result = {key: info.get(key) for key in allowed_fields if info.get(key) is not None}
    webpage_url = result.get("webpage_url")
    if isinstance(webpage_url, str):
        try:
            result["webpage_url"] = normalize_url(webpage_url)
        except InputError:
            result.pop("webpage_url", None)
    result.update(
        {
            "schema_version": 1,
            "requested_url": requested_url,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "native_subtitles": summarize_tracks(info.get("subtitles")),
            "automatic_captions": summarize_tracks(info.get("automatic_captions")),
        }
    )
    return result


def classify_failure(message: str) -> str:
    lowered = message.lower()
    if "412" in lowered or "precondition failed" in lowered or "risk control" in lowered:
        return "anti_bot"
    if any(term in lowered for term in ("login", "log in", "sign in", "cookie", "会员", "登录")):
        return "auth_required"
    if any(term in lowered for term in ("geo", "country", "region", "地区")):
        return "geo_restricted"
    if any(term in lowered for term in ("unavailable", "deleted", "private", "不存在", "已失效")):
        return "video_unavailable"
    if any(
        term in lowered
        for term in ("timed out", "timeout", "network", "connection", "dns", "name resolution")
    ):
        return "network_error"
    return "yt_dlp_error"


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def probe(args: argparse.Namespace) -> int:
    try:
        url = normalize_url(args.url)
    except InputError as exc:
        emit({"status": "invalid_url", "message": str(exc)})
        return 2

    yt_dlp = find_yt_dlp()
    if not yt_dlp:
        emit(
            {
                "status": "tool_missing",
                "missing": "yt-dlp",
                "message": "Install yt-dlp or make its executable available on PATH.",
            }
        )
        return 3

    metadata_path = args.output_root / video_key(url) / "metadata.json"
    if metadata_path.is_file() and not args.refresh:
        emit({"status": "cached", "metadata_path": str(metadata_path.resolve())})
        return 0

    command = [
        *yt_dlp,
        "--dump-single-json",
        "--skip-download",
        "--no-playlist",
        "--no-warnings",
        "--retries",
        "1",
        "--extractor-retries",
        "1",
        "--socket-timeout",
        str(args.socket_timeout),
    ]
    if args.cookies_from_browser:
        command.extend(["--cookies-from-browser", args.cookies_from_browser])
    command.extend(["--", url])

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=args.process_timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        emit(
            {
                "status": "network_error",
                "message": f"yt-dlp exceeded the {args.process_timeout}s process timeout",
            }
        )
        return 4

    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()[-1500:]
        emit({"status": classify_failure(detail), "message": detail})
        return 4

    try:
        info = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        emit({"status": "yt_dlp_error", "message": f"invalid JSON from yt-dlp: {exc}"})
        return 4

    if not isinstance(info, dict):
        emit({"status": "yt_dlp_error", "message": "yt-dlp returned a non-object JSON value"})
        return 4

    metadata = sanitize_metadata(info, url)
    atomic_write_json(metadata_path, metadata)
    emit(
        {
            "status": "ok",
            "metadata_path": str(metadata_path.resolve()),
            "title": metadata.get("title"),
            "duration": metadata.get("duration"),
            "native_subtitle_languages": [
                track["language"] for track in metadata["native_subtitles"]
            ],
            "automatic_caption_languages": [
                track["language"] for track in metadata["automatic_captions"]
            ],
            "used_browser_cookies": bool(args.cookies_from_browser),
        }
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe one Bilibili video with yt-dlp and cache sanitized metadata."
    )
    parser.add_argument("url", nargs="?", help="bilibili.com or b23.tv video URL")
    parser.add_argument("--check-tools", action="store_true", help="report local tools and exit")
    parser.add_argument("--refresh", action="store_true", help="ignore cached metadata")
    parser.add_argument(
        "--cookies-from-browser",
        choices=("edge", "chrome", "firefox", "brave"),
        help="use only after the user explicitly approves browser-cookie access",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--socket-timeout", type=int, default=20)
    parser.add_argument("--process-timeout", type=int, default=90)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.check_tools:
        emit(tool_report())
        return 0
    if not args.url:
        emit({"status": "invalid_url", "message": "provide a URL or use --check-tools"})
        return 2
    return probe(args)


if __name__ == "__main__":
    raise SystemExit(main())
