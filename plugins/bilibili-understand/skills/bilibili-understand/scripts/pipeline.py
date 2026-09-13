"""Prepare and query one cached Bilibili transcript with a single command."""

from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stdout
import html
from io import StringIO
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import probe_bilibili


DEFAULT_OUTPUT_ROOT = Path("outputs") / "bilibili-understand"
SCRIPT_DIR = Path(__file__).resolve().parent
PROBE_SCRIPT = SCRIPT_DIR / "probe_bilibili.py"
TRANSCRIBE_SCRIPT = SCRIPT_DIR / "transcribe_audio.py"
RETRYABLE_STATUSES = {"network_error"}
RETRYABLE_ASR_STATUSES = {"asr_timeout", "process_error"}
TEMP_AUDIO_SUFFIXES = {".part", ".tmp", ".ytdl"}


class PipelineBusy(RuntimeError):
    """Another process currently owns this video's download/ASR lock."""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def parse_time(value: str) -> float:
    """Parse seconds, MM:SS, or HH:MM:SS into non-negative seconds."""
    raw = value.strip()
    if re.fullmatch(r"\d+(?:\.\d+)?", raw):
        return float(raw)

    parts = raw.split(":")
    if len(parts) not in (2, 3):
        raise ValueError(f"invalid time {value!r}; use seconds, MM:SS, or HH:MM:SS")
    try:
        seconds = float(parts[-1])
        minutes = int(parts[-2])
        hours = int(parts[0]) if len(parts) == 3 else 0
    except ValueError as exc:
        raise ValueError(f"invalid time {value!r}") from exc
    if seconds < 0 or seconds >= 60 or minutes < 0:
        raise ValueError(f"invalid time {value!r}")
    if len(parts) == 3 and minutes >= 60:
        raise ValueError(f"invalid time {value!r}")
    return hours * 3600 + minutes * 60 + seconds


def resolve_interval(start: str, end: str | None, duration: str | None) -> tuple[float, float]:
    start_s = parse_time(start)
    if (end is None) == (duration is None):
        raise ValueError("provide exactly one of --end or --duration")
    end_s = parse_time(end) if end is not None else start_s + parse_time(duration or "")
    if end_s <= start_s:
        raise ValueError("the interval end must be after its start")
    return start_s, end_s


def load_segments(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                compact = {
                    "start_s": float(record["start_s"]),
                    "end_s": float(record["end_s"]),
                    "text": str(record["text"]).strip(),
                }
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid transcript record at line {line_number}") from exc
            if record.get("source"):
                compact["source"] = str(record["source"])
            records.append(compact)
    return records


def parse_subtitle_time(value: str) -> float:
    raw = value.strip().replace(",", ".")
    parts = raw.split(":")
    if len(parts) == 2:
        hours = 0
        minutes, seconds = parts
    elif len(parts) == 3:
        hours, minutes, seconds = parts
    else:
        raise ValueError(f"invalid subtitle timestamp {value!r}")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def clean_subtitle_text(lines: Iterable[str]) -> str:
    text = " ".join(line.strip() for line in lines if line.strip())
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\{[^}]+\}", "", text)
    return html.unescape(text).strip()


def load_subtitle_segments(path: Path) -> list[dict[str, Any]]:
    """Normalize VTT/SRT cues into the same compact segment contract as ASR."""
    lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    records: list[dict[str, Any]] = []
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if "-->" not in line:
            index += 1
            continue
        left, right = (part.strip() for part in line.split("-->", 1))
        right = right.split(maxsplit=1)[0]
        try:
            start_s = parse_subtitle_time(left.split(maxsplit=1)[0])
            end_s = parse_subtitle_time(right)
        except (ValueError, IndexError):
            index += 1
            continue
        index += 1
        text_lines = []
        while index < len(lines) and lines[index].strip():
            text_lines.append(lines[index])
            index += 1
        text = clean_subtitle_text(text_lines)
        if text and end_s > start_s:
            records.append(
                {
                    "start_s": round(start_s, 3),
                    "end_s": round(end_s, 3),
                    "text": text,
                    "source": f"subtitle:{path.suffix.lstrip('.')}",
                }
            )
        index += 1
    return records


def subtitle_segments_path(video_dir: Path) -> Path:
    return video_dir / "subtitle_segments.jsonl"


def subtitle_cache_available(video_dir: Path) -> bool:
    path = subtitle_segments_path(video_dir)
    return path.is_file() and path.stat().st_size > 0


def select_segments(
    records: Iterable[dict[str, Any]], start_s: float, end_s: float
) -> list[dict[str, Any]]:
    """Select transcript segments overlapping the half-open interval [start, end)."""
    return [
        record
        for record in records
        if float(record["end_s"]) > start_s and float(record["start_s"]) < end_s
    ]


def compact_transcript(source: Path, destination: Path) -> int:
    records = load_segments(source)
    content = "".join(
        json.dumps(
            {
                "start_s": record["start_s"],
                "end_s": record["end_s"],
                "text": record["text"],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n"
        for record in records
    )
    atomic_write(destination, content)
    return len(records)


def ensure_compact_transcript(source: Path, destination: Path) -> tuple[int, bool]:
    if (
        destination.is_file()
        and destination.stat().st_size > 0
        and destination.stat().st_mtime_ns >= source.stat().st_mtime_ns
    ):
        records = load_segments(destination)
        if all("source" not in record for record in records):
            return len(records), True
    return compact_transcript(source, destination), False


def normalized_hotwords(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def resolve_asr_config(
    args: argparse.Namespace, existing: dict[str, Any] | None
) -> dict[str, Any]:
    existing = existing or {}
    requested_word_timestamps = getattr(args, "word_timestamps", None)
    requested_beam_size = getattr(args, "beam_size", None)
    existing_language = existing.get(
        "requested_language", existing.get("language", "zh")
    )
    return {
        "model": args.model or existing.get("model") or "small",
        "language": args.language or existing_language or "zh",
        "vad_filter": (
            bool(args.vad_filter)
            if args.vad_filter is not None
            else bool(existing.get("vad_filter", False))
        ),
        "hotwords": (
            normalized_hotwords(args.hotwords)
            if args.hotwords is not None
            else normalized_hotwords(existing.get("hotwords"))
        ),
        "word_timestamps": (
            bool(requested_word_timestamps)
            if requested_word_timestamps is not None
            else bool(existing.get("word_timestamps", False))
        ),
        "beam_size": (
            int(requested_beam_size)
            if requested_beam_size is not None
            else int(existing.get("beam_size", 5))
        ),
    }


def audio_signature(path: Path | None) -> dict[str, int] | None:
    if path is None:
        return None
    try:
        stat = path.stat()
    except OSError:
        return None
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


@contextmanager
def video_lock(
    video_dir: Path, timeout_s: float = 60.0, stale_after_s: float | None = None
):
    """Serialize mutating work for one video with a recoverable lock file."""
    video_dir.mkdir(parents=True, exist_ok=True)
    lock_path = video_dir / ".pipeline.lock"
    deadline = time.monotonic() + max(0.0, timeout_s)
    stale_after_s = stale_after_s or max(600.0, timeout_s * 10)
    while True:
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(json.dumps({"pid": os.getpid(), "created_at": now_iso()}))
            break
        except FileExistsError:
            try:
                stale = time.time() - lock_path.stat().st_mtime > stale_after_s
            except OSError:
                stale = False
            if stale:
                try:
                    lock_path.unlink()
                    continue
                except OSError:
                    pass
            if time.monotonic() >= deadline:
                raise PipelineBusy(f"video pipeline is busy: {video_dir}")
            time.sleep(0.2)
    try:
        yield
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def asr_cache_matches(
    transcript_path: Path,
    metadata_path: Path,
    requested: dict[str, Any],
) -> bool:
    if not transcript_path.is_file() or transcript_path.stat().st_size == 0:
        return False
    metadata = read_json(metadata_path)
    if metadata is None:
        return False

    actual_language = metadata.get("requested_language", metadata.get("language"))
    config_matches = (
        metadata.get("model") == requested["model"]
        and actual_language == requested["language"]
        and bool(metadata.get("vad_filter")) == requested["vad_filter"]
        and normalized_hotwords(metadata.get("hotwords")) == requested["hotwords"]
    )
    if not config_matches:
        return False
    # New caches bind ASR output to the exact local audio artifact. Older v0.2
    # metadata has no signature and remains readable for backward compatibility.
    defaults = {"word_timestamps": False, "beam_size": 5}
    for field, default in defaults.items():
        if metadata.get(field, default) != requested.get(field, default):
            return False
    current_signature = audio_signature(find_audio(transcript_path.parent))
    saved_signature = metadata.get("audio_signature")
    return saved_signature is None or saved_signature == current_signature


def clip_index_path(video_dir: Path) -> Path:
    return video_dir / "clips" / "index.json"


def load_clip_index(video_dir: Path) -> list[dict[str, Any]]:
    index = read_json(clip_index_path(video_dir)) or {}
    clips = index.get("clips", [])
    return [clip for clip in clips if isinstance(clip, dict)]


def asr_configs_match(actual: Any, requested: dict[str, Any]) -> bool:
    if not isinstance(actual, dict):
        return False
    defaults = {"word_timestamps": False, "beam_size": 5}
    return all(actual.get(key, defaults.get(key)) == value for key, value in requested.items())


def clip_segments_path(video_dir: Path, entry: dict[str, Any]) -> Path | None:
    relative = entry.get("segments_path")
    if not isinstance(relative, str):
        return None
    candidate = (video_dir / relative).resolve()
    try:
        candidate.relative_to(video_dir.resolve())
    except ValueError:
        return None
    return candidate


def covered_clip_entries(
    video_dir: Path,
    start_s: float,
    end_s: float,
    requested: dict[str, Any] | None = None,
    requested_audio_signature: dict[str, int] | None = None,
) -> list[dict[str, Any]] | None:
    candidates = []
    for entry in load_clip_index(video_dir):
        if requested is not None and not asr_configs_match(entry.get("config"), requested):
            continue
        if (
            requested_audio_signature is not None
            and entry.get("audio_signature") is not None
            and entry.get("audio_signature") != requested_audio_signature
        ):
            continue
        path = clip_segments_path(video_dir, entry)
        if (
            path is None
            or not path.is_file()
            or not isinstance(entry.get("coverage_start_s"), (int, float))
            or not isinstance(entry.get("coverage_end_s"), (int, float))
        ):
            continue
        candidates.append(entry)

    groups: list[list[dict[str, Any]]] = []
    if requested is not None:
        groups = [candidates]
    else:
        by_config: dict[str, list[dict[str, Any]]] = {}
        for entry in candidates:
            config_key = json.dumps(entry.get("config"), sort_keys=True)
            by_config.setdefault(config_key, []).append(entry)
        groups = list(by_config.values())

    for group in groups:
        cursor = start_s
        selected = []
        for entry in sorted(group, key=lambda item: float(item["coverage_start_s"])):
            clip_start = float(entry["coverage_start_s"])
            clip_end = float(entry["coverage_end_s"])
            if clip_end <= cursor:
                continue
            if clip_start > cursor + 0.001:
                break
            selected.append(entry)
            cursor = max(cursor, clip_end)
            if cursor >= end_s - 0.001:
                return selected
    return None


def records_from_clip_entries(
    video_dir: Path, entries: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    records = []
    seen = set()
    for entry in entries:
        path = clip_segments_path(video_dir, entry)
        if path is None:
            continue
        for record in load_segments(path):
            key = (record["start_s"], record["end_s"], record["text"])
            if key not in seen:
                seen.add(key)
                records.append(record)
    return sorted(records, key=lambda record: (record["start_s"], record["end_s"]))


def save_clip_entry(video_dir: Path, entry: dict[str, Any]) -> None:
    clips = [
        old
        for old in load_clip_index(video_dir)
        if old.get("segments_path") != entry.get("segments_path")
    ]
    clips.append(entry)
    atomic_write_json(clip_index_path(video_dir), {"schema_version": 1, "clips": clips})


def full_cache_matches(
    video_dir: Path, args: argparse.Namespace, *, allow_legacy: bool = True
) -> bool:
    """Return whether a complete transcript can answer a run request as-is."""
    transcript_path = video_dir / "transcript.jsonl"
    segments_path = video_dir / "segments.jsonl"
    if not transcript_path.is_file() and not segments_path.is_file():
        return False
    if getattr(args, "refresh_asr", False):
        return False

    metadata_path = video_dir / "asr_metadata.json"
    metadata = read_json(metadata_path)
    if metadata is None:
        # v0.2 caches created before semantic ASR metadata existed are still useful.
        return allow_legacy
    requested = resolve_asr_config(args, metadata)
    return asr_cache_matches(transcript_path, metadata_path, requested)


def find_audio(video_dir: Path) -> Path | None:
    for path in sorted(video_dir.glob("audio.*")):
        if (
            path.is_file()
            and path.suffix.lower() not in TEMP_AUDIO_SUFFIXES
            and path.stat().st_size > 0
        ):
            return path
    return None


def target_key(target: str) -> str:
    if "://" in target:
        return probe_bilibili.video_key(probe_bilibili.normalize_url(target))
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", target):
        raise probe_bilibili.InputError("expected a Bilibili URL or a cached video key")
    return target


def run_json_command(
    command: list[str], timeout: int, timeout_status: str = "network_error"
) -> tuple[int, dict[str, Any]]:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return 4, {"status": timeout_status, "message": f"process exceeded {timeout}s"}

    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        detail = (completed.stderr or completed.stdout).strip()[-2000:]
        payload = {
            "status": "process_error",
            "message": detail or f"process exited with code {completed.returncode}",
        }
    return completed.returncode, payload


def should_retry(status: str) -> bool:
    return status in RETRYABLE_STATUSES


def run_probe(args: argparse.Namespace, url: str, force_refresh: bool) -> tuple[dict[str, Any], int]:
    command = [
        sys.executable,
        str(PROBE_SCRIPT),
        url,
        "--output-root",
        str(args.output_root),
        "--socket-timeout",
        str(args.socket_timeout),
        "--process-timeout",
        str(args.probe_timeout),
    ]
    if force_refresh:
        command.append("--refresh")
    if args.cookies_from_browser:
        command.extend(["--cookies-from-browser", args.cookies_from_browser])

    payload: dict[str, Any] = {"status": "process_error"}
    for attempt in range(1, args.network_attempts + 1):
        _, payload = run_json_command(command, args.probe_timeout + 5)
        if not should_retry(str(payload.get("status"))):
            return payload, attempt
    return payload, args.network_attempts


def download_audio(
    args: argparse.Namespace, url: str, video_dir: Path
) -> tuple[dict[str, Any], Path | None, int]:
    yt_dlp = probe_bilibili.find_yt_dlp()
    if not yt_dlp:
        return (
            {
                "status": "tool_missing",
                "message": "Install yt-dlp in the same Python environment.",
            },
            None,
            0,
        )

    command = [
        *yt_dlp,
        "--no-playlist",
        "--format",
        "bestaudio[ext=m4a]/bestaudio/best",
        "--retries",
        "1",
        "--fragment-retries",
        "1",
        "--socket-timeout",
        str(args.socket_timeout),
        "--no-overwrites",
        "--quiet",
        "--output",
        str(video_dir / "audio.%(ext)s"),
    ]
    if args.cookies_from_browser:
        command.extend(["--cookies-from-browser", args.cookies_from_browser])
    command.extend(["--", url])

    payload: dict[str, Any] = {"status": "download_error"}
    for attempt in range(1, args.network_attempts + 1):
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=args.download_timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            payload = {
                "status": "network_error",
                "message": f"audio download exceeded {args.download_timeout}s",
            }
        else:
            audio = find_audio(video_dir)
            if completed.returncode == 0 and audio is not None:
                return {"status": "ok"}, audio, attempt
            detail = (completed.stderr or completed.stdout).strip()[-2000:]
            payload = {
                "status": probe_bilibili.classify_failure(detail),
                "message": detail or f"yt-dlp exited with code {completed.returncode}",
            }
        if not should_retry(str(payload.get("status"))):
            return payload, None, attempt
    return payload, None, args.network_attempts


def download_subtitles(
    args: argparse.Namespace, url: str, video_dir: Path
) -> tuple[dict[str, Any], Path | None, int]:
    """Fetch native captions without exposing signed subtitle URLs in metadata."""
    yt_dlp = probe_bilibili.find_yt_dlp()
    if not yt_dlp:
        return {"status": "tool_missing", "message": "yt-dlp is unavailable"}, None, 0

    existing = sorted(video_dir.glob("subtitle.*.vtt")) + sorted(video_dir.glob("subtitle.vtt"))
    if existing:
        return {"status": "cached"}, existing[0], 0

    base_command = [
        *yt_dlp,
        "--skip-download",
        "--no-playlist",
        "--no-warnings",
        "--sub-langs",
        "zh.*,zh,chi_sim,chi_tra,en.*",
        "--sub-format",
        "vtt",
        "--retries",
        "1",
        "--socket-timeout",
        str(args.socket_timeout),
        "--output",
        str(video_dir / "subtitle.%(language)s.%(ext)s"),
    ]
    if args.cookies_from_browser:
        base_command.extend(["--cookies-from-browser", args.cookies_from_browser])

    for automatic in (False, True):
        command = [*base_command, "--write-auto-subs" if automatic else "--write-subs", "--", url]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=args.download_timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {
                "status": "network_error",
                "message": f"subtitle download exceeded {args.download_timeout}s",
            }, None, 1
        subtitle_files = sorted(video_dir.glob("subtitle.*.vtt")) + sorted(
            video_dir.glob("subtitle.vtt")
        )
        if completed.returncode == 0 and subtitle_files:
            return {"status": "ok", "automatic": automatic}, subtitle_files[0], 1
        if completed.returncode != 0 and not automatic:
            detail = (completed.stderr or completed.stdout).strip()[-2000:]
            if probe_bilibili.classify_failure(detail) in {"anti_bot", "auth_required"}:
                return {"status": probe_bilibili.classify_failure(detail), "message": detail}, None, 1
    return {"status": "no_subtitles"}, None, 1


def save_subtitle_cache(video_dir: Path, source_path: Path) -> int:
    records = load_subtitle_segments(source_path)
    if not records:
        raise ValueError("subtitle file contained no usable cues")
    destination = subtitle_segments_path(video_dir)
    atomic_write(
        destination,
        "".join(
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            for record in records
        ),
    )
    atomic_write_json(
        video_dir / "subtitle_metadata.json",
        {
            "schema_version": 1,
            "source_path": source_path.name,
            "segment_count": len(records),
            "fetched_at": now_iso(),
        },
    )
    return len(records)


def run_asr(
    args: argparse.Namespace,
    audio_path: Path,
    video_dir: Path,
    clip_start: float | None = None,
    clip_end: float | None = None,
) -> tuple[int, dict[str, Any]]:
    model_cache = args.model_cache or args.output_root / "_models"
    command = [
        sys.executable,
        str(TRANSCRIBE_SCRIPT),
        str(audio_path),
        "--output-dir",
        str(video_dir),
        "--model",
        args.model,
        "--model-cache",
        str(model_cache),
        "--language",
        args.language,
        "--device",
        args.device,
        "--compute-type",
        args.compute_type,
        "--beam-size",
        str(args.beam_size),
    ]
    if args.vad_filter:
        command.append("--vad-filter")
    if args.hotwords:
        command.extend(["--hotwords", args.hotwords])
    if args.word_timestamps:
        command.append("--word-timestamps")
    if clip_start is not None and clip_end is not None:
        command.extend(["--clip-start", str(clip_start), "--clip-end", str(clip_end)])
    if not args.cpu_fallback:
        command.append("--no-cpu-fallback")
    return run_json_command(command, args.asr_timeout, timeout_status="asr_timeout")


def run_asr_with_retry(
    args: argparse.Namespace,
    audio_path: Path,
    output_dir: Path,
    clip_start: float | None = None,
    clip_end: float | None = None,
) -> tuple[int, dict[str, Any], int]:
    attempts = max(1, int(args.asr_attempts))
    result: dict[str, Any] = {"status": "process_error"}
    code = 4
    for attempt in range(1, attempts + 1):
        code, result = run_asr(
            args,
            audio_path,
            output_dir,
            clip_start=clip_start,
            clip_end=clip_end,
        )
        if code == 0 and result.get("status") == "ok":
            return code, result, attempt
        if result.get("status") not in RETRYABLE_ASR_STATUSES or attempt == attempts:
            return code, result, attempt
        time.sleep(min(2.0, 0.25 * attempt))
    return code, result, attempts


def mark_stage(
    state: dict[str, Any],
    name: str,
    status: str,
    started: float,
    **details: Any,
) -> None:
    stage = {"status": status, "elapsed_s": round(time.perf_counter() - started, 3)}
    stage.update({key: value for key, value in details.items() if value is not None})
    state["stages"][name] = stage


def save_state(path: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = now_iso()
    atomic_write_json(path, state)


def finish_prepare(
    state_path: Path,
    state: dict[str, Any],
    status: str,
    total_started: float,
    exit_code: int,
    **details: Any,
) -> int:
    state["status"] = status
    state["total_elapsed_s"] = round(time.perf_counter() - total_started, 3)
    state.update({key: value for key, value in details.items() if value is not None})
    save_state(state_path, state)
    response = {
        "status": status,
        "video_key": state["video_key"],
        "video_dir": state["video_dir"],
        "stages": {
            name: stage["status"] for name, stage in state["stages"].items()
        },
        "timings_s": {
            name: stage["elapsed_s"] for name, stage in state["stages"].items()
        },
        "total_s": state["total_elapsed_s"],
    }
    response.update({key: value for key, value in details.items() if value is not None})
    emit(response)
    return exit_code


def prepare(args: argparse.Namespace) -> int:
    total_started = time.perf_counter()
    try:
        url = probe_bilibili.normalize_url(args.url)
    except probe_bilibili.InputError as exc:
        emit({"status": "invalid_url", "message": str(exc)})
        return 2
    if args.network_attempts < 1 or args.asr_attempts < 1:
        emit(
            {
                "status": "invalid_argument",
                "message": "--network-attempts and --asr-attempts must be >= 1",
            }
        )
        return 2

    key = probe_bilibili.video_key(url)
    video_dir = args.output_root / key
    video_dir.mkdir(parents=True, exist_ok=True)
    state_path = video_dir / "pipeline_state.json"
    state: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at": now_iso(),
        "video_key": key,
        "source_url": url,
        "video_dir": str(video_dir.resolve()),
        "stages": {},
    }
    save_state(state_path, state)

    metadata_path = video_dir / "metadata.json"
    stage_started = time.perf_counter()
    metadata = None if args.refresh_metadata else read_json(metadata_path)
    if metadata is not None:
        mark_stage(state, "probe", "cached", stage_started, attempts=0)
    else:
        print("[probe] fetching metadata", file=sys.stderr, flush=True)
        probe_result, attempts = run_probe(
            args, url, force_refresh=args.refresh_metadata or metadata_path.exists()
        )
        probe_status = str(probe_result.get("status"))
        metadata = read_json(metadata_path) if probe_status in {"ok", "cached"} else None
        if metadata is None:
            mark_stage(
                state,
                "probe",
                probe_status,
                stage_started,
                attempts=attempts,
                message=probe_result.get("message"),
            )
            return finish_prepare(
                state_path,
                state,
                probe_status,
                total_started,
                3 if probe_status == "tool_missing" else 4,
                message=probe_result.get("message"),
            )
        mark_stage(state, "probe", probe_status, stage_started, attempts=attempts)
    save_state(state_path, state)

    transcript_path = video_dir / "transcript.jsonl"
    asr_metadata_path = video_dir / "asr_metadata.json"
    segments_path = video_dir / "segments.jsonl"
    requested_asr = resolve_asr_config(args, read_json(asr_metadata_path))
    args.model = requested_asr["model"]
    args.language = requested_asr["language"]
    args.vad_filter = requested_asr["vad_filter"]
    args.hotwords = requested_asr["hotwords"]
    if requested_asr["beam_size"] < 1:
        return finish_prepare(
            state_path,
            state,
            "invalid_argument",
            total_started,
            2,
            message="--beam-size must be >= 1",
        )
    state["requested_asr"] = requested_asr
    save_state(state_path, state)
    if subtitle_cache_available(video_dir) and not args.force_asr:
        stage_started = time.perf_counter()
        try:
            segment_count = len(load_segments(subtitle_segments_path(video_dir)))
        except (OSError, ValueError):
            segment_count = 0
        if segment_count:
            mark_stage(
                state,
                "subtitles",
                "cached",
                stage_started,
                path=str(subtitle_segments_path(video_dir).resolve()),
            )
            return finish_prepare(
                state_path,
                state,
                "ready",
                total_started,
                0,
                cached=True,
                source="subtitle_cache",
                segments_path=str(subtitle_segments_path(video_dir).resolve()),
                segment_count=segment_count,
            )
    if not args.refresh_asr and asr_cache_matches(
        transcript_path, asr_metadata_path, requested_asr
    ):
        stage_started = time.perf_counter()
        audio_path = find_audio(video_dir)
        mark_stage(
            state,
            "audio",
            "cached" if audio_path else "not_needed",
            stage_started,
            path=str(audio_path.resolve()) if audio_path else None,
        )
        stage_started = time.perf_counter()
        mark_stage(
            state,
            "asr",
            "cached",
            stage_started,
            transcript_path=str(transcript_path.resolve()),
        )
        stage_started = time.perf_counter()
        try:
            segment_count, compact_cached = ensure_compact_transcript(
                transcript_path, segments_path
            )
        except (OSError, ValueError) as exc:
            mark_stage(state, "compact", "transcript_error", stage_started, message=str(exc))
            return finish_prepare(
                state_path,
                state,
                "transcript_error",
                total_started,
                4,
                message=str(exc),
            )
        mark_stage(
            state,
            "compact",
            "cached" if compact_cached else "ok",
            stage_started,
            segment_count=segment_count,
            path=str(segments_path.resolve()),
        )
        return finish_prepare(
            state_path,
            state,
            "ready",
            total_started,
            0,
            cached=True,
            segments_path=str(segments_path.resolve()),
            segment_count=segment_count,
        )

    subtitle_languages = []
    for field in ("native_subtitles", "automatic_captions"):
        tracks = metadata.get(field)
        if isinstance(tracks, list):
            subtitle_languages.extend(
                track["language"]
                for track in tracks
                if isinstance(track, dict) and track.get("language")
            )
    if subtitle_languages and not args.force_asr:
        if getattr(args, "prefer_subtitles", False):
            stage_started = time.perf_counter()
            print("[subtitles] downloading captions", file=sys.stderr, flush=True)
            subtitle_result, subtitle_path, attempts = download_subtitles(args, url, video_dir)
            if subtitle_path is not None:
                try:
                    segment_count = save_subtitle_cache(video_dir, subtitle_path)
                except (OSError, ValueError) as exc:
                    subtitle_result = {"status": "subtitle_error", "message": str(exc)}
                else:
                    mark_stage(
                        state,
                        "subtitles",
                        "ok",
                        stage_started,
                        attempts=attempts,
                        path=str(subtitle_segments_path(video_dir).resolve()),
                        segment_count=segment_count,
                    )
                    return finish_prepare(
                        state_path,
                        state,
                        "ready",
                        total_started,
                        0,
                        cached=False,
                        source="subtitle_cache",
                        segments_path=str(subtitle_segments_path(video_dir).resolve()),
                        segment_count=segment_count,
                    )
            mark_stage(
                state,
                "subtitles",
                str(subtitle_result.get("status", "subtitle_error")),
                stage_started,
                attempts=attempts,
                message=subtitle_result.get("message"),
            )
            if getattr(args, "allow_asr_fallback", False):
                args.force_asr = True
            else:
                return finish_prepare(
                    state_path,
                    state,
                    "subtitles_available",
                    total_started,
                    0,
                    subtitle_languages=subtitle_languages,
                    message="subtitle download failed; use --force-asr to run local ASR",
                )
        else:
            return finish_prepare(
                state_path,
                state,
                "subtitles_available",
                total_started,
                0,
                subtitle_languages=subtitle_languages,
                message="subtitle tracks detected; use --prefer-subtitles to normalize them",
            )

    stage_started = time.perf_counter()
    audio_path = find_audio(video_dir)
    if audio_path is not None:
        mark_stage(
            state, "audio", "cached", stage_started, attempts=0, path=str(audio_path.resolve())
        )
    else:
        print("[audio] downloading", file=sys.stderr, flush=True)
        download_result, audio_path, attempts = download_audio(args, url, video_dir)
        download_status = str(download_result.get("status"))
        if audio_path is None:
            mark_stage(
                state,
                "audio",
                download_status,
                stage_started,
                attempts=attempts,
                message=download_result.get("message"),
            )
            return finish_prepare(
                state_path,
                state,
                download_status,
                total_started,
                3 if download_status == "tool_missing" else 4,
                message=download_result.get("message"),
            )
        mark_stage(
            state,
            "audio",
            "ok",
            stage_started,
            attempts=attempts,
            path=str(audio_path.resolve()),
        )
    save_state(state_path, state)

    if args.audio_only:
        return finish_prepare(
            state_path,
            state,
            "audio_ready",
            total_started,
            0,
            audio_path=str(audio_path.resolve()),
        )

    stage_started = time.perf_counter()
    print("[asr] transcribing", file=sys.stderr, flush=True)
    asr_code, asr_result, asr_attempts = run_asr_with_retry(args, audio_path, video_dir)
    asr_status = str(asr_result.get("status"))
    if asr_code != 0 or asr_status != "ok":
        mark_stage(
            state,
            "asr",
            asr_status,
            stage_started,
            attempts=asr_attempts,
            message=asr_result.get("message"),
        )
        return finish_prepare(
            state_path,
            state,
            asr_status,
            total_started,
            3 if asr_status == "tool_missing" else 5,
            message=asr_result.get("message"),
        )
    mark_stage(
        state,
        "asr",
        "ok",
        stage_started,
        attempts=asr_attempts,
        transcript_path=str(transcript_path.resolve()),
    )
    save_state(state_path, state)

    stage_started = time.perf_counter()
    try:
        segment_count = compact_transcript(transcript_path, segments_path)
    except (OSError, ValueError) as exc:
        mark_stage(state, "compact", "transcript_error", stage_started, message=str(exc))
        return finish_prepare(
            state_path,
            state,
            "transcript_error",
            total_started,
            5,
            message=str(exc),
        )
    mark_stage(
        state,
        "compact",
        "ok",
        stage_started,
        segment_count=segment_count,
        path=str(segments_path.resolve()),
    )
    return finish_prepare(
        state_path,
        state,
        "ready",
        total_started,
        0,
        cached=False,
        segments_path=str(segments_path.resolve()),
        segment_count=segment_count,
    )


def emit_query_result(
    video_dir: Path,
    key: str,
    start_s: float,
    end_s: float,
    records: list[dict[str, Any]],
    source: str,
    timings: dict[str, float] | None = None,
) -> int:
    selected = select_segments(records, start_s, end_s)
    if not selected:
        emit(
            {
                "status": "no_segments",
                "video_key": key,
                "start_s": start_s,
                "end_s": end_s,
                "source": source,
            }
        )
        return 0

    excerpt_path = video_dir / "queries" / f"{round(start_s * 1000)}-{round(end_s * 1000)}.jsonl"
    atomic_write(
        excerpt_path,
        "".join(
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            for record in selected
        ),
    )
    response = {
        "status": "ok",
        "video_key": key,
        "start_s": start_s,
        "end_s": end_s,
        "segment_count": len(selected),
        "excerpt_path": str(excerpt_path.resolve()),
        "source": source,
        "segments": selected,
    }
    if timings:
        response["timings_s"] = timings
    emit(response)
    return 0


def query(args: argparse.Namespace) -> int:
    try:
        key = target_key(args.target)
        start_s, end_s = resolve_interval(args.start, args.end, args.duration)
    except (probe_bilibili.InputError, ValueError) as exc:
        emit({"status": "invalid_argument", "message": str(exc)})
        return 2

    video_dir = args.output_root / key
    compact_path = video_dir / "segments.jsonl"
    transcript_path = video_dir / "transcript.jsonl"
    if compact_path.is_file():
        source, source_name = compact_path, "full_cache"
    elif transcript_path.is_file():
        source, source_name = transcript_path, "full_cache"
    elif subtitle_cache_available(video_dir):
        source, source_name = subtitle_segments_path(video_dir), "subtitle_cache"
    else:
        entries = covered_clip_entries(video_dir, start_s, end_s)
        if entries is None:
            emit(
                {
                    "status": "transcript_missing",
                    "video_key": key,
                    "message": "run the run command for this range or prepare the full transcript first",
                }
            )
            return 3
        try:
            return emit_query_result(
                video_dir,
                key,
                start_s,
                end_s,
                records_from_clip_entries(video_dir, entries),
                "clip_cache",
            )
        except (OSError, ValueError) as exc:
            emit({"status": "transcript_error", "video_key": key, "message": str(exc)})
            return 4

    try:
        records = load_segments(source)
    except (OSError, ValueError) as exc:
        emit({"status": "transcript_error", "video_key": key, "message": str(exc)})
        return 4
    return emit_query_result(video_dir, key, start_s, end_s, records, source_name)


def capture_prepare(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    output = StringIO()
    with redirect_stdout(output):
        code = prepare(args)
    lines = [line for line in output.getvalue().splitlines() if line.strip()]
    if not lines:
        return code, {"status": "prepare_error", "message": "prepare returned no JSON"}
    try:
        return code, json.loads(lines[-1])
    except json.JSONDecodeError:
        return code, {"status": "prepare_error", "message": lines[-1][-2000:]}


def _run_unlocked(args: argparse.Namespace) -> int:
    if args.full and any(value is not None for value in (args.start, args.end, args.duration)):
        emit({"status": "invalid_argument", "message": "--full cannot be combined with a time range"})
        return 2
    if not args.full:
        if args.start is None:
            emit({"status": "invalid_argument", "message": "provide --start or use --full"})
            return 2
        try:
            start_s, end_s = resolve_interval(args.start, args.end, args.duration)
        except ValueError as exc:
            emit({"status": "invalid_argument", "message": str(exc)})
            return 2
        if args.context_before < 0 or args.context_after < 0:
            emit({"status": "invalid_argument", "message": "context padding must be non-negative"})
            return 2

    try:
        key = target_key(args.url)
    except probe_bilibili.InputError as exc:
        emit({"status": "invalid_argument", "message": str(exc)})
        return 2
    video_dir = args.output_root / key

    # Do not probe or download anything when an existing full/clip cache can
    # already answer the request. This is the hot path for repeated questions.
    if not args.full:
        requested_config = resolve_asr_config(
            args, read_json(video_dir / "asr_metadata.json")
        )
        if not args.force_asr and subtitle_cache_available(video_dir):
            return query(
                argparse.Namespace(
                    target=args.url,
                    output_root=args.output_root,
                    start=args.start,
                    end=args.end,
                    duration=args.duration,
                )
            )
        if full_cache_matches(video_dir, args):
            return query(
                argparse.Namespace(
                    target=args.url,
                    output_root=args.output_root,
                    start=args.start,
                    end=args.end,
                    duration=args.duration,
                )
            )
        if not args.refresh_asr:
            cached_entries = covered_clip_entries(
                video_dir,
                start_s,
                end_s,
                requested_config,
                audio_signature(find_audio(video_dir)),
            )
            if cached_entries is not None:
                return emit_query_result(
                    video_dir,
                    key,
                    start_s,
                    end_s,
                    records_from_clip_entries(video_dir, cached_entries),
                    "clip_cache",
                )

    prepare_args = argparse.Namespace(
        url=args.url,
        output_root=args.output_root,
        model=args.model,
        model_cache=args.model_cache,
        language=args.language,
        device=args.device,
        compute_type=args.compute_type,
        vad_filter=args.vad_filter,
        hotwords=args.hotwords,
        word_timestamps=args.word_timestamps,
        beam_size=args.beam_size,
        force_asr=args.force_asr,
        prefer_subtitles=not args.force_asr,
        allow_asr_fallback=True,
        refresh_metadata=args.refresh_metadata,
        refresh_asr=args.refresh_asr,
        cookies_from_browser=args.cookies_from_browser,
        network_attempts=args.network_attempts,
        socket_timeout=args.socket_timeout,
        probe_timeout=args.probe_timeout,
        download_timeout=args.download_timeout,
        asr_timeout=args.asr_timeout,
        asr_attempts=args.asr_attempts,
        cpu_fallback=args.cpu_fallback,
        audio_only=not args.full,
    )
    prepare_code, prepare_result = capture_prepare(prepare_args)
    if args.full:
        prepare_result["mode"] = "full"
        emit(prepare_result)
        return prepare_code
    if prepare_code != 0:
        emit(
            {
                "status": "run_failed",
                "phase": "prepare",
                "video_key": prepare_result.get("video_key"),
                "detail": prepare_result,
            }
        )
        return prepare_code

    requested_config = {
        "model": prepare_args.model,
        "language": prepare_args.language,
        "vad_filter": bool(prepare_args.vad_filter),
        "hotwords": normalized_hotwords(prepare_args.hotwords),
        "word_timestamps": bool(prepare_args.word_timestamps),
        "beam_size": int(prepare_args.beam_size),
    }

    if not args.force_asr and subtitle_cache_available(video_dir):
        return query(
            argparse.Namespace(
                target=args.url,
                output_root=args.output_root,
                start=args.start,
                end=args.end,
                duration=args.duration,
            )
        )

    if full_cache_matches(video_dir, prepare_args):
        return query(
            argparse.Namespace(
                target=args.url,
                output_root=args.output_root,
                start=args.start,
                end=args.end,
                duration=args.duration,
            )
        )

    if not args.refresh_asr:
        cached_entries = covered_clip_entries(
            video_dir,
            start_s,
            end_s,
            requested_config,
            audio_signature(find_audio(video_dir)),
        )
    else:
        cached_entries = None
    if cached_entries is not None:
        return emit_query_result(
            video_dir,
            key,
            start_s,
            end_s,
            records_from_clip_entries(video_dir, cached_entries),
            "clip_cache",
        )

    audio_path = find_audio(video_dir)
    if audio_path is None:
        emit({"status": "run_failed", "phase": "audio", "message": "audio cache is missing"})
        return 4
    metadata = read_json(video_dir / "metadata.json") or {}
    duration = metadata.get("duration")
    clip_start = max(0.0, start_s - args.context_before)
    clip_end = end_s + args.context_after
    if isinstance(duration, (int, float)):
        if start_s >= float(duration):
            emit({"status": "invalid_argument", "message": "start is beyond the video duration"})
            return 2
        clip_end = min(clip_end, float(duration))
    if clip_end <= clip_start:
        emit({"status": "invalid_argument", "message": "clip interval is empty"})
        return 2

    clip_name = f"{round(clip_start * 1000)}-{round(clip_end * 1000)}"
    clip_dir = video_dir / "clips" / clip_name
    print(f"[clip] transcribing {clip_start:.3f}s-{clip_end:.3f}s", file=sys.stderr, flush=True)
    asr_started = time.perf_counter()
    asr_code, asr_result, asr_attempts = run_asr_with_retry(
        prepare_args,
        audio_path,
        clip_dir,
        clip_start=clip_start,
        clip_end=clip_end,
    )
    if asr_code != 0 or asr_result.get("status") != "ok":
        emit(
            {
                "status": "run_failed",
                "phase": "asr",
                "detail": asr_result,
                "clip_start_s": clip_start,
                "clip_end_s": clip_end,
                "attempts": asr_attempts,
            }
        )
        return 5

    transcript_path = clip_dir / "transcript.jsonl"
    segments_path = clip_dir / "segments.jsonl"
    compact_started = time.perf_counter()
    try:
        compact_transcript(transcript_path, segments_path)
        save_clip_entry(
            video_dir,
            {
                "coverage_start_s": clip_start,
                "coverage_end_s": clip_end,
                "segments_path": str(segments_path.relative_to(video_dir)),
                "config": requested_config,
                "audio_signature": audio_signature(audio_path),
            },
        )
        records = load_segments(segments_path)
    except (OSError, ValueError) as exc:
        emit({"status": "run_failed", "phase": "compact", "message": str(exc)})
        return 5
    timings = {
        "asr": round(time.perf_counter() - asr_started, 3),
        "compact": round(time.perf_counter() - compact_started, 3),
    }
    timings["total"] = round(sum(timings.values()), 3)
    state_path = video_dir / "pipeline_state.json"
    state = read_json(state_path) or {"schema_version": 1, "stages": {}}
    state["status"] = "ready"
    state["last_run"] = {
        "mode": "range",
        "status": "ready",
        "start_s": start_s,
        "end_s": end_s,
        "clip_start_s": clip_start,
        "clip_end_s": clip_end,
        "source": "clip_new",
        "attempts": asr_attempts,
        "timings_s": timings,
    }
    save_state(state_path, state)
    return emit_query_result(
        video_dir, key, start_s, end_s, records, "clip_new", timings=timings
    )


def run(args: argparse.Namespace) -> int:
    try:
        key = target_key(args.url)
    except probe_bilibili.InputError as exc:
        emit({"status": "invalid_argument", "message": str(exc)})
        return 2
    try:
        with video_lock(
            args.output_root / key,
            float(getattr(args, "lock_timeout", 60.0)),
            stale_after_s=max(
                float(getattr(args, "asr_timeout", 14400)),
                float(getattr(args, "download_timeout", 3600)),
                3600.0,
            )
            * 2,
        ):
            return _run_unlocked(args)
    except PipelineBusy as exc:
        emit(
            {
                "status": "pipeline_busy",
                "video_key": key,
                "message": str(exc),
                "retryable": True,
            }
        )
        return 6


def status(args: argparse.Namespace) -> int:
    try:
        key = target_key(args.target)
    except probe_bilibili.InputError as exc:
        emit({"status": "invalid_argument", "message": str(exc)})
        return 2

    video_dir = args.output_root / key
    metadata = video_dir / "metadata.json"
    transcript = video_dir / "transcript.jsonl"
    segments = video_dir / "segments.jsonl"
    subtitle_segments = subtitle_segments_path(video_dir)
    asr_metadata = video_dir / "asr_metadata.json"
    audio = find_audio(video_dir)
    state = read_json(video_dir / "pipeline_state.json")
    artifacts = {
        "metadata": metadata.is_file() and metadata.stat().st_size > 0,
        "audio": audio is not None,
        "transcript": transcript.is_file() and transcript.stat().st_size > 0,
        "segments": segments.is_file() and segments.stat().st_size > 0,
        "subtitle_segments": subtitle_segments.is_file() and subtitle_segments.stat().st_size > 0,
        "asr_metadata": asr_metadata.is_file() and asr_metadata.stat().st_size > 0,
    }
    if artifacts["segments"] or artifacts["transcript"] or artifacts["subtitle_segments"]:
        cache_status = "ready"
    elif artifacts["metadata"] or artifacts["audio"]:
        cache_status = "partial"
    else:
        cache_status = "missing"
    emit(
        {
            "status": cache_status,
            "video_key": key,
            "video_dir": str(video_dir.resolve()),
            "pipeline_status": state.get("status") if state else None,
            "artifacts": artifacts,
        }
    )
    return 0


def add_prepare_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--model", help="Whisper model; defaults to an existing cache model or small"
    )
    parser.add_argument("--model-cache", type=Path)
    parser.add_argument(
        "--language", help="speech language; defaults to an existing cache language or zh"
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--vad-filter", action="store_true", default=None)
    parser.add_argument("--hotwords")
    parser.add_argument(
        "--prefer-subtitles",
        action="store_true",
        help="download and normalize available captions before local ASR",
    )
    parser.add_argument(
        "--word-timestamps",
        action="store_true",
        default=None,
        help="retain word-level timestamps; off by default for faster, smaller transcripts",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=None,
        help="Whisper beam size; lower values trade accuracy for speed",
    )
    parser.add_argument("--force-asr", action="store_true")
    parser.add_argument("--refresh-metadata", action="store_true")
    parser.add_argument("--refresh-asr", action="store_true")
    parser.add_argument(
        "--cookies-from-browser", choices=("edge", "chrome", "firefox", "brave")
    )
    parser.add_argument("--network-attempts", type=int, default=2)
    parser.add_argument("--socket-timeout", type=int, default=20)
    parser.add_argument("--probe-timeout", type=int, default=90)
    parser.add_argument("--download-timeout", type=int, default=3600)
    parser.add_argument("--asr-timeout", type=int, default=14400)
    parser.add_argument(
        "--asr-attempts",
        type=int,
        default=1,
        help="bounded retries for transient local ASR process failures",
    )
    parser.add_argument(
        "--no-cpu-fallback", action="store_false", dest="cpu_fallback", default=True
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version="%(prog)s 0.4.0")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare_parser = commands.add_parser(
        "prepare", help="probe, download if needed, transcribe, and cache compact segments"
    )
    prepare_parser.add_argument("url")
    add_prepare_options(prepare_parser)
    prepare_parser.add_argument("--audio-only", action="store_true", help=argparse.SUPPRESS)
    prepare_parser.set_defaults(handler=prepare, audio_only=False)

    run_parser = commands.add_parser(
        "run", help="transcribe only a requested range, or prepare the full transcript"
    )
    run_parser.add_argument("url")
    add_prepare_options(run_parser)
    run_parser.add_argument("--full", action="store_true")
    run_parser.add_argument("--start")
    run_interval = run_parser.add_mutually_exclusive_group()
    run_interval.add_argument("--end")
    run_interval.add_argument("--duration")
    run_parser.add_argument("--context-before", type=float, default=3.0)
    run_parser.add_argument("--context-after", type=float, default=3.0)
    run_parser.add_argument(
        "--lock-timeout",
        type=float,
        default=60.0,
        help="seconds to wait for another process handling the same video",
    )
    run_parser.set_defaults(handler=run)

    query_parser = commands.add_parser(
        "query", help="return only transcript segments overlapping one time range"
    )
    query_parser.add_argument("target", help="Bilibili URL or cached video key")
    query_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    query_parser.add_argument("--start", required=True)
    interval = query_parser.add_mutually_exclusive_group(required=True)
    interval.add_argument("--end")
    interval.add_argument("--duration")
    query_parser.set_defaults(handler=query)

    status_parser = commands.add_parser("status", help="inspect cached pipeline stages")
    status_parser.add_argument("target", help="Bilibili URL or cached video key")
    status_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    status_parser.set_defaults(handler=status)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
