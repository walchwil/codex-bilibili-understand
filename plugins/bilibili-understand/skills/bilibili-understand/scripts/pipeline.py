"""Prepare and query one cached Bilibili transcript with a single command."""

from __future__ import annotations

import argparse
import json
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
TEMP_AUDIO_SUFFIXES = {".part", ".tmp", ".ytdl"}


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
    }


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
    return (
        metadata.get("model") == requested["model"]
        and actual_language == requested["language"]
        and bool(metadata.get("vad_filter")) == requested["vad_filter"]
        and normalized_hotwords(metadata.get("hotwords")) == requested["hotwords"]
    )


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


def run_asr(
    args: argparse.Namespace, audio_path: Path, video_dir: Path
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
    ]
    if args.vad_filter:
        command.append("--vad-filter")
    if args.hotwords:
        command.extend(["--hotwords", args.hotwords])
    if not args.cpu_fallback:
        command.append("--no-cpu-fallback")
    return run_json_command(command, args.asr_timeout, timeout_status="asr_timeout")


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
    if args.network_attempts < 1:
        emit({"status": "invalid_argument", "message": "--network-attempts must be >= 1"})
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
    state["requested_asr"] = requested_asr
    save_state(state_path, state)
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
        return finish_prepare(
            state_path,
            state,
            "subtitles_available",
            total_started,
            0,
            subtitle_languages=subtitle_languages,
            message="v0.2 detects subtitle tracks but does not normalize them; use --force-asr if local ASR is desired.",
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

    stage_started = time.perf_counter()
    print("[asr] transcribing", file=sys.stderr, flush=True)
    asr_code, asr_result = run_asr(args, audio_path, video_dir)
    asr_status = str(asr_result.get("status"))
    if asr_code != 0 or asr_status != "ok":
        mark_stage(
            state,
            "asr",
            asr_status,
            stage_started,
            attempts=1,
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
        attempts=1,
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
    source = compact_path if compact_path.is_file() else transcript_path
    if not source.is_file():
        emit(
            {
                "status": "transcript_missing",
                "video_key": key,
                "message": "run the prepare command first",
            }
        )
        return 3

    try:
        selected = select_segments(load_segments(source), start_s, end_s)
    except (OSError, ValueError) as exc:
        emit({"status": "transcript_error", "video_key": key, "message": str(exc)})
        return 4
    if not selected:
        emit(
            {
                "status": "no_segments",
                "video_key": key,
                "start_s": start_s,
                "end_s": end_s,
            }
        )
        return 0

    excerpt_path = (
        video_dir
        / "queries"
        / f"{round(start_s * 1000)}-{round(end_s * 1000)}.jsonl"
    )
    atomic_write(
        excerpt_path,
        "".join(
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            for record in selected
        ),
    )
    emit(
        {
            "status": "ok",
            "video_key": key,
            "start_s": start_s,
            "end_s": end_s,
            "segment_count": len(selected),
            "excerpt_path": str(excerpt_path.resolve()),
            "segments": selected,
        }
    )
    return 0


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
    asr_metadata = video_dir / "asr_metadata.json"
    audio = find_audio(video_dir)
    state = read_json(video_dir / "pipeline_state.json")
    artifacts = {
        "metadata": metadata.is_file() and metadata.stat().st_size > 0,
        "audio": audio is not None,
        "transcript": transcript.is_file() and transcript.stat().st_size > 0,
        "segments": segments.is_file() and segments.stat().st_size > 0,
        "asr_metadata": asr_metadata.is_file() and asr_metadata.stat().st_size > 0,
    }
    if artifacts["segments"] or artifacts["transcript"]:
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version="%(prog)s 0.2.0")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare_parser = commands.add_parser(
        "prepare", help="probe, download if needed, transcribe, and cache compact segments"
    )
    prepare_parser.add_argument("url")
    prepare_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    prepare_parser.add_argument(
        "--model", help="Whisper model; defaults to an existing cache model or small"
    )
    prepare_parser.add_argument("--model-cache", type=Path)
    prepare_parser.add_argument(
        "--language", help="speech language; defaults to an existing cache language or zh"
    )
    prepare_parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    prepare_parser.add_argument("--compute-type", default="float16")
    prepare_parser.add_argument("--vad-filter", action="store_true", default=None)
    prepare_parser.add_argument("--hotwords")
    prepare_parser.add_argument("--force-asr", action="store_true")
    prepare_parser.add_argument("--refresh-metadata", action="store_true")
    prepare_parser.add_argument("--refresh-asr", action="store_true")
    prepare_parser.add_argument(
        "--cookies-from-browser", choices=("edge", "chrome", "firefox", "brave")
    )
    prepare_parser.add_argument("--network-attempts", type=int, default=2)
    prepare_parser.add_argument("--socket-timeout", type=int, default=20)
    prepare_parser.add_argument("--probe-timeout", type=int, default=90)
    prepare_parser.add_argument("--download-timeout", type=int, default=3600)
    prepare_parser.add_argument("--asr-timeout", type=int, default=14400)
    prepare_parser.add_argument(
        "--no-cpu-fallback", action="store_false", dest="cpu_fallback"
    )
    prepare_parser.set_defaults(handler=prepare, cpu_fallback=True)

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
