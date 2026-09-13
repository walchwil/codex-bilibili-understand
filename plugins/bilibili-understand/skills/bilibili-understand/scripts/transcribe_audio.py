"""Transcribe one local audio file into timestamped JSONL and Markdown."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
from pathlib import Path
from typing import Any


DEFAULT_MODEL_CACHE = Path("outputs") / "bilibili-understand" / "_models"
_WINDOWS_DLL_HANDLES: list[Any] = []
_RUNTIME_READY = False


def timestamp(seconds: float) -> str:
    milliseconds = round(seconds * 1000)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def prepare_runtime(device: str) -> None:
    global _RUNTIME_READY
    if sys.platform != "win32" or _RUNTIME_READY:
        return

    # Initialize Conda's BLAS runtime before CTranslate2 initializes its OpenMP runtime.
    import numpy as np

    _ = np.ones((80, 201), dtype=np.float32) @ np.ones(
        (201, 3001), dtype=np.float32
    )

    if device == "cuda":
        candidates = (
            Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib",
            Path(sys.prefix) / "bin",
            Path(sys.prefix) / "Library" / "bin",
        )
        for directory in candidates:
            libraries = (
                directory / "cublasLt64_12.dll",
                directory / "cublas64_12.dll",
            )
            if all(library.is_file() for library in libraries):
                _WINDOWS_DLL_HANDLES.append(os.add_dll_directory(str(directory)))
                _WINDOWS_DLL_HANDLES.extend(
                    ctypes.CDLL(str(library)) for library in libraries
                )
                break

    _RUNTIME_READY = True


def run_whisper(
    audio_path: Path,
    model_name: str,
    model_cache: Path,
    language: str,
    device: str,
    compute_type: str,
    vad_filter: bool,
    hotwords: str | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prepare_runtime(device)
    from faster_whisper import WhisperModel

    model = WhisperModel(
        model_name,
        device=device,
        compute_type=compute_type,
        download_root=str(model_cache),
    )
    segments, info = model.transcribe(
        str(audio_path),
        language=language,
        vad_filter=vad_filter,
        word_timestamps=True,
        hotwords=hotwords,
    )

    records = []
    for segment in segments:
        words = [
            {
                "start_s": round(word.start, 3),
                "end_s": round(word.end, 3),
                "text": word.word,
                "probability": round(word.probability, 4),
            }
            for word in (segment.words or [])
        ]
        records.append(
            {
                "start_s": round(segment.start, 3),
                "end_s": round(segment.end, 3),
                "text": segment.text.strip(),
                "words": words,
                "source": f"asr:faster-whisper/{model_name}",
            }
        )

    details = {
        "model": model_name,
        "requested_language": language,
        "device": device,
        "compute_type": compute_type,
        "vad_filter": vad_filter,
        "hotwords": hotwords,
        "language": getattr(info, "language", language),
        "language_probability": getattr(info, "language_probability", None),
        "duration_s": getattr(info, "duration", None),
        "duration_after_vad_s": getattr(info, "duration_after_vad", None),
        "segment_count": len(records),
    }
    return records, details


def is_cuda_runtime_error(error: Exception) -> bool:
    message = str(error).lower()
    return any(token in message for token in ("cuda", "cublas", "cudnn", "gpu"))


def transcribe(args: argparse.Namespace) -> int:
    if not args.audio.is_file():
        print(json.dumps({"status": "invalid_audio", "path": str(args.audio)}))
        return 2

    try:
        records, details = run_whisper(
            args.audio,
            args.model,
            args.model_cache,
            args.language,
            args.device,
            args.compute_type,
            args.vad_filter,
            args.hotwords,
        )
    except ImportError:
        print(
            json.dumps(
                {
                    "status": "tool_missing",
                    "missing": "faster-whisper",
                    "message": "Install faster-whisper in the same Python environment.",
                }
            )
        )
        return 3
    except (OSError, RuntimeError) as error:
        if args.device != "cuda" or not args.cpu_fallback or not is_cuda_runtime_error(error):
            raise
        print(
            json.dumps(
                {"status": "cuda_failed", "message": str(error), "fallback": "cpu/int8"},
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        records, details = run_whisper(
            args.audio,
            args.model,
            args.model_cache,
            args.language,
            "cpu",
            "int8",
            args.vad_filter,
            args.hotwords,
        )

    if not records:
        print(json.dumps({"status": "asr_error", "message": "ASR returned no speech segments"}))
        return 4

    output_dir = args.output_dir or args.audio.parent
    jsonl_path = output_dir / "transcript.jsonl"
    markdown_path = output_dir / "transcript.md"
    details_path = output_dir / "asr_metadata.json"

    jsonl = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    markdown = "".join(
        f"[{timestamp(record['start_s'])} --> {timestamp(record['end_s'])}] {record['text']}\n"
        for record in records
    )
    atomic_write(jsonl_path, jsonl)
    atomic_write(markdown_path, markdown)
    atomic_write(details_path, json.dumps(details, ensure_ascii=False, indent=2) + "\n")

    print(
        json.dumps(
            {
                "status": "ok",
                "transcript_jsonl": str(jsonl_path.resolve()),
                "transcript_markdown": str(markdown_path.resolve()),
                **details,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model", default="small")
    parser.add_argument("--model-cache", type=Path, default=DEFAULT_MODEL_CACHE)
    parser.add_argument("--language", default="zh")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument(
        "--vad-filter",
        action="store_true",
        help="enable Silero VAD; off by default for continuous speech",
    )
    parser.add_argument(
        "--hotwords",
        help="space-separated terms known from the title or user-provided context",
    )
    parser.add_argument(
        "--no-cpu-fallback",
        action="store_false",
        dest="cpu_fallback",
        help="fail instead of retrying with CPU int8 when the CUDA runtime is unavailable",
    )
    parser.set_defaults(cpu_fallback=True)
    return parser


if __name__ == "__main__":
    raise SystemExit(transcribe(build_parser().parse_args()))
