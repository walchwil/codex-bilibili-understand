"""Check the isolated runtime without touching the network or video cache."""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any


def check_import(name: str, package: str, required: bool = True) -> dict[str, Any]:
    try:
        module = importlib.import_module(package)
    except Exception as exc:  # pragma: no cover - depends on local environment
        return {
            "name": name,
            "status": "error" if required else "warning",
            "required": required,
            "detail": f"{type(exc).__name__}: {exc}",
        }
    version = getattr(module, "__version__", None)
    return {
        "name": name,
        "status": "ok",
        "required": required,
        "detail": version or "imported",
    }


def collect() -> dict[str, Any]:
    checks: list[dict[str, Any]] = [
        {
            "name": "uv",
            "status": "ok" if shutil.which("uv") else "error",
            "required": True,
            "detail": shutil.which("uv") or "not found on PATH",
        },
        {
            "name": "python",
            "status": "ok" if sys.version_info >= (3, 10) else "error",
            "required": True,
            "detail": sys.version.split()[0],
        },
        {
            "name": "isolated_environment",
            "status": "ok" if sys.prefix != sys.base_prefix else "error",
            "required": True,
            "detail": sys.prefix,
        },
        check_import("yt-dlp", "yt_dlp"),
        check_import("faster-whisper", "faster_whisper"),
        check_import("ctranslate2", "ctranslate2"),
        {
            "name": "ffmpeg",
            "status": "ok" if shutil.which("ffmpeg") else "warning",
            "required": False,
            "detail": shutil.which("ffmpeg") or "not found (optional for the current m4a path)",
        },
        {
            "name": "ffprobe",
            "status": "ok" if shutil.which("ffprobe") else "warning",
            "required": False,
            "detail": shutil.which("ffprobe") or "not found (optional)",
        },
    ]

    cuda_devices = None
    cuda_detail = "CPU fallback available"
    try:
        import ctranslate2

        cuda_devices = ctranslate2.get_cuda_device_count()
        cuda_detail = f"{cuda_devices} CUDA device(s) visible"
    except Exception as exc:  # pragma: no cover - hardware/runtime dependent
        cuda_detail = f"CUDA unavailable; CPU fallback available ({type(exc).__name__})"
    checks.append(
        {
            "name": "cuda",
            "status": "ok" if cuda_devices else "warning",
            "required": False,
            "detail": cuda_detail,
        }
    )

    errors = [item for item in checks if item["status"] == "error"]
    return {
        "status": "error" if errors else "ok",
        "python_executable": sys.executable,
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()
    payload = collect()
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    else:
        print(f"doctor: {payload['status']} ({payload['python_executable']})")
        for item in payload["checks"]:
            print(f"- {item['status']}: {item['name']} — {item['detail']}")
    return 1 if payload["status"] == "error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
