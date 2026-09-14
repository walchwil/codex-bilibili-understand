#!/usr/bin/env sh
set -eu

plugin_root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
doctor="$plugin_root/skills/bilibili-understand/scripts/doctor.py"

if ! command -v uv >/dev/null 2>&1; then
  printf '%s\n' 'uv is required. Install it from https://docs.astral.sh/uv/getting-started/installation/' >&2
  exit 1
fi

uv sync --project "$plugin_root" --locked
uv run --project "$plugin_root" --locked python "$doctor"
