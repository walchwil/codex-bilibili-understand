$ErrorActionPreference = "Stop"

$pluginRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$doctor = Join-Path $pluginRoot "skills\bilibili-understand\scripts\doctor.py"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv is required. Install it with: winget install --id=astral-sh.uv -e"
}

uv sync --project $pluginRoot --locked
uv run --project $pluginRoot --locked python $doctor
