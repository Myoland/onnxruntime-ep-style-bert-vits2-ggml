$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot
uv run python scripts/build_runtime_bundle.py @args
