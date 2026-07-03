@echo off
setlocal
cd /d "%~dp0"
uv run python scripts\build_runtime_bundle.py %*
