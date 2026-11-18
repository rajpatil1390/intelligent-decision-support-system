# Start the IDSS API (FastAPI + uvicorn, auto-reload) on http://localhost:8000 using the project venv.
# Usage, from anywhere:  powershell -ExecutionPolicy Bypass -File scripts\run_api.ps1 [-Port 8000]
# Interactive docs: http://localhost:8000/docs
param(
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    Write-Error "Virtual environment not found at $Python. Create it and run: .venv\Scripts\python.exe -m pip install -r requirements.txt"
    exit 1
}

Set-Location $ProjectRoot
Write-Host "IDSS API on http://localhost:$Port  (docs: http://localhost:$Port/docs)  - press Ctrl+C to stop"
& $Python -m uvicorn api.main:app --reload --host 127.0.0.1 --port $Port --reload-dir api --reload-dir idss --reload-dir configs
exit $LASTEXITCODE
