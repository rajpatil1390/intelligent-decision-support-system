# Start the IDSS Streamlit UI on http://localhost:8501 using the project venv.
# Usage, from anywhere:  powershell -ExecutionPolicy Bypass -File scripts\run_ui.ps1 [-Port 8501] [-ApiUrl http://localhost:8000] [-Headless]
# The UI talks only to the backend (start it first with scripts\run_api.ps1); its address comes from IDSS_API_URL.
param(
    [int]$Port = 8501,
    [string]$ApiUrl = "",
    [switch]$Headless
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    Write-Error "Virtual environment not found at $Python. Create it and run: .venv\Scripts\python.exe -m pip install -r requirements.txt"
    exit 1
}

Set-Location $ProjectRoot  # so Streamlit picks up .streamlit\config.toml
if ($ApiUrl) { $env:IDSS_API_URL = $ApiUrl }
$Backend = if ($env:IDSS_API_URL) { $env:IDSS_API_URL } else { "http://localhost:8000" }
Write-Host "IDSS UI on http://localhost:$Port  (backend: $Backend)  - press Ctrl+C to stop"
$StreamlitArgs = @("-m", "streamlit", "run", (Join-Path "ui" "app.py"), "--server.port", "$Port")
if ($Headless) { $StreamlitArgs += @("--server.headless", "true") }
& $Python @StreamlitArgs
exit $LASTEXITCODE
