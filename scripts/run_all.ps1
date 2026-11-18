# Start the whole IDSS demo: the API in a new PowerShell window, then the Streamlit UI in this one.
# Usage, from anywhere:  powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1 [-ApiPort 8000] [-UiPort 8501]
# Stop the UI with Ctrl+C here; close the API window (or press Ctrl+C in it) to stop the API.
param(
    [int]$ApiPort = 8000,
    [int]$UiPort = 8501
)

$ErrorActionPreference = "Stop"
$ApiScript = Join-Path $PSScriptRoot "run_api.ps1"
$UiScript = Join-Path $PSScriptRoot "run_ui.ps1"

Write-Host "Starting the IDSS API in a new window on http://localhost:$ApiPort ..."
# One quoted argument string: the project path contains spaces.
Start-Process powershell -ArgumentList "-NoExit -ExecutionPolicy Bypass -File `"$ApiScript`" -Port $ApiPort"

$HealthUrl = "http://127.0.0.1:$ApiPort/health"
$Deadline = (Get-Date).AddSeconds(60)
$Up = $false
while (-not $Up -and (Get-Date) -lt $Deadline) {
    try {
        Invoke-RestMethod -Uri $HealthUrl -TimeoutSec 2 | Out-Null
        $Up = $true
    } catch {
        Start-Sleep -Milliseconds 700
    }
}
if ($Up) {
    Write-Host "API is up."
} else {
    Write-Warning "The API is not answering at $HealthUrl yet; the UI shows 'Backend not running' until it is."
}

& $UiScript -Port $UiPort -ApiUrl "http://localhost:$ApiPort"
exit $LASTEXITCODE
