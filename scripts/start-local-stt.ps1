param(
    [string]$RepoRoot = (Split-Path -Parent $PSScriptRoot),
    [int]$Port = 8000,
    [switch]$LaunchBrowser
)

Set-Location $RepoRoot

$venvScripts = Join-Path (Join-Path $RepoRoot ".venv") "Scripts"
$pythonExe = Join-Path $venvScripts "python.exe"

if (-not (Test-Path $pythonExe)) {
    Write-Error "Python interpreter not found at $pythonExe. Activate the virtual environment or install dependencies first."
    exit 1
}

$daemonScript = Join-Path (Join-Path $RepoRoot "scripts") "start-hotkey-daemon.ps1"
if (-not (Test-Path $daemonScript)) {
    Write-Error "Missing daemon launcher script at $daemonScript."
    exit 1
}

$env:PYTHONPATH = $RepoRoot

$uvicornArgs = @(
    "-m","uvicorn",
    "backend.main:app",
    "--host","0.0.0.0",
    "--port",$Port
)

Write-Host "Launching FastAPI server and hotkey daemon..." -ForegroundColor Cyan

$uvicornProcess = Start-Process -FilePath $pythonExe `
    -ArgumentList $uvicornArgs `
    -WorkingDirectory $RepoRoot `
    -WindowStyle Minimized `
    -PassThru

Write-Host "- FastAPI server PID: $($uvicornProcess.Id) (http://localhost:$Port/ui/)"

$daemonProcess = Start-Process -FilePath "powershell.exe" `
    -ArgumentList "-ExecutionPolicy","Bypass","-File",$daemonScript `
    -WorkingDirectory $RepoRoot `
    -WindowStyle Minimized `
    -PassThru

Write-Host "- Hotkey daemon PID: $($daemonProcess.Id)"

if ($LaunchBrowser.IsPresent) {
    Start-Sleep -Seconds 2
    Start-Process "http://localhost:$Port/ui/"
    Write-Host "- Browser opened at http://localhost:$Port/ui/"
}

Write-Host ""; Write-Host "Press Ctrl+C to stop all services." -ForegroundColor Yellow

try {
    Wait-Process -Id @($uvicornProcess.Id, $daemonProcess.Id)
}
catch {
    Write-Host "Shutdown requested..." -ForegroundColor Yellow
}
finally {
    foreach ($proc in @($uvicornProcess, $daemonProcess)) {
        if ($proc -and -not $proc.HasExited) {
            try {
                Stop-Process -Id $proc.Id -Force -ErrorAction Stop
            }
            catch {}
        }
    }
    Write-Host "Services stopped." -ForegroundColor Cyan
    Write-Host "Press Enter to close this window." -ForegroundColor Gray
    try { [void](Read-Host) } catch {}
}
