param(
    [string]$RepoRoot = (Split-Path -Parent $PSScriptRoot)
)

Set-Location $RepoRoot

$venvScripts = Join-Path $RepoRoot ".venv" | Join-Path -ChildPath "Scripts"
$pythonExe = Join-Path $venvScripts "python.exe"

if (-not (Test-Path $pythonExe)) {
    Write-Error "Python interpreter not found at $pythonExe. Activate the virtual environment or install dependencies first."
    exit 1
}

$env:PYTHONPATH = $RepoRoot

& $pythonExe -m backend.hotkey_daemon
