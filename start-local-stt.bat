@echo off
setlocal
cd /d "%~dp0"
set PYTHONPATH=%~dp0

echo Starting voice dictation daemon...
echo Hold AltGr+P to dictate, release to transcribe to clipboard.
echo Close this window to stop the service.
echo.

if exist "%~dp0.venv\Scripts\python.exe" (
    "%~dp0.venv\Scripts\python.exe" -m backend.hotkey_daemon
) else (
    python -m backend.hotkey_daemon
)

echo.
echo Voice dictation service stopped.
pause
