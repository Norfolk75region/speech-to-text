@echo off
rem Zapusk raspoznavaniya rechi.
rem Mozhno dvazhdy shchelknut po etomu failu ili peretashchit na nego audiofaily.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1"
    if errorlevel 1 (
        echo.
        pause
        exit /b 1
    )
)
".venv\Scripts\python.exe" main.py %*
echo.
pause
