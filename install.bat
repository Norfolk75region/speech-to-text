@echo off
rem Ustanovka: skachivaet Python i biblioteki v papku programmy.
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1"
echo.
pause
