@echo off
cd /d "%~dp0"
echo.
echo ============================================
echo   Titanic DS App - Local Server
echo ============================================
echo.

REM Check if Node.js is installed
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Node.js is not installed or not in PATH.
    echo.
    echo Please install Node.js from: https://nodejs.org
    echo Download the LTS version, install it, then try again.
    echo.
    pause
    exit /b 1
)

echo Node.js found:
node --version
echo.
echo Starting server... 
echo Once you see the checkmark, open your browser and go to:
echo http://localhost:3131
echo.
node server.js
echo.
echo Server stopped.
pause
