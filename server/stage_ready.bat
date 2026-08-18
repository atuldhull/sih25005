@echo off
REM ============================================================
REM  SIH25005 - ONE COMMAND before going on stage:
REM    1. resets + reseeds the demo story (fresh dates)
REM    2. runs the preflight GO / NO-GO checks
REM    3. starts the server (via run_server.bat)
REM  Safe to re-run any time; it is the stage reset button.
REM ============================================================
cd /d "%~dp0"

REM MongoDB must be up before seeding - reuse run_server's logic
tasklist /FI "IMAGENAME eq mongod.exe" 2>nul | find /I "mongod.exe" >nul
if errorlevel 1 (
    echo Starting MongoDB...
    if not exist "D:\mongodb\data" mkdir "D:\mongodb\data"
    start "" /B "D:\mongodb-windows-x86_64-8.3.2\mongodb-win32-x86_64-windows-8.3.2\bin\mongod.exe" --dbpath D:\mongodb\data --port 27017
    timeout /t 3 /nobreak >nul
)

echo.
echo ===== 1/3 stage reset: reseeding the demo story =====
venv\Scripts\python demo_seed.py
if errorlevel 1 (
    echo SEEDING FAILED - fix the error above before the demo.
    pause
    exit /b 1
)

echo.
echo ===== 2/3 preflight checks =====
venv\Scripts\python preflight.py
if errorlevel 1 (
    echo.
    echo NO-GO - fix the [FAIL] lines above, then run this again.
    pause
    exit /b 1
)

echo.
echo ===== 3/3 starting the server =====
call run_server.bat
