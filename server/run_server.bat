@echo off
REM ============================================================
REM  SIH25005 server - one-click start for demo / integration
REM  Starts MongoDB (portable) if needed, then the API on ALL
REM  network interfaces so Person 1's phone can reach it over
REM  the same hotspot/WiFi.
REM ============================================================

REM --- firewall: without this rule the phone TIMES OUT even though
REM --- everything runs. Check it and shout if it is missing.
netsh advfirewall firewall show rule name="SIH25005 server" >nul 2>&1
if errorlevel 1 (
    echo ************************************************************
    echo  WARNING: firewall rule missing - the PHONE WILL TIME OUT.
    echo  Run this ONCE in an ADMIN command prompt, then relaunch:
    echo.
    echo  netsh advfirewall firewall add rule name="SIH25005 server" dir=in action=allow protocol=TCP localport=8000
    echo ************************************************************
    echo.
)

REM --- MongoDB: create dbpath if missing, start if not running,
REM --- then VERIFY the port actually answers before continuing.
if not exist "D:\mongodb\data" mkdir "D:\mongodb\data"
tasklist /FI "IMAGENAME eq mongod.exe" 2>nul | find /I "mongod.exe" >nul
if errorlevel 1 (
    echo Starting MongoDB...
    start "" /B "D:\mongodb-windows-x86_64-8.3.2\mongodb-win32-x86_64-windows-8.3.2\bin\mongod.exe" --dbpath D:\mongodb\data --port 27017
    timeout /t 3 /nobreak >nul
) else (
    echo MongoDB already running.
)
powershell -NoProfile -Command "exit (1 - [int](Test-NetConnection 127.0.0.1 -Port 27017 -InformationLevel Quiet -WarningAction SilentlyContinue))"
if errorlevel 1 (
    echo ERROR: MongoDB is not answering on port 27017. Check D:\mongodb\data permissions/disk.
    pause
    exit /b 1
)

cd /d "%~dp0"

echo.
echo Your laptop's addresses (give Person 1 the hotspot one):
powershell -NoProfile -Command "Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -ne '127.0.0.1' -and $_.IPAddress -notlike '169.*' } | ForEach-Object { '   http://' + $_.IPAddress + ':8000   (' + $_.InterfaceAlias + ')' }"
echo.
echo Interactive API docs: add /docs to any address above.
echo If the phone times out: it is the firewall rule above.
echo The ML pipeline (ml/) is re-checked every 30s - no restart needed
echo when Person 2's code lands.
echo Press Ctrl+C to stop the server. MongoDB keeps running.
echo.

venv\Scripts\python -m uvicorn main:app --host 0.0.0.0 --port 8000

REM keep the window open so a crash/bind error stays readable
pause
