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
REM journal recovery after a crash can take longer than 3s - retry
set /a _tries=0
:mongo_wait
powershell -NoProfile -Command "exit (1 - [int](Test-NetConnection 127.0.0.1 -Port 27017 -InformationLevel Quiet -WarningAction SilentlyContinue))"
if not errorlevel 1 goto mongo_ok
set /a _tries+=1
if %_tries% geq 10 (
    echo ERROR: MongoDB is not answering on port 27017 after ~20s. Check D:\mongodb\data permissions/disk.
    pause
    exit /b 1
)
echo   waiting for MongoDB ^(%_tries%/10^)...
timeout /t 2 /nobreak >nul
goto mongo_wait
:mongo_ok
echo MongoDB is answering.

cd /d "%~dp0"

REM --- locate the Python environment -------------------------------------
REM  This folder has no venv of its own: the working environment (torch,
REM  ultralytics, pymongo, whisper) lives in the sibling checkout. Hard-coding
REM  "venv\Scripts\python" meant these scripts failed the moment they were run
REM  from the integration repo, which is the only place they matter now.
REM  Prefer a local venv if someone creates one; fall back to the real one.
set "PY=%~dp0venv\Scripts\python.exe"
if not exist "%PY%" set "PY=D:\sih25005\server\venv\Scripts\python.exe"
if not exist "%PY%" (
    echo ERROR: no Python environment found.
    echo   looked in %~dp0venv\Scripts\python.exe
    echo   and in    D:\sih25005\server\venv\Scripts\python.exe
    pause
    exit /b 1
)
echo Using Python: %PY%



REM mid-demo recovery guard: if the server is already up, do not bind
REM a second instance on the same port
powershell -NoProfile -Command "try { Invoke-WebRequest -Uri http://127.0.0.1:8000/ping -UseBasicParsing -TimeoutSec 3 | Out-Null; exit 0 } catch { exit 1 }"
if not errorlevel 1 (
    echo Server is ALREADY RUNNING on port 8000 - not starting a second one.
    echo MongoDB has been checked/restarted, so you are done.
    pause
    exit /b 0
)

echo.
echo Your laptop's addresses (give Person 1 the hotspot one):
powershell -NoProfile -Command "Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -ne '127.0.0.1' -and $_.IPAddress -notlike '169.*' } | ForEach-Object { '   http://' + $_.IPAddress + ':8000   (' + $_.InterfaceAlias + ')' }"
echo.
echo Interactive API docs: add /docs to any address above.
echo Chat interface:       add /chat-ui to any address above.
echo If the phone times out: it is the firewall rule above.
echo.
echo HARD RULE: connect this laptop ONLY to the team's own hotspot -
echo NEVER to open venue WiFi (no auth on the API; strangers could
echo burn the free-tier LLM quota and read the demo data).
echo VOICE NOTE: the mic in /chat-ui works on http://localhost:8000
echo on THIS laptop; phones over plain http get text chat only
echo (browsers require HTTPS for microphones).
echo The ML pipeline (ml/) is re-checked every 30s, and keys.json is
echo hot-reloaded - no restart needed for either.
echo Press Ctrl+C to stop the server. MongoDB keeps running.
echo.

"%PY%" -m uvicorn main:app --host 0.0.0.0 --port 8000

REM keep the window open so a crash/bind error stays readable
pause
