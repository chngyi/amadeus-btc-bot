@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
rem Restart her whenever she stops (crash, lost connection at startup). To stop for good:
rem Ctrl+C, then answer Y to "Terminate batch job (Y/N)?".
:loop
python bot.py
echo [%date% %time%] Amadeus stopped (exit code %errorlevel%) - restarting in 10s. Ctrl+C then Y to quit.
rem ping instead of timeout: timeout fails when started without a keyboard (e.g. from Task Scheduler).
ping -n 11 127.0.0.1 >nul
goto loop
