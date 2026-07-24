@echo off
REM ============================================================
REM  Markowitz Portfolio Optimizer - launcher
REM  Double-click this file to start the Streamlit app.
REM  It runs the local server and opens the app in your browser.
REM ============================================================

title Markowitz Portfolio Optimizer

REM Switch to the folder this .bat file lives in (handles spaces in path).
cd /d "%~dp0"

REM Make sure the virtual environment exists.
if not exist "venv\Scripts\python.exe" (
    echo.
    echo [ERROR] Virtual environment not found at venv\Scripts\python.exe
    echo Create it first with:  python -m venv venv
    echo and install dependencies:  venv\Scripts\python -m pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

echo.
echo Starting Markowitz Portfolio Optimizer ...
echo The app will open in your browser at http://localhost:8501
echo Keep this window open. Close it (or press Ctrl+C) to stop the app.
echo.

REM Run Streamlit through the venv's Python. Force a browser to open
REM (config.toml sets headless=true, which would otherwise open nothing).
"venv\Scripts\python.exe" -m streamlit run app.py --server.headless false --server.port 8501

REM If Streamlit exits (error or manual stop), keep the window open so the
REM message stays readable.
echo.
echo App stopped.
pause
