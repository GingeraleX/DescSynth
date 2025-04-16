@echo off
echo.
echo ========================================
echo   Descriptor Synth - Windows Launcher
echo ========================================
echo.

REM --- Check Python version ---
for /f "tokens=2 delims==." %%A in ('py -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"') do set PYVER=%%A
if %PYVER% LSS 3 (
    echo Python 3.10 or higher is required.
    pause
    exit /b
)

REM --- Install required packages ---
echo Installing required Python packages...
py -m pip install --quiet --upgrade pip
py -m pip install --quiet numpy scipy matplotlib librosa soundfile sounddevice

REM --- Check for script.py ---
if not exist script.py (
    echo Could not find script.py in the current folder.
    pause
    exit /b
)

REM --- Run the main script ---
echo Launching GUI...
py script.py
if errorlevel 1 (
    echo script.py exited with an error.
    pause
    exit /b
)

echo.
echo Done.
pause
