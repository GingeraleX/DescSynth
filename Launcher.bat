@echo off
setlocal EnableDelayedExpansion

:: Enable ANSI colors on Windows 10+
for /f "tokens=2 delims=[]" %%i in ('ver') do set VERSION=%%i
if %VERSION:~0,2% GEQ 10 (
    reg add HKCU\Console /v VirtualTerminalLevel /t REG_DWORD /d 1 /f >nul 2>&1
)

:: === Style ===
set "RED=[31m"
set "GRN=[32m"
set "YLW=[33m"
set "NC=[0m"

echo.
echo %GRN%Descriptor Synth Launcher%NC%
echo ----------------------------------------

:: --- Check Python ---
where py >nul 2>nul
if errorlevel 1 (
    echo %RED%Python not found.%NC%
    echo %YLW%Installing Python 3.10...%NC%
    powershell -Command "Invoke-WebRequest https://www.python.org/ftp/python/3.10.10/python-3.10.10-amd64.exe -OutFile python_installer.exe"
    python_installer.exe /quiet InstallAllUsers=1 PrependPath=1 Include_test=0
    del python_installer.exe
)

:: --- Version Check ---
py -c "import sys; exit(0) if sys.version_info >= (3,10) else exit(1)"
if errorlevel 1 (
    echo %RED%Python 3.10+ required. Please reinstall manually.%NC%
    pause
    exit /b
)
echo %GRN%Python 3.10+ detected%NC%

:: --- pip check ---
py -m ensurepip >nul 2>&1
py -m pip install --upgrade pip >nul 2>&1
echo %GRN%pip available%NC%

:: --- Check packages ---
set "missing="
for %%m in (numpy scipy matplotlib librosa soundfile sounddevice) do (
    py -c "import %%m" 2>nul || set missing=!missing! %%m
)

if defined missing (
    echo %YLW%Installing: %missing%%NC%
    py -m pip install %missing%
) else (
    echo %GRN%All dependencies already installed%NC%
)

:: --- Run GUI ---
if not exist ".\Code\GUI.py" (
    echo %RED%GUI.py not found in \Code%NC%
    pause
    exit /b
)
echo %GRN%Launching GUI...%NC%
py .\Code\GUI.py
