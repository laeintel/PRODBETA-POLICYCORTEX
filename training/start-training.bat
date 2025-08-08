@echo off
REM PolicyCortex AI Training Launcher for Windows

echo =====================================
echo PolicyCortex Domain Expert AI Training
echo =====================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.11+ from python.org
    pause
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo.
echo Installing training dependencies...
pip install -q --upgrade pip
pip install -q -r requirements.txt

REM Run training setup
echo.
echo Starting PolicyCortex AI Training Setup...
echo.
python start_training.py %*

echo.
echo =====================================
echo Training setup complete!
echo =====================================
pause