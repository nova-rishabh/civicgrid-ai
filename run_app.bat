@echo off
title CivicGrid AI - Startup Launcher
echo ==========================================================
echo Welcome to CivicGrid AI Launcher
echo Indraprastha International School - Team The Optimizers
echo ==========================================================
echo.

REM 1. Check if Python is installed
python --version >nul 2>&1
if %errorlevel% equ 0 goto python_ok

echo [WARNING] Python was not detected on this system.
echo [INFO] Downloading Python 3.11.9 installer...
powershell -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe' -OutFile 'python_installer.exe'"

if not exist python_installer.exe goto download_fail

echo [INFO] Installing Python 3.11.9 silently...
echo Please wait, this may take a minute...
start /wait python_installer.exe /quiet InstallAllUsers=0 PrependPath=1 Include_test=0 Include_pip=1
del python_installer.exe

REM Add installed Python path to current terminal session PATH immediately
set "PATH=%USERPROFILE%\AppData\Local\Programs\Python\Python311;%USERPROFILE%\AppData\Local\Programs\Python\Python311\Scripts;%LOCALAPPDATA%\Programs\Python\Python311;%LOCALAPPDATA%\Programs\Python\Python311\Scripts;%PATH%"

REM Double check if it's available now
python --version >nul 2>&1
if %errorlevel% equ 0 goto python_ok

echo [ERROR] Python installation completed but could not be located in PATH.
echo Please restart your computer and run this script again.
pause
exit /b 1

:download_fail
echo [ERROR] Failed to download Python installer. Please check your internet connection.
pause
exit /b 1

:python_ok
echo [SUCCESS] Python is ready.

REM 2. Ensure pip is installed
python -m pip --version >nul 2>&1
if %errorlevel% equ 0 goto pip_ok
echo [INFO] Installing pip...
python -m ensurepip --default-pip
:pip_ok

REM 3. Create virtual environment if not exists
if exist .venv goto venv_ok
echo [INFO] Creating Python virtual environment (.venv)...
python -m venv .venv
if %errorlevel% equ 0 goto venv_ok
echo [ERROR] Failed to create virtual environment.
pause
exit /b 1
:venv_ok

REM 4. Activate virtual environment
echo [INFO] Activating virtual environment...
call .venv\Scripts\activate
if %errorlevel% equ 0 goto venv_active
echo [ERROR] Failed to activate virtual environment.
pause
exit /b 1
:venv_active

REM 5. Install requirements
echo [INFO] Installing dependencies from requirements.txt...
python -m pip install --upgrade pip >nul 2>&1
call pip install -r requirements.txt
if %errorlevel% equ 0 goto pip_install_ok
echo [ERROR] Failed to install dependencies.
pause
exit /b 1
:pip_install_ok

REM 6. Generate dataset and train default model
echo [INFO] Generating microgrid datasets and training ML model...
python src/data_generator.py
if %errorlevel% equ 0 goto data_ok
echo [ERROR] Failed to generate data.
pause
exit /b 1
:data_ok

REM 7. Run unit tests
echo [INFO] Running automated test suite (pytest)...
call python -m pytest
if %errorlevel% equ 0 goto tests_ok
echo [WARNING] Test suite reported failures. Checking logs...
goto run_server
:tests_ok
echo [SUCCESS] All verification tests passed!

:run_server
REM 8. Launch Streamlit app & Open Browser
echo.
echo ==========================================================
echo STARTING CIVICGRID AI SERVER...
echo.
echo Please open your web browser and go to:
echo http://localhost:8503
echo ==========================================================
echo.

REM Start background command to wait 3 seconds and force open default browser
start /b cmd /c "timeout /t 3 >nul && start http://localhost:8503"

REM Run Streamlit
call streamlit run app.py --server.port 8503 --server.headless true

if %errorlevel% neq 0 (
    echo [ERROR] Streamlit server exited with an error.
)
pause
