@echo off
rem Start PPG2ECG Studio.  run.bat --console  keeps a console window open for error messages.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo PPG2ECG Studio is not set up yet: double-click setup.bat first.
  pause
  exit /b 1
)
if /i "%~1"=="--console" (
  ".venv\Scripts\python.exe" -m studio %2 %3 %4
  pause
  exit /b
)
start "" ".venv\Scripts\pythonw.exe" -m studio %*
