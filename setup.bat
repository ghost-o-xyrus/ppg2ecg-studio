@echo off
rem PPG2ECG Studio setup. Double-click to install everything, or run from a terminal with flags:
rem   setup.bat --zhu --capnobase D:\data\capnobase      setup.bat --list      setup.bat --check
setlocal EnableExtensions
cd /d "%~dp0"
title PPG2ECG Studio setup
set "ARGS=%*"
if "%ARGS%"=="" set "ARGS=--all"

rem 1. uv: fetches a private Python 3.11 and the packages (no admin rights needed)
set "UV="
where uv >nul 2>nul && set "UV=uv"
if not defined UV if exist "%~dp0.tools\uv\uv.exe" set "UV=%~dp0.tools\uv\uv.exe"
if not defined UV if exist "%~dp0.tools\uv\bin\uv.exe" set "UV=%~dp0.tools\uv\bin\uv.exe"
if not defined UV (
  echo Installing uv into .tools\uv ...
  powershell -NoProfile -ExecutionPolicy Bypass -Command "$env:UV_INSTALL_DIR='%~dp0.tools\uv'; $env:UV_NO_MODIFY_PATH='1'; irm https://astral.sh/uv/install.ps1 | iex"
)
if not defined UV if exist "%~dp0.tools\uv\uv.exe" set "UV=%~dp0.tools\uv\uv.exe"
if not defined UV if exist "%~dp0.tools\uv\bin\uv.exe" set "UV=%~dp0.tools\uv\bin\uv.exe"
if not defined UV (
  echo Could not install uv. Check the internet connection, or install Python 3.11 and run:
  echo   py -3.11 -m venv .venv  ^&^&  .venv\Scripts\pip install -r requirements.txt  ^&^&  .venv\Scripts\python setup.py --all
  goto :fail
)

rem 2. private environment in .venv
if not exist ".venv\Scripts\python.exe" (
  echo Creating .venv with Python 3.11 ...
  "%UV%" venv --python 3.11 .venv || goto :fail
)
echo Installing packages ...
"%UV%" pip install --python ".venv\Scripts\python.exe" -r requirements.txt || goto :fail

rem 3. models, weights, self-test, shortcuts
".venv\Scripts\python.exe" setup.py %ARGS% || goto :fail
echo.
echo Done. Start the app with run.bat or the "PPG2ECG Studio" shortcut.
if "%~1"=="" pause
exit /b 0

:fail
echo.
echo Setup did not finish - see the messages above.
if "%~1"=="" pause
exit /b 1
