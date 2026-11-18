@echo off
rem Run the tests.  Optional: set BIDMC_DIR=...\bidmc_csv  to also check the numbers against Track 1.
cd /d "%~dp0"
".venv\Scripts\python.exe" -m pytest -q tests %*
if "%~1"=="" pause
