@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    python -m venv .venv
    if errorlevel 1 goto fail
)

call ".venv\Scripts\activate.bat"
python -m pip install -r requirements.txt
if errorlevel 1 goto fail

python -m uvicorn main:app --host 0.0.0.0 --port 8000
goto end

:fail
echo.
echo Server setup failed. Check that Python is installed and available as "python".
pause

:end
