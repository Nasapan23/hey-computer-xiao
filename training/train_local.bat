@echo off
setlocal

cd /d "%~dp0\.."

if not exist "training\.venv\Scripts\python.exe" (
    python -m venv training\.venv
    if errorlevel 1 goto fail
)

call "training\.venv\Scripts\activate.bat"
python -m pip install --upgrade pip
if errorlevel 1 goto fail

python -m pip install -r training\requirements.txt
if errorlevel 1 goto fail

python training\train_local_wake_word.py
if errorlevel 1 goto fail

if not exist "esp32_tinyml_wake_word\model_data.h" goto missing_model

echo.
echo Training finished.
pause
goto end

:missing_model
echo.
echo Training ended, but esp32_tinyml_wake_word\model_data.h was not created.
echo The TFLite export probably failed. Read the error above.
pause
goto end

:fail
echo.
echo Training failed. Read the error above.
pause

:end
