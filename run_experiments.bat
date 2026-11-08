@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  py -3.13 -m venv .venv
  .venv\Scripts\python.exe -m pip install -r requirements.txt
)
if errorlevel 1 goto end
.venv\Scripts\python.exe -m experiments.prepare_data
if errorlevel 1 goto end
if not exist "configs\frozen_params.yaml" .venv\Scripts\python.exe -m experiments.tune
if errorlevel 1 goto end
.venv\Scripts\python.exe -m experiments.benchmark
if errorlevel 1 goto end
.venv\Scripts\python.exe -m experiments.analyze
:end
pause
