@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  py -3.13 -m venv .venv
  .venv\Scripts\python.exe -m pip install -r requirements.txt
)
if errorlevel 1 goto end
if not exist "data\astronaut.npy" .venv\Scripts\python.exe -m experiments.prepare_data
if not exist "configs\frozen_params.yaml" .venv\Scripts\python.exe -m experiments.tune
.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
:end
pause
