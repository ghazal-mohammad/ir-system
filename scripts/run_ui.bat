@echo off
REM Run the IR system UI locally (Windows)
REM Data files must be in ir-system\data (or set IR_DATA_DIR)
cd /d "%~dp0\.."
streamlit run ui/app.py
