@echo off
REM Run the API Gateway locally (Windows)
cd /d "%~dp0\.."
uvicorn api.main:app --port 8000
