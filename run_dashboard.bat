@echo off
cd /d "%~dp0"
echo Starting Ensemble Learning Dashboard...
streamlit run dashboard\app.py
pause
