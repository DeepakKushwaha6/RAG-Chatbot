@echo off
REM Start the DocuMind RAG chatbot.
cd /d "%~dp0"
echo.
echo   DocuMind RAG chatbot  ->  http://127.0.0.1:8000
echo.
python run.py
pause
