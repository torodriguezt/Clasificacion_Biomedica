@echo off
echo ==================================================
echo   🧠 Clasificador de Artículos Biomédicos con IA
echo ==================================================
echo.
echo Iniciando Backend (FastAPI)...
echo Puerto: http://localhost:8000
echo.

cd /d "%~dp0backend"
py -3.11 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

pause
