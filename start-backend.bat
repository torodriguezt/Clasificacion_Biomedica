@echo off
echo ==================================================
echo   🧠 Clasificador de Artículos Biomédicos con IA
echo ==================================================
echo.
echo Iniciando Backend (FastAPI)...
echo Puerto: http://localhost:8000
echo.

cd /d "%~dp0backend"
py -3.11 main.py

pause
