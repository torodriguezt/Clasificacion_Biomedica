@echo off
echo ==================================================
echo   🚀 Configuración Inicial - Clasificador Biomédico
echo ==================================================
echo.

echo [1/4] Verificando Python 3.11...
py -3.11 --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python 3.11 no encontrado. Por favor instálalo primero.
    pause
    exit /b 1
)
echo ✅ Python 3.11 encontrado

echo.
echo [2/4] Instalando todas las dependencias del proyecto...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ❌ Error instalando dependencias del proyecto
    pause
    exit /b 1
)
echo ✅ Todas las dependencias instaladas correctamente
echo.
echo [3/4] Verificando Node.js...
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Node.js no encontrado. Por favor instálalo primero.
    pause
    exit /b 1
)
echo ✅ Node.js encontrado

echo.
echo [4/4] Instalando dependencias del frontend...
cd frontend
npm install --legacy-peer-deps
if %errorlevel% neq 0 (
    echo ❌ Error instalando dependencias del frontend
    pause
    exit /b 1
)
echo ✅ Dependencias del frontend instaladas

cd ..
echo.
echo ================================================== 
echo   ✅ ¡Configuración completada exitosamente!
echo ==================================================
echo.
echo Para ejecutar la aplicación:
echo   • Backend:  start-backend.bat
echo   • Frontend: start-frontend.bat
echo.
echo URLs:
echo   • Frontend: http://localhost:3000
echo   • Backend:  http://localhost:8000
echo   • Docs API: http://localhost:8000/docs
echo.
pause
