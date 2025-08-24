@echo off
echo ==================================================
echo   🔧 Scripts - Clasificador Biomédico
echo ==================================================
echo.
echo Herramientas disponibles:
echo   • preprocess    - Preprocesar datos médicos
echo   • train         - Entrenar modelo
echo   • evaluate      - Evaluar modelo
echo   • visualize     - Generar visualizaciones
echo   • augment       - Aumentar datos
echo   • device-info   - Información del sistema
echo.
echo Uso: start-scripts.bat [comando] [argumentos]
echo Ejemplo: start-scripts.bat device-info
echo.

if "%1"=="" (
    echo Mostrando ayuda...
    py -3.11 -m scripts --help
) else (
    echo Ejecutando: %*
    py -3.11 -m scripts %*
)

echo.
pause
