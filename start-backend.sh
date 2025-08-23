#!/bin/bash

echo "=================================================="
echo "  🧠 Clasificador de Artículos Biomédicos con IA"
echo "=================================================="
echo ""
echo "Iniciando Backend (FastAPI)..."
echo "Puerto: http://localhost:8000"
echo ""

cd "$(dirname "$0")/backend"
python3.11 main.py
