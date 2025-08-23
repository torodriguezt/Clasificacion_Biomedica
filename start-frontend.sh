#!/bin/bash

echo "=================================================="
echo "  🎯 Frontend - Clasificador Biomédico"
echo "=================================================="
echo ""
echo "Iniciando Frontend (Next.js)..."
echo "Puerto: http://localhost:3000"
echo ""

cd "$(dirname "$0")/frontend"
npm run dev
