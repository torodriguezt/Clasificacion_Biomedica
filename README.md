# 🧠 Clasificador de Textos Biomédicos con IA

Un sistema integral de clasificación de textos biomédicos que utiliza ImprovedMedicalBERT para categorizar literatura médica en las categorías: cardiovascular, hepatorenal, neurológica y oncológica.

[DEMO](http://159.65.106.247:3000/)

## 📊 Resultados Globales

| Métrica              | Valor   |
|-----------------------|---------|
| **F1 Score Weighted** | 0.939  |
| **F1 Score Macro**    | 0.949  |
| **F1 Score Micro**    | 0.943  |

El modelo logra un **F1 Score Weighted de 0.9428**, mostrando un rendimiento muy alto y estable en un escenario de clasificación multi-etiqueta desbalanceado.

## 🏗️ Estructura del Proyecto

```
Clasificacion_Biomedica/
├── 📊 data/                    # Almacenamiento de datasets
│   └── challenge_data-18-ago.csv
├── 🎛️ config/                 # Archivos de configuración
│   └── settings.py
├── 🧠 models/                  # Modelos entrenados y artefactos
│   └── trained_model/
│       ├── config.json
│       ├── model.pt
│       ├── mlb.pkl
│       ├── best_threshold.json
│       └── archivos del tokenizer...
├── 📓 notebooks/               # Notebooks de análisis Jupyter
│   └── Medical_Classification_Analysis.ipynb
├── 🔧 scripts/                 # Utilidades Python modulares
│   ├── data_processing.py      # Carga y preprocesamiento de datos
│   ├── visualization.py        # Gráficos y visualización (incluye curvas ROC)
│   ├── model_utils.py          # Arquitecturas de modelos y datasets
│   ├── training_utils.py       # Entrenamiento y evaluación
│   ├── evaluation_utils.py     # Métricas integrales
│   └── text_augmentation.py    # Aumento de datos
├── 🚀 backend/                 # Servicio web FastAPI
│   ├── main.py                 # Endpoints de la API
│   ├── improved_medical_bert.py # Implementación del modelo personalizado
│   ├── requirements.txt
│   └── Dockerfile
├── 🌐 frontend/                # Interfaz web Next.js
│   ├── app/
│   │   ├── page.tsx           # Página principal de clasificación
│   │   ├── rendimiento/       # Página de métricas del modelo
│   │   └── layout.tsx
│   ├── components/            # Componentes UI reutilizables
│   ├── package.json
│   └── Dockerfile
├── 📚 docs/                    # Documentación
├── ⚙️ setup.bat                # Configuración del entorno
├── 🎯 start-backend.bat        # Lanzador del backend
├── 🎯 start-frontend.bat       # Lanzador del frontend
├── 🐳 docker-compose.yml       # Orquestación de contenedores
├── 🔧 .flake8                  # Configuración de linting
├── 🔧 pyproject.toml           # Configuración de herramientas Python
└── 📋 README.md               # Este archivo
```

## 🚀 Inicio Rápido - Configuración Reproducible

Este proyecto está diseñado para **reproducibilidad completa** usando archivos batch de Windows. Sigue estos pasos:

### 1. Configuración del Entorno
```cmd
# Clona el repositorio
git clone <repository-url>
cd Clasificacion_Biomedica

# Ejecuta la configuración automatizada
setup.bat
```

El archivo `setup.bat` realizará:
- Crear y activar un entorno conda
- Instalar todas las dependencias de Python
- Configurar el entorno de desarrollo
- Verificar las instalaciones

### 2. Servicio Backend
```cmd
# Inicia el servidor FastAPI backend
start-backend.bat
```

El archivo `start-backend.bat` realizará:
- Activar el entorno conda
- Navegar al directorio backend
- Iniciar el servidor FastAPI en `http://localhost:8000`
- Cargar el modelo ImprovedMedicalBERT entrenado
- Habilitar CORS para comunicación con el frontend

### 3. Interfaz Frontend
```cmd
# Inicia el frontend Next.js (en una nueva terminal)
start-frontend.bat
```

El archivo `start-frontend.bat` realizará:
- Navegar al directorio frontend
- Instalar dependencias de Node.js (si es necesario)
- Iniciar el servidor de desarrollo en `http://localhost:3000`
- Habilitar hot reloading para desarrollo

### 4. Despliegue con Docker (Alternativa)
```cmd
# Para despliegue containerizado
docker-compose up --build
```

## 🔬 Arquitectura del Modelo

### ImprovedMedicalBERT
- **Modelo Base**: microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract
- **Arquitectura**: Transformer con attention pooling
- **Etiquetas**: 4 categorías médicas (clasificación multi-etiqueta)
- **Entrada**: Títulos médicos + abstracts
- **Salida**: Puntuaciones de probabilidad para cada categoría

### Características Clave:
- Mecanismo de attention pooling personalizado
- Regularización con dropout (0.3)
- Ponderación de clases positivas para datos desbalanceados
- Umbrales optimizados por categoría
- Soporte para aceleración GPU

## 📈 Métricas de Rendimiento

El sistema rastrea métricas integrales de clasificación multi-etiqueta:

- **F1 Score**: Promedios Macro, Micro y Ponderado
- **Precisión/Recall**: Por clase y promediados
- **ROC-AUC**: Curvas individuales y macro-promediadas
- **Hamming Loss**: Métrica específica para multi-etiqueta
- **Exact Match Ratio**: Todas las etiquetas correctas
- **Average Precision**: Área bajo la curva PR

## 🚀 Uso de la API

### Verificación de Salud
```bash
curl http://localhost:8000/health
```

### Clasificación Individual
```bash
curl -X POST "http://localhost:8000/classify" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Análisis de Arritmias Cardíacas",
    "abstract": "Este estudio examina arritmias cardíacas en pacientes..."
  }'
```

### Clasificación por Lotes
```bash
curl -X POST "http://localhost:8000/batch_classify" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      {"title": "Estudio 1", "abstract": "Abstract 1..."},
      {"title": "Estudio 2", "abstract": "Abstract 2..."}
    ]
  }'
```

## 🐳 Soporte Docker

### Contenedor Backend
```dockerfile
FROM python:3.11.5-slim

# Establecer directorio de trabajo
WORKDIR /app

# Copiar archivos de dependencias primero (para cache de Docker)
COPY backend/requirements.txt .

# Instalar dependencias del sistema y Python
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -r requirements.txt

# Copiar código del backend
COPY backend/ .

# Copiar modelos entrenados
COPY models/ ./models/

# Crear usuario no-root para seguridad
RUN adduser --disabled-password --gecos '' appuser && \
    chown -R appuser:appuser /app
USER appuser

# Exponer puerto
EXPOSE 8000

# Comando para ejecutar la aplicación
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Contenedor Frontend
```dockerfile
FROM node:18-alpine

# Establecer directorio de trabajo
WORKDIR /app

# Copiar archivos de dependencias primero (para cache de Docker)
COPY frontend/package*.json ./

# Instalar dependencias
RUN npm ci --only=production

# Copiar código fuente
COPY frontend/ .

# Construir la aplicación
RUN npm run build

# Crear usuario no-root para seguridad
RUN addgroup -g 1001 -S nodejs && \
    adduser -S nextjs -u 1001
USER nextjs

# Exponer puerto
EXPOSE 3000

# Comando para ejecutar la aplicación
CMD ["npm", "start"]
```

### Docker Compose
```yaml
version: '3.8'

services:
  backend:
    build:
      context: .
      dockerfile: backend/Dockerfile
    ports:
      - "8000:8000"
    environment:
      - PYTHONPATH=/app
    volumes:
      - ./models:/app/models:ro
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  frontend:
    build:
      context: .
      dockerfile: frontend/Dockerfile
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://backend:8000
    depends_on:
      - backend
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "wget", "--no-verbose", "--tries=1", "--spider", "http://localhost:3000"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - frontend
      - backend
    restart: unless-stopped
```

### 🚀 Despliegue en DigitalOcean

```bash
# Clonar el repositorio
git clone <repository-url>
cd Clasificacion_Biomedica

# Construir y ejecutar con Docker Compose
docker-compose up -d

# Verificar que los servicios funcionan
curl http://localhost/api/health  # Backend health check
curl http://localhost             # Frontend

# Ver logs de los contenedores
docker-compose logs -f backend
docker-compose logs -f frontend

# Detener los servicios
docker-compose down
```

### 📋 Características de los Contenedores

#### **Optimizaciones de Seguridad**
- ✅ **Usuarios no-root** en ambos contenedores
- ✅ **Dependencias mínimas** en imágenes slim/alpine
- ✅ **Health checks** para monitoreo automático
- ✅ **Volúmenes de solo lectura** para modelos

#### **Optimizaciones de Performance**
- 🚀 **Cache de Docker layers** optimizado
- 🚀 **Build multi-stage** para imágenes más pequeñas
- 🚀 **Restart policies** para alta disponibilidad
- 🚀 **Nginx reverse proxy** para balanceo de carga

## 📄 Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo LICENSE para detalles.


---
