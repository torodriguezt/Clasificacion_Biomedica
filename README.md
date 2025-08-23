# 🧠 Clasificador de Textos Biomédicos con IA

Un sistema integral de clasificación de textos biomédicos que utiliza ImprovedMedicalBERT para categorizar literatura médica en las categorías: cardiovascular, hepatorenal, neurológica y oncológica.

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

## 📊 Análisis de Datos

### Resumen del Dataset
- **Fuente**: Abstracts de literatura médica
- **Categorías**: Cardiovascular, Hepatorenal, Neurológica, Oncológica
- **Formato**: Clasificación multi-etiqueta
- **Tamaño**: Variable según datos disponibles

### Características del Análisis:
- Análisis de curvas ROC con puntuaciones AUC
- Visualización de distribución de etiquetas
- Análisis de longitud de textos
- Matrices de confusión por categoría
- Métricas de evaluación integrales

## 🛠️ Flujo de Desarrollo

### 1. Procesamiento de Datos
```python
from scripts.data_processing import load_medical_data, clean_medical_text

# Cargar y preprocesar datos
df = load_medical_data("data/challenge_data-18-ago.csv")
df = clean_medical_text(df)
```

### 2. Entrenamiento del Modelo
```python
from scripts.training_utils import train_bert_model
from scripts.model_utils import ImprovedMedicalBERT

# Entrenar el modelo
model, tokenizer, metrics = train_bert_model(
    train_texts, train_labels,
    model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
)
```

### 3. Evaluación
```python
from scripts.evaluation_utils import compute_multilabel_metrics
from scripts.visualization import plot_roc_curves

# Evaluación integral
metrics = compute_multilabel_metrics(y_true, y_pred, y_probs)
plot_roc_curves(y_true, y_probs, class_names)
```

### 4. Aumento de Datos
```python
from scripts.text_augmentation import MedicalTextAugmenter

# Aumentar datos de entrenamiento
augmenter = MedicalTextAugmenter()
augmented_df = augmenter.augment_dataset(df, "text", label_columns)
```

## 🔧 Configuración

### Variables de Entorno
```python
# config/settings.py contiene configuración centralizada
MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
MAX_LENGTH = 512
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 4
```

### Configuración del Modelo
```json
{
  "base_model_name": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
  "num_labels": 4,
  "dropout": 0.3,
  "use_attn": true,
  "pos_weight": [2.5, 3.0, 2.0, 1.8]
}
```

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
FROM python:3.9-slim
COPY backend/ /app/
COPY models/ /app/models/
RUN pip install -r requirements.txt
EXPOSE 8000
CMD ["python", "main.py"]
```

### Contenedor Frontend
```dockerfile
FROM node:18-alpine
COPY frontend/ /app/
RUN npm install && npm run build
EXPOSE 3000
CMD ["npm", "start"]
```

## 🔍 Monitoreo y Depuración

### Logs
- Logs del backend: Revisar salida de consola de `start-backend.bat`
- Logs del frontend: Revisar salida de consola de `start-frontend.bat`
- Carga del modelo: Logging verboso en el inicio de FastAPI

### Problemas Comunes
1. **CUDA/GPU**: El modelo automáticamente recurre a CPU
2. **Memoria**: Ajustar batch size en config para datasets grandes
3. **Dependencias**: Ejecutar `setup.bat` para reinstalar entorno

## 🧪 Pruebas

### Pruebas Unitarias
```cmd
# Ejecutar suite de pruebas
python -m pytest scripts/tests/
```

### Pruebas de API
```cmd
# Probar endpoints de API
python scripts/test_api.py
```

## 📚 Investigación y Citas

### Modelos Base
- **PubMedBERT**: Modelo de lenguaje biomédico específico del dominio
- **Arquitectura Transformer**: Redes neuronales basadas en atención
- **Clasificación Multi-etiqueta**: Predicción simultánea de múltiples categorías

### Métodos de Evaluación
- **Análisis ROC**: Curvas características operador-receptor
- **Validación cruzada**: Validación K-fold para evaluación robusta
- **Optimización de umbrales**: Umbrales óptimos específicos por categoría

## 🤝 Contribuciones

1. **Estándares de Código**: Cumplimiento PEP8, type hints, docstrings
2. **Pruebas**: Agregar pruebas para nuevas características
3. **Documentación**: Actualizar README y comentarios en línea
4. **Reproducibilidad**: Asegurar que archivos .bat funcionen para configuración

## 📄 Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo LICENSE para detalles.

## 🆘 Soporte

### Solución de Problemas
1. Ejecutar `setup.bat` para reiniciar entorno
2. Revisar logs en salidas de consola
3. Verificar archivos del modelo en `models/trained_model/`
4. Asegurar que puertos 8000 y 3000 estén disponibles

### Contacto
Para problemas y preguntas, por favor crear un issue en el repositorio.

---

**Hecho con ❤️ para investigación biomédica y aplicaciones de salud impulsadas por IA.**
