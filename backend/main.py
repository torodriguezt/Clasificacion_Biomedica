from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict
import logging
import json
import torch
import joblib
import pandas as pd
from io import StringIO
from transformers import AutoTokenizer
from improved_medical_bert import ImprovedMedicalBERT  # tu clase custom
from pathlib import Path
from sklearn.metrics import confusion_matrix, f1_score
import numpy as np

# =======================
# Configuración
# =======================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Medical AI Classifier",
    description="Clasificador biomédico usando ImprovedMedicalBERT (custom)",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =======================
# Modelos de datos
# =======================
class MedicalDocument(BaseModel):
    title: str
    abstract: str


class ClassificationResult(BaseModel):
    cardiovascular: float
    hepatorenal: float
    neurological: float
    oncological: float
    dominant_category: str
    confidence: float


# =======================
# Clasificador
# =======================
class MedicalClassifier:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_loaded = False
        self.threshold = 0.36  # THRESHOLD REAL DEL MODELO
        self.categories = [
            "cardiovascular",
            "hepatorenal",
            "neurological",
            "oncological",
        ]

        self._load_bundle()
        self.model_loaded = True

    def _load_bundle(self):
        logger.info(f"Cargando bundle desde: {self.model_dir}")

        cfg_path = self.model_dir / "config.json"
        ckpt_path = self.model_dir / "model.pt"
        mlb_path = self.model_dir / "mlb.pkl"
        thr_path = self.model_dir / "best_threshold.json"

        if not cfg_path.exists() or not ckpt_path.exists():
            raise FileNotFoundError("Faltan config.json o model.pt")

        # cargar tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_dir), local_files_only=True
        )

        # cargar config
        with open(cfg_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # Extraer parámetros del config.json
        model_name = config.get(
            "base_model_name", "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
        )
        num_labels = int(config.get("num_labels", 4))
        dropout_rate = float(config.get("dropout", 0.3))
        use_attn = bool(config.get("use_attn", True))
        pos_weight = config.get("pos_weight", None)

        # Inicializar ImprovedMedicalBERT
        self.model = ImprovedMedicalBERT(
            model_name=model_name,
            num_labels=num_labels,
            pos_weight=pos_weight,
            dropout_rate=dropout_rate,
            use_attn=use_attn,
        )

        # cargar pesos
        sd = torch.load(ckpt_path, map_location="cpu")
        try:
            missing, unexpected = self.model.load_state_dict(sd, strict=True)
            if missing or unexpected:
                logger.warning(f"Missing: {missing}, Unexpected: {unexpected}")
        except Exception as e:
            logger.warning(f"Strict=True falló, reintentando strict=False: {e}")
            self.model.load_state_dict(sd, strict=False)

        self.model.to(self.device).eval()

        # categorías desde mlb.pkl si existe
        if mlb_path.exists():
            try:
                mlb = joblib.load(mlb_path)
                if hasattr(mlb, "classes_"):
                    self.categories = list(mlb.classes_)
                    logger.info(f"✅ Clases cargadas desde mlb.pkl: {self.categories}")
            except Exception as e:
                logger.warning(f"No se pudo cargar mlb.pkl: {e}")

        # threshold si existe
        if thr_path.exists():
            try:
                with open(thr_path, "r", encoding="utf-8") as f:
                    self.threshold = float(json.load(f)["threshold"])
            except Exception as e:
                logger.warning(f"No se pudo leer best_threshold.json: {e}")

        logger.info(
            f"Modelo listo en {self.device}, categorías: {self.categories}, threshold={self.threshold}"
        )

    @torch.inference_mode()
    def predict(self, title: str, abstract: str) -> Dict[str, float]:
        if not self.model_loaded:
            raise RuntimeError("El modelo no está cargado correctamente")

        text = f"{title}. {abstract}".strip()
        enc = self.tokenizer(
            text, max_length=512, truncation=True, padding=True, return_tensors="pt"
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        outputs = self.model(**enc)
        logits = outputs["logits"]

        probs = torch.sigmoid(logits).detach().cpu().numpy()[0]
        preds = {
            self.categories[i]: float(probs[i]) for i in range(len(self.categories))
        }
        return preds


# =======================
# Inicialización
# =======================
MODEL_DIR = Path(__file__).parent.parent / "models" / "trained_model"
classifier = None


@app.on_event("startup")
def _startup():
    global classifier
    classifier = MedicalClassifier(MODEL_DIR)


# =======================
# Endpoints
# =======================
@app.get("/", response_class=HTMLResponse)
async def root():
    return "<h1>Medical AI Classifier API</h1><p>Usa POST /classify</p>"


@app.get("/health")
async def health():
    return {
        "status": "healthy" if classifier and classifier.model_loaded else "error",
        "categories": classifier.categories if classifier else [],
        "device": str(classifier.device) if classifier else "N/A",
        "threshold": getattr(classifier, "threshold", None),
    }


@app.post("/classify", response_model=ClassificationResult)
async def classify(document: MedicalDocument):
    if not document.title.strip() and not document.abstract.strip():
        raise HTTPException(
            status_code=400, detail="Título y abstract no pueden estar vacíos"
        )

    preds = classifier.predict(document.title, document.abstract)
    dom = max(preds, key=lambda k: preds[k])
    conf = preds[dom]

    return ClassificationResult(
        cardiovascular=round(preds.get("cardiovascular", 0.0), 3),
        hepatorenal=round(preds.get("hepatorenal", 0.0), 3),
        neurological=round(preds.get("neurological", 0.0), 3),
        oncological=round(preds.get("oncological", 0.0), 3),
        dominant_category=dom,
        confidence=round(conf, 3),
    )


@app.post("/batch_classify")
async def batch_classify(documents: List[MedicalDocument]):
    if len(documents) > 50:
        raise HTTPException(status_code=400, detail="Máximo 50 documentos por lote")
    results = []
    for doc in documents:
        preds = classifier.predict(doc.title, doc.abstract)
        dom = max(preds, key=lambda k: preds[k])
        results.append(
            {
                "title": doc.title[:100] + "..." if len(doc.title) > 100 else doc.title,
                "predictions": preds,
                "dominant_category": dom,
                "confidence": preds[dom],
            }
        )
    return {"results": results, "total_processed": len(documents)}


@app.post("/csv_classify")
async def csv_classify(file: UploadFile = File(...)):
    try:
        # Read CSV file
        contents = await file.read()
        data = StringIO(contents.decode("utf-8"))
        df = pd.read_csv(data)

        if "title" not in df.columns or "abstract" not in df.columns:
            raise HTTPException(status_code=400, detail="CSV debe contener 'title' y 'abstract'")

        # Process each row in the CSV file
        results = []
        predicted_labels = []
        
        for _, row in df.iterrows():
            preds = classifier.predict(row['title'], row['abstract'])
            dom = max(preds, key=lambda k: preds[k])
            predicted_labels.append(dom)
            
            results.append(
                {
                    "title": row["title"],
                    "predictions": preds,
                    "dominant_category": dom,
                    "confidence": preds[dom],
                }
            )

        # Add predicted group to the DataFrame
        df["group_predicted"] = predicted_labels
        
        # Calculate metrics if we have true labels
        metrics = {}
        if "group" in df.columns:
            true_labels = df["group"].tolist()
            
            # Calculate confusion matrix
            cm = confusion_matrix(true_labels, predicted_labels, labels=classifier.categories)
            
            # Calculate F1 scores
            f1_macro = f1_score(true_labels, predicted_labels, labels=classifier.categories, average='macro')
            f1_micro = f1_score(true_labels, predicted_labels, labels=classifier.categories, average='micro')
            f1_weighted = f1_score(true_labels, predicted_labels, labels=classifier.categories, average='weighted')
            
            # F1 score por clase
            f1_per_class = f1_score(true_labels, predicted_labels, labels=classifier.categories, average=None)
            f1_class_dict = {classifier.categories[i]: float(f1_per_class[i]) for i in range(len(classifier.categories))}
            
            metrics = {
                "confusion_matrix": {
                    "matrix": cm.tolist(),
                    "labels": classifier.categories,
                    "format": "rows=true_labels, cols=predicted_labels"
                },
                "f1_scores": {
                    "macro": float(f1_macro),
                    "micro": float(f1_micro),
                    "weighted": float(f1_weighted),
                    "per_class": f1_class_dict
                },
                "accuracy": float(np.mean(np.array(true_labels) == np.array(predicted_labels)))
            }
            
            logger.info(f"Confusion Matrix:\n{cm}")
            logger.info(f"F1 Score (macro): {f1_macro:.4f}")
            logger.info(f"F1 Score (micro): {f1_micro:.4f}")
            logger.info(f"F1 Score (weighted): {f1_weighted:.4f}")
            logger.info(f"Accuracy: {metrics['accuracy']:.4f}")

        response = {
            "predictions": results, 
            "data": df.to_dict(orient="records"),
            "total_processed": len(results)
        }
        
        # Only add metrics if we have true labels
        if metrics:
            response["evaluation_metrics"] = metrics

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar el archivo CSV: {str(e)}")
    
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)