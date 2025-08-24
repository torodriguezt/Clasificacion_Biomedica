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
MODEL_DIR = Path(__file__).parent.parent / "models"
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
    """Process a CSV file with multilabel medical articles for classification and evaluation."""
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Solo se permiten archivos CSV")
    
    try:
        # Read CSV content
        content = await file.read()
        csv_content = content.decode('utf-8')
        df = pd.read_csv(StringIO(csv_content))
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Validate required columns
        required_columns = ['title', 'abstract', 'group']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise HTTPException(
                status_code=400, 
                detail=f"Columnas faltantes en el CSV: {missing_columns}. Se requieren: {required_columns}"
            )
        
        # Process each row
        results = []
        predicted_labels = []
        true_labels = []
        
        # Helper function to clean NaN and inf values
        def clean_metric_value(value):
            """Convert NaN and inf values to 0.0 for JSON serialization"""
            import numpy as np
            if np.isnan(value) or np.isinf(value):
                return 0.0
            return float(value)
        
        for _, row in df.iterrows():
            # Get predictions for this row
            preds = classifier.predict(row['title'], row['abstract'])
            
            # Clean predictions to ensure no NaN/inf values
            cleaned_preds = {k: clean_metric_value(v) for k, v in preds.items()}
            
            # Get predicted labels above threshold
            threshold = 0.36
            pred_labels = [label for label, score in cleaned_preds.items() if score > threshold]
            if not pred_labels:  # If no label above threshold, take the highest
                pred_labels = [max(cleaned_preds, key=cleaned_preds.get)]
            
            # Get true labels (split by |)
            true_group = str(row['group']).strip()
            if true_group and true_group != 'nan':
                true_label_list = [label.strip() for label in true_group.split('|') if label.strip()]
            else:
                true_label_list = []
            
            predicted_labels.append(pred_labels)
            true_labels.append(true_label_list)
            
            results.append({
                "title": row["title"],
                "predictions": cleaned_preds,
                "dominant_category": max(cleaned_preds, key=cleaned_preds.get),
                "confidence": max(cleaned_preds.values()),
                "predicted_labels": pred_labels,
                "true_labels": true_label_list
            })
        
        # Add predicted labels to dataframe (join with |)
        df["group_predicted"] = ["|".join(labels) for labels in predicted_labels]
        
        # Calculate metrics using multilabel approach
        metrics = {}
        if true_labels and any(true_labels):
            from sklearn.preprocessing import MultiLabelBinarizer
            from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, hamming_loss, jaccard_score
            import numpy as np
            
            # Get all possible labels
            all_categories = classifier.categories
            
            # Use MultiLabelBinarizer
            mlb = MultiLabelBinarizer(classes=all_categories)
            
            # Transform to binary matrices
            y_true_binary = mlb.fit_transform(true_labels)
            y_pred_binary = mlb.transform(predicted_labels)
            
            # Calculate multilabel-specific metrics with additional validation
            try:
                # Subset accuracy (exact match ratio)
                subset_accuracy = np.mean([np.array_equal(true_row, pred_row) for true_row, pred_row in zip(y_true_binary, y_pred_binary)])
                
                # Hamming loss (fraction of wrong labels)
                hamming_loss_score = hamming_loss(y_true_binary, y_pred_binary)
                
                # Jaccard similarity (IoU for multilabel) - handle empty predictions
                if np.any(y_true_binary) or np.any(y_pred_binary):
                    jaccard_similarity = jaccard_score(y_true_binary, y_pred_binary, average='samples', zero_division=0)
                else:
                    jaccard_similarity = 0.0
                
                # F1 scores with better handling
                precision = precision_score(y_true_binary, y_pred_binary, average='weighted', zero_division=0)
                recall = recall_score(y_true_binary, y_pred_binary, average='weighted', zero_division=0)
                f1_weighted = f1_score(y_true_binary, y_pred_binary, average='weighted', zero_division=0)
                f1_macro = f1_score(y_true_binary, y_pred_binary, average='macro', zero_division=0)
                f1_micro = f1_score(y_true_binary, y_pred_binary, average='micro', zero_division=0)
                f1_samples = f1_score(y_true_binary, y_pred_binary, average='samples', zero_division=0)
                
                # Per-class F1 scores
                f1_per_class = f1_score(y_true_binary, y_pred_binary, average=None, zero_division=0)
                f1_class_dict = {all_categories[i]: clean_metric_value(f1_per_class[i]) for i in range(len(all_categories))}
                
            except Exception as metric_error:
                logger.warning(f"Error calculating some metrics: {metric_error}")
                # Set default values if metric calculation fails
                subset_accuracy = 0.0
                hamming_loss_score = 1.0
                jaccard_similarity = 0.0
                precision = 0.0
                recall = 0.0
                f1_weighted = 0.0
                f1_macro = 0.0
                f1_micro = 0.0
                f1_samples = 0.0
                f1_class_dict = {cat: 0.0 for cat in all_categories}
            
            # For confusion matrix, we'll use label-based approach
            # Convert to single-label format for confusion matrix (using dominant labels)
            y_true_single = []
            y_pred_single = []
            
            for i, (true_labs, pred_labs) in enumerate(zip(true_labels, predicted_labels)):
                if true_labs:
                    # Take the label with highest prediction score among true labels
                    true_dominant = max(true_labs, key=lambda x: results[i]["predictions"].get(x, 0))
                    y_true_single.append(true_dominant)
                else:
                    y_true_single.append("unknown")
                    
                if pred_labs:
                    # Take the label with highest prediction score among predicted labels
                    pred_dominant = max(pred_labs, key=lambda x: results[i]["predictions"].get(x, 0))
                    y_pred_single.append(pred_dominant)
                else:
                    y_pred_single.append("unknown")
            
            # Create confusion matrix only for known labels
            valid_indices = [i for i, (t, p) in enumerate(zip(y_true_single, y_pred_single)) if t != "unknown" and p != "unknown"]
            if valid_indices:
                y_true_cm = [y_true_single[i] for i in valid_indices]
                y_pred_cm = [y_pred_single[i] for i in valid_indices]
                cm = confusion_matrix(y_true_cm, y_pred_cm, labels=all_categories)
            else:
                cm = np.zeros((len(all_categories), len(all_categories)))
            
            metrics = {
                "confusion_matrix": {
                    "matrix": cm.tolist(),
                    "labels": all_categories,
                    "format": "rows=true_labels, cols=predicted_labels"
                },
                "f1_scores": {
                    "macro": clean_metric_value(f1_macro),
                    "micro": clean_metric_value(f1_micro),
                    "weighted": clean_metric_value(f1_weighted),
                    "samples": clean_metric_value(f1_samples),
                    "per_class": f1_class_dict
                },
                "accuracy": {
                    "subset_accuracy": clean_metric_value(subset_accuracy),
                    "jaccard_similarity": clean_metric_value(jaccard_similarity)
                },
                "hamming_loss": clean_metric_value(hamming_loss_score),
                "precision": clean_metric_value(precision),
                "recall": clean_metric_value(recall)
            }
            
            logger.info(f"Confusion Matrix:\n{cm}")
            logger.info(f"F1 Score (weighted): {clean_metric_value(f1_weighted):.4f}")
            logger.info(f"Subset Accuracy: {clean_metric_value(subset_accuracy):.4f}")
            logger.info(f"Jaccard Similarity: {clean_metric_value(jaccard_similarity):.4f}")
            logger.info(f"Hamming Loss: {clean_metric_value(hamming_loss_score):.4f}")

        response = {
            "predictions": results, 
            "data": df.to_dict(orient="records"),
            "total_processed": len(results)
        }
        
        # Only add metrics if we have true labels
        if metrics:
            response["evaluation_metrics"] = metrics

        # Final cleanup of the entire response to ensure no NaN/inf values
        def deep_clean_response(obj):
            """Recursively clean all NaN and inf values from a nested structure"""
            import numpy as np
            if isinstance(obj, dict):
                return {k: deep_clean_response(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [deep_clean_response(v) for v in obj]
            elif isinstance(obj, (int, float, np.number)):
                if np.isnan(obj) or np.isinf(obj):
                    return 0.0
                return float(obj)
            else:
                return obj
        
        cleaned_response = deep_clean_response(response)
        return cleaned_response

    except Exception as e:
        logger.error(f"Error processing CSV: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error al procesar el archivo CSV: {str(e)}")
        # Read CSV file
        contents = await file.read()
        data = StringIO(contents.decode("utf-8"))
        df = pd.read_csv(data)

        # Clean column names (remove extra spaces)
        df.columns = df.columns.str.strip()

        # Ensure 'title' and 'abstract' columns are present
        if "title" not in df.columns or "abstract" not in df.columns:
            raise HTTPException(status_code=400, detail="CSV debe contener 'title' y 'abstract'")

        # Ensure 'group' column is present, and handle missing values
        if "group" not in df.columns:
            raise HTTPException(status_code=400, detail="CSV debe contener 'group'")

        # Split group values into lists and handle missing or empty groups
        # Check if groups contain "|" separator for multi-label, otherwise treat as single label
        df["group"] = df["group"].apply(lambda x: 
            [label.strip() for label in x.split("|")] if isinstance(x, str) and x.strip() != "" and "|" in x 
            else [x.strip()] if isinstance(x, str) and x.strip() != "" 
            else []
        )

        # Debugging: Log the processed 'group' column
        logger.info(f"Processed 'group' column: {df['group'].head()}")

        # Process each row in the CSV file
        results = []
        predicted_labels = []
        threshold = 0.36  # Adjust threshold for multi-label classification

        for _, row in df.iterrows():
            preds = classifier.predict(row['title'], row['abstract'])
            predicted_labels.append([label for label, score in preds.items() if score > threshold])  # Multi-label prediction
            
            results.append(
                {
                    "title": row["title"],
                    "predictions": preds,
                    "dominant_category": max(preds, key=lambda k: preds[k]),  # Dominant category for reporting
                    "confidence": preds[max(preds, key=lambda k: preds[k])],
                }
            )

        # Add predicted group to the DataFrame
        df["group_predicted"] = predicted_labels

        # Initialize metrics dictionary
        metrics = {}

        # Calculate metrics if we have true labels
        if "group" in df.columns and df["group"].notnull().any():
            true_labels = df["group"].tolist()

            # Debugging: Log true labels for metrics calculation
            logger.info(f"True labels for metrics calculation: {true_labels[:5]}")

            if not any(true_labels):  # If true_labels is empty, handle gracefully
                raise HTTPException(status_code=400, detail="No valid labels found in the 'group' column.")

            # Check if this is truly multi-label or single-label
            is_multi_label = any(len(labels) > 1 for labels in true_labels) or any(len(labels) > 1 for labels in predicted_labels)
            
            if is_multi_label:
                # Convert to binary format for multi-label classification
                from sklearn.preprocessing import MultiLabelBinarizer
                
                mlb = MultiLabelBinarizer()
                # Combine all unique labels from true and predicted
                all_labels = set()
                for labels in true_labels + predicted_labels:
                    all_labels.update(labels)
                # Ensure we include all classifier categories
                if hasattr(classifier, 'categories'):
                    all_labels.update(classifier.categories)
                mlb.fit([list(all_labels)])
                
                # Transform true and predicted labels to binary format
                y_true_binary = mlb.transform(true_labels)
                y_pred_binary = mlb.transform(predicted_labels)
                classes = mlb.classes_
            else:
                # Handle single-label classification
                # Flatten the single-element lists
                y_true_flat = [labels[0] if labels else 'unknown' for labels in true_labels]
                y_pred_flat = [labels[0] if labels else 'unknown' for labels in predicted_labels]
                
                # Convert to binary format for consistent metric calculation
                from sklearn.preprocessing import LabelBinarizer
                lb = LabelBinarizer()
                
                # Get all unique labels
                all_labels = list(set(y_true_flat + y_pred_flat))
                if hasattr(classifier, 'categories'):
                    all_labels.extend([cat for cat in classifier.categories if cat not in all_labels])
                lb.fit(all_labels)
                
                y_true_binary = lb.transform(y_true_flat)
                y_pred_binary = lb.transform(y_pred_flat)
                
                # If binary classification, reshape to 2D
                if len(all_labels) == 2:
                    y_true_binary = np.column_stack([1 - y_true_binary.ravel(), y_true_binary.ravel()])
                    y_pred_binary = np.column_stack([1 - y_pred_binary.ravel(), y_pred_binary.ravel()])
                
                classes = lb.classes_
            
            # Get the classes (categories) used by the binarizer
            # classes already defined above in both cases

            # Calculate F1 scores for multi-label classification
            f1_samples = f1_score(y_true_binary, y_pred_binary, average='samples', zero_division=1)
            f1_macro = f1_score(y_true_binary, y_pred_binary, average='macro', zero_division=1)
            f1_micro = f1_score(y_true_binary, y_pred_binary, average='micro', zero_division=1)
            f1_weighted = f1_score(y_true_binary, y_pred_binary, average='weighted', zero_division=1)

            # F1 score per class (multi-label)
            f1_per_class = f1_score(y_true_binary, y_pred_binary, average=None, zero_division=1)
            f1_class_dict = {classes[i]: float(f1_per_class[i]) for i in range(len(classes))}

            # Calculate precision and recall
            from sklearn.metrics import precision_score, recall_score, accuracy_score
            
            precision_samples = precision_score(y_true_binary, y_pred_binary, average='samples', zero_division=1)
            precision_macro = precision_score(y_true_binary, y_pred_binary, average='macro', zero_division=1)
            precision_micro = precision_score(y_true_binary, y_pred_binary, average='micro', zero_division=1)
            precision_weighted = precision_score(y_true_binary, y_pred_binary, average='weighted', zero_division=1)
            
            recall_samples = recall_score(y_true_binary, y_pred_binary, average='samples', zero_division=1)
            recall_macro = recall_score(y_true_binary, y_pred_binary, average='macro', zero_division=1)
            recall_micro = recall_score(y_true_binary, y_pred_binary, average='micro', zero_division=1)
            recall_weighted = recall_score(y_true_binary, y_pred_binary, average='weighted', zero_division=1)

            # Subset accuracy (exact match ratio)
            subset_accuracy = accuracy_score(y_true_binary, y_pred_binary)
            
            # Hamming loss (fraction of labels that are incorrectly predicted)
            from sklearn.metrics import hamming_loss
            hamming_loss_score = hamming_loss(y_true_binary, y_pred_binary)

            # Create confusion matrix for each label
            confusion_matrices = {}
            for i, label in enumerate(classes):
                cm_label = confusion_matrix(y_true_binary[:, i], y_pred_binary[:, i])
                confusion_matrices[label] = cm_label.tolist()

            metrics = {
                "classification_type": "multi_label" if is_multi_label else "single_label",
                "confusion_matrices": {
                    "per_label": confusion_matrices,
                    "format": "Binary confusion matrix per label (rows=true, cols=predicted)"
                },
                "f1_scores": {
                    "samples": float(f1_samples),
                    "macro": float(f1_macro),
                    "micro": float(f1_micro),
                    "weighted": float(f1_weighted),
                    "per_class": f1_class_dict
                },
                "precision": {
                    "samples": float(precision_samples),
                    "macro": float(precision_macro),
                    "micro": float(precision_micro),
                    "weighted": float(precision_weighted)
                },
                "recall": {
                    "samples": float(recall_samples),
                    "macro": float(recall_macro),
                    "micro": float(recall_micro),
                    "weighted": float(recall_weighted)
                },
                "accuracy": {
                    "subset_accuracy": float(subset_accuracy),  # Exact match
                    "hamming_loss": float(hamming_loss_score)   # Label-wise accuracy
                },
                "sample_count": len(true_labels),
                "labels_used": classes.tolist()
            }

            # Log metrics
            logger.info(f"Classification type: {'Multi-label' if is_multi_label else 'Single-label'}")
            logger.info(f"F1 Score (samples): {f1_samples:.4f}")
            logger.info(f"F1 Score (macro): {f1_macro:.4f}")
            logger.info(f"F1 Score (micro): {f1_micro:.4f}")
            logger.info(f"F1 Score (weighted): {f1_weighted:.4f}")
            logger.info(f"Subset Accuracy: {subset_accuracy:.4f}")
            logger.info(f"Hamming Loss: {hamming_loss_score:.4f}")

        # Prepare the response
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
        logger.error(f"Error al procesar el archivo CSV: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error al procesar el archivo CSV: {str(e)}")
    
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)