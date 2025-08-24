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
    
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")
    
    try:
        # Read and decode CSV content
        content = await file.read()
        csv_content = content.decode('utf-8')
        
        # Try to detect CSV delimiter automatically
        try:
            import csv
            sniffer = csv.Sniffer()
            delimiter = sniffer.sniff(csv_content[:1024]).delimiter
            logger.info(f"Detected CSV delimiter: '{delimiter}'")
        except:
            delimiter = ','  # Default to comma
            logger.info("Using default comma delimiter")
        
        # Read CSV with multiple fallback strategies
        df = None
        read_strategies = [
            # Strategy 1: Use detected delimiter
            {'delimiter': delimiter, 'quotechar': '"', 'skipinitialspace': True},
            # Strategy 2: Standard comma-separated with quotes
            {'delimiter': ',', 'quotechar': '"', 'skipinitialspace': True},
            # Strategy 3: Semicolon-separated (European standard)
            {'delimiter': ';', 'quotechar': '"', 'skipinitialspace': True},
            # Strategy 4: Tab-separated
            {'delimiter': '\t', 'quotechar': '"', 'skipinitialspace': True},
            # Strategy 5: Pipe-separated
            {'delimiter': '|', 'quotechar': '"', 'skipinitialspace': True},
        ]
        
        for i, strategy in enumerate(read_strategies):
            try:
                df = pd.read_csv(StringIO(csv_content), **strategy, on_bad_lines='skip')
                logger.info(f"Successfully read CSV with strategy {i+1}: {strategy}")
                break
            except Exception as e:
                logger.warning(f"Strategy {i+1} failed: {e}")
                continue
        
        if df is None:
            raise HTTPException(status_code=400, detail="Could not parse CSV file. Please check the format.")
        
        # Clean column names by stripping spaces
        df.columns = df.columns.str.strip()
        
        # Log basic CSV info
        logger.info(f"CSV loaded: {len(df)} rows, {len(df.columns)} columns")
        logger.info(f"Columns: {list(df.columns)}")
        
        # Validate required columns
        required_columns = ['title', 'abstract', 'group']
        
        # Check if we have the right columns or if they're combined in one column
        if len(df.columns) == 1 and any(';' in col for col in df.columns):
            # Data is likely semicolon-separated but read as one column
            logger.info("Detected semicolon-separated data in single column, re-parsing...")
            df = pd.read_csv(StringIO(csv_content), delimiter=';', quotechar='"', skipinitialspace=True)
            df.columns = df.columns.str.strip()
            logger.info(f"Re-parsed CSV: {len(df)} rows, {len(df.columns)} columns")
            logger.info(f"New columns: {list(df.columns)}")
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            # Additional debug info
            logger.error(f"Missing columns: {missing_columns}")
            logger.error(f"Available columns: {list(df.columns)}")
            logger.error(f"CSV sample (first 200 chars): {csv_content[:200]}")
            raise HTTPException(
                status_code=400, 
                detail=f"Missing columns in CSV: {missing_columns}. Required columns are: {required_columns}. Available columns: {list(df.columns)}"
            )
        
        # Clean and validate data
        # Fill NaN values with empty strings for text columns
        df['title'] = df['title'].fillna('')
        df['abstract'] = df['abstract'].fillna('')
        df['group'] = df['group'].fillna('')
        
        # Remove rows where both title and abstract are empty
        initial_rows = len(df)
        df = df[(df['title'].str.strip() != '') | (df['abstract'].str.strip() != '')]
        if len(df) != initial_rows:
            logger.info(f"Removed {initial_rows - len(df)} rows with empty title and abstract")
        
        # Helper function to clean NaN and inf values for JSON serialization
        def clean_metric_value(value):
            import numpy as np
            if np.isnan(value) or np.isinf(value):
                return 0.0
            return float(value)
        
        # Process each row in the CSV
        results = []
        predicted_labels = []
        true_labels = []
        
        try:
            for idx, row in df.iterrows():
                try:
                    # Predict labels
                    preds = classifier.predict(row['title'], row['abstract'])
                    cleaned_preds = {k: clean_metric_value(v) for k, v in preds.items()}
                    
                    # Apply threshold to predicted labels
                    threshold = 0.36
                    pred_labels = [label for label, score in cleaned_preds.items() if score > threshold]
                    if not pred_labels:
                        pred_labels = [max(cleaned_preds, key=cleaned_preds.get)]
                    
                    # Sort predicted labels for consistent ordering
                    pred_labels.sort()
                    
                    # Process true labels with flexible separator detection
                    true_group = str(row['group']).strip()
                    true_label_list = []
                    
                    if true_group and true_group != 'nan' and true_group.lower() != 'none':
                        # Try to detect the separator used
                        # Priority: | (pipe) > , (comma) > ; (semicolon)
                        if '|' in true_group:
                            # Split by pipe separator
                            true_label_list = [label.strip() for label in true_group.split('|') if label.strip()]
                        elif ',' in true_group:
                            # Split by comma separator
                            true_label_list = [label.strip() for label in true_group.split(',') if label.strip()]
                        elif ';' in true_group:
                            # Split by semicolon separator
                            true_label_list = [label.strip() for label in true_group.split(';') if label.strip()]
                        else:
                            # Single value - ensure it's valid
                            clean_label = true_group.strip()
                            if clean_label and clean_label.lower() not in ['', 'none', 'null', 'nan']:
                                true_label_list = [clean_label]
                        
                        # Validate labels against known categories
                        valid_categories = set(classifier.categories)
                        validated_labels = []
                        for label in true_label_list:
                            # Clean and normalize the label
                            clean_label = label.lower().strip()
                            # Try exact match first
                            if clean_label in valid_categories:
                                validated_labels.append(clean_label)
                            else:
                                # Try partial matching
                                for valid_cat in valid_categories:
                                    if clean_label in valid_cat or valid_cat in clean_label:
                                        validated_labels.append(valid_cat)
                                        break
                        
                        true_label_list = list(set(validated_labels))  # Remove duplicates
                        true_label_list.sort()  # Sort for consistent ordering
                        
                        # Log the processing for debugging (first 5 rows)
                        if len(results) < 5:
                            separator_used = '|' if '|' in true_group else ',' if ',' in true_group else ';' if ';' in true_group else 'none'
                            logger.info(f"Row {len(results)+1}: Original='{true_group}' -> Separator='{separator_used}' -> Labels={true_label_list}")
                    
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
                    
                except Exception as row_error:
                    logger.error(f"Error processing row {idx}: {row_error}")
                    logger.error(f"Row data: {dict(row)}")
                    continue  # Skip this row and continue with the next
                    
        except Exception as processing_error:
            logger.error(f"Error in main processing loop: {processing_error}")
            raise HTTPException(status_code=500, detail=f"Error processing CSV data: {str(processing_error)}")
        
        if not results:
            raise HTTPException(status_code=400, detail="No valid rows could be processed from the CSV file")
        
        # Add predicted labels to the dataframe
        # Sort labels for consistent output
        df["group_predicted"] = ["|".join(sorted(labels)) for labels in predicted_labels]
        
        # Initialize metrics dictionary
        metrics = {}
        if true_labels and any(true_labels):
            # Multi-label evaluation
            from sklearn.preprocessing import MultiLabelBinarizer
            from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, hamming_loss, jaccard_score
            import numpy as np
            
            # Get all possible labels
            all_categories = classifier.categories
            mlb = MultiLabelBinarizer(classes=all_categories)
            
            # Transform labels to binary format
            y_true_binary = mlb.fit_transform(true_labels)
            y_pred_binary = mlb.transform(predicted_labels)
            
            try:
                # Calculate metrics
                subset_accuracy = np.mean([np.array_equal(true_row, pred_row) for true_row, pred_row in zip(y_true_binary, y_pred_binary)])
                hamming_loss_score = hamming_loss(y_true_binary, y_pred_binary)
                jaccard_similarity = jaccard_score(y_true_binary, y_pred_binary, average='samples', zero_division=0) if np.any(y_true_binary) or np.any(y_pred_binary) else 0.0
                
                # F1 scores
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
                logger.warning(f"Error calculating metrics: {metric_error}")
                # Set default values if calculation fails
                subset_accuracy = 0.0
                hamming_loss_score = 1.0
                jaccard_similarity = 0.0
                precision = recall = f1_weighted = f1_macro = f1_micro = f1_samples = 0.0
                f1_class_dict = {cat: 0.0 for cat in all_categories}
            
            # Confusion Matrix (per label)
            y_true_single = [max(true_labs, key=lambda x: results[i]["predictions"].get(x, 0)) if true_labs else "unknown" for i, true_labs in enumerate(true_labels)]
            y_pred_single = [max(pred_labs, key=lambda x: results[i]["predictions"].get(x, 0)) if pred_labs else "unknown" for i, pred_labs in enumerate(predicted_labels)]
            valid_indices = [i for i, (t, p) in enumerate(zip(y_true_single, y_pred_single)) if t != "unknown" and p != "unknown"]
            
            if valid_indices:
                y_true_cm = [y_true_single[i] for i in valid_indices]
                y_pred_cm = [y_pred_single[i] for i in valid_indices]
                cm = confusion_matrix(y_true_cm, y_pred_cm, labels=all_categories)
            else:
                cm = np.zeros((len(all_categories), len(all_categories)))
            
            metrics = {
                "confusion_matrix": {"matrix": cm.tolist(), "labels": all_categories, "format": "rows=true_labels, cols=predicted_labels"},
                "f1_scores": {"macro": f1_macro, "micro": f1_micro, "weighted": f1_weighted, "samples": f1_samples, "per_class": f1_class_dict},
                "accuracy": {"subset_accuracy": subset_accuracy, "jaccard_similarity": jaccard_similarity},
                "hamming_loss": hamming_loss_score,
                "precision": precision,
                "recall": recall
            }
            
            logger.info(f"Confusion Matrix:\n{cm}")
            logger.info(f"F1 Score (weighted): {f1_weighted:.4f}")
            logger.info(f"Subset Accuracy: {subset_accuracy:.4f}")
            logger.info(f"Jaccard Similarity: {jaccard_similarity:.4f}")
            logger.info(f"Hamming Loss: {hamming_loss_score:.4f}")
        
        # Prepare final response
        response = {
            "predictions": results, 
            "data": df.to_dict(orient="records"),
            "total_processed": len(results)
        }
        
        if metrics:
            response["evaluation_metrics"] = metrics
        
        # Clean response to remove NaN/inf values
        def deep_clean_response(obj):
            import numpy as np
            if isinstance(obj, dict):
                return {k: deep_clean_response(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [deep_clean_response(v) for v in obj]
            elif isinstance(obj, (int, float, np.number)):
                return float(obj) if not (np.isnan(obj) or np.isinf(obj)) else 0.0
            else:
                return obj
        
        cleaned_response = deep_clean_response(response)
        return cleaned_response

    except Exception as e:
        logger.error(f"Error processing CSV: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing the CSV file: {str(e)}")

    
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)