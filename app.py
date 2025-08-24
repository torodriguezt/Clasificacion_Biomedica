"""
Medical Text Classification Web Application

A Streamlit web application for medical text classification using a trained BERT model.
Allows users to upload CSV files and get predictions with performance metrics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
import json
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    f1_score,
    accuracy_score,
    precision_score,
    recall_score
)
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

# Import our custom model
import sys
sys.path.append(str(Path(__file__).parent))
from backend.improved_medical_bert import ImprovedMedicalBERT

# Page configuration
st.set_page_config(
    page_title="Medical Text Classifier",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

class MedicalClassifierApp:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.categories = ["cardiovascular", "hepatorenal", "neurological", "oncological"]
        self.threshold = 0.36
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = Path("models/trained_model")
        
    @st.cache_resource
    def load_model(_self):
        """Load the trained model and tokenizer."""
        try:
            model_dir = _self.model_dir
            
            # Load config
            config_path = model_dir / "config.json"
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
            
            # Initialize model
            model_name = config.get("base_model_name", "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
            num_labels = int(config.get("num_labels", 4))
            dropout_rate = float(config.get("dropout", 0.3))
            use_attn = bool(config.get("use_attn", True))
            pos_weight = config.get("pos_weight", None)
            
            model = ImprovedMedicalBERT(
                model_name=model_name,
                num_labels=num_labels,
                pos_weight=pos_weight,
                dropout_rate=dropout_rate,
                use_attn=use_attn,
            )
            
            # Load weights
            checkpoint_path = model_dir / "model.pt"
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(state_dict, strict=False)
            model.to(_self.device)
            model.eval()
            
            # Load threshold if available
            threshold_path = model_dir / "best_threshold.json"
            if threshold_path.exists():
                with open(threshold_path, 'r') as f:
                    _self.threshold = float(json.load(f)["threshold"])
            
            # Load categories if available
            mlb_path = model_dir / "mlb.pkl"
            if mlb_path.exists():
                try:
                    mlb = joblib.load(mlb_path)
                    if hasattr(mlb, 'classes_'):
                        _self.categories = list(mlb.classes_)
                except:
                    pass
            
            return model, tokenizer
            
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            st.error("Please ensure the trained model is available in 'models/trained_model/' directory")
            return None, None
    
    def predict_batch(self, texts, batch_size=16):
        """Predict on a batch of texts."""
        if self.model is None or self.tokenizer is None:
            return None
        
        predictions = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize batch
                encoding = self.tokenizer(
                    batch_texts,
                    max_length=512,
                    truncation=True,
                    padding=True,
                    return_tensors="pt"
                )
                
                # Move to device
                encoding = {k: v.to(self.device) for k, v in encoding.items()}
                
                # Get predictions
                outputs = self.model(**encoding)
                logits = outputs["logits"]
                probs = torch.sigmoid(logits).cpu().numpy()
                
                predictions.extend(probs)
        
        return np.array(predictions)
    
    def calculate_metrics(self, y_true, y_pred):
        """Calculate comprehensive metrics."""
        # Convert to binary predictions using threshold
        y_pred_binary = (y_pred > self.threshold).astype(int)
        
        # Calculate metrics
        metrics = {}
        
        # Overall metrics
        metrics['weighted_f1'] = f1_score(y_true, y_pred_binary, average='weighted', zero_division=0)
        metrics['macro_f1'] = f1_score(y_true, y_pred_binary, average='macro', zero_division=0)
        metrics['micro_f1'] = f1_score(y_true, y_pred_binary, average='micro', zero_division=0)
        metrics['accuracy'] = accuracy_score(y_true, y_pred_binary)
        metrics['weighted_precision'] = precision_score(y_true, y_pred_binary, average='weighted', zero_division=0)
        metrics['weighted_recall'] = recall_score(y_true, y_pred_binary, average='weighted', zero_division=0)
        
        # Per-class metrics
        class_report = classification_report(
            y_true, y_pred_binary, 
            target_names=self.categories, 
            output_dict=True,
            zero_division=0
        )
        
        return metrics, class_report
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix for each class."""
        y_pred_binary = (y_pred > self.threshold).astype(int)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        for i, category in enumerate(self.categories):
            cm = confusion_matrix(y_true[:, i], y_pred_binary[:, i])
            
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'],
                ax=axes[i]
            )
            axes[i].set_title(f'{category.title()} - Confusion Matrix')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        plt.tight_layout()
        return fig

def main():
    st.title("🏥 Medical Text Classification System")
    st.markdown("---")
    
    # Initialize app
    app = MedicalClassifierApp()
    
    # Sidebar
    st.sidebar.title("📋 Instructions")
    st.sidebar.markdown("""
    ### How to use:
    1. **Upload CSV file** with columns: `title`, `abstract`, `group`
    2. **Preview** your data
    3. **Run predictions** to get classifications
    4. **View results** with metrics and confusion matrix
    
    ### Expected format:
    - **title**: Paper title
    - **abstract**: Paper abstract
    - **group**: True category (cardiovascular, hepatorenal, neurological, oncological)
    """)
    
    # Load model
    with st.spinner("Loading model..."):
        app.model, app.tokenizer = app.load_model()
    
    if app.model is None:
        st.error("❌ Model could not be loaded. Please check if the model files exist.")
        st.stop()
    
    st.success("✅ Model loaded successfully!")
    st.info(f"📊 Using threshold: {app.threshold:.3f} | Device: {app.device}")
    
    # File upload
    st.header("📁 Upload Data")
    uploaded_file = st.file_uploader(
        "Choose a CSV file", 
        type="csv",
        help="Upload a CSV file with columns: title, abstract, group"
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            
            # Validate columns
            required_cols = ['title', 'abstract', 'group']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"❌ Missing required columns: {missing_cols}")
                st.stop()
            
            # Display data info
            st.header("📊 Data Preview")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Samples", len(df))
            with col2:
                st.metric("Unique Groups", df['group'].nunique())
            with col3:
                st.metric("Columns", len(df.columns))
            
            # Show sample data
            st.subheader("Sample Data")
            st.dataframe(df.head(10))
            
            # Show class distribution
            st.subheader("Class Distribution")
            class_counts = df['group'].value_counts()
            fig, ax = plt.subplots(figsize=(10, 6))
            class_counts.plot(kind='bar', ax=ax)
            ax.set_title('Distribution of Classes in Dataset')
            ax.set_xlabel('Class')
            ax.set_ylabel('Count')
            plt.xticks(rotation=45)
            st.pyplot(fig)
            
            # Prediction button
            st.header("🔮 Run Predictions")
            
            if st.button("Start Classification", type="primary"):
                # Prepare texts
                texts = (df['title'].fillna('') + '. ' + df['abstract'].fillna('')).tolist()
                
                # Run predictions
                with st.spinner("Running predictions..."):
                    predictions = app.predict_batch(texts)
                
                if predictions is not None:
                    # Convert predictions to categories
                    pred_binary = (predictions > app.threshold).astype(int)
                    
                    # Get predicted categories
                    predicted_groups = []
                    for pred in pred_binary:
                        active_categories = [app.categories[i] for i, val in enumerate(pred) if val == 1]
                        if active_categories:
                            predicted_groups.append(active_categories[0])  # Take first if multiple
                        else:
                            # If no category above threshold, take the one with highest probability
                            max_idx = np.argmax(predictions[len(predicted_groups)])
                            predicted_groups.append(app.categories[max_idx])
                    
                    # Add predictions to dataframe
                    df_results = df.copy()
                    df_results['group_predicted'] = predicted_groups
                    
                    # Add probability scores
                    for i, category in enumerate(app.categories):
                        df_results[f'{category}_score'] = predictions[:, i]
                    
                    # Calculate metrics if we have true labels
                    if 'group' in df.columns:
                        # Create binary encoding for true labels
                        mlb = MultiLabelBinarizer(classes=app.categories)
                        y_true = mlb.fit_transform([[group] for group in df['group']])
                        
                        # Calculate metrics
                        metrics, class_report = app.calculate_metrics(y_true, predictions)
                        
                        # Display results
                        st.header("📈 Results")
                        
                        # Overall metrics
                        st.subheader("Overall Performance")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Weighted F1-Score", f"{metrics['weighted_f1']:.3f}")
                        with col2:
                            st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
                        with col3:
                            st.metric("Weighted Precision", f"{metrics['weighted_precision']:.3f}")
                        with col4:
                            st.metric("Weighted Recall", f"{metrics['weighted_recall']:.3f}")
                        
                        # Per-class metrics
                        st.subheader("Per-Class Performance")
                        class_df = pd.DataFrame(class_report).T
                        class_df = class_df.drop(['accuracy', 'macro avg', 'weighted avg'], errors='ignore')
                        st.dataframe(class_df.round(3))
                        
                        # Confusion Matrix
                        st.subheader("Confusion Matrices")
                        fig = app.plot_confusion_matrix(y_true, predictions)
                        st.pyplot(fig)
                    
                    # Results table
                    st.header("📋 Detailed Results")
                    st.dataframe(df_results)
                    
                    # Download button
                    csv = df_results.to_csv(index=False)
                    st.download_button(
                        label="💾 Download Results CSV",
                        data=csv,
                        file_name="medical_classification_results.csv",
                        mime="text/csv"
                    )
                    
                    st.success("✅ Classification completed successfully!")
                    
                else:
                    st.error("❌ Error during prediction")
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("**Medical Text Classification System** | Built with Streamlit & PyTorch")

if __name__ == "__main__":
    main()
