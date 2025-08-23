"""
Configuration settings for the Medical Text Classification system.

This module contains all configuration parameters for the biomedical
text classification project.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Data files
DATASET_FILE = DATA_DIR / "challenge_data-18-ago.csv"

# Model configuration
MODEL_CONFIG = {
    "model_name": "bert-base-uncased",
    "num_labels": 4,
    "max_length": 512,
    "batch_size": 16,
    "learning_rate": 2e-5,
    "epochs": 3,
    "dropout": 0.1,
    "optimal_threshold": 0.6
}

# Label configuration
LABEL_COLUMNS = ["cardiovascular", "hepatorenal", "neurologico", "oncologico"]

# Training configuration
TRAINING_CONFIG = {
    "test_size": 0.2,
    "validation_split": 0.2,
    "random_state": 42,
    "stratify": True
}

# API configuration
API_CONFIG = {
    "host": "localhost",
    "port": 8000,
    "reload": True,
    "cors_origins": ["http://localhost:3000"]
}

# Frontend configuration
FRONTEND_CONFIG = {
    "host": "localhost",
    "port": 3000,
    "api_url": "http://localhost:8000"
}

# Paths for model artifacts
TRAINED_MODEL_PATH = MODELS_DIR / "trained_model"
MODEL_WEIGHTS = TRAINED_MODEL_PATH / "model.pt"
MODEL_CONFIG_FILE = TRAINED_MODEL_PATH / "config.json"
TOKENIZER_PATH = TRAINED_MODEL_PATH
THRESHOLD_FILE = TRAINED_MODEL_PATH / "best_threshold.json"
METRICS_FILE = TRAINED_MODEL_PATH / "metrics.json"
