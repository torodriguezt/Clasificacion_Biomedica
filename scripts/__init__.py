"""
Medical Text Classification Scripts Package

This package provides a comprehensive set of utilities for medical text
classification tasks including data processing, model training, evaluation,
and visualization.

Modules:
    data_processing: Data loading, cleaning, and preprocessing utilities
    model_utils: Model architectures and dataset utilities  
    training_utils: Training and fine-tuning utilities
    evaluation_utils: Comprehensive evaluation metrics
    visualization: Plotting and visualization utilities
    text_augmentation: Data augmentation techniques

Usage:
    # Command line interface
    python -m scripts --help
    
    # Import individual modules
    from scripts.data_processing import load_medical_data
    from scripts.visualization import plot_confusion_matrices
"""

__version__ = "1.0.0"
__author__ = "Medical AI Classification Team"

# Make key functions available at package level
from .data_processing import load_medical_data, preprocess_text
from .model_utils import get_device_info
from .evaluation_utils import calculate_comprehensive_metrics
from .visualization import plot_confusion_matrices, plot_roc_curves

__all__ = [
    "load_medical_data",
    "preprocess_text", 
    "get_device_info",
    "calculate_comprehensive_metrics",
    "plot_confusion_matrices",
    "plot_roc_curves"
]
