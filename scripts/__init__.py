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
try:
    from .data_processing import load_medical_data, clean_medical_text, split_medical_data
    from .model_utils import MedicalDataset, create_data_loaders, calculate_class_weights
    from .evaluation_utils import compute_multilabel_metrics, find_optimal_thresholds, plot_confusion_matrices
    from .visualization import plot_roc_curves, plot_label_distribution, plot_text_length_distribution
    from .text_augmentation import augment_medical_texts
    
    __all__ = [
        "load_medical_data",
        "clean_medical_text",
        "split_medical_data",
        "MedicalDataset",
        "create_data_loaders", 
        "calculate_class_weights",
        "compute_multilabel_metrics",
        "find_optimal_thresholds",
        "plot_confusion_matrices",
        "plot_roc_curves",
        "plot_label_distribution",
        "plot_text_length_distribution",
        "augment_medical_texts"
    ]
except ImportError as e:
    print(f"Warning: Some dependencies not available. Install requirements.txt. Error: {e}")
    __all__ = []
