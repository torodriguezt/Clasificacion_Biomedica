#!/usr/bin/env python3
"""
Main entry point for the Medical Text Classification Scripts.

This module provides a command-line interface to execute various
data processing, model training, and evaluation tasks.

Usage:
    python -m scripts --help
    python -m scripts train --data-path data/challenge_data-18-ago.csv
    python -m scripts evaluate --model-path models/trained_model/
    python -m scripts visualize --results-path results/
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from scripts.data_processing import load_medical_data, clean_medical_text, split_medical_data
    from scripts.model_utils import MedicalDataset, create_data_loaders, calculate_class_weights
    from scripts.training_utils import train_model, evaluate_model
    from scripts.evaluation_utils import compute_multilabel_metrics, find_optimal_thresholds, plot_confusion_matrices
    from scripts.visualization import plot_roc_curves, plot_label_distribution, plot_text_length_distribution
    from scripts.text_augmentation import augment_medical_texts
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some dependencies not available. Install requirements.txt first. Error: {e}")
    DEPENDENCIES_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scripts.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def setup_parser() -> argparse.ArgumentParser:
    """Setup command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Medical Text Classification Scripts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Load and preprocess data
  python -m scripts preprocess --input data/challenge_data-18-ago.csv --output data/processed/

  # Train a new model
  python -m scripts train --data data/processed/ --output models/new_model/

  # Evaluate existing model
  python -m scripts evaluate --model models/trained_model/ --data data/processed/

  # Generate visualizations
  python -m scripts visualize --model models/trained_model/ --output results/plots/

  # Show device information
  python -m scripts device-info
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Preprocess command
    preprocess_parser = subparsers.add_parser('preprocess', help='Preprocess medical text data')
    preprocess_parser.add_argument('--input', '-i', required=True, help='Input CSV file path')
    preprocess_parser.add_argument('--output', '-o', required=True, help='Output directory')
    preprocess_parser.add_argument('--test-size', type=float, default=0.2, help='Test split ratio (default: 0.2)')
    preprocess_parser.add_argument('--random-state', type=int, default=42, help='Random state for reproducibility')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a medical classification model')
    train_parser.add_argument('--data', '-d', required=True, help='Processed data directory')
    train_parser.add_argument('--output', '-o', required=True, help='Model output directory')
    train_parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    train_parser.add_argument('--batch-size', type=int, default=16, help='Training batch size')
    train_parser.add_argument('--learning-rate', type=float, default=2e-5, help='Learning rate')
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate a trained model')
    evaluate_parser.add_argument('--model', '-m', required=True, help='Model directory path')
    evaluate_parser.add_argument('--data', '-d', required=True, help='Test data path')
    evaluate_parser.add_argument('--output', '-o', help='Output directory for results')
    
    # Visualize command
    visualize_parser = subparsers.add_parser('visualize', help='Generate visualization plots')
    visualize_parser.add_argument('--model', '-m', required=True, help='Model directory path')
    visualize_parser.add_argument('--data', '-d', help='Data path for analysis')
    visualize_parser.add_argument('--output', '-o', required=True, help='Output directory for plots')
    visualize_parser.add_argument('--plot-type', choices=['confusion', 'roc', 'metrics', 'all'], 
                                 default='all', help='Type of plots to generate')
    
    # Augment command
    augment_parser = subparsers.add_parser('augment', help='Augment medical text data')
    augment_parser.add_argument('--input', '-i', required=True, help='Input data file')
    augment_parser.add_argument('--output', '-o', required=True, help='Output augmented data file')
    augment_parser.add_argument('--augment-ratio', type=float, default=0.3, help='Augmentation ratio')
    
    # Device info command
    subparsers.add_parser('device-info', help='Show device and environment information')
    
    return parser


def cmd_preprocess(args) -> None:
    """Execute data preprocessing command."""
    logger.info(f"Loading data from {args.input}")
    df = load_medical_data(args.input)
    
    logger.info("Preprocessing text data")
    df = clean_medical_text(df)
    
    logger.info(f"Creating train/test split (test_size={args.test_size})")
    train_data, test_data = split_medical_data(
        df, test_size=args.test_size, random_state=args.random_state
    )
    
    # Save processed data
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    train_data.to_csv(output_path / 'train_data.csv', index=False)
    test_data.to_csv(output_path / 'test_data.csv', index=False)
    
    logger.info(f"Processed data saved to {output_path}")
    logger.info(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}")


def cmd_train(args) -> None:
    """Execute model training command."""
    logger.info(f"Training model with data from {args.data}")
    logger.info(f"Training parameters: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.learning_rate}")
    
    # Implementation would go here
    logger.warning("Training implementation not yet complete. Please use the Jupyter notebook for training.")


def cmd_evaluate(args) -> None:
    """Execute model evaluation command."""
    logger.info(f"Evaluating model from {args.model}")
    logger.info(f"Using test data from {args.data}")
    
    # Implementation would go here
    logger.warning("Evaluation implementation not yet complete. Please use the Jupyter notebook for evaluation.")


def cmd_visualize(args) -> None:
    """Execute visualization generation command."""
    logger.info(f"Generating {args.plot_type} plots")
    logger.info(f"Model: {args.model}, Output: {args.output}")
    
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if args.plot_type in ['confusion', 'all']:
        logger.info("Generating confusion matrix plots")
        # Implementation would call plot_confusion_matrices()
        
    if args.plot_type in ['roc', 'all']:
        logger.info("Generating ROC curve plots")
        # Implementation would call plot_roc_curves()
        
    if args.plot_type in ['metrics', 'all']:
        logger.info("Generating metrics comparison plots")
        # Implementation would call plot_label_distribution()
    
    logger.warning("Visualization implementation not yet complete. Please use the Jupyter notebook for plots.")


def cmd_augment(args) -> None:
    """Execute text augmentation command."""
    logger.info(f"Augmenting data from {args.input}")
    logger.info(f"Augmentation ratio: {args.augment_ratio}")
    
    # Implementation would go here
    logger.warning("Augmentation implementation not yet complete.")


def get_device_info() -> dict:
    """Get basic device and environment information."""
    import sys
    import platform
    
    info = {
        "Python Version": sys.version.split()[0],
        "Platform": platform.platform(),
    }
    
    try:
        import torch
        info["PyTorch Version"] = torch.__version__
        info["CUDA Available"] = torch.cuda.is_available()
        
        if torch.cuda.is_available():
            info["CUDA Version"] = torch.version.cuda
            info["GPU Count"] = torch.cuda.device_count()
            info["GPU Name"] = torch.cuda.get_device_name(0)
    except ImportError:
        info["PyTorch"] = "Not installed"
    
    return info


def cmd_device_info(args) -> None:
    """Show device and environment information."""
    device_info = get_device_info()
    
    print("\n" + "="*50)
    print("🖥️  DEVICE AND ENVIRONMENT INFORMATION")
    print("="*50)
    
    for key, value in device_info.items():
        print(f"{key}: {value}")
    
    print("="*50 + "\n")


def main():
    """Main entry point."""
    parser = setup_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Allow device-info command to work without full dependencies
    if args.command != 'device-info' and not DEPENDENCIES_AVAILABLE:
        logger.error("Required dependencies not available. Please install requirements: pip install -r requirements.txt")
        sys.exit(1)
    
    logger.info(f"Executing command: {args.command}")
    
    try:
        if args.command == 'preprocess':
            cmd_preprocess(args)
        elif args.command == 'train':
            cmd_train(args)
        elif args.command == 'evaluate':
            cmd_evaluate(args)
        elif args.command == 'visualize':
            cmd_visualize(args)
        elif args.command == 'augment':
            cmd_augment(args)
        elif args.command == 'device-info':
            cmd_device_info(args)
        else:
            logger.error(f"Unknown command: {args.command}")
            sys.exit(1)
            
        logger.info(f"Command '{args.command}' completed successfully")
        
    except Exception as e:
        logger.error(f"Error executing command '{args.command}': {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
