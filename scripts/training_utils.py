"""
Training utilities for medical text classification.

This module contains functions for training, evaluating, and managing
medical text classification models.
"""

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss
from sklearn.model_selection import StratifiedKFold
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import json
import joblib
import logging
from pathlib import Path
from tqdm import tqdm

logger = logging.getLogger(__name__)


def train_bert_model(
    model: Any,
    train_loader: Any,
    val_loader: Any,
    num_epochs: int = 4,
    learning_rate: float = 2e-5,
    warmup_steps: int = 500,
    weight_decay: float = 0.01,
    device: torch.device = None,
    save_path: Optional[str] = None,
    patience: int = 3,
) -> Tuple[Any, Dict[str, List[float]]]:
    """
    Train a BERT model for medical text classification.

    Args:
        model: The model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        warmup_steps: Number of warmup steps
        weight_decay: Weight decay for optimizer
        device: Device to train on
        save_path: Path to save best model
        patience: Early stopping patience

    Returns:
        Tuple of (trained_model, training_history)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    # Setup optimizer and scheduler
    optimizer = AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay, eps=1e-8
    )

    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_f1_macro": [],
        "val_f1_weighted": [],
        "learning_rate": [],
    }

    best_f1 = 0
    patience_counter = 0

    logger.info(f"Starting training for {num_epochs} epochs on {device}")

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]")
        for batch in train_pbar:
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, labels)
            loss = outputs["loss"]

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation phase
        val_loss, val_metrics = evaluate_model(model, val_loader, device)

        # Update history
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1_macro"].append(val_metrics["f1_macro"])
        history["val_f1_weighted"].append(val_metrics["f1_weighted"])
        history["learning_rate"].append(scheduler.get_last_lr()[0])

        # Logging
        logger.info(
            f"Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val F1 Macro: {val_metrics['f1_macro']:.4f}"
        )

        # Early stopping check
        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            patience_counter = 0

            # Save best model
            if save_path:
                torch.save(model.state_dict(), save_path)
                logger.info(f"Saved best model with F1: {best_f1:.4f}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.info(f"Early stopping triggered after {epoch + 1} epochs")
            break

    return model, history


def evaluate_model(
    model: Any, data_loader: Any, device: torch.device, threshold: float = 0.5
) -> Tuple[float, Dict[str, float]]:
    """
    Evaluate a model on a dataset.

    Args:
        model: Model to evaluate
        data_loader: Data loader for evaluation
        device: Device to evaluate on
        threshold: Classification threshold

    Returns:
        Tuple of (average_loss, metrics_dict)
    """
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask, labels)
            loss = outputs["loss"]
            logits = outputs["logits"]

            total_loss += loss.item()

            # Convert to probabilities and predictions
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).cpu().numpy()

            all_predictions.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader)

    # Calculate metrics
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    metrics = {
        "f1_macro": f1_score(all_labels, all_predictions, average="macro"),
        "f1_micro": f1_score(all_labels, all_predictions, average="micro"),
        "f1_weighted": f1_score(all_labels, all_predictions, average="weighted"),
        "precision_macro": precision_score(
            all_labels, all_predictions, average="macro"
        ),
        "recall_macro": recall_score(all_labels, all_predictions, average="macro"),
        "hamming_loss": hamming_loss(all_labels, all_predictions),
    }

    return avg_loss, metrics


def cross_validate_model(
    texts: List[str],
    labels: np.ndarray,
    model_class: Any,
    model_params: Dict[str, Any],
    tokenizer: Any,
    n_folds: int = 5,
    batch_size: int = 16,
    num_epochs: int = 3,
    random_state: int = 42,
) -> Dict[str, List[float]]:
    """
    Perform cross-validation for model evaluation.

    Args:
        texts: Input texts
        labels: Binary labels
        model_class: Model class to instantiate
        model_params: Parameters for model initialization
        tokenizer: Tokenizer for text encoding
        n_folds: Number of CV folds
        batch_size: Batch size
        num_epochs: Number of epochs per fold
        random_state: Random seed

    Returns:
        Dictionary with CV metrics
    """
    from .model_utils import create_data_loaders

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    # Convert multilabel to single label for stratification
    label_sums = labels.sum(axis=1)

    cv_results = {"f1_macro": [], "f1_weighted": [], "hamming_loss": [], "val_loss": []}

    for fold, (train_idx, val_idx) in enumerate(skf.split(texts, label_sums)):
        logger.info(f"Training fold {fold + 1}/{n_folds}")

        # Split data
        train_texts = [texts[i] for i in train_idx]
        val_texts = [texts[i] for i in val_idx]
        train_labels = labels[train_idx]
        val_labels = labels[val_idx]

        # Create data loaders
        train_loader, val_loader = create_data_loaders(
            train_texts, train_labels, val_texts, val_labels, tokenizer, batch_size
        )

        # Initialize model
        model = model_class(**model_params)

        # Train model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, _ = train_bert_model(
            model, train_loader, val_loader, num_epochs=num_epochs, device=device
        )

        # Evaluate
        val_loss, metrics = evaluate_model(model, val_loader, device)

        # Store results
        cv_results["f1_macro"].append(metrics["f1_macro"])
        cv_results["f1_weighted"].append(metrics["f1_weighted"])
        cv_results["hamming_loss"].append(metrics["hamming_loss"])
        cv_results["val_loss"].append(val_loss)

        # Clean up
        del model
        torch.cuda.empty_cache()

    # Calculate statistics
    for metric in cv_results:
        values = cv_results[metric]
        cv_results[f"{metric}_mean"] = np.mean(values)
        cv_results[f"{metric}_std"] = np.std(values)

    return cv_results


def save_model_artifacts(
    model: Any,
    tokenizer: Any,
    label_binarizer: Any,
    metrics: Dict[str, float],
    save_dir: str,
    config: Optional[Dict[str, Any]] = None,
):
    """
    Save all model artifacts for deployment.

    Args:
        model: Trained model
        tokenizer: Tokenizer
        label_binarizer: Label binarizer
        metrics: Model metrics
        save_dir: Directory to save artifacts
        config: Model configuration
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Save model state dict
    torch.save(model.state_dict(), save_path / "model.pt")

    # Save tokenizer
    tokenizer.save_pretrained(save_path)

    # Save label binarizer
    joblib.dump(label_binarizer, save_path / "mlb.pkl")

    # Save metrics
    with open(save_path / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save config if provided
    if config:
        with open(save_path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

    logger.info(f"Model artifacts saved to {save_path}")


def load_model_artifacts(
    model_class: Any, load_dir: str, device: torch.device = None
) -> Tuple[Any, Any, Any, Dict[str, float]]:
    """
    Load all model artifacts for inference.

    Args:
        model_class: Model class to instantiate
        load_dir: Directory containing artifacts
        device: Device to load model on

    Returns:
        Tuple of (model, tokenizer, label_binarizer, metrics)
    """
    load_path = Path(load_dir)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config
    with open(load_path / "config.json", "r") as f:
        config = json.load(f)

    # Initialize model
    model = model_class(**config)

    # Load model state dict
    model.load_state_dict(torch.load(load_path / "model.pt", map_location=device))
    model.to(device)
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(load_path)

    # Load label binarizer
    label_binarizer = joblib.load(load_path / "mlb.pkl")

    # Load metrics
    with open(load_path / "metrics.json", "r") as f:
        metrics = json.load(f)

    logger.info(f"Model artifacts loaded from {load_path}")

    return model, tokenizer, label_binarizer, metrics


def optimize_batch_size(
    model: Any, sample_loader: Any, device: torch.device, max_batch_size: int = 64
) -> int:
    """
    Find optimal batch size that fits in memory.

    Args:
        model: Model to test
        sample_loader: Sample data loader
        device: Device to test on
        max_batch_size: Maximum batch size to test

    Returns:
        Optimal batch size
    """
    model.to(device)
    model.eval()

    optimal_batch_size = 1

    for batch_size in [2, 4, 8, 16, 32, 64]:
        if batch_size > max_batch_size:
            break

        try:
            # Get a sample batch
            sample_batch = next(iter(sample_loader))

            # Simulate batch of target size
            input_ids = sample_batch["input_ids"][:batch_size].to(device)
            attention_mask = sample_batch["attention_mask"][:batch_size].to(device)

            with torch.no_grad():
                _ = model(input_ids, attention_mask)

            optimal_batch_size = batch_size
            logger.info(f"Batch size {batch_size} works")

        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.warning(f"Batch size {batch_size} caused OOM")
                break
            else:
                raise e

    logger.info(f"Optimal batch size: {optimal_batch_size}")
    return optimal_batch_size


def create_optimizer_scheduler(
    model: Any,
    train_loader: Any,
    num_epochs: int,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    scheduler_type: str = "linear",
) -> Tuple[Any, Any]:
    """
    Create optimizer and learning rate scheduler.

    Args:
        model: Model to optimize
        train_loader: Training data loader
        num_epochs: Number of training epochs
        learning_rate: Initial learning rate
        weight_decay: Weight decay
        warmup_ratio: Warmup ratio
        scheduler_type: Type of scheduler ("linear" or "cosine")

    Returns:
        Tuple of (optimizer, scheduler)
    """
    # Optimizer
    optimizer = AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay, eps=1e-8
    )

    # Scheduler
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)

    if scheduler_type == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
    elif scheduler_type == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    return optimizer, scheduler


def gradual_unfreezing(
    model: Any,
    train_loader: Any,
    val_loader: Any,
    device: torch.device,
    num_layers: int = 12,
    epochs_per_stage: int = 1,
) -> Tuple[Any, Dict[str, List[float]]]:
    """
    Train model with gradual unfreezing of layers.

    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on
        num_layers: Number of transformer layers
        epochs_per_stage: Epochs to train each stage

    Returns:
        Tuple of (trained_model, training_history)
    """
    from .model_utils import freeze_bert_layers, unfreeze_all_layers

    history = {"train_loss": [], "val_loss": [], "val_f1_macro": [], "stage": []}

    # Stage 1: Train only classifier
    logger.info("Stage 1: Training classifier only")
    freeze_bert_layers(model, num_layers)

    model, stage_history = train_bert_model(
        model, train_loader, val_loader, num_epochs=epochs_per_stage, device=device
    )

    # Update history
    for key in ["train_loss", "val_loss", "val_f1_macro"]:
        history[key].extend(stage_history[key])
    history["stage"].extend([1] * len(stage_history["train_loss"]))

    # Stage 2: Unfreeze top layers
    logger.info("Stage 2: Unfreezing top layers")
    freeze_bert_layers(model, num_layers // 2)

    model, stage_history = train_bert_model(
        model,
        train_loader,
        val_loader,
        num_epochs=epochs_per_stage,
        device=device,
        learning_rate=1e-5,  # Lower learning rate
    )

    # Update history
    for key in ["train_loss", "val_loss", "val_f1_macro"]:
        history[key].extend(stage_history[key])
    history["stage"].extend([2] * len(stage_history["train_loss"]))

    # Stage 3: Fine-tune all layers
    logger.info("Stage 3: Fine-tuning all layers")
    unfreeze_all_layers(model)

    model, stage_history = train_bert_model(
        model,
        train_loader,
        val_loader,
        num_epochs=epochs_per_stage,
        device=device,
        learning_rate=5e-6,  # Even lower learning rate
    )

    # Update history
    for key in ["train_loss", "val_loss", "val_f1_macro"]:
        history[key].extend(stage_history[key])
    history["stage"].extend([3] * len(stage_history["train_loss"]))

    return model, history
