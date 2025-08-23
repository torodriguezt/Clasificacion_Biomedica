"""
Evaluation utilities for biomedical text classification.

This module contains comprehensive evaluation functions and metrics
for multilabel medical text classification.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    hamming_loss,
    average_precision_score,
    roc_auc_score,
    multilabel_confusion_matrix,
    ConfusionMatrixDisplay,
)


def compute_multilabel_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    average_types: List[str] = ["micro", "macro", "weighted"],
) -> Dict[str, Any]:
    """
    Compute comprehensive multilabel classification metrics.

    Args:
        y_true: Ground truth binary labels
        y_pred: Predicted binary labels
        y_probs: Predicted probabilities (optional)
        class_names: Names of classes
        average_types: Types of averaging for metrics

    Returns:
        Dictionary containing all computed metrics
    """
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(y_true.shape[1])]

    metrics = {}

    # Basic multilabel metrics
    for avg in average_types:
        metrics[f"f1_{avg}"] = f1_score(y_true, y_pred, average=avg, zero_division=0)
        metrics[f"precision_{avg}"] = precision_score(
            y_true, y_pred, average=avg, zero_division=0
        )
        metrics[f"recall_{avg}"] = recall_score(
            y_true, y_pred, average=avg, zero_division=0
        )

    # Hamming loss (multilabel-specific)
    metrics["hamming_loss"] = hamming_loss(y_true, y_pred)

    # Exact match ratio (all labels correct)
    metrics["exact_match_ratio"] = (y_true == y_pred).all(axis=1).mean()

    # Label-based metrics
    metrics["avg_labels_true"] = y_true.sum(axis=1).mean()
    metrics["avg_labels_pred"] = y_pred.sum(axis=1).mean()
    metrics["label_coverage"] = (y_pred.sum(axis=0) > 0).mean()

    # Per-class metrics
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)

    metrics["per_class_metrics"] = {}
    for i, class_name in enumerate(class_names):
        metrics["per_class_metrics"][class_name] = {
            "f1": per_class_f1[i],
            "precision": per_class_precision[i],
            "recall": per_class_recall[i],
            "support": y_true[:, i].sum(),
        }

    # Probability-based metrics (if probabilities provided)
    if y_probs is not None:
        try:
            metrics["average_precision_macro"] = average_precision_score(
                y_true, y_probs, average="macro"
            )
            metrics["average_precision_micro"] = average_precision_score(
                y_true, y_probs, average="micro"
            )
        except Exception:
            metrics["average_precision_macro"] = 0.0
            metrics["average_precision_micro"] = 0.0

        try:
            metrics["roc_auc_macro"] = roc_auc_score(y_true, y_probs, average="macro")
            metrics["roc_auc_micro"] = roc_auc_score(y_true, y_probs, average="micro")
        except Exception:
            metrics["roc_auc_macro"] = 0.0
            metrics["roc_auc_micro"] = 0.0

    return metrics


def find_optimal_thresholds(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    metric: str = "f1_macro",
    threshold_range: Tuple[float, float] = (0.1, 0.9),
    step: float = 0.05,
) -> Dict[str, Any]:
    """
    Find optimal classification thresholds for multilabel classification.

    Args:
        y_true: Ground truth binary labels
        y_probs: Predicted probabilities
        metric: Metric to optimize
        threshold_range: Range of thresholds to test
        step: Step size for threshold search

    Returns:
        Dictionary with optimal thresholds and metrics
    """
    thresholds = np.arange(threshold_range[0], threshold_range[1] + step, step)

    best_threshold = 0.5
    best_score = -1.0
    threshold_results = []

    for threshold in thresholds:
        y_pred = (y_probs >= threshold).astype(int)

        # Calculate target metric
        if metric == "f1_macro":
            score = f1_score(y_true, y_pred, average="macro", zero_division=0)
        elif metric == "f1_weighted":
            score = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        elif metric == "f1_micro":
            score = f1_score(y_true, y_pred, average="micro", zero_division=0)
        elif metric == "hamming_loss":
            score = -hamming_loss(y_true, y_pred)  # Negative because lower is better
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        threshold_results.append(
            {
                "threshold": threshold,
                "score": score,
                "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
                "f1_weighted": f1_score(
                    y_true, y_pred, average="weighted", zero_division=0
                ),
                "hamming_loss": hamming_loss(y_true, y_pred),
            }
        )

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return {
        "best_threshold": best_threshold,
        "best_score": best_score,
        "all_results": threshold_results,
    }


def plot_confusion_matrices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrices by Class",
    figsize: Tuple[int, int] = None,
) -> None:
    """
    Plot confusion matrices for each class in multilabel classification.

    Args:
        y_true: Ground truth binary labels
        y_pred: Predicted binary labels
        class_names: Names of classes
        title: Plot title
        figsize: Figure size
    """
    n_classes = len(class_names)

    if figsize is None:
        cols = min(3, n_classes)
        rows = int(np.ceil(n_classes / cols))
        figsize = (5 * cols, 4 * rows)

    # Calculate confusion matrices
    confusion_matrices = multilabel_confusion_matrix(y_true, y_pred)

    # Create subplots
    cols = min(3, n_classes)
    rows = int(np.ceil(n_classes / cols))

    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    # Handle single subplot case
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    else:
        axes = axes.flatten()

    for i, (cm, class_name) in enumerate(zip(confusion_matrices, class_names)):
        if i >= len(axes):
            break

        # Calculate metrics for this class
        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        # Plot confusion matrix
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=[f"Not {class_name}", class_name]
        )
        disp.plot(ax=axes[i], cmap="Blues", colorbar=False, values_format="d")

        # Customize title with metrics
        axes[i].set_title(
            f"{class_name}\nF1: {f1:.3f} | P: {precision:.3f} | R: {recall:.3f}",
            fontsize=10,
            fontweight="bold",
        )

    # Remove empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.show()


def plot_threshold_analysis(
    threshold_results: List[Dict], title: str = "Threshold Analysis"
) -> None:
    """
    Plot threshold analysis results.

    Args:
        threshold_results: Results from threshold optimization
        title: Plot title
    """
    thresholds = [r["threshold"] for r in threshold_results]
    f1_macro = [r["f1_macro"] for r in threshold_results]
    f1_weighted = [r["f1_weighted"] for r in threshold_results]
    hamming_loss = [r["hamming_loss"] for r in threshold_results]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # F1 Macro
    axes[0].plot(thresholds, f1_macro, "b-", linewidth=2, marker="o", markersize=4)
    best_macro_idx = np.argmax(f1_macro)
    axes[0].axvline(thresholds[best_macro_idx], color="red", linestyle="--", alpha=0.7)
    axes[0].set_xlabel("Threshold")
    axes[0].set_ylabel("F1 Macro Score")
    axes[0].set_title("F1 Macro vs Threshold")
    axes[0].grid(True, alpha=0.3)
    axes[0].text(
        thresholds[best_macro_idx],
        f1_macro[best_macro_idx] + 0.01,
        f"Best: {thresholds[best_macro_idx]:.2f}",
        ha="center",
    )

    # F1 Weighted
    axes[1].plot(thresholds, f1_weighted, "g-", linewidth=2, marker="o", markersize=4)
    best_weighted_idx = np.argmax(f1_weighted)
    axes[1].axvline(
        thresholds[best_weighted_idx], color="red", linestyle="--", alpha=0.7
    )
    axes[1].set_xlabel("Threshold")
    axes[1].set_ylabel("F1 Weighted Score")
    axes[1].set_title("F1 Weighted vs Threshold")
    axes[1].grid(True, alpha=0.3)
    axes[1].text(
        thresholds[best_weighted_idx],
        f1_weighted[best_weighted_idx] + 0.01,
        f"Best: {thresholds[best_weighted_idx]:.2f}",
        ha="center",
    )

    # Hamming Loss
    axes[2].plot(thresholds, hamming_loss, "r-", linewidth=2, marker="o", markersize=4)
    best_hamming_idx = np.argmin(hamming_loss)
    axes[2].axvline(
        thresholds[best_hamming_idx], color="blue", linestyle="--", alpha=0.7
    )
    axes[2].set_xlabel("Threshold")
    axes[2].set_ylabel("Hamming Loss")
    axes[2].set_title("Hamming Loss vs Threshold")
    axes[2].grid(True, alpha=0.3)
    axes[2].text(
        thresholds[best_hamming_idx],
        hamming_loss[best_hamming_idx] + 0.001,
        f"Best: {thresholds[best_hamming_idx]:.2f}",
        ha="center",
    )

    fig.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.show()


def analyze_prediction_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    class_names: List[str],
    texts: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Analyze prediction errors to identify patterns.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_probs: Predicted probabilities
        class_names: Names of classes
        texts: Original texts (optional)

    Returns:
        Dictionary with error analysis
    """
    error_analysis = {}

    # Overall error statistics
    exact_matches = (y_true == y_pred).all(axis=1)
    error_analysis["exact_match_accuracy"] = exact_matches.mean()
    error_analysis["total_errors"] = (~exact_matches).sum()

    # Per-class error analysis
    per_class_errors = {}
    for i, class_name in enumerate(class_names):
        y_true_class = y_true[:, i]
        y_pred_class = y_pred[:, i]
        y_probs_class = y_probs[:, i]

        # False positives and negatives
        false_positives = (y_true_class == 0) & (y_pred_class == 1)
        false_negatives = (y_true_class == 1) & (y_pred_class == 0)

        # High confidence errors (problematic)
        fp_high_conf = false_positives & (y_probs_class > 0.8)
        fn_low_conf = false_negatives & (y_probs_class < 0.2)

        per_class_errors[class_name] = {
            "false_positives": false_positives.sum(),
            "false_negatives": false_negatives.sum(),
            "fp_high_confidence": fp_high_conf.sum(),
            "fn_low_confidence": fn_low_conf.sum(),
            "avg_prob_true_positives": (
                y_probs_class[y_true_class == 1].mean()
                if (y_true_class == 1).any()
                else 0
            ),
            "avg_prob_true_negatives": (
                (1 - y_probs_class[y_true_class == 0]).mean()
                if (y_true_class == 0).any()
                else 0
            ),
        }

    error_analysis["per_class_errors"] = per_class_errors

    return error_analysis


def plot_metrics_comparison(
    results_list: List[Dict],
    model_names: List[str],
    metrics_to_plot: List[str] = ["f1_weighted", "f1_macro", "hamming_loss"],
) -> None:
    """
    Plot comparison of metrics across different models.

    Args:
        results_list: List of metric dictionaries
        model_names: Names of models
        metrics_to_plot: Metrics to include in comparison
    """
    if len(results_list) != len(model_names):
        raise ValueError("Number of results must match number of model names")

    # Prepare data for plotting
    comparison_data = {}
    for metric in metrics_to_plot:
        comparison_data[metric] = [results.get(metric, 0) for results in results_list]

    # Create comparison plot
    n_metrics = len(metrics_to_plot)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6))

    if n_metrics == 1:
        axes = [axes]

    colors = ["#3498DB", "#E74C3C", "#2ECC71", "#F39C12", "#9B59B6"]

    for i, metric in enumerate(metrics_to_plot):
        values = comparison_data[metric]
        bars = axes[i].bar(
            model_names,
            values,
            color=[colors[j % len(colors)] for j in range(len(model_names))],
            alpha=0.8,
        )

        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[i].text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.001,
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        axes[i].set_title(metric.replace("_", " ").title(), fontweight="bold")
        axes[i].set_ylabel("Score")
        axes[i].tick_params(axis="x", rotation=45)
        axes[i].grid(axis="y", alpha=0.3)

        # Adjust y-axis for better visualization
        if metric == "hamming_loss":
            axes[i].set_ylim(0, max(values) * 1.2)
        else:
            axes[i].set_ylim(0, 1.0)

    plt.tight_layout()
    plt.show()
