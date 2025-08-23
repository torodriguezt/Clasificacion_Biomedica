"""
Visualization utilities for medical text classification.

This module provides plotting and visualization functions for analyzing
medical text classification models and datasets.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import warnings

warnings.filterwarnings("ignore")

# Set style for better plots
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


def plot_label_distribution(
    df: pd.DataFrame,
    label_columns: List[str],
    title: str = "Label Distribution",
    figsize: Tuple[int, int] = (12, 6),
) -> None:
    """
    Plot the distribution of labels in the dataset.

    Args:
        df: DataFrame containing labels
        label_columns: Names of label columns
        title: Plot title
        figsize: Figure size
    """
    # Calculate label counts
    label_counts = df[label_columns].sum().sort_values(ascending=True)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Bar plot
    colors = sns.color_palette("viridis", len(label_counts))
    bars = ax1.barh(range(len(label_counts)), label_counts.values, color=colors)
    ax1.set_yticks(range(len(label_counts)))
    ax1.set_yticklabels(label_counts.index)
    ax1.set_xlabel("Number of Samples")
    ax1.set_title("Absolute Counts")

    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, label_counts.values)):
        ax1.text(
            value + max(label_counts) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{value}",
            va="center",
            fontweight="bold",
        )

    # Pie chart
    ax2.pie(
        label_counts.values,
        labels=label_counts.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=colors,
    )
    ax2.set_title("Relative Distribution")

    plt.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_text_length_distribution(
    df: pd.DataFrame,
    text_column: str = "text",
    title: str = "Text Length Distribution",
    figsize: Tuple[int, int] = (15, 5),
) -> None:
    """
    Plot the distribution of text lengths.

    Args:
        df: DataFrame containing text
        text_column: Name of text column
        title: Plot title
        figsize: Figure size
    """
    # Calculate text lengths
    text_lengths = df[text_column].astype(str).str.len()
    word_counts = df[text_column].astype(str).str.split().str.len()

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Character length distribution
    axes[0].hist(text_lengths, bins=50, alpha=0.7, color="skyblue", edgecolor="black")
    axes[0].axvline(text_lengths.mean(), color="red", linestyle="--", linewidth=2)
    axes[0].set_xlabel("Character Count")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Character Length Distribution")
    axes[0].text(
        0.7,
        0.9,
        f"Mean: {text_lengths.mean():.0f}",
        transform=axes[0].transAxes,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Word count distribution
    axes[1].hist(word_counts, bins=50, alpha=0.7, color="lightgreen", edgecolor="black")
    axes[1].axvline(word_counts.mean(), color="red", linestyle="--", linewidth=2)
    axes[1].set_xlabel("Word Count")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Word Count Distribution")
    axes[1].text(
        0.7,
        0.9,
        f"Mean: {word_counts.mean():.0f}",
        transform=axes[1].transAxes,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Box plot
    box_data = [text_lengths, word_counts]
    box_labels = ["Characters", "Words"]
    bp = axes[2].boxplot(box_data, labels=box_labels, patch_artist=True)
    colors = ["skyblue", "lightgreen"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    axes[2].set_ylabel("Count")
    axes[2].set_title("Length Statistics")

    plt.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_roc_curves(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    class_names: List[str],
    title: str = "ROC Curves by Class",
    figsize: Tuple[int, int] = (12, 8),
) -> None:
    """
    Plot ROC curves for each class in multilabel classification.

    Args:
        y_true: Ground truth binary labels
        y_probs: Predicted probabilities
        class_names: Names of classes
        title: Plot title
        figsize: Figure size
    """
    n_classes = len(class_names)
    colors = sns.color_palette("husl", n_classes)

    plt.figure(figsize=figsize)

    # Plot ROC curve for each class
    for i, (class_name, color) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)

        plt.plot(
            fpr,
            tpr,
            color=color,
            lw=2,
            label=f"{class_name} (AUC = {roc_auc:.3f})",
        )

    # Plot diagonal line
    plt.plot([0, 1], [0, 1], "k--", lw=2, alpha=0.5)

    # Customize plot
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)

    # Add macro-average ROC curve
    all_fpr = np.unique(
        np.concatenate(
            [roc_curve(y_true[:, i], y_probs[:, i])[0] for i in range(n_classes)]
        )
    )
    mean_tpr = np.zeros_like(all_fpr)

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_probs[:, i])
        mean_tpr += np.interp(all_fpr, fpr, tpr)

    mean_tpr /= n_classes
    mean_auc = auc(all_fpr, mean_tpr)

    plt.plot(
        all_fpr,
        mean_tpr,
        color="navy",
        linestyle=":",
        linewidth=3,
        label=f"Macro-average (AUC = {mean_auc:.3f})",
    )

    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.show()


def plot_precision_recall_curves(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    class_names: List[str],
    title: str = "Precision-Recall Curves",
    figsize: Tuple[int, int] = (12, 8),
) -> None:
    """
    Plot precision-recall curves for each class.

    Args:
        y_true: Ground truth binary labels
        y_probs: Predicted probabilities
        class_names: Names of classes
        title: Plot title
        figsize: Figure size
    """
    n_classes = len(class_names)
    colors = sns.color_palette("husl", n_classes)

    plt.figure(figsize=figsize)

    for i, (class_name, color) in enumerate(zip(class_names, colors)):
        precision, recall, _ = precision_recall_curve(y_true[:, i], y_probs[:, i])
        avg_precision = np.trapz(precision, recall)

        plt.plot(
            recall,
            precision,
            color=color,
            lw=2,
            label=f"{class_name} (AP = {avg_precision:.3f})",
        )

    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_training_history(
    history: Dict[str, List[float]],
    title: str = "Training History",
    figsize: Tuple[int, int] = (15, 5),
) -> None:
    """
    Plot training history metrics.

    Args:
        history: Dictionary with training metrics
        title: Plot title
        figsize: Figure size
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Loss plot
    axes[0].plot(
        epochs, history["train_loss"], "b-", label="Training Loss", linewidth=2
    )
    if "val_loss" in history:
        axes[0].plot(
            epochs, history["val_loss"], "r-", label="Validation Loss", linewidth=2
        )
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # F1 Score plot
    if "val_f1_macro" in history:
        axes[1].plot(
            epochs, history["val_f1_macro"], "g-", label="F1 Macro", linewidth=2
        )
    if "val_f1_weighted" in history:
        axes[1].plot(
            epochs, history["val_f1_weighted"], "m-", label="F1 Weighted", linewidth=2
        )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("F1 Score")
    axes[1].set_title("Validation F1 Scores")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Learning rate plot
    if "learning_rate" in history:
        axes[2].plot(epochs, history["learning_rate"], "orange", linewidth=2)
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Learning Rate")
        axes[2].set_title("Learning Rate Schedule")
        axes[2].set_yscale("log")
        axes[2].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_class_performance_comparison(
    metrics_dict: Dict[str, Dict[str, float]],
    title: str = "Class Performance Comparison",
    figsize: Tuple[int, int] = (12, 8),
) -> None:
    """
    Plot performance comparison across classes.

    Args:
        metrics_dict: Dictionary with per-class metrics
        title: Plot title
        figsize: Figure size
    """
    classes = list(metrics_dict.keys())
    metrics = ["f1", "precision", "recall"]

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    # Extract data for plotting
    data = {
        metric: [metrics_dict[cls][metric] for cls in classes] for metric in metrics
    }

    # Individual metric plots
    for i, metric in enumerate(metrics):
        colors = sns.color_palette("viridis", len(classes))
        bars = axes[i].bar(classes, data[metric], color=colors)
        axes[i].set_title(f"{metric.capitalize()} by Class")
        axes[i].set_ylabel(metric.capitalize())
        axes[i].tick_params(axis="x", rotation=45)

        # Add value labels on bars
        for bar, value in zip(bars, data[metric]):
            axes[i].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

    # Combined radar plot
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    ax = plt.subplot(2, 2, 4, projection="polar")
    colors = sns.color_palette("husl", len(classes))

    for i, cls in enumerate(classes):
        values = [metrics_dict[cls][metric] for metric in metrics]
        values += values[:1]  # Complete the circle

        ax.plot(angles, values, "o-", linewidth=2, label=cls, color=colors[i])
        ax.fill(angles, values, alpha=0.25, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.set_title("Performance Radar Chart")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    plt.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(
    df: pd.DataFrame,
    columns: List[str],
    title: str = "Feature Correlation Matrix",
    figsize: Tuple[int, int] = (10, 8),
) -> None:
    """
    Plot correlation matrix for specified columns.

    Args:
        df: DataFrame containing data
        columns: Columns to include in correlation
        title: Plot title
        figsize: Figure size
    """
    correlation_matrix = df[columns].corr()

    plt.figure(figsize=figsize)
    mask = np.triu(correlation_matrix)

    sns.heatmap(
        correlation_matrix,
        mask=mask,
        annot=True,
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )

    plt.title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


def save_plot(filename: str, dpi: int = 300, bbox_inches: str = "tight") -> None:
    """
    Save the current plot to file.

    Args:
        filename: Output filename
        dpi: Resolution in dots per inch
        bbox_inches: Bounding box inches
    """
    plt.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)
    print(f"Plot saved as {filename}")


def create_results_summary_plot(
    results: Dict[str, Any],
    title: str = "Model Performance Summary",
    figsize: Tuple[int, int] = (16, 10),
) -> None:
    """
    Create a comprehensive summary plot of model results.

    Args:
        results: Dictionary containing model results
        title: Plot title
        figsize: Figure size
    """
    plt.figure(figsize=figsize)

    # Overall metrics (top left)
    ax1 = plt.subplot(2, 3, 1)
    metrics = ["f1_macro", "f1_weighted", "f1_micro", "hamming_loss"]
    values = [results.get(metric, 0) for metric in metrics]
    colors = ["#3498DB", "#2ECC71", "#F39C12", "#E74C3C"]

    bars = ax1.bar(metrics, values, color=colors, alpha=0.8)
    ax1.set_title("Overall Metrics")
    ax1.set_ylabel("Score")
    ax1.tick_params(axis="x", rotation=45)

    for bar, value in zip(bars, values):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Per-class F1 scores (top middle)
    if "per_class_metrics" in results:
        ax2 = plt.subplot(2, 3, 2)
        per_class = results["per_class_metrics"]
        classes = list(per_class.keys())
        f1_scores = [per_class[cls]["f1"] for cls in classes]

        bars = ax2.bar(
            classes, f1_scores, color=sns.color_palette("viridis", len(classes))
        )
        ax2.set_title("F1 Score by Class")
        ax2.set_ylabel("F1 Score")
        ax2.tick_params(axis="x", rotation=45)

        for bar, value in zip(bars, f1_scores):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

    # Model info (top right)
    ax3 = plt.subplot(2, 3, 3)
    ax3.axis("off")
    info_text = f"""
    Model Performance Summary

    Total Samples: {results.get('total_samples', 'N/A')}
    Exact Match Ratio: {results.get('exact_match_ratio', 0):.3f}
    Average Labels/Sample: {results.get('avg_labels_true', 0):.2f}

    Best Metrics:
    • F1 Macro: {results.get('f1_macro', 0):.3f}
    • F1 Weighted: {results.get('f1_weighted', 0):.3f}
    • Hamming Loss: {results.get('hamming_loss', 0):.3f}
    """

    ax3.text(
        0.1,
        0.9,
        info_text,
        transform=ax3.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
    )

    plt.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.show()
