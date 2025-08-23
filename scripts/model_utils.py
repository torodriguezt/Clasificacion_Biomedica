"""
Model utilities for medical text classification.

This module contains the ImprovedMedicalBERT model implementation and
related utilities for biomedical text classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import logging

logger = logging.getLogger(__name__)


class MedicalDataset(Dataset):
    """Custom dataset for medical text classification."""

    def __init__(
        self,
        texts: List[str],
        labels: Optional[np.ndarray] = None,
        tokenizer: Any = None,
        max_length: int = 512,
    ):
        """
        Initialize the dataset.

        Args:
            texts: List of input texts
            labels: Array of binary labels (optional for inference)
            tokenizer: Tokenizer for text encoding
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])

        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
        }

        # Add labels if available
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)

        return item


class AttentionPooling(nn.Module):
    """Attention-based pooling mechanism for sequence representations."""

    def __init__(self, hidden_size: int):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply attention pooling to hidden states.

        Args:
            hidden_states: Hidden states from transformer [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Pooled representation [batch_size, hidden_size]
        """
        # Compute attention weights
        attention_weights = self.attention(hidden_states)  # [batch_size, seq_len, 1]
        attention_weights = attention_weights.squeeze(-1)  # [batch_size, seq_len]

        # Mask padding tokens
        attention_weights = attention_weights.masked_fill(
            attention_mask == 0, float("-inf")
        )

        # Apply softmax
        attention_weights = F.softmax(
            attention_weights, dim=-1
        )  # [batch_size, seq_len]
        attention_weights = self.dropout(attention_weights)

        # Apply attention weights
        pooled_output = torch.sum(
            hidden_states * attention_weights.unsqueeze(-1), dim=1
        )  # [batch_size, hidden_size]

        return pooled_output


class ImprovedMedicalBERT(nn.Module):
    """
    Improved BERT model for medical text classification.

    Features:
    - Attention-based pooling
    - Multiple dropout layers
    - Residual connections
    - Positive class weighting
    """

    def __init__(
        self,
        model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        num_labels: int = 4,
        pos_weight: Optional[List[float]] = None,
        dropout_rate: float = 0.3,
        use_attn: bool = True,
    ):
        """
        Initialize the model.

        Args:
            model_name: Name of the pre-trained model
            num_labels: Number of classification labels
            pos_weight: Positive class weights for imbalanced data
            dropout_rate: Dropout probability
            use_attn: Whether to use attention pooling
        """
        super(ImprovedMedicalBERT, self).__init__()

        self.num_labels = num_labels
        self.use_attn = use_attn

        # Load pre-trained model
        self.bert = AutoModel.from_pretrained(model_name)
        self.config = self.bert.config
        hidden_size = self.config.hidden_size

        # Pooling layer
        if use_attn:
            self.pooler = AttentionPooling(hidden_size)
        else:
            self.pooler = None

        # Classification layers
        self.dropout1 = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.dropout2 = nn.Dropout(dropout_rate * 0.5)

        # Loss function with class weighting
        if pos_weight is not None:
            pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float)
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        else:
            self.loss_fn = nn.BCEWithLogitsLoss()

        # Initialize classifier weights
        self._init_weights()

    def _init_weights(self):
        """Initialize classifier weights."""
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Target labels (optional)

        Returns:
            Dictionary containing logits and loss (if labels provided)
        """
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )

        # Pool representations
        if self.use_attn and self.pooler is not None:
            pooled_output = self.pooler(outputs.last_hidden_state, attention_mask)
        else:
            pooled_output = outputs.pooler_output

        # Apply dropout and classification
        pooled_output = self.dropout1(pooled_output)
        logits = self.classifier(pooled_output)
        logits = self.dropout2(logits)

        result = {"logits": logits}

        # Calculate loss if labels provided
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            result["loss"] = loss

        return result


class EnsembleClassifier:
    """Ensemble of multiple medical BERT models."""

    def __init__(self, models: List[ImprovedMedicalBERT]):
        """
        Initialize ensemble.

        Args:
            models: List of trained models
        """
        self.models = models
        self.device = next(models[0].parameters()).device

    def predict(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Make ensemble predictions.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask

        Returns:
            Averaged predictions
        """
        predictions = []

        for model in self.models:
            model.eval()
            with torch.no_grad():
                outputs = model(input_ids, attention_mask)
                probs = torch.sigmoid(outputs["logits"])
                predictions.append(probs)

        # Average predictions
        ensemble_pred = torch.stack(predictions).mean(dim=0)
        return ensemble_pred

    def to(self, device: torch.device):
        """Move ensemble to device."""
        for model in self.models:
            model.to(device)
        self.device = device
        return self


def create_data_loaders(
    train_texts: List[str],
    train_labels: np.ndarray,
    val_texts: List[str],
    val_labels: np.ndarray,
    tokenizer: Any,
    batch_size: int = 16,
    max_length: int = 512,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders.

    Args:
        train_texts: Training texts
        train_labels: Training labels
        val_texts: Validation texts
        val_labels: Validation labels
        tokenizer: Tokenizer for text encoding
        batch_size: Batch size
        max_length: Maximum sequence length
        num_workers: Number of data loading workers

    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = MedicalDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = MedicalDataset(val_texts, val_labels, tokenizer, max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader


def calculate_class_weights(labels: np.ndarray) -> List[float]:
    """
    Calculate positive class weights for imbalanced data.

    Args:
        labels: Binary label matrix [n_samples, n_classes]

    Returns:
        List of positive class weights
    """
    n_samples, n_classes = labels.shape
    pos_weights = []

    for i in range(n_classes):
        pos_count = labels[:, i].sum()
        neg_count = n_samples - pos_count

        if pos_count == 0:
            weight = 1.0
        else:
            weight = neg_count / pos_count

        pos_weights.append(weight)

    return pos_weights


def prepare_labels(
    df: pd.DataFrame, label_columns: List[str]
) -> Tuple[np.ndarray, MultiLabelBinarizer]:
    """
    Prepare labels for multilabel classification.

    Args:
        df: DataFrame containing labels
        label_columns: Names of label columns

    Returns:
        Tuple of (binary_labels, label_binarizer)
    """
    # Extract labels
    labels = df[label_columns].values.astype(int)

    # Create MultiLabelBinarizer for consistency
    mlb = MultiLabelBinarizer()
    mlb.fit([label_columns])
    mlb.classes_ = np.array(label_columns)

    return labels, mlb


def get_model_summary(model: ImprovedMedicalBERT) -> Dict[str, Any]:
    """
    Get summary information about the model.

    Args:
        model: The model to summarize

    Returns:
        Dictionary with model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_mb": total_params * 4 / (1024 * 1024),  # Assuming float32
        "num_labels": model.num_labels,
        "uses_attention_pooling": model.use_attn,
        "hidden_size": model.config.hidden_size,
        "num_layers": model.config.num_hidden_layers,
        "num_attention_heads": model.config.num_attention_heads,
    }


def freeze_bert_layers(model: ImprovedMedicalBERT, num_layers_to_freeze: int = 6):
    """
    Freeze bottom layers of BERT for transfer learning.

    Args:
        model: Model to modify
        num_layers_to_freeze: Number of bottom layers to freeze
    """
    # Freeze embeddings
    for param in model.bert.embeddings.parameters():
        param.requires_grad = False

    # Freeze bottom encoder layers
    for i in range(num_layers_to_freeze):
        for param in model.bert.encoder.layer[i].parameters():
            param.requires_grad = False

    logger.info(f"Frozen {num_layers_to_freeze} bottom layers of BERT")


def unfreeze_all_layers(model: ImprovedMedicalBERT):
    """
    Unfreeze all model parameters.

    Args:
        model: Model to modify
    """
    for param in model.parameters():
        param.requires_grad = True

    logger.info("Unfrozen all model layers")
