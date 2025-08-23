# improved_medical_bert.py
# ======================================================
# Implementación de ImprovedMedicalBERT
# Encoder Hugging Face + MultiheadAttention + MLP
# ======================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class ImprovedMedicalBERT(nn.Module):
    """
    Modelo customizado para clasificación multi-etiqueta biomédica.
    Combina un encoder BERT de Hugging Face con una capa de atención ligera
    y un clasificador MLP profundo.

    Retorna un diccionario con:
        {
            "loss": (opcional, si se pasan labels),
            "logits": tensor [batch, num_labels]
        }
    """

    def __init__(
        self,
        model_name: str,
        num_labels: int,
        pos_weight=None,
        dropout_rate: float = 0.3,
        use_attn: bool = True,
    ):
        super().__init__()

        # Cargar encoder base (ej: PubMedBERT)
        self.encoder = AutoModel.from_pretrained(model_name)

        # Desactivar salidas extra innecesarias
        if hasattr(self.encoder.config, "output_hidden_states"):
            self.encoder.config.output_hidden_states = False
        if hasattr(self.encoder.config, "output_attentions"):
            self.encoder.config.output_attentions = False
        if hasattr(self.encoder.config, "use_cache"):
            self.encoder.config.use_cache = False

        hidden = self.encoder.config.hidden_size  # ej: 768
        self.use_attn = use_attn
        self.dropout = nn.Dropout(dropout_rate)

        # Atención ligera
        if use_attn:
            self.attn = nn.MultiheadAttention(
                embed_dim=hidden, num_heads=8, dropout=dropout_rate, batch_first=True
            )
            self.norm1 = nn.LayerNorm(hidden)

        # Clasificador MLP
        self.fc1 = nn.Linear(hidden, hidden // 2)
        self.norm2 = nn.LayerNorm(hidden // 2)
        self.fc2 = nn.Linear(hidden // 2, hidden // 4)
        self.classifier = nn.Linear(hidden // 4, num_labels)

        # Ponderación opcional para BCEWithLogitsLoss
        if pos_weight is not None:
            self.register_buffer(
                "pos_weight", torch.as_tensor(pos_weight, dtype=torch.float32)
            )
        else:
            self.pos_weight = None

        # Inicialización de pesos
        self._init_weights()

    def _init_weights(self):
        for m in [self.fc1, self.fc2, self.classifier]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        """
        input_ids: [batch, seq_len]
        attention_mask: [batch, seq_len]
        token_type_ids: (opcional)
        labels: [batch, num_labels] (opcional)

        Retorna un dict con {"loss", "logits"}
        """
        enc = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )

        seq = enc.last_hidden_state  # [B, T, H]

        # Atención ligera
        if self.use_attn:
            kpm = ~attention_mask.bool() if attention_mask is not None else None
            attn_out, _ = self.attn(seq, seq, seq, key_padding_mask=kpm)
            seq = self.norm1(seq + attn_out)

        # Pooling (token [CLS])
        pooled = seq[:, 0, :]  # [B, H]

        # Clasificador
        x = self.dropout(pooled)
        x = F.relu(self.fc1(x))
        x = self.norm2(x)
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        logits = self.classifier(x)

        # Cálculo opcional de la loss
        loss = None
        if labels is not None:
            if self.pos_weight is not None:
                loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
            else:
                loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}
