"""LSTM classifier for binary toxic-chat detection."""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

import config


class ToxicChatLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = config.EMBED_DIM,
        hidden_dim: int = config.HIDDEN_DIM,
        num_layers: int = config.NUM_LAYERS,
        dropout: float = config.DROPOUT,
    ):
        super().__init__()
        # padding_idx zeros the <pad> embedding and freezes its gradient so
        # padding never influences what the model learns.
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            # Inter-layer dropout is only meaningful with a stacked LSTM.
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        # Packing prevents right-padding from overwriting the message summary.
        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden, _) = self.lstm(packed)
        # Return logits for numerically stable BCEWithLogitsLoss. Callers apply
        # sigmoid only when probabilities are needed for metrics or inference.
        return self.fc(self.dropout(hidden[-1])).squeeze(-1)
