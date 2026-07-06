"""LSTM classifier for binary toxic-chat detection."""
import torch
import torch.nn as nn

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
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        # hidden[-1] is the final layer's last-step state: the sequence summary.
        out = self.fc(self.dropout(hidden[-1]))
        # Squeeze the singleton output dim so probs align with [batch] labels.
        return self.sigmoid(out).squeeze(-1)
