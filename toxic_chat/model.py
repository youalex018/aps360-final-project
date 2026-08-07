"""LSTM classifier for binary toxic-chat detection."""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from toxic_chat import config


class ToxicChatLSTM(nn.Module):
    """Configurable packed-LSTM toxic-chat classifier."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = config.EMBED_DIM,
        hidden_dim: int = config.HIDDEN_DIM,
        num_layers: int = config.NUM_LAYERS,
        dropout: float = config.DROPOUT,
        *,
        bidirectional: bool = False,
        pooling: str = "last",
        pretrained_embeddings: torch.Tensor | None = None,
        freeze_embeddings: bool = False,
    ):
        super().__init__()
        if pooling not in {"last", "mean_max"}:
            raise ValueError(f"Unsupported pooling mode: {pooling}")
        self.bidirectional = bidirectional
        self.pooling = pooling
        # padding_idx zeros the <pad> embedding and freezes its gradient so
        # padding never influences what the model learns.
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            if tuple(pretrained_embeddings.shape) != (vocab_size, embed_dim):
                raise ValueError(
                    "Pretrained embedding shape must match "
                    f"({vocab_size}, {embed_dim})."
                )
            with torch.no_grad():
                self.embedding.weight.copy_(pretrained_embeddings)
            self.embedding.weight.requires_grad = not freeze_embeddings
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            # Inter-layer dropout is only meaningful with a stacked LSTM.
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(dropout)
        directions = 2 if bidirectional else 1
        classifier_dim = hidden_dim * directions
        if pooling == "mean_max":
            classifier_dim *= 2
        self.fc = nn.Linear(classifier_dim, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        # Packing prevents right-padding from overwriting the message summary.
        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, (hidden, _) = self.lstm(packed)
        if self.pooling == "mean_max":
            output, _ = pad_packed_sequence(
                packed_output,
                batch_first=True,
                total_length=x.size(1),
            )
            positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
            mask = positions < lengths.to(x.device).unsqueeze(1)
            float_mask = mask.unsqueeze(-1).to(output.dtype)
            mean_pool = (output * float_mask).sum(dim=1) / float_mask.sum(
                dim=1
            ).clamp_min(1.0)
            max_pool = output.masked_fill(~mask.unsqueeze(-1), -torch.inf).max(
                dim=1
            ).values
            features = torch.cat([mean_pool, max_pool], dim=1)
        elif self.bidirectional:
            hidden = hidden.view(
                self.lstm.num_layers,
                2,
                x.size(0),
                self.lstm.hidden_size,
            )
            features = torch.cat([hidden[-1, 0], hidden[-1, 1]], dim=1)
        else:
            features = hidden[-1]
        # Return logits for numerically stable BCEWithLogitsLoss. Callers apply
        # sigmoid only when probabilities are needed for metrics or inference.
        return self.fc(self.dropout(features)).squeeze(-1)
