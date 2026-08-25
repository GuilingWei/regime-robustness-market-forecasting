"""
MLPLOB: a simple MLP-based architecture, adapted from the LOB forecasting
literature (the TLOB paper's finding that a well-tuned MLP can match or beat
more complex sequence models) to OHLCV-derived tabular features rather than
raw order-book input.

Kept deliberately small given the limited training data available in the
calm regime window (~521 rows) -- a large network would almost certainly
overfit.
"""

import torch
import torch.nn as nn


class MLPLOB(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...] = (128,64),
                 dropout: float = 0.2):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
