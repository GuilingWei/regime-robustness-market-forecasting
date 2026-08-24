"""
MLPLOB: a simple MLP-based architecture adapted from LOB forecasting literature
(TLOB paper) to OHLCV-derived features instead of raw order-book input.
"""

import torch.nn as nn


class MLPLOB(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 1):
        super().__init__()
        raise NotImplementedError
