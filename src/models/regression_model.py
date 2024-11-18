from typing import Optional, Dict

import torch
from torch import nn


class RegressionModel(nn.Module):
    """A simple feed-forward neural network for regression tasks."""

    def __init__(self, n_features: int, n_hidden: int):
        super().__init__()
        self.lin1 = nn.Linear(n_features, n_hidden)
        self.lin2 = nn.Linear(n_hidden, 1)
        self.loss_fn = nn.MSELoss()

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        out = torch.relu(self.lin1(x))
        out = self.lin2(out).squeeze(1)
        result = {'pred': out}
        if y is not None:
            result['loss'] = self.loss(out, y)
        return result

    def loss(self, pred, y):
        return self.loss_fn(pred, y)
