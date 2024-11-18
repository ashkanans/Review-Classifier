from typing import Optional, Dict

import torch
from torch import nn


class BinaryClassificationModel(nn.Module):
    """A model for binary classification with a sigmoid activation."""

    def __init__(self, n_features: int, n_hidden: int):
        super().__init__()
        self.lin1 = nn.Linear(n_features, n_hidden)
        self.lin2 = nn.Linear(n_hidden, 1)
        self.loss_fn = nn.BCELoss()

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        out = self.lin1(x)
        out = torch.relu(out)
        out = self.lin2(out).squeeze(1)
        out = torch.sigmoid(out)

        result = {'pred': out}
        if y is not None:
            loss = self.loss(out, y)
            result['loss'] = loss

        return result

    def loss(self, pred, y):
        return self.loss_fn(pred, y)
