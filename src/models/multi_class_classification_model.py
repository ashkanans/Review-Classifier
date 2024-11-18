from typing import Optional, Dict

import torch
from torch import nn


class MultiClassClassificationModel(nn.Module):
    """A model for multi-class classification with softmax activation."""

    def __init__(self, n_features: int, n_hidden: int, n_classes: int = 5):
        super().__init__()
        self.lin1 = nn.Linear(n_features, n_hidden)
        self.lin2 = nn.Linear(n_hidden, n_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        out = self.lin1(x)
        out = torch.relu(out)
        out = self.lin2(out).squeeze(1)

        logits = out
        pred = torch.softmax(out, dim=-1)

        result = {'logits': logits, 'pred': pred}
        if y is not None:
            loss = self.loss(logits, y)
            result['loss'] = loss

        return result

    def loss(self, logits, y):
        return self.loss_fn(logits, y)
