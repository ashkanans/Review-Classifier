# src/models/models.py

import torch
from torch import nn
from typing import Optional, Dict

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


class BinaryClassificationModel(nn.Module):
    """A model for binary classification with a sigmoid activation."""
    def __init__(self, n_features: int, n_hidden: int):
        super().__init__()
        self.lin1 = nn.Linear(n_features, n_hidden)
        self.lin2 = nn.Linear(n_hidden, 1)
        self.loss_fn = torch.nn.BCELoss()

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:

        # actual forward
        out = self.lin1(x)
        out = torch.relu(out)
        out = self.lin2(out).squeeze(1) # removes the second dimension if it is of size 1
        # we need to apply a sigmoid activation function
        out = torch.sigmoid(out)

        result = {'pred': out}

        # compute loss
        if y is not None:
            loss = self.loss(out, y)
            result['loss'] = loss

        return result

    def loss(self, pred, y):
        return self.loss_fn(pred, y)


class MultiClassClassificationModel(nn.Module):
    """A model for multi-class classification with softmax activation."""
    def __init__(self, n_features: int, n_hidden: int, n_classes: int = 5):
        super().__init__()
        self.lin1 = nn.Linear(n_features, n_hidden)
        self.lin2 = nn.Linear(n_hidden, n_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        # actual forward
        out = self.lin1(x)
        out = torch.relu(out)
        out = self.lin2(out).squeeze(1)

        # compute logits (which are simply the out variable) and the actual probability distribution (pred, as it is the predicted distribution)
        logits = out
        pred = torch.softmax(out, dim=-1)

        result = {'logits': logits, 'pred': pred}

        # compute loss
        if y is not None:
            # while mathematically the CrossEntropyLoss takes as input the probability distributions,
            # torch optimizes its computation internally and takes as input the logits instead
            loss = self.loss(logits, y)
            result['loss'] = loss

        return result

    def loss(self, logits, y):
        return self.loss_fn(logits, y)
