# src/train/train.py

from typing import Callable, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.binary_classification_model import BinaryClassificationModel
from src.models.multi_class_classification_model import MultiClassClassificationModel
from src.models.regression_model import RegressionModel


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    adapt: Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
    epochs: int = 5
):
    """Training loop for the models."""
    for epoch in range(epochs):
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for x, y in progress_bar:
            x, y = adapt(x, y)
            optimizer.zero_grad()
            batch_out = model(x, y)
            loss = batch_out['loss']
            loss.backward()
            optimizer.step()
            progress_bar.set_postfix(loss=loss.item())
    print("Training completed.")

def evaluate_model(model: torch.nn.Module, test_loader: DataLoader, adapt: Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]) -> float:
    """Evaluate the models and return accuracy on test data."""
    model.eval()
    n_correct, n_total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = adapt(x, y)
            output = model(x)['pred']
            predictions = torch.round(output) if len(output.shape) == 1 else output.argmax(dim=1)
            n_correct += (predictions == y).sum().item()
            n_total += y.size(0)
    accuracy = n_correct / n_total
    print(f"Test Accuracy: {accuracy:.2f}")
    return accuracy

def initialize_model(model_type: str, n_features: int, n_hidden: int) -> torch.nn.Module:
    """Initialize a specific model type based on input model_type."""
    if model_type == "regression":
        return RegressionModel(n_features, n_hidden)
    elif model_type == "binary_classification":
        return BinaryClassificationModel(n_features, n_hidden)
    elif model_type == "multi_class_classification":
        return MultiClassClassificationModel(n_features, n_hidden)
    else:
        raise ValueError("Invalid model type provided.")

