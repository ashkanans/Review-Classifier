# src/data/dataset.py

import torch
from torch.utils.data import Dataset
from typing import Callable

class ReviewDataset(Dataset):
    """A PyTorch Dataset class for loading and vectorizing review data."""

    def __init__(self, dataset_path: str, feature_extraction_function: Callable[[str], torch.Tensor]):
        self.dataset_path = dataset_path
        self.feature_extraction_function = feature_extraction_function
        self._load_data()

    def _load_data(self):
        """Loads and processes data from a TSV file."""
        self.samples = []
        with open(self.dataset_path, 'r', encoding="UTF-8") as f:
            for line in f:
                star, review = line.strip().split('\t')
                # Convert review to vector and cast star rating to an integer tensor
                self.samples.append((self.feature_extraction_function(review), torch.tensor(int(star))))

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx):
        """Returns the idx-th sample."""
        return self.samples[idx]
