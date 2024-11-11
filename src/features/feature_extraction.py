# src/features/feature_extraction.py

import torch

# Define marker sets for positive, neutral, and negative words
positive_markers = {'fantastic', 'amazing', 'excellent', 'very good'}
neutral_markers = {'adequate', 'fine but', 'good but', 'ok but'}
negative_markers = {'returning', 'sucks', 'waste'}

# Combined marker set
markers = positive_markers | neutral_markers | negative_markers
marker2idx = {marker: idx for idx, marker in enumerate(markers)}

def review2vector(review: str):
    """Converts a review into a vector representation based on marker presence."""
    vector = torch.zeros(len(marker2idx), dtype=torch.float)
    for marker, idx in marker2idx.items():
        if marker in review:
            vector[idx] = 1
    return vector

def get_markers():
    """Returns marker sets and marker-to-index dictionary for use in other modules."""
    return marker2idx
