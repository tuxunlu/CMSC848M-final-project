import torch
import numpy as np

def entropy_loss(min_encoding_indices, n_embeddings, eps=1e-10):
    """
    Computes entropy loss to encourage codebook diversity.

    Args:
        min_encoding_indices: Tensor of shape (B, 1200), dtype torch.long
        n_embeddings: Total number of embeddings (K)
        eps: Small constant to avoid log(0)

    Returns:
        entropy_loss: scalar tensor
    """
    # Flatten to (B * 1200)
    indices = min_encoding_indices.view(-1)

    # Count token usage
    counts = torch.bincount(indices, minlength=n_embeddings).float()

    # Normalize to get empirical distribution
    probs = counts / counts.sum()

    # Compute entropy
    entropy = -torch.sum(probs * torch.log(probs + eps))

    # Normalize: maximum entropy is log(K)
    max_entropy = np.log(n_embeddings)
    normalized_entropy_loss = (max_entropy - entropy) / max_entropy

    return normalized_entropy_loss
