import torch.nn.functional as F
import torch
import numpy as np
from collections import Counter

def translation_loss(logits, targets, pad_token_id=0):
    """
    Compute cross-entropy loss for machine translation.
    
    Args:
        logits: [B, T, V] decoder output logits
        targets: [B, T] ground-truth token indices
        pad_token_id: token id used for padding

    Returns:
        loss: scalar tensor
    """
    B, T, V = logits.shape

    # Flatten logits and targets to match CrossEntropyLoss input shape
    logits = logits.view(B * T, V)
    targets = targets.view(B * T)

    loss = F.cross_entropy(logits, targets, ignore_index=pad_token_id)
    return loss


def evaluate_token_diversity(generated_tokens: torch.Tensor, vocab_size: int = 512, topk: int = 10):
    """
    Evaluate token diversity metrics for generated token sequences.

    Args:
        generated_tokens: Tensor of shape (B, T), each entry in [0, vocab_size)
        vocab_size: Size of vocabulary
        topk: for Top-k coverage

    Returns:
        Dictionary of diversity metrics
    """
    # Flatten and convert to int64
    tokens = generated_tokens.view(-1).cpu().detach().numpy().astype(np.int64)

    # Silently filter out invalid tokens
    tokens = tokens[(tokens >= 0) & (tokens < vocab_size)]
    total_tokens = len(tokens)

    if total_tokens == 0:
        return {
            'unique_token_ratio': 0.0,
            'token_entropy': 0.0,
            f'top_{topk}_token_coverage': 0.0
        }

    # Count frequency of each token
    counts = np.bincount(tokens, minlength=vocab_size)
    probs = counts / total_tokens  # empirical distribution

    # Metric 1: Unique Token Ratio
    unique_token_ratio = np.sum(counts > 0) / vocab_size

    # Metric 2: Entropy of token distribution
    nonzero_probs = probs[probs > 0]
    entropy = -np.sum(nonzero_probs * np.log2(nonzero_probs))

    # Metric 3: Top-k token coverage
    topk_token_mass = np.sum(np.sort(probs)[-topk:])

    return {
        'unique_token_ratio': unique_token_ratio,
        'token_entropy': entropy,
        f'top_{topk}_token_coverage': topk_token_mass
    }
