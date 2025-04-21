import torch.nn.functional as F

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
