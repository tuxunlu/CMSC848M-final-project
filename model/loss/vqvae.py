import torch

def reconstruction_loss(pred, ground_truth, train_var=1.0):
    return torch.mean((pred - ground_truth) ** 2) / train_var