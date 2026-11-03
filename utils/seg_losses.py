"""Minimal segmentation losses used by train_MSPCNet.py (self-contained)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def Binary_dice_loss(predictive, target, ep=1e-8):
    intersection = 2 * torch.sum(predictive * target) + ep
    union = torch.sum(predictive) + torch.sum(target) + ep
    return 1 - intersection / union


def mse_loss(input1, input2):
    return torch.mean((input1 - input2) ** 2)


def gaussian_aux_weight(iter_num: int, max_iterations: int, sigma_factor: float = 0.5) -> float:
    """
    Paper Eq. (25): Gaussian progressive schedule over full training length T.
    Returns a multiplier in (0, 1], approaching 1 near the end.
    g(t) = exp( - (1 - t/T)^2 / (2 * sigma_factor^2) )
    """
    if max_iterations <= 0:
        return 1.0
    t = min(max(iter_num, 0), max_iterations) / float(max_iterations)
    return float(torch.exp(torch.tensor(-((1.0 - t) ** 2) / (2.0 * sigma_factor ** 2))).item())
