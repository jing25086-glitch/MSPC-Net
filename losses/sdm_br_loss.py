"""
SDM-BR: Signed Distance Map Boundary Regularization (MSPC-Net Sec. 3.5)

Convention (Eq. 16): interior negative, boundary zero, exterior positive.
Target from GT (labeled) or stop-grad teacher pseudo (unlabeled).
Loss: squared L2 between predicted continuous field and target (Eq. 19).
"""

from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


def compute_signed_distance_map(
    mask_binary: np.ndarray,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """
    mask_binary: bool/uint8 array, True = foreground.
    Returns float32 SDM: inside < 0, outside > 0, boundary ~ 0.
    """
    from scipy.ndimage import distance_transform_edt

    mask = np.asarray(mask_binary).astype(bool)
    if mask.sum() == 0 or (~mask).sum() == 0:
        return np.zeros(mask.shape, dtype=np.float32)
    dist_out = distance_transform_edt(~mask, sampling=spacing)  # exterior
    dist_in = distance_transform_edt(mask, sampling=spacing)  # interior
    sdm = dist_out.astype(np.float32) - dist_in.astype(np.float32)
    return sdm


def masks_to_sdm_tensor(
    masks: torch.Tensor,
    num_fg: int = 1,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
) -> torch.Tensor:
    """
    masks: (B, D, H, W) long — class ids
    returns: (B, C_fg, D, H, W) float SDM targets
    """
    B = masks.shape[0]
    device = masks.device
    outs = []
    masks_np = masks.detach().cpu().numpy()
    for b in range(B):
        chans = []
        for c in range(1, num_fg + 1):
            if num_fg == 1:
                fg = masks_np[b] > 0
            else:
                fg = masks_np[b] == c
            chans.append(compute_signed_distance_map(fg, spacing=spacing))
        outs.append(np.stack(chans, axis=0))
    arr = np.stack(outs, axis=0)  # (B,C_fg,D,H,W)
    return torch.from_numpy(arr).float().to(device)


def compute_sdm_br_loss(
    sdm_pred: torch.Tensor,
    target_mask: torch.Tensor,
    num_fg: int = 1,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
) -> torch.Tensor:
    """
    sdm_pred: (B, C_fg, D, H, W) linear head output (can be any real values)
    target_mask: (B, D, H, W) long GT or teacher pseudo (stop-grad outside)
    """
    if sdm_pred is None:
        return torch.tensor(0.0, device=target_mask.device)
    with torch.no_grad():
        target = masks_to_sdm_tensor(target_mask, num_fg=num_fg, spacing=spacing)
        if target.shape[2:] != sdm_pred.shape[2:]:
            target = F.interpolate(target, size=sdm_pred.shape[2:], mode="trilinear", align_corners=False)
    return F.mse_loss(sdm_pred, target)
