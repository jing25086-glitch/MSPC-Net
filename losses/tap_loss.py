"""
TAP: Topology-aware Anatomical Prototype Alignment (MSPC-Net Sec. 3.3)

- Deep stages: global class-mean prototypes (Eq. 4)
- Shallow stages: 3x3x3 local affinity-weighted prototypes on L2-normalized
  features with Gaussian bandwidth sigma=0.5 (Eq. 5-8)
- Align labeled vs unlabeled prototypes (Eq. 9)
"""

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


def _resize_mask(mask: torch.Tensor, size: Sequence[int], nearest: bool = True) -> torch.Tensor:
    """mask: (B,D,H,W) long/float -> (B,1,D',H',W') float."""
    m = mask.float().unsqueeze(1)
    if tuple(m.shape[2:]) == tuple(size):
        return m
    if nearest:
        return F.interpolate(m, size=size, mode="nearest")
    return F.interpolate(m, size=size, mode="trilinear", align_corners=False)


def global_class_prototypes(
    feat: torch.Tensor,
    mask: torch.Tensor,
    num_classes: int = 2,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Eq. (4): class-wise masked mean over spatial dims, then batch-mean.
    Returns (num_classes, C).
    """
    B, C = feat.shape[:2]
    mask_rs = _resize_mask(mask, feat.shape[2:], nearest=True)  # (B,1,...)
    protos = []
    for c in range(num_classes):
        if num_classes == 2 and c == 0:
            m = (mask_rs < 0.5).float()
        elif num_classes == 2 and c == 1:
            m = (mask_rs >= 0.5).float()
        else:
            m = (mask_rs == c).float()
        w = m.sum(dim=(2, 3, 4)).clamp_min(eps)  # (B,1)
        p = (feat * m).sum(dim=(2, 3, 4)) / w  # (B,C)
        valid = (m.sum(dim=(2, 3, 4)).view(B) > 0).float().unsqueeze(1)
        if valid.sum() < eps:
            protos.append(torch.zeros(C, device=feat.device, dtype=feat.dtype))
        else:
            protos.append((p * valid).sum(dim=0) / valid.sum().clamp_min(eps))
    return torch.stack(protos, dim=0)


def local_affinity_prototypes(
    feat: torch.Tensor,
    mask: torch.Tensor,
    num_classes: int = 2,
    sigma: float = 0.5,
    eps: float = 1e-6,
    max_centers: int = 256,
) -> torch.Tensor:
    """
    Shallow local prototypes (Eq. 5-8).
    For tractability: sample centers, build 3x3x3 neighborhoods via unfold,
    affinity A = exp(-||f_hat_p - f_hat_q||^2 / (2 sigma^2)), then
    class-masked affinity aggregation; finally mean over centers -> (K, C).
    """
    B, C, D, H, W = feat.shape
    device = feat.device
    # L2 normalize features (Eq. 6)
    feat_n = F.normalize(feat, p=2, dim=1, eps=eps)
    mask_rs = _resize_mask(mask, (D, H, W), nearest=True).squeeze(1)  # (B,D,H,W)

    # 3x3x3 neighborhood via spatial shifts (paper Eq. 5)
    shifts = [(dz, dy, dx) for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
    neigh_feats, neigh_masks = [], []
    for dz, dy, dx in shifts:
        f_s = torch.roll(feat_n, shifts=(dz, dy, dx), dims=(2, 3, 4))
        m_s = torch.roll(mask_rs, shifts=(dz, dy, dx), dims=(1, 2, 3))
        if dz < 0:
            f_s[:, :, dz:, :, :] = 0
            m_s[:, dz:, :, :] = -1
        elif dz > 0:
            f_s[:, :, :dz, :, :] = 0
            m_s[:, :dz, :, :] = -1
        if dy < 0:
            f_s[:, :, :, dy:, :] = 0
            m_s[:, :, dy:, :] = -1
        elif dy > 0:
            f_s[:, :, :, :dy, :] = 0
            m_s[:, :, :dy, :] = -1
        if dx < 0:
            f_s[:, :, :, :, dx:] = 0
            m_s[:, :, :, dx:] = -1
        elif dx > 0:
            f_s[:, :, :, :, :dx] = 0
            m_s[:, :, :, :dx] = -1
        neigh_feats.append(f_s)
        neigh_masks.append(m_s)

    neigh_f = torch.stack(neigh_feats, dim=2)  # (B,C,27,D,H,W)
    neigh_m = torch.stack(neigh_masks, dim=1)  # (B,27,D,H,W)
    center_f = feat_n.unsqueeze(2)

    dist2 = ((center_f - neigh_f) ** 2).sum(dim=1)  # (B,27,D,H,W)
    affinity = torch.exp(-dist2 / (2.0 * sigma * sigma))

    N = D * H * W
    n_samp = min(max_centers, N)
    idx = torch.randperm(N, device=device)[:n_samp]

    neigh_f_flat = neigh_f.permute(0, 3, 4, 5, 2, 1).reshape(B, N, 27, C)[:, idx]
    neigh_m_flat = neigh_m.permute(0, 2, 3, 4, 1).reshape(B, N, 27)[:, idx]
    aff_flat = affinity.permute(0, 2, 3, 4, 1).reshape(B, N, 27)[:, idx]

    protos = []
    for c in range(num_classes):
        if num_classes == 2:
            class_m = (
                (neigh_m_flat >= 0.5).float()
                if c == 1
                else ((neigh_m_flat >= 0) & (neigh_m_flat < 0.5)).float()
            )
        else:
            class_m = (neigh_m_flat == c).float()
        w = aff_flat * class_m
        w_sum = w.sum(dim=2, keepdim=True).clamp_min(eps)
        local = (w.unsqueeze(-1) * neigh_f_flat).sum(dim=2) / w_sum
        valid = (w.sum(dim=2) > eps).float().unsqueeze(-1)
        if valid.sum() < eps:
            protos.append(torch.zeros(C, device=device, dtype=feat.dtype))
        else:
            protos.append((local * valid).sum(dim=(0, 1)) / valid.sum().clamp_min(eps))
    return torch.stack(protos, dim=0)


def compute_tap_loss(
    feats_labeled: List[torch.Tensor],
    feats_unlabeled: List[torch.Tensor],
    mask_labeled: torch.Tensor,
    mask_unlabeled: torch.Tensor,
    deep_indices: Sequence[int] = (2, 3),
    shallow_indices: Sequence[int] = (0, 1),
    lambda_deep: float = 1.0,
    lambda_shallow: float = 1.0,
    num_classes: int = 2,
    sigma: float = 0.5,
    max_centers: int = 256,
) -> torch.Tensor:
    """
    Eq. (9): weighted sum of deep-global and shallow-local prototype MSE.
    feats_*: list length 4, order shallow->deep [x8, x7, x6, x5].
    """
    device = feats_labeled[0].device
    losses = []

    for i in deep_indices:
        if i >= len(feats_labeled):
            continue
        p_l = global_class_prototypes(feats_labeled[i], mask_labeled, num_classes)
        p_u = global_class_prototypes(feats_unlabeled[i], mask_unlabeled, num_classes)
        losses.append(lambda_deep * F.mse_loss(p_l, p_u))

    for i in shallow_indices:
        if i >= len(feats_labeled):
            continue
        p_l = local_affinity_prototypes(
            feats_labeled[i], mask_labeled, num_classes, sigma=sigma, max_centers=max_centers
        )
        p_u = local_affinity_prototypes(
            feats_unlabeled[i], mask_unlabeled, num_classes, sigma=sigma, max_centers=max_centers
        )
        losses.append(lambda_shallow * F.mse_loss(p_l, p_u))

    if not losses:
        return torch.tensor(0.0, device=device)
    return sum(losses) / len(losses)
