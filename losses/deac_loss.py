"""
DEAC: Dynamic Entropy-Aware Contrastive learning (MSPC-Net Sec. 3.4)

- Shannon entropy U_i from teacher probs (Eq. 10)
- Confidence gate: max_prob >= 0.90 (Eq. 11)
- Entropy weight omega = 1 + 0.5 * U / log C (Eq. 12)
- Dynamic temperature tau = tau0 * (1 + U), tau0=0.10 (Eq. 13)
- Prototype InfoNCE (Eq. 14-15)
"""

from typing import Optional

import math
import torch
import torch.nn.functional as F


def teacher_entropy_and_conf(teacher_logits: torch.Tensor):
    """
    teacher_logits: (B, C, D, H, W)
    Returns:
      probs (B,C,...), entropy U (B,...), max_prob (B,...), pseudo (B,...)
    """
    probs = F.softmax(teacher_logits, dim=1)
    log_p = torch.log(probs.clamp_min(1e-8))
    entropy = -(probs * log_p).sum(dim=1)  # (B,D,H,W)
    max_prob, pseudo = probs.max(dim=1)
    return probs, entropy, max_prob, pseudo


def compute_deac_loss(
    feat: torch.Tensor,
    teacher_logits: torch.Tensor,
    prototypes: torch.Tensor,
    conf_thresh: float = 0.90,
    tau0: float = 0.10,
    max_samples: int = 512,
    num_classes: int = 2,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Unlabeled DEAC InfoNCE against class prototypes.
    feat: (B,C,D,H,W) student feature (prefer shallow/high-res)
    prototypes: (num_classes, C)
    """
    B, C, D, H, W = feat.shape
    device = feat.device
    _, entropy, max_prob, pseudo = teacher_entropy_and_conf(teacher_logits.detach())

    # resize teacher maps to feature resolution
    if entropy.shape[1:] != (D, H, W):
        entropy = F.interpolate(entropy.unsqueeze(1), size=(D, H, W), mode="trilinear", align_corners=False).squeeze(1)
        max_prob = F.interpolate(max_prob.unsqueeze(1), size=(D, H, W), mode="trilinear", align_corners=False).squeeze(1)
        pseudo = F.interpolate(pseudo.float().unsqueeze(1), size=(D, H, W), mode="nearest").squeeze(1).long()

    gate = max_prob >= conf_thresh  # Eq. 11
    if gate.sum() == 0:
        return torch.tensor(0.0, device=device)

    feat_flat = feat.permute(0, 2, 3, 4, 1).reshape(-1, C)
    ent_flat = entropy.reshape(-1)
    pse_flat = pseudo.reshape(-1)
    gate_flat = gate.reshape(-1)

    feat_v = feat_flat[gate_flat]
    ent_v = ent_flat[gate_flat]
    lab_v = pse_flat[gate_flat]

    n = feat_v.shape[0]
    if n == 0:
        return torch.tensor(0.0, device=device)
    if n > max_samples:
        sel = torch.randperm(n, device=device)[:max_samples]
        feat_v, ent_v, lab_v = feat_v[sel], ent_v[sel], lab_v[sel]
        n = max_samples

    # Eq. 12-13
    log_c = math.log(max(num_classes, 2))
    omega = 1.0 + 0.5 * (ent_v / log_c)  # in [1, 1.5]
    tau = (tau0 * (1.0 + ent_v)).clamp_min(1e-4)  # per-anchor temperature

    feat_n = F.normalize(feat_v, dim=1, eps=eps)
    proto_n = F.normalize(prototypes, dim=1, eps=eps)

    # sim / tau : (N, K)
    sim = torch.mm(feat_n, proto_n.t())  # (N,K)
    sim = sim / tau.unsqueeze(1)

    log_prob = F.log_softmax(sim, dim=1)
    # clamp labels
    lab_v = lab_v.clamp(0, prototypes.size(0) - 1)
    nll = -log_prob[torch.arange(n, device=device), lab_v]  # Eq. 14
    loss = (omega * nll).sum() / omega.sum().clamp_min(eps)  # Eq. 15
    return loss


def compute_deac_loss_labeled(
    feat: torch.Tensor,
    labels: torch.Tensor,
    prototypes: torch.Tensor,
    tau0: float = 0.10,
    max_samples: int = 512,
    num_classes: int = 2,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Labeled DEAC without teacher gate (fixed tau=tau0, omega=1)."""
    B, C = feat.shape[:2]
    D, H, W = feat.shape[2:]
    device = feat.device
    if labels.shape[1:] != (D, H, W):
        labels_rs = F.interpolate(labels.float().unsqueeze(1), size=(D, H, W), mode="nearest").squeeze(1).long()
    else:
        labels_rs = labels.long()

    feat_flat = feat.permute(0, 2, 3, 4, 1).reshape(-1, C)
    lab_flat = labels_rs.reshape(-1)
    valid = (lab_flat >= 0) & (lab_flat < num_classes)
    feat_v, lab_v = feat_flat[valid], lab_flat[valid]
    n = feat_v.shape[0]
    if n == 0:
        return torch.tensor(0.0, device=device)
    if n > max_samples:
        sel = torch.randperm(n, device=device)[:max_samples]
        feat_v, lab_v = feat_v[sel], lab_v[sel]
        n = max_samples

    feat_n = F.normalize(feat_v, dim=1, eps=eps)
    proto_n = F.normalize(prototypes, dim=1, eps=eps)
    sim = torch.mm(feat_n, proto_n.t()) / tau0
    log_prob = F.log_softmax(sim, dim=1)
    return -log_prob[torch.arange(n, device=device), lab_v].mean()
