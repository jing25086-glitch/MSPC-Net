"""
TTA-like Augmentation Utils for Train-Time Enhancement
用于训练时模拟 TTA 增强效果的工具函数
"""
import torch
import random
import torch.nn.functional as F


def random_tta_augment(image, label=None, scale_min=0.9, scale_max=1.1):
    """在训练时应用类似 TTA 的随机增强，兼容 2D [B,C,H,W] 和 3D [B,C,D,H,W]。

    Args:
        image: [B, C, H, W] 或 [B, C, D, H, W]
        label: [B, H, W] 或 [B, D, H, W]（可选）
        scale_min: 最小缩放比例
        scale_max: 最大缩放比例
    Returns:
        augmented_image, augmented_label
    """
    is_2d = (image.dim() == 4)  # [B, C, H, W]
    interp_img  = 'bilinear'  if is_2d else 'trilinear'
    original_size = image.shape[2:]  # (H,W) 或 (D,H,W)

    # 1. Random flip along last dim (W)
    if random.random() > 0.5:
        image = torch.flip(image, dims=[-1])
        if label is not None:
            label = torch.flip(label, dims=[-1])

    # 2. Random flip along second-to-last dim (H)
    if random.random() > 0.5:
        image = torch.flip(image, dims=[-2])
        if label is not None:
            label = torch.flip(label, dims=[-2])

    # 3. Random flip along depth dim (only 3D)
    if not is_2d and random.random() > 0.5:
        image = torch.flip(image, dims=[-3])
        if label is not None:
            label = torch.flip(label, dims=[-3])

    # 4. Random 90° rotation in H-W plane
    k = random.choice([0, 1, 2, 3])
    if k > 0:
        image = torch.rot90(image, k, dims=(-2, -1))
        if label is not None:
            label = torch.rot90(label, k, dims=(-2, -1))
        # rot90 可能交换 H 和 W，恢复到原始尺寸
        if image.shape[2:] != original_size:
            image = F.interpolate(image, size=original_size, mode=interp_img,
                                  align_corners=False)
            if label is not None:
                label = F.interpolate(
                    label.float().unsqueeze(1), size=original_size, mode='nearest'
                ).squeeze(1).long()

    # 5. Random scale jitter
    scale = random.uniform(scale_min, scale_max)
    if abs(scale - 1.0) > 0.01:
        scaled_size = tuple(max(1, int(s * scale)) for s in original_size)
        image = F.interpolate(image, size=scaled_size, mode=interp_img, align_corners=False)
        image = F.interpolate(image, size=original_size, mode=interp_img, align_corners=False)
        if label is not None:
            label = F.interpolate(
                label.float().unsqueeze(1), size=scaled_size, mode='nearest'
            )
            label = F.interpolate(label, size=original_size, mode='nearest').squeeze(1).long()

    # 最终保证尺寸一致
    if image.shape[2:] != original_size:
        image = F.interpolate(image, size=original_size, mode=interp_img, align_corners=False)
        if label is not None:
            label = F.interpolate(
                label.float().unsqueeze(1), size=original_size, mode='nearest'
            ).squeeze(1).long()

    return image, label


def tta_consistency_loss(pred1, pred2, mode='kl'):
    """计算两个预测之间的一致性损失，兼容 2D/3D 输出。

    Args:
        pred1: [B, C, ...] 第一个预测（logits）
        pred2: [B, C, ...] 第二个预测（logits）
        mode: 'kl' 或 'mse'
    Returns:
        consistency loss (scalar)
    """
    if mode == 'kl':
        p1 = F.log_softmax(pred1, dim=1)
        p2 = F.softmax(pred2, dim=1).detach()
        loss = F.kl_div(p1, p2, reduction='batchmean')
    elif mode == 'mse':
        p1 = F.softmax(pred1, dim=1)
        p2 = F.softmax(pred2, dim=1).detach()
        loss = F.mse_loss(p1, p2)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return loss


def apply_tta_train(model, image, label, args, deepsup=True):
    """
    训练时应用 TTA 增强的完整流程
    
    Args:
        model: 网络模型
        image: [B, C, D, H, W] 输入图像
        label: [B, D, H, W] 标签
        args: 参数配置
        deepsup: 是否使用深度监督
    
    Returns:
        dict: {
            'outputs_orig': 原始输出,
            'outputs_tta': TTA增强输出,
            'loss_tta': TTA数据的监督损失,
            'loss_consistency': 一致性损失
        }
    """
    # 应用 TTA 增强
    image_tta, label_tta = random_tta_augment(
        image, label, 
        scale_min=args.tta_scale_min, 
        scale_max=args.tta_scale_max
    )
    
    # TTA 增强数据的前向传播
    if deepsup:
        outputs_tta, features_tta = model(image_tta, deepsup)
    else:
        outputs_tta = model(image_tta)
        features_tta = None
    
    return outputs_tta, features_tta, label_tta


def get_tta_ensemble_loss(outputs_orig, outputs_tta, mode='kl'):
    """
    计算原始预测和 TTA 预测之间的集成一致性损失
    
    适用于深度监督的多尺度输出
    
    Args:
        outputs_orig: list of tensors [B, C, D, H, W]
        outputs_tta: list of tensors [B, C, D, H, W]
        mode: 'kl' 或 'mse'
    
    Returns:
        averaged consistency loss
    """
    if not isinstance(outputs_orig, list):
        outputs_orig = [outputs_orig]
        outputs_tta = [outputs_tta]
    
    consistency_losses = []
    for out_orig, out_tta in zip(outputs_orig, outputs_tta):
        loss = tta_consistency_loss(out_orig, out_tta, mode=mode)
        if not torch.isnan(loss):
            consistency_losses.append(loss)
    
    if len(consistency_losses) > 0:
        return sum(consistency_losses) / len(consistency_losses)
    else:
        return torch.tensor(0.0).to(outputs_orig[0].device)

