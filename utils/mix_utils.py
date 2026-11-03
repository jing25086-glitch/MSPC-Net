import random
from typing import Iterable, Sequence, Tuple, Union

import torch


def _as_patch_tuple(
    patch_size: Union[int, Sequence[int]],
    spatial_dims: int,
) -> Tuple[int, ...]:
    if isinstance(patch_size, Iterable) and not isinstance(patch_size, (str, bytes)):
        patch_tuple = tuple(int(max(1, p)) for p in patch_size)
        if len(patch_tuple) != spatial_dims:
            raise ValueError(
                f"patch_size length ({len(patch_tuple)}) must match spatial dims ({spatial_dims})"
            )
        return patch_tuple
    patch_value = int(max(1, patch_size))
    return (patch_value,) * spatial_dims


def _build_axis_intervals(axis_dim: int, patch_dim: int) -> Tuple[Tuple[int, int], ...]:
    if patch_dim >= axis_dim:
        return ((0, axis_dim),)
    intervals = []
    start = 0
    while start + patch_dim <= axis_dim:
        intervals.append((start, start + patch_dim))
        start += patch_dim
    if intervals[-1][1] < axis_dim:
        intervals.append((axis_dim - patch_dim, axis_dim))
    return tuple(intervals)


def _unravel_index(index: int, dims: Sequence[int]) -> Tuple[int, ...]:
    coords = []
    for dim in reversed(dims):
        coords.append(index % dim)
        index //= dim
    return tuple(reversed(coords))


def generate_patchfusion_mask(
    img_shape: torch.Size,
    patch_size: Union[int, Sequence[int]],
    device: torch.device = None,
) -> torch.Tensor:
    """
    生成固定位置的 patch 掩码，用于 Patch-level 融合。

    Args:
        img_shape: 输入图像的形状，支持 [B, C, H, W] 或 [B, C, W, H, D]。
        patch_size: patch 边长（int）或各维 patch 尺寸序列（长度与空间维一致）。
        device: 可选，指定返回掩码的设备；若为空则使用当前 CUDA/CPU 默认。

    Returns:
        patch 掩码，shape = [B, 1, ...]，掩码值为 {0,1}。
    """

    if len(img_shape) < 4:
        raise ValueError(f"img_shape 必须至少包含 4 个维度，当前为 {img_shape}")

    spatial_shape = tuple(int(s) for s in img_shape[2:])
    spatial_dims = len(spatial_shape)
    patch_tuple = _as_patch_tuple(patch_size, spatial_dims)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = int(img_shape[0])
    mask = torch.zeros((batch_size, 1, *spatial_shape), dtype=torch.float32, device=device)

    axis_intervals = tuple(
        _build_axis_intervals(axis_dim, patch_dim)
        for axis_dim, patch_dim in zip(spatial_shape, patch_tuple)
    )
    grid_counts = [len(intervals) for intervals in axis_intervals]
    total_patches = 1
    for count in grid_counts:
        total_patches *= count

    for b in range(batch_size):
        n_selected = max(1, total_patches // 3)
        selected_indices = random.sample(range(total_patches), n_selected)
        for flat_idx in selected_indices:
            coords = _unravel_index(flat_idx, grid_counts)
            slices = [
                slice(*axis_intervals[axis][coord]) for axis, coord in enumerate(coords)
            ]
            mask[(b, 0, *slices)] = 1.0

    return mask


