from __future__ import annotations

from typing import Optional

import torch
import torchmetrics as tm


def make_auroc_metrics():
    """
    Compatible with older torchmetrics (perceiver-io pins <0.10).
    """
    pixel_auroc = tm.AUROC(pos_label=1)
    image_auroc = tm.AUROC(pos_label=1)
    return pixel_auroc, image_auroc


def flatten_pixel_map(x: torch.Tensor) -> torch.Tensor:
    """
    (B, H, W, 1) or (B, 1, H, W) -> (B*H*W,)
    """
    if x.ndim != 4:
        raise ValueError(f"Expected 4D tensor, got {tuple(x.shape)}")

    if x.shape[-1] == 1:
        return x.reshape(-1)

    if x.shape[1] == 1:
        x = x.permute(0, 2, 3, 1).contiguous()
        return x.reshape(-1)

    raise ValueError(f"Unsupported pixel map shape: {tuple(x.shape)}")


@torch.no_grad()
def update_pixel_auroc(pixel_auroc, pred_prob_map: torch.Tensor, true_mask: torch.Tensor) -> bool:
    pred = flatten_pixel_map(pred_prob_map).float()
    true = flatten_pixel_map(true_mask).long()

    # 空张量保护 跳过单类数据 全0 全1 直接返回False 
    if true.numel() == 0 or true.min() == true.max():
        return False

    pixel_auroc.update(pred, true)
    return True


@torch.no_grad()
def update_image_auroc(image_auroc, image_score: torch.Tensor, true_label: torch.Tensor) -> bool:
    pred = image_score.float().view(-1)
    true = true_label.long().view(-1)

    if true.numel() == 0 or true.min() == true.max():
        return False

    image_auroc.update(pred, true)
    return True


def compute_and_reset(metric) -> Optional[torch.Tensor]:
    try:
        value = metric.compute()
    except Exception:
        value = None
    metric.reset()
    return value