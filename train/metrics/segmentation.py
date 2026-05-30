"""分割评估指标。"""

from __future__ import annotations

import torch


def pixel_accuracy(pred_bin: torch.Tensor, target: torch.Tensor) -> float:
    correct = (pred_bin == target).float().mean()
    return float(correct.item())


def segmentation_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    prob = torch.sigmoid(logits)
    pred = (prob > threshold).float()
    inter = (pred * target).sum()
    union = ((pred + target) > 0).float().sum()
    dice_denom = pred.sum() + target.sum()
    iou = float((inter / (union + 1e-6)).item())
    dice = float((2.0 * inter / (dice_denom + 1e-6)).item())
    acc = pixel_accuracy(pred, target)
    return {"iou": iou, "dice": dice, "acc": acc}
