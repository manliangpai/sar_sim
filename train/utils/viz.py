"""训练/评估可视化。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def save_triplet_figure(
    bp_mag: np.ndarray,
    gt: np.ndarray,
    pred_prob: np.ndarray,
    *,
    name: str,
    out_path: Path,
    dpi: int = 120,
) -> None:
    """GT | 输入 BP magnitude | 模型概率，统一 viridis。"""
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    panels = [
        (gt, "GT"),
        (bp_mag, "BP magnitude"),
        (pred_prob, "Prediction"),
    ]
    for ax, (img, title) in zip(axes, panels):
        im = ax.imshow(img, cmap="viridis", vmin=0.0, vmax=1.0, interpolation="nearest")
        ax.set_title(f"{title} — {name}")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor="white")
    plt.close(fig)


def logits_to_prob(logits: torch.Tensor) -> np.ndarray:
    prob = torch.sigmoid(logits).detach().cpu().numpy()
    return prob[0, 0]
