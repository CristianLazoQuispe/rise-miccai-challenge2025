"""
Utility functions: loss definitions and evaluation metrics for 3D segmentation.

This module implements a combined Dice and weighted cross‑entropy loss to
address class imbalance in 3D hippocampus segmentation.  It also provides
functions to compute the Dice similarity coefficient and the Hausdorff
distance between predicted and ground truth segmentations.  These
implementations avoid external dependencies beyond PyTorch and SciPy; if
SciPy is unavailable, Hausdorff distance computation will fall back to a
dummy implementation that returns ``NaN``.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Sequence, Tuple

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None  # type: ignore
    F = None  # type: ignore


def one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Convert a tensor of labels to one‑hot encoding along a new channel axis.

    Parameters
    ----------
    labels : torch.Tensor
        Tensor of shape ``(N, 1, D, H, W)`` containing integer class labels.
    num_classes : int
        Number of classes for the one‑hot representation.

    Returns
    -------
    torch.Tensor
        One‑hot encoded tensor of shape ``(N, num_classes, D, H, W)``.
    """
    # Remove the channel dimension and apply one_hot; then permute back
    # to (N, D, H, W, C) and finally reorder to (N, C, D, H, W)
    return F.one_hot(labels.squeeze(1).long(), num_classes=num_classes).permute(0, 4, 1, 2, 3)


def dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: Optional[Sequence[float]] = None,
    smooth: float = 1e-5,
) -> torch.Tensor:
    """Compute weighted multi‑class Dice loss.

    Parameters
    ----------
    logits : torch.Tensor
        Raw output from the network of shape ``(N, C, D, H, W)``.
    targets : torch.Tensor
        Ground truth labels of shape ``(N, 1, D, H, W)``.
    class_weights : Sequence[float], optional
        Per‑class weights to compensate for class imbalance.  If
        ``None``, equal weighting is used.  The length of
        ``class_weights`` must be equal to the number of classes ``C``.
    smooth : float, optional
        Small constant added to the numerator and denominator to avoid
        division by zero.  Default is ``1e-5``.

    Returns
    -------
    torch.Tensor
        Scalar tensor representing the averaged Dice loss across classes.
    """
    if torch is None:
        raise RuntimeError("PyTorch must be installed to compute the Dice loss.")
    num_classes = logits.shape[1]
    probs = F.softmax(logits, dim=1)
    target_onehot = one_hot(targets, num_classes=num_classes).float()
    dims = (0, 2, 3, 4)  # sum over batch and spatial dims
    intersection = (probs * target_onehot).sum(dim=dims)
    union = probs.sum(dim=dims) + target_onehot.sum(dim=dims)
    dice_per_class = (2.0 * intersection + smooth) / (union + smooth)
    if class_weights is not None:
        weight_tensor = torch.tensor(class_weights, device=dice_per_class.device, dtype=dice_per_class.dtype)
        loss = 1.0 - (dice_per_class * weight_tensor).sum() / weight_tensor.sum()
    else:
        loss = 1.0 - dice_per_class.mean()
    return loss


def weighted_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: Optional[Sequence[float]] = None,
) -> torch.Tensor:
    """Compute weighted multi‑class cross‑entropy loss.

    Parameters
    ----------
    logits : torch.Tensor
        Raw output from the network of shape ``(N, C, D, H, W)``.
    targets : torch.Tensor
        Ground truth labels of shape ``(N, 1, D, H, W)``.
    class_weights : Sequence[float], optional
        Per‑class weights.  The length of ``class_weights`` must equal
        the number of classes.  If ``None``, all classes are weighted equally.

    Returns
    -------
    torch.Tensor
        Scalar tensor representing the mean cross‑entropy loss.
    """
    if torch is None:
        raise RuntimeError(
            "PyTorch must be installed to compute the cross‑entropy loss."
        )
    weights = None
    if class_weights is not None:
        weights = torch.tensor(class_weights, device=logits.device, dtype=logits.dtype)
    # targets must be of shape (N, D, H, W) for F.cross_entropy
    ce_loss = F.cross_entropy(logits, targets.squeeze(1).long(), weight=weights, reduction="mean")
    return ce_loss


class CombinedLoss(torch.nn.Module):
    """Combine Dice loss and cross‑entropy loss with configurable weights."""

    def __init__(
        self,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        class_weights: Optional[Sequence[float]] = None,
    ) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.class_weights = class_weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        dloss = dice_loss(logits, targets, class_weights=self.class_weights)
        celoss = weighted_cross_entropy(logits, targets, class_weights=self.class_weights)
        return self.dice_weight * dloss + self.ce_weight * celoss


def compute_dice_per_class(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    smooth: float = 1e-5,
) -> np.ndarray:
    """Compute Dice scores for each class for a batch of volumes.

    Parameters
    ----------
    preds : torch.Tensor
        Predicted probability maps or logits of shape ``(N, C, D, H, W)``.
    targets : torch.Tensor
        Ground truth labels of shape ``(N, 1, D, H, W)``.
    num_classes : int
        Number of segmentation classes.
    smooth : float, optional
        Smoothing term to avoid division by zero.

    Returns
    -------
    np.ndarray
        Dice score for each class averaged over the batch.
    """
    if torch is None:
        raise RuntimeError(
            "PyTorch must be installed to compute Dice scores."
        )
    with torch.no_grad():
        # Convert logits to hard predictions
        pred_labels = torch.argmax(preds, dim=1, keepdim=True)
        # Convert to one‑hot for computing intersection and union
        pred_onehot = one_hot(pred_labels, num_classes=num_classes).float()
        tgt_onehot = one_hot(targets, num_classes=num_classes).float()
        dims = (0, 2, 3, 4)
        intersection = (pred_onehot * tgt_onehot).sum(dim=dims)
        union = pred_onehot.sum(dim=dims) + tgt_onehot.sum(dim=dims)
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return dice.cpu().numpy()


def hausdorff_distance(
    pred: np.ndarray,
    target: np.ndarray,
    spacing: Optional[Tuple[float, float, float]] = None,
) -> float:
    """Compute the Hausdorff distance between two binary volumes.

    The Hausdorff distance measures the greatest of all the distances
    from a point in one set to the closest point in the other set.  This
    implementation uses SciPy's directed Hausdorff function.  If SciPy
    cannot be imported, the function returns ``NaN``.

    Parameters
    ----------
    pred : np.ndarray
        Binary prediction of shape ``(D, H, W)``.
    target : np.ndarray
        Binary ground truth of the same shape as ``pred``.
    spacing : Tuple[float, float, float], optional
        Voxel spacing used to scale distances.  If ``None``, unit spacing
        is assumed.

    Returns
    -------
    float
        Hausdorff distance.  Returns ``NaN`` if SciPy is unavailable.
    """
    try:
        from scipy.spatial.distance import directed_hausdorff
    except ImportError:
        return float("nan")
    # Extract coordinates of the foreground voxels
    pred_points = np.argwhere(pred > 0)
    target_points = np.argwhere(target > 0)
    if pred_points.size == 0 or target_points.size == 0:
        # If one of the sets is empty, the Hausdorff distance is undefined
        return float("nan")
    if spacing is not None:
        pred_points = pred_points * np.array(spacing)
        target_points = target_points * np.array(spacing)
    d1 = directed_hausdorff(pred_points, target_points)[0]
    d2 = directed_hausdorff(target_points, pred_points)[0]
    return max(d1, d2)
