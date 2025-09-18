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
    """Convert a tensor of integer labels to one‑hot encoding along a new channel axis.

    This helper removes the singleton channel dimension often present on
    label volumes and uses :func:`torch.nn.functional.one_hot` to
    construct a one‑hot representation.  The resulting tensor is
    rearranged to the shape ``(N, C, D, H, W)`` expected by common
    loss functions.

    Parameters
    ----------
    labels : torch.Tensor
        Tensor of shape ``(N, 1, D, H, W)`` containing integer class
        labels.
    num_classes : int
        Total number of classes in the segmentation.  Must be greater
        than the maximum label value present in ``labels``.

    Returns
    -------
    torch.Tensor
        One‑hot encoded tensor of shape ``(N, num_classes, D, H, W)``.

    Notes
    -----
    The returned tensor is placed on the same device as the input
    ``labels`` and uses the default tensor datatype for one‑hot
    encodings (usually ``torch.int64``).  If you require a different
    datatype, call ``to(dtype=...)`` on the result.
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


def tversky_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.3,
    beta: float = 0.7,
    class_weights: Optional[Sequence[float]] = None,
    gamma: float = 1.0,
    smooth: float = 1e-5,
) -> torch.Tensor:
    """Compute the (focal) Tversky loss for multi‑class segmentation.

    The Tversky index is a generalisation of the Dice coefficient that
    introduces separate penalties for false positives and false
    negatives via the ``alpha`` and ``beta`` parameters.  The focal
    version of the loss raises the complement of the Tversky index to
    the power ``gamma`` to focus training on harder examples, which is
    particularly beneficial when segmenting very small structures.

    Parameters
    ----------
    logits : torch.Tensor
        Raw output from the network of shape ``(N, C, D, H, W)``.
    targets : torch.Tensor
        Ground truth labels of shape ``(N, 1, D, H, W)``.
    alpha : float, optional
        Weight for false positives.  A higher value penalises false
        positives more strongly.  Default is 0.3.
    beta : float, optional
        Weight for false negatives.  A higher value penalises false
        negatives more strongly.  Default is 0.7.
    class_weights : sequence of floats, optional
        Optional per‑class weights applied to the loss.  If
        ``None``, all classes contribute equally.
    gamma : float, optional
        Exponent applied to the Tversky complement to form the focal
        Tversky loss.  When ``gamma=1``, this reduces to the plain
        Tversky loss.  Values greater than 1 emphasise hard to
        segment voxels.  Default is 1.0.
    smooth : float, optional
        Smoothing factor added to numerator and denominator to avoid
        division by zero.  Default is ``1e-5``.

    Returns
    -------
    torch.Tensor
        Scalar tensor representing the average focal Tversky loss.
    """
    if torch is None:
        raise RuntimeError("PyTorch must be installed to compute the Tversky loss.")
    num_classes = logits.shape[1]
    # Convert logits to probabilities
    probs = F.softmax(logits, dim=1)
    target_onehot = one_hot(targets, num_classes=num_classes).float()
    dims = (0, 2, 3, 4)
    tp = (probs * target_onehot).sum(dim=dims)
    fp = (probs * (1 - target_onehot)).sum(dim=dims)
    fn = ((1 - probs) * target_onehot).sum(dim=dims)
    tversky_index = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    # Convert to loss
    if class_weights is not None:
        weight_tensor = torch.tensor(class_weights, device=tversky_index.device, dtype=tversky_index.dtype)
        per_class = torch.pow(1.0 - tversky_index, gamma) * weight_tensor
        return per_class.sum() / weight_tensor.sum()
    else:
        return torch.pow(1.0 - tversky_index, gamma).mean()


class FocalTverskyLoss(torch.nn.Module):
    """PyTorch module for the focal Tversky loss.

    This class wraps :func:`tversky_loss` into a ``torch.nn.Module``
    compatible interface.  Use this loss when the target structures
    occupy only a small fraction of the image volume; the focal
    exponent ``gamma`` down‑weights easy examples and improves
    sensitivity to small objects.

    Parameters
    ----------
    alpha : float, optional
        Weight for false positives.  Default is 0.3.
    beta : float, optional
        Weight for false negatives.  Default is 0.7.
    gamma : float, optional
        Focusing exponent; values greater than 1 put more emphasis on
        misclassified voxels.  Default is 1.33.
    class_weights : sequence of floats, optional
        Per‑class weights to mitigate class imbalance.  Default is
        ``None``.
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 1.33,
        class_weights: Optional[Sequence[float]] = None,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.class_weights = class_weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return tversky_loss(
            logits,
            targets,
            alpha=self.alpha,
            beta=self.beta,
            class_weights=self.class_weights,
            gamma=self.gamma,
        )


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


import torch
import numpy as np
from monai.metrics import DiceMetric, HausdorffDistanceMetric, SurfaceDiceMetric
from monai.metrics.utils import get_surface_distance
from monai.transforms import AsDiscrete

# ---------------------------
# Metrics initializers
# ---------------------------

def get_metrics_3d(device):
    """
    Returns a dictionary of 3D medical segmentation metrics
    including Dice, Hausdorff95, ASSD and Surface Dice.
    """
    metrics = {
        "dice": DiceMetric(include_background=False, reduction="mean", get_not_nans=False),#.to(device),
        "hd95": HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean"),#.to(device),
        "surf_dice": SurfaceDiceMetric(include_background=False, reduction="mean", distance_metric="euclidean",
                                                   class_thresholds=[1.0]  # for 1 class (foreground)
                                                   ),#.to(device),
    }
    return metrics


# ---------------------------
# Metric computation
# ---------------------------

def compute_metrics(pred, target, metrics):
    """
    Compute all metrics for given prediction and target masks.

    Args:
        pred (torch.Tensor): predicted segmentation, shape (B, C, H, W, D)
        target (torch.Tensor): ground truth segmentation, shape (B, C, H, W, D)
        metrics (dict): dictionary of MONAI metrics from get_metrics_3d()

    Returns:
        dict: results of all metrics
    """
    # binarize predictions and targets
    pred_discrete = AsDiscrete(threshold=0.5)(pred)
    target_discrete = AsDiscrete(to_onehot=pred.shape[1])(target)

    results = {}
    with torch.no_grad():
        for name, metric in metrics.items():
            metric.reset()
            metric(y_pred=pred_discrete, y=target_discrete)
            results[name] = metric.aggregate().item()
            metric.reset()

    return results


# ---------------------------
# Example usage inside training
# ---------------------------

"""
from utils import get_metrics_3d, compute_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
metrics = get_metrics_3d(device)

# inside validation loop
with torch.no_grad():
    outputs = model(inputs)
    results = compute_metrics(outputs, labels, metrics)
    print("Dice:", results["dice"], "HD95:", results["hd95"], "SurfaceDice:", results["surf_dice"])
"""
