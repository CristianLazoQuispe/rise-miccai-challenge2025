"""
cascade_utils.py
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple


def get_roi_bbox_from_labels(labels: torch.Tensor, margin: int = 8) -> List[Tuple[int, int, int, int, int, int]]:
    """Compute a bounding box around the foreground voxels of each sample.

    Parameters
    ----------
    labels : torch.Tensor
        Tensor of shape ``[B,1,D,H,W]`` containing integer labels.
    margin : int, optional
        Number of voxels to extend the bounding box in each direction.

    Returns
    -------
    List[Tuple[int,int,int,int,int,int]]
        For each batch element, a 6‑tuple ``(z0,y0,x0,z1,y1,x1)``
        defining the start (inclusive) and end (exclusive) indices of
        the bounding box in ``(z,y,x)`` order.  If no foreground is
        present, the bounding box spans the entire volume.
    """
    bboxes: List[Tuple[int,int,int,int,int,int]] = []
    B, C, D, H, W = labels.shape
    for i in range(B):
        mask = (labels[i] > 0).squeeze(0)
        coords = mask.nonzero(as_tuple=False)
        if coords.numel() == 0:
            bboxes.append((0, 0, 0, D, H, W))
            continue
        z0, y0, x0 = coords.min(dim=0).values.tolist()
        z1, y1, x1 = coords.max(dim=0).values.tolist()
        z0 = max(z0 - margin, 0)
        y0 = max(y0 - margin, 0)
        x0 = max(x0 - margin, 0)
        z1 = min(z1 + margin + 1, D)
        y1 = min(y1 + margin + 1, H)
        x1 = min(x1 + margin + 1, W)
        bboxes.append((z0, y0, x0, z1, y1, x1))
    return bboxes


def get_roi_bbox_from_logits(logits: torch.Tensor, thr: float = 0.2, margin: int = 8) -> List[Tuple[int, int, int, int, int, int]]:
    """Compute bounding boxes from coarse model logits.

    The bounding boxes are derived by thresholding the softmax
    probabilities and finding the union of foreground voxels across
    classes 1 and 2 (assuming class 0 is background).  A margin is
    added to enlarge the ROI.  If the model predicts no voxels above
    threshold, the full volume is used.

    Parameters
    ----------
    logits : torch.Tensor
        Tensor of shape ``[B,C,D,H,W]`` containing raw logits.
    thr : float, optional
        Threshold on the foreground probability; voxels with probability
        greater than ``thr`` for either class 1 or 2 are considered
        foreground.  Lower this value (e.g. 0.1) early in training.
    margin : int, optional
        Number of voxels to extend the bounding box in each direction.

    Returns
    -------
    List[Tuple[int,int,int,int,int,int]]
        Bounding boxes for each batch element, as in
        :func:`get_roi_bbox_from_labels`.
    """
    probs = torch.softmax(logits, dim=1)
    foreground = probs[:, 1:].sum(dim=1)  # sum of class 1 and 2
    bboxes: List[Tuple[int,int,int,int,int,int]] = []
    B, D, H, W = foreground.shape
    for i in range(B):
        #print(foreground[i].mean(),foreground[i].min(),foreground[i].max(),foreground[i].std())
        mask = (foreground[i] > thr)
        coords = mask.nonzero(as_tuple=False)
        if coords.numel() == 0:
            bboxes.append((0, 0, 0, D, H, W))
            continue
        z0, y0, x0 = coords.min(dim=0).values.tolist()
        z1, y1, x1 = coords.max(dim=0).values.tolist()
        z0 = max(z0 - margin, 0)
        y0 = max(y0 - margin, 0)
        x0 = max(x0 - margin, 0)
        z1 = min(z1 + margin + 1, D)
        y1 = min(y1 + margin + 1, H)
        x1 = min(x1 + margin + 1, W)
        bboxes.append((z0, y0, x0, z1, y1, x1))
    return bboxes


def crop_to_bbox(volume: torch.Tensor, bbox: Tuple[int, int, int, int, int, int]) -> torch.Tensor:
    """Crop a 5D tensor ``[B,C,D,H,W]`` to the given bounding box.

    The bounding box is specified as ``(z0,y0,x0,z1,y1,x1)``.  The
    function supports batching; only the spatial dimensions are
    cropped.  The batch dimension is preserved.

    Parameters
    ----------
    volume : torch.Tensor
        A tensor of shape ``[B,C,D,H,W]``.
    bbox : Tuple[int,int,int,int,int,int]
        Bounding box coordinates.

    Returns
    -------
    torch.Tensor
        The cropped tensor of shape ``[B,C,z1-z0,y1-y0,x1-x0]``.
    """
    z0, y0, x0, z1, y1, x1 = bbox
    return volume[:, :, z0:z1, y0:y1, x0:x1]


def resize_volume(volume: torch.Tensor, spatial_size: Tuple[int, int, int], mode: str = "trilinear") -> torch.Tensor:
    """Resize a 5D tensor spatially to a fixed size.

    Uses ``torch.nn.functional.interpolate`` on the spatial dimensions.

    Parameters
    ----------
    volume : torch.Tensor
        Tensor of shape ``[B,C,D,H,W]``.
    spatial_size : Tuple[int,int,int]
        Target spatial size ``(depth,height,width)``.
    mode : str, optional
        Interpolation mode.  Use ``"trilinear"`` for images and
        ``"nearest"`` for labels.

    Returns
    -------
    torch.Tensor
        The resized tensor.
    """
    if volume.dim() != 5:
        raise ValueError(f"Expected 5D tensor [B,C,D,H,W], got {volume.shape}")
    return F.interpolate(volume, size=spatial_size, mode=mode, align_corners=False if mode != "nearest" else None)


class CascadeDataset(torch.utils.data.Dataset):
    """Dataset wrapper for cascade refinement.

    It loads image/label pairs, applies a coarse model to obtain a
    preliminary segmentation and uses it to compute a ROI for
    refinement.  During early epochs, or with a specified probability,
    the ROI is derived from the ground truth labels instead of the
    coarse prediction to avoid compounding errors.  The cropped
    images and labels are then resampled to ``roi_size`` and returned.

    Parameters
    ----------
    base_ds : torch.utils.data.Dataset
        A dataset returning dictionaries with keys ``"image"`` and
        ``"label"``.  These volumes should already be resampled to the
        coarse model resolution.
    coarse_model : torch.nn.Module
        A pretrained coarse segmentation model.  It should take a
        tensor of shape ``[B,1,D,H,W]`` and return logits of shape
        ``[B,C,D,H,W]``.  The model will be set to ``eval()`` and
        gradients disabled inside the ``__getitem__``.
    roi_size : Tuple[int,int,int]
        Spatial size of the cropped ROI to feed into the refinement
        model.
    margin : int
        Number of voxels to extend each bounding box edge.
    use_gt_prob : float
        Probability of using the ground‑truth bounding box instead of
        the predicted bounding box for each sample.  This can be
        linearly annealed during training to implement a scheduled
        sampling strategy.

    Note
    ----
    Use this dataset only after the coarse model has been trained
    sufficiently to provide reasonable localisation.
    """
    def __init__(self, base_ds, coarse_model: torch.nn.Module, roi_size: Tuple[int, int, int] = (160, 160, 160),
                 margin: int = 8, use_gt_prob: float = 0.5):
        super().__init__()
        self.base_ds = base_ds
        self.coarse_model = coarse_model.eval()
        for p in self.coarse_model.parameters():
            p.requires_grad = False
        self.roi_size = roi_size
        self.margin = margin
        self.use_gt_prob = use_gt_prob

    def __len__(self) -> int:
        return len(self.base_ds)

    def __getitem__(self, idx):
        item = self.base_ds[idx]
        img = item["image"]  # [C,D,H,W] from base dataset (without batch dim)
        lbl = item["label"]  # [C,D,H,W]
        # ensure batch dim
        img_b = img.unsqueeze(0)
        lbl_b = lbl.unsqueeze(0)
        with torch.no_grad():
            logits = self.coarse_model(img_b)  # [1,C,D,H,W]
        # choose bbox from GT or pred based on probability
        if torch.rand(1).item() < self.use_gt_prob:
            bboxes = get_roi_bbox_from_labels(lbl_b, margin=self.margin)
        else:
            bboxes = get_roi_bbox_from_logits(logits, thr=0.2, margin=self.margin)
        bbox = bboxes[0]
        # crop and resize
        img_roi = crop_to_bbox(img_b, bbox)
        lbl_roi = crop_to_bbox(lbl_b, bbox)
        img_roi = resize_volume(img_roi, self.roi_size, mode="trilinear")
        lbl_roi = resize_volume(lbl_roi, self.roi_size, mode="nearest")
        # return without batch dim
        return {"image": img_roi.squeeze(0), "label": lbl_roi.squeeze(0)}