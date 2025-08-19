"""
Dataset utilities for the RISE‑MICCAI LISA 2025 hippocampus segmentation task.

This module defines a PyTorch Dataset class to load CISO images and their
corresponding merged HF/LF hippocampal segmentations from NIfTI files.  It
also provides helper functions to perform group‑wise cross‑validation splits
using the subject identifiers contained in the CSV metadata.

Note that this code relies on external libraries (`torch`, `nibabel`,
`numpy`, `pandas` and `scikit‑learn`) which are not available in the
current environment.  The provided implementation illustrates how to
structure the data loading and splitting logic – you should run it on a
machine with the required dependencies installed.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional

# External dependencies required for full functionality
try:
    import torch
    from torch.utils.data import Dataset
    import nibabel as nib
    from sklearn.model_selection import GroupKFold
except ImportError:
    # When running in an environment without these libraries, define
    # placeholders so that type hints remain valid.  The actual runtime
    # functionality will not work until the dependencies are installed.
    torch = None  # type: ignore
    nib = None  # type: ignore
    Dataset = object  # type: ignore
    GroupKFold = None  # type: ignore


class HippocampusDataset(Dataset):
    """PyTorch dataset for loading 3D MR volumes and hippocampus segmentations.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing at least the columns ``filepath`` (path to the
        CISO image) and, if labels are available, ``filepath_label`` (path
        to the merged hippocampal segmentation).  The DataFrame should also
        include an ``ID`` column identifying the subject; this is used for
        group‑wise cross‑validation but is not accessed inside the dataset.
    is_test : bool, optional
        If ``True``, the dataset will return only images and omit labels.
        Use this for inference on the validation/test set where ground truth
        segmentations are not available.  Default is ``False``.
    resample_spacing : Optional[Tuple[float, float, float]], optional
        Desired voxel spacing (in millimetres) to resample the images and
        labels to.  If provided, both image and label volumes will be
        resampled using linear and nearest interpolation, respectively.
        Passing ``None`` disables resampling.  Note that resampling
        requires access to the image header; if nibabel is not installed
        this option will have no effect.  Default is ``None``.
    spatial_size : Tuple[int, int, int]
        Spatial size (depth, height, width) to which the resampled volumes
        will be cropped or padded.  The default (96, 96, 96) matches the
        input size expected by the pretrained UNesT network.  Larger
        volumes will be centre‑cropped and smaller volumes will be zero‑padded.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        is_test: bool = False,
        resample_spacing: Optional[Tuple[float, float, float]] = None,
        spatial_size: Tuple[int, int, int] = (96, 96, 96),
    ) -> None:
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.is_test = is_test
        self.resample_spacing = resample_spacing
        self.spatial_size = spatial_size

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict:
        if torch is None or nib is None:
            raise RuntimeError(
                "torch and nibabel must be installed to use the HippocampusDataset."
            )
        row = self.df.iloc[index]
        img_path = row["filepath"]
        # Load the image volume using nibabel
        img_nii = nib.load(img_path)
        img = img_nii.get_fdata().astype(np.float32)

        # Optional intensity normalisation: scale to [0,1]
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)

        # Resample to desired spacing if requested
        if self.resample_spacing is not None:
            # Using nibabel's affine to compute new shape.  This is a simple
            # nearest‑neighbour resampling; for higher quality consider using
            # scipy.ndimage.zoom or SimpleITK.  Here we avoid extra
            # dependencies by implementing a basic rescale based on voxel
            # spacing.
            orig_spacing = np.abs(img_nii.header.get_zooms()[:3])
            scale_factors = orig_spacing / np.array(self.resample_spacing)
            new_shape = np.round(np.array(img.shape) * scale_factors).astype(int)
            # Use numpy's zoom for simple interpolation.  order=1 for linear.
            import scipy.ndimage
            img = scipy.ndimage.zoom(img, zoom=scale_factors, order=1)

        # Resize/crop/pad to fixed spatial size
        img = self._resize_volume(img, self.spatial_size)
        # Add channel dimension
        img_tensor = torch.from_numpy(img[None, ...])  # shape: (1, D, H, W)

        sample = {"image": img_tensor}

        if not self.is_test:
            label_path = row["filepath_label"]
            lab_nii = nib.load(label_path)
            lab = lab_nii.get_fdata().astype(np.int64)
            if self.resample_spacing is not None:
                orig_spacing = np.abs(lab_nii.header.get_zooms()[:3])
                scale_factors = orig_spacing / np.array(self.resample_spacing)
                import scipy.ndimage
                lab = scipy.ndimage.zoom(
                    lab, zoom=scale_factors, order=0
                )  # nearest for labels
            lab = self._resize_volume(lab, self.spatial_size)
            lab_tensor = torch.from_numpy(lab[None, ...])  # shape: (1, D, H, W)
            sample["label"] = lab_tensor
        return sample

    @staticmethod
    def _resize_volume(volume: np.ndarray, target_size: Tuple[int, int, int]) -> np.ndarray:
        """Center‑crop or pad a 3D volume to the target spatial size.

        Parameters
        ----------
        volume : np.ndarray
            Input volume of shape (D, H, W).
        target_size : Tuple[int, int, int]
            Desired output size (D, H, W).
        Returns
        -------
        np.ndarray
            Resized volume of shape ``target_size``.
        """
        current_shape = volume.shape
        out = np.zeros(target_size, dtype=volume.dtype)
        # Compute cropping or padding indices
        for i in range(3):
            excess = current_shape[i] - target_size[i]
            if excess > 0:
                # Crop
                start = excess // 2
                end = start + target_size[i]
                slc = slice(start, end)
                if i == 0:
                    volume = volume[slc, :, :]
                elif i == 1:
                    volume = volume[:, slc, :]
                else:
                    volume = volume[:, :, slc]
                current_shape = volume.shape
        # After cropping, compute padding
        pad_sizes = [max(t - s, 0) for s, t in zip(volume.shape, target_size)]
        pad_before = [p // 2 for p in pad_sizes]
        pad_after = [p - b for p, b in zip(pad_sizes, pad_before)]
        out_slice = tuple(
            slice(b, b + volume.shape[i]) for i, b in enumerate(pad_before)
        )
        out[out_slice] = volume
        return out


def create_group_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate group‑wise k‑fold splits based on the ``ID`` column of a DataFrame.

    This function returns a list of tuples ``(train_idx, val_idx)`` where
    ``train_idx`` and ``val_idx`` are NumPy arrays of indices into ``df``.
    The splits are produced by ``GroupKFold`` so that all samples with the
    same ``ID`` appear in only one of the training or validation sets.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing an ``ID`` column.
    n_splits : int, optional
        Number of folds for cross‑validation.  Default is 5.
    random_state : int, optional
        Shuffling seed passed to the internal random number generator before
        splitting.  ``GroupKFold`` itself does not support shuffling, so if
        you want reproducible splits you should shuffle ``df`` prior to
        calling this function.  Default is 42.

    Returns
    -------
    List[Tuple[np.ndarray, np.ndarray]]
        List of ``(train_idx, val_idx)`` index arrays.
    """
    if GroupKFold is None:
        raise RuntimeError(
            "scikit‑learn must be installed to use create_group_splits()."
        )
    df_shuffled = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    groups = df_shuffled["ID"].values
    gkf = GroupKFold(n_splits=n_splits)
    splits = []
    for train_index, val_index in gkf.split(df_shuffled, groups=groups):
        splits.append((train_index, val_index))
    return splits


def compute_class_weights(df: pd.DataFrame, num_classes: int = 3) -> np.ndarray:
    """Compute per‑class weights to mitigate class imbalance.

    The hippocampi occupy a very small fraction of each MR volume.  This
    function sums the number of voxels of each label across all segmentations
    referenced in ``df['filepath_label']`` and computes a weight inversely
    proportional to the class frequency.  The weights are scaled so that
    their sum equals the number of classes.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with a ``filepath_label`` column pointing to NIfTI label files.
    num_classes : int, optional
        Number of segmentation classes (including background).  Default is 3.

    Returns
    -------
    np.ndarray
        Array of shape ``(num_classes,)`` containing the weight for each class.
    """
    if nib is None:
        raise RuntimeError(
            "nibabel must be installed to compute class weights from NIfTI files."
        )
    voxel_counts = np.zeros(num_classes, dtype=np.float64)
    for _, row in df.iterrows():
        label_path = row.get("filepath_label")
        if not label_path or not os.path.exists(label_path):
            continue
        lab = nib.load(label_path).get_fdata().astype(np.int64)
        for c in range(num_classes):
            voxel_counts[c] += np.sum(lab == c)
    # Avoid division by zero
    voxel_counts = np.clip(voxel_counts, a_min=1.0, a_max=None)
    frequencies = voxel_counts / np.sum(voxel_counts)
    # Inverse log weighting; adjust exponent for more or less smoothing
    weights = 1.0 / (np.log(1.0 + frequencies))
    # Normalise so that sum(weights) = num_classes
    weights = weights / np.sum(weights) * num_classes
    return weights
