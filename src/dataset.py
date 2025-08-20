"""
Dataset and data utilities for the RISE‑MICCAI LISA 2025 hippocampus segmentation task.

This module leverages MONAI's data loading and transformation pipeline to
streamline preprocessing, resampling, cropping/padding and augmentation of
3D volumetric MRI data.  Unlike the original implementation, which
performed these steps manually with ``nibabel``, ``numpy`` and ``scipy``,
the functions here delegate these responsibilities to MONAI transforms.

The key entry points are:

* :func:`get_data_list` – convert a ``pandas.DataFrame`` with filepaths into
  a list of dictionaries compatible with MONAI datasets.
* :func:`get_train_transform` and :func:`get_val_transform` – assemble
  deterministic and random transforms for training and validation/test
  pipelines, including spacing normalisation, intensity scaling and
  optional augmentation.
* :func:`get_dataset` – wrap a data list and transform into a
  ``monai.data.Dataset``.  You may substitute ``monai.data.CacheDataset``
  or ``PersistentDataset`` here to accelerate repeated data loading.
* :func:`create_group_splits` – generate group‑wise cross‑validation
  splits using subject identifiers to prevent leakage between HF/LF
  scans of the same patient.
* :func:`compute_class_weights` – compute inverse‑frequency class weights
  from the segmentation volumes.  This function still uses ``nibabel`` to
  read the label files because MONAI transforms operate within the
  training loop; computing weights beforehand avoids loading the
  entire dataset into memory multiple times.

To use this module, install MONAI and nibabel in your environment.  The
functions will raise an exception at runtime if these dependencies are
missing.
"""

from __future__ import annotations

import os
from typing import List, Tuple, Optional, Sequence

import numpy as np
import pandas as pd
from monai.transforms import RandRotated, RandZoomd, Rand3DElasticd
from monai.transforms import RandBiasFieldd, RandHistogramShiftd, RandAdjustContrastd
from monai.transforms import RandCoarseDropoutd




try:
    # Core MONAI components for data loading and transforms
    from monai.data import Dataset
    from monai.transforms import (
        Compose,
        LoadImaged,
        EnsureChannelFirstd,
        Spacingd,
        ScaleIntensityRanged,
        ResizeWithPadOrCropd,
        RandFlipd,
        RandRotate90d,
        RandAffined,
        RandGaussianNoised,
        ToTensord,
        EnsureTyped,
    )
except ImportError as e:
    Dataset = None  # type: ignore
    Compose = LoadImaged = EnsureChannelFirstd = Spacingd = ScaleIntensityRanged = None  # type: ignore
    ResizeWithPadOrCropd = RandFlipd = RandRotate90d = RandAffined = RandGaussianNoised = ToTensord = None  # type: ignore
    EnsureTyped = None  # type: ignore

try:
    import nibabel as nib
except ImportError:
    nib = None  # type: ignore

try:
    from sklearn.model_selection import GroupKFold
except ImportError:
    GroupKFold = None  # type: ignore


def get_data_list(df: pd.DataFrame, is_test: bool = False) -> List[dict]:
    """Convert a DataFrame of filepaths into a list of dictionaries for MONAI.

    Each entry in the returned list has keys ``"image"`` and, if
    ``is_test`` is ``False``, ``"label"``.  MONAI's ``LoadImaged``
    transform recognises these keys and loads the corresponding NIfTI
    volumes.  The ``ID`` column is ignored at this stage but can be
    used externally to perform group‑wise splits.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing at least the column ``filepath`` and
        optionally ``filepath_label`` when labels are available.
    is_test : bool, optional
        If ``True``, the data list will include only the ``image`` key.

    Returns
    -------
    List[dict]
        A list of dictionaries mapping key names to filepaths.
    """
    data: List[dict] = []
    for _, row in df.iterrows():
        item = {"image": row["filepath"]}
        if not is_test and "filepath_label" in row and pd.notna(row["filepath_label"]):
            item["label"] = row["filepath_label"]
        data.append(item)
    return data


def get_train_transform(
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    spatial_size: Tuple[int, int, int] = (96, 96, 96),
    augment: bool = True,
) -> Compose:
    """Create a MONAI transform pipeline for training.

    The pipeline includes loading images/labels, ensuring channel order,
    resampling to the specified voxel spacing, scaling intensities to
    [0, 1], resizing with padding/cropping to the target spatial size and
    optional random augmentations.  Finally, it converts the numpy arrays
    into PyTorch tensors.

    Parameters
    ----------
    spacing : tuple of floats
        Target voxel spacing (mm) for resampling.  Must match the spacing
        used during training and inference.
    spatial_size : tuple of ints
        Desired output size (depth, height, width).
    augment : bool, optional
        If ``True``, include random flips, rotations, affine
        transformations and additive noise.  Otherwise only deterministic
        preprocessing is applied.  Default is ``True``.

    Returns
    -------
    monai.transforms.Compose
        A composition of transforms to be applied sequentially.
    """
    if Compose is None:
        raise ImportError("MONAI is required to build the transform pipeline.")
    transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        # Resample to the desired spacing.  Image uses trilinear interpolation;
        # label uses nearest neighbour to preserve discrete values.
        Spacingd(keys=["image", "label"], pixdim=spacing, mode=("bilinear", "nearest")),
        # Scale intensities from min/max range to [0, 1]
        ScaleIntensityRanged(
            keys=["image"],
            a_min=0.0,
            a_max=1.0,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        # Resize or pad volumes to the target spatial size
        ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=spatial_size),
    ]
    if augment:
        transforms += [
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandRotate90d(keys=["image", "label"], prob=0.3, max_k=3),
            RandAffined(
                keys=["image", "label"],
                prob=0.3,
                rotate_range=(0.1, 0.1, 0.1),
                translate_range=(20, 20,10),  # traslación máx en voxeles en (x,y,z)
                scale_range=(0.1, 0.1, 0.1),
                mode=("bilinear", "nearest"),
                padding_mode="border",
            ),
            RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.01),

            RandRotated(keys=["image", "label"], prob=0.2, range_x=0.26, range_y=0.26, range_z=0.26),  # ±15°
            RandZoomd(keys=["image", "label"], prob=0.2, min_zoom=0.9, max_zoom=1.1, mode=("trilinear", "nearest")),
            Rand3DElasticd(keys=["image", "label"], prob=0.1, sigma_range=(5,7), magnitude_range=(50,100), mode=("bilinear", "nearest")),

            RandBiasFieldd(keys=["image"], prob=0.2, coeff_range=(0.0, 0.5)),
            RandHistogramShiftd(keys=["image"], prob=0.3, num_control_points=5, shift_range=(-0.05, 0.05)),
            RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.7, 1.5)),

            RandCoarseDropoutd(
                keys=["image", "label"],
                prob=0.15,
                holes=2,
                spatial_size=(32, 32, 32),
                fill_value=0,
                max_holes=5
            ),

        ]
    # Convert to MetaTensor and preserve metadata for inverse transforms. Do not
    # call ToTensord here, as it strips metadata. DataLoader will convert
    # to tensors automatically.
    # Convert to MetaTensor and preserve metadata for inverse transforms. Do not call
    # ToTensord here, as it strips metadata. The DataLoader will handle
    # conversion to torch.Tensor. Returning Compose once is sufficient.
    transforms += [EnsureTyped(keys=["image", "label"], track_meta=True)]
    return Compose(transforms)


def get_val_transform(
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    spatial_size: Tuple[int, int, int] = (96, 96, 96),
) -> Compose:
    """Create a MONAI transform pipeline for validation and testing.

    This pipeline performs only deterministic preprocessing: loading,
    resampling, intensity scaling, resizing/cropping and conversion to
    tensors.  No random augmentations are applied.

    Parameters
    ----------
    spacing : tuple of floats
        Target voxel spacing (mm) for resampling.  Must match the spacing
        used during training.
    spatial_size : tuple of ints
        Desired output size (depth, height, width).

    Returns
    -------
    monai.transforms.Compose
        A composition of transforms for validation or test datasets.
    """
    if Compose is None:
        raise ImportError("MONAI is required to build the transform pipeline.")
    transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=spacing, mode=("bilinear", "nearest")),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=0.0,
            a_max=1.0,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=spatial_size),
        EnsureTyped(keys=["image", "label"], track_meta=True),
    ]
    return Compose(transforms)


def get_test_transform(
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    spatial_size: Tuple[int, int, int] = (96, 96, 96),
) -> Compose:
    """Create a MONAI transform pipeline for inference on unseen data.

    The test transform is identical to the validation transform except
    that it does not expect a ``label`` key and therefore omits label
    operations.  If your DataFrame includes a ``filepath_label`` column
    for a back‑test set, you may use the validation transform instead.
    """
    if Compose is None:
        raise ImportError("MONAI is required to build the transform pipeline.")
    # Inference transform: load image, ensure channel, resample to specified spacing,
    # normalize intensities, and keep metadata. We avoid padding/cropping here and
    # let sliding_window_inference handle patch-wise processing on the full volume.
    transforms = [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Spacingd(keys=["image"], pixdim=spacing, mode="bilinear"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=0.0,
            a_max=1.0,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        # Track metadata for inversion. DataLoader will convert to tensors automatically.
        EnsureTyped(keys=["image"], track_meta=True),
    ]
    return Compose(transforms)


def get_dataset(
    df: pd.DataFrame,
    is_train: bool,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    spatial_size: Tuple[int, int, int] = (96, 96, 96),
    augment: bool = True,
    is_test: bool = False,
) -> Dataset:
    """Create a MONAI dataset with the appropriate transforms.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing filepaths.
    is_train : bool
        Whether the dataset will be used for training.  Determines the
        transform used.
    spacing : tuple of floats
        Voxel spacing for resampling.
    spatial_size : tuple of ints
        Output volume size.
    augment : bool, optional
        When ``is_train`` is ``True``, controls whether random
        augmentation is added.  Ignored for validation/test datasets.
    is_test : bool, optional
        If ``True``, create a dataset without labels (for inference on
        unlabeled data).  The ``is_train`` flag is ignored in this case.

    Returns
    -------
    monai.data.Dataset
        A dataset yielding dictionaries with keys ``image`` and
        optionally ``label``.
    """
    if Dataset is None:
        raise ImportError("MONAI must be installed to use get_dataset().")
    data_list = get_data_list(df, is_test=is_test)
    if is_test:
        transform = get_test_transform(spacing=spacing, spatial_size=spatial_size)
    else:
        if is_train:
            transform = get_train_transform(spacing=spacing, spatial_size=spatial_size, augment=augment)
        else:
            transform = get_val_transform(spacing=spacing, spatial_size=spatial_size)
    return Dataset(data=data_list, transform=transform)


def create_group_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate group‑wise k‑fold splits based on the ``ID`` column of a DataFrame.

    This function uses scikit‑learn's ``GroupKFold`` to ensure that all
    samples belonging to the same subject are placed in the same fold.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing an ``ID`` column identifying the subject.
    n_splits : int, optional
        Number of cross‑validation folds.  Default is 5.
    random_state : int, optional
        Seed used to shuffle the data before splitting.  Note that
        ``GroupKFold`` itself does not support shuffling, so the shuffle
        happens on the DataFrame indices prior to splitting.  Default is 42.

    Returns
    -------
    List[Tuple[np.ndarray, np.ndarray]]
        A list of tuples ``(train_idx, val_idx)`` containing the indices
        for each fold.
    """
    if GroupKFold is None:
        raise RuntimeError("scikit‑learn must be installed to create group splits.")
    df_shuffled = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    groups = df_shuffled["ID"].values
    gkf = GroupKFold(n_splits=n_splits)
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for train_index, val_index in gkf.split(df_shuffled, groups=groups):
        splits.append((train_index, val_index))
    return splits


def compute_class_weights(df: pd.DataFrame, num_classes: int = 3) -> np.ndarray:
    """Compute inverse‑frequency class weights from the label volumes.

    This helper iterates over the ``filepath_label`` column of the
    provided DataFrame, loads each NIfTI label using ``nibabel`` and
    counts the number of voxels belonging to each class.  It then
    computes weights inversely proportional to the logarithm of the
    class frequencies, normalising the weights so that they sum to
    ``num_classes``.  These weights can be passed to loss functions to
    mitigate class imbalance.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with a ``filepath_label`` column pointing to label
        volumes.  Rows without labels are ignored.
    num_classes : int, optional
        Total number of segmentation classes (including background).  Default is 3.

    Returns
    -------
    np.ndarray
        An array of shape ``(num_classes,)`` containing the weight for
        each class.
    """
    if nib is None:
        raise RuntimeError("nibabel must be installed to compute class weights.")
    voxel_counts = np.zeros(num_classes, dtype=np.float64)
    for _, row in df.iterrows():
        label_path = row.get("filepath_label")
        if not label_path or not os.path.exists(label_path):
            continue
        lab = nib.load(label_path).get_fdata().astype(np.int64)
        for c in range(num_classes):
            voxel_counts[c] += np.sum(lab == c)
    # Prevent division by zero
    voxel_counts = np.clip(voxel_counts, a_min=1.0, a_max=None)
    frequencies = voxel_counts / np.sum(voxel_counts)
    # Use inverse log weighting to dampen large class weight differences
    weights = 1.0 / (np.log(1.0 + frequencies))
    # Normalise so that weights sum to the number of classes
    weights = weights / np.sum(weights) * num_classes
    return weights
