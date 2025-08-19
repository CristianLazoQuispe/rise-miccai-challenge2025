"""
Inference script for the RISE‑MICCAI LISA 2025 hippocampus segmentation task.

This script loads a trained model from disk and generates hippocampus
segmentations for a set of unseen CISO images listed in a CSV file.  For
each subject in the CSV, the corresponding NIfTI image is loaded,
resampled and resized to the input size expected by the network.  The
predicted segmentation is then mapped back to the original resolution and
saved with the naming convention required by the challenge.

Example usage:

```bash
python inference.py \
  --test_csv results/preprocessed_data/task2/df_test_hipp.csv \
  --model_dir /data/cristian/projects/med_data/rise-miccai/task-2/3d_models/results/model_unest_01/fold_0 \
  --model unest \
  --output_dir /data/cristian/projects/med_data/rise-miccai/task-2/3d_models/predictions/model_unest_01/
```
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Optional

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import DataLoader
    import nibabel as nib
except ImportError:
    torch = None  # type: ignore
    DataLoader = None  # type: ignore
    nib = None  # type: ignore

from src.dataset import HippocampusDataset
from src.models import get_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate hippocampus segmentations on unseen data.")
    parser.add_argument("--test_csv", type=str, required=True, help="Path to the CSV file listing test images.")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing the trained model (best_model.pth).")
    parser.add_argument("--model", type=str, default="unest", choices=["unest", "segresnet", "autoencoder"], help="Architecture used during training.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to write the predicted NIfTI files.")
    parser.add_argument("--resample_spacing", type=float, nargs=3, default=None, help="Voxel spacing used during training for resampling. Must match the training script.")
    parser.add_argument("--gpu", type=int, default=None, help="CUDA device index.  Defaults to CPU.")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of workers for DataLoader.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if torch is None or nib is None:
        raise RuntimeError(
            "PyTorch and nibabel are required for inference. Please install the missing dependencies."
        )
    # Prepare device
    if args.gpu is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    # Load test DataFrame
    df = pd.read_csv(args.test_csv)
    # Instantiate dataset and dataloader
    test_ds = HippocampusDataset(
        df,
        is_test=True,
        resample_spacing=tuple(args.resample_spacing) if args.resample_spacing else None,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    # Build the model and load the checkpoint
    model = get_model(name=args.model, num_classes=3, device=str(device))
    checkpoint_path = os.path.join(args.model_dir, "best_model.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    # Inference loop
    for idx, sample in enumerate(test_loader):
        img_tensor = sample["image"].to(device)
        row = df.iloc[idx]
        with torch.no_grad():
            logits = model(img_tensor)
            pred_labels = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        # Load original image for shape and affine
        img_path = row["filepath"]
        orig_img_nii = nib.load(img_path)
        orig_shape = orig_img_nii.shape
        orig_affine = orig_img_nii.affine
        # Upsample prediction to original shape using nearest neighbour interpolation
        # Compute zoom factors as ratio of original to current size
        current_size = pred_labels.shape
        zoom_factors = [o / c for o, c in zip(orig_shape, current_size)]
        try:
            import scipy.ndimage
            pred_resized = scipy.ndimage.zoom(pred_labels, zoom=zoom_factors, order=0)
        except ImportError:
            # If SciPy is unavailable, fall back to simple padding/cropping
            # This will work correctly only if current_size == orig_shape
            if current_size != orig_shape:
                raise RuntimeError(
                    "scipy.ndimage is required to rescale prediction to original shape."
                )
            pred_resized = pred_labels
        # Ensure type is integer
        pred_resized = pred_resized.astype(np.uint8)
        # Construct output filename: LISAHF<ID>segprediction.nii.gz
        subject_id = row["ID"]
        filename   = row["filename"]
        # Remove any prefix/suffix from ID (e.g., LISA_VALIDATION_0001 -> 0001)
        numeric_id = re.findall(r"\d+", str(subject_id))
        numeric_id_str = numeric_id[-1] if numeric_id else str(subject_id)
        out_fname = f"LISAHF{numeric_id_str}segprediction.nii.gz"
        
        out_path = os.path.join(args.output_dir, out_fname)
        # Save NIfTI with the original affine
        pred_nii = nib.Nifti1Image(pred_resized, affine=orig_affine)
        nib.save(pred_nii, out_path)
        print(f"Saved prediction for subject {subject_id} to {out_path}")


if __name__ == "__main__":
    main()
