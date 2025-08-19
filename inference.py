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
  --model_dirs /data/cristian/projects/med_data/rise-miccai/task-2/3d_models/results/model_unest_02/fold_0,/data/cristian/projects/med_data/rise-miccai/task-2/3d_models/results/model_unest_02/fold_1,/data/cristian/projects/med_data/rise-miccai/task-2/3d_models/results/model_unest_02/fold_2 \
  --model unest \
  --output_dir /data/cristian/projects/med_data/rise-miccai/task-2/3d_models/predictions/model_unest_02/
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

from src.dataset import get_dataset
from src.models import get_model
from src.utils import compute_dice_per_class
import tqdm


def parse_args() -> argparse.Namespace:
    """Parse arguments for inference.

    In addition to the baseline options, this parser allows for model ensembling by
    passing multiple ``--model_dirs`` and corresponding ``--models``.  When
    multiple models are supplied, the predictions from each network are combined
    by averaging their softmax probabilities.
    """
    parser = argparse.ArgumentParser(description="Generate hippocampus segmentations on unseen data.")
    parser.add_argument("--test_csv", type=str, required=True, help="Path to the CSV file listing test images.")
    # Backwards compatibility: allow a single model directory
    parser.add_argument("--model_dir", type=str, default=None, help="Directory containing a trained model (best_model.pth).")
    parser.add_argument("--model", type=str, default=None, choices=["unest", "segresnet", "autoencoder"], help="Architecture used by the single model.")
    # New: support multiple model directories and architectures
    parser.add_argument(
        "--model_dirs",
        type=str,
        default=None,
        help="Comma‑separated list of directories, each containing a best_model.pth checkpoint.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma‑separated list of architectures corresponding to --model_dirs (e.g. 'unest,segresnet,autoencoder').",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to write the predicted NIfTI files.")
    parser.add_argument(
        "--spacing",
        type=float,
        nargs=3,
        default=(1.0, 1.0, 1.0),
        help="Voxel spacing used during training for resampling. Must match the training script.",
    )
    parser.add_argument(
        "--spatial_size",
        type=int,
        nargs=3,
        default=(96, 96, 96),
        help="Spatial size (depth, height, width) used during training. Must match the network input size.",
    )
    parser.add_argument("--gpu", type=int, default=None, help="CUDA device index.  Defaults to CPU.")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of workers for DataLoader.")
    parser.add_argument("--eval", action="store_true", help="If set and ground truth labels are available, compute and print Dice metrics.")
    parser.add_argument(
        "--unest_cache",
        type=str,
        default=None,
        help="Local cache directory for the pretrained UNesT weights. Only used when loading UNesT models.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if torch is None or nib is None:
        raise RuntimeError(
            "PyTorch and nibabel are required for inference. Please install the missing dependencies."
        )
    # Choose device
    if args.gpu is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    # Read CSV
    df = pd.read_csv(args.test_csv)
    # Determine which model directories and architectures to use
    model_dirs: list[str] = []
    model_names: list[str] = []
    # If multiple model_dirs specified, parse them; otherwise fall back to single
    if args.model_dirs:
        model_dirs = [m.strip() for m in args.model_dirs.split(",") if m.strip()]
        if args.models:
            model_names = [m.strip() for m in args.models.split(",") if m.strip()]
            if len(model_names) != len(model_dirs):
                raise ValueError("--models must have the same number of entries as --model_dirs")
        else:
            # If no individual models given, replicate either args.model or default to 'unest'
            base_model = args.model if args.model is not None else "unest"
            model_names = [base_model] * len(model_dirs)
    else:
        # Fallback to single model
        if args.model_dir is None:
            raise ValueError("Must specify either --model_dir or --model_dirs")
        if args.model is None:
            raise ValueError("For a single model, --model must also be specified")
        model_dirs = [args.model_dir]
        model_names = [args.model]
    # Instantiate the dataset.  If ``args.eval`` is True and a ``filepath_label`` column is present,
    # labels will be loaded; otherwise the dataset will omit the ``label`` key.
    is_test_dataset = not args.eval or ("filepath_label" not in df.columns)
    test_ds = get_dataset(
        df,
        is_train=False,
        spacing=tuple(args.spacing),
        spatial_size=tuple(args.spatial_size),
        augment=False,
        is_test=is_test_dataset,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    # Preload all models
    models_list: list[torch.nn.Module] = []
    use_cuda = device.type == "cuda"
    for md, name in zip(model_dirs, model_names):
        # Build model.  When loading UNesT, provide cache_dir if specified.
        model_kwargs: dict = {}
        if name == "unest" and args.unest_cache:
            model_kwargs["cache_dir"] = args.unest_cache
        model = get_model(name=name, num_classes=3, device=str(device), **model_kwargs)
        # Load weights
        ckpt_path = os.path.join(md, "best_model.pth")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        models_list.append(model)
        print(f"Loaded model '{name}' from {ckpt_path}")
    # Inference loop with optional evaluation
    dice_scores_all: list[np.ndarray] = []
    pbar = tqdm.tqdm(test_loader, desc="Inference", total=len(test_loader), dynamic_ncols=True)
    for idx, sample in enumerate(pbar):
        img_tensor = sample["image"].to(device)
        # Average predictions from all models
        probs_sum = None
        with torch.no_grad():
            for model in models_list:
                with torch.cuda.amp.autocast(enabled=use_cuda):
                    logits = model(img_tensor)
                    probs = torch.softmax(logits, dim=1)
                probs_sum = probs if probs_sum is None else probs_sum + probs
            avg_probs = probs_sum / len(models_list)
            pred_labels = torch.argmax(avg_probs, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        # For evaluation (if labels exist and requested), compute Dice
        if args.eval and ("filepath_label" in df.columns):
            labels_tensor = sample.get("label")
            if labels_tensor is not None:
                labels_tensor = labels_tensor.to(device)
                # We need logits for compute_dice_per_class.  Use avg_probs converted to logits via logit transform
                # but compute_dice_per_class uses argmax, so we can wrap avg_probs as logits
                avg_logits = torch.log(avg_probs + 1e-8)
                dice = compute_dice_per_class(avg_logits, labels_tensor, num_classes=3)
                dice_scores_all.append(dice)
        # Rescale prediction back to original shape
        row = df.iloc[idx]
        img_path = row["filepath"]
        orig_img_nii = nib.load(img_path)
        orig_shape = orig_img_nii.shape
        orig_affine = orig_img_nii.affine
        current_size = pred_labels.shape
        zoom_factors = [o / c for o, c in zip(orig_shape, current_size)]
        try:
            import scipy.ndimage
            pred_resized = scipy.ndimage.zoom(pred_labels, zoom=zoom_factors, order=0)
        except ImportError:
            if current_size != orig_shape:
                raise RuntimeError(
                    "scipy.ndimage is required to rescale prediction to original shape."
                )
            pred_resized = pred_labels
        pred_resized = pred_resized.astype(np.uint8)
        # Construct output filename following challenge spec
        subject_id = row["ID"]
        numeric_id = re.findall(r"\d+", str(subject_id))
        numeric_id_str = numeric_id[-1] if numeric_id else str(subject_id)
        out_fname = f"LISAHF{numeric_id_str}segprediction.nii.gz"
        out_path = os.path.join(args.output_dir, out_fname)
        pred_nii = nib.Nifti1Image(pred_resized, affine=orig_affine)
        nib.save(pred_nii, out_path)
        pbar.set_postfix({"subject": str(subject_id)})
    # If evaluating, summarise and report mean Dice
    if args.eval and dice_scores_all:
        dice_array = np.stack(dice_scores_all, axis=0)
        mean_dice_per_class = dice_array.mean(axis=0)
        mean_dice_overall = float(np.mean(dice_array))
        print("\nEvaluation Summary:")
        print(f"Mean Dice per class: {mean_dice_per_class}")
        print(f"Mean Dice across all classes and subjects: {mean_dice_overall:.4f}")



if __name__ == "__main__":
    main()
