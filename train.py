"""
Training script for the RISE‑MICCAI LISA 2025 hippocampus segmentation task.

This script trains one of the provided 3D segmentation models (UNesT,
SegResNet or Autoencoder) using k‑fold group cross‑validation.  The data
are loaded from a CSV file listing the paths to the CISO images and the
merged HF/LF hippocampus segmentations.  Splits are created using the
subject identifiers (``ID`` column) to prevent leakage between high‑ and
low‑field volumes of the same subject.

The training loop combines a weighted Dice loss with weighted cross‑entropy
to mitigate class imbalance and uses an AdamW optimiser with a cosine
annealing learning rate schedule.  The best model for each fold (in terms
of mean Dice score across classes) is saved to disk.

Example usage:

```bash
python train.py \
  --train_csv results/preprocessed_data/task2/df_train_hipp.csv \
  --output_dir /data/cristian/projects/med_data/rise-miccai/task-2/3d_models/results/model_unest_01 \
  --model unest \
  --folds 3 \
  --epochs 50 \
  --batch_size 4 \
  --gpu 5
```
"""

from __future__ import annotations

import argparse
import os
import json
import time
from typing import List

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
import tqdm

from src.dataset import HippocampusDataset, create_group_splits, compute_class_weights
from src.models import get_model
from src.utils import CombinedLoss, compute_dice_per_class


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train 3D segmentation models for the RISE‑MICCAI challenge.")
    parser.add_argument("--train_csv", type=str, required=True, help="Path to the training CSV file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save trained models and logs.")
    parser.add_argument("--model", type=str, default="unest", choices=["unest", "segresnet", "autoencoder"], help="Model architecture to train.")
    parser.add_argument("--folds", type=int, default=5, help="Number of cross‑validation folds.")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs per fold.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training.  3D volumes are memory intensive; adjust accordingly.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay for AdamW optimiser.")
    parser.add_argument("--dice_weight", type=float, default=1.0, help="Weight of the Dice loss component.")
    parser.add_argument("--ce_weight", type=float, default=1.0, help="Weight of the cross‑entropy loss component.")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of worker processes for data loading.")
    parser.add_argument("--resample_spacing", type=float, nargs=3, default=None, help="Optional voxel spacing to resample images/labels to, e.g. 1.0 1.0 1.0.")
    parser.add_argument("--gpu", type=int, default=None, help="CUDA device index to use.  If omitted, CPU is used.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--unest_cache", type=str, default=None, help="Local directory to cache the pretrained UNesT weights.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    import random
    random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def train_fold(
    fold: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    """Train a single fold of the cross‑validation experiment.

    Parameters
    ----------
    fold : int
        Index of the current fold (0‑based).
    train_df : pandas.DataFrame
        DataFrame containing the training split.
    val_df : pandas.DataFrame
        DataFrame containing the validation split.
    args : argparse.Namespace
        Parsed command‑line arguments.
    device : torch.device
        Device on which to perform training and inference.
    """
    # Create output directory for this fold
    fold_dir = os.path.join(args.output_dir, f"fold_{fold}")
    os.makedirs(fold_dir, exist_ok=True)

    # Compute class weights for this fold
    class_weights = compute_class_weights(train_df, num_classes=3)
    # Save class weights to file for reference
    np.save(os.path.join(fold_dir, "class_weights.npy"), class_weights)

    # Initialise datasets
    train_ds = HippocampusDataset(
        train_df,
        is_test=False,
        resample_spacing=tuple(args.resample_spacing) if args.resample_spacing else None,
    )
    val_ds = HippocampusDataset(
        val_df,
        is_test=False,
        resample_spacing=tuple(args.resample_spacing) if args.resample_spacing else None,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Build the chosen model
    model_kwargs = {}
    if args.model == "unest" and args.unest_cache:
        model_kwargs["cache_dir"] = args.unest_cache
    model = get_model(
        name=args.model,
        num_classes=3,
        device=str(device),
        **model_kwargs,
    )

    # Define loss and optimiser
    criterion = CombinedLoss(
        dice_weight=args.dice_weight,
        ce_weight=args.ce_weight,
        class_weights=class_weights,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_mean_dice = 0.0
    log_file = open(os.path.join(fold_dir, "training_log.jsonl"), "w")
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        # ---- Training ----
        model.train()
        train_losses: List[float] = []
        pbar = tqdm.tqdm(train_loader, desc=f"Train Fold {fold} Epoch {epoch}", leave=False, dynamic_ncols=True)
        for step,batch in enumerate(pbar, start=1):
        #for batch in train_loader:
            imgs = batch["image"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        scheduler.step()

        # ---- Validation ----
        model.eval()
        dice_scores: List[np.ndarray] = []
        with torch.no_grad():
            #pbar = tqdm.tqdm(train_loader, desc=f"Val  Fold {fold} Epoch {epoch}", leave=False, dynamic_ncols=True)
            for batch in val_loader:
                #for step,batch in enumerate(val_loader, start=1):
                imgs = batch["image"].to(device)
                labels = batch["label"].to(device)
                outputs = model(imgs)
                dice = compute_dice_per_class(outputs, labels, num_classes=3)
                dice_scores.append(dice)
        # Compute mean Dice per class and overall
        dice_array = np.stack(dice_scores, axis=0)  # shape (B, C)
        mean_dice_per_class = dice_array.mean(axis=0).tolist()
        mean_dice = float(np.mean(dice_array))

        # Save best model
        if mean_dice > best_mean_dice:
            best_mean_dice = mean_dice
            model_path = os.path.join(fold_dir, "best_model.pth")
            torch.save({"model_state_dict": model.state_dict(), "mean_dice": best_mean_dice}, model_path)

        # Logging
        log_entry = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)) if train_losses else None,
            "mean_dice": mean_dice,
            "dice_per_class": mean_dice_per_class,
            "elapsed_sec": time.time() - epoch_start,
        }
        log_file.write(json.dumps(log_entry) + "\n")
        log_file.flush()
        print(f"Fold {fold} | Epoch {epoch}/{args.epochs}: loss={log_entry['train_loss']:.4f}, mean_dice={mean_dice:.4f}")
    log_file.close()


def main() -> None:
    args = parse_args()
    if torch is None:
        raise RuntimeError(
            "PyTorch must be installed to run training. Please install torch and its dependencies."
        )
    # Set random seed
    set_seed(args.seed)
    # Prepare device
    if args.gpu is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    # Read CSV
    df = pd.read_csv(args.train_csv)
    # Create folds
    splits = create_group_splits(df, n_splits=args.folds, random_state=args.seed)
    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f"Starting fold {fold + 1}/{len(splits)}...")
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        train_fold(fold, train_df, val_df, args, device)


if __name__ == "__main__":
    main()
