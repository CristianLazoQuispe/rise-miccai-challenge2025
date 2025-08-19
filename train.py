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
  --output_dir /data/cristian/projects/med_data/rise-miccai/task-2/3d_models/results/model_unest_02 \
  --model unest \
  --folds 3 \
  --epochs 100 \
  --batch_size 4 \
  --gpu 5

python train.py \
  --train_csv results/preprocessed_data/task2/df_train_hipp.csv \
  --output_dir /data/cristian/projects/med_data/rise-miccai/task-2/3d_models/results/model_dynunet_02 \
  --model dynunet \
  --folds 3 \
  --epochs 100 \
  --batch_size 4 \
  --gpu 4
  

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

from src.dataset import (
    get_dataset,
    create_group_splits,
    compute_class_weights,
)
from sklearn.model_selection import GroupShuffleSplit

# Loss functions and metrics
from src.utils import (
    CombinedLoss,
    compute_dice_per_class,
    FocalTverskyLoss,
    weighted_cross_entropy,
    tversky_loss,
)
from src.models import get_model

from src.utils import get_metrics_3d, compute_metrics


def parse_args() -> argparse.Namespace:
    """Parse command‑line arguments for training.

    This function exposes additional options compared to the baseline implementation:

    * ``--test_split`` controls the fraction of subjects held out as a fixed back‑testing set.
    * ``--augment`` toggles data augmentation using MONAI transforms for the training set.
    """
    parser = argparse.ArgumentParser(description="Train 3D segmentation models for the RISE‑MICCAI challenge.")
    parser.add_argument("--train_csv", type=str, required=True, help="Path to the training CSV file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save trained models and logs.")
    parser.add_argument("--model", type=str, default="unest", choices=["unest", "swinunetr", "dynunet"], help="Model architecture to train.")
    parser.add_argument("--folds", type=int, default=5, help="Number of cross‑validation folds to generate from the non‑held‑out data.")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs per fold.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training.  3D volumes are memory intensive; adjust accordingly.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate for the optimiser.")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay for AdamW optimiser.")
    parser.add_argument("--dice_weight", type=float, default=1.0, help="Relative weight of the Dice loss component.")
    parser.add_argument("--ce_weight", type=float, default=1.0, help="Relative weight of the cross‑entropy loss component.")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of worker processes for data loading.")
    parser.add_argument(
        "--spacing",
        type=float,
        nargs=3,
        default=(1.0, 1.0, 1.0),
        help="Target voxel spacing for resampling (e.g. 1.0 1.0 1.0). Must match between training and inference.",
    )
    parser.add_argument(
        "--spatial_size",
        type=int,
        nargs=3,
        default=(96, 96, 96),
        help="Output spatial size (depth, height, width) for the network. Should match the model's expected input size.",
    )
    parser.add_argument("--gpu", type=int, default=None, help="CUDA device index.  Defaults to CPU when unspecified.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--unest_cache", type=str, default=None, help="Local cache directory for the pretrained UNesT weights.")
    parser.add_argument("--test_split", type=float, default=0.1, help="Fraction of the data reserved for back‑testing.  Must be between 0 and 1.")
    parser.add_argument("--augment", action="store_true", help="Enable random data augmentation for the training set.")
    parser.add_argument(
        "--loss",
        type=str,
        default="dice_ce",
        choices=["dice_ce", "focal_tversky", "tversky_ce"],
        help="Loss function to use: 'dice_ce' for combined Dice + cross-entropy, 'focal_tversky' for focal Tversky loss, or 'tversky_ce' for focal Tversky plus weighted cross-entropy.",
    )
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
) -> tuple[str, float]:
    """Train a single fold of the cross‑validation experiment.

    This function trains the specified model on ``train_df``, validates on ``val_df``
    each epoch and saves the model with the highest mean Dice score.
    Datasets and transforms are created internally using the MONAI
    utilities defined in ``src.dataset.get_dataset``.  Data
    augmentation is controlled via the ``augment`` flag in ``args``.

    Parameters
    ----------
    fold : int
        Zero‑based index of the current fold.
    train_df : pandas.DataFrame
        Training split for this fold.
    val_df : pandas.DataFrame
        Validation split for this fold.
    args : argparse.Namespace
        Parsed command‑line arguments.
    device : torch.device
        Target device for computation.

    Returns
    -------
    tuple[str, float]
        A tuple containing the path to the best model (saved checkpoint) and
        the highest mean Dice score achieved on the validation set.
    """
    # Create output directory for this fold
    fold_dir = os.path.join(args.output_dir, f"fold_{fold}")
    os.makedirs(fold_dir, exist_ok=True)

    # Compute class weights for this fold
    class_weights = compute_class_weights(train_df, num_classes=3)
    # Persist class weights for reference
    np.save(os.path.join(fold_dir, "class_weights.npy"), class_weights)

    # Instantiate datasets using MONAI utilities.  For training, ``augment`` controls whether
    # random augmentations are applied.  For validation, no augmentation is used.
    train_ds = get_dataset(
        train_df,
        is_train=True,
        spacing=tuple(args.spacing),
        spatial_size=tuple(args.spatial_size),
        augment=args.augment,
        is_test=False,
    )
    val_ds = get_dataset(
        val_df,
        is_train=False,
        spacing=tuple(args.spacing),
        spatial_size=tuple(args.spatial_size),
        augment=False,
        is_test=False,
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

    # Build the model.  When using UNesT, optionally supply a cache dir
    model_kwargs = {}
    if args.model == "unest" and args.unest_cache:
        model_kwargs["cache_dir"] = args.unest_cache
    model = get_model(
        name=args.model,
        num_classes=3,
        device=str(device),
        **model_kwargs,
    )

    metrics = get_metrics_3d(device)


    # Set up loss, optimiser, scheduler and AMP scaler
    # Choose loss function based on user specification.  For small objects, focal
    # Tversky loss can improve sensitivity to the hippocampi.  If combining
    # Tversky with cross‑entropy, the cross‑entropy term is scaled by
    # ``args.ce_weight``.
    if args.loss == "dice_ce":
        criterion = CombinedLoss(
            dice_weight=args.dice_weight,
            ce_weight=args.ce_weight,
            class_weights=class_weights,
        )
    elif args.loss == "focal_tversky":
        criterion = FocalTverskyLoss(
            alpha=0.3,
            beta=0.7,
            gamma=1.33,
            class_weights=class_weights,
        )
    elif args.loss == "tversky_ce":
        # Combine focal Tversky with weighted cross‑entropy.  The Tversky
        # component is weighted by ``args.dice_weight`` and the
        # cross‑entropy component by ``args.ce_weight``.
        focal_tversky = FocalTverskyLoss(
            alpha=0.3,
            beta=0.7,
            gamma=1.33,
            class_weights=class_weights,
        )
        def combined_loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            return args.dice_weight * focal_tversky(logits, targets) + args.ce_weight * weighted_cross_entropy(logits, targets, class_weights)
        # Wrap the callable in a nn.Module so that it can be used like a loss
        class _WrappedLoss(torch.nn.Module):
            def forward(self, logits, targets):
                return combined_loss_fn(logits, targets)
        criterion = _WrappedLoss()
    else:
        raise ValueError(f"Unknown loss type: {args.loss}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    use_cuda = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)

    best_mean_dice = 0.0
    best_model_path = os.path.join(fold_dir, "best_model.pth")
    # Open a JSONL log file for per‑epoch metrics
    log_file = open(os.path.join(fold_dir, "training_log.jsonl"), "w")

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        # ---- Training ----
        model.train()
        train_losses: List[float] = []
        pbar = tqdm.tqdm(train_loader, desc=f"Train Fold {fold} Epoch {epoch}", leave=False, dynamic_ncols=True)
        for batch in pbar:
            imgs = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad()
            # AMP forward/backward
            with torch.cuda.amp.autocast(enabled=use_cuda):
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(loss.item())
            pbar.set_postfix({"train_loss": f"{np.mean(train_losses):.4f}"})
        scheduler.step()

        # ---- Validation ----
        model.eval()
        dice_scores: List[np.ndarray] = []
        hd95_scores: List[float] = []
        surf_dice_scores: List[float] = []
        val_losses: List[float] = []
        pbar_val = tqdm.tqdm(val_loader, desc=f"Val   Fold {fold} Epoch {epoch}", leave=False, dynamic_ncols=True)
        with torch.no_grad():
            for batch in pbar_val:
                imgs = batch["image"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=use_cuda):
                    outputs = model(imgs)
                    # Compute loss for monitoring; use same criterion
                    val_loss = criterion(outputs, labels).item()
                val_losses.append(val_loss)
                #print(f"outputs shape: {outputs.shape}, labels shape: {labels.shape}")
                dice = compute_dice_per_class(outputs, labels, num_classes=3)
                #results = compute_metrics(outputs, labels, metrics)
                #print("Dice:", results["dice"], "HD95:", results["hd95"], "SurfaceDice:", results["surf_dice"])
                #dice = results["dice"]
                hd95 =  0 #results["hd95"]
                surf_dice = 0 #results["surf_dice"]
                # Store the Dice scores for each class
                dice_scores.append(dice)
                hd95_scores.append(hd95)
                surf_dice_scores.append(surf_dice)
                pbar_val.set_postfix({"val_loss": f"{np.mean(val_losses):.4f} | HD95: {hd95:.2f} | SurfDice: {surf_dice:.2f}    | Dice: {dice.mean():.4f}"})

        # Compute mean Dice per class and overall
        dice_array = np.stack(dice_scores, axis=0) if dice_scores else np.zeros((0, 3))
        mean_dice_per_class = dice_array.mean(axis=0).tolist() if dice_scores else [0.0, 0.0, 0.0]
        mean_dice = float(np.mean(dice_array)) if dice_scores else 0.0
        mean_train_loss = float(np.mean(train_losses)) if train_losses else float('nan')
        mean_val_loss = float(np.mean(val_losses)) if val_losses else float('nan')

        mean_hd95 = float(np.mean(hd95_scores)) if hd95_scores else float('nan')
        mean_surf_dice = float(np.mean(surf_dice_scores)) if surf_dice_scores else float('nan')
        print(
            f"Fold {fold} | Epoch {epoch}/{args.epochs}: "
            f"train_loss={mean_train_loss:.4f}, "
            f"val_loss={mean_val_loss:.4f}, "
            f"mean_dice={mean_dice:.4f}, "
            #f"mean_hd95={mean_hd95:.2f}, "
            #f"mean_surf_dice={mean_surf_dice:.2f}, "
            f"mean_dice_per_class={mean_dice_per_class}"
        )
        # Save the best model based on validation mean Dice
        if mean_dice > best_mean_dice:
            best_mean_dice = mean_dice
            torch.save({"model_state_dict": model.state_dict(), "mean_dice": best_mean_dice}, best_model_path)

        # Log the metrics
        log_entry = {
            "epoch": epoch,
            "train_loss": mean_train_loss,
            "val_loss": mean_val_loss,
            "mean_dice": mean_dice,
            "mean_hd95": mean_hd95,
            "mean_surf_dice": mean_surf_dice,
            "fold": fold,
            "mean_dice_per_class": mean_dice_per_class,
            "dice_per_class": mean_dice_per_class,
            "elapsed_sec": time.time() - epoch_start,
        }
        log_file.write(json.dumps(log_entry) + "\n")
        log_file.flush()
        # Print a concise summary to stdout
        print(
            f"Fold {fold} | Epoch {epoch}/{args.epochs}: train_loss={mean_train_loss:.4f}, val_loss={mean_val_loss:.4f}, mean_dice={mean_dice:.4f}"
        )
    log_file.close()
    return best_model_path, best_mean_dice


def evaluate_on_dataset(
    model_path: str,
    model_name: str,
    df: pd.DataFrame,
    args: argparse.Namespace,
    device: torch.device,
) -> float:
    """Evaluate a saved model checkpoint on a dataset and return the mean Dice.

    Parameters
    ----------
    model_path : str
        Path to the ``.pth`` checkpoint containing ``model_state_dict``.
    model_name : str
        Name of the model architecture ("unest", "segresnet", "autoencoder").
    df : pandas.DataFrame
        DataFrame with columns ``filepath`` and ``filepath_label``.
    args : argparse.Namespace
        Command‑line arguments used for training; only ``spacing``, ``spatial_size``
        and ``unest_cache`` are relevant here.
    device : torch.device
        Device on which to perform inference.
    Returns
    -------
    float
        Mean Dice score across all subjects and classes.
    """
    # If no labels present (is_test), we cannot compute Dice
    if "filepath_label" not in df.columns:
        raise ValueError("Back‑test evaluation requires ground truth labels.")
    # Instantiate dataset and loader
    # Create a validation dataset using the same spacing and spatial size as training.
    ds = get_dataset(
        df,
        is_train=False,
        spacing=tuple(args.spacing),
        spatial_size=tuple(args.spatial_size),
        augment=False,
        is_test=False,
    )
    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    # Build model and load weights
    model_kwargs: dict = {}
    if model_name == "unest" and args.unest_cache:
        model_kwargs["cache_dir"] = args.unest_cache
    model = get_model(name=model_name, num_classes=3, device=str(device), **model_kwargs)
    ckpt = torch.load(model_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    use_cuda = device.type == "cuda"
    dice_scores: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device)
            labels = batch["label"].to(device)
            with torch.cuda.amp.autocast(enabled=use_cuda):
                outputs = model(imgs)
            dice = compute_dice_per_class(outputs, labels, num_classes=3)
            dice_scores.append(dice)
    if not dice_scores:
        return 0.0
    dice_array = np.stack(dice_scores, axis=0)
    return float(np.mean(dice_array))


def main() -> None:
    args = parse_args()
    if torch is None:
        raise RuntimeError(
            "PyTorch must be installed to run training. Please install torch and its dependencies."
        )
    # Set random seed across numpy, random and torch
    set_seed(args.seed)
    # Prepare computation device
    if args.gpu is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    # Read training CSV
    df = pd.read_csv(args.train_csv)
    # Sanity checks
    if not 0.0 < args.test_split < 1.0:
        raise ValueError("--test_split must be a float between 0 and 1.")
    # Perform a group shuffle split to create a hold‑out back‑test set
    gss = GroupShuffleSplit(n_splits=1, test_size=args.test_split, random_state=args.seed)
    # Use subject IDs to avoid leakage
    groups = df["ID"].values
    train_val_idx, test_back_idx = next(gss.split(df, groups=groups))
    train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
    test_back_df = df.iloc[test_back_idx].reset_index(drop=True)
    # Save the back‑test split index list for reproducibility
    split_info = {
        "test_back_indices": test_back_idx.tolist(),
        "train_val_indices": train_val_idx.tolist(),
    }
    with open(os.path.join(args.output_dir, "split_indices.json"), "w") as f:
        json.dump(split_info, f, indent=2)
    print(f"Back‑test set size: {len(test_back_df)} subjects (\u2248{len(test_back_df)/len(df):.1%} of data)")
    # Create cross‑validation splits on the remaining data
    splits = create_group_splits(train_val_df, n_splits=args.folds, random_state=args.seed)
    # Note: All data preprocessing and augmentation are handled inside
    # ``get_dataset``.  The ``augment`` flag controls whether random
    # augmentations are applied to the training set.
    # Evaluate each fold; collect metrics
    cv_dice_scores: list[float] = []
    back_dice_scores: list[float] = []
    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f"Starting fold {fold + 1}/{len(splits)}...")
        sub_train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
        sub_val_df = train_val_df.iloc[val_idx].reset_index(drop=True)
        best_model_path, best_cv_dice = train_fold(
            fold=fold,
            train_df=sub_train_df,
            val_df=sub_val_df,
            args=args,
            device=device,
        )
        cv_dice_scores.append(best_cv_dice)
        # Evaluate on the held‑out back‑test set using the best model of this fold
        back_dice = evaluate_on_dataset(
            model_path=best_model_path,
            model_name=args.model,
            df=test_back_df,
            args=args,
            device=device,
        )
        back_dice_scores.append(back_dice)
        print(f"Fold {fold} complete.  Best CV mean Dice: {best_cv_dice:.4f} | Back‑test mean Dice: {back_dice:.4f}")
    # Summarise results across folds
    cv_mean = float(np.mean(cv_dice_scores)) if cv_dice_scores else 0.0
    back_mean = float(np.mean(back_dice_scores)) if back_dice_scores else 0.0
    summary = {
        "cv_mean_dice": cv_mean,
        "fold_cv_dice_scores": cv_dice_scores,
        "back_test_mean_dice": back_mean,
        "fold_back_dice_scores": back_dice_scores,
    }
    # Write summary to a JSON file
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\nTraining complete.")
    print(f"Average CV mean Dice across folds: {cv_mean:.4f}")
    print(f"Average Back‑test mean Dice across folds: {back_mean:.4f}")


if __name__ == "__main__":
    main()
