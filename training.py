# training.py - VERSION COMPLETA Y ROBUSTA CON TODAS LAS FEATURES
import os, time, gc
import wandb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold, train_test_split
import matplotlib.pyplot as plt
import tqdm
import imageio
from collections import defaultdict

from src.models import create_model
from src import metrics
from src import dataset
from src import losses
from src.cascade_utils import (
    get_roi_bbox_from_labels,
    get_roi_bbox_from_logits,
    crop_to_bbox,
    resize_volume,
)
from dotenv import load_dotenv

os.environ["WANDB_DIR"] = "/data/cristian/paper_2025/wandb_dir"
os.environ["WANDB_CACHE_DIR"] = "/data/cristian/paper_2025/wandb_dir"
os.environ["WANDB_ARTIFACT_DIR"] = "/data/cristian/paper_2025/wandb_dir"

SPACING = (1.0, 1.0, 1.0)

# ======================== VISUALIZACIONES ========================

def create_volume_animation(x, y, pred_base, pred_fine, epoch, exp_name, phase="val", fold=1):
    """Crea animaciones GIF para las 3 vistas anatómicas"""
    
    vis_dir = f"images/{exp_name}/fold_{fold}/{phase}/animations"
    os.makedirs(vis_dir, exist_ok=True)
    
    # Convertir a numpy
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()
    
    img_vol = x[0, 0] if x.ndim == 5 else x[0]
    gt_mask = y[0, 0] if y.ndim == 5 else y[0]
    pred_base_vol = pred_base[0] if pred_base is not None else np.zeros_like(gt_mask)
    pred_fine_vol = pred_fine[0] if pred_fine is not None else pred_base_vol
    
    # Calcular métricas
    dice_l_base = metrics.dice_score(pred_base_vol, gt_mask, 1)
    dice_r_base = metrics.dice_score(pred_base_vol, gt_mask, 2)
    dice_l_fine = metrics.dice_score(pred_fine_vol, gt_mask, 1)
    dice_r_fine = metrics.dice_score(pred_fine_vol, gt_mask, 2)
    
    # Animación combinada de las 3 vistas
    frames_combined = []
    num_steps = 20
    
    for step in range(num_steps):
        z_idx = int(step * img_vol.shape[0] / num_steps)
        x_idx = int(step * img_vol.shape[1] / num_steps)
        y_idx = int(step * img_vol.shape[2] / num_steps)
        
        fig, axes = plt.subplots(3, 6, figsize=(20, 12))
        
        # Plot para cada vista (axial, sagital, coronal)
        for view_idx, (slice_idx, axis_name) in enumerate([
            (z_idx, f'Axial z={z_idx}'),
            (x_idx, f'Sagital x={x_idx}'),
            (y_idx, f'Coronal y={y_idx}')
        ]):
            if view_idx == 0:  # Axial
                img_slice = img_vol[slice_idx]
                gt_slice = gt_mask[slice_idx]
                base_slice = pred_base_vol[slice_idx]
                fine_slice = pred_fine_vol[slice_idx]
            elif view_idx == 1:  # Sagital
                img_slice = img_vol[:, slice_idx, :]
                gt_slice = gt_mask[:, slice_idx, :]
                base_slice = pred_base_vol[:, slice_idx, :]
                fine_slice = pred_fine_vol[:, slice_idx, :]
            else:  # Coronal
                img_slice = img_vol[:, :, slice_idx]
                gt_slice = gt_mask[:, :, slice_idx]
                base_slice = pred_base_vol[:, :, slice_idx]
                fine_slice = pred_fine_vol[:, :, slice_idx]
            
            # Input
            axes[view_idx, 0].imshow(img_slice, cmap='gray')
            axes[view_idx, 0].set_title(axis_name)
            axes[view_idx, 0].axis('off')
            
            # GT
            axes[view_idx, 1].imshow(gt_slice, cmap='viridis', vmin=0, vmax=2)
            axes[view_idx, 1].set_title('GT')
            axes[view_idx, 1].axis('off')
            
            # Base
            axes[view_idx, 2].imshow(base_slice, cmap='viridis', vmin=0, vmax=2)
            axes[view_idx, 2].set_title('Base')
            axes[view_idx, 2].axis('off')
            
            # Fine
            axes[view_idx, 3].imshow(fine_slice, cmap='viridis', vmin=0, vmax=2)
            axes[view_idx, 3].set_title('Fine')
            axes[view_idx, 3].axis('off')
            
            # GT Overlay
            axes[view_idx, 4].imshow(img_slice, cmap='gray')
            masked_gt = np.ma.masked_where(gt_slice == 0, gt_slice)
            axes[view_idx, 4].imshow(masked_gt, cmap='jet', alpha=0.5, vmin=1, vmax=2)
            axes[view_idx, 4].set_title('GT Overlay')
            axes[view_idx, 4].axis('off')
            
            # Pred Overlay
            axes[view_idx, 5].imshow(img_slice, cmap='gray')
            masked_pred = np.ma.masked_where(fine_slice == 0, fine_slice)
            axes[view_idx, 5].imshow(masked_pred, cmap='jet', alpha=0.5, vmin=1, vmax=2)
            axes[view_idx, 5].set_title('Pred Overlay')
            axes[view_idx, 5].axis('off')
        
        title = f'{phase.upper()} - Epoch {epoch}\n'
        title += f'Base: L={dice_l_base:.3f} R={dice_r_base:.3f} | '
        title += f'Fine: L={dice_l_fine:.3f} R={dice_r_fine:.3f}'
        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames_combined.append(image)
        plt.close()
    
    gif_path = os.path.join(vis_dir, f"{phase}_epoch{epoch:03d}_combined.gif")
    imageio.mimsave(gif_path, frames_combined, duration=0.3)
    print(f"  Animation saved: {gif_path}")
    
    return gif_path

def save_training_visualization(x, y, pred_base, pred_fine, epoch, batch_idx, 
                               exp_name, phase="train", fold=1):
    """Visualización estática con GT overlay - versión mejorada"""
    
    vis_dir = f"images/{exp_name}/fold_{fold}/{phase}"
    os.makedirs(vis_dir, exist_ok=True)
    
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()
    
    def find_best_slice(mask, axis):
        sums = []
        for i in range(mask.shape[axis]):
            if axis == 0:
                slice_sum = np.sum(mask[i, :, :] > 0)
            elif axis == 1:
                slice_sum = np.sum(mask[:, i, :] > 0)
            else:
                slice_sum = np.sum(mask[:, :, i] > 0)
            sums.append(slice_sum)
        
        if max(sums) == 0:
            return mask.shape[axis] // 2
        return np.argmax(sums)
    
    gt_mask = y[0, 0] if y.ndim == 5 else y[0]
    
    best_axial = find_best_slice(gt_mask, 0)
    best_sagittal = find_best_slice(gt_mask, 1)
    best_coronal = find_best_slice(gt_mask, 2)
    
    fig, axes = plt.subplots(3, 6, figsize=(20, 12))
    
    img_vol = x[0, 0] if x.ndim == 5 else x[0]
    pred_base_vol = pred_base[0] if pred_base is not None else np.zeros_like(gt_mask)
    pred_fine_vol = pred_fine[0] if pred_fine is not None else pred_base_vol
    
    def plot_row(ax_row, img_slice, gt_slice, pred_base_slice, pred_fine_slice, view_name):
        # Input
        ax_row[0].imshow(img_slice, cmap='gray')
        ax_row[0].set_title(f'{view_name}')
        ax_row[0].axis('off')
        
        # GT
        im = ax_row[1].imshow(gt_slice, cmap='viridis', vmin=0, vmax=2)
        ax_row[1].set_title('GT')
        ax_row[1].axis('off')
        
        # Base
        ax_row[2].imshow(pred_base_slice, cmap='viridis', vmin=0, vmax=2)
        ax_row[2].set_title('Base')
        ax_row[2].axis('off')
        
        # Fine
        ax_row[3].imshow(pred_fine_slice, cmap='viridis', vmin=0, vmax=2)
        ax_row[3].set_title('Fine')
        ax_row[3].axis('off')
        
        # GT Overlay
        ax_row[4].imshow(img_slice, cmap='gray')
        masked_gt = np.ma.masked_where(gt_slice == 0, gt_slice)
        ax_row[4].imshow(masked_gt, cmap='jet', alpha=0.5, vmin=1, vmax=2)
        ax_row[4].set_title('GT Overlay')
        ax_row[4].axis('off')
        
        # Pred Overlay
        ax_row[5].imshow(img_slice, cmap='gray')
        pred_to_show = pred_fine_slice if pred_fine is not None else pred_base_slice
        masked_pred = np.ma.masked_where(pred_to_show == 0, pred_to_show)
        ax_row[5].imshow(masked_pred, cmap='jet', alpha=0.5, vmin=1, vmax=2)
        ax_row[5].set_title('Pred Overlay')
        ax_row[5].axis('off')
        
        return im
    
    im = plot_row(axes[0], img_vol[best_axial], gt_mask[best_axial],
                  pred_base_vol[best_axial], pred_fine_vol[best_axial], f'Axial z={best_axial}')
    
    plot_row(axes[1], img_vol[:, best_sagittal], gt_mask[:, best_sagittal],
             pred_base_vol[:, best_sagittal], pred_fine_vol[:, best_sagittal], f'Sagittal x={best_sagittal}')
    
    plot_row(axes[2], img_vol[:, :, best_coronal], gt_mask[:, :, best_coronal],
             pred_base_vol[:, :, best_coronal], pred_fine_vol[:, :, best_coronal], f'Coronal y={best_coronal}')
    
    # Calcular métricas
    dice_l_base = metrics.dice_score(pred_base_vol, gt_mask, 1) if pred_base is not None else 0
    dice_r_base = metrics.dice_score(pred_base_vol, gt_mask, 2) if pred_base is not None else 0
    dice_l_fine = metrics.dice_score(pred_fine_vol, gt_mask, 1) if pred_fine is not None else 0
    dice_r_fine = metrics.dice_score(pred_fine_vol, gt_mask, 2) if pred_fine is not None else 0
    
    title = f'{phase.upper()} - Epoch {epoch} - Batch {batch_idx}\n'
    title += f'Base: L={dice_l_base:.3f} R={dice_r_base:.3f} | '
    title += f'Fine: L={dice_l_fine:.3f} R={dice_r_fine:.3f}'
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    filename = f"{phase}_epoch{epoch:03d}_batch{batch_idx:03d}.png"
    save_path = os.path.join(vis_dir, filename)
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    return save_path

def plot_metrics_history(history, save_path):
    """Plot training history metrics"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Loss curves
    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Val')
    axes[0, 0].plot(history['test_loss'], label='Test')
    axes[0, 0].set_title('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Dice Average
    axes[0, 1].plot(history['val_dice_avg'], label='Val')
    axes[0, 1].plot(history['test_dice_avg'], label='Test')
    axes[0, 1].set_title('Dice Average')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Dice L
    axes[0, 2].plot(history['val_dice_L'], label='Val L')
    axes[0, 2].plot(history['test_dice_L'], label='Test L')
    axes[0, 2].set_title('Dice Left')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # Dice R
    axes[1, 0].plot(history['val_dice_R'], label='Val R')
    axes[1, 0].plot(history['test_dice_R'], label='Test R')
    axes[1, 0].set_title('Dice Right')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Learning Rate
    axes[1, 1].plot(history['lr'], label='LR')
    axes[1, 1].set_title('Learning Rate')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True)
    
    # Overfitting Gap
    if 'overfitting_gap' in history:
        axes[1, 2].plot(history['overfitting_gap'], label='Val-Train Loss')
        axes[1, 2].set_title('Overfitting Gap')
        axes[1, 2].axhline(y=0, color='r', linestyle='--')
        axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    return save_path

# ======================== MIXUP 3D ========================

def mixup_3d(images, labels, alpha=0.2):
    """MixUp augmentation for 3D volumes"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = images.size(0)
    index = torch.randperm(batch_size).to(images.device)
    
    mixed_images = lam * images + (1 - lam) * images[index]
    labels_a, labels_b = labels, labels[index]
    
    return mixed_images, labels_a, labels_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Loss for mixup samples"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ======================== WARMUP SCHEDULER ========================

class WarmupCosineScheduler:
    """Custom scheduler with warmup + cosine annealing"""
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, warmup_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.warmup_lr = warmup_lr
        self.current_epoch = 0
        
    def step(self):
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup
            lr = self.warmup_lr + (self.base_lr - self.warmup_lr) * (self.current_epoch / self.warmup_epochs)
        else:
            # Cosine annealing
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.base_lr * 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr

# ======================== DEEP SUPERVISION ========================

class DeepSupervisionLoss(nn.Module):
    """Deep supervision for intermediate layers"""
    def __init__(self, loss_fn, weights=[1.0, 0.5, 0.3]):
        super().__init__()
        self.loss_fn = loss_fn
        self.weights = weights
    
    def forward(self, outputs, targets):
        """outputs should be a list of predictions at different scales"""
        if not isinstance(outputs, (list, tuple)):
            return self.loss_fn(outputs, targets)
        
        total_loss = 0
        for i, (output, weight) in enumerate(zip(outputs, self.weights)):
            # Resize output to target size if needed
            if output.shape[2:] != targets.shape[2:]:
                output = torch.nn.functional.interpolate(
                    output, 
                    size=targets.shape[2:], 
                    mode='trilinear', 
                    align_corners=False
                )
            total_loss += weight * self.loss_fn(output, targets)
        
        return total_loss

# ======================== TRAINING FUNCTIONS ========================

def train_cascade_sequential(df, args):
    """Entrenamiento completo con TODAS las features"""
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    results = {"val_best": [], "test_best": []}
    
    # Split train/test con estratificación
    list_ids = df["ID"].unique().tolist()
    print("*"*30)
    print("LEN IDS : ",len(list_ids))
    list_ids_train, list_ids_backtest = train_test_split(
        list_ids, test_size=0.15, random_state=42
    )
    df_backtest = df[df["ID"].isin(list_ids_backtest)].reset_index(drop=True)
    df_train = df[~df["ID"].isin(list_ids_backtest)].reset_index(drop=True)
    
    print(f"Train: {len(df_train)} samples | Test: {len(df_backtest)} samples")
    print(f"Train IDs: {len(list_ids_train)} | Test IDs: {len(list_ids_backtest)}")
    
    # Test dataset
    test_ds = dataset.MRIDataset3D(
        df_backtest,
        transform=dataset.get_val_transforms(SPACING, (args.dim, args.dim, args.dim))
    )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4)
    
    # K-Fold Cross Validation
    gkf = GroupKFold(n_splits=args.num_folds)
    df_train["fold"] = -1
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(df_train, groups=df_train["ID"])):
        print(f"\n{'='*60}")
        print(f"FOLD {fold+1}/{args.num_folds}")
        print(f"{'='*60}")
        
        # Prepare fold data
        df_train.loc[val_idx, "fold"] = fold
        df_train_fold = df_train[df_train["fold"] != fold].reset_index(drop=True)
        df_val_fold = df_train[df_train["fold"] == fold].reset_index(drop=True)
        
        print(f"Fold {fold+1}: Train={len(df_train_fold)} Val={len(df_val_fold)}")
        
        # Datasets
        train_ds = dataset.MRIDataset3D(
            df_train_fold,
            transform=dataset.get_train_transforms_hippocampus(SPACING, (args.dim, args.dim, args.dim))
        )
        val_ds = dataset.MRIDataset3D(
            df_val_fold,
            transform=dataset.get_val_transforms(SPACING, (args.dim, args.dim, args.dim))
        )
        
        train_loader = DataLoader(
            train_ds, 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=8,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_ds, 
            batch_size=1, 
            shuffle=False, 
            num_workers=4,
            pin_memory=True
        )
        
        # Create directories
        fold_dir = os.path.join(args.root_dir, f"fold_{fold+1}")
        os.makedirs(fold_dir, exist_ok=True)
        os.makedirs(f"images/{args.experiment_name}/fold_{fold+1}", exist_ok=True)
        
        # ========== ETAPA 1: BASE MODEL ==========
        if os.path.exists(os.path.join(fold_dir, "best_model.pth")) and not args.force_retrain:
            print("✓ Base model exists, loading...")
            base_model = create_model(args.model_name, device).to(device)
            base_model.load_state_dict(torch.load(os.path.join(fold_dir, "best_model.pth")))
            best_dice_base = 0.5  # Placeholder
        else:
            print("\n>>> STAGE 1: Training BASE model with warmup")
            
            base_model = create_model(args.model_name, device).to(device)
            
            # Loss con deep supervision si está disponible
            if args.use_deep_supervision:
                loss_fn_base = DeepSupervisionLoss(
                    losses.create_loss_function(args.loss_function),
                    weights=[1.0, 0.5, 0.3]
                )
            else:
                loss_fn_base = losses.create_loss_function(args.loss_function)
            
            # Optimizer con gradient accumulation
            optimizer_base = torch.optim.AdamW(
                base_model.parameters(), 
                lr=args.lr,
                weight_decay=args.weight_decay,
                betas=(0.9, 0.999)
            )
            
            # Scheduler con warmup
            warmup_epochs = min(10, args.num_epochs // 10)
            scheduler_base = WarmupCosineScheduler(
                optimizer_base, 
                warmup_epochs=warmup_epochs,
                total_epochs=args.num_epochs,
                base_lr=args.lr,
                warmup_lr=args.lr * 0.01
            )
            
            scaler_base = GradScaler()
            
            # Training state
            best_dice_base = 0.0
            patience_counter = 0
            MIN_IMPROVEMENT = 0.001
            MAX_PATIENCE = 50
            GRADIENT_ACCUMULATION = args.gradient_accumulation if hasattr(args, 'gradient_accumulation') else 2
            
            # History tracking
            history = defaultdict(list)
            
            # Initialize wandb
            if args.use_wandb:
                run_base = wandb.init(
                    project=os.getenv('PROJECT_WANDB'),
                    entity=os.getenv('ENTITY'),
                    name=f"{args.experiment_name}_fold{fold}_base",
                    group=args.experiment_name,
                    config=vars(args),
                    reinit=True,
                )
            
            print(f"Training for {args.num_epochs} epochs with warmup={warmup_epochs}")
            
            for epoch in range(args.num_epochs):
                # ===== TRAINING =====
                base_model.train()
                epoch_loss = 0
                batch_count = 0
                
                progress_bar = tqdm.tqdm(train_loader, desc=f"Base Epoch {epoch+1}/{args.num_epochs}")
                
                for batch_idx, batch in enumerate(progress_bar):
                    x = batch["image"].to(device)
                    y = batch["label"].to(device)
                    
                    flag_mixup = args.use_mixup and np.random.random() > 0.5
                    # MixUp augmentation
                    if flag_mixup:
                        x, y_a, y_b, lam = mixup_3d(x, y, alpha=0.2)
                        
                        with autocast():
                            logits = base_model(x)
                            loss = mixup_criterion(loss_fn_base, logits, y_a, y_b, lam)
                    else:
                        with autocast():
                            logits = base_model(x)
                            loss = loss_fn_base(logits, y)
                    
                    # Gradient accumulation
                    loss = loss / GRADIENT_ACCUMULATION
                    scaler_base.scale(loss).backward()
                    
                    if (batch_idx + 1) % GRADIENT_ACCUMULATION == 0:
                        scaler_base.unscale_(optimizer_base)
                        torch.nn.utils.clip_grad_norm_(base_model.parameters(), 1.0)
                        scaler_base.step(optimizer_base)
                        scaler_base.update()
                        optimizer_base.zero_grad()
                    
                    epoch_loss += loss.item() * GRADIENT_ACCUMULATION
                    batch_count += 1
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        'loss': f'{epoch_loss/batch_count:.4f}',
                        'lr': f'{optimizer_base.param_groups[0]["lr"]:.2e}'
                    })
                    
                    # Periodic visualization
                    if batch_idx % 100 == 0 and epoch % 10 == 0:
                        with torch.no_grad():
                            if isinstance(logits, (list, tuple)):
                                pred = torch.argmax(torch.softmax(logits[0], dim=1), dim=1)
                            else:
                                pred = torch.argmax(torch.softmax(logits, dim=1), dim=1)
                            
                            img_path = save_training_visualization(
                                x, y if not flag_mixup else y_a, 
                                pred.cpu().numpy(), None,
                                epoch, batch_idx, args.experiment_name, 
                                phase="train_base", fold=fold+1
                            )
                            if args.use_wandb:
                                run_base.log({
                                    "train/visualization": wandb.Image(img_path),
                                    "epoch": epoch
                                })
                
                epoch_loss /= batch_count
                
                # Update learning rate
                current_lr = scheduler_base.step()
                
                # ===== VALIDATION =====
                val_metrics = evaluate_base_model(base_model, val_loader, device, loss_fn_base)
                test_metrics = evaluate_base_model(base_model, test_loader, device, loss_fn_base)
                
                val_dice = val_metrics["dice_avg"]
                val_loss = val_metrics["loss"]
                test_loss = test_metrics["loss"]
                overfitting_gap = val_loss - epoch_loss
                
                # Save history
                history['train_loss'].append(epoch_loss)
                history['val_loss'].append(val_loss)
                history['test_loss'].append(test_loss)
                history['val_dice_avg'].append(val_dice)
                history['val_dice_L'].append(val_metrics["dice_L"])
                history['val_dice_R'].append(val_metrics["dice_R"])
                history['test_dice_avg'].append(test_metrics["dice_avg"])
                history['test_dice_L'].append(test_metrics["dice_L"])
                history['test_dice_R'].append(test_metrics["dice_R"])
                history['lr'].append(current_lr)
                history['overfitting_gap'].append(overfitting_gap)
                
                print(f"\nBase Epoch {epoch+1}/{args.num_epochs}:")
                print(f"  Loss: Train={epoch_loss:.4f}, Val={val_loss:.4f}, Test={test_loss:.4f}")
                print(f"  Dice Val: Avg={val_dice:.4f} (L={val_metrics['dice_L']:.3f}, R={val_metrics['dice_R']:.3f})")
                print(f"  Dice Test: Avg={test_metrics['dice_avg']:.4f} (L={test_metrics['dice_L']:.3f}, R={test_metrics['dice_R']:.3f})")
                print(f"  LR: {current_lr:.2e}, Overfitting Gap: {overfitting_gap:.4f}")
                
                if args.use_wandb:
                    run_base.log({
                        "base/train_loss": epoch_loss,
                        "base/val_loss": val_loss,
                        "base/test_loss": test_loss,
                        "base/overfitting_gap": overfitting_gap,
                        "base/val_dice_avg": val_dice,
                        "base/val_dice_L": val_metrics["dice_L"],
                        "base/val_dice_R": val_metrics["dice_R"],
                        "base/test_dice_avg": test_metrics["dice_avg"],
                        "base/test_dice_L": test_metrics["dice_L"],
                        "base/test_dice_R": test_metrics["dice_R"],
                        "base/lr": current_lr,
                        "epoch": epoch
                    })
                
                # Visualization and animations every N epochs
                if epoch % 10 == 0 or epoch == args.num_epochs - 1:
                    with torch.no_grad():
                        # Validation visualization
                        for i, val_batch in enumerate(val_loader):
                            if i == 0:
                                x_val = val_batch["image"].to(device)
                                y_val = val_batch["label"].to(device)
                                logits_val = base_model(x_val)
                                if isinstance(logits_val, (list, tuple)):
                                    logits_val = logits_val[0]
                                pred_val = torch.argmax(torch.softmax(logits_val, dim=1), dim=1)
                                
                                # Static visualization
                                img_path = save_training_visualization(
                                    x_val, y_val, pred_val.cpu().numpy(), None,
                                    epoch, 0, args.experiment_name,
                                    phase="val_base", fold=fold+1
                                )
                                
                                # Animation
                                gif_path = create_volume_animation(
                                    x_val, y_val, pred_val.cpu().numpy(), None,
                                    epoch, args.experiment_name, phase="val_base", fold=fold+1
                                )
                                
                                if args.use_wandb:
                                    run_base.log({
                                        "val/visualization": wandb.Image(img_path),
                                        "val/animation": wandb.Video(gif_path),
                                        "epoch": epoch
                                    })
                                break
                    
                    # Plot metrics history
                    plot_path = os.path.join(fold_dir, f"metrics_base_epoch{epoch}.png")
                    plot_metrics_history(history, plot_path)
                    if args.use_wandb:
                        run_base.log({"metrics_plot": wandb.Image(plot_path), "epoch": epoch})
                
                # Early stopping
                if val_dice > best_dice_base + MIN_IMPROVEMENT:
                    best_dice_base = val_dice
                    patience_counter = 0
                    torch.save(base_model.state_dict(), os.path.join(fold_dir, "best_model.pth"))
                    # Save full checkpoint
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': base_model.state_dict(),
                        'optimizer_state_dict': optimizer_base.state_dict(),
                        'scheduler_state': scheduler_base.current_epoch,
                        'best_dice': best_dice_base,
                        'history': dict(history)
                    }, os.path.join(fold_dir, "checkpoint_base.pth"))
                    print(f"  ✓ Saved best base model: Dice={best_dice_base:.4f}")
                else:
                    patience_counter += 1
                    if patience_counter > MAX_PATIENCE:
                        print(f"Early stopping at epoch {epoch}")
                        break
                
                # Memory cleanup
                if epoch % 10 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
            
            if args.use_wandb:
                run_base.finish()
        
        # ========== ETAPA 2: FINE MODEL ==========
        print(f"\n>>> STAGE 2: Fine-tuning (Base Dice: {best_dice_base:.3f})")
        
        # Freeze base model
        base_model.eval()
        for param in base_model.parameters():
            param.requires_grad = False
        
        fine_model = create_model(args.model_name, device).to(device)
        
        # Copy weights from base model as initialization (optional)
        if args.init_fine_from_base:
            fine_model.load_state_dict(base_model.state_dict())
            print("  Initialized fine model with base model weights")
        
        # Simple loss for fine-tuning
        loss_fn_fine = losses.create_loss_function("dice_ce_spatial_balanced")
        
        optimizer_fine = torch.optim.AdamW(
            fine_model.parameters(),
            lr=args.lr * 0.5,  # Lower LR for fine-tuning
            weight_decay=args.weight_decay
        )
        
        scheduler_fine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer_fine,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )
        
        scaler_fine = GradScaler()
        
        # Fine-tuning parameters
        best_dice_fine = 0.0
        roi_margin = 25
        roi_size_fine = (args.dim//2, args.dim//2, args.dim//2)  # Same size, no reduction
        MAX_ROIS_PER_BATCH = 4
        MIN_VOXELS_IN_ROI = 100
        
        # History for fine-tuning
        history_fine = defaultdict(list)
        
        if args.use_wandb:
            run_fine = wandb.init(
                project=os.getenv('PROJECT_WANDB'),
                entity=os.getenv('ENTITY'),
                name=f"{args.experiment_name}_fold{fold}_fine",
                group=args.experiment_name,
                config=vars(args),
                reinit=True
            )
        
        fine_epochs = min(100, args.num_epochs // 2)
        print(f"Fine-tuning for {fine_epochs} epochs")
        
        for epoch in range(fine_epochs):
            fine_model.train()
            epoch_loss = 0
            n_batches = 0
            total_rois_processed = 0
            total_rois_from_gt = 0
            total_rois_from_pred = 0
            
            # Scheduled sampling strategy
            if epoch < 20:
                use_gt_prob = 1.0  # 100% GT boxes first 20 epochs
            elif epoch < 40:
                use_gt_prob = 0.7 - (epoch - 20) * 0.02  # Gradual decrease
            else:
                use_gt_prob = 0.3  # Minimum 30% GT
            
            progress_bar = tqdm.tqdm(train_loader, desc=f"Fine Epoch {epoch+1}/{fine_epochs}")
            
            for batch_idx, batch in enumerate(progress_bar):
                x = batch["image"].to(device)
                y = batch["label"].to(device)
                
                # Get base model predictions
                with torch.no_grad():
                    base_logits = base_model(x)
                    if isinstance(base_logits, (list, tuple)):
                        base_logits = base_logits[0]
                    base_pred = torch.argmax(torch.softmax(base_logits, dim=1), dim=1)
                
                # Collect ROIs with improved strategy
                all_img_rois = []
                all_lbl_rois = []
                roi_info = []
                
                for i in range(x.shape[0]):
                    # Decide whether to use GT or predicted boxes
                    use_gt = np.random.random() < use_gt_prob
                    
                    if use_gt:
                        bboxes_i = get_roi_bbox_from_labels(y[i:i+1], margin=roi_margin)
                        total_rois_from_gt += len(bboxes_i)
                    else:
                        # Try multiple thresholds
                        bboxes_i = []
                        for thr in [0.15, 0.1, 0.05, 0.02]:
                            bboxes_temp = get_roi_bbox_from_logits(
                                base_logits[i:i+1], 
                                thr=thr, 
                                margin=roi_margin
                            )
                            if bboxes_temp and bboxes_temp[0] is not None:
                                # Verify ROI contains labels
                                bb_test = bboxes_temp[0]
                                lbl_test = crop_to_bbox(y[i:i+1], bb_test)
                                if lbl_test.sum() > MIN_VOXELS_IN_ROI:
                                    bboxes_i = bboxes_temp
                                    break
                        
                        if not bboxes_i or bboxes_i[0] is None:
                            # Fallback to GT
                            bboxes_i = get_roi_bbox_from_labels(y[i:i+1], margin=roi_margin)
                            total_rois_from_gt += 1
                        else:
                            total_rois_from_pred += len(bboxes_i)
                    
                    # Process each bounding box
                    for bb in bboxes_i:
                        if bb is None:
                            continue
                        
                        img_roi = crop_to_bbox(x[i:i+1], bb)
                        lbl_roi = crop_to_bbox(y[i:i+1], bb)
                        
                        # Verify minimum voxels
                        n_voxels = lbl_roi.sum().item()
                        if n_voxels < MIN_VOXELS_IN_ROI:
                            continue
                        
                        # Resize if needed
                        if img_roi.shape[2:] != roi_size_fine:
                            img_roi = resize_volume(img_roi, roi_size_fine, mode="trilinear")
                            lbl_roi = resize_volume(lbl_roi.float(), roi_size_fine, mode="nearest").long()
                        
                        all_img_rois.append(img_roi)
                        all_lbl_rois.append(lbl_roi)
                        roi_info.append((i, bb))
                        total_rois_processed += 1
                
                # Debug info
                if epoch == 0 and batch_idx == 0:
                    print(f"\n  === DEBUG Fine Training ===")
                    print(f"  Batch size: {x.shape[0]}")
                    print(f"  ROIs found: {len(all_img_rois)}")
                    print(f"  Use GT prob: {use_gt_prob:.1%}")
                    if len(all_img_rois) > 0:
                        print(f"  ROI shapes: {[roi.shape for roi in all_img_rois[:2]]}")
                        print(f"  Voxels in labels: {[roi.sum().item() for roi in all_lbl_rois[:3]]}")
                
                if len(all_img_rois) == 0:
                    continue
                
                # Process in mini-batches with gradient accumulation
                optimizer_fine.zero_grad()
                batch_loss = 0
                n_mini = 0
                
                for start_idx in range(0, len(all_img_rois), MAX_ROIS_PER_BATCH):
                    end_idx = min(start_idx + MAX_ROIS_PER_BATCH, len(all_img_rois))
                    
                    img_roi_batch = torch.cat(all_img_rois[start_idx:end_idx], dim=0)
                    lbl_roi_batch = torch.cat(all_lbl_rois[start_idx:end_idx], dim=0)
                    
                    # MixUp for ROIs
                    if args.use_mixup and np.random.random() > 0.7:
                        img_roi_batch, lbl_a, lbl_b, lam = mixup_3d(img_roi_batch, lbl_roi_batch, alpha=0.1)
                        
                        with autocast():
                            logits_fine_batch = fine_model(img_roi_batch)
                            loss = mixup_criterion(loss_fn_fine, logits_fine_batch, lbl_a, lbl_b, lam)
                    else:
                        with autocast():
                            logits_fine_batch = fine_model(img_roi_batch)
                            loss = loss_fn_fine(logits_fine_batch, lbl_roi_batch)
                    
                    scaler_fine.scale(loss).backward()
                    batch_loss += loss.item()
                    n_mini += 1
                
                if n_mini > 0:
                    scaler_fine.unscale_(optimizer_fine)
                    torch.nn.utils.clip_grad_norm_(fine_model.parameters(), 1.0)
                    scaler_fine.step(optimizer_fine)
                    scaler_fine.update()
                    
                    epoch_loss += batch_loss / n_mini
                    n_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{epoch_loss/max(n_batches, 1):.4f}',
                    'rois': total_rois_processed,
                    'lr': f'{optimizer_fine.param_groups[0]["lr"]:.2e}'
                })
                
                # Periodic visualization
                if batch_idx % 100 == 0 and epoch % 10 == 0:
                    with torch.no_grad():
                        # Reconstruct full prediction for visualization
                        fine_preds = torch.zeros_like(base_pred)
                        for (batch_i, bb), roi_idx in zip(roi_info[:MAX_ROIS_PER_BATCH], range(len(roi_info[:MAX_ROIS_PER_BATCH]))):
                            if roi_idx < len(all_img_rois):
                                logits_roi = fine_model(all_img_rois[roi_idx])
                                pred_roi = torch.argmax(torch.softmax(logits_roi, dim=1), dim=1)
                                
                                z0, y0, x0, z1, y1, x1 = bb
                                if pred_roi.shape[2:] != (z1-z0, y1-y0, x1-x0):
                                    pred_roi_up = resize_volume(
                                        pred_roi.unsqueeze(0).float(),
                                        (z1-z0, y1-y0, x1-x0),
                                        mode="nearest"
                                    )[0].long()
                                else:
                                    pred_roi_up = pred_roi[0]
                                fine_preds[batch_i:batch_i+1, z0:z1, y0:y1, x0:x1] = pred_roi_up
                        
                        img_path = save_training_visualization(
                            x, y, base_pred.cpu().numpy(), fine_preds.cpu().numpy(),
                            epoch, batch_idx, args.experiment_name,
                            phase="train_fine", fold=fold+1
                        )
                        if args.use_wandb:
                            run_fine.log({
                                "train/visualization": wandb.Image(img_path),
                                "epoch": epoch
                            })
            
            epoch_loss /= max(n_batches, 1)
            scheduler_fine.step()
            current_lr = optimizer_fine.param_groups[0]['lr']
            
            print(f"\nFine Epoch {epoch+1}/{fine_epochs}:")
            print(f"  ROIs: Total={total_rois_processed}, GT={total_rois_from_gt}, Pred={total_rois_from_pred}")
            print(f"  Loss: {epoch_loss:.4f}, LR: {current_lr:.2e}")
            
            # ===== VALIDATION CASCADE =====
            val_metrics = evaluate_cascade(
                base_model, fine_model, val_loader, device,
                loss_fn_fine, roi_margin, roi_size_fine, args.dim
            )
            test_metrics = evaluate_cascade(
                base_model, fine_model, test_loader, device,
                loss_fn_fine, roi_margin, roi_size_fine, args.dim
            )
            
            val_dice_fine = val_metrics["dice_fine_avg"]
            
            # Save history
            history_fine['train_loss'].append(epoch_loss)
            history_fine['val_loss'].append(val_metrics["loss_fine"])
            history_fine['test_loss'].append(test_metrics["loss_fine"])
            history_fine['val_dice_base'].append(val_metrics["dice_avg"])
            history_fine['val_dice_fine'].append(val_dice_fine)
            history_fine['val_dice_fine_L'].append(val_metrics["dice_fine_L"])
            history_fine['val_dice_fine_R'].append(val_metrics["dice_fine_R"])
            history_fine['test_dice_fine'].append(test_metrics["dice_fine_avg"])
            history_fine['lr'].append(current_lr)
            
            print(f"  Val Dice: Base={val_metrics['dice_avg']:.4f}, Fine={val_dice_fine:.4f}")
            print(f"  Val Fine: L={val_metrics['dice_fine_L']:.3f}, R={val_metrics['dice_fine_R']:.3f}")
            print(f"  Test Dice: Base={test_metrics['dice_avg']:.4f}, Fine={test_metrics['dice_fine_avg']:.4f}")
            print(f"  Test Fine: L={test_metrics['dice_fine_L']:.3f}, R={test_metrics['dice_fine_R']:.3f}")
            
            if args.use_wandb:
                run_fine.log({
                    "fine/train_loss": epoch_loss,
                    "fine/val_loss": val_metrics["loss_fine"],
                    "fine/test_loss": test_metrics["loss_fine"],
                    "fine/val_dice_base": val_metrics["dice_avg"],
                    "fine/val_dice_fine": val_dice_fine,
                    "fine/val_dice_fine_L": val_metrics["dice_fine_L"],
                    "fine/val_dice_fine_R": val_metrics["dice_fine_R"],
                    "fine/test_dice_base": test_metrics["dice_avg"],
                    "fine/test_dice_fine": test_metrics["dice_fine_avg"],
                    "fine/test_dice_fine_L": test_metrics["dice_fine_L"],
                    "fine/test_dice_fine_R": test_metrics["dice_fine_R"],
                    "fine/rois_processed": total_rois_processed,
                    "fine/lr": current_lr,
                    "epoch": epoch
                })
            
            # Visualization and animations
            if epoch % 10 == 0 or epoch == fine_epochs - 1:
                with torch.no_grad():
                    for i, val_batch in enumerate(val_loader):
                        if i == 0:
                            x_val = val_batch["image"].to(device)
                            y_val = val_batch["label"].to(device)
                            
                            # Full cascade inference
                            logits_base = base_model(x_val)
                            if isinstance(logits_base, (list, tuple)):
                                logits_base = logits_base[0]
                            pred_base_val = torch.argmax(torch.softmax(logits_base, dim=1), dim=1)
                            
                            # Get ROIs and apply fine model
                            bboxes = get_roi_bbox_from_logits(logits_base, thr=0.1, margin=roi_margin)
                            logits_fine = torch.zeros_like(logits_base)
                            
                            for j, bb in enumerate(bboxes):
                                if bb is None:
                                    bb_gt = get_roi_bbox_from_labels(y_val[j:j+1], margin=roi_margin)
                                    if bb_gt and bb_gt[0] is not None:
                                        bb = bb_gt[0]
                                    else:
                                        continue
                                
                                img_roi = crop_to_bbox(x_val[j:j+1], bb)
                                if img_roi.shape[2:] != roi_size_fine:
                                    img_roi = resize_volume(img_roi, roi_size_fine, mode="trilinear")
                                
                                logits2 = fine_model(img_roi)
                                
                                z0, y0, x0, z1, y1, x1 = bb
                                if logits2.shape[2:] != (z1-z0, y1-y0, x1-x0):
                                    logits2_up = resize_volume(logits2, (z1-z0, y1-y0, x1-x0), mode="trilinear")[0]
                                else:
                                    logits2_up = logits2[0]
                                logits_fine[j:j+1, :, z0:z1, y0:y1, x0:x1] = logits2_up.cpu()
                            
                            pred_fine_val = torch.argmax(torch.softmax(logits_fine, dim=1), dim=1)
                            
                            # Static visualization
                            img_path = save_training_visualization(
                                x_val, y_val, pred_base_val.cpu().numpy(), pred_fine_val.cpu().numpy(),
                                epoch, 0, args.experiment_name,
                                phase="val_fine", fold=fold+1
                            )
                            
                            # Animation
                            gif_path = create_volume_animation(
                                x_val, y_val, pred_base_val.cpu().numpy(), pred_fine_val.cpu().numpy(),
                                epoch, args.experiment_name, phase="val_fine", fold=fold+1
                            )
                            
                            if args.use_wandb:
                                run_fine.log({
                                    "val/visualization": wandb.Image(img_path),
                                    "val/animation": wandb.Video(gif_path),
                                    "epoch": epoch
                                })
                            break
                
                # Plot metrics
                plot_path = os.path.join(fold_dir, f"metrics_fine_epoch{epoch}.png")
                
                # Combine base and fine histories for plotting
                combined_history = {
                    'train_loss': history_fine['train_loss'],
                    'val_loss': history_fine['val_loss'],
                    'test_loss': history_fine['test_loss'],
                    'val_dice_avg': history_fine['val_dice_fine'],
                    'val_dice_L': history_fine['val_dice_fine_L'],
                    'val_dice_R': history_fine['val_dice_fine_R'],
                    'test_dice_avg': history_fine['test_dice_fine'],
                    'test_dice_L': history_fine.get('test_dice_fine_L', []),
                    'test_dice_R': history_fine.get('test_dice_fine_R', []),
                    'lr': history_fine['lr']
                }
                plot_metrics_history(combined_history, plot_path)
                
                if args.use_wandb:
                    run_fine.log({"metrics_plot": wandb.Image(plot_path), "epoch": epoch})
            
            # Save best model
            if val_dice_fine > best_dice_fine:
                best_dice_fine = val_dice_fine
                torch.save(fine_model.state_dict(), os.path.join(fold_dir, "best_fine_model.pth"))
                # Save full checkpoint
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': fine_model.state_dict(),
                    'optimizer_state_dict': optimizer_fine.state_dict(),
                    'best_dice': best_dice_fine,
                    'history': dict(history_fine)
                }, os.path.join(fold_dir, "checkpoint_fine.pth"))
                print(f"  ✓ Saved best fine model: Dice={best_dice_fine:.4f}")
            
            # Memory cleanup
            if epoch % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache()
        
        if args.use_wandb:
            run_fine.finish()
        
        # ===== FINAL TEST EVALUATION =====
        print(f"\n>>> Final Test Results for Fold {fold+1}")
        print(f"  Base Dice: {test_metrics['dice_avg']:.4f} (L={test_metrics['dice_L']:.3f}, R={test_metrics['dice_R']:.3f})")
        print(f"  Fine Dice: {test_metrics['dice_fine_avg']:.4f} (L={test_metrics['dice_fine_L']:.3f}, R={test_metrics['dice_fine_R']:.3f})")
        print(f"  Improvement: +{test_metrics['dice_fine_avg'] - test_metrics['dice_avg']:.4f}")
        
        results["test_best"].append(test_metrics)
        
        # Save final results
        results_df = pd.DataFrame(results["test_best"])
        results_df.to_csv(os.path.join(args.root_dir, f"fold_{fold+1}_results.csv"), index=False)
    
    # ===== AGGREGATE RESULTS =====
    print(f"\n{'='*60}")
    print("FINAL RESULTS ACROSS ALL FOLDS")
    print(f"{'='*60}")
    
    test_dices_base = [r['dice_avg'] for r in results["test_best"]]
    test_dices_fine = [r['dice_fine_avg'] for r in results["test_best"]]
    
    print(f"Base Model:")
    print(f"  Mean Dice: {np.mean(test_dices_base):.4f} ± {np.std(test_dices_base):.4f}")
    print(f"Fine Model:")
    print(f"  Mean Dice: {np.mean(test_dices_fine):.4f} ± {np.std(test_dices_fine):.4f}")
    
    return results

# ======================== EVALUATION FUNCTIONS ========================

def evaluate_base_model(model, loader, device, loss_fn):
    """Evaluation with comprehensive metrics"""
    model.eval()
    all_dice = {"L": [], "R": []}
    epoch_loss = 0.0
    
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            
            with autocast():
                logits = model(x)
                if isinstance(logits, (list, tuple)):
                    # Deep supervision - use only main output for evaluation
                    loss = loss_fn(logits, y) if hasattr(loss_fn, 'weights') else loss_fn(logits[0], y)
                    logits = logits[0]
                else:
                    loss = loss_fn(logits, y)
            
            epoch_loss += loss.item()
            
            pred = torch.argmax(torch.softmax(logits, dim=1), dim=1)
            pred_np = pred.cpu().numpy()
            y_np = y.squeeze(1).cpu().numpy()
            
            for p, g in zip(pred_np, y_np):
                for cls, key in [(1, "L"), (2, "R")]:
                    dice = metrics.dice_score(p, g, cls)
                    if not np.isnan(dice):
                        all_dice[key].append(dice)
    
    epoch_loss /= len(loader)
    
    return {
        "loss": epoch_loss,
        "dice_L": np.mean(all_dice["L"]) if all_dice["L"] else 0,
        "dice_R": np.mean(all_dice["R"]) if all_dice["R"] else 0,
        "dice_avg": np.mean(all_dice["L"] + all_dice["R"]) if (all_dice["L"] or all_dice["R"]) else 0
    }

def evaluate_cascade(base_model, fine_model, loader, device, loss_fn, 
                     roi_margin, roi_size_fine, dim):
    """Cascade evaluation with robust ROI detection"""
    base_model.eval()
    fine_model.eval()
    
    metrics_base = {"L": [], "R": []}
    metrics_fine = {"L": [], "R": []}
    epoch_loss_fine = 0.0
    n_rois = 0
    
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            
            # Base model prediction
            with autocast():
                logits_base = base_model(x)
                if isinstance(logits_base, (list, tuple)):
                    logits_base = logits_base[0]
            
            # Try multiple thresholds for ROI detection
            bboxes = None
            for thr in [0.1, 0.05, 0.02]:
                bboxes_temp = get_roi_bbox_from_logits(logits_base, thr=thr, margin=roi_margin)
                if bboxes_temp and bboxes_temp[0] is not None:
                    bboxes = bboxes_temp
                    break
            
            # Fallback to GT if no ROI found
            if bboxes is None or bboxes[0] is None:
                bboxes = get_roi_bbox_from_labels(y, margin=roi_margin)
            
            # Apply fine model to ROIs
            logits_fine = torch.zeros_like(logits_base)
            
            for i, bb in enumerate(bboxes):
                if bb is None:
                    continue
                
                img_roi = crop_to_bbox(x[i:i+1], bb)
                lbl_roi = crop_to_bbox(y[i:i+1], bb)
                
                # Resize if needed
                if img_roi.shape[2:] != roi_size_fine:
                    img_roi = resize_volume(img_roi, roi_size_fine, mode="trilinear")
                    lbl_roi = resize_volume(lbl_roi.float(), roi_size_fine, mode="nearest").long()
                
                with autocast():
                    logits2 = fine_model(img_roi)
                    loss_roi = loss_fn(logits2, lbl_roi)
                
                epoch_loss_fine += loss_roi.item()
                n_rois += 1
                
                # Resize back to original size
                z0, y0, x0, z1, y1, x1 = bb
                if logits2.shape[2:] != (z1-z0, y1-y0, x1-x0):
                    logits2_up = resize_volume(
                        logits2, 
                        (z1-z0, y1-y0, x1-x0), 
                        mode="trilinear"
                    )[0]
                else:
                    logits2_up = logits2[0]
                
                logits_fine[i:i+1, :, z0:z1, y0:y1, x0:x1] = logits2_up.cpu()
            
            # Compute metrics
            pred_base = torch.argmax(torch.softmax(logits_base, dim=1), dim=1).cpu().numpy()
            pred_fine = torch.argmax(torch.softmax(logits_fine, dim=1), dim=1).cpu().numpy()
            y_np = y.squeeze(1).cpu().numpy()
            
            for p_base, p_fine, g in zip(pred_base, pred_fine, y_np):
                for cls, key in [(1, "L"), (2, "R")]:
                    dice_base = metrics.dice_score(p_base, g, cls)
                    dice_fine = metrics.dice_score(p_fine, g, cls)
                    
                    if not np.isnan(dice_base):
                        metrics_base[key].append(dice_base)
                    if not np.isnan(dice_fine):
                        metrics_fine[key].append(dice_fine)
    
    epoch_loss_fine /= max(n_rois, 1)
    
    return {
        "loss_fine": epoch_loss_fine,
        "dice_L": np.mean(metrics_base["L"]) if metrics_base["L"] else 0,
        "dice_R": np.mean(metrics_base["R"]) if metrics_base["R"] else 0,
        "dice_avg": np.mean(metrics_base["L"] + metrics_base["R"]) if metrics_base["L"] else 0,
        "dice_fine_L": np.mean(metrics_fine["L"]) if metrics_fine["L"] else 0,
        "dice_fine_R": np.mean(metrics_fine["R"]) if metrics_fine["R"] else 0,
        "dice_fine_avg": np.mean(metrics_fine["L"] + metrics_fine["R"]) if metrics_fine["L"] else 0,
    }

# ======================== MAIN ========================
"""

python training.py \
  --model_name=eff-b2 \
  --device=cuda:0 \
  --root_dir=/data/cristian/projects/med_data/rise-miccai/task-2/3d_models/predictions/eff-b2-dice_ce_balanced-v3-pruebis/fold_models \
  --num_epochs=100 \
  --num_folds=3 \
  --use_mixup=1 \
  --experiment_name=eff-b2-dice_ce_balanced \
  --lr=5e-4 \
  --weight_decay=1e-4 \
  --loss_function=dice_ce_balanced \
  --dim=192 \
  --batch_size=2 \
  --use_wandb=1 \
  --use_deep_supervision=0 \
  --init_fine_from_base=1 \
  --gradient_accumulation=2 \
  --force_retrain=0
  
"""
if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="eff-b2")
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_epochs', type=int, default=150)
    parser.add_argument('--num_folds', type=int, default=5)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--root_dir', type=str, required=True)
    parser.add_argument('--dim', type=int, default=192)
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--loss_function', type=str, default="dice_ce_spatial_balanced")
    parser.add_argument('--use_mixup', type=int, default=1)
    parser.add_argument('--use_wandb', type=int, default=1)
    parser.add_argument('--use_deep_supervision', type=int, default=0)
    parser.add_argument('--init_fine_from_base', type=int, default=1)
    parser.add_argument('--gradient_accumulation', type=int, default=2)
    parser.add_argument('--force_retrain', type=int, default=0)
    
    args = parser.parse_args()
    load_dotenv()
    
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    df = pd.read_csv("results/preprocessed_data/task2/df_train_hipp.csv")
    
    print(f"\n{'='*60}")
    print(f"Starting training with configuration:")
    print(f"  Model: {args.model_name}")
    print(f"  Dims: {args.dim}x{args.dim}x{args.dim}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Loss: {args.loss_function}")
    print(f"  MixUp: {'Yes' if args.use_mixup else 'No'}")
    print(f"  Deep Supervision: {'Yes' if args.use_deep_supervision else 'No'}")
    print(f"  Device: {args.device}")
    print(f"{'='*60}\n")
    
    results = train_cascade_sequential(df, args)
    print("\nTraining completed successfully!")