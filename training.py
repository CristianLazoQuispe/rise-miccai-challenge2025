# training.py - VERSION COMPLETA CON ANIMACIONES Y MÉTRICAS
import os, time
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
        
        # Axial
        axes[0,0].imshow(img_vol[z_idx], cmap='gray')
        axes[0,0].set_title(f'Axial z={z_idx}')
        axes[0,0].axis('off')
        
        axes[0,1].imshow(gt_mask[z_idx], cmap='viridis', vmin=0, vmax=2)
        axes[0,1].set_title('GT')
        axes[0,1].axis('off')
        
        axes[0,2].imshow(pred_base_vol[z_idx], cmap='viridis', vmin=0, vmax=2)
        axes[0,2].set_title('Base')
        axes[0,2].axis('off')
        
        axes[0,3].imshow(pred_fine_vol[z_idx], cmap='viridis', vmin=0, vmax=2)
        axes[0,3].set_title('Fine')
        axes[0,3].axis('off')
        
        # GT Overlay
        axes[0,4].imshow(img_vol[z_idx], cmap='gray')
        masked_gt = np.ma.masked_where(gt_mask[z_idx] == 0, gt_mask[z_idx])
        axes[0,4].imshow(masked_gt, cmap='jet', alpha=0.5, vmin=1, vmax=2)
        axes[0,4].set_title('GT Overlay')
        axes[0,4].axis('off')
        
        # Pred Overlay
        axes[0,5].imshow(img_vol[z_idx], cmap='gray')
        masked_pred = np.ma.masked_where(pred_fine_vol[z_idx] == 0, pred_fine_vol[z_idx])
        axes[0,5].imshow(masked_pred, cmap='jet', alpha=0.5, vmin=1, vmax=2)
        axes[0,5].set_title('Pred Overlay')
        axes[0,5].axis('off')
        
        # Sagital
        axes[1,0].imshow(img_vol[:, x_idx, :], cmap='gray')
        axes[1,0].set_title(f'Sagital x={x_idx}')
        axes[1,0].axis('off')
        
        axes[1,1].imshow(gt_mask[:, x_idx, :], cmap='viridis', vmin=0, vmax=2)
        axes[1,1].axis('off')
        
        axes[1,2].imshow(pred_base_vol[:, x_idx, :], cmap='viridis', vmin=0, vmax=2)
        axes[1,2].axis('off')
        
        axes[1,3].imshow(pred_fine_vol[:, x_idx, :], cmap='viridis', vmin=0, vmax=2)
        axes[1,3].axis('off')
        
        axes[1,4].imshow(img_vol[:, x_idx, :], cmap='gray')
        masked_gt = np.ma.masked_where(gt_mask[:, x_idx, :] == 0, gt_mask[:, x_idx, :])
        axes[1,4].imshow(masked_gt, cmap='jet', alpha=0.5, vmin=1, vmax=2)
        axes[1,4].axis('off')
        
        axes[1,5].imshow(img_vol[:, x_idx, :], cmap='gray')
        masked_pred = np.ma.masked_where(pred_fine_vol[:, x_idx, :] == 0, pred_fine_vol[:, x_idx, :])
        axes[1,5].imshow(masked_pred, cmap='jet', alpha=0.5, vmin=1, vmax=2)
        axes[1,5].axis('off')
        
        # Coronal
        axes[2,0].imshow(img_vol[:, :, y_idx], cmap='gray')
        axes[2,0].set_title(f'Coronal y={y_idx}')
        axes[2,0].axis('off')
        
        axes[2,1].imshow(gt_mask[:, :, y_idx], cmap='viridis', vmin=0, vmax=2)
        axes[2,1].axis('off')
        
        axes[2,2].imshow(pred_base_vol[:, :, y_idx], cmap='viridis', vmin=0, vmax=2)
        axes[2,2].axis('off')
        
        axes[2,3].imshow(pred_fine_vol[:, :, y_idx], cmap='viridis', vmin=0, vmax=2)
        axes[2,3].axis('off')
        
        axes[2,4].imshow(img_vol[:, :, y_idx], cmap='gray')
        masked_gt = np.ma.masked_where(gt_mask[:, :, y_idx] == 0, gt_mask[:, :, y_idx])
        axes[2,4].imshow(masked_gt, cmap='jet', alpha=0.5, vmin=1, vmax=2)
        axes[2,4].axis('off')
        
        axes[2,5].imshow(img_vol[:, :, y_idx], cmap='gray')
        masked_pred = np.ma.masked_where(pred_fine_vol[:, :, y_idx] == 0, pred_fine_vol[:, :, y_idx])
        axes[2,5].imshow(masked_pred, cmap='jet', alpha=0.5, vmin=1, vmax=2)
        axes[2,5].axis('off')
        
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
    """Visualización estática con GT overlay"""
    
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

def train_cascade_sequential(df, args):
    """Entrenamiento completo con animaciones y métricas de overfitting"""
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    results = {"val_best": [], "test_best": []}
    
    # Split train/test
    list_ids = df["ID"].unique().tolist()
    list_ids_train, list_ids_backtest = train_test_split(
        list_ids, test_size=0.1, random_state=42
    )
    df_backtest = df[df["ID"].isin(list_ids_backtest)].reset_index(drop=True)
    df_train = df[~df["ID"].isin(list_ids_backtest)].reset_index(drop=True)
    
    print(f"Train: {len(df_train)} | Test: {len(df_backtest)}")
    
    test_ds = dataset.MRIDataset3D(
        df_backtest,
        transform=dataset.get_val_transforms(SPACING, (args.dim, args.dim, args.dim))
    )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4)
    
    gkf = GroupKFold(n_splits=args.num_folds)
    df_train["fold"] = -1
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(df_train, groups=df_train["ID"])):
        print(f"\n{'='*50}")
        print(f"FOLD {fold+1}/{args.num_folds}")
        print(f"{'='*50}")
        
        df_train.loc[val_idx, "fold"] = fold
        df_train_fold = df_train[df_train["fold"] != fold].reset_index(drop=True)
        df_val_fold = df_train[df_train["fold"] == fold].reset_index(drop=True)
        
        train_ds = dataset.MRIDataset3D(
            df_train_fold,
            transform=dataset.get_train_transforms_hippocampus(SPACING, (args.dim, args.dim, args.dim))
        )
        val_ds = dataset.MRIDataset3D(
            df_val_fold,
            transform=dataset.get_val_transforms(SPACING, (args.dim, args.dim, args.dim))
        )
        
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=8)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4)
        
        fold_dir = os.path.join(args.root_dir, f"fold_{fold+1}")
        os.makedirs(fold_dir, exist_ok=True)
        os.makedirs(f"images/{args.experiment_name}/fold_{fold+1}", exist_ok=True)
        
        # ========== ETAPA 1: BASE MODEL ==========
        if os.path.exists(os.path.join(fold_dir, "best_model.pth")):
            print("✓ Base model existe, cargando...")
            base_model = create_model(args.model_name, device).to(device)
            base_model.load_state_dict(torch.load(os.path.join(fold_dir, "best_model.pth")))
            best_dice_base = 0.5
        else:
            print("\n>>> ETAPA 1: Entrenando modelo BASE")
            
            base_model = create_model(args.model_name, device).to(device)
            loss_fn_base = losses.create_loss_function(args.loss_function)
            optimizer_base = torch.optim.AdamW(
                base_model.parameters(), 
                lr=args.lr, 
                weight_decay=args.weight_decay
            )
            #scheduler_base = torch.optim.lr_scheduler.ReduceLROnPlateau(
            #    optimizer_base, mode='max', factor=0.9, patience=10
            #)
            # En lugar de ReduceLROnPlateau, usar CosineAnnealingWarmRestarts
            scheduler_base = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer_base, 
                T_0=20,  # Restart cada 20 epochs
                T_mult=2,
                eta_min=1e-6
            )


            scaler_base = GradScaler()
            
            best_dice_base = 0.0
            patience_counter = 0
            warmup_epochs = min(100, args.num_epochs // 3)
            
            if args.use_wandb:
                run_base = wandb.init(
                    project=os.getenv('PROJECT_WANDB'),
                    entity=os.getenv('ENTITY'),
                    name=f"{args.experiment_name}_fold{fold}_base",
                    group=args.experiment_name,
                    config=vars(args),
                    reinit=True,
                )
            
            for epoch in range(warmup_epochs):
                base_model.train()
                epoch_loss = 0
                
                for batch_idx, batch in enumerate(tqdm.tqdm(train_loader, desc=f"Base Epoch {epoch+1}/{warmup_epochs}")):
                    x = batch["image"].to(device)
                    y = batch["label"].to(device)
                    
                    optimizer_base.zero_grad()
                    
                    with autocast():
                        logits = base_model(x)
                        loss = loss_fn_base(logits, y)
                    
                    scaler_base.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(base_model.parameters(), 1.0)
                    scaler_base.step(optimizer_base)
                    scaler_base.update()
                    
                    epoch_loss += loss.item()
                    
                    # Visualización periódica
                    if batch_idx % 50 == 0 and epoch % 10 == 0:
                        with torch.no_grad():
                            pred = torch.argmax(torch.softmax(logits, dim=1), dim=1)
                            img_path = save_training_visualization(
                                x, y, pred.cpu().numpy(), None,
                                epoch, batch_idx, args.experiment_name, 
                                phase="train_base", fold=fold+1
                            )
                            if args.use_wandb:
                                run_base.log({
                                    "train/visualization": wandb.Image(img_path),
                                    "epoch": epoch
                                })
                
                epoch_loss /= len(train_loader)
                
                # VALIDATION & TEST
                val_metrics = evaluate_base_model(base_model, val_loader, device, loss_fn_base)
                test_metrics = evaluate_base_model(base_model, test_loader, device, loss_fn_base)
                
                val_dice = val_metrics["dice_avg"]
                val_loss = val_metrics["loss"]
                test_loss = test_metrics["loss"]
                
                print(f"Base Epoch {epoch+1}:")
                print(f"  Loss: Train={epoch_loss:.4f}, Val={val_loss:.4f}, Test={test_loss:.4f}")
                print(f"  Dice: Val={val_dice:.4f}, Test={test_metrics['dice_avg']:.4f}")
                print(f"  Overfitting gap: {val_loss - epoch_loss:.4f}")
                
                if args.use_wandb:
                    run_base.log({
                        "base/train_loss": epoch_loss,
                        "base/val_loss": val_loss,
                        "base/test_loss": test_loss,
                        "base/overfitting_gap": val_loss - epoch_loss,
                        "base/val_dice_avg": val_dice,
                        "base/val_dice_L": val_metrics["dice_L"],
                        "base/val_dice_R": val_metrics["dice_R"],
                        "base/test_dice_avg": test_metrics["dice_avg"],
                        "epoch": epoch
                    })
                
                # Visualización y animación en validación
                if epoch % 10 == 0:
                    with torch.no_grad():
                        for i, val_batch in enumerate(val_loader):
                            if i == 0:
                                x_val = val_batch["image"].to(device)
                                y_val = val_batch["label"].to(device)
                                logits_val = base_model(x_val)
                                pred_val = torch.argmax(torch.softmax(logits_val, dim=1), dim=1)
                                
                                img_path = save_training_visualization(
                                    x_val, y_val, pred_val.cpu().numpy(), None,
                                    epoch, 0, args.experiment_name,
                                    phase="val_base", fold=fold+1
                                )
                                
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
                
                scheduler_base.step(val_dice)
                if val_dice > best_dice_base:
                    best_dice_base = val_dice
                    patience_counter = 0
                    torch.save(base_model.state_dict(), os.path.join(fold_dir, "best_model.pth"))
                    print(f"✓ Saved best base model: Dice={best_dice_base:.4f}")
                else:
                    patience_counter += 1
                    if patience_counter > 30:
                        print("Early stopping")
                        break
            
            if args.use_wandb:
                run_base.finish()
        
        # ========== ETAPA 2: FINE MODEL ==========
        print(f"\n>>> ETAPA 2: Fine-tuning")
        
        base_model.eval()
        for param in base_model.parameters():
            param.requires_grad = False
        
        fine_model = create_model(args.model_name, device).to(device)
        loss_fn_fine = losses.create_loss_function(args.loss_function)
        optimizer_fine = torch.optim.AdamW(
            fine_model.parameters(),
            lr=args.lr * 0.5,
            weight_decay=args.weight_decay
        )
        #scheduler_fine = torch.optim.lr_scheduler.CosineAnnealingLR(
        #    optimizer_fine, T_max=50, eta_min=1e-6
        #)
        # O OneCycleLR para fine-tuning
        scheduler_fine = torch.optim.lr_scheduler.OneCycleLR(
            optimizer_fine,
            max_lr=args.lr,
            epochs=50,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
            anneal_strategy='cos'
        )        
        scaler_fine = GradScaler()
        
        best_dice_fine = 0.0
        roi_margin = 16
        roi_size_fine = (args.dim//2, args.dim//2, args.dim//2)
        MAX_ROIS_PER_BATCH = 8
        
        if args.use_wandb:
            run_fine = wandb.init(
                project=os.getenv('PROJECT_WANDB'),
                entity=os.getenv('ENTITY'),
                name=f"{args.experiment_name}_fold{fold}_fine",
                group=args.experiment_name,
                config=vars(args),
                reinit=True
            )
        
        for epoch in range(50):
            fine_model.train()
            epoch_loss = 0
            n_batches = 0
            
            for batch_idx, batch in enumerate(tqdm.tqdm(train_loader, desc=f"Fine Epoch {epoch+1}/50")):
                x = batch["image"].to(device)
                y = batch["label"].to(device)
                
                with torch.no_grad():
                    base_logits = base_model(x)
                    base_pred = torch.argmax(torch.softmax(base_logits, dim=1), dim=1)
                
                # Recolectar ROIs (optimizado con batch)
                all_img_rois = []
                all_lbl_rois = []
                roi_info = []
                
                for i in range(x.shape[0]):
                    bboxes_i = get_roi_bbox_from_logits(base_logits[i:i+1], thr=0.2, margin=roi_margin)
                    
                    for bb in bboxes_i:
                        if bb is None:
                            continue
                        
                        img_roi = crop_to_bbox(x[i:i+1], bb)
                        lbl_roi = crop_to_bbox(y[i:i+1], bb)
                        
                        if lbl_roi.sum() < 10:
                            continue
                        
                        img_roi = resize_volume(img_roi, roi_size_fine, mode="trilinear")
                        lbl_roi = resize_volume(lbl_roi.float(), roi_size_fine, mode="nearest").long()
                        
                        all_img_rois.append(img_roi)
                        all_lbl_rois.append(lbl_roi)
                        roi_info.append((i, bb))
                
                if len(all_img_rois) == 0:
                    continue
                
                # Procesar en mini-batches
                optimizer_fine.zero_grad()
                batch_loss = 0
                n_mini = 0
                
                for start_idx in range(0, len(all_img_rois), MAX_ROIS_PER_BATCH):
                    end_idx = min(start_idx + MAX_ROIS_PER_BATCH, len(all_img_rois))
                    
                    img_roi_batch = torch.cat(all_img_rois[start_idx:end_idx], dim=0)
                    lbl_roi_batch = torch.cat(all_lbl_rois[start_idx:end_idx], dim=0)
                    
                    with autocast():
                        logits_fine_batch = fine_model(img_roi_batch)
                        loss = loss_fn_fine(logits_fine_batch, lbl_roi_batch)
                    
                    scaler_fine.scale(loss).backward()
                    batch_loss += loss.item()
                    n_mini += 1
                
                torch.nn.utils.clip_grad_norm_(fine_model.parameters(), 1.0)
                scaler_fine.step(optimizer_fine)
                scaler_fine.update()
                
                epoch_loss += batch_loss / n_mini if n_mini > 0 else 0
                n_batches += 1
                
                # Visualización periódica
                if batch_idx % 50 == 0 and epoch % 10 == 0:
                    with torch.no_grad():
                        # Reconstruir predicción para visualización
                        fine_preds = torch.zeros_like(base_pred)
                        for (batch_i, bb), roi in zip(roi_info[:MAX_ROIS_PER_BATCH], all_img_rois[:MAX_ROIS_PER_BATCH]):
                            logits_roi = fine_model(roi)
                            pred_roi = torch.argmax(torch.softmax(logits_roi, dim=1), dim=1)
                            
                            z0, y0, x0, z1, y1, x1 = bb
                            pred_roi_up = resize_volume(
                                pred_roi.unsqueeze(0).float(),
                                (z1-z0, y1-y0, x1-x0),
                                mode="nearest"
                            )[0].long()
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
            
            # VALIDATION & TEST CASCADE
            val_metrics = evaluate_cascade(
                base_model, fine_model, val_loader, device,
                loss_fn_fine, roi_margin, roi_size_fine, args.dim
            )
            test_metrics = evaluate_cascade(
                base_model, fine_model, test_loader, device,
                loss_fn_fine, roi_margin, roi_size_fine, args.dim
            )
            
            val_dice_fine = val_metrics["dice_fine_avg"]
            val_loss_fine = val_metrics["loss_fine"]
            test_loss_fine = test_metrics["loss_fine"]
            
            print(f"Fine Epoch {epoch+1}:")
            print(f"  Loss: Train={epoch_loss:.4f}, Val={val_loss_fine:.4f}, Test={test_loss_fine:.4f}")
            print(f"  Dice Fine: Val={val_dice_fine:.4f}, Test={test_metrics['dice_fine_avg']:.4f}")
            print(f"  Overfitting gap: {val_loss_fine - epoch_loss:.4f}")
            
            if args.use_wandb:
                run_fine.log({
                    "fine/train_loss": epoch_loss,
                    "fine/val_loss": val_loss_fine,
                    "fine/test_loss": test_loss_fine,
                    "fine/overfitting_gap": val_loss_fine - epoch_loss,
                    "fine/val_dice_base": val_metrics["dice_avg"],
                    "fine/val_dice_fine": val_dice_fine,
                    "fine/val_dice_fine_L": val_metrics["dice_fine_L"],
                    "fine/val_dice_fine_R": val_metrics["dice_fine_R"],
                    "fine/test_dice_fine": test_metrics["dice_fine_avg"],
                    "epoch": epoch
                })
            
            # Visualización y animación en validación
            if epoch % 10 == 0:
                with torch.no_grad():
                    for i, val_batch in enumerate(val_loader):
                        if i == 0:
                            x_val = val_batch["image"].to(device)
                            y_val = val_batch["label"].to(device)
                            
                            # Cascade completo
                            logits_base = base_model(x_val)
                            pred_base_val = torch.argmax(torch.softmax(logits_base, dim=1), dim=1)
                            bboxes = get_roi_bbox_from_logits(logits_base, thr=0.2, margin=roi_margin)
                            
                            logits_fine = torch.zeros_like(logits_base)
                            for j, bb in enumerate(bboxes):
                                img_roi = crop_to_bbox(x_val[j:j+1], bb)
                                img_roi = resize_volume(img_roi, roi_size_fine, mode="trilinear")
                                logits2 = fine_model(img_roi)
                                
                                z0, y0, x0, z1, y1, x1 = bb
                                logits2_up = resize_volume(logits2, (z1-z0, y1-y0, x1-x0), mode="trilinear")[0]
                                logits_fine[j:j+1, :, z0:z1, y0:y1, x0:x1] = logits2_up.cpu()
                            
                            pred_fine_val = torch.argmax(torch.softmax(logits_fine, dim=1), dim=1)
                            
                            img_path = save_training_visualization(
                                x_val, y_val, pred_base_val.cpu().numpy(), pred_fine_val.cpu().numpy(),
                                epoch, 0, args.experiment_name,
                                phase="val_fine", fold=fold+1
                            )
                            
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
            
            if val_dice_fine > best_dice_fine:
                best_dice_fine = val_dice_fine
                torch.save(fine_model.state_dict(), os.path.join(fold_dir, "best_fine_model.pth"))
                print(f"✓ Saved best fine model: Dice={best_dice_fine:.4f}")
        
        if args.use_wandb:
            run_fine.finish()
        
        # Test final
        print(f"\n>>> Test final Fold {fold+1}")
        results["test_best"].append(test_metrics)
        print(f"Final: Base={test_metrics['dice_avg']:.4f}, Fine={test_metrics['dice_fine_avg']:.4f}")
    
    return results

def evaluate_base_model(model, loader, device, loss_fn):
    """Evaluación con métricas de loss"""
    model.eval()
    all_dice = {"L": [], "R": []}
    epoch_loss = 0.0
    
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            
            logits = model(x)
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
    """Evaluación cascade con loss"""
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
            
            logits_base = base_model(x)
            bboxes = get_roi_bbox_from_logits(logits_base, thr=0.2, margin=roi_margin)
            
            logits_fine = torch.zeros_like(logits_base)
            for i, bb in enumerate(bboxes):
                img_roi = crop_to_bbox(x[i:i+1], bb)
                lbl_roi = crop_to_bbox(y[i:i+1], bb)
                
                img_roi = resize_volume(img_roi, roi_size_fine, mode="trilinear")
                lbl_roi = resize_volume(lbl_roi.float(), roi_size_fine, mode="nearest").long()
                
                logits2 = fine_model(img_roi)
                
                loss_roi = loss_fn(logits2, lbl_roi)
                epoch_loss_fine += loss_roi.item()
                n_rois += 1
                
                z0, y0, x0, z1, y1, x1 = bb
                logits2_up = resize_volume(
                    logits2, 
                    (z1-z0, y1-y0, x1-x0), 
                    mode="trilinear"
                )[0]
                logits_fine[i:i+1, :, z0:z1, y0:y1, x0:x1] = logits2_up.cpu()
            
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
"""
python training.py   --model_name=eff-b2   --device=cuda:3   --root_dir=/data/cristian/projects/med_data/rise-miccai/task-2/3d_models/predictions/eff-b2-cascade-balance_ce_focal_192_v2/fold_models   --num_epochs=150   --num_folds=5   --use_mixup=0   --experiment_name=eff-b2-cascade-balance_ce_focal_192   --lr=1e-4   --weight_decay=1e-5   --loss_function=balance_ce_focal   --dim=192   --batch_size=2   --use_wandb=1

"""

# Main
if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="eff-b2")
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_epochs', type=int, default=150)
    parser.add_argument('--num_folds', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--root_dir', type=str, required=True)
    parser.add_argument('--dim', type=int, default=192)
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--loss_function', type=str, default="dice_focal_spatial")
    parser.add_argument('--use_mixup', type=int, default=0)
    parser.add_argument('--use_wandb', type=int, default=1)
    
    args = parser.parse_args()
    load_dotenv()
    
    df = pd.read_csv("results/preprocessed_data/task2/df_train_hipp.csv")
    
    results = train_cascade_sequential(df, args)
    print("\nFinal results:", results)
