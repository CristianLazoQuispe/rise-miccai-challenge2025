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

# Configuración
SPACING = (1.0, 1.0, 1.0)
def save_training_visualization(x, y, pred_base, pred_fine, epoch, batch_idx, 
                               exp_name, phase="train", fold=1):
    """Guarda visualización de las 3 vistas con slices que contengan hipocampo"""
    
    # Crear directorio
    vis_dir = f"images/{exp_name}/fold_{fold}/{phase}"
    os.makedirs(vis_dir, exist_ok=True)
    
    # Convertir a numpy si es necesario
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()
    
    # Encontrar slices con contenido en ground truth para cada vista
    def find_best_slice(mask, axis):
        """Encuentra el slice con más contenido de hipocampo"""
        sums = []
        for i in range(mask.shape[axis]):
            if axis == 0:
                slice_sum = np.sum(mask[i, :, :] > 0)
            elif axis == 1:
                slice_sum = np.sum(mask[:, i, :] > 0)
            else:  # axis == 2
                slice_sum = np.sum(mask[:, :, i] > 0)
            sums.append(slice_sum)
        
        # Si no hay contenido, usar el centro
        if max(sums) == 0:
            return mask.shape[axis] // 2
        
        # Encontrar el slice con más contenido
        best_idx = np.argmax(sums)
        
        # Si el mejor slice está muy al borde, usar uno más centrado
        center = mask.shape[axis] // 2
        if abs(best_idx - center) > mask.shape[axis] // 3:
            # Buscar el slice más cercano al centro con contenido
            for offset in range(mask.shape[axis] // 3):
                if center + offset < len(sums) and sums[center + offset] > 0:
                    return center + offset
                if center - offset >= 0 and sums[center - offset] > 0:
                    return center - offset
        
        return best_idx
    
    # Ground truth mask
    gt_mask = y[0, 0] if y.ndim == 5 else y[0]
    
    # Encontrar mejores slices para cada vista
    best_axial = find_best_slice(gt_mask, 0)
    best_sagittal = find_best_slice(gt_mask, 1)
    best_coronal = find_best_slice(gt_mask, 2)
    
    # Crear figura 3x6 (3 vistas x 6 visualizaciones)
    fig, axes = plt.subplots(3, 6, figsize=(20, 12))
    
    # Preparar datos
    img_vol = x[0, 0] if x.ndim == 5 else x[0]
    pred_base_vol = pred_base[0] if pred_base is not None else np.zeros_like(gt_mask)
    pred_fine_vol = pred_fine[0] if pred_fine is not None else np.zeros_like(gt_mask)
    
    # Función helper para plotear una fila
    def plot_row(ax_row, img_slice, gt_slice, pred_base_slice, pred_fine_slice, view_name):
        # Columna 1: Imagen original
        ax_row[0].imshow(img_slice, cmap='gray')
        ax_row[0].set_title(f'{view_name} - Input')
        ax_row[0].axis('off')
        
        # Columna 2: Ground Truth
        im = ax_row[1].imshow(gt_slice, cmap='viridis', vmin=0, vmax=2)
        ax_row[1].set_title('Ground Truth')
        ax_row[1].axis('off')
        
        # Columna 3: Predicción Base
        ax_row[2].imshow(pred_base_slice, cmap='viridis', vmin=0, vmax=2)
        ax_row[2].set_title('Base Prediction')
        ax_row[2].axis('off')
        
        # Columna 4: Predicción Fine
        ax_row[3].imshow(pred_fine_slice, cmap='viridis', vmin=0, vmax=2)
        ax_row[3].set_title('Fine Prediction')
        ax_row[3].axis('off')
        
        # Columna 5: Overlay GT
        ax_row[4].imshow(img_slice, cmap='gray')
        masked_gt = np.ma.masked_where(gt_slice == 0, gt_slice)
        ax_row[4].imshow(masked_gt, cmap='jet', alpha=0.5, vmin=1, vmax=2)
        ax_row[4].set_title('GT Overlay')
        ax_row[4].axis('off')
        
        # Columna 6: Overlay Predicción
        ax_row[5].imshow(img_slice, cmap='gray')
        masked_pred = np.ma.masked_where(pred_fine_slice == 0, pred_fine_slice)
        ax_row[5].imshow(masked_pred, cmap='jet', alpha=0.5, vmin=1, vmax=2)
        ax_row[5].set_title('Prediction Overlay')
        ax_row[5].axis('off')
        
        return im
    
    # Vista Axial (slice horizontal)
    im = plot_row(
        axes[0],
        img_vol[best_axial, :, :],
        gt_mask[best_axial, :, :],
        pred_base_vol[best_axial, :, :],
        pred_fine_vol[best_axial, :, :],
        f'Axial (z={best_axial})'
    )
    
    # Vista Sagital (slice lateral)
    plot_row(
        axes[1],
        img_vol[:, best_sagittal, :],
        gt_mask[:, best_sagittal, :],
        pred_base_vol[:, best_sagittal, :],
        pred_fine_vol[:, best_sagittal, :],
        f'Sagittal (x={best_sagittal})'
    )
    
    # Vista Coronal (slice frontal)
    plot_row(
        axes[2],
        img_vol[:, :, best_coronal],
        gt_mask[:, :, best_coronal],
        pred_base_vol[:, :, best_coronal],
        pred_fine_vol[:, :, best_coronal],
        f'Coronal (y={best_coronal})'
    )
    
    # Agregar colorbar
    cbar_ax = fig.add_axes([0.92, 0.35, 0.02, 0.3])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Label (0: BG, 1: Left, 2: Right)', rotation=270, labelpad=20)
    
    # Calcular métricas globales (no por slice)
    dice_l_base = metrics.dice_score(pred_base_vol, gt_mask, 1)  if pred_base is not None else 0
    dice_r_base = metrics.dice_score(pred_base_vol, gt_mask, 2)  if pred_base is not None else 0
    dice_l_fine = metrics.dice_score(pred_fine_vol, gt_mask, 1)  if pred_fine is not None else 0
    dice_r_fine = metrics.dice_score(pred_fine_vol, gt_mask, 2)  if pred_fine is not None else 0
    
    # Título con métricas
    title = f'{phase.upper()} - Epoch {epoch} - Batch {batch_idx}\n'
    title += f'Base Dice: L={dice_l_base:.3f} R={dice_r_base:.3f} | '
    title += f'Fine Dice: L={dice_l_fine:.3f} R={dice_r_fine:.3f}'
    
    # Agregar información sobre contenido de GT
    gt_voxels_l = np.sum(gt_mask == 1)
    gt_voxels_r = np.sum(gt_mask == 2)
    title += f'\nGT Voxels: L={gt_voxels_l} R={gt_voxels_r}'
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    
    # Guardar
    filename = f"{phase}_epoch{epoch:03d}_batch{batch_idx:03d}.png"
    save_path = os.path.join(vis_dir, filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Debug: imprimir información
    print(f"Visualization saved: {save_path}")
    print(f"  GT has content: L={gt_voxels_l>0}, R={gt_voxels_r>0}")
    print(f"  Slices selected: Axial={best_axial}, Sagittal={best_sagittal}, Coronal={best_coronal}")
    
    return save_path

# También agregar una función para crear GIF animado de todo el volumen
def create_volume_animation(x, y, pred_base, pred_fine, epoch, exp_name, phase="val", fold=1):
    """Crea una animación GIF mostrando todos los slices"""
    import imageio
    from PIL import Image
    
    vis_dir = f"images/{exp_name}/fold_{fold}/{phase}/animations"
    os.makedirs(vis_dir, exist_ok=True)
    
    frames = []
    
    # Convertir a numpy
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()
    
    img_vol = x[0, 0] if x.ndim == 5 else x[0]
    gt_mask = y[0, 0] if y.ndim == 5 else y[0]
    pred_vol = pred_fine[0] if pred_fine is not None else pred_base[0]
    
    # Crear frames para cada slice axial
    for z in range(0, img_vol.shape[0], 2):  # Cada 2 slices para reducir tamaño
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        axes[0].imshow(img_vol[z], cmap='gray')
        axes[0].set_title(f'Input (z={z})')
        axes[0].axis('off')
        
        axes[1].imshow(gt_mask[z], cmap='viridis', vmin=0, vmax=2)
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        axes[2].imshow(pred_vol[z], cmap='viridis', vmin=0, vmax=2)
        axes[2].set_title('Prediction')
        axes[2].axis('off')
        
        axes[3].imshow(img_vol[z], cmap='gray')
        masked = np.ma.masked_where(pred_vol[z] == 0, pred_vol[z])
        axes[3].imshow(masked, cmap='jet', alpha=0.5, vmin=1, vmax=2)
        axes[3].set_title('Overlay')
        axes[3].axis('off')
        
        plt.suptitle(f'{phase.upper()} - Epoch {epoch}')
        plt.tight_layout()
        
        # Convertir a imagen
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)
        plt.close()
    
    # Guardar como GIF
    gif_path = os.path.join(vis_dir, f"{phase}_epoch{epoch:03d}.gif")
    imageio.mimsave(gif_path, frames, duration=0.1)
    
    return gif_path

# Modificar la función de evaluación para usar la visualización mejorada
def evaluate_base_model_with_vis(model, loader, device, loss_fn, epoch, exp_name, fold, use_wandb, run):
    """Evaluación del modelo base con visualización mejorada"""
    metrics = evaluate_base_model(model, loader, device, loss_fn)
    
    model.eval()
    with torch.no_grad():
        # Buscar un batch con contenido en GT
        for i, batch in enumerate(loader):
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            
            # Verificar si este batch tiene hipocampo
            if y.sum() > 100:  # Tiene contenido significativo
                logits = model(x)
                pred = torch.argmax(torch.softmax(logits, dim=1), dim=1)
                
                img_path = save_training_visualization(
                    x, y, pred.cpu().numpy(), None,
                    epoch, i, exp_name, phase="val_base", fold=fold
                )
                
                # Crear animación cada 20 epochs
                if epoch % 20 == 0:
                    gif_path = create_volume_animation(
                        x, y, pred.cpu().numpy(), None,
                        epoch, exp_name, phase="val_base", fold=fold
                    )
                    print(f"Animation saved: {gif_path}")
                
                if use_wandb and run is not None:
                    run.log({
                        f"val/visualization": wandb.Image(img_path),
                        "epoch": epoch
                    })
                    if epoch % 20 == 0:
                        run.log({
                            f"val/animation": wandb.Video(gif_path),
                            "epoch": epoch
                        })
                break
    
    return metrics


def train_cascade_sequential(df, args):
    """Entrenamiento secuencial con visualización"""
    
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
    
    # Test dataset común
    test_ds = dataset.MRIDataset3D(
        df_backtest,
        transform=dataset.get_val_transforms(SPACING, (args.dim, args.dim, args.dim))
    )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4)
    
    # GroupKFold
    gkf = GroupKFold(n_splits=args.num_folds)
    df_train["fold"] = -1
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(df_train, groups=df_train["ID"])):
        print(f"\n{'='*50}")
        print(f"FOLD {fold+1}/{args.num_folds}")
        print(f"{'='*50}")
        
        df_train.loc[val_idx, "fold"] = fold
        df_train_fold = df_train[df_train["fold"] != fold].reset_index(drop=True)
        df_val_fold = df_train[df_train["fold"] == fold].reset_index(drop=True)
        
        # Datasets
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
        
        # Directorio para guardar modelos y visualizaciones
        fold_dir = os.path.join(args.root_dir, f"fold_{fold+1}")
        os.makedirs(fold_dir, exist_ok=True)
        os.makedirs(f"images/{args.experiment_name}/fold_{fold+1}", exist_ok=True)
        
        # ========== ETAPA 1: ENTRENAR MODELO BASE ==========
        print("\n>>> ETAPA 1: Entrenando modelo BASE")
        
        base_model = create_model(args.model_name, device).to(device)
        loss_fn_base = losses.create_loss_function(args.loss_function)
        optimizer_base = torch.optim.AdamW(
            base_model.parameters(), 
            lr=args.lr, 
            weight_decay=args.weight_decay
        )
        scheduler_base = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_base, mode='max', factor=0.9, patience=10
        )
        scaler_base = GradScaler()
        
        best_dice_base = 0.0
        patience_counter = 0
        warmup_epochs = min(100, args.num_epochs // 3)
        
        # Wandb para base model
        if args.use_wandb:
            exp_name = args.experiment_name
            run_base = wandb.init(
                project=os.getenv('PROJECT_WANDB'),
                entity=os.getenv('ENTITY'),
                name=f"{args.experiment_name}_fold{fold}_base",
                group=exp_name,
                config=vars(args),
                save_code=True,
                reinit=True,
            )
        
        for epoch in range(warmup_epochs):
            # Training
            base_model.train()
            epoch_loss = 0
            
            for batch_idx, batch in enumerate(tqdm.tqdm(train_loader, desc=f"Base Epoch {epoch+1}/{warmup_epochs}")):
                x = batch["image"].to(device)
                y = batch["label"].to(device)
                
                # MixUp opcional (solo primeras epochs)
                use_mixup = args.use_mixup and epoch < 50 and np.random.rand() < 0.3
                if use_mixup:
                    lam = np.random.beta(0.4, 0.4)
                    idx = torch.randperm(x.size(0)).to(device)
                    x_mixed = lam * x + (1 - lam) * x[idx]
                    y1, y2 = y, y[idx]
                else:
                    x_mixed = x
                    y1 = y
                
                optimizer_base.zero_grad()
                
                with autocast():
                    logits = base_model(x_mixed)
                    if use_mixup:
                        loss = lam * loss_fn_base(logits, y1) + (1 - lam) * loss_fn_base(logits, y2)
                    else:
                        loss = loss_fn_base(logits, y1)
                
                scaler_base.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(base_model.parameters(), 1.0)
                scaler_base.step(optimizer_base)
                scaler_base.update()
                
                epoch_loss += loss.item()
                
                # Visualización periódica (cada 50 batches)
                if batch_idx % 50 == 0 and epoch % 10 == 0:
                    with torch.no_grad():
                        pred = torch.argmax(torch.softmax(logits, dim=1), dim=1)
                        img_path = save_training_visualization(
                            x, y1, pred.cpu().numpy(), None,
                            epoch, batch_idx, args.experiment_name, 
                            phase="train_base", fold=fold+1
                        )
                        if args.use_wandb:
                            run_base.log({
                                f"train/visualization": wandb.Image(img_path),
                                "epoch": epoch
                            })
            
            epoch_loss /= len(train_loader)
            
            # Validation
            base_model.eval()
            val_metrics = evaluate_base_model_with_vis(
                base_model, val_loader, device, loss_fn_base,
                epoch, args.experiment_name, fold+1, args.use_wandb, run_base if args.use_wandb else None
            )
            test_metrics = evaluate_base_model(base_model, test_loader, device, loss_fn_base)
            
            val_dice = val_metrics["dice_avg"]
            val_loss = val_metrics["loss"]
            test_loss = test_metrics["loss"]
            
            print(f"Base Epoch {epoch+1}: Train Loss={epoch_loss:.4f}, "
                  f"Val Loss={val_loss:.4f}, Test Loss={test_loss:.4f}, "
                  f"Val Dice={val_dice:.4f}, Test Dice={test_metrics['dice_avg']:.4f}")
            
            # Logging
            if args.use_wandb:
                run_base.log({
                    "train/loss": epoch_loss,
                    "val/dice_avg": val_dice,
                    "val/dice_L": val_metrics["dice_L"],
                    "val/dice_R": val_metrics["dice_R"],
                    "test/dice_avg": test_metrics["dice_avg"],
                    "epoch": epoch
                })
            
            # Save best
            scheduler_base.step(val_dice)
            if val_dice > best_dice_base:
                best_dice_base = val_dice
                patience_counter = 0
                torch.save(base_model.state_dict(), os.path.join(fold_dir, "best_model.pth"))
                print(f"✓ Saved best base model: Dice={best_dice_base:.4f}")
            else:
                patience_counter += 1
                if patience_counter > 30:
                    print("Early stopping base model")
                    break
        
        if args.use_wandb:
            run_base.finish()
        
        # ========== ETAPA 2: ENTRENAR MODELO FINO ==========
        print(f"\n>>> ETAPA 2: Entrenando modelo FINO (Base Dice={best_dice_base:.4f})")
        
        # Cargar mejor modelo base
        base_model.load_state_dict(torch.load(os.path.join(fold_dir, "best_model.pth")))
        base_model.eval()
        for param in base_model.parameters():
            param.requires_grad = False
        
        # Crear modelo fino
        fine_model = create_model(args.model_name, device).to(device)
        loss_fn_fine = losses.create_loss_function(args.loss_function)
        optimizer_fine = torch.optim.AdamW(
            fine_model.parameters(),
            lr=args.lr * 0.5,
            weight_decay=args.weight_decay
        )
        scheduler_fine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_fine, T_max=50, eta_min=1e-6
        )
        scaler_fine = GradScaler()
        
        best_dice_fine = 0.0
        roi_margin = 16
        roi_size_fine = (args.dim//2, args.dim//2, args.dim//2)
        
        # Wandb para fine model
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
            
            for batch_idx, batch in enumerate(tqdm.tqdm(train_loader, desc=f"Fine Epoch {epoch+1}/50")):
                x = batch["image"].to(device)
                y = batch["label"].to(device)
                
                # Obtener ROIs del modelo base
                with torch.no_grad():
                    base_logits = base_model(x)
                    bboxes = get_roi_bbox_from_logits(base_logits, thr=0.2, margin=roi_margin)
                    base_pred = torch.argmax(torch.softmax(base_logits, dim=1), dim=1)
                
                # Entrenar en cada ROI
                optimizer_fine.zero_grad()
                total_loss = 0
                valid_rois = 0
                fine_preds = torch.zeros_like(base_pred)
                
                for i, bb in enumerate(bboxes):
                    img_roi = crop_to_bbox(x[i:i+1], bb)
                    lbl_roi = crop_to_bbox(y[i:i+1], bb)
                    
                    if lbl_roi.sum() < 10:
                        continue
                    
                    img_roi = resize_volume(img_roi, roi_size_fine, mode="trilinear")
                    lbl_roi = resize_volume(lbl_roi.float(), roi_size_fine, mode="nearest").long()
                    
                    with autocast():
                        logits_fine = fine_model(img_roi)
                        loss = loss_fn_fine(logits_fine, lbl_roi)
                    
                    # Guardar predicción para visualización
                    if batch_idx % 50 == 0 and epoch % 10 == 0:
                        with torch.no_grad():
                            z0, y0, x0, z1, y1, x1 = bb
                            pred_fine = torch.argmax(torch.softmax(logits_fine, dim=1), dim=1)
                            pred_fine_up = resize_volume(
                                pred_fine.unsqueeze(0).float(),
                                (z1-z0, y1-y0, x1-x0),
                                mode="nearest"
                            )[0].long()
                            fine_preds[i:i+1, z0:z1, y0:y1, x0:x1] = pred_fine_up
                    
                    total_loss += loss
                    valid_rois += 1
                
                if valid_rois > 0:
                    avg_loss = total_loss / valid_rois
                    scaler_fine.scale(avg_loss).backward()
                    torch.nn.utils.clip_grad_norm_(fine_model.parameters(), 1.0)
                    scaler_fine.step(optimizer_fine)
                    scaler_fine.update()
                    
                    epoch_loss += avg_loss.item()
                    
                    # Visualización periódica
                    if batch_idx % 50 == 0 and epoch % 10 == 0:
                        img_path = save_training_visualization(
                            x, y, base_pred.cpu().numpy(), fine_preds.cpu().numpy(),
                            epoch, batch_idx, args.experiment_name,
                            phase="train_fine", fold=fold+1
                        )
                        if args.use_wandb:
                            run_fine.log({
                                f"train/visualization": wandb.Image(img_path),
                                "epoch": epoch
                            })
            
            epoch_loss /= len(train_loader)
            scheduler_fine.step()
            
            # Validation con cascade
            val_metrics = evaluate_cascade_with_vis(
                base_model, fine_model, val_loader, device,
                loss_fn_base, roi_margin, roi_size_fine, args.dim,
                epoch, args.experiment_name, fold+1, args.use_wandb, 
                run_fine if args.use_wandb else None
            )
            
            val_dice_fine = val_metrics["dice_fine_avg"]
            
            print(f"Fine Epoch {epoch+1}: Loss={epoch_loss:.4f}, "
                  f"Fine Dice={val_dice_fine:.4f}")
            
            # Logging
            if args.use_wandb:
                run_fine.log({
                    "train/loss_fine": epoch_loss,
                    "val/dice_fine_avg": val_dice_fine,
                    "val/dice_fine_L": val_metrics["dice_fine_L"],
                    "val/dice_fine_R": val_metrics["dice_fine_R"],
                    "epoch": epoch
                })
            
            # Save best fine model
            if val_dice_fine > best_dice_fine:
                best_dice_fine = val_dice_fine
                torch.save(fine_model.state_dict(), os.path.join(fold_dir, "best_fine_model.pth"))
                print(f"✓ Saved best fine model: Dice={best_dice_fine:.4f}")
        
        if args.use_wandb:
            run_fine.finish()
        
        # Evaluación final con visualización
        print(f"\n>>> Evaluación final Fold {fold+1}")
        base_model.load_state_dict(torch.load(os.path.join(fold_dir, "best_model.pth")))
        fine_model.load_state_dict(torch.load(os.path.join(fold_dir, "best_fine_model.pth")))
        
        final_metrics = evaluate_cascade_with_vis(
            base_model, fine_model, test_loader, device,
            loss_fn_base, roi_margin, roi_size_fine, args.dim,
            999, args.experiment_name, fold+1, False, None  # epoch 999 para indicar final
        )
        
        results["test_best"].append(final_metrics)
        print(f"Final Test Dice: Base={final_metrics['dice_avg']:.4f}, "
              f"Fine={final_metrics['dice_fine_avg']:.4f}")
    
    return results

def evaluate_base_model(model, loader, device, loss_fn):
    """Evaluación simple del modelo base"""
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

def evaluate_base_model_with_vis(model, loader, device, loss_fn, epoch, exp_name, fold, use_wandb, run):
    """Evaluación del modelo base con visualización"""
    metrics = evaluate_base_model(model, loader, device, loss_fn)
    
    # Visualizar un batch aleatorio
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i == min(3, len(loader)-1):  # Visualizar el 3er batch o el último
                x = batch["image"].to(device)
                y = batch["label"].to(device)
                logits = model(x)
                pred = torch.argmax(torch.softmax(logits, dim=1), dim=1)
                
                img_path = save_training_visualization(
                    x, y, pred.cpu().numpy(), None,
                    epoch, i, exp_name, phase="val_base", fold=fold
                )
                
                if use_wandb and run is not None:
                    run.log({
                        f"val/visualization": wandb.Image(img_path),
                        "epoch": epoch
                    })
                break
    
    return metrics

def evaluate_cascade(base_model, fine_model, loader, device, loss_fn, 
                    roi_margin, roi_size_fine, dim):
    """Evaluación del cascade completo"""
    base_model.eval()
    fine_model.eval()
    
    metrics_base = {"L": [], "R": []}
    metrics_fine = {"L": [], "R": []}
    
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            
            # Base prediction
            logits_base = base_model(x)
            bboxes = get_roi_bbox_from_logits(logits_base, thr=0.2, margin=roi_margin)
            
            # Fine prediction
            logits_fine = torch.zeros_like(logits_base)
            for i, bb in enumerate(bboxes):
                img_roi = crop_to_bbox(x[i:i+1], bb)
                img_roi = resize_volume(img_roi, roi_size_fine, mode="trilinear")
                
                logits2 = fine_model(img_roi)
                
                z0, y0, x0, z1, y1, x1 = bb
                logits2_up = resize_volume(
                    logits2, 
                    (z1-z0, y1-y0, x1-x0), 
                    mode="trilinear"
                )[0]
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
                        metrics_base[key].append(dice_base )
                    if not np.isnan(dice_fine):
                        metrics_fine[key].append(dice_fine )
    
    return {
        "dice_L": np.mean(metrics_base["L"]) if metrics_base["L"] else 0,
        "dice_R": np.mean(metrics_base["R"]) if metrics_base["R"] else 0,
        "dice_avg": np.mean(metrics_base["L"] + metrics_base["R"]) if metrics_base["L"] else 0,
        "dice_fine_L": np.mean(metrics_fine["L"]) if metrics_fine["L"] else 0,
        "dice_fine_R": np.mean(metrics_fine["R"]) if metrics_fine["R"] else 0,
        "dice_fine_avg": np.mean(metrics_fine["L"] + metrics_fine["R"]) if metrics_fine["L"] else 0,
    }

def evaluate_cascade_with_vis(base_model, fine_model, loader, device, loss_fn,
                             roi_margin, roi_size_fine, dim, epoch, exp_name, 
                             fold, use_wandb, run):
    """Evaluación del cascade con visualización"""
    metrics = evaluate_cascade(base_model, fine_model, loader, device, loss_fn,
                              roi_margin, roi_size_fine, dim)
    
    # Visualizar un batch
    base_model.eval()
    fine_model.eval()
    
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i == min(3, len(loader)-1):
                x = batch["image"].to(device)
                y = batch["label"].to(device)
                
                # Base prediction
                logits_base = base_model(x)
                pred_base = torch.argmax(torch.softmax(logits_base, dim=1), dim=1)
                bboxes = get_roi_bbox_from_logits(logits_base, thr=0.2, margin=roi_margin)
                
                # Fine prediction
                logits_fine = torch.zeros_like(logits_base)
                for j, bb in enumerate(bboxes):
                    img_roi = crop_to_bbox(x[j:j+1], bb)
                    img_roi = resize_volume(img_roi, roi_size_fine, mode="trilinear")
                    logits2 = fine_model(img_roi)
                    
                    z0, y0, x0, z1, y1, x1 = bb
                    logits2_up = resize_volume(logits2, (z1-z0, y1-y0, x1-x0), mode="trilinear")[0]
                    logits_fine[j:j+1, :, z0:z1, y0:y1, x0:x1] = logits2_up.cpu()
                
                pred_fine = torch.argmax(torch.softmax(logits_fine, dim=1), dim=1)
                
                phase = "test" if epoch == 999 else "val_cascade"
                img_path = save_training_visualization(
                    x, y, pred_base.cpu().numpy(), pred_fine.cpu().numpy(),
                    epoch if epoch != 999 else "final", i, exp_name, phase=phase, fold=fold
                )
                
                if use_wandb and run is not None:
                    run.log({
                        f"{phase}/visualization": wandb.Image(img_path),
                        "epoch": epoch
                    })
                break
    
    return metrics

# Main execution
if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="efficientnet-b7")
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=150)
    parser.add_argument('--num_folds', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--root_dir', type=str, required=True)
    parser.add_argument('--dim', type=int, default=96)
    parser.add_argument('--experiment_name', type=str, default="hippocampus_cascade")
    parser.add_argument('--loss_function', type=str, default="dice_focal_hippocampus")
    parser.add_argument('--use_mixup', type=int, default=1)
    parser.add_argument('--use_wandb', type=int, default=1)
    
    args = parser.parse_args()
    load_dotenv()
    
    # Load data
    df = pd.read_csv("results/preprocessed_data/task2/df_train_hipp.csv")
    
    # Train
    results = train_cascade_sequential(df, args)
    print("\nFinal results:", results)