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
import scipy.ndimage as ndimage
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

# Configuración
SPACING = (1.0, 1.0, 1.0)

def correct_laterality_post(prediction):
    """
    Corrige etiquetas L/R solo si están mal posicionadas anatómicamente.
    """
    if prediction.max() == 0:
        return prediction
    
    corrected = prediction.copy()
    center_w = prediction.shape[2] / 2
    
    # Procesar cada clase por separado
    for class_id in [1, 2]:
        class_mask = (prediction == class_id)
        if not class_mask.any():
            continue
            
        # Encontrar componentes de esta clase
        labeled, n_features = ndimage.label(class_mask)
        
        for comp_id in range(1, n_features + 1):
            component = (labeled == comp_id)
            com = ndimage.center_of_mass(component)
            
            # Verificar si necesita corrección
            if class_id == 1:  # Left hippocampus
                # Debería estar en el lado izquierdo (com[2] < center)
                if com[2] > center_w:
                    # Está mal, cambiar a right
                    corrected[component] = 2
            elif class_id == 2:  # Right hippocampus  
                # Debería estar en el lado derecho (com[2] > center)
                if com[2] < center_w:
                    # Está mal, cambiar a left
                    corrected[component] = 1
    
    return corrected


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
    pred_post_vol = correct_laterality_post(pred_fine_vol.copy())
    
    # Calcular métricas una sola vez
    dice_l_base = metrics.dice_score(pred_base_vol, gt_mask, 1)
    dice_r_base = metrics.dice_score(pred_base_vol, gt_mask, 2)
    dice_l_fine = metrics.dice_score(pred_fine_vol, gt_mask, 1)
    dice_r_fine = metrics.dice_score(pred_fine_vol, gt_mask, 2)
    dice_l_post = metrics.dice_score(pred_post_vol, gt_mask, 1)
    dice_r_post = metrics.dice_score(pred_post_vol, gt_mask, 2)
    
    # Función helper para crear frames de una vista
    def create_frames_for_view(view_axis, view_name):
        frames = []
        
        if view_axis == 0:  # Axial
            num_slices = img_vol.shape[0]
            step = max(1, num_slices // 30)  # ~30 frames max
            
            for z in range(0, num_slices, step):
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                
                # Primera fila
                axes[0,0].imshow(img_vol[z], cmap='gray')
                axes[0,0].set_title(f'Input ({view_name} z={z})')
                axes[0,0].axis('off')
                
                axes[0,1].imshow(gt_mask[z], cmap='viridis', vmin=0, vmax=2)
                axes[0,1].set_title('Ground Truth')
                axes[0,1].axis('off')
                
                axes[0,2].imshow(pred_base_vol[z], cmap='viridis', vmin=0, vmax=2)
                axes[0,2].set_title('Base Prediction')
                axes[0,2].axis('off')
                
                # Segunda fila
                axes[1,0].imshow(pred_fine_vol[z], cmap='viridis', vmin=0, vmax=2)
                axes[1,0].set_title('Fine Prediction')
                axes[1,0].axis('off')
                
                axes[1,1].imshow(pred_post_vol[z], cmap='viridis', vmin=0, vmax=2)
                axes[1,1].set_title('Post-processed')
                axes[1,1].axis('off')
                
                # Overlay
                axes[1,2].imshow(img_vol[z], cmap='gray')
                masked_post = np.ma.masked_where(pred_post_vol[z] == 0, pred_post_vol[z])
                axes[1,2].imshow(masked_post, cmap='jet', alpha=0.5, vmin=1, vmax=2)
                axes[1,2].set_title('Overlay')
                axes[1,2].axis('off')
                
                title = f'{phase.upper()} - Epoch {epoch} - {view_name} View\n'
                title += f'Base: L={dice_l_base:.3f} R={dice_r_base:.3f} | '
                title += f'Fine: L={dice_l_fine:.3f} R={dice_r_fine:.3f} | '
                title += f'Post: L={dice_l_post:.3f} R={dice_r_post:.3f}'
                
                plt.suptitle(title, fontsize=14)
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                
                # Convertir a imagen
                fig.canvas.draw()
                image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
                image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                frames.append(image)  # AQUÍ estaba el problema - debe estar dentro del loop
                plt.close()
                
        elif view_axis == 1:  # Sagital
            num_slices = img_vol.shape[1]
            step = max(1, num_slices // 30)
            
            for x_idx in range(0, num_slices, step):
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                
                axes[0,0].imshow(img_vol[:, x_idx, :], cmap='gray')
                axes[0,0].set_title(f'Input ({view_name} x={x_idx})')
                axes[0,0].axis('off')
                
                axes[0,1].imshow(gt_mask[:, x_idx, :], cmap='viridis', vmin=0, vmax=2)
                axes[0,1].set_title('Ground Truth')
                axes[0,1].axis('off')
                
                axes[0,2].imshow(pred_base_vol[:, x_idx, :], cmap='viridis', vmin=0, vmax=2)
                axes[0,2].set_title('Base Prediction')
                axes[0,2].axis('off')
                
                axes[1,0].imshow(pred_fine_vol[:, x_idx, :], cmap='viridis', vmin=0, vmax=2)
                axes[1,0].set_title('Fine Prediction')
                axes[1,0].axis('off')
                
                axes[1,1].imshow(pred_post_vol[:, x_idx, :], cmap='viridis', vmin=0, vmax=2)
                axes[1,1].set_title('Post-processed')
                axes[1,1].axis('off')
                
                axes[1,2].imshow(img_vol[:, x_idx, :], cmap='gray')
                masked_post = np.ma.masked_where(pred_post_vol[:, x_idx, :] == 0, pred_post_vol[:, x_idx, :])
                axes[1,2].imshow(masked_post, cmap='jet', alpha=0.5, vmin=1, vmax=2)
                axes[1,2].set_title('Overlay')
                axes[1,2].axis('off')
                
                title = f'{phase.upper()} - Epoch {epoch} - {view_name} View\n'
                title += f'Base: L={dice_l_base:.3f} R={dice_r_base:.3f} | '
                title += f'Fine: L={dice_l_fine:.3f} R={dice_r_fine:.3f} | '
                title += f'Post: L={dice_l_post:.3f} R={dice_r_post:.3f}'
                
                plt.suptitle(title, fontsize=14)
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                
                fig.canvas.draw()
                image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
                image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                frames.append(image)
                plt.close()
                
        else:  # Coronal (view_axis == 2)
            num_slices = img_vol.shape[2]
            step = max(1, num_slices // 30)
            
            for y_idx in range(0, num_slices, step):
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                
                axes[0,0].imshow(img_vol[:, :, y_idx], cmap='gray')
                axes[0,0].set_title(f'Input ({view_name} y={y_idx})')
                axes[0,0].axis('off')
                
                axes[0,1].imshow(gt_mask[:, :, y_idx], cmap='viridis', vmin=0, vmax=2)
                axes[0,1].set_title('Ground Truth')
                axes[0,1].axis('off')
                
                axes[0,2].imshow(pred_base_vol[:, :, y_idx], cmap='viridis', vmin=0, vmax=2)
                axes[0,2].set_title('Base Prediction')
                axes[0,2].axis('off')
                
                axes[1,0].imshow(pred_fine_vol[:, :, y_idx], cmap='viridis', vmin=0, vmax=2)
                axes[1,0].set_title('Fine Prediction')
                axes[1,0].axis('off')
                
                axes[1,1].imshow(pred_post_vol[:, :, y_idx], cmap='viridis', vmin=0, vmax=2)
                axes[1,1].set_title('Post-processed')
                axes[1,1].axis('off')
                
                axes[1,2].imshow(img_vol[:, :, y_idx], cmap='gray')
                masked_post = np.ma.masked_where(pred_post_vol[:, :, y_idx] == 0, pred_post_vol[:, :, y_idx])
                axes[1,2].imshow(masked_post, cmap='jet', alpha=0.5, vmin=1, vmax=2)
                axes[1,2].set_title('Overlay')
                axes[1,2].axis('off')
                
                title = f'{phase.upper()} - Epoch {epoch} - {view_name} View\n'
                title += f'Base: L={dice_l_base:.3f} R={dice_r_base:.3f} | '
                title += f'Fine: L={dice_l_fine:.3f} R={dice_r_fine:.3f} | '
                title += f'Post: L={dice_l_post:.3f} R={dice_r_post:.3f}'
                
                plt.suptitle(title, fontsize=14)
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                
                fig.canvas.draw()
                image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
                image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                frames.append(image)
                plt.close()
        
        return frames
    
    # Crear animaciones para las 3 vistas
    gif_paths = []
    
    # Vista Axial
    frames_axial = create_frames_for_view(0, "Axial")
    if frames_axial:  # Solo guardar si hay frames
        gif_path_axial = os.path.join(vis_dir, f"{phase}_epoch{epoch:03d}_axial.gif")
        imageio.mimsave(gif_path_axial, frames_axial, duration=0.3)  # Más lento: 0.3s por frame
        gif_paths.append(gif_path_axial)
        print(f"  Axial GIF saved with {len(frames_axial)} frames")
    
    # Vista Sagital  
    frames_sagital = create_frames_for_view(1, "Sagital")
    if frames_sagital:
        gif_path_sagital = os.path.join(vis_dir, f"{phase}_epoch{epoch:03d}_sagital.gif")
        imageio.mimsave(gif_path_sagital, frames_sagital, duration=0.3)
        gif_paths.append(gif_path_sagital)
        print(f"  Sagital GIF saved with {len(frames_sagital)} frames")
    
    # Vista Coronal
    frames_coronal = create_frames_for_view(2, "Coronal")
    if frames_coronal:
        gif_path_coronal = os.path.join(vis_dir, f"{phase}_epoch{epoch:03d}_coronal.gif")
        imageio.mimsave(gif_path_coronal, frames_coronal, duration=0.3)
        gif_paths.append(gif_path_coronal)
        print(f"  Coronal GIF saved with {len(frames_coronal)} frames")
    
    # Animación combinada (opcional, más compacta)
    create_combined = True
    if create_combined:
        frames_combined = []
        num_steps = 15  # Reducir a 15 frames para que sea más manejable
        
        for step in range(num_steps):
            # Calcular índices
            z_idx = int(step * img_vol.shape[0] / num_steps)
            x_idx = int(step * img_vol.shape[1] / num_steps)
            y_idx = int(step * img_vol.shape[2] / num_steps)
            
            fig, axes = plt.subplots(3, 5, figsize=(18, 12))
            
            # Fila 1: Axial
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
            
            axes[0,4].imshow(pred_post_vol[z_idx], cmap='viridis', vmin=0, vmax=2)
            axes[0,4].set_title('Post')
            axes[0,4].axis('off')
            
            # Fila 2: Sagital
            axes[1,0].imshow(img_vol[:, x_idx, :], cmap='gray')
            axes[1,0].set_title(f'Sagital x={x_idx}')
            axes[1,0].axis('off')
            
            axes[1,1].imshow(gt_mask[:, x_idx, :], cmap='viridis', vmin=0, vmax=2)
            axes[1,1].axis('off')
            
            axes[1,2].imshow(pred_base_vol[:, x_idx, :], cmap='viridis', vmin=0, vmax=2)
            axes[1,2].axis('off')
            
            axes[1,3].imshow(pred_fine_vol[:, x_idx, :], cmap='viridis', vmin=0, vmax=2)
            axes[1,3].axis('off')
            
            axes[1,4].imshow(pred_post_vol[:, x_idx, :], cmap='viridis', vmin=0, vmax=2)
            axes[1,4].axis('off')
            
            # Fila 3: Coronal
            axes[2,0].imshow(img_vol[:, :, y_idx], cmap='gray')
            axes[2,0].set_title(f'Coronal y={y_idx}')
            axes[2,0].axis('off')
            
            axes[2,1].imshow(gt_mask[:, :, y_idx], cmap='viridis', vmin=0, vmax=2)
            axes[2,1].axis('off')
            
            axes[2,2].imshow(pred_base_vol[:, :, y_idx], cmap='viridis', vmin=0, vmax=2)
            axes[2,2].axis('off')
            
            axes[2,3].imshow(pred_fine_vol[:, :, y_idx], cmap='viridis', vmin=0, vmax=2)
            axes[2,3].axis('off')
            
            axes[2,4].imshow(pred_post_vol[:, :, y_idx], cmap='viridis', vmin=0, vmax=2)
            axes[2,4].axis('off')
            
            title = f'{phase.upper()} - Epoch {epoch} - All Views\n'
            title += f'Dice L: Base={dice_l_base:.3f} Fine={dice_l_fine:.3f} Post={dice_l_post:.3f} | '
            title += f'Dice R: Base={dice_r_base:.3f} Fine={dice_r_fine:.3f} Post={dice_r_post:.3f}'
            plt.suptitle(title, fontsize=14)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames_combined.append(image)
            plt.close()
        
        gif_path_combined = os.path.join(vis_dir, f"{phase}_epoch{epoch:03d}_combined.gif")
        imageio.mimsave(gif_path_combined, frames_combined, duration=0.4)  # Más lento aún para el combinado
        print(f"  Combined GIF saved with {len(frames_combined)} frames")
    
    return gif_paths if gif_paths else None


def save_training_visualization_with_post(x, y, pred_base, pred_fine, epoch, batch_idx, 
                                         exp_name, phase="train", fold=1):
    """Visualización con post-procesamiento"""
    
    vis_dir = f"images/{exp_name}/fold_{fold}/{phase}"
    os.makedirs(vis_dir, exist_ok=True)
    
    # Convertir a numpy
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
        
        best_idx = np.argmax(sums)
        center = mask.shape[axis] // 2
        
        if abs(best_idx - center) > mask.shape[axis] // 3:
            for offset in range(mask.shape[axis] // 3):
                if center + offset < len(sums) and sums[center + offset] > 0:
                    return center + offset
                if center - offset >= 0 and sums[center - offset] > 0:
                    return center - offset
        
        return best_idx
    
    gt_mask = y[0, 0] if y.ndim == 5 else y[0]
    
    best_axial = find_best_slice(gt_mask, 0)
    best_sagittal = find_best_slice(gt_mask, 1)
    best_coronal = find_best_slice(gt_mask, 2)
    
    fig, axes = plt.subplots(3, 7, figsize=(24, 12))
    
    img_vol = x[0, 0] if x.ndim == 5 else x[0]
    pred_base_vol = pred_base[0] if pred_base is not None else np.zeros_like(gt_mask)
    pred_fine_vol = pred_fine[0] if pred_fine is not None else np.zeros_like(gt_mask)
    pred_post_vol = correct_laterality_post(pred_fine_vol.copy())
    
    def plot_row(ax_row, img_slice, gt_slice, pred_base_slice, pred_fine_slice, pred_post_slice, view_name):
        ax_row[0].imshow(img_slice, cmap='gray')
        ax_row[0].set_title(f'{view_name} - Input')
        ax_row[0].axis('off')
        
        im = ax_row[1].imshow(gt_slice, cmap='viridis', vmin=0, vmax=2)
        ax_row[1].set_title('Ground Truth')
        ax_row[1].axis('off')
        
        ax_row[2].imshow(pred_base_slice, cmap='viridis', vmin=0, vmax=2)
        ax_row[2].set_title('Base Pred')
        ax_row[2].axis('off')
        
        ax_row[3].imshow(pred_fine_slice, cmap='viridis', vmin=0, vmax=2)
        ax_row[3].set_title('Fine Pred')
        ax_row[3].axis('off')
        
        ax_row[4].imshow(pred_post_slice, cmap='viridis', vmin=0, vmax=2)
        ax_row[4].set_title('Post-processed')
        ax_row[4].axis('off')
        
        ax_row[5].imshow(img_slice, cmap='gray')
        masked_gt = np.ma.masked_where(gt_slice == 0, gt_slice)
        ax_row[5].imshow(masked_gt, cmap='jet', alpha=0.5, vmin=1, vmax=2)
        ax_row[5].set_title('GT Overlay')
        ax_row[5].axis('off')
        
        ax_row[6].imshow(img_slice, cmap='gray')
        masked_post = np.ma.masked_where(pred_post_slice == 0, pred_post_slice)
        ax_row[6].imshow(masked_post, cmap='jet', alpha=0.5, vmin=1, vmax=2)
        ax_row[6].set_title('Post Overlay')
        ax_row[6].axis('off')
        
        return im
    
    im = plot_row(
        axes[0],
        img_vol[best_axial, :, :],
        gt_mask[best_axial, :, :],
        pred_base_vol[best_axial, :, :],
        pred_fine_vol[best_axial, :, :],
        pred_post_vol[best_axial, :, :],
        f'Axial (z={best_axial})'
    )
    
    plot_row(
        axes[1],
        img_vol[:, best_sagittal, :],
        gt_mask[:, best_sagittal, :],
        pred_base_vol[:, best_sagittal, :],
        pred_fine_vol[:, best_sagittal, :],
        pred_post_vol[:, best_sagittal, :],
        f'Sagittal (x={best_sagittal})'
    )
    
    plot_row(
        axes[2],
        img_vol[:, :, best_coronal],
        gt_mask[:, :, best_coronal],
        pred_base_vol[:, :, best_coronal],
        pred_fine_vol[:, :, best_coronal],
        pred_post_vol[:, :, best_coronal],
        f'Coronal (y={best_coronal})'
    )
    
    cbar_ax = fig.add_axes([0.92, 0.35, 0.02, 0.3])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Label (0: BG, 1: Left, 2: Right)', rotation=270, labelpad=20)
    
    dice_l_base = metrics.dice_score(pred_base_vol, gt_mask, 1) if pred_base is not None else 0
    dice_r_base = metrics.dice_score(pred_base_vol, gt_mask, 2) if pred_base is not None else 0
    dice_l_fine = metrics.dice_score(pred_fine_vol, gt_mask, 1) if pred_fine is not None else 0
    dice_r_fine = metrics.dice_score(pred_fine_vol, gt_mask, 2) if pred_fine is not None else 0
    dice_l_post = metrics.dice_score(pred_post_vol, gt_mask, 1)
    dice_r_post = metrics.dice_score(pred_post_vol, gt_mask, 2)
    
    title = f'{phase.upper()} - Epoch {epoch} - Batch {batch_idx}\n'
    title += f'Base: L={dice_l_base:.3f} R={dice_r_base:.3f} | '
    title += f'Fine: L={dice_l_fine:.3f} R={dice_r_fine:.3f} | '
    title += f'Post: L={dice_l_post:.3f} R={dice_r_post:.3f}'
    
    gt_voxels_l = np.sum(gt_mask == 1)
    gt_voxels_r = np.sum(gt_mask == 2)
    title += f'\nGT Voxels: L={gt_voxels_l} R={gt_voxels_r}'
    
    plt.suptitle(title, fontsize=12)
    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    
    filename = f"{phase}_epoch{epoch:03d}_batch{batch_idx:03d}.png"
    save_path = os.path.join(vis_dir, filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved: {save_path}")
    
    return save_path

def evaluate_base_model_with_vis(model, loader, device, loss_fn, epoch, exp_name, fold, use_wandb, run):
    """Evaluación base con visualización"""
    metrics = evaluate_base_model(model, loader, device, loss_fn)
    
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            
            # Solo visualizar si hay contenido
            if y.sum() > 100:
                logits = model(x)
                pred = torch.argmax(torch.softmax(logits, dim=1), dim=1)
                
                img_path = save_training_visualization_with_post(
                    x, y, pred.cpu().numpy(), None,
                    epoch, i, exp_name, phase="val_base", fold=fold
                )
                
                # Crear animación cada 10 epochs
                if epoch % 10 == 0:
                    gif_paths = create_volume_animation(
                        x, y, pred.cpu().numpy(), None,
                        epoch, exp_name, phase="val_base", fold=fold
                    )

                    for i,gif_path in enumerate(gif_paths):
                        print(f"Animation saved {i}: {gif_path}")
                        
                        if use_wandb and run is not None:
                            run.log({
                                f"val/animation_{i}": wandb.Video(gif_path),
                                "epoch": epoch
                            })
                
                if use_wandb and run is not None:
                    run.log({
                        f"val/visualization": wandb.Image(img_path),
                        "epoch": epoch
                    })
                break
    
    return metrics

def evaluate_cascade_with_vis(base_model, fine_model, loader, device, loss_fn,
                             roi_margin, roi_size_fine, dim, epoch, exp_name, 
                             fold, use_wandb, run):
    """Evaluación cascade con visualización"""
    metrics = evaluate_cascade_with_post(
        base_model, fine_model, loader, device,
        loss_fn, roi_margin, roi_size_fine, dim
    )
    
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
                
                # Visualización estática
                img_path = save_training_visualization_with_post(
                    x, y, pred_base.cpu().numpy(), pred_fine.cpu().numpy(),
                    epoch if epoch != 999 else "final", i, exp_name, phase=phase, fold=fold
                )
                
                # Animación cada 10 epochs o al final
                if epoch % 10 == 0 or epoch == 999:
                    gif_paths = create_volume_animation(
                        x, y, pred_base.cpu().numpy(), pred_fine.cpu().numpy(),
                        epoch if epoch != 999 else "final", exp_name, phase=phase, fold=fold
                    )

                    for i,gif_path in enumerate(gif_paths):
                        print(f"Animation saved {i}: {gif_path}")

                        
                        if use_wandb and run is not None:
                            run.log({
                                f"{phase}/animation_{i}": wandb.Video(gif_path),
                                "epoch": epoch
                            })
                
                if use_wandb and run is not None:
                    run.log({
                        f"{phase}/visualization": wandb.Image(img_path),
                        "epoch": epoch
                    })
                break
    
    return metrics

def evaluate_cascade_with_post(base_model, fine_model, loader, device, loss_fn, 
                               roi_margin, roi_size_fine, dim, use_post_process=True):
    """Evaluación del cascade con post-procesamiento"""
    base_model.eval()
    fine_model.eval()
    
    metrics_base = {"L": [], "R": []}
    metrics_fine = {"L": [], "R": []}
    metrics_post = {"L": [], "R": []}
    
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            
            logits_base = base_model(x)
            bboxes = get_roi_bbox_from_logits(logits_base, thr=0.2, margin=roi_margin)
            
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
            
            pred_base = torch.argmax(torch.softmax(logits_base, dim=1), dim=1).cpu().numpy()
            pred_fine = torch.argmax(torch.softmax(logits_fine, dim=1), dim=1).cpu().numpy()
            y_np = y.squeeze(1).cpu().numpy()
            
            pred_post = np.zeros_like(pred_fine)
            for i in range(pred_fine.shape[0]):
                if use_post_process:
                    pred_post[i] = correct_laterality_post(pred_fine[i])
                else:
                    pred_post[i] = pred_fine[i]
            
            for p_base, p_fine, p_post, g in zip(pred_base, pred_fine, pred_post, y_np):
                for cls, key in [(1, "L"), (2, "R")]:
                    dice_base = metrics.dice_score(p_base, g, cls)
                    dice_fine = metrics.dice_score(p_fine, g, cls)
                    dice_post = metrics.dice_score(p_post, g, cls)
                    
                    if not np.isnan(dice_base):
                        metrics_base[key].append(dice_base)
                    if not np.isnan(dice_fine):
                        metrics_fine[key].append(dice_fine)
                    if not np.isnan(dice_post):
                        metrics_post[key].append(dice_post)
    
    return {
        "dice_L": np.mean(metrics_base["L"]) if metrics_base["L"] else 0,
        "dice_R": np.mean(metrics_base["R"]) if metrics_base["R"] else 0,
        "dice_avg": np.mean(metrics_base["L"] + metrics_base["R"]) if metrics_base["L"] else 0,
        "dice_fine_L": np.mean(metrics_fine["L"]) if metrics_fine["L"] else 0,
        "dice_fine_R": np.mean(metrics_fine["R"]) if metrics_fine["R"] else 0,
        "dice_fine_avg": np.mean(metrics_fine["L"] + metrics_fine["R"]) if metrics_fine["L"] else 0,
        "dice_post_L": np.mean(metrics_post["L"]) if metrics_post["L"] else 0,
        "dice_post_R": np.mean(metrics_post["R"]) if metrics_post["R"] else 0,
        "dice_post_avg": np.mean(metrics_post["L"] + metrics_post["R"]) if metrics_post["L"] else 0,
    }

def train_cascade_sequential(df, args):
    """Entrenamiento secuencial con todas las visualizaciones"""
    
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
        
        if os.path.exists(os.path.join(fold_dir, "best_model.pth")):
            print("PATH EXIST! :",os.path.join(fold_dir, "best_model.pth"))
        else:
            for epoch in range(warmup_epochs):
                base_model.train()
                epoch_loss = 0
                
                for batch_idx, batch in enumerate(tqdm.tqdm(train_loader, desc=f"Base Epoch {epoch+1}/{warmup_epochs}")):
                    x = batch["image"].to(device)
                    y = batch["label"].to(device)
                    
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
                    
                    # Visualización cada 50 batches y cada 10 epochs
                    if batch_idx % 50 == 0 and epoch % 10 == 0:
                        with torch.no_grad():
                            pred = torch.argmax(torch.softmax(logits, dim=1), dim=1)
                            img_path = save_training_visualization_with_post(
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
                
                # Validation con visualización
                val_metrics = evaluate_base_model_with_vis(
                    base_model, val_loader, device, loss_fn_base,
                    epoch, args.experiment_name, fold+1, args.use_wandb, 
                    run_base if args.use_wandb else None
                )
                test_metrics = evaluate_base_model(base_model, test_loader, device, loss_fn_base)
                
                val_dice = val_metrics["dice_avg"]
                val_loss = val_metrics["loss"]
                test_loss = test_metrics["loss"]
                
                print(f"Base Epoch {epoch+1}: Train Loss={epoch_loss:.4f}, "
                    f"Val Loss={val_loss:.4f}, Test Loss={test_loss:.4f}, "
                    f"Val Dice={val_dice:.4f}, Test Dice={test_metrics['dice_avg']:.4f}")
                
                if args.use_wandb:
                    run_base.log({
                        "train/loss": epoch_loss,
                        "val/loss": val_loss,
                        "test/loss": test_loss,
                        "val/dice_avg": val_dice,
                        "val/dice_L": val_metrics["dice_L"],
                        "val/dice_R": val_metrics["dice_R"],
                        "test/dice_avg": test_metrics["dice_avg"],
                        "epoch": epoch
                    })
                
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
        
        base_model.load_state_dict(torch.load(os.path.join(fold_dir, "best_model.pth")))
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
        scheduler_fine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_fine, T_max=50, eta_min=1e-6
        )
        scaler_fine = GradScaler()
        
        best_dice_fine = 0.0
        roi_margin = 16
        roi_size_fine = (args.dim//2, args.dim//2, args.dim//2)
        
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
                
                with torch.no_grad():
                    base_logits = base_model(x)
                    bboxes = get_roi_bbox_from_logits(base_logits, thr=0.2, margin=roi_margin)
                    base_pred = torch.argmax(torch.softmax(base_logits, dim=1), dim=1)
                
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
                    
                    if batch_idx % 50 == 0 and epoch % 10 == 0:
                        img_path = save_training_visualization_with_post(
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
            
            # Validation con cascade y visualización
            val_metrics = evaluate_cascade_with_vis(
                base_model, fine_model, val_loader, device,
                loss_fn_base, roi_margin, roi_size_fine, args.dim,
                epoch, args.experiment_name, fold+1, args.use_wandb, 
                run_fine if args.use_wandb else None
            )
            
            val_dice_fine = val_metrics["dice_fine_avg"]
            val_dice_post = val_metrics["dice_post_avg"]
            
            print(f"Fine Epoch {epoch+1}: Loss={epoch_loss:.4f}, "
                  f"Fine Dice={val_dice_fine:.4f}, Post Dice={val_dice_post:.4f}")
            
            if args.use_wandb:
                run_fine.log({
                    "train/loss_fine": epoch_loss,
                    "val/dice_fine_avg": val_dice_fine,
                    "val/dice_fine_L": val_metrics["dice_fine_L"],
                    "val/dice_fine_R": val_metrics["dice_fine_R"],
                    "val/dice_post_avg": val_dice_post,
                    "val/dice_post_L": val_metrics["dice_post_L"],
                    "val/dice_post_R": val_metrics["dice_post_R"],
                    "epoch": epoch
                })
            
            if val_dice_post > best_dice_fine:
                best_dice_fine = val_dice_post
                torch.save(fine_model.state_dict(), os.path.join(fold_dir, "best_fine_model.pth"))
                print(f"✓ Saved best fine model: Post Dice={best_dice_fine:.4f}")
        
        if args.use_wandb:
            run_fine.finish()
        
        # Evaluación final con visualización
        print(f"\n>>> Evaluación final Fold {fold+1}")
        base_model.load_state_dict(torch.load(os.path.join(fold_dir, "best_model.pth")))
        fine_model.load_state_dict(torch.load(os.path.join(fold_dir, "best_fine_model.pth")))
        
        final_metrics = evaluate_cascade_with_vis(
            base_model, fine_model, test_loader, device,
            loss_fn_base, roi_margin, roi_size_fine, args.dim,
            999, args.experiment_name, fold+1, False, None
        )
        
        results["test_best"].append(final_metrics)
        print(f"Final Test Dice: Base={final_metrics['dice_avg']:.4f}, "
              f"Fine={final_metrics['dice_fine_avg']:.4f}, "
              f"Post={final_metrics['dice_post_avg']:.4f}")
    
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
    parser.add_argument('--loss_function', type=str, default="dice_focal_spatial")
    parser.add_argument('--use_mixup', type=int, default=0)
    parser.add_argument('--use_wandb', type=int, default=1)
    
    args = parser.parse_args()
    load_dotenv()
    
    df = pd.read_csv("results/preprocessed_data/task2/df_train_hipp.csv")
    
    results = train_cascade_sequential(df, args)
    print("\nFinal results:", results)


"""

# Comando con loss espacial y augmentaciones corregidas
python training.py \
  --model_name=eff-b2 \
  --device=cuda:5 \
  --root_dir=/data/cristian/projects/med_data/rise-miccai/task-2/3d_models/predictions/eff-b2-cascade-focal_spatial_192/fold_models \
  --num_epochs=150 \
  --num_folds=5 \
  --use_mixup=0 \
  --experiment_name=eff-b2-cascade-focal_spatial_192 \
  --lr=1e-4 \
  --weight_decay=1e-5 \
  --loss_function=dice_focal_spatial \
  --dim=192 \
  --batch_size=2 \
  --use_wandb=1


  python training.py \
  --model_name=eff-b2 \
  --device=cuda:3 \
  --root_dir=/data/cristian/projects/med_data/rise-miccai/task-2/3d_models/predictions/eff-b2-cascade-focal_spatial_192/fold_models \
  --num_epochs=150 \
  --num_folds=5 \
  --use_mixup=0 \
  --experiment_name=eff-b2-cascade-focal_spatial_192_Pruebis \
  --lr=1e-4 \
  --weight_decay=1e-5 \
  --loss_function=dice_focal_spatial \
  --dim=192 \
  --batch_size=2 \
  --use_wandb=1


  python training.py \
  --model_name=eff-b2 \
  --device=cuda:1 \
  --root_dir=/data/cristian/projects/med_data/rise-miccai/task-2/3d_models/predictions/eff-b2-cascade-focal_tversky_192/fold_models \
  --num_epochs=150 \
  --num_folds=5 \
  --use_mixup=0 \
  --experiment_name=eff-b2-cascade-focal_tversky_192 \
  --lr=1e-4 \
  --weight_decay=1e-5 \
  --loss_function=focal_tversky \
  --dim=192 \
  --batch_size=2 \
  --use_wandb=1

  """

pass