# =========================
# Inference CASCADE 3D con TTA Robusto - FIXED
# =========================
import os, re, gc
import numpy as np
import nibabel as nib
import pandas as pd
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
import scipy.ndimage as ndimage
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from monai.data import Dataset
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    ScaleIntensityRanged, EnsureTyped, CropForegroundd, Resized
)

import sys
sys.path.append("../")
from src.models import create_model
from src.cascade_utils import (
    get_roi_bbox_from_logits,
    crop_to_bbox,
    resize_volume,
)

# ---------- Configuración ----------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_CLASSES = 3

def remove_small_components(prediction, min_size=100):
    """
    Elimina componentes pequeños (ruido) de la predicción.
    """
    cleaned = np.zeros_like(prediction)
    
    for class_id in [1, 2]:
        class_mask = (prediction == class_id)
        if not class_mask.any():
            continue
            
        labeled, n_features = ndimage.label(class_mask)
        
        # Mantener solo componentes grandes
        for comp_id in range(1, n_features + 1):
            component = (labeled == comp_id)
            if component.sum() >= min_size:
                cleaned[component] = class_id
    
    return cleaned

def apply_morphological_operations(prediction, iterations=1):
    """
    Aplica operaciones morfológicas para suavizar la segmentación.
    """
    processed = prediction.copy()
    
    for class_id in [1, 2]:
        class_mask = (prediction == class_id).astype(np.uint8)
        if not class_mask.any():
            continue
        
        # Closing para cerrar huecos
        struct = ndimage.generate_binary_structure(3, 1)
        class_mask = ndimage.binary_closing(class_mask, structure=struct, iterations=iterations)
        
        # Opening para eliminar protuberancias pequeñas
        class_mask = ndimage.binary_opening(class_mask, structure=struct, iterations=iterations)
        
        # Actualizar predicción
        processed[class_mask > 0] = class_id
    
    return processed

# ---------- Transforms ----------
def get_test_transforms(spacing, spatial_size):
    """Transforms para test - sin augmentation"""
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"], 
            a_min=0.0, a_max=16.0,
            b_min=0.0, b_max=1.0, 
            clip=True
        ),
        CropForegroundd(
            keys=["image"], 
            source_key="image",
            margin=20,
            allow_smaller=True
        ),
        Spacingd(keys=["image"], pixdim=spacing, mode="bilinear"),
        Resized(keys=["image"], spatial_size=spatial_size, mode="trilinear"),
        EnsureTyped(keys=["image"], track_meta=True),
    ])

# ---------- Dataset ----------
class MRIDataset3DTest(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        data = []
        for _, r in df.iterrows():
            item = {
                "image": r["filepath"],
                "image_path": r["filepath"]
            }
            if "ID" in r:
                item["ID"] = r["ID"]
            data.append(item)
        super().__init__(data, transform=transform)

# ---------- Carga de modelos ----------
def load_cascade_models(models_root, model_name, device):
    """Carga modelos base y fine de todos los folds"""
    fold_dirs = []
    
    # Buscar directorios de folds
    for d in sorted(os.listdir(models_root)):
        p = os.path.join(models_root, d)
        if os.path.isdir(p) and d.lower().startswith("fold_"):
            base_path = os.path.join(p, "best_model.pth")
            fine_path = os.path.join(p, "best_fine_model.pth")
            
            if os.path.exists(base_path) and os.path.exists(fine_path):
                fold_dirs.append(p)
            else:
                print(f"⚠ Warning: Missing models in {p}")
            break

    if len(fold_dirs) == 0:
        raise ValueError(f"No valid fold directories found in {models_root}")
    
    print(f"Found {len(fold_dirs)} valid folds in {models_root}")
    
    base_models = []
    fine_models = []
    
    for fd in fold_dirs:
        # Modelo base
        base_ckpt = os.path.join(fd, "best_model.pth")
        base_model = create_model(model_name, device).to(device)
        base_model.load_state_dict(torch.load(base_ckpt, map_location=device))
        base_model.eval()
        base_models.append(base_model)
        print(f"✓ Loaded base model: {base_ckpt}")
        
        # Modelo fine
        fine_ckpt = os.path.join(fd, "best_fine_model.pth")
        fine_model = create_model(model_name, device).to(device)
        fine_model.load_state_dict(torch.load(fine_ckpt, map_location=device))
        fine_model.eval()
        fine_models.append(fine_model)
        print(f"✓ Loaded fine model: {fine_ckpt}")
    
    return base_models, fine_models

# ---------- TTA con flips seguros ----------
def apply_tta_cascade(image, base_model, fine_model, roi_size, sw_batch, 
                      roi_margin, roi_size_fine, device, use_tta=True, tta_flips=None, verbose=False):
    """
    Aplica TTA con flips que preservan lateralidad.
    
    Args:
        tta_flips: Lista de configuraciones de flip. Por defecto usa flips seguros.
        verbose: Si True, imprime información de debug
    """
    
    if use_tta:
        if tta_flips is None:
            # Configuraciones de TTA que PRESERVAN lateralidad L/R
            tta_configs = [
                (0, 0, 0),  # Original - sin flips
                (0, 1, 0),  # Flip coronal (anterior-posterior) - SEGURO
                (0, 0, 1),  # Flip axial (superior-inferior) - SEGURO
                #(0, 1, 1),  # Flip coronal + axial - SEGURO
            ]
        else:
            tta_configs = tta_flips
    else:
        tta_configs = [(0, 0, 0)]
    
    all_probs = []
    
    for fx, fy, fz in tta_configs:
        # Aplicar flips a la imagen
        img_flipped = image.clone()
        if fx: 
            img_flipped = img_flipped.flip(2)  # Flip sagital - EVITAR para L/R
        if fy: 
            img_flipped = img_flipped.flip(3)  # Flip coronal - SEGURO
        if fz: 
            img_flipped = img_flipped.flip(4)  # Flip axial - SEGURO
        
        # BASE: Sliding window inference
        with autocast():
            logits_base = sliding_window_inference(
                inputs=img_flipped,
                roi_size=roi_size,
                sw_batch_size=sw_batch,
                predictor=base_model,
                overlap=0.25,
                mode="gaussian",
                sw_device=device,
                device=device
            )
        print("logits_base:",logits_base.min(),logits_base.max())

        # Detectar ROIs con threshold adaptativo
        bboxes = None
        threshold_used = None
        for thr in [0.1, 0.05, 0.02]:
            bboxes = get_roi_bbox_from_logits(logits_base, thr=thr, margin=roi_margin)
            if bboxes and bboxes[0] is not None:
                threshold_used = thr
                break
        
        if verbose:
            print(f"  ROI threshold used: {threshold_used}")
            if bboxes and bboxes[0] is not None:
                z0, y0, x0, z1, y1, x1 = bboxes[0]
                print(f"  ROI size: {z1-z0}x{y1-y0}x{x1-x0}")
        
        # Si no encuentra ROI, usar imagen completa
        if bboxes is None or bboxes[0] is None:
            if verbose:
                print("  Warning: No ROI found, using full image")
            B, C, D, H, W = img_flipped.shape
            bboxes = [(0, 0, 0, D, H, W)]
        
        # FINE: Procesar cada ROI
        logits_fine = torch.zeros_like(logits_base)
        
        for i, bb in enumerate(bboxes):
            if bb is None:
                continue
                
            # Crop ROI
            img_roi = crop_to_bbox(img_flipped[i:i+1], bb)
            
            # Verificar tamaño mínimo del ROI
            if img_roi.numel() < 100:
                continue
            print("img_roi:",img_roi.shape)
            # Resize si es necesario
            if img_roi.shape[2:] != roi_size_fine:
                img_roi = resize_volume(img_roi, roi_size_fine, mode="trilinear")
            
            # Predicción fine con sliding window si el ROI es grande
            with autocast():
                if roi_size_fine[0] > 128:
                    print("1 PREDICT")
                    logits_roi = sliding_window_inference(
                        inputs=img_roi,
                        roi_size=roi_size,#(96, 96, 96),
                        sw_batch_size=sw_batch,
                        predictor=fine_model,
                        overlap=0.25,
                        mode="gaussian"
                    )
                else:
                    print("2 PREDICT")
                    logits_roi = fine_model(img_roi)
            aux = torch.argmax(torch.softmax(logits_roi, dim=1), dim=1)
            print("aux:",aux.min(),aux.max())#,aux.std())
            # Resize back a tamaño original del ROI
            z0, y0, x0, z1, y1, x1 = bb
            target_shape = (z1-z0, y1-y0, x1-x0)
            
            if logits_roi.shape[2:] != target_shape:
                logits_roi_resized = resize_volume(
                    logits_roi, 
                    target_shape, 
                    mode="trilinear"
                )
            else:
                logits_roi_resized = logits_roi
            
            print("logits_roi_resized:",logits_roi_resized.min(),logits_roi_resized.max())
            # Asegurar dimensiones correctas
            if logits_roi_resized.dim() == 5:
                logits_roi_resized = logits_roi_resized[0]
            
            # Colocar en la posición correcta
            logits_fine[i, :, z0:z1, y0:y1, x0:x1] = logits_roi_resized
        
        # Deshacer flips en logits
        if fz: 
            logits_fine = logits_fine.flip(4)
        if fy: 
            logits_fine = logits_fine.flip(3)
            # Si se flipeó coronalmente, intercambiar canales L/R
            #logits_fine = logits_fine[:, [0, 2, 1], ...]  
        if fx: 
            logits_fine = logits_fine.flip(2)
            # Si se flipeó sagitalmente, intercambiar canales L/R
            #logits_fine = logits_fine[:, [0, 2, 1], ...]
        
        # Softmax y guardar
        probs = torch.softmax(logits_fine, dim=1)
        all_probs.append(probs)
    
    # Promediar probabilidades
    if len(all_probs) > 0:
        print("promedio 1")
        avg_probs = torch.stack(all_probs).mean(dim=0)
    else:
        print("promedio 2")
        # Fallback a solo background si algo falla
        avg_probs = torch.zeros_like(logits_base)
        avg_probs[:, 0] = 1.0
    
    print("avg_probs:",avg_probs.min(),avg_probs.max())
    return avg_probs

def inverse_transform_prediction(probs, orig_shape, test_tfm, img_meta, device):
    """
    Invierte las transformaciones aplicadas a la predicción para volver al espacio original.
    
    Args:
        probs: Probabilidades predichas [C, D, H, W]
        orig_shape: Forma original de la imagen
        test_tfm: Transformaciones aplicadas
        img_meta: Metadata de la imagen
        device: Device
        
    Returns:
        Predicción en el espacio original
    """
    import torch.nn.functional as F
    from monai.transforms import Spacing, Resize
    
    # Asegurar que probs esté en CPU
    if probs.is_cuda:
        probs = probs.cpu()
    
    # Debug info
    print(f"    Probs shape before inverse: {probs.shape}")
    print(f"    Original image shape: {orig_shape}")
    
    # Si las dimensiones ya coinciden, no hacer nada
    if probs.shape[1:] == orig_shape:
        print("    Shapes already match, no inverse needed")
        return probs.numpy()
    
    # Método 1: Resize directo a la forma original
    # Agregar batch dimension si no está presente
    if probs.dim() == 4:
        probs = probs.unsqueeze(0)  # [1, C, D, H, W]
    
    # Resize usando interpolación trilineal
    probs_resized = F.interpolate(
        probs,
        size=orig_shape,
        mode='trilinear',
        align_corners=False
    )
    
    # Quitar batch dimension y convertir a numpy
    probs_resized = probs_resized.squeeze(0).numpy()
    
    print(f"    Probs shape after inverse: {probs_resized.shape}")
    
    return probs_resized

# ---------- Inference principal ----------
def run_cascade_inference(
    df_test_csv,
    models_root,
    output_dir,
    model_name="eff-b2",
    dim=192,
    use_tta=True,
    use_post_process=True,
    save_per_fold=False,
    batch_size=1,
    submission_format="LISAHF",  # or "standard"
    debug_mode=False
):
    """
    Inferencia completa con cascade, TTA y post-procesamiento.
    
    Args:
        submission_format: "LISAHF" para el formato de submission del challenge,
                          "standard" para nombres estándar.
        debug_mode: Si True, guarda información de debug
    """
    
    # Crear directorios de salida
    os.makedirs(output_dir, exist_ok=True)
    if save_per_fold:
        os.makedirs(os.path.join(output_dir, "per_fold"), exist_ok=True)
    if debug_mode:
        os.makedirs(os.path.join(output_dir, "debug"), exist_ok=True)
    
    # Configuración
    SPACING = (1.0, 1.0, 1.0)
    SPATIAL_SIZE = (dim, dim, dim)
    ROI_SIZE = SPATIAL_SIZE
    ROI_SIZE_FINE = SPATIAL_SIZE  # Mismo tamaño, no reducir
    SW_BATCH = 2
    ROI_MARGIN = 25
    
    # Cargar datos
    df_test = pd.read_csv(df_test_csv)
    print(f"Processing {len(df_test)} test cases")
    
    test_tfm = get_test_transforms(SPACING, SPATIAL_SIZE)
    test_ds = MRIDataset3DTest(df_test, transform=test_tfm)
    test_loader = DataLoader(
        test_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    # Cargar modelos
    try:
        base_models, fine_models = load_cascade_models(models_root, model_name, DEVICE)
    except Exception as e:
        print(f"Error loading models: {e}")
        return
    
    n_folds = len(base_models)
    print(f"Using {n_folds} folds for ensemble")
    
    # Tracking de predicciones
    predictions_log = []
    
    # Procesar cada caso
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_loader, desc="Processing cases")):
            try:
                # Cargar imagen original para guardar con affine correcto
                img_path = batch["image_path"][0] if batch_size == 1 else batch["image_path"][idx]
                nii = nib.load(img_path)
                orig_img = nii.get_fdata()
                orig_aff = nii.affine
                orig_header = nii.header
                orig_shape = orig_img.shape
                
                # Determinar ID para nombre de salida
                subject_id = batch.get("ID", [None])[0] if "ID" in batch else None
                
                if subject_id is None:
                    # Extraer ID del nombre del archivo
                    basename = os.path.basename(img_path)
                    # Buscar patrones como "sub-XXX" o números
                    import re
                    match = re.search(r'sub[-_]?(\d+)', basename)
                    if match:
                        subject_id = match.group(1)
                    else:
                        # Buscar cualquier secuencia de números
                        nums = re.findall(r'\d+', basename)
                        subject_id = nums[0] if nums else f"unknown_{idx}"
                
                # Generar nombre de salida según formato
                if submission_format == "LISAHF":
                    out_name = f"LISAHF{subject_id}segprediction.nii.gz"
                else:
                    out_name = f"sub-{subject_id}_pred.nii.gz"
                
                print(f"\n[{idx+1}/{len(test_ds)}] Processing: {os.path.basename(img_path)}")
                print(f"  Output name: {out_name}")
                print(f"  Original shape: {orig_shape}")
                
                # Imagen procesada
                img_tensor = batch["image"].to(DEVICE)
                img_meta = batch["image"].meta if hasattr(batch["image"], 'meta') else None
                
                print(f"  Processed shape: {img_tensor.shape}")
                
                # Ensemble sobre folds
                prob_sum = None
                fold_predictions = []
                
                for k, (base_model, fine_model) in enumerate(zip(base_models, fine_models), 1):
                    try:
                        # Aplicar cascade con TTA
                        # Verbose en el primer caso para debugging
                        verbose = (idx == 0)  # Verbose en el primer caso de todos los folds
                        probs_fold = apply_tta_cascade(
                            img_tensor, base_model, fine_model,
                            ROI_SIZE, SW_BATCH, ROI_MARGIN, ROI_SIZE_FINE,
                            DEVICE, use_tta=use_tta, verbose=verbose
                        )
                        
                        # Debug: verificar predicciones antes de invertir
                        probs_cpu = probs_fold.cpu()[0]  # Quitar batch dim [C, D, H, W]
                        pred_before = torch.argmax(probs_cpu, dim=0).numpy()
                        n_pred_before = np.sum(pred_before > 0)
                        print(f"  Fold {k}: Predictions before inverse: {n_pred_before} voxels")
                        
                        # Invertir transformaciones manualmente
                        probs_orig = inverse_transform_prediction(
                            probs_cpu, 
                            orig_shape, 
                            test_tfm, 
                            img_meta,
                            DEVICE
                        )
                        
                        # Debug: verificar predicciones después de invertir
                        pred_after = np.argmax(probs_orig, axis=0)
                        n_pred_after = np.sum(pred_after > 0)
                        print(f"  Fold {k}: Predictions after inverse: {n_pred_after} voxels")
                        
                        # Acumular probabilidades
                        prob_sum = probs_orig if prob_sum is None else (prob_sum + probs_orig)
                        
                        # Guardar predicción por fold (opcional)
                        if save_per_fold:
                            pred_fold = np.argmax(probs_orig, axis=0).astype(np.uint8)
                            
                            if use_post_process:
                                pred_fold = remove_small_components(pred_fold, min_size=100)
                                pred_fold = apply_morphological_operations(pred_fold, iterations=1)
                            
                            # Aplicar máscara cerebral
                            brain_mask = (orig_img > 0.5).astype(np.uint8)
                            pred_fold = (pred_fold * brain_mask).astype(np.uint8)
                            
                            fold_dir = os.path.join(output_dir, "per_fold", f"fold_{k}")
                            os.makedirs(fold_dir, exist_ok=True)
                            out_path_k = os.path.join(fold_dir, out_name)
                            
                            # Asegurar dtype uint8
                            nii_out = nib.Nifti1Image(pred_fold.astype(np.uint8), orig_aff, orig_header)
                            nii_out.set_data_dtype(np.uint8)
                            nib.save(nii_out, out_path_k)
                            
                            fold_predictions.append({
                                'fold': k,
                                'n_left': np.sum(pred_fold == 1),
                                'n_right': np.sum(pred_fold == 2)
                            })
                        
                        print(f"  Fold {k}/{n_folds} completed")
                        
                    except Exception as e:
                        print(f"  ⚠ Error in fold {k}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
                
                # Predicción final del ensemble
                if prob_sum is not None:
                    prob_avg = prob_sum / float(n_folds)
                    pred_final = np.argmax(prob_avg, axis=0).astype(np.uint8)
                    
                    print(f"  Ensemble prediction before post-processing: {np.sum(pred_final > 0)} voxels")
                    
                    # Debug: guardar predicción sin post-procesamiento
                    if debug_mode:
                        debug_path = os.path.join(output_dir, "debug", f"raw_{out_name}")
                        nii_debug = nib.Nifti1Image(pred_final.astype(np.uint8), orig_aff, orig_header)
                        nib.save(nii_debug, debug_path)
                        print(f"  Debug: Raw prediction saved to {debug_path}")
                    
                    # Post-procesamiento (sin corrección de lateralidad)
                    if use_post_process:
                        # Solo eliminar componentes pequeños y operaciones morfológicas
                        pred_final = remove_small_components(pred_final, min_size=100)
                        pred_final = apply_morphological_operations(pred_final, iterations=1)
                        print(f"  After post-processing: {np.sum(pred_final > 0)} voxels")
                    
                    # Aplicar máscara cerebral
                    brain_mask = (orig_img > 0.5).astype(np.uint8)
                    pred_final = (pred_final * brain_mask).astype(np.uint8)
                    print(f"  After brain mask: {np.sum(pred_final > 0)} voxels")
                    
                    # Asegurar dtype uint8 antes de guardar
                    pred_final = pred_final.astype(np.uint8)
                    
                    # Guardar predicción final
                    out_path = os.path.join(output_dir, out_name)
                    nii_out = nib.Nifti1Image(pred_final, orig_aff, orig_header)
                    nii_out.set_data_dtype(np.uint8)
                    nib.save(nii_out, out_path)
                    
                    # Estadísticas
                    n_left = np.sum(pred_final == 1)
                    n_right = np.sum(pred_final == 2)
                    total_voxels = n_left + n_right
                    
                    print(f"  ✓ Saved: {out_path}")
                    print(f"  Statistics: L={n_left:,} R={n_right:,} Total={total_voxels:,}")
                    
                    # Log para tracking
                    predictions_log.append({
                        'id': subject_id,
                        'filename': out_name,
                        'n_left': int(n_left),
                        'n_right': int(n_right),
                        'total_voxels': int(total_voxels),
                        'fold_predictions': fold_predictions
                    })
                else:
                    print(f"  ⚠ No valid predictions for {subject_id}")
                
                # Liberar memoria periódicamente
                if idx % 10 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"  ✗ Error processing case {idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Guardar log de predicciones
    if len(predictions_log) > 0:
        log_df = pd.DataFrame(predictions_log)
        log_path = os.path.join(output_dir, "predictions_log.csv")
        log_df.to_csv(log_path, index=False)
        print(f"\nPredictions log saved to: {log_path}")
        
        # Imprimir resumen
        print("\n" + "="*60)
        print("INFERENCE SUMMARY")
        print("="*60)
        print(f"Total cases processed: {len(predictions_log)}")
        print(f"Average left hippocampus voxels: {log_df['n_left'].mean():.0f}")
        print(f"Average right hippocampus voxels: {log_df['n_right'].mean():.0f}")
        print(f"Average total voxels: {log_df['total_voxels'].mean():.0f}")
        
        # Casos con problemas
        zero_cases = log_df[log_df['total_voxels'] == 0]
        if len(zero_cases) > 0:
            print(f"\n⚠ WARNING: {len(zero_cases)} cases with ZERO predictions:")
            for _, row in zero_cases.iterrows():
                print(f"  - {row['filename']}")
        
    print("\n✓ Inference completed!")

# ---------- Función de entrada simplificada ----------
def run_inference_simple(
    test_csv_path,
    models_path,
    output_path,
    model_name="eff-b2",
    image_size=192,
    enable_tta=True,
    enable_post_processing=True,
    debug=False
):
    """
    Función simplificada para correr inferencia.
    
    Args:
        test_csv_path: Path al CSV con los casos de test
        models_path: Path a la carpeta con los modelos (fold_1, fold_2, etc.)
        output_path: Path donde guardar las predicciones
        model_name: Nombre del modelo a usar
        image_size: Tamaño de la imagen (192 o 256)
        enable_tta: Usar Test Time Augmentation
        enable_post_processing: Aplicar post-procesamiento
        debug: Guardar información de debug
    """
    
    run_cascade_inference(
        df_test_csv=test_csv_path,
        models_root=models_path,
        output_dir=output_path,
        model_name=model_name,
        dim=image_size,
        use_tta=enable_tta,
        use_post_process=enable_post_processing,
        save_per_fold=False,  # Solo guardar predicción final
        batch_size=1,
        submission_format="LISAHF",  # Formato para submission
        debug_mode=debug
    )

# ---------- Ejecución ----------
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Cascade inference for hippocampus segmentation")
    parser.add_argument('--test_csv', type=str, required=True, help='Path to test CSV')
    parser.add_argument('--models_dir', type=str, required=True, help='Path to models directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to output directory')
    parser.add_argument('--model_name', type=str, default='eff-b2', help='Model architecture')
    parser.add_argument('--dim', type=int, default=192, help='Image dimension')
    parser.add_argument('--use_tta', type=int, default=1, help='Use TTA (0/1)')
    parser.add_argument('--use_post', type=int, default=1, help='Use post-processing (0/1)')
    parser.add_argument('--save_folds', type=int, default=0, help='Save per-fold predictions (0/1)')
    parser.add_argument('--format', type=str, default='LISAHF', choices=['LISAHF', 'standard'],
                       help='Output filename format')
    parser.add_argument('--debug', type=int, default=0, help='Enable debug mode (0/1)')
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("CASCADE INFERENCE - FIXED VERSION")
    print(f"{'='*60}")
    print(f"Test CSV: {args.test_csv}")
    print(f"Models: {args.models_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Model: {args.model_name}")
    print(f"Dimension: {args.dim}x{args.dim}x{args.dim}")
    print(f"TTA: {'Enabled' if args.use_tta else 'Disabled'}")
    print(f"Post-processing: {'Enabled' if args.use_post else 'Disabled'}")
    print(f"Format: {args.format}")
    print(f"Debug: {'Enabled' if args.debug else 'Disabled'}")
    print(f"{'='*60}\n")
    
    run_cascade_inference(
        df_test_csv=args.test_csv,
        models_root=args.models_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        dim=args.dim,
        use_tta=bool(args.use_tta),
        use_post_process=bool(args.use_post),
        save_per_fold=bool(args.save_folds),
        submission_format=args.format,
        debug_mode=bool(args.debug)
    )

"""

python inference_cascade.py \
  --test_csv ./results/preprocessed_data/task2/df_test_hipp.csv \
  --models_dir /data/cristian/projects/med_data/rise-miccai/task-2/3d_models/predictions/eff-dice_focal_spatial/fold_models \
  --output_dir /data/cristian/projects/med_data/rise-miccai/task-2/3d_models/predictions/eff-dice_focal_spatial/submissions_fixed_v2 \
  --model_name eff-b2 \
  --dim 192 \
  --use_tta 0 \
  --use_post 1 \
  --save_folds 0 \
  --format LISAHF \
  --debug 1

"""
pass