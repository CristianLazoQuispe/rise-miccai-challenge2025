# =========================
# Inference CASCADE 3D con TTA
# =========================
import os, re
import numpy as np
import nibabel as nib
import pandas as pd
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
import scipy.ndimage as ndimage

from monai.data import Dataset
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    ScaleIntensityRanged, EnsureTyped, CropForegroundd, Resized, Invertd
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

def is_positive_background(img):
    """Returns boolean where positive values are True"""
    return img > 0.5

def correct_laterality_post(prediction):
    """Corrige etiquetas L/R basándose en posición anatómica."""
    if prediction.max() == 0:
        return prediction
    
    corrected = prediction.copy()
    center_w = prediction.shape[2] / 2
    
    # Procesar cada clase por separado
    for class_id in [1, 2]:
        class_mask = (prediction == class_id)
        if not class_mask.any():
            continue
            
        labeled, n_features = ndimage.label(class_mask)
        
        for comp_id in range(1, n_features + 1):
            component = (labeled == comp_id)
            com = ndimage.center_of_mass(component)
            
            if class_id == 1:  # Left hippocampus
                if com[2] > center_w:  # Está en lado derecho, cambiar a right
                    corrected[component] = 2
            elif class_id == 2:  # Right hippocampus  
                if com[2] < center_w:  # Está en lado izquierdo, cambiar a left
                    corrected[component] = 1
    
    return corrected

# ---------- Transforms ----------
def get_test_transforms(spacing, spatial_size):
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
    for d in sorted(os.listdir(models_root)):
        p = os.path.join(models_root, d)
        if os.path.isdir(p) and d.lower().startswith("fold_"):
            if os.path.exists(os.path.join(p, "best_model.pth")) and \
               os.path.exists(os.path.join(p, "best_fine_model.pth")):
                fold_dirs.append(p)
    
    print(f"Encontrados {len(fold_dirs)} folds en {models_root}")
    
    base_models = []
    fine_models = []
    
    for fd in fold_dirs:
        # Modelo base
        base_ckpt = os.path.join(fd, "best_model.pth")
        base_model = create_model(model_name, device).to(device)
        base_model.load_state_dict(torch.load(base_ckpt, map_location=device))
        base_model.eval()
        base_models.append(base_model)
        print(f"✓ Cargado base model: {base_ckpt}")
        
        # Modelo fine
        fine_ckpt = os.path.join(fd, "best_fine_model.pth")
        fine_model = create_model(model_name, device).to(device)
        fine_model.load_state_dict(torch.load(fine_ckpt, map_location=device))
        fine_model.eval()
        fine_models.append(fine_model)
        print(f"✓ Cargado fine model: {fine_ckpt}")
    
    return base_models, fine_models

# ---------- TTA con flips ----------
def apply_tta_cascade(image, base_model, fine_model, roi_size, sw_batch, 
                      roi_margin, roi_size_fine, device, use_tta=True):
    """Aplica TTA con flips y cascade"""
    
    if use_tta:
        # Flips conservadores que preservan anatomía
        tta_configs = [
            (0, 0, 0),  # Original
            (0, 1, 0),  # Flip coronal (anterior-posterior)
            (0, 0, 1),  # Flip axial (superior-inferior)
            (0, 1, 1),  # Flip coronal + axial
        ]
    else:
        tta_configs = [(0, 0, 0)]
    
    all_probs = []
    
    for fx, fy, fz in tta_configs:
        # Aplicar flips a la imagen
        img_flipped = image.clone()
        if fx: img_flipped = img_flipped.flip(2)  # NO flip sagital (preserva L/R)
        if fy: img_flipped = img_flipped.flip(3)  # Flip coronal
        if fz: img_flipped = img_flipped.flip(4)  # Flip axial
        
        # BASE: Sliding window inference
        with autocast():
            logits_base = sliding_window_inference(
                inputs=img_flipped,
                roi_size=roi_size,
                sw_batch_size=sw_batch,
                predictor=base_model,
                overlap=0.25,
                mode="gaussian",
            )
        
        # Detectar ROIs
        bboxes = get_roi_bbox_from_logits(logits_base, thr=0.2, margin=roi_margin)
        
        # FINE: Procesar cada ROI
        logits_fine = torch.zeros_like(logits_base)
        for i, bb in enumerate(bboxes):
            if bb is None:
                continue
                
            # Crop ROI
            img_roi = crop_to_bbox(img_flipped[i:i+1], bb)
            img_roi = resize_volume(img_roi, roi_size_fine, mode="trilinear")
            
            # Predicción fine
            with autocast():
                logits_roi = fine_model(img_roi)
            
            # Resize back y colocar en posición
            z0, y0, x0, z1, y1, x1 = bb
            logits_roi_up = resize_volume(
                logits_roi, 
                (z1-z0, y1-y0, x1-x0), 
                mode="trilinear"
            )[0]
            logits_fine[i:i+1, :, z0:z1, y0:y1, x0:x1] = logits_roi_up
        
        # Deshacer flips en logits
        if fz: logits_fine = logits_fine.flip(4)
        if fy: 
            logits_fine = logits_fine.flip(3)
            # Intercambiar canales L/R si se flipó coronalmente
            logits_fine = logits_fine[:, [0, 2, 1], ...]  
        if fx: logits_fine = logits_fine.flip(2)
        
        # Softmax y guardar
        probs = torch.softmax(logits_fine, dim=1)
        all_probs.append(probs)
    
    # Promediar probabilidades
    avg_probs = torch.stack(all_probs).mean(dim=0)
    return avg_probs

# ---------- Inference principal ----------
def run_cascade_inference(
    df_test_csv,
    models_root,
    output_dir,
    model_name="eff-b2",
    dim=192,
    use_tta=True,
    use_post_process=True,
    save_per_fold=False
):
    """
    Inferencia completa con cascade, TTA y post-procesamiento
    """
    
    os.makedirs(output_dir, exist_ok=True)
    if save_per_fold:
        os.makedirs(os.path.join(output_dir, "per_fold"), exist_ok=True)
    
    # Configuración
    SPACING = (1.0, 1.0, 1.0)
    SPATIAL_SIZE = (dim, dim, dim)
    ROI_SIZE = SPATIAL_SIZE
    ROI_SIZE_FINE = (dim//2, dim//2, dim//2)
    SW_BATCH = 2
    ROI_MARGIN = 16
    
    # Cargar datos
    df_test = pd.read_csv(df_test_csv)
    print(f"Procesando {len(df_test)} casos de test")
    
    test_tfm = get_test_transforms(SPACING, SPATIAL_SIZE)
    test_ds = MRIDataset3DTest(df_test, transform=test_tfm)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4)
    
    # Cargar modelos
    base_models, fine_models = load_cascade_models(models_root, model_name, DEVICE)
    n_folds = len(base_models)
    print(f"Usando {n_folds} folds para ensemble")
    
    # Procesar cada caso
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            # Cargar imagen original para guardar con affine correcto
            img_path = batch["image_path"][0]
            nii = nib.load(img_path)
            orig_img = nii.get_fdata()
            orig_aff = nii.affine
            
            # ID para nombre de salida
            subject_id = batch.get("ID", [None])[0]
            if subject_id is None:
                nums = re.findall(r"\d+", os.path.basename(img_path))
                subject_id = nums[-1] if nums else "unknown"
            out_name = f"LISAHF{subject_id}segprediction.nii.gz"
            
            # Imagen procesada
            img_tensor = batch["image"].to(DEVICE)
            img_meta = batch["image"].meta
            
            # Ensemble sobre folds
            prob_sum = None
            
            for k, (base_model, fine_model) in enumerate(zip(base_models, fine_models), 1):
                print(f"  Fold {k}/{n_folds}...")
                
                # Aplicar cascade con TTA
                probs_fold = apply_tta_cascade(
                    img_tensor, base_model, fine_model,
                    ROI_SIZE, SW_BATCH, ROI_MARGIN, ROI_SIZE_FINE,
                    DEVICE, use_tta=use_tta
                )
                
                # Invertir transformaciones para volver al espacio original
                probs_cpu = probs_fold.cpu()[0]  # Quitar batch dim
                
                # Crear diccionario para Invertd
                elem = {
                    "image": batch["image"][0],
                    "image_meta_dict": img_meta[0] if isinstance(img_meta, list) else img_meta,
                    "pred": probs_cpu,
                    "pred_meta_dict": img_meta[0] if isinstance(img_meta, list) else img_meta,
                }
                
                # Aplicar Invertd
                inv = Invertd(
                    keys="pred",
                    transform=test_tfm,
                    orig_keys="image",
                    meta_keys="pred_meta_dict",
                    orig_meta_keys="image_meta_dict",
                    nearest_interp=False,  # Lineal para probabilidades
                    to_tensor=True,
                )
                elem = inv(elem)
                
                # Probabilidades en espacio original
                probs_orig = elem["pred"].numpy()
                prob_sum = probs_orig if prob_sum is None else (prob_sum + probs_orig)
                
                # Guardar por fold (opcional)
                if save_per_fold:
                    pred_fold = np.argmax(probs_orig, axis=0).astype(np.uint8)
                    if use_post_process:
                        pred_fold = correct_laterality_post(pred_fold)
                    
                    # Aplicar máscara cerebral
                    brain_mask = (orig_img > 0.5).astype(np.uint8)
                    pred_fold = (pred_fold * brain_mask).astype(np.uint8)
                    
                    fold_dir = os.path.join(output_dir, "per_fold", f"fold_{k}")
                    os.makedirs(fold_dir, exist_ok=True)
                    out_path_k = os.path.join(fold_dir, out_name)
                    nib.save(nib.Nifti1Image(pred_fold, orig_aff), out_path_k)
            
            # Predicción final del ensemble
            prob_avg = prob_sum / float(n_folds)
            pred_final = np.argmax(prob_avg, axis=0).astype(np.uint8)
            
            # Post-procesamiento
            if use_post_process:
                pred_final = correct_laterality_post(pred_final)
            
            # Aplicar máscara cerebral
            brain_mask = (orig_img > 0.5).astype(np.uint8)
            pred_final = (pred_final * brain_mask).astype(np.uint8)
            
            # Guardar predicción final
            out_path = os.path.join(output_dir, out_name)
            nib.save(nib.Nifti1Image(pred_final, orig_aff), out_path)
            print(f"[{idx+1}/{len(test_ds)}] Guardado: {out_path}")
            
            # Estadísticas
            n_left = np.sum(pred_final == 1)
            n_right = np.sum(pred_final == 2)
            print(f"  Voxels: L={n_left}, R={n_right}")

# ---------- Ejecución ----------
if __name__ == "__main__":
    
    # Para tu modelo en entrenamiento
    run_cascade_inference(
        df_test_csv="./results/preprocessed_data/task2/df_test_hipp.csv",
        models_root="/data/cristian/projects/med_data/rise-miccai/task-2/3d_models/predictions/eff-b2-cascade-focal_spatial_192/fold_models",
        output_dir="/data/cristian/projects/med_data/rise-miccai/task-2/3d_models/predictions/eff-b2-cascade-focal_spatial_192/submissions",
        model_name="eff-b2",
        dim=192,
        use_tta=True,
        use_post_process=True,
        save_per_fold=True
    )