# %%
import os, copy, time
import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from monai.data import Dataset, decollate_batch
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
    CropForegroundd, Spacingd, ScaleIntensityRanged, Resized,
    RandFlipd, RandRotate90d, RandAffineD, RandBiasFieldd, RandGaussianNoised,
    RandAdjustContrastd, EnsureTyped
)
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet
from monai.losses import DiceCELoss

import tqdm

import sys
import os

#sys.path.append("../")
from src.models import get_model

################################################################################
# UTILIDADES PARA MÉTRICAS (scipy disponible en este entorno)
################################################################################
from scipy.spatial.distance import cdist
def dice_score(pred, gt, cls):
    """
    Dice para una clase específica.
    pred, gt: arrays 3D de ints (0,1,2); cls: entero de clase.
    Devuelve NaN si en GT no hay voxels de esa clase.
    """
    pred_bin = (pred == cls)
    gt_bin   = (gt == cls)
    inter = np.logical_and(pred_bin, gt_bin).sum()
    denom = pred_bin.sum() + gt_bin.sum()
    if denom == 0:
        return np.nan  # No voxels en ninguna => ignorar
    return 2.0 * inter / denom

def hausdorff_distance(pred, gt, cls, percentile=100):
    """
    HD o HD95 (si percentile=95). pred y gt de shape (Z,H,W).
    Si alguna máscara está vacía devuelve NaN.
    """
    p = (pred == cls)
    g = (gt   == cls)
    if p.sum() == 0 or g.sum() == 0:
        return np.nan
    coords_p = np.argwhere(p)
    coords_g = np.argwhere(g)
    dists = cdist(coords_p, coords_g)
    d_p_to_g = dists.min(axis=1)
    d_g_to_p = dists.min(axis=0)
    all_d = np.concatenate([d_p_to_g, d_g_to_p])
    return np.percentile(all_d, percentile)

def assd(pred, gt, cls):
    """
    Average Symmetric Surface Distance (ASSD). 
    Calcula distancias entre voxels de superficie.
    Devuelve NaN si alguna máscara vacía.
    """
    p = (pred == cls)
    g = (gt   == cls)
    if p.sum() == 0 or g.sum() == 0:
        return np.nan
    coords_p = np.argwhere(p)
    coords_g = np.argwhere(g)
    dists = cdist(coords_p, coords_g)
    d_p_to_g = dists.min(axis=1)
    d_g_to_p = dists.min(axis=0)
    return (d_p_to_g.mean() + d_g_to_p.mean()) / 2.0

def rve(pred, gt, cls):
    """
    Relative Volume Error: (Vol_pred - Vol_gt) / Vol_gt.
    Devuelve NaN si Vol_gt=0.
    """
    v_pred = (pred == cls).sum()
    v_gt   = (gt   == cls).sum()
    if v_gt == 0:
        return np.nan
    return (v_pred - v_gt) / v_gt

################################################################################
# TRANSFORMS
################################################################################
SPACING      = (1.0, 1.0, 1.0)
SPATIAL_SIZE = (120, 120, 120)
SPATIAL_SIZE = (96,96,96)

def get_train_transforms():
    return Compose([
        LoadImaged(keys=["image","label"]),
        EnsureChannelFirstd(keys=["image","label"]),
        Orientationd(keys=["image","label"], axcodes="RAS"),
        CropForegroundd(keys=["image","label"], source_key="image", allow_smaller=True),
        Spacingd(keys=["image","label"], pixdim=SPACING, mode=("bilinear","nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=15.0, b_min=0.0, b_max=1.0, clip=True),
        Resized(keys=["image","label"], spatial_size=SPATIAL_SIZE, mode=("trilinear","nearest")),
        # Ejemplo de augmentación: flips y rotaciones aleatorias
        RandFlipd(keys=["image","label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image","label"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image","label"], prob=0.5, max_k=3),
        RandAffineD(
            keys=["image","label"], prob=0.5,
            rotate_range=(0.4,0,0), translate_range=(0,20,20),
            scale_range=(0.1,0.1,0.1), padding_mode="zeros",
            mode=("bilinear","nearest")
        ),
        RandBiasFieldd(keys=["image"], prob=0.5, coeff_range=(0.0,0.05)),
        RandGaussianNoised(keys=["image"], prob=0.5, std=0.01),
        RandAdjustContrastd(keys=["image"], prob=0.5, gamma=(0.7,1.5)),
        EnsureTyped(keys=["image","label"], track_meta=True),
    ])

def get_val_transforms():
    return Compose([
        LoadImaged(keys=["image","label"]),
        EnsureChannelFirstd(keys=["image","label"]),
        Orientationd(keys=["image","label"], axcodes="RAS"),
        CropForegroundd(keys=["image","label"], source_key="image", allow_smaller=True),
        Spacingd(keys=["image","label"], pixdim=SPACING, mode=("bilinear","nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=15.0, b_min=0.0, b_max=1.0, clip=True),
        Resized(keys=["image","label"], spatial_size=SPATIAL_SIZE, mode=("trilinear","nearest")),
        EnsureTyped(keys=["image","label"], track_meta=True),
    ])

################################################################################
# DATASET y DATALOADER
################################################################################
class MRIDataset3D(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        data = []
        for _, r in df.iterrows():
            item = {"image": r["filepath"]}
            if "filepath_label" in r and pd.notna(r["filepath_label"]):
                item["label"] = r["filepath_label"]
            data.append(item)
        super().__init__(data, transform=transform)

################################################################################
# MODELOS
################################################################################
def create_model(model_name="unet",device="cuda:5"):
    """
    Devuelve un modelo MONAI:
      - 'unet'   → UNet 3D simple.
      - 'unetr'  → Ejemplo transformer (necesita monai >= 1.0).
    Añade aquí otras arquitecturas.
    """
    if model_name.lower() == "unet":
        return  UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=3,       # binario
            channels=(16, 32, 64, 128),
            strides=(2, 2, 2),
        )
    elif model_name.lower() == "unest":
        return get_model(name=model_name, num_classes=3, device=device)

    elif model_name.lower() == "swinunetr":
        from monai.networks.nets import SwinUNETR
        from huggingface_hub import hf_hub_download
        """
        model = SwinUNETR(in_channels=1, out_channels=14, feature_size=48, use_checkpoint=True)
        # 📦 Descargar pesos preentrenados
        model_path = hf_hub_download(
            repo_id="MONAI/swin_unetr_btcv_segmentation",
            filename="models/model.pt",
            local_dir="/data/cristian/projects/med_data/rise-miccai/pretrained_models/swin_unetr_btcv_segmentation/",
            local_dir_use_symlinks=False
        )

        # ✅ Cargar directamente el state_dict (no usar load_from aquí)
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        """
        import torch
        import torch.nn as nn
        from monai.networks.nets import SwinUNETR
        from huggingface_hub import hf_hub_download


        class AdaptedSwinUNETR(nn.Module):
            def __init__(
                self,
                img_size=(96, 96, 96),
                in_channels=1,
                num_classes=3,
                pretrained=True,
                freeze_stage=0,  # 0: none, 1: freeze all, 2: freeze encoder only
                bottleneck=32,
                use_checkpoint=True,
                repo_id="MONAI/swin_unetr_btcv_segmentation",
                filename="models/model.pt",
                cache_dir="/data/cristian/projects/med_data/rise-miccai/pretrained_models/swin_unetr_btcv_segmentation/",
                device="cuda:0",
            ):
                super().__init__()

                # Backbone con salida original (14 clases BTCV)
                self.backbone = SwinUNETR(
                    #img_size=img_size,
                    in_channels=in_channels,
                    out_channels=14,  # Pretrained
                    feature_size=48,
                    use_checkpoint=use_checkpoint,
                )

                if pretrained:
                    ckpt_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        local_dir=cache_dir,
                        local_dir_use_symlinks=False,
                    )
                    state_dict = torch.load(ckpt_path, map_location="cpu")
                    self.backbone.load_state_dict(state_dict)
                    print("Cargados pesos preentrenados desde:", ckpt_path)

                # ❄️ Freeze encoder si se requiere (solo backbone, no head)
                # 🔁 Fijar pesos si se requiere
                if freeze_stage == 1:
                    for param in self.backbone.parameters():
                        param.requires_grad = False
                elif freeze_stage == 2:
                    for name, param in self.backbone.named_parameters():
                        if "decoder" not in name and "up" not in name and "out" not in name:
                            print("freeze param:", name)
                            param.requires_grad = False

                # 🎯 Adapter elegante con bottleneck
                self.adapter = nn.Sequential(
                    nn.Conv3d(14, bottleneck, kernel_size=1, bias=False),
                    nn.GroupNorm(num_groups=8, num_channels=bottleneck),
                    nn.ReLU(inplace=True),
                    nn.Dropout3d(p=0.1),
                    nn.Conv3d(bottleneck, num_classes, kernel_size=1, bias=True)
                )

                self.to(device)

            def forward(self, x):
                x = self.backbone(x)  # [B, 14, D, H, W]
                x = self.adapter(x)   # [B, 3, D, H, W]
                return x


        model = AdaptedSwinUNETR(
                img_size=(96, 96, 96),
                in_channels=1,
                num_classes=3,
                freeze_stage=0,  # Puedes controlar qué congelar (0: nada, 1: todo, 2: encoder NO FUNCIONO)
                pretrained=True,
                device=device,
            )

        return model

    elif model_name.lower() == "segresnet":
        from monai.networks.nets import SegResNet
        from huggingface_hub import hf_hub_download
        """
        model = SegResNet(spatial_dims=3, in_channels=1, out_channels=105, init_filters=32, blocks_down=[1,2,2,4])
        # 📦 Descargar pesos preentrenados
        model_path = hf_hub_download(
            repo_id="MONAI/wholeBody_ct_segmentation",
            filename="models/model.pt",
            local_dir="/data/cristian/projects/med_data/rise-miccai/pretrained_models/wholeBody_ct_segmentation/",
            local_dir_use_symlinks=False
        )
        # ✅ Cargar directamente el state_dict (no usar load_from aquí)
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        """
        import torch
        import torch.nn as nn
        from monai.networks.nets import SegResNet
        from huggingface_hub import hf_hub_download


        class AdaptedSegResNetV2(nn.Module):
            def __init__(
                self,
                pretrained=True,
                repo_id="MONAI/wholeBody_ct_segmentation",
                filename="models/model.pt",
                cache_dir="/data/cristian/projects/med_data/rise-miccai/pretrained_models/wholeBody_ct_segmentation/",
                freeze_stage=0,  # 0: none, 1: freeze all, 2: freeze encoder only
                bottleneck=32,
                num_classes=3,
                device="cuda:0",
            ):
                super().__init__()

                # Backbone con salida original (105 clases del modelo preentrenado)
                self.backbone = SegResNet(
                    spatial_dims=3,
                    in_channels=1,
                    out_channels=105,
                    init_filters=32,
                    blocks_down=[1, 2, 2, 4],
                )

                if pretrained:
                    ckpt_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        local_dir=cache_dir,
                        local_dir_use_symlinks=False,
                    )
                    state_dict = torch.load(ckpt_path, map_location="cpu")
                    self.backbone.load_state_dict(state_dict)
                    print("Cargados pesos preentrenados desde:", ckpt_path)

                # 🔁 Fijar pesos si se requiere
                if freeze_stage == 1:
                    for param in self.backbone.parameters():
                        param.requires_grad = False
                elif freeze_stage == 2:
                    for name, param in self.backbone.named_parameters():
                        if "decoder" not in name and "upsample" not in name:
                            print("freeze param:", name)
                            param.requires_grad = False

                # 🔄 Adapter más sofisticado que simple conv
                self.adapter = nn.Sequential(
                    nn.Conv3d(105, bottleneck, kernel_size=1, bias=False),
                    nn.GroupNorm(num_groups=8, num_channels=bottleneck),
                    nn.ReLU(inplace=True),
                    nn.Dropout3d(p=0.1),
                    nn.Conv3d(bottleneck, num_classes, kernel_size=1, bias=True)
                )

                self.to(device)

            def forward(self, x):
                x = self.backbone(x)
                x = self.adapter(x)
                return x


        model = AdaptedSegResNetV2(
                pretrained=True,
                freeze_stage=0,  # Puedes controlar qué congelar (0: nada, 1: todo, 2: encoder NO FUNCIONO)
                bottleneck=32,
                num_classes=3,
                device=device
            )
    
        return model
    else:
        raise ValueError(f"Modelo no soportado: {model_name}")

################################################################################
# ENTRENAMIENTO CON 5 FOLDS
################################################################################
def train_and_evaluate(df: pd.DataFrame, num_folds=5, num_epochs=50, model_name="unet",early_stopping_patience=50,
                       batch_size=1, lr=1e-4, weight_decay=1e-5, root_dir="./models",device = "cuda:5"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
        print(f"\n===== Fold {fold+1}/{num_folds} =====")
        df_train_fold = df.iloc[train_idx].reset_index(drop=True)
        df_val_fold   = df.iloc[val_idx].reset_index(drop=True)

        train_ds = MRIDataset3D(df_train_fold, transform=get_train_transforms())
        val_ds   = MRIDataset3D(df_val_fold,   transform=get_val_transforms())

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

        model = create_model(model_name,device).to(device)
        loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scaler = GradScaler()

        # para guardar el mejor modelo del fold
        best_dice = 0.0
        fold_dir = os.path.join(root_dir, f"fold_{fold+1}")
        os.makedirs(fold_dir, exist_ok=True)
        
        early_stopping_counter = 0
        #early_stopping_patience = 50  # detener si no hay mejora en 10
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0
            pbar = tqdm.tqdm(train_loader, desc=f"Train Fold {fold} Epoch {epoch}", leave=False, dynamic_ncols=True)
            #for batch in train_loader:
            for batch in pbar:
                x = batch["image"].to(device)
                y = batch["label"].to(device)

                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=True):
                    #print("x,y:",x.shape, y.shape)  # Añadido para depuración
                    logits = model(x)
                    #print("logits:",logits.shape)  # Añadido para depuración
                    loss   = loss_function(logits, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()

            epoch_loss /= len(train_loader)
            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}")

            # VALIDACIÓN
            model.eval()
            fold_metrics = {"dice_L": [], "dice_R": [], "hd_L": [], "hd_R": [],
                            "hd95_L": [], "hd95_R": [], "assd_L": [], "assd_R": [],
                            "rve_L": [], "rve_R": []}
            with torch.no_grad():
                for batch in val_loader:
                    val_img = batch["image"].to(device)
                    val_lbl = batch["label"].to(device)

                    with torch.cuda.amp.autocast(enabled=True):
                        logits = sliding_window_inference(val_img, SPATIAL_SIZE, 2, model)
                    # pred -> (B, C, D,H,W); argmax -> (B,D,H,W)
                    preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)
                    gts   = val_lbl.squeeze(1)

                    preds_np = preds.cpu().numpy()
                    gts_np   = gts.cpu().numpy()

                    # computar métricas por batch (B==1 aquí)
                    for p, g in zip(preds_np, gts_np):
                        # clases: 1 = L, 2 = R
                        for cls, key in [(1, "L"), (2, "R")]:
                            dsc  = dice_score(p, g, cls)
                            hd   = hausdorff_distance(p, g, cls, percentile=100)
                            hd95 = hausdorff_distance(p, g, cls, percentile=95)
                            dist = assd(p, g, cls)
                            vol  = rve(p, g, cls)

                            if not np.isnan(dsc):  fold_metrics[f"dice_{key}"].append(dsc)
                            if not np.isnan(hd):   fold_metrics[f"hd_{key}"].append(hd)
                            if not np.isnan(hd95): fold_metrics[f"hd95_{key}"].append(hd95)
                            if not np.isnan(dist): fold_metrics[f"assd_{key}"].append(dist)
                            if not np.isnan(vol):  fold_metrics[f"rve_{key}"].append(vol)

            # promediamos métricas de este epoch
            if len(fold_metrics["dice_L"]) > 0:
                dice_L   = np.mean(fold_metrics["dice_L"])
                dice_R   = np.mean(fold_metrics["dice_R"])
                dice_avg = (dice_L + dice_R) / 2.0

                hd_L   = np.mean(fold_metrics["hd_L"]) if len(fold_metrics["hd_L"]) > 0 else np.nan
                hd_R   = np.mean(fold_metrics["hd_R"]) if len(fold_metrics["hd_R"]) > 0 else np.nan
                hd95_L = np.mean(fold_metrics["hd95_L"]) if len(fold_metrics["hd95_L"]) > 0 else np.nan
                hd95_R = np.mean  (fold_metrics["hd95_R"]) if len(fold_metrics["hd95_R"]) > 0 else np.nan
                assd_L = np.mean(fold_metrics["assd_L"]) if len(fold_metrics["assd_L"]) > 0 else np.nan 
                assd_R = np.mean(fold_metrics["assd_R"]) if len(fold_metrics["assd_R"]) > 0 else np.nan
                rve_L  = np.mean(fold_metrics["rve_L"]) if len(fold_metrics["rve_L"]) > 0 else np.nan
                rve_R  = np.mean(fold_metrics["rve_R"]) if len(fold_metrics["rve_R"]) > 0 else np.nan

                print(f"Val Dice L: {dice_L:.4f}  Dice R: {dice_R:.4f}  Dice Avg: {dice_avg:.4f}" 
                      f"  HD L: {hd_L:.4f}  HD R: {hd_R:.4f}  HD95 L: {hd95_L:.4f}  HD95 R: {hd95_R:.4f}"
                      f"  ASSD L: {assd_L:.4f}  ASSD R: {assd_R:.4f}  RVE L: {rve_L:.4f}  RVE R: {rve_R:.4f}")

                # checkpoint: guarda si supera mejor dice promedio
                if dice_avg > best_dice:
                    early_stopping_counter = 0  # reset counter si hay mejora
                    best_dice = dice_avg
                    torch.save(model.state_dict(), os.path.join(fold_dir, "best_model.pth"))
                    print(f"Nuevo mejor modelo guardado - Avg Dice: {best_dice:.4f}",os.path.join(fold_dir, "best_model.pth"))
                else:
                    early_stopping_counter += 1
                    if early_stopping_counter >= early_stopping_patience:
                        print(f"No hay mejora en {early_stopping_patience} epochs. Deteniendo entrenamiento.")
                        break
            #else:
            #    print("No se pudieron calcular métricas de validación (posiblemente máscaras vacías).")
        # tras entrenamiento, evaluamos todo el fold (best checkpoint)
        # cargamos mejor modelo
        model.load_state_dict(torch.load(os.path.join(fold_dir, "best_model.pth")))
        model.eval()
        fold_results = {}
        for key in ["dice", "hd", "hd95", "assd", "rve"]:
            fold_results[key+"_L"] = []
            fold_results[key+"_R"] = []

        with torch.no_grad():
            for batch in val_loader:
                val_img = batch["image"].to(device)
                val_lbl = batch["label"].to(device)
                with torch.cuda.amp.autocast(enabled=True):
                    logits = sliding_window_inference(val_img, SPATIAL_SIZE, 2, model)
                preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)
                gts   = val_lbl.squeeze(1)
                preds_np = preds.cpu().numpy()
                gts_np   = gts.cpu().numpy()
                for p, g in zip(preds_np, gts_np):
                    for cls, key in [(1,"L"), (2,"R")]:
                        fold_results["dice_"+key].append(dice_score(p,g,cls))
                        fold_results["hd_"+key].append(hausdorff_distance(p,g,cls,100))
                        fold_results["hd95_"+key].append(hausdorff_distance(p,g,cls,95))
                        fold_results["assd_"+key].append(assd(p,g,cls))
                        fold_results["rve_"+key].append(rve(p,g,cls))

        # resumimos fold
        summary = {}
        for metric in ["dice","hd","hd95","assd","rve"]:
            for cls in ["L","R"]:
                vals = [v for v in fold_results[f"{metric}_{cls}"] if not np.isnan(v)]
                summary[f"{metric}_{cls}"] = np.mean(vals) if len(vals)>0 else np.nan
            # promedio sobre L y R
            summary[f"{metric}_Avg"] = np.nanmean([summary[f"{metric}_L"], summary[f"{metric}_R"]])
        print("Resumen métricas fold:", summary)
        results.append(summary)

    # promediamos sobre folds
    final = {}
    for metric in ["dice","hd","hd95","assd","rve"]:
        for key in ["L","R","Avg"]:
            vals = [res[f"{metric}_{key}"] for res in results]
            final[f"{metric}_{key}"] = np.nanmean(vals)
    print("\n=== Resultado final promedio de 5 folds ===")
    for metric in ["dice","hd","hd95","assd","rve"]:
        print(f"{metric.upper()}_L  : {final[metric+'_L']:.4f}")
        print(f"{metric.upper()}_R  : {final[metric+'_R']:.4f}")
        print(f"{metric.upper()}_Avg: {final[metric+'_Avg']:.4f}\n")

    return final

################################################################################
# EJECUCIÓN PRINCIPAL
################################################################################

# %%



# %%

# Entrenamiento y evaluación con UNet
"""
final_metrics = train_and_evaluate(
    df=df,
    num_folds=5,
    num_epochs=1000,         # ajusta según tus recursos
    model_name="unest",      # o "unetr" si quieres transformer
    batch_size=4,           # 1 para sliding_window, puedes aumentar si tu GPU lo permite
    lr=1e-4,
    weight_decay=1e-5,
    device="cuda:5",
    root_dir="/data/cristian/projects/med_data/rise-miccai/task-2/3d_models/predictions/model_unest_04_test/fold_models"
)


final_metrics = train_and_evaluate(
    df=df,
    num_folds=5,
    num_epochs=1000,         # ajusta según tus recursos
    model_name="swinunetr",      # o "unetr" si quieres transformer
    batch_size=4,           # 1 para sliding_window, puedes aumentar si tu GPU lo permite
    lr=1e-4,
    weight_decay=1e-5,
    device="cuda:3",
    root_dir="/data/cristian/projects/med_data/rise-miccai/task-2/3d_models/predictions/model_swinunetr_04_test/fold_models"
)

final_metrics = train_and_evaluate(
    df=df,
    num_folds=5,
    num_epochs=1000,         # ajusta según tus recursos
    model_name="unest",      # o "unetr" si quieres transformer
    batch_size=4,           # 1 para sliding_window, puedes aumentar si tu GPU lo permite
    lr=1e-4,
    weight_decay=1e-5,
    device="cuda:5",
    root_dir="/data/cristian/projects/med_data/rise-miccai/task-2/3d_models/predictions/model_unest_04_test/fold_models"
)


final_metrics = train_and_evaluate(
    df=df,
    num_folds=5,
    num_epochs=1000,         # ajusta según tus recursos
    model_name="swinunetr",      # o "unetr" si quieres transformer
    batch_size=4,           # 1 para sliding_window, puedes aumentar si tu GPU lo permite
    lr=1e-4,
    weight_decay=1e-5,
    device="cuda:3",
    root_dir="/data/cristian/projects/med_data/rise-miccai/task-2/3d_models/predictions/model_swinunetr_04_test/fold_models"
)



    num_folds=5,
    num_epochs=1000,         # ajusta según tus recursos
    model_name="swinunetr",      # o "unetr" si quieres transformer
    batch_size=4,           # 1 para sliding_window, puedes aumentar si tu GPU lo permite
    lr=1e-4,
    weight_decay=1e-5,
    device="cuda:3",
    root_dir="/data/cristian/projects/med_data/rise-miccai/task-2/3d_models/predictions/model_swinunetr_04_test/fold_models"


    python training.py --model_name=swinunetr --device=cuda:3 --root_dir=/data/cristian/projects/med_data/rise-miccai/task-2/3d_models/predictions/model_swinunetr_04_test/fold_models

    python training.py --model_name=segresnet --device=cuda:5 --root_dir=/data/cristian/projects/med_data/rise-miccai/task-2/3d_models/predictions/model_segresnet_04_test/fold_models --early_stopping_patience=50
    
"""




import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train 3D segmentation model for LISA Challenge")

    parser.add_argument('--model_name', type=str, default="swinunetr", choices=["swinunetr", "unetr","segresnet", "unest"],
                        help="Model architecture to use")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size for training")
    parser.add_argument('--early_stopping_patience', type=int, default=50, help="Batch size for training")
    
    parser.add_argument('--num_epochs', type=int, default=1000, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=1e-5, help="Weight decay (L2 regularization)")
    parser.add_argument('--device', type=str, default="cuda:0", help="Device to use for training")
    parser.add_argument('--num_folds', type=int, default=5, help="Number of cross-validation folds")
    parser.add_argument('--root_dir', type=str, required=True, help="Directory to save models and logs")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()


    # %%
    # Carga tu CSV con rutas de imágenes y máscaras
    # Debe tener columnas: 'filepath' y 'filepath_label'
    csv_path = "results/preprocessed_data/task2/df_train_hipp.csv"
    df = pd.read_csv(csv_path)
    df.head()

    final_metrics = train_and_evaluate(
        df=df,
        num_folds=args.num_folds,
        num_epochs=args.num_epochs,
        model_name=args.model_name,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        root_dir=args.root_dir
    )

    print("Métricas finales:", final_metrics)
    print("hello")
# %%



