"""
inference_cascade.py
====================

Este script realiza la inferencia con una cascada coarse→fine.  Para
cada volumen del set de test, se aplica el modelo coarse mediante
``sliding_window_inference`` y se obtienen regiones de interés (ROIs)
bilaterales mediante umbralado de la probabilidad.  Cada ROI se
reescalada al tamaño esperado por el refinador, se procesa con el
modelo fine y luego se reproyecta a la resolución coarse.  Las
logits resultantes se promedian entre todos los folds disponibles.
Finalmente se aplican las transformaciones inversas para regresar al
espacio original y se guarda un archivo NIfTI.

Uso básico::

    python inference_cascade.py --csv test.csv \
        --models_root /path/a/folds --outdir ./preds \
        --coarse_model swinunetr --fine_model unest

Los checkpoints deben llamarse ``best_model_coarse.pth`` y
``best_model_fine.pth`` dentro de cada carpeta ``fold_i`` bajo
``models_root``.  Si sólo tienes ``best_model.pth``, el loader lo
usará para ambos modelos.
"""

from __future__ import annotations

import argparse
import os
import re
import copy
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple

import nibabel as nib
import pandas as pd
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from monai.data import Dataset
from monai.inferers import sliding_window_inference
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd, ScaleIntensityRanged, EnsureTyped, Invertd,CropForegroundd

from src.cascade_utils import (
    get_roi_bbox_from_logits,
    crop_to_bbox,
    resize_volume,
)
from src.models import create_model
import gc

DEVICE = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")

def is_positive_background(img):
    """
    Returns a boolean version of `img` where the positive values are converted into True, the other values are False.
    """
    return img > 0.5

def build_preproc(spacing=(1.0, 1.0, 1.0)) -> Compose:
    """Transformaciones para el volumen completo (sin recortar ni redimensionar)."""
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        CropForegroundd(keys=["image"], source_key="image", allow_smaller=False,select_fn=is_positive_background),
        Spacingd(keys=["image"], pixdim=spacing, mode=("bilinear",)),
        ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=15.0, b_min=0.0, b_max=1.0, clip=True),
        EnsureTyped(keys=["image"], track_meta=True),
    ])


class SimpleTestDS(Dataset):
    """Dataset mínimo para inferencia: carga rutas de un DataFrame."""
    def __init__(self, df: pd.DataFrame, transform: Compose) -> None:
        data = []
        for _, r in df.iterrows():
            d = {"image": r["filepath"], "filepath": r["filepath"]}
            if "ID" in r:
                d["ID"] = r["ID"]
            data.append(d)
        super().__init__(data, transform=transform)


def load_models(models_root: str, coarse_name: str, fine_name: str) -> List[Tuple[torch.nn.Module, torch.nn.Module]]:
    """Carga modelos coarse y fine para cada fold en ``models_root``.

    Busca ficheros ``best_model_coarse.pth`` y ``best_model_fine.pth``
    en cada carpeta ``fold_*``.  Si no se encuentran, usa
    ``best_model.pth`` para ambos.
    """
    models = []
    for d in sorted(os.listdir(models_root)):
        fold_dir = os.path.join(models_root, d)
        if not os.path.isdir(fold_dir):
            continue
        ck_coarse = os.path.join(fold_dir, "best_model_coarse.pth")
        ck_fine   = os.path.join(fold_dir, "best_model_fine.pth")
        if not os.path.exists(ck_coarse):
            ck_coarse = os.path.join(fold_dir, "best_model.pth")
        if not os.path.exists(ck_fine):
            ck_fine = os.path.join(fold_dir, "best_model.pth")
        if not os.path.exists(ck_coarse) or not os.path.exists(ck_fine):
            continue
        coarse = create_model(coarse_name).to(DEVICE).eval()
        fine   = create_model(fine_name).to(DEVICE).eval()
        print("Reading coarse:",ck_coarse)
        print("Reading fine  :",ck_fine)
        coarse.load_state_dict(torch.load(ck_coarse, map_location=DEVICE))
        fine.load_state_dict(torch.load(ck_fine,   map_location=DEVICE))
        coarse.eval(); fine.eval()
        models.append((coarse, fine))
    if not models:
        raise ValueError(f"No se encontraron modelos en {models_root}")
    return models


def build_outname(subject_id_or_path: str) -> str:
    base = str(subject_id_or_path)
    nums = re.findall(r"\d+", base)
    nid = nums[-1] if nums else os.path.basename(base).split('.')[0]
    return f"LISAHF{nid}segprediction.nii.gz"


def infer_volume(
    elem: dict,
    models: List[Tuple[torch.nn.Module, torch.nn.Module]],
    preproc_tfm: Compose,
    size_fine: Tuple[int, int, int] = (96, 96, 96),
    thr_roi: float = 0.2,
    roi_margin: int = 16,
    overlap: float = 0.75,
    tta_flips: List[Tuple[int, int, int]] = [(0, 0, 0)],
) -> Tuple[np.ndarray, dict]:
    """Procesa un solo volumen y devuelve la etiqueta predicha y meta."""
    e_img  = elem["image"][0]  # (1, Zp, Yp, Xp)
    e_meta = {k: (v[0] if isinstance(v, (list, tuple)) else v) for k, v in elem["image"].meta.items()}

    logits_sum = torch.zeros((1, 3) + tuple(e_img.shape[-3:]), device="cpu")
    for coarse, fine in models:
        # coarse con SWI + TTA
        def predict_tta(model, x: torch.Tensor) -> torch.Tensor:
            outs = []
            for fx, fy, fz in tta_flips:
                xx = x
                if fx: xx = xx.flip(2)
                if fy: xx = xx.flip(3)
                if fz: xx = xx.flip(4)
                lo = model(xx)
                if fz: lo = lo.flip(4)
                if fy: lo = lo.flip(3)
                if fx: lo = lo.flip(2)
                outs.append(lo.cpu())
            ans = torch.stack(outs, 0).mean(0).to(DEVICE)
            del outs, lo, xx
            return ans

        with autocast():
            logits1 = sliding_window_inference(
                inputs=e_img.unsqueeze(0).to(DEVICE),
                roi_size=(128, 128, 128),
                sw_batch_size=1,
                predictor=lambda inp: predict_tta(coarse, inp),
                overlap=overlap,
                mode="constant",
                sw_device=DEVICE,
                device=DEVICE,
                progress=False,                
            )  # (1, C, Zp, Yp, Xp)
        bbs = get_roi_bbox_from_logits(logits1, thr=thr_roi, margin=roi_margin)
        logits_fold = torch.zeros_like(logits_sum)
        for bb in bbs:
            z0, y0, x0, z1, y1, x1 = bb
            img_roi = crop_to_bbox(e_img.unsqueeze(0), bb)
            img_roi = resize_volume(img_roi, size_fine, mode="trilinear")
            with autocast():
                logits2 = fine(img_roi.to(DEVICE))
            logits2_up = resize_volume(logits2, (z1 - z0, y1 - y0, x1 - x0), mode="trilinear")[0]
            # coloca de vuelta
            logits_fold[:, :, z0:z1, y0:y1, x0:x1] = logits2_up.cpu()
        logits_sum += logits_fold

    logits_avg = logits_sum / float(len(models))
    probs_pre  = torch.softmax(logits_avg, dim=1)[0].cpu().detach().numpy()  # (C,Z,Y,X)
    # invierte al espacio original
    elem_inv = {
        "image": e_img,
        "image_meta_dict": copy.deepcopy(e_meta),
        "pred": probs_pre,
        "pred_meta_dict": copy.deepcopy(e_meta),
    }
    inv = Invertd(
        keys="pred",
        transform=preproc_tfm,
        orig_keys="image",
        meta_keys="pred_meta_dict",
        orig_meta_keys="image_meta_dict",
        nearest_interp=False,
        to_tensor=True,
    )
    elem_inv = inv(elem_inv)
    pred_lab = np.argmax(elem_inv["pred"], axis=0).astype(np.uint8)
    return pred_lab, e_meta


def run_inference_cascade(
    csv_path: str,
    models_root: str,
    outdir: str,
    coarse_model: str = "swinunetr",
    fine_model: str = "unest",
    size_fine: Tuple[int, int, int] = (96, 96, 96),
    thr_roi: float = 0.2,
    roi_margin: int = 16,
    overlap: float = 0.75,
    use_tta: bool = True,
):
    os.makedirs(outdir, exist_ok=True)
    df = pd.read_csv(csv_path)
    assert "filepath" in df.columns
    preproc = build_preproc()
    test_ds = SimpleTestDS(df, transform=preproc)
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    tta_flips = [(0, 0, 0)]
    if use_tta:
        tta_flips = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)]
    models = load_models(models_root, coarse_model, fine_model)
    for i, elem in enumerate(test_dl):
        sid = elem.get("ID", [None])[0]
        out_name = build_outname(sid if sid is not None else elem["filepath"][0])
        pred_lab, meta = infer_volume(
            elem,
            models,
            preproc,
            size_fine=size_fine,
            thr_roi=thr_roi,
            roi_margin=roi_margin,
            overlap=overlap,
            tta_flips=tta_flips,
        )
        aff = np.array(meta["original_affine"])
        nib.save(nib.Nifti1Image(pred_lab.astype(np.uint8), aff), os.path.join(outdir, out_name))
        print(f"[{i+1}/{len(test_ds)}] guardado: {out_name}")
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()

# python inference_cascade.py --csv  "results/preprocessed_data/task2/df_test_hipp.csv" --models_root "/data/cristian/projects/med_data/rise-miccai/task-2/3d_models/predictions/efficientnet-b7_mixup_hard_dice_ce_symmetry_fine/fold_models" --outdir "/data/cristian/projects/med_data/rise-miccai/task-2/3d_models/predictions/efficientnet-b7_mixup_hard_dice_ce_symmetry_fine/predictions" --coarse_model  "efficientnet-b7" --fine_model "efficientnet-b7"
                        
def main():
    parser = argparse.ArgumentParser(description="Inferencia de cascada coarse→fine")
    parser.add_argument("--csv", type=str, required=True, help="CSV de test con columna filepath")
    parser.add_argument("--models_root", type=str, required=True, help="Directorio con fold_*/best_model*.pth")
    parser.add_argument("--outdir", type=str, required=True, help="Carpeta de salida")
    parser.add_argument("--coarse_model", type=str, default="unest")
    parser.add_argument("--fine_model", type=str, default="unest")
    parser.add_argument("--size_fine", type=str, default="96,96,96", help="Tamaño ROI fine, ej: 96,96,96")
    parser.add_argument("--thr_roi", type=float, default=0.2)
    parser.add_argument("--roi_margin", type=int, default=16)
    parser.add_argument("--overlap", type=float, default=0.75)
    parser.add_argument("--no_tta", action="store_true", help="Desactiva el test-time augmentation por flips")
    args = parser.parse_args()
    size_f = tuple(int(s) for s in args.size_fine.split(','))
    run_inference_cascade(
        csv_path=args.csv,
        models_root=args.models_root,
        outdir=args.outdir,
        coarse_model=args.coarse_model,
        fine_model=args.fine_model,
        size_fine=size_f,
        thr_roi=args.thr_roi,
        roi_margin=args.roi_margin,
        overlap=args.overlap,
        use_tta=not args.no_tta,
    )


if __name__ == "__main__":
    main()