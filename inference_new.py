#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference for RISE-MICCAI LISA 2025 (hippocampus 3-class) con:
- sliding_window_inference (parches 3D)
- Invertd usando meta interno (MetaTensor) → vuelve a shape/affine ORIGINAL
- Ensemble opcional (promedio de logits)


python inference.py   --test_csv results/preprocessed_data/task2/df_test_hipp.csv   --model_dirs /data/cristian/projects/med_data/rise-miccai/task-2/3d_models/results/model_unest_02/fold_0   --model unest   --output_dir /data/cristian/projects/med_data/rise-miccai/task-2/3d_models/predictions/model_unest_02/

"""

from __future__ import annotations
import argparse, os, re
from typing import List, Tuple
import numpy as np
import pandas as pd
import tqdm

import torch
from torch.utils.data import DataLoader
import nibabel as nib

from monai.data import MetaTensor
from monai.metrics import DiceMetric
from monai.networks.utils import one_hot
from monai.inferers import sliding_window_inference
from monai.transforms import Compose, Activationsd, AsDiscreted, Invertd

# imports del proyecto
try:
    from src.dataset import get_dataset
    from src.models import get_model
except Exception:
    from dataset import get_dataset  # type: ignore
    from models import get_model     # type: ignore


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("RISE-MICCAI LISA 2025 - inference")
    p.add_argument("--test_csv", required=True, type=str)
    p.add_argument("--output_dir", required=True, type=str)

    # single
    p.add_argument("--model_dir", type=str, default=None)
    p.add_argument("--model", type=str, default=None,
                   choices=["unest", "swinunetr", "dynunet", "segresnet", "autoencoder"])

    # ensemble
    p.add_argument("--model_dirs", type=str, default=None)
    p.add_argument("--models", type=str, default=None)

    # SWI / preprocess (deben calzar con train)
    p.add_argument("--spacing", type=float, nargs=3, default=(1.0, 1.0, 1.0))
    p.add_argument("--roi_size", type=int, nargs=3, default=(96, 96, 96))
    p.add_argument("--overlap", type=float, default=0.5)
    p.add_argument("--sw_batch_size", type=int, default=4)

    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--eval", action="store_true")
    p.add_argument("--unest_cache", type=str, default=None)
    return p.parse_args()


def _resolve_models(args: argparse.Namespace) -> Tuple[List[str], List[str]]:
    if args.model_dirs:
        dirs = [d.strip() for d in args.model_dirs.split(",") if d.strip()]
        if args.models:
            names = [m.strip().lower() for m in args.models.split(",") if m.strip()]
            if len(names) != len(dirs):
                raise ValueError("--models y --model_dirs deben tener misma longitud")
        else:
            base = (args.model or "unest").lower()
            names = [base] * len(dirs)
        return dirs, names
    else:
        if not (args.model_dir and args.model):
            raise ValueError("Modo single: especifica --model_dir y --model")
        return [args.model_dir], [args.model.lower()]


def _subject_outname(subject_id: str | int) -> str:
    s = str(subject_id)
    nums = re.findall(r"\d+", s)
    numeric = nums[-1] if nums else s
    return f"LISAHF{numeric}segprediction.nii.gz"


def main() -> None:
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}") if (args.gpu is not None and torch.cuda.is_available()) else torch.device("cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.test_csv)
    if "filepath" not in df.columns:
        raise ValueError("CSV debe tener columna 'filepath'")
    has_labels = ("filepath_label" in df.columns) and df["filepath_label"].notna().any()

    # dataset/loader (si --eval y hay labels → is_test=False para cargar labels)
    is_test_dataset = not args.eval or (not has_labels)
    test_ds = get_dataset(
        df=df,
        is_train=False,
        spacing=tuple(args.spacing),
        spatial_size=tuple(args.roi_size),  # sólo para train/val; SWI evita reescalar global en test
        augment=False,
        is_test=is_test_dataset,
    )
    loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                        num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    # post: softmax -> Invertd (usando meta interno) -> argmax
    pre_transform = test_ds.transform
    post_transforms = Compose([
        Activationsd(keys="pred", softmax=True),   # logits -> probs
        Invertd(
            keys="pred",
            transform=pre_transform,
            orig_keys="image",
            # MUY IMPORTANTE: usar meta interno del MetaTensor (no *_meta_dict):
            meta_keys=None,
            orig_meta_keys=None,
            nearest_interp=False,  # lineal sobre probs antes de argmax
            to_tensor=True,        # salida sigue siendo Tensor/MetaTensor
        ),
        AsDiscreted(keys="pred", argmax=True),     # probs -> etiquetas 0/1/2
    ])

    # cargar modelos
    model_dirs, model_names = _resolve_models(args)
    models_list: List[torch.nn.Module] = []
    for md, nm in zip(model_dirs, model_names):
        ckpt = os.path.join(md, "best_model.pth")
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"No se encontró: {ckpt}")
        kwargs = {}
        if nm == "unest" and args.unest_cache:
            kwargs["cache_dir"] = args.unest_cache
        model = get_model(name=nm, num_classes=3, device=str(device), **kwargs)
        sd = torch.load(ckpt, map_location=device)
        state = sd.get("model_state_dict", sd)
        model.load_state_dict(state, strict=False)
        model.eval()
        models_list.append(model)
        print(f"[OK] {nm} <- {ckpt}")

    dice_metric = DiceMetric(include_background=False, reduction="mean") if (args.eval and has_labels) else None
    dice_list = []
    use_amp = (device.type == "cuda")

    pbar = tqdm.tqdm(loader, total=len(loader), dynamic_ncols=True, desc="Inference")
    for i, batch in enumerate(pbar):
        # batch["image"] es MetaTensor con meta/applied_operations
        img = batch["image"].to(device)  # [1,1,D,H,W]

        # --- ensemble: promedio de LOGITS (softmax en post_transforms) ---
        with torch.no_grad():
            logits_sum = None
            for m in models_list:
                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = sliding_window_inference(
                        inputs=img,
                        roi_size=tuple(args.roi_size),
                        sw_batch_size=args.sw_batch_size,
                        predictor=m,
                        overlap=args.overlap,
                        mode="gaussian",
                        sigma_scale=0.125,
                        progress=False,
                        #cache_roi_weight_map=True,
                    )  # [1,3,D,H,W] mismo tamaño que 'img'
                logits_sum = logits if logits_sum is None else (logits_sum + logits)
            avg_logits = logits_sum / len(models_list)

        # --- eval opcional (en espacio transformado) ---
        if dice_metric is not None and ("label" in batch):
            pred_t = torch.argmax(avg_logits, dim=1)     # [1,D,H,W]
            lab_t  = batch["label"].to(device)           # [1,1,D,H,W]
            dice_metric(y_pred=one_hot(pred_t, 3), y=one_hot(lab_t, 3))
            dval = dice_metric.aggregate().item()
            dice_metric.reset()
            dice_list.append(dval)
            pbar.set_postfix({"dice": f"{dval:.3f}"})

        # --- construir 'sample' por-sujeto usando MetaTensor para 'pred' ---
        img_mt: MetaTensor = batch["image"][0].cpu()                          # MetaTensor
        # ¡Clave!: copiar el meta del image (incluye applied_operations)
        pred_mt = MetaTensor(avg_logits[0].cpu(), meta=img_mt.meta.copy())    # [3,D,H,W] con meta
        sample = {"image": img_mt, "pred": pred_mt}
        sample = post_transforms(sample)                                      # softmax -> Invertd -> argmax

        pred_orig = sample["pred"].cpu().numpy().astype(np.uint8)             # [D0,H0,W0] original
        meta = sample["image"].meta
        affine = meta.get("original_affine", meta.get("affine"))

        subject_id = df.iloc[i]["ID"] if "ID" in df.columns else f"{i:04d}"
        out_name = _subject_outname(subject_id)
        out_path = os.path.join(args.output_dir, out_name)
        nib.save(nib.Nifti1Image(pred_orig, affine), out_path)
        pbar.set_postfix({"saved": out_name})

    if dice_list:
        print("\nEvaluation summary (Dice sin background):")
        print(f"  Subjects: {len(dice_list)}")
        print(f"  Mean Dice: {np.mean(dice_list):.4f}")


if __name__ == "__main__":
    main()
