
import os, copy, time
import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupKFold

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from monai.inferers import sliding_window_inference

import tqdm

import sys
import os

#sys.path.append("../")
from src.models import create_model
from src import metrics
from src import dataset
from src import losses

from dotenv import load_dotenv

os.environ["WANDB_DIR"] = "/data/cristian/paper_2025/wandb_dir"  # Aquí se guardarán los archivos temporales y logs
os.environ["WANDB_CACHE_DIR"] = "/data/cristian/paper_2025/wandb_dir"
os.environ["WANDB_ARTIFACT_DIR"] = "/data/cristian/paper_2025/wandb_dir"



################################################################################
# TRANSFORMS
################################################################################
SPACING      = (1.0, 1.0, 1.0)
SPATIAL_SIZE = (120, 120, 120)
SPATIAL_SIZE = (96,96,96)

################################################################################
# ENTRENAMIENTO CON 5 FOLDS
################################################################################

def evaluate_model(model, val_loader, device,prefix="val",loss_function=None,fold=None,epoch=None,full=True,show=False):
    # VALIDACIÓN
    model.eval()
    fold_metrics = {"dice_L": [], "dice_R": [], "hd_L": [], "hd_R": [],
                    "hd95_L": [], "hd95_R": [], "assd_L": [], "assd_R": [],
                    "rve_L": [], "rve_R": []}

    fold_metrics_means = {}
    epoch_loss = 0
    with torch.no_grad():
        pbar = tqdm.tqdm(val_loader, desc=f"{prefix} Fold {fold} Epoch {epoch}", leave=False, dynamic_ncols=True)
        #for batch in val_loader:
        for batch in pbar:        
            val_img = batch["image"].to(device)
            val_lbl = batch["label"].to(device)

            with torch.cuda.amp.autocast(enabled=True):
                #logits = sliding_window_inference(val_img, SPATIAL_SIZE, 2, model)
                logits = model(val_img)
                loss   = loss_function(logits, val_lbl)
            epoch_loss += loss.item()

            # pred -> (B, C, D,H,W); argmax -> (B,D,H,W)
            preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)
            gts   = val_lbl.squeeze(1)

            preds_np = preds.cpu().numpy()
            gts_np   = gts.cpu().numpy()

            # computar métricas por batch (B==1 aquí)
            for p, g in zip(preds_np, gts_np):
                # clases: 1 = L, 2 = R
                for cls, key in [(1, "L"), (2, "R")]:
                    dsc  = metrics.dice_score(p, g, cls)
                    if full:
                        hd   = metrics.hausdorff_distance(p, g, cls, percentile=100)
                        hd95 = metrics.hausdorff_distance(p, g, cls, percentile=95)
                        dist = metrics.assd(p, g, cls)
                        vol  = metrics.rve(p, g, cls)

                    if not np.isnan(dsc):  fold_metrics[f"dice_{key}"].append(dsc)
                    if full:
                        if not np.isnan(hd):   fold_metrics[f"hd_{key}"].append(hd)
                        if not np.isnan(hd95): fold_metrics[f"hd95_{key}"].append(hd95)
                        if not np.isnan(dist): fold_metrics[f"assd_{key}"].append(dist)
                        if not np.isnan(vol):  fold_metrics[f"rve_{key}"].append(vol)
        epoch_loss /= len(val_loader)

    # promediamos métricas de este epoch
    if len(fold_metrics["dice_L"]) > 0:
        dice_L   = np.mean(fold_metrics["dice_L"])
        dice_R   = np.mean(fold_metrics["dice_R"])
        dice_avg = (dice_L + dice_R) / 2.0

        if full:
            hd_L   = np.mean(fold_metrics["hd_L"])
            hd_R   = np.mean(fold_metrics["hd_R"])
            hd_avg = (hd_L + hd_R) / 2.0
            hd95_L = np.mean(fold_metrics["hd95_L"])
            hd95_R = np.mean(fold_metrics["hd95_R"]) 
            hd95_avg = (hd95_L + hd95_R) / 2.0
            assd_L = np.mean(fold_metrics["assd_L"]) 
            assd_R = np.mean(fold_metrics["assd_R"]) 
            assd_avg = (assd_L + assd_R) / 2.0
            rve_L  = np.mean(fold_metrics["rve_L"]) 
            rve_R  = np.mean(fold_metrics["rve_R"]) 
            rve_avg = (rve_L + rve_R) / 2.0
        # fold metrics means with prefix
        fold_metrics_means = {
            f"{prefix}_dice_L": dice_L,
            f"{prefix}_dice_R": dice_R,
            f"{prefix}_dice_Avg": dice_avg,
            f"{prefix}_loss": epoch_loss
        }
        if full:
            fold_metrics_means.update({
            f"{prefix}_hd_L": hd_L,
            f"{prefix}_hd_R": hd_R,
            f"{prefix}_hd95_L": hd95_L,
            f"{prefix}_hd95_R": hd95_R,
            f"{prefix}_assd_L": assd_L,
            f"{prefix}_assd_R": assd_R,
            f"{prefix}_rve_L": rve_L,
            f"{prefix}_rve_R": rve_R,

            f"{prefix}_hd_Avg": hd_avg,
            f"{prefix}_hd95_Avg": hd95_avg,
            f"{prefix}_assd_Avg": assd_avg,
            f"{prefix}_rve_Avg": rve_avg,
            })

        if show:
            print(f"{prefix} Dice L: {dice_L:.4f} Dice R: {dice_R:.4f} Dice Avg: {dice_avg:.4f}" ,end=" ")
            if full:
                print(f" HD L: {hd_L:.4f} HD R: {hd_R:.4f}"
                    f"HD95 L: {hd95_L:.4f} HD95 R: {hd95_R:.4f}"
                    f" ASSD L: {assd_L:.4f} ASSD R: {assd_R:.4f} RVE L: {rve_L:.4f} RVE R: {rve_R:.4f}",end= " ")
            print("")
    return fold_metrics_means

def train_and_evaluate(df: pd.DataFrame, num_folds=5, num_epochs=50, model_name="unet",early_stopping_patience=50,
                       batch_size=1, lr=1e-4, weight_decay=1e-5, root_dir="./models",device = "cuda:5",aug_method="lite",use_mixup= False,args={}):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    results = {"val_best": [], "test_best": []}

    use_wandb = bool(os.getenv('WANDB_API_KEY')) and bool(os.getenv('PROJECT_WANDB'))

    # Remove 10% of HF_hipp samples from df for backtesting, keep the rest for training
    #df_hipp = df[df["source_label"] == "HF_hipp"].reset_index(drop=True)
    list_ids = df["ID"].unique().tolist()
    print(f"Total samples: {len(df)} | Unique IDs: {len(list_ids)}")
    #train test split of IDs 10% of list_ids
    list_ids_train, list_ids_backtest = train_test_split(list_ids, test_size=0.1, random_state=42)
    df_backtest = df[(df["ID"].isin(list_ids_backtest))].reset_index(drop=True).copy()
    df_train    = df[~(df["ID"].isin(list_ids_backtest))].reset_index(drop=True).copy()
    print(f"Total samples: {len(df)} | Train samples: {len(df_train)} | Backtest samples: {len(df_backtest)}")

    ids_train = set(df_train["ID"])
    ids_val = set(df_backtest["ID"])
    ids_comunes = ids_train.intersection(ids_val)
    print(f"Train    size: {len(df_train)}")
    print(f"BackTest size:   {len(df_backtest)}")
    print(f"IDs en común (data leakage): {len(ids_comunes)}")
    if len(ids_comunes) > 0:
        print(f"IDs duplicados: {list(ids_comunes)}")


    df = df_train
    test_ds      = dataset.MRIDataset3D(df_backtest,   transform=dataset.get_val_transforms(SPACING=SPACING,SPATIAL_SIZE=SPATIAL_SIZE))
    test_loader  = DataLoader(test_ds,   batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    
    # Verifica que 'ID' existe
    assert "ID" in df.columns, "Falta la columna 'ID'"

    # Número de folds
    num_folds = 5

    # Inicializa columna de folds
    df["fold"] = -1

    # GroupKFold asegura que IDs no se mezclan entre train y val
    gkf = GroupKFold(n_splits=num_folds)

    for fold, (train_idx, val_idx) in enumerate(gkf.split(df, groups=df["ID"])):
        
        use_train = True
        run = None
        if use_wandb and use_train:
            import wandb
            exp_name = args.get('experiment_name', 'rise_experiment')
            run = wandb.init(
                project=os.getenv('PROJECT_WANDB'),
                entity=os.getenv('ENTITY'),
                name=f"{exp_name}_fold{fold}",
                group=exp_name,
                config=args,
                save_code=True,
                reinit=True,
            )

        df.loc[val_idx, "fold"] = fold

        df_train_fold = df[df["fold"] != fold].reset_index(drop=True)
        df_val_fold = df[df["fold"] == fold].reset_index(drop=True)

        ids_train = set(df_train_fold["ID"])
        ids_val = set(df_val_fold["ID"])
        ids_comunes = ids_train.intersection(ids_val)

        print(f"\n===== Fold {fold+1}/{num_folds} =====")
        print(f"Train size: {len(df_train_fold)}")
        print(f"Val size:   {len(df_val_fold)}")
        print(f"IDs en común (data leakage): {len(ids_comunes)}")
        if len(ids_comunes) > 0:
            print(f"IDs duplicados: {list(ids_comunes)}")
    
        #raise

    
        train_ds = dataset.MRIDataset3D(df_train_fold,
                                         transform=dataset.get_train_transforms_lite(SPACING=SPACING,SPATIAL_SIZE=SPATIAL_SIZE) if aug_method=="lite" else dataset.get_train_transforms_hard(SPACING=SPACING,SPATIAL_SIZE=SPATIAL_SIZE))
        val_ds   = dataset.MRIDataset3D(df_val_fold,   transform=dataset.get_val_transforms(SPACING=SPACING,SPATIAL_SIZE=SPATIAL_SIZE))

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=16, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

        model = create_model(model_name,device).to(device)
        #loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
        loss_function = losses.create_loss_function(args.get('loss_function','dice_ce'))
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler   = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.95, patience= args.get('scheduler_patience',5)
        )
        scaler = GradScaler()

        # para guardar el mejor modelo del fold
        best_dice = 0.0
        fold_dir = os.path.join(root_dir, f"fold_{fold+1}")
        os.makedirs(fold_dir, exist_ok=True)

        if use_train:
            torch.save(model.state_dict(), os.path.join(fold_dir, "best_model.pth"))

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

                    if y.ndim == 4:
                        y = y.unsqueeze(1)  # Asegura shape [B, 1, D, H, W]

                    use_mixup_now = use_mixup and np.random.rand() < 0.5  # 50% de los batches

                    if use_mixup_now:
                        # --- Aplicamos MixUp ---
                        lam = np.random.beta(0.4, 0.4)
                        index = torch.randperm(x.size(0)).to(x.device)

                        x_mix = lam * x + (1 - lam) * x[index]
                        y1 = y
                        y2 = y[index]
                    else:
                        # --- No MixUp ---
                        x_mix = x
                        y1 = y
                        y2 = None  # no se usará

                    optimizer.zero_grad()
                    with torch.cuda.amp.autocast(enabled=True):
                        logits = model(x_mix)

                        if use_mixup_now:
                            loss = lam * loss_function(logits, y1) + (1 - lam) * loss_function(logits, y2)
                        else:
                            loss = loss_function(logits, y1)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    epoch_loss += loss.item()

                epoch_loss /= len(train_loader)
                #print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}")

                val_fold_metrics = evaluate_model(model, val_loader, device,prefix="val",loss_function=loss_function,fold=fold,epoch=epoch,full=False)
                val_dice_avg = val_fold_metrics["val_dice_Avg"]
                val_loss = val_fold_metrics["val_loss"]
                test_fold_metrics = evaluate_model(model, test_loader, device,prefix="test",loss_function=loss_function,fold=fold,epoch=epoch,full=False)
                test_loss = test_fold_metrics["test_loss"]
                test_dice_avg = test_fold_metrics["test_dice_Avg"]
                print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_loss:.4f} Val Loss: {val_loss:.4f} Test Loss: {test_loss:.4f}"
                    f" Val Dice Avg: {val_dice_avg:.4f}"
                    f" Test Dice Avg: {test_dice_avg:.4f}")
            
                if use_wandb and run is not None:
                    step_epoch = epoch  # local step for this fold's run
                    log_data= {
                        'epoch': epoch,
                        'train/train_loss': epoch_loss,
                        'val/val_loss': val_loss,
                        'test/test_loss': test_loss,
                    }
                    for key, value in val_fold_metrics.items():
                        log_data[f'val/{key}'] = value
                    for key, value in test_fold_metrics.items():
                        log_data[f'test/{key}'] = value
                    run.log(log_data, step=step_epoch)
                # checkpoint: guarda si supera mejor dice promedio
                scheduler.step(val_dice_avg)
                if val_dice_avg > best_dice:
                    early_stopping_counter = 0  # reset counter si hay mejora
                    best_dice = val_dice_avg
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
        val_fold_metrics = evaluate_model(model, val_loader, device, prefix="val_best", loss_function=loss_function,show=True,full=True)
        results["val_best"].append(val_fold_metrics)
        test_fold_metrics = evaluate_model(model, test_loader, device, prefix="test_best", loss_function=loss_function,show=True,full=True)
        results["test_best"].append(test_fold_metrics)

        
        if use_wandb and run is not None:
            step_epoch = epoch  # local step for this fold's run
            log_data= {
                'epoch': epoch,
            }
            for key, value in val_fold_metrics.items():
                log_data[f'val_best/{key}'] = value
            for key, value in test_fold_metrics.items():
                log_data[f'test_best/{key}'] = value
            run.log(log_data, step=step_epoch)
            run.finish()



    if use_wandb:
        import wandb
        exp_name = args.get('experiment_name', 'rise_experiment')
        run = wandb.init(
            project=os.getenv('PROJECT_WANDB'),
            entity=os.getenv('ENTITY'),
            name=f"{exp_name}_final",
            group=exp_name,
            config=args,
            reinit=True,
        )
    # promediamos sobre folds
    final = {"val_best":{},"test_best":{}}
    for prefix in ["val_best","test_best"]:
        llaves = results[prefix][0].keys()  # keys from first fold
        vals = []
        for results_fold in results[prefix]:
            for llave in llaves:
                if llave not in final[prefix]:
                    final[prefix][llave] = []
                if llave in results_fold:
                    val = results_fold[llave]
                    if isinstance(val, (int, float)):
                        final[prefix][llave].append(val)
                    elif isinstance(val, list):
                        final[prefix][llave].extend(val)
                    else:
                        print(f"Tipo de dato no soportado para {llave}: {type(val)}")
        # ahora promediamos los valores
        for llave, vals in final[prefix].items():
            if len(vals) > 0:
                if isinstance(vals[0], (int, float)):
                    final[prefix][llave] = np.mean(vals)
                elif isinstance(vals[0], list):
                    final[prefix][llave] = [np.mean([v[i] for v in vals]) for i in range(len(vals[0]))]
                else:
                    print(f"Tipo de dato no soportado para promediar {llave}: {type(vals[0])}")
            else:
                final[prefix][llave] = np.nan

        print("\n=== Resultado final promedio de 5 folds ===")
        for metric in ["dice","hd","hd95","assd","rve"]:
            for key in ["L","R","Avg"]:
                llave = f"{prefix}_{metric}_{key}"
                if llave in final:
                    print(f"{llave}  : {final[llave]:.4f}")
    print("final:", final)
    if use_wandb and run is not None:
        step_epoch = 0  # local step for this fold's run
        log_data= {
            'epoch': 0,
        }
        for key, value in final["val_best"].items():
            log_data[f'val_best/{key}'] = value
        for key, value in final["test_best"].items():
            log_data[f'test_best/{key}'] = value
        run.log(log_data, step=0)
    run.finish()
    return final

################################################################################
# EJECUCIÓN PRINCIPAL
################################################################################

# Entrenamiento y evaluación con UNet
"""
    UNET NO CONVERGE , DIFICIL DE ENTRENAR FROM SCRATCH, MEJOR CON PRETRAINED
    python training.py --model_name=unet --device=cuda:2 --root_dir=/data/cristian/projects/med_data/rise-miccai/task-2/3d_models/predictions/unet_mixup_lite/fold_models --num_epochs=5000 --num_folds=5 --use_mixup=1 --aug_method=lite --experiment_name=unet_mixup_lite --lr=1e-4 --weight_decay=1e-5 --early_stopping_patience=60 

    
    python training.py --model_name=unest --device=cuda:5 --root_dir=/data/cristian/projects/med_data/rise-miccai/task-2/3d_models/predictions/unest_mixup_lite/fold_models --num_epochs=5000 --num_folds=5 --use_mixup=1 --aug_method=lite --experiment_name=unest_mixup_lite --lr=1e-4 --weight_decay=1e-5
    
    python training.py --model_name=swinunetr --device=cuda:4 --root_dir=/data/cristian/projects/med_data/rise-miccai/task-2/3d_models/predictions/swinunetr_mixup_lite/fold_models --num_epochs=5000 --num_folds=5 --use_mixup=1 --aug_method=lite --experiment_name=swinunetr_mixup_lite --lr=1e-4 --weight_decay=1e-5

    python training.py --model_name=segresnet --device=cuda:3 --root_dir=/data/cristian/projects/med_data/rise-miccai/task-2/3d_models/predictions/segresnet_mixup_lite/fold_models --num_epochs=5000 --num_folds=5 --use_mixup=1 --aug_method=lite --experiment_name=segresnet_mixup_lite --lr=1e-4 --weight_decay=1e-5

    python training.py --model_name=unest --device=cuda:2 --root_dir=/data/cristian/projects/med_data/rise-miccai/task-2/3d_models/predictions/unest_mixup_hard/fold_models --num_epochs=5000 --num_folds=5 --use_mixup=1 --aug_method=hard --experiment_name=unest_mixup_hard --lr=1e-4 --weight_decay=1e-5


    

    python training.py --model_name=unest --device=cuda:0 --root_dir=/data/cristian/projects/med_data/rise-miccai/task-2/3d_models/predictions/unest_mixup_hard_dice_lovasz_softmax/fold_models --num_epochs=5000 --num_folds=5 --use_mixup=1 --aug_method=hard --experiment_name=unest_mixup_hard_lovasz_softmax --lr=1e-4 --weight_decay=1e-5 --loss_function=lovasz_softmax

    
    python training.py --model_name=unest --device=cuda:1 --root_dir=/data/cristian/projects/med_data/rise-miccai/task-2/3d_models/predictions/unest_mixup_hard_dice_ce_hd95/fold_models --num_epochs=5000 --num_folds=5 --use_mixup=1 --aug_method=hard --experiment_name=unest_mixup_hard_dice_ce_hd95 --lr=1e-4 --weight_decay=1e-5 --loss_function=dice_ce_hd95

    python training.py --model_name=unest --device=cuda:2 --root_dir=/data/cristian/projects/med_data/rise-miccai/task-2/3d_models/predictions/unest_mixup_hard_gdl_focal/fold_models --num_epochs=5000 --num_folds=5 --use_mixup=1 --aug_method=hard --experiment_name=unest_mixup_hard_gdl_focal --lr=1e-4 --weight_decay=1e-5 --loss_function=gdl_focal

    python training.py --model_name=unest --device=cuda:3 --root_dir=/data/cristian/projects/med_data/rise-miccai/task-2/3d_models/predictions/unest_mixup_hard_focal_tversky/fold_models --num_epochs=5000 --num_folds=5 --use_mixup=1 --aug_method=hard --experiment_name=unest_mixup_hard_focal_tversky --lr=1e-4 --weight_decay=1e-5 --loss_function=focal_tversky

    python training.py --model_name=unest --device=cuda:4 --root_dir=/data/cristian/projects/med_data/rise-miccai/task-2/3d_models/predictions/unest_mixup_hard_dice_ce_symmetry/fold_models --num_epochs=5000 --num_folds=5 --use_mixup=1 --aug_method=hard --experiment_name=unest_mixup_hard_dice_ce_symmetry --lr=1e-4 --weight_decay=1e-5 --loss_function=dice_ce_symmetry

    python training.py --model_name=unest --device=cuda:5 --root_dir=/data/cristian/projects/med_data/rise-miccai/task-2/3d_models/predictions/unest_mixup_hard_dice_ce/fold_models --num_epochs=5000 --num_folds=5 --use_mixup=1 --aug_method=hard --experiment_name=unest_mixup_hard_dice_ce --lr=1e-4 --weight_decay=1e-5 --loss_function=dice_ce

            
"""




import argparse
from sklearn.model_selection import StratifiedKFold

def parse_args():
    parser = argparse.ArgumentParser(description="Train 3D segmentation model for LISA Challenge")

    parser.add_argument('--model_name', type=str, default="swinunetr", choices=["swinunetr", "unet","segresnet", "unest"],
                        help="Model architecture to use")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size for training")
    parser.add_argument('--early_stopping_patience', type=int, default=80, help="Batch size for training")
    parser.add_argument('--experiment_name', type=str, default="base", help="Batch size for training")
    parser.add_argument('--scheduler_patience', type=int, default=25, help="scheduler_patience")

    parser.add_argument('--use_mixup', type=int, default=0, help="Batch size for training")
    
    parser.add_argument('--num_epochs', type=int, default=5000, help="Number of training epochs")
    parser.add_argument('--aug_method', type=str, default="lite", help="Number of training epochs")

    parser.add_argument('--loss_function', type=str, default="dice_ce", help="Number of training epochs")

    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=1e-5, help="Weight decay (L2 regularization)")
    parser.add_argument('--device', type=str, default="cuda:0", help="Device to use for training")
    parser.add_argument('--num_folds', type=int, default=5, help="Number of cross-validation folds")
    parser.add_argument('--root_dir', type=str, required=True, help="Directory to save models and logs")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    load_dotenv()

    
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
        root_dir=args.root_dir,
        aug_method=args.aug_method,
        use_mixup = args.use_mixup,
        early_stopping_patience=args.early_stopping_patience,
        args= vars(args).copy()
    )

    print("Métricas finales:", final_metrics)
    print("Métricas finales:", final_metrics)
    print("hello")

