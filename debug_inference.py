# debug_cascade.py
import torch
import numpy as np
from src.models import create_model
from src.cascade_utils import get_roi_bbox_from_logits, crop_to_bbox, resize_volume
from monai.transforms import Compose, LoadImage, EnsureChannelFirst, Orientation, ScaleIntensityRange, Spacing, Resize

def debug_cascade(base_path, fine_path, test_image_path, device="cuda"):
    """Debug específico del cascade"""
    
    # Cargar modelos
    base_model = create_model("eff-b2", device).to(device)
    base_model.load_state_dict(torch.load(base_path, map_location=device))
    base_model.eval()
    
    fine_model = create_model("eff-b2", device).to(device)
    fine_model.load_state_dict(torch.load(fine_path, map_location=device))
    fine_model.eval()
    
    # Procesar imagen
    transforms = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        Orientation(axcodes="RAS"),
        ScaleIntensityRange(a_min=0.0, a_max=16.0, b_min=0.0, b_max=1.0, clip=True),
        Spacing(pixdim=(1.0, 1.0, 1.0)),
        Resize(spatial_size=(192, 192, 192))
    ])
    
    img = transforms(test_image_path)
    img_tensor = torch.tensor(img).unsqueeze(0).to(device)
    
    print("="*60)
    print("STEP 1: BASE MODEL")
    print("="*60)
    
    # Base prediction
    with torch.no_grad():
        base_logits = base_model(img_tensor)
        base_probs = torch.softmax(base_logits, dim=1)
        base_pred = torch.argmax(base_probs, dim=1)
    
    print(f"Base prediction: L={(base_pred==1).sum()}, R={(base_pred==2).sum()}")
    
    # Encontrar ROIs
    for thr in [0.2, 0.1, 0.05, 0.01]:
        bboxes = get_roi_bbox_from_logits(base_logits, thr=thr, margin=20)
        if bboxes and bboxes[0] is not None:
            print(f"ROI found with threshold {thr}")
            break
    
    if not bboxes or bboxes[0] is None:
        print("ERROR: No ROI found!")
        return
    
    print(f"BBox: {bboxes[0]}")
    z0, y0, x0, z1, y1, x1 = bboxes[0]
    print(f"ROI size: {z1-z0} x {y1-y0} x {x1-x0}")
    
    print("\n" + "="*60)
    print("STEP 2: FINE MODEL ON ROI")
    print("="*60)
    
    # Crop ROI
    img_roi = crop_to_bbox(img_tensor, bboxes[0])
    print(f"Cropped ROI shape: {img_roi.shape}")
    print(f"ROI value range: [{img_roi.min():.3f}, {img_roi.max():.3f}]")
    
    # Resize ROI si es necesario
    roi_size_fine = (192, 192, 192)  # O el tamaño que uses
    if img_roi.shape[2:] != roi_size_fine:
        print(f"Resizing ROI from {img_roi.shape[2:]} to {roi_size_fine}")
        img_roi = resize_volume(img_roi, roi_size_fine, mode="trilinear")
    
    # Fine prediction en ROI
    with torch.no_grad():
        fine_logits = fine_model(img_roi)
        fine_probs = torch.softmax(fine_logits, dim=1)
        fine_pred = torch.argmax(fine_probs, dim=1)
    
    print(f"Fine logits shape: {fine_logits.shape}")
    print(f"Fine logits range: [{fine_logits.min():.3f}, {fine_logits.max():.3f}]")
    print(f"Fine prediction in ROI: L={(fine_pred==1).sum()}, R={(fine_pred==2).sum()}")
    
    # Verificar probabilidades
    print(f"Max prob per class in ROI: BG={fine_probs[0,0].max():.3f}, L={fine_probs[0,1].max():.3f}, R={fine_probs[0,2].max():.3f}")
    
    # Ver si hay alguna predicción no-background
    if (fine_pred > 0).sum() == 0:
        print("\n⚠️ WARNING: Fine model predicts only background in ROI!")
        print("Debugging probabilities...")
        # Encontrar los top-5 valores más altos para cada clase
        for c in range(3):
            top5 = torch.topk(fine_probs[0, c].flatten(), 5)
            print(f"  Class {c}: top 5 probs = {top5.values.cpu().numpy()}")
    
    return base_pred, fine_pred

# Ejecutar
debug_cascade(
    base_path="/data/cristian/projects/med_data/rise-miccai/task-2/3d_models/predictions/eff-dice_ce_balanced/fold_models/fold_1/best_model.pth",
    fine_path="/data/cristian/projects/med_data/rise-miccai/task-2/3d_models/predictions/eff-dice_ce_balanced/fold_models/fold_1/best_fine_model.pth",
    test_image_path="/data/cristian/projects/med_data/rise-miccai/task-2-val/552/158977552/LISA_VALIDATION_0001_ciso.nii.gz"
)