import numpy as np
from monai.data import Dataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
    CropForegroundd, Spacingd, ScaleIntensityRanged, Resized,
    RandFlipd, RandRotate90d, RandAffined, RandGaussianNoised,
    RandAdjustContrastd, EnsureTyped, RandShiftIntensityd,
    RandScaleIntensityd, RandGaussianSmoothd, Lambdad
)
import pandas as pd

def is_positive_background(img):
    """Returns boolean where positive values are True"""
    return img > 0.5

def get_train_transforms_hippocampus(SPACING=(1.0, 1.0, 1.0), SPATIAL_SIZE=(96, 96, 96)):
    """Augmentaciones conservadoras para hipocampo pequeño"""
    return Compose([
        LoadImaged(keys=["image","label"]),
        EnsureChannelFirstd(keys=["image","label"]),
        Orientationd(keys=["image","label"], axcodes="RAS"),
        
        # Normalización robusta
        ScaleIntensityRanged(
            keys=["image"], 
            a_min=0.0, a_max=16.0,  # Ajustado a tu rango de datos
            b_min=0.0, b_max=1.0, 
            clip=True
        ),
        
        # Crop centrado en foreground
        CropForegroundd(
            keys=["image","label"], 
            source_key="image",
            margin=20,  # Margen alrededor del cerebro
            allow_smaller=True
        ),
        
        # Spacing isotrópico
        Spacingd(
            keys=["image","label"], 
            pixdim=SPACING, 
            mode=("bilinear","nearest")
        ),
        
        # Resize al tamaño objetivo
        Resized(
            keys=["image"], 
            spatial_size=SPATIAL_SIZE,
            mode="trilinear", 
            align_corners=False
        ),
        Resized(
            keys=["label"], 
            spatial_size=SPATIAL_SIZE,
            mode="nearest"
        ),
        
        # === AUGMENTACIONES CONSERVADORAS ===
        
        # NO flip en eje lateral (preserva L/R)
        RandFlipd(keys=["image","label"], prob=0.3, spatial_axis=0),  # Sagital OK
        RandFlipd(keys=["image","label"], prob=0.3, spatial_axis=1),  # Coronal OK
        # NO flip en eje 2 (axial) para no confundir L/R
        
        # Rotaciones pequeñas solo
        RandAffined(
            keys=["image","label"],
            prob=0.4,
            rotate_range=(0.1, 0.05, 0.05),  # Max 5.7° en x, 2.8° en y,z
            translate_range=(5, 5, 5),  # Max 5 voxels shift
            scale_range=(0.05, 0.05, 0.05),  # ±5% scaling
            padding_mode="border",  # Importante: no zeros
            mode=("bilinear","nearest")
        ),
        
        # Augmentaciones de intensidad (no afectan posición)
        RandShiftIntensityd(keys=["image"], prob=0.3, offsets=0.1),
        RandScaleIntensityd(keys=["image"], prob=0.3, factors=0.1),
        RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.01),
        RandGaussianSmoothd(
            keys=["image"], 
            prob=0.2,
            sigma_x=(0.5, 1.0),
            sigma_y=(0.5, 1.0),
            sigma_z=(0.5, 1.0)
        ),
        
        EnsureTyped(keys=["image","label"], track_meta=True),
    ])

def get_val_transforms(SPACING=(1.0, 1.0, 1.0), SPATIAL_SIZE=(96, 96, 96)):
    """Sin augmentación para validación - mantiene inversión"""
    return Compose([
        LoadImaged(keys=["image","label"]),
        EnsureChannelFirstd(keys=["image","label"]),
        Orientationd(keys=["image","label"], axcodes="RAS"),
        
        ScaleIntensityRanged(
            keys=["image"], 
            a_min=0.0, a_max=16.0,
            b_min=0.0, b_max=1.0, 
            clip=True
        ),
        
        CropForegroundd(
            keys=["image","label"], 
            source_key="image",
            margin=20,
            allow_smaller=True
        ),
        
        Spacingd(
            keys=["image","label"], 
            pixdim=SPACING, 
            mode=("bilinear","nearest")
        ),
        
        Resized(keys=["image"], spatial_size=SPATIAL_SIZE, mode="trilinear"),
        Resized(keys=["label"], spatial_size=SPATIAL_SIZE, mode="nearest"),
        
        EnsureTyped(keys=["image","label"], track_meta=True),
    ])

class MRIDataset3D(Dataset):
    """Dataset que preserva metadata para inverse transforms"""
    def __init__(self, df: pd.DataFrame, transform=None, is_train=True):
        data = []
        for _, r in df.iterrows():
            item = {
                "image": r["filepath"],
                "image_path": r["filepath"]  # Preserva path para inference
            }
            if "filepath_label" in r and pd.notna(r["filepath_label"]):
                item["label"] = r["filepath_label"]
            if "ID" in r:
                item["ID"] = r["ID"]
            data.append(item)
        super().__init__(data, transform=transform)
        self.is_train = is_train