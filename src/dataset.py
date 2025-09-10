
from monai.data import Dataset, decollate_batch
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
    CropForegroundd, Spacingd, ScaleIntensityRanged, Resized,
    RandFlipd, RandRotate90d, RandAffineD, RandBiasFieldd, RandGaussianNoised,
    RandAdjustContrastd, EnsureTyped, OneOf, AsDiscreted, ResampleToMatchd,RandRotated,RandAffined,Lambdad,
)
import pandas as pd


def is_positive_background(img):
    """
    Returns a boolean version of `img` where the positive values are converted into True, the other values are False.
    """
    return img > 0.5



"""    
        # Ejemplo de augmentación: flips y rotaciones aleatorias
        OneOf([
            RandFlipd(keys=["image","label"], prob=0.5, spatial_axis=1),  #  si usar
            RandFlipd(keys=["image","label"], prob=0.5, spatial_axis=2),  # si usar
            RandRotate90d(keys=["image","label"], prob=0.5, max_k=3), # si usar
        ]),


        OneOf([
            RandRotated( # si usar RandRotated,RandAffined
                keys=["image","label"], prob=0.5,
                range_z=(-0.8, 0.8),
                mode=("bilinear","nearest")),
        ]),
"""

def get_train_transforms_lite(SPACING,SPATIAL_SIZE):
    return Compose([
        LoadImaged(keys=["image","label"]),
        EnsureChannelFirstd(keys=["image","label"]),
        Orientationd(keys=["image","label"], axcodes="RAS"),
        CropForegroundd(keys=["image","label"], source_key="image", allow_smaller=False,select_fn=is_positive_background),
        Spacingd(keys=["image","label"], pixdim=SPACING, mode=("bilinear","nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=15.0, b_min=0.0, b_max=1.0, clip=True),
        #
        # ---- Reescalado correcto ----
        Resized(keys=["image"], spatial_size=SPATIAL_SIZE,
                mode="trilinear", align_corners=False, anti_aliasing=True),
        Resized(keys=["label"], spatial_size=SPATIAL_SIZE,
                mode="nearest", anti_aliasing=False),        


        # 🔥 Binarizar el label
        Lambdad(keys="label", func=lambda x: (x > 0).astype(x.dtype)),
        
        # Ejemplo de augmentación: flips y rotaciones aleatorias
        OneOf([
            RandFlipd(keys=["image","label"], prob=0.5, spatial_axis=1),  #  si usar
            RandFlipd(keys=["image","label"], prob=0.5, spatial_axis=2),  # si usar
            RandRotate90d(keys=["image","label"], prob=0.5, max_k=3), # si usar
        ]),


        OneOf([
            RandRotated( # si usar RandRotated,RandAffined
                keys=["image","label"], prob=0.5,
                range_z=(-0.8, 0.8),
                mode=("bilinear","nearest")),
            
            RandAffined( # si usar
                keys=["image","label"], prob=0.5,
                rotate_range=(0.4,0,0),
                translate_range=(0, 20 , 20),
                scale_range=(0.10,0.10,0.10),
                padding_mode="zeros",
                mode=("bilinear","nearest"),
            ),
        ]),

        OneOf([
            RandBiasFieldd(keys=["image"], prob=0.5, coeff_range=(0.0,0.05)),
            RandGaussianNoised(keys=["image"], prob=0.5, std=0.01),
            RandAdjustContrastd(keys=["image"], prob=0.5, gamma=(0.7,1.5)),
        ]),


        EnsureTyped(keys=["image","label"], track_meta=True),
    ])


def get_train_transforms_lite1(SPACING,SPATIAL_SIZE):
    return Compose([
        LoadImaged(keys=["image","label"]),
        EnsureChannelFirstd(keys=["image","label"]),
        Orientationd(keys=["image","label"], axcodes="RAS"),
        CropForegroundd(keys=["image","label"], source_key="image", allow_smaller=False,select_fn=is_positive_background),
        Spacingd(keys=["image","label"], pixdim=SPACING, mode=("bilinear","nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=15.0, b_min=0.0, b_max=1.0, clip=True),
        #
        # ---- Reescalado correcto ----
        Resized(keys=["image"], spatial_size=SPATIAL_SIZE,
                mode="trilinear", align_corners=False, anti_aliasing=True),
        Resized(keys=["label"], spatial_size=SPATIAL_SIZE,
                mode="nearest", anti_aliasing=False),     
        # 🔥 Binarizar el label
        Lambdad(keys="label", func=lambda x: (x > 0).astype(x.dtype)),


        RandFlipd(keys=["image","label"], prob=0.5, spatial_axis=1),  #  si usar
        RandFlipd(keys=["image","label"], prob=0.5, spatial_axis=2),  # si usar


        RandRotated( # si usar RandRotated,RandAffined
            keys=["image","label"], prob=0.5,
            range_z=(-0.8, 0.8),
            mode=("bilinear","nearest")),

        RandAffined( # si usar
            keys=["image","label"], prob=0.5,
            rotate_range=(0.6,0,0),
            translate_range=(0, 40,40),
            scale_range=(0.20,0.20,0.20),
            padding_mode="zeros",
            mode=("bilinear","nearest"),
        ),

        #OneOf([
        RandBiasFieldd(keys=["image"], prob=0.5, coeff_range=(0.1,0.5)),
        RandGaussianNoised(keys=["image"], prob=0.5, std=0.01),
        RandAdjustContrastd(keys=["image"], prob=0.5, gamma=(0.1,2)),
        #]),


        EnsureTyped(keys=["image","label"], track_meta=True),
    ])


def get_train_transforms_hard(SPACING,SPATIAL_SIZE):
    return Compose([
        LoadImaged(keys=["image","label"]),
        EnsureChannelFirstd(keys=["image","label"]),
        Orientationd(keys=["image","label"], axcodes="RAS"),
        CropForegroundd(keys=["image","label"], source_key="image", allow_smaller=False,select_fn=is_positive_background),
        Spacingd(keys=["image","label"], pixdim=SPACING, mode=("bilinear","nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=15.0, b_min=0.0, b_max=1.0, clip=True),
        #
        # ---- Reescalado correcto ----
        Resized(keys=["image"], spatial_size=SPATIAL_SIZE,
                mode="trilinear", align_corners=False, anti_aliasing=True),
        Resized(keys=["label"], spatial_size=SPATIAL_SIZE,
                mode="nearest", anti_aliasing=False),   
        # 🔥 Binarizar el label
        Lambdad(keys="label", func=lambda x: (x > 0).astype(x.dtype)),

        # Ejemplo de augmentación: flips y rotaciones aleatorias
        #OneOf([
        RandFlipd(keys=["image","label"], prob=0.5, spatial_axis=0),  #  si usar
        RandFlipd(keys=["image","label"], prob=0.5, spatial_axis=1),  #  si usar
        RandFlipd(keys=["image","label"], prob=0.5, spatial_axis=2),  # si usar
        RandRotate90d(keys=["image","label"], prob=0.5, max_k=3), # si usar
        #]),
        OneOf([

            RandRotated( # si usar RandRotated,RandAffined
                keys=["image","label"], prob=0.5,
                range_z=(-0.8, 0.8),
                mode=("bilinear","nearest")),
            
            RandAffined( # si usar
                keys=["image","label"], prob=0.5,
                rotate_range=(0.4,0.4,0.4),
                translate_range=(20, 20 , 20),
                scale_range=(0.10,0.10,0.10),
                padding_mode="zeros",
                mode=("bilinear","nearest"),
            ),
        ]),

        #OneOf([
        RandBiasFieldd(keys=["image"], prob=0.5, coeff_range=(0.1,0.5)),
        RandGaussianNoised(keys=["image"], prob=0.5, std=0.01),
        RandAdjustContrastd(keys=["image"], prob=0.5, gamma=(0.1,2)),
        #]),


        EnsureTyped(keys=["image","label"], track_meta=True),
    ])


def get_val_transforms(SPACING,SPATIAL_SIZE):
    return Compose([
        LoadImaged(keys=["image","label"]),
        EnsureChannelFirstd(keys=["image","label"]),
        Orientationd(keys=["image","label"], axcodes="RAS"),
        CropForegroundd(keys=["image","label"], source_key="image", allow_smaller=False,select_fn=is_positive_background),
        Spacingd(keys=["image","label"], pixdim=SPACING, mode=("bilinear","nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=15.0, b_min=0.0, b_max=1.0, clip=True),
        #
        # ---- Reescalado correcto ----
        Resized(keys=["image"], spatial_size=SPATIAL_SIZE,
                mode="trilinear", align_corners=False, anti_aliasing=True),
        Resized(keys=["label"], spatial_size=SPATIAL_SIZE,
                mode="nearest", anti_aliasing=False),        
        # 🔥 Binarizar el label
        Lambdad(keys="label", func=lambda x: (x > 0).astype(x.dtype)),

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
