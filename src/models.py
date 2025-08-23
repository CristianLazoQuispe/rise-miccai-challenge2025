
from monai.networks.nets import UNet
from .seg_models.unest import UNesT133Adapter
from .seg_models.swinunetr import AdaptedSwinUNETR
from .seg_models.segresnet import AdaptedSegResNetV2
import torch
import torch.nn as nn
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
        return UNesT133Adapter(device=device, num_classes = 3)

    elif model_name.lower() == "swinunetr":
        return AdaptedSwinUNETR(
                img_size=(96, 96, 96),
                in_channels=1,
                num_classes=3,
                freeze_stage=0,  # Puedes controlar qué congelar (0: nada, 1: todo, 2: encoder NO FUNCIONO)
                pretrained=True,
                device=device,
            )

    elif model_name.lower() == "segresnet":
        return AdaptedSegResNetV2(
        pretrained=True,
        freeze_stage=0,  # Puedes controlar qué congelar (0: nada, 1: todo, 2: encoder NO FUNCIONO)
        bottleneck=32,
        num_classes=3,
        device=device
    )
    else:
        raise ValueError(f"Modelo no soportado: {model_name}")
