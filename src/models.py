
from monai.networks.nets import UNet
from .seg_models.unest import UNesT133Adapter
from .seg_models.unestpn import UNesT133PANetASPPAdapter
from .seg_models.swinunetr import AdaptedSwinUNETR
from .seg_models.segresnet import AdaptedSegResNetV2
import torch
import torch.nn as nn
import segmentation_models_pytorch_3d as smp
import torch

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

    elif model_name.lower() == "dpn131":

        return smp.Unet(
            encoder_name="dpn131", # choose encoder, e.g. resnet34
            in_channels=1,                  # model input channels (1 for gray-scale volumes, 3 for RGB, etc.)
            classes=3,                      # model output channels (number of classes in your dataset)
        )
    
    elif model_name.lower() == "resnet152":

        return smp.Unet(
            encoder_name="resnet152", # choose encoder, e.g. resnet34
            in_channels=1,                  # model input channels (1 for gray-scale volumes, 3 for RGB, etc.)
            classes=3,                      # model output channels (number of classes in your dataset)
        )
    	

    elif model_name.lower() == "efficientnet-b7":

        return smp.Unet(
            encoder_name="efficientnet-b7", # choose encoder, e.g. resnet34
            in_channels=1,                  # model input channels (1 for gray-scale volumes, 3 for RGB, etc.)
            classes=3,                      # model output channels (number of classes in your dataset)
        )


    elif model_name.lower() == "unest":
        return UNesT133Adapter(device=device, num_classes = 3)

    elif model_name.lower() == "unestpn":
        return UNesT133PANetASPPAdapter(device=device, num_classes = 3)

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
