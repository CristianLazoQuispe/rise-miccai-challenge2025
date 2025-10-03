
from monai.networks.nets import UNet
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
    	

    elif model_name.lower() == "eff-b7":

        return smp.Unet(
            encoder_name="efficientnet-b7", # choose encoder, e.g. resnet34
            in_channels=1,                  # model input channels (1 for gray-scale volumes, 3 for RGB, etc.)
            encoder_weights=None,  # NO pretrained weights - they're for 2D ImageNet
            classes=3,                      # model output channels (number of classes in your dataset)
        )


    elif model_name.lower() == "eff-b2":

        return smp.Unet(
            encoder_name="efficientnet-b1", # choose encoder, e.g. resnet34
            in_channels=1,                  # model input channels (1 for gray-scale volumes, 3 for RGB, etc.)
            encoder_weights=None,  # NO pretrained weights - they're for 2D ImageNet
            classes=3,                      # model output channels (number of classes in your dataset)
        )
    else:
        raise ValueError(f"Modelo no soportado: {model_name}")
