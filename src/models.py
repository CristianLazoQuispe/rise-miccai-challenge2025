"""
Model definitions for the RISE‑MICCAI LISA 2025 hippocampus segmentation challenge.

This module defines three different 3D segmentation networks:

* **UNesT133Adapter** – a wrapper around the UNesT transformer with a
  133‑channel output head from the official MONAI whole brain model.  A
  1×1×1 convolution adapts the 133 feature channels to 3 semantic
  classes.  The backbone can optionally be frozen to reduce over‑fitting.

* **SwinUNETR** – a Swin Transformer based U‑Net with windowed
  self‑attention.  It is a strong baseline for 3D medical image
  segmentation.  We expose a simple factory function to instantiate
  the network and optionally load pretrained weights.

* **DynUNet** – a dynamic U‑Net with flexible depth and width, based on
  convolutional blocks.  It is well suited for volumetric data and
  provides an alternative to transformer architectures.

All factories return PyTorch modules with 3 output channels (0=background,
1=left hippocampus, 2=right hippocampus) and 1 input channel.  Pretrained
weights can be loaded for UNesT and SwinUNETR via the Hugging Face
repository if desired.

Note: To run these models you need to install the following dependencies
outside of this sandbox environment:

- ``torch`` for tensor operations and training
- ``monai`` for network architectures and transforms
- ``huggingface_hub`` if you wish to download pretrained weights

"""

from __future__ import annotations

import os
from typing import Optional, Dict, Any

import torch
import torch.nn as nn

try:
    # Hugging Face helper for downloading pretrained weights
    from huggingface_hub import hf_hub_download
except Exception:
    hf_hub_download = None  # type: ignore

try:
    # MONAI network definitions
    from monai.networks.nets import SwinUNETR, DynUNet
except Exception:
    SwinUNETR = None  # type: ignore
    DynUNet = None  # type: ignore


def _to_device(model: nn.Module, device: Optional[str] = None) -> nn.Module:
    """Utility to move a model to the specified device if provided."""
    if device is not None:
        model = model.to(device)
    return model


class UNesT133Adapter(nn.Module):
    """UNesT transformer with an adapter for 3 output classes.

    This class instantiates the UNesT architecture with 133 output
    channels (as used in the MONAI whole brain model) and inserts an
    additional 1×1×1 convolution to map those channels down to 3.  The
    pretrained weights can be downloaded from the Hugging Face hub.  You
    may freeze the backbone to perform linear probing on your target
    dataset.
    """

    def __init__(
        self,
        repo_id: str = "MONAI/wholeBrainSeg_Large_UNEST_segmentation",
        filename: str = "models/model.pt",
        cache_dir: Optional[str] = None,
        freeze_backbone: bool = False,
        bottleneck: int = 32,
        dropout: float = 0.0,
        num_classes: int = 3,
        verbose: bool = False,
        device: Optional[str] = None,
    ) -> None:
        super().__init__()
        # Import the UNesT architecture from the user's codebase
        try:
            from .seg_models.unets_scripts.networks.unest_base_patch_4 import UNesT
        except Exception as e:
            raise ImportError(
                "Failed to import UNesT. Ensure that 'unest_base_patch_4.py' is on the Python path."
            ) from e
        # Build the backbone with 133 output channels
        self.backbone = UNesT(
            in_channels=1,
            out_channels=133,
            img_size=(96, 96, 96),
            num_heads=(4, 8, 16),
            depths=(2, 2, 8),
            embed_dim=(128, 256, 512),
            patch_size=4,
        )
        # Load pretrained weights if available
        if hf_hub_download is None:
            raise RuntimeError(
                "huggingface_hub is required to download pretrained UNesT weights. Install it via pip."
            )
        ckpt_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=cache_dir,
            local_dir_use_symlinks=False,
        )
        state = torch.load(ckpt_path, map_location="cpu")
        state_dict = state.get("model", state)
        # Remove the original classification head weights if present
        filtered_state = {k: v for k, v in state_dict.items() if not k.startswith("out.")}
        missing, unexpected = self.backbone.load_state_dict(filtered_state, strict=False)
        if verbose:
            print("[UNesT133Adapter] missing keys:", missing)
            print("[UNesT133Adapter] unexpected keys:", unexpected)
        # Optionally freeze the backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        # Define the adapter: bottleneck→ReLU→conv to num_classes
        if bottleneck and bottleneck > 0:
            self.adapter = nn.Sequential(
                nn.Conv3d(133, bottleneck, kernel_size=1, bias=False),
                nn.GroupNorm(num_groups=8, num_channels=bottleneck),
                nn.ReLU(inplace=True),
                nn.Dropout3d(p=dropout) if dropout > 0 else nn.Identity(),
                nn.Conv3d(bottleneck, num_classes, kernel_size=1, bias=True),
            )
        else:
            self.adapter = nn.Conv3d(133, num_classes, kernel_size=1, bias=True)
        # Initialise adapter weights
        for m in self.adapter.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        # Move to device if specified
        _to_device(self, device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits_133 = self.backbone(x)
        logits_3 = self.adapter(logits_133)
        return logits_3


def build_swin_unetr(
    num_classes: int = 3,
    img_size: Tuple[int, int, int] = (96, 96, 96),
    feature_size: int = 48,
    pretrained: bool = False,
    pretrained_checkpoint: Optional[str] = None,
    device: Optional[str] = None,
) -> nn.Module:
    """Instantiate a SwinUNETR model for 3D segmentation.

    Parameters
    ----------
    num_classes : int, optional
        Number of output segmentation classes.  Default is 3 (background,
        left hippocampus, right hippocampus).
    img_size : tuple of ints, optional
        Input patch size for the network.  Must match the spatial
        dimensions of the training volumes.  Default is (96, 96, 96).
    feature_size : int, optional
        Base feature dimension for the Swin transformer.  Increasing
        ``feature_size`` increases model capacity.  Default is 48.
    pretrained : bool, optional
        If ``True``, attempts to load pretrained weights from
        ``pretrained_checkpoint``.  Default is ``False``.
    pretrained_checkpoint : str, optional
        Path to a checkpoint file for SwinUNETR.  This parameter is
        ignored if ``pretrained`` is ``False``.  You may obtain
        pretrained weights from the MONAI Model Zoo or the Hugging Face
        hub.  The number of output channels in the checkpoint must
        match ``num_classes`` or the load will be partial.
    device : str, optional
        Device identifier (e.g. 'cuda:0') on which to place the model.

    Returns
    -------
    torch.nn.Module
        A SwinUNETR instance ready for training or inference.
    """

    # 📦 Descargar pesos preentrenados
    model_path = hf_hub_download(
        repo_id="MONAI/swin_unetr_btcv_segmentation",
        filename="models/model.pt",
        local_dir="/data/cristian/projects/med_data/rise-miccai/pretrained_models/swin_unetr_btcv_segmentation/",
        local_dir_use_symlinks=False
    )

    # 🧠 Cargar modelo compatible con Swin (NO UNETR)
    model = SwinUNETR(
        in_channels=1,
        out_channels=14,  # Debe coincidir con preentrenado BTCV
        #img_size=(96, 96, 96),
        feature_size=48,
        use_checkpoint=True
    )

    # ✅ Cargar directamente el state_dict (no usar load_from aquí)
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)


    class SegHead(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.head = nn.Sequential(
                nn.Conv3d(in_ch, in_ch // 2, kernel_size=3, padding=1),
                nn.BatchNorm3d(in_ch // 2),
                nn.ReLU(inplace=True),
                nn.Dropout3d(0.2),
                nn.Conv3d(in_ch // 2, out_ch, kernel_size=1)
            )

        def forward(self, x):
            return self.head(x)

    class MySwinUNETR(nn.Module):
        def __init__(self, out_channels=3):
            super().__init__()
            self.encoder = model
            self.out = SegHead(in_ch=768, out_ch=out_channels)

        def forward(self, x):
            features = self.encoder(x)
            return self.out(features)

    model = MySwinUNETR(out_channels=num_classes)
    return _to_device(model, device)


def build_dynunet(
    num_classes: int = 3,
    in_channels: int = 1,
    spatial_dims: int = 3,
    filters: Optional[Sequence[int]] = None,
    kernel_size: Optional[Sequence[int]] = None,
    strides: Optional[Sequence[int]] = None,
    upsample_kernel_size: Optional[Sequence[int]] = None,
    deep_supervision: bool = False,
    device: Optional[str] = None,
) -> nn.Module:
    """Instantiate a dynamic U‑Net (DynUNet) for 3D segmentation.

    The DynUNet architecture allows flexible specification of depth,
    kernel sizes and strides.  The defaults provided here follow a
    typical 4‑level 3D U‑Net configuration suitable for volumes of
    approximately 96×96×96 voxels.

    Parameters
    ----------
    num_classes : int, optional
        Number of output segmentation classes.  Default is 3.
    in_channels : int, optional
        Number of input channels.  Default is 1.
    spatial_dims : int, optional
        Number of spatial dimensions (always 3 for volumetric data).  Default is 3.
    filters : sequence of ints, optional
        Number of convolutional filters at each level.  If ``None``, uses
        ``[32, 64, 128, 256]``.
    kernel_size : sequence of ints, optional
        Convolution kernel sizes per level.  If ``None``, uses ``[3, 3, 3, 3]``.
    strides : sequence of ints, optional
        Downsampling strides between levels.  If ``None``, uses
        ``[1, 2, 2, 2]``; the first stride must be 1 so that the input
        resolution is preserved at the first level.
    upsample_kernel_size : sequence of ints, optional
        Kernel sizes for transposed convolutions used to upsample feature
        maps in the decoder.  If ``None``, uses ``[2, 2, 2]`` for each
        decoder level.
    deep_supervision : bool, optional
        Whether to include auxiliary output heads at intermediate levels
        for deep supervision.  Default is ``False``.
    device : str, optional
        Device identifier (e.g. 'cuda:0') on which to place the model.

    Returns
    -------
    torch.nn.Module
        A DynUNet instance ready for training or inference.
    """
    if DynUNet is None:
        raise ImportError("MONAI is required to build DynUNet.")
    if filters is None:
        filters = (32, 64, 128, 256)
    if kernel_size is None:
        kernel_size = (3, 3, 3, 3)
    if strides is None:
        # First stride must be 1
        strides = (1, 2, 2, 2)
    if upsample_kernel_size is None:
        upsample_kernel_size = (2, 2, 2)
    model = DynUNet(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=num_classes,
        kernel_size=kernel_size,
        strides=strides,
        upsample_kernel_size=upsample_kernel_size,
        filters=filters,
        deep_supervision=deep_supervision,
    )
    return _to_device(model, device)


def get_model(
    name: str,
    device: Optional[str] = None,
    **kwargs: Any,
) -> nn.Module:
    """Factory function to construct one of the available 3D segmentation models.

    The ``name`` parameter selects between ``"unest"``, ``"swinunetr"``
    and ``"dynunet"``.  Additional keyword arguments are forwarded to
    the respective builder functions.

    Parameters
    ----------
    name : str
        Name of the model to construct.  Supported values are
        ``"unest"``, ``"swinunetr"`` and ``"dynunet"``.
    device : str, optional
        Device on which to place the model (e.g. ``"cuda:0"``).  If
        ``None``, the model remains on the default device.
    **kwargs : dict
        Additional arguments passed to the model constructor.

    Returns
    -------
    torch.nn.Module
        Instantiated segmentation model.
    """
    name = name.lower()
    if name == "unest":
        return UNesT133Adapter(device=device, **kwargs)
    if name == "swinunetr":
        return build_swin_unetr(device=device, **kwargs)
    if name == "dynunet":
        return build_dynunet(device=device, **kwargs)
    raise ValueError(
        "Unknown model name '{}'. Available models: unest, swinunetr, dynunet.".format(name)
    )



