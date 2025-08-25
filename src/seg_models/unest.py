from __future__ import annotations

import os
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

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
            from .unets_scripts.networks.unest_base_patch_4 import UNesT
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



