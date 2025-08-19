"""
models.py — 3D segmentation (hippocampus) with an adapter over a 133-class UNesT.

- UNesT133Adapter: mantiene la cabeza de 133 canales del checkpoint y añade
  un adaptador 1x1x1 (o cuello 133→32→3) para producir 3 clases (0,1,2).
  Puedes congelar el backbone y entrenar solo el adaptador para evitar
  sobreajuste y acelerar.

- SegResNet (MONAI) con 3 salidas.
- Autoencoder 3D ligero con 3 salidas.

Requisitos en tu entorno (no en este sandbox):
    torch, monai, huggingface_hub

Ajusta el import de UNesT si tu ruta difiere.
"""

from __future__ import annotations
from typing import Optional, Dict, Any

import torch
import torch.nn as nn

# ==== HuggingFace (para pesos UNesT) ====
try:
    from huggingface_hub import hf_hub_download
except Exception:  # pragma: no cover
    hf_hub_download = None  # type: ignore

# ==== MONAI (para SegResNet) ====
try:
    from monai.networks.nets import SegResNet
except Exception:  # pragma: no cover
    SegResNet = None  # type: ignore


# ---------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------
def _to_device(m: nn.Module, device: Optional[str]) -> nn.Module:
    if device is not None:
        m = m.to(device)
    return m


# ---------------------------------------------------------------------
# UNesT con cabeza original de 133 + Adaptador a 3 clases
# ---------------------------------------------------------------------
class UNesT133Adapter(nn.Module):
    """
    Envuelve UNesT (out=133) y proyecta a 3 clases con un adaptador 1x1x1.

    forward(x):
        logits133 = unest(x)           # [B, 133, D, H, W]
        logits3   = adapter(logits133) # [B,   3, D, H, W]
    """
    def __init__(
        self,
        repo_id: str = "MONAI/wholeBrainSeg_Large_UNEST_segmentation",
        filename: str = "models/model.pt",
        cache_dir: Optional[str] = None,
        device: Optional[str] = None,
        freeze_backbone: bool = False,
        bottleneck: int = 32,      # 0 => proyección lineal 133->3 directa
        dropout: float = 0.0,
        num_classes  : int = 3,
        verbose: bool = True,
    ) -> None:
        super().__init__()

        # IMPORTA TU ARQUITECTURA UNesT (out=133). NO la modifiques.
        # Ruta típica basada en lo que compartiste:
        from .seg_models.unets_scripts.networks.unest_base_patch_4 import UNesT

        # 1) Construye UNesT exactamente con out_channels=133
        self.backbone = UNesT(
            in_channels=1,
            out_channels=133,            # ← SE MANTIENE
            img_size=(96, 96, 96),
            num_heads=(4, 8, 16),
            depths=(2, 2, 8),
            embed_dim=(128, 256, 512),
            patch_size=4,
        )

        # 2) Carga pesos preentrenados (ignorando diferencias si las hubiese)
        if hf_hub_download is None:
            raise RuntimeError("huggingface_hub no está disponible para descargar los pesos.")

        ckpt_path = hf_hub_download(
            repo_id="MONAI/wholeBrainSeg_Large_UNEST_segmentation",
            filename="models/model.pt",
            local_dir="/data/cristian/projects/med_data/rise-miccai/pretrained_models/wholeBrainSeg_Large_UNEST_segmentation/",
        local_dir_use_symlinks=False
        )
        print("model_path:",ckpt_path)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state: Dict[str, Any] = ckpt.get("model", ckpt)

        # Cargamos con strict=True normalmente; si tu ckpt tiene claves distintas
        # puedes pasar a strict=False. Aquí usamos strict=False por robustez.
        missing, unexpected = self.backbone.load_state_dict(state, strict=False)
        if verbose:
            print("[UNesT133Adapter] missing keys:", missing)
            print("[UNesT133Adapter] unexpected keys:", unexpected)

        # 3) (Opcional) Congelar backbone y entrenar solo el adaptador
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # 4) Adaptador 133 -> 3
        if bottleneck and bottleneck > 0:
            self.adapter = nn.Sequential(
                nn.Conv3d(133, bottleneck, kernel_size=1, bias=False),
                nn.GroupNorm(1, bottleneck),
                nn.ReLU(inplace=True),
                nn.Dropout3d(p=dropout) if dropout > 0 else nn.Identity(),
                nn.Conv3d(bottleneck, num_classes, kernel_size=1, bias=True),
            )
        else:
            # Proyección directa (simple y eficiente)
            self.adapter = nn.Conv3d(133, num_classes, kernel_size=1, bias=True)

        # Inicialización razonable para la proyección final
        for m in self.adapter.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        _to_device(self, device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits_133 = self.backbone(x)   # [B,133,D,H,W]
        logits_3   = self.adapter(logits_133)
        return logits_3


# ---------------------------------------------------------------------
# SegResNet (3 clases)
# ---------------------------------------------------------------------
def build_segresnet(
    num_classes: int = 3,
    init_filters: int = 32,
    dropout_prob: float = 0.0,
    device: Optional[str] = None,
) -> nn.Module:
    if SegResNet is None:
        raise ImportError("MONAI no está disponible (instala `monai`) para usar SegResNet.")
    model = SegResNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=num_classes,  # ← 3
        init_filters=init_filters,
        dropout_prob=dropout_prob,
    )
    return _to_device(model, device)


# ---------------------------------------------------------------------
# Autoencoder ligero (3 clases)
# ---------------------------------------------------------------------
class AutoencoderSeg(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, base_filters=16) -> None:
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv3d(in_channels, base_filters, 3, 2, 1),
            nn.BatchNorm3d(base_filters),
            nn.ReLU(inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv3d(base_filters, base_filters*2, 3, 2, 1),
            nn.BatchNorm3d(base_filters*2),
            nn.ReLU(inplace=True),
        )
        self.enc3 = nn.Sequential(
            nn.Conv3d(base_filters*2, base_filters*4, 3, 2, 1),
            nn.BatchNorm3d(base_filters*4),
            nn.ReLU(inplace=True),
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose3d(base_filters*4, base_filters*2, 2, 2),
            nn.BatchNorm3d(base_filters*2),
            nn.ReLU(inplace=True),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose3d(base_filters*2, base_filters, 2, 2),
            nn.BatchNorm3d(base_filters),
            nn.ReLU(inplace=True),
        )
        self.dec3 = nn.ConvTranspose3d(base_filters, out_channels, 2, 2)  # ← 3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.enc1(x); x = self.enc2(x); x = self.enc3(x)
        x = self.dec1(x); x = self.dec2(x); x = self.dec3(x)
        return x


def build_autoencoder(num_classes: int = 3, device: Optional[str] = None) -> nn.Module:
    model = AutoencoderSeg(in_channels=1, out_channels=num_classes, base_filters=16)
    return _to_device(model, device)


# ---------------------------------------------------------------------
# Fábrica
# ---------------------------------------------------------------------
def get_model(
    name: str,
    device: Optional[str] = None,
    **kwargs,
) -> nn.Module:
    """
    name in {"unest", "segresnet", "autoencoder"}.

    Ejemplos:
    - get_model("unest", device="cuda:0", freeze_backbone=True, bottleneck=32)
    - get_model("segresnet", device="cuda:0")
    - get_model("autoencoder", device="cuda:0")
    """
    n = name.lower()
    if n == "unest":
        return UNesT133Adapter(device=device, **kwargs)
    if n == "segresnet":
        return build_segresnet(device=device, **kwargs)
    if n == "autoencoder":
        return build_autoencoder(device=device, **kwargs)
    raise ValueError("Modelos disponibles: 'unest', 'segresnet', 'autoencoder'")
