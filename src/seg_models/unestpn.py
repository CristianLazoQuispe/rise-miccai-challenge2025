from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Tuple
from huggingface_hub import hf_hub_download

# --- Bloques utilitarios ---

class ConvGNReLU3d(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)

class SE3D(nn.Module):
    """Squeeze-Excite 3D muy ligero (opcional)."""
    def __init__(self, ch, r=16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(ch, ch//r, 1, bias=True), nn.ReLU(inplace=True),
            nn.Conv3d(ch//r, ch, 1, bias=True), nn.Sigmoid()
        )
    def forward(self, x):
        w = self.fc(self.avg(x))
        return x * w

# --- Head FPN ---

class FPNHead3D(nn.Module):
    """
    FPN 3D con 3 niveles: C1 (alto detalle), C2, C3 (semántica profunda).
    Produce logits finales y (opcional) salidas auxiliares en C2/C3.
    """
    def __init__(
        self,
        in_channels: List[int],     # p.ej. [C1, C2, C3] = [128, 256, 512]
        num_classes: int = 3,
        fpn_ch: int = 128,
        deep_supervision: bool = True,
        use_se: bool = True,
        use_hr_branch: bool = True,
        hr_in_ch: int = 1,
        hr_ch: int = 32
    ):
        super().__init__()
        assert len(in_channels) == 3, "Se esperan 3 niveles para la pirámide"

        # proyecciones laterales
        self.lat1 = nn.Conv3d(in_channels[0], fpn_ch, 1, bias=False)
        self.lat2 = nn.Conv3d(in_channels[1], fpn_ch, 1, bias=False)
        self.lat3 = nn.Conv3d(in_channels[2], fpn_ch, 1, bias=False)

        # suavizado tras la suma top-down
        self.smooth1 = ConvGNReLU3d(fpn_ch, fpn_ch)
        self.smooth2 = ConvGNReLU3d(fpn_ch, fpn_ch)

        # rama semántica desde los 133 logits del backbone (se inyecta fuera)
        self.sem_proj = nn.Conv3d(133, fpn_ch, 1, bias=False)

        # rama HR (alta resolución) sobre la entrada
        self.use_hr_branch = use_hr_branch
        if use_hr_branch:
            self.hr = nn.Sequential(
                ConvGNReLU3d(hr_in_ch, hr_ch),
                ConvGNReLU3d(hr_ch, hr_ch),
                ConvGNReLU3d(hr_ch, fpn_ch)
            )

        self.use_se = use_se
        if use_se:
            self.se1 = SE3D(fpn_ch)
            self.se2 = SE3D(fpn_ch)
            self.se3 = SE3D(fpn_ch)

        # cabeza final
        self.head = nn.Sequential(
            ConvGNReLU3d(fpn_ch, fpn_ch),
            nn.Conv3d(fpn_ch, num_classes, 1)
        )

        # deep supervision
        self.deep_supervision = deep_supervision
        if deep_supervision:
            self.aux2 = nn.Conv3d(fpn_ch, num_classes, 1)
            self.aux3 = nn.Conv3d(fpn_ch, num_classes, 1)

    def forward(
        self,
        feats: List[torch.Tensor],   # [C1, C2, C3]
        logits133: torch.Tensor,     # salida 133 del backbone
        x: torch.Tensor              # entrada original (para rama HR)
    ) -> Dict[str, torch.Tensor]:
        C1, C2, C3 = feats  # tamaños: C1> C2> C3 (en resolución)

        # laterales
        P3 = self.lat3(C3)
        P2 = self.lat2(C2) + F.interpolate(P3, size=C2.shape[-3:], mode="trilinear", align_corners=False)
        P1 = self.lat1(C1) + F.interpolate(P2, size=C1.shape[-3:], mode="trilinear", align_corners=False)

        # suavizados
        P2 = self.smooth2(P2)
        P1 = self.smooth1(P1)

        # atención (opcional)
        if self.use_se:
            P3 = self.se3(P3)
            P2 = self.se2(P2)
            P1 = self.se1(P1)

        # inyectar semántica de los 133 canales
        Sem = self.sem_proj(logits133)
        Sem = F.interpolate(Sem, size=P1.shape[-3:], mode="trilinear", align_corners=False)
        P1 = P1 + Sem

        # rama HR
        if self.use_hr_branch:
            HR = self.hr(x)
            HR = F.interpolate(HR, size=P1.shape[-3:], mode="trilinear", align_corners=False)
            P1 = P1 + HR

        out = {"logits": self.head(P1)}
        if self.deep_supervision:
            out["aux2"] = F.interpolate(self.aux2(P2), size=P1.shape[-3:], mode="trilinear", align_corners=False)
            out["aux3"] = F.interpolate(self.aux3(P3), size=P1.shape[-3:], mode="trilinear", align_corners=False)
        return out

def _to_device(model: nn.Module, device: Optional[str] = None) -> nn.Module:
    """Utility to move a model to the specified device if provided."""
    if device is not None:
        model = model.to(device)
    return model

import torch, torch.nn as nn

def infer_feature_dims_ordered(backbone: nn.Module, hook_layers, in_shape=(1,1,96,96,96)):
    feats, handles = [], []
    def _hook(_m, _i, o): feats.append(o)
    name2mod = dict(backbone.named_modules())
    for n in hook_layers:
        assert n in name2mod, f"No existe el módulo: {n}"
        handles.append(name2mod[n].register_forward_hook(_hook))
    with torch.no_grad():
        backbone.eval()
        _ = backbone(torch.zeros(in_shape))
    for h in handles: h.remove()

    # Ordena por resolución (volumen espacial) de mayor a menor
    order = sorted(range(len(feats)),
                   key=lambda i: feats[i].shape[-1]*feats[i].shape[-2]*feats[i].shape[-3],
                   reverse=True)
    dims_ordered = [feats[i].shape[1] for i in order]
    shapes_ordered = [tuple(feats[i].shape[-3:]) for i in order]
    print("[probe] feats shapes (alto→bajo):", shapes_ordered,
          "| chans:", dims_ordered)
    return dims_ordered


# --- Adaptador completo sobre tu UNesT ---
class UNesT133FPNAdapter(nn.Module):
    """
    Reemplazo de tu UNesT133Adapter con FPN + HR + deep supervision.
    Usa hooks para extraer 3 features intermedias de UNesT.
    """
    def __init__(
        self,
        #hook_layers: List[str],        # nombres de módulos a "enganchar" para C1,C2,C3
        #feature_dims: List[int],       # canales de C1,C2,C3 (p.ej. [128,256,512])
        hook_layers = [
            "decoder1.conv_block",  # C1: alta resolución
            "decoder3.conv_block",  # C2: media
            "decoder5.conv_block",  # C3: baja (profunda)
        ],
        #feature_dims = [32, 128, 512],

        repo_id: str = "MONAI/wholeBrainSeg_Large_UNEST_segmentation",
        filename: str = "models/model.pt",
        cache_dir: Optional[str] = None,
        num_classes: int = 3,
        fpn_ch: int = 128,
        deep_supervision: bool = False,
        use_se: bool = True,
        freeze_backbone: bool = False,
        use_hr_branch: bool = True,
        verbose: bool = False,
        device: Optional[str] = None,
    ):
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
        
        print("*"*30)
        feature_dims = infer_feature_dims_ordered(self.backbone, hook_layers)  # ← usa la nueva función
        print("feature_dims =", feature_dims)
        print("*"*30)
        print([n for n,_ in self.backbone.named_modules()])

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

        assert len(hook_layers) == 3 and len(feature_dims) == 3
        self._feats: List[torch.Tensor] = []
        self._hooks = []
        self._register_hooks(hook_layers)
        self.head = FPNHead3D(
            in_channels=feature_dims,
            num_classes=num_classes,
            fpn_ch=fpn_ch,
            deep_supervision=deep_supervision,
            use_se=use_se,
            use_hr_branch=use_hr_branch,
            hr_in_ch=1,
            hr_ch=fpn_ch // 4,
        )
        _to_device(self, device)

    def _get_module_by_name(self, name: str) -> nn.Module:
        mod = dict(self.backbone.named_modules()).get(name, None)
        if mod is None:
            raise ValueError(f"No se encontró el módulo '{name}' en el backbone. "
                             f"Imprime los módulos con: [n for n,_ in backbone.named_modules()]")
        return mod

    def _hook_fn(self, _module, _inp, out):
        self._feats.append(out)

    def _register_hooks(self, names: List[str]):
        for n in names:
            m = self._get_module_by_name(n)
            self._hooks.append(m.register_forward_hook(self._hook_fn))

    def remove_hooks(self):
        for h in self._hooks: h.remove()
        self._hooks = []

    def forward(self, x: torch.Tensor) -> torch.Tensor | Dict[str, torch.Tensor]:
        self._feats = []
        logits133 = self.backbone(x)          # aquí se llenan los hooks
        if len(self._feats) != 3:
            raise RuntimeError(f"Se esperaban 3 features hookeadas y se capturaron {len(self._feats)}. "
                               f"Revisa 'hook_layers'.")
        # ordenar de alta a baja resolución si hace falta
        # dentro de UNesT133FPNAdapter.forward (después de self._feats = [])
        feats = sorted(self._feats, key=lambda t: t.shape[-1], reverse=True)  # alto→medio→bajo
        #print("FPN feats ch =", [f.shape[1] for f in feats])  # debería calzar con feature_dims
        return self.head(feats, logits133, x)["logits"]
