# --- Bloques utilitarios ---
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Tuple
from huggingface_hub import hf_hub_download

# ----------------- Utils -----------------

class ConvGNReLU3d(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.GroupNorm(num_groups=min(groups, out_ch), num_channels=out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)

class SE3D(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        mid = max(1, ch // r)
        self.avg = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(ch, mid, 1), nn.ReLU(inplace=True),
            nn.Conv3d(mid, ch, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.fc(self.avg(x))

class CoordConv3D(nn.Module):
    """Concatena coordenadas normalizadas (z,y,x) a la entrada."""
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = ConvGNReLU3d(in_ch+3, out_ch, k=k, s=s, p=p)
    def forward(self, x):
        B, _, D, H, W = x.shape
        z = torch.linspace(-1, 1, D, device=x.device).view(1,1,D,1,1).expand(B,1,D,H,W)
        y = torch.linspace(-1, 1, H, device=x.device).view(1,1,1,H,1).expand(B,1,D,H,W)
        xg = torch.linspace(-1, 1, W, device=x.device).view(1,1,1,1,W).expand(B,1,D,H,W)
        return self.conv(torch.cat([x, z, y, xg], dim=1))

class ASPP3D(nn.Module):
    """Atrous Spatial Pyramid Pooling 3D sencillo."""
    def __init__(self, in_ch, out_ch, dilations=(1, 2, 4, 8), use_gp=True, drop_p=0.1):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=d, dilation=d, bias=False),
                nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
                nn.ReLU(inplace=True)
            ) for d in dilations
        ])
        self.use_gp = use_gp
        if use_gp:
            self.gp = nn.Sequential(
                nn.AdaptiveAvgPool3d(1),
                nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.ReLU(inplace=True)
            )
        self.proj = nn.Sequential(
            nn.Conv3d(out_ch*(len(dilations)+(1 if use_gp else 0)), out_ch, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=drop_p)
        )
    def forward(self, x):
        outs = [b(x) for b in self.branches]
        if self.use_gp:
            gp = self.gp(x)
            gp = F.interpolate(gp, size=x.shape[-3:], mode="trilinear", align_corners=False)
            outs.append(gp)
        x = torch.cat(outs, dim=1)
        return self.proj(x)

# ----------------- Head: FPN + PAN + ASPP + HR -----------------

class FPN_PAN_ASPP_Head3D(nn.Module):
    """
    Entradas: C1 (alta res), C2 (media), C3 (baja). Canalizado a fpn_ch.
    Top-down (FPN) -> bottom-up (PAN) -> fusión (+ASPP, HR, Sem) -> logits.
    """
    def __init__(
        self,
        in_channels: List[int],        # [C1, C2, C3]
        num_classes: int = 3,
        fpn_ch: int = 96,              # un poco menor para regularizar
        deep_supervision: bool = False,
        use_se: bool = True,
        use_hr_branch: bool = True,
        hr_in_ch: int = 1,
        hr_ch: int = 24,
        drop_p: float = 0.1,
    ):
        super().__init__()
        assert len(in_channels) == 3
        C1, C2, C3 = in_channels

        # Lateral 1x1 a fpn_ch
        self.lat1 = nn.Conv3d(C1, fpn_ch, 1, bias=False)
        self.lat2 = nn.Conv3d(C2, fpn_ch, 1, bias=False)
        self.lat3 = nn.Conv3d(C3, fpn_ch, 1, bias=False)

        # Top-down smoothing
        self.s1 = ConvGNReLU3d(fpn_ch, fpn_ch)
        self.s2 = ConvGNReLU3d(fpn_ch, fpn_ch)

        # Bottom-up (PAN): downsample conv stride 2
        self.down12 = ConvGNReLU3d(fpn_ch, fpn_ch, k=3, s=1, p=1)
        self.down23 = ConvGNReLU3d(fpn_ch, fpn_ch, k=3, s=1, p=1)

        # ASPP sobre fusión final
        self.aspp = ASPP3D(fpn_ch, fpn_ch, dilations=(1,2,4,8), drop_p=drop_p)

        # Semántica desde los 133 logits
        self.sem_proj = nn.Conv3d(133, fpn_ch, 1, bias=False)

        # HR branch con CoordConv (mejor prior espacial)
        self.use_hr_branch = use_hr_branch
        if use_hr_branch:
            self.hr = nn.Sequential(
                CoordConv3D(hr_in_ch, hr_ch, k=3, s=1, p=1),
                ConvGNReLU3d(hr_ch, hr_ch, k=3, s=1, p=1),
                nn.Conv3d(hr_ch, fpn_ch, kernel_size=1, bias=False),
            )

        self.use_se = use_se
        if use_se:
            self.se1 = SE3D(fpn_ch); self.se2 = SE3D(fpn_ch); self.se3 = SE3D(fpn_ch)

        self.drop = nn.Dropout3d(p=drop_p)

        # Head final
        self.head = nn.Sequential(
            ConvGNReLU3d(fpn_ch, fpn_ch),
            nn.Conv3d(fpn_ch, num_classes, 1)
        )

        # Deep supervision (opcional)
        self.deep_supervision = deep_supervision
        if deep_supervision:
            self.aux2 = nn.Conv3d(fpn_ch, num_classes, 1)
            self.aux3 = nn.Conv3d(fpn_ch, num_classes, 1)

    def forward(self, feats: List[torch.Tensor], logits133: torch.Tensor, x: torch.Tensor):
        C1, C2, C3 = feats  # alta -> media -> baja

        # FPN (top-down)
        P3 = self.lat3(C3)
        P2 = self.lat2(C2) + F.interpolate(P3, size=C2.shape[-3:], mode="trilinear", align_corners=False)
        P1 = self.lat1(C1) + F.interpolate(P2, size=C1.shape[-3:], mode="trilinear", align_corners=False)

        P2 = self.s2(P2)
        P1 = self.s1(P1)

        if self.use_se:
            P3 = self.se3(P3); P2 = self.se2(P2); P1 = self.se1(P1)

        # PAN (bottom-up): agrega camino inverso para mejor contexto
        N2 = P2 + self.down12(P1)  # baja a escala de P2
        N3 = P3 + self.down23(N2)  # baja a escala de P3

        # Fusión a alta resolución
        F2 = F.interpolate(N2, size=P1.shape[-3:], mode="trilinear", align_corners=False)
        F3 = F.interpolate(N3, size=P1.shape[-3:], mode="trilinear", align_corners=False)
        Ff = (P1 + F2 + F3) / 3.0

        # Inyecta semántica (133) + HR + ASPP + Dropout
        Sem = self.sem_proj(logits133)
        Sem = F.interpolate(Sem, size=Ff.shape[-3:], mode="trilinear", align_corners=False)
        Ff = Ff + Sem

        if self.use_hr_branch:
            HR = self.hr(x)
            HR = F.interpolate(HR, size=Ff.shape[-3:], mode="trilinear", align_corners=False)
            Ff = Ff + HR

        Ff = self.aspp(Ff)
        Ff = self.drop(Ff)

        out: Dict[str, torch.Tensor] = {"logits": self.head(Ff)}
        if self.deep_supervision:
            out["aux2"] = F.interpolate(self.aux2(N2), size=Ff.shape[-3:], mode="trilinear", align_corners=False)
            out["aux3"] = F.interpolate(self.aux3(N3), size=Ff.shape[-3:], mode="trilinear", align_corners=False)
        return out

# -------------- Adapter sobre UNesT --------------

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
    # ordenar alto→bajo por tamaño espacial
    order = sorted(range(len(feats)), key=lambda i: feats[i].shape[-1]*feats[i].shape[-2]*feats[i].shape[-3], reverse=True)
    return [feats[i].shape[1] for i in order]

class UNesT133PANetASPPAdapter(nn.Module):
    def __init__(
        self,
        hook_layers = ["decoder1.conv_block", "decoder3.conv_block", "decoder5.conv_block"],
        repo_id: str = "MONAI/wholeBrainSeg_Large_UNEST_segmentation",
        filename: str = "models/model.pt",
        cache_dir: Optional[str] = None,
        num_classes: int = 3,
        fpn_ch: int = 96,
        deep_supervision: bool = False,
        use_se: bool = True,
        freeze_backbone: bool = False,
        use_hr_branch: bool = True,
        drop_p: float = 0.1,
        return_dict: bool = False,     # ← por defecto devuelve solo logits
        verbose: bool = False,
        device: Optional[str] = None,
    ):
        super().__init__()
        try:
            from .unets_scripts.networks.unest_base_patch_4 import UNesT
        except Exception as e:
            raise ImportError("UNeST import failed. Checa 'unest_base_patch_4.py' en tu PYTHONPATH.") from e

        self.backbone = UNesT(
            in_channels=1, out_channels=133, img_size=(96,96,96),
            num_heads=(4,8,16), depths=(2,2,8), embed_dim=(128,256,512), patch_size=4
        )

        # Carga pesos
        ckpt_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=cache_dir, local_dir_use_symlinks=False)
        state = torch.load(ckpt_path, map_location="cpu")
        state_dict = state.get("model", state)
        filtered_state = {k: v for k, v in state_dict.items() if not k.startswith("out.")}
        self.backbone.load_state_dict(filtered_state, strict=False)

        if freeze_backbone:
            for p in self.backbone.parameters(): p.requires_grad = False

        # Hooks
        self._feats: List[torch.Tensor] = []
        self._hooks = []
        self._register_hooks(hook_layers)

        # Dims por probe (alto→medio→bajo)
        feature_dims = infer_feature_dims_ordered(self.backbone, hook_layers)

        self.head = FPN_PAN_ASPP_Head3D(
            in_channels=feature_dims,
            num_classes=num_classes,
            fpn_ch=fpn_ch,
            deep_supervision=deep_supervision,
            use_se=use_se,
            use_hr_branch=use_hr_branch,
            hr_in_ch=1,
            hr_ch=max(16, fpn_ch//4),
            drop_p=drop_p,
        )
        self.return_dict = return_dict
        if device is not None: self.to(device)

    def _get_module_by_name(self, name: str) -> nn.Module:
        mod = dict(self.backbone.named_modules()).get(name, None)
        if mod is None:
            raise ValueError(f"No se encontró el módulo '{name}'. Usa: [n for n,_ in backbone.named_modules()]")
        return mod

    def _hook_fn(self, _module, _inp, out): self._feats.append(out)

    def _register_hooks(self, names: List[str]):
        for n in names:
            m = self._get_module_by_name(n)
            self._hooks.append(m.register_forward_hook(self._hook_fn))

    def remove_hooks(self):
        for h in self._hooks: h.remove()
        self._hooks = []

    def forward(self, x: torch.Tensor):
        self._feats = []
        logits133 = self.backbone(x)        # llena hooks
        feats = sorted(self._feats, key=lambda t: t.shape[-1], reverse=True)  # alto→bajo
        out = self.head(feats, logits133, x)
        return out if self.return_dict else out["logits"]
