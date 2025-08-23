from monai.networks.nets import SwinUNETR
from huggingface_hub import hf_hub_download
"""
model = SwinUNETR(in_channels=1, out_channels=14, feature_size=48, use_checkpoint=True)
# 📦 Descargar pesos preentrenados
model_path = hf_hub_download(
    repo_id="MONAI/swin_unetr_btcv_segmentation",
    filename="models/model.pt",
    local_dir="/data/cristian/projects/med_data/rise-miccai/pretrained_models/swin_unetr_btcv_segmentation/",
    local_dir_use_symlinks=False
)

# ✅ Cargar directamente el state_dict (no usar load_from aquí)
state_dict = torch.load(model_path, map_location="cpu")
model.load_state_dict(state_dict)
"""
import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR
from huggingface_hub import hf_hub_download


class AdaptedSwinUNETR(nn.Module):
    def __init__(
        self,
        img_size=(96, 96, 96),
        in_channels=1,
        num_classes=3,
        pretrained=True,
        freeze_stage=0,  # 0: none, 1: freeze all, 2: freeze encoder only
        bottleneck=32,
        use_checkpoint=True,
        repo_id="MONAI/swin_unetr_btcv_segmentation",
        filename="models/model.pt",
        cache_dir="/data/cristian/projects/med_data/rise-miccai/pretrained_models/swin_unetr_btcv_segmentation/",
        device="cuda:0",
    ):
        super().__init__()

        # Backbone con salida original (14 clases BTCV)
        self.backbone = SwinUNETR(
            #img_size=img_size,
            in_channels=in_channels,
            out_channels=14,  # Pretrained
            feature_size=48,
            use_checkpoint=use_checkpoint,
        )

        if pretrained:
            ckpt_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=cache_dir,
                local_dir_use_symlinks=False,
            )
            state_dict = torch.load(ckpt_path, map_location="cpu")
            self.backbone.load_state_dict(state_dict)
            print("Cargados pesos preentrenados desde:", ckpt_path)

        # ❄️ Freeze encoder si se requiere (solo backbone, no head)
        # 🔁 Fijar pesos si se requiere
        if freeze_stage == 1:
            for param in self.backbone.parameters():
                param.requires_grad = False
        elif freeze_stage == 2:
            for name, param in self.backbone.named_parameters():
                if "decoder" not in name and "up" not in name and "out" not in name:
                    print("freeze param:", name)
                    param.requires_grad = False

        # 🎯 Adapter elegante con bottleneck
        self.adapter = nn.Sequential(
            nn.Conv3d(14, bottleneck, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=bottleneck),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=0.1),
            nn.Conv3d(bottleneck, num_classes, kernel_size=1, bias=True)
        )

        self.to(device)

    def forward(self, x):
        x = self.backbone(x)  # [B, 14, D, H, W]
        x = self.adapter(x)   # [B, 3, D, H, W]
        return x

