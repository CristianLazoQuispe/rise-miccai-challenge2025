from monai.networks.nets import SegResNet
from huggingface_hub import hf_hub_download
"""
model = SegResNet(spatial_dims=3, in_channels=1, out_channels=105, init_filters=32, blocks_down=[1,2,2,4])
# 📦 Descargar pesos preentrenados
model_path = hf_hub_download(
    repo_id="MONAI/wholeBody_ct_segmentation",
    filename="models/model.pt",
    local_dir="/data/cristian/projects/med_data/rise-miccai/pretrained_models/wholeBody_ct_segmentation/",
    local_dir_use_symlinks=False
)
# ✅ Cargar directamente el state_dict (no usar load_from aquí)
state_dict = torch.load(model_path, map_location="cpu")
model.load_state_dict(state_dict)
"""
import torch
import torch.nn as nn
from monai.networks.nets import SegResNet
from huggingface_hub import hf_hub_download


class AdaptedSegResNetV2(nn.Module):
    def __init__(
        self,
        pretrained=True,
        repo_id="MONAI/wholeBody_ct_segmentation",
        filename="models/model.pt",
        cache_dir="/data/cristian/projects/med_data/rise-miccai/pretrained_models/wholeBody_ct_segmentation/",
        freeze_stage=0,  # 0: none, 1: freeze all, 2: freeze encoder only
        bottleneck=32,
        num_classes=3,
        device="cuda:0",
    ):
        super().__init__()

        # Backbone con salida original (105 clases del modelo preentrenado)
        self.backbone = SegResNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=105,
            init_filters=32,
            blocks_down=[1, 2, 2, 4],
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

        # 🔁 Fijar pesos si se requiere
        if freeze_stage == 1:
            for param in self.backbone.parameters():
                param.requires_grad = False
        elif freeze_stage == 2:
            for name, param in self.backbone.named_parameters():
                if "decoder" not in name and "upsample" not in name:
                    print("freeze param:", name)
                    param.requires_grad = False

        # 🔄 Adapter más sofisticado que simple conv
        self.adapter = nn.Sequential(
            nn.Conv3d(105, bottleneck, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=bottleneck),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=0.1),
            nn.Conv3d(bottleneck, num_classes, kernel_size=1, bias=True)
        )

        self.to(device)

    def forward(self, x):
        x = self.backbone(x)
        x = self.adapter(x)
        return x

