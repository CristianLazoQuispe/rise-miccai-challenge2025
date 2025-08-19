import torch
import torch.nn as nn
from monai.networks.nets import AutoencoderKL
import torch
import torch.nn as nn
from monai.networks.nets import AutoencoderKL



class Latent3DClassifier_old(nn.Module):
    def __init__(self, model_path: str, device="cuda:0", num_labels=7, num_classes=3, dropout_p=0.3):
        super().__init__()
        self.num_labels = num_labels
        self.num_classes = num_classes
        self.device = torch.device(device)

        # 🧠 Definir Autoencoder (estructura esperada del modelo preentrenado)
        self.encoder = AutoencoderKL(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            latent_channels=8,
            channels=[64, 128, 256],
            num_res_blocks=2,
            norm_num_groups=32,
            norm_eps=1e-6,
            attention_levels=[False, False, False],
            with_encoder_nonlocal_attn=False,
            with_decoder_nonlocal_attn=False,
            include_fc=False
        ).encoder  # solo cargamos encoder

        # 📥 Cargar pesos desde archivo
        state_dict = torch.load(model_path, map_location=self.device)

        # 🔁 Renombrar claves conflictivas si es necesario
        new_state_dict = {}
        for k, v in state_dict.items():
            if "decoder.blocks.3.conv.conv" in k:
                k = k.replace("decoder.blocks.3.conv.conv", "decoder.blocks.3.postconv.conv")
            if "decoder.blocks.6.conv.conv" in k:
                k = k.replace("decoder.blocks.6.conv.conv", "decoder.blocks.6.postconv.conv")
            new_state_dict[k] = v

        # ✅ Cargar pesos al encoder
        missing, unexpected = self.encoder.load_state_dict(
            {k.replace("encoder.", ""): v for k, v in new_state_dict.items() if k.startswith("encoder.")},
            strict=False
        )
        print("🔍 Missing keys (encoder):", missing)
        print("🔍 Unexpected keys (encoder):", unexpected)

        # 🧊 Congelar pesos
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

        # 🔮 Clasificador sobre z: (B, 8, d, h, w)
        self.classifier = nn.Sequential(
            nn.Conv3d(8, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(32),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(128, num_labels * num_classes)
        )

    def forward(self, x):  # x: (B, 1, D, H, W)
        with torch.no_grad():
            z = self.encoder(x)  # z: (B, 8, d, h, w)
            print(z.shape)
        out = self.classifier(z)
        return out.view(-1, self.num_labels, self.num_classes)



class Latent3DClassifier(nn.Module):
    def __init__(self,num_labels=7, num_classes=3, dropout_p=0.3,pretrained=True,freeze_backbone=False):
        super().__init__()
        
        self.num_labels = num_labels
        self.num_classes = num_classes

        # 🧠 Inicializar el autoencoder completo
        self.autoencoder = AutoencoderKL(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            latent_channels=8,
            channels=[64, 128, 256],
            num_res_blocks=2,
            norm_num_groups=32,
            norm_eps=1e-6,
            attention_levels=[False, False, False],
            with_encoder_nonlocal_attn=False,
            with_decoder_nonlocal_attn=False,
            include_fc=False
        )

        if pretrained:
            model_path ="/data/cristian/projects/med_data/rise-miccai/pretrained_models/brats_mri_generative_diffusion/models/model_autoencoder.pt"

            # 📥 Cargar y ajustar el state_dict
            state_dict = torch.load(model_path)
            new_state_dict = {}
            for k, v in state_dict.items():
                if "decoder.blocks.3.conv.conv" in k:
                    k = k.replace("decoder.blocks.3.conv.conv", "decoder.blocks.3.postconv.conv")
                if "decoder.blocks.6.conv.conv" in k:
                    k = k.replace("decoder.blocks.6.conv.conv", "decoder.blocks.6.postconv.conv")
                new_state_dict[k] = v

            missing, unexpected = self.autoencoder.load_state_dict(new_state_dict, strict=True)
            print("🔍 Missing keys:", missing)
            print("🔍 Unexpected keys:", unexpected)

        if freeze_backbone:
            self.autoencoder.eval()
            for p in self.autoencoder.parameters():
                p.requires_grad = False
                # True  epoch 18 ✅ Model saved (F1=0.334)

        # 🔮 Clasificador sobre z_mu
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Conv3d(8, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(32),
            nn.Dropout(p=dropout_p),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(128, num_labels * num_classes)
        )

    def forward(self, x):  # x: (B, 1, D, H, W)
        with torch.no_grad():
            z_mu, _ = self.autoencoder.encode(x)  # usar solo la media latente
            #print(z_mu.shape)
        out = self.classifier(z_mu)
        return out.view(-1, self.num_labels, self.num_classes)
