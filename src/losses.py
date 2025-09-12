import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceLoss, FocalLoss, TverskyLoss, DiceCELoss

def create_loss_function(loss_name="dice_ce_balanced"):
    """Loss functions optimizadas para hippocampus segmentation - Device agnostic"""
    
    if loss_name == "dice_ce_balanced":
        """DiceCE balanceado - MÁS ESTABLE Y RECOMENDADO"""
        class BalancedDiceCELoss(nn.Module):
            def __init__(self):
                super().__init__()
                self.dice = DiceLoss(
                    include_background=False,
                    to_onehot_y=True,
                    softmax=True,
                    batch=True,
                    smooth_nr=1e-5,
                    smooth_dr=1e-5
                )
                # Pesos se crearán dinámicamente en el device correcto
                self.ce_weight = None
                
            def forward(self, logits, targets):
                # Crear pesos en el mismo device que los logits
                if self.ce_weight is None or self.ce_weight.device != logits.device:
                    self.ce_weight = torch.tensor([0.1, 1.2, 1.2], device=logits.device, dtype=logits.dtype)
                
                # Dice loss
                dice_loss = self.dice(logits, targets)
                
                # Cross entropy con pesos
                ce_loss = F.cross_entropy(
                    logits, 
                    targets.squeeze(1).long(), 
                    weight=self.ce_weight
                )
                
                # Combinación balanceada
                return dice_loss + 0.5 * ce_loss
        
        return BalancedDiceCELoss()
    
    elif loss_name == "dice_focal_spatial":
        """Versión ARREGLADA del dice_focal_spatial"""
        class DiceFocalSpatialLoss(nn.Module):
            def __init__(self):
                super().__init__()
                self.dice = DiceLoss(
                    include_background=False,
                    to_onehot_y=True,
                    softmax=True,
                    batch=True
                )
                # FocalLoss sin weights inicialmente
                self.focal = FocalLoss(
                    include_background=True,
                    to_onehot_y=True,
                    use_softmax=True,
                    gamma=2.0
                )
                self.focal_weight = None
            
            def forward(self, logits, targets):
                # Crear pesos en el mismo device que los logits
                if self.focal_weight is None or self.focal_weight.device != logits.device:
                    self.focal_weight = torch.tensor([0.1, 1.2, 1.2], device=logits.device, dtype=logits.dtype)
                
                # Losses base
                dice_loss = self.dice(logits, targets)
                
                # Actualizar weights del FocalLoss
                self.focal.weight = self.focal_weight
                focal_loss = self.focal(logits, targets)
                
                # Penalización espacial (simplificada)
                B, C, D, H, W = logits.shape
                probs = torch.softmax(logits, dim=1)
                center_w = W // 2
                
                spatial_penalty = torch.tensor(0.0, device=logits.device)
                # Penalizar left en lado derecho
                if C > 1:
                    left_wrong = probs[:, 1, :, :, center_w:].mean()
                    spatial_penalty = spatial_penalty + left_wrong * 0.3
                # Penalizar right en lado izquierdo
                if C > 2:
                    right_wrong = probs[:, 2, :, :, :center_w].mean()
                    spatial_penalty = spatial_penalty + right_wrong * 0.3
                
                # Combinación con pesos ajustados
                return 0.5 * dice_loss + 0.3 * focal_loss + 0.2 * spatial_penalty
        
        return DiceFocalSpatialLoss()
    
    elif loss_name == "focal_tversky_balanced":
        """Focal Tversky para estructuras pequeñas"""
        class FocalTverskyBalanced(nn.Module):
            def __init__(self):
                super().__init__()
                self.alpha = 0.7  # Penaliza más false negatives
                self.beta = 0.3   # Menos peso a false positives
                self.gamma = 1.5  # Focal weight
                
            def forward(self, logits, targets):
                device = logits.device
                dtype = logits.dtype
                num_classes = logits.shape[1]
                probs = F.softmax(logits, dim=1)
                
                # One-hot encoding
                targets_one_hot = F.one_hot(
                    targets.squeeze(1).long(), 
                    num_classes
                ).permute(0, 4, 1, 2, 3).float().to(device=device, dtype=dtype)
                
                # Skip background para Tversky
                probs_fg = probs[:, 1:]
                targets_fg = targets_one_hot[:, 1:]
                
                # Tversky components
                tp = (probs_fg * targets_fg).sum(dim=(2, 3, 4))
                fp = (probs_fg * (1 - targets_fg)).sum(dim=(2, 3, 4))
                fn = ((1 - probs_fg) * targets_fg).sum(dim=(2, 3, 4))
                
                # Tversky index
                tversky = (tp + 1e-5) / (tp + self.alpha * fn + self.beta * fp + 1e-5)
                
                # Focal Tversky loss
                focal_tversky = torch.pow(1.0 - tversky, self.gamma)
                
                return focal_tversky.mean()
        
        return FocalTverskyBalanced()
    
    elif loss_name == "dice_only": # NO FUNCIONA BIEN, MEJOR NO USARLO
        """Solo Dice loss - simple pero efectivo"""
        return DiceLoss(
            include_background=False,
            to_onehot_y=True,
            softmax=True,
            batch=True,
            smooth_nr=1e-5,
            smooth_dr=1e-5
        )
    
    elif loss_name == "dice_ce":
        """DiceCE de MONAI sin modificaciones"""
        return DiceCELoss(
            include_background=False,
            to_onehot_y=True,
            softmax=True,
            lambda_dice=1.0,
            lambda_ce=1.0
        )
    
    elif loss_name == "balance_ce_focal":
        """Version alternativa con balance dinámico"""
        class BalancedDiceFocalLoss(nn.Module):
            def __init__(self):
                super().__init__()
                self.dice = DiceLoss(
                    include_background=False,
                    to_onehot_y=True,
                    softmax=True,
                    squared_pred=True
                )
                # FocalLoss sin weights inicialmente
                self.focal = FocalLoss(
                    include_background=False,
                    to_onehot_y=True,
                    gamma=2.0
                )
                self.focal_weight = None
                
            def forward(self, logits, targets):
                device = logits.device
                dtype = logits.dtype
                
                # Crear pesos dinámicamente
                if self.focal_weight is None or self.focal_weight.device != device:
                    self.focal_weight = torch.tensor([1.2, 1.5], device=device, dtype=dtype)
                
                # Aplicar pesos
                self.focal.weight = self.focal_weight
                
                # Calcular losses
                dice_loss = self.dice(logits, targets)
                focal_loss = self.focal(logits, targets)
                
                # Balance adaptativo
                with torch.no_grad():
                    pred = torch.argmax(logits, dim=1)
                    n_left = (pred == 1).sum()
                    n_right = (pred == 2).sum()
                    balance_factor = torch.clamp(n_left / (n_right + 1e-5), 0.5, 2.0)
                
                return dice_loss + focal_loss * balance_factor
        
        return BalancedDiceFocalLoss()
    
    elif loss_name == "dice_ce_spatial":
        """DiceCE con constraint espacial"""
        class DiceCESpatialLoss(nn.Module):
            def __init__(self):
                super().__init__()
                self.dice_ce = DiceCELoss(to_onehot_y=True, softmax=True)
                
            def forward(self, logits, targets):
                device = logits.device
                base_loss = self.dice_ce(logits, targets)
                
                # Agregar penalización espacial
                B, C, D, H, W = logits.shape
                probs = torch.softmax(logits, dim=1)
                center = W // 2
                
                # Penalizaciones
                spatial_loss = torch.tensor(0.0, device=device)
                
                # Left hippocampus should be on left side
                if C > 1:  # Si tenemos clase 1
                    left_prob = probs[:, 1]
                    wrong_side = left_prob[:, :, :, center:].mean()
                    spatial_loss = spatial_loss + wrong_side
                
                # Right hippocampus should be on right side  
                if C > 2:  # Si tenemos clase 2
                    right_prob = probs[:, 2]
                    wrong_side = right_prob[:, :, :, :center].mean()
                    spatial_loss = spatial_loss + wrong_side
                
                return base_loss + 0.3 * spatial_loss
        
        return DiceCESpatialLoss()
    
    elif loss_name == "focal_spatial": # NO FUNCIONA BIEN, MEJOR NO USARLO
        """Focal Tversky con awareness espacial"""
        class FocalTverskySpatialLoss(nn.Module):
            def __init__(self):
                super().__init__()
                self.alpha = 0.7  # Peso para false negatives
                self.beta = 0.3   # Peso para false positives
                self.gamma = 2.0  # Focal parameter
                
            def forward(self, logits, y):
                device = logits.device
                dtype = logits.dtype
                num_classes = logits.shape[1]
                B, C, D, H, W = logits.shape
                
                # One-hot encoding
                y_onehot = F.one_hot(y.long().squeeze(1), num_classes).permute(0,4,1,2,3).float()
                y_onehot = y_onehot.to(device=device, dtype=dtype)
                
                # Skip background
                y_onehot = y_onehot[:,1:]
                logits_fg = logits[:,1:]
                
                p = torch.softmax(logits_fg, dim=1)
                
                # Tversky components
                tp = (p * y_onehot).sum(dim=(2,3,4))
                fp = (p * (1 - y_onehot)).sum(dim=(2,3,4))
                fn = ((1 - p) * y_onehot).sum(dim=(2,3,4))
                
                tversky = (tp + 1e-5) / (tp + self.alpha*fn + self.beta*fp + 1e-5)
                focal_tversky = torch.pow(1.0 - tversky, self.gamma)
                
                # Spatial penalty
                center = W // 2
                probs_full = torch.softmax(logits, dim=1)
                
                spatial_penalty = torch.tensor(0.0, device=device)
                if num_classes > 1:
                    left_wrong = probs_full[:, 1, :, :, center:].mean()
                    spatial_penalty = spatial_penalty + left_wrong * 0.5
                if num_classes > 2:
                    right_wrong = probs_full[:, 2, :, :, :center].mean()
                    spatial_penalty = spatial_penalty + right_wrong * 0.5
                
                return focal_tversky.mean() + 0.25 * spatial_penalty
        
        return FocalTverskySpatialLoss()
    
    else:
        print(f"Warning: Unknown loss {loss_name}, using dice_ce_balanced")
        return create_loss_function("dice_ce_balanced")