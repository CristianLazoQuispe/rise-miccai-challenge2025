import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceLoss, FocalLoss, TverskyLoss, DiceCELoss

def create_loss_function(loss_name="dice_focal_spatial"):
    """Loss functions con awareness espacial para L/R"""
    
    if loss_name == "dice_focal_spatial":
        """Loss que penaliza predicciones en el lado incorrecto"""
        class DiceFocalSpatialLoss(nn.Module):
            def __init__(self):
                super().__init__()
                self.dice = DiceLoss(
                    include_background=False,
                    to_onehot_y=True,
                    softmax=True,
                    squared_pred=True
                )
                self.focal = FocalLoss(
                    include_background=False,
                    to_onehot_y=True,
                    use_softmax=True,
                    gamma=2.5,
                    weight=torch.tensor([1.0, 1.0])  # ← CORREGIDO: Solo 2 pesos para L y R
                )
            
            def forward(self, logits, targets):
                # Losses base
                dice_loss = self.dice(logits, targets)
                focal_loss = self.focal(logits, targets)
                
                # Penalización espacial
                B, C, D, H, W = logits.shape
                probs = torch.softmax(logits, dim=1)
                
                # El centro en el eje W (último) separa L/R
                center_w = W // 2
                
                # Penalizar clase 1 (left) en lado derecho
                left_probs = probs[:, 1, :, :, :]  # Probabilidades de clase left
                right_side_mask = torch.zeros_like(left_probs)
                right_side_mask[:, :, :, center_w:] = 1.0
                wrong_left = (left_probs * right_side_mask).mean()
                
                # Penalizar clase 2 (right) en lado izquierdo
                right_probs = probs[:, 2, :, :, :]  # Probabilidades de clase right
                left_side_mask = torch.zeros_like(right_probs)
                left_side_mask[:, :, :, :center_w] = 1.0
                wrong_right = (right_probs * left_side_mask).mean()
                
                spatial_penalty = wrong_left + wrong_right
                
                # Combinación ponderada
                total_loss = 0.4 * dice_loss + 0.4 * focal_loss + 0.2 * spatial_penalty
                
                return total_loss
        
        return DiceFocalSpatialLoss()
    
    elif loss_name == "dice_ce_spatial":
        """DiceCE con constraint espacial"""
        class DiceCESpatialLoss(nn.Module):
            def __init__(self):
                super().__init__()
                self.dice_ce = DiceCELoss(to_onehot_y=True, softmax=True)
                
            def forward(self, logits, targets):
                base_loss = self.dice_ce(logits, targets)
                
                # Agregar penalización espacial
                B, C, D, H, W = logits.shape
                probs = torch.softmax(logits, dim=1)
                center = W // 2
                
                # Penalizaciones
                spatial_loss = 0
                
                # Left hippocampus should be on left side
                if C > 1:  # Si tenemos clase 1
                    left_prob = probs[:, 1]
                    wrong_side = left_prob[:, :, :, center:].mean()
                    spatial_loss += wrong_side
                
                # Right hippocampus should be on right side  
                if C > 2:  # Si tenemos clase 2
                    right_prob = probs[:, 2]
                    wrong_side = right_prob[:, :, :, :center].mean()
                    spatial_loss += wrong_side
                
                return base_loss + 0.3 * spatial_loss
        
        return DiceCESpatialLoss()
    
    elif loss_name == "focal_tversky_spatial":
        """Focal Tversky con awareness espacial"""
        class FocalTverskySpatialLoss(nn.Module):
            def __init__(self):
                super().__init__()
                self.alpha = 0.7  # Peso para false negatives
                self.beta = 0.3   # Peso para false positives
                self.gamma = 2.0  # Focal parameter
                
            def forward(self, logits, y):
                num_classes = logits.shape[1]
                B, C, D, H, W = logits.shape
                
                # One-hot encoding
                y_onehot = F.one_hot(y.long().squeeze(1), num_classes).permute(0,4,1,2,3).float()
                
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
                
                spatial_penalty = 0
                if num_classes > 1:
                    left_wrong = probs_full[:, 1, :, :, center:].mean()
                    spatial_penalty += left_wrong
                if num_classes > 2:
                    right_wrong = probs_full[:, 2, :, :, :center].mean()
                    spatial_penalty += right_wrong
                
                return focal_tversky.mean() + 0.25 * spatial_penalty
        
        return FocalTverskySpatialLoss()
    
    # Mantén las originales si las necesitas
    elif loss_name == "dice_ce":
        return DiceCELoss(to_onehot_y=True, softmax=True)
    
    elif loss_name == "focal_tversky":
        class FocalTverskyLoss(nn.Module):
            def __init__(self):
                super().__init__()
                
            def forward(self, logits, y):
                num_classes = logits.shape[1]
                y_onehot = F.one_hot(y.long().squeeze(1), num_classes).permute(0,4,1,2,3).float()
                
                # Skip background
                y_onehot = y_onehot[:,1:]
                logits_fg = logits[:,1:]
                
                p = torch.softmax(logits_fg, dim=1)
                
                tp = (p * y_onehot).sum(dim=(2,3,4))
                fp = (p * (1 - y_onehot)).sum(dim=(2,3,4))
                fn = ((1 - p) * y_onehot).sum(dim=(2,3,4))
                
                alpha, beta = 0.7, 0.3
                tversky = (tp + 1e-5) / (tp + alpha*fn + beta*fp + 1e-5)
                
                focal_tversky = torch.pow(1.0 - tversky, 2.0)
                
                return focal_tversky.mean()
        
        return FocalTverskyLoss()
    
    # Default con spatial awareness
    return create_loss_function("dice_focal_spatial")