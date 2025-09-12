import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceLoss, FocalLoss, TverskyLoss

def create_loss_function(loss_name="dice_focal_hippocampus"):
    """Loss functions optimizadas para hipocampo"""
    
    if loss_name == "dice_focal_hippocampus":
        """Loss optimizada para estructuras pequeñas y desbalanceadas"""
        class DiceFocalHippocampusLoss(nn.Module):
            def __init__(self):
                super().__init__()
                # Pesos: bajo para background, alto para L/R hippocampus
                self.class_weights = torch.tensor([0.05, 1.0, 1.0])
                
                self.dice = DiceLoss(
                    include_background=False,  # Ignora background
                    to_onehot_y=True,
                    softmax=True,
                    squared_pred=True,  # Más estable
                    smooth_nr=1e-5,
                    smooth_dr=1e-5
                )
                
                self.focal = FocalLoss(
                    include_background=False,
                    to_onehot_y=True,
                    use_softmax=True,
                    gamma=2.5,  # Alto para estructuras pequeñas
                    weight=self.class_weights
                )
            
            def forward(self, logits, targets):
                dice_loss = self.dice(logits, targets)
                focal_loss = self.focal(logits, targets)
                return 0.5 * dice_loss + 0.5 * focal_loss
        
        return DiceFocalHippocampusLoss()
    
    elif loss_name == "tversky_hippocampus":
        """Tversky loss con peso alto en false negatives"""
        class TverskyHippocampusLoss(nn.Module):
            def __init__(self):
                super().__init__()
                self.tversky = TverskyLoss(
                    include_background=False,
                    to_onehot_y=True,
                    softmax=True,
                    alpha=0.7,  # Peso alto para FN (miss hippocampus)
                    beta=0.3,   # Peso bajo para FP
                    smooth_nr=1e-5,
                    smooth_dr=1e-5
                )
                
            def forward(self, logits, targets):
                return self.tversky(logits, targets)
        
        return TverskyHippocampusLoss()
    
    elif loss_name == "dice_ce_boundary":
        """Combinación con boundary loss para bordes precisos"""
        class DiceCEBoundaryLoss(nn.Module):
            def __init__(self):
                super().__init__()
                self.dice = DiceLoss(
                    include_background=False,
                    to_onehot_y=True,
                    softmax=True
                )
                self.ce = nn.CrossEntropyLoss(
                    weight=torch.tensor([0.1, 1.0, 1.0])
                )
                
            def forward(self, logits, targets):
                dice_loss = self.dice(logits, targets)
                
                # CE loss
                targets_squeezed = targets.squeeze(1).long()
                ce_loss = self.ce(logits, targets_squeezed)
                
                # Simple boundary weighting
                with torch.no_grad():
                    # Detecta bordes con max pooling
                    kernel_size = 3
                    targets_float = targets.float()
                    dilated = F.max_pool3d(targets_float, kernel_size, stride=1, padding=1)
                    eroded = -F.max_pool3d(-targets_float, kernel_size, stride=1, padding=1)
                    boundary = ((dilated - eroded) > 0).float()
                    weights = 1.0 + 2.0 * boundary  # Peso extra en bordes
                
                # Weighted loss
                total_loss = dice_loss + 0.3 * ce_loss
                weighted_loss = total_loss * weights.mean()
                
                return weighted_loss
        
        return DiceCEBoundaryLoss()
    
    # Mantén tus losses existentes
    from monai.losses import DiceCELoss
    
    if loss_name == "dice_ce":
        return DiceCELoss(to_onehot_y=True, softmax=True)
    
    elif loss_name == "dice_ce_symmetry":
        dicece = DiceCELoss(to_onehot_y=True, softmax=True)
        
        def symmetry_loss(logits, y, lam_sym=0.05):
            p = torch.softmax(logits, dim=1)
            # Flip en eje lateral (último)
            p_flip = torch.flip(p, dims=[4])
            # Swap canales L/R (1 y 2)
            p_flip_swapped = torch.stack([p_flip[:,0], p_flip[:,2], p_flip[:,1]], dim=1)
            # MSE solo en hippocampus, no background
            return dicece(logits, y) + lam_sym * F.mse_loss(p[:,1:], p_flip_swapped[:,1:])
        
        return symmetry_loss
    
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
                
                # Tversky components
                tp = (p * y_onehot).sum(dim=(2,3,4))
                fp = (p * (1 - y_onehot)).sum(dim=(2,3,4))
                fn = ((1 - p) * y_onehot).sum(dim=(2,3,4))
                
                # High alpha for hippocampus detection
                alpha, beta = 0.7, 0.3
                tversky = (tp + 1e-5) / (tp + alpha*fn + beta*fp + 1e-5)
                
                # Focal component
                focal_tversky = torch.pow(1.0 - tversky, 2.0)
                
                return focal_tversky.mean()
        
        return FocalTverskyLoss()
    
    # Default fallback
    return DiceCELoss(to_onehot_y=True, softmax=True)