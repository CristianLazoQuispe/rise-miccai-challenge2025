from monai.losses import DiceCELoss
from monai.losses import DiceCELoss
from monai.losses import GeneralizedDiceLoss, FocalLoss
from monai.losses import DiceFocalLoss
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceCELoss, HausdorffDTLoss
import torch.nn.functional as F
from monai.losses import DiceFocalLoss
import torch
import torch.nn.functional as F


def create_loss_function(loss_name="dice_ce"):

    if loss_name == "dice_ce":
        return DiceCELoss(to_onehot_y=True, softmax=True)
        
    elif loss_name == "dice_ce_hd95":
        # y debe ser [B,1,D,H,W] con índices {0,1,2}
        dicece = DiceCELoss(
            to_onehot_y=True,    # convierte la GT a one-hot
            softmax=True,        # aplica softmax a logits
            include_background=True
        )

        # Aproxima Hausdorff via Distance Transform (no hay "percentile" aquí)
        hd = HausdorffDTLoss(
            alpha=2.0,                 # potencia del DT (2.0 suele ir bien)
            include_background=False,  # sólo L y R
            to_onehot_y=True,
            softmax=True,
            reduction="mean",
            batch=True                 # agrega por batch (más estable con clases pequeñas)
        )

        def loss_dicece_hd95(logits, y, w_dice=0.7, w_hd=0.3):
            # combinación estable
            return w_dice * dicece(logits, y) + w_hd * hd(logits, y)

        return loss_dicece_hd95

    elif loss_name == "gdl_focal":

        gdl = GeneralizedDiceLoss(include_background=True, to_onehot_y=True, softmax=True)
        fl  = FocalLoss(to_onehot_y=True, use_softmax=True, gamma=2.0)

        def loss_gdl_focal(logits, y):
            return 0.6 * gdl(logits, y) + 0.4 * fl(logits, y)
        return loss_gdl_focal
    
    elif loss_name == "focal_tversky":


        class FocalTverskyLoss(nn.Module):
            def __init__(self, alpha=0.7, beta=0.3, gamma=4/3, eps=1e-6, include_background=True):
                super().__init__()
                self.alpha, self.beta, self.gamma = alpha, beta, gamma
                self.eps = eps
                self.include_background = include_background

            def forward(self, logits, y):
                # y: (B,1,D,H,W) con índices 0/1/2 -> onehot
                num_classes = logits.shape[1]
                y_onehot = F.one_hot(y.long().squeeze(1), num_classes=num_classes).permute(0,4,1,2,3).float()
                if not self.include_background and num_classes > 1:
                    y_onehot = y_onehot[:,1:]
                    logits   = logits[:,1:]
                p = torch.softmax(logits, dim=1)

                dims = tuple(range(2, p.ndim))
                tp = (p * y_onehot).sum(dims)
                fp = (p * (1 - y_onehot)).sum(dims)
                fn = ((1 - p) * y_onehot).sum(dims)

                tversky = (tp + self.eps) / (tp + self.alpha*fn + self.beta*fp + self.eps)
                focal_tversky = torch.pow((1.0 - tversky), self.gamma)
                return focal_tversky.mean()

        ftl = FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=1.33, include_background=True)
        return ftl
    

    elif loss_name == "dice_ce_symmetry":

        dicece = DiceCELoss(to_onehot_y=True, softmax=True)

        def symmetry_regularizer(logits, axis=3):  # axis: W en [B,C,D,H,W]
            p = torch.softmax(logits, dim=1)
            p_flip = torch.flip(p, dims=[axis])
            # swap L<->R channels (asumiendo [0:bg,1:L,2:R])
            p_flip_swapped = torch.stack([p_flip[:,0], p_flip[:,2], p_flip[:,1]], dim=1)
            return F.mse_loss(p[:,1:], p_flip_swapped[:,1:])  # compara solo L/R

        def loss_dicece_sym(logits, y, lam_sym=0.1):
            return dicece(logits, y) + lam_sym * symmetry_regularizer(logits, axis=4)  # eje=4 si W es el último

        return loss_dicece_sym

    elif loss_name == "lovasz_softmax":

        def lovasz_grad(gt_sorted):
            # gradiente de Lovasz extension para IoU
            gts = gt_sorted.sum()
            if gts == 0:
                return gt_sorted * 0.0
            intersection = gts - gt_sorted.cumsum(0)
            union = gts + (1 - gt_sorted).cumsum(0)
            jaccard = 1.0 - intersection / union
            if gt_sorted.numel() > 1:
                jaccard[1:] = jaccard[1:] - jaccard[:-1]
            return jaccard

        def flatten_probas(probas, labels, ignore=None):
            # probas: [B,C,*], labels: [B,1,*] o [B,*]
            if labels.ndim == probas.ndim:
                labels = labels.squeeze(1)
            B, C = probas.shape[:2]
            probas = probas.permute(0, *range(2, probas.ndim), 1).contiguous().view(-1, C)  # [N,C]
            labels = labels.contiguous().view(-1)  # [N]
            if ignore is None:
                return probas, labels
            valid = labels != ignore
            return probas[valid], labels[valid]

        def lovasz_softmax_flat(probas, labels, classes='present'):
            """
            probas: [N, C] probs (softmax ya aplicado)
            labels: [N] int64
            classes: 'all' | 'present' | lista de ints
            """
            C = probas.size(1)
            losses = []
            class_to_sum = range(C) if classes in ['all', 'present'] else classes
            for c in class_to_sum:
                fg = (labels == c).float()            # foreground para clase c
                if (classes == 'present') and (fg.sum() == 0):
                    continue
                if fg.numel() == 0:
                    continue
                # error absoluto por clase c (1 - p_c para pixeles de clase c; p_c para resto)
                pc = probas[:, c]
                errors = (fg - pc).abs()
                # ordenar por error desc
                errors_sorted, perm = torch.sort(errors, descending=True)
                fg_sorted = fg[perm]
                grad = lovasz_grad(fg_sorted)
                loss_c = torch.dot(F.relu(errors_sorted), grad)
                losses.append(loss_c)
            if len(losses) == 0:
                return probas.sum() * 0.0
            return torch.mean(torch.stack(losses))

        def lovasz_softmax(logits, labels, classes='present', per_image=False, ignore_index=None):
            """
            logits: [B,C,D,H,W], labels: [B,1,D,H,W] (índices)
            """
            if per_image:
                loss = 0.0
                for logit, label in zip(logits, labels):
                    probas = torch.softmax(logit.unsqueeze(0), dim=1)  # [1,C,D,H,W]
                    probas, lab = flatten_probas(probas, label.unsqueeze(0), ignore_index)
                    loss += lovasz_softmax_flat(probas, lab, classes=classes)
                return loss / logits.shape[0]
            else:
                probas = torch.softmax(logits, dim=1)
                probas, lab = flatten_probas(probas, labels, ignore_index)
                return lovasz_softmax_flat(probas, lab, classes=classes)


        dicece = DiceCELoss(to_onehot_y=True, softmax=True, include_background=True)

        def lovasz_softmax_loss(logits, y, classes="present"):
            # logits: [B,C,D,H,W], y: [B,1,D,H,W] con índices {0,1,2}
            return lovasz_softmax(logits, y, classes=classes, per_image=False, ignore_index=None)

        def loss_fn(logits, y, w_dice=0.5, w_lovasz=0.5):
            return w_dice * dicece(logits, y) + w_lovasz * lovasz_softmax_loss(logits, y)

        return loss_fn
