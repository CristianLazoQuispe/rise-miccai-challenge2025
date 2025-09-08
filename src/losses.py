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

    elif loss_name == "dice_ce_edge":
        dicece = DiceCELoss(to_onehot_y=True, softmax=True)

        # Sobel 3D kernels (aprox) para gradiente |∇p|
        def sobel_3d():
            g = torch.tensor([[-1,0,1]], dtype=torch.float32)
            kx = g.view(1,1,1,1,3)
            ky = g.view(1,1,1,3,1)
            kz = g.view(1,1,3,1,1)
            return kx, ky, kz

        def edge_map_from_labels(y):
            # y: [B,1,D,H,W] -> contornos binarios (dilatación - erosión soft)
            # Usamos max/avg-pooling para aproximar gradiente morfológico 3D
            with torch.no_grad():
                yb = (y > 0).float()  # foreground (incluye L/R)
                k = 1
                pad = (k, k, k)
                d = F.max_pool3d(yb, kernel_size=3, stride=1, padding=1)
                e = -F.max_pool3d(-yb, kernel_size=3, stride=1, padding=1)
                edge = (d - e).clamp(min=0)
                return edge

        def edge_alignment_loss(logits, y, w_edge=1.0):
            p = torch.softmax(logits, dim=1)  # [B,3,D,H,W]
            pf = p[:,1:]                      # solo L/R
            pf = pf.sum(dim=1, keepdim=True)  # foreground prob [B,1,D,H,W]

            kx, ky, kz = sobel_3d()
            kx = kx.to(pf.device); ky = ky.to(pf.device); kz = kz.to(pf.device)

            # derivadas
            gx = F.conv3d(pf, kx, padding=(0,0,1), groups=1)
            gy = F.conv3d(pf, ky, padding=(0,1,0), groups=1)
            gz = F.conv3d(pf, kz, padding=(1,0,0), groups=1)
            grad_mag = torch.sqrt(gx*gx + gy*gy + gz*gz + 1e-8)

            edge = edge_map_from_labels(y)
            # MSE en bordes (ponderado)
            return w_edge * F.mse_loss(grad_mag, grad_mag.detach()*0 + edge)

        def loss_fn(logits, y, w_main=0.8, w_edge=0.2):
            return w_main * dicece(logits, y) + edge_alignment_loss(logits, y, w_edge=w_edge)

        return loss_fn

    elif loss_name == "gdl_volbal":
        gdl = GeneralizedDiceLoss(include_background=True, to_onehot_y=True, softmax=True)

        def volume_balance_loss(logits, y, eps=1e-6):
            with torch.no_grad():
                # fracciones de volumen en GT (L y R)
                gt = y.long().squeeze(1)  # [B,D,H,W]
                gL = (gt==1).float().sum(dim=(1,2,3)) + eps
                gR = (gt==2).float().sum(dim=(1,2,3)) + eps
                gsum = gL + gR
                rL = gL / (gsum + eps)
                rR = gR / (gsum + eps)

            p = torch.softmax(logits, dim=1)
            pL = p[:,1].sum(dim=(1,2,3)) + eps
            pR = p[:,2].sum(dim=(1,2,3)) + eps
            psum = pL + pR
            qL = pL / (psum + eps)
            qR = pR / (psum + eps)

            # KL simétrica entre (rL,rR) y (qL,qR)
            kl1 = (rL * torch.log((rL+eps)/(qL+eps)) + rR * torch.log((rR+eps)/(qR+eps)))
            kl2 = (qL * torch.log((qL+eps)/(rL+eps)) + qR * torch.log((qR+eps)/(rR+eps)))
            return 0.5*(kl1 + kl2).mean()

        def loss_fn(logits, y, w_gdl=0.9, w_vol=0.1):
            return w_gdl * gdl(logits, y) + w_vol * volume_balance_loss(logits, y)

        return loss_fn

    elif loss_name == "dice_bootce":

        def bootstrapped_ce(logits, y, topk=0.2):
            # CE por-vóxel, tomamos el top-k% más difícil
            num_classes = logits.shape[1]
            y_flat = y.long().squeeze(1)  # [B,D,H,W]
            ce = F.cross_entropy(logits, y_flat, reduction='none')  # [B,D,H,W]
            ce_flat = ce.view(ce.shape[0], -1)
            k = (ce_flat.shape[1] * topk)
            k = max(1, int(k))
            vals, _ = torch.topk(ce_flat, k, dim=1)
            return vals.mean()

        def loss_fn(logits, y, w_dice=0.6, w_bce=0.4, topk=0.2):
            # separo el Dice de DiceCE: lo obtengo como Dice + CE con w_ce=0, implementando Dice puro vía DiceCELoss con CE reducido
            # Truco: reuse DiceCELoss pero restando CE. Más simple: computo Dice manual o uso dicece con weight bajo en CE.
            # Para mantenerlo estable, reutilizamos DiceCE “completo” y le sumamos bootstrap CE adicional pero con menor peso.
            return w_dice * DiceCELoss(to_onehot_y=True, softmax=True)(logits, y) + w_bce * bootstrapped_ce(logits, y, topk=topk)

        return loss_fn

    elif loss_name == "focal_foreground_lovasz":
        # Lovasz ya lo tienes implementado arriba como funciones utilitarias:
        #  - lovasz_softmax, lovasz_softmax_loss (si no, copia ese bloque)
        dicece = DiceCELoss(to_onehot_y=True, softmax=True, include_background=True)

        def focal_foreground_ce(logits, y, gamma=2.0, eps=1e-6):
            # CE focal sólo donde y>0
            y_flat = y.long().squeeze(1)  # [B,D,H,W]
            ce = F.cross_entropy(logits, y_flat, reduction='none')  # [B,D,H,W]
            with torch.no_grad():
                mask_fg = (y_flat > 0).float()
            p = torch.softmax(logits, dim=1)
            # prob de clase GT
            p_t = p.gather(1, y_flat.unsqueeze(1)).squeeze(1) + eps
            focal = ((1 - p_t)**gamma) * ce
            # promediamos solo en foreground
            denom = mask_fg.sum().clamp_min(1.0)
            return (focal * mask_fg).sum() / denom

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

        # Usa tu lovasz_softmax_loss definido en tu archivo (bloque 'lovasz_softmax')
        def lovasz_only(logits, y):
            return lovasz_softmax(logits, y, classes="present", per_image=False, ignore_index=None)

        def loss_fn(logits, y, w_lovasz=0.6, w_focalfg=0.4):
            return w_lovasz * lovasz_only(logits, y) + w_focalfg * focal_foreground_ce(logits, y)

        return loss_fn

    elif loss_name == "dice_surface":
        dicece = DiceCELoss(to_onehot_y=True, softmax=True)

        def soft_contour_map(y):
            # mapa de borde suavizado a partir de GT (morph grad + blur local)
            with torch.no_grad():
                yb = (y > 0).float()  # foreground (L|R)
                # gradiente morfológico aproximado
                d = F.max_pool3d(yb, kernel_size=3, stride=1, padding=1)
                e = -F.max_pool3d(-yb, kernel_size=3, stride=1, padding=1)
                edge = (d - e).clamp(min=0)
                # "ensanchamos" el borde para que soporte pequeñas mis-registraciones
                soft = F.max_pool3d(edge, kernel_size=3, stride=1, padding=1)
                return soft

        def surface_loss(logits, y, alpha=1.0):
            p = torch.softmax(logits, dim=1)
            pf = p[:,1:].sum(dim=1, keepdim=True)  # prob foreground
            target = soft_contour_map(y)
            # L1 mejor que L2 en contornos finos
            return alpha * F.l1_loss(pf, target)

        def loss_fn(logits, y, w_main=0.8, w_surf=0.2):
            return w_main * dicece(logits, y) + w_surf * surface_loss(logits, y)

        return loss_fn


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
