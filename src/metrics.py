
import numpy as np
################################################################################
# UTILIDADES PARA MÉTRICAS (scipy disponible en este entorno)
################################################################################
from scipy.spatial.distance import cdist
def dice_score(pred, gt, cls):
    """
    Dice para una clase específica.
    pred, gt: arrays 3D de ints (0,1,2); cls: entero de clase.
    Devuelve NaN si en GT no hay voxels de esa clase.
    """
    pred_bin = (pred == cls)
    gt_bin   = (gt == cls)
    inter = np.logical_and(pred_bin, gt_bin).sum()
    denom = pred_bin.sum() + gt_bin.sum()
    if denom == 0:
        return np.nan  # No voxels en ninguna => ignorar
    return 2.0 * inter / denom

def hausdorff_distance(pred, gt, cls, percentile=100):
    """
    HD o HD95 (si percentile=95). pred y gt de shape (Z,H,W).
    Si alguna máscara está vacía devuelve NaN.
    """
    p = (pred == cls)
    g = (gt   == cls)
    if p.sum() == 0 or g.sum() == 0:
        return np.nan
    coords_p = np.argwhere(p)
    coords_g = np.argwhere(g)
    dists = cdist(coords_p, coords_g)
    d_p_to_g = dists.min(axis=1)
    d_g_to_p = dists.min(axis=0)
    all_d = np.concatenate([d_p_to_g, d_g_to_p])
    return np.percentile(all_d, percentile)

def assd(pred, gt, cls):
    """
    Average Symmetric Surface Distance (ASSD). 
    Calcula distancias entre voxels de superficie.
    Devuelve NaN si alguna máscara vacía.
    """
    p = (pred == cls)
    g = (gt   == cls)
    if p.sum() == 0 or g.sum() == 0:
        return np.nan
    coords_p = np.argwhere(p)
    coords_g = np.argwhere(g)
    dists = cdist(coords_p, coords_g)
    d_p_to_g = dists.min(axis=1)
    d_g_to_p = dists.min(axis=0)
    return (d_p_to_g.mean() + d_g_to_p.mean()) / 2.0

def rve(pred, gt, cls):
    """
    Relative Volume Error: (Vol_pred - Vol_gt) / Vol_gt.
    Devuelve NaN si Vol_gt=0.
    """
    v_pred = (pred == cls).sum()
    v_gt   = (gt   == cls).sum()
    if v_gt == 0:
        return np.nan
    return (v_pred - v_gt) / v_gt