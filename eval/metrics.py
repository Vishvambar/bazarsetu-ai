"""
BazarSetu V3.0 - Evaluation Metrics Harness
Implements IoU, Boundary F1, and CIEDE2000 Delta E00.
"""

from __future__ import annotations
from typing import Optional, Tuple
import cv2
import numpy as np

def compute_mask_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Computes Intersection over Union for binary segmentation masks."""
    p_bin = pred_mask > 128
    g_bin = gt_mask > 128
    intersection = np.logical_and(p_bin, g_bin).sum()
    union = np.logical_or(p_bin, g_bin).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection / union)

def compute_boundary_f1(pred_mask: np.ndarray, gt_mask: np.ndarray, tolerance_px: int = 4) -> float:
    """
    Standard Boundary F1-score (BF-score).
    Evaluates contour alignment with spatial tolerance theta.
    """
    p_bin = (pred_mask > 128).astype(np.uint8)
    g_bin = (gt_mask > 128).astype(np.uint8)

    k1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    b_pred = cv2.morphologyEx(p_bin, cv2.MORPH_GRADIENT, k1) > 0
    b_gt = cv2.morphologyEx(g_bin, cv2.MORPH_GRADIENT, k1) > 0

    pred_count = b_pred.sum()
    gt_count = b_gt.sum()

    if pred_count == 0 and gt_count == 0:
        return 1.0
    if pred_count == 0 or gt_count == 0:
        return 0.0

    # Tolerance dilation window (standard spatial slack theta)
    k_tol = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * tolerance_px + 1, 2 * tolerance_px + 1))
    b_pred_dil = cv2.dilate(b_pred.astype(np.uint8), k_tol) > 0
    b_gt_dil = cv2.dilate(b_gt.astype(np.uint8), k_tol) > 0

    precision = np.logical_and(b_pred, b_gt_dil).sum() / (pred_count + 1e-6)
    recall = np.logical_and(b_gt, b_pred_dil).sum() / (gt_count + 1e-6)

    if precision + recall == 0:
        return 0.0
    return float(2 * (precision * recall) / (precision + recall))

def compute_ciede2000_delta(orig_rgb: np.ndarray, polished_rgb: np.ndarray, mask: np.ndarray) -> float:
    """Computes mean Delta E00 across foreground pixels."""
    fg = mask > 128
    if not np.any(fg):
        return 0.0
    
    p1 = orig_rgb[fg]
    p2 = polished_rgb[fg]

    try:
        from skimage import color
        lab1 = color.rgb2lab(p1.reshape(-1, 1, 3) / 255.0)
        lab2 = color.rgb2lab(p2.reshape(-1, 1, 3) / 255.0)
        deltas = color.deltaE_ciede2000(lab1, lab2)
        return float(np.mean(deltas))
    except ImportError:
        lab1 = cv2.cvtColor(p1.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
        lab2 = cv2.cvtColor(p2.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
        return float(np.mean(np.linalg.norm(lab1 - lab2, axis=2)))

