"""zed_mask :: black out everything outside the table workspace in the fixed ZED view, identically for training
features, the twin and the real robot (ALOHA Lightning-style masking of what differs between training and deployment).
The quad is the sim table footprint (slightly dilated) in the 432x240 ZED frame; the camera is fixed, so the mask is too."""
import cv2
import numpy as np

QUAD_432 = np.array([[162, 85], [280, 85], [305, 199], [160, 203]], np.float64)
_cache = {}


def mask(h, w):
    if (h, w) not in _cache:
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [np.round(QUAD_432 * [w / 432.0, h / 240.0]).astype(np.int32)], 1)
        _cache[(h, w)] = m
    return _cache[(h, w)]


def apply(img):
    """HxWx3 uint8 (any channel order) -> copy with the off-table pixels set to zero"""
    return img * mask(*img.shape[:2])[:, :, None]
