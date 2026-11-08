import numpy as np
from scipy.ndimage import median_filter, uniform_filter
from numpy.lib.stride_tricks import sliding_window_view
from .image_io import luminance

def detect_and_repair(image, window=3, threshold=.6):
    if window not in (3, 5) or not 0 <= threshold <= 1:
        raise ValueError('无效的检测参数')
    y = luminance(image)
    pad = window//2
    patches = sliding_window_view(np.pad(y, pad, mode='symmetric'), (window, window))
    med = np.median(patches, axis=(-2, -1))
    mad = np.median(np.abs(patches-med[..., None, None]), axis=(-2, -1))
    scale = 1.4826*mad
    eps = 2/255
    if image.ndim == 2:
        extreme = (image <= eps) | (image >= 1-eps)
    else:
        extreme = np.all(image <= eps, axis=-1) | np.all(image >= 1-eps, axis=-1)
    ordinary = (~extreme).astype(np.float32)
    support = (uniform_filter(ordinary, size=window, mode='reflect')*window**2-ordinary)/(window**2-1)
    confidence = extreme * np.clip(support, 0, 1) * np.clip((np.abs(y-med)-2*scale)/(scale+1/255), 0, 1)
    mask = confidence > threshold
    whole = median_filter(image, size=window if image.ndim == 2 else (window, window, 1), mode='reflect')
    repaired = image.copy()
    repaired[mask] = whole[mask]
    return repaired, {'mask': mask, 'confidence': confidence.astype(np.float32),
                      'median_image': whole.astype(np.float32),
                      'estimated_impulse_rate': float(mask.mean()),
                      'replacement_rate': float(mask.mean()),
                      'repair_enabled': bool(mask.any())}
