import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.filters import sobel
from .image_io import luminance

def quality(clean, output):
    if clean.shape != output.shape:
        raise ValueError('参考图与输出图尺寸不一致')
    return {
        'psnr': float(peak_signal_noise_ratio(clean, output, data_range=1)),
        'ssim': float(structural_similarity(clean, output, data_range=1,
                                           channel_axis=-1 if clean.ndim == 3 else None)),
        'gradient_mae': float(np.mean(np.abs(sobel(luminance(clean))-sobel(luminance(output))))),
    }

def detection_scores(predicted, selected):
    tp = int(np.count_nonzero(predicted & selected))
    fp = int(np.count_nonzero(predicted & ~selected))
    fn = int(np.count_nonzero(~predicted & selected))
    tn = int(np.count_nonzero(~predicted & ~selected))
    present = bool(selected.any())
    return {'tp':tp, 'fp':fp, 'fn':fn, 'tn':tn,
            'precision': (tp/(tp+fp) if tp+fp else 0.) if present else None,
            'recall': tp/(tp+fn) if present else None,
            'f1': (2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.) if present else None,
            'false_positive_rate': fp/(fp+tn) if fp+tn else None}
