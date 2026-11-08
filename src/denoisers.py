import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
from skimage.restoration import denoise_nl_means, denoise_wavelet

def nlm(image, h):
    return denoise_nl_means(image, h=float(h), patch_size=5, patch_distance=6,
                           fast_mode=True, sigma=0.0, preserve_range=True,
                           channel_axis=-1 if image.ndim == 3 else None).astype(np.float32)

def adaptive_h(sigma, k):
    return float(np.clip(k*sigma, 1/255, .25))

def run_baseline(image, method, params):
    if method == 'gaussian':
        s = params['sigma']
        out = gaussian_filter(image, sigma=s if image.ndim == 2 else (s,s,0), mode='reflect')
    elif method == 'median':
        w = params['window']
        out = median_filter(image, size=w if image.ndim == 2 else (w,w,1), mode='reflect')
    elif method == 'bilateral':
        out = cv2.bilateralFilter(np.ascontiguousarray(image), d=7,
                    sigmaColor=params['sigma_color'], sigmaSpace=params['sigma_space'], borderType=cv2.BORDER_REFLECT)
    elif method == 'wavelet':
        out = denoise_wavelet(image, sigma=None, wavelet=params['wavelet'], method=params['method'],
                    mode='soft', convert2ycbcr=False, rescale_sigma=True,
                    channel_axis=-1 if image.ndim == 3 else None)
    elif method == 'nlm':
        out = nlm(image, params['h'])
    else:
        raise ValueError('未知算法: '+method)
    return np.clip(out,0,1).astype(np.float32)
