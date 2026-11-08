import numpy as np
import pywt

def estimate_noise(image):
    """Centered HH1 Haar MAD, symmetric boundary; RGB arithmetic mean."""
    channels = [image] if image.ndim == 2 else [image[..., i] for i in range(3)]
    sigmas = []
    for ch in channels:
        _, (_, _, hh) = pywt.dwt2(ch, 'haar', mode='symmetric')
        sigmas.append(float(np.median(np.abs(hh - np.median(hh))) / .6744897501960817))
    return float(np.mean(sigmas)), sigmas
