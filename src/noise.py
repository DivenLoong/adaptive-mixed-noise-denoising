"""Synthetic data generation. Truth objects must never enter a denoiser."""
import numpy as np
from .image_io import validate_image

CONDITIONS = {
    'G10': (10, 0), 'G25': (25, 0), 'G50': (50, 0),
    'SP05': (0, .05), 'SP15': (0, .15),
    'MIX-L': (15, .05), 'MIX-H': (35, .15),
}

def generate_noise(clean, sigma_255, probability, seed):
    clean = validate_image(clean)
    if not 0 <= sigma_255 <= 100 or not 0 <= probability <= 1:
        raise ValueError('噪声参数超出允许范围')
    gaussian_rng, impulse_rng = [np.random.default_rng(s) for s in np.random.SeedSequence(int(seed)).spawn(2)]
    gaussian = gaussian_rng.normal(0, sigma_255 / 255, clean.shape)
    clipped = np.clip(clean + gaussian, 0, 1).astype(np.float32)
    draws = impulse_rng.random(clean.shape[:2])
    selected = draws < probability
    salt = draws < probability/2
    noisy = clipped.copy()
    noisy[selected] = 0
    noisy[salt] = 1
    truth = {
        'selected_mask': selected, 'gaussian_clipped': clipped,
        'effective_disturbance': clipped - clean,
        'sigma_g_255': float(sigma_255), 'probability': float(probability),
    }
    return noisy, truth
