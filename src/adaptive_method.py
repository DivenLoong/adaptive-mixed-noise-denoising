from time import perf_counter
import numpy as np
from .image_io import validate_image
from .noise_estimation import estimate_noise
from .impulse_detection import detect_and_repair
from .denoisers import nlm, adaptive_h
from .common import array_hash

def prepare(image, params):
    image = validate_image(image)
    t = perf_counter()
    repaired, decision = detect_and_repair(image, params['window'], params['threshold'])
    repair_ms = (perf_counter()-t)*1000
    t = perf_counter(); raw, raw_channels = estimate_noise(image); raw_ms=(perf_counter()-t)*1000
    t = perf_counter(); post, post_channels = estimate_noise(repaired); post_ms=(perf_counter()-t)*1000
    decision.update(sigma_raw=raw, sigma_repaired=post, sigma_raw_channels=raw_channels,
                    sigma_repaired_channels=post_channels, repair_ms=repair_ms,
                    estimate_raw_ms=raw_ms, estimate_repaired_ms=post_ms,
                    pre_hash=array_hash(repaired))
    return repaired, decision

def denoise(image, params):
    """Public inference API: no clean reference or noise truth accepted."""
    t = perf_counter()
    repaired, d = prepare(image, params)
    h = adaptive_h(d['sigma_repaired'], params['k'])
    out = np.clip(nlm(repaired, h),0,1).astype(np.float32)
    d.update(h=h, total_ms=(perf_counter()-t)*1000)
    return out, d
