from io import BytesIO
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from skimage.transform import resize

def validate_image(x):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim not in (2, 3) or (x.ndim == 3 and x.shape[-1] != 3):
        raise ValueError('仅支持灰度或 RGB 图像')
    if min(x.shape[:2]) < 8 or not np.isfinite(x).all():
        raise ValueError('图像宽高至少为 8，像素必须为有限值')
    if x.min() < 0 or x.max() > 1:
        raise ValueError('图像数值范围必须为 [0,1]')
    return np.ascontiguousarray(x)

def resize_image(x, max_side=256):
    x = validate_image(x)
    if max(x.shape[:2]) <= max_side:
        return x.copy()
    scale = max_side / max(x.shape[:2])
    shape = (max(8, round(x.shape[0]*scale)), max(8, round(x.shape[1]*scale)))
    if x.ndim == 3:
        shape += (3,)
    return resize(x, shape, order=1, mode='reflect', anti_aliasing=True, preserve_range=True).astype(np.float32)

def load_image(source, max_side=256):
    try:
        with Image.open(source) as im:
            im = ImageOps.exif_transpose(im)
            # Transparent images are composited on white, not silently discarded.
            if im.mode in ('RGBA', 'LA') or 'transparency' in im.info:
                rgba = im.convert('RGBA')
                bg = Image.new('RGBA', rgba.size, 'white')
                im = Image.alpha_composite(bg, rgba).convert('RGB')
            elif im.mode not in ('L', 'RGB'):
                im = im.convert('RGB')
            x = np.asarray(im, dtype=np.float32) / 255.0
        return resize_image(x, max_side)
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError('无法解码图像，请选择有效 PNG 或 JPEG 文件') from exc

def png_bytes(x):
    b = BytesIO()
    Image.fromarray(np.round(np.clip(x, 0, 1)*255).astype('uint8')).save(b, format='PNG')
    return b.getvalue()

def save_image(path, x):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(png_bytes(x))

def luminance(x):
    return x if x.ndim == 2 else np.einsum('ijk,k->ij', x, np.array([.2126,.7152,.0722],dtype=np.float32))
