from pathlib import Path
import inspect
import numpy as np
from skimage import data, img_as_float32
from src.common import ROOT, write_json, array_hash
from src.image_io import resize_image, save_image

NAMES = ['camera','coffee','astronaut','chelsea','coins','moon','page','text']
def main():
    folder=ROOT/'data'; folder.mkdir(exist_ok=True)
    manifest=[]; sources=['# 图像来源与使用说明','',
        '图像来自 scikit-image 0.26.0 的公开示例数据。各图使用条件分别列于官方 API，不将软件整体许可证自动等同于每张图像的许可。',
        '官方页面：https://scikit-image.org/docs/stable/api/skimage.data.html','',
        '仅用于本地课程实验；公开仓库前应逐图确认再分发条件。输入缓存与 PNG 可由本脚本重建。','']
    for name in NAMES:
        original=img_as_float32(getattr(data,name)())
        x=resize_image(original,256)
        np.save(folder/f'{name}.npy',x)
        save_image(folder/'preview'/f'{name}.png',x)
        manifest.append({'name':name,'split':'dev' if name in NAMES[:2] else 'test',
              'original_shape':list(original.shape),'processed_shape':list(x.shape),
              'sha256':array_hash(x),'interpolation':'bilinear order=1 anti_aliasing=True mode=reflect',
              'source':f'https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.{name}'})
        sources.extend([f'## {name}', '', inspect.getdoc(getattr(data,name)), ''])
    write_json(folder/'manifest.json',manifest)
    (folder/'DATA_SOURCES.md').write_text('\n'.join(sources),encoding='utf-8')
    print('Prepared 8 images, max side 256, dev/test fixed.',flush=True)
if __name__=='__main__': main()
