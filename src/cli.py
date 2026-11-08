import argparse
import yaml
from .common import ROOT, write_json
from .image_io import load_image, save_image
from .adaptive_method import denoise

def main():
    p=argparse.ArgumentParser(description='Reference-free adaptive mixed-noise denoising')
    p.add_argument('input'); p.add_argument('output'); p.add_argument('--max-side',type=int,default=256)
    a=p.parse_args(); cfg=yaml.safe_load((ROOT/'configs/frozen_params.yaml').read_text(encoding='utf-8'))
    x=load_image(a.input,a.max_side); out,d=denoise(x,cfg['selected']['adaptive'])
    save_image(a.output,out)
    write_json(a.output+'.json',{k:v for k,v in d.items() if k not in ['mask','confidence','median_image']})
    print('Saved',a.output)
if __name__=='__main__': main()
