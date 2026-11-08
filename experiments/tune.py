import argparse
import itertools
import json
import time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import yaml
from src.common import ROOT, set_threads, write_json, file_hash
from src.noise import CONDITIONS, generate_noise
from src.denoisers import run_baseline
from src.adaptive_method import denoise
from src.metrics import quality

def grid(values):
    return [dict(zip(values, v)) for v in itertools.product(*values.values())]

def main():
    set_threads()
    config_path=ROOT/'configs/experiment.yaml'
    cfg=yaml.safe_load(config_path.read_text(encoding='utf-8'))
    protocol_hash=file_hash(config_path)
    cases=[]
    for name in cfg['development_images']:
        clean=np.load(ROOT/'data'/f'{name}.npy')
        for condition,(sigma,p) in CONDITIONS.items():
            noisy,_=generate_noise(clean,sigma,p,cfg['development_seed'])
            cases.append((name,condition,clean,noisy))
    rows=[]; selected={}; summary=[]
    grids=dict(cfg['baseline_grids']); grids['adaptive']=cfg['adaptive_grid']
    for method, values in grids.items():
        candidates=grid(values)
        for number,params in enumerate(candidates):
            t=time.perf_counter(); current=[]
            for name,condition,clean,noisy in cases:
                out=denoise(noisy,params)[0] if method=='adaptive' else run_baseline(noisy,method,params)
                score=quality(clean,out)
                record={'method':method,'candidate':number,'params':json.dumps(params,sort_keys=True),
                        'image':name,'condition':condition,**score}
                rows.append(record); current.append(record)
            mean_psnr=float(np.mean([x['psnr'] for x in current])); mean_ssim=float(np.mean([x['ssim'] for x in current]))
            summary.append({'method':method,'candidate':number,'params':params,'psnr':mean_psnr,'ssim':mean_ssim})
            print(f'{method} {number+1}/{len(candidates)} PSNR={mean_psnr:.4f} SSIM={mean_ssim:.4f} seconds={time.perf_counter()-t:.2f}',flush=True)
        best=max([x for x in summary if x['method']==method],key=lambda x:(x['psnr'],x['ssim'],-x['candidate']))
        selected[method]=best['params']
    outdir=ROOT/'results/raw'; outdir.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(outdir/'tuning.csv',index=False,encoding='utf-8-sig')
    frozen={'created_utc':datetime.now(timezone.utc).isoformat(),
            'protocol_sha256':protocol_hash,'selection':'development macro PSNR, tie SSIM, then candidate order',
            'development_images':cfg['development_images'],'development_seed':cfg['development_seed'],
            'selected':selected,'candidate_summaries':summary,
            'nlm':cfg['nlm']}
    frozen_path=ROOT/'configs/frozen_params.yaml'
    if frozen_path.exists():
        raise RuntimeError('冻结配置已经存在。为避免覆盖正式实验，需在新的研究目录重新选参。')
    frozen_path.write_text(yaml.safe_dump(frozen,allow_unicode=True,sort_keys=False),encoding='utf-8')
    write_json(outdir/'freeze_receipt.json',{'frozen_sha256':file_hash(frozen_path),'protocol_sha256':protocol_hash,
                'source_sha256':{str(p.relative_to(ROOT)):file_hash(p) for p in sorted((ROOT/'src').glob('*.py'))},
                'test_not_used':True,'selected':selected,'created_utc':frozen['created_utc']})
    print('FROZEN',selected,flush=True)
if __name__=='__main__': main()
