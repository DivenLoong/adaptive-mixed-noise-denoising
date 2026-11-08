import argparse
import json
import platform
import time
from datetime import datetime, timezone
from src.common import ROOT, set_threads, write_json, read_json, file_hash, array_hash
# Set thread environment BEFORE cv2/NumPy/SciPy load any BLAS runtime.
set_threads()
import numpy as np
import pandas as pd
import yaml
from threadpoolctl import threadpool_info
from src.noise import CONDITIONS, generate_noise
from src.denoisers import run_baseline, nlm, adaptive_h
from src.adaptive_method import prepare
from src.metrics import quality, detection_scores
from src.image_io import save_image

def timed(call):
    t=time.perf_counter(); out=call()
    return np.clip(out,0,1).astype(np.float32),(time.perf_counter()-t)*1000

def verify_freeze():
    receipt=read_json(ROOT/'results/raw/freeze_receipt.json')
    if file_hash(ROOT/'configs/frozen_params.yaml')!=receipt['frozen_sha256']:
        raise RuntimeError('冻结配置已变化，禁止继续正式测试')
    if file_hash(ROOT/'configs/experiment.yaml')!=receipt['protocol_sha256']:
        raise RuntimeError('实验协议已变化，禁止继续正式测试')
    for name,sha in receipt['source_sha256'].items():
        if file_hash(ROOT/name)!=sha: raise RuntimeError('冻结后算法发生变化: '+name)
    return receipt

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args(); set_threads()
    pools=threadpool_info()
    assert all(p['num_threads']==1 for p in pools), pools
    receipt=verify_freeze()
    cfg=yaml.safe_load((ROOT/'configs/experiment.yaml').read_text(encoding='utf-8'))
    frozen=yaml.safe_load((ROOT/'configs/frozen_params.yaml').read_text(encoding='utf-8'))
    selected=frozen['selected']; ap=selected['adaptive']; fixed_h=selected['nlm']['h']
    folder=ROOT/'results'; cache=folder/'cache'; outputs=folder/'outputs'
    cache.mkdir(parents=True,exist_ok=True); outputs.mkdir(parents=True,exist_ok=True)
    names=cfg['test_images'][:2] if args.smoke else cfg['test_images']
    conditions=['G25','MIX-L'] if args.smoke else list(CONDITIONS)
    seeds=cfg['test_seeds'][:1] if args.smoke else cfg['test_seeds']
    # All methods warm up once outside formal timers; single-thread CPU processing.
    warm=np.load(ROOT/'data/astronaut.npy')[:32,:32].copy()
    for method in ['gaussian','median','bilateral','wavelet','nlm']:
        run_baseline(warm,method,selected[method])
    prepare(warm,ap); nlm(warm,.08)
    started=time.perf_counter(); start_utc=datetime.now(timezone.utc).isoformat()
    quality_rows=[]; mechanism_rows=[]; manifest=[]
    total=len(names)*len(conditions)*len(seeds); done=0
    for name in names:
        clean=np.load(ROOT/'data'/f'{name}.npy')
        for condition in conditions:
            sigma,p=CONDITIONS[condition]
            for seed in seeds:
                stem=f'{name}__{condition}__{seed}'
                noisy,truth=generate_noise(clean,sigma,p,seed)
                input_sha=array_hash(noisy)
                np.savez_compressed(cache/f'{stem}.npz',noisy=noisy,selected_mask=truth['selected_mask'],
                    gaussian_clipped=truth['gaussian_clipped'])
                base={'image':name,'condition':condition,'seed':seed,'input_sha256':input_sha,
                      'frozen_sha256':receipt['frozen_sha256']}
                result={'noisy':(noisy,0.)}
                for method in ['gaussian','median','bilateral','wavelet','nlm']:
                    result[method]=timed(lambda m=method:run_baseline(noisy,m,selected[m]))
                pre,d=prepare(noisy,ap)
                immutable_pre=pre.copy(); pre_hash=array_hash(pre)
                hr=adaptive_h(d['sigma_raw'],ap['k']); hp=adaptive_h(d['sigma_repaired'],ap['k'])
                result['A0']=result['nlm']
                median_image,median_ms=timed(lambda:run_baseline(noisy,'median',{'window':ap['window']}))
                assert np.array_equal(median_image,d['median_image']), 'H1 median boundary mismatch'
                result['pre_median']=(median_image,median_ms)
                result['pre_selective']=(pre,d['repair_ms'])
                a1,ms=timed(lambda:nlm(median_image,fixed_h)); result['A1']=(a1,ms+median_ms)
                a2,ms=timed(lambda:nlm(pre,fixed_h)); result['A2']=(a2,ms+d['repair_ms'])
                a3,ms=timed(lambda:nlm(noisy,hr)); result['A3']=(a3,ms+d['estimate_raw_ms'])
                a4,ms=timed(lambda:nlm(pre,hp)); result['A4']=(a4,ms+d['repair_ms']+d['estimate_repaired_ms'])
                assert np.array_equal(pre,immutable_pre) and array_hash(pre)==pre_hash
                a5,ms=timed(lambda:nlm(pre,hr)); result['A5']=(a5,ms+d['repair_ms']+d['estimate_raw_ms'])
                assert np.array_equal(pre,immutable_pre) and array_hash(pre)==d['pre_hash']
                residual=pre-clean; eff=truth['effective_disturbance']
                nonimp=~truth['selected_mask']
                mech={**base,'sigma_g_255':sigma,'p_nominal':p,'p_selected':float(truth['selected_mask'].mean()),
                    'sigma_effective_255':float(np.std(eff)*255),
                    'sigma_effective_nonimpulse_255':float(np.std(eff[nonimp])*255),
                    'residual_std_255':float(np.std(residual)*255),
                    'residual_rmse_255':float(np.sqrt(np.mean(residual**2))*255),
                    'sigma_raw_255':d['sigma_raw']*255,'sigma_repaired_255':d['sigma_repaired']*255,
                    'sigma_raw_channels_255':json.dumps([s*255 for s in d['sigma_raw_channels']]),
                    'sigma_repaired_channels_255':json.dumps([s*255 for s in d['sigma_repaired_channels']]),
                    'sigma_raw_error_255':abs(d['sigma_raw']*255-sigma),
                    'sigma_repaired_error_255':abs(d['sigma_repaired']*255-sigma),
                    'h_A4':hp,'h_A5':hr,'h_fixed':fixed_h,'pre_hash_A4':pre_hash,'pre_hash_A5':array_hash(pre),
                    'replacement_rate':d['replacement_rate'],'repair_enabled':d['repair_enabled'],
                    'repair_ms':d['repair_ms'],**detection_scores(d['mask'],truth['selected_mask'])}
                mechanism_rows.append(mech)
                save={}
                for method,(out,ms) in result.items():
                    quality_rows.append({**base,'method':method,'ms':ms,**quality(clean,out)})
                    save[method]=out
                # Exact float outputs permit independent metric verification.
                np.savez_compressed(outputs/f'{stem}.npz',**save,mask=d['mask'],confidence=d['confidence'])
                manifest.append({**base,'cache':str((cache/f'{stem}.npz').relative_to(ROOT)),
                                 'outputs':str((outputs/f'{stem}.npz').relative_to(ROOT))})
                done+=1
                write_json(folder/'progress.json',{'status':'running','done':done,'total':total,'elapsed_seconds':round(time.perf_counter()-started,2)})
                print(f'{done}/{total} {stem} A4={quality_rows[-2]["psnr"]:.3f} elapsed={time.perf_counter()-started:.1f}s',flush=True)
    suffix='_smoke' if args.smoke else ''
    pd.DataFrame(quality_rows).to_csv(folder/'raw'/f'quality{suffix}.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(mechanism_rows).to_csv(folder/'raw'/f'mechanisms{suffix}.csv',index=False,encoding='utf-8-sig')
    import cv2, scipy, skimage, pywt
    write_json(folder/'raw'/f'run_manifest{suffix}.json',{'started_utc':start_utc,
        'completed_utc':datetime.now(timezone.utc).isoformat(),'seconds':time.perf_counter()-started,
        'cases':manifest,'cpu':platform.processor(),'platform':platform.platform(),'threads':1,'threadpools':pools,
        'timing':'warmup excluded; perf_counter; IO/metrics excluded; shared preprocessing charged per logical pipeline; A0 aliases nlm',
        'versions':{'numpy':np.__version__,'opencv':cv2.__version__,'scipy':scipy.__version__,'skimage':skimage.__version__,'PyWavelets':pywt.__version__},
        'freeze_receipt':receipt})
    write_json(folder/'progress.json',{'status':'complete','done':done,'total':total,'seconds':time.perf_counter()-started})
    print('BENCHMARK COMPLETE',flush=True)
if __name__=='__main__': main()
