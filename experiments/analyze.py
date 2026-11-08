import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.common import ROOT, write_json, read_json
from src.noise import CONDITIONS

MAIN=['noisy','gaussian','median','bilateral','wavelet','nlm','A4']
LABELS={'noisy':'带噪输入','gaussian':'高斯滤波','median':'中值滤波','bilateral':'双边滤波',
        'wavelet':'小波去噪','nlm':'固定 NLM','A0':'A0 固定 NLM','A1':'A1 全图中值+固定 NLM',
        'A2':'A2 选择修复+固定 NLM','A3':'A3 原图估计+自适应 NLM','A4':'A4 完整方法',
        'A5':'A5 修复+原图估计','pre_median':'仅全图中值','pre_selective':'仅选择修复'}

def figure_style():
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,
        'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
        'figure.facecolor':'white','savefig.facecolor':'white','savefig.dpi':180})

def main():
    figure_style()
    raw=ROOT/'results/raw'; figs=ROOT/'results/figures'; figs.mkdir(parents=True,exist_ok=True)
    q=pd.read_csv(raw/'quality.csv'); m=pd.read_csv(raw/'mechanisms.csv')
    # Unit of cross-image variation: image, after averaging seeds first.
    per=q.groupby(['image','condition','method'],as_index=False)[['psnr','ssim','gradient_mae','ms']].mean()
    per.to_csv(raw/'per_image_seedmean.csv',index=False,encoding='utf-8-sig')
    bycond=per.groupby(['condition','method']).agg(
        psnr_mean=('psnr','mean'),psnr_sd=('psnr','std'),ssim_mean=('ssim','mean'),
        ssim_sd=('ssim','std'),gradient_mean=('gradient_mae','mean'),ms_mean=('ms','mean')).reset_index()
    bycond.to_csv(raw/'summary_by_condition.csv',index=False,encoding='utf-8-sig')
    pi=per.groupby(['image','method'],as_index=False)[['psnr','ssim','gradient_mae','ms']].mean()
    overall=pi.groupby('method').agg(psnr_mean=('psnr','mean'),psnr_sd=('psnr','std'),
        ssim_mean=('ssim','mean'),ssim_sd=('ssim','std'),gradient_mean=('gradient_mae','mean'),ms_mean=('ms','mean')).reset_index()
    overall.to_csv(raw/'summary_overall.csv',index=False,encoding='utf-8-sig')
    pivot=per.pivot(index=['image','condition'],columns='method',values='psnr')
    delta=(pivot['A4']-pivot['nlm']).unstack('condition').reindex(columns=CONDITIONS)
    delta.to_csv(raw/'delta_heatmap.csv',encoding='utf-8-sig')
    ranks=per.copy(); ranks['rank']=ranks.groupby(['image','condition'])['psnr'].rank(ascending=False,method='min')
    ranks.to_csv(raw/'all_method_ranks.csv',index=False,encoding='utf-8-sig')
    main_ranks=per[per.method.isin(MAIN[1:])].copy()
    main_ranks['rank']=main_ranks.groupby(['image','condition'])['psnr'].rank(ascending=False,method='min')
    main_ranks.to_csv(raw/'main_method_ranks.csv',index=False,encoding='utf-8-sig')
    fig,ax=plt.subplots(figsize=(9,3.8)); bound=max(abs(delta.to_numpy()).max(),.1)
    im=ax.imshow(delta,vmin=-bound,vmax=bound,cmap='RdBu',aspect='auto')
    ax.set_xticks(range(7),delta.columns); ax.set_yticks(range(6),delta.index)
    for i in range(6):
        for j in range(7):
            v=delta.iloc[i,j]; ax.text(j,i,f'{v:+.2f}',ha='center',va='center',color='white' if abs(v)>.65*bound else '#142431')
    ax.set_title('完整方法相对固定 NLM 的 PSNR 差值（先平均三个种子）')
    fig.colorbar(im,ax=ax,label='ΔPSNR / dB'); fig.tight_layout(); fig.savefig(figs/'delta_heatmap.png'); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.6))
    for method in MAIN[1:]:
        b=bycond[bycond.method==method].set_index('condition').reindex(CONDITIONS)
        axes[0].plot(list(CONDITIONS),b.psnr_mean,marker='o',label=LABELS[method])
        axes[1].plot(list(CONDITIONS),b.ssim_mean,marker='o',label=LABELS[method])
    axes[0].set_ylabel('PSNR / dB'); axes[1].set_ylabel('SSIM'); axes[0].legend(fontsize=8,ncol=2)
    for ax in axes: ax.grid(alpha=.2); ax.tick_params(axis='x',rotation=25)
    fig.tight_layout(); fig.savefig(figs/'quality_by_condition.png'); plt.close(fig)
    mp=m.drop(columns=['seed']).groupby(['image','condition'],as_index=False).mean(numeric_only=True)
    means=mp.groupby('condition').mean(numeric_only=True).reindex(CONDITIONS)
    means.to_csv(raw/'mechanism_summary.csv',encoding='utf-8-sig')
    fig,axes=plt.subplots(1,2,figsize=(10,3.6)); xs=np.arange(7)
    axes[0].bar(xs-.18,means.sigma_raw_error_255,.36,label='直接 MAD')
    axes[0].bar(xs+.18,means.sigma_repaired_error_255,.36,label='修复后 MAD')
    axes[0].set_ylabel('生成参数恢复 MAE / 灰度级'); axes[0].set_xticks(xs,list(CONDITIONS),rotation=25); axes[0].legend()
    contam=means.loc[['SP05','SP15','MIX-L','MIX-H']]
    for metric,label in [('precision','Precision'),('recall','Recall'),('f1','F1')]:
        axes[1].plot(contam.index,contam[metric],marker='o',label=label)
    axes[1].set_ylim(0,1.05); axes[1].set_ylabel('检测指标（逐输入计算后平均）'); axes[1].legend()
    fig.tight_layout(); fig.savefig(figs/'noise_and_detection.png'); plt.close(fig)
    ab=bycond[bycond.method.isin(['A0','A1','A2','A3','A4','A5'])].pivot(index='condition',columns='method',values='psnr_mean').reindex(CONDITIONS)
    ax=ab.plot.bar(figsize=(10,3.6),width=.83); ax.set_ylabel('PSNR / dB'); ax.set_xlabel(''); ax.tick_params(axis='x',rotation=0); ax.legend(ncol=6)
    ax.figure.tight_layout(); ax.figure.savefig(figs/'ablation.png'); plt.close(ax.figure)
    ms=overall.set_index('method').loc[MAIN[1:],'ms_mean']
    fig,ax=plt.subplots(figsize=(8,3.2)); ax.barh([LABELS[x] for x in ms.index],ms,color='#208681')
    ax.set_xlabel('平均单图计算时间 / ms（不含 IO 与评价）'); fig.tight_layout(); fig.savefig(figs/'runtime.png'); plt.close(fig)
    tuning=pd.read_csv(raw/'tuning.csv'); tuning['parsed']=tuning.params.map(json.loads)
    import yaml
    frozen=yaml.safe_load((ROOT/'configs/frozen_params.yaml').read_text(encoding='utf-8'))['selected']['adaptive']
    tuning=tuning[(tuning.method=='adaptive') & tuning.parsed.map(lambda p:p.get('window')==frozen['window'] and p.get('threshold')==frozen['threshold'])].copy()
    tuning['k']=tuning.parsed.map(lambda p:p['k']); sens=tuning.groupby('k').psnr.mean()
    fig,ax=plt.subplots(figsize=(7,3)); ax.plot(sens.index,sens.values,'o-',color='#208681'); ax.set_xlabel('NLM 系数 k'); ax.set_ylabel('开发集平均 PSNR / dB'); ax.set_title('固定所选检测参数的开发集敏感性（复用选参记录）')
    fig.tight_layout(); fig.savefig(figs/'sensitivity.png'); plt.close(fig)
    contam_per=per[per.condition.isin(['SP05','SP15','MIX-L','MIX-H'])]
    h1p=contam_per.pivot(index=['image','condition'],columns='method',values=['psnr','gradient_mae'])
    h1grad=h1p['gradient_mae']['pre_median']-h1p['gradient_mae']['pre_selective']
    h1psnr=h1p['psnr']['pre_selective']-h1p['psnr']['pre_median']
    mixed=mp[mp.condition.isin(['MIX-L','MIX-H'])]
    overall_index=overall.set_index('method')
    stats={'cases':len(m),'quality_rows':len(q),'test_images':6,'noise_conditions':7,'seeds':3,
       'overall':overall.set_index('method').to_dict(orient='index'),
       'condition_means':{c:bycond[bycond.condition==c].set_index('method').to_dict(orient='index') for c in CONDITIONS},
       'H1':{'gradient_error_reduction':float(h1grad.mean()),'psnr_delta':float(h1psnr.mean()),
             'positive_gradient_cases':int((h1grad>0).sum()),'image_condition_pairs':len(h1grad)},
       'H2':{'raw_mae':float(mixed.sigma_raw_error_255.mean()),'repaired_mae':float(mixed.sigma_repaired_error_255.mean()),
             'positive_pairs':int((mixed.sigma_raw_error_255>mixed.sigma_repaired_error_255).sum()),'pairs':len(mixed)},
       'H3':{'A3_minus_A0':float((pivot.A3-pivot.A0).mean()),'A4_minus_A2':float((pivot.A4-pivot.A2).mean()),
             'A4_minus_A5':float((pivot.A4-pivot.A5).mean())},
       'A4_minus_fixed':float(delta.to_numpy().mean()),'positive_pairs':int((delta>0).to_numpy().sum()),
       'total_pairs':int(delta.size),'A4_A5_preidentical':bool((m.pre_hash_A4==m.pre_hash_A5).all()),
       'best_pair':{'image':delta.stack().idxmax()[0],'condition':delta.stack().idxmax()[1],'delta':float(delta.stack().max())},
       'worst_pair':{'image':delta.stack().idxmin()[0],'condition':delta.stack().idxmin()[1],'delta':float(delta.stack().min())}}
    write_json(raw/'analysis_summary.json',stats)
    # Exact representative cases: first seed, best/worst aggregate pair, plus mandatory text edges.
    examples=[('best',stats['best_pair']['image'],stats['best_pair']['condition']),
              ('worst',stats['worst_pair']['image'],stats['worst_pair']['condition']),
              ('text','text','MIX-H'),('page','page','SP15')]
    for label,name,condition in examples:
        clean=np.load(ROOT/'data'/f'{name}.npy'); stem=f'{name}__{condition}__42'
        with np.load(ROOT/'results/outputs'/f'{stem}.npz') as result:
            methods=['noisy','median','nlm','A4']; imgs=[clean]+[result[x] for x in methods]
            titles=['干净参考']+[LABELS[x] for x in methods]
            fig,axes=plt.subplots(2,5,figsize=(11,4.6))
            height,width=clean.shape[:2]; cy,cx=height//2,width//2; r=max(10,min(height,width)//5)
            for j,(img,title) in enumerate(zip(imgs,titles)):
                axes[0,j].imshow(img,cmap='gray',vmin=0,vmax=1); axes[0,j].set_title(title,fontsize=10)
                axes[1,j].imshow(img[max(0,cy-r):cy+r,max(0,cx-r):cx+r],cmap='gray',vmin=0,vmax=1)
                for ax in axes[:,j]: ax.axis('off')
            fig.suptitle(f'{name} / {condition} / seed 42，上排全图，下排中心局部',fontsize=11)
            fig.tight_layout(); fig.savefig(figs/f'case_{label}.png'); plt.close(fig)
    print(json.dumps(stats,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__': main()
