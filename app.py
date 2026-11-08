from pathlib import Path
from io import BytesIO
import json
import subprocess
import sys
import zipfile
import numpy as np
import pandas as pd
import streamlit as st
import yaml
from src.common import ROOT, read_json
from src.image_io import load_image, png_bytes, luminance
from src.noise import generate_noise, CONDITIONS
from src.adaptive_method import denoise
from src.denoisers import run_baseline
from src.metrics import quality

st.set_page_config(page_title='混合噪声去噪实验室',page_icon='◈',layout='wide')
st.markdown('''<style>
.stApp {background:#f5f8fb;} .block-container {max-width:1280px;padding-top:2rem;}
h1,h2,h3 {color:#17394c;} [data-testid="stMetric"] {background:white;padding:14px;border-radius:8px;border:1px solid #e4ebf0;}
div.stButton > button[kind="primary"] {background:#147d83;border:0;}
</style>''',unsafe_allow_html=True)
st.title('混合噪声去噪实验室')
st.caption('选择性脉冲修复 · Haar MAD 噪声估计 · 自适应非局部均值 | 无需训练的课程研究系统')
frozen_path=ROOT/'configs/frozen_params.yaml'
if not frozen_path.exists():
    st.info('请先按 README 运行数据准备与开发集选参。'); st.stop()
frozen=yaml.safe_load(frozen_path.read_text(encoding='utf-8'))
params=frozen['selected']
tab1,tab2,tab3=st.tabs(['图像实验','批量结果','方法与复现'])

def safe_decision(d):
    return {k:v for k,v in d.items() if k not in ['mask','confidence','median_image']}

with tab1:
    control,display=st.columns([1,2.5],gap='large')
    with control:
        st.subheader('输入与设置')
        uploaded=st.file_uploader('上传 PNG / JPEG',type=['png','jpg','jpeg'])
        example=st.selectbox('内置示例',['astronaut','chelsea','coins','moon','page','text','camera','coffee'])
        try:
            clean=load_image(uploaded) if uploaded is not None else np.load(ROOT/'data'/f'{example}.npy')
        except (ValueError,FileNotFoundError) as exc:
            st.error(str(exc)); st.stop()
        simulate=st.checkbox('模拟噪声（保留干净参考）',value=True)
        condition=st.selectbox('噪声条件',list(CONDITIONS),index=5,disabled=not simulate)
        seed=st.number_input('随机种子',min_value=0,max_value=2147483647,value=42,step=1,disabled=not simulate)
        if simulate:
            sigma,p=CONDITIONS[condition]; noisy,_=generate_noise(clean,sigma,p,int(seed))
            st.caption(f'高斯 σ={sigma} / 255，椒盐 p={p:.0%}；先裁剪，再覆盖。')
        else:
            noisy=clean.copy(); st.caption('直接处理输入；无干净参考，不提供 PSNR / SSIM。')
        names={'完整自适应方法':'adaptive','固定 NLM':'nlm','中值滤波':'median','高斯滤波':'gaussian','双边滤波':'bilateral','小波去噪':'wavelet'}
        choice=st.selectbox('去噪方法',list(names))
        manual=st.checkbox('手动调整参数',value=False)
        ap=dict(params['adaptive']); fixed_h=params['nlm']['h']
        if manual:
            with st.expander('参数调整（不修改正式实验配置）',expanded=True):
                ap['window']=st.selectbox('修复窗口',[3,5],index=1 if ap['window']==5 else 0)
                ap['threshold']=st.slider('置信度门槛',.1,.9,float(ap['threshold']),.1)
                ap['k']=st.slider('自适应系数 k',.2,1.5,float(ap['k']),.1)
                fixed_h=st.slider('固定 NLM 强度 h',.005,.25,float(fixed_h),.005)
        run=st.button('运行去噪与对比',type='primary',width='stretch')
    with display:
        if run:
            with st.spinner('正在检测污染并执行去噪…'):
                adaptive,decision=denoise(noisy,ap)
                nlm=run_baseline(noisy,'nlm',{'h':fixed_h})
                chosen=adaptive if names[choice]=='adaptive' else (nlm if names[choice]=='nlm' else run_baseline(noisy,names[choice],params[names[choice]]))
                st.session_state['result']={'noisy':noisy,'adaptive':adaptive,'fixed':nlm,'chosen':chosen,
                    'reference':clean if simulate else None,'decision':decision,'choice':choice,
                    'condition':condition if simulate else '真实输入','ap':ap}
        result=st.session_state.get('result')
        if result is None:
            a,b=st.columns(2); a.image(noisy,caption='当前输入预览',clamp=True,width='stretch')
            b.info('点击“运行去噪与对比”，查看输出、决策记录和局部细节。')
        else:
            st.subheader('处理结果')
            st.caption(f'已运行：{result["choice"]} · {result["condition"]}。修改左侧设置后请重新运行。')
            cols=st.columns(3)
            for c,key,title in zip(cols,['noisy','fixed','adaptive'],['带噪 / 输入','固定 NLM','完整自适应方法']):
                c.image(result[key],caption=title,clamp=True,width='stretch')
            d=result['decision']; cols=st.columns(4)
            cols[0].metric('估计脉冲污染率',f'{d["estimated_impulse_rate"]:.1%}')
            cols[1].metric('修复后 σ × 255',f'{d["sigma_repaired"]*255:.2f}')
            cols[2].metric('自适应强度 h',f'{d["h"]:.4f}')
            cols[3].metric('完整推理耗时',f'{d["total_ms"]:.1f} ms')
            if result['reference'] is not None:
                scores=[]
                for key,label in [('noisy','带噪输入'),('fixed','固定 NLM'),('adaptive','完整方法')]:
                    scores.append({'方法':label,**quality(result['reference'],result[key])})
                st.dataframe(pd.DataFrame(scores).rename(columns={'psnr':'PSNR / dB','ssim':'SSIM','gradient_mae':'梯度 MAE'}),hide_index=True,width='stretch')
            else:
                st.info('当前没有干净参考，未计算 PSNR / SSIM。')
            if result['choice'] not in ['完整自适应方法','固定 NLM']:
                st.image(result['chosen'],caption='所选方法：'+result['choice'],width=300)
            with st.expander('局部放大与污染检测',expanded=False):
                h,w=result['noisy'].shape[:2]; x=st.slider('局部中心 X',0,w-1,w//2); y=st.slider('局部中心 Y',0,h-1,h//2)
                r=max(8,min(h,w)//6); cs=st.columns(3)
                for c,key,title in zip(cs,['noisy','fixed','adaptive'],['输入局部','固定 NLM 局部','自适应局部']):
                    c.image(result[key][max(0,y-r):min(h,y+r),max(0,x-r):min(w,x+r)],caption=title,clamp=True,width='stretch')
                c1,c2=st.columns(2)
                c1.image(d['mask'].astype(np.float32),caption='实际修复掩膜',clamp=True,width='stretch')
                residual=np.abs(luminance(result['noisy'])-luminance(result['adaptive']))
                c2.image(np.clip(residual*3,0,1),caption='输入与输出差值 × 3（非真实噪声）',clamp=True,width='stretch')
            with st.expander('决策记录'):
                st.json(safe_decision(d))
            b=BytesIO()
            with zipfile.ZipFile(b,'w',zipfile.ZIP_DEFLATED) as z:
                for key in ['noisy','fixed','adaptive','chosen']:
                    z.writestr(key+'.png',png_bytes(result[key]))
                z.writestr('decision.json',json.dumps(safe_decision(d),ensure_ascii=False,indent=2))
                if result['reference'] is not None:
                    z.writestr('metrics.csv',pd.DataFrame(scores).to_csv(index=False).encode('utf-8-sig'))
            st.download_button('下载本次图像与决策记录',data=b.getvalue(),file_name='denoising_result.zip',mime='application/zip')

with tab2:
    st.subheader('冻结实验结果')
    st.caption('批量计算由独立 CLI 进程执行；此页读取已完成的 CSV，不在页面线程中运行全套实验。')
    progress=read_json(ROOT/'results/progress.json') if (ROOT/'results/progress.json').exists() else {}
    if progress: st.json(progress,expanded=False)
    if (ROOT/'results/raw/summary_by_condition.csv').exists():
        summary=pd.read_csv(ROOT/'results/raw/summary_by_condition.csv')
        method=st.multiselect('显示算法',['gaussian','median','bilateral','wavelet','nlm','A4','A1','A2','A3','A5'],default=['nlm','A4'])
        st.dataframe(summary[summary.method.isin(method)].round(4),hide_index=True,width='stretch')
        for file in ['quality_by_condition.png','delta_heatmap.png','noise_and_detection.png','ablation.png']:
            st.image(str(ROOT/'results/figures'/file),width='stretch')
        st.download_button('下载全部逐项指标 CSV',(ROOT/'results/raw/quality.csv').read_bytes(),'quality.csv','text/csv')
        st.download_button('下载机制分析 CSV',(ROOT/'results/raw/mechanisms.csv').read_bytes(),'mechanisms.csv','text/csv')
    else: st.info('正式实验完成并运行分析脚本后，这里显示真实结果。')
    if st.button('启动独立批量实验',disabled=progress.get('status')=='running'):
        log=open(ROOT/'results/batch_cli.log','w',encoding='utf-8')
        proc=subprocess.Popen([sys.executable,'-m','experiments.benchmark'],cwd=str(ROOT),stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        log.close(); st.success(f'独立实验进程已启动，PID {proc.pid}；完成后运行 analyze 更新图表。')
with tab3:
    st.subheader('可解释的四阶段处理')
    st.markdown('**污染置信度 → 选择性修复 → Haar MAD → 自适应 NLM**\n\n算法只接收带噪图像。干净参考和污染真值用于离线评估，不参与推理。')
    st.code('h = clip(k × sigma_hat, 1/255, 0.25)\nNLM: patch_size=5, patch_distance=6, fast_mode=True, sigma=0.0')
    st.write('H1：修复前处理的结构保护；H2：混合条件下生成参数恢复；H3：自适应强度的跨条件表现。')
    st.json(params)
    st.code('python -m experiments.prepare_data\npython -m experiments.tune  # 仅首次开发选参\npython -m experiments.benchmark\npython -m experiments.analyze\npython -m pytest -q',language='bash')
    st.caption('六张测试图像属于课程级受控实验。真实噪声、细线、强纹理和高污染率可能导致检测或估计失效。')
