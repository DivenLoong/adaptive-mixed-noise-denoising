"""Checks against actual frozen experiment artifacts, not mock results."""
import numpy as np
import pandas as pd
import pytest
from src.common import ROOT, array_hash, read_json
from src.metrics import quality
from experiments.benchmark import verify_freeze

@pytest.fixture(scope='module')
def artifacts():
    if not (ROOT/'results/raw/quality.csv').exists(): pytest.skip('Formal experiment not run yet')
    return pd.read_csv(ROOT/'results/raw/quality.csv'),pd.read_csv(ROOT/'results/raw/mechanisms.csv')

def test_frozen_algorithm_unchanged(artifacts):
    assert verify_freeze()['test_not_used']

def test_complete_design_and_no_development_images(artifacts):
    q,m=artifacts
    assert len(m)==126 and len(q)==1764
    assert not set(m.image)&{'camera','coffee'}
    assert not m.duplicated(['image','condition','seed']).any()
    assert set(m.seed)=={42,2026,1108}

def test_A4_A5_same_preprocessing(artifacts):
    _,m=artifacts
    assert (m.pre_hash_A4==m.pre_hash_A5).all()

def test_every_saved_float_output_matches_metrics(artifacts):
    q,m=artifacts
    if not (ROOT/'results/outputs').exists() or not (ROOT/'results/cache').exists():
        pytest.skip('紧凑提交包省略精确缓存；运行 benchmark 后可执行完整浮点审计')
    for row in m.itertuples():
        stem=f'{row.image}__{row.condition}__{row.seed}'
        clean=np.load(ROOT/'data'/f'{row.image}.npy')
        with np.load(ROOT/'results/cache'/f'{stem}.npz') as cached:
            assert array_hash(cached['noisy'])==row.input_sha256
        with np.load(ROOT/'results/outputs'/f'{stem}.npz') as outputs:
            part=q[(q.image==row.image)&(q.condition==row.condition)&(q.seed==row.seed)]
            for r in part.itertuples():
                score=quality(clean,outputs[r.method])
                assert abs(score['psnr']-r.psnr)<1e-8
                assert abs(score['ssim']-r.ssim)<1e-8
                assert abs(score['gradient_mae']-r.gradient_mae)<1e-8
            assert np.array_equal(outputs['A0'],outputs['nlm'])

def test_statistics_unit_is_image(artifacts):
    q,m=artifacts
    summary=read_json(ROOT/'results/raw/analysis_summary.json')
    avg=q.groupby(['image','condition','method']).psnr.mean().unstack('method')
    assert abs((avg.A4-avg.nlm).mean()-summary['A4_minus_fixed'])<1e-10
