import inspect
import numpy as np
import pytest
from src.noise import generate_noise
from src.impulse_detection import detect_and_repair
from src.noise_estimation import estimate_noise
from src.metrics import detection_scores, quality
from src.adaptive_method import denoise
from src.denoisers import nlm
from src.image_io import load_image, save_image, validate_image

@pytest.mark.parametrize('color',[False,True])
def test_noise_reproducible_and_shared_mask(color):
    x=np.full((32,40,3) if color else (32,40),.45,dtype=np.float32)
    y,t=generate_noise(x,25,.2,42); y2,t2=generate_noise(x,25,.2,42)
    assert np.array_equal(y,y2) and np.array_equal(t['selected_mask'],t2['selected_mask'])
    assert y.dtype==np.float32 and y.min()>=0 and y.max()<=1
    z=y[t['selected_mask']]
    assert np.all((z==0)|(z==1))
    if color: assert np.all(z==z[:,0,None])

def test_selected_label_even_without_change():
    x=np.zeros((32,32),np.float32); y,t=generate_noise(x,0,1,1)
    assert t['selected_mask'].all() and np.any(y==x)

def test_homogeneous_extremes_preserved():
    x=np.zeros((32,32),np.float32); x[:,16:]=1
    z,d=detect_and_repair(x,3,.6)
    assert np.array_equal(z,x) and not d['mask'].any()

def test_isolated_impulse_repaired_and_ordinary_untouched():
    x=np.full((32,32),.5,np.float32); x[16,16]=1
    z,d=detect_and_repair(x,3,.6)
    assert d['mask'][16,16] and z[16,16]==.5
    assert np.array_equal(x[~d['mask']],z[~d['mask']])

def test_sigma_estimation_on_gaussian_field():
    rng=np.random.default_rng(9); x=rng.normal(.5,.04,(256,256)).astype(np.float32)
    s,channels=estimate_noise(x)
    assert abs(s-.04)<.002 and len(channels)==1

def test_rgb_sigma_arithmetic_mean():
    rng=np.random.default_rng(12); x=np.stack([rng.normal(.5,s,(128,128)) for s in [.02,.04,.06]],axis=-1).astype(np.float32)
    s,cs=estimate_noise(x)
    assert len(cs)==3 and s==np.mean(cs)

def test_reference_free_api_and_nlm_semantics():
    assert list(inspect.signature(denoise).parameters)==['image','params']
    x=np.full((16,20,3),.5,np.float32); out=nlm(x,.08)
    assert out.shape==x.shape and out.dtype==np.float32
    assert np.max(abs(out-x))<1e-5

def test_detection_zero_cases():
    zeros=np.zeros((8,8),bool)
    assert detection_scores(zeros,zeros)['f1'] is None
    truth=zeros.copy(); truth[2,2]=True
    d=detection_scores(zeros,truth)
    assert d['precision']==d['recall']==d['f1']==0

def test_chinese_path_and_invalid_file(tmp_path):
    p=tmp_path/'中文图像.png'; x=np.full((20,24,3),.5,np.float32)
    save_image(p,x); assert load_image(p).shape==x.shape
    p=tmp_path/'坏图.png'; p.write_bytes(b'not an image')
    with pytest.raises(ValueError): load_image(p)

def test_invalid_input():
    with pytest.raises(ValueError): validate_image(np.zeros((3,3)))
    with pytest.raises(ValueError): validate_image(np.full((10,10),np.nan))
