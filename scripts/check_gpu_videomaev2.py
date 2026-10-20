"""Small offline VideoMAEv2 CUDA smoke test on an unchanged real train input."""
import importlib.util
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from run_protocol_pilot import digest
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'review/generated/gpu-setup-5090'

def main():
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);torch.manual_seed(20261017)
    assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    a=torch.randn(256,256,device='cuda');b=a@a.T;torch.cuda.synchronize();assert torch.isfinite(b).all()
    cfg=json.loads((ROOT/'data/action_videomaev2.json').read_text());vendor=ROOT/'scripts/vendor/videomaev2/modeling_finetune.py'
    assert digest(vendor)==cfg['implementation_sha256']
    spec=importlib.util.spec_from_file_location('fortethon_official_videomaev2',vendor);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    path=Path(hf_hub_download(cfg['repo'],cfg['filename'],revision=cfg['revision'],local_files_only=True))
    ref=json.loads((ROOT/'review/generated/action-videomaev2/inference.json').read_text());assert digest(path)==ref['weights_sha256']
    state=torch.load(path,map_location='cpu',weights_only=True);state=state.get('module',state.get('model',state))
    model=module.vit_base_patch16_224(num_classes=710,all_frames=16,tubelet_size=2,use_mean_pooling=True)
    model.load_state_dict(state,strict=True);del state
    model=model.eval().requires_grad_(False).cuda()
    data=json.loads((ROOT/'review/generated/action-videomaev2/prepared.json').read_text());i=next(i for i,w in enumerate(data['windows']) if w['index']==3);w=data['windows'][i];assert w['split']=='train'
    item=data['inputs'][w['video']];inputpath=ROOT/'review/generated/action-pilot'/item['file'];assert digest(inputpath)==item['sha256']
    frames=np.load(inputpath)['frames'][w['frames']]
    pixels=torch.from_numpy(frames.copy()).permute(3,0,1,2)[None].to('cuda',torch.float32)/255
    mean=torch.tensor(cfg['image_mean'],device='cuda')[None,:,None,None,None];std=torch.tensor(cfg['image_std'],device='cuda')[None,:,None,None,None];pixels=(pixels-mean)/std
    timings=[];vectors=[]
    with torch.inference_mode():
        for _ in range(3):
            torch.cuda.synchronize();start=time.monotonic();v=model.forward_features(pixels);torch.cuda.synchronize();timings.append(time.monotonic()-start)
            assert v.shape==(1,768) and torch.isfinite(v).all();vectors.append(torch.nn.functional.normalize(v,dim=1).cpu().numpy()[0])
    assert all(np.allclose(vectors[0],v,atol=1e-6) for v in vectors)
    expected=np.load(ROOT/'review/generated/action-videomaev2/features.npz')['videomaev2'][i]
    cosine=float(np.dot(expected,vectors[-1]));max_error=float(np.max(np.abs(expected-vectors[-1])))
    assert cosine>.9999 and max_error<1e-3, 'Unexpected difference from L4 reference feature'
    versions={name:importlib.metadata.version(name) for name in ['torch','torchvision','transformers','timm','numpy','scikit-learn','huggingface-hub','Pillow','matplotlib','openpyxl','opencv-python-headless']}
    result=dict(gpu=torch.cuda.get_device_name(),compute_capability=torch.cuda.get_device_capability(),cuda=torch.version.cuda,
        torch_architectures=torch.cuda.get_arch_list(),python=platform.python_version(),versions=versions,
        train_window=dict(index=w['index'],video=w['video'],start=w['start'],end=w['end']),weights_sha256=digest(path),input_sha256=digest(inputpath),
        shape=list(v.shape),dtype=str(v.dtype),device=str(v.device),finite=True,strict_weights=True,frozen=True,
        repeat_seconds=timings,l4_reference_cosine=cosine,l4_max_abs_error=max_error,test_evaluated=False)
    (OUT/'smoke-test.json').write_text(json.dumps(result,indent=2)+'\n')
    (OUT/'pip-freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
