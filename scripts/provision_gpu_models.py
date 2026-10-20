"""Download exact official checkpoints used by our baselines; verify all weight hashes."""
import hashlib
import json
from pathlib import Path
import urllib.request
from huggingface_hub import hf_hub_download,snapshot_download
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'review/generated/gpu-setup-5090'

def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    OUT.mkdir(parents=True,exist_ok=True);models=json.loads((ROOT/'data/gpu_models.json').read_text());results=[]
    hub=Path.home()/'.cache/huggingface/hub';torch=Path.home()/'.cache/torch/hub/checkpoints';torch.mkdir(parents=True,exist_ok=True)
    for m in models:
        print('Preparing',m['name'],flush=True)
        if 'url' in m:
            path=torch/m['filename']
            if not path.exists():
                partial=path.with_suffix('.partial');urllib.request.urlretrieve(m['url'],partial)
                assert sha(partial)==m['sha256'];partial.replace(path)
        elif m['snapshot']:
            folder=snapshot_download(m['repo'],revision=m['revision'],cache_dir=hub,allow_patterns=['*.json','*.txt','*.safetensors'])
            path=Path(folder)/m['filename']
        else:path=Path(hf_hub_download(m['repo'],m['filename'],revision=m['revision'],cache_dir=hub))
        actual=sha(path);assert actual==m['sha256'],f"Weight mismatch: {m['name']}"
        results.append(dict(**m,path=str(path),bytes=path.stat().st_size,verified=True));print('Verified',m['name'],actual,flush=True)
    (OUT/'models.json').write_text(json.dumps(results,indent=2)+'\n')
if __name__=='__main__':main()
