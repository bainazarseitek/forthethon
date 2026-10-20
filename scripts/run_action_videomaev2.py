"""Frozen official VideoMAEv2 base, original full-frame inputs, train-only scaling."""
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from run_protocol_pilot import digest, normalize, save_json
from run_action_pilot import metric, split_audit
from run_action_normalization import transform, check as scaler_check

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'review/generated/action-pilot'
BASELINE=ROOT/'review/generated/action-normalization'
OUT=ROOT/'review/generated/action-videomaev2'
CONFIG=ROOT/'data/action_videomaev2.json'
VENDOR=ROOT/'scripts/vendor/videomaev2/modeling_finetune.py'


def prepare():
    assert not (OUT/'prepared.json').exists(), 'Do not overwrite a prepared experiment'
    cfg=json.loads(CONFIG.read_text()); data=json.loads((SOURCE/'prepared.json').read_text())
    manifest=ROOT/'review/generated/manifest.json'
    assert digest(manifest)==data['manifest_sha256']
    split_audit(data['config'],json.loads(manifest.read_text()))
    data['windows']=[w for w in data['windows'] if w['split'] in ('train','validation')]
    data['intervals']=[r for r in data['intervals'] if r['split'] in ('train','validation')]
    videos={w['video'] for w in data['windows']}
    data['inputs']={k:v for k,v in data['inputs'].items() if k in videos}
    data['sources']={k:v for k,v in data['sources'].items() if k in videos}
    data['v2']=cfg;data['v2_config_sha256']=digest(CONFIG);data['original_prepared_sha256']=digest(SOURCE/'prepared.json')
    save_json(OUT/'prepared.json',data)
    check()


def check():
    scaler_check()
    data=json.loads((OUT/'prepared.json').read_text()); original=json.loads((SOURCE/'prepared.json').read_text())
    assert digest(CONFIG)==data['v2_config_sha256']
    assert digest(VENDOR)==data['v2']['implementation_sha256']
    assert digest(SOURCE/'prepared.json')==data['original_prepared_sha256']
    assert data['windows']==[w for w in original['windows'] if w['split'] in ('train','validation')]
    assert data['assignments']==original['assignments']
    assert len(data['windows'])==160 and sum(w['split']=='train' for w in data['windows'])==123
    assert all(s['split']!='test' for s in data['sources'].values())
    for item in data['inputs'].values(): assert digest(SOURCE/item['file'])==item['sha256']
    print('OK: unchanged full-frame input bytes, windows, labels, PTS, split; test excluded',flush=True)


def infer():
    import torch
    import timm
    from huggingface_hub import hf_hub_download
    import importlib.util
    check()
    data=json.loads((OUT/'prepared.json').read_text());cfg=data['v2']
    assert torch.cuda.is_available() and timm.__version__==cfg['timm']
    torch.manual_seed(data['config']['seed']);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    weights=Path(hf_hub_download(cfg['repo'],cfg['filename'],revision=cfg['revision'],local_files_only=True))
    module_spec=importlib.util.spec_from_file_location('fortethon_official_videomaev2',VENDOR)
    module=importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name]=module
    module_spec.loader.exec_module(module)
    started=time.monotonic()
    checkpoint=torch.load(weights,map_location='cpu',weights_only=True)
    state=checkpoint.get('module',checkpoint.get('model',checkpoint))
    assert isinstance(state,dict) and all(isinstance(v,torch.Tensor) for v in state.values())
    assert state['head.weight'].shape==(710,768)
    model=module.vit_base_patch16_224(num_classes=710,all_frames=16,tubelet_size=2,use_mean_pooling=True)
    model.load_state_dict(state,strict=True)
    del checkpoint,state
    model=model.eval().requires_grad_(False).to('cuda')
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    mean=torch.tensor(cfg['image_mean'],device='cuda')[None,:,None,None,None]
    std=torch.tensor(cfg['image_std'],device='cuda')[None,:,None,None,None]
    load_seconds=time.monotonic()-started
    vectors=[];timings=[];current=None;first_check=False
    for w in data['windows']:
        if w['video']!=current:
            item=data['inputs'][w['video']];frames=np.load(SOURCE/item['file'])['frames'];current=w['video']
        pixels=torch.from_numpy(frames[w['frames']].copy()).permute(3,0,1,2)[None].to('cuda',torch.float32)/255
        assert pixels.shape==(1,3,16,224,224)
        pixels=(pixels-mean)/std
        torch.cuda.synchronize();tick=time.monotonic()
        with torch.inference_mode():
            vector=model.forward_features(pixels)
        torch.cuda.synchronize();timings.append(time.monotonic()-tick)
        assert vector.shape==(1,768) and torch.isfinite(vector).all()
        if not first_check:
            with torch.inference_mode():
                assert torch.allclose(model(pixels),model.head(vector),atol=1e-5,rtol=1e-5)
                assert torch.allclose(vector,model.forward_features(pixels),atol=1e-6,rtol=1e-6)
            first_check=True
        vectors.append(vector.cpu().numpy()[0])
        if len(vectors)%40==0: print(f'videomaev2: {len(vectors)}/160',flush=True)
    x=normalize(np.array(vectors));assert x.shape==(160,768) and np.isfinite(x).all()
    np.savez_compressed(OUT/'features.npz',videomaev2=x)
    save_json(OUT/'inference.json',dict(model=cfg,weights_sha256=digest(weights),implementation_sha256=digest(VENDOR),
        prepared_sha256=digest(OUT/'prepared.json'),features_sha256=digest(OUT/'features.npz'),script_sha256=digest(__file__),
        gpu=torch.cuda.get_device_name(),torch=torch.__version__,timm=timm.__version__,dtype='float32',
        parameters=sum(p.numel() for p in model.parameters()),trainable_encoder_parameters=0,
        load_seconds=load_seconds,forward_seconds=sum(timings),total_seconds=time.monotonic()-started,
        strict_weights=True,feature_head_equivalence_checked=first_check,
        pooling='Official forward_features: mean of final patch tokens then learned fc_norm; classifier not used; L2 vector',
        view='Unchanged cached full-frame 224 padding, identical 16 frames/2 s/PTS as v1 and CLIP; no test inputs',test_evaluated=False))


def fit():
    import sklearn
    from sklearn.linear_model import LogisticRegression
    check()
    data=json.loads((OUT/'prepared.json').read_text());run=json.loads((OUT/'inference.json').read_text())
    base=json.loads((BASELINE/'results.json').read_text())
    assert run['prepared_sha256']==digest(OUT/'prepared.json') and run['features_sha256']==digest(OUT/'features.npz')
    for file,sha in base['source_hashes'].items(): assert digest(SOURCE/file)==sha
    rows=data['windows'];ids=[c['id'] for c in data['config']['classes']]
    masks={s:np.array([w['split']==s for w in rows]) for s in ('train','validation')}
    y={s:np.array([ids.index(w['label']) for w in rows if w['split']==s]) for s in masks}
    indexes=np.array([w['index'] for w in rows]);full=np.load(SOURCE/'features.npz');v2=np.load(OUT/'features.npz')['videomaev2']
    results={};params={}
    for name,x in [('videomae',full['videomae'][indexes]),('clip',full['clip'][indexes]),('videomaev2',v2)]:
        assert x.shape==(160,768) and np.isfinite(x).all()
        train,val,scaler=transform(x[masks['train']],x[masks['validation']],True)
        assert scaler.n_samples_seen_==123
        grid=[];best=None
        for C in data['config']['C_grid']:
            head=LogisticRegression(C=C,class_weight='balanced',solver='lbfgs',max_iter=3000,random_state=data['config']['seed']).fit(train,y['train'])
            assert max(head.n_iter_)<head.max_iter and np.array_equal(head.classes_,np.arange(6))
            scores=head.predict_proba(val)
            row=dict(C=C,train=metric(y['train'],head.predict(train),6),validation=metric(y['validation'],scores.argmax(1),6))
            grid.append(row)
            if best is None or row['validation']['macro_f1']>best[0]['validation']['macro_f1']+1e-12: best=(row,head,scores)
        row,head,scores=best
        if name in base['models']:
            old=base['models'][name]['variants']['after']
            assert row['C']==old['C'] and np.allclose(scores,old['scores'],atol=1e-10)
            assert all(abs(a['validation']['macro_f1']-b['validation']['macro_f1'])<1e-12 for a,b in zip(grid,old['grid']))
        for k,v in dict(mean=scaler.mean_,scale=scaler.scale_,var=scaler.var_,n_samples_seen=scaler.n_samples_seen_,coef=head.coef_,intercept=head.intercept_).items():params[name+'_'+k]=v
        logits=((x[masks['validation']]-scaler.mean_)/scaler.scale_)@head.coef_.T+head.intercept_
        prob=np.exp(logits-logits.max(1,keepdims=True));prob/=prob.sum(1,keepdims=True)
        assert np.allclose(prob,scores,atol=1e-6)
        results[name]=dict(**row,grid=grid,scores=scores.tolist(),predictions=scores.argmax(1).tolist())
        print(name,'C',row['C'],'validation F1',row['validation']['macro_f1'],'accuracy',row['validation']['accuracy'],flush=True)
    np.savez(OUT/'pipelines.npz',**params)
    result=dict(models=results,classes=data['config']['classes'],validation_windows=[w for w in rows if w['split']=='validation'],
        train_windows=123,validation_count=37,test_evaluated=False,checkpoint=data['v2'],numpy=np.__version__,sklearn=sklearn.__version__,
        prepared_sha256=digest(OUT/'prepared.json'),features_sha256=digest(OUT/'features.npz'),pipelines_sha256=digest(OUT/'pipelines.npz'),
        baseline_sha256=digest(BASELINE/'results.json'),script_sha256=digest(__file__))
    for file,sha in base['source_hashes'].items(): assert digest(SOURCE/file)==sha
    save_json(OUT/'results.json',result)
    with (OUT/'scores.csv').open('w',newline='') as f:
        wr=csv.writer(f,lineterminator='\n');wr.writerow(['index','video','start','end','truth','model','prediction']+ids)
        for name,v in results.items():
            for i,w in enumerate(result['validation_windows']):wr.writerow([w['index'],w['video'],w['start'],w['end'],w['label'],name,ids[v['predictions'][i]]]+v['scores'][i])
    report()


def report():
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'review/generated/action-crop/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    r=json.loads((OUT/'results.json').read_text());fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    names={'videomae':'VideoMAE v1','clip':'CLIP','videomaev2':'VideoMAEv2 base K710'}
    for i,(name,v) in enumerate(r['models'].items()):
        axes[0].bar(np.arange(6)+(i-1)*.25,v['validation']['f1'],.25,label=f"{names[name]}: {v['validation']['macro_f1']:.3f}")
        axes[1].plot([g['C'] for g in v['grid']],[g['validation']['macro_f1'] for g in v['grid']],marker='o',label=names[name])
    axes[0].set(xticks=np.arange(6),xticklabels=[c['short'] for c in r['classes']],ylim=(0,1.15),ylabel='Validation class F1');axes[0].legend(fontsize=8)
    axes[1].set(xscale='log',ylim=(0,1),xlabel='C',ylabel='Validation macro F1');axes[1].legend(fontsize=8)
    fig.suptitle('Same full-frame inputs | 123 train / 37 validation | train-only scaling | no test')
    fig.savefig(OUT/'validation.png',dpi=160);plt.close(fig)
    rows=''.join(f"<tr><td>{names[k]}</td><td>{v['C']}</td><td>{v['train']['macro_f1']:.3f}</td><td>{v['validation']['macro_f1']:.3f}</td><td>{round(v['validation']['accuracy']*37)}/37</td></tr>" for k,v in r['models'].items())
    (OUT/'report.html').write_text(f'''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>VideoMAEv2: validation</title><style>body{{font:16px system-ui;color:#172b35;max-width:1100px;margin:30px auto;padding:0 16px}}img{{max-width:100%}}table{{border-collapse:collapse}}td,th{{padding:10px;border-bottom:1px solid #ddd}}.scroll{{overflow:auto}}</style><a href="../../actions.html">← Все эксперименты</a><h1>VideoMAEv2, VideoMAE v1 и CLIP</h1><p>Полный кадр, те же 160 окон и исходные кадры. StandardScaler обучен отдельно для каждого encoder только на 123 train-окнах. Прежняя сетка C; выбор по 37 validation-окнам. Test не использован.</p><div class="scroll"><table><tr><th>Encoder</th><th>C</th><th>Train F1</th><th>Validation F1</th><th>Верно</th></tr>{rows}</table></div><p><strong>Сравниваются конкретные checkpoints:</strong> v2 — официальный distilled ViT-base K710; v1 — self-supervised K400. Различаются данные/способ предобучения и pooling. Этот опыт не выделяет эффект архитектуры v2.</p><img src="validation.png" alt="F1 по классам и сетка C"><p>Классы: ручной пересчёт, машинка, ноутбук, документ, сейф, уход. Данных мало, окна зависимы; validation уже использовался в предыдущих опытах.</p><p><a href="results.json">Все метрики</a> · <a href="scores.csv">Предсказания CSV</a> · <a href="inference.json">Checkpoint, хеши и время</a></p></html>''')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['prepare','infer','check','fit','report']);args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        globals()[args.mode]()
