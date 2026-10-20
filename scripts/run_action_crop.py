"""Fixed train-selected workspace crop; frozen encoders; train-only scaling; validation only."""
import argparse
import copy
import csv
import fcntl
import json
import os
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw
from run_action_pilot import infer, metric, split_audit
from run_action_normalization import transform, check as scaler_check
from run_protocol_pilot import digest, save_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'review/generated/action-crop'
CONFIG = ROOT / 'data/action_crop.json'


def load():
    cfg = json.loads(CONFIG.read_text())
    source = ROOT / cfg['source']
    data = json.loads((source/'prepared.json').read_text())
    manifest = json.loads((ROOT/'review/generated/manifest.json').read_text())
    assert digest(ROOT/'review/generated/manifest.json') == data['manifest_sha256']
    split_audit(data['config'], manifest)
    return cfg, source, data, manifest


def prepare():
    cfg, source, original, manifest = load()
    assert not (OUT/'prepared.json').exists(), 'Use a new experiment directory for a different crop'
    data = copy.deepcopy(original)
    data['windows'] = [w for w in data['windows'] if w['split'] in ('train','validation')]
    assert len(data['windows']) == 160
    x0,y0,x1,y1 = cfg['crop_xyxy']
    assert 0 <= x0 < x1 <= 464 and 0 <= y0 < y1 <= 848
    data['inputs'], data['sources'] = {}, {}
    data['view'] = cfg['view']
    data['crop'] = cfg
    data['crop_config_sha256'] = digest(CONFIG)
    data['original_prepared_sha256'] = digest(source/'prepared.json')
    data['intervals'] = [r for r in data['intervals'] if r['split'] in ('train','validation')]
    for folder in ('inputs','thumbnails'): (OUT/folder).mkdir(exist_ok=True)
    examples = []
    for video in sorted({w['video'] for w in data['windows']}, key=lambda v:int(v[2:])):
        c = next(c for c in manifest['clips'] if c['id']==video)
        src = ROOT/c['source']
        assert digest(src)==c['sha256']
        local = [w for w in data['windows'] if w['video']==video]
        indexes = sorted({i for w in local for i in w['source_frames']})
        expression = '+'.join(f'eq(n\\,{i})' for i in indexes)
        filters = (f"select='{expression}',crop={x1-x0}:{y1-y0}:{x0}:{y0},"
            'scale=224:224:force_original_aspect_ratio=decrease:flags=bicubic,'
            'pad=224:224:(ow-iw)/2:(oh-ih)/2:color=0x7b7468,format=rgb24')
        raw = subprocess.check_output(['ffmpeg','-v','error','-i',str(src),'-vf',filters,
            '-fps_mode','vfr','-f','rawvideo','pipe:1'])
        frames = np.frombuffer(raw,np.uint8).reshape(-1,224,224,3)
        assert len(frames)==len(indexes)
        mapping = {n:i for i,n in enumerate(indexes)}
        for w in local:
            w['frames'] = [mapping[i] for i in w['source_frames']]
            Image.fromarray(np.concatenate([frames[w['frames'][i]] for i in (0,7,15)],axis=1)).save(OUT/w['thumbnail'],quality=90)
        file = f'inputs/{video}.npz'
        np.savez_compressed(OUT/file,frames=frames)
        data['inputs'][video] = dict(file=file,sha256=digest(OUT/file),frames=len(frames))
        data['sources'][video] = original['sources'][video]
        if video in ('X-2','X-3','X-8','X-11','X-30','X-43'):
            w=local[0]; old=next(r for r in original['windows'] if r['index']==w['index'])
            full=np.load(source/original['inputs'][video]['file'])['frames'][old['frames'][7]]
            examples.append((video,full,frames[w['frames'][7]]))
        print(video,len(local),local[0]['split'],flush=True)
    sheet=Image.new('RGB',(3*448,2*254),'white'); draw=ImageDraw.Draw(sheet)
    for i,(v,full,crop) in enumerate(examples):
        x=(i%3)*448;y=(i//3)*254
        sheet.paste(Image.fromarray(full),(x,y+30));sheet.paste(Image.fromarray(crop),(x+224,y+30))
        draw.text((x+8,y+8),v+' TRAIN | full frame -> fixed crop',fill='black')
    sheet.save(OUT/'train-inputs.jpg')
    save_json(OUT/'prepared.json',data)
    check()


def check():
    scaler_check()
    cfg, source, original, _ = load()
    data = json.loads((OUT/'prepared.json').read_text())
    assert data['original_prepared_sha256']==digest(source/'prepared.json')
    assert data['crop_config_sha256']==digest(CONFIG)
    assert data['assignments']==original['assignments']
    expected=[w for w in original['windows'] if w['split'] in ('train','validation')]
    assert len(expected)==len(data['windows'])==160
    for a,b in zip(expected,data['windows']):
        for key in ('index','video','row','split','label','start','end','source_frames','source_pts'):
            assert a[key]==b[key],key
        assert len(b['frames'])==16
    assert all(v['split']!='test' for v in data['sources'].values())
    for item in data['inputs'].values(): assert digest(OUT/item['file'])==item['sha256']
    print('OK: identical train/validation windows, labels, source frames, PTS and split; test excluded')


def fit():
    import sklearn
    from sklearn.linear_model import LogisticRegression
    check()
    cfg, source, original, _ = load()
    data=json.loads((OUT/'prepared.json').read_text()); run=json.loads((OUT/'inference.json').read_text())
    baseline=json.loads((ROOT/cfg['baseline']/'results.json').read_text())
    for file,sha in baseline['source_hashes'].items(): assert digest(source/file)==sha
    assert run['prepared_sha256']==digest(OUT/'prepared.json')
    assert run['features_sha256']==digest(OUT/'features.npz')
    assert baseline['source_hashes']['features.npz']==digest(source/'features.npz')
    assert baseline['source_hashes']['prepared.json']==digest(source/'prepared.json')
    previous=json.loads((source/'inference.json').read_text())
    for key in ('torch','transformers'): assert run[key]==previous[key]
    for name in ('videomae','clip'):
        assert run['models'][name]['weights_sha256']==previous['models'][name]['weights_sha256']
        for key in ('image_mean','image_std'): assert run['models'][name][key]==previous['models'][name][key]
    windows=data['windows']; ids=[c['id'] for c in data['config']['classes']]
    mask={s:np.array([w['split']==s for w in windows]) for s in ('train','validation')}
    y={s:np.array([ids.index(w['label']) for w in windows if w['split']==s]) for s in mask}
    results={}; parameters={}
    old_indexes=np.array([w['index'] for w in windows])
    with np.load(source/'features.npz') as full, np.load(OUT/'features.npz') as crop:
        for name in ('videomae','clip'):
            variants={}
            for view,features in (('full',full[name][old_indexes]),('crop',crop[name])):
                assert features.shape==(160,768) and np.isfinite(features).all()
                train,val,scaler=transform(features[mask['train']],features[mask['validation']],True)
                assert scaler.n_samples_seen_==123 and len(val)==37
                grid=[];best=None
                for C in data['config']['C_grid']:
                    head=LogisticRegression(C=C,class_weight='balanced',solver='lbfgs',max_iter=3000,random_state=data['config']['seed']).fit(train,y['train'])
                    assert max(head.n_iter_)<head.max_iter
                    assert np.array_equal(head.classes_,np.arange(6))
                    scores=head.predict_proba(val)
                    row=dict(C=C,train=metric(y['train'],head.predict(train),6),validation=metric(y['validation'],scores.argmax(1),6))
                    grid.append(row)
                    if best is None or row['validation']['macro_f1']>best[0]['validation']['macro_f1']+1e-12: best=(row,head,scores)
                row,head,scores=best
                if view=='full':
                    old=baseline['models'][name]['variants']['after']
                    assert row['C']==old['C'] and np.allclose(scores,old['scores'],atol=1e-10)
                    assert all(abs(a['validation']['macro_f1']-b['validation']['macro_f1'])<1e-12 for a,b in zip(grid,old['grid']))
                prefix=name+'_'+view
                for k,v in dict(mean=scaler.mean_,scale=scaler.scale_,var=scaler.var_,coef=head.coef_,intercept=head.intercept_,n_samples_seen=scaler.n_samples_seen_).items(): parameters[prefix+'_'+k]=v
                logits=((features[mask['validation']]-scaler.mean_)/scaler.scale_)@head.coef_.T+head.intercept_
                prob=np.exp(logits-logits.max(1,keepdims=True));prob/=prob.sum(1,keepdims=True)
                assert np.allclose(prob,scores,atol=1e-6)
                variants[view]=dict(**row,grid=grid,scores=scores.tolist(),predictions=scores.argmax(1).tolist())
            oldC=variants['full']['C']
            variants['crop_at_full_C']=next(r for r in variants['crop']['grid'] if r['C']==oldC)
            results[name]=variants
            print(name,[(v,variants[v]['C'],variants[v]['validation']['macro_f1']) for v in ('full','crop')],flush=True)
    np.savez(OUT/'pipelines.npz',**parameters)
    result=dict(models=results,classes=data['config']['classes'],validation_windows=[w for w in windows if w['split']=='validation'],
        test_evaluated=False,train_windows=123,validation_count=37,crop=cfg,sklearn=sklearn.__version__,numpy=np.__version__,
        prepared_sha256=digest(OUT/'prepared.json'),features_sha256=digest(OUT/'features.npz'),pipelines_sha256=digest(OUT/'pipelines.npz'),
        baseline_results_sha256=digest(ROOT/cfg['baseline']/'results.json'),script_sha256=digest(__file__))
    for file,sha in baseline['source_hashes'].items(): assert digest(source/file)==sha
    save_json(OUT/'results.json',result)
    with (OUT/'scores.csv').open('w',newline='') as f:
        writer=csv.writer(f,lineterminator='\n');writer.writerow(['index','video','start','end','truth','model','view','prediction']+ids)
        for name,variants in results.items():
            for view in ('full','crop'):
                v=variants[view]
                for i,w in enumerate(result['validation_windows']): writer.writerow([w['index'],w['video'],w['start'],w['end'],w['label'],name,view,ids[v['predictions'][i]]]+v['scores'][i])
    report()


def report():
    os.environ.setdefault('MPLCONFIGDIR',str(OUT/'matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    r=json.loads((OUT/'results.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for ax,(name,vs) in zip(axes,r['models'].items()):
        x=np.arange(6)
        for offset,view,color in ((-.18,'full','#8794a4'),(.18,'crop','#267b86')):
            ax.bar(x+offset,vs[view]['validation']['f1'],.35,label=f"{view}: macro F1 {vs[view]['validation']['macro_f1']:.3f}",color=color)
        ax.set(xticks=x,xticklabels=[c['short'] for c in r['classes']],ylim=(0,1.1),title=name.upper(),ylabel='Validation class F1')
        ax.legend(fontsize=9);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Same 123 train / 37 validation windows | train-only standardization | test not evaluated')
    fig.savefig(OUT/'validation.png',dpi=160);plt.close(fig)
    rows=''
    for name,vs in r['models'].items():
        for view in ('full','crop'):
            v=vs[view];rows+=f"<tr><td>{name}</td><td>{view}</td><td>{v['C']}</td><td>{v['train']['macro_f1']:.3f}</td><td>{v['validation']['macro_f1']:.3f}</td><td>{v['validation']['accuracy']:.3f}</td></tr>"
    fixed='; '.join(f"{name}: C={vs['full']['C']}, full F1={vs['full']['validation']['macro_f1']:.3f}, crop F1={vs['crop_at_full_C']['validation']['macro_f1']:.3f}" for name,vs in r['models'].items())
    body=f'''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Кроп рабочего стола</title><style>body{{font:16px system-ui;margin:30px auto;padding:0 16px;max-width:1100px;color:#172b35}}img{{max-width:100%;height:auto}}table{{border-collapse:collapse}}td,th{{padding:10px;border-bottom:1px solid #ddd}}.scroll{{overflow:auto}}</style><a href="../../actions.html">← Все эксперименты</a><h1>Полный кадр и кроп рабочего стола</h1><p>Один кроп [0,240,440,640] выбран по train до инференса и одинаков для всех видео. Те же 16 кадров, PTS, окна и split. StandardScaler обучен отдельно для каждого encoder/вида только на 123 train-окнах. C выбран по прежней сетке на 37 validation-окнах. Test не использован.</p><div class="scroll"><table><tr><th>Encoder</th><th>Вид</th><th>C</th><th>Train F1</th><th>Validation F1</th><th>Accuracy</th></tr>{rows}</table></div><p>При фиксированном C полного кадра: {fixed}.</p><h2>Что изменилось по действиям</h2><img src="validation.png" alt="F1 по классам для полного кадра и кропа"><h2>Входы из train</h2><p>Слева полный кадр, справа кроп. Увеличение деталей одновременно удаляет часть контекста.</p><img src="train-inputs.jpg" alt="Примеры полного кадра и кропа только из train"><p>Это исследовательская validation-проверка на знакомой камере: окна зависимы, редкие классы представлены несколькими примерами. Увеличение F1 не доказывает устойчивость к новым съёмкам.</p><p><a href="results.json">Все C, метрики и вероятности</a> · <a href="scores.csv">Предсказания CSV</a> · <a href="inference.json">Инференс и веса</a></p></html>'''
    (OUT/'report.html').write_text(body)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['prepare','infer','fit','check','report']);args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.mode=='infer': infer(OUT)
        else: globals()[args.mode]()
