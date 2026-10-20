"""Recompute frozen full-frame encoders on RTX 5090; train/validation only."""
import csv
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
import run_action_pilot as v1
import run_action_videomaev2 as v2
from run_action_normalization import transform
from run_protocol_pilot import digest, save_json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'review/generated/action-5090'


def main():
    assert '5090' in torch.cuda.get_device_name()
    assert not OUT.exists(), 'Use a new output directory; never overwrite a benchmark'
    OUT.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.backends.cudnn.allow_tf32=False
    # Reuse the exact original input selection and inference code, with isolated outputs.
    v2.OUT=OUT/'v2';v2.OUT.mkdir();v2.prepare()
    data=json.loads((v2.OUT/'prepared.json').read_text());rows=data['windows']
    assert {w['split'] for w in rows}=={'train','validation'}
    baseline={p:digest(ROOT/p) for p in ['review/generated/action-pilot/prepared.json','review/generated/action-normalization/results.json','review/generated/action-videomaev2/results.json']}
    first=OUT/'v1-clip';first.mkdir()
    save_json(first/'prepared.json',data)
    (first/'inputs').mkdir()
    for item in data['inputs'].values():
        target=first/item['file'];target.parent.mkdir(parents=True,exist_ok=True)
        target.symlink_to(v2.SOURCE/item['file'])
    v1.infer(first)
    v2.infer()
    a=np.load(first/'features.npz');b=np.load(v2.OUT/'features.npz')
    features={name:a[name] for name in ('videomae','clip')};features['videomaev2']=b['videomaev2']
    np.savez_compressed(OUT/'features.npz',**features)
    save_json(OUT/'prepared.json',data)
    runs={'v1_clip':json.loads((first/'inference.json').read_text()),'v2':json.loads((v2.OUT/'inference.json').read_text())}
    expected={m['name']:m['sha256'] for m in json.loads((ROOT/'data/gpu_models.json').read_text())}
    for name in ('videomae','clip'): assert runs['v1_clip']['models'][name]['weights_sha256']==expected[name]
    assert runs['v2']['weights_sha256']==expected['videomaev2']
    save_json(OUT/'inference.json',dict(**runs,test_evaluated=False,script_sha256=digest(__file__)))
    ids=[c['id'] for c in data['config']['classes']]
    masks={s:np.array([w['split']==s for w in rows]) for s in ('train','validation')}
    y={s:np.array([ids.index(w['label']) for w in rows if w['split']==s]) for s in masks}
    assert len(y['train'])==123 and len(y['validation'])==37
    models={};params={}
    for name,x in features.items():
        assert x.shape==(160,768) and np.isfinite(x).all()
        train,val,scaler=transform(x[masks['train']],x[masks['validation']],True)
        grid=[];best=None
        for C in data['config']['C_grid']:
            head=LogisticRegression(C=C,class_weight='balanced',solver='lbfgs',max_iter=3000,random_state=data['config']['seed']).fit(train,y['train'])
            assert max(head.n_iter_)<3000 and np.array_equal(head.classes_,np.arange(6))
            scores=head.predict_proba(val)
            row=dict(C=C,train=v1.metric(y['train'],head.predict(train),6),validation=v1.metric(y['validation'],scores.argmax(1),6))
            grid.append(row)
            if best is None or row['validation']['macro_f1']>best[0]['validation']['macro_f1']+1e-12:best=(row,head,scores)
        row,head,scores=best
        for k,v in dict(mean=scaler.mean_,scale=scaler.scale_,var=scaler.var_,n_samples_seen=scaler.n_samples_seen_,coef=head.coef_,intercept=head.intercept_).items():params[name+'_'+k]=v
        logits=((x[masks['validation']]-scaler.mean_)/scaler.scale_)@head.coef_.T+head.intercept_
        prob=np.exp(logits-logits.max(1,keepdims=True));prob/=prob.sum(1,keepdims=True)
        assert np.allclose(prob,scores,atol=1e-6)
        models[name]=dict(**row,grid=grid,scores=scores.tolist(),predictions=scores.argmax(1).tolist())
        print(name,row,flush=True)
    np.savez(OUT/'pipelines.npz',**params)
    previous=json.loads((ROOT/'review/generated/action-videomaev2/results.json').read_text())
    comparison={name:dict(previous_macro_f1=previous['models'][name]['validation']['macro_f1'],delta_macro_f1=m['validation']['macro_f1']-previous['models'][name]['validation']['macro_f1'],changed_predictions=sum(a!=b for a,b in zip(m['predictions'],previous['models'][name]['predictions']))) for name,m in models.items()}
    result=dict(models=models,classes=data['config']['classes'],validation_windows=[w for w in rows if w['split']=='validation'],train_windows=123,validation_count=37,test_evaluated=False,gpu=torch.cuda.get_device_name(),l4_comparison=comparison,baseline_hashes=baseline,prepared_sha256=digest(OUT/'prepared.json'),features_sha256=digest(OUT/'features.npz'),pipelines_sha256=digest(OUT/'pipelines.npz'))
    save_json(OUT/'results.json',result)
    with (OUT/'scores.csv').open('w',newline='') as f:
        wr=csv.writer(f);wr.writerow(['index','video','start','end','truth','model','prediction']+ids)
        for name,m in models.items():
            for i,w in enumerate(result['validation_windows']):wr.writerow([w['index'],w['video'],w['start'],w['end'],w['label'],name,ids[m['predictions'][i]]]+m['scores'][i])
    v2.OUT=OUT;v2.report()
    html=(OUT/'report.html').read_text().replace('<h1>','<h1>RTX 5090 · ',1)
    details='<h2>Precision / Recall / F1 по каждому действию</h2><table><tr><th>Действие</th><th>Окон</th><th>Модель</th><th>Precision</th><th>Recall</th><th>F1</th></tr>'
    for i,c in enumerate(result['classes']):
        for name,m in models.items():
            metric=m['validation'];details+=f"<tr><td>{c['title']}</td><td>{metric['support'][i]}</td><td>{name}</td>"+''.join(f'<td>{metric[k][i]:.3f}</td>' for k in ('precision','recall','f1'))+'</tr>'
    details+='</table><p>Все три encoder повторно запущены на RTX 5090. Время inference и сравнение с L4 сохранены в JSON. Test не обрабатывался.</p>'
    (OUT/'report.html').write_text(html.replace('</html>',details+'</html>'))
    for p,sha in baseline.items():assert digest(ROOT/p)==sha
    print('PASS: frozen split, train-only scaler, no test windows, reproducible saved heads, baseline unchanged',flush=True)

if __name__=='__main__':main()
