"""Read-only error analysis: train neighbours, class coverage, binary diagnostic and review clips."""
from collections import Counter
import json
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image,ImageDraw
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from run_protocol_pilot import digest,save_json
from run_action_pilot import metric

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'review/generated/action-videomaev2'
OUT=ROOT/'review/generated/action-diagnosis'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    source_hashes={f:digest(SRC/f) for f in ('prepared.json','features.npz','pipelines.npz','results.json')}
    p=json.loads((SRC/'prepared.json').read_text());r=json.loads((SRC/'results.json').read_text())
    m=json.loads((ROOT/'review/generated/manifest.json').read_text());notes=json.loads((ROOT/'data/action_diagnosis.json').read_text())
    assert source_hashes['prepared.json']==r['prepared_sha256'] and source_hashes['features.npz']==r['features_sha256']
    assert source_hashes['pipelines.npz']==r['pipelines_sha256']
    ws=p['windows'];x=np.load(SRC/'features.npz')['videomaev2'];params=np.load(SRC/'pipelines.npz')
    tr=np.array([i for i,w in enumerate(ws) if w['split']=='train']);hand=[i for i in tr if ws[i]['label']=='hand_count']
    queries=[i for i,w in enumerate(ws) if w['split']=='validation' and w['label']=='hand_count']
    assert len(hand)==3 and len(queries)==5 and x.shape==(160,768)
    z=(x-params['videomaev2_mean'])/params['videomaev2_scale'];zn=z/np.linalg.norm(z,axis=1,keepdims=True)
    assert np.allclose(params['videomaev2_mean'],x[tr].astype(float).mean(0),atol=1e-12)
    ids=[c['id'] for c in r['classes']];results=[];needed=set(hand+queries)
    for i in queries:
        w=ws[i];vi=next(j for j,v in enumerate(r['validation_windows']) if v['index']==w['index'])
        v=r['models']['videomaev2'];row=dict(index=w['index'],prediction=ids[v['predictions'][vi]],scores=v['scores'][vi],spaces={})
        for space,f in [('raw',x),('standardized',zn)]:
            sim=f[tr]@f[i];order=np.argsort(-sim,kind='stable')
            candidates=[dict(index=ws[tr[k]]['index'],cosine=float(sim[k]),rank=j+1) for j,k in enumerate(order)]
            groups={label:[c for c in candidates if ws[next(t for t in tr if ws[t]['index']==c['index'])]['label']==label] for label in ('hand_count','machine')}
            row['spaces'][space]=dict(top=candidates[:5],best_hand=groups['hand_count'][0],best_machine=groups['machine'][0],all_hand=groups['hand_count'])
            for c in candidates[:5]+[groups['hand_count'][0],groups['machine'][0]]:needed.add(next(t for t in tr if ws[t]['index']==c['index']))
        results.append(row)
    binary_train=np.array([i for i in tr if ws[i]['label'] in ('hand_count','machine')]);binary_val=np.array([i for i,w in enumerate(ws) if w['split']=='validation' and w['label'] in ('hand_count','machine')])
    y=np.array([int(w['label']=='hand_count') for w in ws]);scaler=StandardScaler().fit(x[binary_train].astype(float))
    head=LogisticRegression(C=notes['binary_probe_C'],class_weight='balanced',max_iter=3000,random_state=p['config']['seed']).fit(scaler.transform(x[binary_train]),y[binary_train])
    assert scaler.n_samples_seen_==49 and max(head.n_iter_)<3000
    probabilities=head.predict_proba(scaler.transform(x[binary_val]));pred=probabilities.argmax(1)
    binary=dict(C=head.C,train_count=49,train_class_counts=dict(Counter(ws[i]['label'] for i in binary_train)),
        validation_count=len(binary_val),validation=metric(y[binary_val],pred,2),class_order=['machine','hand_count'],
        predictions=[dict(index=ws[i]['index'],truth=ws[i]['label'],prediction='hand_count' if pr else 'machine',hand_probability=float(probabilities[j,1])) for j,(i,pr) in enumerate(zip(binary_val,pred))])
    np.savez(OUT/'binary-pipeline.npz',mean=scaler.mean_,scale=scaler.scale_,coef=head.coef_,intercept=head.intercept_,train_indices=binary_train)
    # Export source-resolution inspection clips; these crops never enter an encoder.
    for i in sorted(needed):
        w=ws[i];c=next(c for c in m['clips'] if c['id']==w['video']);src=ROOT/c['source']
        assert digest(src)==c['sha256']
        idx=w['index'];video=OUT/f'{idx:04d}.mp4';image=OUT/f'{idx:04d}.jpg'
        if not video.exists():
            subprocess.run(['ffmpeg','-v','error','-i',str(src),'-ss',str(w['start']),'-t','2','-vf','crop=440:400:0:240','-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],check=True)
        if not image.exists():
            positions=[0,3,6,9,12,15];selected=[w['source_frames'][j] for j in positions];expr='+'.join(f'eq(n\\,{j})' for j in selected)
            raw=subprocess.check_output(['ffmpeg','-v','error','-i',str(src),'-vf',f"select='{expr}',crop=440:400:0:240,scale=330:300,format=rgb24",'-fps_mode','vfr','-f','rawvideo','pipe:1'])
            frames=np.frombuffer(raw,np.uint8).reshape(-1,300,330,3);assert len(frames)==6
            sheet=Image.new('RGB',(990,640),'white');draw=ImageDraw.Draw(sheet)
            for j,f in enumerate(frames):
                xx=(j%3)*330;yy=(j//3)*320;sheet.paste(Image.fromarray(f),(xx,yy+20));draw.text((xx+5,yy+4),f"{w['video']} {w['source_pts'][positions[j]]:.2f}s",fill='black')
            sheet.save(image)
    coverage={label:dict(windows=sum(ws[i]['label']==label for i in tr),videos=len({ws[i]['video'] for i in tr if ws[i]['label']==label}),intervals=len({(ws[i]['video'],ws[i]['row']) for i in tr if ws[i]['label']==label})) for label in ids}
    # Scenario components from the frozen pair graph, never from class predictions.
    parent={v:v for v in p['assignments']}
    def find(v):
        while parent[v]!=v:v=parent[v]
        return v
    for a,b,*_ in p['pairs']:parent[find(a)]=find(b)
    for label in ids:coverage[label]['scenario_groups']=len({find(ws[i]['video']) for i in tr if ws[i]['label']==label})
    result=dict(classes=r['classes'],queries=results,train_hands=[ws[i]['index'] for i in hand],coverage=coverage,
        train_hand_cosine=(x[hand]@x[hand].T).tolist(),windows={str(ws[i]['index']):ws[i] for i in sorted(needed)},notes=notes,binary=binary,
        source_hashes=source_hashes,script_sha256=digest(__file__),test_evaluated=False,
        caveat='Nearest train neighbours are descriptive, not classifier explanations. Standardized cosine is not logistic-regression distance. More machine candidates bias maximum similarity; no causal attribution is possible from this small retrospective set.')
    assert source_hashes=={f:digest(SRC/f) for f in source_hashes}
    assert all(result['windows'][str(q['spaces'][s][k]['index'])]['split']=='train' for q in results for s in ('raw','standardized') for k in ('best_hand','best_machine'))
    save_json(OUT/'results.json',result);(OUT/'results.js').write_text('window.ACTION_DIAGNOSIS = '+json.dumps(result,ensure_ascii=False)+';\n')
    print('Diagnosis:',coverage,'binary',binary['validation'],'clips',len(needed),flush=True)

if __name__=='__main__':main()
