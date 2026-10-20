"""Grounding DINO cash/person observations, cached CLIP actions, unchanged causal memory.

infer on gpu-host, analyze locally. Previous experiment and original annotations stay intact.
"""
import argparse
import fcntl
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from run_protocol_pilot import apply_memory, check as check_memory, digest, normalize, save_json

ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = ROOT / "review/generated/protocol-pilot"
OUT = ROOT / "review/generated/detector-pilot"
CONFIG = ROOT / "data/detector_pilot.json"
TABLE_VIEW = "wide"


def inside(point, polygon):
    x, y = point
    hit = False
    for a, b in zip(polygon, polygon[1:] + polygon[:1]):
        if (a[1] > y) != (b[1] > y) and x < (b[0]-a[0])*(y-a[1])/(b[1]-a[1])+a[0]:
            hit = not hit
    return hit


def center(box):
    return [(box[0]+box[2])/2, (box[1]+box[3])/2]


def in_rect(point, box):
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def overlap_fraction(target, other):
    w = max(0, min(target[2], other[2])-max(target[0], other[0]))
    h = max(0, min(target[3], other[3])-max(target[1], other[1]))
    return w*h / max(1e-9, (target[2]-target[0])*(target[3]-target[1]))


def support(score, low, high):
    return float(np.clip((score-low)/(high-low), 0, 1))


def observe(detections, config):
    """Only current-frame boxes and geometry; no temporal annotations or video IDs."""
    people = [b for b in detections if b["kind"] == "person"]
    hands = [b for b in detections if b["kind"] == "hand" and b["score"] >= config["hand_score"]]
    boxes = []
    for b in detections:
        b = dict(b)
        if b["kind"] == "person":
            b["in_zone"] = in_rect(center(b["box"]), config["cashier_zone"])
        if b["kind"] == "cash":
            b["on_surface"] = inside(center(b["box"]), config["table_surface"])
            b["hand_overlap"] = max((overlap_fraction(b["box"], h["box"]) for h in hands), default=0)
            b["on_table"] = b["on_surface"] and b["hand_overlap"] < config["cash_hand_overlap"]
        boxes.append(b)
    cash = max((b["score"] for b in boxes if b["kind"] == "cash" and b["on_table"]), default=0)
    person = max((b["score"] for b in boxes if b["kind"] == "person" and b["in_zone"]), default=0)
    covered = max((overlap_fraction(config["visibility_patch"], h["box"]) for h in hands), default=0)
    table = ("cash" if cash >= config["cash_high"] else "occluded" if covered >= config["occluded_patch_fraction"] else
             "clear" if cash <= config["cash_low"] else "unknown")
    presence = "attending" if person >= config["person_high"] else "away" if person <= config["person_low"] else "unknown"
    # Adapter supplies explicit states to the old memory function; these are not cosine scores.
    state_scores = {k: float(table == k) for k in ["cash", "clear", "occluded", "paper"]}
    return dict(boxes=boxes, cash_raw=cash, person_raw=person, hand_coverage=covered,
        table=state_scores, presence_observation=presence,
        cash_support=support(cash, config["cash_low"], config["cash_high"]),
        away_support=1-support(person, config["person_low"], config["person_high"]))


def infer():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    import transformers
    from torchvision.ops import nms, box_convert
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    started = time.monotonic()
    config = json.loads(CONFIG.read_text())
    config["table_view"] = TABLE_VIEW
    if TABLE_VIEW == "patch": config["table_crop"] = config["visibility_patch"]
    prepared = json.loads((PREVIOUS / "prepared.json").read_text())
    checkpoint = Path.home()/".cache/huggingface/hub"/("models--"+config["model"].replace("/", "--"))/"snapshots"/config["revision"]
    assert (checkpoint/"model.safetensors").is_file(), "Download the pinned official safetensors snapshot first"
    torch.set_num_threads(4)
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(checkpoint, local_files_only=True,
        use_safetensors=True, trust_remote_code=False).eval().to(device)
    if device == "cuda": torch.cuda.synchronize()
    load_seconds = time.monotonic()-started
    print(f"Loaded {config['model']} on {device}; {len(prepared['decisions'])} current-frame updates", flush=True)

    def detect(image, phrases, offset):
        caption, spans = "", {}
        for key, variants in phrases.items():
            spans[key] = []
            for phrase in variants:
                spans[key].append((len(caption), len(caption)+len(phrase)))
                caption += phrase + ". "
        token_offsets = processor.tokenizer(caption, return_offsets_mapping=True)["offset_mapping"]
        inputs = processor(images=image, text=caption, return_tensors="pt").to(device)
        with torch.inference_mode():
            outputs = model(**inputs)
        probabilities = outputs.logits[0].sigmoid().float()
        boxes = box_convert(outputs.pred_boxes[0].float(), "cxcywh", "xyxy")
        boxes *= torch.tensor([image.width, image.height, image.width, image.height], device=device)
        boxes[:,[0,2]] = boxes[:,[0,2]].clamp(0,image.width)
        boxes[:,[1,3]] = boxes[:,[1,3]].clamp(0,image.height)
        result = []
        for key, intervals in spans.items():
            positions = [i for i,(a,b) in enumerate(token_offsets) if b>a and any(a<end and b>start for start,end in intervals)]
            assert positions
            scores = probabilities[:,positions].max(-1).values
            valid = (scores >= config["retain_box_score"]) & ((boxes[:,2]-boxes[:,0]) > 0) & ((boxes[:,3]-boxes[:,1]) > 0)
            selected_boxes, selected_scores = boxes[valid], scores[valid]
            keep = nms(selected_boxes, selected_scores, config["nms_iou"])[:20]
            for box, score in zip(selected_boxes[keep].cpu().tolist(), selected_scores[keep].cpu().tolist()):
                result.append(dict(kind=key, score=float(score), box=[box[0]+offset[0],box[1]+offset[1],box[2]+offset[0],box[3]+offset[1]]))
        return result

    detections, latencies, frame_hashes = [], [], {}
    for d in prepared["decisions"]:
        frame = prepared["frames"][d["frame"]]
        assert frame["pts"] <= d["t"] + 1e-7
        path = PREVIOUS / frame["file"]
        frame_hashes[frame["file"]] = digest(path)
        with Image.open(path) as image:
            tick = time.monotonic()
            boxes = detect(image, config["person_phrases"], [0,0])
            crop = config["table_crop"]
            boxes += detect(image.crop(crop), config["table_phrases"], crop[:2])
            if device == "cuda": torch.cuda.synchronize()
            latencies.append(time.monotonic()-tick)
        detections.append(dict(d, pts=frame["pts"], image=frame["file"], detections=boxes))
        if len(detections) % 5 == 0: print(f"{len(detections)}/{len(prepared['decisions'])} cash + person detections", flush=True)
    run = dict(model=config["model"], revision=config["revision"], device=device,
        gpu=torch.cuda.get_device_name() if device=="cuda" else device, dtype="float32",
        torch=torch.__version__, transformers=transformers.__version__, model_load_seconds=load_seconds,
        detector_seconds=sum(latencies), per_update_seconds=latencies,
        weights_sha256=digest(checkpoint/"model.safetensors"), script_sha256=digest(__file__),
        config_sha256=digest(CONFIG), prepared_sha256=digest(PREVIOUS/"prepared.json"),
        frame_sha256=frame_hashes, total_seconds=time.monotonic()-started,
        preprocessing="Official GroundingDINO resize/normalization; full source frame for people, configured fixed crop for cash/hand/paper. Class score=max sigmoid over category word tokens; per-class NMS 0.5; all candidate boxes >=0.1 retained.")
    save_json(OUT/"detections.json", dict(config=config, run=run, rows=detections))
    print(f"Saved {len(detections)} updates; detector seconds {sum(latencies):.2f}", flush=True)


def check():
    check_memory(OUT)
    config = json.loads(CONFIG.read_text())
    assert inside([220,405],config["table_surface"]) and not inside([200,300],config["table_surface"])
    assert overlap_fraction([0,0,10,10],[0,0,5,10]) == .5
    def box(kind, score, coords): return dict(kind=kind,score=score,box=coords)
    cash = box("cash",.8,[210,398,233,410])
    seated = box("person",.9,[140,290,235,510])
    door = box("person",.9,[80,180,170,370])
    a=observe([cash,seated],config); b=observe([cash,door],config)
    assert a["cash_support"]==1 and a["away_support"]==0
    assert b["away_support"]==1, "Person near exit is not at desk"
    hand=box("hand",.8,[200,385,242,420])
    h=observe([cash,hand],config)
    assert h["cash_support"]==0 and h["table"]["occluded"]==1, "Money overlapping hand is not a confirmed pile on table"
    unclear=observe([box("cash",.25,[210,398,233,410])],config)
    assert max(unclear["table"].values())==0, "Weak detection is unknown"
    print("Passed: cash/person geometry, hand overlap, uncertain detections", flush=True)


def metrics(rows, key, threshold):
    from sklearn.metrics import roc_auc_score
    valid=[r for r in rows if r["label"]!="unknown"]
    truth=np.array([r["label"]=="violation" for r in valid])
    scores=np.array([r[key] for r in valid])
    alarms=scores>=threshold
    return dict(auc=float(roc_auc_score(truth,scores)), true_positive=int((alarms & truth).sum()),
        false_negative=int((~alarms & truth).sum()), false_positive=int((alarms & ~truth).sum()),
        true_negative=int((~alarms & ~truth).sum()), first_alarm={v:next((r["t"] for r in rows
            if r["video"]==v and r[key]>=threshold),None) for v in ["X-26","X-27"]})


def analyze():
    check()
    raw=json.loads((OUT/"detections.json").read_text())
    config=raw["config"]
    data=json.loads((PREVIOUS/"prepared.json").read_text())
    assert raw["run"]["prepared_sha256"]==digest(PREVIOUS/"prepared.json")
    assert raw["run"]["config_sha256"]==digest(CONFIG)
    assert len(raw["rows"])==len(data["decisions"])
    features=np.load(PREVIOUS/"features.npz")
    action_text={}
    offset=0
    for key,texts in data["config"]["prompts"]["actions"].items():
        action_text[key]=normalize(features["text"][offset:offset+len(texts)].mean(0))
        offset+=len(texts)
    observations=[]
    for d in raw["rows"]:
        assert d["pts"]<=d["t"]+1e-7
        assert max(data["frames"][i]["pts"] for i in d["window"])<=d["t"]+1e-7
        assert digest(PREVIOUS/d["image"])==raw["run"]["frame_sha256"][d["image"]]
        vector=normalize(features["images"][d["window"],0].mean(0))
        actions={k:float(vector@v) for k,v in action_text.items()}
        observations.append(dict(video=d["video"],t=d["t"],pts=d["pts"],frame=d["frame"],
            image=d["image"],window=d["window"],actions=actions,**observe(d["detections"],config)))
    rows=[]
    for video in ["X-26","X-27"]:
        selected=[r for r in observations if r["video"]==video]
        result=apply_memory(selected,config)
        for n in range(1,len(selected)+1): assert apply_memory(selected[:n],config)==result[:n]
        rows+=result
    # Load labels only after prediction; cached CLIP scores are checked rather than refitted.
    previous=json.loads((PREVIOUS/"results.json").read_text())
    previous_rows={(r["video"],r["t"]):r for r in previous["rows"]}
    for row in rows:
        old=previous_rows[(row["video"],row["t"])]
        assert row["actions"]==old["actions"], "Action observations must be identical to the prior run"
        for key in ["label","table_truth","presence_truth","manual_note"]: row[key]=old[key]
        row["clip_without_memory"]=old["without_memory"]
        row["clip_with_memory"]=old["with_memory"]
    comparison={k:metrics(rows,k,config["alarm_threshold"]) for k in ["clip_without_memory","without_memory","with_memory"]}
    def audit(selected, kind):
        truth=[r["table_truth"]=="cash" if kind=="cash" else r["presence_truth"]=="away" for r in selected]
        score="cash_support" if kind=="cash" else "away_support"
        state="table_observation" if kind=="cash" else "presence_observation"
        truth_key="table_truth" if kind=="cash" else "presence_truth"
        baseline=[previous_rows[(r["video"],r["t"])] for r in selected]
        return dict(total=len(selected),
            clip_binary_correct=sum((r[score]>=.5)==y for r,y in zip(baseline,truth)),
            detector_binary_correct=sum((r[score]>=.5)==y for r,y in zip(selected,truth)),
            detector_state_correct=sum(r[state]==r[truth_key] for r in selected),
            detector_unknown=sum(r[state] in ["unknown","occluded"] for r in selected),
            points=[dict(video=r["video"],t=r["t"],truth=r[truth_key],prediction=r[state]) for r in selected])
    cash_rows=[r for r in rows if r["table_truth"]!="unknown"]
    presence_rows=[r for r in rows if r["presence_truth"]!="unknown"]
    # Same annotated late intervals and seated prefix, excluding coarse departure boundaries.
    presence_clear=[r for r in presence_rows if r["t"]<=7.5 or r["table_truth"]!="unknown"]
    audits=dict(cash=audit(cash_rows,"cash"),presence=audit(presence_rows,"presence"),
                presence_clear=audit(presence_clear,"presence"))
    for i,row in enumerate(rows):
        render_boxes(row,i,config)
    result=dict(config=config,run=raw["run"],rows=rows,metrics=comparison,audits=audits,
        sources=data["sources"],action_prompts=data["config"]["prompts"]["actions"],
        changed_scores=sum(abs(r["with_memory"]-r["without_memory"])>1e-6 for r in rows),
        changed_alarms=sum((r["with_memory"]>=.5)!=(r["without_memory"]>=.5) for r in rows),
        memory_active_points=sum(r["memory_evidence_t"] is not None for r in rows),
        confirmed_cash_points=sum(r["table_observation"]=="cash" for r in rows),
        analysis_script_sha256=digest(__file__),memory_script_sha256=digest(ROOT/"scripts/run_protocol_pilot.py"),
        previous_results_sha256=digest(PREVIOUS/"results.json"),features_sha256=digest(PREVIOUS/"features.npz"),
        annotations_sha256=digest(ROOT/"docs/annotations.md"))
    save_json(OUT/"results.json",result)
    view=config.get("table_view","wide")
    (OUT/"results.js").write_text("window.DETECTOR_PILOTS=window.DETECTOR_PILOTS||{};\nwindow.DETECTOR_PILOTS["+json.dumps(view)+"]="+json.dumps(result,ensure_ascii=False)+";\n")
    import csv
    with (OUT/"scores.csv").open("w") as f:
        columns=["video","t","pts","label","cash_raw","person_raw","hand_coverage","cash_support","away_support",
                 "clip_without_memory","without_memory","with_memory","table_observation","presence_observation","stage","memory_used"]
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction="ignore")
        writer.writeheader();writer.writerows(rows)
    plots(result)
    print(json.dumps(dict(metrics=comparison,audits={k:{a:b for a,b in v.items() if a!="points"} for k,v in audits.items()},
        changed_scores=result["changed_scores"],changed_alarms=result["changed_alarms"]),indent=2),flush=True)


def render_boxes(row,index,config):
    (OUT/"frames").mkdir(exist_ok=True)
    with Image.open(PREVIOUS/row["image"]) as source:
        image=source.copy()
    draw=ImageDraw.Draw(image)
    draw.rectangle(config["cashier_zone"],outline="#2f9bce",width=2)
    draw.line(config["table_surface"]+[config["table_surface"][0]],fill="#d47b26",width=2)
    for b in row["boxes"]:
        threshold=config["person_high"] if b["kind"]=="person" else config["hand_score"] if b["kind"]=="hand" else config["cash_high"]
        weak_cash=b["kind"]=="cash" and b.get("on_surface") and b["score"]>=config["cash_low"] and b["score"]<threshold
        if (b["score"]<threshold and not weak_cash) or b["kind"]=="paper": continue
        color=("#23a889" if b["in_zone"] else "#558bb5") if b["kind"]=="person" else "#dd7734" if b["kind"]=="cash" else "#aa5fa7"
        if weak_cash: color="#777777"
        draw.rectangle(b["box"],outline=color,width=2)
        label=f"{b['kind']}{'?' if weak_cash else ''} {b['score']:.2f}"+(" / table" if b.get("on_table") else "")
        x,y=b["box"][:2]
        draw.rectangle([x,max(0,y-13),min(464,x+len(label)*6+4),y],fill="#ffffff")
        draw.text((x+2,max(0,y-12)),label,fill="#172633")
    row["annotated_image"]=f"frames/{index:03d}.jpg"
    row["table_image"]=f"frames/{index:03d}-table.jpg"
    image.save(OUT/row["annotated_image"],quality=94)
    crop=image.crop(config["table_crop"])
    crop.resize((round(260*crop.width/crop.height),260)).save(OUT/row["table_image"],quality=94)


def plots(result):
    os.environ.setdefault("MPLCONFIGDIR",str(PREVIOUS/"matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,2,figsize=(15,13),sharex="col",layout="constrained")
    config=result["config"]
    for col,video in enumerate(["X-26","X-27"]):
        rows=[r for r in result["rows"] if r["video"]==video];t=[r["t"] for r in rows]
        axes[0,col].set_title(video+(" / manual violation from 9 s" if video=="X-26" else " / manual clean"))
        for k in rows[0]["actions"]: axes[0,col].plot(t,[r["actions"][k] for r in rows],label=k,linewidth=1.3)
        axes[0,col].set_ylabel("CLIP action cosine (unchanged)");axes[0,col].legend(fontsize=8,ncol=3)
        axes[1,col].plot(t,[r["cash_raw"] for r in rows],label="Cash box on table",color="#b75b2e")
        axes[1,col].axhline(config["cash_high"],label="Positive threshold",color="#999",linestyle=":")
        axes[1,col].plot(t,[r["cash_support"] for r in rows],label="Current support",color="#af6572",alpha=.65)
        axes[1,col].plot(t,[r["remembered_cash"] for r in rows],label="Remembered support",color="#237b66",linestyle="--")
        axes[1,col].set_ylabel("Cash: detector score / support");axes[1,col].legend(fontsize=8,ncol=2)
        axes[2,col].plot(t,[r["person_raw"] for r in rows],label="Person in cashier zone",color="#2867a1")
        axes[2,col].plot(t,[r["away_support"] for r in rows],label="Away support",color="#af6572")
        axes[2,col].axhline(config["person_high"],label="Presence threshold",color="#999",linestyle=":")
        axes[2,col].set_ylabel("Presence: detector score / support");axes[2,col].legend(fontsize=8)
        for key,label,color,style in [("clip_without_memory","Previous: CLIP only","#9a9a9a",":"),
            ("without_memory","Detectors, no memory","#af6572","-"),("with_memory","Detectors + memory","#237b66","--")]:
            axes[3,col].plot(t,[r[key] for r in rows],label=label,color=color,linestyle=style,linewidth=2)
        axes[3,col].axhline(.5,color="#555",linestyle=":")
        axes[3,col].set_ylabel("Unattended cash score");axes[3,col].legend(fontsize=8)
        axes[3,col].set_xlabel("Decision time, seconds; only past frames")
        for ax in axes[:,col]:
            ax.grid(alpha=.18)
            if video=="X-26":
                ax.axvspan(9,result["sources"][video]["duration"],color="#b54b45",alpha=.06)
                ax.axvline(9,color="#b54b45",alpha=.3)
        for ax in axes[1:,col]: ax.set_ylim(-.04,1.04)
    fig.suptitle("Separate cash/person detection + unchanged CLIP actions and causal memory\nFixed camera geometry / fixed heuristic thresholds / one pair, no held-out calibration",fontsize=15)
    fig.savefig(OUT/"timelines.png",dpi=160);plt.close(fig)


def main():
    global OUT, TABLE_VIEW
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["infer","analyze","check"])
    parser.add_argument("--table-view",choices=["wide","patch"],default="wide")
    args=parser.parse_args()
    TABLE_VIEW=args.table_view
    if TABLE_VIEW=="patch": OUT=ROOT/"review/generated/detector-pilot-patch"
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/".lock").open("w") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        globals()[args.stage]()


if __name__=="__main__": main()
