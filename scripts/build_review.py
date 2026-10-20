"""Build the local clip review assets without changing any source labels.

Run with Python + openpyxl + Pillow, with ffmpeg/ffprobe on PATH.
"""
import bisect
import hashlib
import json
import re
import subprocess
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openpyxl import load_workbook
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "review/generated"
NUM = r"\d+(?:\.\d+)?"


def canonical(value):
    return str(value).replace("Х", "X").strip()


def time_value(value):
    if value is None:
        return None
    values = [float(x) for x in re.findall(NUM, str(value))]
    if "мин" in str(value):
        return 60 * values[0] + (values[1] if len(values) > 1 else 0)
    return values[0]


def annotations():
    result = {}
    for section in re.split(r"^## ", (ROOT / "docs/annotations.md").read_text(), flags=re.M)[1:]:
        header = section.splitlines()[0]
        vid = canonical(header.split()[0])
        rows = []
        for line in section.splitlines():
            if not line.startswith("|"):
                continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) != 5 or not re.match(r"^\*?\*?\d", cells[0]):
                continue
            values = [float(x) for x in re.findall(NUM, cells[0])]
            rows.append(dict(start=min(values), end=max(values), time_text=cells[0],
                             kind=cells[1] or "context", types=cells[2], seen=cells[3], rule=cells[4]))
        meta = {}
        for key in ["operation", "people", "status", "сцена"]:
            match = re.search(r"^- " + key + r": (.*)$", section, re.M)
            meta[key] = match.group(1) if match else ""
        result[vid] = dict(**meta, rows=rows, source_text=section.strip())
    result["X-48"] = dict(result["X-5"], inherited_from="X-5")
    return result


def run(*args):
    return subprocess.check_output(args, text=True)


def nearest(values, target):
    index = bisect.bisect_left(values, target)
    return min([max(0, index - 1), min(len(values) - 1, index)], key=lambda i: abs(values[i] - target))


def prepare(clip):
    path = ROOT / clip["source"]
    info = json.loads(run("ffprobe", "-v", "error", "-select_streams", "v:0",
                          "-show_streams", "-show_format", "-of", "json", str(path)))
    stream = info["streams"][0]
    timing = json.loads(run("ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
                            "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", str(path)))
    pts = [float(f["best_effort_timestamp_time"]) for f in timing["frames"]]
    assert pts and pts == sorted(pts), path
    assert abs(pts[0]) < 0.001, "Nonzero media start requires an explicit timeline offset"
    duration = float(stream["duration"])
    targets = list(range(0, int(duration) + 1, 2)) + [pts[-1]]
    targets += [r["start"] for r in clip["annotation"]["rows"]]
    if clip["official_time"] is not None:
        targets += [clip["official_time"] + t / 2 for t in range(-4, 5)]
    indexes = sorted({nearest(pts, min(max(t, 0), pts[-1])) for t in targets})
    folder = OUT / "frames" / clip["id"]
    folder.mkdir(parents=True, exist_ok=True)
    expression = "+".join(f"eq(n\\,{i})" for i in indexes)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(path), "-vf",
                    f"select='{expression}',scale=256:-2", "-fps_mode", "vfr", "-q:v", "3",
                    str(folder / "%04d.jpg")], check=True)
    files = sorted(folder.glob("*.jpg"))
    assert len(files) == len(indexes), (clip["id"], len(files), len(indexes))
    frames = [dict(t=round(pts[i], 6), file=f"generated/frames/{clip['id']}/{f.name}")
              for i, f in zip(indexes, files)]
    increments = Counter(round(b - a, 4) for a, b in zip(pts, pts[1:]))
    clip.update(duration=duration, width=stream["width"], height=stream["height"],
                codec=stream["codec_name"], frame_count=len(pts), frame_deltas=dict(increments),
                frames=frames, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    # Remux preserves the original decoded frames and PTS; MP4 is portable across browsers.
    mp4 = OUT / "video" / f"{clip['id']}.mp4"
    mp4.parent.mkdir(exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(path), "-map", "0:v:0", "-c:v", "copy",
                    "-an", "-movflags", "+faststart", str(mp4)], check=True)
    clip["video"] = f"generated/video/{clip['id']}.mp4"
    print(f"{clip['id']}: {duration:.1f}s, {len(frames)} review frames", flush=True)
    return clip


def sheets(clips):
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 15)
    except OSError:
        font = ImageFont.load_default()
    folder = OUT / "contact-sheets"
    folder.mkdir(exist_ok=True)
    unique = [c for c in clips if not c.get("duplicate_of")]
    for batch in range(0, len(unique), 4):
        group = unique[batch:batch + 4]
        canvas = Image.new("RGB", (8 * 150, len(group) * 320), "#101a27")
        draw = ImageDraw.Draw(canvas)
        for row, clip in enumerate(group):
            times = [clip["duration"] * i / 8 for i in range(8)]
            if clip["official_time"] is not None:
                times[nearest(times, clip["official_time"])] = clip["official_time"]
            title = f"{clip['id']} | {'LABEL '+str(clip['official_time'])+'s' if clip['official_time'] is not None else 'NO VIOLATION LABEL'} | {clip['duration']:.1f}s"
            draw.text((6, row * 320 + 5), title, font=font, fill="white")
            frame_times = [f["t"] for f in clip["frames"]]
            for col, t in enumerate(sorted(times)):
                frame = clip["frames"][nearest(frame_times, t)]
                im = Image.open(ROOT / "review" / frame["file"])
                im.thumbnail((146, 267))
                canvas.paste(im, (col * 150 + 2, row * 320 + 29))
                draw.text((col * 150 + 4, row * 320 + 299), f"{frame['t']:.3f}s", font=font, fill="#d1d9e2")
        canvas.save(folder / f"{batch // 4 + 1:02d}.jpg", quality=88)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (ROOT / "assets/videos").mkdir(exist_ok=True)
    with zipfile.ZipFile(ROOT / "open.zip") as archive:
        for entry in archive.namelist():
            if entry.endswith(".mov"):
                path = ROOT / "assets/videos" / Path(entry).name
                if not path.exists():
                    path.write_bytes(archive.read(entry))
        assert archive.read("open/open_labels.xlsx") == (ROOT / "assets/open_labels.xlsx").read_bytes()
    book = load_workbook(ROOT / "assets/open_labels.xlsx", read_only=True, data_only=True)
    official = list(book.active.values)[1:]
    manual = load_workbook(ROOT / "assets/team_labels.xlsx", read_only=True, data_only=True)
    notes = {canonical(r[0]): r[3] or "" for r in list(manual.active.values)[1:]}
    ann = annotations()
    truth = json.loads((ROOT / "data/ground_truth.json").read_text())
    ground = {canonical(v["video"]): v for v in truth["videos"]}
    clips = []
    for name, timestamp, label in official:
        vid = canonical(name)
        path = ROOT / "assets/videos" / f"{name}.mov"
        assert path.is_file(), path
        clips.append(dict(id=vid, original_id=name, source=str(path.relative_to(ROOT)),
                          official_time=time_value(timestamp), label="clean" if label == "без нарушения" else "violation",
                          manual_note=notes[vid], annotation=ann[vid], ground_truth=ground[vid]))
    assert len(clips) == 42 and len({c["id"] for c in clips}) == 42
    with ThreadPoolExecutor(max_workers=4) as pool:
        clips = list(pool.map(prepare, clips))
    groups = defaultdict(list)
    for clip in clips:
        groups[clip["sha256"]].append(clip["id"])
    for clip in clips:
        clip["same_content"] = groups[clip["sha256"]]
        if clip["same_content"][0] != clip["id"]:
            clip["duplicate_of"] = clip["same_content"][0]
        assert clip["official_time"] is None or 0 <= clip["official_time"] < clip["duration"]
        assert clip["ground_truth"]["label_t_sec"] == clip["official_time"], clip["id"]
    # These are curated comparison candidates, not automatically inferred causal matches.
    pairs = [("X-2", "X-3", "Document before cash"), ("X-1", "X-3", "Cash handover"),
             ("X-4", "X-5", "Store cash before next customer"), ("X-11", "X-12", "Leaving the workstation"),
             ("X-13", "X-14", "Counting before storage"), ("X-17", "X-16", "Bag acceptance"),
             ("X-18", "X-19", "Finish the first customer"), ("X-20", "X-21", "Safe door"),
             ("X-22", "X-23", "Remove the safe key"), ("X-24", "X-25", "Store stamps and journal"),
             ("X-26", "X-27", "Store cash before leaving"), ("X-28", "X-29", "Workstation screen"),
             ("X-30", "X-31", "Close the entrance"), ("X-32", "X-31", "Close the entrance"),
             ("X-33", "X-31", "Entrance and second person"), ("X-34", "X-35", "Count with customer present"),
             ("X-34", "X-37", "Count with customer present (short example)"),
             ("X-38", "X-39", "Store previous customer's cash"), ("X-44", "X-43", "Check separated notes"),
             ("X-44", "X-45", "Check separated notes (second example)")]
    summary = dict(clips=len(clips), unique=len(groups),
                   labels=dict(Counter(c["label"] for c in clips)),
                   duration=sum(c["duration"] for c in clips),
                   unique_duration=sum(c["duration"] for c in clips if not c.get("duplicate_of")),
                   duplicates=[g for g in groups.values() if len(g) > 1],
                   frames=sum(len(c["frames"]) for c in clips))
    result = dict(summary=summary, clips=clips, pairs=pairs,
                  rules=json.loads((ROOT / "data/rules.json").read_text()))
    (OUT / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    (OUT / "manifest.js").write_text("window.REVIEW_DATA = " + json.dumps(result, ensure_ascii=False).replace("</", "<\\/") + ";\n")
    sheets(clips)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    assert time_value("1минута 8 секунда") == 68
    assert time_value("22 секунда") == 22
    assert canonical("Х-47") == "X-47"
    assert nearest([0, .033333, .1], .06) == 1
    main()
