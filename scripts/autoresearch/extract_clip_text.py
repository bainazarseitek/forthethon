"""CLIP ViT-L/14 text-prompt similarity features (pre-registered prompt set; do not iterate).

Two stages:
  --encode  (any host with the pinned CLIP snapshot; text encoder only):
            text_projection(text_model(...).pooler_output), L2 -> (P, 768) float32,
            written to review/generated/autoresearch/clip-text/ (new dir, never overwritten).
  --sims    (local): cosine similarity of saved image projections with the text embeddings ->
            review/generated/autoresearch/features/cliptext.npz
              sim_mean   (160, P) = features.npz 'clip' (L2 window mean) @ T.T
              sim_frames (160, P) = mean over 16 frames of clipframes 'frames' @ T.T
Only the 160 train/val windows are used. Prompt list fixed before any result was seen.
"""
import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "review/generated/action-5090"
OUT = ROOT / "review/generated/autoresearch/clip-text"
FEAT = ROOT / "review/generated/autoresearch/features"

# Pre-registered: 4 paraphrases x 6 classes, label order as evaluate.LABELS.
PROMPTS = {
    "hand_count": [
        "a cashier counting banknotes by hand",
        "a person counting paper money with their fingers",
        "hands flipping through a stack of banknotes on a desk",
        "a bank teller manually counting cash",
    ],
    "machine": [
        "a banknote counting machine counting money",
        "a cash counting machine on a desk with banknotes",
        "a person putting a stack of money into a money counter machine",
        "an electronic bill counter running",
    ],
    "laptop": [
        "a person typing on a laptop",
        "a cashier working on a computer keyboard",
        "hands on a laptop keyboard at a desk",
        "a bank teller using a laptop",
    ],
    "document": [
        "a person checking an ID document",
        "a cashier examining a passport",
        "a person reading a paper document at a desk",
        "hands holding an identity card",
    ],
    "safe": [
        "a person opening a safe",
        "a cashier putting money into a metal safe",
        "a person taking cash out of a safe",
        "an open safe door in a bank office",
    ],
    "leave": [
        "an empty chair at a cashier desk",
        "an empty bank teller workplace with nobody there",
        "a desk with no person sitting at it",
        "an empty office room",
    ],
}
FLAT = [(c, p) for c, ps in PROMPTS.items() for p in ps]


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def encode():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from transformers import CLIPModel, CLIPTokenizer

    assert not OUT.exists(), "never overwrite an output dir"
    spec = json.loads((SRC / "prepared.json").read_text())["config"]["models"]["clip"]
    path = Path.home() / ".cache/huggingface/hub" / ("models--" + spec["id"].replace("/", "--")) / "snapshots" / spec["revision"]
    expected = {m["name"]: m["sha256"] for m in json.loads((ROOT / "data/gpu_models.json").read_text())}
    sha = digest(path / "model.safetensors")
    assert sha == expected["clip"]
    model = CLIPModel.from_pretrained(path, local_files_only=True, use_safetensors=True).eval().requires_grad_(False)
    tok = CLIPTokenizer.from_pretrained(path, local_files_only=True)
    tokens = tok([p for _, p in FLAT], padding=True, return_tensors="pt")
    with torch.inference_mode():
        t = model.text_projection(model.text_model(**tokens).pooler_output)
        t = torch.nn.functional.normalize(t.float(), dim=-1).numpy().astype(np.float32)
    assert t.shape == (len(FLAT), 768) and np.isfinite(t).all()
    tmp = Path(tempfile.mkdtemp(dir=OUT.parent, prefix=".clip-text-tmp-"))
    np.savez(tmp / "text.npz", text=t, classes=np.array([c for c, _ in FLAT]), prompts=np.array([p for _, p in FLAT]))
    (tmp / "provenance.json").write_text(json.dumps(dict(
        model=spec, safetensors_sha256=sha, torch=torch.__version__, script_sha256=digest(__file__),
        n_prompts=len(FLAT), normalization="text_projection(pooler_output), L2"), indent=1))
    os.rename(tmp, OUT)
    print("wrote", OUT, t.shape)


def sims():
    dst = FEAT / "cliptext.npz"
    assert not dst.exists(), "never overwrite"
    d = np.load(OUT / "text.npz")
    assert list(d["prompts"]) == [p for _, p in FLAT], "prompt list changed after encoding"
    t = d["text"]
    clip = np.load(SRC / "features.npz")["clip"]
    frames = np.load(FEAT / "clipframes.npz")["frames"]
    assert clip.shape == (160, 768) and frames.shape == (160, 16, 768)
    sim_mean = (clip @ t.T).astype(np.float32)
    sim_frames = np.einsum("nfd,pd->np", frames, t).astype(np.float32) / frames.shape[1]
    fd, tmp = tempfile.mkstemp(dir=FEAT.parent, suffix=".npz")
    os.close(fd)
    np.savez(tmp, sim_mean=sim_mean, sim_frames=sim_frames)
    os.rename(tmp, dst)
    print("wrote", dst, sim_mean.shape, sim_frames.shape)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--encode", action="store_true")
    g.add_argument("--sims", action="store_true")
    a = ap.parse_args()
    encode() if a.encode else sims()
