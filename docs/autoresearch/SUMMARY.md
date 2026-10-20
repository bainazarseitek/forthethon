# Autoresearch summary: cashier action recognition (6 classes)

Date: 2026-10-17 · Branch: `baseline/action-recognition-protocol-memory` (PR #3).
Scope: the frozen 160 train/validation windows only. **Test windows/videos were never loaded, encoded or viewed.**
Detailed record: [`log.md`](log.md) · every run: [`ledger.jsonl`](ledger.jsonl) · rules: [`PROTOCOL.md`](PROTOCOL.md).

## TL;DR

- 51 pre-registered candidates over 6 iterations, run by 24 subagents on two GPUs (RTX 5090, L4) plus local CPU.
- **One benchmark win:** late fusion of a CLIP head and a SigLIP head. Cross-validated macro F1 0.601 → **0.732**, frozen validation 0.661 → **0.677**, hand_count 0.105 → 0.500. It passed every in-benchmark robustness check.
- **Not confirmed on fresh data:** on 21 never-used windows it ties the CLIP baseline (macro F1 0.778 vs 0.767, accuracy 16/21 vs 17/21). The fresh set is machine-heavy (17/21) and has one hand_count window, so it can neither confirm nor refute the gain.
- **Root failure mode is session identity:** CLIP gets hand_count F1 0.552 when the paired recording session is in training, 0.105 when it is not.
- "Scenario groups" are recording sessions on only ~4 camera setups; no split in the dataset, including test, measures generalisation to a new camera or branch.
- Classifier capacity, motion features, hand crops, label-noise filtering, sub-window training and flips do not fix it.
- The 160-window benchmark and the existing annotations are exhausted as a selection tool. Further progress needs **new labelled hand_count sessions and footage from new camera setups**.

## Setup

**Data.** 160 windows (123 train / 37 validation) of 2 s, 16 frames, from the frozen action manifest (`review/generated/action-5090/prepared.json`). Class counts train+val: hand_count 8, machine 52, laptop 35, document 6, safe 43, leave 16. hand_count spans 5 videos in 3 scenario groups; any training fold sees at most 2 of them.

**Starting point.** CLIP ViT-L/14 window-mean embedding → train-only `StandardScaler` → balanced logistic regression, C=10 (`candidates/baseline_clip.py`). It reproduces the 5090 benchmark exactly (validation macro F1 0.6614).

**Evaluation** (`scripts/autoresearch/evaluate.py`):

| Element | Definition |
|---|---|
| CV | leave-one-scenario-group-out over train+val, 15 groups (connected components of the `pairs` graph), pooled macro F1 |
| Frozen val | fit on train, score the 37 validation windows (comparable to earlier reports) |
| Noise check | paired bootstrap over scenario groups, 2000 resamples, of CV Δ macro F1 vs the champion |
| Win rule | CV Δ CI95 lower bound > 0 **and** val Δ ≥ 0 |
| Confirmation of a win | deterministic rerun, neighbouring constants form no isolated spike, leave-one-video-out not worse, then a fresh-data check |
| Discipline | constants fixed before running (pre-registration), hyperparameters only by inner grouped CV on the training fold, stochastic candidates average ≥5 seeds, test labels and label-derived metadata hidden from candidates |

## Results by direction

Δ values are against the CLIP baseline unless stated. Full table in the appendix.

| Direction | Candidates | Best result | Verdict | Takeaway |
|---|---|---|---|---|
| Classifier (NCM, shrinkage LDA, inner-CV LR, RBF SVM, MLP) | 5 | MLP: CV +0.004 | noise / NCM loss | Head capacity is not the bottleneck |
| Encoder fusion (VideoMAE v1/v2, DINOv2, X-CLIP) | 8 | X-CLIP late fusion: CV +0.016, val −0.072 | noise | Encoders weaker than CLIP add noise when fused |
| **SigLIP (+ CLIP late fusion)** | 2 | **CV +0.130 [+0.008, +0.211], val +0.016** | **win** (unconfirmed on fresh data) | Two equally strong image–text encoders trained on different data make partly different session-specific errors |
| Motion (CLIP frame diffs, compact motion, VideoMAEv2 temporal tokens, pixel/optical flow) | 12 | motion_blend: CV +0.038, val +0.010 | noise; two-stage and v2-only losses | CLIP frame differences lift hand_count in single runs, never confirmed; pixel and v2 motion carry no hand signal |
| Session invariance (NAP, per-video scene embedding, scene-offset augmentation, text-prompt similarities, bagging) | 14 | NAP K=8: CV +0.097, val −0.034 (with motion: CV +0.104, val −0.103) | noise | Several shrink the seen/unseen gap, but gains sit on the same few windows; NAP's peak is an isolated K spike |
| Augmentation (sub-window MIL, horizontal flip) | 6 | flip train-aug: CV +0.049, val +0.001 | noise | Flip helps a little as jitter (not explained by row duplication) but leaves session identity intact |
| Label-noise handling (boundary down-weighting, confident learning) | 2 | centrality: CV −0.000 | noise | Confident learning deletes scene-novel windows, not mislabelled ones (val −0.240) |
| Hand crops (GroundingDINO "hands", top-2 boxes) | 2 | late fusion vs CLIP: CV +0.055, val −0.129 | noise | Hands found in 90% of frames, but localisation is inconsistent across sessions; hand_count unchanged |

## The win and its confirmation

`candidates/siglip_late_clip.py`: equal-weight average of class probabilities from the CLIP head and a SigLIP so400m head (window mean of per-frame embeddings decoded at 384 px), each `StandardScaler` + balanced LR C=10. Pre-registered before any SigLIP evaluation.

| Check | Result |
|---|---|
| Evaluator verdict | CV 0.732, Δ +0.130 [+0.008, +0.211], p(Δ≤0) 0.009; val 0.677, Δ +0.016 → **win** |
| Deterministic rerun | identical predictions |
| Bootstrap seed | CI95 lower bound > 0 for 5/5 seeds (+0.0075…+0.0105) |
| Neighbouring constants | smooth hump over fusion weight 0.25–0.75 and C 1–100; all CV Δ positive; pre-registered values kept |
| Leave-one-video-out | 0.728 vs 0.702 (not worse) |
| Group jackknife | Δ stays positive with any one group removed (min +0.061 without X-21) |
| Concentration | 11 windows fixed, 1 broken, all in 3 groups (X-21 hand ×5, X-31 machine ×5, X-14 ×1); hand precision 0.375 |
| **Fresh windows (confirm-v1)** | 21 windows from unused train/val annotation rows, labels from a type→class table fixed before any model output; each predicted by models trained without its scenario group |

Fresh-data result (evaluated once):

| Protocol | CLIP+SigLIP | CLIP | Δ macro F1 [CI95] | Δ accuracy |
|---|---|---|---|---|
| Primary: own group held out | 0.778, 16/21 | 0.767, 17/21 | +0.012 [−0.083, +0.030], p(Δ≤0) 0.85 | −1 window |
| Secondary: all 160 for training | 0.710, 16/21 | 0.792, 17/21 | −0.082 [−0.249, 0.000] | −1 window |

Pre-registered decision: "consistent" (Δ > 0, CI includes 0). In practice a tie: the machine part of the gain does not replicate, and the fusion again over-predicts hand_count on long machine episodes. `champion.txt` still points to the fusion, because it won under the pre-registered rule, but the gain is **unconfirmed as generalisation**.

## Key findings

1. **Session identity is the failure mode.** Each window's CLIP vector has cosine 0.975 with its video's whole-video mean, so the embedding is mostly "which session".

   | Model | Unseen session (LOGO) macro / hand | Paired session in training (LOVO) macro / hand | Gap |
   |---|---|---|---|
   | CLIP | 0.601 / 0.105 | 0.702 / 0.552 | 0.101 |
   | NAP K=8 | 0.698 / 0.588 | 0.739 / 0.593 | 0.040 |
   | Scene-relative concat | 0.649 / 0.435 | 0.663 / 0.615 | 0.014 |
   | CLIP+SigLIP | 0.732 / 0.500 | 0.728 / 0.516 | −0.004 |

2. **Groups are sessions, not scenes.** Annotation scene notes show ~4 camera setups in train/val: side view (X-1..3), behind the cashier (X-16..19), front view (X-8, X-10..15, X-20..25, X-30..33, X-40, X-43..46), behind the cashier with a client (X-38, X-39). Test videos reuse them. Nothing measures a new camera or branch.
3. **hand_count F1 has a ±0.3 noise floor.** In one candidate, changing only the augmentation seed moved hand F1 between 0.09 and 0.40. Single-run hand gains of that size are noise.
4. **Validation is burned.** ~51 comparisons on 37 windows; a Δ of ±0.05 is one or two windows. Under ~51 looks, a single per-candidate win can be a false positive, which is why the fresh check was added.
5. **Annotations are exhausted for these six classes.** Of 147 unused train/val rows, 23 map to a class by type, but 17 of those are too short for a 2 s window; the other 6 give the 21 fresh windows. The 70 remaining rows long enough for a window are mostly client actions, bag/door states, transitions and idle time. Description-based labelling would add only ~15–20 windows, mostly machine and leave.
6. **Frozen labels are not fully consistent.** X-20 `safe_left_open` is labelled hand_count; X-13 `stored_without_count` is labelled both safe and leave. Mixed intervals (X-8 10–18 s `machine_unload, hand_count`; X-14 48–112 s) need boundary review.
7. **CLIP is sensitive to tiny shifts.** The prepared frames are padded 50 px left / 51 px right; a 1 px shift moves the window embedding about half as much as a horizontal mirror (cosine 0.991 vs 0.980).

## Process notes and corrections

- **Evaluator hardening.** Test folds hid `y`, but their `meta` still carried `label`, `seen` and annotation bounds. An audit found no candidate reading them; `evaluate.py` now strips them (`blind()`), and re-running two candidates gave identical scores.
- **Corrections during the work.** hand_count spans 3 scenario groups, not 2. X-10's hand window is the same front camera as X-8 (a new session, not a new scene). "Scene identity" in early entries means session identity. The orchestrator's pre-result critique of the SigLIP experiment was wrong about fusion.
- **Operational issues.** fp32 SDPA attention was not repeatable on the RTX 5090 (Blackwell), so SigLIP uses eager attention. One evaluation crashed on a half-copied feature file, which led to the atomic-`mv` rule. One agent was interrupted by an API rate limit and resumed without rerunning its finished extraction.

## Recommendations / open decisions

1. **Data:** label new, independent hand_count sessions and record footage from new camera setups. This is the only lever left that should move hand counting and the only way to test deployment generalisation.
2. **Boundary review** of mixed intervals (X-8 10–18 s, X-14 48–112 s) and the inconsistent frozen labels above, kept independent of model predictions.
3. **Model choice:** keep CLIP as the practical baseline and CLIP+SigLIP as the leading candidate until fresh data confirms it (SigLIP adds a 3.5 GB model and a second forward pass).
4. **Test evaluation:** only with explicit authorisation, pre-registered, both models, run once.

## Reproduce

```sh
# score a candidate against the champion (appends to ledger.jsonl)
python3 scripts/autoresearch/evaluate.py scripts/autoresearch/candidates/<name>.py --note "<hypothesis>"
# against a specific baseline, without logging
python3 scripts/autoresearch/evaluate.py scripts/autoresearch/candidates/<name>.py --vs scripts/autoresearch/candidates/baseline_clip.py --no-log
```

| Path | Content |
|---|---|
| `scripts/autoresearch/evaluate.py` | evaluator (CV, val, bootstrap, verdict, ledger) |
| `scripts/autoresearch/candidates/` | 52 candidate files (`fit_predict(train, test)`) incl. `baseline_clip.py` |
| `scripts/autoresearch/champion.txt` | current champion pointer |
| `scripts/autoresearch/extract_*.py` | GPU feature extraction (CLIP frames/flip/scene/text, SigLIP, DINOv2, VideoMAEv2 tokens, X-CLIP, hand crops, pixel motion) |
| `scripts/autoresearch/build_confirm_set.py`, `extract_confirm_features.py`, `confirm_eval.py` | fresh-window confirmation (`confirm_eval.py` refuses a second run) |
| [`docs/autoresearch/results/`](results/) | committed snapshot: fresh-window set and evaluation (`confirm-v1/`), feature-extraction provenance (`provenance/`), hand-crop contact sheet |
| `review/generated/autoresearch/features/*.npz` | 160-row features (local only): clipframes, clipflip, clipscene, cliptext, dinov2, handcrop, pixmotion, siglip, v2tokens, xclip |
| `review/generated/autoresearch/confirm-v1/` | fresh-window features (local only) |

`review/generated/` is git-ignored: the feature arrays (~55 MB) exist only locally and on the GPU hosts, so rerunning the evaluator requires regenerating them with the `extract_*.py` scripts. The RTX 5090 container has no persistent volume; everything needed is already copied locally.

Model weights used (official copies, hashes checked):

| Model | Repository @ revision | SHA-256 (prefix) |
|---|---|---|
| CLIP ViT-L/14 | openai/clip-vit-large-patch14 @ 32bd6428 | a2bf730a |
| SigLIP so400m | google/siglip-so400m-patch14-384 @ 9fdffc58 | ea2abad2 |
| DINOv2 base | facebook/dinov2-base @ f9e44c81 | d73036b5 |
| X-CLIP B/16 | microsoft/xclip-base-patch16-zero-shot @ 6746d6a5 | 0057077d |
| VideoMAEv2 ViT-B (K710 distilled) | OpenGVLab/VideoMAE2 @ 706cc172 | 8141a695 |
| VideoMAE v1 base | MCG-NJU/videomae-base @ dc740ced | bc053ca2 |
| GroundingDINO base | IDEA-Research/grounding-dino-base @ 12bdfa31 | 5548f844 |

## Appendix: all 51 ledger runs

"vs" is the champion at run time. Hand = cross-validated hand_count F1.

| # | Candidate | vs | CV F1 | CV Δ [CI95] | Val F1 | Val Δ | Hand | Verdict |
|---|---|---|---|---|---|---|---|---|
| 1 | `fewshot_ncm` | CLIP | 0.476 | −0.126 [−0.293, −0.006] | 0.230 | −0.431 | 0.167 | loss |
| 2 | `fuse_concat_clip_v2` | CLIP | 0.562 | −0.039 [−0.129, +0.040] | 0.619 | −0.043 | 0.154 | noise |
| 3 | `fewshot_lda` | CLIP | 0.566 | −0.036 [−0.179, +0.047] | 0.626 | −0.036 | 0.000 | noise |
| 4 | `fuse_late_clip_v2` | CLIP | 0.601 | −0.000 [−0.073, +0.063] | 0.668 | +0.007 | 0.111 | noise |
| 5 | `fewshot_lr_innercv` | CLIP | 0.582 | −0.020 [−0.066, +0.019] | 0.625 | −0.036 | 0.143 | noise |
| 6 | `fuse_late_innercv` | CLIP | 0.610 | +0.009 [−0.052, +0.065] | 0.596 | −0.066 | 0.143 | noise |
| 7 | `temporal_meanstd` | CLIP | 0.552 | −0.049 [−0.105, +0.030] | 0.494 | −0.167 | 0.182 | noise |
| 8 | `temporal_meandiff` | CLIP | 0.615 | +0.014 [−0.061, +0.108] | 0.613 | −0.049 | 0.462 | noise |
| 9 | `temporal_halves` | CLIP | 0.588 | −0.014 [−0.150, +0.073] | 0.578 | −0.083 | 0.364 | noise |
| 10 | `mil_frame_lr` | CLIP | 0.613 | +0.011 [−0.061, +0.082] | 0.564 | −0.097 | 0.091 | noise |
| 11 | `motion_pca16` | CLIP | 0.631 | +0.030 [+0.000, +0.064] | 0.687 | +0.026 | 0.200 | noise |
| 12 | `mil_chunk4_lr` | CLIP | 0.608 | +0.006 [−0.014, +0.054] | 0.555 | −0.107 | 0.100 | noise |
| 13 | `motion_scalars` | CLIP | 0.623 | +0.022 [−0.056, +0.112] | 0.715 | +0.053 | 0.364 | noise |
| 14 | `motion_twostage` | CLIP | 0.549 | −0.053 [−0.104, −0.026] | 0.588 | −0.074 | 0.056 | loss |
| 15 | `mil_innercv` | CLIP | 0.606 | +0.004 [−0.034, +0.057] | 0.629 | −0.032 | 0.118 | noise |
| 16 | `v2temporal_meandiff` | CLIP | 0.387 | −0.215 [−0.366, −0.045] | 0.513 | −0.148 | 0.000 | loss |
| 17 | `v2temporal_clip_pcadiff` | CLIP | 0.601 | +0.000 [+0.000, +0.000] | 0.652 | −0.010 | 0.105 | noise |
| 18 | `v2temporal_clip_pcadiff_innercv` | CLIP | 0.603 | +0.002 [−0.017, +0.018] | 0.601 | −0.060 | 0.133 | noise |
| 19 | `motion_blend` | CLIP | 0.640 | +0.038 [−0.046, +0.141] | 0.672 | +0.010 | 0.364 | noise |
| 20 | `pixmotion_w3` | CLIP | 0.499 | −0.102 [−0.216, +0.039] | 0.536 | −0.125 | 0.222 | noise |
| 21 | `pixmotion_innercv` | CLIP | 0.477 | −0.124 [−0.253, +0.012] | 0.625 | −0.036 | 0.000 | noise |
| 22 | `dino_cls_mean` | CLIP | 0.548 | −0.054 [−0.240, +0.051] | 0.318 | −0.344 | 0.250 | noise |
| 23 | `dino_late_clip` | CLIP | 0.571 | −0.030 [−0.125, +0.029] | 0.633 | −0.028 | 0.211 | noise |
| 24 | `dino_cls_patchdiff_pca16` | CLIP | 0.531 | −0.070 [−0.267, +0.033] | 0.318 | −0.344 | 0.194 | noise |
| 25 | `sceneinv_weights` | CLIP | 0.599 | −0.002 [−0.005, +0.000] | 0.715 | +0.053 | 0.105 | noise |
| 26 | `sceneinv_nap` | CLIP | 0.698 | +0.097 [+0.000, +0.163] | 0.628 | −0.034 | 0.588 | noise |
| 27 | `sceneinv_combined` | CLIP | 0.682 | +0.080 [+0.000, +0.140] | 0.628 | −0.034 | 0.471 | noise |
| 28 | `text_sim_lr` | CLIP | 0.542 | −0.060 [−0.189, +0.031] | 0.485 | −0.176 | 0.500 | noise |
| 29 | `text_clip_concat` | CLIP | 0.634 | +0.033 [−0.015, +0.085] | 0.482 | −0.179 | 0.286 | noise |
| 30 | `text_late_fusion` | CLIP | 0.626 | +0.024 [−0.035, +0.081] | 0.610 | −0.052 | 0.316 | noise |
| 31 | `scene_sub` | CLIP | 0.592 | −0.010 [−0.188, +0.119] | 0.775 | +0.114 | 0.148 | noise |
| 32 | `scene_sub_std` | CLIP | 0.558 | −0.044 [−0.220, +0.094] | 0.599 | −0.062 | 0.087 | noise |
| 33 | `scene_concat` | CLIP | 0.649 | +0.047 [−0.090, +0.143] | 0.773 | +0.112 | 0.435 | noise |
| 34 | `nap_kensemble` | CLIP | 0.640 | +0.039 [+0.000, +0.075] | 0.628 | −0.034 | 0.211 | noise |
| 35 | `aug_scenetransfer` | CLIP | 0.650 | +0.049 [−0.010, +0.104] | 0.672 | +0.010 | 0.333 | noise |
| 36 | `nap_motion` | CLIP | 0.705 | +0.104 [+0.004, +0.170] | 0.559 | −0.103 | 0.588 | noise |
| 37 | `nonlin_rbfsvm_innercv` | CLIP | 0.482 | −0.120 [−0.349, +0.073] | 0.600 | −0.062 | 0.000 | noise |
| 38 | `bag_dropgroup` | CLIP | 0.605 | +0.004 [+0.000, +0.013] | 0.706 | +0.045 | 0.111 | noise |
| 39 | `nonlin_mlp64_os_ens5` | CLIP | 0.605 | +0.004 [−0.034, +0.049] | 0.686 | +0.025 | 0.125 | noise |
| 40 | `nap_innercv` | CLIP | 0.652 | +0.050 [+0.000, +0.110] | 0.625 | −0.036 | 0.286 | noise |
| 41 | `noise_centrality` | CLIP | 0.601 | −0.000 [−0.011, +0.007] | 0.706 | +0.045 | 0.118 | noise |
| 42 | `noise_conflearn` | CLIP | 0.587 | −0.014 [−0.132, +0.065] | 0.421 | −0.240 | 0.067 | noise |
| 43 | `xclip_video` | CLIP | 0.507 | −0.095 [−0.231, +0.051] | 0.464 | −0.197 | 0.000 | noise |
| 44 | `xclip_late_clip` | CLIP | 0.617 | +0.016 [−0.049, +0.089] | 0.589 | −0.072 | 0.111 | noise |
| 45 | `flip_trainaug` | CLIP | 0.650 | +0.049 [−0.010, +0.105] | 0.662 | +0.001 | 0.316 | noise |
| 46 | `flip_tta` | CLIP | 0.642 | +0.040 [−0.006, +0.106] | 0.655 | −0.007 | 0.222 | noise |
| 47 | `flip_sym` | CLIP | 0.592 | −0.010 [−0.060, +0.041] | 0.608 | −0.054 | 0.118 | noise |
| 48 | `siglip_mean` | CLIP | 0.633 | +0.032 [−0.144, +0.176] | 0.587 | −0.074 | 0.345 | noise |
| 49 | **`siglip_late_clip`** | CLIP | **0.732** | **+0.130 [+0.008, +0.211]** | **0.677** | **+0.016** | 0.500 | **win** |
| 50 | `handcrop_concat` | CLIP+SigLIP | 0.639 | −0.092 [−0.175, +0.012] | 0.478 | −0.200 | 0.167 | noise |
| 51 | `handcrop_late` | CLIP+SigLIP | 0.656 | −0.075 [−0.155, +0.042] | 0.532 | −0.145 | 0.105 | noise |
