# Autoresearch log

## Iteration 0 — baseline
baseline_clip.py: CV macro F1 0.6014, val 0.6614 (matches 5090 report). Champion set.

## Iteration 1
### A. Few-shot classifiers (CLIP features)
Problem: hand_count CV F1 0.105; LR at C=10 may overfit rare classes.
Hypothesis: shrinkage / prototype classifiers handle 8-window classes better.
- fewshot_ncm (cosine nearest class mean): CV 0.476, Δ -0.126 [-0.294, -0.006], val Δ -0.431 → **loss**
- fewshot_lda (Ledoit-Wolf LDA): CV 0.566, Δ -0.036 [-0.179, 0.047], val Δ -0.036 → noise (hand 0.0)
- fewshot_lr_innercv (PCA×C by inner GroupKFold): CV 0.582, Δ -0.020 [-0.066, 0.019], val Δ -0.036 → noise
Interpretation: classifier isn't the bottleneck; hand_count stays 0–0.17 everywhere → features don't separate hand from machine. Rejected.

### B. Encoder fusion (CLIP + VideoMAEv2 [+v1])
Problem: on frozen val, v2 wins safe/laptop and CLIP wins hand — looks complementary.
Hypothesis: fusing encoders beats CLIP alone.
- fuse_concat_clip_v2 (per-encoder scaler, concat, LR C=10): CV 0.562, Δ -0.039 [-0.129, 0.040], val Δ -0.043 → noise
- fuse_late_clip_v2 (avg probs): CV 0.601, Δ -0.000 [-0.073, 0.063], val Δ +0.007 → noise
- fuse_late_innercv (encoder set + C by inner CV): CV 0.610, Δ +0.009 [-0.052, 0.065], val Δ -0.066 → noise
Interpretation: val-set complementarity doesn't survive CV; hand_count unchanged (0.11–0.15). Rejected.

### C. Temporal CLIP features (per-frame CLIP from 5090; sanity cos ≥ 0.9999999 vs saved mean)
Problem: window-mean discards motion; hand_count vs machine is a motion difference.
Hypothesis: temporal statistics of per-frame embeddings improve macro F1, especially hand_count.
- temporal_meanstd: CV 0.552, Δ -0.049 [-0.105, 0.030], val Δ -0.167 → noise
- temporal_meandiff (mean + mean|Δframe|): CV 0.615, Δ +0.014 [-0.061, 0.108], val Δ -0.049 → noise; **hand_count CV F1 0.105 → 0.462**
- temporal_halves: CV 0.588, Δ -0.014 [-0.150, 0.073], val Δ -0.083 → noise
Interpretation: motion carries a real hand_count signal (only 8 windows/2 groups, so not confirmable), but +768 dims at fixed C overfits and costs other classes. Lead for iteration 2: compact motion.

**Iteration 1 result: no win (1/3 toward stop rule). Champion unchanged.**

## Iteration 2
### A. Compact motion (constants fixed a priori: PCA k=16, W=3, C=10)
Problem: temporal_meandiff lifted hand_count but +768 dims overfit.
Hypothesis: a compact motion signal appended to CLIP mean keeps the gain without hurting other classes.
- motion_pca16 (CLIP mean + train-fit PCA16 of mean|Δframe|): CV 0.631, Δ +0.030 [0.000, 0.064], p≤0 0.13, val Δ +0.026 → noise (**borderline**, first candidate positive on both CV and val)
- motion_scalars (CLIP mean + 6 motion stats ×3): CV 0.623, Δ +0.022 [-0.056, 0.112], val Δ +0.053 → noise; hand 0.364
- motion_twostage (binary motion head re-decides hand/machine): CV 0.549, Δ -0.053 [-0.104, -0.026], val Δ -0.074 → **loss**
Interpretation: motion helps only in combination with CLIP appearance; on its own it can't separate hand from machine. pca16 sits on the win threshold but is not confirmed.

### B. Sub-window MIL (per-frame / 4-frame chunk CLIP samples, mean log-prob aggregation)
Problem: rare classes have few training windows.
Hypothesis: sub-window samples (4–16×) + probability aggregation improve macro F1.
- mil_frame_lr (16×, C=10): CV 0.613, Δ +0.012 [-0.061, 0.082], val Δ -0.097 → noise
- mil_chunk4_lr (4×, C=10): CV 0.608, Δ +0.007 [-0.014, 0.054], val Δ -0.107 → noise
- mil_innercv (chunk×C by inner GroupKFold): CV 0.606, Δ +0.005 [-0.034, 0.057], val Δ -0.032 → noise
Interpretation: frames within a window are near-duplicates; effective sample size unchanged; hand_count 0.09–0.12. Rejected.

### C. VideoMAEv2 temporal tokens (L4; sanity cos ≥ 0.99999988)
Problem: v2 window mean pools away its 8 temporal positions.
Hypothesis: v2 temporal structure (mean + position diff) encodes motion better than independent CLIP frames.
- v2temporal_meandiff (v2 alone): CV 0.387, Δ -0.215 [-0.366, -0.045], val Δ -0.149 → **loss**
- v2temporal_clip_pcadiff (CLIP + PCA16 v2 diff): CV 0.601, Δ 0.000 [0, 0], val Δ -0.010 → noise (no effect)
- v2temporal_clip_pcadiff_innercv: CV 0.603, Δ +0.002 [-0.017, 0.018], val Δ -0.061 → noise
Interpretation: v2 temporal diffs carry no hand signal (0.0–0.13). CLIP per-frame diffs remain the only motion lead. Rejected.

**Iteration 2 result: no win (2/3 toward stop rule).** ~21 candidates evaluated so far; with that many looks, one borderline CI (motion_pca16, lower bound 0.000) is consistent with chance. Next iteration must be pre-registered and test genuinely new signals.

## Iteration 3
### A. Pixel motion (38-d: 4×4 frame-diff grid, 4×4 Farneback grid, 6 desk stats; constants pre-registered)
Problem: embedding models may be invariant to small hand motions.
Hypothesis: low-level pixel motion appended to CLIP mean improves macro F1, esp. hand_count.
- pixmotion_w3 (W=3, C=10): CV 0.499, Δ -0.102 [-0.216, 0.039], val Δ -0.125 → noise (negative)
- pixmotion_innercv (W,C by inner CV): CV 0.477, Δ -0.124 [-0.253, 0.012], val Δ -0.036 → noise (negative)
Interpretation: desk motion energy is nearly identical for hand_count and machine (flow 0.056 vs 0.050) — running machine and moving notes look alike in magnitude. The CLIP diff signal is about *what* changes, not how much. Refuted.

### B. Pre-registered motion blend + robustness check
Hypothesis (one evaluation, spec fixed in docstring): CLIP mean + PCA16 diff + 6 motion scalars beats champion.
- motion_blend: CV 0.640, Δ +0.038 [-0.046, 0.141], p≤0 0.28, val Δ +0.010 → noise
Robustness (6 bootstrap seeds × {leave-one-group-out, leave-one-video-out} × {group, video} resampling) for pca16/scalars/blend: CI lower bound > 0 in **0 of all configurations**. pca16's lower bound is a floor at exactly 0 (gain rests on 1–2 groups). Under leave-one-video-out (paired scene visible in train) every model reaches hand F1 0.55–0.59 and the motion advantage vanishes (±0.004).
Interpretation: CLIP motion is not a confirmed effect. The leave-one-video-out jump shows hand_count *is* recognisable once a related scene is in training → the bottleneck is scene diversity of hand_count examples (2 groups), not features. Stop tuning motion.
Orchestrator check (no ledger): champion under leave-one-group-out vs leave-one-video-out: macro F1 0.601 → 0.702, **hand_count 0.105 → 0.552**. The plain CLIP champion already recognises hand counting when the paired scene is in training — confirms the hand_count failure is scene generalisation with 2 groups, not missing features.

### C. DINOv2-base per-frame (5090; rev f9e44c81, safetensors d73036b5…)
Problem: CLIP may lose fine hand/object detail.
Hypothesis: DINOv2 self-supervised features separate actions better, alone or fused.
- dino_cls_mean: CV 0.548, Δ -0.054 [-0.240, 0.051], val Δ -0.344 → noise (val collapses to 0.32)
- dino_late_clip: CV 0.571, Δ -0.030 [-0.125, 0.029], val Δ -0.028 → noise
- dino_cls_patchdiff_pca16: CV 0.531, Δ -0.071 [-0.267, 0.033], val Δ -0.344 → noise
Interpretation: CLIP's semantic (caption-trained) features matter more than fine spatial detail. Rejected.

**Iteration 3 result: no win (3/3 — stop rule reached). User said "continue" → proceeding to iteration 4 targeting scene generalisation.**

## Iteration 4
### A. Scene-invariant head (CPU; K=8, C=10 pre-registered)
Problem: champion hand F1 0.552 when paired scene in train vs 0.105 on unseen scenes → head learns scene identity.
Hypothesis: equalising (video,class) weights and/or projecting out within-class between-video directions (NAP, fit on train only) improves cross-scene F1.
- sceneinv_weights: CV 0.599, Δ -0.002 [-0.005, 0.0], val Δ +0.053 → noise
- sceneinv_nap: CV **0.698**, Δ +0.097 [0.000, 0.163], p≤0 0.061, val Δ -0.034 → noise; LOGO hand 0.105 → 0.588, LOGO/LOVO gap 0.101 → 0.040
- sceneinv_combined: CV 0.682, Δ +0.080 [0.000, 0.140], val Δ -0.034 → noise
Code reviewed by orchestrator: NAP basis fit on training fold only; no leakage.
Orchestrator K-sensitivity (diagnostic only, not selection): K 0/1/2/4/8/12/16/24/32 → CV 0.601/0.607/0.607/0.632/**0.698**/0.633/0.623/0.621/0.617; hand 0.105…0.200/**0.588**/0.222/0.100…; val 0.661/0.654/0.715/0.690/0.628/0.571/0.571/0.615/0.658.
Interpretation: a modest CV gain for K≥4 (+0.02–0.03) is consistent, but the big K=8 spike (hand 0.588) is isolated — fragile, likely partly luck on 3 groups. NAP direction is promising, not confirmed. Do not retune K; test K-ensemble / inner-CV K as new pre-registered hypotheses.

**Correction:** hand_count = 8 windows, 5 videos, **3** scenario groups (train: X-2/X-3 group 2 windows, X-8 1 window; val: X-20/X-21 group 5 windows). Any LOGO fold trains on at most 2 hand groups; when the X-21 group is held out, only 3 train windows remain. Earlier "2 groups" referred to the train split only.

### B. Per-video scene embedding (5090; whole-video CLIP mean at 2 fps, label-free; window·scene cosine 0.975 mean)
Problem: window embeddings are dominated by scene identity.
Hypothesis: subtracting the per-video scene mean removes scene identity and improves cross-scene F1.
- scene_sub L2(clip − scene): CV 0.592, Δ -0.010 [-0.188, 0.119], val Δ **+0.114** (val 0.775) → noise
- scene_sub_std (÷ per-video std): CV 0.558, Δ -0.044 [-0.220, 0.094], val Δ -0.063 → noise
- scene_concat [clip, L2(clip − scene)]: CV 0.649, Δ +0.047 [-0.090, 0.143], val Δ **+0.112** (val 0.773) → noise; LOGO hand 0.435, LOGO/LOVO gap 0.101 → **0.014**
Interpretation: subtraction alone raises seen-scene scores but not unseen; concatenation nearly closes the scene gap and has the largest val gain of the loop, but the CV CI is wide. Lead.

### C. CLIP text-prompt similarity (24 fixed prompts, 4/class; text on L4, sims local)
Problem: a linear head on 1–2 hand scenes learns scene identity; text priors are scene-independent.
Hypothesis: learned head on image–text similarity scores generalises across scenes.
- text_sim_lr (24-d sims only, C=1): CV 0.542, Δ -0.060 [-0.189, 0.031], val Δ -0.176 → noise; LOGO hand **0.500**, hand gap 0.446 → 0.060, but machine 0.247
- text_clip_concat (CLIP + 3×sims): CV 0.634, Δ +0.033 [-0.015, 0.085], val Δ -0.179 → noise
- text_late_fusion: CV 0.626, Δ +0.024 [-0.035, 0.081], val Δ -0.052 → noise
Zero-shot argmax reference: macro 0.09.
Interpretation: text features are scene-invariant for hand (unseen ≈ seen) but confuse hand with machine; adding image features brings scene dependence back. Not confirmed.

**Iteration 4 pattern:** three different scene-invariance approaches (NAP, scene_concat, text sims) each shrink the LOGO/LOVO gap → strong mechanistic support that scene identity is the failure mode. None passes the noise check alone; CV and val disagree in sign between NAP (CV+, val−) and scene_sub (CV−, val+).

## Iteration 5 (scale-up: ~11 parallel agents)
### NAP K-ensemble {4,8,12,16} (pre-registered, one evaluation)
Hypothesis: averaging NAP over K removes the K=8 luck and keeps a stable gain.
- nap_kensemble: CV 0.641, Δ +0.039 [0.000, 0.075], p≤0 0.039, val Δ -0.034 → noise; hand 0.211
Interpretation: confirms the K=8 hand spike (0.588) was luck. NAP gives a small, consistent but unconfirmed CV gain (+0.04) on a few groups; val still drops. Not adopted.

### NAP K=8 + motion PCA16 (pre-registered, one evaluation)
Hypothesis: scene removal (NAP) and motion act on different causes and add up, with val more stable than NAP alone.
- nap_motion: CV 0.705, Δ +0.104 [**0.004**, 0.170], p≤0 0.02, val Δ **-0.103** → noise (fails val ≥ 0)
Interpretation: per-class CV identical to sceneinv_nap except safe/leave (1–2 windows); hand 0.588 unchanged → the CV gain is NAP's isolated K=8 spike, not new signal. Val worse than either parent. First strictly positive CI lower bound in the loop, but it rests on the known-fragile K=8 and fails val; with ~40 looks, consistent with chance. Refuted.

### Scene-offset transfer augmentation (pre-registered, S=3, C=10, one evaluation)
Hypothesis: synthetic samples z + (μ_v' − μ_v) (train-only class-residual video offsets, label kept) teach the head that scene offsets don't change the action.
- aug_scenetransfer: CV 0.650, Δ +0.049 [-0.010, 0.104], p≤0 0.11, val Δ **+0.010** → noise; hand 0.105 → 0.333, 5/6 classes up, document 0.632 → 0.556
Interpretation: broad modest gain positive on both CV and val, with no tuned subspace size. Not confirmed by the bootstrap.
Orchestrator stability diagnostic (no ledger; S ∈ {1,3,5,10} × 5 RNG seeds, scratch script): CV Δ is positive in **20/20** configs (+0.005…+0.066, typically +0.03), CI lower bound ≤ 0.000 in all; **val Δ negative in 14/20** (down to -0.178) — the pre-registered seed's val +0.010 was luck. hand F1 swings 0.09…0.40 **from the RNG seed alone**. LOVO 0.710–0.735 / hand 0.615–0.640 (champion 0.702 / 0.552), not worse.
Interpretation: a consistent but small CV gain that val doesn't support. Not adopted. Key noise-floor fact: **hand_count F1 differences up to ~0.3 are within seed noise** (8 windows); single-run hand gains of that size in earlier candidates should be read as noise.

### Drop-one-group bagging (pre-registered, one evaluation)
- bag_dropgroup: CV 0.605, Δ +0.004 [0.000, 0.013], val Δ +0.045 → noise; changes 1/160 CV and 2/37 val predictions vs champion
Interpretation: every head still sees nearly all scenes, so the scene bias isn't averaged out; effectively the champion. Rejected.

### Nonlinear heads on CLIP (pre-registered, 2 candidates)
Problem: only linear heads (and LDA/NCM) had been tried; machine/document may be multi-modal across scenes.
Hypothesis: a nonlinear head on standardized CLIP improves macro F1.
- nonlin_rbfsvm_innercv (RBF SVC, balanced, C by inner GroupKFold; C=1 in 13/15 folds): CV 0.482, Δ -0.120 [-0.349, 0.073], val Δ -0.062 → noise (negative; leave 0.364, hand 0.0)
- nonlin_mlp64_os_ens5 (MLP 64, alpha 1e-2, class oversampling, 5-seed prob average): CV 0.605, Δ +0.004 [-0.034, 0.049], val Δ +0.025 → noise; per-class ≈ champion
Interpretation: head capacity is not the bottleneck (consistent with iteration 1A). Rejected.

### NAP with K, C chosen by inner leave-one-group-out (pre-registered, one evaluation)
- nap_innercv: CV 0.652, Δ +0.050 [0.000, 0.110], p≤0 0.13, val Δ -0.036 → noise; hand 0.286
Chosen K over 15 outer folds: {0:1, 2:1, 4:1, 8:11, 16:1}; frozen-val fit chose K=0 (plain LR). Inner CV prefers K=8 only when all rare-class groups (X-3, X-8, X-21, X-19, X-23) are in training; drop any one and K moves.
Interpretation: honest nested selection keeps about half the K=8 gain and the choice is unstable → the K=8 spike is a joint property of a few groups, not a selectable setting. NAP line closed: small, unconfirmed gain.

### Label-noise handling on TRAIN windows (pre-registered, 2 candidates)
Problem: broad annotation intervals mix actions. Hypothesis: down-weighting/filtering likely-mixed train windows helps.
- noise_centrality (weight by distance from interval boundary): CV 0.601, Δ -0.000 [-0.011, 0.007], val Δ +0.045 (2/37 windows) → noise; 2/160 CV predictions change. 17/49 intervals have a single window; the visually flagged mixed windows (idx 20, 179) get the *highest* weight → position is a poor proxy for mixing.
- noise_conflearn (drop bottom-10% given-label OOF probability per class): CV 0.587, Δ -0.014 [-0.132, 0.065], val Δ **-0.240** → noise, harmful; it drops scene-novel windows (X-8 hand, all X-1 machine), not mislabeled ones.
Interpretation: label noise near boundaries isn't the limit; model-confidence filtering deletes exactly the scarce scene diversity. Mixed windows need real boundary review, not a proxy. Rejected.

### Evaluator hardening (orchestrator)
Audit found test-fold `meta` still carried `label`/`seen`/`row`/annotation bounds (only `test["y"]` was hidden). Audit of all candidates: none read test labels (only nap_innercv reads test video names for logging). evaluate.py now strips label-derived meta fields from test folds (`blind()`); re-running sceneinv_nap and noise_centrality gives identical scores → no prior result is affected.

### X-CLIP video–text (L4; microsoft/xclip-base-patch16-zero-shot rev 6746d6a5, safetensors 0057077d…)
Hypothesis: a video–text model beats or complements CLIP.
- xclip_video: CV 0.507, Δ -0.095 [-0.231, 0.051], val Δ -0.197 → noise (negative); hand 0.0
- xclip_late_clip: CV 0.617, Δ +0.016 [-0.049, 0.089], val Δ -0.072 → noise; hand 0.111
Encoder-alone CV (same head): v1 0.375, v2 0.418, X-CLIP 0.507, CLIP-L 0.601. Text supervision lifts a video model above v1/v2 but not to CLIP-L (backbone confound B/16 vs L/14). Rejected.

### Orchestrator finding: unused hand_count annotations in TRAIN videos
The frozen manifest (annotation snapshot b4cb042) uses 5 hand_count intervals. Current docs/annotations.md has more:
- **X-10 13–15.6 s `hand_count`** — train, own scenario group → a **new independent hand scene** (would make train hand groups 3 instead of 2).
- **X-8 10–18 s `machine_unload, hand_count`** — train, compound/mixed row (needs boundary review), up to ~7 windows.
- Test videos (X-4, X-5, X-26, X-27, X-35, X-47) also have hand-counting rows — off-limits.
Every model-side experiment so far tries to generalise hand_count from 3 train windows while one more independent train scene sits unused. Next step (needs a versioned manifest v2; v1 stays frozen): add these train windows, extract CLIP, re-evaluate on the original 160 evaluation windows.

### Horizontal flip (L4; user-suggested; pre-registered, one evaluation each)
Padding is NOT symmetric (50 px left / 51 right); the agent mirrored only the content columns (a naive full flip would add a 1-px shift). Unflipped re-encode on the L4 matches the 5090 clipframes (cos ≥ 0.9999998).
Original-vs-flipped window cosine 0.980 (min 0.960): flips stay inside their own scene cluster (same video+label 0.988; same label, other video 0.909).
- flip_trainaug (train on orig+flipped): CV 0.650, Δ +0.049 [-0.010, 0.105], p≤0 0.085, val Δ +0.001 → noise
- flip_tta (+ average over flipped test window): CV 0.642, Δ +0.040 [-0.006, 0.106], val Δ -0.007 → noise
- flip_sym (flip-symmetrised features): CV 0.592, Δ -0.010 [-0.060, 0.041], val Δ -0.054 → noise
Controls (diagnostic): plain row duplication +0.003 → the train-aug gain is not the doubled row count. CI lower bound -0.006…-0.014 in all 24 seed/protocol/resampling configs (sign-robust, never confirmed). Gain sits in X-21/X-19; the hand gain is 2 windows (X-21 rows 88–89, laptop → hand). Scene gap unchanged (0.101 → 0.098). Champion predicting on mirrored test windows: LOVO 0.702 → 0.656 → part of the seen-scene advantage is memorised left/right layout.
Interpretation: flip helps a little as label-preserving jitter, not as a scene-generalisation fix. Side fact: a 1-px shift moves CLIP-L/14 window embeddings about half as much as a mirror (cos 0.991) → translation jitter is a cheap follow-up.
**Meta-note:** flip_trainaug, aug_scenetransfer, NAP and motion all gain mostly on the same few X-21/X-19 windows → their "consistent +0.03–0.05" is not independent evidence.

### SigLIP so400m-patch14-384 (5090; rev 9fdffc58, safetensors ea2abad2…; same 16 source frames decoded at 384, bit-exact frame check at 224; eager attention because fp32 SDPA was non-repeatable on Blackwell)
Hypothesis: a stronger image–text encoder beats or complements CLIP.
- siglip_mean (SigLIP alone): CV 0.633, Δ +0.032 [-0.144, 0.176], val Δ -0.074 → noise
- **siglip_late_clip** (0.5/0.5 probability average of CLIP head + SigLIP head, both scaler + balanced LR C=10, pre-registered): CV **0.732**, Δ **+0.130 [0.008, 0.211]**, p≤0 0.009, val **0.677**, Δ +0.016 → **WIN**
Confirmation (orchestrator, independent scratch script + agent diagnostics):
(a) deterministic rerun identical; CI lower bound > 0 for 5/5 bootstrap seeds (+0.0075…+0.0105).
(b) neighbouring constants form a smooth hump (all CV Δ positive for w ∈ {0.25…0.75}, C ∈ {1,10,100}); no isolated spike. Not re-selected: pre-registered w=0.5, C=10 kept.
(c) LOVO 0.728 vs champion 0.702 (not worse; hand 0.516 vs 0.552). LOGO/LOVO gap 0.101 → -0.004.
Jackknife: Δ stays positive with any single group removed (min +0.061 without X-21).
Caveats: 11 fixed / 1 broken windows, all in 3 groups (X-21 hand ×5, X-31 machine ×5, X-14 ×1); hand precision only 0.375 (10 false positives); p is uncorrected for ~49 looks; val Δ = +1 window net.
Interpretation: unlike CLIP+v2/DINOv2/X-CLIP, SigLIP is about as strong as CLIP alone but trained on different data, so the two heads' scene-specific errors are partly decorrelated and averaging cancels some of them. The orchestrator's roast of this experiment ("answers a settled question") was wrong about fusion.

**CHAMPION → candidates/siglip_late_clip.py (CV 0.7318, val 0.6773).** Later evaluate.py runs compare against it (ledger `baseline` field records which). Independent confirmation on fresh data (annotations added after the frozen snapshot, train/val videos only) is the next priority.

### Orchestrator finding: "scenario groups" are recording sessions, not independent scenes
Annotation scene notes (`сцена`) show only ~4 camera setups in train/val: side view as X-1 (X-1..3), behind-cashier as X-1 (X-16..19), front view as X-8 (X-8, X-10..15, X-20..25, X-30..33, X-40, X-43..46, chained via "ракурс как в …"), behind-cashier with client as X-34 (X-38, X-39). Test videos share these setups too.
Consequences:
- LOGO holds out a *session* (a paired re-enactment), not a camera/room. E.g., hand_count val X-20/X-21 is the same front setup as train X-8. The LOGO vs LOVO gap is about **session identity** (near-duplicate paired re-enactments), not camera identity; earlier "scene identity" wording should be read that way.
- **Correction:** X-10 13–15.6 hand_count is the same front setup as X-8 ("ракурс как в Х-8") — a new *session*, not a new independent camera scene.
- No split in this dataset (including test) measures generalisation to a new camera/branch. Any deployment claim needs footage from new setups.

## Loop stopped (2026-10-17)
44 candidates, 0 wins; champion stays `baseline_clip.py`. The stop rule was hit at iteration 3 and ignored for two more iterations. The finding holds from several angles: hand_count fails because of **scene diversity** (3 groups, 8 windows), not features or head; hand F1 differences up to ~0.3 are seed noise. No new feature/head candidates until new independent hand_count scenes and reviewed action boundaries exist. Next work moves to the protocol layer: `scripts/protocol_gold.py` (rules on gold `ok` facts).

### Hand-region CLIP via GroundingDINO (5090; resumed after rate limit — extraction had already completed, hashes verified, nothing rerun)
Problem: global motion can't separate hand counting from a running machine; a fixed desk crop hurt CLIP.
Hypothesis: CLIP on detected hand regions ("hands. banknotes.", box 0.3 / text 0.25 fixed a priori, top-2 hand boxes merged, 20% pad, full-frame fallback) improves macro F1, esp. hand_count.
Detection: hands in 90% of frames (leave 28% — cashier absent), banknotes 18%; median crop 1.2% of the frame.
- handcrop_concat (CLIP + crop mean): CV 0.639 vs champion Δ -0.093 [-0.175, 0.012], val Δ -0.200 → noise; vs CLIP baseline Δ +0.038 [-0.064, 0.124]
- handcrop_late (late fusion): CV 0.656 vs champion Δ -0.076 [-0.155, 0.042], val Δ -0.145 → noise; vs CLIP baseline Δ +0.055 [-0.009, 0.117], p 0.057
Diagnostic (no ledger): crop-only CV 0.703 (Δ +0.102 vs CLIP, machine/safe/leave up) but val 0.454 (Δ -0.207); hand_count 0.105 in every variant.
Contact sheet: tight, correct crops in front-camera sessions (X-8, X-11), but in the X-2/X-3 session (2 of 3 train hand windows) the "hands" box covers the client's whole upper body.
Interpretation: hand crops carry signal for machine/safe/leave on CV but lose 0.13–0.21 on val, and hand_count is untouched — inconsistent hand localisation across sessions adds variance rather than removing session identity. Refuted for hand_count; not adopted.

## Iteration 6 — independent confirmation of the champion (fresh windows)
Fresh set confirm-v1: TRAIN/VAL annotation rows not in `row_mapping`, labelled by a type→class table fixed before any model output (review/generated/autoresearch/confirm-v1/). 23 primary rows → **21 windows** (machine 17, leave 2, hand_count 1, document 1; 5 groups). Features via the same CLIP/SigLIP code paths (sanity cos ≥ 0.9999998; 224 decode pixel-exact). Each fresh window predicted by models trained on the 160 frozen windows minus its scenario group; evaluated once.
| protocol | champion (CLIP+SigLIP) | baseline (CLIP) | Δ macro F1 [CI95] | Δ accuracy |
|---|---|---|---|---|
| primary (own group out) | 0.778, 16/21 | 0.767, 17/21 | +0.012 [-0.083, +0.030], p≤0 0.85 | −1 window |
| secondary (all 160) | 0.710, 16/21 | 0.792, 17/21 | −0.082 [-0.249, 0.0] | −1 window |
Pre-registered decision: **"consistent"** (Δ > 0, CI includes 0) — in practice a tie that does **not** reproduce the +0.130 CV gain. The positive Δ is one window (where the baseline's error lands on leave). Champion again over-predicts hand_count on long machine episodes (5 vs 3 windows; X-14 row 4 is a known mixed episode). The machine part of the CV gain (X-31) does not replicate on fresh machine windows (machine F1 0.828 vs 0.867); hand cannot be assessed (1 window).
Status: champion.txt unchanged (it won by the pre-registered rule; the fresh check is not "not confirmed"), but the gain is **unconfirmed** as generalisation.

Annotation-exhaustion check (orchestrator): the 70 excluded TRAIN/VAL rows long enough for a window are mostly client actions, bag/door states, transitions and idle; by their descriptions at most ~15–20 more windows would fit the six classes (mostly machine X-38 r1, leave/cashier-absent rows), with no clear new hand_count. The existing annotations cannot supply a decisive confirmation set.

## Loop summary (stopped after iteration 6)
Protocol: leave-one-scenario-group-out CV (15 groups) + frozen 37-window val + paired group bootstrap; win = CV CI95 lower > 0 and val Δ ≥ 0; wins re-checked (rerun, neighbouring constants, LOVO, jackknife) and then on fresh windows. Test never loaded.
~51 candidates: classifiers (NCM, LDA, inner-CV LR, RBF SVM, MLP), encoder fusion (VideoMAE v1/v2, DINOv2, X-CLIP, SigLIP), motion (CLIP frame diffs, v2 temporal tokens, pixel/optical flow), session invariance (NAP, per-video scene embedding, scene-offset augmentation, text-prompt similarities, bagging), augmentation (sub-window MIL, horizontal flip), label-noise handling, hand crops.
One win: CLIP+SigLIP late fusion, CV 0.601 → 0.732, val 0.661 → 0.677; robust on all in-benchmark checks; **tie on 21 fresh windows → unconfirmed**.
What we learned:
1. Failure mode is session identity: the CLIP baseline gets hand_count 0.552 when the paired session is in training (LOVO) vs 0.105 when not (LOGO).
2. "Scenario groups" are recording sessions on ~4 camera setups; no split (incl. test) measures generalisation to a new camera/branch.
3. hand_count F1 has a ±0.3 noise floor from RNG seeds alone (8 windows, 3 groups); single-run hand gains are noise.
4. Head capacity, motion features, hand crops and label-noise filtering don't help; several session-invariance ideas shrink the LOGO/LOVO gap but gains sit on the same few X-21/X-19 windows and are not independent evidence.
5. The 37-window val set is burned (~50 comparisons); the 160-window benchmark can no longer discriminate reliably; existing annotations are exhausted for the six classes.
Recommendations: (a) collect/label new independent hand_count sessions and footage from new camera setups; (b) review boundaries of mixed intervals (X-8 r3, X-14 r4); (c) keep CLIP as the practical baseline and CLIP+SigLIP as the leading candidate until fresh data confirms it; (d) a final test evaluation only with explicit authorisation — pre-registered, both models, run once.
