# Baseline and next experiments

This is a research baseline for teammates to inspect and extend, not a validated violation detector. The completed experiments and their failure cases are documented in [the experiment log](embedding-pilot.md). Machine-readable results and plots are in [the dated snapshot](baseline-results/2026-10-17/README.md).

## Main idea

Video observations → visible actions and object states → memory of facts → protocol-condition checks with timestamped evidence.

Detect cash and cashier presence separately from action recognition. Track what was observed, its confidence and when it was last seen. A flexible protocol can allow several action orders: accepting documents after a customer places money on the counter is not automatically the same as leaving accepted cash unattended. Legal interpretation and transaction context must come from the reviewed rules, not embedding distance alone.

Represent unobserved or occluded states as unknown. Failure to recognize an action does not prove it was omitted. Check obligations at the relevant deadline, and retain evidence of earlier violations even if the final state becomes normal. The existing two-video memory pilot implements only a limited cash/presence hypothesis, not this complete protocol engine.

## What the baseline establishes

Frozen VideoMAE v1 and CLIP feed balanced six-class logistic heads. Whole videos, paired scenarios and exact duplicates stay in the same partition. There are 123 train, 37 validation and 69 test windows. Standardization is fitted on train only; the encoder is unchanged.

| Validation macro F1 | Original L2 features | Train-standardized features |
|---|---:|---:|
| VideoMAE v1 | 0.2824 | 0.4031 |
| CLIP | 0.5682 | 0.6614 |

C is selected on validation from the same grid. At the original C, standardized VideoMAE scores 0.3372 and CLIP 0.6614. Both standardized heads reach train F1 = 1.0. Normalized models have **not** been evaluated on test. Earlier unstandardized test results are separate and must not be confused with validation selection scores.

Memory did not improve the X-26/X-27 detection pilot because its confirmation condition never activated. This is not evidence that memory is unnecessary. Selected action windows are a closed-set task; unmapped footage is not evaluated as background or unknown. No result establishes OOD generalization.

## Hypotheses, not conclusions

Small hands, notes and documents lose detail in the full-frame 224-pixel input. Features may capture recording-specific context. Sparse, correlated windows and coarse labels can also explain the train/validation gap: hand counting has only three training windows. Normalization helps optimization but does not settle these alternatives.

## Next controlled experiments

1. Compare full-frame inputs with a fixed cashier/hands/work-surface crop, holding labels, splits and normalization constant. Specify crop selection without using validation labels; record camera dependence. Inspect class-level gains and failures.
2. Compare a suitable frozen VideoMAE v2 checkpoint with the same linear probe and data. Pin checkpoint, preprocessing, licence, compute and feature dimensions. Do not attribute checkpoint-specific gains to architecture alone.
3. Refine action intervals and add independent training recordings, including background and unknown cases. Preserve the existing benchmark version; publish new annotation versions separately.
4. Once continuous features and reliable boundaries exist, test temporal localization. Report segment metrics and boundary errors, not just window F1. For an online version, explicitly prohibit future-frame access and measure latency.
5. Evaluate protocol memory with identical observations in both variants, including occlusion, reordering, delayed actions and repeated actions. Calibrate thresholds on train/validation and reserve independent recordings for final evaluation.

## Repositories suggested by the team

- [VideoMAE v2](https://github.com/OpenGVLab/VideoMAEv2): video representation models, pretrained/distilled checkpoints and temporal-action feature extraction. Our completed run uses VideoMAE **v1**, not v2.
- [ActionFormer fork supplied by the team](https://github.com/myccver/actionformer), [original implementation](https://github.com/happyharrycn/actionformer_release): temporal action localization over feature sequences, predicting categories and start/end boundaries. It does not itself decide legal compliance or guarantee causal online inference. Compare the fork with upstream before adoption.
- [MMAction2](https://github.com/open-mmlab/mmaction2): training/evaluation toolbox for video understanding. Consider it if it reduces experiment maintenance; it is not an additional model in this baseline.

These projects are proposed follow-ups. None has been integrated or benchmarked here yet.
