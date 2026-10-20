# Baseline snapshot — 17 October 2026

Review these files without model downloads or access to the GPU host:

- [Original action results](action-pilot/results.json), [predictions](action-pilot/scores.csv), [confusion matrices](action-pilot/confusion.png).
- [Train-only normalization results](action-normalization/results.json), [validation predictions](action-normalization/scores.csv), [C-grid plot](action-normalization/validation.png).
- [Prepared action inputs metadata](action-pilot/prepared.json), [inference provenance](action-pilot/inference.json), [head fitting results](action-pilot/fit.json).
- [Review manifest used for these experiments](manifest.json). This includes the original annotation snapshot, PTS references and source hashes, but no image/video bytes.
- [File checksums](sha256.json).

The source annotation version used for these results was based on repository commit `b4cb042`. Later main-branch annotation/rule changes are preserved in the PR. Regenerating the viewer from current annotations is a new input version and need not reproduce these historical metrics. Action labels use explicit row indices, so re-audit their meaning whenever annotations change.

For the historical action experiment, first build the review assets from the separately supplied `open.zip`, then copy this snapshot's `manifest.json` to `review/generated/manifest.json` **before** running action `check` and `prepare`. Preserve existing generated experiments elsewhere before replacing their manifest. Follow the root README for extraction, inference, fitting and normalization. The snapshot's prepared metadata records the authoritative sampled windows and source timestamps; check them against the new preparation.

Videos, extracted frames, model weights, embeddings and trained heads are intentionally not committed. Recomputing features requires the source archive, exact cached checkpoints and the documented runtime. JSON/CSV reports suffice to review the recorded metrics but are not a standalone inference package. HTML viewers need the generated local assets.

Normalization results contain train/validation only. The older action report includes the original unstandardized test evaluation. The test footage had already been inspected in earlier exploratory work; it is not a pristine external benchmark. Windows overlap and are not independent samples.

Committed CSV copies use LF line endings; numeric values are unchanged. Checksums describe the committed snapshot bytes.

## Added in the PR update (validation only)

- [Fixed hands/workspace crop](action-crop/results.json), [predictions](action-crop/scores.csv), [plot](action-crop/validation.png), [provenance](action-crop/inference.json).
- [VideoMAEv2 base, K710 distilled](action-videomaev2/results.json), [predictions](action-videomaev2/scores.csv), [plot](action-videomaev2/validation.png), [provenance](action-videomaev2/inference.json).
- [RTX 5090 rerun of all three encoders](action-5090/results.json), [predictions](action-5090/scores.csv), [plot](action-5090/validation.png), [provenance](action-5090/inference.json). All 37 validation predictions per encoder match the earlier L4 run.
- [Hand-count confusion diagnosis](action-diagnosis/results.json) and its [preview](action-diagnosis/preview.png). Visual notes are provisional and change no labels.

These experiments use the same 123 train / 37 validation windows as the normalization snapshot and never read test windows. The autoresearch loop that followed is summarised in [docs/autoresearch/SUMMARY.md](../../autoresearch/SUMMARY.md); its own lightweight results are in [docs/autoresearch/results/](../../autoresearch/results/).
