# Dataset inspection and first review

Inspected on 2026-10-17, including the pull that added the rules catalogue and ground truth. The provided names are `open.zip`, `open_labels.xlsx`, and `annotations.md`; there are no `.dat` files in this checkout.

## What each source file does

| File | Role |
|---|---|
| `open.zip` | Raw release: 42 MOV clips and one official label workbook. |
| `Кейс_ForteBank.pdf` | Seven-page brief, scoring, output/API contract, constraints and FAQ. |
| `assets/open_labels.xlsx` | Official clip verdict and one timestamp for each positive clip. Sheet `labels`, 42 data rows. Byte-identical to the archived workbook. |
| `assets/team_labels.xlsx` | Same 42 label rows plus human interpretations in column D. Sheet `Sheet1`. Some interpretations differ from the later Markdown review. |
| `docs/annotations.md` | Detailed chronological observations, correct steps, flagged events, tentative interpretations and source references. Each clip is marked verified in the existing source; this inspection does not independently re-certify that status. |
| `docs/model-contract.md` | Proposed model/backend interface, event intervals, evidence frames, confidence, normalized boxes, offline inference, and use of decoded timestamps. |
| `docs/research/README.md` | Regulatory source index, proposed event types, cautions, and citation checklist. |
| `docs/research/01-nbrk-cash-operations.md` | Cash operations, deposits, withdrawals, bag receipt, and what video cannot establish. |
| `docs/research/02-collection-and-security.md` | Collection procedures, physical security, room access, and visibility limitations. |
| `docs/research/03-banknotes-exchange-client.md` | Banknote handling, doubtful/worn notes, exchange operations and customer presence. |
| `docs/research/04-banks-and-violation-catalogue.md` | Earlier candidate catalogue and false-positive risks; some names differ from the newer catalogue. |
| `docs/research/05-bank-internal-rules.md` | Search for missing legal support, including non-Kazakhstan analogies that should not be passed off as Kazakhstan law. |
| `data/rules.json` | 29 structured candidate types, document references, direct/partial/internal basis, cautions and dataset links. |
| `data/ground_truth.json` | 42 derived clip records with flagged intervals, key times, primary-event guesses, correct-step sets, pairs and duplicates. Contains 19 event types. |
| `scripts/build_ground_truth.py` | Converts Markdown into that JSON. Groups rows by type, spans earliest to latest time, heuristically chooses a primary event, extracts pair references, and copies X-5 labels to X-48. |
| `README.md` | Originally just a project title; now includes local viewer instructions. |
| `.gitignore` | Excludes local video assets, generated review assets and Python caches. |
| `.git/` | Version-control metadata. Not dataset content. |

The requested `.agent/rules` and `.agent/workflows` paths are absent.

## Verified inventory

- 42 files, 40 unique byte contents. SHA-256 confirms X-4 = X-47 and X-5 = X-48.
- Official labels: 22 positive and 20 no-violation. Unique contents: 21 positive and 19 no-violation.
- Total runtime: 24.07 minutes; unique runtime: 21.66 minutes.
- Every clip is 464 × 848, H.264, with no audio stream in the inspected release. Lengths range from about 11.6 to 142.5 seconds.
- Frame intervals include approximately 33.3 and 66.7 milliseconds. Use presentation timestamps; frame index / 30 can drift.
- Names mix Latin `X` and Cyrillic `Х`. The viewer uses ASCII identifiers while retaining original filenames and source provenance. It does not rename originals.
- The viewer contains 1,060 extracted frames, generally every two seconds, plus event starts and denser sampling around official timestamps. These frames are for review, not an unbiased evaluation sample.

Visual inspection covered sampled chronological contact sheets of every unique clip, with closer examination of selected event frames. This is not a second frame-by-frame legal adjudication. Small objects, display contents, seals, and short hand movements still require full-motion review at source resolution.

## Clip-by-clip review guide

Times in the second column are official labels. Descriptions summarize the existing manual interpretation, supported where possible by sampled frames; they are not new legal findings. “No violation” is the supplied clip label.

| Clip | Official time | What to inspect / useful comparison |
|---|---:|---|
| X-1 | 22 s | Direct handover of returned notes around 22.3–23.3 s. Why they are returned is not observable. Compare cash handling with X-3. |
| X-2 | 10 s | Cash acceptance without a visible preceding document step. Compare X-3 at 6.8–11.6 s. Absence needs coverage of the preceding interaction. |
| X-3 | No violation | Document handover and inspection, then cash and counting. |
| X-4 | 57 s | First customer's cash remains when the next arrives. Compare X-5 at 45.8–52.5 s. Also inspect 68 s via X-47. |
| X-5 | No violation | Cash stored before the next customer's service. |
| X-8 | No violation | Cashier counts alone, stores cash and writes in a journal. Important counterexample to “no customer means violation.” |
| X-10 | No violation | Cash remains on the table while the cashier is working. Important counterexample to “cash on table means violation.” |
| X-11 | 18 s | Departure with laptop open; inspect display and preceding actions. Compare X-12. Brightness changes when lights go out. |
| X-12 | No violation | Laptop lid closed before departure. Closure is visible; security lock state is not automatically established. |
| X-13 | 47 s | Bundles move from bag/table to safe without a visible counting-machine step. Compare X-14. Legal basis still unresolved in catalogue. |
| X-14 | No violation | Similar bag handling, with a long counting phase at 48–112 s before storage. |
| X-15 | 15 s | A green bag remains as the cashier leaves. Its contents are not visible. |
| X-16 | No violation | Document interaction before bag handover and inspection. |
| X-17 | 7 s | Bag accepted before visible document verification; second person leaves early. Compare X-16. |
| X-18 | 20 s | Interaction with second customer's document while first is still present. Markdown contradicts the spreadsheet's claim about the first ID check. Compare X-19. |
| X-19 | No violation | First interaction finishes before the second starts. |
| X-20 | 5 s | Safe door remains open during work at the table. Compare X-21. |
| X-21 | No violation | Safe closed before work continues. |
| X-22 | 14 s | Alleged key left in safe after departure. Small-object detail needs close playback. Compare X-23. |
| X-23 | No violation | Key-removal gesture before departure; inspect at original resolution. |
| X-24 | 8 s | Journal/stamps remain after departure. Compare X-25. |
| X-25 | No violation | Objects moved to storage before departure. |
| X-26 | 9 s | Cash remains on table as cashier leaves. Compare X-27. |
| X-27 | No violation | Cash moved into safe before departure. |
| X-28 | 7 s | Workstation left; distinguish screen appearance from actual lock state. Compare X-29. |
| X-29 | No violation | Annotated screen-lock action before departure; requires screen-specific evidence. |
| X-30 | 13 s | Open entrance followed by second person's entry. Presence alone does not establish authorization. Compare X-31. |
| X-31 | No violation | Door closed after entry. |
| X-32 | 5 s | Door left open; second person need not enter for this label. Compare X-31. |
| X-33 | 6 s | Open door, later approach/entry of another person. Inspect boundary crossing rather than assuming role. |
| X-34 | 13 s | Customer leaves before the counting-machine phase. Compare X-35 and X-37. |
| X-35 | No violation | Counting while customer remains present. |
| X-37 | No violation | Shorter counting-with-customer example; clip boundaries matter. |
| X-38 | 33 s | Previous cash remains across the customer transition. Compare X-39. |
| X-39 | No violation | Previous cash stored before next interaction. |
| X-40 | 17 s | Bag and coat brought to and placed on the working desk. The suggested X-43 pair is speculative and is not offered as a matched pair in the viewer. |
| X-43 | No violation | Separated notes receive additional checking before combination. Compare X-44; detector interaction needs close review. |
| X-44 | 16 s | Notes from separate machine pockets are combined without the annotated extra check. Compare X-43 and X-45. Pocket semantics are an assumption to verify. |
| X-45 | No violation | Additional detector checks before notes are combined/sorted. |
| X-46 | 11 s | Cup taken and used at desk. Action is visible; legal interpretation needs separate support. |
| X-47 | 68 s | Same bytes as X-4, but label points to mixing the two customers' cash. Preserve both event annotations for this content. |
| X-48 | No violation | Same bytes as X-5. Not an independent example. |

## Label issues to resolve before training

1. **Official label, visual observation, and legal conclusion are separate fields.** In X-1, handover is visible; suspicious/counterfeit status and obligation to create an act are not established. In X-15, a bag is visible; its contents are not. In X-11/X-28, an open illuminated laptop does not prove an unlocked session.
2. **The brief and research have an unresolved mismatch.** FAQ 32 says the dataset contains no violations supported only by unpublished internal rules. The catalogue still has `internal` bases, including `stored_without_count` and `serving_two_clients`. This means interpretation or sourcing needs further work, not that a missing legal source should be invented.
3. **A timestamp is incomplete supervision.** X-4/X-47 prove that one content can have multiple relevant events. The annotation notes also intentionally omit some candidate events when another event matches the official label. Unannotated time must not automatically become negative training data.
4. **Do not split copies or nearby frames across train/test.** Group by content hash at minimum; preferably keep related staged scenarios and their contrasting takes together for the harder evaluation. There are only 40 independent files, not thousands of independent frame examples.
5. **Derived `ok_steps` are not a sequence.** The current builder alphabetically sorts a set and leaves comma-separated names combined. Use the chronological Markdown rows for temporal learning. The viewer preserves their order and intervals.
6. **Derived pairs need review.** X-37 has an empty pair entry. X-40 → X-43 came from a tentative remark, yet the clips show different tasks. The viewer uses an explicit comparison list and excludes that speculative pairing.
7. **Grouping by type can hide gaps.** The current ground-truth builder merges all rows of one type into an earliest-to-latest interval. Treat that as an envelope, not proof that every frame inside it is a violation. Primary-event selection is also a heuristic, not a human event label.
8. **Some type names overstate their visible evidence.** Preserve observations separately from interpretation and uncertainty, even when the source section says `verified`.

## ML framing to test next

The task is **temporal event detection with contextual constraints and unknown-event handling**. A clip-level binary classifier is useful as a measured baseline, but its output alone cannot supply multiple event types, times, evidence frames or grounded explanations required by the brief.

Start with pretrained local visual/video representations and compare a small binary head against an event/state approach. Forty unique clips do not justify training a video backbone from scratch. This is a proposed experiment, not a claim of measured accuracy. VideoMAE is one published example of learning reusable video representations; no particular checkpoint, licence, hardware requirement or inference budget has been selected or validated here. [VideoMAE paper](https://arxiv.org/abs/2203.12602).

Represent visible facts such as `cash_on_desk`, `cashier_present`, `customer_present`, `safe_open`, `cash_moved_to_safe`, `document_presented`, and `counting`. Attach timestamps and visibility/confidence to each fact. Preserve `unknown` for occluded or unreadable states. Track objects/roles only within the clip; do not identify faces.

Then test contextual relations instead of one exact memorized action order:

- Cash visible on the desk AND cashier departs before storage → candidate unattended-cash event.
- New customer starts AND previous customer's cash still remains → candidate separation/storage event.
- Customer transaction counting occurs AFTER that customer has left → candidate customer-presence event.
- A mandatory earlier action is missing only when the relevant interval was actually observed and a meaningful deadline/transition has occurred.

These are candidate visual event formulations; their legal applicability still comes from independently verified rules. Reordering irrelevant actions, pausing, changing walking paths, or moving a chair should not by itself change the verdict. Cashier-alone counting in X-8/X-10 is a key contextual counterexample.

For out-of-distribution behavior, distinguish:

| Situation | Appropriate handling |
|---|---|
| Known event, new order/actor/view | Test whether the same visible state relation is recovered. |
| New violation made from known visible facts | A relevant existing constraint may still apply. Verify evidence and applicability. |
| Unseen action or unfamiliar scene | Novelty can prioritize review, but is not itself proof of a violation. |
| Occluded step, cropped beginning, unreadable display | Mark internal state unknown; absence of observation is not observation of absence. |

Keep novelty/visibility separate from violation confidence. An unknown score should not be converted automatically into “violation,” nor does high classifier confidence guarantee familiar input. Open-set recognition explicitly studies rejecting unknown inputs; it does not provide a bank-compliance guarantee. [Bendale and Boult, CVPR 2016](https://openaccess.thecvf.com/content_cvpr_2016/html/Bendale_Towards_Open_Set_CVPR_2016_paper.html).

Internally allow uncertain/review-required states. The submitted API still has the brief's two verdicts; define and evaluate that mapping explicitly. Do not emit an unsupported third verdict or equate “no supported violation detected” with “proved compliant.” The backend contract's empty event list deserves the same distinction.

## Evaluation that answers the OOD question

- First evaluate with gold visual facts. If temporal rules fail with correct facts, fix the formulation before tuning vision.
- Compare binary baseline and event/state approach on grouped held-out clips. Keep calibration and threshold selection inside training/validation groups.
- Hold out entire event families to test unknown types. Exclude their examples, descriptions derived from held-out videos, and nearest-neighbour entries from training/prompt demonstrations. Document which general regulatory rules remain available.
- Hold out scenarios and verified camera setups separately. A true new-angle recording is necessary; crops and flips are only stress tests, not evidence of cross-camera generalization.
- Include normal-but-unusual sequences and omissions hidden by clip boundaries. Measure their false positives separately.
- Report clip verdict precision/recall, event recall and false alarms, timestamp error, evidence-frame relevance, type correctness, and unknown-event behavior separately. Report raw counts with such a small sample.
- Benchmark offline latency on the actual target machine, including decoding and evidence extraction, against the five-minute total limit. No latency result has been measured here.

The next review should prioritize X-2/X-3, X-4/X-5/X-47, X-26/X-27, then X-43/X-44/X-45. Together they cover a missing step, cross-customer memory, a state at departure, and a subtle multi-step action.

## Added review files

`scripts/build_review.py` builds the manifest, thumbnails, contact sheets and browser video copies. `scripts/serve_review.py` serves them on loopback with byte-range support for seeking. `review/index.html`, `review/app.js`, and `review/style.css` provide the three views. Everything in `review/generated/` is derived and ignored by Git. No model was trained and no original labels were edited.
