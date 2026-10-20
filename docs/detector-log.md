# detector log

verdict truth here counts every annotated event, including `labeled: false` no_id_check (team rule), so Х-35, Х-37, Х-39 count as violations. 42 files (Х-47/Х-48 are copies of Х-4/Х-5).
bench: `uv run python -m app.detector_vlm.bench out/<tag> videos <workers>` on the server, direct calls without the backend, threshold 0.5 (other 0.7).

| run | setup | acc | FP | FN | types hit | latency |
|---|---|---|---|---|---|---|
| v2 | v1 with checklist prompt and rules catalog | 0.524 | 9 | 11 | 8/25 | med 38 s, max 61 s (2 workers) |
| v4 | facts: one bundled yes/no prompt per frame at 1 fps, logprob P(yes), smoothing, rules over the timeline (91087ba) | 0.643 | 13 | 2 | 7/25 | med 21 s, max 160 s (shared GPU) |
| v6 | v4 plus desk/safe zoom questions on frames where the cashier is away (fef4eb5) | 0.690 | 10 | 3 | 9/25 | med 12 s, max 60 s |
| v7 | v6 plus a verify question on candidates | 0.524 | 8 | 12 | 8/25 | med 13 s, max 109 s |
| v9 | f5d7cb8: grounded crops per small-object fact, door and phone dropped, second_person only without client, plus actions b561d5c | 0.810 | 3 | 5 | 15/25 | med 37 s, max 150 s (GPU shared by 5 clients) |
| v9 clean | 8c22e5f alone, 13 videos incl. 10 longest | 12/13 | 0 | 1 | 7/8 | med 35 s, max 58 s (Х-14, 145 s) |
| v9 c5 | d5de086 via backend, concurrency 5, 5 longest | | | | | 39, 88, 133, 177, 223 s, all under 270 |
| v10 | a60faae: eng3 crop facts (cashier_absent, desk_items, safe_open, laptop, two_clients), door-closing protocol after entry, previous_cash across sessions, no_id_check skipped for bag intake, actions 0785810 | 0.905 | 2 | 2 | 23/25 | med 42 s, max 126 s (shared GPU) |
| v10 + 993b406 + 70c96df | replay: safe must stay open to the end (cut 0.9), strict away safe, departure/onset retiming | 0.952 | 0 | 2 (X-1, Х-44) | 23/25 | tc median 1.0 s |
| v1 | Qwen3-VL-32B-AWQ, vLLM, whole video 2 fps 448 px as mp4, free-form violations list with self-reported confidence | 0.50 | 12 | 9 | 7/25 | med 18 s, max 36 s (2 workers) |

notes
- v0 (Qwen3-VL-8B, interleaved images with text timestamps) degenerated: repeated timeline lines, wrong times. dropped after 2 videos.
- v1: the model describes the timeline well but hallucinates door_left_open and workstation_left_unlocked, misses subtle actions (X-1 return, Х-4 cash on desk).
- v6 per-type thresholds: fit on all 42 gives 0.833, leave-one-out 0.667, odd->even 0.524, even->odd 0.429, while a flat 0.5 scores 0.714/0.667 on the same halves. fitted thresholds overfit, so they are not shipped.
- v7: verify killed the second_person hits that had been carrying the no_id_check videos (the client was read as a second person). those videos now rely on actions.py.
- v8/v9: door dropped (the VLM says "open" on every probe), phone dropped (no ground truth, misreads), second_person counts only while no client is at the desk and none arrives within 4 s; small-object facts get one question per request on a grounded crop upscaled to about 1200 visual tokens (fallback: full frame at 2x); leave events are timed at the moment the cashier left.
