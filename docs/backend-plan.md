# Backend plan

Source of truth: `Кейс_ForteBank.pdf` (sections 4, 7, 10 and FAQ), `docs/model-contract.md`, `data/rules.json`, `data/ground_truth.json`.

## Hard requirements from the case

| requirement | source |
|---|---|
| `POST /api/v1/analyze`, multipart field `file`, `.mov` about 29 MB, returns `job_id` | §10 |
| `GET /api/v1/jobs/{job_id}`, `status` is `processing`, `done` or `failed`, result in the §10 schema | §10 |
| `X-API-Key` header, requests without the right key are rejected | §10 |
| result within 5 minutes of upload, otherwise the video counts as unprocessed | §10, FAQ 14 |
| `verdict` is `violation` or `no_violation`; `violations` empty when clean | §10 |
| 1 to 3 JPEG evidence frames per violation, base64 | §10 |
| public URL, web UI without login | FAQ 23, 27 |
| all models local, no cloud AI calls, no internet needed after weights are loaded | §7 |
| no face recognition, no audio | FAQ 28, 29 |

## Stack

FastAPI, one process, SQLite job table, files on local disk, PyAV for frames, Pillow for JPEG. One GPU worker thread consumes jobs in order. The detector runs in the same container.

Why one worker: one GPU, and a second model copy does not fit. Throughput comes from a faster detector, not from parallel jobs.

## Layout

```
app/
  main.py        routes, auth, startup recovery
  jobs.py        SQLite job store and the worker loop
  pipeline.py    detector call → verdict → rules join → evidence frames → §10 json
  frames.py      decode by pts, JPEG, base64
  detector_stub.py
  static/        frontend build, served by FastAPI
scripts/eval.py  runs data/ground_truth.json through the live API
tests/
Dockerfile, compose.yaml
```

## API

| method | path | auth | notes |
|---|---|---|---|
| POST | `/api/v1/analyze` | key | validate extension and size (limit 2048 MB), probe with PyAV, reject undecodable files with 400. store, enqueue, return `{"job_id", "status": "processing"}` |
| GET | `/api/v1/jobs/{id}` | key | §10 schema. 404 for unknown id. on `failed` add `error` |
| POST | `/app/analyze` | none | same handler for the UI |
| GET | `/app/jobs/{id}` | none | same as above plus `queue_position`, `progress`, `boxes`, `rule` |
| GET | `/app/jobs/{id}/video` | none | browser-playable copy for the UI (remux `.mov` to `.mp4` with `-c copy`), Range requests |
| GET | `/healthz` | none | model loaded, queue length |

Key comparison uses `hmac.compare_digest`. Key comes from env `API_KEY`. Wrong or missing key gives 401.

Extra fields in the §10 output are allowed (organisers: what is not forbidden is allowed): each violation also gets `rule` (`document`, `clause`, `url`), `confidence` and `boxes`. The citation is also appended to `description`, in case the parser drops unknown fields.

## Pipeline

1. `detector.analyze(path)` returns events per `docs/model-contract.md`. Selected by env `DETECTOR=stub|<module>`.
2. Verdict: `violation` if any event has `confidence >= THRESHOLD` (env, tuned with `scripts/eval.py`).
3. Rules join by `type_id` from `data/rules.json`: `type` = `name`; `description` = `<observation>. <rule> (<document title>, <clause>)`. Unknown `type_id` is treated as `other`. Never emit `fallback_non_kz`.
4. Every event above the threshold is a violation, `no_id_check` included (team decision: the ID check before cash is required by S1 п. 19 пп. 1). Note for eval: it also fires on Х-35, Х-37, Х-39, which organisers labeled clean.
5. Evidence: for each event, decode the frames at `frames[].t_sec` by pts, 1 to 3 per event, JPEG quality 85, rotation applied. Frames stay clean; boxes go into a separate field.
6. `timecode_sec` = first evidence frame time.

## Reliability

- Job state persists in SQLite. On startup, jobs left `processing` are re-queued.
- Per-job watchdog at 270 s: mark `failed` with `error: "timeout"` and move on.
- Uploaded files and results are deleted after 24 h. Test videos must not leave the server (§10).
- Structured log line per job: id, size, duration, queue wait, detector time, total time, verdict.

## Main risk: queue wait counts against 5 minutes

The jury may upload several videos at once. With one worker, the last job waits for all the earlier ones. With 5 videos submitted together, the detector must average under 60 s per video, including a 145 s video. `scripts/eval.py --concurrency 5` measures this, and the result goes back to the model teammate as a budget.

## Tests

- auth: missing key, wrong key, right key
- upload validation: wrong extension, empty file, garbage bytes
- job lifecycle with the stub: processing → done, JSON matches §10, base64 decodes to a JPEG
- frames: synthetic VFR video from `ffmpeg testsrc` with dropped frames, decoded time matches pts
- restart recovery: a job left `processing` finishes after restart
- rules join: every `type_id` in `data/rules.json` renders a description with a citation

The real dataset is not in git (`assets/videos/` is ignored). Tests use synthetic videos only.

## eval

`scripts/eval.py --url --key --concurrency N` posts every video from `data/ground_truth.json`, polls, and reports:
- verdict accuracy, false positives, false negatives
- type accuracy on labeled events
- timecode error to the nearest event
- latency p50 and max, including queue wait

Х-47 and Х-48 are byte copies of Х-4 and Х-5. Count them once.

## Deploy

Docker image with the weights baked in, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`. `compose.yaml` with the GPU reservation and a volume for jobs. HTTPS via Caddy or a cloudflared tunnel. Rented GPU box, provider still to be chosen.

## Order

1. API, job store and worker with the stub detector, plus tests
2. evidence frames and rules join
3. UI endpoints and video remux
4. `scripts/eval.py`
5. Dockerfile, compose, deploy on the GPU box with the stub, verify with curl from outside
6. swap in the real detector, tune `THRESHOLD` and the time budget
