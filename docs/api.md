# API

Base URL in local development is `http://127.0.0.1:8000`. Start the server with:

```bash
API_KEY=<key> uv run uvicorn app.main:create_app --factory --port 8000
```

Two groups of endpoints share one handler:

- `/api/v1/*` is the jury API from §10 of the case. It needs the `X-API-Key` header.
- `/app/*` is for the web UI. It needs no key and returns a few extra fields.

All bodies are JSON in UTF-8. Every error response has a short code and one line of text, never a traceback or a server path:

```json
{"error": "invalid_video", "detail": "The file could not be decoded as a video."}
```

Branch on `error`, show `detail` to the user.

## POST /app/analyze

Uploads a video. Multipart form, field `file`. The server accepts `.mov` and `.mp4` up to 2048 MB.

```bash
curl -X POST http://127.0.0.1:8000/app/analyze -F "file=@X-1.mov"
```

`200 OK`

```json
{"job_id": "39b8a40d609e6247", "status": "processing"}
```

| status | `error` | when |
|---|---|---|
| 400 | `wrong_extension` | the file name does not end in `.mov` or `.mp4` |
| 400 | `empty_file` | zero bytes |
| 400 | `invalid_video` | PyAV cannot decode a frame |
| 413 | `file_too_large` | over 2048 MB. The server checks `Content-Length` before reading, and counts bytes while a chunked body streams |
| 422 | `file_missing` | no `file` field in the form |
| 429 | `queue_full` | 3 UI uploads are already uploading or waiting. Show `detail` and let the user retry later |
| 503 | `uploads_disabled` | UI uploads are turned off with `APP_UPLOADS_ENABLED=false`. The GET endpoints keep working |

Jobs from the jury API always run before UI jobs, so a UI job can wait behind jury uploads that arrive after it.

## GET /app/jobs/{job_id}

Poll this every 1 to 2 seconds until `status` is `done`. The server returns 404 for an unknown id. It deletes jobs 24 hours after upload, so old ids also return 404.

Fields present in every state:

| field | type | meaning |
|---|---|---|
| `job_id` | string | |
| `status` | `processing`, `done` | `failed` is never returned, see below |
| `queue_position` | int | number of jobs ahead of this one, counting jury API jobs that run first. `0` means the worker is on it or it is finished |
| `progress` | float 0 to 1 | `0` queued, `0.05` started, `0.8` detector finished, `0.95` frames cut, `1` done |
| `filename` | string | original upload name |
| `duration_sec` | float | video length from the container |
| `video_url` | string | path of the browser-playable copy, see below |

### Processing

```json
{
  "job_id": "171dd83cc57b7a0c",
  "status": "processing",
  "queue_position": 2,
  "progress": 0.0,
  "filename": "X-3.mov",
  "duration_sec": 20.033,
  "video_url": "/app/jobs/171dd83cc57b7a0c/video"
}
```

### Done, violation found

`verdict` is `violation` or `no_violation`. `violations` is sorted by `timecode_sec` and is empty when the verdict is `no_violation`.

```json
{
  "job_id": "39b8a40d609e6247",
  "status": "done",
  "verdict": "violation",
  "violations": [
    {
      "type": "Ключ оставлен в замке сейфа",
      "timecode_sec": 5.367,
      "description": "<observation>. Ключи от хранилищ и сейфов хранятся у лиц, ответственных за сохранность наличных денег. Порядок хранения ключей должен исключать их несанкционированное использование (Правила осуществления кассовых операций и операций по инкассации банкнот, монет и ценностей, постановление правления НБРК от 29.11.2019 № 231, п. 76, п. 80; внутренние правила банка).",
      "evidence_frames": [
        {"timestamp_sec": 5.367, "image_base64": "/9j/4AAQSkZJRgABAQAAAQAB..."},
        {"timestamp_sec": 5.7, "image_base64": "/9j/4AAQSkZJRgABAQAAAQAB..."},
        {"timestamp_sec": 6.033, "image_base64": "/9j/4AAQSkZJRgABAQAAAQAB..."}
      ],
      "type_id": "key_left_in_safe",
      "confidence": 0.62,
      "t_start_sec": 5.362,
      "t_end_sec": 6.362,
      "rule": {
        "document": "Правила осуществления кассовых операций и операций по инкассации банкнот, монет и ценностей, постановление правления НБРК от 29.11.2019 № 231",
        "clause": "п. 76, п. 80",
        "url": "https://adilet.zan.kz/rus/docs/V1900019680"
      },
      "boxes": [
        {"timestamp_sec": 5.367, "boxes": [{"x": 0.25, "y": 0.4, "w": 0.2, "h": 0.15, "label": "купюры"}]},
        {"timestamp_sec": 5.7, "boxes": []},
        {"timestamp_sec": 6.033, "boxes": []}
      ]
    }
  ],
  "queue_position": 0,
  "progress": 1.0,
  "filename": "X-1.mov",
  "duration_sec": 25.933,
  "video_url": "/app/jobs/39b8a40d609e6247/video"
}
```

Per violation:

| field | type | notes |
|---|---|---|
| `type` | string | human-readable type name from `data/rules.json` |
| `timecode_sec` | float | equals `evidence_frames[0].timestamp_sec` |
| `description` | string | what is visible, then the rule, then the citation in brackets |
| `evidence_frames` | 1 to 3 items | `timestamp_sec` is the pts of the decoded frame. `image_base64` is a clean JPEG with no data-URL prefix: render it as `data:image/jpeg;base64,<value>` |
| `type_id` | string | catalog id, `other` for a violation outside the catalog |
| `confidence` | float 0 to 1 | detector score |
| `t_start_sec`, `t_end_sec` | float | event interval, useful for a timeline marker |
| `rule.document` | string | document title, or `внутренние правила банка` |
| `rule.clause` | string or null | null when the source has no clause |
| `rule.url` | string or null | adilet link, null for internal bank rules |
| `boxes` | list | one entry per evidence frame, same order and `timestamp_sec` |
| `boxes[].boxes[]` | `{x, y, w, h, label}` | fractions 0 to 1 of the upright frame, origin top left. May be empty |

The JPEG has rotation applied. Box coordinates use the same upright frame, so the UI can draw them directly over the image or over the video at that time.

### Done, clean

```json
{
  "job_id": "7b6a464a3ed0c648",
  "status": "done",
  "verdict": "no_violation",
  "violations": [],
  "queue_position": 0,
  "progress": 1.0,
  "filename": "X-3.mov",
  "duration_sec": 20.033,
  "video_url": "/app/jobs/7b6a464a3ed0c648/video"
}
```

### Not finished

Every job must finish within 285 seconds of its upload. A job whose analysis did not finish is still returned as `done`, so a client never reads "no violation" for a video nobody checked. It gets `verdict: "violation"` and one violation with `type` «Требуется ручная проверка» (`type_id: "other"`, `confidence: 0`, timecode and frame at the middle of the video). The reason is kept in `error`, a short code, and `message`, one line to show the user:

| error | message |
|---|---|
| `timeout` | The detector did not finish in time and was stopped. |
| `deadline` | The job could not finish within the time limit after upload. |
| `crashed` | The server restarted twice while processing this video. |
| `detector_not_ready` | The detection model was still loading. |
| `detector_failed` | The detector failed on this video. |
| `result_invalid` | The detector returned a result that could not be saved. |
| `processing_failed` | Processing failed. |

`deadline` covers two cases: the job waited so long that less than 20 s of detector time would be left, or the job finished after its deadline. The detector stops 15 s before the deadline, which leaves time to cut the frames.

```json
{
  "job_id": "a580742ff3ecbccb",
  "status": "done",
  "verdict": "violation",
  "violations": [{"type": "Требуется ручная проверка", "type_id": "other", "confidence": 0.0, "timecode_sec": 12.967, "description": "...", "evidence_frames": [...], "rule": {...}, "boxes": [...]}],
  "error": "timeout",
  "message": "The detector did not finish in time and was stopped.",
  "queue_position": 0,
  "progress": 0.05,
  "filename": "X-1.mov",
  "duration_sec": 25.933,
  "video_url": "/app/jobs/a580742ff3ecbccb/video"
}
```

## GET /app/jobs/{job_id}/video

Returns the uploaded video as `video/mp4`. The server copies the H.264 stream from `.mov` into `.mp4` without re-encoding, keeps the rotation flag, and puts the index at the start of the file. The first request does the copy, which takes well under a second for a 29 MB file. Later requests serve the cached file.

The endpoint supports `Range`, so `<video src="/app/jobs/{id}/video">` can seek:

```
GET /app/jobs/39b8a40d609e6247/video
Range: bytes=0-1023

HTTP/1.1 206 Partial Content
content-type: video/mp4
accept-ranges: bytes
content-length: 1024
content-range: bytes 0-1023/5253233
```

The video is available while the job is processing, so the UI can show it before the result arrives. The server returns 404 for an unknown job.

## Reviewer feedback

The reviewer can confirm, reject or mark each model violation as uncertain, and add a violation the model missed: type from the catalog, start (and optional end) time in the video, and a comment saying what happens. Feedback is stored on the server in the local SQLite database, not in the browser. It is kept after the job and its video are deleted (labels only) and **does not change the model**; the export exists so the collected cases can be reviewed later.

### GET /app/catalog

Violation types for the form: `{"types": [{"type_id": "cash_unattended", "name": "Наличные оставлены без присмотра", "tier": "a"}, ...]}`, catalog order, `other` included.

### GET /app/jobs/{job_id}/review

```json
{"job_id": "39b8a40d609e6247", "version": 0, "decisions": {}, "added": [], "updated_at": null}
```

Version 0 means nobody has reviewed the video yet. 404 for an unknown job without a review.

### PUT /app/jobs/{job_id}/review

Send the whole review and the `version` you loaded. Keys in `decisions` are the UI's event keys (`event-<i>`, index in the time-ordered `violations` of `GET /app/jobs/{id}`).

```json
{
  "version": 0,
  "decisions": {"event-0": {"status": "rejected", "reason": "не то нарушение", "comment": "купюры у кассира в руке"}},
  "added": [{"id": "r1", "type_id": "drink_in_cash_zone", "t_start_sec": 5.0, "t_end_sec": 8.0, "comment": "на столе кружка рядом с деньгами"}]
}
```

| field | rule |
|---|---|
| `decisions[*].status` | `accepted`, `rejected`, `uncertain` or `unreviewed`; `reason` ≤ 300 chars, `comment` ≤ 2000 |
| `added[*].type_id` | a `type_id` from `data/rules.json`, `other` included |
| `added[*].t_start_sec`, `t_end_sec` | seconds from the start, inside the video; `t_end_sec` optional, not before the start |
| `added[*].comment` | required, ≤ 2000 chars |
| limits | ≤ 200 decisions, ≤ 50 added violations |

The response is the stored review with `version` + 1. The server adds `model_event` (type, timecode, confidence) to each decision, so the decision stays meaningful after the job is deleted. Errors: 409 `review_conflict` when someone saved a newer version (reload and retry), 409 `job_not_done` while processing, 422 `invalid_review` for invalid input.

### GET /api/v1/reviews/export

Team only (`X-API-Key`). All reviews of web-UI jobs, with added violations in the `data/ground_truth.json` event format (`labeled: "reviewer"`) and the decisions on model events. Jury API jobs are never exported: test videos must not be used for training.

## Cameras (live view)

Camera streams for the «Камеры» page. Off by default; someone has to press «Подключить камеры». The camera list comes from `data/cameras.json` (`CAMERAS_CONFIG`): each camera has either a `url` (`rtsp://…`, read as it arrives) or a `playlist` of video files (played back at real speed, the demo). Every finished segment becomes a normal job with source `stream`, analysed by the same detector and pipeline. Details and the demo procedure: [cameras.md](cameras.md).

`GET /app/cameras` returns the state. Each call also counts as someone watching.

```json
{
  "enabled": true, "running": true, "started_at": 1792287128.78, "stopped_reason": null,
  "now": 1792287190.1, "idle_stop_sec": 120, "jury_waiting": false,
  "cameras": [{"id": "kassa-1", "name": "Касса 1", "kind": "playlist", "state": "live", "error": null, "clip": "Х-3", "recording_since": 1792287185.2}],
  "segments": [
    {"job_id": "608f0a7b9c6599a8", "camera_id": "kassa-1", "clip": "Х-26", "started_at": 1792287128.8, "ended_at": 1792287148.6,
     "duration_sec": 19.8, "state": "done", "reason": null, "verdict": "violation", "analysed_at": 1792287172.4,
     "violations": [{"index": 0, "type": "Касса оставлена без присмотра", "type_id": "cash_unattended", "confidence": 0.81,
                     "timecode_sec": 6.2, "at": 1792287135.0, "has_frame": true}]}
  ]
}
```

- `segments` are newest first. `state` is `recording`, `queued`, `running`, `done`, `failed` or `skipped`. `reason` on a skipped segment is `busy` (a newer segment from the same camera replaced it while it waited) or `jury` (a jury API video was waiting or running, or arrived in the last 10 minutes). Times are Unix seconds; `at` is the moment of the violation on the camera clock (`started_at + timecode_sec`).
- `violations[].index` is the position in the job's own result: `/app/jobs/{job_id}` opens the segment like any other job, and `GET /app/cameras/evidence/{job_id}/{index}` returns its first evidence frame as `image/jpeg`.

`POST /app/cameras/start` and `POST /app/cameras/stop` return the same body. Start fails with 503 `cameras_disabled` when `CAMERAS_ENABLED=false`, and 503 `cameras_unavailable` when the config is missing, lists no cameras or names a clip that does not exist.

`WS /app/cameras/preview` sends every camera's preview, about 8 frames per second at 480 px height, as binary messages `<camera id>\n<jpeg bytes>`. An open socket counts as someone watching. With nobody watching for `CAMERAS_IDLE_SEC` (120 s), the cameras stop on their own (`stopped_reason: "idle"`).

## GET /healthz

```json
{"status": "ok", "detector": "stub", "model_loaded": true, "worker_alive": true, "detector_pid": 9, "queue_length": 0, "app_uploads_enabled": true, "cameras_running": false}
```

`queue_length` counts jobs that are queued or running. The endpoint returns `503` while the model is loading or after the detector process died (`"status": "model_not_loaded"`), and when the worker thread is dead (`"status": "worker_dead"`). If the worker loop dies, the process exits with code 1 so the container restart policy starts it again.

## Jury API

`POST /api/v1/analyze` and `GET /api/v1/jobs/{job_id}` take the same input (the jury API has no queue cap) and return the same body as the `/app` pair, without `queue_position`, `progress`, `filename`, `duration_sec` and `video_url`. They require `X-API-Key`. A missing or wrong key returns 401 with `{"error": "unauthorized", "detail": "Invalid or missing X-API-Key."}`.

```bash
curl -X POST https://<host>/api/v1/analyze -H "X-API-Key: <key>" -F "file=@video_017.mov"
curl https://<host>/api/v1/jobs/<job_id> -H "X-API-Key: <key>"
```

## Static UI

If `app/static/index.html` exists, the server serves `app/static/` at `/`. API routes take priority over static files.
