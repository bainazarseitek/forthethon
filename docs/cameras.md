# Cameras: live analysis without uploads

The «Камеры» button in the header opens a monitoring page. The server reads camera streams, cuts them into segments and sends each finished segment to the same detector the jury API uses. Violations appear in a feed with the frame, the camera and the time; each opens the usual review page.

There is no real camera at the hackathon, so the demo camera plays dataset recordings at real speed. To the server it behaves like a camera: frames arrive at the camera's pace, nothing is uploaded, and the code that records and analyses segments is the same one that reads an `rtsp://` camera (tested with MediaMTX, see below). Say this out loud in the pitch: the jury knows the dataset.

## How it works

```
camera (rtsp://…)  ─or─  playlist of clips played at real speed
        │  PyAV reads packets
        ▼
camera thread ──► preview: JPEG ~8 fps ──► WebSocket ──► «Камеры» page
        │
        └─► segment: packets copied to .mov without re-encoding
              playlist: one clip = one segment;  stream: cut at a keyframe after segment_sec
                  ▼
            job, source "stream", last in the queue ──► detector ──► alert in the feed
```

- Recording copies the H.264 packets, so the detector sees the same frames as in the source clip with the same timestamps. The accuracy measured on the dataset clips carries over to the demo.
- One clip per segment also means a cut between two clips never falls inside a segment, so it is never mistaken for the cashier leaving.

## Safety around the jury run

- Off by default. It starts only from the button and stops itself after 2 minutes with nobody on the page (`CAMERAS_IDLE_SEC`).
- Queue order is jury API, then the web UI, then camera segments.
- While a jury video is uploading, queued or running, and for 10 minutes after the last one arrived, finished segments are dropped, not queued (the feed shows «пропущен: идёт проверка через API»). A jury that sends videos one after another never finds a segment in the gap between two of its videos.
- A segment that is already running when a jury video arrives still finishes. That adds up to one segment's analysis time (about 20–40 s, capped at 90 s by the detector budget for segments) to that jury video. **Keep the cameras off during the jury testing window**, or set `CAMERAS_ENABLED=false` for the backend program in supervisord.

## Capacity

A 20 s clip takes about 20–40 s to analyse on the RTX 5090 (detector log, shared GPU), so one GPU keeps up with roughly one camera. To stay live, each camera keeps at most one segment waiting: a newer segment replaces it (`пропущен: модель занята`). With more cameras, more segments get skipped. In production the answer is motion gating (skip segments where nothing moves, which is most of the day at a till) and more GPUs.

From the end of an event to the alert: rest of the segment + analysis, typically 30–60 s.

## Run the demo

1. The 13 clips listed in `data/cameras.json` must exist under `assets/videos/` on the server (the folder is not in git). Copy them, or point `CAMERAS_CONFIG` at a file with absolute paths.
2. About 2 minutes before presenting, open `http://<host>/#/cameras` and press «Подключить камеры». By the time you present, the feed already has alerts and new ones keep arriving.
3. Show the live picture (camera name, LIVE, clock, REC timer), the segment strip under it, an alert coming in. Click it: the review page with frames, legal norm, decision buttons.
4. Press «Остановить» when done. Closing the tab also stops it within 2 minutes.

The playlist is 10 violation clips the model detects reliably and 3 clean ones, about 4 minutes in a loop. It leaves out Х-1 and Х-44 (missed by the model) and Х-35/37/39 (the organisers label them clean).

## Connect a real camera

Add an entry with a `url` instead of a `playlist`:

```json
{"id": "kassa-2", "name": "Касса 2", "url": "rtsp://user:pass@10.0.0.21:554/stream1"}
```

`segment_sec` (default 30) sets the segment length for streams. The reader uses RTSP over TCP and reconnects every 5 s if the camera drops. H.264 cameras are recorded without re-encoding.

## Test an RTSP camera locally

```bash
brew install mediamtx ffmpeg
mediamtx &                                         # RTSP server on :8554
printf "file '%s'\n" "$PWD/assets/videos/Х-15.mov" "$PWD/assets/videos/Х-22.mov" > /tmp/list.txt
ffmpeg -re -stream_loop -1 -f concat -safe 0 -i /tmp/list.txt -an \
  -c:v libx264 -preset ultrafast -tune zerolatency -g 30 -f rtsp rtsp://127.0.0.1:8554/kassa2 &
```

Then put `{"id": "kassa-2", "name": "Касса 2", "url": "rtsp://127.0.0.1:8554/kassa2"}` into a copy of `data/cameras.json`, start the backend with `CAMERAS_CONFIG=<copy>` and press the button. The same URL opens in VLC.

## Limits

- Audio is dropped. Rotation metadata of the source is not copied into segments (the dataset and typical CCTV cameras have none).
- An event that spans two stream segments may be cut in two or missed. Playlist segments do not have this problem. For streams, overlapping segments would fix it.
- Segment history lives in memory: a backend restart clears the feed and drops segments still waiting. The last 40 segments per camera are kept; older ones are deleted to save disk.
