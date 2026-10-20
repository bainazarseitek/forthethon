# Forte Review frontend

Russian-language review UI for the `/app/*` endpoints in [docs/api.md](../docs/api.md). React + TypeScript + Vite.

The reviewer uploads a `.mov` or `.mp4`, sees queue position and progress, then the verdict, each violation with timecode, description, rule citation and evidence frames with box overlays. Clicking a violation seeks the video. Reviewer decisions are stored only in the browser's `localStorage`; the backend has no review storage.

## Build

The FastAPI server serves the built files from `app/static/`, so the server needs no Node. After changing anything in `src/`, rebuild and commit `app/static/`:

```sh
cd frontend
npm ci
npm run build
```

Routes use the URL hash (`/#/jobs/<id>`), so `StaticFiles` needs no SPA fallback.

## Develop

```sh
API_KEY=dev uv run uvicorn app.main:create_app --factory --port 8000
cd frontend && npm run dev
```

Vite proxies `/app` to `API_ORIGIN` (default `http://127.0.0.1:8000`). `/#/demo` shows a synthetic example in dev only, or with `VITE_ENABLE_DEMO=true` at build time.

## Test

```sh
npm test
npx playwright install chromium
SAMPLE_VIDEO=/path/to/X-1.mov npm run test:e2e
```

`test:e2e` builds, starts the backend with the stub detector on port 8092 and a temporary `DATA_DIR`, uploads the sample video through the UI, and covers error, queue, failed, clean and many-violation states with mocked `/app` responses. Without `SAMPLE_VIDEO` (default `../assets/videos/X-1.mov`) the real-upload test is skipped.
