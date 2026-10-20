# Product plan: an evidence-review workspace

Status: proposed implementation plan, not an implemented feature set. Based on `main` at `486829e` and the [case brief](../../Кейс_ForteBank.pdf). See [research and sources](ux-research.md) and [copy-paste build prompt](frontend-build-prompt.md).

## Product decision

Build a Russian-first workspace for a bank reviewer to answer: **What did the system notice, what can I verify in the video, and what is my decision?**

Core journey:

`Выбрать видео → загрузка → анализ → проверка событий → сохранённые решения`

The differentiator is a short, credible path from prediction to evidence to human judgment. Keep the original machine result and human decisions separately visible. Do not promise winning the competition through presentation alone.

## Screens and behavior

### 1. Upload

Title: «Проверка кассовых операций». Explanation: «Загрузите видео. Система найдёт возможные нарушения, а вы проверите их по кадрам и правилам.»

- One prominent «Выбрать видео» button with a labelled native file input. Optional drop area: «Или перетащите файл сюда».
- Show `.mov`, the configured maximum size, and the fact that choosing a valid file starts upload and analysis. Do not invent a 29 MB limit.
- Single-file MVP. Show the selected filename and real upload progress when measurable. No decorative sample statistics or compulsory registration.
- Validate early for helpful feedback; backend still validates content, size, and decodability. An empty MIME type alone must not reject a valid MOV.
- Specific messages: «Этот формат не поддерживается. Выберите файл .mov»; «Файл превышает допустимый размер: …»; «Не удалось прочитать видео. Попробуйте другой файл».
- Do not claim “nothing leaves your computer”: footage is uploaded to the team's service. Once verified, say «Анализ выполняется на сервере команды без обращения к облачным AI-сервисам».

### 2. Upload and processing

- Separate byte upload from server analysis. Actual upload percentage may use XHR progress. Analysis is indeterminate unless the backend provides real progress.
- With the minimum API, show «Видео загружено. Выполняется анализ» and elapsed time. Do not simulate frame-by-frame progress, model stages, or a countdown.
- Link the job to `/jobs/:jobId`; refresh resumes fetching an existing job instead of submitting again.
- Poll while processing, stop on `done`/`failed`; retry transient connection failures without calling them “no violations.” Do not automatically retry a non-idempotent upload.
- At the five-minute deadline, show a clear delay/failure state and preserve the job reference. Do not silently report success. A late result, if supported, is explicitly late; it does not erase the missed jury deadline.
- Do not show server cancellation unless supported. Aborting a local upload and cancelling an already accepted job are different operations.

### 3. Review

Desktop composition, intended for 1280 px and wider:

```text
Проверка кассовых операций                     Загрузить другое видео
filename.mov · duration · job reference
Обнаружены признаки возможных нарушений
Предложения системы: 3       Проверка: 1 из 3 рассмотрено
─────────────────────────────────────────────────────────────────────
События              Видео / исходный контекст      Выбранное событие
00:13 Не рассмотрено [preserved video aspect ratio] Название · 00:13
00:22 Подтверждено   [play / seek / speed]           Что видно
00:46 Не рассмотрено[clickable event timeline]      Почему отмечено
                                                    Кадры: 1 · 2 · 3
                                                    Правило и источник
                                                    Ограничения вывода
                                                    [Подтвердить признак]
                                                    [Отклонить]
                                                    [Недостаточно данных]
─────────────────────────────────────────────────────────────────────
Решение сохранено                         К следующему нерассмотренному
```

Widths are content-driven: event rail roughly 220–260 px, portrait player roughly 300–400 px, explanation panel gets remaining space. Preserve both portrait and landscape aspect ratios. At narrower widths, collapse the event rail into a labelled event selector; at phone widths use one column, not a squeezed three-column layout.

- Use a chronological event list with textual review status, not a wall of cards or a red incident count.
- Selecting an event selects its evidence and seeks the player near the supplied timestamp. Offer a context playback action starting about three seconds before it, clamped to video bounds. No autoplay on initial page load.
- Timeline markers must also be accessible buttons/list entries. Overlapping markers must remain individually selectable through the list.
- Display backend-derived times. Never calculate original timestamps as `frame_index / 30`.
- Evidence frames are authoritative snapshots from the original video. Browser seeking is contextual playback, not proof of exact-frame alignment. Show the exact evidence timestamp separately.
- Show 1–3 supplied frames, with enlarge/zoom and optional bounding-box toggle. Do not crop evidence by default or permanently paint boxes into the source image.
- Boxes use normalized display-oriented coordinates. Test rotation, letterboxing, and resizing; clip overlays to the actual image area.

The explanation panel has four logically distinct parts, only when data supports them:

1. **Что видно** — the supplied visible observation, without speculation about intent or off-camera actions.
2. **Почему отмечено** — the supplied reasoning connecting that observation to a possible breach.
3. **Правило** — backend-supplied document, clause when available, and source link. Source links open on user request; inference must not depend on live source access.
4. **Ограничения** — relevant uncertainty, such as authorization not being visible. Display supplied limitations or a generic caution, not invented case-specific conclusions.

If the endpoint supplies only `description`, show «Описание системы» verbatim as text. Do not use a frontend LLM or brittle string splitting to invent the four sections. Ask the backend for structured fields as an extension.

No confidence meter by default. Numeric confidence is optional expert detail only if actually exposed and appropriately explained/calibrated. Never fabricate certainty labels or severity scores.

### 4. Human decisions

All events start «Не рассмотрено». A decision applies to one event, not the entire video.

- «Подтвердить признак»: reviewer agrees the visible evidence supports the proposed possible breach. It is not a legal or disciplinary conclusion.
- «Отклонить»: reviewer disagrees. Offer a quick reason: not visible, wrong classification, wrong time, insufficient rule support, or other; comment optional. Do not force a long form.
- «Недостаточно данных»: reviewer cannot decide from available evidence. This is separate from rejection and remains explicitly unresolved in the final summary.
- Give confirm/reject comparable prominence. No preselection, automatic acceptance, or MVP “accept all.”
- Show saving/saved/failed honestly. A failed save cannot increment the persisted-review count. Retry should be safe. Allow changing/undoing a decision and keep its audit history.
- After saving, offer the next unreviewed event without unexpectedly moving keyboard focus or playing video.
- Distinguish “considered” from “resolved.” A review summary might say «Подтверждено: 1 · Отклонено: 1 · Недостаточно данных: 1 · Не рассмотрено: 0». Do not label it fully resolved.
- A completion action requires no unreviewed events; if uncertainty remains, label it «Завершить с нерешёнными вопросами», not a clean approval.

Original predictions remain immutable. If all predictions are rejected, the model header still reports what the model found and the review summary reports that none was accepted. Reviewer decisions never rewrite the jury API verdict. Feedback is not automatic training data, especially for jury test videos.

### 5. Empty and degraded results

| State | Required treatment |
|---|---|
| Successful analysis, no events | Exact heading «Нарушение не обнаружено»; explain this is a model result, not proof that every procedure was followed; keep full-video access. |
| Processing failed / unavailable | Explicit error and recovery action; never a no-violation result. |
| Event has missing evidence | Keep event visible, label evidence unavailable, offer context video and retry where meaningful; do not create a placeholder “proof” frame. |
| Source missing / internal-only | Show «Публичное основание не подтверждено» where applicable and route for rules-owner review. Do not invent a legal citation. |
| Unknown `other` event | Show supplied observation and «Тип требует уточнения»; an analogous rule is labelled as analogy, not direct proof. |
| Preview unsupported / failed | Evidence and review stay usable; clearly explain unavailable playback. Generate a compatible preview locally where possible. |
| Lost network during review | Preserve unsaved text in memory, mark not saved, allow retry. Do not imply synchronization. |
| Re-analysis changes output | New analysis revision; never silently attach old decisions to new predictions. |

## Visual direction

Working direction: **a precise, calm inspection desk**. This is a proposed design, not an official ForteBank brand guide.

- Warm off-white canvas, charcoal text, one deep burgundy action accent. Amber for awaiting review, restrained green for accepted observations. Always accompany color with text/icon.
- Compact typography hierarchy and strong alignment; tabular numerals for timecodes. Start with a local system sans-serif stack supporting Cyrillic; a licensed, self-hosted font can follow after visual testing.
- Flat, well-separated panels, fine rules, modest radii, and generous space around evidence. Avoid putting every label in a rounded card.
- Dark neutral player surface distinguishes the source footage from surrounding explanations. Original footage remains visually unmodified by default.
- No gradient hero, glass panels, glowing AI orb, chatbot entry point, fake analytics, decorative charts, or unnecessary sidebar navigation.
- Motion only for orientation and state feedback; respect reduced-motion settings. No animated page entrance that delays work.

This direction is distinctive because of its evidence hierarchy and video-aware composition, not because it adds unrelated decoration.

## Stack decision

There is no existing frontend stack on the inspected main branch. Re-check before implementation; reuse compatible work if teammates add it.

| Option | Advantages here | Costs / risks | Decision |
|---|---|---|---|
| React + TypeScript + Vite SPA | Good fit for synchronized events/video/review state; static deployment beside Python; flexible visual design | Must explicitly implement routing and server-state handling | **Recommended** |
| Next.js | Useful integrated routing and server features; reasonable if team already has it | SSR/SEO provide little value here; another server boundary alongside Python unless carefully simplified | Not justified for current repository |
| Vue or Svelte SPA | Also capable of this workflow; compact component code | Switching gives little task-specific benefit; team familiarity dominates | Choose instead only if team is materially faster in it |
| Server templates / HTMX | Small initial setup, natural Python integration | Synchronized playback, overlays, and rich review state need substantial client logic anyway | Viable for a simpler demo, less suitable for this workspace |
| Streamlit / Gradio | Fast model demo and internal debugging | Less control over exact evidence interactions and visual composition | Keep for internal model tooling, not primary jury-facing UX |

Proposed implementation: React, TypeScript, Vite; React Router for job URLs; TanStack Query for polling and review mutations; CSS custom properties + CSS Modules for styling; native HTML controls and selective Radix primitives; native `<video>` plus an accessible custom event timeline and SVG overlays. No global state library initially: local/reducer state for selection, server-state cache for jobs/reviews.

Tests: Vitest for adapters and review state, Playwright for user flows and browser screenshots. Validate API responses at the boundary (a small schema validator such as Zod is optional). Resolve compatible package versions at implementation time, check commercial-use licenses, and commit a lockfile. Serve the static build from the existing/planned Python service or same-origin reverse proxy; a separate Node production server is not required.

## API boundaries and integration work

The required jury API is `POST /api/v1/analyze` (multipart `file`, `X-API-Key`) and `GET /api/v1/jobs/{job_id}`. Status: `processing | done | failed`. A completed result contains `verdict`, and `violations[]` with `type`, `timecode_sec`, `description`, and `evidence_frames[{timestamp_sec,image_base64}]`.

FAQ 15 mentions an insufficient-data verdict, but the explicit API schema has only two verdict values. Keep uncertainty in the human-review layer; seek organizer clarification before changing the jury contract.

The [model contract](../model-contract.md) additionally describes rule metadata, intervals, observations, confidence, and boxes at different layers. Do not assume every internal model field reaches the browser.

| UI need | Current documented minimum | Proposed additional contract, to agree with backend |
|---|---|---|
| Stable review target | No event ID or revision in jury schema | `analysis_revision` and stable `event_id` in a separate UI response or compatible extension |
| Separate explanatory sections | Combined `description` | Optional `observation`, `rationale`, `limitations`, structured `rule` |
| Playback after reload | No media endpoint specified | Protected, range-capable preview URL, duration, display dimensions, and documented time mapping |
| Saved decisions | No review endpoint | Read/write review API keyed by job + revision + event, with decision, optional reason/comment, server timestamp, review revision |
| Honest processing phases | Only coarse status | Optional real phase/progress; indeterminate UI works without it |

Proposed review writes are idempotent/version-checked so retries or two tabs do not silently overwrite a newer decision. A lightweight SQLite store may suffice for the hackathon; use existing backend storage if present. Keep immutable analysis records and a review-change log. A reviewer session is not verified employee identity; do not claim otherwise.

Build against a small adapter interface until backend extensions are agreed. Development fixtures are explicit and synthetic. If persistence is absent, label a prototype decision «Черновик — не сохранён на сервере»; do not present browser-only storage as a completed feature. Do not store frames or video in localStorage.

Never embed the shared jury API key in the browser bundle or a `VITE_*` environment variable. Agree a same-origin, separately scoped UI session/API path; the jury retains its required keyed endpoints. Protect job/media access, bound uploads and concurrency, and avoid making a public proxy an unrestricted path to the private key. Full enterprise accounts are not required for the MVP.

Browser compatibility: use a local preview conversion/remux when the actual codec requires it; retain the original video and original PTS-based evidence. Preview generation must preserve/document time mapping and not consume the model's latency budget unexpectedly. No FFmpeg-in-browser dependency for the first version. Serve app assets locally and exclude third-party analytics/session replay on video pages.

## Delivery sequence and ownership

| Step | AI can implement or prepare | Team must supply or verify | Exit condition |
|---|---|---|---|
| 1. Contract alignment | Inventory fields; draft adapters, state types, fixtures | Backend owner agrees preview/review endpoints; ML owner confirms time/box semantics; rules owner resolves citation gaps | Minimum and extension schemas explicitly agreed |
| 2. Review-screen prototype | Layout, tokens, timeline, synthetic event states, responsive screenshots | Product reviewer chooses visual direction and checks Russian wording | One event can be examined and a reversible draft decision made |
| 3. Upload → job → result | Validated upload, real progress, routing, polling, errors, result adapter | Backend supplies working endpoints and limits | Fresh video runs without manual intervention; refresh resumes job |
| 4. Evidence + persistent review | Playback integration, frames, sources, review controls and save states | Backend persistence/access controls; human verifies timing against actual footage | Decisions survive reload; original model response remains unchanged |
| 5. Verification | Automated flow tests, keyboard checks, visual screenshots, network audit | 3–5 unfamiliar teammates perform tasks; team judges evidence/source accuracy | No critical errors; usability issues fixed and test outcomes recorded |
| 6. Demo hardening | Reproducible build, smoke-test script, failure recovery checks | Team tests deployed URL/API, model latency, hardware and concurrency | End-to-end results meet five-minute limit on target hardware |

Implement the core vertical slice before history, export, batch upload, manual event creation, correction of event time/type, or reviewer accounts. Those are P1, not dependencies of the MVP. Do not merge or publish branches automatically as part of design work.

## Acceptance and usability checks

Functional fixtures: 0/1/many events, overlapping timestamps, `other`, missing sources/frames, portrait/landscape/rotated footage, upload rejection, corrupt video, expired job access, processing failure, delayed result, transient network error, failed review save, rapid repeat clicks, and reload after review.

- `no_violation` is reachable only from successful valid analysis; malformed/failed responses are not treated as empty results.
- Confirm/reject/uncertain/undo work per event; replaying save does not duplicate decisions. All-rejected and partially unresolved summaries remain accurate.
- Required API responses retain their schema and original verdict after human review.
- Local PTS evidence matches the event; video context and boxes are correctly oriented. Browser playback is tested with real representative codecs.
- Keyboard can upload, select events, seek, open/close evidence, read sources, make/reverse decisions. Dialog focus returns correctly; visible focus is not obscured.
- Check WCAG AA text contrast, non-color status cues, meaningful labels, live status announcements, reduced motion, and 200% zoom. Primary controls target 44 px; test narrow reflow as well as desktop.
- Inspect screenshots at 1280, 1440, and 1920 widths plus a narrow viewport. Test Chromium and Firefox, and Safari/WebKit where available.
- Inspect deployed network requests: no cloud model calls, third-party media uploads, analytics capture, remote fonts, or leaked API key. Rule links are user-initiated external navigation.
- Measure the deployed request-to-complete-result path, including queueing, media preparation, inference, evidence extraction, and serialization. A fast UI or a model-only benchmark does not demonstrate the five-minute service requirement.

Suggested moderated tasks: find upload, submit a video, locate the first suspected event, explain why it was flagged, inspect its source, reject it and undo, refresh to verify saving, and recover from a failed request. Ask “Is this the model's suggestion or a final finding?” without teaching the answer first.

Proposed targets, not measured results: primary upload found within 5 seconds; first evidence opened within 10 seconds of results; every participant understands that review is required; no lost saved decisions. Record task success, time, confusion, and errors. Three to five participants can reveal usability issues but cannot establish statistical performance claims.
