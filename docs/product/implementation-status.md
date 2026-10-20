# Frontend implementation status

Verified locally on 2026-10-17, on branch `design/evidence-review-ux`. Nothing committed, pushed, or deployed by this implementation.

Superseded on 2026-10-18: the frontend now talks to the real `/app/*` backend and is served from `app/static/`. See [frontend/README.md](../../frontend/README.md); the notes below describe the earlier mock-only state.

## Delivered

- React/TypeScript/Vite application in `frontend/`, with a dependency lockfile.
- Russian-language upload screen with file validation, native chooser, drag/drop, actual upload progress, and explicit API failure handling.
- Job URL, polling, refresh/resume, five-minute waiting state, retry, and failure/no-violation separation.
- Responsive event rail, video context, timestamp navigation, evidence JPEG dialogs, per-frame box overlays, backend-provided explanations and safe source links.
- Confirm/reject/uncertain/reset workflow, counts independent of the machine verdict, and optional reasons/comments.
- Review API adapter with version/revision checking, server acknowledgement validation, save failure/conflict recovery, and honest draft fallback when persistence is unavailable.
- Explicit synthetic demo, without dataset footage or fabricated model outputs for real uploads. Demo route is production opt-in.
- Keyboard-accessible dialogs, status announcements, reduced-motion handling, contrast fixes, and locally bundled assets.
- [Backend contract and run instructions](../../frontend/README.md).

The Sites skill was used for local implementation and verification, preserving the existing repository and avoiding separate hosted deployment. The native preview browser was unavailable; verification used the application's Playwright browser test suite and inspected screenshot artifacts instead.

## Verification evidence

- `npm test`: **18 passed**.
- `npm run build`: TypeScript check and production build passed.
- `npm run test:e2e`: **40 passed, 2 intentionally skipped**. The extra 320/1920 viewport sweep is Chromium-only; regular flows run in Chromium, Firefox, and mobile Chromium.
- Automated axe checks on upload and demo review: no reported violations for the configured WCAG A/AA tags. This is not a full accessibility certification.
- Inspected desktop/mobile browser screenshots. Fixed a duplicate-key rendering issue, help-button accessible name, selected-state contrast, and skip-link screenshot clipping during verification.
- Browser tests cover simulated API upload/polling, no-result/failure/malformed states, five-minute wait, evidence dialogs, server save failures, conflicts, analysis-revision mismatch, and immutable model verdict after all events are rejected.
- Production dependency license metadata in the lockfile: MIT, ISC, or 0BSD. Preserve upstream license notices when distributing; no third-party fonts or media assets added.
- `git diff --check`: clean for tracked changes. New source/config files formatted with Prettier.

## Not implemented or not verified

- No detector/backend application exists in this checkout. Real analysis, queueing, authentication, review storage/audit history, and local preview generation are backend integration work, not simulated production features.
- Server review persistence is tested against synthetic intercepted responses, not a deployed database. Live review without the required capability is explicitly an in-memory draft.
- Actual MOV codec playback, original PTS alignment, model accuracy, legal source correctness, and five-minute end-to-end latency require testing with the real pipeline and target hardware.
- No final sign-off API, exports, history, batch upload, employee identity, or automatic feedback training.
- No production deployment or human usability study. Safari/WebKit was not tested; the mobile run uses Chromium device emulation.

Next: backend owner implements `/api/ui` and the optional review/media contracts, then the team verifies the deployed vertical slice with actual footage locally.
