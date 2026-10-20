# UX research: evidence review for ForteBank

Research date: 2026-10-17. Planning only; no user study has been conducted and no frontend has been implemented by this change.

Read next: [product and implementation plan](ui-ux-plan.md), then [AI build prompt](frontend-build-prompt.md).

## What the case actually requires

Primary source: [case brief](../../Кейс_ForteBank.pdf), sections 1–10 and FAQ. Repository integration source: [model contract](../model-contract.md). These take precedence over general design advice.

- A working public website and backend; upload and processing must not require manual code execution.
- Local inference, with no cloud AI processing of video, frames, or descriptions. After weights are loaded, inference must work without internet.
- Upload, processing status, a result, and multiple possible events. Each event needs a type, time, description, 1–3 evidence JPEGs, and an applicable rule/source explanation.
- Required result language: «Нарушение не обнаружено» or «Обнаружены признаки возможных нарушений».
- Results support a human reviewer; they are not final legal or disciplinary findings.
- The jury API must return results within **five minutes maximum**, not minimum. The repository allocates up to 240 seconds to the model to leave backend overhead.
- Dataset examples are `.mov`, approximately 10–145 seconds and around 29 MB. These descriptions are not documented upload-size limits. Test footage may use another angle.
- Face identification and audio analysis are out of scope. Test material cannot be used for training, published, or transferred to third parties.
- Accounts, history, export, and batch upload are optional. They should not delay the core workflow.

Scoring: detection 25, localization 15, classification 15, rule grounding 15, robustness 10, product readiness 10, technical solution/locality 10. UX can make evidence, reasoning, and reliability demonstrable; it cannot compensate for weak model evaluation.

The user additionally requires accepting/rejecting predictions. That is an explicit product requirement, even though a saved review workflow is not specified in the jury API.

## Research and application

The right-hand column is our design judgment, not a claim that a source tested this banking workflow.

| Primary resource | Relevant guidance | Application to this product |
|---|---|---|
| [Google PAIR: explainability and trust](https://pair.withgoogle.com/guidebook-v2/chapter/explainability-trust/) | Explanations should support decisions; numerical confidence can be misunderstood and should be tested. | Lead with the actual frame, context, and applicable rule. Do not turn an uncalibrated score into “82% certain.” |
| [Google PAIR: feedback and control](https://pair.withgoogle.com/guidebook-v2/chapter/feedback-controls/) | Feedback needs clear meaning and an understandable effect. | State that accepting/rejecting saves this review; it does not automatically retrain the model or rewrite its output. |
| [GOV.UK: file upload](https://design-system.service.gov.uk/components/file-upload/) | Use a labelled file input, optional drag-and-drop, associated hints, and specific error messages. | A visible «Выбрать видео» button is the primary action. Dragging is an alternative, never the only path. |
| [W3C: status messages](https://www.w3.org/WAI/WCAG22/Understanding/status-messages.html) | Status changes should be available to assistive technology without requiring focus changes. | Announce upload completion, analysis completion, and save failures politely; avoid announcing every polling tick. |
| [W3C: target size](https://www.w3.org/WAI/WCAG22/Understanding/target-size-minimum) | WCAG 2.2 AA specifies a 24 CSS-pixel minimum subject to exceptions. | Prefer 44-pixel-high primary controls as our usability target; do not misstate 44 pixels as the AA minimum. |
| [Anthropic: Claude Code best practices](https://code.claude.com/docs/en/best-practices) | Supply concrete context and verifiable checks; separate exploration, planning, and implementation. | Give the agent the case, API contract, state matrix, and acceptance tests. Require browser verification before declaring the interface finished. |
| [Anthropic: frontend-design plugin](https://claude.com/marketplace/plugins/frontend-design) | Establish purpose, audience, and visual direction before generating a distinctive interface. | Specify a restrained evidence-review workspace. Borrow the intentional design process, not decorative motion inappropriate for reviewing evidence. This plugin was researched, not installed. |
| [Anthropic: harness design for long-running apps](https://www.anthropic.com/engineering/harness-design-long-running-apps) | Explicit design criteria and evaluation loops help critique generic or incomplete generated work. | Run a separate critique pass against screenshots and tasks. A teammate supplies an independent judgment; this plan does not require a multi-agent system. |
| [React: building an app from scratch](https://react.dev/learn/build-a-react-app-from-scratch) | React generally recommends frameworks, but documents Vite-based setups and the routing/data-fetching work they require. | A small SPA fits this specific Python-backed tool without SSR/SEO requirements. This is a contextual choice, not a universal rejection of frameworks. |
| [TanStack Query: auto-refetching example](https://tanstack.com/query/latest/docs/framework/react/examples/auto-refetching) | Supports interval refetching for changing server data. | Poll jobs with bounded retries, stop on terminal status, and keep polling errors separate from model failure. |
| [Radix: accessibility](https://www.radix-ui.com/primitives/docs/overview/accessibility) | Primitives provide established keyboard/focus behavior; authors still provide meaningful labels. | Use primitives selectively for dialogs and other complex controls, with a custom visual system. They do not guarantee the entire application is accessible. |
| [MDN: media containers](https://developer.mozilla.org/en-US/docs/Web/Media/Guides/Formats/Containers) | A container and its codecs jointly affect playback compatibility; MOV and MP4 are not interchangeable. | Keep the original for inference/evidence and generate a browser-playable preview locally when necessary. Test actual codecs in target browsers. |
| [Playwright: visual comparisons](https://playwright.dev/docs/test-snapshots) | Browser screenshots can be compared against approved baselines. | Use deterministic synthetic fixtures and stable browser environments for visual regression; pair screenshots with functional checks. |

## How to get useful frontend work from Claude or another coding AI

1. Supply a bounded context pack: this research, the plan, the build prompt, case requirements, model contract, and relevant code. Do not dump all videos, unrelated research, or historical PR arguments into the prompt.
2. Choose one visual direction and a small reference set. Here the references have different jobs: GOV.UK for upload clarity, PAIR for human control, and the plan for the visual composition. They are not templates to copy wholesale.
3. Build the review screen first with explicitly synthetic data. It exposes the hardest layout and interaction problems sooner than a polished landing page.
4. Implement one complete interaction slice at a time. Inspect its browser rendering, keyboard flow, and failure state before expanding scope.
5. Review against concrete criteria: visible primary action, legible portrait footage, traceable evidence, reversible decisions, no fabricated data, and predictable recovery.
6. Keep a short project instruction file only when implementation starts: commands, constraints, architecture, and verification expectations. Avoid an enormous permanent prompt.

Do not send bank footage, evidence frames, test data, or sensitive observations to cloud coding/design tools. Use synthetic placeholders for remote AI-assisted design and local human inspection for real footage. Do not use cloud AI inside the deployed analysis pipeline. A public application URL does not imply cloud inference.

## Uncertainties to resolve, not hide with UI

- The repository currently documents the API but contains no main-branch frontend application. Written contracts are not proof that endpoints are deployed.
- The catalogue contains internal-rule references, while FAQ 32 says the dataset does not depend exclusively on unpublished internal rules. The rules owner must resolve that gap; the frontend must not invent an Adilet citation.
- FAQ 15 mentions a “Недостаточно данных” verdict, but the explicit jury JSON schema allows only `violation` and `no_violation`. Follow the explicit schema unless organizers clarify otherwise. Our uncertain human-review state is separate and does not add a third API verdict.
- Review semantics need team confirmation: “confirm an observable possible breach” is the proposed meaning, not “make a final legal finding.”
- Playback codecs, server upload limit, retention policy, concurrent job capacity, and access-control approach require implementation decisions.
- The visual direction and task-time targets in the plan are hypotheses. Test them with people unfamiliar with the project before claiming usability results.
