# Design B: Evidence dossier

## Concept

1. The result is a case file, not a dashboard. Each check gets a case number (`Дело 7B4A·98B1`).
2. Light warm paper palette, Forte raspberry only for findings, pins and the norm.
3. Source Serif 4 for verdicts and headings, Inter for UI, JetBrains Mono for timecodes. All bundled via @fontsource, no CDN.
4. The verdict is one of the two required phrases, set large at the top, with the "hint, not a legal conclusion" line under it.
5. The player sits on a dark stage with an ambient glow sampled from the video, so the portrait CCTV frame does not float in black bars.
6. The timeline is chaptered: every finding is a numbered pin plus a shaded chapter that covers the 3 s lead-in and the event interval.
7. Beside the player (under it on mobile) is the finding card, a three-step story: what is seen, why it may violate, the norm with clause and adilet link.
8. The right rail is the case index: number, type, timecode, confidence, the reviewer's decision.
9. The report is a printable A4 dossier: a cover with the verdict, facts, table of contents and signature lines, then one page per finding with frames.
10. Motion is short and directional: the card slides in the step direction, the seek flashes once, boxes fade in. Reduced motion turns it all off.

## Key interactions

- Upload by drag and drop or file picker. Validation and upload progress happen inside the drop zone, and API errors show in Russian.
- Processing shows a five-stage stepper driven by `progress` and `queue_position`, plus elapsed time.
- Clicking a pin, the index or N/P jumps to the exact event frame and pauses it, with the detector boxes drawn over the video.
- R replays from 3 s before the event to 1 s after it. L loops that window.
- Evidence frames open in a lightbox with click-to-zoom (2.2x, follows the pointer), a boxes toggle, frame stepping and "Показать в видео".
- Reviewer decisions (Подтверждаю / Нужна проверка / Отклоняю) are stored in localStorage. They show in the rail, on the pins and in the report.
- Report: `#/jobs/{id}/report`, then "Печать / PDF" (window.print, exactly 1 + N A4 pages).

## Keyboard

| Key | Action |
|---|---|
| Space, K | play / pause |
| N, P | next / previous finding |
| 1-9 | jump to finding |
| R | replay 3 s before the finding |
| L | loop the finding |
| ← → (Shift) | seek 1 s (5 s) |
| , . | frame step |
| 0, Home | to start |
| S | speed 1x / 0.5x / 0.25x |
| B | boxes over video |
| F | fullscreen |
| ? | shortcut sheet |

The keys also work in the Russian layout.

## What stands out

- The finding reads as an argument (observation, reason, norm). The model's `description` is split client-side, and the bracketed citation is dropped because the norm block already shows it.
- Boxes are always an overlay, on the video and on the frames, and never burned in. They can be toggled.
- The report is a real document: it prints on A4 and has a signature block for the reviewer.
- It works offline: no external requests, and fonts are bundled.
