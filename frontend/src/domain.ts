import { z } from "zod";

const seconds = z.number().finite().nonnegative();
export const boxSchema = z.object({
  x: z.number().finite(),
  y: z.number().finite(),
  w: z.number().finite().nonnegative(),
  h: z.number().finite().nonnegative(),
  label: z.string(),
});
export const eventSchema = z
  .object({
    event_id: z.string().min(1).optional(),
    type: z.string().min(1),
    type_id: z.string().optional(),
    timecode_sec: seconds,
    description: z.string(),
    observation: z.string().optional(),
    rationale: z.string().optional(),
    limitations: z.string().optional(),
    confidence: z.number().min(0).max(1).optional(),
    t_start_sec: seconds.optional(),
    t_end_sec: seconds.optional(),
    evidence_frames: z
      .array(
        z.object({
          timestamp_sec: seconds,
          image_base64: z.string(),
          boxes: z.array(boxSchema).optional(),
        }),
      )
      .max(3),
    boxes: z
      .array(z.object({ timestamp_sec: seconds, boxes: z.array(boxSchema) }))
      .optional(),
    rule: z
      .object({
        document: z.string(),
        clause: z.string().nullable().optional(),
        url: z.string().nullable().optional(),
      })
      .nullable()
      .optional(),
  })
  .transform(({ boxes, ...event }) => ({
    ...event,
    evidence_frames: event.evidence_frames.map((frame, i) => ({
      ...frame,
      boxes: frame.boxes ?? boxes?.[i]?.boxes,
    })),
  }));
const common = z.object({
  job_id: z.string().min(1),
  filename: z.string().optional(),
  duration_sec: seconds.optional(),
  video_url: z.string().optional(),
  queue_position: z.number().int().nonnegative().optional(),
  progress: z.number().min(0).max(1).optional(),
});
export const jobSchema = z
  .discriminatedUnion("status", [
    common.extend({ status: z.literal("processing") }),
    common.extend({
      status: z.literal("failed"),
      error: z.string().optional(),
      message: z.string().optional(),
    }),
    common.extend({
      status: z.literal("done"),
      verdict: z.enum(["violation", "no_violation"]),
      violations: z.array(eventSchema),
    }),
  ])
  .superRefine((job, ctx) => {
    if (job.status !== "done") return;
    if ((job.verdict === "no_violation") !== (job.violations.length === 0)) {
      ctx.addIssue({
        code: "custom",
        message: "Verdict and event list disagree",
      });
    }
    const ids = job.violations.flatMap((e) => (e.event_id ? [e.event_id] : []));
    if (new Set(ids).size !== ids.length)
      ctx.addIssue({ code: "custom", message: "Duplicate event IDs" });
  });
export type Job = z.infer<typeof jobSchema>;
export type CompletedJob = Extract<Job, { status: "done" }>;
export type Event = z.infer<typeof eventSchema>;
export type Frame = Event["evidence_frames"][number];
export const decisionSchema = z.object({
  status: z.enum(["accepted", "rejected", "uncertain", "unreviewed"]),
  reason: z.string().max(300).optional(),
  comment: z.string().max(2000).optional(),
});
export type Decision = z.infer<typeof decisionSchema>;
export const reviewSchema = z.object({
  version: z.number().int().nonnegative(),
  decisions: z.record(z.string(), decisionSchema),
});
export type Review = z.infer<typeof reviewSchema>;
export const statusLabels: Record<Decision["status"], string> = {
  accepted: "Подтверждено",
  rejected: "Отклонено",
  uncertain: "Недостаточно данных",
  unreviewed: "Не рассмотрено",
};
export function eventKey(event: Event, index: number) {
  return event.event_id ?? `event-${index}`;
}
export function summary(events: Event[], decisions: Review["decisions"]) {
  const counts = { accepted: 0, rejected: 0, uncertain: 0, unreviewed: 0 };
  events.forEach(
    (e, i) => counts[decisions[eventKey(e, i)]?.status ?? "unreviewed"]++,
  );
  return counts;
}
export function timecode(time: number, precise = false) {
  const safe = Number.isFinite(time) ? Math.max(0, time) : 0;
  const ms = Math.round(safe * 1000);
  const base = `${Math.floor(ms / 60000)
    .toString()
    .padStart(2, "0")}:${Math.floor((ms / 1000) % 60)
    .toString()
    .padStart(2, "0")}`;
  return precise ? `${base}.${(ms % 1000).toString().padStart(3, "0")}` : base;
}
export function safeSourceUrl(value?: string | null) {
  try {
    const url = new URL(value ?? "");
    return ["https:", "http:"].includes(url.protocol) ? url.href : undefined;
  } catch {
    return undefined;
  }
}
export function safeMediaUrl(value: string | undefined, origin: string) {
  if (!value) return undefined;
  try {
    const url = new URL(value, origin);
    return url.origin === origin && ["https:", "http:"].includes(url.protocol)
      ? url.href
      : undefined;
  } catch {
    return undefined;
  }
}
export function validateFile(
  file: Pick<File, "name" | "size">,
  maxBytes?: number,
) {
  if (!/\.(mov|mp4)$/i.test(file.name))
    return "Этот формат не поддерживается. Выберите файл .mov или .mp4.";
  if (file.size === 0) return "Файл пуст. Выберите видео с записью.";
  if (maxBytes && file.size > maxBytes)
    return `Файл слишком большой. Максимум — ${Math.floor(maxBytes / 1024 / 1024)} МБ.`;
  return null;
}
export function evidenceSrc(
  frame: Partial<Frame> & Pick<Frame, "image_base64">,
) {
  const raw = frame.image_base64;
  return raw.startsWith("/9j/") && /^[A-Za-z0-9+/=\s]+$/.test(raw)
    ? `data:image/jpeg;base64,${raw}`
    : undefined;
}
export function story(event: Event) {
  if (event.observation)
    return { seen: event.observation, why: event.rationale ?? "" };
  const body = stripCitation(event.description.trim());
  const match = body.match(/^(.+?[.!?])\s+(?=[«"A-ZА-ЯЁ])(.+)$/s);
  if (!match) return { seen: body, why: "" };
  return { seen: match[1], why: match[2].replace(/([^.!?])$/, "$1.") };
}
function stripCitation(text: string) {
  const end = text.replace(/\.$/, "");
  if (!end.endsWith(")")) return text;
  let depth = 0;
  for (let i = end.length - 1; i >= 0; i--) {
    if (end[i] === ")") depth++;
    if (end[i] === "(" && --depth === 0) return end.slice(0, i).trim();
  }
  return text;
}
export function confidencePercent(value: number) {
  return `${Math.min(99, Math.round(value * 100))}%`;
}
export function confidenceLevel(value?: number) {
  if (value === undefined) return undefined;
  if (value >= 0.8) return "высокая";
  if (value >= 0.6) return "средняя";
  return "низкая";
}
export function episodeRange(event: Event, duration: number) {
  const start = event.t_start_sec ?? event.timecode_sec;
  const end = Math.max(event.t_end_sec ?? start + 1, start + 0.4);
  return { start, end: Math.min(end, duration || end) };
}
// Reviewer feedback stored on the server (GET/PUT /app/jobs/{id}/review).
export const addedSchema = z.object({
  id: z.string().min(1).max(64),
  type_id: z.string().min(1).max(64),
  t_start_sec: seconds,
  t_end_sec: seconds.nullable(),
  comment: z.string().min(1).max(2000),
});
export type Added = z.infer<typeof addedSchema>;
export const serverReviewSchema = z.object({
  job_id: z.string().min(1),
  version: z.number().int().nonnegative(),
  added: z.array(addedSchema),
});
export type ServerReview = z.infer<typeof serverReviewSchema>;
export const catalogSchema = z.object({
  types: z.array(
    z.object({ type_id: z.string().min(1), name: z.string().min(1) }),
  ),
});
export type CatalogType = z.infer<typeof catalogSchema>["types"][number];
// Alphabetical by name, the catch-all "other" last.
export function sortCatalog(types: CatalogType[]) {
  return [...types].sort((a, b) =>
    a.type_id === "other"
      ? 1
      : b.type_id === "other"
        ? -1
        : a.name.localeCompare(b.name, "ru"),
  );
}
export function checkAdded(
  draft: { type_id: string; start: number; end?: number; comment: string },
  duration: number,
) {
  if (!draft.type_id) return "Выберите тип нарушения.";
  if (draft.end !== undefined && draft.end < draft.start)
    return "Конец не может быть раньше начала.";
  if (duration && (draft.end ?? draft.start) > duration + 1)
    return "Время выходит за пределы видео.";
  if (draft.comment.trim().length < 3)
    return "Опишите, что происходит на записи.";
  return null;
}
