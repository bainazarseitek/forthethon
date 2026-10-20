import { z } from "zod";
import { ApiError, parsePayload } from "./api";

const violationSchema = z.object({
  index: z.number().int().nonnegative(),
  type: z.string(),
  type_id: z.string().nullable().optional(),
  confidence: z.number().nullable().optional(),
  timecode_sec: z.number(),
  at: z.number(),
  has_frame: z.boolean(),
});
const segmentSchema = z.object({
  job_id: z.string().min(1),
  camera_id: z.string(),
  clip: z.string().nullable(),
  started_at: z.number(),
  ended_at: z.number().nullable(),
  duration_sec: z.number(),
  state: z.enum(["recording", "queued", "running", "done", "failed", "skipped"]),
  reason: z.string().nullable(),
  verdict: z.string().nullable(),
  analysed_at: z.number().nullable(),
  violations: z.array(violationSchema),
});
const cameraSchema = z.object({
  id: z.string(),
  name: z.string(),
  kind: z.string(),
  state: z.string(),
  error: z.string().nullable(),
  clip: z.string().nullable(),
  recording_since: z.number().nullable(),
});
export const cameraStatusSchema = z.object({
  enabled: z.boolean(),
  running: z.boolean(),
  started_at: z.number().nullable(),
  stopped_reason: z.string().nullable(),
  now: z.number(),
  idle_stop_sec: z.number(),
  jury_waiting: z.boolean(),
  cameras: z.array(cameraSchema),
  segments: z.array(segmentSchema),
});
export type CameraStatus = z.infer<typeof cameraStatusSchema>;
export type Segment = z.infer<typeof segmentSchema>;
export type Camera = z.infer<typeof cameraSchema>;

const messages: Record<string, string> = {
  cameras_disabled: "Трансляция камер отключена на этом сервере.",
  cameras_unavailable:
    "Список камер на сервере не найден или содержит ошибку. Проверьте data/cameras.json.",
};

async function request(path: string, method = "GET", signal?: AbortSignal) {
  let response: Response;
  try {
    response = await fetch(`/app/cameras${path}`, {
      method,
      cache: "no-store",
      signal: signal
        ? AbortSignal.any([signal, AbortSignal.timeout(15_000)])
        : AbortSignal.timeout(15_000),
      headers: { Accept: "application/json" },
    });
  } catch {
    throw new ApiError("Сервер недоступен. Проверьте соединение.");
  }
  const body: unknown = await response.json().catch(() => undefined);
  if (!response.ok) {
    const code = (body as { error?: string } | undefined)?.error ?? "";
    throw new ApiError(
      messages[code] ?? "Сервер не смог выполнить запрос к камерам.",
      response.status,
    );
  }
  return parsePayload(body, cameraStatusSchema);
}
export const getCameras = (signal?: AbortSignal) => request("", "GET", signal);
export const startCameras = () => request("/start", "POST");
export const stopCameras = () => request("/stop", "POST");

export function previewSocketUrl(location: Location) {
  return `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/app/cameras/preview`;
}

export function clock(epochSec: number) {
  return new Date(epochSec * 1000).toLocaleTimeString("ru-RU", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
}

export type Alert = Segment["violations"][number] & {
  segment: Segment;
  camera: string;
};

// Every violation found on any camera, newest first.
export function cameraAlerts(status: CameraStatus): Alert[] {
  const names = new Map(status.cameras.map((c) => [c.id, c.name]));
  return status.segments
    .filter((s) => s.state === "done")
    .flatMap((segment) =>
      segment.violations.map((v) => ({
        ...v,
        segment,
        camera: names.get(segment.camera_id) ?? segment.camera_id,
      })),
    )
    .sort((a, b) => b.at - a.at);
}

export type Tone = "live" | "wait" | "work" | "clean" | "alert" | "off";

export function segmentLabel(s: Segment): { text: string; tone: Tone } {
  switch (s.state) {
    case "recording":
      return { text: "идёт запись", tone: "live" };
    case "queued":
      return { text: "ждёт анализа", tone: "wait" };
    case "running":
      return { text: "анализ", tone: "work" };
    case "done":
      if (!s.violations.length) return { text: "чисто", tone: "clean" };
      return {
        text:
          s.violations.length === 1
            ? "нарушение"
            : `нарушений: ${s.violations.length}`,
        tone: "alert",
      };
    case "skipped":
      return {
        text:
          s.reason === "jury"
            ? "пропущен: идёт проверка через API"
            : "пропущен: модель занята",
        tone: "off",
      };
    default:
      return { text: "не проанализирован", tone: "off" };
  }
}

export function cameraCounts(segments: Segment[]) {
  return {
    analysed: segments.filter((s) => s.state === "done").length,
    alerts: segments.filter((s) => s.state === "done" && s.violations.length)
      .length,
    skipped: segments.filter((s) => s.state === "skipped").length,
  };
}

// Seconds from the moment of the violation to the model's verdict.
export function detectionDelay(alert: Alert) {
  return alert.segment.analysed_at === null
    ? undefined
    : Math.max(0, Math.round(alert.segment.analysed_at - alert.at));
}
