import { describe, expect, it } from "vitest";
import {
  boxSchema,
  confidenceLevel,
  confidencePercent,
  story,
  evidenceSrc,
  jobSchema,
  safeMediaUrl,
  safeSourceUrl,
  summary,
  timecode,
  validateFile,
  type CompletedJob, checkAdded, sortCatalog } from "./domain";
import { parsePayload } from "./api";

const job: CompletedJob = {
  job_id: "test",
  status: "done",
  verdict: "violation",
  violations: [
    {
      event_id: "a",
      type: "Synthetic event",
      timecode_sec: 22.033,
      description: "Test only",
      evidence_frames: [],
    },
  ],
};
describe("jury response boundary", () => {
  it("accepts the minimum jury shape without invented extensions", () =>
    expect(jobSchema.parse(job)).toEqual(job));
  it("accepts processing and failure as separate states", () => {
    expect(
      jobSchema.parse({ job_id: "test", status: "processing" }).status,
    ).toBe("processing");
    expect(jobSchema.parse({ job_id: "test", status: "failed" }).status).toBe(
      "failed",
    );
  });
  it("rejects missing violations instead of silently showing no events", () =>
    expect(
      jobSchema.safeParse({
        job_id: "test",
        status: "done",
        verdict: "no_violation",
      }).success,
    ).toBe(false));
  it("rejects inconsistent verdicts", () =>
    expect(
      jobSchema.safeParse({ ...job, verdict: "no_violation" }).success,
    ).toBe(false));
  it("rejects a positive result with zero events", () =>
    expect(jobSchema.safeParse({ ...job, violations: [] }).success).toBe(
      false,
    ));
  it("rejects duplicate event IDs", () =>
    expect(
      jobSchema.safeParse({
        ...job,
        violations: [...job.violations, ...job.violations],
      }).success,
    ).toBe(false));
  it("rejects invalid timestamps", () =>
    expect(
      jobSchema.safeParse({
        ...job,
        violations: [{ ...job.violations[0], timecode_sec: -2 }],
      }).success,
    ).toBe(false));
  it("reports malformed responses with a user-safe error", () =>
    expect(() => parsePayload("<html>not API</html>", jobSchema)).toThrow(
      "контракту",
    ));
});
describe("review isolation", () => {
  it("counts decisions without modifying predictions", () => {
    const before = JSON.stringify(job);
    expect(
      summary(job.violations, {
        a: { status: "rejected" },
        stale: { status: "accepted" },
      }),
    ).toEqual({ accepted: 0, rejected: 1, uncertain: 0, unreviewed: 0 });
    expect(JSON.stringify(job)).toBe(before);
    expect(job.verdict).toBe("violation");
  });
  it("keeps uncertainty separate from acceptance and missing decisions", () => {
    expect(summary(job.violations, {}).unreviewed).toBe(1);
    expect(
      summary(job.violations, { a: { status: "uncertain" } }).uncertain,
    ).toBe(1);
  });
});
describe("untrusted data and media", () => {
  it("rejects script and malformed source URLs", () => {
    expect(safeSourceUrl("javascript:alert(1)")).toBeUndefined();
    expect(safeSourceUrl("data:text/html,x")).toBeUndefined();
    expect(safeSourceUrl("not a link")).toBeUndefined();
    expect(safeSourceUrl("https://adilet.zan.kz/rus/docs/example")).toContain(
      "https://adilet",
    );
  });
  it("permits only same-origin media", () => {
    expect(safeMediaUrl("/app/jobs/test/video", "https://team.test")).toBe(
      "https://team.test/app/jobs/test/video",
    );
    expect(
      safeMediaUrl("https://external.test/video.mov", "https://team.test"),
    ).toBeUndefined();
    expect(
      safeMediaUrl("javascript:alert(1)", "https://team.test"),
    ).toBeUndefined();
  });
  it("never treats SVG or external links as JPEG evidence", () => {
    expect(
      evidenceSrc({
        timestamp_sec: 0,
        image_base64: "data:image/svg+xml,evil",
      }),
    ).toBeUndefined();
    expect(
      evidenceSrc({
        timestamp_sec: 0,
        image_base64: "https://external.test/a.jpg",
      }),
    ).toBeUndefined();
    expect(evidenceSrc({ timestamp_sec: 0, image_base64: "/9j/AA==" })).toBe(
      "data:image/jpeg;base64,/9j/AA==",
    );
  });
  it("attaches per-frame boxes from the backend to their frames", () => {
    const box = { x: 0.1, y: 0.2, w: 0.3, h: 0.4, label: "купюры" };
    const parsed = jobSchema.parse({
      ...job,
      violations: [
        {
          ...job.violations[0],
          evidence_frames: [
            { timestamp_sec: 1, image_base64: "/9j/AA==" },
            { timestamp_sec: 2, image_base64: "/9j/AA==" },
          ],
          boxes: [
            { timestamp_sec: 1, boxes: [box] },
            { timestamp_sec: 2, boxes: [] },
          ],
        },
      ],
    });
    if (parsed.status !== "done") throw new Error("unexpected status");
    expect(parsed.violations[0].evidence_frames.map((f) => f.boxes)).toEqual([
      [box],
      [],
    ]);
    expect(boxSchema.safeParse(box).success).toBe(true);
  });
});
describe("upload and time", () => {
  it("allows MOV and MP4 irrespective of filename case", () => {
    expect(validateFile({ name: "касса.MOV", size: 200 })).toBeNull();
    expect(validateFile({ name: "video.mp4", size: 200 })).toBeNull();
  });
  it("rejects other types and empty files", () => {
    expect(validateFile({ name: "video.avi", size: 200 })).toContain(".mov");
    expect(validateFile({ name: "empty.mov", size: 0 })).toContain("пуст");
  });
  it("only applies a size ceiling when configured", () => {
    expect(validateFile({ name: "file.mov", size: 80_000_000 })).toBeNull();
    expect(
      validateFile({ name: "file.mov", size: 80_000_000 }, 40_000_000),
    ).toContain("большой");
  });
  it("preserves millisecond timestamps without FPS assumptions", () => {
    expect(timecode(22.033, true)).toBe("00:22.033");
    expect(timecode(59.9999, true)).toBe("01:00.000");
    expect(timecode(145)).toBe("02:25");
    expect(timecode(NaN)).toBe("00:00");
  });
});
describe("finding story", () => {
  const base = { type: "t", timecode_sec: 1, evidence_frames: [] };
  it("splits observation, rule and drops the bracketed citation", () => {
    const s = story({
      ...base,
      description:
        "Кассир передаёт пачку купюр клиенту. Наличные выдаются только через окно (Правила № 231, п. 19)",
    });
    expect(s.seen).toBe("Кассир передаёт пачку купюр клиенту.");
    expect(s.why).toBe("Наличные выдаются только через окно.");
  });
  it("drops a citation with nested brackets", () =>
    expect(
      story({
        ...base,
        description:
          "Кассир передаёт пачку. Касса оборудуется лотком (Правила № 14, п. 16 пп. 4 (в редакции до 01.01.2027); Правила № 231, п. 20)",
      }).why,
    ).toBe("Касса оборудуется лотком."));
  it("keeps a single sentence as the observation", () =>
    expect(story({ ...base, description: "Ключ в замке сейфа" })).toEqual({
      seen: "Ключ в замке сейфа",
      why: "",
    }));
  it("prefers explicit observation fields", () =>
    expect(
      story({ ...base, description: "x", observation: "o", rationale: "r" }),
    ).toEqual({ seen: "o", why: "r" }));
  it("labels confidence", () => {
    expect(confidenceLevel(0.85)).toBe("высокая");
    expect(confidenceLevel(0.62)).toBe("средняя");
    expect(confidenceLevel(0.3)).toBe("низкая");
  });
  it("never shows full certainty", () => {
    expect(confidencePercent(1)).toBe("99%");
    expect(confidencePercent(0.996)).toBe("99%");
    expect(confidencePercent(0.854)).toBe("85%");
    expect(confidencePercent(0)).toBe("0%");
  });
});

describe("reviewer-added violations", () => {
  const draft = { type_id: "cash_unattended", start: 9, comment: "пачка купюр на столе" };
  it("accepts a complete draft", () => {
    expect(checkAdded(draft, 20)).toBeNull();
    expect(checkAdded({ ...draft, end: 18 }, 20)).toBeNull();
  });
  it("explains what is missing", () => {
    expect(checkAdded({ ...draft, type_id: "" }, 20)).toMatch(/тип/);
    expect(checkAdded({ ...draft, comment: "  " }, 20)).toMatch(/Опишите/);
    expect(checkAdded({ ...draft, end: 5 }, 20)).toMatch(/раньше начала/);
    expect(checkAdded({ ...draft, start: 30 }, 20)).toMatch(/пределы/);
  });
  it("sorts the catalog by name with «other» last", () => {
    const sorted = sortCatalog([
      { type_id: "other", name: "Возможное нарушение" },
      { type_id: "b", name: "Сейф оставлен открытым" },
      { type_id: "a", name: "Дверь кассы не закрыта" },
    ]);
    expect(sorted.map((t) => t.type_id)).toEqual(["a", "b", "other"]);
  });
});
