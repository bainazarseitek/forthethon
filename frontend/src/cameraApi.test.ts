import { describe, expect, it } from "vitest";
import {
  cameraAlerts,
  cameraCounts,
  cameraStatusSchema,
  detectionDelay,
  previewSocketUrl,
  segmentLabel,
  type CameraStatus,
  type Segment,
} from "./cameraApi";

function segment(over: Partial<Segment>): Segment {
  return {
    job_id: "j",
    camera_id: "c1",
    clip: "Х-26",
    started_at: 1000,
    ended_at: 1020,
    duration_sec: 20,
    state: "done",
    reason: null,
    verdict: "no_violation",
    analysed_at: 1045,
    violations: [],
    ...over,
  };
}

const status: CameraStatus = cameraStatusSchema.parse({
  enabled: true,
  running: true,
  started_at: 990,
  stopped_reason: null,
  now: 1100,
  idle_stop_sec: 120,
  jury_waiting: false,
  cameras: [
    { id: "c1", name: "Касса 1", kind: "playlist", state: "live", error: null, clip: "Х-3", recording_since: 1080 },
  ],
  segments: [
    segment({ job_id: "rec", state: "recording", started_at: 1080, ended_at: null, analysed_at: null, verdict: null }),
    segment({
      job_id: "b",
      started_at: 1040,
      analysed_at: 1090,
      verdict: "violation",
      violations: [
        { index: 1, type: "Сейф открыт", timecode_sec: 12, at: 1052, has_frame: true },
        { index: 0, type: "Касса без присмотра", timecode_sec: 3, at: 1043, has_frame: true },
      ],
    }),
    segment({ job_id: "skip", state: "skipped", reason: "busy", analysed_at: null, verdict: null }),
    segment({ job_id: "a" }),
  ],
});

describe("camera alerts", () => {
  it("lists every violation on done segments, newest first, with the camera name", () => {
    const alerts = cameraAlerts(status);
    expect(alerts.map((a) => [a.segment.job_id, a.index, a.camera])).toEqual([
      ["b", 1, "Касса 1"],
      ["b", 0, "Касса 1"],
    ]);
    expect(detectionDelay(alerts[0])).toBe(38);
  });

  it("counts analysed, alerting and skipped segments", () => {
    expect(cameraCounts(status.segments)).toEqual({ analysed: 2, alerts: 1, skipped: 1 });
  });

  it("labels each segment state for the strip", () => {
    expect(status.segments.map((s) => segmentLabel(s).tone)).toEqual(["live", "alert", "off", "clean"]);
    expect(segmentLabel(segment({ state: "skipped", reason: "jury" })).text).toMatch(/API/);
  });

  it("builds the preview socket address from the page address", () => {
    expect(previewSocketUrl({ protocol: "http:", host: "1.2.3.4:10171" } as Location)).toBe(
      "ws://1.2.3.4:10171/app/cameras/preview",
    );
    expect(previewSocketUrl({ protocol: "https:", host: "x.kz" } as Location)).toBe(
      "wss://x.kz/app/cameras/preview",
    );
  });
});
