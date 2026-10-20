import { test, expect, type Page } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";

const now = Date.now() / 1000;
const camera = {
  id: "kassa-1",
  name: "Касса 1",
  kind: "playlist",
  state: "live",
  error: null,
  clip: "Х-3",
  recording_since: now - 5,
};
function segment(over: Record<string, unknown>) {
  return {
    camera_id: "kassa-1",
    clip: "Х-26",
    ended_at: now - 10,
    duration_sec: 20,
    reason: null,
    verdict: null,
    analysed_at: null,
    violations: [],
    ...over,
  };
}
const status = {
  enabled: true,
  running: true,
  started_at: now - 90,
  stopped_reason: null,
  now,
  idle_stop_sec: 120,
  jury_waiting: false,
  cameras: [camera],
  segments: [
    segment({ job_id: "rec", started_at: now - 5, ended_at: null, duration_sec: 0, state: "recording" }),
    segment({
      job_id: "seg2",
      started_at: now - 30,
      state: "done",
      verdict: "violation",
      analysed_at: now - 4,
      violations: [
        {
          index: 0,
          type: "Касса оставлена без присмотра",
          type_id: "cash_unattended",
          confidence: 0.8,
          timecode_sec: 6,
          at: now - 24,
          has_frame: true,
        },
      ],
    }),
    segment({ job_id: "seg1", started_at: now - 60, state: "skipped", reason: "busy" }),
  ],
};

async function mockCameras(page: Page, body: unknown) {
  await page.route("**/app/cameras", (route) => route.fulfill({ json: body }));
}
async function jpeg(page: Page) {
  await page.goto("/");
  const b64 = await page.evaluate(() => {
    const canvas = document.createElement("canvas");
    canvas.width = 180;
    canvas.height = 320;
    const ctx = canvas.getContext("2d")!;
    ctx.fillStyle = "#c9cfc4";
    ctx.fillRect(0, 0, 180, 320);
    return canvas.toDataURL("image/jpeg").split(",")[1];
  });
  return Buffer.from(b64, "base64");
}

test("header opens the cameras page, which starts switched off", async ({ page }) => {
  await mockCameras(page, { ...status, running: false, cameras: [{ ...camera, state: "off" }], segments: [] });
  await page.goto("/");
  await page.getByRole("link", { name: "Камеры", exact: true }).click();
  await expect(page).toHaveURL(/#\/cameras$/);
  await expect(page.getByRole("heading", { name: "Камеры не подключены" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Подключить камеры" })).toBeEnabled();
  await expect(page.getByText("Демо-режим")).toContainText("записи из датасета");
});

test("running cameras show the live picture, segments and alerts", async ({ page }) => {
  const frame = await jpeg(page);
  await mockCameras(page, status);
  await page.route("**/app/cameras/evidence/**", (route) =>
    route.fulfill({ contentType: "image/jpeg", body: frame }),
  );
  await page.routeWebSocket("**/app/cameras/preview", (ws) => {
    ws.send(Buffer.concat([Buffer.from("kassa-1\n"), frame]));
  });
  await page.goto("/#/cameras");

  const picture = page.getByAltText("Трансляция: Касса 1");
  await expect
    .poll(() => picture.evaluate((img: HTMLImageElement) => img.naturalWidth))
    .toBeGreaterThan(0);
  await expect(page.locator(".cam-osd-top")).toContainText("Касса 1");
  await expect(page.locator(".cam-osd-top")).toContainText("LIVE");
  await expect(page.locator(".cam-seg")).toHaveText([/идёт запись/, /нарушение/, /пропущен/]);

  const alert = page.locator(".alert-card");
  await expect(alert).toContainText("Касса оставлена без присмотра");
  await expect(alert).toContainText("найдено через 20 с");
  await expect(alert).toHaveAttribute("href", "#/jobs/seg2");
  await expect(page.getByRole("button", { name: "Остановить" })).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  const audit = await new AxeBuilder({ page }).include(".cams").analyze();
  expect(audit.violations.map((v) => v.id)).toEqual([]);
});

test("jury videos pause the analysis and say so", async ({ page }) => {
  await mockCameras(page, { ...status, jury_waiting: true });
  await page.goto("/#/cameras");
  await expect(page.getByRole("status").filter({ hasText: "через API" })).toBeVisible();
});
