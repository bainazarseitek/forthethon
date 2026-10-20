import { test, expect, type Page } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import { existsSync } from "node:fs";

const sample = process.env.SAMPLE_VIDEO ?? "../assets/videos/X-1.mov";
const rule = {
  document:
    "Правила осуществления кассовых операций и операций по инкассации банкнот, монет и ценностей, постановление правления НБРК от 29.11.2019 № 231",
  clause: "п. 76, п. 80",
  url: "https://adilet.zan.kz/rus/docs/V1900019680",
};
const event = {
  type: "Ключ оставлен в замке сейфа",
  type_id: "key_left_in_safe",
  timecode_sec: 22.033,
  description: "Синтетическое описание для теста интерфейса.",
  evidence_frames: [],
  confidence: 0.7,
  t_start_sec: 22,
  t_end_sec: 23,
  rule,
  boxes: [],
};
const done = {
  job_id: "test",
  status: "done",
  verdict: "violation",
  violations: [event],
  queue_position: 0,
  progress: 1,
  filename: "X-1.mov",
  duration_sec: 25.9,
};
async function mockJob(page: Page, job: unknown, status = 200) {
  await page.route("**/app/jobs/test", (route) =>
    route.fulfill({ status, json: job }),
  );
}
async function noOverflow(page: Page) {
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBe(true);
}
async function syntheticJpeg(page: Page) {
  return page.evaluate(() => {
    const canvas = document.createElement("canvas");
    canvas.width = 240;
    canvas.height = 320;
    const ctx = canvas.getContext("2d")!;
    ctx.fillStyle = "#d9ded4";
    ctx.fillRect(0, 0, 240, 320);
    return canvas.toDataURL("image/jpeg").split(",")[1];
  });
}

test("upload page validates files and explains the tool", async ({
  page,
}, info) => {
  await page.goto("/");
  await page.getByLabel("Видео для проверки").setInputFiles({
    name: "wrong.avi",
    mimeType: "video/x-msvideo",
    buffer: Buffer.from("test"),
  });
  await expect(page.getByRole("alert")).toContainText(".mov или .mp4");
  await expect(page.locator(".drop-hint")).toContainText("2048 МБ");
  await noOverflow(page);
  await page.screenshot({
    path: info.outputPath("upload.png"),
    fullPage: true,
  });
});

test("real backend: upload, poll, verdict, evidence and seeking", async ({
  page,
}, info) => {
  test.skip(!existsSync(sample), `sample video missing: ${sample}`);
  test.setTimeout(120_000);
  await page.goto("/");
  await page.getByLabel("Видео для проверки").setInputFiles(sample);
  await expect(page).toHaveURL(/#\/jobs\/[0-9a-f]+/, { timeout: 60_000 });
  const heading = page.getByRole("heading", { level: 1 });
  await expect(heading).toHaveText(
    /^(Обнаружены признаки возможных нарушений|Нарушение не обнаружено)$/,
    { timeout: 60_000 },
  );
  await expect(page.locator(".verdict-sub")).toContainText(
    "не юридическое заключение",
  );
  const events = page.locator(".index-item");
  if ((await heading.textContent())?.startsWith("Нарушение")) {
    await expect(events).toHaveCount(0);
    return;
  }
  await expect(page.locator(".trail-file")).toContainText("X-1.mov");
  await expect(page.locator(".finding .frame-btn").first()).toBeVisible();
  await expect(page.locator(".finding .frame-btn .box").first()).toBeVisible();
  await expect(
    page.getByRole("link", { name: /Открыть на adilet.zan.kz/ }),
  ).toHaveAttribute("href", /^https:\/\/adilet\.zan\.kz\//);
  const video = page.locator("video");
  await expect
    .poll(() => video.evaluate((v: HTMLVideoElement) => v.readyState))
    .toBeGreaterThan(0);
  const count = await events.count();
  await events.nth(count - 1).click();
  const target = await events
    .nth(count - 1)
    .locator(".mono")
    .textContent();
  const [m, s] = (target ?? "0:0").split(":").map(Number);
  await expect
    .poll(() => video.evaluate((v: HTMLVideoElement) => v.currentTime))
    .toBeCloseTo(m * 60 + s, 0);
  await expect(page.locator(".episode-bar")).toContainText(
    `${String(count).padStart(2, "0")}/`,
  );
  await expect(page.locator(".finding .story p").first()).toHaveText(
    /^[^a-zа-яё]/,
  );
  await page.getByRole("button", { name: "Подтверждаю" }).click();
  await page.screenshot({
    path: info.outputPath("result.png"),
    fullPage: true,
  });
  await page.reload();
  await events.nth(count - 1).click();
  await expect(
    page.getByRole("button", { name: "Подтверждаю" }),
  ).toHaveAttribute("aria-pressed", "true");
  await noOverflow(page);
  await page.getByRole("link", { name: "Отчёт" }).click();
  await expect(page.locator(".sheet")).toHaveCount(count + 1);
  await expect(page.locator(".cover")).toContainText("подтверждено 1");
});

for (const [status, error, text] of [
  [429, "queue_full", "в очереди уже несколько видео"],
  [503, "uploads_disabled", "временно отключена"],
  [413, "file_too_large", "Максимум 2048 МБ"],
  [400, "invalid_video", "не смог прочитать видео"],
] as const) {
  test(`upload error ${status} shows a Russian message`, async ({ page }) => {
    await page.route("**/app/analyze", (route) =>
      route.fulfill({ status, json: { error, detail: "English detail" } }),
    );
    await page.goto("/");
    await page.getByLabel("Видео для проверки").setInputFiles({
      name: "test.mov",
      mimeType: "",
      buffer: Buffer.from("test"),
    });
    await expect(page.getByRole("alert")).toContainText(text);
    await expect(page).not.toHaveURL(/jobs/);
  });
}

test("queued job shows its position", async ({ page }) => {
  await mockJob(page, {
    job_id: "test",
    status: "processing",
    queue_position: 2,
    progress: 0,
  });
  await page.goto("/#/jobs/test");
  await expect(page.locator(".stages")).toContainText("перед вами 2");
});

test("failed job explains the error and is not a clean result", async ({
  page,
}, info) => {
  await mockJob(page, {
    job_id: "test",
    status: "failed",
    error: "timeout",
    message: "The detector did not finish in time and was stopped.",
    queue_position: 0,
    progress: 0.05,
  });
  await page.goto("/#/jobs/test");
  await expect(page.getByRole("heading", { level: 1 })).toHaveText(
    "Видео не удалось проанализировать",
  );
  await expect(page.locator(".failure p")).toContainText(
    "не успела обработать",
  );
  await page.screenshot({
    path: info.outputPath("failed.png"),
    fullPage: true,
  });
});

test("unknown job and malformed response are reported", async ({ page }) => {
  await mockJob(
    page,
    { error: "job_not_found", detail: "No job with this id." },
    404,
  );
  await page.goto("/#/jobs/test");
  await expect(page.getByRole("alert")).toContainText("хранятся 24 часа");
  await expect(
    page.getByRole("link", { name: "Новая проверка" }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Повторить запрос" }),
  ).toHaveCount(0);
  await page.unroute("**/app/jobs/test");
  await mockJob(page, {
    job_id: "test",
    status: "done",
    verdict: "no_violation",
  });
  await page.reload();
  await expect(page.getByRole("alert")).toContainText("контракту");
});

test("clean result explains its limits", async ({ page }, info) => {
  await mockJob(page, { ...done, verdict: "no_violation", violations: [] });
  await page.goto("/#/jobs/test");
  await expect(page.getByRole("heading", { level: 1 })).toHaveText(
    "Нарушение не обнаружено",
  );
  await expect(
    page.getByText("Модель может ошибаться", { exact: false }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Подтверждаю" }),
  ).not.toBeVisible();
  await noOverflow(page);
  await page.screenshot({ path: info.outputPath("clean.png"), fullPage: true });
});

test("many violations with long texts keep the layout", async ({
  page,
}, info) => {
  await page.goto("/");
  const image = await syntheticJpeg(page);
  const long =
    "Очень длинное описание наблюдения, которое продолжается и продолжается, ".repeat(
      8,
    );
  await mockJob(page, {
    ...done,
    filename: "Очень-длинное-имя-файла-с-записью-кассы-отделения-номер-17.mov",
    violations: Array.from({ length: 9 }, (_, i) => ({
      ...event,
      type:
        i % 2
          ? "Содержимое сумки убрано без пересчёта на счётной машине и без внесения в систему"
          : "Иное нарушение",
      type_id: i % 2 ? "bag_not_counted" : "other",
      timecode_sec: i * 2.5,
      description: long,
      evidence_frames: [0, 1, 2].map((k) => ({
        timestamp_sec: i * 2.5 + k / 3,
        image_base64: image,
      })),
      boxes: [0, 1, 2].map((k) => ({
        timestamp_sec: i * 2.5 + k / 3,
        boxes: [{ x: 0.1, y: 0.4, w: 0.8, h: 0.2, label: "купюры" }],
      })),
      rule:
        i % 2
          ? { document: "внутренние правила банка", clause: null, url: null }
          : rule,
    })),
  });
  await page.goto("/#/jobs/test");
  await expect(page.locator(".index-item")).toHaveCount(9);
  await expect(page.locator(".finding .frame-btn .box")).toHaveCount(3);
  await expect(page.getByText("Тип вне каталога правил")).toBeVisible();
  await noOverflow(page);
  await page.screenshot({ path: info.outputPath("many.png"), fullPage: true });
  await page.locator(".index-item").nth(1).click();
  await expect(page.locator(".finding .citation")).toContainText(
    "Внутренний документ банка",
  );
  await page.locator(".finding .frame-btn").first().click();
  await expect(page.getByRole("dialog")).toContainText("00:02.500");
  await expect(page.getByRole("dialog").locator(".box")).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(page.locator(".finding .frame-btn").first()).toBeFocused();
});

test("unsafe source links are not rendered", async ({ page }) => {
  await mockJob(page, {
    ...done,
    violations: [{ ...event, rule: { ...rule, url: "javascript:alert(1)" } }],
  });
  await page.goto("/#/jobs/test");
  await expect(page.locator(".finding")).toBeVisible();
  await expect(
    page.getByRole("link", { name: /Открыть на adilet/ }),
  ).not.toBeVisible();
});

test("accessibility audit on upload and review", async ({ page }) => {
  await page.goto("/");
  await page.waitForTimeout(800);
  let audit = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa", "wcag21aa", "wcag22aa"])
    .analyze();
  expect(audit.violations).toEqual([]);
  await mockJob(page, done);
  await page.goto("/#/jobs/test");
  await expect(page.locator(".finding")).toBeVisible();
  await page.waitForTimeout(800);
  audit = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa", "wcag21aa", "wcag22aa"])
    .analyze();
  expect(audit.violations).toEqual([]);
});

test("320px reflow keeps controls and skip link works", async ({
  page,
}, info) => {
  test.skip(info.project.name !== "chromium", "desktop project only");
  await mockJob(page, done);
  await page.setViewportSize({ width: 320, height: 800 });
  await page.goto("/#/jobs/test");
  await expect(page.getByRole("button", { name: "Подтверждаю" })).toBeVisible();
  await noOverflow(page);
  await page.screenshot({
    path: info.outputPath("review-320.png"),
    fullPage: true,
  });
  await page.keyboard.press("Tab");
  const skip = page.getByRole("link", { name: "Перейти к содержимому" });
  await expect(skip).toBeFocused();
  await page.keyboard.press("Enter");
  await expect(page).toHaveURL(/#\/jobs\/test$/);
  await expect(page.locator("#main")).toBeFocused();
});

test("episode bar and control row page violations, keys seek", async ({
  page,
}, info) => {
  await mockJob(page, {
    ...done,
    violations: [
      { ...event, timecode_sec: 4, t_start_sec: 3.5, t_end_sec: 5 },
      { ...event, t_start_sec: 22.033, t_end_sec: 22.033 },
    ],
  });
  await page.goto("/#/jobs/test");
  const bar = page.locator(".episode-bar");
  await expect(bar).toContainText("01/02");
  await expect(page.locator(".toast")).toHaveCount(0);
  await page.evaluate(() =>
    document.body.dispatchEvent(
      new KeyboardEvent("keydown", { key: "т", bubbles: true }),
    ),
  );
  await expect(bar).toContainText("02/02");
  await expect(bar).toContainText("00:22.033");
  if (info.project.name === "chromium") {
    await expect(bar.locator(".episode-bar-tc span")).toHaveText("22.0 с");
  }
  await bar.getByRole("button", { name: "Предыдущее нарушение (P)" }).click();
  await expect(bar).toContainText("01/02");
  const row = page.getByRole("toolbar", { name: "Управление видео" });
  await expect(row.getByRole("button")).toHaveCount(
    info.project.name === "chromium" ? 6 : 4,
  );
  await expect(
    row.getByRole("button", { name: "Добавить нарушение" }),
  ).toBeVisible();
  if (info.project.name === "chromium") {
    await row.getByRole("button", { name: "Следующее нарушение" }).click();
    await expect(bar).toContainText("02/02");
    await row.getByRole("button", { name: "Предыдущее нарушение" }).click();
    await expect(bar).toContainText("01/02");
  }
  const tc = page.locator(".stage-tc");
  await expect(tc).toHaveText("00:04");
  await page.keyboard.press("l");
  await expect(tc).toHaveText("00:09");
  await page.keyboard.press("j");
  await expect(tc).toHaveText("00:04");
  await page.keyboard.press("ArrowRight");
  await expect(tc).toHaveText("00:05");
  await page.keyboard.press("?");
  await expect(page.getByRole("dialog")).toContainText("Кадр назад / вперёд");
});

test("lightbox hides the boxes toggle when frames have no boxes", async ({
  page,
}) => {
  await page.goto("/");
  const image = await syntheticJpeg(page);
  await mockJob(page, {
    ...done,
    violations: [
      {
        ...event,
        evidence_frames: [{ timestamp_sec: 22, image_base64: image }],
      },
    ],
  });
  await page.goto("/#/jobs/test");
  await page.locator(".finding .frame-btn").first().click();
  await expect(page.getByRole("dialog")).toContainText("Показать в видео");
  await expect(
    page.getByRole("dialog").getByRole("button", { name: /рамки/ }),
  ).toHaveCount(0);
});

test("lightbox frame fills the viewport height and zooms on click", async ({
  page,
}) => {
  await page.goto("/");
  const image = await syntheticJpeg(page);
  await mockJob(page, {
    ...done,
    violations: [
      {
        ...event,
        evidence_frames: [{ timestamp_sec: 22, image_base64: image }],
      },
    ],
  });
  await page.goto("/#/jobs/test");
  await page.locator(".finding .frame-btn").first().click();
  const img = page.getByRole("dialog").locator(".lightbox-frame img");
  await expect(img).toBeVisible();
  await expect
    .poll(async () => (await img.boundingBox())!.height)
    .toBeGreaterThan(page.viewportSize()!.height * 0.65);
  const box = (await img.boundingBox())!;
  expect(box.width / box.height).toBeCloseTo(240 / 320, 2);
  await page.mouse.click(box.x + box.width / 2, box.y + box.height / 2);
  await expect(page.locator(".lightbox-frame")).toHaveCSS(
    "transform",
    "matrix(2, 0, 0, 2, 0, 0)",
  );
});

test("portrait video switches to the side-by-side layout", async ({
  page,
}, info) => {
  test.skip(!existsSync(sample), `sample video missing: ${sample}`);
  await mockJob(page, done);
  await page.route("**/app/jobs/test/video", (route) =>
    route.fulfill({ path: sample, contentType: "video/quicktime" }),
  );
  await page.goto("/#/jobs/test");
  await expect(page.locator(".review")).toHaveAttribute(
    "data-orientation",
    "portrait",
  );
  if (info.project.name !== "chromium") return;
  const stage = (await page.locator(".stage").boundingBox())!;
  const finding = (await page.locator(".finding-col").boundingBox())!;
  expect(finding.y).toBeCloseTo(stage.y, 0);
  expect(finding.x).toBeGreaterThan(stage.x + stage.width);
  expect(stage.width / stage.height).toBeLessThan(1);
  await noOverflow(page);
});

test("reviewer adds a missed violation, it is saved on the server", async ({
  page,
}) => {
  test.skip(!existsSync(sample), `sample video missing: ${sample}`);
  test.setTimeout(120_000);
  await page.goto("/");
  await page.getByLabel("Видео для проверки").setInputFiles(sample);
  await expect(page).toHaveURL(/#\/jobs\/[0-9a-f]+/, { timeout: 60_000 });
  await expect(page.getByRole("heading", { level: 1 })).toBeVisible({
    timeout: 60_000,
  });
  const add = page.getByRole("button", { name: /Добавить нарушение/ }).first();
  await add.click();
  await expect(
    page.getByRole("heading", { name: "Добавить нарушение" }),
  ).toBeVisible();
  const audit = await new AxeBuilder({ page })
    .include(".add-violation")
    .withTags(["wcag2a", "wcag2aa", "wcag21aa", "wcag22aa"])
    .analyze();
  expect(audit.violations).toEqual([]);
  await page.getByRole("button", { name: "Сохранить нарушение" }).click();
  await expect(page.getByRole("alert")).toContainText("Выберите тип");
  await page
    .getByLabel("Тип нарушения")
    .selectOption({ label: "Наличные оставлены без присмотра" });
  await expect(page.getByRole("alert")).toHaveCount(0);
  await page
    .getByLabel("Что происходит")
    .fill("Кассир вышел, пачка купюр осталась на столе");
  await page.getByRole("button", { name: "Сохранить нарушение" }).click();
  const added = page.getByRole("navigation", { name: "Добавлено проверяющим" });
  await expect(added).toContainText("Наличные оставлены без присмотра");
  await expect(page.locator(".pin-added")).toHaveCount(1);
  await noOverflow(page);
  await page.reload();
  await expect(added.locator(".added-item")).toHaveCount(1);
  await page.getByRole("button", { name: /^Удалить:/ }).click();
  await page.getByRole("button", { name: "Удалить", exact: true }).click();
  await expect(added).toHaveCount(0);
});

test("clean result offers to add a violation the model missed", async ({
  page,
}) => {
  await mockJob(page, { ...done, verdict: "no_violation", violations: [] });
  await page.route("**/app/jobs/test/review", (route) =>
    route.fulfill({
      json: { job_id: "test", version: 0, decisions: {}, added: [], updated_at: null },
    }),
  );
  await page.goto("/#/jobs/test");
  await expect(
    page.getByText("Видите нарушение, которое модель пропустила?"),
  ).toBeVisible();
  await page
    .locator(".finding-clean")
    .getByRole("button", { name: "Добавить нарушение" })
    .click();
  await expect(page.getByLabel("Тип нарушения")).toBeVisible();
  await noOverflow(page);
});
