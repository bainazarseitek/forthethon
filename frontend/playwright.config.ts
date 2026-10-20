import { defineConfig, devices } from "@playwright/test";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const port = Number(process.env.E2E_PORT ?? 8092);
const dataDir = mkdtempSync(join(tmpdir(), "forte-e2e-"));
export default defineConfig({
  testDir: "./tests",
  fullyParallel: true,
  retries: 0,
  workers: 3,
  reporter: [["list"], ["html", { open: "never" }]],
  use: {
    baseURL: `http://127.0.0.1:${port}`,
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  projects: [
    {
      name: "chromium",
      use: {
        ...devices["Desktop Chrome"],
        viewport: { width: 1440, height: 1000 },
      },
    },
    {
      name: "mobile",
      use: { ...devices["iPhone 13"], defaultBrowserType: "chromium" },
    },
  ],
  webServer: {
    command: `uv run uvicorn app.main:create_app --factory --port ${port}`,
    cwd: "..",
    env: { API_KEY: "e2e", DATA_DIR: dataDir, APP_QUEUE_CAP: "20" },
    url: `http://127.0.0.1:${port}/healthz`,
    reuseExistingServer: false,
  },
});
