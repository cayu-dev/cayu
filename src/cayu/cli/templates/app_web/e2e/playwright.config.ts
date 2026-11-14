import { randomBytes } from "node:crypto";
import { existsSync } from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";
import { defineConfig } from "@playwright/test";

/*
 * GENERATED. The smoke test runs against the project's own `cayu serve --dev`, which serves the
 * committed web/static at / and the Developer dashboard at /cayu/, on a port of its own
 * (SMOKE_PORT, else 4317). Run `uv sync` in the project first.
 */
const project = fileURLToPath(new URL("../..", import.meta.url));
const port = Number(process.env.SMOKE_PORT ?? 4317);
const environment: Record<string, string> = {};
for (const [name, value] of Object.entries(process.env)) if (value !== undefined) environment[name] = value;
// With automatic memory, the app needs a memory evidence key. A fresh checkout, such as a CI
// job, has neither CAYU_MEMORY_EVIDENCE_KEY nor the git-ignored data/memory-evidence.key, so
// the server gets a throwaway one, as the project's tests do.
if (!environment.CAYU_MEMORY_EVIDENCE_KEY && !existsSync(path.join(project, "data", "memory-evidence.key"))) {
  environment.CAYU_MEMORY_EVIDENCE_KEY = randomBytes(48).toString("base64url");
}

export default defineConfig({
  testDir: ".",
  forbidOnly: Boolean(process.env.CI),
  workers: 1,
  reporter: "list",
  use: { baseURL: `http://127.0.0.1:${port}`, browserName: "chromium" },
  webServer: {
    command: `uv run --no-sync cayu serve --dev --host 127.0.0.1 --port ${port}`,
    cwd: project,
    url: `http://127.0.0.1:${port}/`,
    env: environment,
    reuseExistingServer: false,
    timeout: 120_000,
    stdout: "ignore",
    stderr: "pipe",
  },
});
