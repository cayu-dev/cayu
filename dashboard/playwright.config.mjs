import { defineConfig, devices } from "@playwright/test"

// Browser checks for the prototype build. `npm run test:prototype` builds it into
// dist/prototype (CAYU_PROTOTYPE=1) first; this config only serves and drives it.
const PORT = 4178

export default defineConfig({
  testDir: "tests",
  testMatch: "*.spec.mjs",
  outputDir: "dist/test-results",
  workers: 1,
  forbidOnly: !!process.env.CI,
  reporter: "list",
  use: { baseURL: `http://127.0.0.1:${PORT}`, trace: "retain-on-failure" },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: {
    command: `npx vite preview --host 127.0.0.1 --port ${PORT} --strictPort`,
    env: { CAYU_PROTOTYPE: "1" },
    url: `http://127.0.0.1:${PORT}/prototype.html`,
    reuseExistingServer: false,
  },
})
