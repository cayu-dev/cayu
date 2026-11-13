import { defineConfig, devices } from "@playwright/test"

// Serves the consumer's production build (`vite build`) and drives it in one browser.
const PORT = 4179

export default defineConfig({
  testDir: ".",
  testMatch: "consumer.spec.mjs",
  outputDir: "test-results",
  workers: 1,
  reporter: "list",
  use: { baseURL: `http://127.0.0.1:${PORT}` },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: {
    command: `npx vite preview --host 127.0.0.1 --port ${PORT} --strictPort`,
    url: `http://127.0.0.1:${PORT}/`,
    reuseExistingServer: false,
  },
})
