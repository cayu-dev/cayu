import { onTestFinished, vi } from "vite-plus/test"

// Fake setTimeout and clearTimeout for one test.
export function useFakeTimeouts() {
  vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] })
  onTestFinished(() => vi.useRealTimers())
}
