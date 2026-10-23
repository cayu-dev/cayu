import assert from "node:assert/strict"
import test from "node:test"

globalThis.window = { __CAYU_DASHBOARD_CONFIG__: { apiBaseUrl: "/nested/api" } }
const { fetchSessionsPage } = await import("../src/lib/api.ts")
test("canonical list forwards complete selection and cancellation without losing empty continuation", async () => {
  const original = globalThis.fetch
  const signal = new AbortController().signal
  const expected = { sessions: [], overviews: [], total_count: null, next_cursor: "opaque" }
  try {
    globalThis.fetch = async (input, options) => {
      assert.equal(options.signal, signal)
      const url = new URL(input, "https://example.test")
      assert.equal(url.pathname, "/nested/api/sessions")
      assert.deepEqual(url.searchParams.getAll("overview_state"), [
        "waiting_input",
        "needs_recovery",
      ])
      assert.equal(url.searchParams.get("include_eval"), "false")
      assert.equal(url.searchParams.get("include_forks"), "false")
      assert.deepEqual(url.searchParams.getAll("label_selector"), ["!hidden", "tier in (a,b)"])
      return new Response(JSON.stringify(expected))
    }
    assert.deepEqual(
      await fetchSessionsPage(
        {
          include: ["overview"],
          overview_state: ["waiting_input", "needs_recovery"],
          include_eval: false,
          include_forks: false,
          label_selector: ["!hidden", "tier in (a,b)"],
        },
        signal,
      ),
      expected,
    )
  } finally {
    globalThis.fetch = original
  }
})
