import assert from "node:assert/strict"
import test from "node:test"
import { fileURLToPath } from "node:url"
import { rolldown } from "rolldown"

const bundle = await rolldown({
  cwd: fileURLToPath(new URL("..", import.meta.url)),
  input: "src/lib/session-message-queue.ts",
  platform: "node",
})
const output = await bundle.generate({ format: "esm", codeSplitting: false })
await bundle.close()
globalThis.window = { __CAYU_DASHBOARD_CONFIG__: { apiBaseUrl: "/api" } }
const { SessionMessageAcceptance } = await import(
  `data:text/javascript;base64,${Buffer.from(output.output[0].code).toString("base64")}`
)

test("lost acceptance retries immutable content and key without Last-Event-ID", async () => {
  const requests = []
  const owner = new SessionMessageAcceptance("s", "message", "key", {
    attempt: async (request, callbacks) => {
      requests.push(request)
      if (requests.length === 1)
        return { kind: "network_failed", accepted: true, error: new Error("lost") }
      callbacks.onEvent(
        { type: "session.message.queued", session_id: "s", payload: { queue_id: "q" } },
        "event",
      )
      return { kind: "closed", accepted: true }
    },
  })
  assert.equal((await owner.submit(new AbortController().signal)).phase, "unknown")
  assert.equal((await owner.submit(new AbortController().signal)).phase, "accepted")
  assert.deepEqual(requests[0], requests[1])
  assert.equal(requests[0].lastEventId, null)
  assert.equal(requests[0].body.idempotency_key, "key")
})

test("authorization rejection after uncertain dispatch cannot erase uncertainty", async () => {
  let attempts = 0
  const owner = new SessionMessageAcceptance("s", "message", "key", {
    attempt: async () =>
      ++attempts === 1
        ? { kind: "network_failed", accepted: false, error: new Error("lost") }
        : { kind: "http_failed", accepted: false, status: 403, error: new Error("denied") },
  })
  assert.equal((await owner.submit(new AbortController().signal)).phase, "unknown")
  assert.equal((await owner.submit(new AbortController().signal)).phase, "unknown")
})

test("initial rejection is definitive and unrelated queued event does not prove acceptance", async () => {
  const denied = new SessionMessageAcceptance("s", "message", "key", {
    attempt: async () => ({
      kind: "http_failed",
      accepted: false,
      status: 403,
      error: new Error("denied"),
    }),
  })
  assert.equal((await denied.submit(new AbortController().signal)).phase, "rejected")
  const unrelated = new SessionMessageAcceptance("s", "message", "key", {
    attempt: async (_request, callbacks) => {
      callbacks.onEvent(
        { type: "session.message.queued", session_id: "other", payload: { queue_id: "q" } },
        "event",
      )
      return { kind: "closed", accepted: true }
    },
  })
  assert.equal((await unrelated.submit(new AbortController().signal)).phase, "unknown")
})

test("server failure is uncertain and overlapping submits never duplicate the attempt", async () => {
  let release
  let calls = 0
  const owner = new SessionMessageAcceptance("s", "message", "key", {
    attempt: async () => {
      calls++
      await new Promise((resolve) => {
        release = resolve
      })
      return { kind: "http_failed", accepted: false, status: 503, error: new Error("unavailable") }
    },
  })
  const pending = owner.submit(new AbortController().signal)
  await assert.rejects(owner.submit(new AbortController().signal), /already being observed/)
  assert.equal(calls, 1)
  release()
  assert.equal((await pending).phase, "unknown")
})
