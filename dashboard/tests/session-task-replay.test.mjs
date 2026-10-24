import assert from "node:assert/strict"
import test from "node:test"
import { fileURLToPath } from "node:url"
import { rolldown } from "rolldown"

const builder = await rolldown({
  cwd: fileURLToPath(new URL("..", import.meta.url)),
  input: "src/lib/session-task-replay.ts",
  platform: "node",
})
const bundled = await builder.generate({ format: "esm", codeSplitting: false })
await builder.close()
globalThis.window = { __CAYU_DASHBOARD_CONFIG__: { apiBaseUrl: "/api" } }
const { prepareSessionTaskReplay } = await import(
  `data:text/javascript;base64,${Buffer.from(bundled.output[0].code).toString("base64")}`
)
const source = {
  task_id: "source",
  created_at: "2026-10-24T00:00:00Z",
  root_invocation_id: "root",
  restart_count: 0,
  session_id: "session",
  session_instance_id: "instance",
}

test("task replay retries the exact operation after lost acknowledgement without reloading source", async () => {
  const original = globalThis.fetch
  const reads = [],
    bodies = []
  globalThis.fetch = async (url, init) => {
    if (init.method === "POST") {
      bodies.push(JSON.parse(init.body))
      if (bodies.length === 1) throw new Error("lost acknowledgement")
      return Response.json({ task_id: "fresh" })
    }
    reads.push(url)
    return Response.json(
      url.endsWith("task-replay") ? { available: true, source } : { type: "declared" },
    )
  }
  try {
    const operation = await prepareSessionTaskReplay("session", new AbortController().signal)
    assert.equal((await operation.submit(new AbortController().signal)).phase, "unknown")
    assert.deepEqual(await operation.submit(new AbortController().signal), {
      phase: "accepted",
      taskId: "fresh",
      message: null,
    })
    assert.equal(reads.length, 2)
    assert.deepEqual(bodies[0], bodies[1])
    assert.deepEqual(bodies[0].source, source)
    assert.equal(bodies[0].type, "declared")
    assert.ok(bodies[0].operation_id)
    await operation.submit(new AbortController().signal)
    assert.equal(bodies.length, 2)
  } finally {
    globalThis.fetch = original
  }
})

test("unavailable task replay never loads input or dispatches a specialist run", async () => {
  const original = globalThis.fetch
  const calls = []
  globalThis.fetch = async (url) => {
    calls.push(url)
    return Response.json({
      available: false,
      unavailable_reason: "task_type_not_declared",
      source: null,
    })
  }
  try {
    await assert.rejects(
      prepareSessionTaskReplay("session", new AbortController().signal),
      /task_type_not_declared/,
    )
    assert.equal(calls.length, 1)
  } finally {
    globalThis.fetch = original
  }
})
