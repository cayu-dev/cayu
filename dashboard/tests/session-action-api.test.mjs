import assert from "node:assert/strict"
import test from "node:test"
import { fileURLToPath } from "node:url"
import { rolldown } from "rolldown"

const builder = await rolldown({
  cwd: fileURLToPath(new URL("..", import.meta.url)),
  input: "src/next/api/real.ts",
  platform: "node",
})
const bundled = await builder.generate({ format: "esm", codeSplitting: false })
await builder.close()
globalThis.window = { __CAYU_DASHBOARD_CONFIG__: { apiBaseUrl: "/control/api" } }
const { realApi } = await import(
  `data:text/javascript;base64,${Buffer.from(bundled.output[0].code).toString("base64")}`
)

test("reply and interrupt preserve observer callbacks and definitive rejected snapshots", async () => {
  const original = globalThis.fetch
  globalThis.fetch = async () => {
    throw new Error("Unexpected network read outside injected transport")
  }
  try {
    for (const action of ["reply", "interrupt"]) {
      let controller
      const phases = []
      const options = {
        visibility: { isVisible: () => true, subscribe: () => () => {} },
        onController: (owner) => {
          controller = owner
        },
        onChange: (snapshot) => phases.push(snapshot.phase),
        dependencies: {
          readSessionEvents: async () => ({ events: [], next_cursor: null, has_more: false }),
          fetchEventSource: async (_url, init) => {
            await init.onopen(
              new Response(JSON.stringify({ detail: "Rejected" }), {
                status: 422,
                headers: { "content-type": "application/json" },
              }),
            )
          },
        },
      }
      const result = await realApi[action]("session-action", "keep draft", options)
      assert.equal(result.phase, "request_failed")
      assert.ok(controller)
      assert.ok(phases.includes("request_failed"))
      controller.dispose()
    }
  } finally {
    globalThis.fetch = original
  }
})

test("manual recovery requires exact attention identity and finds it beyond the first native page", async () => {
  const original = globalThis.fetch
  const reads = []
  let posted
  globalThis.fetch = async (url) => {
    const params = new URL(url, "https://example.test").searchParams
    reads.push(params.get("cursor"))
    return new Response(
      JSON.stringify(
        params.get("cursor")
          ? {
              actions: [
                {
                  kind: "manual_recovery",
                  attention_id: "exact",
                  tool_call_id: "call-exact",
                  round_id: "round-exact",
                },
              ],
              next_cursor: null,
            }
          : {
              actions: [
                {
                  kind: "manual_recovery",
                  attention_id: "other",
                  tool_call_id: "call-other",
                  round_id: "round-other",
                },
              ],
              next_cursor: "next",
            },
      ),
      { headers: { "content-type": "application/json" } },
    )
  }
  try {
    await assert.rejects(
      realApi.recover("session-action", { outcome: "failed", message: "verified" }),
      /attention identity/,
    )
    assert.deepEqual(reads, [])
    const result = await realApi.recover(
      "session-action",
      { attention_id: "exact", outcome: "failed", message: "verified" },
      {
        visibility: { isVisible: () => true, subscribe: () => () => {} },
        dependencies: {
          readSessionEvents: async () => ({ events: [], next_cursor: null, has_more: false }),
          fetchEventSource: async (_url, init) => {
            posted = JSON.parse(init.body)
            await init.onopen(
              new Response(JSON.stringify({ detail: "Rejected" }), {
                status: 422,
                headers: { "content-type": "application/json" },
              }),
            )
          },
        },
      },
    )
    assert.deepEqual(reads, [null, "next"])
    assert.equal(posted.tool_call_id, "call-exact")
    assert.equal(posted.round_id, "round-exact")
    assert.equal(result.phase, "request_failed")
  } finally {
    globalThis.fetch = original
  }
})

test("specialist rerun reads the first user message by role and preserves labels and observation", async () => {
  const original = globalThis.fetch
  let body
  globalThis.fetch = async (input) => {
    const url = new URL(input, "https://example.test")
    if (!url.pathname.endsWith("/transcript"))
      return Response.json({
        agent_name: "writer",
        labels: { team: "finance" },
        invocation: { source: "http_run" },
      })
    assert.equal(url.searchParams.get("role"), "user")
    assert.equal(url.searchParams.get("limit"), "1")
    return Response.json({
      messages: [{ role: "user", content: [{ type: "text", text: "First request" }] }],
    })
  }
  try {
    const result = await realApi.runAgain("source", {
      visibility: { isVisible: () => true, subscribe: () => () => {} },
      dependencies: {
        fetchEventSource: async (_url, init) => {
          body = JSON.parse(init.body)
          await init.onopen(
            new Response(JSON.stringify({ detail: "Rejected" }), {
              status: 422,
              headers: { "content-type": "application/json" },
            }),
          )
        },
      },
    })
    assert.equal(body.prompt, "First request")
    assert.deepEqual(body.labels, { team: "finance" })
    assert.equal(result.snapshot.phase, "request_failed")
  } finally {
    globalThis.fetch = original
  }
})
