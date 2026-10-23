import assert from "node:assert/strict"
import test from "node:test"
import {
  browserViewFailureMessage,
  createBrowserControlClient,
} from "../src/lib/browser-control-client.ts"

test("browser control preserves only a fixed role denial through its private transport boundary", async () => {
  const client = createBrowserControlClient(
    new URL("https://example.test/api/browser-control"),
    "session",
    async () =>
      new Response(
        JSON.stringify({ detail: { code: "role_forbidden", message: "secret-canary" } }),
        { status: 403 },
      ),
  )
  try {
    await assert.rejects(client.discover(), (error) => {
      assert.equal(
        browserViewFailureMessage(error),
        "Your role can read and decide, but this action requires full access.",
      )
      assert.ok(!String(error).includes("secret-canary"))
      return true
    })
  } finally {
    client.dispose()
  }
})
