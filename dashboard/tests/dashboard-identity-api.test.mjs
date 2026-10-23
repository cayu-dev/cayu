import assert from "node:assert/strict"
import test from "node:test"

globalThis.window = { __CAYU_DASHBOARD_CONFIG__: { apiBaseUrl: "/nested/api" } }
const api = await import("../src/lib/api.ts")

test("identity client uses configured API and validates authority before publishing", async () => {
  const original = globalThis.fetch
  const expected = {
    name: "Agent",
    runtime_version: "0.14.0",
    role: "custom-role",
    full_access: false,
  }
  try {
    globalThis.fetch = async (input, options) => {
      assert.equal(String(input), "/nested/api/app")
      assert.equal(options.credentials, "same-origin")
      return new Response(JSON.stringify(expected), {
        headers: { "content-type": "application/json" },
      })
    }
    assert.deepEqual(await api.fetchDashboardIdentity(), expected)
    globalThis.fetch = async () =>
      new Response(JSON.stringify({ ...expected, full_access: "true" }), {
        headers: { "content-type": "application/json" },
      })
    await assert.rejects(api.fetchDashboardIdentity(), /invalid dashboard identity/)
  } finally {
    globalThis.fetch = original
  }
})

test("public request errors preserve detail/status and explain403 role_forbidden", async () => {
  const original = globalThis.fetch
  const detail = { code: "role_forbidden", message: "Your role can't use this." }
  try {
    globalThis.fetch = async () =>
      new Response(JSON.stringify({ detail }), {
        status: 403,
        headers: { "content-type": "application/json" },
      })
    await assert.rejects(api.fetchSystemDiagnostics(), (error) => {
      assert.equal(error.status, 403)
      assert.deepEqual(error.detail, detail)
      assert.equal(
        error.message,
        "Your role can read and decide, but this action requires full access.",
      )
      return true
    })
  } finally {
    globalThis.fetch = original
  }
})

test("identity snapshots revoke authority on denial and cannot revive it on transient failure", async () => {
  const { QueryClient } = await import("@tanstack/react-query")
  const { fetchDashboardIdentitySnapshot } = await import("../src/lib/dashboard-identity-query.ts")
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } })
  const original = globalThis.fetch
  const query = { queryKey: ["identity-refetch"], queryFn: fetchDashboardIdentitySnapshot }
  const identity = { name: "Agent", runtime_version: "0.14.0", role: "admin", full_access: true }
  const respond =
    (value, status = 200) =>
    async () =>
      new Response(JSON.stringify(value), {
        status,
        headers: { "content-type": "application/json" },
      })
  try {
    globalThis.fetch = respond(identity)
    await client.fetchQuery(query)
    for (const failedFetch of [
      respond({}, 408),
      respond({}, 429),
      respond({}, 503),
      async () => {
        throw new TypeError("offline")
      },
    ]) {
      globalThis.fetch = failedFetch
      await assert.rejects(client.fetchQuery(query))
      assert.deepEqual(client.getQueryData(query.queryKey).identity, identity)
    }
    for (const status of [401, 403]) {
      globalThis.fetch = respond({ detail: "denied" }, status)
      assert.equal((await client.fetchQuery(query)).identity, null)
      for (const transientStatus of [408, 429, 502]) {
        globalThis.fetch = respond({}, transientStatus)
        await assert.rejects(client.fetchQuery(query))
        assert.equal(client.getQueryData(query.queryKey).identity, null)
      }
    }
    globalThis.fetch = respond({ ...identity, full_access: "true" })
    assert.equal((await client.fetchQuery(query)).identity, null)
    globalThis.fetch = respond({ ...identity, role: "member", full_access: false })
    assert.equal((await client.fetchQuery(query)).identity.full_access, false)
  } finally {
    globalThis.fetch = original
    client.clear()
  }
})
