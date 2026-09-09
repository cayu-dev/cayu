import assert from "node:assert/strict"
import test from "node:test"
import { createBrowserControlClient } from "../src/lib/browser-control-client.ts"

const root = new URL("https://dashboard.example/api/browser-control")
const browser = {
  identity: {
    profile_checkpoint_policy: "unavailable",
    operator_purpose: { code: "login", expected_origins: ["https://site.test"] },
    session_id: "session",
    session_instance_id: "instance",
    run_epoch: 1,
    interaction_id: "interaction",
    execution_profile_fingerprint: "a".repeat(64),
    environment_name: "browser",
    allocation_fingerprint: "b".repeat(64),
    browser_session_id: "browser-session",
    worker_instance_id: "worker",
  },
  revision: 2,
  control_epoch: 1,
  state: "agent_controlled",
  sensitive_entry: false,
  sensitive_entry_pending: false,
  fresh_observation_required: false,
  owned_request: null,
}

test("unsafe transport endpoints are refused before authentication", () => {
  for (const url of [
    "http://dashboard.example/api/browser-control",
    "https://user:private-canary@dashboard.example/api/browser-control",
    "https://dashboard.example/api/browser-control?token=private-canary",
    "https://dashboard.example/api/browser-control#private-canary",
  ]) {
    let calls = 0
    assert.throws(
      () =>
        createBrowserControlClient(new URL(url), "session", async () => {
          calls++
          return Response.json({})
        }),
      /unavailable/,
    )
    assert.equal(calls, 0)
  }
})

test("discovery, pages and viewer ticket retain exact authority without URL credentials", async () => {
  const calls = []
  const page = { page_id: "page", revision: "revision", control_epoch: 1 }
  const results = [
    { operator_session_token: "private-continuity-canary" },
    { browsers: [browser] },
    { pages: [page], locations: [{ page, origin: "https://site.test" }], active_page_id: "page" },
    { ticket: "private-viewer-canary" },
  ]
  const client = createBrowserControlClient(root, "session", async (url, init) => {
    calls.push({ url: url.href, ...init })
    return Response.json(results.shift())
  })
  const [discovered] = await client.discover()
  const {
    pages: [selected],
    active_page_id: active,
    locations,
  } = await client.pages(discovered)
  assert.equal(active, "page")
  assert.deepEqual(locations, [{ page, origin: "https://site.test" }])
  assert.equal(await client.viewerTicket(discovered, selected), "private-viewer-canary")
  assert.deepEqual(JSON.parse(calls[3].body), {
    identity: browser.identity,
    expected_record_revision: 2,
    page,
  })
  assert.equal(client.viewerUrl(), "wss://dashboard.example/api/browser-control/viewer")
  for (const [index, call] of calls.entries()) {
    assert.equal(call.cache, "no-store")
    assert.equal(call.redirect, "error")
    assert.equal(call.referrerPolicy, "no-referrer")
    assert.equal(call.url.includes("canary"), false)
    assert.equal(
      call.headers["X-Cayu-Browser-Operator"],
      index ? "private-continuity-canary" : undefined,
    )
  }
  client.dispose()
  await assert.rejects(client.discover(), /unavailable/)
  assert.equal(calls.length, 4)
})

test("discovery rejects malformed application purpose without reflecting its values", async () => {
  for (const purpose of [
    undefined,
    { code: "private-canary", expected_origins: ["https://site.test"] },
    { code: "login", expected_origins: [] },
    { code: "login", expected_origins: ["https://site.test/private-canary"] },
    { code: "login", expected_origins: ["https://private-canary@site.test"] },
    { code: "login", expected_origins: ["http://site.test"] },
    { code: "login", expected_origins: ["https://site.test", "https://site.test"] },
    { code: "login", expected_origins: ["https://z.test", "https://a.test"] },
  ]) {
    const client = createBrowserControlClient(root, "session", async (url) =>
      Response.json(
        url.pathname.endsWith("operator-session")
          ? { operator_session_token: "continuity" }
          : {
              browsers: [
                { ...browser, identity: { ...browser.identity, operator_purpose: purpose } },
              ],
            },
      ),
    )
    try {
      await assert.rejects(client.discover(), (error) => {
        assert.match(error.message, /unavailable/)
        assert.equal(String(error).includes("private-canary"), false)
        return true
      })
    } finally {
      client.dispose()
    }
  }
})

test("page locations require exact page identity and origin-only values", async () => {
  const page = { page_id: "page", revision: "revision", control_epoch: 1 }
  for (const locations of [
    undefined,
    [],
    [{ page: { ...page, revision: "different" }, origin: "https://site.test" }],
    [{ page, origin: "https://site.test/private-canary" }],
    [{ page, origin: "https://private-canary@site.test" }],
    [{ page, origin: "not a URL private-canary" }],
    [{ page, origin: "http://site.test" }],
  ]) {
    const client = createBrowserControlClient(root, "session", async () =>
      Response.json({ pages: [page], locations, active_page_id: "page" }),
    )
    try {
      await assert.rejects(client.pages(browser), (error) => {
        assert.match(error.message, /unavailable/)
        assert.equal(String(error).includes("private-canary"), false)
        return true
      })
    } finally {
      client.dispose()
    }
  }
})

test("discovery preserves each admitted checkpoint policy in subsequent authority", async () => {
  for (const policy of ["unavailable", "disabled", "on_close", "after_terminal_operation"]) {
    const identity = { ...browser.identity, profile_checkpoint_policy: policy }
    const calls = []
    const client = createBrowserControlClient(root, "session", async (_url, init) => {
      calls.push(init)
      return Response.json(
        calls.length === 1
          ? { operator_session_token: "continuity" }
          : calls.length === 2
            ? { browsers: [{ ...browser, identity }] }
            : { pages: [], locations: [], active_page_id: null },
      )
    })
    try {
      const [discovered] = await client.discover()
      assert.equal(discovered.identity.profile_checkpoint_policy, policy)
      await client.pages(discovered)
      assert.deepEqual(JSON.parse(calls[2].body).identity, identity)
    } finally {
      client.dispose()
    }
  }
})

test("discovery rejects missing or malformed checkpoint policy without echoing it", async () => {
  for (const policy of [
    undefined,
    null,
    false,
    1,
    "private-canary",
    { secret: "private-canary" },
  ]) {
    let calls = 0
    const client = createBrowserControlClient(root, "session", async () =>
      Response.json(
        calls++
          ? {
              browsers: [
                {
                  ...browser,
                  identity: { ...browser.identity, profile_checkpoint_policy: policy },
                },
              ],
            }
          : { operator_session_token: "continuity" },
      ),
    )
    try {
      await assert.rejects(client.discover(), (error) => {
        assert.match(error.message, /unavailable/)
        assert.equal(String(error).includes("private-canary"), false)
        return true
      })
      assert.equal(calls, 2)
    } finally {
      client.dispose()
    }
  }
})

test("disposal aborts an in-flight request and never admits its late response", async () => {
  let release
  let signal
  const client = createBrowserControlClient(root, "session", async (_url, init) => {
    signal = init.signal
    return new Promise((resolve) => {
      release = resolve
    })
  })
  const pending = client.discover()
  client.dispose()
  assert.equal(signal.aborted, true)
  release(Response.json({ operator_session_token: "late-private-canary" }))
  await assert.rejects(pending, (error) => {
    assert.equal(String(error).includes("canary"), false)
    return true
  })
})

test("malformed, oversized and denied responses expose only fixed diagnostics", async () => {
  for (const response of [
    new Response("private-canary", { status: 403 }),
    new Response("private-canary"),
    new Response("x".repeat(65537)),
    Response.json({ operator_session_token: { secret: "private-canary" } }),
  ]) {
    const client = createBrowserControlClient(root, "session", async () => response)
    await assert.rejects(client.discover(), (error) => {
      assert.match(error.message, /Protected browser control is unavailable/)
      assert.equal(String(error).includes("canary"), false)
      return true
    })
    client.dispose()
  }
})

test("wrong session and malformed counters are not forwarded as browser authority", async () => {
  for (const invalid of [
    { ...browser, identity: { ...browser.identity, session_id: "other" } },
    { ...browser, revision: true },
    { ...browser, control_epoch: Number.MAX_SAFE_INTEGER + 1 },
    { ...browser, sensitive_entry: "false" },
  ]) {
    let count = 0
    const client = createBrowserControlClient(root, "session", async () =>
      Response.json(count++ ? { browsers: [invalid] } : { operator_session_token: "token" }),
    )
    await assert.rejects(client.discover(), /unavailable/)
    assert.equal(count, 2)
    client.dispose()
  }
})

test("takeover acknowledgement loss preserves continuity for readback without replay", async () => {
  const calls = []
  const client = createBrowserControlClient(root, "session", async (url, init) => {
    calls.push({ path: url.pathname, ...init })
    if (url.pathname.endsWith("operator-session"))
      return Response.json({ operator_session_token: "continuity-canary" })
    if (url.pathname.endsWith("takeover")) throw Error("lost-acknowledgement-canary")
    return Response.json({ browsers: [browser] })
  })
  const [discovered] = await client.discover()
  const pages = [
    { page_id: "z", revision: "z-revision", control_epoch: 3 },
    { page_id: "a", revision: "a-revision", control_epoch: 2 },
  ]
  await assert.rejects(client.takeover(discovered, pages, "undecided"), /unavailable/)
  await client.discover()
  const requests = calls.filter((call) => call.path.endsWith("takeover"))
  assert.equal(requests.length, 1)
  const request = JSON.parse(requests[0].body)
  assert.deepEqual(request.identity, browser.identity)
  assert.equal(request.expected_record_revision, browser.revision)
  assert.equal(request.expected_control_epoch, browser.control_epoch)
  assert.match(request.request_id, /^bt_[a-f0-9]{32}$/)
  assert.equal(request.expires_at_ms - request.requested_at_ms, 30_000)
  assert.equal(request.maximum_until_ms - request.requested_at_ms, 900_000)
  assert.deepEqual(request.pages, [pages[1], pages[0]])
  assert.deepEqual(
    pages.map((page) => page.page_id),
    ["z", "a"],
  )
  assert.equal(request.checkpoint_consent, "undecided")
  assert.equal(calls.filter((call) => call.path.endsWith("operator-session")).length, 1)
  assert.equal(calls.at(-1).headers["X-Cayu-Browser-Operator"], "continuity-canary")
  client.dispose()
})

test("takeover binds an explicit finite checkpoint decision before requesting authority", async (t) => {
  for (const consent of [
    "allow",
    "deny",
    "undecided",
    undefined,
    null,
    true,
    "ALLOW",
    "private-consent-canary",
    {},
  ]) {
    await t.test(String(consent), async () => {
      const calls = []
      const client = createBrowserControlClient(root, "session", async (url, init) => {
        calls.push({ path: url.pathname, ...init })
        if (url.pathname.endsWith("operator-session"))
          return Response.json({ operator_session_token: "continuity" })
        return Response.json({ state: "takeover_requested" })
      })
      const pages = [{ page_id: "page", revision: "revision", control_epoch: 1 }]
      if (["allow", "deny", "undecided"].includes(consent)) {
        await client.takeover(browser, pages, consent)
        const request = JSON.parse(calls.at(-1).body)
        assert.equal(request.checkpoint_consent, consent)
        assert.equal(request.purpose_code, browser.identity.operator_purpose.code)
        assert.deepEqual(request.identity, browser.identity)
        assert.equal(request.expected_control_epoch, browser.control_epoch)
        assert.equal(calls.filter((call) => call.path.endsWith("takeover")).length, 1)
      } else {
        await assert.rejects(client.takeover(browser, pages, consent), (error) => {
          assert.equal(String(error).includes("private-consent-canary"), false)
          return /unavailable/.test(String(error))
        })
        assert.equal(calls.length, 0)
      }
      client.dispose()
    })
  }
})

test("handback and sensitive entry require and forward the discovered self-owned request", async () => {
  const calls = []
  const controlled = {
    ...browser,
    state: "operator_controlled",
    revision: 4,
    control_epoch: 2,
    owned_request: {
      request_id: `bt_${"c".repeat(32)}`,
      expires_at_ms: 1000,
      maximum_until_ms: 5000,
      lease_until_ms: 3000,
      pending_lease_until_ms: null,
      settled_input_sequence: 0,
      pending_input_sequence: null,
      checkpoint_consent: "undecided",
    },
  }
  const client = createBrowserControlClient(root, "session", async (url, init) => {
    calls.push({ path: url.pathname, ...init })
    if (url.pathname.endsWith("operator-session"))
      return Response.json({ operator_session_token: "token" })
    return Response.json({ browsers: [controlled] })
  })
  await assert.rejects(client.handback(browser), /unavailable/)
  await assert.rejects(client.sensitiveEntry(browser), /unavailable/)
  assert.equal(calls.length, 0)
  const [owned] = await client.discover()
  assert.deepEqual(owned.owned_request, controlled.owned_request)
  await client.sensitiveEntry(owned)
  await client.handback(owned)
  for (const call of calls.slice(2)) {
    assert.deepEqual(JSON.parse(call.body), {
      identity: controlled.identity,
      expected_record_revision: 4,
      expected_control_epoch: 2,
      request_id: controlled.owned_request.request_id,
    })
  }
  client.dispose()
})

test("renewal preserves the source lease and never extends the takeover maximum", async (t) => {
  t.mock.method(Date, "now", () => 2000)
  const controlled = {
    ...browser,
    state: "operator_controlled",
    owned_request: {
      request_id: `bt_${"d".repeat(32)}`,
      expires_at_ms: 3000,
      maximum_until_ms: 40_000,
      lease_until_ms: 10_000,
      pending_lease_until_ms: null,
      settled_input_sequence: 0,
      pending_input_sequence: null,
      checkpoint_consent: "undecided",
    },
  }
  const calls = []
  const client = createBrowserControlClient(root, "session", async (url, init) => {
    calls.push({ path: url.pathname, ...init })
    return Response.json({})
  })
  await client.renew(controlled)
  assert.deepEqual(JSON.parse(calls[0].body), {
    identity: browser.identity,
    expected_record_revision: browser.revision,
    expected_control_epoch: browser.control_epoch,
    request_id: controlled.owned_request.request_id,
    expected_lease_until_ms: 10_000,
    lease_until_ms: 40_000,
  })
  for (const update of [
    { lease_until_ms: 2000 },
    { pending_lease_until_ms: 40_000 },
    { maximum_until_ms: 10_000 },
  ]) {
    await assert.rejects(
      client.renew({ ...controlled, owned_request: { ...controlled.owned_request, ...update } }),
      /unavailable/,
    )
  }
  assert.equal(calls.length, 1)
  client.dispose()
})

test("private input ticket contains only exact control authority and requires settled capture", async (t) => {
  t.mock.method(Date, "now", () => 2000)
  const controlled = {
    ...browser,
    state: "operator_controlled",
    sensitive_entry: true,
    owned_request: {
      request_id: `bt_${"e".repeat(32)}`,
      expires_at_ms: 3000,
      maximum_until_ms: 60000,
      lease_until_ms: 10000,
      pending_lease_until_ms: null,
      settled_input_sequence: 3,
      pending_input_sequence: null,
      checkpoint_consent: "undecided",
    },
  }
  const page = { page_id: "page", revision: "page-revision", control_epoch: 1 }
  const calls = []
  const client = createBrowserControlClient(root, "session", async (url, init) => {
    calls.push({ path: url.pathname, ...init })
    return Response.json({ ticket: "private-input-ticket" })
  })
  const grant = await client.inputTicket(controlled, page)
  assert.deepEqual(grant.expected, {
    revision: browser.revision + 2,
    control_epoch: browser.control_epoch,
    settled_input_sequence: 4,
  })
  assert.deepEqual(JSON.parse(calls[0].body), {
    identity: browser.identity,
    expected_record_revision: browser.revision,
    expected_control_epoch: browser.control_epoch,
    request_id: controlled.owned_request.request_id,
    input_sequence: 4,
    page,
  })
  assert.equal(client.inputUrl(), "wss://dashboard.example/api/browser-control/input")
  for (const invalid of [
    { ...controlled, sensitive_entry: false },
    { ...controlled, sensitive_entry_pending: true },
    { ...controlled, owned_request: { ...controlled.owned_request, pending_input_sequence: 4 } },
    {
      ...controlled,
      owned_request: { ...controlled.owned_request, pending_lease_until_ms: 12000 },
    },
    { ...controlled, owned_request: { ...controlled.owned_request, lease_until_ms: 2000 } },
  ])
    await assert.rejects(client.inputTicket(invalid, page), /unavailable/)
  assert.equal(calls.length, 1)
  await client.inputTicket(controlled, page, "tab")
  assert.equal(JSON.parse(calls[1].body).input_kind, "tab")
  client.dispose()
})

// Exercise the same refresh sequence used by the explicit View page handler.
test("fresh views bind new revisions and refuse replacements, missing pages and races", async (t) => {
  const { requestFreshBrowserView, browserViewFailureMessage } = await import(
    "../src/lib/browser-control-client.ts"
  )
  const fresh = { ...browser, revision: 9, control_epoch: 3 }
  const page = { page_id: "page", revision: "navigated", control_epoch: 3 }
  const replacements = Object.entries(browser.identity).map(([key, value]) => ({
    ...fresh,
    identity: {
      ...browser.identity,
      [key]:
        key === "operator_purpose"
          ? { code: "other", expected_origins: ["https://site.test"] }
          : key === "profile_checkpoint_policy"
            ? "disabled"
            : typeof value === "number"
              ? value + 1
              : `${value}-replacement`,
    },
  }))
  const cases = [
    { name: "fresh evidence", success: true, count: 4 },
    { name: "retired", browsers: [], count: 2 },
    { name: "ambiguous", browsers: [fresh, fresh], count: 2 },
    ...replacements.map((replacement, index) => ({
      name: `identity fence ${index}`,
      browsers: [replacement],
      count: 2,
    })),
    {
      name: "missing page",
      pages: [{ ...page, page_id: "other" }],
      count: 3,
      message: /page is no longer available/,
    },
    {
      name: "closed",
      browsers: [{ ...fresh, state: "closed" }],
      count: 2,
      message: /closed or unavailable/,
    },
    {
      name: "sensitive",
      browsers: [{ ...fresh, sensitive_entry: true }],
      count: 2,
      message: /paused/,
    },
    {
      name: "pending sensitive",
      browsers: [{ ...fresh, sensitive_entry_pending: true }],
      count: 2,
      message: /paused/,
    },
    ...["operator-session", "session", "pages", "view-ticket"].map((failAt, index) => ({
      name: `conflict at ${failAt}`,
      failAt,
      count: index + 1,
    })),
    { name: "closed during discovery", cancelAt: "session", count: 2 },
    { name: "closed during pages", cancelAt: "pages", count: 3 },
  ]
  for (const scenario of cases)
    await t.test(scenario.name, async () => {
      const calls = []
      let current = true
      const client = createBrowserControlClient(root, "session", async (url, init) => {
        const path = url.pathname.split("/").at(-1)
        calls.push({ path, body: init.body && JSON.parse(init.body) })
        if (path === scenario.cancelAt) current = false
        if (path === scenario.failAt) return new Response("private-canary", { status: 409 })
        if (path === "operator-session")
          return Response.json({ operator_session_token: "continuity" })
        if (path === "session") return Response.json({ browsers: scenario.browsers ?? [fresh] })
        if (path === "pages") {
          const pages = scenario.pages ?? [page]
          return Response.json({
            pages,
            locations: pages.map((page) => ({ page, origin: null })),
            active_page_id: pages[0]?.page_id ?? null,
          })
        }
        assert.equal(path, "view-ticket")
        return Response.json({ ticket: "fresh-ticket" })
      })
      try {
        const pending = requestFreshBrowserView(client, browser, "page", () => current)
        if (scenario.success) {
          assert.equal(await pending, "fresh-ticket")
          assert.deepEqual(calls[2].body, {
            identity: browser.identity,
            expected_record_revision: 9,
          })
          assert.deepEqual(calls[3].body, {
            identity: browser.identity,
            expected_record_revision: 9,
            page,
          })
          assert.equal(browser.revision, 2)
        } else if (scenario.cancelAt) {
          assert.equal(await pending, "")
        } else {
          await assert.rejects(pending, (error) => {
            assert.match(
              browserViewFailureMessage(error),
              scenario.message ?? /unavailable|no longer available/,
            )
            assert.doesNotMatch(browserViewFailureMessage(error), /canary|settle|repeat input/)
            return true
          })
        }
        assert.deepEqual(
          calls.map((call) => call.path),
          ["operator-session", "session", "pages", "view-ticket"].slice(0, scenario.count),
        )
      } finally {
        client.dispose()
      }
    })
})
