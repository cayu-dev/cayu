import assert from "node:assert/strict"
import test from "node:test"
import {
  buildTrace,
  findSpan,
  flatten,
  formatSpanDuration,
  isHiddenValue,
} from "../src/next/app/trace-model.ts"

const T0 = Date.parse("2026-10-21T12:00:00Z")
let seq = 0
const ev = (ms, type, payload = {}, extra = {}) => ({
  sequence: ++seq,
  id: `e${seq}`,
  type,
  session_id: "s1",
  interaction_id: "i1",
  agent_name: "a",
  environment_name: "local",
  tool_name: null,
  payload,
  timestamp: new Date(T0 + ms).toISOString(),
  ...extra,
})
const step = { model_step_id: "m1", model_attempt_id: "a1", step: 1 }
const call = { ...step, tool_round_id: "r1", tool_call_id: "c1" }

test("buildTrace nests turn, model step, tool round, tool call and command", () => {
  seq = 0
  const root = buildTrace(
    "s1",
    [
      ev(0, "session.started", {}, { interaction_id: null }),
      ev(10, "interaction.started"),
      ev(20, "model.started", { ...step, model: "gpt" }),
      ev(900, "model.completed", { ...step, model: "gpt", status: "completed" }),
      ev(950, "tool.call.started", call, { tool_name: "read_file" }),
      ev(
        960,
        "runner.exec.started",
        { ...call, execution_id: "x1", command: "cat a.py" },
        { tool_name: "read_file" },
      ),
      ev(
        1200,
        "runner.exec.completed",
        { ...call, execution_id: "x1", exit_code: 0 },
        { tool_name: "read_file" },
      ),
      ev(1250, "tool.call.completed", call, { tool_name: "read_file" }),
      ev(1300, "interaction.completed", { status: "completed" }),
      ev(1310, "session.completed", {}, { interaction_id: null }),
    ],
    T0 + 5000,
  )

  assert.equal(root.status, "ok")
  assert.equal(root.end - root.start, 1310)
  const [turn] = root.children
  assert.equal(turn.name, "Turn 1")
  const [model] = turn.children
  assert.equal(model.kind, "model")
  assert.equal(model.detail, "gpt")
  const [tool] = model.children
  assert.equal(tool.kind, "tool", "a single-call round shows the call directly")
  assert.equal(tool.name, "read_file")
  assert.equal(tool.end - tool.start, 300)
  assert.equal(tool.children[0].kind, "exec")
  assert.equal(tool.children[0].detail, "cat a.py")
  assert.deepEqual(
    flatten(root, new Set()).map((r) => [r.span.kind, r.depth]),
    [
      ["session", 0],
      ["interaction", 1],
      ["model", 2],
      ["tool", 3],
      ["exec", 4],
    ],
  )
  assert.deepEqual(flatten(root, new Set(["model:m1"])).length, 3)
})

test("a failed tool marks its round, and an approval request opens a wait until now", () => {
  seq = 0
  const now = T0 + 60 * 60_000
  const root = buildTrace(
    "s1",
    [
      ev(0, "tool.call.started", call, { tool_name: "search" }),
      ev(50, "tool.call.started", { ...call, tool_call_id: "c3" }, { tool_name: "read" }),
      ev(80, "tool.call.completed", { ...call, tool_call_id: "c3" }, { tool_name: "read" }),
      ev(100, "tool.call.failed", call, { tool_name: "search" }),
      ev(
        200,
        "tool.approval.requested",
        { ...call, tool_call_id: "c2" },
        { tool_name: "hold_payment" },
      ),
    ],
    now,
  )
  assert.equal(findSpan(root, "round:r1").status, "error")
  const wait = root.children[0].children.find((s) => s.kind === "wait")
  assert.equal(wait.status, "waiting")
  assert.equal(wait.detail, "hold_payment")
  assert.equal(wait.end, now)
})

test("open spans in a finished session are closed with the session's outcome", () => {
  seq = 0
  const root = buildTrace(
    "s1",
    [
      ev(0, "tool.call.started", call, { tool_name: "post" }),
      ev(500, "session.interrupted", {}, { interaction_id: null }),
    ],
    T0 + 99_000,
    "error",
  )
  assert.equal(findSpan(root, "tool:c1").status, "error")
  assert.equal(findSpan(root, "tool:c1").end, T0)
})

test("hidden IDs are linked by order: step, then its calls, completions matched by tool", () => {
  seq = 0
  const hidden = "[PRIVATE_EVENT_AUTHORITY]"
  const alias = (field) => `cayu_event_${seq + 1}:${field}`
  const started = (ms, tool) =>
    ev(
      ms,
      "tool.call.started",
      {
        model_step_id: hidden,
        tool_call_id: alias("tool_call_id"),
        tool_round_id: alias("tool_round_id"),
      },
      { tool_name: tool },
    )
  const done = (ms, tool) =>
    ev(
      ms,
      "tool.call.completed",
      {
        model_step_id: hidden,
        tool_call_id: alias("tool_call_id"),
        tool_round_id: alias("tool_round_id"),
      },
      { tool_name: tool },
    )
  const root = buildTrace(
    "s1",
    [
      ev(0, "model.started", { model_step_id: hidden, step: 1 }),
      ev(500, "model.completed", { model_step_id: hidden, step: 1 }),
      started(510, "read"),
      started(520, "search"),
      done(900, "search"),
      done(1000, "read"),
      ev(1100, "model.started", { model_step_id: hidden, step: 2 }),
      ev(1500, "model.completed", { model_step_id: hidden, step: 2 }),
    ],
    T0 + 5000,
    "ok",
  )
  assert.equal(root.linkedByOrder, true)
  const [turn] = root.children
  assert.deepEqual(
    turn.children.map((s) => s.kind),
    ["model", "model"],
  )
  const [round] = turn.children[0].children
  assert.equal(round.kind, "round")
  const durations = Object.fromEntries(round.children.map((t) => [t.name, t.end - t.start]))
  assert.deepEqual(durations, { read: 490, search: 380 })
})

test("isHiddenValue spots withheld values and nothing else", () => {
  assert.equal(isHiddenValue("[PRIVATE_EVENT_AUTHORITY]"), true)
  assert.equal(isHiddenValue("cayu_event_287097:tool_call_id"), true)
  assert.equal(isHiddenValue("cayu_event_287050"), false)
  assert.equal(isHiddenValue("call_abc"), false)
  assert.equal(isHiddenValue(42), false)
})

test("formatSpanDuration", () => {
  assert.equal(formatSpanDuration(240), "240ms")
  assert.equal(formatSpanDuration(1500), "1.5s")
  assert.equal(formatSpanDuration(42_000), "42s")
  assert.equal(formatSpanDuration(125_000), "2m 5s")
  assert.equal(formatSpanDuration(2 * 3600_000 + 60_000), "2h 1m")
})

const approvalEvents = ({ nextTurn = false, runtime = false, approvalId = "approval-1" } = {}) => {
  seq = 0
  const hidden = "[PRIVATE_EVENT_AUTHORITY]"
  const payload = () => ({
    model_step_id: hidden,
    tool_call_id: `cayu_event_${seq + 1}:tool_call_id`,
    tool_round_id: `cayu_event_${seq + 1}:tool_round_id`,
    ...(approvalId ? { approval_id: approvalId } : {}),
  })
  return [
    ev(0, "model.started", { model_step_id: hidden }),
    ev(50, "model.completed", { model_step_id: hidden }),
    ev(60, runtime ? "tool.call.approval_requested" : "tool.approval.requested", payload(), {
      tool_name: "write_file",
    }),
    ev(100, runtime ? "tool.call.approved" : "tool.approval.resolved", {
      ...payload(),
      decision: "approve",
    }),
    ev(110, "tool.call.started", payload(), {
      tool_name: "write_file",
      interaction_id: nextTurn ? "i2" : "i1",
    }),
    ev(200, "tool.call.completed", payload(), {
      tool_name: "write_file",
      interaction_id: nextTurn ? "i2" : "i1",
    }),
  ]
}

for (const runtime of [false, true]) {
  for (const nextTurn of [false, true]) {
    test(`approved hidden call stays one completed span (${runtime ? "runtime" : "prototype"}, ${nextTurn ? "new" : "same"} turn)`, () => {
      const events = approvalEvents({ runtime, nextTurn })
      const root = buildTrace("s1", events, T0 + 99999)
      const spans = flatten(root, new Set()).map(({ span }) => span)
      const tools = spans.filter((s) => s.kind === "tool")
      assert.equal(tools.length, 1)
      assert.equal(tools[0].status, "ok")
      assert.equal(tools[0].start, T0 + 60)
      assert.equal(tools[0].end, T0 + 200)
      assert.deepEqual(
        tools[0].events.map((e) => e.type),
        [
          events[2].type,
          ...(runtime ? ["tool.call.approved"] : []),
          "tool.call.started",
          "tool.call.completed",
        ],
      )
      const waits = spans.filter((s) => s.kind === "wait")
      assert.equal(waits.length, 1)
      assert.equal(waits[0].end, T0 + 100)
    })
  }
}

test("a unique hidden approval can be linked without approval identifiers", () => {
  const root = buildTrace("s1", approvalEvents({ approvalId: null }), T0 + 99999)
  const tools = flatten(root, new Set())
    .map(({ span }) => span)
    .filter((s) => s.kind === "tool")
  assert.equal(tools.length, 1)
  assert.equal(tools[0].status, "ok")
})

test("ambiguous parallel approvals are not consumed by a guessed same-tool match", () => {
  const events = approvalEvents({ approvalId: null })
  events.splice(3, 0, { ...events[2], id: "second-approval", sequence: 3.5 })
  const root = buildTrace("s1", events, T0 + 99999)
  const tools = flatten(root, new Set())
    .map(({ span }) => span)
    .filter((s) => s.kind === "tool")
  assert.equal(tools.length, 3)
  assert.equal(tools.filter((s) => s.status === "ok").length, 1)
  assert.equal(tools.find((s) => s.status === "ok").start, T0 + 110)
})

test("a conflicting explicit approval ID never falls back to another approval", () => {
  const events = approvalEvents()
  events[4].payload.approval_id = "another-approval"
  events[5].payload.approval_id = "another-approval"
  const root = buildTrace("s1", events, T0 + 99999)
  const tools = flatten(root, new Set())
    .map(({ span }) => span)
    .filter((s) => s.kind === "tool")
  assert.equal(tools.length, 2)
  assert.equal(tools.find((s) => s.status === "ok").start, T0 + 110)
})

for (const type of [
  "tool.call.approval_denied",
  "tool.call.approval_expired",
  "tool.approval.resolved",
  "tool.approval.recovered",
]) {
  test(`${type} closes the approval call and its wait without a start`, () => {
    const events = approvalEvents({ runtime: type.startsWith("tool.call.") }).slice(0, 4)
    events[3].type = type
    events[3].payload.decision = "deny"
    if (type.endsWith("recovered")) {
      delete events[3].payload.decision
      events[3].payload.outcome = "failed"
    }
    const spans = flatten(buildTrace("s1", events, T0 + 99999), new Set()).map(({ span }) => span)
    const tools = spans.filter((s) => s.kind === "tool")
    assert.equal(tools.length, 1)
    assert.equal(tools[0].status, "error")
    assert.equal(tools[0].end, T0 + 100)
    assert.equal(spans.find((s) => s.kind === "wait").end, T0 + 100)
  })
}

test("a denial followed by tool failure still describes one call", () => {
  const events = approvalEvents({ runtime: true })
  events[3].type = "tool.call.approval_denied"
  events.splice(4, 1)
  events[4].type = "tool.call.failed"
  const tools = flatten(buildTrace("s1", events, T0 + 99999), new Set())
    .map(({ span }) => span)
    .filter((s) => s.kind === "tool")
  assert.equal(tools.length, 1)
  assert.equal(tools[0].status, "error")
  assert.equal(tools[0].end, T0 + 200)
})

test("a recovered completed call settles without pretending it executed again", () => {
  const events = approvalEvents().slice(0, 4)
  events[3].type = "tool.approval.recovered"
  events[3].payload.outcome = "completed"
  const tools = flatten(buildTrace("s1", events, T0 + 99999), new Set())
    .map(({ span }) => span)
    .filter((s) => s.kind === "tool")
  assert.equal(tools.length, 1)
  assert.equal(tools[0].status, "ok")
  assert.equal(tools[0].end, T0 + 100)
})

test("per-event approval aliases are not treated as stable IDs", () => {
  const events = approvalEvents({ nextTurn: true })
  for (const event of events.slice(2))
    event.payload.approval_id = `cayu_event_${event.sequence}:approval_id`
  const tools = flatten(buildTrace("s1", events, T0 + 99999), new Set())
    .map(({ span }) => span)
    .filter((s) => s.kind === "tool")
  assert.equal(tools.length, 1)
  assert.equal(tools[0].status, "ok")
})

test("parallel approvals use explicit IDs for reversed decisions and settle both waits", () => {
  const events = approvalEvents({ runtime: true })
  const first = events[2]
  events.splice(3, 0, {
    ...first,
    id: "approval-b",
    sequence: 3.5,
    payload: { ...first.payload, approval_id: "approval-b" },
  })
  for (const event of events.slice(4)) event.payload.approval_id = "approval-b"
  events.push(
    ev(300, "tool.call.approved", {
      approval_id: "approval-1",
      tool_call_id: "[PRIVATE_EVENT_AUTHORITY]",
    }),
    ev(
      310,
      "tool.call.started",
      { approval_id: "approval-1", tool_call_id: "[PRIVATE_EVENT_AUTHORITY]" },
      { tool_name: "write_file", interaction_id: "i2" },
    ),
    ev(
      400,
      "tool.call.completed",
      { approval_id: "approval-1", tool_call_id: "[PRIVATE_EVENT_AUTHORITY]" },
      { tool_name: "write_file", interaction_id: "i2" },
    ),
  )
  const spans = flatten(buildTrace("s1", events, T0 + 99999), new Set()).map(({ span }) => span)
  assert.deepEqual(
    spans
      .filter((s) => s.kind === "tool")
      .map((s) => s.end)
      .sort(),
    [T0 + 200, T0 + 400],
  )
  assert.deepEqual(
    spans
      .filter((s) => s.kind === "wait")
      .map((s) => s.end)
      .sort(),
    [T0 + 100, T0 + 300],
  )
})

test("parallel hidden started calls do not guess a completion by queue order", () => {
  seq = 0
  const events = [
    ev(0, "tool.call.started", {}, { tool_name: "write_file" }),
    ev(10, "tool.call.started", {}, { tool_name: "write_file" }),
    ev(20, "tool.call.completed", {}, { tool_name: "write_file" }),
  ]
  const tools = flatten(buildTrace("s1", events, T0 + 99999), new Set())
    .map(({ span }) => span)
    .filter((s) => s.kind === "tool")
  assert.equal(tools.length, 3)
  assert.equal(tools.find((s) => s.status === "ok").start, T0 + 20)
})

test("a real call ID appearing at start is preserved and replaces the inferred approval ID", () => {
  const events = approvalEvents()
  events[4].payload.tool_call_id = "real-call"
  events[5].payload.tool_call_id = "real-call"
  const spans = flatten(buildTrace("s1", events, T0 + 99999), new Set()).map(({ span }) => span)
  const tools = spans.filter((s) => s.kind === "tool")
  assert.equal(tools.length, 1)
  assert.equal(tools[0].id, "tool:real-call")
  assert.equal(tools[0].status, "ok")
  assert.equal(spans.find((s) => s.kind === "wait").end, T0 + 100)
})

for (const stepIds of ["hidden", "missing", "stable"]) {
  test(`an ambiguous round does not poison later sequential calls (${stepIds} model steps)`, () => {
    seq = 0
    const stepPayload = (step) =>
      stepIds === "missing"
        ? {}
        : {
            model_step_id: stepIds === "stable" ? `step-${step}` : "[PRIVATE_EVENT_AUTHORITY]",
          }
    const callPayload = (step) => ({
      ...stepPayload(step),
      tool_call_id: `cayu_event_${seq + 1}:tool_call_id`,
      tool_round_id: `cayu_event_${seq + 1}:tool_round_id`,
    })
    const toolEvent = (ms, type, step) => ev(ms, type, callPayload(step), { tool_name: "read" })
    const events = [
      ev(0, "model.started", stepPayload(1)),
      ev(10, "model.completed", stepPayload(1)),
      toolEvent(20, "tool.call.started", 1),
      toolEvent(30, "tool.call.started", 1),
      toolEvent(40, "tool.call.completed", 1),
      toolEvent(50, "tool.call.completed", 1),
      ev(60, "model.started", stepPayload(2)),
      ev(70, "model.completed", stepPayload(2)),
      toolEvent(80, "tool.call.started", 2),
      toolEvent(85, "runner.exec.started", 2),
      toolEvent(86, "runner.exec.completed", 2),
      toolEvent(90, "tool.call.completed", 2),
      ev(100, "model.started", stepPayload(3)),
      ev(110, "model.completed", stepPayload(3)),
      toolEvent(120, "tool.call.started", 3),
      toolEvent(130, "tool.call.failed", 3),
    ]
    const root = buildTrace("s1", events, T0 + 99999)
    const turn = root.children[0]
    const models = turn.children.filter((s) => s.kind === "model")
    assert.deepEqual(
      models.map((s) => [s.start, s.end, s.status]),
      [
        [T0, T0 + 10, "ok"],
        [T0 + 60, T0 + 70, "ok"],
        [T0 + 100, T0 + 110, "ok"],
      ],
    )
    assert.deepEqual(
      models.map((s) => s.events.filter((e) => e.type.startsWith("model.")).map((e) => e.type)),
      Array.from({ length: 3 }, () => ["model.started", "model.completed"]),
    )
    assert.equal(
      turn.events.some((e) => e.type.startsWith("model.")),
      false,
    )
    assert.deepEqual(
      models.map((s) => s.children.map((c) => c.kind)),
      [["round"], ["tool"], ["tool"]],
    )
    assert.equal(
      models[0].children[0].status,
      "running",
      "original parallel ambiguity stays visible",
    )
    assert.equal(models[1].children[0].status, "ok")
    assert.equal(models[2].children[0].status, "error")
    const tools = flatten(root, new Set())
      .map(({ span }) => span)
      .filter((s) => s.kind === "tool")
    assert.equal(
      tools.length,
      6,
      "four separate ambiguous observations plus two linked sequential calls",
    )
    const ambiguous = tools.filter((s) => s.start < T0 + 60)
    assert.equal(ambiguous.length, 4)
    assert.equal(ambiguous.filter((s) => s.status === "running").length, 2)
    const later = tools.filter((s) => s.start >= T0 + 60)
    assert.equal(later[0].children.length, 1, "the later command belongs only to its scoped call")
    assert.equal(later[0].children[0].end, T0 + 86)
    assert.deepEqual(
      later.map((s) => [s.start, s.end, s.status, s.events.length]),
      [
        [T0 + 80, T0 + 90, "ok", 4],
        [T0 + 120, T0 + 130, "error", 2],
      ],
    )
  })
}

test("distinct public rounds scope hidden calls even without a model-start boundary", () => {
  seq = 0
  const toolEvent = (ms, type, round) =>
    ev(ms, type, { tool_round_id: round }, { tool_name: "read" })
  const events = [
    toolEvent(0, "tool.call.started", "round-1"),
    toolEvent(10, "tool.call.started", "round-1"),
    toolEvent(20, "tool.call.completed", "round-1"),
    toolEvent(30, "tool.call.completed", "round-1"),
    toolEvent(40, "tool.call.started", "round-2"),
    toolEvent(50, "tool.call.completed", "round-2"),
  ]
  const tools = flatten(buildTrace("s1", events, T0 + 99999), new Set())
    .map(({ span }) => span)
    .filter((s) => s.kind === "tool")
  const later = tools.filter((s) => s.start >= T0 + 40)
  assert.equal(later.length, 1)
  assert.equal(later[0].end, T0 + 50)
  assert.equal(later[0].status, "ok")
})

for (const ids of ["missing", "hidden", "explicit"]) {
  for (const ending of ["model.completed", "model.failed"]) {
    for (const terminal of [null, "ok", "error"]) {
      test(`model spans own their outcomes and events (${ids}, ${ending}, session ${terminal ?? "live"})`, () => {
        seq = 0
        const identity = (step) =>
          ids === "missing"
            ? {}
            : {
                model_step_id: ids === "explicit" ? `model-${step}` : "[PRIVATE_EVENT_AUTHORITY]",
              }
        const events = [
          ev(0, "interaction.started"),
          ev(10, "model.started", identity(1)),
          ev(20, "model.text.delta", { ...identity(1), text: "first" }),
          ev(30, ending, {
            ...identity(1),
            status: ending === "model.failed" ? "failed" : "completed",
          }),
          ev(40, "model.started", identity(2)),
          ev(50, "model.text.delta", { ...identity(2), text: "second" }),
          ev(60, "model.completed", { ...identity(2), status: "completed" }),
          ev(70, "interaction.completed", { status: "completed" }),
        ]
        if (terminal)
          events.push(
            ev(
              80,
              terminal === "ok" ? "session.completed" : "session.failed",
              {},
              { interaction_id: null },
            ),
          )
        const root = buildTrace("s1", events, T0 + 99999, terminal)
        const tree = flatten(root, new Set()).map(({ span, depth }) => [span.kind, depth])
        assert.deepEqual(tree, [
          ["session", 0],
          ["interaction", 1],
          ["model", 2],
          ["model", 2],
        ])
        const turn = root.children[0]
        const [first, second] = turn.children
        assert.deepEqual(
          [first.start, first.end, first.status],
          [T0 + 10, T0 + 30, ending === "model.failed" ? "error" : "ok"],
        )
        assert.deepEqual([second.start, second.end, second.status], [T0 + 40, T0 + 60, "ok"])
        assert.deepEqual(
          first.events.map((e) => e.id),
          events.slice(1, 4).map((e) => e.id),
        )
        assert.deepEqual(
          second.events.map((e) => e.id),
          events.slice(4, 7).map((e) => e.id),
        )
        assert.deepEqual(
          turn.events.map((e) => e.type),
          ["interaction.started", "interaction.completed"],
        )
        assert.deepEqual([turn.start, turn.end, turn.status], [T0, T0 + 70, "ok"])
        assert.deepEqual(
          [root.status, root.end],
          [terminal ?? "running", terminal ? T0 + 80 : T0 + 99999],
        )
        if (ids === "explicit")
          assert.deepEqual([first.id, second.id], ["model:model-1", "model:model-2"])
      })
    }
  }
}

test("missing model terminal IDs retain an explicit start and never cross interactions", () => {
  seq = 0
  const events = [
    ev(0, "model.started", { model_step_id: "real-model" }),
    ev(10, "model.completed", { status: "failed" }),
    ev(20, "model.completed", {}, { interaction_id: "other-turn" }),
  ]
  const root = buildTrace("s1", events, T0 + 99999, "ok")
  const [first, other] = root.children
  assert.equal(first.children.length, 1)
  assert.equal(first.children[0].id, "model:real-model")
  assert.equal(first.children[0].status, "error")
  assert.equal(first.children[0].end, T0 + 10)
  assert.deepEqual(
    first.children[0].events.map((e) => e.id),
    events.slice(0, 2).map((e) => e.id),
  )
  assert.equal(other.children.length, 0, "a terminal without a start in its turn remains unlinked")
  assert.deepEqual(
    other.events.map((e) => e.id),
    [events[2].id],
  )
})
