import assert from "node:assert/strict"
import test from "node:test"
import {
  argPreview,
  buildBlocks,
  constantColumns,
  formatDuration,
  relativeTime,
  stepLabel,
  workSummary,
} from "../src/next/app/session-model.ts"

const correlation = { model_step_id: "step", model_attempt_id: "attempt", tool_round_id: "round" }
const msg = (index, role, content) => ({ index, interaction_id: null, role, content })

test("buildBlocks groups tool calls and thinking between messages into one work block", () => {
  const blocks = buildBlocks([
    msg(0, "system", [{ type: "text", text: "You are helpful." }]),
    msg(1, "user", [{ type: "text", text: "Check the ledger." }]),
    msg(2, "assistant", [
      { type: "thinking", text: "**Looking at accounts**" },
      {
        type: "tool_call",
        ...correlation,
        tool_call_id: "a",
        tool_name: "query",
        arguments: { account: "2100" },
      },
    ]),
    msg(3, "tool", [
      {
        type: "tool_result",
        ...correlation,
        tool_call_id: "a",
        tool_name: "query",
        content: "214 lines",
      },
    ]),
    msg(4, "assistant", [
      { type: "tool_call", ...correlation, tool_call_id: "b", tool_name: "query", arguments: {} },
    ]),
    msg(5, "tool", [
      {
        type: "tool_result",
        ...correlation,
        tool_call_id: "b",
        tool_name: "query",
        content: "boom",
        is_error: true,
      },
    ]),
    msg(6, "assistant", [{ type: "text", text: "Done." }]),
  ])
  assert.deepEqual(
    blocks.map((b) => b.kind),
    ["user", "work", "text"],
  )
  const work = blocks[1]
  assert.deepEqual(work.thoughts, ["Looking at accounts"])
  assert.equal(work.tools.length, 2)
  assert.equal(work.tools[0].result, "214 lines")
  assert.equal(work.tools[1].error, true)
})

test("buildBlocks renders public structured-output parts from assistant and tool messages", () => {
  for (const role of ["assistant", "tool"]) {
    for (const output of [{ findings: [] }, [1, 2], "ready", false, 0, null]) {
      const blocks = buildBlocks([
        msg(0, "user", [{ type: "text", text: "Review it." }]),
        msg(1, role, [{ type: "structured_output", name: "Answer", output }]),
      ])
      assert.deepEqual(blocks.at(-1), { kind: "output", key: "m1.0", value: output })
    }
  }
})

test("structured display evidence is independent of tool names and preserves user parts as raw evidence", () => {
  const blocks = buildBlocks([
    msg(0, "user", [{ type: "structured_output", output: { untrusted: true } }]),
    msg(1, "assistant", [
      {
        type: "tool_call",
        ...correlation,
        tool_call_id: "s",
        tool_name: "__cayu_submit_structured_output",
        arguments: {},
      },
    ]),
    msg(2, "tool", [
      {
        type: "tool_result",
        ...correlation,
        tool_call_id: "s",
        content: "accepted",
        structured: { output: { guessed: true } },
      },
    ]),
  ])
  assert.deepEqual(
    blocks.map((block) => block.kind),
    ["part", "work"],
  )
  assert.deepEqual(blocks[0].part, { type: "structured_output", output: { untrusted: true } })
  assert.equal(blocks[1].tools[0].result, "accepted")
})

test("a tool call without a result is still pending", () => {
  const [, work] = buildBlocks([
    msg(0, "user", [{ type: "text", text: "Hold it." }]),
    msg(1, "assistant", [
      {
        type: "tool_call",
        ...correlation,
        tool_call_id: "h",
        tool_name: "hold_payment",
        arguments: { invoice: "INV-1" },
      },
    ]),
  ])
  assert.equal(work.tools[0].running, true)
  assert.equal(work.tools[0].result, null)
})

test("workSummary phrases each tool from its display label, or its name", () => {
  const t = (name) => ({ name })
  const displays = {
    read_file: { verb: "Read", noun: "file" },
    search_text: { verb: "Searched" },
    list_files: { verb: "Listed files" },
  }
  assert.equal(
    workSummary(
      [t("read_file"), t("search_text"), t("read_file"), t("list_files"), t("search_text")],
      displays,
    ),
    "Read 2 files, searched 2 times, listed files",
  )
  assert.equal(workSummary([t("read_file")], displays), "Read a file")
  assert.equal(
    workSummary([t("load")], { load: { verb: "Loaded", noun: "invoice batch" } }),
    "Loaded an invoice batch",
  )
  assert.equal(workSummary([t("query"), t("query")]), "Used query 2 times")
})

test("stepLabel uses the display verb and the first string argument", () => {
  const tool = { name: "read_file", args: { path: "src/app.py" } }
  assert.deepEqual(stepLabel(tool, { read_file: { verb: "Read", noun: "file" } }), {
    verb: "Read",
    target: "src/app.py",
  })
  assert.deepEqual(stepLabel(tool), { verb: "read_file", target: "src/app.py" })
})

test("argPreview shows the first string argument", () => {
  assert.equal(argPreview({ limit: 10, path: "src/a.py" }), "src/a.py")
  assert.equal(argPreview({ limit: 10 }), '{"limit":10}')
  assert.equal(argPreview({}), "")
})

test("relativeTime and formatDuration", () => {
  const now = Date.parse("2026-10-22T04:58:00Z")
  assert.equal(relativeTime(now - 20_000, now), "now")
  assert.equal(relativeTime(now - 10 * 60_000, now), "10m")
  assert.equal(relativeTime(Date.parse("2026-10-21T06:00:00Z"), now), "Yesterday")
  assert.equal(relativeTime(Date.parse("2026-10-14T18:00:00Z"), now), "Oct 14")
  assert.equal(formatDuration(null), "–")
  assert.equal(formatDuration(42), "42s")
  assert.equal(formatDuration(138), "2m 18s")
})

test("constantColumns hides columns that never vary", () => {
  const rows = [
    { agent: "a", model: "m" },
    { agent: "b", model: "m" },
  ]
  const result = constantColumns(rows, { Specialist: (r) => r.agent, Model: (r) => r.model })
  assert.deepEqual(result, { constant: { Model: "m" }, varying: ["Specialist"] })
})

test("structured output stays hidden when its public capability is unavailable", () => {
  const messages = [
    msg(0, "assistant", [
      { type: "text", text: "Original answer" },
      { type: "structured_output", name: null, output: { answer: "validated" } },
    ]),
  ]
  assert.deepEqual(buildBlocks(messages, false), [
    { kind: "text", key: "m0.0", text: "Original answer" },
  ])
})

test("conversation retains public evidence and unmatched tool results at page boundaries", () => {
  const citation = { type: "citation", url: "https://example.test/source", title: "Source" }
  const hosted = { type: "hosted_tool_call", status: "completed", action: { sources: [] } }
  const orphan = {
    type: "tool_result",
    ...correlation,
    tool_call_id: "previous-page",
    tool_name: "read",
    content: "Historical result",
  }
  const blocks = buildBlocks([
    msg(100, "assistant", [citation, hosted]),
    msg(101, "tool", [orphan]),
  ])
  assert.deepEqual(
    blocks.map((block) => block.part),
    [citation, hosted, orphan],
  )
})

test("conversation matches repeated tool IDs only within their exact public correlation tuple", () => {
  const messages = ["first", "second"].flatMap((round, i) => [
    msg(i * 2, "assistant", [
      {
        type: "tool_call",
        ...correlation,
        tool_call_id: "same",
        tool_name: "read",
        tool_round_id: round,
        arguments: {},
      },
    ]),
    msg(i * 2 + 1, "tool", [
      {
        type: "tool_result",
        ...correlation,
        tool_call_id: "same",
        tool_name: "read",
        tool_round_id: round,
        content: round,
      },
    ]),
  ])
  const tools = buildBlocks(messages).flatMap((block) => block.tools ?? [])
  assert.deepEqual(
    tools.map((tool) => tool.result),
    ["first", "second"],
  )
  assert.equal(new Set(tools.map((tool) => tool.key)).size, 2)
  assert.deepEqual(
    tools.map((tool) => [tool.messageIndex, tool.partIndex]),
    [
      [0, 0],
      [2, 0],
    ],
  )
})

test("conversation retains ambiguous, hidden and mismatched tool results independently", () => {
  for (const identity of ["duplicate", "[REDACTED_SECRET]", "prefix[REDACTED_SECRET]", ""]) {
    const call = {
      type: "tool_call",
      ...correlation,
      tool_call_id: identity,
      tool_name: "read",
      arguments: {},
    }
    const result = {
      type: "tool_result",
      ...correlation,
      tool_call_id: identity,
      tool_name: "read",
      content: "evidence",
    }
    const blocks = buildBlocks([
      msg(0, "assistant", identity === "duplicate" ? [call, call] : [call]),
      msg(1, "tool", [result]),
    ])
    assert.deepEqual(
      blocks.flatMap((block) => block.tools ?? []).map((tool) => tool.result),
      identity === "duplicate" ? [null, null] : [null],
    )
    assert.deepEqual(
      blocks.filter((block) => block.kind === "part").map((block) => block.part),
      [result],
    )
  }
  const blocks = buildBlocks([
    msg(0, "assistant", [
      {
        type: "tool_call",
        ...correlation,
        tool_call_id: "unique",
        tool_name: "read",
        model_attempt_id: "a",
      },
    ]),
    msg(1, "tool", [
      {
        type: "tool_result",
        ...correlation,
        tool_call_id: "unique",
        tool_name: "read",
        model_attempt_id: "b",
        content: "wrong attempt",
      },
    ]),
  ])
  assert.equal(blocks[0].tools[0].result, null)
  assert.equal(blocks[1].part.content, "wrong attempt")
})

test("conversation preserves suppressed argument state without presenting an empty argument object", () => {
  const blocks = buildBlocks([
    msg(0, "assistant", [
      {
        type: "tool_call",
        ...correlation,
        tool_call_id: "private-call",
        tool_name: "read",
        arguments: {},
        arguments_state: "unavailable",
      },
    ]),
  ])
  assert.equal(blocks[0].tools[0].argumentsUnavailable, true)
  assert.equal(blocks[0].tools[0].args, null)
  assert.equal(stepLabel(blocks[0].tools[0]).target, "")
})

test("hiding thinking preserves exact native part positions and all other evidence", () => {
  const messages = [
    {
      index: 4,
      role: "assistant",
      content: [
        { type: "thinking", text: "hidden" },
        { type: "tool_call", tool_call_id: "a", tool_name: "read", arguments: {} },
        { type: "tool_call", tool_call_id: "b", tool_name: "read", arguments: {} },
      ],
    },
    {
      index: 5,
      role: "user",
      content: [
        { type: "thinking", text: "also hidden" },
        { type: "file", file_url: "asset" },
      ],
    },
  ]
  const blocks = buildBlocks(messages, true, false)
  assert.deepEqual(blocks[0].thoughts, [])
  assert.deepEqual(
    blocks[0].tools.map((tool) => [tool.messageIndex, tool.partIndex]),
    [
      [4, 1],
      [4, 2],
    ],
  )
  assert.equal(blocks[1].key, "m5.1")
  assert.equal(blocks[1].part.type, "file")
})
