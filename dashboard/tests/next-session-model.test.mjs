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

const msg = (index, role, content) => ({ index, interaction_id: null, role, content })

test("buildBlocks groups tool calls and thinking between messages into one work block", () => {
  const blocks = buildBlocks([
    msg(0, "system", [{ type: "text", text: "You are helpful." }]),
    msg(1, "user", [{ type: "text", text: "Check the ledger." }]),
    msg(2, "assistant", [
      { type: "thinking", text: "**Looking at accounts**" },
      { type: "tool_call", tool_call_id: "a", tool_name: "query", arguments: { account: "2100" } },
    ]),
    msg(3, "tool", [
      { type: "tool_result", tool_call_id: "a", tool_name: "query", content: "214 lines" },
    ]),
    msg(4, "assistant", [
      { type: "tool_call", tool_call_id: "b", tool_name: "query", arguments: {} },
    ]),
    msg(5, "tool", [
      {
        type: "tool_result",
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

test("buildBlocks turns the structured-output tool into an output block", () => {
  const blocks = buildBlocks([
    msg(0, "user", [{ type: "text", text: "Review it." }]),
    msg(1, "assistant", [
      {
        type: "tool_call",
        tool_call_id: "s",
        tool_name: "__cayu_submit_structured_output",
        arguments: {},
      },
    ]),
    msg(2, "tool", [
      {
        type: "tool_result",
        tool_call_id: "s",
        tool_name: "__cayu_submit_structured_output",
        content: "ok",
        structured: { output: { findings: [] } },
      },
    ]),
  ])
  assert.deepEqual(blocks.at(-1), { kind: "output", key: "m1.0", value: { findings: [] } })
})

test("a tool call without a result is still pending", () => {
  const [, work] = buildBlocks([
    msg(0, "user", [{ type: "text", text: "Hold it." }]),
    msg(1, "assistant", [
      {
        type: "tool_call",
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
