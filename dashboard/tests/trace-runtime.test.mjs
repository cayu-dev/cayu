import assert from "node:assert/strict"
import test from "node:test"
import {
  buildRuntimeTrace,
  mergeTracePages,
  traceSpanForPosition,
} from "../src/next/app/trace-runtime.ts"

const span = (id, kind, parent_id, extra = {}) => ({
  id,
  kind,
  parent_id,
  name: kind,
  status: "completed",
  started_at: "2026-01-01T00:00:00Z",
  ended_at: "2026-01-01T00:00:01Z",
  event_ids: [],
  transcript_links: [],
  ...extra,
})
const page = (spans, events = [], extra = {}) => ({
  session_id: "s",
  session_instance_id: "i",
  root_span_id: "session",
  through_sequence: 5,
  next_sequence: 5,
  has_more: false,
  spans,
  events,
  ...extra,
})

test("exact trace preserves reused call identities, null timing and every outcome", () => {
  const source = page([
    span("session", "session", null),
    span("tool:1", "tool", "session", {
      status: "interrupted",
      started_at: null,
      transcript_links: [{ message_index: 2, part_index: 0, tool_call_id: "same" }],
    }),
    span("tool:2", "tool", "session", {
      status: "blocked",
      transcript_links: [{ message_index: 4, part_index: 0, tool_call_id: "same" }],
    }),
    span("knowledge:3", "knowledge", "session"),
  ])
  const tree = buildRuntimeTrace(source)
  assert.equal(tree.linkedByOrder, false)
  assert.equal(tree.children[0].status, "interrupted")
  assert.equal(tree.children[0].startKnown, false)
  assert.equal(tree.children[1].status, "blocked")
  assert.equal(tree.children[2].kind, "knowledge")
  assert.equal(traceSpanForPosition(source, { messageIndex: 2, partIndex: 0 }), "tool:1")
  assert.equal(traceSpanForPosition(source, { messageIndex: 4, partIndex: 0 }), "tool:2")
  assert.equal(traceSpanForPosition(source, { messageIndex: 3, partIndex: 0 }), undefined)
})

test("page merge unions exact event membership and rejects another incarnation", () => {
  const root = span("session", "session", null)
  const first = page(
    [root, span("tool:1", "tool", "session", { event_ids: ["a"] })],
    [{ id: "a", sequence: 1 }],
  )
  const next = page(
    [root, span("tool:1", "tool", "session", { event_ids: ["b"] })],
    [{ id: "b", sequence: 2 }],
  )
  const merged = mergeTracePages(first, next)
  assert.deepEqual(merged.spans[1].event_ids, ["a", "b"])
  assert.deepEqual(
    buildRuntimeTrace(merged).children[0].events.map((e) => e.id),
    ["a", "b"],
  )
  assert.throws(() => mergeTracePages(first, { ...next, session_instance_id: "new" }), /changed/)
})

test("older history cannot regress a live span's terminal evidence", () => {
  const root = span("session", "session", null)
  const live = page(
    [root, span("tool:1", "tool", "session", { status: "completed", event_ids: ["done"] })],
    [],
    { through_sequence: 20 },
  )
  const older = page(
    [
      root,
      span("tool:1", "tool", "session", {
        status: "running",
        ended_at: null,
        event_ids: ["start"],
      }),
    ],
    [],
    { through_sequence: 10 },
  )
  const result = mergeTracePages(live, older)
  assert.equal(result.through_sequence, 20)
  assert.equal(result.spans[1].status, "completed")
  assert.deepEqual(result.spans[1].event_ids, ["done", "start"])
})
