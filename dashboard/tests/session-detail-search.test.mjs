import assert from "node:assert/strict"
import test from "node:test"
import {
  historySearchPatch,
  validateSessionDetailSearch,
  workflowSearchFromSession,
  workflowSearchPatch,
} from "../src/lib/session-detail-search.ts"

test("Session preserves canonical workflow expansion and filters alongside event history", () => {
  const result = validateSessionDetailSearch({
    expanded_session_id: ["child-b", "child-a"],
    expanded_task_id: "task",
    status: ["failed", "running"],
    node_type: "task",
    range: "7d",
    event_type: "tool_call_started",
    transcript_role: "assistant",
    event_id: "event",
  })
  assert.deepEqual(result.expanded_session_id, ["child-a", "child-b"])
  assert.deepEqual(result.expanded_task_id, ["task"])
  assert.deepEqual(result.status, ["running", "failed"])
  assert.deepEqual(result.node_type, ["task"])
  assert.equal(result.range, "7d")
  assert.equal(result.event_type, "tool_call_started")
  assert.equal(result.transcript_role, "assistant")
  assert.equal(result.view, "trace")
})

test("invalid workflow authority is retained as invalid instead of broadening silently", () => {
  const result = validateSessionDetailSearch({ expanded_session_id: [true], status: "future" })
  assert.equal(result.invalid, true)
})

test("an explicit Conversation view retains exact event filters without reopening Trace", () => {
  const result = validateSessionDetailSearch({
    view: "conversation",
    event_id: "event",
    event_type: "model.completed",
  })
  assert.equal(result.view, "conversation")
  assert.equal(result.event_id, "event")
  assert.equal(result.event_type, "model.completed")
})

test("workflow specialist and environment never leak into exact event filters", () => {
  const result = validateSessionDetailSearch({
    agent_name: "event-agent",
    environment_name: "event-env",
    workflow_agent_name: "child-agent",
    workflow_environment_name: "child-env",
  })
  assert.equal(result.agent_name, "event-agent")
  assert.equal(result.environment_name, "event-env")
  assert.equal(result.workflow_agent_name, "child-agent")
  assert.equal(result.workflow_environment_name, "child-env")
})

test("replacing either filter owner preserves its sibling and clears invalid workflow state", () => {
  const initial = validateSessionDetailSearch({
    view: "trace",
    event_type: "tool_call_started",
    transcript_role: "user",
    agent_name: "event-agent",
    workflow_agent_name: "child-agent",
    status: "bad",
  })
  const workflowReset = validateSessionDetailSearch({ ...initial, ...workflowSearchPatch({}) })
  assert.equal(workflowReset.invalid, undefined)
  assert.equal(workflowReset.agent_name, "event-agent")
  assert.equal(workflowSearchFromSession(workflowReset).agent_name, undefined)
  const historyReset = validateSessionDetailSearch({ ...initial, ...historySearchPatch({}) })
  assert.equal(historyReset.event_type, undefined)
  assert.equal(historyReset.agent_name, undefined)
  assert.equal(historyReset.event, undefined)
  assert.equal(historyReset.transcript_role, "user")
  assert.equal(workflowSearchFromSession(historyReset).agent_name, "child-agent")
})

test("legacy event links open the log, while explicit subsequent views remain authoritative", () => {
  const legacy = validateSessionDetailSearch({ event_id: "event" })
  assert.equal(legacy.view, "trace")
  assert.equal(legacy.log, "1")
  assert.equal(legacy.event, "event")
  assert.equal(validateSessionDetailSearch({ ...legacy, log: undefined }).log, undefined)
  const conversation = validateSessionDetailSearch({ ...legacy, view: "conversation" })
  assert.equal(conversation.view, "conversation")
  assert.equal(validateSessionDetailSearch({ event_id: "event", view: "trace" }).log, undefined)
})
