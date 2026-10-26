import assert from "node:assert/strict"
import test from "node:test"
import {
  applyViewerAccess,
  parseDashboardIdentity,
  ROLE_FORBIDDEN_REASON,
  roleForbiddenMessage,
} from "../src/lib/dashboard-access.ts"

const available = { enabled: true, unavailable_reason: null }
const member = { name: "Agent", runtime_version: "0.14.0", role: "member", full_access: false }

test("authenticated full_access, not a role spelling or null, grants write authority", () => {
  const write = { kind: "mutation", mutation: "session_execution" }
  for (const role of [null, "admin", "developer", "custom-owner"]) {
    assert.equal(
      applyViewerAccess(available, write, { ...member, role, full_access: true }).enabled,
      true,
    )
    assert.equal(
      applyViewerAccess(available, write, { ...member, role, full_access: false }).enabled,
      false,
    )
  }
  for (const identity of [null, undefined, {}, { ...member, full_access: "true" }]) {
    assert.equal(applyViewerAccess(available, write, identity).enabled, false)
  }
})

test("members keep read-and-decide while full-access operations are unavailable", () => {
  for (const mutation of ["pending_action_resolution", "knowledge_review"]) {
    assert.equal(applyViewerAccess(available, { kind: "mutation", mutation }, member).enabled, true)
  }
  for (const mutation of [
    "session_execution",
    "session_interruption",
    "session_annotations",
    "provider_operation_resolution",
    "task_lifecycle",
    "future_mutation",
  ]) {
    assert.deepEqual(applyViewerAccess(available, { kind: "mutation", mutation }, member), {
      enabled: false,
      unavailable_reason: "role_forbidden",
    })
  }
  for (const surface of [
    "dashboard",
    "tasks",
    "usage",
    "artifacts",
    "workflow",
    "reviewed_knowledge",
  ]) {
    assert.equal(applyViewerAccess(available, { kind: "surface", surface }, member).enabled, true)
  }
  for (const surface of ["evals", "evaluation_promotion", "browser_recordings"]) {
    assert.equal(applyViewerAccess(available, { kind: "surface", surface }, member).enabled, false)
  }
  assert.equal(
    applyViewerAccess(available, { kind: "surface", surface: "tasks", operation: "mutate" }, member)
      .enabled,
    false,
  )
})

test("deployment availability remains authoritative for privileged viewers", () => {
  const operation = { enabled: false, unavailable_reason: "not_configured" }
  assert.deepEqual(
    applyViewerAccess(
      operation,
      { kind: "mutation", mutation: "session_execution" },
      { ...member, full_access: true },
    ),
    operation,
  )
})

test("identity requires positive typed authority and keeps configured roles", () => {
  for (const role of [null, "member", "custom-reviewer"]) {
    const identity = { ...member, role }
    assert.deepEqual(parseDashboardIdentity(identity), identity)
  }
  for (const identity of [
    null,
    {},
    { ...member, full_access: 1 },
    { ...member, role: false },
    { ...member, name: null },
    { ...member, runtime_version: [] },
  ]) {
    assert.throws(() => parseDashboardIdentity(identity), /identity/)
  }
})

test("only a structured role_forbidden403 becomes the useful role reason", () => {
  assert.equal(
    roleForbiddenMessage(403, { code: "role_forbidden", message: "Your role can't use this." }),
    ROLE_FORBIDDEN_REASON,
  )
  assert.equal(roleForbiddenMessage(403, "role_forbidden"), null)
  assert.equal(roleForbiddenMessage(401, { code: "role_forbidden" }), null)
  assert.equal(roleForbiddenMessage(403, { code: "other" }), null)
})

test("members read Inbox query counts without mutation authority", () => {
  assert.equal(
    applyViewerAccess(available, { kind: "surface", surface: "inbox_queries" }, member).enabled,
    true,
  )
  assert.equal(
    applyViewerAccess(
      available,
      { kind: "surface", surface: "inbox_queries", operation: "mutate" },
      member,
    ).enabled,
    false,
  )
})

test("members may read native tool usage but never gain tool mutations", () => {
  const read = { kind: "surface", surface: "tool_usage" }
  assert.equal(applyViewerAccess(available, read, member).enabled, true)
  assert.equal(
    applyViewerAccess({ enabled: false, unavailable_reason: "unsupported" }, read, member).enabled,
    false,
  )
  assert.equal(
    applyViewerAccess(available, { ...read, operation: "mutate" }, member).enabled,
    false,
  )
})
