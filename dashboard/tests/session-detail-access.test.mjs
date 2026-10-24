import assert from "node:assert/strict"
import test from "node:test"
import { applyViewerAccess } from "../src/lib/dashboard-access.ts"

test("members can read structured transcript evidence without acquiring execution rights", () => {
  const member = { role: "member", full_access: false }
  assert.equal(
    applyViewerAccess(
      { enabled: true },
      { kind: "surface", surface: "structured_output_transcript" },
      member,
    ).enabled,
    true,
  )
  assert.equal(
    applyViewerAccess(
      { enabled: false, unavailable_reason: "not_configured" },
      { kind: "surface", surface: "structured_output_transcript" },
      member,
    ).enabled,
    false,
  )
  assert.equal(
    applyViewerAccess(
      { enabled: true },
      { kind: "mutation", mutation: "session_execution" },
      member,
    ).enabled,
    false,
  )
})
