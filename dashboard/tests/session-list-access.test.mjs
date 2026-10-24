import assert from "node:assert/strict"
import test from "node:test"
import { applyViewerAccess } from "../src/lib/dashboard-access.ts"

test("members read canonical session list surfaces but cannot start", () => {
  const member = { role: "member", full_access: false }
  for (const surface of [
    "session_trace",
    "session_overview",
    "session_overview_queries",
    "session_display_metadata",
  ]) {
    assert.equal(
      applyViewerAccess({ enabled: true }, { kind: "surface", surface }, member).enabled,
      true,
    )
  }
  assert.equal(
    applyViewerAccess(
      { enabled: true },
      { kind: "mutation", mutation: "session_execution" },
      member,
    ).enabled,
    false,
  )
})
