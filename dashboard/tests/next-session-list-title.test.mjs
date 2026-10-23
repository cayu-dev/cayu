import assert from "node:assert/strict"
import test from "node:test"
import { sessionDisplayTitle } from "../src/next/app/session-model.ts"

const user = (content) => ({ role: "user", content })

test("session title precedence is authored title, labels, first user text, then ID", () => {
  const record = { id: "session-1", labels: { tenant: "acme", case: "17" } }
  const message = user([{ type: "text", text: "Inspect the report" }])
  assert.equal(sessionDisplayTitle(record, "Quarterly review", message), "Quarterly review")
  assert.equal(sessionDisplayTitle(record, null, message), "tenant=acme · case=17")
  assert.equal(sessionDisplayTitle({ ...record, labels: {} }, null, message), "Inspect the report")
  assert.equal(sessionDisplayTitle({ ...record, labels: {} }, null, null), "session-1")
})

test("fallback uses only text from the first user message, never other roles or tool payloads", () => {
  const record = { id: "session-1", labels: {} }
  assert.equal(
    sessionDisplayTitle(record, null, {
      role: "system",
      content: [{ type: "text", text: "secret preamble" }],
    }),
    "session-1",
  )
  assert.equal(
    sessionDisplayTitle(record, null, user([{ type: "image", text: "not a text part" }])),
    "session-1",
  )
  assert.equal(sessionDisplayTitle(record, null, user([{ type: "text", text: "  " }])), "session-1")
  assert.equal(
    sessionDisplayTitle(
      record,
      null,
      user([
        { type: "text", text: "First " },
        { type: "text", text: "message" },
      ]),
    ),
    "First message",
  )
})
