import assert from "node:assert/strict"
import test from "node:test"
import { sessionDisplayTitle, shortSessionTitle } from "../src/next/app/session-model.ts"

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

test("a title from the first message is its first line, cut at a word", () => {
  const record = { id: "session-1", labels: {} }
  const prompt =
    "Review 2026-09 close. Investigate the supplied records, reconcile every account and report."
  const title = sessionDisplayTitle(record, null, user([{ type: "text", text: prompt }]))
  assert.equal(title, "Review 2026-09 close. Investigate the supplied records…")
  assert.ok(title.length <= 61)
  assert.equal(
    sessionDisplayTitle(record, null, user([{ type: "text", text: "Close September\n\nDetails" }])),
    "Close September",
  )
  assert.equal(shortSessionTitle("x".repeat(80)), `${"x".repeat(60)}…`)
  // A cut counts characters, so it never splits an emoji into half a surrogate pair.
  assert.equal(shortSessionTitle(`a${"😀".repeat(80)}`), `a${"😀".repeat(59)}…`)
  assert.equal(sessionDisplayTitle(record, "An authored title ".repeat(5), null).length, 90)
})
