import assert from "node:assert/strict"
import test from "node:test"

import { hostedSearchSourceRows } from "../src/lib/hosted-search-sources.ts"

test("API evidence retains its name without becoming a link", () => {
  const rows = hostedSearchSourceRows([
    { type: "api", name: "redacted" },
    { type: "api", name: "redacted-other", url: "https://example.com" },
  ])
  assert.deepEqual(
    rows.map(({ label, url }) => ({ label, url })),
    [
      { label: "redacted", url: null },
      { label: "redacted-other", url: null },
    ],
  )
})

test("duplicate API and URL sources keep distinct stable keys", () => {
  const sources = [
    { type: "api", name: "redacted" },
    { type: "api", name: "redacted" },
    { type: "api", name: "redacted-other" },
    { type: "url", url: "https://example.com", title: "redacted" },
    { url: "https://example.com", title: "redacted" },
  ]
  const keys = hostedSearchSourceRows(sources).map((row) => row.key)
  assert.equal(new Set(keys).size, sources.length)
  assert.deepEqual(
    hostedSearchSourceRows(sources).map((row) => row.key),
    keys,
  )
})

test("URL and omitted-type evidence retain link destinations and labels", () => {
  const rows = hostedSearchSourceRows([
    { type: "url", url: "https://example.com", title: "Example" },
    { url: "https://example.org" },
  ])
  assert.deepEqual(
    rows.map(({ label, url }) => ({ label, url })),
    [
      { label: "Example", url: "https://example.com" },
      { label: "https://example.org", url: "https://example.org" },
    ],
  )
})
