import assert from "node:assert/strict"
import test from "node:test"
import * as history from "../src/lib/session-history.ts"

test("latest transcript probes then reads a bounded tail beyond ten thousand messages", async () => {
  const reads = []
  const page = await history.loadTranscriptWindow(null, 100, 3, async (query) => {
    reads.push(query)
    return { ...query, total_messages: 12001, has_more: query.offset + query.limit < 12001 }
  })
  assert.deepEqual(reads, [
    { offset: 0, limit: 1 },
    { offset: 11901, limit: 100 },
  ])
  assert.equal(page.has_more, false)
})

test("explicit history pages are read once without chasing the latest tail", async () => {
  const reads = []
  const page = await history.loadTranscriptWindow(
    { offset: 200, limit: 50 },
    100,
    3,
    async (query) => {
      reads.push(query)
      return { ...query, total_messages: 12001, has_more: true }
    },
  )
  assert.deepEqual(reads, [{ offset: 200, limit: 50 }])
  assert.equal(page.offset, 200)
})

test("empty and one-message transcripts keep the probe without another request", async () => {
  for (const total_messages of [0, 1]) {
    let reads = 0
    await history.loadTranscriptWindow(null, 100, 3, async () => {
      reads++
      return { total_messages, has_more: false }
    })
    assert.equal(reads, 1)
  }
})

test("aborting after the probe propagates the read cancellation and never publishes a partial window", async () => {
  const controller = new AbortController()
  let reads = 0
  await assert.rejects(
    history.loadTranscriptWindow(null, 100, 3, async () => {
      controller.signal.throwIfAborted()
      reads++
      controller.abort()
      return { total_messages: 12001, has_more: true }
    }),
    { name: "AbortError" },
  )
  assert.equal(reads, 1)
})

test("continuous appends exhaust the existing bounded tail retry instead of looping", async () => {
  let reads = 0
  await assert.rejects(
    history.loadTranscriptWindow(null, 100, 3, async () => {
      reads++
      return { total_messages: 12000 + reads * 100, has_more: true }
    }),
    /growing too quickly/,
  )
  assert.equal(reads, 4)
})
