import assert from "node:assert/strict"
import test from "node:test"

import { QueryClient, QueryObserver } from "@tanstack/react-query"

import { queryReadErrorIsFatal } from "../src/lib/session-history.ts"

test("a failed detail refetch keeps confirmed query data usable", async () => {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  let shouldFail = false
  const observer = new QueryObserver(client, {
    queryKey: ["session", "session-1"],
    queryFn: async () => {
      if (shouldFail) throw new Error("temporary read failure")
      return { id: "session-1" }
    },
  })
  const unsubscribe = observer.subscribe(() => {})

  try {
    await observer.refetch()
    shouldFail = true
    await observer.refetch()

    const result = observer.getCurrentResult()
    assert.equal(result.isError, true)
    assert.equal(result.isRefetchError, true)
    assert.deepEqual(result.data, { id: "session-1" })
    assert.equal(queryReadErrorIsFatal(result.isError, result.data !== undefined), false)
  } finally {
    unsubscribe()
    client.clear()
  }
})

test("a successful empty reconciliation clears a retained refetch error", async () => {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  let shouldFail = false
  const key = ["session-events", "session-1", "latest"]
  const observer = new QueryObserver(client, {
    queryKey: key,
    queryFn: async () => {
      if (shouldFail) throw new Error("temporary read failure")
      return { events: [], has_more: false }
    },
  })
  const unsubscribe = observer.subscribe(() => {})

  try {
    await observer.refetch()
    shouldFail = true
    await observer.refetch()
    assert.equal(observer.getCurrentResult().isError, true)

    client.setQueryData(key, (existing) => existing)

    const result = observer.getCurrentResult()
    assert.equal(result.isError, false)
    assert.deepEqual(result.data, { events: [], has_more: false })
  } finally {
    unsubscribe()
    client.clear()
  }
})

test("a failed immutable summary can be retried explicitly", async () => {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  let shouldFail = true
  const observer = new QueryObserver(client, {
    queryKey: ["session-summary", "session-1", "terminal-revision"],
    queryFn: async () => {
      if (shouldFail) throw new Error("temporary summary failure")
      return { events: { total_events: 3 } }
    },
    staleTime: Number.POSITIVE_INFINITY,
  })
  const unsubscribe = observer.subscribe(() => {})

  try {
    await observer.refetch()
    assert.equal(observer.getCurrentResult().isError, true)

    shouldFail = false
    await observer.refetch()

    assert.equal(observer.getCurrentResult().isError, false)
    assert.deepEqual(observer.getCurrentResult().data, { events: { total_events: 3 } })
  } finally {
    unsubscribe()
    client.clear()
  }
})
