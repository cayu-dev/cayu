import assert from "node:assert/strict"
import test from "node:test"
import { startPrivateBrowserViewer } from "../src/lib/browser-private-viewer.ts"

function frame(width = 16, height = 16) {
  const buffer = new ArrayBuffer(24)
  new Uint8Array(buffer).set([137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82])
  new DataView(buffer).setUint32(16, width)
  new DataView(buffer).setUint32(20, height)
  return buffer
}

test("intentional replacement waits for decode disposal and server purge acknowledgement", async () => {
  const sent = []
  let release
  let closed = 0
  let disposed = 0
  let retired = false
  const socket = { send: (value) => sent.push(value), close: () => closed++ }
  const viewer = startPrivateBrowserViewer(
    socket,
    "ticket",
    () => assert.fail(),
    () => {},
    () =>
      new Promise((resolve) => {
        release = resolve
      }),
  )
  socket.onopen()
  socket.onmessage({ data: "ready" })
  socket.onmessage({ data: frame() })
  await Promise.resolve()
  const retirement = viewer.retire().then(() => {
    retired = true
  })
  assert.equal(sent.at(-1), "retire")
  socket.onmessage({ data: `purge:${"a".repeat(32)}` })
  assert.equal(closed, 0)
  assert.equal(retired, false)
  release({ width: 16, height: 16, close: () => disposed++ })
  await new Promise((resolve) => setImmediate(resolve))
  assert.equal(disposed, 1)
  assert.equal(sent.at(-1), `purged:${"a".repeat(32)}`)
  assert.equal(retired, false)
  socket.onmessage({ data: "retired" })
  await retirement
  assert.equal(retired, true)
  assert.equal(closed, 1)
  assert.equal(sent.filter((value) => value === "frame").length, 1)
})

test("abrupt disconnect cannot acknowledge intentional retirement", async () => {
  const socket = { send() {}, close() {} }
  const viewer = startPrivateBrowserViewer(
    socket,
    "ticket",
    () => {},
    () => {},
    async () => assert.fail(),
  )
  socket.onopen()
  socket.onmessage({ data: "ready" })
  const retirement = viewer.retire()
  socket.onclose()
  await assert.rejects(retirement)
})

test("purge joins an in-flight decode before acknowledgement and never renders it", async () => {
  const sent = []
  let release
  let draws = 0
  let clears = 0
  let disposed = 0
  const socket = { send: (value) => sent.push(value), close() {} }
  const viewer = startPrivateBrowserViewer(
    socket,
    "private-ticket",
    () => draws++,
    () => clears++,
    () =>
      new Promise((resolve) => {
        release = resolve
      }),
  )
  socket.onopen()
  socket.onmessage({ data: "ready" })
  socket.onmessage({ data: frame() })
  await Promise.resolve()
  socket.onmessage({ data: `purge:${"a".repeat(32)}` })
  assert.deepEqual(sent, ["private-ticket", "frame"])
  release({ width: 16, height: 16, close: () => disposed++ })
  await new Promise((resolve) => setImmediate(resolve))
  assert.equal(draws, 0)
  assert.equal(disposed, 1)
  assert.ok(clears >= 2)
  assert.equal(sent.at(-1), `purged:${"a".repeat(32)}`)
  viewer.stop()
})

test("disconnect prevents late bitmap publication", async () => {
  let release
  let draws = 0
  let disposed = 0
  const socket = { send() {}, close() {} }
  const viewer = startPrivateBrowserViewer(
    socket,
    "ticket",
    () => draws++,
    () => {},
    () =>
      new Promise((resolve) => {
        release = resolve
      }),
  )
  socket.onopen()
  socket.onmessage({ data: "ready" })
  socket.onmessage({ data: frame() })
  await Promise.resolve()
  viewer.stop()
  release({ width: 16, height: 16, close: () => disposed++ })
  await new Promise((resolve) => setImmediate(resolve))
  assert.equal(draws, 0)
  assert.equal(disposed, 1)
})

test("a second ready after purge cannot reactivate frame capture", async () => {
  const sent = []
  let closed = 0
  const socket = { send: (value) => sent.push(value), close: () => closed++ }
  startPrivateBrowserViewer(
    socket,
    "ticket",
    () => assert.fail(),
    () => {},
    async () => assert.fail(),
  )
  socket.onopen()
  socket.onmessage({ data: "ready" })
  socket.onmessage({ data: `purge:${"a".repeat(32)}` })
  await new Promise((resolve) => setImmediate(resolve))
  socket.onmessage({ data: "ready" })
  assert.equal(closed, 1)
  assert.equal(sent.filter((value) => value === "frame").length, 1)
})

test("malformed and oversized dimensions are rejected before decoding", () => {
  for (const data of [new ArrayBuffer(24), frame(0), frame(1921), frame(16, 1081)]) {
    let decodes = 0
    let closed = 0
    const socket = {
      send() {},
      close() {
        closed++
      },
    }
    startPrivateBrowserViewer(
      socket,
      "ticket",
      () => {},
      () => {},
      () => {
        decodes++
        throw Error()
      },
    )
    socket.onopen()
    socket.onmessage({ data: "ready" })
    socket.onmessage({ data })
    assert.equal(decodes, 0)
    assert.equal(closed, 1)
  }
})

test("decoder failure closes the viewer without publishing diagnostics", async () => {
  let closed = 0
  const socket = {
    send() {},
    close() {
      closed++
    },
  }
  startPrivateBrowserViewer(
    socket,
    "ticket",
    () => assert.fail(),
    () => {},
    () => {
      throw Error("private-decoder-canary")
    },
  )
  socket.onopen()
  socket.onmessage({ data: "ready" })
  socket.onmessage({ data: frame() })
  await new Promise((resolve) => setImmediate(resolve))
  assert.equal(closed, 1)
})
