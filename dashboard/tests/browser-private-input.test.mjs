import assert from "node:assert/strict"
import test from "node:test"
import {
  encodePrivateBrowserText,
  startPrivateBrowserInput,
} from "../src/lib/browser-private-input.ts"

const expected = { revision: 5, control_epoch: 2, settled_input_sequence: 1 }
function socket() {
  return {
    sent: [],
    closed: 0,
    send(value) {
      this.sent.push(value instanceof Uint8Array ? value.slice() : value)
    },
    close() {
      this.closed++
    },
  }
}

test("text sends once only after ready and is erased before settlement", async () => {
  const transport = socket()
  const payload = encodePrivateBrowserText("private-text-canary")
  const input = startPrivateBrowserInput(transport, "ticket", payload, expected)
  transport.onopen()
  assert.deepEqual(transport.sent, ["ticket"])
  transport.onmessage({ data: "ready" })
  assert.equal(new TextDecoder().decode(transport.sent[1]), "private-text-canary")
  assert.ok(payload.every((byte) => byte === 0))
  transport.onmessage({ data: JSON.stringify({ state: "settled", ...expected }) })
  await input.settled
  assert.equal(transport.closed, 1)
  assert.equal(transport.onmessage, null)
})

test("duplicate ready cannot replay input", async () => {
  const transport = socket()
  const input = startPrivateBrowserInput(
    transport,
    "ticket",
    encodePrivateBrowserText("canary"),
    expected,
  )
  transport.onopen()
  transport.onmessage({ data: "ready" })
  transport.onmessage({ data: "ready" })
  await assert.rejects(input.settled, /do not resend/)
  assert.equal(transport.sent.filter((value) => value instanceof Uint8Array).length, 1)
})

test("stop before admission erases unsent payload without claiming settlement", async () => {
  const transport = socket()
  const payload = encodePrivateBrowserText("canary")
  const input = startPrivateBrowserInput(transport, "ticket", payload, expected)
  input.stop()
  await assert.rejects(input.settled, /did not settle/)
  assert.equal(transport.sent.length, 0)
  assert.ok(payload.every((byte) => byte === 0))
})

test("wrong-generation and malformed acknowledgements never settle input", async () => {
  for (const data of [
    "private-response-canary",
    JSON.stringify({ state: "settled", ...expected, revision: 6 }),
    JSON.stringify({ state: "settled", ...expected, control_epoch: 3 }),
    JSON.stringify({ state: "settled", ...expected, settled_input_sequence: true }),
    JSON.stringify({ state: "settled", ...expected, reflected: "private-response-canary" }),
  ]) {
    const transport = socket()
    const input = startPrivateBrowserInput(
      transport,
      "ticket",
      encodePrivateBrowserText("canary"),
      expected,
    )
    transport.onopen()
    transport.onmessage({ data: "ready" })
    transport.onmessage({ data })
    await assert.rejects(input.settled, (error) => {
      assert.match(error.message, /did not settle/)
      assert.equal(String(error).includes("canary"), false)
      return true
    })
  }
})

test("deadline after send closes waiting and never retransmits", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] })
  const transport = socket()
  const input = startPrivateBrowserInput(
    transport,
    "ticket",
    encodePrivateBrowserText("canary"),
    expected,
  )
  transport.onopen()
  transport.onmessage({ data: "ready" })
  t.mock.timers.tick(30_000)
  await assert.rejects(input.settled, /do not resend/)
  assert.equal(transport.sent.length, 2)
  assert.equal(transport.closed, 1)
})

test("invalid private text is rejected without reflecting it", () => {
  for (const value of ["", "canary\0", "canary\ud800", "a".repeat(4097)]) {
    assert.throws(
      () => encodePrivateBrowserText(value),
      (error) => {
        assert.equal(String(error).includes("canary"), false)
        return true
      },
    )
  }
  assert.equal(encodePrivateBrowserText("😀".repeat(4096)).length, 16384)
})

test("missing or invalid expected receipt cannot authenticate an acknowledgement", async () => {
  for (const authority of [
    {},
    { ...expected, revision: true },
    { ...expected, control_epoch: 0 },
  ]) {
    const transport = socket()
    const payload = encodePrivateBrowserText("private-canary")
    const input = startPrivateBrowserInput(transport, "ticket", payload, authority)
    await assert.rejects(input.settled, /did not settle/)
    assert.equal(transport.sent.length, 0)
    assert.ok(payload.every((byte) => byte === 0))
  }
})
