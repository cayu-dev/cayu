import assert from "node:assert/strict"
import test from "node:test"
import { TaskOperation } from "../src/next/app/task-operation.ts"

test("lost acknowledgement retains exact operation and copied input until retry", async () => {
  const calls = []
  const operation = new TaskOperation(async (body, id) => {
    calls.push({ body: structuredClone(body), id })
    if (calls.length === 1) throw new Error("connection lost")
    return { task_id: "created" }
  })
  const input = { type: "report", input: { amount: 1 } }
  await operation.submit(input)
  input.input.amount = 9
  assert.equal(operation.state.phase, "unknown")
  operation.clear()
  assert.equal(operation.state.phase, "unknown")
  await operation.submit({ type: "wrong", input: {} })
  assert.deepEqual(calls[0], calls[1])
  assert.equal(operation.state.taskId, "created")
})

test("pending dispatch fences duplicate submission and clear until its exact result", async () => {
  let finish
  let calls = 0
  const owner = new TaskOperation(async () => {
    calls++
    return await new Promise((resolve) => {
      finish = resolve
    })
  })
  const pending = owner.submit({ type: "report", input: {} })
  owner.clear()
  await owner.submit({ type: "different", input: {} })
  assert.equal(owner.state.phase, "pending")
  assert.equal(calls, 1)
  finish({ task_id: "durable" })
  await pending
  assert.equal(owner.state.phase, "succeeded")
  assert.equal(owner.state.taskId, "durable")
})

test("a recovery rejection cannot establish that the original unknown operation failed", async () => {
  const calls = []
  const owner = new TaskOperation(async (body, id) => {
    calls.push({ body, id })
    if (calls.length === 1) throw new Error("lost committed response")
    throw { status: 409 }
  })
  await owner.submit({ type: "report", input: { value: 1 } })
  await owner.submit()
  assert.equal(owner.state.phase, "unknown")
  owner.clear()
  await owner.submit({ type: "other", input: {} })
  assert.deepEqual(calls[0], calls[2])
})
