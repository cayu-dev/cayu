import assert from "node:assert/strict"
import test from "node:test"
import {
  isProfileChangedConflict,
  PROFILE_ADOPTION_REASON,
  withProfileAdoption,
} from "../src/next/app/inbox-adoption.ts"

test("only a 409 profile_changed settle offers adoption", () => {
  assert.equal(isProfileChangedConflict({ status: 409, detail: { code: "profile_changed" } }), true)
  assert.equal(isProfileChangedConflict({ status: 409, detail: { code: "conflict" } }), false)
  assert.equal(
    isProfileChangedConflict({ status: 400, detail: { code: "profile_changed" } }),
    false,
  )
  assert.equal(isProfileChangedConflict({ status: 409, detail: "profile_changed" }), false)
  assert.equal(isProfileChangedConflict(null), false)
})

test("the retried decision keeps every field and adds one adoption intent", () => {
  const body = { decision: "approve", reason: "Looks right", arguments: { title: "x" } }
  assert.deepEqual(withProfileAdoption(body, "key-1"), {
    ...body,
    profile_adoption: { idempotency_key: "key-1", reason: PROFILE_ADOPTION_REASON },
  })
  assert.equal("profile_adoption" in body, false)
})
