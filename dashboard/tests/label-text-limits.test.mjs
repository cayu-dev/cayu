import assert from "node:assert/strict"
import { readFileSync } from "node:fs"
import test from "node:test"

import { unicodeScalarLength } from "../src/lib/durable-text.ts"
import { LABEL_LIMITS } from "../src/lib/generated/label-limits.ts"
import { parseSessionLabelsDraft } from "../src/lib/session-editing.ts"
import { usageRollupRequestState } from "../src/lib/usage-rollup.ts"
import { validateUsageRollupSearch } from "../src/lib/usage-rollup-search.ts"

const cases = JSON.parse(
  readFileSync(new URL("./fixtures/label-text-boundaries.json", import.meta.url)),
)

test("generated label limits retain Unicode scalar measurement", () => {
  assert.equal(LABEL_LIMITS.unit, "unicode_scalar_values")
  assert.equal(unicodeScalarLength("a😀e\u0301"), 4)
})

test("label editing and Usage counts follow the declared limits", () => {
  for (const offset of [0, 1]) {
    const accepted = offset === 0
    for (const field of ["key", "value"]) {
      const text = "😀".repeat(LABEL_LIMITS[field] + offset)
      const key = field === "key" ? text : "key"
      const value = field === "value" ? text : "value"
      assert.equal(parseSessionLabelsDraft(JSON.stringify({ [key]: value })).ok, accepted)
      assert.equal(
        usageRollupRequestState(validateUsageRollupSearch({ label: [`${key}=${value}`] })).ok,
        accepted,
      )
    }
    const inputs = [
      { label: Array.from({ length: LABEL_LIMITS.filters + offset }, (_, index) => `k${index}=v`) },
      {
        label_selector: Array.from(
          { length: LABEL_LIMITS.selectors + offset },
          (_, index) => `k${index}`,
        ),
      },
      {
        label_selector: [
          `key in (${Array.from({ length: LABEL_LIMITS.selectorValues + offset }, (_, index) => `v${index}`).join(",")})`,
        ],
      },
    ]
    for (const input of inputs) {
      assert.equal(usageRollupRequestState(validateUsageRollupSearch(input)).ok, accepted)
    }
  }
})

for (const boundary of cases) {
  test(`label editor and filters preserve shared boundary: ${boundary.id}`, () => {
    const text = boundary.text.repeat(boundary.repeat)
    const key = boundary.field === "key" ? text : "key"
    const value = boundary.field === "value" ? text : "value"
    const edited = parseSessionLabelsDraft(JSON.stringify({ [key]: value }))
    assert.equal(edited.ok, boundary.accepted, "editor")
    if (edited.ok) assert.deepEqual(edited.value, { [key]: value })
    const exact = usageRollupRequestState(validateUsageRollupSearch({ label: [`${key}=${value}`] }))
    assert.equal(exact.ok, boundary.accepted, "exact filter")
    const selectors = [
      `${key}=${value}`,
      `${key}==${value}`,
      `${key}!=${value}`,
      `${key} in (${value})`,
      `${key} notin (${value})`,
    ]
    if (boundary.field === "key") selectors.push(key, `!${key}`)
    for (const selector of selectors) {
      const result = usageRollupRequestState(
        validateUsageRollupSearch({ label_selector: [selector] }),
      )
      assert.equal(result.ok, boundary.accepted, selector)
    }
  })
}
