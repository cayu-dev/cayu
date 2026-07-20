import assert from "node:assert/strict"
import test from "node:test"

import {
  describeUsageRollupScope,
  USAGE_ROLLUP_SESSION_LIMIT,
} from "../src/lib/usage-rollup-scope.ts"

const englishCount = new Intl.NumberFormat("en-US")
const formatEnglishCount = (value) => englishCount.format(value)

test("usage scope identifies a complete bounded rollup", () => {
  assert.deepEqual(
    describeUsageRollupScope(
      { hasMore: false, loadedCount: 125, totalCount: 125 },
      formatEnglishCount,
    ),
    {
      kind: "complete",
      label: "Complete loaded scope",
      description:
        "All 125 matching sessions reported by this paginated request are included in the client-side rollup. The loader is bounded to 10 pages of 1,000 sessions (10,000 maximum).",
      metricQualifier: "all 125 loaded matches",
    },
  )
})

test("usage scope exposes known and unknown truncation", () => {
  assert.deepEqual(
    describeUsageRollupScope(
      {
        hasMore: true,
        loadedCount: USAGE_ROLLUP_SESSION_LIMIT,
        totalCount: 12_345,
      },
      formatEnglishCount,
    ),
    {
      kind: "partial",
      label: "Partial loaded scope",
      description:
        "The latest 10,000 of 12,345 matching sessions by updated time are included. Every figure below excludes older matches; narrow the filters to inspect a complete loaded set.",
      metricQualifier: "10,000 of 12,345 loaded matches",
    },
  )
  assert.deepEqual(
    describeUsageRollupScope(
      { hasMore: true, loadedCount: 10_000, totalCount: null },
      formatEnglishCount,
    ),
    {
      kind: "partial",
      label: "Partial loaded scope",
      description:
        "The latest 10,000 matching sessions by updated time are included, and more matches are available. Every figure below excludes older matches; narrow the filters to inspect a complete loaded set.",
      metricQualifier: "10,000 loaded matches; more available",
    },
  )
})

test("usage scope remains conservative when pagination metadata disagrees", () => {
  assert.deepEqual(
    describeUsageRollupScope(
      { hasMore: true, loadedCount: 100, totalCount: 100 },
      formatEnglishCount,
    ),
    {
      kind: "partial",
      label: "Partial loaded scope",
      description:
        "The latest 100 matching sessions by updated time are included, and more matches are available. Every figure below excludes older matches; narrow the filters to inspect a complete loaded set.",
      metricQualifier: "100 loaded matches; more available",
    },
  )
  assert.equal(
    describeUsageRollupScope(
      { hasMore: false, loadedCount: 90, totalCount: 100 },
      formatEnglishCount,
    ).kind,
    "indeterminate",
  )
  assert.deepEqual(
    describeUsageRollupScope(
      { hasMore: false, loadedCount: 100, totalCount: 99 },
      formatEnglishCount,
    ),
    {
      kind: "indeterminate",
      label: "Bounded loaded scope",
      description:
        "100 matching sessions are included, but a reliable matching total was not available. Every figure below covers only this loaded set, which is bounded to 10,000 sessions.",
      metricQualifier: "100 loaded matches; total unavailable",
    },
  )
})
