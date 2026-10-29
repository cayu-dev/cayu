import assert from "node:assert/strict"
import test from "node:test"
import { mockTimeBuckets } from "../src/next/mock/time-series.ts"
import { histogramRequest, joinedSeriesBuckets } from "../src/next/routes/usage/usage-series.ts"

const body = {
  start_at: "2026-07-01T00:00:59.999123Z",
  end_at: "2026-07-02T08:30:00.123456+06:00",
  session_filter: {
    agent_name: "writer",
    environment_name: "Review / λ",
    provider_name: "p",
    model: "m",
  },
}

test("one histogram request preserves exact bounds, all supported filters, resolution and zone", () => {
  assert.deepEqual(histogramRequest(body, "hour", "America/New_York"), {
    since: body.start_at,
    until: body.end_at,
    ...body.session_filter,
    bucket_unit: "hour",
    timezone: "America/New_York",
  })
})

test("unsupported scopes never silently become an unfiltered chart", () => {
  for (const extra of [
    { labels: { team: "qa" } },
    { label_selectors: [{ key: "team", operator: "exists" }] },
    { parent_session_id: "parent" },
    { causal_budget_id: "budget" },
  ])
    assert.equal(
      histogramRequest(
        { ...body, session_filter: { ...body.session_filter, ...extra } },
        "day",
        "UTC",
      ),
      null,
    )
  assert.ok(
    histogramRequest(
      { ...body, session_filter: { labels: {}, label_selectors: [], parent_session_id: null } },
      "day",
      "UTC",
    ),
  )
})

test("native boundaries, start counts and activity tokens retain distinct populations", () => {
  const counts = {
    unit: "hour",
    timezone: "America/New_York",
    buckets: [
      {
        start_at: "2026-11-01T05:00:00+00:00",
        end_at: "2026-11-01T06:00:00+00:00",
        total_count: "0",
        failed: "0",
      },
      {
        start_at: "2026-11-01T06:00:00+00:00",
        end_at: "2026-11-01T07:00:00+00:00",
        total_count: "2",
        failed: "1",
      },
    ],
  }
  const activity = {
    ...counts,
    buckets: counts.buckets.map((b) => ({ ...b, totals: { usage: { total_tokens: "900" } } })),
  }
  const result = joinedSeriesBuckets(counts, activity)
  assert.deepEqual(
    result.map((b) => [b.start, b.end, b.total, b.failed, b.activityTokens]),
    [
      [Date.parse(counts.buckets[0].start_at), Date.parse(counts.buckets[0].end_at), 0, 0, "900"],
      [Date.parse(counts.buckets[1].start_at), Date.parse(counts.buckets[1].end_at), 2, 1, "900"],
    ],
  )
  for (const mismatch of [
    null,
    { ...activity, unit: "day" },
    { ...activity, timezone: "UTC" },
    {
      ...activity,
      buckets: [
        { ...activity.buckets[0], end_at: "2026-11-01T06:00:00.000001+00:00" },
        activity.buckets[1],
      ],
    },
  ])
    assert.ok(joinedSeriesBuckets(counts, mismatch).every((b) => b.activityTokens === null))
})

test("prototype viewer-local buckets preserve repeated hours and calendar days", () => {
  const previous = process.env.TZ
  process.env.TZ = "America/New_York"
  try {
    const hours = mockTimeBuckets(
      "2026-11-01T05:30:00Z",
      "2026-11-01T07:30:00Z",
      "hour",
      "America/New_York",
    )
    assert.deepEqual(
      hours.map((b) => b.end - b.start),
      [1800000, 3600000, 1800000],
    )
    const folded = mockTimeBuckets(
      "2026-11-01T06:30:00Z",
      "2026-11-01T07:30:00Z",
      "hour",
      "America/New_York",
    )
    assert.deepEqual(
      folded.map((b) => b.end - b.start),
      [1800000, 1800000],
    )
    for (const [start, end, hours] of [
      ["2026-03-07T05:00:00Z", "2026-03-10T04:00:00Z", [24, 23, 24]],
      ["2026-10-31T04:00:00Z", "2026-11-03T05:00:00Z", [24, 25, 24]],
    ])
      assert.deepEqual(
        mockTimeBuckets(start, end, "day", "America/New_York").map(
          (b) => (b.end - b.start) / 3600000,
        ),
        hours,
      )
  } finally {
    if (previous === undefined) delete process.env.TZ
    else process.env.TZ = previous
  }
})
