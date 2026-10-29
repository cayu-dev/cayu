/** Synthetic series for the prototype's viewer-local requests; never used by production. */
import { compareUsageTimestamps } from "../../lib/usage-rollup.ts"
import type { BucketUnit } from "../routes/usage/usage-window"

export function mockTimeBuckets(
  startAt: string,
  endAt: string,
  unit: BucketUnit,
  timezone: string,
) {
  // The prototype only makes viewer-local requests. Do not claim another zone's calendar.
  if (timezone !== Intl.DateTimeFormat().resolvedOptions().timeZone)
    throw new Error("The prototype supports series in the viewer's time zone only.")
  const start = Date.parse(startAt)
  const end = Date.parse(endAt)
  const local = new Date(start)
  const hour =
    start - local.getMinutes() * 60_000 - local.getSeconds() * 1000 - local.getMilliseconds()
  const buckets = []
  let at = start
  for (let n = 1; at < end; n++) {
    const boundary =
      unit === "hour"
        ? hour + n * 3600_000
        : new Date(
            local.getFullYear(),
            local.getMonth(),
            local.getDate() + n * (unit === "week" ? 7 : 1),
          ).getTime()
    if (boundary <= at) continue
    if (buckets.length === 512) throw new Error("Series cannot contain more than 512 buckets.")
    const until = Math.min(boundary, end)
    buckets.push({ start: at, end: until })
    at = until
  }
  if (start === end) buckets.push({ start, end })
  return buckets.map((bucket, index) => ({
    ...bucket,
    start_at: index === 0 ? startAt : new Date(bucket.start).toISOString(),
    end_at: index === buckets.length - 1 ? endAt : new Date(bucket.end).toISOString(),
  }))
}

/** Numeric milliseconds are only for calendar/display; membership retains every digit. */
export function inMockTimeWindow(timestamp: string, startAt: string, endAt: string): boolean {
  return (
    compareUsageTimestamps(timestamp, startAt) >= 0 && compareUsageTimestamps(timestamp, endAt) < 0
  )
}
