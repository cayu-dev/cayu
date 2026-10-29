/**
 * The Usage page's time window and resolution, in the viewer's local time. Preset
 * ranges end now and start on a whole hour or day, so the headline numbers and the bars
 * cover exactly the same span.
 */

import { resolveUsageWindow } from "../../../lib/usage-rollup.ts"
import { DEFAULT_USAGE_RANGE } from "../../../lib/usage-rollup-search.ts"

export type UsageRangeId = "24h" | "7d" | "30d" | "90d" | "365d" | "custom"
export type BucketUnit = "hour" | "day" | "week"
export type Bucket = { start: number; end: number }
export type UsageWindow = {
  start: number
  end: number
  startAt: string
  endAt: string
  unit: BucketUnit
}

export const DEFAULT_RANGE: UsageRangeId = DEFAULT_USAGE_RANGE
/** The choices the picker offers; 90d and 365d still work when a link carries them. */
export const RANGE_CHOICES: { id: UsageRangeId; label: string }[] = [
  { id: "24h", label: "Last 24 hours" },
  { id: "7d", label: "Last 7 days" },
  { id: "30d", label: "Last 30 days" },
  { id: "custom", label: "Custom" },
]
const PRESET_DAYS: Record<string, number> = { "7d": 7, "30d": 30, "90d": 90, "365d": 365 }
const HOUR = 3600_000
const DAY = 24 * HOUR

export function rangeOf(value: string | undefined): UsageRangeId {
  // `Object.hasOwn`, not `in`: `?range=constructor` mustn't match Object.prototype.
  return value === "24h" || value === "custom" || (value && Object.hasOwn(PRESET_DAYS, value))
    ? (value as UsageRangeId)
    : DEFAULT_RANGE
}

export function rangeLabel(range: UsageRangeId): string {
  if (range === "24h") return "Last 24 hours"
  if (range === "custom") return "Custom range"
  return `Last ${PRESET_DAYS[range]} days`
}

function startOfHour(ms: number) {
  const d = new Date(ms)
  d.setMinutes(0, 0, 0)
  return d.getTime()
}

function startOfDay(ms: number) {
  const d = new Date(ms)
  d.setHours(0, 0, 0, 0)
  return d.getTime()
}

/** Calendar steps, so a day stays a day across daylight-saving changes. */
function step(ms: number, unit: BucketUnit, n = 1) {
  if (unit === "hour") return ms + n * HOUR
  const d = new Date(ms)
  d.setDate(d.getDate() + n * (unit === "week" ? 7 : 1))
  return d.getTime()
}

export function resolveWindow(
  range: UsageRangeId,
  startAt: string | undefined,
  endAt: string | undefined,
  now: number,
): UsageWindow | { error: string } {
  if (!Number.isFinite(now))
    return { error: "The current time is unavailable; reload before requesting usage." }
  if (range === "24h") {
    const start = startOfHour(now) - 23 * HOUR
    return {
      start,
      end: now,
      startAt: new Date(start).toISOString(),
      endAt: new Date(now).toISOString(),
      unit: "hour",
    }
  }
  if (range !== "custom") {
    const days = PRESET_DAYS[range] ?? 7
    const start = step(startOfDay(now), "day", -(days - 1))
    const unit = days > 62 ? "week" : "day"
    return {
      start,
      end: now,
      startAt: new Date(start).toISOString(),
      endAt: new Date(now).toISOString(),
      unit,
    }
  }
  try {
    const exact = resolveUsageWindow(
      { range: "custom", start_at: startAt, end_at: endAt },
      new Date(now),
    )
    const start = Date.parse(exact.startAt)
    const end = Date.parse(exact.endAt)
    const unit: BucketUnit =
      end - start <= 2 * DAY ? "hour" : end - start <= 62 * DAY ? "day" : "week"
    return { start, end, ...exact, unit }
  } catch (error) {
    return { error: error instanceof Error ? error.message : "The usage window is invalid." }
  }
}

/** A short label under a bar. */
export function bucketLabel(bucket: Bucket, unit: BucketUnit): string {
  const d = new Date(bucket.start)
  if (unit === "hour") return d.toLocaleTimeString([], { hour: "numeric" })
  return d.toLocaleDateString([], { month: "short", day: "numeric" })
}

/** The full span of a bar, for its tooltip. */
export function bucketSpan(bucket: Bucket, unit: BucketUnit): string {
  const day = (ms: number) =>
    new Date(ms).toLocaleDateString([], { weekday: "short", month: "short", day: "numeric" })
  const time = (ms: number) =>
    new Date(ms).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })
  if (unit === "hour") return `${day(bucket.start)}, ${time(bucket.start)}–${time(bucket.end)}`
  if (unit === "day") return day(bucket.start)
  return `${day(bucket.start)} – ${day(bucket.end - 1)}`
}

/** A `datetime-local` value in the viewer's time zone. */
export function localInputValue(ms: number): string {
  const d = new Date(ms)
  return new Date(ms - d.getTimezoneOffset() * 60_000).toISOString().slice(0, 23)
}

export function formatWhen(ms: number): string {
  return new Date(ms).toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  })
}
