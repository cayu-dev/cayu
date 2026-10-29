import type { UsageRollupBody } from "@/lib/api"
import type { SessionStartSeries, UsageTimeSeries } from "@/lib/generated/server-api"
import type { OverviewCountsQuery } from "../../api/types"
import type { BucketUnit } from "./usage-window"

/** The start-time API supports these four current-session attributes only. */
export function histogramRequest(
  body: UsageRollupBody,
  unit: BucketUnit,
  timezone: string,
): OverviewCountsQuery | null {
  const filter = body.session_filter
  if (
    filter?.parent_session_id ||
    filter?.causal_budget_id ||
    Object.keys(filter?.labels ?? {}).length ||
    filter?.label_selectors?.length
  )
    return null
  return {
    since: body.start_at,
    until: body.end_at,
    agent_name: filter?.agent_name ?? undefined,
    environment_name: filter?.environment_name ?? undefined,
    provider_name: filter?.provider_name ?? undefined,
    model: filter?.model ?? undefined,
    bucket_unit: unit,
    timezone,
  }
}

/** Join only identical server spans; activity belongs to events, not the start cohort. */
export function joinedSeriesBuckets(starts: SessionStartSeries, activity?: UsageTimeSeries | null) {
  const aligned =
    activity?.unit === starts.unit &&
    activity.timezone === starts.timezone &&
    activity.buckets.length === starts.buckets.length &&
    activity.buckets.every(
      (bucket, index) =>
        bucket.start_at === starts.buckets[index]?.start_at &&
        bucket.end_at === starts.buckets[index]?.end_at,
    )
  return starts.buckets.map((bucket, index) => ({
    start: Date.parse(bucket.start_at),
    end: Date.parse(bucket.end_at),
    total: Number(bucket.total_count),
    failed: Number(bucket.failed),
    activityTokens: aligned ? (activity?.buckets[index]?.totals.usage.total_tokens ?? null) : null,
  }))
}
