import type { UsageRollupBody } from "@/lib/api"
import type { SessionStartSeries, UsageTimeSeries } from "@/lib/generated/server-api"
import type { UsageRollupSearch } from "@/lib/usage-rollup-search"
import type { OverviewCountsQuery } from "../../api/types"
import type { BucketUnit } from "./usage-window"

/**
 * The start-time request for the headline's filters. Label, parent-session and
 * causal-budget filters need `session_overview_count_filters`: an older server ignores
 * them, so without it the chart is unavailable rather than silently unfiltered.
 */
export function histogramRequest(
  body: UsageRollupBody,
  search: Pick<UsageRollupSearch, "label" | "label_selector">,
  unit: BucketUnit,
  timezone: string,
  countFilters: boolean,
): OverviewCountsQuery | null {
  const filter = body.session_filter
  const related =
    !!filter?.parent_session_id ||
    !!filter?.causal_budget_id ||
    Object.keys(filter?.labels ?? {}).length > 0 ||
    (filter?.label_selectors?.length ?? 0) > 0
  if (related && !countFilters) return null
  return {
    since: body.start_at,
    until: body.end_at,
    agent_name: filter?.agent_name ?? undefined,
    environment_name: filter?.environment_name ?? undefined,
    provider_name: filter?.provider_name ?? undefined,
    model: filter?.model ?? undefined,
    ...(related
      ? {
          parent_session_id: filter?.parent_session_id ?? undefined,
          causal_budget_id: filter?.causal_budget_id ?? undefined,
          // The same text the headline's filter was parsed from; the server parses it
          // with the rules GET /api/sessions uses, which the headline parser mirrors.
          label: search.label,
          label_selector: search.label_selector,
        }
      : {}),
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
