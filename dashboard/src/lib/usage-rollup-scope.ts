export const USAGE_ROLLUP_PAGE_LIMIT = 1000
export const USAGE_ROLLUP_MAX_PAGES = 10
export const USAGE_ROLLUP_SESSION_LIMIT = USAGE_ROLLUP_PAGE_LIMIT * USAGE_ROLLUP_MAX_PAGES

export type UsageRollupScope = {
  kind: "complete" | "partial" | "indeterminate"
  label: string
  description: string
  metricQualifier: string
}

function formatScopeCount(value: number) {
  return value.toLocaleString()
}

export function describeUsageRollupScope(
  {
    hasMore,
    loadedCount,
    totalCount,
  }: {
    hasMore: boolean
    loadedCount: number
    totalCount: number | null
  },
  formatCount: (value: number) => string = formatScopeCount,
): UsageRollupScope {
  const loaded = formatCount(loadedCount)
  const reliableTotal =
    totalCount !== null && Number.isSafeInteger(totalCount) && totalCount >= loadedCount
      ? totalCount
      : null

  if (hasMore) {
    if (reliableTotal !== null && reliableTotal > loadedCount) {
      const total = formatCount(reliableTotal)
      return {
        kind: "partial",
        label: "Partial loaded scope",
        description: `The latest ${loaded} of ${total} matching sessions by updated time are included. Every figure below excludes older matches; narrow the filters to inspect a complete loaded set.`,
        metricQualifier: `${loaded} of ${total} loaded matches`,
      }
    }
    return {
      kind: "partial",
      label: "Partial loaded scope",
      description: `The latest ${loaded} matching sessions by updated time are included, and more matches are available. Every figure below excludes older matches; narrow the filters to inspect a complete loaded set.`,
      metricQualifier: `${loaded} loaded matches; more available`,
    }
  }

  if (reliableTotal === loadedCount) {
    return {
      kind: "complete",
      label: "Complete loaded scope",
      description: `All ${loaded} matching sessions reported by this paginated request are included in the client-side rollup. The loader is bounded to ${USAGE_ROLLUP_MAX_PAGES} pages of ${formatCount(USAGE_ROLLUP_PAGE_LIMIT)} sessions (${formatCount(USAGE_ROLLUP_SESSION_LIMIT)} maximum).`,
      metricQualifier: `all ${loaded} loaded matches`,
    }
  }

  return {
    kind: "indeterminate",
    label: "Bounded loaded scope",
    description: `${loaded} matching sessions are included, but a reliable matching total was not available. Every figure below covers only this loaded set, which is bounded to ${formatCount(USAGE_ROLLUP_SESSION_LIMIT)} sessions.`,
    metricQualifier: `${loaded} loaded matches; total unavailable`,
  }
}
