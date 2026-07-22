export const DEFAULT_USAGE_RANGE = "30d" as const
export const USAGE_RANGE_OPTIONS = ["24h", "7d", "30d", "90d", "365d", "custom"] as const

export type UsageRange = (typeof USAGE_RANGE_OPTIONS)[number]

export type UsageRollupSearch = {
  range?: UsageRange
  start_at?: string
  end_at?: string
  agent_name?: string
  provider_name?: string
  model?: string
  environment_name?: string
  parent_session_id?: string
  causal_budget_id?: string
  label?: string[]
  label_selector?: string[]
}

function optionalSearchValue(value: unknown): string | undefined {
  if (typeof value !== "string") return undefined
  const trimmed = value.trim()
  return trimmed === "" ? undefined : trimmed
}

function searchList(value: unknown): string[] | undefined {
  const values = Array.isArray(value) ? value : typeof value === "string" ? [value] : []
  const normalized = values
    .map(optionalSearchValue)
    .filter((item): item is string => item !== undefined)
  return normalized.length === 0 ? undefined : normalized
}

function allowedRange(value: unknown): UsageRange | undefined {
  return typeof value === "string" && USAGE_RANGE_OPTIONS.includes(value as UsageRange)
    ? (value as UsageRange)
    : undefined
}

export function validateUsageRollupSearch(search: Record<string, unknown>): UsageRollupSearch {
  const range = allowedRange(search.range)
  const custom = range === "custom"
  return {
    range: range === DEFAULT_USAGE_RANGE ? undefined : range,
    start_at: custom ? optionalSearchValue(search.start_at) : undefined,
    end_at: custom ? optionalSearchValue(search.end_at) : undefined,
    agent_name: optionalSearchValue(search.agent_name),
    provider_name: optionalSearchValue(search.provider_name),
    model: optionalSearchValue(search.model),
    environment_name: optionalSearchValue(search.environment_name),
    parent_session_id: optionalSearchValue(search.parent_session_id),
    causal_budget_id: optionalSearchValue(search.causal_budget_id),
    label: searchList(search.label),
    label_selector: searchList(search.label_selector),
  }
}

export function usageRollupRange(search: UsageRollupSearch): UsageRange {
  return search.range ?? DEFAULT_USAGE_RANGE
}
