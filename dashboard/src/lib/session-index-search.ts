import type { SessionsSummaryQuery } from "./api"

export const SESSION_INDEX_PAGE_LIMIT = 100
export const DEFAULT_SESSION_ORDER = "updated_at_desc" as const

export const SESSION_INDEX_STATUSES = [
  "pending",
  "running",
  "interrupting",
  "completed",
  "failed",
  "interrupted",
] as const

export const SESSION_INDEX_DEBUG_STATES = [
  "needs_attention",
  "session_failure",
  "tool_issue",
  "interruption",
] as const

export const SESSION_INDEX_ORDERS = [
  "updated_at_desc",
  "updated_at_asc",
  "created_at_desc",
  "created_at_asc",
] as const

export type SessionIndexStatus = (typeof SESSION_INDEX_STATUSES)[number]
export type SessionIndexDebugState = (typeof SESSION_INDEX_DEBUG_STATES)[number]
export type SessionIndexOrder = (typeof SESSION_INDEX_ORDERS)[number]

export type SessionIndexSearch = {
  q?: string
  status?: SessionIndexStatus
  debug_state?: SessionIndexDebugState
  agent_name?: string
  provider_name?: string
  model?: string
  environment_name?: string
  parent_session_id?: string
  causal_budget_id?: string
  order_by?: SessionIndexOrder
  label?: string[]
  label_selector?: string[]
  cursor?: string
}

export type SessionIndexFilterPatch = Omit<Partial<SessionIndexSearch>, "cursor">

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

function allowedValue<const T extends readonly string[]>(
  value: unknown,
  allowed: T,
): T[number] | undefined {
  return typeof value === "string" && allowed.includes(value) ? value : undefined
}

export function validateSessionIndexSearch(search: Record<string, unknown>): SessionIndexSearch {
  const order = allowedValue(search.order_by, SESSION_INDEX_ORDERS)
  return {
    q: optionalSearchValue(search.q),
    status: allowedValue(search.status, SESSION_INDEX_STATUSES),
    debug_state: allowedValue(search.debug_state, SESSION_INDEX_DEBUG_STATES),
    agent_name: optionalSearchValue(search.agent_name),
    provider_name: optionalSearchValue(search.provider_name),
    model: optionalSearchValue(search.model),
    environment_name: optionalSearchValue(search.environment_name),
    parent_session_id: optionalSearchValue(search.parent_session_id),
    causal_budget_id: optionalSearchValue(search.causal_budget_id),
    order_by: order === DEFAULT_SESSION_ORDER ? undefined : order,
    label: searchList(search.label),
    label_selector: searchList(search.label_selector),
    cursor: optionalSearchValue(search.cursor),
  }
}

export function sessionIndexQuery(search: SessionIndexSearch): SessionsSummaryQuery {
  return {
    limit: SESSION_INDEX_PAGE_LIMIT,
    cursor: search.cursor,
    q: search.q,
    status: search.status,
    debug_state: search.debug_state,
    agent_name: search.agent_name,
    provider_name: search.provider_name,
    model: search.model,
    environment_name: search.environment_name,
    parent_session_id: search.parent_session_id,
    causal_budget_id: search.causal_budget_id,
    order_by: search.order_by ?? DEFAULT_SESSION_ORDER,
    label: search.label,
    label_selector: search.label_selector,
  }
}

export function sessionIndexSearchWithFilters(
  search: SessionIndexSearch,
  patch: SessionIndexFilterPatch,
): SessionIndexSearch {
  return validateSessionIndexSearch({ ...search, ...patch, cursor: undefined })
}

export function sessionIndexSearchWithCursor(
  search: SessionIndexSearch,
  cursor: string | null | undefined,
): SessionIndexSearch {
  return validateSessionIndexSearch({ ...search, cursor })
}

export function sessionIndexSearchWithoutFilters(search: SessionIndexSearch): SessionIndexSearch {
  return validateSessionIndexSearch({ order_by: search.order_by })
}

export function multilineSessionFilters(value: string): string[] | undefined {
  return searchList(value.split(/\r?\n/g))
}

export function sessionIndexFilterCount(search: SessionIndexSearch): number {
  const scalarFilters = [
    search.q,
    search.status,
    search.debug_state,
    search.agent_name,
    search.provider_name,
    search.model,
    search.environment_name,
    search.parent_session_id,
    search.causal_budget_id,
  ]
  return (
    scalarFilters.filter((value) => value !== undefined).length +
    (search.label?.length ?? 0) +
    (search.label_selector?.length ?? 0)
  )
}

export function sessionIndexAdvancedFilterKey(search: SessionIndexSearch): string {
  return JSON.stringify([
    search.agent_name ?? "",
    search.provider_name ?? "",
    search.model ?? "",
    search.environment_name ?? "",
    search.parent_session_id ?? "",
    search.causal_budget_id ?? "",
    search.label ?? [],
    search.label_selector ?? [],
  ])
}
