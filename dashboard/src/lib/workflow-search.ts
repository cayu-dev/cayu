import { stringifyDashboardSearch } from "./search-params.ts"
import { resolveUsageWindow } from "./usage-rollup.ts"
import { DEFAULT_USAGE_RANGE, USAGE_RANGE_OPTIONS, type UsageRange } from "./usage-rollup-search.ts"

// The server accepts 50 expanded session parents in total. The Workflow view
// always includes the focus session, so shareable state may add at most 49 more.
export const WORKFLOW_SESSION_EXPANSION_LIMIT = 49
export const WORKFLOW_TASK_EXPANSION_LIMIT = 50
export const WORKFLOW_IDENTIFIER_MAX_BYTES = 1024
export const WORKFLOW_FILTER_MAX_BYTES = 1024
export const WORKFLOW_TIMESTAMP_MAX_BYTES = 64
// Keep the complete request line below common 8 KiB proxy limits after percent
// encoding. A maximum-size focus-session path can itself consume roughly 3 KiB.
export const WORKFLOW_URL_MAX_BYTES = 4 * 1024

export const WORKFLOW_STATUS_FILTERS = [
  "pending",
  "claimed",
  "running",
  "interrupting",
  "paused",
  "blocked",
  "needs_attention",
  "completed",
  "failed",
  "interrupted",
  "cancelled",
] as const

export const WORKFLOW_NODE_TYPE_FILTERS = ["session", "task"] as const

export type WorkflowStatusFilter = (typeof WORKFLOW_STATUS_FILTERS)[number]
export type WorkflowNodeTypeFilter = (typeof WORKFLOW_NODE_TYPE_FILTERS)[number]

export type WorkflowSearch = {
  status?: WorkflowStatusFilter[]
  agent_name?: string
  environment_name?: string
  node_type?: WorkflowNodeTypeFilter[]
  expanded_session_id?: string[]
  expanded_task_id?: string[]
  focus_collapsed?: true
  range?: UsageRange
  start_at?: string
  end_at?: string
  invalid?: true
}

type SearchListResult = {
  values: string[] | undefined
  invalid: boolean
}

function optionalSearchValue(value: unknown): { value: string | undefined; invalid: boolean } {
  if (value === undefined) return { value: undefined, invalid: false }
  if (typeof value !== "string") return { value: undefined, invalid: true }
  const trimmed = value.trim()
  if (trimmed === "") return { value: undefined, invalid: false }
  return { value: trimmed, invalid: trimmed !== value }
}

function boundedOptionalSearchValue(
  value: unknown,
  maxBytes: number,
): { value: string | undefined; invalid: boolean } {
  if (typeof value === "string" && value.length > maxBytes) {
    return { value: undefined, invalid: true }
  }
  const parsed = optionalSearchValue(value)
  if (parsed.value === undefined || parsed.invalid) return parsed
  if (
    containsUnsafeSearchText(parsed.value) ||
    new TextEncoder().encode(parsed.value).byteLength > maxBytes
  ) {
    return { value: undefined, invalid: true }
  }
  return parsed
}

function searchList(value: unknown, limit = 100, maxCodeUnits?: number): SearchListResult {
  if (value === undefined) return { values: undefined, invalid: false }
  const items = Array.isArray(value) ? value : [value]
  const normalized: string[] = []
  let invalid = items.length > limit
  for (const item of items.slice(0, limit)) {
    if (typeof item === "string" && maxCodeUnits !== undefined && item.length > maxCodeUnits) {
      invalid = true
      continue
    }
    const parsed = optionalSearchValue(item)
    invalid ||= parsed.invalid
    if (parsed.value !== undefined) normalized.push(parsed.value)
  }
  return {
    values: normalized.length === 0 ? undefined : normalized,
    invalid,
  }
}

function canonicalAllowedList<T extends string>(
  value: unknown,
  allowed: readonly T[],
): { values: T[] | undefined; invalid: boolean } {
  const parsed = searchList(value, allowed.length)
  if (parsed.values === undefined) return parsed as { values: undefined; invalid: boolean }
  const allowedValues = new Set<T>(allowed)
  const values: T[] = []
  let invalid = parsed.invalid
  for (const item of parsed.values) {
    if (!allowedValues.has(item as T)) {
      invalid = true
      continue
    }
    if (!values.includes(item as T)) values.push(item as T)
  }
  values.sort((left, right) => allowed.indexOf(left) - allowed.indexOf(right))
  return { values: values.length === 0 ? undefined : values, invalid }
}

function containsUnsafeSearchText(value: string): boolean {
  for (let index = 0; index < value.length; index += 1) {
    const codeUnit = value.charCodeAt(index)
    if (codeUnit >= 0xd800 && codeUnit <= 0xdbff) {
      const next = value.charCodeAt(index + 1)
      if (!(next >= 0xdc00 && next <= 0xdfff)) return true
      index += 1
      continue
    }
    if (codeUnit >= 0xdc00 && codeUnit <= 0xdfff) return true
    const codePoint = codeUnit
    if (codePoint <= 0x1f || (codePoint >= 0x7f && codePoint <= 0x9f)) return true
  }
  return false
}

function canonicalIdentifierList(value: unknown, limit: number): SearchListResult {
  const parsed = searchList(value, limit, WORKFLOW_IDENTIFIER_MAX_BYTES)
  if (parsed.values === undefined) return parsed
  const encoder = new TextEncoder()
  const unique = new Set<string>()
  let invalid = parsed.invalid
  for (const item of parsed.values) {
    if (
      containsUnsafeSearchText(item) ||
      encoder.encode(item).byteLength > WORKFLOW_IDENTIFIER_MAX_BYTES
    ) {
      invalid = true
      continue
    }
    unique.add(item)
  }
  if (unique.size > limit) invalid = true
  const values = [...unique]
    .sort((left, right) => (left < right ? -1 : left > right ? 1 : 0))
    .slice(0, limit)
  return { values: values.length === 0 ? undefined : values, invalid }
}

function usageRange(value: unknown): { value: UsageRange | undefined; invalid: boolean } {
  if (value === undefined || value === DEFAULT_USAGE_RANGE) {
    return { value: undefined, invalid: false }
  }
  if (typeof value !== "string" || !USAGE_RANGE_OPTIONS.includes(value as UsageRange)) {
    return { value: undefined, invalid: true }
  }
  return { value: value as UsageRange, invalid: false }
}

function trueSearchFlag(value: unknown): { value: true | undefined; invalid: boolean } {
  if (value === undefined || value === false || value === "false" || value === "0") {
    return { value: undefined, invalid: false }
  }
  if (value === true || value === "true" || value === "1") {
    return { value: true, invalid: false }
  }
  return { value: undefined, invalid: true }
}

export function validateWorkflowSearch(search: Record<string, unknown>): WorkflowSearch {
  const statuses = canonicalAllowedList(search.status, WORKFLOW_STATUS_FILTERS)
  const nodeTypes = canonicalAllowedList(search.node_type, WORKFLOW_NODE_TYPE_FILTERS)
  const agentName = boundedOptionalSearchValue(search.agent_name, WORKFLOW_FILTER_MAX_BYTES)
  const environmentName = boundedOptionalSearchValue(
    search.environment_name,
    WORKFLOW_FILTER_MAX_BYTES,
  )
  const expandedSessions = canonicalIdentifierList(
    search.expanded_session_id,
    WORKFLOW_SESSION_EXPANSION_LIMIT,
  )
  const expandedTasks = canonicalIdentifierList(
    search.expanded_task_id,
    WORKFLOW_TASK_EXPANSION_LIMIT,
  )
  const range = usageRange(search.range)
  const focusCollapsed = trueSearchFlag(search.focus_collapsed)
  const inheritedInvalid = search.invalid !== undefined
  const custom = range.value === "custom"
  const startAt = custom
    ? boundedOptionalSearchValue(search.start_at, WORKFLOW_TIMESTAMP_MAX_BYTES)
    : { value: undefined, invalid: false }
  const endAt = custom
    ? boundedOptionalSearchValue(search.end_at, WORKFLOW_TIMESTAMP_MAX_BYTES)
    : { value: undefined, invalid: false }
  let customWindowInvalid = false
  if (custom && !startAt.invalid && !endAt.invalid) {
    try {
      resolveUsageWindow({ range: "custom", start_at: startAt.value, end_at: endAt.value })
    } catch {
      customWindowInvalid = true
    }
  }
  const invalid = [
    statuses.invalid,
    nodeTypes.invalid,
    agentName.invalid,
    environmentName.invalid,
    expandedSessions.invalid,
    expandedTasks.invalid,
    range.invalid,
    focusCollapsed.invalid,
    inheritedInvalid,
    startAt.invalid,
    endAt.invalid,
    customWindowInvalid,
  ].some(Boolean)

  const canonical: WorkflowSearch = {
    status: statuses.values,
    agent_name: agentName.value,
    environment_name: environmentName.value,
    node_type: nodeTypes.values,
    expanded_session_id: expandedSessions.values,
    expanded_task_id: expandedTasks.values,
    focus_collapsed: focusCollapsed.value,
    range: range.value,
    start_at: customWindowInvalid ? undefined : startAt.value,
    end_at: customWindowInvalid ? undefined : endAt.value,
    invalid: invalid ? true : undefined,
  }
  const encodedBytes = new TextEncoder().encode(
    stringifyDashboardSearch(canonical as Record<string, unknown>),
  ).byteLength
  return encodedBytes > WORKFLOW_URL_MAX_BYTES ? { invalid: true } : canonical
}

export function workflowSearchForUrl(search: WorkflowSearch): WorkflowSearch {
  if (search.invalid) {
    throw new Error("Invalid Workflow search state cannot be serialized into a shareable URL.")
  }
  const canonical = validateWorkflowSearch(search as Record<string, unknown>)
  if (canonical.invalid) {
    throw new Error("Invalid Workflow search state cannot be serialized into a shareable URL.")
  }
  const { invalid: _invalid, ...safe } = canonical
  return safe
}
