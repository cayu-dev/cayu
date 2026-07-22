import type {
  LabelSelectorRequirement,
  PriceBook,
  SessionAggregateFilter,
  UsageRollupRequest,
} from "./generated/server-api"
import {
  type UsageRange,
  type UsageRollupSearch,
  usageRollupRange,
  validateUsageRollupSearch,
} from "./usage-rollup-search.ts"

export type UsageRollupRequestState =
  | { ok: true; request: UsageRollupRequest }
  | { ok: false; error: string }

const HOUR_MS = 60 * 60 * 1000
const DAY_MS = 24 * HOUR_MS
const MAX_WINDOW_SECONDS = 366n * 24n * 60n * 60n
const MAX_LABEL_FILTERS = 50
const MAX_LABEL_SELECTORS = 25
const MAX_LABEL_SELECTOR_VALUES = 100
const MAX_LABEL_KEY_LENGTH = 128
const MAX_LABEL_VALUE_LENGTH = 512

const RANGE_DURATION_MS: Record<Exclude<UsageRange, "custom">, number> = {
  "24h": DAY_MS,
  "7d": 7 * DAY_MS,
  "30d": 30 * DAY_MS,
  "90d": 90 * DAY_MS,
  "365d": 365 * DAY_MS,
}

const UTC_DATETIME_INPUT_PATTERN =
  /^(\d{4})-(\d{2})-(\d{2})[tT](\d{2}):(\d{2})(?::(\d{2})(?:\.(\d{1,6}))?)?$/
const AWARE_TIMESTAMP_PATTERN =
  /^(\d{4})-(\d{2})-(\d{2})[tT](\d{2}):(\d{2}):(\d{2})(?:\.(\d{1,6}))?([zZ]|([+-])(\d{2}):(\d{2}))$/

type ParsedAwareTimestamp = {
  epochSecond: bigint
  fraction: string
}

function searchList(value: unknown): string[] | undefined {
  const values = Array.isArray(value) ? value : typeof value === "string" ? [value] : []
  const normalized = values
    .map((item) => (typeof item === "string" ? item.trim() : ""))
    .filter((item) => item !== "")
  return normalized.length === 0 ? undefined : normalized
}

export function usageRollupSearchWithoutFilters(search: UsageRollupSearch): UsageRollupSearch {
  return validateUsageRollupSearch({
    range: search.range,
    start_at: search.start_at,
    end_at: search.end_at,
  })
}

export function multilineUsageFilters(value: string): string[] | undefined {
  return searchList(value.split(/\r?\n/g))
}

export function usageTimestampInputValue(value: string | undefined): string {
  if (!value) return ""
  try {
    const parsed = parseAwareTimestamp(value, "Timestamp")
    const milliseconds = Number(parsed.epochSecond) * 1000 + fractionalMilliseconds(parsed.fraction)
    return new Date(milliseconds).toISOString().slice(0, 23)
  } catch {
    return ""
  }
}

export function usageTimestampFromUtcInput(
  value: string,
  label: string,
  {
    originalValue,
    originalInputValue,
  }: {
    originalValue?: string
    originalInputValue?: string
  } = {},
): string {
  if (originalValue !== undefined && value === originalInputValue) {
    parseAwareTimestamp(originalValue, label)
    return originalValue
  }
  if (!value) throw new Error(`${label} is required.`)
  const match = UTC_DATETIME_INPUT_PATTERN.exec(value)
  if (!match) {
    throw new Error(`${label} must be a valid UTC date and time with at most 6 fractional digits.`)
  }
  const seconds = match[6] === undefined ? ":00" : `:${match[6]}`
  const fraction = match[7] === undefined ? ".000" : `.${match[7]}`
  const timestamp = `${match[1]}-${match[2]}-${match[3]}T${match[4]}:${match[5]}${seconds}${fraction}Z`
  parseAwareTimestamp(timestamp, label)
  return timestamp
}

export function usageRollupFilterCount(search: UsageRollupSearch): number {
  return (
    [
      search.agent_name,
      search.provider_name,
      search.model,
      search.environment_name,
      search.parent_session_id,
      search.causal_budget_id,
    ].filter((value) => value !== undefined).length +
    (search.label?.length ?? 0) +
    (search.label_selector?.length ?? 0)
  )
}

export function usageRollupRequestState(
  search: UsageRollupSearch,
  {
    now = new Date(),
    pricing = null,
    groupLimit = 20,
    pricingInputLimit = 1000,
  }: {
    now?: Date
    pricing?: PriceBook | null
    groupLimit?: number
    pricingInputLimit?: number
  } = {},
): UsageRollupRequestState {
  try {
    return {
      ok: true,
      request: buildUsageRollupRequest(search, {
        now,
        pricing,
        groupLimit,
        pricingInputLimit,
      }),
    }
  } catch (error) {
    return {
      ok: false,
      error: error instanceof Error ? error.message : "The usage query is invalid.",
    }
  }
}

export function buildUsageRollupRequest(
  search: UsageRollupSearch,
  {
    now = new Date(),
    pricing = null,
    groupLimit = 20,
    pricingInputLimit = 1000,
  }: {
    now?: Date
    pricing?: PriceBook | null
    groupLimit?: number
    pricingInputLimit?: number
  } = {},
): UsageRollupRequest {
  const { startAt, endAt } = resolveUsageWindow(search, now)
  const sessionFilter = buildSessionAggregateFilter(search)
  return {
    start_at: startAt,
    end_at: endAt,
    session_filter: sessionFilter,
    group_limit: groupLimit,
    pricing_input_limit: pricingInputLimit,
    pricing,
  }
}

export function resolveUsageWindow(
  search: UsageRollupSearch,
  now: Date = new Date(),
): { startAt: string; endAt: string } {
  if (!Number.isFinite(now.getTime())) {
    throw new Error("The current time is unavailable; reload before requesting usage.")
  }
  const range = usageRollupRange(search)
  if (range !== "custom") {
    const end = now.getTime()
    return {
      startAt: new Date(end - RANGE_DURATION_MS[range]).toISOString(),
      endAt: new Date(end).toISOString(),
    }
  }

  const startAt = search.start_at
  const endAt = search.end_at
  const start = parseAwareTimestamp(startAt, "Custom start")
  const end = parseAwareTimestamp(endAt, "Custom end")
  if (compareParsedTimestamps(start, end) >= 0) {
    throw new Error("Custom usage start must be before the end.")
  }
  if (windowExceedsMaximum(start, end)) {
    throw new Error("Custom usage windows cannot exceed 366 days.")
  }
  return {
    startAt: startAt!,
    endAt: endAt!,
  }
}

function parseAwareTimestamp(value: string | undefined, label: string): ParsedAwareTimestamp {
  if (!value) throw new Error(`${label} is required.`)
  const parts = AWARE_TIMESTAMP_PATTERN.exec(value)
  if (!parts) {
    throw new Error(
      `${label} must be an RFC 3339 timestamp with a timezone and at most 6 fractional digits.`,
    )
  }
  const [
    ,
    yearText,
    monthText,
    dayText,
    hourText,
    minuteText,
    secondText,
    fraction = "",
    ,
    offsetSign,
    offsetHourText,
    offsetMinuteText,
  ] = parts
  const year = Number(yearText)
  const month = Number(monthText)
  const day = Number(dayText)
  const hour = Number(hourText)
  const minute = Number(minuteText)
  const second = Number(secondText)
  const calendar = new Date(0)
  calendar.setUTCFullYear(year, month - 1, day)
  if (
    year < 1 ||
    calendar.getUTCFullYear() !== year ||
    calendar.getUTCMonth() !== month - 1 ||
    calendar.getUTCDate() !== day ||
    hour > 23 ||
    minute > 59 ||
    second > 59 ||
    (offsetHourText !== undefined && (Number(offsetHourText) > 23 || Number(offsetMinuteText) > 59))
  ) {
    throw new Error(`${label} is not a valid timestamp.`)
  }
  calendar.setUTCHours(hour, minute, second, 0)
  const localEpochMilliseconds = calendar.getTime()
  if (!Number.isSafeInteger(localEpochMilliseconds)) {
    throw new Error(`${label} is not a valid timestamp.`)
  }
  const offsetSeconds =
    offsetHourText === undefined
      ? 0
      : (Number(offsetHourText) * 60 + Number(offsetMinuteText)) *
        60 *
        (offsetSign === "-" ? -1 : 1)
  return {
    epochSecond: BigInt(localEpochMilliseconds / 1000 - offsetSeconds),
    fraction,
  }
}

function fractionalMilliseconds(fraction: string): number {
  return fraction === "" ? 0 : Number(`${fraction}000`.slice(0, 3))
}

function compareParsedTimestamps(left: ParsedAwareTimestamp, right: ParsedAwareTimestamp): number {
  if (left.epochSecond < right.epochSecond) return -1
  if (left.epochSecond > right.epochSecond) return 1
  return compareFractions(left.fraction, right.fraction)
}

function compareFractions(left: string, right: string): number {
  const width = Math.max(left.length, right.length)
  const normalizedLeft = left.padEnd(width, "0")
  const normalizedRight = right.padEnd(width, "0")
  return normalizedLeft < normalizedRight ? -1 : normalizedLeft > normalizedRight ? 1 : 0
}

function windowExceedsMaximum(start: ParsedAwareTimestamp, end: ParsedAwareTimestamp): boolean {
  const wholeSeconds = end.epochSecond - start.epochSecond
  if (wholeSeconds > MAX_WINDOW_SECONDS) return true
  if (wholeSeconds < MAX_WINDOW_SECONDS) return false
  return compareFractions(end.fraction, start.fraction) > 0
}

export function buildSessionAggregateFilter(search: UsageRollupSearch): SessionAggregateFilter {
  return {
    agent_name: search.agent_name,
    provider_name: search.provider_name,
    model: search.model,
    environment_name: search.environment_name,
    parent_session_id: search.parent_session_id,
    causal_budget_id: search.causal_budget_id,
    labels: parseExactLabels(search.label),
    label_selectors: parseLabelSelectors(search.label_selector),
  }
}

export function usageRollupSessionSearch(search: UsageRollupSearch) {
  return {
    agent_name: search.agent_name,
    provider_name: search.provider_name,
    model: search.model,
    environment_name: search.environment_name,
    parent_session_id: search.parent_session_id,
    causal_budget_id: search.causal_budget_id,
    label: search.label,
    label_selector: search.label_selector,
  }
}

function parseExactLabels(values: string[] | undefined): Record<string, string> {
  const labels = new Map<string, string>()
  for (const expression of values ?? []) {
    const separator = expression.indexOf("=")
    if (separator <= 0) {
      throw new Error(`Label filter ${JSON.stringify(expression)} must use key=value.`)
    }
    const key = cleanBoundedText(expression.slice(0, separator), "Label key", MAX_LABEL_KEY_LENGTH)
    const value = cleanBoundedText(
      expression.slice(separator + 1),
      `Label ${key}`,
      MAX_LABEL_VALUE_LENGTH,
    )
    if (labels.has(key)) {
      throw new Error(`Duplicate label filter: ${key}.`)
    }
    labels.set(key, value)
  }
  if (labels.size > MAX_LABEL_FILTERS) {
    throw new Error(`Usage filters cannot contain more than ${MAX_LABEL_FILTERS} exact labels.`)
  }
  return Object.fromEntries(labels)
}

function parseLabelSelectors(values: string[] | undefined): LabelSelectorRequirement[] {
  const selectors = (values ?? []).flatMap(splitLabelSelector).map(parseLabelSelector)
  if (selectors.length > MAX_LABEL_SELECTORS) {
    throw new Error(`Usage filters cannot contain more than ${MAX_LABEL_SELECTORS} selectors.`)
  }
  const valueCount = selectors.reduce(
    (count, selector) => count + (selector.values?.length ?? 0),
    0,
  )
  if (valueCount > MAX_LABEL_SELECTOR_VALUES) {
    throw new Error(
      `Usage filters cannot contain more than ${MAX_LABEL_SELECTOR_VALUES} selector values.`,
    )
  }
  return selectors
}

function splitLabelSelector(value: string): string[] {
  const parts: string[] = []
  let start = 0
  let depth = 0
  for (let index = 0; index < value.length; index += 1) {
    const char = value[index]
    if (char === "(") depth += 1
    if (char === ")") {
      depth -= 1
      if (depth < 0) throw new Error(`Invalid label selector: ${value}.`)
    }
    if (char === "," && depth === 0) {
      const part = value.slice(start, index).trim()
      if (!part) throw new Error(`Invalid label selector: ${value}.`)
      parts.push(part)
      start = index + 1
    }
  }
  if (depth !== 0) throw new Error(`Invalid label selector: ${value}.`)
  const part = value.slice(start).trim()
  if (!part) throw new Error(`Invalid label selector: ${value}.`)
  parts.push(part)
  return parts
}

function parseLabelSelector(expression: string): LabelSelectorRequirement {
  if (expression.startsWith("!")) {
    return selectorRequirement(expression.slice(1), "not_exists")
  }
  const notIn = expression.indexOf(" notin ")
  if (notIn >= 0) {
    return selectorRequirement(
      expression.slice(0, notIn),
      "not_in",
      selectorValues(expression.slice(notIn + " notin ".length)),
    )
  }
  const inOperator = expression.indexOf(" in ")
  if (inOperator >= 0) {
    return selectorRequirement(
      expression.slice(0, inOperator),
      "in",
      selectorValues(expression.slice(inOperator + " in ".length)),
    )
  }
  for (const [operatorText, operator] of [
    ["!=", "not_in"],
    ["==", "in"],
    ["=", "in"],
  ] as const) {
    const separator = expression.indexOf(operatorText)
    if (separator >= 0) {
      return selectorRequirement(expression.slice(0, separator), operator, [
        cleanBoundedText(
          expression.slice(separator + operatorText.length).trim(),
          "Label selector value",
          MAX_LABEL_VALUE_LENGTH,
        ),
      ])
    }
  }
  return selectorRequirement(expression, "exists")
}

function selectorRequirement(
  key: string,
  operator: LabelSelectorRequirement["operator"],
  values: string[] = [],
): LabelSelectorRequirement {
  return {
    key: cleanBoundedText(key.trim(), "Label selector key", MAX_LABEL_KEY_LENGTH),
    operator,
    values,
  }
}

function selectorValues(value: string): string[] {
  const trimmed = value.trim()
  if (!trimmed.startsWith("(") || !trimmed.endsWith(")")) {
    throw new Error("Label selector values must use (a,b).")
  }
  const values = trimmed
    .slice(1, -1)
    .split(",")
    .map((item) => cleanBoundedText(item.trim(), "Label selector value", MAX_LABEL_VALUE_LENGTH))
  if (new Set(values).size !== values.length) {
    throw new Error("Label selector values must be distinct.")
  }
  return values
}

function cleanBoundedText(value: string, label: string, maxLength: number): string {
  const trimmed = value.trim()
  if (!trimmed) throw new Error(`${label} cannot be blank.`)
  if (trimmed !== value) throw new Error(`${label} cannot start or end with whitespace.`)
  if (trimmed.length > maxLength) {
    throw new Error(`${label} cannot exceed ${maxLength} characters.`)
  }
  return trimmed
}
