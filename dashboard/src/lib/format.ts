export function formatDateTime(value: string | null | undefined) {
  if (!value) return "-"
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? "-" : date.toLocaleString()
}

function exactInteger(value: string | number): bigint | null {
  if (typeof value === "number") {
    return Number.isSafeInteger(value) ? BigInt(value) : null
  }
  if (!/^-?\d+$/.test(value)) return null
  try {
    return BigInt(value)
  } catch {
    return null
  }
}

export function formatCount(value: string | number | null | undefined) {
  if (typeof value === "string") {
    return exactInteger(value)?.toLocaleString() ?? "0"
  }
  return typeof value === "number" && Number.isFinite(value) ? value.toLocaleString() : "0"
}

export function sumCounts(...values: Array<string | number>): string {
  let total = 0n
  for (const value of values) {
    const integer = exactInteger(value)
    if (integer === null) return "0"
    total += integer
  }
  return total.toString()
}

export function formatBytes(value: number | null | undefined) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "-"
  if (value < 1024) return `${formatCount(value)} B`
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`
  return `${(value / (1024 * 1024)).toFixed(1)} MB`
}

function localizedExactDecimal(
  sign: string,
  integerDigits: string,
  fractionDigits: string | undefined,
  locales?: Intl.LocalesArgument,
) {
  const integerFormatter = new Intl.NumberFormat(locales)
  const magnitude = BigInt(integerDigits)
  const localizedInteger =
    sign === "-"
      ? integerFormatter.format(magnitude === 0n ? -0 : -magnitude)
      : integerFormatter.format(magnitude)
  const fraction = fractionDigits?.replace(/0+$/, "")
  if (!fraction) return localizedInteger

  const decimalSeparator = new Intl.NumberFormat(locales, {
    useGrouping: false,
    minimumFractionDigits: 1,
    maximumFractionDigits: 1,
  })
    .formatToParts(1.1)
    .find((part) => part.type === "decimal")?.value
  if (!decimalSeparator) return `${localizedInteger}.${fraction}`

  const digitFormatter = new Intl.NumberFormat(locales, {
    useGrouping: false,
    maximumFractionDigits: 0,
  })
  const localizedFraction = [...fraction]
    .map((digit) => digitFormatter.format(Number(digit)))
    .join("")
  return `${localizedInteger}${decimalSeparator}${localizedFraction}`
}

export function formatDecimal(
  value: string | number | null | undefined,
  locales?: Intl.LocalesArgument,
) {
  if (value === null || value === undefined || value === "") return "-"
  if (typeof value === "string") {
    const match = /^([+-]?)(\d+)(?:\.(\d+))?$/.exec(value)
    if (!match) return value
    const sign = match[1] ?? ""
    const integerDigits = match[2] ?? "0"
    const fractionDigits = match[3]
    return localizedExactDecimal(sign, integerDigits, fractionDigits, locales)
  }
  const numeric = value
  if (!Number.isFinite(numeric)) return String(value)
  return numeric.toLocaleString(locales, {
    maximumFractionDigits: 6,
  })
}

export function formatTime(value: string | null | undefined) {
  if (!value) return "-"
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? "-" : date.toLocaleTimeString()
}

export function numericValue(value: unknown) {
  if (typeof value === "number") return Number.isFinite(value) ? value : 0
  if (typeof value === "string" && value.trim() !== "") {
    const numberValue = Number(value)
    return Number.isFinite(numberValue) ? numberValue : 0
  }
  return 0
}

export function payloadObject(payload: unknown): Record<string, unknown> {
  return payload && typeof payload === "object" && !Array.isArray(payload)
    ? (payload as Record<string, unknown>)
    : {}
}

export function modelUsagePayload(payload: Record<string, unknown>) {
  const usageMetrics = payloadObject(payload.usage_metrics)
  if (Object.keys(usageMetrics).length > 0) return usageMetrics
  return payloadObject(payload.usage)
}
