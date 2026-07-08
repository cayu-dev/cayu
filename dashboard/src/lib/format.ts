export function formatDateTime(value: string | null | undefined) {
  if (!value) return "-"
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? "-" : date.toLocaleString()
}

export function formatCount(value: number | null | undefined) {
  return typeof value === "number" && Number.isFinite(value) ? value.toLocaleString() : "0"
}

export function formatDecimal(value: string | number | null | undefined) {
  if (value === null || value === undefined || value === "") return "-"
  const numeric = typeof value === "number" ? value : Number(value)
  if (!Number.isFinite(numeric)) return String(value)
  return numeric.toLocaleString(undefined, {
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
