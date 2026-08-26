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
  trimFraction = true,
) {
  const integerFormatter = new Intl.NumberFormat(locales)
  const magnitude = BigInt(integerDigits)
  const localizedInteger =
    sign === "-"
      ? integerFormatter.format(magnitude === 0n ? -0 : -magnitude)
      : integerFormatter.format(magnitude)
  const fraction = trimFraction ? fractionDigits?.replace(/0+$/, "") : fractionDigits
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

const MAX_FORMATTED_DECIMAL_DIGITS = 4096
const MAX_FORMATTED_DECIMAL_EXPONENT = 4096n

function exactDecimalParts(value: string) {
  const match = /^([+-]?)(\d+)(?:\.(\d+))?(?:[eE]([+-]?\d+))?$/.exec(value)
  if (!match) return null
  const sourceInteger = match[2] ?? "0"
  const sourceFraction = match[3] ?? ""
  const coefficient = `${sourceInteger}${sourceFraction}`
  if (coefficient.length > MAX_FORMATTED_DECIMAL_DIGITS) return null

  let exponent = 0
  if (match[4] !== undefined) {
    const exponentSign = match[4].startsWith("-") ? "-" : ""
    const exponentDigits = match[4].replace(/^[+-]?0*/, "") || "0"
    if (exponentDigits.length > MAX_FORMATTED_DECIMAL_EXPONENT.toString().length) return null
    const exactExponent = BigInt(`${exponentSign}${exponentDigits}`)
    if (
      exactExponent < -MAX_FORMATTED_DECIMAL_EXPONENT ||
      exactExponent > MAX_FORMATTED_DECIMAL_EXPONENT
    ) {
      return null
    }
    exponent = Number(exactExponent)
  }

  const decimalIndex = sourceInteger.length + exponent
  const normalizedDigits = Math.max(decimalIndex, coefficient.length) - Math.min(decimalIndex, 0)
  if (normalizedDigits > MAX_FORMATTED_DECIMAL_DIGITS) return null

  const integer =
    decimalIndex <= 0
      ? "0"
      : decimalIndex >= coefficient.length
        ? `${coefficient}${"0".repeat(decimalIndex - coefficient.length)}`
        : coefficient.slice(0, decimalIndex)
  const fraction =
    decimalIndex <= 0
      ? `${"0".repeat(-decimalIndex)}${coefficient}`
      : decimalIndex < coefficient.length
        ? coefficient.slice(decimalIndex)
        : undefined
  return {
    sign: match[1] === "-" ? "-" : "",
    integer: integer.replace(/^0+(?=\d)/, ""),
    fraction,
    nonzero: /[1-9]/.test(coefficient),
  }
}

function roundedExactDecimal(value: string, fractionDigits: number) {
  const parsed = exactDecimalParts(value)
  if (!parsed) return null
  const fraction = parsed.fraction ?? ""
  const retained = fraction.slice(0, fractionDigits).padEnd(fractionDigits, "0")
  const omitted = fraction[fractionDigits]
  let magnitude = BigInt(`${parsed.integer}${retained}` || "0")
  if (omitted !== undefined && omitted >= "5") magnitude += 1n

  const digits = magnitude.toString().padStart(fractionDigits + 1, "0")
  const roundedInteger =
    fractionDigits === 0 ? digits : digits.slice(0, Math.max(1, digits.length - fractionDigits))
  const roundedFraction = fractionDigits === 0 ? undefined : digits.slice(-fractionDigits)
  return {
    sign: parsed.sign === "-" && magnitude !== 0n ? "-" : "",
    integer: roundedInteger,
    fraction: roundedFraction,
    nonzero: parsed.nonzero,
    roundedToZero: magnitude === 0n,
  }
}

function currencyFractionDigits(currency: string, locales?: Intl.LocalesArgument) {
  return (
    new Intl.NumberFormat(locales, {
      style: "currency",
      currency,
      currencyDisplay: "narrowSymbol",
    }).resolvedOptions().maximumFractionDigits ?? 2
  )
}

function formatExactCurrency(
  value: string,
  currency: string,
  locales: Intl.LocalesArgument | undefined,
  currencyDisplay: "narrowSymbol" | "code",
) {
  try {
    const baseDigits = currencyFractionDigits(currency, locales)
    const baseRounded = roundedExactDecimal(value, baseDigits)
    if (!baseRounded) return `${formatDecimal(value, locales)} ${currency}`
    const digits =
      baseRounded.nonzero && baseRounded.roundedToZero ? Math.max(baseDigits, 4) : baseDigits
    const rounded = roundedExactDecimal(value, digits)
    if (!rounded) return `${formatDecimal(value, locales)} ${currency}`

    const shell = new Intl.NumberFormat(locales, {
      style: "currency",
      currency,
      currencyDisplay,
      minimumFractionDigits: digits,
      maximumFractionDigits: digits,
    }).formatToParts(1)
    const numericPartTypes = new Set(["integer", "group", "decimal", "fraction"])
    const firstNumeric = shell.findIndex((part) => numericPartTypes.has(part.type))
    if (firstNumeric < 0) return `${formatDecimal(value, locales)} ${currency}`
    let lastNumeric = firstNumeric
    while (true) {
      const next = shell[lastNumeric + 1]
      if (!next || !numericPartTypes.has(next.type)) break
      lastNumeric += 1
    }
    const prefix = shell
      .slice(0, firstNumeric)
      .map((part) => part.value)
      .join("")
    const suffix = shell
      .slice(lastNumeric + 1)
      .map((part) => part.value)
      .join("")
    const belowDisplayPrecision = rounded.nonzero && rounded.roundedToZero
    const formatted = `${prefix}${localizedExactDecimal(
      belowDisplayPrecision ? "" : rounded.sign,
      rounded.integer,
      belowDisplayPrecision ? `${"0".repeat(Math.max(0, digits - 1))}1` : rounded.fraction,
      locales,
      false,
    )}${suffix}`
    if (belowDisplayPrecision) return `<${formatted}`
    return formatted
  } catch {
    return `${formatDecimal(value, locales)} ${currency}`
  }
}

export function formatCurrency(value: string, currency: string, locales?: Intl.LocalesArgument) {
  return formatExactCurrency(value, currency, locales, "narrowSymbol")
}

export function formatCurrencyWithCode(
  value: string,
  currency: string,
  locales?: Intl.LocalesArgument,
) {
  return formatExactCurrency(value, currency, locales, "code")
}

export function formatDecimal(
  value: string | number | null | undefined,
  locales?: Intl.LocalesArgument,
) {
  if (value === null || value === undefined || value === "") return "-"
  if (typeof value === "string") {
    const parsed = exactDecimalParts(value)
    if (!parsed) return value
    return localizedExactDecimal(parsed.sign, parsed.integer, parsed.fraction, locales)
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
