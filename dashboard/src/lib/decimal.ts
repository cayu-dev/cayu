export function addDecimalStrings(left: string, right: string) {
  const split = (value: string) => {
    const trimmed = value.trim()
    const sign = trimmed.startsWith("-") ? -1n : 1n
    const unsigned = trimmed.replace(/^[+-]/, "")
    const [mantissa = "0", exponentText] = unsigned.toLowerCase().split("e")
    const exponent = exponentText ? Number.parseInt(exponentText, 10) : 0
    const [wholeRaw = "0", fractionalRaw = ""] = mantissa.split(".")
    const digits = `${wholeRaw}${fractionalRaw}`.replace(/^0+(?=\d)/, "") || "0"
    const decimalPlaces = fractionalRaw.length - exponent
    if (decimalPlaces <= 0) {
      return { sign, whole: `${digits}${"0".repeat(Math.abs(decimalPlaces))}`, fractional: "" }
    }
    if (digits.length > decimalPlaces) {
      const splitAt = digits.length - decimalPlaces
      return { sign, whole: digits.slice(0, splitAt), fractional: digits.slice(splitAt) }
    }
    return { sign, whole: "0", fractional: `${"0".repeat(decimalPlaces - digits.length)}${digits}` }
  }
  const leftParts = split(left)
  const rightParts = split(right)
  const precision = Math.max(leftParts.fractional.length, rightParts.fractional.length)
  const scale = 10n ** BigInt(precision)
  const parse = (parts: ReturnType<typeof split>) =>
    parts.sign *
    (BigInt(parts.whole) * scale + BigInt(parts.fractional.padEnd(precision, "0") || "0"))

  const total = parse(leftParts) + parse(rightParts)
  const sign = total < 0n ? "-" : ""
  const absolute = total < 0n ? -total : total
  if (precision === 0) return `${sign}${absolute.toString()}`
  const whole = absolute / scale
  const fractional = (absolute % scale).toString().padStart(precision, "0").replace(/0+$/g, "")
  return fractional ? `${sign}${whole.toString()}.${fractional}` : `${sign}${whole.toString()}`
}
