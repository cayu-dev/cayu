export const MAX_LABEL_KEY_LENGTH = 128
export const MAX_LABEL_VALUE_LENGTH = 512

export function unicodeScalarLength(value: string): number {
  return Array.from(value).length
}

export function invalidDurableText(value: string): string | null {
  if (value.includes("\0")) return "must not contain NUL characters"
  for (const character of value) {
    const codePoint = character.codePointAt(0)
    if (codePoint !== undefined && codePoint >= 0xd800 && codePoint <= 0xdfff) {
      return "must not contain Unicode surrogate code points"
    }
  }
  return null
}
