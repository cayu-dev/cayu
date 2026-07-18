type SearchPrimitive = string | number | boolean

function searchPrimitive(value: unknown): SearchPrimitive | undefined {
  return typeof value === "string" || typeof value === "number" || typeof value === "boolean"
    ? value
    : undefined
}

export function parseDashboardSearch(searchString: string): Record<string, unknown> {
  const query = searchString.startsWith("?") ? searchString.slice(1) : searchString
  const result: Record<string, unknown> = Object.create(null)

  for (const [key, value] of new URLSearchParams(query)) {
    const current = result[key]
    if (current === undefined) {
      result[key] = value
    } else if (Array.isArray(current)) {
      current.push(value)
    } else {
      result[key] = [current, value]
    }
  }
  return result
}

export function stringifyDashboardSearch(search: Record<string, unknown>): string {
  const query = new URLSearchParams()

  for (const [key, value] of Object.entries(search)) {
    if (Array.isArray(value)) {
      for (const item of value) {
        const primitive = searchPrimitive(item)
        if (primitive !== undefined) query.append(key, String(primitive))
      }
      continue
    }
    const primitive = searchPrimitive(value)
    if (primitive !== undefined) query.set(key, String(primitive))
  }

  const encoded = query.toString()
  return encoded === "" ? "" : `?${encoded}`
}
