export function hostedSearchSourceRows(sources: Record<string, unknown>[]) {
  const occurrences = new Map<string, number>()
  return sources.map((source) => {
    const isApi = source.type === "api"
    const url = !isApi && typeof source.url === "string" ? source.url : null
    const label = isApi
      ? typeof source.name === "string"
        ? source.name
        : "API source"
      : (typeof source.title === "string" && source.title) || url || "Source"
    const identity = JSON.stringify([isApi ? "api" : "url", url, label])
    const occurrence = occurrences.get(identity) ?? 0
    occurrences.set(identity, occurrence + 1)
    return { key: `${identity}:${occurrence}`, label, url }
  })
}
