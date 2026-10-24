import type { SessionTraceResponse } from "../../lib/generated/server-api"
import type { Span } from "./trace-model.ts"

export type TranscriptPosition = { messageIndex: number; partIndex: number }

/** Merge only server-authored identities; projected event aliases are never join keys. */
export function mergeTracePages(
  previous: SessionTraceResponse | undefined,
  incoming: SessionTraceResponse,
): SessionTraceResponse {
  if (!previous) return incoming
  if (
    previous.session_instance_id !== incoming.session_instance_id ||
    previous.session_id !== incoming.session_id
  ) {
    throw new Error("The session changed while reading its trace. Reload the trace.")
  }
  const spans = new Map(previous.spans.map((span) => [span.id, span]))
  for (const span of incoming.spans) {
    const existing = spans.get(span.id)
    const links = new Map(
      [...(existing?.transcript_links ?? []), ...span.transcript_links].map((link) => [
        JSON.stringify([link.message_index, link.part_index]),
        link,
      ]),
    )
    spans.set(span.id, {
      ...(existing && incoming.through_sequence < previous.through_sequence ? existing : span),
      event_ids: [...new Set([...(existing?.event_ids ?? []), ...span.event_ids])],
      transcript_links: [...links.values()],
    })
  }
  const events = new Map(previous.events.map((event) => [event.id, event]))
  for (const event of incoming.events) events.set(event.id, event)
  return {
    ...incoming,
    through_sequence: Math.max(previous.through_sequence, incoming.through_sequence),
    spans: [...spans.values()],
    events: [...events.values()].sort((a, b) => a.sequence - b.sequence),
  }
}

export function buildRuntimeTrace(page: SessionTraceResponse): Span {
  const events = new Map(page.events.map((event) => [event.id, event]))
  const nodes = new Map<string, Span>()
  for (const source of page.spans) {
    nodes.set(source.id, {
      id: source.id,
      kind: source.kind,
      name: source.name,
      detail: "",
      start: source.started_at === null ? 0 : Date.parse(source.started_at),
      startKnown: source.started_at !== null,
      end: source.ended_at === null ? null : Date.parse(source.ended_at),
      status:
        source.status === "completed" ? "ok" : source.status === "failed" ? "error" : source.status,
      events: source.event_ids
        .flatMap((id) => {
          const event = events.get(id)
          return event ? [event] : []
        })
        .sort((left, right) => left.sequence - right.sequence),
      children: [],
      transcriptLinks: source.transcript_links,
      linkedByOrder: false,
    })
  }
  const root = nodes.get(page.root_span_id)
  if (!root) throw new Error("The server returned a trace without its session span.")
  for (const source of page.spans) {
    if (source.parent_id === null) continue
    const child = nodes.get(source.id)
    const parent = nodes.get(source.parent_id)
    if (!parent || !child || parent === child)
      throw new Error("The server returned an incomplete trace relationship.")
    parent.children.push(child)
  }
  for (const node of nodes.values())
    node.children.sort((left, right) => {
      const leftAnchor = Number(left.id.split(":").at(-1)) || 0
      const rightAnchor = Number(right.id.split(":").at(-1)) || 0
      return leftAnchor - rightAnchor
    })
  return root
}

export function traceSpanForPosition(
  page: SessionTraceResponse,
  position: TranscriptPosition,
): string | undefined {
  const matching = page.spans.filter((span) =>
    span.transcript_links.some(
      (link) =>
        link.message_index === position.messageIndex && link.part_index === position.partIndex,
    ),
  )
  return matching.length === 1 ? matching[0]?.id : undefined
}

/** Withheld markers are presentation metadata, never user-facing raw values. */
export function displayTraceValue(value: unknown): unknown {
  if (typeof value === "string")
    return value.replace(
      /\[PRIVATE_EVENT_AUTHORITY\]|cayu_event_\d+:[a-z_]+/g,
      "Hidden by the server",
    )
  if (Array.isArray(value)) return value.map(displayTraceValue)
  if (value !== null && typeof value === "object")
    return Object.fromEntries(
      Object.entries(value).map(([key, item]) => [key, displayTraceValue(item)]),
    )
  return value
}
