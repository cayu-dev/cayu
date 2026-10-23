/**
 * A fake runtime server for the pages the rework leaves unchanged (Evals, Usage, Tasks,
 * Environments, Artifacts, System). Those pages call the API through `fetch`, so the
 * prototype answers their `/api/...` requests here, from the current seed's records.
 * Requests nothing handles get a 501, which the pages show as an error.
 */
import type { AppSeed } from "../model"

export type LegacyContext = {
  /** The current seed's live records. */
  records: () => AppSeed
  /** The seed's clock, in milliseconds. */
  now: () => number
}

export type LegacyRequest = {
  method: string
  path: string
  /** Values for the `:name` parts of the route's path. */
  params: Record<string, string>
  query: URLSearchParams
  body: unknown
  headers: Headers
}

export type LegacyRoute = {
  method: "GET" | "POST" | "PATCH" | "PUT" | "DELETE"
  /** The path after `/api`, with `:name` parts, e.g. `/evals/runs/:runId`. */
  path: string
  /** Returns the JSON body, or a Response for anything else (downloads). */
  handle: (request: LegacyRequest, context: LegacyContext) => unknown
}

export class HttpError extends Error {
  readonly status: number
  readonly detail: string
  constructor(status: number, detail: string) {
    super(detail)
    this.status = status
    this.detail = detail
  }
}

const LATENCY_MS = 120

function match(pattern: string, path: string): Record<string, string> | null {
  const want = pattern.split("/")
  const got = path.split("/")
  if (want.length !== got.length) return null
  const params: Record<string, string> = {}
  for (const [i, part] of want.entries()) {
    const value = got[i] ?? ""
    if (part.startsWith(":")) params[part.slice(1)] = decodeURIComponent(value)
    else if (part !== value) return null
  }
  return params
}

const json = (status: number, body: unknown) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  })

/** Answers `/api/...` requests from `routes`; everything else goes to the real fetch. */
export function installLegacyServer(routes: LegacyRoute[], context: LegacyContext) {
  const realFetch = window.fetch.bind(window)
  window.fetch = async (input, init) => {
    const request = new Request(input, init)
    const url = new URL(request.url, window.location.href)
    const at = url.pathname.indexOf("/api/")
    if (url.origin !== window.location.origin || at === -1) return realFetch(input, init)
    const path = url.pathname.slice(at + "/api".length)
    const text = await request.text()
    let body: unknown = null
    try {
      body = text ? JSON.parse(text) : null
    } catch {
      body = text
    }
    await new Promise((resolve) => setTimeout(resolve, LATENCY_MS))
    for (const route of routes) {
      if (route.method !== request.method) continue
      const params = match(route.path, path)
      if (!params) continue
      try {
        const result = route.handle(
          {
            method: request.method,
            path,
            params,
            query: url.searchParams,
            body,
            headers: request.headers,
          },
          context,
        )
        return result instanceof Response ? result : json(200, structuredClone(result))
      } catch (error) {
        if (error instanceof HttpError) return json(error.status, { detail: error.detail })
        throw error
      }
    }
    return json(501, { detail: `The prototype doesn't fake ${request.method} /api${path}.` })
  }
}
