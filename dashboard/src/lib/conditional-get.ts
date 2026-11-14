/**
 * Conditional GETs for JSON reads. When a response carries an `ETag`, its body is kept
 * and the next GET of the same URL sends `If-None-Match`. A `304 Not Modified` is then
 * answered from the kept body as an ordinary 200, so callers (and React Query's
 * structural sharing) see the same data without the server sending it again.
 *
 * Endpoints that send no ETag are unaffected: nothing is kept and no header is added.
 */

type Entry = { etag: string; body: string; contentType: string }

export type ConditionalGetOptions = {
  /** Most URLs kept at once; the least recently used is dropped first. */
  maxEntries?: number
  /** Bodies larger than this many characters aren't kept. */
  maxBodyLength?: number
}

/** Response header set on a 200 rebuilt from a 304, for tests and diagnostics. */
export const REVALIDATED_HEADER = "x-cayu-revalidated"

export class ConditionalGetCache {
  readonly #entries = new Map<string, Entry>()
  readonly #maxEntries: number
  readonly #maxBodyLength: number

  constructor({ maxEntries = 200, maxBodyLength = 2_000_000 }: ConditionalGetOptions = {}) {
    this.#maxEntries = maxEntries
    this.#maxBodyLength = maxBodyLength
  }

  get size(): number {
    return this.#entries.size
  }

  clear(): void {
    this.#entries.clear()
  }

  /** `fetch` with If-None-Match for GETs whose last response had an ETag. */
  async fetch(
    url: string,
    init: RequestInit = {},
    fetchImpl: typeof fetch = globalThis.fetch,
  ): Promise<Response> {
    const method = (init.method ?? "GET").toUpperCase()
    if (method !== "GET") return fetchImpl(url, init)
    const headers = new Headers(init.headers)
    // A caller that sets its own validator owns the 304.
    const managed = !headers.has("If-None-Match")
    const kept = managed ? this.#entries.get(url) : undefined
    if (kept) headers.set("If-None-Match", kept.etag)
    const response = await fetchImpl(url, { ...init, headers })
    if (!managed) return response
    if (response.status === 304 && kept) {
      this.#touch(url, kept)
      return new Response(kept.body, {
        status: 200,
        headers: { "content-type": kept.contentType, etag: kept.etag, [REVALIDATED_HEADER]: "1" },
      })
    }
    if (!response.ok) {
      // An error says nothing about the kept body, except that access may have changed.
      if (response.status === 401 || response.status === 403) this.#entries.delete(url)
      return response
    }
    const etag = response.headers.get("etag")
    const contentType = response.headers.get("content-type") ?? ""
    if (!etag || !contentType.includes("json")) {
      this.#entries.delete(url)
      return response
    }
    const body = await response.text()
    if (body.length <= this.#maxBodyLength) this.#touch(url, { etag, body, contentType })
    else this.#entries.delete(url)
    return new Response(body, {
      status: response.status,
      statusText: response.statusText,
      headers: response.headers,
    })
  }

  #touch(url: string, entry: Entry): void {
    this.#entries.delete(url)
    this.#entries.set(url, entry)
    while (this.#entries.size > this.#maxEntries) {
      const oldest = this.#entries.keys().next().value
      if (oldest === undefined) break
      this.#entries.delete(oldest)
    }
  }
}

/** The dashboard's shared cache, used by every JSON read through `requestJson`. */
export const conditionalGets = new ConditionalGetCache()
