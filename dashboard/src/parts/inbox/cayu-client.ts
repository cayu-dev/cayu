/**
 * The Inbox's client from client.js, the browser client every Cayu server serves at
 * `/client.js`: `inboxApiFromCayuClient(await connect())` is a host's `inbox.api`.
 */
import type { InboxApi, InboxSearch } from "../host.tsx"
import type { InboxCounts, InboxItemState, InboxPage, InboxSettleRequest } from "../types.ts"

type Filters = InboxSearch & { signal?: AbortSignal }

/** The client.js methods the Inbox calls; a `CayuClient` from `connect()` has them all. */
export interface CayuInboxClient {
  listInbox(options?: Filters & { limit?: number; cursor?: string }): Promise<unknown>
  getInboxCounts(options?: Omit<Filters, "kind">): Promise<unknown>
  settleInboxItem(itemId: string, decision: InboxSettleRequest): Promise<unknown>
  watchInbox?(options?: InboxSearch & { onChange?: () => void }): () => void
}

type SettleResult =
  | { settled: true; item: InboxItemState }
  | { settled: false; code: string; message: string; item: InboxItemState | null }

/**
 * The error a refused settle rejects with: the HTTP status and the server's code, such as
 * 409 `profile_changed`, and the item as it stands now when the server sent it.
 */
export class InboxSettleError extends Error {
  readonly status: number
  readonly code: string
  readonly item: InboxItemState | null

  constructor(code: string, message: string, item: InboxItemState | null, status = 409) {
    super(message || `The inbox item couldn't be settled (${code}).`)
    this.name = "InboxSettleError"
    this.status = status
    this.code = code
    this.item = item
  }
}

export function inboxApiFromCayuClient(client: CayuInboxClient): InboxApi {
  const api: InboxApi = {
    list: async ({ cursor, ...query }, signal) =>
      (await client.listInbox({ ...query, cursor: cursor ?? undefined, signal })) as InboxPage,
    counts: async (query, signal) =>
      (await client.getInboxCounts({ ...query, signal })) as InboxCounts,
    settle: async (itemId, body) => {
      const result = (await client.settleInboxItem(itemId, body)) as SettleResult
      if (!result.settled) throw new InboxSettleError(result.code, result.message, result.item)
      return result.item
    },
  }
  const { watchInbox } = client
  if (watchInbox)
    api.watch = (filters, onChange) => watchInbox.call(client, { ...filters, onChange })
  return api
}
