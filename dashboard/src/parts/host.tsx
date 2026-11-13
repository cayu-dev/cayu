/**
 * What a part needs from the page that hosts it, and nothing else: the clock, its links
 * out, a hook for after it changed something, and one section per part with that part's
 * data client and what the viewer may do.
 *
 * The Developer dashboard provides it once, from its own API client, router and server
 * contract (src/next/app/parts-host.tsx). An app that imports a part provides its own: a
 * client for its own routes, links into its own pages or to the Developer dashboard, and
 * what its signed-in person may do. A page that shows only the Inbox supplies only the
 * `inbox` section.
 */
import { type ComponentType, createContext, type ReactNode, useContext } from "react"
import type {
  ApiPendingAction,
  HumanReviewReference,
  InboxCounts,
  InboxItemState,
  InboxPage,
  InboxQuery,
  InboxSettleRequest,
  PendingKnowledgeDetailResponse,
  RecoveryRequest,
} from "./types"

/** Where a part's links go. The host renders them with its own router. */
export type PartTarget = { session: string } | { task: string }
export type PartLink = ComponentType<{
  to: PartTarget
  className?: string
  children: ReactNode
}>

/** What a part changed, for the host to refresh its own views. */
export type PartChange = {
  part: "inbox"
  /** The settled item. For a recovery, the listed item, pending until the Inbox reads again. */
  item: InboxItemState
  decision: InboxSettleRequest["decision"] | "recover"
}

/** The Inbox's filters, as the host keeps them (usually in the URL). */
export type InboxSearch = {
  q?: string
  kind?: InboxQuery["kind"]
  session_id?: string
  agent_name?: string
}

export type InboxCountsQuery = Pick<InboxQuery, "session_id" | "agent_name" | "q">

/** Cayu's inbox, as the host reaches it: the dashboard's `/api`, or an app's own routes. */
export interface InboxApi {
  list(query: InboxQuery, signal?: AbortSignal): Promise<InboxPage>
  counts(query: InboxCountsQuery, signal?: AbortSignal): Promise<InboxCounts>
  /**
   * Settle one item. A refusal rejects with an error carrying the HTTP `status` and the
   * server's `code`, such as 409 `profile_changed`.
   */
  settle(itemId: string, body: InboxSettleRequest): Promise<InboxItemState>
  /** Titles for the sessions items belong to. Without it, an item shows its session's ID. */
  sessionTitles?(ids: string[]): Promise<{ session_id: string; title?: string | null }[]>
  /**
   * A knowledge proposal's full text, which the person reads before keeping or dropping it.
   * Without it, the card shows the summary the inbox lists and decides from that.
   */
  knowledgeEntry?(entryId: string): Promise<PendingKnowledgeDetailResponse>
  /** The exact pending recovery behind a recovery item, for recovering it in its session. */
  sessionRecovery?(
    sessionId: string,
    attentionId: string | null,
    signal?: AbortSignal,
  ): Promise<ApiPendingAction | null>
  /**
   * Settle a recovery right from the card, with the evidence the person gives. The host
   * finds the pending call it names (the dashboard's prototype settles its fake data).
   * Without it, recovery happens in the session (`InboxCard`'s `onRecover`).
   */
  recover?(sessionId: string, body: RecoveryRequest): Promise<unknown>
  /** client.js's `watchInbox`, for hosts that follow changes instead of polling. */
  watch?(filters: InboxSearch, onChange: () => void): () => void
}

/** A decision made through protected human review, ready for the host's settle. */
export type ProtectedReviewDecision = {
  decision: "approve" | "deny" | "answer"
  /** The reviewed content's reference; settling without it is refused. */
  review_reference: HumanReviewReference
  answer?: string
  reason?: string | null
}
/**
 * Protected human review, which then decides approvals and questions instead of the
 * buttons: usually `ProtectedReviewPart` with the host's `ReviewApi` and purpose.
 */
export type ProtectedReview = ComponentType<{
  sessionId: string
  kind: "user_input" | "tool_approval"
  disabled: boolean
  onDecision: (decision: ProtectedReviewDecision) => Promise<void>
}>

export interface InboxHost {
  api: InboxApi
  /** The server lists the inbox (`inbox_queries`). */
  enabled: boolean
  /** The viewer may decide approvals and questions (`pending_action_resolution`). */
  decide: boolean
  /** The viewer may keep or drop knowledge proposals (`knowledge_review`). */
  reviewKnowledge: boolean
  /** The viewer has full access: may approve with changed arguments, and recover in a session. */
  fullAccess: boolean
  /** When the deployment configures protected human review. */
  ProtectedReview?: ProtectedReview
  /** Refresh of the newest page, in ms; false when `api.watch` pushes changes. Default 5000. */
  pollMs?: number | false
}

export interface PartsHost {
  /** The clock. */
  now(): number
  Link: PartLink
  /** After a part changed something and refreshed its own queries; the host refreshes its own. */
  onSettled?(change: PartChange): void | Promise<void>
  inbox?: InboxHost
}

const PartsHostContext = createContext<PartsHost | null>(null)

export function PartsHostProvider({ value, children }: { value: PartsHost; children: ReactNode }) {
  return <PartsHostContext.Provider value={value}>{children}</PartsHostContext.Provider>
}

export function usePartsHost(): PartsHost {
  const host = useContext(PartsHostContext)
  if (!host) throw new Error("A part must be rendered inside <PartsHostProvider>.")
  return host
}

/** The host with the Inbox's section, which every Inbox part needs. */
export function useInboxHost(): PartsHost & { inbox: InboxHost } {
  const host = usePartsHost()
  if (!host.inbox)
    throw new Error(
      'This part needs the host\'s "inbox" section. Pass `inbox` to <PartsHostProvider>.',
    )
  return host as PartsHost & { inbox: InboxHost }
}
