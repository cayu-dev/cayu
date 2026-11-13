/*
 * An app that imports @cayu-tech/dashboard-parts the way a generated app does: its own
 * kit as `@kit`, a fake host with no router or services, and the record of what the
 * parts asked the host to do on `window.consumer` for the Playwright test.
 */
import "./styles.css"
import {
  type HumanReviewReference,
  type InboxItem,
  type InboxItemState,
  type PartsHost,
  PartsHostProvider,
  type ProtectedReview,
  ProtectedReviewPart,
  type ReviewApi,
} from "@cayu-tech/dashboard-parts"
import { InboxCard, type InboxSearch, InboxPart } from "@cayu-tech/dashboard-parts/inbox"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { Component, type ReactNode, StrictMode, useState } from "react"
import { createRoot } from "react-dom/client"

const params = new URLSearchParams(location.search)
const NOW = Date.parse("2026-11-12T18:00:00Z")
const ITEMS: InboxItem[] = [
  {
    id: "approval-1",
    kind: "approval",
    title: "Hold a payment",
    created_at: "2026-11-12T17:40:00Z",
    decisions: ["approve", "deny"],
    session_id: "session-1",
    agent_name: "ap-auditor",
    tool_name: "hold_payment",
  },
  {
    id: "question-1",
    kind: "question",
    title: "Which cost center?",
    created_at: "2026-11-12T16:00:00Z",
    decisions: ["answer"],
    session_id: "session-2",
    question: "Which cost center should the accrual post to?",
    options: ["CC-4100", "CC-4300"],
  },
]

const record = {
  settles: [] as { id: string; body: unknown }[],
  settled: [] as unknown[],
  reviews: [] as unknown[][],
  fetches: 0,
}
Object.assign(window, { consumer: record })
// The parts read only through the host's clients, never the network themselves.
const realFetch = window.fetch.bind(window)
window.fetch = (...args) => {
  record.fetches += 1
  return realFetch(...args)
}

const REFERENCE = (purpose: string): HumanReviewReference => ({
  content_tag: "tag-1",
  policy_version: "v1",
  context: { purpose, recipient: "alice", tenant: null },
})

const FakeReview: ProtectedReview = ({ sessionId, onDecision }) => (
  <button
    type="button"
    data-protected-review={sessionId}
    onClick={() => void onDecision({ decision: "approve", review_reference: REFERENCE("stub") })}
  >
    Review protected arguments
  </button>
)

// A review whose reference names `viewPurpose` (the asked purpose unless the URL says).
const reviewApi: ReviewApi = {
  inspect: async (sessionId, purpose) => {
    record.reviews.push(["inspect", sessionId, purpose])
    return {
      session_id: sessionId,
      status: "permitted",
      guidance: "Review all calls before deciding.",
      kind: "tool_approval",
      interaction_id: "approval-1",
      tool_round_id: "round-1",
      tool_call_id: "call-1",
      reference: REFERENCE(params.get("viewPurpose") ?? purpose),
      fields: [{ label: "Amount", text: "$12,400 &amp; fees" }],
      long_fields: [{ key: "memo", label: "Memo", characters: 2400 }],
      calls: [{ tool_call_id: "call-1", tool_name: "hold_payment", on_grant: "eligible" }],
    }
  },
  longField: async (sessionId, reference, key) => {
    record.reviews.push(["longField", sessionId, reference.content_tag, key])
    return { session_id: sessionId, key, status: "permitted", text: "The whole memo." }
  },
}
const purpose = params.get("purpose") === "none" ? null : "consumer"
const PartReview: ProtectedReview = (props) => (
  <ProtectedReviewPart {...props} api={reviewApi} purpose={purpose} />
)

const host: PartsHost = {
  now: () => NOW,
  Link: ({ to, className, children }) => (
    <a className={className} href={`#${"session" in to ? to.session : to.task}`}>
      {children}
    </a>
  ),
  onSettled: (change) => {
    record.settled.push(change)
  },
  inbox:
    params.get("missing") === "1"
      ? undefined
      : {
          api: {
            list: async () => ({ items: ITEMS, has_more: false, next_cursor: null, issues: [] }),
            counts: async () => ({
              counts: { approval: 1, question: 1, recovery: 0, knowledge: 0 },
              total_count: 2,
              complete: true,
              unavailable_reason: null,
            }),
            settle: async (id, body): Promise<InboxItemState> => {
              record.settles.push({ id, body })
              const item = ITEMS.find((candidate) => candidate.id === id)!
              return { id, kind: item.kind, status: "settled", session_id: item.session_id }
            },
          },
          enabled: true,
          decide: params.get("decide") !== "0",
          reviewKnowledge: false,
          fullAccess: false,
          ProtectedReview:
            params.get("protected") === "1"
              ? FakeReview
              : params.get("review") === "1"
                ? PartReview
                : undefined,
          pollMs: false,
        },
}

class Boundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state = { error: null as Error | null }
  static getDerivedStateFromError(error: Error) {
    return { error }
  }
  render() {
    return this.state.error ? <p role="alert">{this.state.error.message}</p> : this.props.children
  }
}

function App() {
  const [search, setSearch] = useState<InboxSearch>({})
  return params.get("card") === "1" || params.get("missing") === "1" ? (
    <InboxCard item={ITEMS[0]!} />
  ) : (
    <InboxPart search={search} onSearch={setSearch} advancedFilters={false} />
  )
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={new QueryClient()}>
      <PartsHostProvider value={host}>
        <main className="mx-auto max-w-3xl p-6">
          <Boundary>
            <App />
          </Boundary>
        </main>
      </PartsHostProvider>
    </QueryClientProvider>
  </StrictMode>,
)
