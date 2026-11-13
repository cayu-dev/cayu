/**
 * One inbox item with its decision buttons, used inline in a session and on the Inbox
 * page. Deciding settles the same record everywhere (Cayu Cloud and the app included).
 * Recovery is the exception: the server only takes it on the session page, with the
 * evidence of what happened, so on the Inbox page a recovery item links to its session.
 */

import {
  BulbIcon,
  HelpCircleIcon,
  Icon,
  type KitIcon,
  RotateLeft01Icon,
  SecurityCheckIcon,
} from "@kit/icons"
import { cn } from "@kit/lib"
import { Button } from "@kit/ui"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { type ReactNode, useState } from "react"
import { relativeTime } from "../format"
import { type PartChange, useInboxHost } from "../host"
import { partKeys } from "../keys"
import { ErrorState } from "../states"
import type {
  InboxItem,
  InboxItemState,
  InboxSettleRequest,
  PendingKnowledgeDetailResponse,
  RecoveryRequest,
} from "../types"
import { isProfileChangedConflict, withProfileAdoption } from "./inbox-adoption"
import { refreshAfterChange } from "./keys"

const KIND: Record<InboxItem["kind"], { icon: KitIcon; label: string }> = {
  approval: { icon: SecurityCheckIcon, label: "Needs your approval" },
  question: { icon: HelpCircleIcon, label: "Needs your answer" },
  recovery: { icon: RotateLeft01Icon, label: "Needs recovery" },
  knowledge: { icon: BulbIcon, label: "Knowledge to review" },
}

/** A listed item as the change a recovery reports; it stays pending until the next read. */
const listedState = (item: InboxItem): InboxItemState => ({
  id: item.id,
  kind: item.kind,
  agent_name: item.agent_name,
  session_id: item.session_id,
  status: "pending",
  item,
})

export function InboxCard({
  item,
  context,
  onRecover,
  mutationBlocked = false,
  recoveryEnabled = false,
  showKnowledgeEvidence = false,
}: {
  item: InboxItem
  /** On the Inbox page, the session the item belongs to. */
  context?: { title: string; sessionId: string }
  onRecover?: (body: RecoveryRequest) => Promise<"observed" | "unknown">
  mutationBlocked?: boolean
  recoveryEnabled?: boolean
  showKnowledgeEvidence?: boolean
}) {
  const host = useInboxHost()
  const { now, Link } = host
  const { api, decide: decisionsEnabled, reviewKnowledge, fullAccess, ProtectedReview } = host.inbox
  const available = item.kind === "knowledge" ? reviewKnowledge : decisionsEnabled
  // Without the full text, the card shows the listed summary and decides from that.
  const readsKnowledge = !!api.knowledgeEntry
  const knowledgeSource = useQuery({
    queryKey: partKeys.knowledgeEntry(item.knowledge?.entry_id, item.knowledge?.revision),
    queryFn: () => api.knowledgeEntry!(item.knowledge!.entry_id),
    enabled: readsKnowledge && item.kind === "knowledge" && !!item.knowledge,
  })
  const knowledgeReady =
    !readsKnowledge ||
    (!!knowledgeSource.data &&
      !knowledgeSource.error &&
      knowledgeSource.data.revision === item.knowledge?.revision)
  // The host settles recoveries itself, with the evidence the person gives.
  const recoversInPlace = !!api.recover
  const recoveryEvidence = useQuery({
    queryKey: partKeys.sessionRecovery(item.session_id, item.attention_id),
    queryFn: ({ signal }) =>
      api.sessionRecovery!(item.session_id!, item.attention_id ?? null, signal),
    enabled:
      !recoversInPlace &&
      !!api.sessionRecovery &&
      !context &&
      item.kind === "recovery" &&
      !!item.session_id &&
      recoveryEnabled &&
      !mutationBlocked,
    retry: false,
  })
  const queryClient = useQueryClient()
  const [answer, setAnswer] = useState("")
  const [reason, setReason] = useState("")
  const [claimTask, setClaimTask] = useState(false)
  const [override, setOverride] = useState(false)
  const [argumentsText, setArgumentsText] = useState("{}")
  const [argumentError, setArgumentError] = useState<string | null>(null)
  const canOverride = fullAccess
  const overrideSuspended = override && !canOverride
  // The Inbox's own queries, then the host's, by contract.
  const refresh = (change: PartChange) => refreshAfterChange(queryClient, host, change)
  const settleItem = useMutation({
    mutationFn: (body: InboxSettleRequest) => api.settle(item.id, body),
    onSuccess: (state, body) => refresh({ part: "inbox", item: state, decision: body.decision }),
  })
  const recover = useMutation({
    mutationFn: async (body: RecoveryRequest) => {
      if (!item.session_id) throw new Error("This recovery item names no session.")
      if (onRecover) return await onRecover(body)
      else if (api.recover) await api.recover(item.session_id, body)
      else throw new Error("Open the session to observe this recovery safely.")
    },
    onSuccess: () => refresh({ part: "inbox", item: listedState(item), decision: "recover" }),
  })
  // Whichever action this card took: settling, or recovering on the session page.
  const settle = item.kind === "recovery" ? recover : settleItem
  // A real pending recovery remains unresolved until the authoritative Inbox read removes it.
  const decided =
    item.kind === "recovery" ? recoversInPlace && recover.isSuccess : settleItem.isSuccess
  const decide = (
    decision: InboxSettleRequest["decision"],
    extra: Partial<InboxSettleRequest> = {},
  ) => {
    if (!available || !item.decisions.includes(decision)) return
    let argumentsValue: Record<string, unknown> | undefined
    if (decision === "approve" && override) {
      if (!canOverride) return
      try {
        const parsed: unknown = JSON.parse(argumentsText)
        if (!parsed || typeof parsed !== "object" || Array.isArray(parsed))
          throw new Error("Expected an object")
        argumentsValue = parsed as Record<string, unknown>
        setArgumentError(null)
      } catch {
        setArgumentError("Changed arguments must be a JSON object.")
        return
      }
    }
    settleItem.mutate({
      decision,
      ...(argumentsValue ? { arguments: argumentsValue } : {}),
      reason: reason.trim() || null,
      ...(claimTask ? { task_continuation: "claim" as const } : {}),
      ...extra,
    })
  }
  // The application changed while this approval waited: offer the same decision again,
  // asking the server to adopt the update.
  const adoptAfterUpdate =
    item.kind === "approval" && settleItem.variables && isProfileChangedConflict(settleItem.error)
      ? settleItem.variables
      : null
  const { icon: kindIcon, label } = KIND[item.kind]
  const busy =
    mutationBlocked ||
    !available ||
    settle.isPending ||
    decided ||
    (item.kind === "knowledge" && !knowledgeReady)

  let body: ReactNode
  let actions: ReactNode
  if (item.kind === "approval") {
    body = (
      <>
        <p>
          {item.agent_name ?? "The specialist"} wants to run{" "}
          <code className="font-mono text-[13px]">{item.tool_name}</code>.
        </p>
        {item.detail && <p className="text-xs text-muted-foreground">{item.detail}</p>}
      </>
    )
    actions = (
      <>
        <Button
          size="sm"
          disabled={busy || overrideSuspended || !item.decisions.includes("approve")}
          onClick={() => decide("approve")}
        >
          Approve
        </Button>
        <Button
          size="sm"
          variant="outline"
          disabled={busy || !item.decisions.includes("deny")}
          onClick={() => decide("deny")}
        >
          Deny
        </Button>
      </>
    )
  } else if (item.kind === "question") {
    body = <p className="leading-relaxed">{item.question}</p>
    actions = (
      <div className="w-full space-y-2">
        {(item.options ?? []).length > 0 && (
          <div className="flex flex-wrap gap-2">
            {item.options?.map((o) => (
              <Button
                key={o}
                size="sm"
                variant="outline"
                disabled={busy}
                onClick={() => decide("answer", { answer: o })}
              >
                {o}
              </Button>
            ))}
          </div>
        )}
        <form
          className="flex gap-2"
          onSubmit={(e) => {
            e.preventDefault()
            if (answer.trim()) decide("answer", { answer: answer.trim() })
          }}
        >
          <label htmlFor={`answer-${item.id}`} className="sr-only">
            Your answer
          </label>
          <input
            id={`answer-${item.id}`}
            value={answer}
            onChange={(e) => setAnswer(e.target.value)}
            placeholder={(item.options ?? []).length ? "Or type an answer" : "Type an answer"}
            className="h-7 min-w-0 flex-1 rounded-lg border border-input bg-background px-2.5 text-sm outline-none focus:border-ring"
          />
          <Button size="sm" variant="secondary" type="submit" disabled={busy || !answer.trim()}>
            Send
          </Button>
        </form>
      </div>
    )
  } else if (item.kind === "recovery") {
    body = (
      <>
        <p>
          <code className="font-mono text-[13px]">{item.tool_name ?? "A tool call"}</code> started
          but never reported a result. The worker probably stopped while it was running, so it may
          or may not have taken effect. Check what really happened, then record it.
        </p>
        {item.question && (
          <p className="text-xs text-muted-foreground">Question: {item.question}</p>
        )}
        {item.detail && (
          <p className="text-xs text-muted-foreground">Server detail: {item.detail}</p>
        )}
      </>
    )
    actions =
      context || !item.session_id ? (
        item.session_id ? (
          <Link
            to={{ session: item.session_id }}
            className="inline-flex h-7 items-center rounded-lg border border-border px-2.5 text-[13px] font-medium hover:bg-muted"
          >
            Recover in the session
          </Link>
        ) : (
          <p className="text-xs text-muted-foreground">
            This item names no session, so it can't be recovered here.
          </p>
        )
      ) : !recoversInPlace && (!fullAccess || !recoveryEnabled) ? (
        <p className="text-xs text-muted-foreground">
          Manual recovery requires full access and recovery capability.
        </p>
      ) : !recoversInPlace &&
        (recoveryEvidence.isPending ||
          recoveryEvidence.error ||
          !recoveryEvidence.data?.tool_call_id ||
          (!recoveryEvidence.data.approval_id &&
            !recoveryEvidence.data.input_id &&
            !recoveryEvidence.data.round_id)) ? (
        <p className="text-xs text-muted-foreground">
          {recoveryEvidence.isPending
            ? "Checking recovery evidence…"
            : "Recovery is unavailable because its exact tool-call authority is not exposed. Inspect the recovery evidence before proceeding."}
        </p>
      ) : (
        <RecoveryForm
          id={item.id}
          question={
            recoversInPlace
              ? (item.question ?? null)
              : recoveryEvidence.data?.input_id
                ? (recoveryEvidence.data.question ?? "The person's answer")
                : null
          }
          busy={busy}
          onRecover={(body) => recover.mutate({ ...body, attention_id: item.attention_id })}
        />
      )
  } else {
    body = (
      <>
        <p>{item.agent_name ?? "The specialist"} wants to remember:</p>
        <blockquote className="rounded-lg bg-muted px-3 py-2">
          {item.detail ?? item.title}
        </blockquote>
      </>
    )
    actions = (
      <>
        <Button
          size="sm"
          disabled={busy || !item.decisions.includes("keep")}
          onClick={() => decide("keep")}
        >
          Keep
        </Button>
        <Button
          size="sm"
          variant="outline"
          disabled={busy || !item.decisions.includes("drop")}
          onClick={() => decide("drop")}
        >
          Drop
        </Button>
      </>
    )
  }

  const protectedDecision =
    !!ProtectedReview && !!item.session_id && (item.kind === "approval" || item.kind === "question")
  if (protectedDecision) {
    actions = (
      <ProtectedReview
        sessionId={item.session_id!}
        kind={item.kind === "approval" ? "tool_approval" : "user_input"}
        disabled={busy || item.decisions.length === 0}
        onDecision={async (decision) => {
          // An answer carries the card's own reason; approve and deny carry the review's.
          const body: InboxSettleRequest =
            decision.decision === "answer"
              ? {
                  decision: "answer",
                  answer: decision.answer,
                  reason: reason.trim() || null,
                  review_reference: decision.review_reference,
                }
              : {
                  decision: decision.decision,
                  reason: decision.reason,
                  review_reference: decision.review_reference,
                }
          if (!item.decisions.includes(body.decision))
            throw new Error("This decision is no longer available.")
          await settleItem.mutateAsync({
            ...body,
            ...(claimTask ? { task_continuation: "claim" as const } : {}),
          })
        }}
      />
    )
  }

  return (
    <div
      data-inbox-item={item.id}
      className={cn(
        "min-w-0 break-words rounded-xl border bg-card",
        decided ? "border-border opacity-70" : "border-warning/50 shadow-xs",
      )}
    >
      <div className="flex items-center gap-2 border-b border-border px-3.5 py-2 text-xs">
        <Icon icon={kindIcon} className="size-3.5 text-warning" aria-hidden="true" />
        <span className="font-medium text-warning">{decided ? "Decided" : label}</span>
        {context && (
          <>
            <span className="text-muted-foreground">·</span>
            <Link
              to={{ session: context.sessionId }}
              className="min-w-0 truncate text-muted-foreground hover:text-foreground hover:underline"
            >
              {context.title}
            </Link>
          </>
        )}
        <span className="ml-auto shrink-0 text-muted-foreground">
          {relativeTime(Date.parse(item.created_at), now())}
        </span>
      </div>
      <div className="space-y-2 px-3.5 py-3 text-sm">
        {body}
        {item.kind === "approval" && overrideSuspended && !decided && (
          <div role="status" className="space-y-2 text-xs">
            <p>
              Your current role cannot change arguments. Your draft is retained. Choose the original
              arguments to approve without changes.
            </p>
            <Button
              size="sm"
              variant="outline"
              disabled={busy}
              onClick={() => {
                setOverride(false)
                setArgumentError(null)
              }}
            >
              Use original arguments
            </Button>
          </div>
        )}
        {!available && item.kind !== "recovery" && (
          <p role="status">Decisions are unavailable for this deployment.</p>
        )}
        {item.kind === "knowledge" && item.knowledge && (
          <KnowledgeSource
            entry={knowledgeSource.data}
            error={knowledgeSource.error}
            revision={item.knowledge.revision}
            shown={readsKnowledge}
            showEvidence={showKnowledgeEvidence}
          />
        )}
        {item.arguments_unavailable_reason === "no_review_policy" && (
          <p className="text-xs text-muted-foreground">
            This agent shows no arguments for approvals. To show selected ones, the agent can
            configure a human review policy such as{" "}
            <code className="font-mono">ToolArgumentReview</code>.
          </p>
        )}
        {item.arguments_unavailable_reason === "quarantined" && (
          <p className="text-xs text-muted-foreground">
            Arguments are not listed in the inbox. The app's review policy decides what protected
            human review shows.
          </p>
        )}
        {!decided && item.kind !== "recovery" && (
          <details className="text-xs" open={override || !!argumentError}>
            <summary className="cursor-pointer text-muted-foreground">Decision options</summary>
            <div className="mt-3 space-y-3">
              {(item.kind === "approval" || item.kind === "question") && !decided && (
                <label className="flex items-start gap-2 text-xs">
                  <input
                    type="checkbox"
                    checked={claimTask}
                    onChange={(event) => setClaimTask(event.target.checked)}
                    disabled={busy}
                  />
                  Continue an interrupted task with this decision. Active workers remain protected.
                </label>
              )}
              {item.kind === "approval" && !protectedDecision && canOverride && !decided && (
                <div className="space-y-2">
                  <label className="flex items-center gap-2 text-xs">
                    <input
                      type="checkbox"
                      checked={override}
                      onChange={(event) => setOverride(event.target.checked)}
                      disabled={busy}
                    />
                    Approve with changed arguments
                  </label>
                  {override && (
                    <label className="block text-xs">
                      Changed arguments
                      <textarea
                        aria-label="Changed arguments"
                        value={argumentsText}
                        onChange={(event) => setArgumentsText(event.target.value)}
                        disabled={busy}
                        rows={4}
                        className="mt-1 block w-full rounded border border-input bg-background px-2 py-1 font-mono"
                      />
                      <span>
                        The proposed arguments remain withheld. This object replaces them.
                      </span>
                    </label>
                  )}
                  {argumentError && <p role="alert">{argumentError}</p>}
                </div>
              )}
              {(!protectedDecision || item.kind === "question") && !decided && (
                <label className="block text-xs">
                  Reason (optional)
                  <input
                    value={reason}
                    onChange={(event) => setReason(event.target.value)}
                    maxLength={4096}
                    disabled={busy}
                    className="mt-1 block w-full rounded border border-input bg-background px-2 py-1"
                  />
                </label>
              )}
            </div>
          </details>
        )}
      </div>
      {!decided && (
        <div className="flex flex-wrap gap-2 border-t border-border px-3.5 py-2.5">{actions}</div>
      )}
      {settle.error && (
        <div className="px-3.5 pb-3">
          <ErrorState error={settle.error} />
        </div>
      )}
      {adoptAfterUpdate && (
        <div className="space-y-2 px-3.5 pb-3 text-xs">
          <p className="text-muted-foreground">
            The application was updated while this approval was waiting. Settling it under the
            update checks the call against the current tool and policy first, and the application's
            execution-profile policy decides.
          </p>
          <Button
            size="sm"
            variant="outline"
            disabled={busy}
            onClick={() =>
              settleItem.mutate(withProfileAdoption(adoptAfterUpdate, crypto.randomUUID()))
            }
          >
            {adoptAfterUpdate.decision === "approve"
              ? "Approve under the update"
              : "Deny under the update"}
          </Button>
        </div>
      )}
    </div>
  )
}

/**
 * The shared manual recovery form: the outcome, the externally verified
 * result or failure reason, and the person's answer again when the call was a question.
 */
function RecoveryForm({
  id,
  question,
  busy,
  onRecover,
}: {
  id: string
  question: string | null
  busy: boolean
  onRecover: (body: Omit<RecoveryRequest, "attention_id">) => void
}) {
  const [outcome, setOutcome] = useState<RecoveryRequest["outcome"]>("completed")
  const [message, setMessage] = useState("")
  const [answer, setAnswer] = useState("")
  const needsAnswer = question !== null
  const ready = !!message.trim() && (!needsAnswer || !!answer.trim())
  const field =
    "h-7 min-w-0 rounded-lg border border-input bg-background px-2.5 text-sm outline-none focus:border-ring disabled:opacity-50"
  return (
    <form
      className="w-full space-y-2"
      onSubmit={(e) => {
        e.preventDefault()
        if (!ready || busy) return
        onRecover({
          outcome,
          message: message.trim(),
          answer: needsAnswer ? answer.trim() : null,
        })
      }}
    >
      <div className="flex flex-wrap items-center gap-2">
        <label htmlFor={`recovery-outcome-${id}`} className="text-xs text-muted-foreground">
          What happened
        </label>
        <select
          id={`recovery-outcome-${id}`}
          value={outcome}
          disabled={busy}
          onChange={(e) => setOutcome(e.target.value as RecoveryRequest["outcome"])}
          className={field}
        >
          <option value="completed">It completed</option>
          <option value="failed">It failed</option>
        </select>
      </div>
      {needsAnswer && (
        <>
          <label htmlFor={`recovery-answer-${id}`} className="sr-only">
            The person's answer
          </label>
          <input
            id={`recovery-answer-${id}`}
            value={answer}
            disabled={busy}
            onChange={(e) => setAnswer(e.target.value)}
            placeholder="The person's answer to the question, again"
            className={cn(field, "w-full")}
          />
        </>
      )}
      <div className="flex gap-2">
        <label htmlFor={`recovery-message-${id}`} className="sr-only">
          Verified result or failure reason
        </label>
        <input
          id={`recovery-message-${id}`}
          value={message}
          disabled={busy}
          onChange={(e) => setMessage(e.target.value)}
          placeholder={
            outcome === "completed"
              ? "The externally verified result"
              : "Why it failed, as verified outside Cayu"
          }
          className={cn(field, "flex-1")}
        />
        <Button size="sm" type="submit" disabled={busy || !ready}>
          Recover
        </Button>
      </div>
    </form>
  )
}

function KnowledgeSource({
  entry,
  error,
  revision,
  shown,
  showEvidence,
}: {
  entry: PendingKnowledgeDetailResponse | undefined
  error: unknown
  revision: number
  /** Whether the host reads proposals' full text. */
  shown: boolean
  showEvidence: boolean
}) {
  if (!shown) return null
  if (error) return <p className="text-xs text-muted-foreground">Source details are unavailable.</p>
  if (!entry) return <p role="status">Loading source details…</p>
  if (entry.revision !== revision)
    return <p role="status">This proposal changed. Refresh the Inbox before deciding.</p>
  return (
    <>
      <h3 className="text-xs font-medium">Full proposal</h3>
      <p className="whitespace-pre-wrap [overflow-wrap:anywhere]">
        {entry.title ?? "Untitled proposal"}
      </p>
      <p className="whitespace-pre-wrap break-words rounded-lg bg-muted px-3 py-2">{entry.text}</p>
      <dl className="grid gap-1 break-words text-xs text-muted-foreground">
        <dt>Namespace</dt>
        <dd>{entry.namespace}</dd>
        <dt>Labels</dt>
        <dd>
          {Object.entries(entry.labels)
            .map(([key, value]) => `${key}=${value}`)
            .join(", ") || "None"}
        </dd>
        <dt>Aspects</dt>
        <dd>{entry.aspects.join(", ") || "None"}</dd>
        <dt>Impact targets</dt>
        <dd>{entry.impact_targets.join(", ") || "None"}</dd>
        <dt>Source type</dt>
        <dd className="[overflow-wrap:anywhere]">{entry.source_type ?? "None"}</dd>
        <dt>Source ID</dt>
        <dd className="[overflow-wrap:anywhere]">{entry.source_id ?? "None"}</dd>
        <dt>Source URI</dt>
        <dd>{entry.source_uri || "None"}</dd>
        <dt>Created by</dt>
        <dd>
          {entry.created_by_type}
          {entry.created_by ? `: ${entry.created_by}` : ""}
        </dd>
      </dl>
      {showEvidence && (
        <details className="min-w-0 text-xs">
          <summary className="cursor-pointer">Proposal evidence</summary>
          <p className="mt-2">
            Visibility: {entry.visibility}. Created {entry.created_at}.
          </p>
          <h4 className="mt-2 font-medium">Chunks</h4>
          <p>
            {entry.chunks.length} loaded / limit {entry.chunk_limit}
          </p>
          {entry.chunks.map((chunk) => (
            <div key={chunk.chunk_id} className="mt-2 min-w-0 rounded-lg border p-2">
              <p className="break-all">
                Chunk {chunk.chunk_index}: {chunk.chunk_id}
              </p>
              <pre className="max-h-48 overflow-auto whitespace-pre-wrap break-words [overflow-wrap:anywhere]">
                {chunk.text}
              </pre>
            </div>
          ))}
          <h4 className="mt-2 font-medium">Metadata</h4>
          <pre className="max-h-72 overflow-auto whitespace-pre-wrap break-words [overflow-wrap:anywhere]">
            {JSON.stringify(entry.metadata, null, 2)}
          </pre>
        </details>
      )}
    </>
  )
}
