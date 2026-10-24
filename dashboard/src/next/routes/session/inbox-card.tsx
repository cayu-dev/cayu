/**
 * One inbox item with its decision buttons, used inline in a session and on the Inbox
 * page. Deciding settles the same record everywhere (Cayu Cloud and the app included).
 * Recovery is the exception: the server only takes it on the session page, with the
 * evidence of what happened, so on the Inbox page a recovery item links to its session.
 */
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { CircleHelp, Lightbulb, RotateCcw, ShieldCheck } from "lucide-react"
import { type ReactNode, useState } from "react"
import { ProtectedHumanReview } from "@/components/dashboard/protected-human-review"
import { useViewer } from "@/components/dashboard/viewer"
import { Button } from "@/components/ui/button"
import { fetchPendingKnowledgeEntry, type KnowledgeEntryDetail } from "@/lib/api"
import { dashboardConfig } from "@/lib/config"
import { cn } from "@/lib/utils"
import type { InboxItem, InboxSettleRequest, RecoveryRequest } from "../../api/types"
import { useApi, useNow, useServices } from "../../app/services"
import { relativeTime } from "../../app/session-model"
import { ErrorState, NavLink, sessionPath } from "../common"

const KIND = {
  approval: { icon: ShieldCheck, label: "Needs your approval" },
  question: { icon: CircleHelp, label: "Needs your answer" },
  recovery: { icon: RotateCcw, label: "Needs recovery" },
  knowledge: { icon: Lightbulb, label: "Knowledge to review" },
} as const

export function InboxCard({
  item,
  context,
}: {
  item: InboxItem
  /** On the Inbox page, the session the item belongs to. */
  context?: { title: string; sessionId: string }
}) {
  const api = useApi()
  const { prototype, decisionsEnabled, knowledgeReviewEnabled } = useServices()
  const available =
    !!prototype ||
    (item.kind === "knowledge" ? knowledgeReviewEnabled === true : decisionsEnabled === true)
  const knowledgeSource = useQuery({
    queryKey: ["knowledge-source", item.knowledge?.entry_id, item.knowledge?.revision],
    queryFn: () => fetchPendingKnowledgeEntry(item.knowledge!.entry_id),
    enabled: !prototype && item.kind === "knowledge" && !!item.knowledge,
  })
  const knowledgeReady =
    !!prototype ||
    (!!knowledgeSource.data &&
      !knowledgeSource.error &&
      knowledgeSource.data.revision === item.knowledge?.revision)
  const now = useNow()
  const queryClient = useQueryClient()
  const [answer, setAnswer] = useState("")
  const [reason, setReason] = useState("")
  const [claimTask, setClaimTask] = useState(false)
  const [override, setOverride] = useState(false)
  const [argumentsText, setArgumentsText] = useState("{}")
  const [argumentError, setArgumentError] = useState<string | null>(null)
  const viewer = useViewer()
  const refresh = () => {
    for (const key of [
      "inbox",
      "inbox-counts",
      "child-actions",
      "home",
      "sessions",
      "session-list",
      "overview",
      "transcript",
      "knowledge",
      "tool-usage",
    ])
      void queryClient.invalidateQueries({ queryKey: [key] })
  }
  const settleItem = useMutation({
    mutationFn: (body: InboxSettleRequest) => api.settle(item.id, body),
    onSuccess: refresh,
  })
  const recover = useMutation({
    mutationFn: (body: RecoveryRequest) => {
      if (!item.session_id) throw new Error("This recovery item names no session.")
      return api.recover(item.session_id, body)
    },
    onSuccess: refresh,
  })
  // Whichever action this card took: settling, or recovering on the session page.
  const settle = item.kind === "recovery" ? recover : settleItem
  const decide = (
    decision: InboxSettleRequest["decision"],
    extra: Partial<InboxSettleRequest> = {},
  ) => {
    if (!available || !item.decisions.includes(decision)) return
    let argumentsValue: Record<string, unknown> | undefined
    if (decision === "approve" && override) {
      if (!prototype && viewer?.full_access !== true) return
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
  const { icon: Icon, label } = KIND[item.kind]
  const busy =
    !available ||
    settle.isPending ||
    settle.isSuccess ||
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
          disabled={busy || !item.decisions.includes("approve")}
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
          <NavLink
            to={sessionPath(item.session_id)}
            className="inline-flex h-7 items-center rounded-lg border border-border px-2.5 text-[13px] font-medium hover:bg-muted"
          >
            Recover in the session
          </NavLink>
        ) : (
          <p className="text-xs text-muted-foreground">
            This item names no session, so it can't be recovered here.
          </p>
        )
      ) : (
        <RecoveryForm
          id={item.id}
          question={item.question ?? null}
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
    dashboardConfig.humanReviewPurpose &&
    item.session_id &&
    (item.kind === "approval" || item.kind === "question")
  if (protectedDecision) {
    actions = (
      <ProtectedHumanReview
        sessionId={item.session_id!}
        purpose={dashboardConfig.humanReviewPurpose!}
        kind={item.kind === "approval" ? "tool_approval" : "user_input"}
        disabled={busy || item.decisions.length === 0}
        unavailableReason={null}
        onDecision={async (decision) => {
          const body: InboxSettleRequest =
            "answer" in decision
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
        settle.isSuccess ? "border-border opacity-70" : "border-warning/50 shadow-xs",
      )}
    >
      <div className="flex items-center gap-2 border-b border-border px-3.5 py-2 text-xs">
        <Icon className="size-3.5 text-warning" aria-hidden="true" />
        <span className="font-medium text-warning">{settle.isSuccess ? "Decided" : label}</span>
        {context && (
          <>
            <span className="text-muted-foreground">·</span>
            <NavLink
              to={sessionPath(context.sessionId)}
              className="min-w-0 truncate text-muted-foreground hover:text-foreground hover:underline"
            >
              {context.title}
            </NavLink>
          </>
        )}
        <span className="ml-auto shrink-0 text-muted-foreground">
          {relativeTime(Date.parse(item.created_at), now())}
        </span>
      </div>
      <div className="space-y-2 px-3.5 py-3 text-sm">
        {body}
        {!available && item.kind !== "recovery" && (
          <p role="status">Decisions are unavailable for this deployment.</p>
        )}
        {item.kind === "knowledge" && item.knowledge && (
          <KnowledgeSource
            entry={knowledgeSource.data}
            error={knowledgeSource.error}
            revision={item.knowledge.revision}
            prototype={!!prototype}
          />
        )}
        {(item.kind === "approval" || item.kind === "question") && !settle.isSuccess && (
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
        {item.arguments_unavailable_reason === "quarantined" && (
          <p className="text-xs text-muted-foreground">
            Arguments are withheld. Use protected human review when this agent provides it.
          </p>
        )}
        {item.kind === "approval" &&
          !protectedDecision &&
          (!!prototype || viewer?.full_access === true) &&
          !settle.isSuccess && (
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
                  <span>The proposed arguments remain withheld. This object replaces them.</span>
                </label>
              )}
              {argumentError && <p role="alert">{argumentError}</p>}
            </div>
          )}
        {item.kind !== "recovery" &&
          (!protectedDecision || item.kind === "question") &&
          !settle.isSuccess && (
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
      {!settle.isSuccess && (
        <div className="flex flex-wrap gap-2 border-t border-border px-3.5 py-2.5">{actions}</div>
      )}
      {settle.error && (
        <div className="px-3.5 pb-3">
          <ErrorState error={settle.error} />
        </div>
      )}
    </div>
  )
}

/**
 * Today's manual recovery form (session-detail.tsx): the outcome, the externally verified
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
  prototype,
}: {
  entry: KnowledgeEntryDetail | undefined
  error: unknown
  revision: number
  prototype: boolean
}) {
  if (prototype) return null
  if (error) return <p className="text-xs text-muted-foreground">Source details are unavailable.</p>
  if (!entry) return <p role="status">Loading source details…</p>
  if (entry.revision !== revision)
    return <p role="status">This proposal changed. Refresh the Inbox before deciding.</p>
  return (
    <>
      <h3 className="text-xs font-medium">Full proposal</h3>
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
        <dt>Source URI</dt>
        <dd>{entry.source_uri || "None"}</dd>
        <dt>Created by</dt>
        <dd>
          {entry.created_by_type}
          {entry.created_by ? `: ${entry.created_by}` : ""}
        </dd>
      </dl>
    </>
  )
}
