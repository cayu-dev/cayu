/**
 * One inbox item with its decision buttons, used inline in a session and on the Inbox
 * page. Deciding settles the same record everywhere (Cayu Cloud and the app included).
 * Recovery is the exception: the server only takes it on the session page, with the
 * evidence of what happened, so on the Inbox page a recovery item links to its session.
 */
import { useMutation, useQueryClient } from "@tanstack/react-query"
import { CircleHelp, Lightbulb, RotateCcw, ShieldCheck } from "lucide-react"
import { type ReactNode, useState } from "react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import type { InboxItem, InboxSettleRequest, RecoveryRequest } from "../../api/types"
import { useApi, useNow } from "../../app/services"
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
  const now = useNow()
  const queryClient = useQueryClient()
  const [answer, setAnswer] = useState("")
  const refresh = () => {
    for (const key of ["inbox", "sessions", "overview", "transcript", "knowledge", "tool-usage"])
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
  ) => settleItem.mutate({ decision, ...extra })
  const { icon: Icon, label } = KIND[item.kind]
  const busy = settle.isPending || settle.isSuccess

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
        <Button size="sm" disabled={busy} onClick={() => decide("approve")}>
          Approve
        </Button>
        <Button size="sm" variant="outline" disabled={busy} onClick={() => decide("deny")}>
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
        <Button size="sm" disabled={busy} onClick={() => decide("keep")}>
          Keep
        </Button>
        <Button size="sm" variant="outline" disabled={busy} onClick={() => decide("drop")}>
          Drop
        </Button>
      </>
    )
  }

  return (
    <div
      className={cn(
        "rounded-xl border bg-card",
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
      <div className="space-y-2 px-3.5 py-3 text-sm">{body}</div>
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
