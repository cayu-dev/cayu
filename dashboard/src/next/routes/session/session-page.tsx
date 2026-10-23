import { useMutation, useQueryClient } from "@tanstack/react-query"
import { useParams, useSearch } from "@tanstack/react-router"
import {
  ChevronRight,
  Copy,
  Ellipsis,
  FlaskConical,
  GitBranch,
  PanelRight,
  RotateCw,
  TriangleAlert,
} from "lucide-react"
import { useEffect, useMemo, useRef, useState } from "react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import type { ToolWithDisplay } from "../../api/types"
import {
  useInbox,
  useOverview,
  useSessionDetail,
  useSessionRows,
  useSpecialists,
  useTranscript,
} from "../../app/queries"
import { useApi, useNow } from "../../app/services"
import {
  buildBlocks,
  formatDuration,
  formatTokens,
  isActive,
  isWaiting,
} from "../../app/session-model"
import { toolSpanId } from "../../app/trace-model"
import { ErrorState, EvalTrialTag, LabelChips, NavLink, StateBadge, TopBar, useGo } from "../common"
import { AddToEvalsDialog } from "./add-to-evals"
import { Composer } from "./composer"
import { Conversation, SubagentRows } from "./conversation"
import { DetailsPanel, PANEL_TABS, type PanelTab } from "./details-panel"
import { InboxCard } from "./inbox-card"
import { TraceView } from "./trace-view"

export function SessionPage() {
  const { sessionId } = useParams({ strict: false }) as { sessionId: string }
  const search = useSearch({ strict: false }) as {
    panel?: string
    view?: string
    span?: string
    log?: string
    event?: string
  }
  const go = useGo()
  const api = useApi()
  const now = useNow()
  const queryClient = useQueryClient()
  const detail = useSessionDetail(sessionId)
  const overview = useOverview(sessionId)
  const state = overview.data?.state
  const transcript = useTranscript(sessionId, isActive(state))
  const inbox = useInbox({ session_id: sessionId })
  const all = useSessionRows()
  const specialists = useSpecialists()
  const [menuOpen, setMenuOpen] = useState(false)
  const [addingToEvals, setAddingToEvals] = useState(false)
  const endRef = useRef<HTMLDivElement>(null)

  const panel = PANEL_TABS.find((t) => t === search.panel) as PanelTab | undefined
  const trace = search.view === "trace"
  const setSearch = (patch: Record<string, string | undefined>) => {
    const next = { ...search, ...patch }
    for (const k of Object.keys(next))
      if (next[k as keyof typeof next] === undefined) delete next[k as keyof typeof next]
    go(`/sessions/${encodeURIComponent(sessionId)}`, next)
  }
  const setPanel = (tab: PanelTab | undefined) => setSearch({ panel: tab })
  const openTrace = (span?: string) => setSearch({ view: "trace", span, log: undefined })
  const displays = useMemo(() => {
    const agent = specialists.data?.agents.find((a) => a.name === detail.data?.agent_name)
    return Object.fromEntries(
      (agent?.tools ?? []).map((t) => [t.name, (t as ToolWithDisplay).display ?? undefined]),
    )
  }, [specialists.data, detail.data])
  const blocks = useMemo(() => buildBlocks(transcript.data?.messages ?? []), [transcript.data])
  const related = useMemo(() => {
    const rows = all.data?.rows ?? []
    const parentId = detail.data?.parent_session_id
    return {
      parent: parentId ? (rows.find((r) => r.session.id === parentId) ?? null) : null,
      children: rows.filter((r) => r.session.parent_session_id === sessionId),
    }
  }, [all.data, detail.data, sessionId])
  const refresh = () => {
    for (const key of ["overview", "transcript", "sessions", "inbox"])
      void queryClient.invalidateQueries({ queryKey: [key] })
  }
  const reply = useMutation({
    mutationFn: (text: string) => api.reply(sessionId, text),
    onSuccess: refresh,
  })
  const runAgain = useMutation({
    mutationFn: () => api.runAgain(sessionId),
    onSuccess: ({ session_id }) => {
      setMenuOpen(false)
      for (const key of ["sessions", "tasks"])
        void queryClient.invalidateQueries({ queryKey: [key] })
      go(`/sessions/${encodeURIComponent(session_id)}`)
    },
  })
  const [confirmingStop, setConfirmingStop] = useState(false)
  const stop = useMutation({
    mutationFn: (reason: string | undefined) => api.interrupt(sessionId, reason),
    onSuccess: () => {
      setConfirmingStop(false)
      refresh()
    },
  })

  // biome-ignore lint/correctness/useExhaustiveDependencies: scroll when the conversation grows
  useEffect(() => {
    endRef.current?.scrollIntoView({ block: "end" })
  }, [sessionId, blocks.length, inbox.data?.items.length])
  // biome-ignore lint/correctness/useExhaustiveDependencies: close the menus when the session changes
  useEffect(() => {
    setMenuOpen(false)
    setConfirmingStop(false)
  }, [sessionId])

  if (detail.error) {
    return (
      <div className="flex min-w-0 flex-1 flex-col">
        <TopBar>Session</TopBar>
        <div className="p-6">
          <ErrorState error={detail.error} />
        </div>
      </div>
    )
  }

  const o = overview.data
  const title = o?.title ?? sessionId
  const running = state === "running"
  const finished = state === "completed" || state === "failed"
  const seconds = o
    ? ((o.ended_at ? Date.parse(o.ended_at) : now()) - Date.parse(o.started_at)) / 1000
    : null
  const items = inbox.data?.items ?? []

  return (
    <div className="flex min-w-0 flex-1">
      <div className="flex min-w-0 flex-1 flex-col">
        <TopBar
          actions={
            <>
              <div
                className="mr-1 flex rounded-lg border border-border p-0.5"
                role="tablist"
                aria-label="Session view"
              >
                {(
                  [
                    ["conversation", "Conversation"],
                    ["trace", "Trace"],
                  ] as const
                ).map(([value, label]) => (
                  <button
                    key={value}
                    type="button"
                    role="tab"
                    aria-selected={(value === "trace") === trace}
                    onClick={() => setSearch({ view: value === "trace" ? "trace" : undefined })}
                    className={cn(
                      "rounded-md px-2.5 py-0.5 text-[13px]",
                      (value === "trace") === trace
                        ? "bg-muted font-medium"
                        : "text-muted-foreground hover:text-foreground",
                    )}
                  >
                    {label}
                  </button>
                ))}
              </div>
              <Button
                variant="ghost"
                size="icon-sm"
                aria-label="Session details"
                aria-pressed={!!panel}
                className={cn(panel && "bg-muted")}
                onClick={() => setPanel(panel ? undefined : "overview")}
              >
                <PanelRight />
              </Button>
              <div className="relative">
                <Button
                  variant="ghost"
                  size="icon-sm"
                  aria-label="More actions"
                  aria-expanded={menuOpen}
                  onClick={() => setMenuOpen(!menuOpen)}
                >
                  <Ellipsis />
                </Button>
                {menuOpen && (
                  <div className="absolute top-9 right-0 z-20 w-56 rounded-lg border border-border bg-popover p-1 text-sm shadow-lg">
                    <MenuItem
                      icon={RotateCw}
                      label={runAgain.isPending ? "Starting…" : "Run again"}
                      onClick={() => runAgain.mutate()}
                    />
                    <MenuItem
                      icon={Copy}
                      label="Copy session ID"
                      onClick={() => {
                        void navigator.clipboard?.writeText(sessionId).catch(() => undefined)
                        setMenuOpen(false)
                      }}
                    />
                    <MenuItem
                      icon={GitBranch}
                      label="Open trace"
                      onClick={() => {
                        setMenuOpen(false)
                        openTrace()
                      }}
                    />
                    {finished && (
                      <MenuItem
                        icon={FlaskConical}
                        label="Add to evals"
                        onClick={() => {
                          setMenuOpen(false)
                          setAddingToEvals(true)
                        }}
                      />
                    )}
                  </div>
                )}
              </div>
            </>
          }
        >
          <span className="flex min-w-0 items-center gap-1.5">
            <NavLink
              to="/sessions"
              className="shrink-0 font-normal text-muted-foreground hover:text-foreground"
            >
              Sessions
            </NavLink>
            <ChevronRight className="size-3.5 shrink-0 text-muted-foreground" aria-hidden="true" />
            <span className="truncate" title={title}>
              {title}
            </span>
          </span>
        </TopBar>

        {trace && detail.data ? (
          <TraceView
            session={detail.data}
            state={state}
            displays={displays}
            subagents={related.children}
            spanId={search.span}
            log={search.log === "1"}
            onSelect={(span) => setSearch({ span })}
            eventId={search.event}
            onSelectEvent={(event) => setSearch({ event })}
            onShowSpan={(span) => setSearch({ log: undefined, span })}
            onLog={(log) => setSearch({ log: log ? "1" : undefined })}
            toolCallOrder={blocks.flatMap((b) =>
              b.kind === "work" ? b.tools.map((t) => t.id) : [],
            )}
          />
        ) : (
          <>
            <div className="min-h-0 flex-1 overflow-y-auto">
              <div className="mx-auto w-full max-w-3xl px-4 pt-5 pb-6 sm:px-6">
                <div className="mb-5 flex flex-wrap items-center gap-x-3 gap-y-1.5 text-xs text-muted-foreground">
                  <span className="font-medium text-foreground">{detail.data?.agent_name}</span>
                  <StateBadge overview={o} />
                  {o?.eval_trial && <EvalTrialTag />}
                  {seconds != null && (
                    <span className="tabular-nums">
                      {running ? "running " : ""}
                      {formatDuration(seconds)}
                    </span>
                  )}
                  {o && (
                    <span className="tabular-nums">
                      {formatTokens(o.tokens.input + o.tokens.output)} tokens
                    </span>
                  )}
                  {o?.cost && <span className="tabular-nums">${o.cost.amount.toFixed(2)}</span>}
                  {detail.data && <LabelChips labels={detail.data.labels} />}
                </div>

                {transcript.error && <ErrorState error={transcript.error} />}
                {transcript.isPending && (
                  <p className="text-sm text-muted-foreground">Loading conversation…</p>
                )}
                <Conversation
                  blocks={blocks}
                  displays={displays}
                  onTrace={(toolCallId) => openTrace(toolSpanId(toolCallId))}
                  pending={
                    state === "waiting_approval"
                      ? "approval"
                      : state === "needs_recovery"
                        ? "recovery"
                        : state === "running" || state === "waiting_input"
                          ? "running"
                          : "stopped"
                  }
                >
                  <SubagentRows rows={related.children} />
                  {state === "failed" && (
                    <p className="flex items-start gap-2 rounded-lg border border-destructive/30 bg-destructive/5 px-3 py-2 text-sm text-destructive">
                      <TriangleAlert className="mt-0.5 size-4 shrink-0" aria-hidden="true" />
                      {o?.reason ?? "This session failed."}
                    </p>
                  )}
                  {state === "interrupted" && items.length === 0 && (
                    <p className="text-sm text-muted-foreground">
                      This session was stopped. Send a message to continue it.
                    </p>
                  )}
                  {items.map((item) => (
                    <InboxCard key={item.id} item={item} />
                  ))}
                  {running && (
                    <p className="flex items-center gap-2 text-sm text-muted-foreground">
                      <span
                        className="size-2 animate-pulse rounded-full bg-info motion-reduce:animate-none"
                        aria-hidden="true"
                      />
                      Working…
                    </p>
                  )}
                </Conversation>
                <div ref={endRef} />
              </div>
            </div>

            <div className="relative mx-auto w-full max-w-3xl px-4 pb-4 sm:px-6">
              {confirmingStop && running && (
                <StopConfirm
                  childCount={related.children.length}
                  busy={stop.isPending}
                  error={stop.error}
                  onCancel={() => {
                    setConfirmingStop(false)
                    stop.reset()
                  }}
                  onStop={(reason) => stop.mutate(reason)}
                />
              )}
              <Composer
                id="session-composer"
                placeholder={
                  running
                    ? `${detail.data?.agent_name ?? "The specialist"} is working…`
                    : isWaiting(state)
                      ? "Answer the card above first, or send a message"
                      : "Message this session…"
                }
                running={running}
                onStop={() => setConfirmingStop(true)}
                disabled={reply.isPending}
                onSend={(text) => reply.mutate(text)}
              />
              {(reply.error || runAgain.error || (stop.error && !confirmingStop)) && (
                <div className="mt-2">
                  <ErrorState error={reply.error ?? runAgain.error ?? stop.error} />
                </div>
              )}
            </div>
          </>
        )}
      </div>

      {panel && detail.data && (
        <>
          <button
            type="button"
            aria-label="Close details"
            className="fixed inset-0 z-30 bg-black/20 xl:hidden"
            onClick={() => setPanel(undefined)}
          />
          <DetailsPanel
            tab={panel}
            setTab={(t) => setPanel(t)}
            onClose={() => setPanel(undefined)}
            session={detail.data}
            overview={o ?? null}
            parent={related.parent}
            onOpenTrace={() => setSearch({ view: "trace", panel: undefined })}
          />
        </>
      )}
      {addingToEvals && detail.data && (
        <AddToEvalsDialog
          sessionId={sessionId}
          specialist={detail.data.agent_name}
          onClose={() => setAddingToEvals(false)}
        />
      )}
    </div>
  )
}

/**
 * Confirm before stopping, as today's session page does: stopping doesn't undo finished
 * tool calls or cancel linked tasks, and active child sessions may be interrupted too.
 * The optional reason goes with the interrupt.
 */
function StopConfirm({
  childCount,
  busy,
  error,
  onCancel,
  onStop,
}: {
  childCount: number
  busy: boolean
  error: unknown
  onCancel: () => void
  onStop: (reason: string | undefined) => void
}) {
  const [reason, setReason] = useState("")
  return (
    <div
      role="dialog"
      aria-labelledby="stop-confirm-title"
      className="absolute right-4 bottom-full z-20 mb-2 w-80 max-w-[calc(100%-2rem)] rounded-xl border border-border bg-popover p-3 text-sm shadow-lg sm:right-6"
      onKeyDown={(e) => {
        if (e.key === "Escape" && !busy) onCancel()
      }}
    >
      <p id="stop-confirm-title" className="font-medium">
        Stop this session?
      </p>
      <p className="mt-1 text-xs text-muted-foreground">
        It ends as stopped. Tool calls that already finished, and their effects, aren't undone, and
        a task linked to it isn't cancelled.
      </p>
      {childCount > 0 && (
        <p className="mt-2 flex items-start gap-1.5 text-xs text-warning">
          <TriangleAlert className="mt-px size-3.5 shrink-0" aria-hidden="true" />
          {childCount === 1
            ? "It has a child session, which may be interrupted too if still active."
            : `It has ${childCount} child sessions, which may be interrupted too if still active.`}
        </p>
      )}
      <label htmlFor="stop-reason" className="mt-3 block text-xs font-medium">
        Reason <span className="font-normal text-muted-foreground">(optional)</span>
      </label>
      <textarea
        id="stop-reason"
        // biome-ignore lint/a11y/noAutofocus: the popover opens on Stop, so typing the reason comes next
        autoFocus
        rows={2}
        value={reason}
        disabled={busy}
        onChange={(e) => setReason(e.target.value)}
        placeholder="Why stop it?"
        className="mt-1 block w-full resize-none rounded-lg border border-input bg-background px-2.5 py-1.5 text-sm outline-none focus:border-ring"
      />
      {error != null && (
        <div className="mt-2">
          <ErrorState error={error} />
        </div>
      )}
      <div className="mt-3 flex justify-end gap-2">
        <Button size="sm" variant="outline" disabled={busy} onClick={onCancel}>
          Keep running
        </Button>
        <Button
          size="sm"
          variant="destructive"
          disabled={busy}
          onClick={() => onStop(reason.trim() || undefined)}
        >
          {busy ? "Stopping…" : "Stop session"}
        </Button>
      </div>
    </div>
  )
}

function MenuItem({
  icon: Icon,
  label,
  onClick,
}: {
  icon: typeof Copy
  label: string
  onClick: () => void
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left hover:bg-muted"
    >
      <Icon className="size-3.5" aria-hidden="true" />
      {label}
    </button>
  )
}
