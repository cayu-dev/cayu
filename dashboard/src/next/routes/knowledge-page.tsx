/**
 * What the agent knows: entries waiting for review, then the knowledge it has kept,
 * where each entry came from, and how often it is used.
 */
import { useMutation, useQueryClient } from "@tanstack/react-query"
import { Search } from "lucide-react"
import { useState } from "react"
import { Button } from "@/components/ui/button"
import type { KnowledgeEntry } from "../api/types"
import { useInbox, useKnowledge, useSessionRows } from "../app/queries"
import { useApi, useNow } from "../app/services"
import { relativeTime } from "../app/session-model"
import { EmptyState, ErrorState, NavLink, PageBody, sessionPath, TopBar } from "./common"
import { InboxCard } from "./session/inbox-card"

export function KnowledgePage() {
  const knowledge = useKnowledge()
  const inbox = useInbox({ kind: "knowledge" })
  const sessions = useSessionRows()
  const now = useNow()
  const [q, setQ] = useState("")
  const titles = new Map(
    (sessions.data?.rows ?? []).map((r) => [r.session.id, r.overview?.title ?? r.session.id]),
  )
  const pending = inbox.data?.items ?? []
  const needle = q.trim().toLowerCase()
  const entries = (knowledge.data ?? []).filter(
    (k) => !needle || `${k.text} ${k.kind} ${k.agent_name ?? ""}`.toLowerCase().includes(needle),
  )

  return (
    <div className="flex min-w-0 flex-1 flex-col">
      <TopBar>Knowledge</TopBar>
      <PageBody width="max-w-3xl">
        <p className="text-sm text-muted-foreground">
          What this agent knows. Specialists propose entries while they work; a person keeps or
          drops each one before it is used.
        </p>

        {pending.length > 0 && (
          <section className="mt-6 space-y-3" aria-labelledby="knowledge-review">
            <h2 id="knowledge-review" className="text-sm font-medium">
              Waiting for review
            </h2>
            {pending.map((item) => (
              <InboxCard
                key={item.id}
                item={item}
                context={
                  item.session_id
                    ? {
                        sessionId: item.session_id,
                        title: titles.get(item.session_id) ?? item.session_id,
                      }
                    : undefined
                }
              />
            ))}
          </section>
        )}

        <section className="mt-6 space-y-3" aria-labelledby="knowledge-active">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <h2 id="knowledge-active" className="text-sm font-medium">
              Kept{knowledge.data ? ` (${knowledge.data.length})` : ""}
            </h2>
            <label className="flex h-8 w-full max-w-xs items-center gap-2 rounded-lg border border-input bg-background px-2.5 text-sm focus-within:border-ring">
              <Search className="size-4 shrink-0 text-muted-foreground" aria-hidden="true" />
              <span className="sr-only">Search knowledge</span>
              <input
                id="knowledge-search"
                value={q}
                onChange={(e) => setQ(e.target.value)}
                placeholder="Search knowledge"
                className="grouped-field-input w-full min-w-0 bg-transparent outline-none placeholder:text-muted-foreground"
              />
            </label>
          </div>
          {knowledge.error && <ErrorState error={knowledge.error} />}
          {knowledge.isSuccess && entries.length === 0 ? (
            <EmptyState title={q ? "No knowledge matches" : "No knowledge kept yet"}>
              {q ? null : "Entries appear here once someone keeps what a specialist proposed."}
            </EmptyState>
          ) : (
            <div className="divide-y divide-border overflow-hidden rounded-xl border border-border bg-card">
              {entries.map((k) => (
                <Entry
                  key={k.id}
                  entry={k}
                  title={k.source_session_id ? titles.get(k.source_session_id) : undefined}
                  now={now()}
                />
              ))}
            </div>
          )}
        </section>
      </PageBody>
    </div>
  )
}

function Entry({ entry, title, now }: { entry: KnowledgeEntry; title?: string; now: number }) {
  const api = useApi()
  const queryClient = useQueryClient()
  const [confirming, setConfirming] = useState(false)
  const remove = useMutation({
    mutationFn: () => api.removeKnowledge(entry.id),
    onSuccess: () => void queryClient.invalidateQueries({ queryKey: ["knowledge"] }),
  })
  return (
    <div className="space-y-2 px-4 py-3">
      <p className="text-sm leading-relaxed">{entry.text}</p>
      <p className="flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-muted-foreground">
        <span className="rounded-md border border-border px-1.5 py-px">{entry.kind}</span>
        {entry.agent_name && <span>{entry.agent_name}</span>}
        <span>
          kept {entry.kept_by ? `by ${entry.kept_by} ` : ""}
          {relativeTime(Date.parse(entry.kept_at), now)}
        </span>
        <span className="tabular-nums">
          {entry.lookups} {entry.lookups === 1 ? "lookup" : "lookups"}
          {entry.last_used_at ? `, last ${relativeTime(Date.parse(entry.last_used_at), now)}` : ""}
        </span>
        {entry.source_session_id && (
          <NavLink
            to={sessionPath(entry.source_session_id)}
            className="truncate hover:text-foreground hover:underline"
          >
            from {title ?? entry.source_session_id}
          </NavLink>
        )}
        <span className="ml-auto flex items-center gap-1">
          {confirming ? (
            <>
              <span>Remove this entry?</span>
              <Button
                size="xs"
                variant="destructive"
                disabled={remove.isPending}
                onClick={() => remove.mutate()}
              >
                Remove
              </Button>
              <Button size="xs" variant="ghost" onClick={() => setConfirming(false)}>
                Cancel
              </Button>
            </>
          ) : (
            <Button size="xs" variant="ghost" onClick={() => setConfirming(true)}>
              Remove
            </Button>
          )}
        </span>
      </p>
      {remove.error && <ErrorState error={remove.error} />}
    </div>
  )
}
