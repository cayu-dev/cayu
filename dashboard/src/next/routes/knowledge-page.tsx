/** Knowledge review and scoped kept entries share the runtime's existing owners. */
import { useMutation, useQueryClient } from "@tanstack/react-query"
import { Search } from "lucide-react"
import { useState } from "react"
import { useViewer } from "@/components/dashboard/viewer"
import { Button } from "@/components/ui/button"
import { ROLE_FORBIDDEN_REASON } from "@/lib/dashboard-access"
import type { ApiKeptKnowledgeItem } from "@/lib/generated/server-api"
import { useKeptKnowledge } from "../app/knowledge-query"
import { useInbox } from "../app/queries"
import { useApi, useNow, useServices } from "../app/services"
import { relativeTime } from "../app/session-model"
import { EmptyState, ErrorState, NavLink, PageBody, sessionPath, TopBar } from "./common"
import { InboxCard } from "./session/inbox-card"

export function KnowledgePage({
  keptUnavailable = null,
  reviewUnavailable = null,
}: {
  keptUnavailable?: string | null
  reviewUnavailable?: string | null
} = {}) {
  const knowledge = useKeptKnowledge(keptUnavailable === null)
  const [q, setQ] = useState("")
  const needle = q.trim().toLowerCase()
  const entries = knowledge.entries.filter(
    (entry) =>
      !needle ||
      `${entry.text_preview} ${entry.kind} ${entry.specialist ?? ""}`
        .toLowerCase()
        .includes(needle),
  )
  return (
    <div className="flex min-h-0 min-w-0 flex-1 flex-col">
      <TopBar>Knowledge</TopBar>
      <PageBody width="max-w-3xl">
        <p className="text-sm text-muted-foreground">
          What this agent knows. Review proposals, then inspect the knowledge available to
          specialists.
        </p>
        {reviewUnavailable ? (
          <p role="status" className="mt-6">
            Review is unavailable. {reviewUnavailable}
          </p>
        ) : (
          <Reviews />
        )}
        <section className="mt-6 space-y-3" aria-labelledby="knowledge-active">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <h2 id="knowledge-active" className="text-sm font-medium">
              Kept
            </h2>
            <label className="flex h-8 w-full max-w-xs items-center gap-2 rounded-lg border border-input bg-background px-2.5 text-sm focus-within:border-ring">
              <Search className="size-4 shrink-0 text-muted-foreground" aria-hidden="true" />
              <span className="sr-only">Search loaded knowledge</span>
              <input
                id="knowledge-search"
                value={q}
                onChange={(event) => setQ(event.target.value)}
                placeholder="Search loaded knowledge"
                className="grouped-field-input w-full min-w-0 bg-transparent outline-none placeholder:text-muted-foreground"
              />
            </label>
          </div>
          {keptUnavailable ? (
            <p role="status">Kept knowledge is unavailable. {keptUnavailable}</p>
          ) : (
            <>
              {knowledge.isPending && (
                <p role="status" className="text-sm text-muted-foreground">
                  Loading kept knowledge…
                </p>
              )}
              {knowledge.error && <ErrorState error={knowledge.error} />}
              {knowledge.data && (
                <p className="text-xs text-muted-foreground">
                  {knowledge.entries.length} loaded
                  {knowledge.hasNextPage ? "; more entries may be available" : ""}. Text is a public
                  preview.
                </p>
              )}
              <p className="text-xs text-muted-foreground">
                Only the current page is displayed and searched. Loading another page replaces these
                entries.
                {knowledge.historical
                  ? " Viewing an older page. Automatic refresh is paused; Refresh knowledge returns to the newest page."
                  : " The newest page refreshes automatically."}
              </p>
              {knowledge.usageUnavailable && (
                <p className="text-xs text-muted-foreground">
                  Usage is unavailable for this read. Unknown usage is not zero.
                </p>
              )}
              {entries.length === 0 &&
                knowledge.data &&
                !knowledge.error &&
                !knowledge.isFetching &&
                (knowledge.complete && !knowledge.historical && !q ? (
                  <EmptyState title="No knowledge kept yet">
                    Entries appear here after they become active.
                  </EmptyState>
                ) : (
                  <p role="status">
                    {q
                      ? "No loaded entries match this search."
                      : knowledge.hasNextPage
                        ? "No visible entries on this page. Continue loading to check more."
                        : "No visible entries on this page. Refresh knowledge to return to the newest page."}
                  </p>
                ))}
              {entries.length > 0 && (
                <div className="divide-y divide-border overflow-hidden rounded-xl border border-border bg-card">
                  {entries.map((entry) => (
                    <Entry key={JSON.stringify([entry.namespace, entry.entry_id])} entry={entry} />
                  ))}
                </div>
              )}
              <div className="flex flex-wrap gap-2">
                {knowledge.hasNextPage && (
                  <Button
                    variant="outline"
                    disabled={knowledge.isFetching}
                    onClick={() => void knowledge.fetchNextPage()}
                  >
                    {knowledge.isFetching ? "Loading…" : "Load more knowledge"}
                  </Button>
                )}
                <Button
                  variant="outline"
                  disabled={knowledge.isFetching}
                  onClick={() => void knowledge.refetch()}
                >
                  Refresh knowledge
                </Button>
              </div>
            </>
          )}
        </section>
      </PageBody>
    </div>
  )
}

function Reviews() {
  const inbox = useInbox({ kind: "knowledge" })
  return (
    <section className="mt-6 space-y-3" aria-labelledby="knowledge-review">
      <h2 id="knowledge-review" className="text-sm font-medium">
        Waiting for review
      </h2>
      <p className="text-xs text-muted-foreground">
        Showing the current review page. Load more replaces this page.
      </p>
      {inbox.historical && (
        <p className="text-xs text-muted-foreground">
          Viewing older proposals. Automatic refresh is paused.
        </p>
      )}
      <Button variant="outline" disabled={inbox.isFetching} onClick={() => void inbox.refetch()}>
        Refresh proposals
      </Button>
      {inbox.isPending && (
        <p role="status" className="text-sm text-muted-foreground">
          Loading knowledge proposals…
        </p>
      )}
      {inbox.error && <ErrorState error={inbox.error} />}
      {inbox.data?.issues.map((issue) => (
        <p role="status" key={JSON.stringify(issue)} className="break-words text-sm text-warning">
          {issue.detail}{" "}
          {issue.session_id ? (
            <NavLink to={sessionPath(issue.session_id)}>Inspect session {issue.session_id}</NavLink>
          ) : null}
        </p>
      ))}
      {inbox.data?.items.map((item) => (
        <InboxCard
          key={item.id}
          item={item}
          showKnowledgeEvidence
          context={
            item.session_id ? { sessionId: item.session_id, title: item.session_id } : undefined
          }
        />
      ))}
      {inbox.isSuccess &&
        !inbox.isFetching &&
        !inbox.data?.items.length &&
        !inbox.data?.has_more &&
        !inbox.data?.issues.length && (
          <p className="text-sm text-muted-foreground">
            {inbox.historical
              ? "No actions on this older page. Refresh to check the newest proposals."
              : "No proposals waiting for review."}
          </p>
        )}
      {inbox.hasNextPage && (
        <Button
          variant="outline"
          disabled={inbox.isFetching}
          onClick={() => void inbox.fetchNextPage()}
        >
          Load more proposals
        </Button>
      )}
    </section>
  )
}

function Entry({ entry }: { entry: ApiKeptKnowledgeItem }) {
  const api = useApi()
  const client = useQueryClient()
  const now = useNow()
  const viewer = useViewer()
  const { prototype } = useServices()
  const canRemove = !!prototype || viewer?.full_access === true
  // Capture the observed revision at confirmation. Polling must not silently replace intent.
  const [revision, setRevision] = useState<number | null>(null)
  const remove = useMutation({
    mutationFn: (expected: number) => api.removeKnowledge(entry.entry_id, expected),
    retry: false,
    onSuccess: () => {
      setRevision(null)
      void client.invalidateQueries({ queryKey: ["knowledge"] })
      void client.invalidateQueries({ queryKey: ["inbox-counts"] })
    },
  })
  return (
    <article className="min-w-0 space-y-2 px-4 py-3" aria-label={`Knowledge ${entry.entry_id}`}>
      <p className="whitespace-pre-wrap break-words text-sm leading-relaxed [overflow-wrap:anywhere]">
        {entry.text_preview}
      </p>
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1 break-words text-xs text-muted-foreground [overflow-wrap:anywhere]">
        <span className="rounded-md border border-border px-1.5 py-px">{entry.kind}</span>
        {entry.specialist && <span>{entry.specialist}</span>}
        <span>
          {entry.kept_at
            ? `Kept${entry.kept_by ? ` by ${entry.kept_by}` : ""} ${relativeTime(Date.parse(entry.kept_at), now())}`
            : "Kept attribution unavailable"}
        </span>
        <span className="tabular-nums">
          {entry.lookups === null
            ? "Usage unavailable"
            : `${entry.lookups} ${entry.lookups === "1" ? "lookup" : "lookups"}`}
          {entry.last_used_at
            ? `, last ${relativeTime(Date.parse(entry.last_used_at), now())}`
            : ""}
        </span>
        {entry.source_session_id && (
          <NavLink
            to={sessionPath(entry.source_session_id)}
            className="min-w-0 break-all hover:underline"
          >
            from {entry.source_session_id}
          </NavLink>
        )}
      </div>
      <details className="text-xs text-muted-foreground">
        <summary className="cursor-pointer">Entry details</summary>
        <dl className="mt-2 grid gap-1 break-all">
          <dt>Entry</dt>
          <dd>{entry.entry_id}</dd>
          <dt>Namespace</dt>
          <dd>{entry.namespace}</dd>
          <dt>Revision</dt>
          <dd>{entry.revision}</dd>
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
          <dd>{entry.source_uri ?? "None"}</dd>
          <dt>Created by</dt>
          <dd>
            {entry.created_by_type}
            {entry.created_by ? `: ${entry.created_by}` : ""}
          </dd>
        </dl>
      </details>
      {canRemove ? (
        <div className="flex flex-wrap items-center gap-2">
          {revision === null ? (
            <Button
              size="xs"
              variant="ghost"
              onClick={() => {
                remove.reset()
                setRevision(entry.revision)
              }}
            >
              Remove
            </Button>
          ) : (
            <>
              <span className="text-xs">Remove this entry from active knowledge?</span>
              <Button
                size="xs"
                variant="destructive"
                disabled={remove.isPending || !!remove.error}
                onClick={() => remove.mutate(revision)}
              >
                Confirm remove
              </Button>
              <Button
                size="xs"
                variant="ghost"
                disabled={remove.isPending}
                onClick={() => {
                  setRevision(null)
                  remove.reset()
                }}
              >
                Cancel
              </Button>
            </>
          )}
        </div>
      ) : (
        <p className="text-xs text-muted-foreground">{ROLE_FORBIDDEN_REASON}</p>
      )}
      {remove.error && (
        <>
          <ErrorState error={remove.error} />
          <p role="status" className="text-xs">
            Removal was not confirmed. Refresh knowledge to inspect the current entry before
            starting a new removal.
          </p>
        </>
      )}
    </article>
  )
}
