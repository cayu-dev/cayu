/**
 * The Inbox's items for some filters: a page at a time, with the sources the server
 * couldn't read, its empty states, and an `InboxCard` per item titled by its session.
 */

import { Button } from "@kit/ui"
import { useQuery } from "@tanstack/react-query"
import { type InboxSearch, useInboxHost } from "../host"
import { EmptyState, ErrorState } from "../states"
import { InboxCard } from "./inbox-card"
import { inboxKeys } from "./keys"
import { useInbox, useInboxCounts } from "./queries"

export function InboxList({ search, pageSize = 50 }: { search: InboxSearch; pageSize?: number }) {
  const {
    Link,
    inbox: { api },
  } = useInboxHost()
  const inbox = useInbox({
    limit: pageSize,
    q: search.q,
    kind: search.kind,
    session_id: search.session_id,
    agent_name: search.agent_name,
  })
  const items = inbox.data?.items ?? []
  // Items carry their session's title; ask the host only for sessions they don't name.
  const ids = [
    ...new Set(
      items.flatMap((item) => (item.session_id && !item.session_title ? [item.session_id] : [])),
    ),
  ]
  const sessions = useQuery({
    queryKey: inboxKeys.sessionTitles(ids),
    queryFn: ({ signal }) => api.sessionTitles!(ids, signal),
    enabled: ids.length > 0 && !!api.sessionTitles,
    // Titles change with the listed items (a new key), not on a timer.
    staleTime: 60_000,
  })
  const titles = new Map(
    (sessions.data ?? []).map((row) => [row.session_id, row.title ?? row.session_id]),
  )
  const all = useInboxCounts({
    q: search.q,
    session_id: search.session_id,
    agent_name: search.agent_name,
  })

  return (
    <div className="mt-4 space-y-3">
      {inbox.isPending && (
        <p role="status" className="text-sm text-muted-foreground">
          Loading inbox…
        </p>
      )}
      {inbox.error && <ErrorState error={inbox.error} />}
      {all.error && <ErrorState error={all.error} />}
      {all.data && all.data.complete !== true && (
        <p role="status" className="text-sm text-muted-foreground">
          Counts unavailable:{" "}
          {all.data.unavailable_reason === "access_limited"
            ? "your access is limited"
            : all.data.unavailable_reason === "source_incomplete"
              ? "a source is incomplete"
              : "the server could not establish a complete total"}
          .
        </p>
      )}
      {inbox.data?.issues?.map((issue) => (
        <p
          key={`${issue.code}-${issue.session_id ?? ""}-${issue.detail}`}
          role="status"
          className="text-sm text-warning [overflow-wrap:anywhere]"
        >
          {issue.detail}
          {issue.session_id && (
            <Link to={{ session: issue.session_id }} className="ml-2 underline">
              Inspect session {issue.session_id}
            </Link>
          )}
        </p>
      ))}
      {inbox.isSuccess && !inbox.isFetching && items.length === 0 && (
        <EmptyState
          title={
            inbox.data?.issues.length
              ? "Some action sources are unavailable"
              : inbox.hasNextPage
                ? "No visible items on this page"
                : inbox.historical
                  ? "No actions on this older page. Refresh to check newest actions."
                  : "Nothing is waiting for you"
          }
        >
          Approvals, questions, recovery and knowledge to review show up here as sessions need them.
        </EmptyState>
      )}
      {items.map((item) => (
        <InboxCard
          key={item.id}
          item={item}
          context={
            item.session_id
              ? {
                  sessionId: item.session_id,
                  title: item.session_title ?? titles.get(item.session_id) ?? item.session_id,
                }
              : undefined
          }
        />
      ))}
      <p className="text-xs text-muted-foreground">
        Showing the current action page. Load more replaces it; older pages pause automatic refresh.
      </p>
      <div className="flex flex-wrap gap-2">
        <Button variant="outline" disabled={inbox.isFetching} onClick={() => void inbox.refetch()}>
          Refresh actions
        </Button>
        {inbox.hasNextPage && (
          <Button
            variant="outline"
            disabled={inbox.isFetching}
            onClick={() => void inbox.fetchNextPage()}
          >
            {items.length === 0 ? "Continue scanning" : "Load more"}
          </Button>
        )}
      </div>
    </div>
  )
}
