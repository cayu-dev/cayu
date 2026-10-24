import { useQuery } from "@tanstack/react-query"
import { useState } from "react"
import { fetchPendingActions } from "@/lib/api"
import { ErrorState, NavLink, sessionPath } from "./common"

/** Delegated actions are inspected in their session; they are not Inbox settlements. */
export function ChildActions({ sessionId, agentName }: { sessionId?: string; agentName?: string }) {
  const [search, setSearch] = useState("")
  const [cursor, setCursor] = useState<string | undefined>()
  const query = useQuery({
    queryKey: ["child-actions", search, cursor, sessionId, agentName],
    queryFn: () =>
      fetchPendingActions({
        kind: "delegated_action",
        session_id: sessionId,
        agent_name: agentName,
        q: search || undefined,
        limit: 25,
        cursor,
      }),
    retry: false,
    refetchInterval: 5000,
  })
  return (
    <section className="mt-8 space-y-3 border-t border-border pt-4">
      <h2 className="text-sm font-medium">Child actions</h2>
      <p className="text-xs text-muted-foreground">
        Delegated actions are inspected in their session.
      </p>
      <label className="block text-sm">
        Search child actions
        <input
          className="mt-1 block h-8 w-full rounded-lg border border-input bg-background px-2.5 text-sm"
          value={search}
          onChange={(event) => {
            setSearch(event.target.value)
            setCursor(undefined)
          }}
        />
      </label>
      {query.error && <ErrorState error={query.error} />}
      {query.data?.issues?.map((issue) => (
        <p
          key={`${issue.code}-${issue.session_id ?? ""}-${issue.detail}`}
          role="status"
          className="[overflow-wrap:anywhere]"
        >
          {issue.detail}
          {issue.session_id && (
            <NavLink to={sessionPath(issue.session_id)} className="ml-2 underline">
              Inspect session {issue.session_id}
            </NavLink>
          )}
        </p>
      ))}
      {query.data?.actions.map((action) => (
        <NavLink
          key={`${action.session.id}-${action.attention_id}-${action.tool_call_id}`}
          to={sessionPath(action.delegated_action?.child_session_id ?? action.session.id)}
          className="block [overflow-wrap:anywhere] rounded border border-border p-3 text-sm hover:bg-muted/60"
        >
          {action.session.id} · {action.tool_name ?? action.kind} · Open child
        </NavLink>
      ))}
      {query.isSuccess && query.data.actions.length === 0 && (
        <p className="text-sm text-muted-foreground">No child actions on this page.</p>
      )}
      <div className="flex gap-2">
        {cursor && (
          <button type="button" onClick={() => setCursor(undefined)}>
            First page
          </button>
        )}
        {query.data?.next_cursor && (
          <button type="button" onClick={() => setCursor(query.data?.next_cursor ?? undefined)}>
            Next child actions
          </button>
        )}
      </div>
    </section>
  )
}
