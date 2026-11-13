import { useSearch } from "@tanstack/react-router"
import { InboxPart, type InboxSearch } from "../../parts/inbox"
import { ChildActions } from "./child-actions"
import { PageBody, TopBar, useGo } from "./common"

/** The Inbox page: the shared Inbox part (src/parts) under the dashboard's top bar. */
export function InboxPage() {
  const search = useSearch({ strict: false }) as InboxSearch
  const go = useGo()
  return (
    <div className="flex min-h-0 min-w-0 flex-1 flex-col">
      <TopBar>Inbox</TopBar>
      <PageBody width="max-w-3xl">
        <InboxPart
          search={search}
          onSearch={(next) => go({ to: "/inbox", search: next })}
          description={
            <p className="text-sm text-muted-foreground">
              Everything in this agent waiting on a person. Deciding here settles the same record in
              Cayu Cloud and in the agent's own UI.
            </p>
          }
          footer={
            <ChildActions
              key={`${search.session_id}-${search.agent_name}`}
              sessionId={search.session_id}
              agentName={search.agent_name}
            />
          }
        />
      </PageBody>
    </div>
  )
}
