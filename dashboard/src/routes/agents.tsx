import { useQuery } from "@tanstack/react-query"
import { Bot, Brain, Search, Wrench } from "lucide-react"
import { type ReactNode, useEffect, useMemo, useState } from "react"
import {
  DataCard,
  Page,
  PageHeader,
  PayloadViewer,
  StateMessage,
} from "../components/dashboard/layout"
import { Badge } from "../components/ui/badge"
import { Button, buttonVariants } from "../components/ui/button"
import { Input } from "../components/ui/input"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "../components/ui/table"
import {
  type AgentSummary,
  fetchAgents,
  fetchSessionsSummary,
  fetchTasks,
  type SessionsSummary,
  type Task,
  type ToolSummary,
} from "../lib/api"
import { formatCount, formatDateTime } from "../lib/format"
import { currentQueryParam, dashboardPath, replaceDashboardLocation } from "../lib/links"
import { cn } from "../lib/utils"

function useDebouncedValue<T>(value: T, delayMs: number): T {
  const [debounced, setDebounced] = useState(value)
  useEffect(() => {
    const handle = window.setTimeout(() => setDebounced(value), delayMs)
    return () => window.clearTimeout(handle)
  }, [delayMs, value])
  return debounced
}

function agentMatches(agent: AgentSummary, query: string) {
  const needle = query.trim().toLowerCase()
  if (!needle) return true
  return [
    agent.name,
    agent.model,
    agent.provider_name,
    ...agent.tools.map((tool) => tool.name),
    ...agent.tools.map((tool) => tool.effect),
  ].some((value) => value?.toLowerCase().includes(needle))
}

function countExternalTools(agent: AgentSummary) {
  return agent.tools.filter((tool) => tool.effect !== "read_only").length
}

function DetailRow({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="grid grid-cols-[7rem_minmax(0,1fr)] gap-3 text-sm">
      <div className="text-muted-foreground">{label}</div>
      <div className="min-w-0 break-words">{children}</div>
    </div>
  )
}

function ToolRow({ tool }: { tool: ToolSummary }) {
  return (
    <div className="rounded-md border border-border p-3">
      <div className="flex min-w-0 flex-wrap items-center gap-2">
        <span className="font-mono text-sm font-medium">{tool.name}</span>
        <Badge variant={tool.effect === "read_only" ? "secondary" : "outline"}>{tool.effect}</Badge>
        <Badge variant="outline">{tool.parallel_safe ? "parallel" : "serialized"}</Badge>
      </div>
      <p className="mt-2 text-sm text-muted-foreground">{tool.description || "No description."}</p>
      <PayloadViewer value={tool.input_schema} className="mt-3" maxHeight="max-h-52" />
    </div>
  )
}

type SessionSummaryItem = SessionsSummary["sessions"][number]

function RecentSessionLink({ item }: { item: SessionSummaryItem }) {
  return (
    <a
      href={dashboardPath(`/sessions/${encodeURIComponent(item.session.id)}`)}
      className="block rounded-md border border-border p-3 transition-colors hover:bg-muted/40"
    >
      <div className="flex min-w-0 items-center justify-between gap-3">
        <span className="truncate font-mono text-xs font-medium">{item.session.id}</span>
        <Badge variant="outline">{item.session.status}</Badge>
      </div>
      <div className="mt-1 flex min-w-0 flex-wrap gap-x-3 gap-y-1 text-xs text-muted-foreground">
        <span>{formatDateTime(item.session.updated_at)}</span>
        <span>{formatCount(item.events.total_events)} events</span>
        {item.events.latest_event && <span>{item.events.latest_event.type}</span>}
      </div>
    </a>
  )
}

function TaskLink({ task }: { task: Task }) {
  return (
    <a
      href={dashboardPath("/tasks", { q: task.id })}
      className="block rounded-md border border-border p-3 transition-colors hover:bg-muted/40"
    >
      <div className="flex min-w-0 items-center justify-between gap-3">
        <span className="truncate text-sm font-medium">{task.title || task.type}</span>
        <Badge variant="outline">{task.status}</Badge>
      </div>
      <div className="mt-1 truncate font-mono text-xs text-muted-foreground">{task.id}</div>
    </a>
  )
}

function AgentDetail({
  agent,
  relatedSessions,
  relatedSessionsLoading,
  relatedTasks,
  relatedTasksLoading,
}: {
  agent: AgentSummary | null
  relatedSessions: SessionsSummary | undefined
  relatedSessionsLoading: boolean
  relatedTasks: Task[] | undefined
  relatedTasksLoading: boolean
}) {
  if (agent === null) {
    return (
      <StateMessage className="py-16">Select an agent to inspect its runtime shape.</StateMessage>
    )
  }

  return (
    <div className="space-y-5 p-4 sm:p-5">
      <div>
        <div className="flex min-w-0 flex-wrap items-center gap-2">
          <Badge variant="secondary">{agent.provider_name ?? "default provider"}</Badge>
          <Badge variant="outline">{agent.model}</Badge>
          {agent.has_system_prompt && <Badge variant="outline">system prompt</Badge>}
          {agent.thinking && <Badge variant="outline">thinking</Badge>}
        </div>
        <h2 className="mt-3 truncate text-xl font-semibold">{agent.name}</h2>
      </div>

      <div className="grid gap-3 sm:grid-cols-2">
        <div className="rounded-md border border-border p-4">
          <div className="flex gap-3">
            <Wrench className="mt-0.5 h-4 w-4 text-primary" />
            <div>
              <div className="text-sm font-medium text-muted-foreground">Tools</div>
              <div className="mt-1 text-2xl font-semibold">{formatCount(agent.tool_count)}</div>
              <div className="mt-1 text-xs text-muted-foreground">
                {formatCount(countExternalTools(agent))} external-effect tools
              </div>
            </div>
          </div>
        </div>
        <div className="rounded-md border border-border p-4">
          <div className="flex gap-3">
            <Brain className="mt-0.5 h-4 w-4 text-primary" />
            <div>
              <div className="text-sm font-medium text-muted-foreground">Model</div>
              <div className="mt-1 truncate text-2xl font-semibold">{agent.model}</div>
              <div className="mt-1 text-xs text-muted-foreground">
                {agent.provider_name ?? "uses app default provider"}
              </div>
            </div>
          </div>
        </div>
      </div>

      <section className="rounded-md border border-border">
        <div className="border-b border-border px-4 py-3">
          <h3 className="text-sm font-semibold">Configuration</h3>
        </div>
        <div className="space-y-3 p-4">
          <DetailRow label="provider">{agent.provider_name ?? "-"}</DetailRow>
          <DetailRow label="model">{agent.model}</DetailRow>
          <DetailRow label="system prompt">
            {agent.has_system_prompt ? "configured" : "-"}
          </DetailRow>
          <DetailRow label="metadata">
            <PayloadViewer value={agent.metadata} maxHeight="max-h-40" />
          </DetailRow>
          <DetailRow label="provider options">
            <PayloadViewer value={agent.provider_options} maxHeight="max-h-40" />
          </DetailRow>
          {agent.thinking && (
            <DetailRow label="thinking">
              <PayloadViewer value={agent.thinking} maxHeight="max-h-40" />
            </DetailRow>
          )}
        </div>
      </section>

      <section className="rounded-md border border-border">
        <div className="flex min-w-0 items-center justify-between gap-3 border-b border-border px-4 py-3">
          <div>
            <h3 className="text-sm font-semibold">Related Work</h3>
            <p className="mt-1 text-xs text-muted-foreground">
              Recent sessions and queued tasks using this agent.
            </p>
          </div>
          <div className="flex shrink-0 gap-2">
            <a
              href={dashboardPath("/sessions", { agent_name: agent.name })}
              className={buttonVariants({ variant: "outline", size: "sm" })}
            >
              Sessions
            </a>
            <a
              href={dashboardPath("/tasks", { assigned_agent_name: agent.name })}
              className={buttonVariants({ variant: "outline", size: "sm" })}
            >
              Tasks
            </a>
          </div>
        </div>
        <div className="grid gap-4 p-4 lg:grid-cols-2">
          <div>
            <div className="mb-2 text-xs font-medium uppercase tracking-wide text-muted-foreground">
              Recent sessions
            </div>
            <div className="space-y-2">
              {relatedSessionsLoading ? (
                <StateMessage className="rounded-md border border-border py-6">
                  Loading sessions...
                </StateMessage>
              ) : (relatedSessions?.sessions ?? []).length > 0 ? (
                relatedSessions?.sessions.map((item) => (
                  <RecentSessionLink key={item.session.id} item={item} />
                ))
              ) : (
                <StateMessage className="rounded-md border border-border py-6">
                  No sessions found for this agent.
                </StateMessage>
              )}
            </div>
          </div>
          <div>
            <div className="mb-2 text-xs font-medium uppercase tracking-wide text-muted-foreground">
              Queue tasks
            </div>
            <div className="space-y-2">
              {relatedTasksLoading ? (
                <StateMessage className="rounded-md border border-border py-6">
                  Loading tasks...
                </StateMessage>
              ) : (relatedTasks ?? []).length > 0 ? (
                relatedTasks?.map((task) => <TaskLink key={task.id} task={task} />)
              ) : (
                <StateMessage className="rounded-md border border-border py-6">
                  No tasks assigned to this agent.
                </StateMessage>
              )}
            </div>
          </div>
        </div>
      </section>

      <section className="rounded-md border border-border">
        <div className="border-b border-border px-4 py-3">
          <h3 className="text-sm font-semibold">Tools</h3>
          <p className="mt-1 text-xs text-muted-foreground">
            Registered tool contracts visible to this agent.
          </p>
        </div>
        <div className="space-y-3 p-4">
          {agent.tools.length === 0 ? (
            <StateMessage className="py-8">No tools registered.</StateMessage>
          ) : (
            agent.tools.map((tool) => <ToolRow key={tool.name} tool={tool} />)
          )}
        </div>
      </section>
    </div>
  )
}

export function AgentsPage() {
  const [search, setSearch] = useState(() => currentQueryParam("q"))
  const [selectedAgentName, setSelectedAgentName] = useState<string | null>(null)
  const debouncedSearch = useDebouncedValue(search, 300)

  const agents = useQuery({
    queryKey: ["agents"],
    queryFn: fetchAgents,
    staleTime: 15_000,
  })

  const filteredAgents = useMemo(() => {
    const items = agents.data?.agents ?? []
    return items.filter((agent) => agentMatches(agent, debouncedSearch))
  }, [agents.data?.agents, debouncedSearch])

  useEffect(() => {
    if (selectedAgentName && filteredAgents.some((agent) => agent.name === selectedAgentName)) {
      return
    }
    setSelectedAgentName(filteredAgents[0]?.name ?? null)
  }, [filteredAgents, selectedAgentName])

  const selectedAgent =
    filteredAgents.find((agent) => agent.name === selectedAgentName) ?? filteredAgents[0] ?? null
  const relatedSessions = useQuery({
    queryKey: ["agent-related-sessions", selectedAgent?.name],
    queryFn: () =>
      fetchSessionsSummary({
        agent_name: selectedAgent?.name,
        limit: 5,
        order_by: "updated_at_desc",
      }),
    enabled: selectedAgent !== null,
    staleTime: 10_000,
  })
  const relatedTasks = useQuery({
    queryKey: ["agent-related-tasks", selectedAgent?.name],
    queryFn: () => fetchTasks({ assigned_agent_name: selectedAgent?.name, limit: 5 }),
    enabled: selectedAgent !== null,
    staleTime: 10_000,
  })
  const error = agents.error instanceof Error ? agents.error.message : null

  return (
    <Page>
      <PageHeader
        title="Agents"
        description="Inspect registered agent models, prompts, tools, and runtime-facing contracts."
      />

      <div className="grid min-h-[calc(100vh-10rem)] gap-4 xl:grid-cols-[minmax(0,0.95fr)_minmax(28rem,1.25fr)]">
        <DataCard
          title="Registered Agents"
          description={`${formatCount(agents.data?.total_count)} configured agents`}
        >
          <div className="border-b border-border p-4">
            <div className="flex min-w-0 gap-2">
              <div className="relative min-w-0 flex-1">
                <Search className="pointer-events-none absolute left-2.5 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                <Input
                  value={search}
                  onChange={(event) => setSearch(event.target.value)}
                  placeholder="Search agents, models, providers, tools..."
                  className="pl-8"
                />
              </div>
              {search.trim() !== "" && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => {
                    setSearch("")
                    replaceDashboardLocation("/agents")
                  }}
                >
                  Clear
                </Button>
              )}
            </div>
          </div>
          {agents.isLoading ? (
            <StateMessage>Loading agents...</StateMessage>
          ) : error ? (
            <StateMessage tone="danger">{error}</StateMessage>
          ) : filteredAgents.length === 0 ? (
            <StateMessage>No agents match the current search.</StateMessage>
          ) : (
            <div className="overflow-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Agent</TableHead>
                    <TableHead>Model</TableHead>
                    <TableHead className="text-right">Tools</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {filteredAgents.map((agent) => (
                    <TableRow
                      key={agent.name}
                      className={cn(
                        "cursor-pointer",
                        selectedAgent?.name === agent.name && "bg-muted/60",
                      )}
                      onClick={() => setSelectedAgentName(agent.name)}
                    >
                      <TableCell>
                        <div className="flex min-w-0 items-center gap-2">
                          <Bot className="h-4 w-4 shrink-0 text-muted-foreground" />
                          <div className="min-w-0">
                            <div className="truncate font-medium">{agent.name}</div>
                            <div className="truncate text-xs text-muted-foreground">
                              {agent.provider_name ?? "default provider"}
                            </div>
                          </div>
                        </div>
                      </TableCell>
                      <TableCell className="min-w-36">
                        <span className="font-mono text-xs">{agent.model}</span>
                      </TableCell>
                      <TableCell className="text-right">{formatCount(agent.tool_count)}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          )}
        </DataCard>

        <DataCard title="Agent Detail" description="Read-only runtime contract.">
          <AgentDetail
            agent={selectedAgent}
            relatedSessions={relatedSessions.data}
            relatedSessionsLoading={relatedSessions.isLoading}
            relatedTasks={relatedTasks.data}
            relatedTasksLoading={relatedTasks.isLoading}
          />
        </DataCard>
      </div>
    </Page>
  )
}
