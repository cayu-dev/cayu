import { useQuery } from "@tanstack/react-query"
import { Link } from "@tanstack/react-router"
import { AlertTriangle, Coins, Database, Hash, Search, Workflow } from "lucide-react"
import { useEffect, useMemo, useState } from "react"
import { DataCard, Page, PageHeader, StateMessage } from "../components/dashboard/layout"
import { Badge } from "../components/ui/badge"
import { Button } from "../components/ui/button"
import { Input } from "../components/ui/input"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "../components/ui/table"
import { fetchSessionsSummary, type SessionsSummary, type SessionsSummaryQuery } from "../lib/api"
import { bedrockBillingRows } from "../lib/bedrock-billing"
import { dashboardConfig } from "../lib/config"
import { addDecimalStrings } from "../lib/decimal"
import { formatCount, formatDateTime, formatDecimal, numericValue } from "../lib/format"
import {
  describeUsageRollupScope,
  USAGE_ROLLUP_MAX_PAGES,
  USAGE_ROLLUP_PAGE_LIMIT,
  USAGE_ROLLUP_SESSION_LIMIT,
} from "../lib/usage-rollup-scope"
import { cn } from "../lib/utils"

type SessionStatusFilter = "all" | Exclude<SessionsSummaryQuery["status"], null | undefined>
type UsageBreakdown = NonNullable<SessionsSummary["provider_breakdown"]>[number]
type UsageSession = SessionsSummary["usage"]["session_summaries"][number]
type UsageSessionCost = NonNullable<SessionsSummary["cost"]>["session_costs"][number]
type UsageCostLineItem = NonNullable<SessionsSummary["cost"]>["line_items"][number]

type UsageRollup = {
  sessions: SessionsSummary["sessions"]
  sessionSummaries: UsageSession[]
  totalCount: number | null
  loadedCount: number
  nextCursor: string | null
  usage: SessionsSummary["usage"]["usage"]
  modelSteps: number
  toolCalls: number
  providerBreakdown: UsageBreakdown[]
  modelBreakdown: UsageBreakdown[]
  cost: SessionsSummary["cost"]
}

const SESSION_STATUSES: SessionStatusFilter[] = [
  "pending",
  "running",
  "interrupting",
  "completed",
  "failed",
  "interrupted",
]

const selectClassName =
  "h-8 min-w-32 rounded-lg border border-input bg-background px-2.5 py-1 text-sm outline-none transition-colors focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50"

function useDebouncedValue<T>(value: T, delayMs: number): T {
  const [debounced, setDebounced] = useState(value)
  useEffect(() => {
    const handle = window.setTimeout(() => setDebounced(value), delayMs)
    return () => window.clearTimeout(handle)
  }, [delayMs, value])
  return debounced
}

function optionalFilter(value: string) {
  const trimmed = value.trim()
  return trimmed === "" ? undefined : trimmed
}

function formatCost(value: string | null | undefined, currency: string | null | undefined) {
  if (!value || !currency) return "-"
  return `${formatDecimal(value)} ${currency}`
}

function addUsage(left: UsageBreakdown["usage"], right: UsageBreakdown["usage"]) {
  return {
    provider_name: left.provider_name ?? right.provider_name ?? null,
    requested_model: left.requested_model ?? right.requested_model ?? null,
    model: left.model ?? right.model ?? null,
    input_tokens: numericValue(left.input_tokens) + numericValue(right.input_tokens),
    output_tokens: numericValue(left.output_tokens) + numericValue(right.output_tokens),
    total_tokens: numericValue(left.total_tokens) + numericValue(right.total_tokens),
    reasoning_output_tokens:
      numericValue(left.reasoning_output_tokens) + numericValue(right.reasoning_output_tokens),
    cache: {
      read_tokens: numericValue(left.cache?.read_tokens) + numericValue(right.cache?.read_tokens),
      write_tokens:
        numericValue(left.cache?.write_tokens) + numericValue(right.cache?.write_tokens),
      write_5m_tokens:
        numericValue(left.cache?.write_5m_tokens) + numericValue(right.cache?.write_5m_tokens),
      write_1h_tokens:
        numericValue(left.cache?.write_1h_tokens) + numericValue(right.cache?.write_1h_tokens),
      write_unknown_ttl_tokens:
        numericValue(left.cache?.write_unknown_ttl_tokens) +
        numericValue(right.cache?.write_unknown_ttl_tokens),
      cached_input_tokens:
        numericValue(left.cache?.cached_input_tokens) +
        numericValue(right.cache?.cached_input_tokens),
      uncached_input_tokens:
        numericValue(left.cache?.uncached_input_tokens) +
        numericValue(right.cache?.uncached_input_tokens),
    },
  }
}

function mergeCost(left: SessionsSummary["cost"], right: SessionsSummary["cost"]) {
  if (!left) return right
  if (!right) return left
  if (left.currency !== right.currency) {
    throw new Error(
      `Cannot merge usage costs with different currencies: ${left.currency}, ${right.currency}`,
    )
  }
  return {
    currency: left.currency,
    session_ids: [...left.session_ids, ...right.session_ids],
    session_count: left.session_count + right.session_count,
    model_steps: left.model_steps + right.model_steps,
    priced_model_steps: left.priced_model_steps + right.priced_model_steps,
    unpriced_model_steps: left.unpriced_model_steps + right.unpriced_model_steps,
    total_cost: addDecimalStrings(left.total_cost, right.total_cost),
    line_items: [...left.line_items, ...right.line_items],
    session_costs: [...left.session_costs, ...right.session_costs],
  }
}

function mergeBreakdown(
  left: UsageBreakdown[],
  right: UsageBreakdown[],
  keyOf: (item: UsageBreakdown) => string,
) {
  const buckets = new Map<string, UsageBreakdown>()
  for (const item of [...left, ...right]) {
    const key = keyOf(item)
    const existing = buckets.get(key)
    if (!existing) {
      buckets.set(key, { ...item, usage: { ...item.usage, cache: { ...item.usage.cache } } })
      continue
    }
    buckets.set(key, {
      provider_name: existing.provider_name,
      model: existing.model,
      session_count: existing.session_count + item.session_count,
      model_steps: existing.model_steps + item.model_steps,
      usage: addUsage(existing.usage, item.usage),
    })
  }
  return [...buckets.values()].sort((a, b) => {
    const tokenDelta = numericValue(b.usage.total_tokens) - numericValue(a.usage.total_tokens)
    if (tokenDelta !== 0) return tokenDelta
    return `${a.provider_name ?? ""}/${a.model ?? ""}`.localeCompare(
      `${b.provider_name ?? ""}/${b.model ?? ""}`,
    )
  })
}

async function fetchUsageRollup(query: SessionsSummaryQuery): Promise<UsageRollup> {
  let cursor: string | null | undefined
  let pages = 0
  const sessions: UsageRollup["sessions"] = []
  const sessionSummaries: UsageSession[] = []
  let totalCount: number | null = null
  let usage: UsageRollup["usage"] = {}
  let modelSteps = 0
  let toolCalls = 0
  let providerBreakdown: UsageBreakdown[] = []
  let modelBreakdown: UsageBreakdown[] = []
  let cost: SessionsSummary["cost"] = null

  do {
    const page = await fetchSessionsSummary(
      {
        ...query,
        limit: USAGE_ROLLUP_PAGE_LIMIT,
        cursor,
        offset: undefined,
      },
      {
        pricing: dashboardConfig.priceBook,
      },
    )
    pages += 1
    sessions.push(...page.sessions)
    sessionSummaries.push(...page.usage.session_summaries)
    totalCount = page.total_count
    usage = addUsage(usage, page.usage.usage)
    modelSteps += page.usage.model_steps
    toolCalls += page.usage.tool_calls
    providerBreakdown = mergeBreakdown(
      providerBreakdown,
      page.provider_breakdown ?? [],
      (item) => item.provider_name ?? "unknown-provider",
    )
    modelBreakdown = mergeBreakdown(
      modelBreakdown,
      page.model_breakdown ?? [],
      (item) => `${item.provider_name ?? "unknown-provider"}\u0000${item.model ?? "unknown-model"}`,
    )
    cost = mergeCost(cost, page.cost)
    cursor = page.next_cursor
  } while (cursor && pages < USAGE_ROLLUP_MAX_PAGES)

  return {
    sessions,
    sessionSummaries,
    totalCount,
    loadedCount: sessions.length,
    nextCursor: cursor ?? null,
    usage,
    modelSteps,
    toolCalls,
    providerBreakdown,
    modelBreakdown,
    cost,
  }
}

function MetricCard({
  icon: Icon,
  label,
  value,
  detail,
  tone = "default",
}: {
  icon: typeof Database
  label: string
  value: string
  detail: string
  tone?: "default" | "warning"
}) {
  return (
    <DataCard contentClassName="p-4">
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0">
          <div className="text-xs font-medium uppercase text-muted-foreground">{label}</div>
          <div className="mt-2 truncate text-2xl font-semibold">{value}</div>
          <div className="mt-1 text-xs text-muted-foreground">{detail}</div>
        </div>
        <div
          className={cn(
            "rounded-md border p-2",
            tone === "warning"
              ? "border-chart-1/30 bg-chart-1/10 text-chart-1"
              : "border-border bg-muted/50 text-muted-foreground",
          )}
        >
          <Icon className="h-4 w-4" />
        </div>
      </div>
    </DataCard>
  )
}

function BreakdownTable({
  title,
  description,
  rows,
  includeModel,
}: {
  title: string
  description: string
  rows: UsageBreakdown[]
  includeModel?: boolean
}) {
  return (
    <DataCard title={title} description={description}>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Provider</TableHead>
            {includeModel && <TableHead>Model</TableHead>}
            <TableHead className="text-right">Steps</TableHead>
            <TableHead className="text-right">Sessions</TableHead>
            <TableHead className="text-right">Input</TableHead>
            <TableHead className="text-right">Output</TableHead>
            <TableHead className="text-right">Total</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.length ? (
            rows.map((row) => (
              <TableRow key={`${row.provider_name ?? "unknown"}:${row.model ?? "all"}`}>
                <TableCell>{row.provider_name ?? "unknown"}</TableCell>
                {includeModel && (
                  <TableCell className="max-w-56 truncate font-mono text-xs">
                    {row.model ?? "unknown"}
                  </TableCell>
                )}
                <TableCell className="text-right">{formatCount(row.model_steps)}</TableCell>
                <TableCell className="text-right">{formatCount(row.session_count)}</TableCell>
                <TableCell className="text-right">{formatCount(row.usage.input_tokens)}</TableCell>
                <TableCell className="text-right">{formatCount(row.usage.output_tokens)}</TableCell>
                <TableCell className="text-right font-medium">
                  {formatCount(row.usage.total_tokens)}
                </TableCell>
              </TableRow>
            ))
          ) : (
            <TableRow>
              <TableCell colSpan={includeModel ? 7 : 6}>
                <StateMessage>No model usage in the current filters.</StateMessage>
              </TableCell>
            </TableRow>
          )}
        </TableBody>
      </Table>
    </DataCard>
  )
}

function BedrockBillingBreakdown({
  lineItems,
  currency,
  scopeDescription,
}: {
  lineItems: UsageCostLineItem[]
  currency: string
  scopeDescription: string
}) {
  const rows = useMemo(() => bedrockBillingRows(lineItems), [lineItems])

  if (!rows.length) return null
  return (
    <DataCard
      title="Bedrock Billing Breakdown"
      description={`Exact invocation identity within ${scopeDescription}; regions, profiles, and tiers are never combined.`}
    >
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Invoked resource</TableHead>
            <TableHead>Source region</TableHead>
            <TableHead>Scope</TableHead>
            <TableHead>Requested / effective tier</TableHead>
            <TableHead>Pricing target</TableHead>
            <TableHead className="text-right">Priced / unpriced</TableHead>
            <TableHead className="text-right">Cost</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.map((row) => (
            <TableRow
              key={JSON.stringify([
                row.identity.invoked_model,
                row.identity.source_region,
                row.identity.resource_type,
                row.identity.profile_scope,
                row.identity.requested_service_tier,
                row.identity.effective_service_tier,
              ])}
            >
              <TableCell className="max-w-80 truncate font-mono text-xs">
                {row.identity.invoked_model}
              </TableCell>
              <TableCell>{row.identity.source_region ?? "unresolved"}</TableCell>
              <TableCell>{row.identity.profile_scope ?? row.identity.resource_type}</TableCell>
              <TableCell>
                {row.identity.requested_service_tier ?? "default"} /{" "}
                {row.identity.effective_service_tier ?? "not reported"}
              </TableCell>
              <TableCell className="max-w-64 truncate font-mono text-xs">
                {row.pricingModel ?? row.missingReason ?? "unpriced"}
              </TableCell>
              <TableCell className="text-right">
                {formatCount(row.priced)} / {formatCount(row.unpriced)}
              </TableCell>
              <TableCell className="text-right">{formatCost(row.totalCost, currency)}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </DataCard>
  )
}

export function UsagePage() {
  const [search, setSearch] = useState("")
  const [status, setStatus] = useState<SessionStatusFilter>("all")
  const debouncedSearch = useDebouncedValue(search, 300)
  const query = useMemo<SessionsSummaryQuery>(
    () => ({
      q: optionalFilter(debouncedSearch),
      status: status === "all" ? undefined : status,
      order_by: "updated_at_desc",
    }),
    [debouncedSearch, status],
  )

  const usage = useQuery({
    queryKey: ["usage-rollup", query],
    queryFn: () => fetchUsageRollup(query),
  })
  const data = usage.data
  const scope = describeUsageRollupScope({
    hasMore: Boolean(data?.nextCursor),
    loadedCount: data?.loadedCount ?? 0,
    totalCount: data?.totalCount ?? null,
  })
  const ScopeIcon = scope.kind === "partial" ? AlertTriangle : Database
  const hasFilters = search.trim() !== "" || status !== "all"
  const costBySessionId = useMemo(() => {
    const bySessionId = new Map<string, UsageSessionCost>()
    for (const item of data?.cost?.session_costs ?? []) {
      bySessionId.set(item.session_id, item)
    }
    return bySessionId
  }, [data?.cost?.session_costs])
  const topSessions = useMemo(() => {
    if (!data) return []
    const sessionsById = new Map(data.sessions.map((item) => [item.session.id, item.session]))
    return [...data.sessionSummaries]
      .sort((a, b) => {
        if (data.cost) {
          const costDelta =
            numericValue(costBySessionId.get(b.session_id)?.total_cost) -
            numericValue(costBySessionId.get(a.session_id)?.total_cost)
          if (costDelta !== 0) return costDelta
        }
        return numericValue(b.usage?.total_tokens) - numericValue(a.usage?.total_tokens)
      })
      .slice(0, 20)
      .map((summary) => ({
        summary,
        session: sessionsById.get(summary.session_id),
        cost: costBySessionId.get(summary.session_id),
      }))
  }, [costBySessionId, data])

  function clearFilters() {
    setSearch("")
    setStatus("all")
  }

  return (
    <Page>
      <PageHeader
        title="Usage"
        description={`Client-side usage rollup over at most ${formatCount(USAGE_ROLLUP_SESSION_LIMIT)} most recently updated matching sessions. The current loaded scope is shown below.`}
      />

      <DataCard>
        <div className="flex min-w-0 flex-wrap items-center gap-2 border-b border-border p-4">
          <div className="relative min-w-56 flex-[2_1_18rem]">
            <Search className="pointer-events-none absolute left-2.5 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
            <Input
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder="Search sessions, agents, providers, models..."
              className="pl-8"
            />
          </div>
          <select
            value={status}
            onChange={(event) => setStatus(event.target.value as SessionStatusFilter)}
            className={selectClassName}
            aria-label="Filter by status"
          >
            <option value="all">All statuses</option>
            {SESSION_STATUSES.map((item) => (
              <option key={item} value={item}>
                {item}
              </option>
            ))}
          </select>
          {hasFilters && (
            <Button variant="outline" size="sm" onClick={clearFilters}>
              Clear
            </Button>
          )}
        </div>
      </DataCard>

      {usage.isError ? (
        <StateMessage tone="danger">
          {usage.error instanceof Error ? usage.error.message : "Failed to load usage."}
        </StateMessage>
      ) : usage.isLoading || !data ? (
        <StateMessage>Loading usage...</StateMessage>
      ) : (
        <>
          <div
            className={cn(
              "flex items-start gap-3 rounded-md border px-4 py-3 text-sm",
              scope.kind === "complete"
                ? "border-border bg-muted/40 text-foreground"
                : "border-chart-1/30 bg-chart-1/5 text-chart-1",
            )}
            data-testid="usage-loaded-scope"
            role={scope.kind === "partial" ? "alert" : "status"}
          >
            <ScopeIcon className="mt-0.5 h-4 w-4 flex-shrink-0" />
            <div className="space-y-1">
              <div className="font-medium">{scope.label}</div>
              <div>{scope.description}</div>
            </div>
          </div>

          <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-5">
            <MetricCard
              icon={Database}
              label="Loaded Sessions"
              value={formatCount(data.loadedCount)}
              detail={scope.metricQualifier}
            />
            <MetricCard
              icon={Hash}
              label="Tokens"
              value={formatCount(data.usage.total_tokens)}
              detail={`${formatCount(data.usage.input_tokens)} in / ${formatCount(data.usage.output_tokens)} out across ${scope.metricQualifier}`}
            />
            <MetricCard
              icon={Workflow}
              label="Model Steps"
              value={formatCount(data.modelSteps)}
              detail={`${formatCount(data.toolCalls)} tool calls across ${scope.metricQualifier}`}
            />
            <MetricCard
              icon={Coins}
              label="Estimated Cost"
              value={data.cost ? formatDecimal(data.cost.total_cost) : "—"}
              detail={
                data.cost
                  ? `${data.cost.currency} across ${scope.metricQualifier}`
                  : `Pricing not supplied; ${scope.metricQualifier}`
              }
              tone={data.cost?.unpriced_model_steps ? "warning" : "default"}
            />
            <MetricCard
              icon={AlertTriangle}
              label="Unpriced Steps"
              value={data.cost ? formatCount(data.cost.unpriced_model_steps) : "—"}
              detail={
                data.cost
                  ? `From the price book across ${scope.metricQualifier}`
                  : `Cost not estimated; ${scope.metricQualifier}`
              }
              tone={data.cost?.unpriced_model_steps ? "warning" : "default"}
            />
          </div>

          <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
            <BreakdownTable
              title="Provider Breakdown"
              description={`Provider-level token totals from model.completed events across ${scope.metricQualifier}.`}
              rows={data.providerBreakdown}
            />
            <BreakdownTable
              title="Model Breakdown"
              description={`Provider/model token totals from model.completed events across ${scope.metricQualifier}.`}
              rows={data.modelBreakdown}
              includeModel
            />
          </div>

          {data.cost && (
            <BedrockBillingBreakdown
              lineItems={data.cost.line_items}
              currency={data.cost.currency}
              scopeDescription={scope.metricQualifier}
            />
          )}

          <DataCard
            title={data.cost ? "Highest Cost Sessions" : "Highest Token Sessions"}
            description={
              data.cost
                ? `Up to 20 sessions by estimated cost within ${scope.metricQualifier}.`
                : `Up to 20 sessions by total tokens within ${scope.metricQualifier}.`
            }
          >
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Session</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead>Provider</TableHead>
                  <TableHead>Model</TableHead>
                  <TableHead className="text-right">Input</TableHead>
                  <TableHead className="text-right">Output</TableHead>
                  <TableHead className="text-right">Total</TableHead>
                  {data.cost && <TableHead className="text-right">Cost</TableHead>}
                  <TableHead>Updated</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {topSessions.length ? (
                  topSessions.map(({ cost, session, summary }) => (
                    <TableRow key={summary.session_id}>
                      <TableCell>
                        <Link
                          to="/sessions/$sessionId"
                          params={{ sessionId: summary.session_id }}
                          className="font-mono text-sm text-primary hover:underline"
                        >
                          {summary.session_id}
                        </Link>
                      </TableCell>
                      <TableCell>
                        {session ? <Badge variant="outline">{session.status}</Badge> : "-"}
                      </TableCell>
                      <TableCell>
                        {session?.provider_name ?? summary.provider_names?.[0] ?? "-"}
                      </TableCell>
                      <TableCell className="max-w-56 truncate font-mono text-xs">
                        {session?.model ?? summary.models?.[0] ?? "-"}
                      </TableCell>
                      <TableCell className="text-right">
                        {formatCount(summary.usage?.input_tokens)}
                      </TableCell>
                      <TableCell className="text-right">
                        {formatCount(summary.usage?.output_tokens)}
                      </TableCell>
                      <TableCell className="text-right font-medium">
                        {formatCount(summary.usage?.total_tokens)}
                      </TableCell>
                      {data.cost && (
                        <TableCell className="text-right font-medium">
                          {formatCost(cost?.total_cost, cost?.currency)}
                        </TableCell>
                      )}
                      <TableCell className="text-muted-foreground">
                        {formatDateTime(session?.updated_at)}
                      </TableCell>
                    </TableRow>
                  ))
                ) : (
                  <TableRow>
                    <TableCell colSpan={data.cost ? 9 : 8}>
                      <StateMessage>
                        {hasFilters
                          ? "No sessions with usage match the current filters."
                          : "No session usage yet."}
                      </StateMessage>
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          </DataCard>
        </>
      )}
    </Page>
  )
}
