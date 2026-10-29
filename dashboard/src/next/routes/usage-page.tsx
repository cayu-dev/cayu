/**
 * What the agent used over a time range: cost (or tokens), sessions, model steps and tool
 * calls; sessions per day; provider/model costs; and the sessions that used the most. Window
 * bounds, accuracy and pricing coverage sit in Debug.
 */
import { hashKey, useQuery, useQueryClient } from "@tanstack/react-query"
import { useSearch } from "@tanstack/react-router"
import { Info, RefreshCw, X } from "lucide-react"
import { type ReactNode, useEffect, useRef, useState } from "react"
import { Button } from "@/components/ui/button"
import { retryAggregateRequest } from "@/lib/aggregate-query"
import { fetchUsageRollup, type UsageRollup, type UsageRollupBody } from "@/lib/api"
import { dashboardConfig } from "@/lib/config"
import { useEnvironments } from "@/lib/environment-query"
import { formatCount, formatCurrencyWithCode } from "@/lib/format"
import type {
  SessionStartSeries,
  UsageAggregateGroup,
  UsageAggregateTotals,
  UsageCurrencyCost,
  UsageSessionCostSummary,
  UsageTimeSeries,
} from "@/lib/generated/server-api"
import { usageRollupRequestState, usageRollupSessionSearch } from "@/lib/usage-rollup"
import type { UsageRollupSearch } from "@/lib/usage-rollup-search"
import { cn } from "@/lib/utils"
import { useSpecialists } from "../app/queries"
import { useApi, useNow, useServices } from "../app/services"
import {
  EmptyState,
  ErrorState,
  NavLink,
  PageBody,
  ROW_CLASS,
  ROW_LINK_CLASS,
  StateDot,
  sessionRoute,
  TopBar,
  useGo,
  useRowClick,
} from "./common"
import { SessionsChart } from "./usage/sessions-chart"
import {
  partialCost,
  rankingExplanation,
  usageCostText,
  validateCostRanking,
} from "./usage/usage-costs"
import { readUsage } from "./usage/usage-query"
import { histogramRequest, joinedSeriesBuckets } from "./usage/usage-series"
import {
  formatWhen,
  localInputValue,
  RANGE_CHOICES,
  rangeLabel,
  rangeOf,
  resolveWindow,
  type UsageRangeId,
  type UsageWindow,
} from "./usage/usage-window"

type UsageSearch = UsageRollupSearch

const TOP_SESSIONS = 10
const compactFormat = new Intl.NumberFormat(undefined, {
  notation: "compact",
  maximumFractionDigits: 1,
})
const compact = (value: string | number) => {
  const n = Number(value)
  return Number.isFinite(n) ? compactFormat.format(n) : "–"
}
const money = (currencies: UsageCurrencyCost[]) =>
  currencies.map((c) => formatCurrencyWithCode(c.total_cost, c.currency)).join(" · ")
const partialSummary = partialCost

export function UsagePage() {
  const search = useSearch({ from: "/usage" })
  const {
    usageCostDetailsEnabled = false,
    usageSeriesEnabled = false,
    sessionOverviewSeriesEnabled = false,
  } = useServices()
  const timezone = Intl.DateTimeFormat().resolvedOptions().timeZone || "UTC"
  const go = useGo()
  const now = useNow()
  const api = useApi()
  const queryClient = useQueryClient()
  const anchor = useRef(now())
  const range = rangeOf(search.range)
  const resolved = resolveWindow(range, search.start_at, search.end_at, anchor.current)
  const window = "error" in resolved ? null : resolved

  const set = (patch: Partial<UsageSearch>) =>
    go({ to: "/usage", search: { ...search, ...patch }, replace: true, resetScroll: false })

  const requestState =
    window &&
    usageRollupRequestState(
      { ...search, range: "custom", start_at: window.startAt, end_at: window.endAt },
      { pricing: dashboardConfig.priceBook },
    )
  const body: UsageRollupBody | null = requestState?.ok
    ? { ...requestState.request, session_group_limit: TOP_SESSIONS }
    : null
  const problem =
    "error" in resolved
      ? resolved.error
      : requestState && !requestState.ok
        ? requestState.error
        : null
  const rollup = useQuery({
    queryKey: [
      "usage",
      "rollup",
      search,
      !!dashboardConfig.priceBook,
      usageSeriesEnabled,
      usageCostDetailsEnabled,
      timezone,
    ],
    queryFn: ({ signal, queryKey }) =>
      readUsage(queryClient, queryKey, signal, async () => {
        const requestedWindow = resolveWindow(range, search.start_at, search.end_at, anchor.current)
        if ("error" in requestedWindow) throw new Error(requestedWindow.error)
        const requested = usageRollupRequestState(
          {
            ...search,
            range: "custom",
            start_at: requestedWindow.startAt,
            end_at: requestedWindow.endAt,
          },
          { pricing: dashboardConfig.priceBook },
        )
        if (!requested.ok) throw new Error(requested.error)
        const requestedBody: UsageRollupBody = {
          ...requested.request,
          session_group_limit: TOP_SESSIONS,
          ...(usageCostDetailsEnabled ? { session_order: "cost" as const } : {}),
          ...(usageSeriesEnabled ? { series: { unit: requestedWindow.unit, timezone } } : {}),
        }
        const result = await fetchUsageRollup(requestedBody, signal)
        return {
          data: usageCostDetailsEnabled ? validateCostRanking(result) : result,
          body: requestedBody,
          window: requestedWindow,
        }
      }),
    enabled: body !== null,
    retry: retryAggregateRequest,
    refetchOnWindowFocus: false,
  })
  // Headline values retain their successful request when a refresh fails.
  const shownWindow = rollup.data?.value?.window ?? window
  const shownBody = rollup.data?.value?.body ?? body
  const countsRequest =
    shownBody && shownWindow ? histogramRequest(shownBody, shownWindow.unit, timezone) : null
  const countsQueryKey = ["usage", "counts", countsRequest, sessionOverviewSeriesEnabled]
  const counts = useQuery({
    queryKey: countsQueryKey,
    queryFn: ({ signal, queryKey }) =>
      readUsage(queryClient, queryKey, signal, () => api.overviewCounts(countsRequest!, signal)),
    enabled: !!countsRequest && sessionOverviewSeriesEnabled,
    retry: retryAggregateRequest,
    refetchOnWindowFocus: false,
  })
  const countsData = counts.data?.value
  const rollupError = rollup.data?.denied ?? rollup.error
  const countsError = counts.data?.denied ?? counts.error
  // A relative refresh changes exact bounds without changing the selected scope.
  // Retain the previous successful histogram, with its own bounds, until its replacement
  // arrives. Filters, range, timezone and capability changes must never borrow that data.
  const histogramScope = hashKey([
    search,
    timezone,
    usageSeriesEnabled,
    sessionOverviewSeriesEnabled,
  ])
  const countsKey = hashKey([countsRequest])
  const [previousHistogram, setPreviousHistogram] = useState<{
    scope: string
    cacheKey: typeof countsQueryKey
    entry: NonNullable<typeof counts.data>
    key: string
    request: NonNullable<typeof countsRequest>
    data: Awaited<ReturnType<typeof api.overviewCounts>>
  } | null>(null)
  useEffect(() => {
    setPreviousHistogram((previous) => {
      if (!sessionOverviewSeriesEnabled || !countsRequest || counts.data?.denied) return null
      if (!countsData?.series) return previous?.scope === histogramScope ? previous : null
      if (
        previous?.scope === histogramScope &&
        previous.key === countsKey &&
        previous.data === countsData
      )
        return previous
      return {
        scope: histogramScope,
        key: countsKey,
        cacheKey: ["usage", "counts", countsRequest, sessionOverviewSeriesEnabled],
        entry: counts.data!,
        request: countsRequest,
        data: countsData,
      }
    })
  }, [
    counts.data,
    countsData,
    countsRequest,
    countsKey,
    histogramScope,
    sessionOverviewSeriesEnabled,
  ])
  const histogram =
    sessionOverviewSeriesEnabled && countsRequest && !counts.data?.denied
      ? countsData?.series
        ? { key: countsKey, request: countsRequest, data: countsData }
        : previousHistogram?.scope === histogramScope &&
            queryClient.getQueryData(previousHistogram.cacheKey) === previousHistogram.entry
          ? previousHistogram
          : null
      : null
  const starts = histogram?.data.series
  const previousWindow = histogram && histogram.key !== countsKey ? histogram.request : null
  // Even identical calendar labels are not proof of equal exact bounds.
  const activity = usageSeriesEnabled && !previousWindow ? rollup.data?.value?.data.series : null
  const data = rollup.data?.value?.data

  const refresh = () => {
    anchor.current = now()
    void rollup.refetch()
    void queryClient.invalidateQueries({ queryKey: ["usage", "counts"] })
  }

  return (
    <div className="flex min-h-0 min-w-0 flex-1 flex-col">
      <TopBar
        actions={
          <Button
            variant="ghost"
            size="icon-sm"
            aria-label="Refresh"
            title="Refresh"
            disabled={rollup.isFetching}
            onClick={refresh}
          >
            <RefreshCw className={cn(rollup.isFetching && "animate-spin")} />
          </Button>
        }
      >
        Usage
      </TopBar>
      <PageBody width="max-w-5xl">
        <RangePicker
          range={range}
          window={shownWindow}
          error={problem}
          onPreset={(id) => set({ range: id, start_at: undefined, end_at: undefined })}
          onCustom={(start, end) => set({ range: "custom", start_at: start, end_at: end })}
        />
        <Filters search={search} set={set} />
        {problem && (
          <p role="alert" className="mt-3 text-sm text-destructive">
            {problem}
          </p>
        )}

        {rollupError && !data && (
          <div className="mt-4 space-y-2">
            <ErrorState error={rollupError} />
            <Button variant="outline" size="sm" onClick={refresh}>
              Try again
            </Button>
          </div>
        )}
        {rollupError && data && (
          <p role="status" className="mt-4 text-sm text-muted-foreground">
            Couldn't refresh; showing the last numbers loaded.
          </p>
        )}
        {body && rollup.isPending && (
          <p role="status" className="mt-6 text-sm text-muted-foreground">
            Loading usage…
          </p>
        )}

        {data && shownWindow && shownBody && body && (
          <div>
            <Headline data={data} />
            <PricingNote data={data} />
            <Section
              title={`Sessions per ${starts?.unit ?? shownWindow.unit}`}
              aside={
                starts ? `${formatCount(histogram?.data.total_count ?? "0")} started` : undefined
              }
            >
              <DayChart
                starts={starts}
                activity={activity}
                error={countsError}
                pending={counts.isFetching}
                previousWindow={previousWindow}
                unavailable={
                  !sessionOverviewSeriesEnabled
                    ? "Session counts are unavailable for this agent."
                    : !countsRequest
                      ? "Session counts are unavailable for label, parent or budget filters."
                      : null
                }
              />
            </Section>
            <ModelSplit data={data} search={search} />
            <TopSessions data={data} />
            <UsageDebug
              data={data}
              body={shownBody}
              window={shownWindow}
              starts={starts}
              timezone={timezone}
            />
          </div>
        )}
      </PageBody>
    </div>
  )
}

function Section({
  title,
  aside,
  children,
}: {
  title: string
  aside?: ReactNode
  children: ReactNode
}) {
  return (
    <section className="mt-8">
      <div className="mb-2 flex items-baseline justify-between gap-3">
        <h2 className="text-sm font-medium">{title}</h2>
        {aside && <span className="text-xs text-muted-foreground tabular-nums">{aside}</span>}
      </div>
      {children}
    </section>
  )
}

/* ---------- range and filters ---------- */

function RangePicker({
  range,
  window,
  error,
  onPreset,
  onCustom,
}: {
  range: UsageRangeId
  window: UsageWindow | null
  error: string | null
  onPreset: (id: UsageRangeId) => void
  onCustom: (start: string, end: string) => void
}) {
  const [editing, setEditing] = useState(false)
  const open = editing || range === "custom"
  return (
    <div>
      <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
        <fieldset className="flex min-w-0 flex-wrap gap-1.5">
          <legend className="sr-only">Time range</legend>
          {RANGE_CHOICES.map((choice) => {
            const on = choice.id === "custom" ? open : range === choice.id && !editing
            return (
              <button
                key={choice.id}
                type="button"
                aria-pressed={on}
                onClick={() => {
                  if (choice.id === "custom") setEditing(true)
                  else {
                    setEditing(false)
                    onPreset(choice.id)
                  }
                }}
                className={cn(
                  "inline-flex h-7 items-center rounded-full border px-3 text-[13px]",
                  on
                    ? "border-foreground bg-foreground text-background"
                    : "border-border hover:bg-muted",
                )}
              >
                {choice.label}
              </button>
            )
          })}
        </fieldset>
        {window && (
          <span className="text-xs text-muted-foreground tabular-nums">
            {range === "custom" || !RANGE_CHOICES.some((c) => c.id === range)
              ? `${rangeLabel(range)}: `
              : ""}
            {formatWhen(window.start)} – {range === "custom" ? formatWhen(window.end) : "now"}
          </span>
        )}
      </div>
      {open && (
        <CustomRange
          key={`${window?.start}-${window?.end}`}
          window={window}
          error={range === "custom" ? error : null}
          onApply={(start, end) => {
            setEditing(false)
            onCustom(start, end)
          }}
        />
      )}
    </div>
  )
}

function CustomRange({
  window,
  error,
  onApply,
}: {
  window: UsageWindow | null
  error: string | null
  onApply: (start: string, end: string) => void
}) {
  const originalStart = window ? localInputValue(window.start) : ""
  const originalEnd = window ? localInputValue(window.end) : ""
  const [start, setStart] = useState(originalStart)
  const [end, setEnd] = useState(originalEnd)
  const [problem, setProblem] = useState<string | null>(null)
  const field =
    "h-8 rounded-lg border border-input bg-background px-2 text-sm outline-none focus:border-ring"
  return (
    <form
      className="mt-3 flex flex-wrap items-end gap-2"
      onSubmit={(e) => {
        e.preventDefault()
        const s = Date.parse(start)
        const t = Date.parse(end)
        const exactStart =
          start === originalStart
            ? window?.startAt
            : Number.isNaN(s)
              ? undefined
              : new Date(s).toISOString()
        const exactEnd =
          end === originalEnd
            ? window?.endAt
            : Number.isNaN(t)
              ? undefined
              : new Date(t).toISOString()
        const check = resolveWindow("custom", exactStart, exactEnd, 0)
        if ("error" in check) return setProblem(check.error)
        setProblem(null)
        onApply(check.startAt, check.endAt)
      }}
    >
      <label className="text-xs text-muted-foreground">
        From
        <input
          id="usage-start"
          type="datetime-local"
          step="0.001"
          value={start}
          onChange={(e) => setStart(e.target.value)}
          className={cn(field, "mt-1 block")}
        />
      </label>
      <label className="text-xs text-muted-foreground">
        To
        <input
          id="usage-end"
          type="datetime-local"
          step="0.001"
          value={end}
          onChange={(e) => setEnd(e.target.value)}
          className={cn(field, "mt-1 block")}
        />
      </label>
      <Button type="submit" variant="outline">
        Show
      </Button>
      {(problem ?? error) && (
        <p role="alert" className="w-full text-sm text-destructive">
          {problem ?? error}
        </p>
      )}
    </form>
  )
}

function Filters({ search, set }: { search: UsageSearch; set: (p: Partial<UsageSearch>) => void }) {
  const specialists = useSpecialists()
  const environments = useEnvironments()
  const options = (names: string[], current: string | undefined) =>
    [...new Set([...names, ...(current ? [current] : [])])].sort()
  const agents = options(
    (specialists.data?.agents ?? []).map((a) => a.name),
    search.agent_name,
  )
  const envs = options(
    (environments.data?.environments ?? []).map((e) => e.name),
    search.environment_name,
  )
  const field =
    "h-8 max-w-full rounded-lg border border-input bg-background px-2 text-sm outline-none focus:border-ring"
  const chips: { label: string; clear: Partial<UsageSearch> }[] = [
    ...(search.provider_name
      ? [{ label: `Provider: ${search.provider_name}`, clear: { provider_name: undefined } }]
      : []),
    ...(search.model ? [{ label: `Model: ${search.model}`, clear: { model: undefined } }] : []),
    ...(search.parent_session_id
      ? [{ label: "Under one parent session", clear: { parent_session_id: undefined } }]
      : []),
    ...(search.causal_budget_id
      ? [{ label: "One budget", clear: { causal_budget_id: undefined } }]
      : []),
    ...(search.label?.map((value) => ({
      label: `Label: ${value}`,
      clear: { label: search.label?.filter((item) => item !== value) },
    })) ?? []),
    ...(search.label_selector?.map((value) => ({
      label: `Labels: ${value}`,
      clear: { label_selector: search.label_selector?.filter((item) => item !== value) },
    })) ?? []),
  ]
  const any = chips.length > 0 || !!search.agent_name || !!search.environment_name
  return (
    <div className="mt-3 flex flex-wrap items-center gap-2">
      <select
        id="usage-agent"
        aria-label="Specialist"
        className={field}
        value={search.agent_name ?? ""}
        onChange={(e) => set({ agent_name: e.target.value || undefined })}
      >
        <option value="">All specialists</option>
        {agents.map((a) => (
          <option key={a} value={a}>
            {a}
          </option>
        ))}
      </select>
      <select
        id="usage-environment"
        aria-label="Environment"
        className={field}
        value={search.environment_name ?? ""}
        onChange={(e) => set({ environment_name: e.target.value || undefined })}
      >
        <option value="">All environments</option>
        {envs.map((name) => (
          <option key={name} value={name}>
            {name}
          </option>
        ))}
      </select>
      {chips.map((chip) => (
        <span
          key={chip.label}
          className="inline-flex h-7 max-w-full items-center gap-1 rounded-full border border-border pr-1 pl-3 text-[13px]"
        >
          <span className="truncate">{chip.label}</span>
          <Button
            variant="ghost"
            size="icon-xs"
            aria-label={`Remove filter ${chip.label}`}
            onClick={() => set(chip.clear)}
          >
            <X />
          </Button>
        </span>
      ))}
      {any && (
        <Button
          variant="ghost"
          size="sm"
          onClick={() =>
            set({
              agent_name: undefined,
              environment_name: undefined,
              provider_name: undefined,
              model: undefined,
              parent_session_id: undefined,
              causal_budget_id: undefined,
              label: undefined,
              label_selector: undefined,
            })
          }
        >
          Clear filters
        </Button>
      )}
    </div>
  )
}

/* ---------- headline and pricing ---------- */

function Stat({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="min-w-0 rounded-xl border border-border bg-card px-4 py-3">
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="mt-1 truncate text-2xl font-semibold tabular-nums">{value}</div>
      {sub && <div className="mt-0.5 truncate text-xs text-muted-foreground">{sub}</div>}
    </div>
  )
}

function Headline({ data }: { data: UsageRollup }) {
  const usage = data.totals.usage
  const priced = data.cost && data.cost.currencies.length > 0 ? data.cost : null
  const tokens = `${compact(usage.total_tokens)} tokens`
  return (
    <div className="mt-5 grid grid-cols-2 gap-3 lg:grid-cols-4">
      {priced ? (
        <Stat
          label={partialSummary(priced) ? "Estimated cost (partial)" : "Estimated cost"}
          value={money(priced.currencies)}
          sub={tokens}
        />
      ) : (
        <Stat
          label="Tokens"
          value={compact(usage.total_tokens)}
          sub={`${compact(usage.input_tokens)} in · ${compact(usage.output_tokens)} out`}
        />
      )}
      <Stat label="Sessions" value={formatCount(data.totals.session_count)} sub="with activity" />
      <Stat label="Model steps" value={formatCount(data.totals.model_steps)} />
      <Stat label="Tool calls" value={formatCount(data.totals.tool_calls)} />
    </div>
  )
}

function Note({ children }: { children: ReactNode }) {
  return (
    <div className="mt-3 flex gap-2 rounded-lg border border-border bg-muted/40 px-3 py-2 text-sm text-muted-foreground">
      <Info className="mt-0.5 size-4 shrink-0" aria-hidden="true" />
      <div className="min-w-0 [overflow-wrap:anywhere]">{children}</div>
    </div>
  )
}

const HOW_TO_PRICE = (
  <>
    Configure a price book with{" "}
    <code className="font-mono text-xs">cayu serve --price-book path/to/prices.json</code> or your
    project's price book setting.
  </>
)

function PricingNote({ data }: { data: UsageRollup }) {
  const cost = data.cost
  if (!cost)
    return <Note>Costs aren't shown because this agent has no price list. {HOW_TO_PRICE}</Note>
  if (!partialSummary(cost)) return null
  return (
    <Note>
      Some usage could not be priced completely. Amounts shown are partial estimates. Check pricing
      coverage in Debug, or narrow the range and filters.
    </Note>
  )
}

/* ---------- chart ---------- */

function DayChart({
  starts,
  activity,
  error,
  pending,
  unavailable,
  previousWindow,
}: {
  starts?: SessionStartSeries | null
  activity?: UsageTimeSeries | null
  error: unknown
  pending: boolean
  unavailable: string | null
  previousWindow: ReturnType<typeof histogramRequest>
}) {
  const buckets = starts
    ? joinedSeriesBuckets(starts, activity).map((bucket) => ({
        ...bucket,
        detail:
          bucket.activityTokens === null
            ? null
            : `${compact(bucket.activityTokens)} activity tokens`,
      }))
    : []
  return (
    <div className="rounded-xl border border-border bg-card px-4 pt-4 pb-3">
      {unavailable ? (
        <p className="text-sm text-muted-foreground">{unavailable}</p>
      ) : starts ? (
        <>
          {error && (
            <p role="status" className="mb-3 text-sm text-muted-foreground">
              Couldn't refresh session counts; showing the last counts loaded.
            </p>
          )}
          {!error && pending && (
            <p role="status" className="mb-3 text-sm text-muted-foreground">
              Refreshing session counts; showing the last counts loaded.
            </p>
          )}
          {previousWindow && (
            <p role="status" className="mb-3 text-sm text-muted-foreground">
              Session counts cover{" "}
              <time dateTime={previousWindow.since} title={previousWindow.since}>
                {formatWhen(Date.parse(previousWindow.since!))}
              </time>{" "}
              –{" "}
              <time dateTime={previousWindow.until} title={previousWindow.until}>
                {formatWhen(Date.parse(previousWindow.until!))}
              </time>{" "}
              ({previousWindow.timezone}). Usage totals above cover the refreshed range.
            </p>
          )}
          <SessionsChart buckets={buckets} unit={starts.unit} />
          {!buckets.some((bucket) => bucket.detail) && (
            <p className="mt-2 text-xs text-muted-foreground">
              Activity token detail is unavailable for this range.
            </p>
          )}
        </>
      ) : error ? (
        <ErrorState error={error} />
      ) : (
        <p role="status" className="text-sm text-muted-foreground">
          {pending ? "Loading session counts…" : "Session counts are unavailable for this range."}
        </p>
      )}
      <p className="mt-2 text-xs text-muted-foreground">
        Counts sessions by when they started, leaving out eval trials and forks. Failures show their
        current state.
        {buckets.some((bucket) => bucket.detail) &&
          " Activity tokens count events in each period, including sessions started earlier."}
      </p>
    </div>
  )
}

function CostValue({ cost }: { cost?: UsageSessionCostSummary | null }) {
  return (
    <span
      className="inline-block min-w-20 whitespace-nowrap"
      title={cost?.accuracy.reason ?? undefined}
    >
      {usageCostText(cost)}
    </span>
  )
}

/* ---------- models ---------- */

type ProviderGroup = {
  provider: string | null
  totals: UsageAggregateTotals | null
  models: UsageAggregateGroup[]
}

function providerGroups(data: UsageRollup): ProviderGroup[] {
  const groups = new Map<string | null, ProviderGroup>()
  for (const p of data.provider_breakdown.groups)
    groups.set(p.provider_name, { provider: p.provider_name, totals: p.totals, models: [] })
  for (const m of data.model_breakdown.groups) {
    const group = groups.get(m.provider_name) ?? {
      provider: m.provider_name,
      totals: null,
      models: [],
    }
    group.models.push(m)
    groups.set(m.provider_name, group)
  }
  return [...groups.values()]
}

function ModelSplit({ data, search }: { data: UsageRollup; search: UsageSearch }) {
  const groups = providerGroups(data)
  const total = Number(data.totals.usage.total_tokens) || 1
  const scope = usageRollupSessionSearch(search)
  const rest = data.model_breakdown.remainder
  const showCost = data.cost != null
  return (
    <Section title="Models">
      {groups.length === 0 ? (
        <EmptyState title="No model calls in this range" />
      ) : (
        <div className="overflow-x-auto rounded-xl border border-border">
          <table aria-label="Cost and usage by provider and model" className="w-full text-sm">
            <thead>
              <tr className="border-b border-border text-left text-xs text-muted-foreground">
                <th className="px-3 py-2 font-medium">Provider and model</th>
                <th className="px-3 py-2 text-right font-medium">Sessions</th>
                <th className="hidden px-3 py-2 text-right font-medium sm:table-cell">Steps</th>
                <th className="px-3 py-2 text-right font-medium">Tokens</th>
                {showCost && <th className="px-3 py-2 text-right font-medium">Estimated cost</th>}
                <th className="hidden w-32 px-3 py-2 font-medium sm:table-cell">
                  <span className="sr-only">Share of tokens</span>
                </th>
              </tr>
            </thead>
            <tbody>
              {groups.map((g) => (
                <ProviderRows
                  key={g.provider ?? "unknown"}
                  group={g}
                  total={total}
                  scope={scope}
                  data={data}
                  showCost={showCost}
                />
              ))}
              {rest && (
                <tr className="border-t border-border text-muted-foreground">
                  <td className="px-3 py-2 pl-7">
                    {formatCount(rest.group_count)} other model
                    {rest.group_count === "1" ? "" : "s"}
                  </td>
                  <Numbers
                    totals={rest.totals}
                    total={total}
                    showCost={showCost}
                    cost={data.model_cost_breakdown?.remainder?.cost}
                  />
                </tr>
              )}
            </tbody>
          </table>
        </div>
      )}
      {data.provider_breakdown.remainder && (
        <p className="mt-2 text-xs text-muted-foreground">
          {formatCount(data.provider_breakdown.remainder.group_count)} other providers ·{" "}
          {compact(data.provider_breakdown.remainder.totals.usage.total_tokens)} tokens
          {showCost && ` · ${usageCostText(data.provider_cost_breakdown?.remainder?.cost)}`}
        </p>
      )}
      {(data.provider_breakdown.accuracy.kind !== "exact" ||
        data.model_breakdown.accuracy.kind !== "exact" ||
        (data.provider_cost_breakdown && data.provider_cost_breakdown.accuracy.kind !== "exact") ||
        (data.model_cost_breakdown && data.model_cost_breakdown.accuracy.kind !== "exact")) && (
        <p className="mt-2 text-xs text-muted-foreground">
          Showing a limited model and provider breakdown. See Debug for coverage.
        </p>
      )}
    </Section>
  )
}

function ProviderRows({
  group,
  total,
  scope,
  data,
  showCost,
}: {
  data: UsageRollup
  showCost: boolean
  group: ProviderGroup
  total: number
  scope: ReturnType<typeof usageRollupSessionSearch>
}) {
  const showProviderTotals = group.totals && (showCost || group.models.length !== 1)
  return (
    <>
      <tr className="border-b border-border bg-muted/30">
        <td className="px-3 py-2 font-medium">
          {group.provider ? (
            <NavLink
              to="/sessions"
              search={{ ...scope, provider_name: group.provider }}
              className={ROW_LINK_CLASS}
              title={`Sessions using ${group.provider}`}
            >
              {group.provider}
            </NavLink>
          ) : (
            "Unknown provider"
          )}
        </td>
        {showProviderTotals && group.totals ? (
          <Numbers
            totals={group.totals}
            total={total}
            muted
            showCost={showCost}
            cost={
              data.provider_cost_breakdown?.groups.find((p) => p.provider_name === group.provider)
                ?.cost
            }
          />
        ) : (
          <>
            <td colSpan={showCost ? 3 : 2} />
            <td colSpan={2} className="hidden sm:table-cell" />
          </>
        )}
      </tr>
      {group.models.map((m) => (
        <tr key={m.model ?? "unknown"} className="border-b border-border last:border-0">
          <td className="w-full max-w-0 px-3 py-2 pl-7">
            {m.model ? (
              <NavLink
                to="/sessions"
                search={{ ...scope, provider_name: group.provider ?? undefined, model: m.model }}
                className={cn("block truncate", ROW_LINK_CLASS)}
                title={`Sessions using ${m.model}`}
              >
                {m.model}
              </NavLink>
            ) : (
              <span className="text-muted-foreground">Unknown model</span>
            )}
          </td>
          <Numbers
            totals={m.totals}
            total={total}
            showCost={showCost}
            cost={
              data.model_cost_breakdown?.groups.find(
                (p) => p.provider_name === m.provider_name && p.model === m.model,
              )?.cost
            }
          />
        </tr>
      ))}
    </>
  )
}

function Numbers({
  totals,
  total,
  muted,
  showCost,
  cost,
}: {
  showCost: boolean
  cost?: UsageSessionCostSummary | null
  totals: UsageAggregateTotals
  total: number
  muted?: boolean
}) {
  const share = Number(totals.usage.total_tokens) / total
  return (
    <>
      <td className="px-3 py-2 text-right tabular-nums">{formatCount(totals.session_count)}</td>
      <td className="hidden px-3 py-2 text-right tabular-nums sm:table-cell">
        {formatCount(totals.model_steps)}
      </td>
      <td
        className="px-3 py-2 text-right tabular-nums"
        title={`${formatCount(totals.usage.input_tokens)} in · ${formatCount(totals.usage.output_tokens)} out`}
      >
        {compact(totals.usage.total_tokens)}
      </td>
      {showCost && (
        <td className="px-3 py-2 text-right tabular-nums">
          <CostValue cost={cost} />
        </td>
      )}
      <td className="hidden px-3 py-2 sm:table-cell">
        <div className="flex items-center gap-2">
          <div className="h-1.5 w-20 overflow-hidden rounded-full bg-muted">
            <div
              className={cn("h-full rounded-full", muted ? "bg-foreground/30" : "bg-foreground/70")}
              style={{ width: `${Math.min(100, share * 100)}%` }}
            />
          </div>
          <span className="w-8 text-right text-xs text-muted-foreground tabular-nums">
            {Math.round(share * 100)}%
          </span>
        </div>
      </td>
    </>
  )
}

/* ---------- top sessions ---------- */

function TopSessions({ data }: { data: UsageRollup }) {
  const { usageCostDetailsEnabled = false } = useServices()
  const rowClick = useRowClick()
  const groups = data.session_breakdown?.groups ?? []
  const title =
    data.session_ranking?.effective_order === "cost"
      ? "Top sessions by cost"
      : "Top sessions by tokens"
  const explanation = rankingExplanation(data, usageCostDetailsEnabled)
  const costs = new Map(
    (data.session_cost_breakdown?.groups ?? []).map((g) => [g.session_id, g.cost]),
  )
  const showCost = data.cost != null
  const rest = data.session_breakdown?.remainder
  if (!data.session_breakdown) return null
  return (
    <Section title={title}>
      {explanation && <p className="mb-2 text-xs text-muted-foreground">{explanation}</p>}
      {groups.length === 0 ? (
        <EmptyState title="No matching sessions in this range" />
      ) : (
        <div className="overflow-x-auto rounded-xl border border-border">
          <table aria-label={title} className="w-full text-sm">
            <thead>
              <tr className="border-b border-border text-left text-xs text-muted-foreground">
                <th className="px-3 py-2 font-medium">Session</th>
                <th className="hidden px-3 py-2 font-medium sm:table-cell">Specialist</th>
                <th className="px-3 py-2 text-right font-medium">Tokens</th>
                {showCost && <th className="px-3 py-2 text-right font-medium">Estimated cost</th>}
              </tr>
            </thead>
            <tbody>
              {groups.map((g) => {
                const title = g.title?.trim() || "Untitled session"
                const specialist = g.agent_name?.trim() || "Unknown specialist"
                const cost = costs.get(g.session_id)
                return (
                  <tr
                    key={g.session_id}
                    onClick={rowClick(sessionRoute(g.session_id))}
                    className={ROW_CLASS}
                  >
                    <td className="w-full max-w-0 px-3 py-2">
                      <div className="flex min-w-0 items-center gap-2">
                        <StateDot
                          state={
                            g.status === "completed" || g.status === "failed" ? g.status : undefined
                          }
                        />
                        <NavLink
                          {...sessionRoute(g.session_id)}
                          className={cn("truncate", ROW_LINK_CLASS)}
                        >
                          {title}
                        </NavLink>
                      </div>
                      <div className="truncate pl-4 text-xs text-muted-foreground sm:hidden">
                        {specialist}
                      </div>
                    </td>
                    <td className="hidden px-3 py-2 whitespace-nowrap text-muted-foreground sm:table-cell">
                      {specialist}
                    </td>
                    <td
                      className="px-3 py-2 text-right tabular-nums"
                      title={`${formatCount(g.totals.model_steps)} model steps · ${formatCount(g.totals.tool_calls)} tool calls`}
                    >
                      {compact(g.totals.usage.total_tokens)}
                    </td>
                    {showCost && (
                      <td className="px-3 py-2 text-right tabular-nums">
                        <CostValue cost={cost} />
                      </td>
                    )}
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
      <p className="mt-2 text-xs text-muted-foreground">
        {data.session_breakdown.accuracy.kind !== "exact" &&
          "Session detail is limited; see Debug for coverage. "}
        {rest &&
          ` ${formatCount(rest.group_count)} more session${rest.group_count === "1" ? "" : "s"} used ${compact(rest.totals.usage.total_tokens)} tokens.`}
        {rest &&
          showCost &&
          ` Estimated cost of the other sessions: ${usageCostText(data.session_cost_breakdown?.remainder?.cost)}.`}
      </p>
    </Section>
  )
}

/* ---------- debug ---------- */

function Row({ k, children, mono }: { k: string; children: ReactNode; mono?: boolean }) {
  return (
    <div className="grid grid-cols-[10rem_1fr] gap-3 py-1.5 text-sm">
      <dt className="text-muted-foreground">{k}</dt>
      <dd className={cn("min-w-0 break-words", mono && "font-mono text-xs leading-5")}>
        {children}
      </dd>
    </div>
  )
}

const accuracy = (
  a: { kind: string; reason?: string | null; limit?: number | null } | null | undefined,
) =>
  a ? `${a.kind}${a.limit ? ` · limit ${a.limit}` : ""}${a.reason ? ` · ${a.reason}` : ""}` : "–"

function UsageDebug({
  data,
  body,
  window,
  starts,
  timezone,
}: {
  data: UsageRollup
  body: UsageRollupBody | null
  window: UsageWindow
  starts?: SessionStartSeries | null
  timezone: string
}) {
  const cost = data.cost
  const reported = data.reported_costs
  return (
    <details className="mt-8 rounded-xl border border-border">
      <summary className="cursor-pointer px-3 py-2 text-sm font-medium text-muted-foreground hover:text-foreground">
        Debug
      </summary>
      <dl className="border-t border-border px-3 py-2">
        <Row k="Window" mono>
          {data.start_at} → {data.end_at} (end excluded)
        </Row>
        <Row k="As of" mono>
          {data.as_of}
        </Row>
        <Row k="Chart buckets" mono>
          {starts?.buckets.length ?? 0} × {window.unit} · {timezone}
        </Row>
        <Row k="Scope" mono>
          {data.scope} · {data.time_basis} · {data.session_filter_basis}
        </Row>
        <Row k="Matching sessions" mono>
          {data.matching_session_count} ({data.active_session_count} active, active{" "}
          {data.includes_active_sessions ? "included" : "excluded"})
        </Row>
        <Row k="Accuracy" mono>
          {accuracy(data.accuracy)}
        </Row>
        <Row k="Providers" mono>
          {accuracy(data.provider_breakdown.accuracy)}
        </Row>
        <Row k="Models" mono>
          {accuracy(data.model_breakdown.accuracy)}
        </Row>
        <Row k="Sessions" mono>
          {accuracy(data.session_breakdown?.accuracy)}
        </Row>
        <Row k="Steps with usage" mono>
          {data.totals.model_steps_with_usage} of {data.totals.model_steps}
        </Row>
        <Row k="Pricing" mono>
          {cost
            ? `${cost.price_book_version} · generated ${cost.price_book_generated_at} · ${accuracy(cost.accuracy)}`
            : "no price book"}
        </Row>
        {cost && (
          <>
            <Row k="Pricing coverage" mono>
              priced {cost.priced_model_steps} · unpriced {cost.unpriced_model_steps} · unevaluated{" "}
              {cost.unevaluated_model_steps} · evaluated {cost.evaluated_model_steps}
            </Row>
            {cost.unpriced_reasons.length > 0 && (
              <Row k="Unpriced reasons" mono>
                {cost.unpriced_reasons.map((r) => `${r.reason} (${r.model_steps})`).join("; ")}
              </Row>
            )}
            <Row k="Billing identities" mono>
              {cost.billing_breakdown.identified_model_steps} steps ·{" "}
              {cost.billing_breakdown.groups.length} groups ·{" "}
              {accuracy(cost.billing_breakdown.accuracy)}
            </Row>
          </>
        )}
        <Row k="Reported costs" mono>
          {reported?.records?.length
            ? `${reported.records.length} record${reported.records.length === 1 ? "" : "s"}${reported.truncated ? " (truncated)" : ""}`
            : "none"}
        </Row>
      </dl>
      <div className="space-y-3 border-t border-border px-3 py-3">
        {!!reported?.records?.length && (
          <div>
            <p className="mb-1 text-xs text-muted-foreground">Reported costs</p>
            <pre className="max-h-60 overflow-auto rounded-lg bg-muted p-3 font-mono text-xs leading-5">
              {reported.records
                .map(
                  (r) =>
                    `${r.timestamp}  ${r.status}  ${r.cost ?? "–"} ${r.currency ?? ""}  ${r.provider_name ?? "?"}/${r.model ?? "?"}  ${r.session_id}`,
                )
                .join("\n")}
            </pre>
          </div>
        )}
        <div>
          <p className="mb-1 text-xs text-muted-foreground">Request</p>
          <pre className="max-h-60 overflow-auto rounded-lg bg-muted p-3 font-mono text-xs leading-5">
            {JSON.stringify(
              body && { ...body, pricing: body.pricing ? "(price book)" : null },
              null,
              2,
            )}
          </pre>
        </div>
        <div>
          <p className="mb-1 text-xs text-muted-foreground">Raw response</p>
          <pre className="max-h-80 overflow-auto rounded-lg bg-muted p-3 font-mono text-xs leading-5">
            {JSON.stringify(data, null, 2)}
          </pre>
        </div>
      </div>
    </details>
  )
}
