import { useQuery } from "@tanstack/react-query"
import { Link, useNavigate, useSearch } from "@tanstack/react-router"
import {
  Activity,
  AlertTriangle,
  CalendarRange,
  Coins,
  Database,
  Hash,
  RefreshCw,
  SlidersHorizontal,
  Workflow,
} from "lucide-react"
import { useMemo, useRef, useState } from "react"
import { DataCard, Page, PageHeader, StateMessage } from "../components/dashboard/layout"
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
import { Textarea } from "../components/ui/textarea"
import { retryAggregateRequest } from "../lib/aggregate-query"
import { fetchUsageRollup, type UsageRollup } from "../lib/api"
import { billingCostRows, billingIdentityCoverage } from "../lib/billing-breakdown"
import { dashboardConfig } from "../lib/config"
import { formatCount, formatDateTime, formatDecimal } from "../lib/format"
import type {
  AggregateAccuracy,
  UsageAggregateBreakdown,
  UsageCostRollup,
} from "../lib/generated/server-api"
import {
  multilineUsageFilters,
  usageRollupFilterCount,
  usageRollupRequestState,
  usageRollupSearchWithoutFilters,
  usageRollupSessionSearch,
  usageTimestampFromUtcInput,
  usageTimestampInputValue,
} from "../lib/usage-rollup"
import {
  type UsageRange,
  type UsageRollupSearch,
  usageRollupRange,
  validateUsageRollupSearch,
} from "../lib/usage-rollup-search"
import { cn } from "../lib/utils"

const RENDERED_DETAIL_LIMIT = 20

const RANGE_LABELS: Record<UsageRange, string> = {
  "24h": "Last 24 hours",
  "7d": "Last 7 days",
  "30d": "Last 30 days",
  "90d": "Last 90 days",
  "365d": "Last 365 days",
  custom: "Custom window",
}

const selectClassName =
  "h-8 min-w-36 rounded-lg border border-input bg-background px-2.5 py-1 text-sm outline-none transition-colors focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50"

type UsageFilterDraft = {
  range: UsageRange
  startAt: string
  endAt: string
  agentName: string
  providerName: string
  model: string
  environmentName: string
  parentSessionId: string
  causalBudgetId: string
  labels: string
  labelSelectors: string
}

type UsageScalarFilterField = keyof Pick<
  UsageFilterDraft,
  "agentName" | "providerName" | "model" | "environmentName" | "parentSessionId" | "causalBudgetId"
>

const USAGE_SCALAR_FILTERS: Array<{
  id: string
  label: string
  field: UsageScalarFilterField
  placeholder: string
}> = [
  { id: "usage-agent-filter", label: "Agent", field: "agentName", placeholder: "agent name" },
  {
    id: "usage-provider-filter",
    label: "Provider",
    field: "providerName",
    placeholder: "provider name",
  },
  { id: "usage-model-filter", label: "Model", field: "model", placeholder: "model" },
  {
    id: "usage-environment-filter",
    label: "Environment",
    field: "environmentName",
    placeholder: "environment",
  },
  {
    id: "usage-parent-filter",
    label: "Parent session",
    field: "parentSessionId",
    placeholder: "parent session ID",
  },
  {
    id: "usage-budget-filter",
    label: "Causal budget",
    field: "causalBudgetId",
    placeholder: "causal budget ID",
  },
]

function optionalFilter(value: string): string | undefined {
  const trimmed = value.trim()
  return trimmed === "" ? undefined : trimmed
}

function filterDraft(search: UsageRollupSearch): UsageFilterDraft {
  return {
    range: usageRollupRange(search),
    startAt: usageTimestampInputValue(search.start_at),
    endAt: usageTimestampInputValue(search.end_at),
    agentName: search.agent_name ?? "",
    providerName: search.provider_name ?? "",
    model: search.model ?? "",
    environmentName: search.environment_name ?? "",
    parentSessionId: search.parent_session_id ?? "",
    causalBudgetId: search.causal_budget_id ?? "",
    labels: search.label?.join("\n") ?? "",
    labelSelectors: search.label_selector?.join("\n") ?? "",
  }
}

function draftKey(draft: UsageFilterDraft): string {
  return JSON.stringify(Object.values(draft).map((value) => value.trim()))
}

function UsageControls({
  search,
  resolvedWindow,
  onApply,
  onClearFilters,
}: {
  search: UsageRollupSearch
  resolvedWindow: { startAt: string; endAt: string } | null
  onApply: (search: UsageRollupSearch) => void
  onClearFilters: () => void
}) {
  const confirmed = filterDraft(search)
  const [draft, setDraft] = useState(confirmed)
  const [error, setError] = useState<string | null>(null)
  const [filtersOpen, setFiltersOpen] = useState(() => usageRollupFilterCount(search) > 0)
  const dirty = draftKey(draft) !== draftKey(confirmed)
  const filterCount = usageRollupFilterCount(search)

  function setField<Key extends keyof UsageFilterDraft>(field: Key, value: UsageFilterDraft[Key]) {
    setDraft((current) => ({ ...current, [field]: value }))
    setError(null)
  }

  function selectRange(range: UsageRange) {
    setDraft((current) => {
      if (range !== "custom" || current.startAt || current.endAt || !resolvedWindow) {
        return { ...current, range }
      }
      return {
        ...current,
        range,
        startAt: usageTimestampInputValue(resolvedWindow.startAt),
        endAt: usageTimestampInputValue(resolvedWindow.endAt),
      }
    })
    setError(null)
  }

  function submit() {
    try {
      const next = validateUsageRollupSearch({
        range: draft.range,
        start_at:
          draft.range === "custom"
            ? usageTimestampFromUtcInput(draft.startAt, "Custom start", {
                originalValue: search.start_at,
                originalInputValue: confirmed.startAt,
              })
            : undefined,
        end_at:
          draft.range === "custom"
            ? usageTimestampFromUtcInput(draft.endAt, "Custom end", {
                originalValue: search.end_at,
                originalInputValue: confirmed.endAt,
              })
            : undefined,
        agent_name: optionalFilter(draft.agentName),
        provider_name: optionalFilter(draft.providerName),
        model: optionalFilter(draft.model),
        environment_name: optionalFilter(draft.environmentName),
        parent_session_id: optionalFilter(draft.parentSessionId),
        causal_budget_id: optionalFilter(draft.causalBudgetId),
        label: multilineUsageFilters(draft.labels),
        label_selector: multilineUsageFilters(draft.labelSelectors),
      })
      const validation = usageRollupRequestState(next)
      if (!validation.ok) throw new Error(validation.error)
      onApply(next)
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "The usage filters are invalid.")
    }
  }

  return (
    <DataCard>
      <form
        className="space-y-4 p-4"
        onSubmit={(event) => {
          event.preventDefault()
          submit()
        }}
      >
        <div className="flex min-w-0 flex-wrap items-end gap-3">
          <label htmlFor="usage-range" className="text-sm text-muted-foreground">
            Time range
            <select
              id="usage-range"
              value={draft.range}
              onChange={(event) => selectRange(event.target.value as UsageRange)}
              className={cn(selectClassName, "mt-1.5 block")}
            >
              {Object.entries(RANGE_LABELS).map(([value, label]) => (
                <option key={value} value={value}>
                  {label}
                </option>
              ))}
            </select>
          </label>
          {draft.range === "custom" && (
            <>
              <label htmlFor="usage-start-at" className="text-sm text-muted-foreground">
                Start (UTC)
                <Input
                  id="usage-start-at"
                  type="datetime-local"
                  step="0.001"
                  value={draft.startAt}
                  onChange={(event) => setField("startAt", event.target.value)}
                  className="mt-1.5"
                />
              </label>
              <label htmlFor="usage-end-at" className="text-sm text-muted-foreground">
                End (UTC)
                <Input
                  id="usage-end-at"
                  type="datetime-local"
                  step="0.001"
                  value={draft.endAt}
                  onChange={(event) => setField("endAt", event.target.value)}
                  className="mt-1.5"
                />
              </label>
            </>
          )}
          <Button type="submit" size="sm" disabled={!dirty}>
            Apply
          </Button>
          {dirty && (
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => {
                setDraft(confirmed)
                setError(null)
              }}
            >
              Discard changes
            </Button>
          )}
          {filterCount > 0 && (
            <Button type="button" variant="outline" size="sm" onClick={onClearFilters}>
              Clear filters
            </Button>
          )}
        </div>

        <details
          className="border-t border-border pt-3"
          open={filtersOpen}
          onToggle={(event) => setFiltersOpen(event.currentTarget.open)}
        >
          <summary className="flex cursor-pointer list-none items-center gap-2 text-sm font-medium">
            <SlidersHorizontal className="h-4 w-4" />
            Session filters
            {filterCount > 0 && <Badge variant="secondary">{filterCount}</Badge>}
          </summary>
          <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {USAGE_SCALAR_FILTERS.map(({ id, label, field, placeholder }) => (
              <label key={id} htmlFor={id} className="text-sm text-muted-foreground">
                {label}
                <Input
                  id={id}
                  value={draft[field]}
                  onChange={(event) => setField(field, event.target.value)}
                  placeholder={placeholder}
                  className="mt-1.5"
                />
              </label>
            ))}
            <label htmlFor="usage-label-filter" className="text-sm text-muted-foreground">
              Exact labels
              <Textarea
                id="usage-label-filter"
                value={draft.labels}
                onChange={(event) => setField("labels", event.target.value)}
                placeholder={"tenant=acme\ntier=critical"}
                className="mt-1.5"
                rows={3}
              />
              <span className="mt-1 block text-xs">One key=value filter per line.</span>
            </label>
            <label htmlFor="usage-selector-filter" className="text-sm text-muted-foreground">
              Label selectors
              <Textarea
                id="usage-selector-filter"
                value={draft.labelSelectors}
                onChange={(event) => setField("labelSelectors", event.target.value)}
                placeholder={"region in (us,eu)\n!archived"}
                className="mt-1.5"
                rows={3}
              />
              <span className="mt-1 block text-xs">One selector expression per line.</span>
            </label>
          </div>
        </details>
        {error && (
          <div className="text-sm text-destructive" role="alert">
            {error}
          </div>
        )}
      </form>
    </DataCard>
  )
}

function AccuracyBadge({ accuracy }: { accuracy: AggregateAccuracy }) {
  return (
    <Badge variant={accuracy.kind === "exact" ? "secondary" : "outline"}>{accuracy.kind}</Badge>
  )
}

function accuracyExplanation(accuracy: AggregateAccuracy, fallback: string): string {
  const reason = accuracy.reason ?? fallback
  return accuracy.limit ? `${reason} Limit: ${formatCount(accuracy.limit)}.` : reason
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
  breakdown,
  includeModel = false,
}: {
  title: string
  description: string
  breakdown: UsageAggregateBreakdown
  includeModel?: boolean
}) {
  const remainder = breakdown.remainder
  return (
    <DataCard
      title={title}
      description={description}
      actions={<AccuracyBadge accuracy={breakdown.accuracy} />}
    >
      {breakdown.accuracy.kind !== "exact" && (
        <div className="border-b border-border bg-chart-1/5 px-4 py-3 text-sm text-chart-1">
          {accuracyExplanation(
            breakdown.accuracy,
            "This breakdown does not cover the complete scope.",
          )}
        </div>
      )}
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
          {breakdown.groups.map((group) => (
            <TableRow key={JSON.stringify([group.provider_name, group.model])}>
              <TableCell>{group.provider_name ?? "unknown"}</TableCell>
              {includeModel && (
                <TableCell className="max-w-56 truncate font-mono text-xs">
                  {group.model ?? "unknown"}
                </TableCell>
              )}
              <TableCell className="text-right">{formatCount(group.totals.model_steps)}</TableCell>
              <TableCell className="text-right">
                {formatCount(group.totals.session_count)}
              </TableCell>
              <TableCell className="text-right">
                {formatCount(group.totals.usage.input_tokens)}
              </TableCell>
              <TableCell className="text-right">
                {formatCount(group.totals.usage.output_tokens)}
              </TableCell>
              <TableCell className="text-right font-medium">
                {formatCount(group.totals.usage.total_tokens)}
              </TableCell>
            </TableRow>
          ))}
          {remainder && (
            <TableRow className="bg-muted/40">
              <TableCell>
                Other {formatCount(remainder.group_count)} {includeModel ? "model" : "provider"}
                {remainder.group_count === "1" ? " group" : " groups"}
              </TableCell>
              {includeModel && <TableCell className="text-muted-foreground">combined</TableCell>}
              <TableCell className="text-right">
                {formatCount(remainder.totals.model_steps)}
              </TableCell>
              <TableCell className="text-right">
                {formatCount(remainder.totals.session_count)}
              </TableCell>
              <TableCell className="text-right">
                {formatCount(remainder.totals.usage.input_tokens)}
              </TableCell>
              <TableCell className="text-right">
                {formatCount(remainder.totals.usage.output_tokens)}
              </TableCell>
              <TableCell className="text-right font-medium">
                {formatCount(remainder.totals.usage.total_tokens)}
              </TableCell>
            </TableRow>
          )}
          {!breakdown.groups.length && !remainder && (
            <TableRow>
              <TableCell colSpan={includeModel ? 7 : 6}>
                <StateMessage>No model usage in this window and filter scope.</StateMessage>
              </TableCell>
            </TableRow>
          )}
        </TableBody>
      </Table>
    </DataCard>
  )
}

function CostSummary({ cost }: { cost: UsageCostRollup | null }) {
  if (!cost) {
    return (
      <DataCard title="Estimated Cost" description="No dashboard price book is configured.">
        <StateMessage>
          Usage remains available, but cost is unavailable rather than displayed as zero.
        </StateMessage>
      </DataCard>
    )
  }

  const currencies = cost.currencies.slice(0, RENDERED_DETAIL_LIMIT)
  const reasons = cost.unpriced_reasons.slice(0, RENDERED_DETAIL_LIMIT)
  const hasCoverageGap = cost.unpriced_model_steps !== "0" || cost.unevaluated_model_steps !== "0"
  return (
    <DataCard
      title="Estimated Cost"
      description={`Price book ${cost.price_book_version}, generated ${cost.price_book_generated_at}; currencies are never combined.`}
      actions={<AccuracyBadge accuracy={cost.accuracy} />}
    >
      <div className="grid gap-3 border-b border-border p-4 sm:grid-cols-3">
        <div>
          <div className="text-xs uppercase text-muted-foreground">Priced steps</div>
          <div className="mt-1 text-xl font-semibold">{formatCount(cost.priced_model_steps)}</div>
        </div>
        <div>
          <div className="text-xs uppercase text-muted-foreground">Unpriced steps</div>
          <div className="mt-1 text-xl font-semibold">{formatCount(cost.unpriced_model_steps)}</div>
        </div>
        <div>
          <div className="text-xs uppercase text-muted-foreground">Unevaluated steps</div>
          <div className="mt-1 text-xl font-semibold">
            {formatCount(cost.unevaluated_model_steps)}
          </div>
        </div>
      </div>
      {(hasCoverageGap || cost.accuracy.kind !== "exact") && (
        <div className="border-b border-border bg-chart-1/5 px-4 py-3 text-sm text-chart-1">
          {accuracyExplanation(
            cost.accuracy,
            "Some model steps could not be priced; displayed currency totals cover priced steps only.",
          )}
        </div>
      )}
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Currency</TableHead>
            <TableHead className="text-right">Priced steps</TableHead>
            <TableHead className="text-right">Estimated cost</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {currencies.map((currency) => (
            <TableRow key={currency.currency}>
              <TableCell>{currency.currency}</TableCell>
              <TableCell className="text-right">{formatCount(currency.model_steps)}</TableCell>
              <TableCell className="text-right font-medium">
                {formatDecimal(currency.total_cost)}
              </TableCell>
            </TableRow>
          ))}
          {!currencies.length && (
            <TableRow>
              <TableCell colSpan={3}>
                <StateMessage>No model steps were priced in this scope.</StateMessage>
              </TableCell>
            </TableRow>
          )}
        </TableBody>
      </Table>
      {cost.currencies.length > currencies.length && (
        <div className="border-t border-border px-4 py-3 text-sm text-muted-foreground">
          {formatCount(cost.currencies.length - currencies.length)} additional currency rows are
          omitted from rendering.
        </div>
      )}
      {reasons.length > 0 && (
        <div className="border-t border-border p-4">
          <div className="text-sm font-medium">Unpriced reasons</div>
          <ul className="mt-2 space-y-1 text-sm text-muted-foreground">
            {reasons.map((reason) => (
              <li key={reason.reason} className="flex justify-between gap-3">
                <span>{reason.reason}</span>
                <span className="shrink-0">{formatCount(reason.model_steps)} steps</span>
              </li>
            ))}
          </ul>
          {cost.unpriced_reasons.length > reasons.length && (
            <div className="mt-2 text-xs text-muted-foreground">
              {formatCount(cost.unpriced_reasons.length - reasons.length)} additional reasons are
              omitted from rendering.
            </div>
          )}
        </div>
      )}
    </DataCard>
  )
}

function BillingIdentityBreakdown({ cost }: { cost: UsageCostRollup | null }) {
  if (!cost) return null
  const breakdown = cost.billing_breakdown
  const rows = billingCostRows(breakdown.groups)
  const coverage = billingIdentityCoverage(cost)
  const hasIdentityDetail = rows.length > 0 || breakdown.remainder !== null
  return (
    <div data-testid="usage-billing-breakdown">
      <DataCard
        title="Billing Identity Breakdown"
        description="Bounded commercial identity for identified pricing inputs. Coverage is reported separately; Bedrock regions, profiles, and tiers remain distinct while other provider evidence stays excluded."
        actions={<AccuracyBadge accuracy={breakdown.accuracy} />}
      >
        <div
          className="grid gap-3 border-b border-border p-4 sm:grid-cols-3"
          data-testid="usage-billing-identity-coverage"
        >
          <div>
            <div className="text-xs uppercase text-muted-foreground">Identified steps</div>
            <div className="mt-1 text-xl font-semibold">
              {formatCount(coverage?.identified ?? breakdown.identified_model_steps)}
            </div>
          </div>
          <div>
            <div className="text-xs uppercase text-muted-foreground">Evaluated steps</div>
            <div className="mt-1 text-xl font-semibold">
              {formatCount(coverage?.evaluated ?? cost.evaluated_model_steps)}
            </div>
          </div>
          <div>
            <div className="text-xs uppercase text-muted-foreground">Without identity</div>
            <div className="mt-1 text-xl font-semibold">
              {coverage ? formatCount(coverage.missing) : "—"}
            </div>
          </div>
        </div>
        {coverage && coverage.missing !== "0" && (
          <div
            className="border-b border-border bg-chart-1/5 px-4 py-3 text-sm text-chart-1"
            data-testid="usage-billing-identity-gap"
          >
            Billing identity is missing for {formatCount(coverage.missing)} of{" "}
            {formatCount(coverage.evaluated)} evaluated model steps. Missing steps are not
            represented in this table, but remain in the aggregate usage and cost accounting above.
          </div>
        )}
        {!coverage && (
          <div className="border-b border-border bg-destructive/5 px-4 py-3 text-sm text-destructive">
            Billing identity coverage is unavailable because the returned counters are inconsistent.
          </div>
        )}
        {breakdown.accuracy.kind !== "exact" && (
          <div className="border-b border-border bg-chart-1/5 px-4 py-3 text-sm text-chart-1">
            {accuracyExplanation(
              breakdown.accuracy,
              "This billing-identity breakdown does not include every identified group.",
            )}
          </div>
        )}
        {hasIdentityDetail ? (
          <>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Provider</TableHead>
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
                      row.identity.provider_name,
                      row.identity.source_region,
                      row.identity.resource_type,
                      row.identity.profile_scope,
                      row.identity.requested_service_tier,
                      row.identity.effective_service_tier,
                      row.pricingProvider,
                      row.pricingModel,
                      row.currency,
                      row.missingReason,
                    ])}
                  >
                    <TableCell>{row.identity.provider_name}</TableCell>
                    <TableCell className="max-w-80 truncate font-mono text-xs">
                      {row.identity.invoked_model}
                    </TableCell>
                    <TableCell>
                      {row.identity.source_region ??
                        (row.identity.provider_name === "bedrock" ? "unresolved" : "not reported")}
                    </TableCell>
                    <TableCell>
                      {row.identity.profile_scope ??
                        row.identity.resource_type ??
                        (row.identity.provider_name === "bedrock" ? "unresolved" : "not reported")}
                    </TableCell>
                    <TableCell>
                      {row.identity.provider_name === "bedrock"
                        ? `${row.identity.requested_service_tier ?? "unresolved"} / ${
                            row.identity.effective_service_tier ?? "not reported"
                          }`
                        : "not reported"}
                    </TableCell>
                    <TableCell className="max-w-64 truncate font-mono text-xs">
                      {row.pricingModel
                        ? `${row.pricingProvider ?? "unknown"}/${row.pricingModel}`
                        : (row.missingReason ?? "unpriced")}
                    </TableCell>
                    <TableCell className="text-right">
                      {formatCount(row.priced)} / {formatCount(row.unpriced)}
                    </TableCell>
                    <TableCell className="text-right">
                      {row.currency ? `${formatDecimal(row.totalCost)} ${row.currency}` : "—"}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
            {breakdown.remainder && (
              <div className="border-t border-border px-4 py-3 text-sm text-muted-foreground">
                {formatCount(breakdown.remainder.group_count)} additional billing identity groups
                across providers ({formatCount(breakdown.remainder.model_steps)} model steps) are
                included in aggregate cost totals but omitted from this bounded table. Narrow the
                provider filter to inspect a crowded provider.
              </div>
            )}
          </>
        ) : (
          <StateMessage>
            {coverage?.evaluated === "0"
              ? "No model steps were evaluated for pricing in this scope."
              : `No billing identity was reported. Evaluated model-step count: ${formatCount(
                  coverage?.evaluated ?? cost.evaluated_model_steps,
                )}.`}
          </StateMessage>
        )}
      </DataCard>
    </div>
  )
}

function UsageScope({ data }: { data: UsageRollup }) {
  const exact = data.accuracy.kind === "exact"
  return (
    <div
      className={cn(
        "flex items-start gap-3 rounded-md border px-4 py-3 text-sm",
        exact
          ? "border-border bg-muted/40 text-foreground"
          : "border-chart-1/30 bg-chart-1/5 text-chart-1",
      )}
      data-testid="usage-aggregate-scope"
      role={exact ? "status" : "alert"}
    >
      {exact ? (
        <Database className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" />
      ) : (
        <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
      )}
      <div className="space-y-1">
        <div className="flex flex-wrap items-center gap-2 font-medium">
          <span>{exact ? "Authoritative store rollup" : `${data.accuracy.kind} store rollup`}</span>
          <AccuracyBadge accuracy={data.accuracy} />
        </div>
        <div className={exact ? "text-muted-foreground" : undefined}>
          Events from {formatDateTime(data.start_at)} inclusive to {formatDateTime(data.end_at)}{" "}
          exclusive, as of {formatDateTime(data.as_of)}. Filters use current session attributes;
          event time is the usage basis. Times are shown in this browser&apos;s local zone.
        </div>
        {!exact && (
          <div>
            {accuracyExplanation(data.accuracy, "This rollup does not cover the complete scope.")}
          </div>
        )}
      </div>
    </div>
  )
}

function UsagePageForSearch({ search }: { search: UsageRollupSearch }) {
  const navigate = useNavigate({ from: "/usage" })
  const queryAnchor = useRef(new Date())
  const searchKey = JSON.stringify(search)
  const requestState = useMemo(
    () =>
      usageRollupRequestState(search, {
        now: queryAnchor.current,
        pricing: dashboardConfig.priceBook,
      }),
    [search],
  )
  const request = requestState.ok ? requestState.request : null
  const usage = useQuery({
    queryKey: ["usage-rollup", searchKey],
    queryFn: ({ signal }) => {
      const currentRequest = usageRollupRequestState(search, {
        now: queryAnchor.current,
        pricing: dashboardConfig.priceBook,
      })
      if (!currentRequest.ok) throw new Error(currentRequest.error)
      return fetchUsageRollup(currentRequest.request, signal)
    },
    enabled: request !== null,
    gcTime: 0,
    refetchOnWindowFocus: false,
    retry: retryAggregateRequest,
  })
  const data = usage.data
  const filterCount = usageRollupFilterCount(search)

  function applySearch(next: UsageRollupSearch) {
    void navigate({ search: next, resetScroll: false })
  }

  function clearFilters() {
    void navigate({
      search: (current) => usageRollupSearchWithoutFilters(current),
      resetScroll: false,
    })
  }

  function refreshUsage() {
    queryAnchor.current = new Date()
    void usage.refetch()
  }

  return (
    <Page>
      <PageHeader
        title="Usage"
        description="Store-native activity, token, and cost totals over an explicit event-time window."
        actions={
          <div className="flex flex-wrap gap-2">
            <Button
              type="button"
              variant="outline"
              onClick={refreshUsage}
              disabled={usage.isFetching}
            >
              <RefreshCw className={cn("h-4 w-4", usage.isFetching && "animate-spin")} />
              Refresh window
            </Button>
            <Link
              to="/sessions"
              search={usageRollupSessionSearch(search)}
              className={buttonVariants({ variant: "outline" })}
            >
              Browse matching sessions
            </Link>
          </div>
        }
      />

      <UsageControls
        search={search}
        resolvedWindow={
          data
            ? { startAt: data.start_at, endAt: data.end_at }
            : request
              ? { startAt: request.start_at, endAt: request.end_at }
              : null
        }
        onApply={applySearch}
        onClearFilters={clearFilters}
      />

      {!requestState.ok ? (
        <StateMessage tone="danger">
          <div role="alert">
            <div className="font-medium">The usage query is invalid.</div>
            <div className="mt-1">{requestState.error}</div>
          </div>
        </StateMessage>
      ) : usage.isError && !data ? (
        <StateMessage tone="danger">
          <div className="space-y-2" role="alert">
            <div>
              {usage.error instanceof Error ? usage.error.message : "Failed to load usage."}
            </div>
            <Button
              type="button"
              variant="outline"
              size="sm"
              disabled={usage.isFetching}
              onClick={() => void usage.refetch()}
            >
              {usage.isFetching ? "Retrying..." : "Retry"}
            </Button>
          </div>
        </StateMessage>
      ) : usage.isLoading || !data ? (
        <StateMessage>Loading the bounded usage rollup...</StateMessage>
      ) : (
        <>
          {usage.isError && (
            <StateMessage tone="danger">
              <div data-testid="usage-retained-error">
                The last confirmed rollup remains visible, but its refresh failed.{" "}
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  disabled={usage.isFetching}
                  onClick={() => void usage.refetch()}
                >
                  Retry
                </Button>
              </div>
            </StateMessage>
          )}
          <UsageScope data={data} />

          <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-5">
            <MetricCard
              icon={Database}
              label="Matching Sessions"
              value={formatCount(data.matching_session_count)}
              detail={`${formatCount(data.active_session_count)} active; ${
                filterCount
                  ? `${filterCount} current-attribute filter${filterCount === 1 ? "" : "s"}`
                  : "all current sessions"
              }; not time-bounded`}
            />
            <MetricCard
              icon={Activity}
              label="Sessions with Activity"
              value={formatCount(data.totals.session_count)}
              detail={
                data.includes_active_sessions
                  ? "model or tool activity in-window; active sessions included"
                  : "model or tool activity in-window; active sessions excluded"
              }
            />
            <MetricCard
              icon={Hash}
              label="Tokens"
              value={formatCount(data.totals.usage.total_tokens)}
              detail={`${formatCount(data.totals.usage.input_tokens)} in / ${formatCount(data.totals.usage.output_tokens)} out`}
              tone={data.accuracy.kind === "exact" ? "default" : "warning"}
            />
            <MetricCard
              icon={Workflow}
              label="Model Steps"
              value={formatCount(data.totals.model_steps)}
              detail={`${formatCount(data.totals.model_steps_with_usage)} reported normalized usage`}
              tone={data.accuracy.kind === "exact" ? "default" : "warning"}
            />
            <MetricCard
              icon={CalendarRange}
              label="Tool Calls"
              value={formatCount(data.totals.tool_calls)}
              detail="tool.call.started events in the window"
              tone={data.accuracy.kind === "exact" ? "default" : "warning"}
            />
          </div>

          <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
            <BreakdownTable
              title="Provider Breakdown"
              description="Top provider groups by normalized total tokens; omitted groups are combined explicitly."
              breakdown={data.provider_breakdown}
            />
            <BreakdownTable
              title="Model Breakdown"
              description="Top provider/model groups by normalized total tokens; omitted groups are combined explicitly."
              breakdown={data.model_breakdown}
              includeModel
            />
          </div>

          <CostSummary cost={data.cost} />
          <BillingIdentityBreakdown cost={data.cost} />

          <div className="flex items-start gap-3 rounded-md border border-border bg-muted/30 px-4 py-3 text-sm text-muted-foreground">
            <Coins className="mt-0.5 h-4 w-4 shrink-0" />
            <div>
              Aggregate responses intentionally omit session histories and per-session cost rows.
              Use{" "}
              <Link
                to="/sessions"
                search={usageRollupSessionSearch(search)}
                className="text-primary underline"
              >
                matching sessions
              </Link>{" "}
              for bounded drill-down; the time window applies only to this aggregate view.
            </div>
          </div>
        </>
      )}
    </Page>
  )
}

export function UsagePage() {
  const search = useSearch({ from: "/usage" })
  const searchKey = JSON.stringify(search)
  return <UsagePageForSearch key={searchKey} search={search} />
}
