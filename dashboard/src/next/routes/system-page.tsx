/**
 * How this agent is set up: version, deployment, who you are, which stores it uses and
 * whether pricing is on, then the features this deployment has turned off. Capability
 * matrices, store fingerprints and the raw diagnostics sit in Debug. Full access only.
 */
import { useQuery } from "@tanstack/react-query"
import { CircleCheck, CircleMinus, CircleSlash, RefreshCw } from "lucide-react"
import type { ReactNode } from "react"
import { useViewer } from "@/components/dashboard/viewer"
import { Button } from "@/components/ui/button"
import { fetchSystemDiagnostics, type SystemDiagnostics } from "@/lib/api"
import type { CapabilityOperation } from "@/lib/generated/server-api"
import { cn } from "@/lib/utils"
import { useApp } from "../app/queries"
import { useNow, useServices } from "../app/services"
import { relativeTime } from "../app/session-model"
import { EmptyState, ErrorState, PageBody, TopBar } from "./common"
import { EVAL_OPERATIONS, evalReason, readable, STORE, turnedOff } from "./system/system-labels"

export function SystemPage() {
  const { prototype } = useServices()
  const viewer = useViewer()
  const allowed = !!prototype || viewer?.full_access === true
  const diagnostics = useQuery({
    queryKey: ["system-diagnostics"],
    queryFn: ({ signal }) => fetchSystemDiagnostics(signal),
    enabled: allowed,
    staleTime: 60_000,
    retry: false,
    refetchOnWindowFocus: false,
  })
  const data = diagnostics.data

  return (
    <div className="flex min-h-0 min-w-0 flex-1 flex-col">
      <TopBar
        actions={
          allowed ? (
            <Button
              variant="ghost"
              size="icon-sm"
              aria-label="Refresh"
              title="Refresh"
              disabled={diagnostics.isFetching}
              onClick={() => void diagnostics.refetch()}
            >
              <RefreshCw className={cn(diagnostics.isFetching && "animate-spin")} />
            </Button>
          ) : undefined
        }
      >
        System
      </TopBar>
      <PageBody width="max-w-3xl">
        {!allowed ? (
          <EmptyState title="Only people with full access can see this">
            Ask someone with full access to this agent if you need its setup details.
          </EmptyState>
        ) : diagnostics.isPending ? (
          <p role="status" className="text-sm text-muted-foreground">
            Loading…
          </p>
        ) : !data ? (
          <ErrorState error={diagnostics.error} />
        ) : (
          <>
            {diagnostics.isRefetchError && (
              <p role="status" className="mb-3 text-sm text-muted-foreground">
                Couldn't refresh; showing what was loaded before.
              </p>
            )}
            <About data={data} />
            <TurnedOff data={data} />
            <SystemDebug data={data} />
          </>
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
    <section className="mt-8 first:mt-0">
      <div className="mb-2 flex items-baseline justify-between gap-3">
        <h2 className="text-sm font-medium">{title}</h2>
        {aside && <span className="text-xs text-muted-foreground">{aside}</span>}
      </div>
      {children}
    </section>
  )
}

function Row({ k, children, mono }: { k: string; children: ReactNode; mono?: boolean }) {
  return (
    <div className="grid grid-cols-[8.5rem_1fr] gap-3 py-1.5 text-sm sm:grid-cols-[10rem_1fr]">
      <dt className="text-muted-foreground">{k}</dt>
      <dd className={cn("min-w-0 break-words", mono && "font-mono text-xs leading-5")}>
        {children}
      </dd>
    </div>
  )
}

const day = (value: string | null | undefined) => {
  const ms = value ? Date.parse(value) : Number.NaN
  return Number.isNaN(ms)
    ? null
    : new Date(ms).toLocaleDateString([], { year: "numeric", month: "short", day: "numeric" })
}

const checked = (ago: string) =>
  ago === "now"
    ? "Checked just now"
    : /^\d+[mh]$/.test(ago)
      ? `Checked ${ago} ago`
      : `Checked ${ago === "Yesterday" ? "yesterday" : `on ${ago}`}`

function About({ data }: { data: SystemDiagnostics }) {
  const app = useApp()
  const viewer = useViewer()
  const now = useNow()
  const { prototype } = useServices()
  const actor = data.capabilities.actor
  const deployment = data.deployment
  const pricing = data.pricing_catalog
  const stores = data.capabilities.configured_store_roles
  const artifactStores = data.artifact_stores.total_count
  const role = viewer?.role ?? (viewer?.full_access || prototype ? "Full access" : null)
  return (
    <Section
      title="About this agent"
      aside={checked(relativeTime(Date.parse(data.observed_at), now()))}
    >
      <dl className="rounded-xl border border-border bg-card px-4 py-2">
        {app.data?.name && <Row k="Name">{app.data.name}</Row>}
        <Row k="Cayu version">{data.versions.cayu ?? "Not reported"}</Row>
        <Row k="Deployment">
          {deployment.name ??
            (deployment.name_status === "omitted" ? "Name hidden" : "No name set")}
          <span className="text-muted-foreground">
            {" · "}
            {deployment.api_access === "authenticated" ? "sign-in required" : "open, no sign-in"}
          </span>
        </Row>
        <Row k="Signed in as">
          {actor ? (
            <>
              {actor.subject}
              {actor.tenant && <span className="text-muted-foreground"> · {actor.tenant}</span>}
            </>
          ) : deployment.api_access === "open" ? (
            <span className="text-muted-foreground">Nobody; this deployment is open</span>
          ) : (
            <span className="text-muted-foreground">Not reported</span>
          )}
        </Row>
        {role && <Row k="Your role">{role}</Row>}
        <Row k="Stores">
          {stores.length ? stores.map((s) => STORE[s] ?? readable(s)).join(", ") : "None reported"}
          {artifactStores > 1 && (
            <span className="text-muted-foreground"> · {artifactStores} artifact stores</span>
          )}
        </Row>
        <Row k="Pricing">
          {pricing.configured ? (
            <>
              On
              {pricing.price_book_version && (
                <span className="text-muted-foreground">
                  {" · "}price list {pricing.price_book_version}
                  {day(pricing.generated_at) && ` from ${day(pricing.generated_at)}`}
                </span>
              )}
            </>
          ) : (
            <>
              Off
              <span className="text-muted-foreground">
                {" · "}Usage shows tokens but no costs. Pass{" "}
                <code className="font-mono text-xs">
                  {'dashboard_config={"priceBook": default_price_book()}'}
                </code>{" "}
                to <code className="font-mono text-xs">mount_cayu</code> to turn it on.
              </span>
            </>
          )}
        </Row>
        <Row k="API docs">
          {deployment.docs_enabled === false
            ? "Off"
            : deployment.docs_enabled
              ? "On"
              : "Not reported"}
        </Row>
      </dl>
    </Section>
  )
}

function TurnedOff({ data }: { data: SystemDiagnostics }) {
  const off = turnedOff(data)
  const evals = EVAL_OPERATIONS.map(([key, label]) => ({
    key,
    label,
    readiness: data.capabilities.evals_readiness[key],
  })).filter((e) => e.readiness)
  const notReady = evals.filter((e) => e.readiness.state !== "ready")
  return (
    <>
      <Section title="What's turned off">
        {off.length === 0 ? (
          <EmptyState title="Everything is on" />
        ) : (
          <ul className="divide-y divide-border rounded-xl border border-border bg-card">
            {off.map((item) => (
              <li key={item.key} className="flex gap-3 px-4 py-2.5 text-sm">
                <CircleSlash
                  className="mt-0.5 size-4 shrink-0 text-muted-foreground"
                  aria-hidden="true"
                />
                <div className="min-w-0">
                  <span>{item.name}</span>
                  <span className="text-muted-foreground"> · {item.why}</span>
                </div>
              </li>
            ))}
          </ul>
        )}
      </Section>
      <Section
        title="Evals readiness"
        aside={
          notReady.length === 0
            ? "All ready"
            : `${evals.length - notReady.length} of ${evals.length} ready`
        }
      >
        <ul className="divide-y divide-border rounded-xl border border-border bg-card">
          {notReady.map(({ key, label, readiness }) => {
            const gated = readiness.state === "gated"
            const Icon = gated ? CircleMinus : CircleSlash
            return (
              <li key={key} className="flex gap-3 px-4 py-2 text-sm">
                <Icon
                  className={cn(
                    "mt-0.5 size-4 shrink-0",
                    gated ? "text-warning" : "text-muted-foreground",
                  )}
                  aria-hidden="true"
                />
                <div className="min-w-0">
                  <span>{label}</span>
                  <span className="text-muted-foreground">
                    {" · "}
                    {evalReason(readiness)}
                  </span>
                </div>
              </li>
            )
          })}
          {notReady.length < evals.length && (
            <li className="flex gap-3 px-4 py-2 text-sm">
              <CircleCheck className="mt-0.5 size-4 shrink-0 text-success" aria-hidden="true" />
              <span className="text-muted-foreground">
                {notReady.length === 0
                  ? "Everything Evals needs is set up: suites, runs, reports and scoring past sessions."
                  : `The other ${evals.length - notReady.length} eval features are ready.`}
              </span>
            </li>
          )}
        </ul>
      </Section>
    </>
  )
}

const opText = (op: CapabilityOperation) => (op.enabled ? "on" : (op.unavailable_reason ?? "off"))

function SystemDebug({ data }: { data: SystemDiagnostics }) {
  const surfaces = Object.entries(data.capabilities.surfaces).filter(
    (e): e is [string, NonNullable<(typeof e)[1]>] => !!e[1],
  )
  return (
    <details className="mt-8 rounded-xl border border-border">
      <summary className="cursor-pointer px-3 py-2 text-sm font-medium text-muted-foreground hover:text-foreground">
        Debug
      </summary>
      <dl className="border-t border-border px-3 py-2">
        <Row k="Observed at" mono>
          {data.observed_at}
        </Row>
        <Row k="Server contract" mono>
          v{data.versions.server_contract}
        </Row>
        <Row k="Cayu version" mono>
          {data.versions.cayu ?? "null"} (capabilities: {data.capabilities.cayu_version ?? "null"})
        </Row>
        <Row k="Deployment" mono>
          name_status={data.deployment.name_status} · api_access={data.deployment.api_access} ·
          dashboard_access={data.deployment.dashboard_access ?? "null"} · dashboard_enabled=
          {String(data.deployment.dashboard_enabled)} · docs_enabled=
          {String(data.deployment.docs_enabled)}
        </Row>
        <Row k="Store roles" mono>
          {data.capabilities.configured_store_roles.join(", ") || "none"}
        </Row>
        <Row k="Pricing catalog" mono>
          configured={String(data.pricing_catalog.configured)} · metadata=
          {data.pricing_catalog.metadata_status} · version=
          {data.pricing_catalog.price_book_version ?? "null"} · generated=
          {data.pricing_catalog.generated_at ?? "null"}
        </Row>
      </dl>
      <div className="space-y-4 border-t border-border px-3 py-3">
        <div>
          <p className="mb-1 text-xs text-muted-foreground">Surfaces</p>
          <div className="overflow-x-auto">
            <table className="w-full font-mono text-xs leading-5">
              <thead className="text-left text-muted-foreground">
                <tr>
                  <th className="pr-3 font-normal">surface</th>
                  <th className="pr-3 font-normal">configured</th>
                  <th className="pr-3 font-normal">read</th>
                  <th className="font-normal">mutate</th>
                </tr>
              </thead>
              <tbody>
                {surfaces.map(([key, s]) => (
                  <tr key={key}>
                    <td className="pr-3">{key}</td>
                    <td className="pr-3">{String(s.configured)}</td>
                    <td className="pr-3">{opText(s.read)}</td>
                    <td>{opText(s.mutate)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
        <div>
          <p className="mb-1 text-xs text-muted-foreground">Mutations</p>
          <table className="font-mono text-xs leading-5">
            <tbody>
              {Object.entries(data.capabilities.mutations).map(([key, op]) => (
                <tr key={key}>
                  <td className="pr-3">{key}</td>
                  <td>{opText(op)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div>
          <p className="mb-1 text-xs text-muted-foreground">Evals readiness</p>
          <table className="font-mono text-xs leading-5">
            <tbody>
              {Object.entries(data.capabilities.evals_readiness).map(([key, r]) => (
                <tr key={key}>
                  <td className="pr-3">{key}</td>
                  <td className="pr-3">{r.state}</td>
                  <td>{r.reason_code ?? ""}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div>
          <p className="mb-1 text-xs text-muted-foreground">
            Artifact stores ({data.artifact_stores.total_count}
            {data.artifact_stores.truncated ? ", list cut short" : ""})
          </p>
          {data.artifact_stores.registrations.length === 0 ? (
            <p className="font-mono text-xs">none</p>
          ) : (
            <ul className="space-y-1 font-mono text-xs leading-5 break-all">
              {data.artifact_stores.registrations.map((r) => (
                <li key={r.fingerprint}>
                  {r.fingerprint}
                  {r.store_contract_operations && (
                    <span className="text-muted-foreground">
                      {" "}
                      [{r.store_contract_operations.join(", ")}]
                    </span>
                  )}
                </li>
              ))}
            </ul>
          )}
        </div>
        <div>
          <p className="mb-1 text-xs text-muted-foreground">Raw diagnostics</p>
          <pre className="max-h-96 overflow-auto rounded-lg bg-muted p-3 font-mono text-xs leading-5">
            {JSON.stringify(data, null, 2)}
          </pre>
        </div>
      </div>
    </details>
  )
}
