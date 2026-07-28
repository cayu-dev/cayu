import { useQuery } from "@tanstack/react-query"
import { Database, RefreshCw, ShieldCheck } from "lucide-react"
import { DataCard, Page, PageHeader, StateMessage } from "../components/dashboard/layout"
import { Badge } from "../components/ui/badge"
import { Button } from "../components/ui/button"
import { fetchSystemDiagnostics, type SystemDiagnostics } from "../lib/api"
import { formatCount, formatDateTime } from "../lib/format"
import type { CapabilityOperation } from "../lib/generated/server-api"

function displayName(value: string): string {
  return value
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ")
}

function booleanLabel(value: boolean | null): string {
  if (value === null) return "Not reported"
  return value ? "Enabled" : "Disabled"
}

function operationLabel(operation: CapabilityOperation): string {
  if (operation.enabled) return "Available"
  if (operation.unavailable_reason === "not_configured") return "Not configured"
  return "Unsupported"
}

function OperationBadge({ operation }: { operation: CapabilityOperation }) {
  return (
    <Badge variant={operation.enabled ? "secondary" : "outline"}>{operationLabel(operation)}</Badge>
  )
}

function DetailRow({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="grid min-w-0 gap-1 border-b border-border py-3 last:border-b-0 sm:grid-cols-[12rem_minmax(0,1fr)] sm:gap-4">
      <dt className="text-sm text-muted-foreground">{label}</dt>
      <dd className="min-w-0 break-words text-sm">{value}</dd>
    </div>
  )
}

function CapabilityTables({ diagnostics }: { diagnostics: SystemDiagnostics }) {
  return (
    <div className="grid gap-5 xl:grid-cols-2">
      <DataCard
        title="Optional surfaces"
        description="Presentation availability reported by the server. Route authorization remains authoritative."
        contentClassName="p-4"
      >
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead className="border-b border-border text-xs text-muted-foreground">
              <tr>
                <th className="pb-2 font-medium">Surface</th>
                <th className="pb-2 font-medium">Configured</th>
                <th className="pb-2 font-medium">Read</th>
                <th className="pb-2 font-medium">Mutate</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(diagnostics.capabilities.surfaces).map(([name, surface]) => {
                if (surface === null || surface === undefined) return null
                return (
                  <tr key={name} className="border-b border-border last:border-b-0">
                    <td className="py-3 pr-3 font-medium">{displayName(name)}</td>
                    <td className="py-3 pr-3">
                      <Badge variant="outline">{surface.configured ? "Yes" : "No"}</Badge>
                    </td>
                    <td className="py-3 pr-3">
                      <OperationBadge operation={surface.read} />
                    </td>
                    <td className="py-3">
                      <OperationBadge operation={surface.mutate} />
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </DataCard>

      <DataCard
        title="Mutation families"
        description="Framework-owned actions advertised to compatible control-plane clients."
        contentClassName="p-4"
      >
        <dl>
          {Object.entries(diagnostics.capabilities.mutations).map(([name, operation]) => (
            <DetailRow
              key={name}
              label={displayName(name)}
              value={<OperationBadge operation={operation} />}
            />
          ))}
        </dl>
      </DataCard>
    </div>
  )
}

function SystemSnapshot({ diagnostics }: { diagnostics: SystemDiagnostics }) {
  const actor = diagnostics.capabilities.actor
  const pricing = diagnostics.pricing_catalog

  return (
    <>
      <div
        className="flex items-start gap-3 rounded-md border border-border bg-muted/40 px-4 py-3 text-sm"
        data-testid="system-snapshot-scope"
      >
        <ShieldCheck className="mt-0.5 h-4 w-4 shrink-0 text-primary" />
        <div>
          <div className="font-medium">Bounded Cayu configuration snapshot</div>
          <div className="mt-1 text-muted-foreground">
            Observed {formatDateTime(diagnostics.observed_at)}. This view does not probe databases,
            workers, networks, or external services and is not a readiness check.
          </div>
        </div>
      </div>

      <div className="grid gap-5 xl:grid-cols-2">
        <DataCard title="Deployment" contentClassName="px-4">
          <dl>
            <DetailRow
              label="Name"
              value={
                diagnostics.deployment.name ??
                (diagnostics.deployment.name_status === "omitted"
                  ? "Omitted by bounds"
                  : "Not provided")
              }
            />
            <DetailRow label="API access" value={displayName(diagnostics.deployment.api_access)} />
            <DetailRow
              label="Dashboard access"
              value={
                diagnostics.deployment.dashboard_access
                  ? displayName(diagnostics.deployment.dashboard_access)
                  : "Not applicable"
              }
            />
            <DetailRow
              label="Dashboard"
              value={booleanLabel(diagnostics.deployment.dashboard_enabled)}
            />
            <DetailRow label="API docs" value={booleanLabel(diagnostics.deployment.docs_enabled)} />
          </dl>
        </DataCard>

        <DataCard title="Runtime identity" contentClassName="px-4">
          <dl>
            <DetailRow label="Cayu version" value={diagnostics.versions.cayu ?? "Not reported"} />
            <DetailRow label="Server contract" value={`v${diagnostics.versions.server_contract}`} />
            <DetailRow label="Actor subject" value={actor?.subject ?? "Anonymous"} />
            <DetailRow label="Actor tenant" value={actor?.tenant ?? "Not reported"} />
            <DetailRow
              label="Configured stores"
              value={
                diagnostics.capabilities.configured_store_roles.length > 0 ? (
                  <div className="flex flex-wrap gap-1.5">
                    {diagnostics.capabilities.configured_store_roles.map((role) => (
                      <Badge key={role} variant="outline">
                        {role}
                      </Badge>
                    ))}
                  </div>
                ) : (
                  "None reported"
                )
              }
            />
          </dl>
        </DataCard>
      </div>

      <CapabilityTables diagnostics={diagnostics} />

      <div className="grid gap-5 xl:grid-cols-2">
        <DataCard
          title="Artifact stores"
          description="Opaque registration identities; no storage connection is probed."
          contentClassName="p-4"
        >
          <div className="mb-3 flex items-center gap-2 text-sm">
            <Database className="h-4 w-4 text-primary" />
            <span>{formatCount(diagnostics.artifact_stores.total_count)} registrations</span>
            {diagnostics.artifact_stores.truncated && <Badge variant="outline">Bounded list</Badge>}
          </div>
          {diagnostics.artifact_stores.registrations.length === 0 ? (
            <StateMessage className="py-6">No artifact stores are registered.</StateMessage>
          ) : (
            <div className="space-y-2">
              {diagnostics.artifact_stores.registrations.map((registration) => (
                <div
                  key={registration.fingerprint}
                  className="rounded-md border border-border px-3 py-2"
                >
                  <div className="break-all font-mono text-xs">{registration.fingerprint}</div>
                  <div className="mt-2 flex flex-wrap gap-1.5">
                    {(registration.store_contract_operations ?? []).map((operation) => (
                      <Badge key={operation} variant="outline">
                        {operation}
                      </Badge>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}
        </DataCard>

        <DataCard
          title="Pricing catalog"
          description="Resolved dashboard pricing availability and bounded provenance."
          contentClassName="px-4"
        >
          <dl>
            <DetailRow label="Configured" value={booleanLabel(pricing.configured)} />
            <DetailRow label="Metadata" value={displayName(pricing.metadata_status)} />
            <DetailRow label="Price book version" value={pricing.price_book_version ?? "-"} />
            <DetailRow label="Generated at" value={pricing.generated_at ?? "-"} />
          </dl>
        </DataCard>
      </div>
    </>
  )
}

export function SystemPage() {
  const diagnostics = useQuery({
    queryKey: ["system-diagnostics"],
    queryFn: ({ signal }) => fetchSystemDiagnostics(signal),
    staleTime: Number.POSITIVE_INFINITY,
    retry: false,
    refetchInterval: false,
    refetchOnMount: false,
    refetchOnReconnect: false,
    refetchOnWindowFocus: false,
  })
  const initialLoading = diagnostics.isLoading && diagnostics.data === undefined
  const error = diagnostics.error instanceof Error ? diagnostics.error.message : null

  return (
    <Page>
      <PageHeader
        title="System"
        description="Inspect bounded framework-owned deployment configuration and capability state."
        actions={
          <Button
            type="button"
            variant="outline"
            disabled={diagnostics.isFetching}
            onClick={() => void diagnostics.refetch()}
          >
            <RefreshCw
              className={`mr-1.5 h-3.5 w-3.5 ${diagnostics.isFetching ? "animate-spin" : ""}`}
            />
            {diagnostics.isFetching ? "Refreshing..." : "Refresh snapshot"}
          </Button>
        }
      />

      {initialLoading ? (
        <StateMessage>Loading the protected system snapshot...</StateMessage>
      ) : diagnostics.data ? (
        <>
          {diagnostics.isRefetchError && (
            <StateMessage tone="danger">
              Refresh failed: {error}. The last confirmed snapshot remains visible.
            </StateMessage>
          )}
          <SystemSnapshot diagnostics={diagnostics.data} />
        </>
      ) : (
        <StateMessage tone="danger">
          <div role="alert">
            <div className="font-medium">System diagnostics unavailable</div>
            <div className="mt-1">{error ?? "The server returned no diagnostic snapshot."}</div>
          </div>
        </StateMessage>
      )}
    </Page>
  )
}
