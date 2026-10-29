import { Dialog } from "@base-ui/react/dialog"
import { useQuery } from "@tanstack/react-query"
import { X } from "lucide-react"
import { type RefObject, useRef, useSyncExternalStore } from "react"
import { BrowserOperator } from "@/components/dashboard/browser-operator"
import { BrowserRecordings } from "@/components/dashboard/browser-recordings"
import { SessionAnnotations } from "@/components/dashboard/session-annotations"
import { useViewer } from "@/components/dashboard/viewer"
import { Button } from "@/components/ui/button"
import { fetchArtifacts } from "@/lib/api"
import { cn } from "@/lib/utils"
import type { SessionDetail, SessionOverview } from "../../api/types"
import type { SessionRow } from "../../app/queries"
import { useServices } from "../../app/services"
import { formatDuration } from "../../app/session-model"
import { LabelChips, NavLink, SessionSummary, sessionRoute } from "../common"
import { environmentRoute } from "../environments/environment-model"

export const PANEL_TABS = ["overview", "usage", "artifacts", "browser"] as const
export type PanelTab = (typeof PANEL_TABS)[number]
const TAB_LABEL: Record<PanelTab, string> = {
  overview: "Overview",
  usage: "Usage",
  artifacts: "Artifacts",
  browser: "Browser",
}

const when = (iso: string | null | undefined) =>
  iso
    ? new Date(iso).toLocaleString("en-US", {
        month: "short",
        day: "numeric",
        hour: "numeric",
        minute: "2-digit",
        second: "2-digit",
      })
    : "–"

function Row({ k, children, mono }: { k: string; children: React.ReactNode; mono?: boolean }) {
  return (
    <div className="grid grid-cols-[7.5rem_1fr] gap-3 py-1.5 text-sm">
      <dt className="text-muted-foreground">{k}</dt>
      <dd className={cn("min-w-0 break-words", mono && "font-mono text-xs leading-5")}>
        {children}
      </dd>
    </div>
  )
}

const widePanel = () => window.matchMedia("(min-width: 1280px)").matches
const subscribePanelWidth = (listener: () => void) => {
  const media = window.matchMedia("(min-width: 1280px)")
  media.addEventListener("change", listener)
  return () => media.removeEventListener("change", listener)
}

export function DetailsPanel({
  returnFocus,
  tab,
  artifactsEnabled = true,
  setTab,
  onClose,
  session,
  overview,
  parent,
  onOpenTrace,
}: {
  returnFocus: RefObject<HTMLButtonElement | null>
  tab: PanelTab
  artifactsEnabled?: boolean
  setTab: (tab: PanelTab) => void
  onClose: () => void
  session: SessionDetail
  overview: SessionOverview | null
  parent: SessionRow | null
  /** Events, workflow and debug detail live in the session's trace. */
  onOpenTrace: () => void
}) {
  const portalContainer = useRef<HTMLDivElement>(null)
  const desktop = useSyncExternalStore(subscribePanelWidth, widePanel, () => false)
  const { prototype } = useServices()
  const viewer = useViewer()
  const artifacts = useQuery({
    queryKey: ["session-artifacts", session.id],
    queryFn: () => fetchArtifacts({ session_id: session.id, limit: 6 }),
    enabled: tab === "artifacts" && artifactsEnabled,
  })
  const tabs = PANEL_TABS.filter(
    (t) => (t !== "browser" || overview?.uses.browser) && (t !== "artifacts" || artifactsEnabled),
  )
  const seconds = overview?.ended_at
    ? (Date.parse(overview.ended_at) - Date.parse(overview.started_at)) / 1000
    : null
  const metadata = Object.entries(session.metadata).filter(([k]) => !k.startsWith("cayu:"))

  return (
    <Dialog.Root
      open
      modal={!desktop}
      disablePointerDismissal={desktop}
      onOpenChange={(open, event) => {
        if (!open) {
          if (desktop && (event.reason === "focus-out" || event.reason === "escape-key")) {
            event.cancel()
            event.allowPropagation()
            return
          }
          onClose()
        }
      }}
    >
      <div ref={portalContainer} className="contents" />
      <Dialog.Portal container={portalContainer} className="contents">
        <Dialog.Popup
          render={<aside />}
          role={desktop ? "complementary" : "dialog"}
          initialFocus={desktop ? false : undefined}
          finalFocus={returnFocus}
          aria-label="Session details"
          className="fixed inset-y-0 right-0 z-40 flex w-full max-w-[400px] flex-col border-l border-border bg-background shadow-xl xl:static xl:z-auto xl:w-[380px] xl:shadow-none"
        >
          <div className="flex h-12 shrink-0 items-center justify-between border-b border-border px-3">
            <span className="text-sm font-medium">Session details</span>
            <Dialog.Close
              render={<Button variant="ghost" size="icon-sm" aria-label="Close details" />}
            >
              <X />
            </Dialog.Close>
          </div>
          <div
            role="tablist"
            className="flex shrink-0 gap-1 overflow-x-auto border-b border-border px-2 py-1.5"
          >
            {tabs.map((t) => (
              <button
                key={t}
                type="button"
                role="tab"
                aria-selected={tab === t}
                onClick={() => setTab(t)}
                className={cn(
                  "rounded-md px-2.5 py-1 text-sm whitespace-nowrap",
                  tab === t
                    ? "bg-muted font-medium"
                    : "text-muted-foreground hover:text-foreground",
                )}
              >
                {TAB_LABEL[t]}
              </button>
            ))}
          </div>
          <div className="min-h-0 flex-1 overflow-y-auto px-4 py-3">
            {tab === "overview" && (
              <dl className="divide-y divide-border">
                <Row k="Session ID" mono>
                  {session.id}
                </Row>
                <Row k="Specialist">
                  <NavLink to="/agents" className="hover:underline">
                    {session.agent_name}
                  </NavLink>
                </Row>
                <Row k="Title">{overview?.title ?? "–"}</Row>
                <Row k="Summary">
                  {overview?.summary ? (
                    <SessionSummary text={overview.summary} tone={overview.summary_tone} />
                  ) : (
                    "–"
                  )}
                </Row>
                <Row k="Started">{when(session.created_at)}</Row>
                <Row k="Last activity">
                  {when(overview?.last_activity_at ?? session.updated_at)}
                </Row>
                <Row k="Model">
                  {session.model ?? "–"}
                  {session.provider_name ? ` via ${session.provider_name}` : ""}
                </Row>
                <Row k="Runtime">
                  {session.runtime_name} {session.runtime_version}
                </Row>
                <Row k="Build" mono>
                  {session.runtime_build_provenance.fingerprint ??
                    session.runtime_build_provenance.availability}
                  <span className="block text-muted-foreground">
                    {session.runtime_build_provenance.availability} ·{" "}
                    {session.runtime_build_provenance.origin} ·{" "}
                    {session.runtime_build_provenance.strength}
                  </span>
                </Row>
                <Row k="Environment">
                  {session.environment_name ? (
                    <NavLink
                      {...environmentRoute(session.environment_name)}
                      className="hover:underline"
                    >
                      {session.environment_name}
                    </NavLink>
                  ) : (
                    "–"
                  )}
                </Row>
                <Row k="Started by">{session.invocation.source.replaceAll("_", " ")}</Row>
                {parent && (
                  <Row k="Parent session">
                    <NavLink {...sessionRoute(parent.session.id)} className="hover:underline">
                      {parent.overview?.title ?? parent.session.id}
                    </NavLink>
                  </Row>
                )}
                {prototype && (
                  <Row k="Labels">
                    {Object.keys(session.labels).length ? (
                      <LabelChips labels={session.labels} />
                    ) : (
                      "–"
                    )}
                  </Row>
                )}
                {prototype &&
                  metadata.map(([k, v]) => (
                    <Row key={k} k={k} mono>
                      {typeof v === "object" ? JSON.stringify(v) : String(v)}
                    </Row>
                  ))}
              </dl>
            )}
            {tab === "overview" && !prototype && (
              <SessionAnnotations session={session} showRuntimeMetadata={false} compact />
            )}

            {tab === "artifacts" && (
              <div className="space-y-3">
                {artifacts.isPending && (
                  <p className="text-sm text-muted-foreground">Loading artifacts…</p>
                )}
                {artifacts.error && (
                  <p role="alert" className="text-sm text-destructive">
                    Artifacts could not be loaded.
                  </p>
                )}
                {artifacts.data?.artifacts.map((artifact) => (
                  <NavLink
                    key={`${artifact.artifact_store_id}:${artifact.id}`}
                    to="/artifacts"
                    search={{
                      session_id: session.id,
                      preview_store_id: artifact.artifact_store_id,
                      artifact: artifact.id,
                    }}
                    className="block rounded-md border border-border p-2 text-sm hover:bg-muted"
                  >
                    {artifact.filename}
                    <span className="block text-xs text-muted-foreground">
                      {artifact.content_type}
                    </span>
                  </NavLink>
                ))}
                {artifacts.data?.artifacts.length === 0 && (
                  <p className="text-sm text-muted-foreground">This session saved no artifacts.</p>
                )}
                <NavLink
                  to="/artifacts"
                  search={{ session_id: session.id }}
                  className="text-sm underline"
                >
                  Open artifacts
                </NavLink>
              </div>
            )}

            {tab === "browser" &&
              (overview?.uses.browser ? (
                <div className="space-y-4">
                  {!prototype && viewer?.full_access === true ? (
                    <BrowserOperator sessionId={session.id} />
                  ) : (
                    <p className="text-sm text-muted-foreground">
                      Browser control requires full access and application authorization.
                    </p>
                  )}
                  {!prototype && <BrowserRecordings sessionId={session.id} />}
                </div>
              ) : (
                <p className="text-sm text-muted-foreground">
                  No browser use is recorded for this session.
                </p>
              ))}

            {tab === "usage" && (
              <dl className="divide-y divide-border">
                <Row k="Duration">
                  {overview?.ended_at ? formatDuration(seconds) : "Still running"}
                </Row>
                <Row k="Model steps">{overview?.model_steps ?? "–"}</Row>
                <Row k="Tool calls">{overview?.tool_calls ?? "–"}</Row>
                <Row k="Input tokens">
                  {overview ? overview.tokens.input.toLocaleString() : "–"}
                  {overview ? (
                    <span className="text-muted-foreground">
                      {" "}
                      ({overview.tokens.cached.toLocaleString()} cached)
                    </span>
                  ) : null}
                </Row>
                <Row k="Output tokens">
                  {overview ? overview.tokens.output.toLocaleString() : "–"}
                </Row>
                <Row k="Cost">
                  {overview?.cost ? (
                    `${overview.cost.currency === "USD" ? "$" : `${overview.cost.currency} `}${overview.cost.amount.toFixed(2)}${overview.cost.complete ? "" : " (partial)"}`
                  ) : (
                    <span className="text-muted-foreground">No price book configured</span>
                  )}
                </Row>
              </dl>
            )}
          </div>
          <div className="shrink-0 border-t border-border p-3">
            <Button variant="outline" size="sm" className="w-full" onClick={onOpenTrace}>
              Open trace: events, workflow and internals
            </Button>
          </div>
        </Dialog.Popup>
      </Dialog.Portal>
    </Dialog.Root>
  )
}
