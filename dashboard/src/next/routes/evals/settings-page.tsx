import { Dialog } from "@base-ui/react/dialog"
import { useQuery } from "@tanstack/react-query"
import { useBlocker, useSearch } from "@tanstack/react-router"
import { Check, ChevronRight, Minus } from "lucide-react"
import { useCallback, useEffect, useRef, useState } from "react"
import { MemoryExperimentReportPanel } from "@/components/dashboard/evals-memory"
import { PayloadViewer } from "@/components/dashboard/layout"
import type { CalibrationState } from "@/components/dashboard/structured-judge-editor"
import { Button } from "@/components/ui/button"
import { type EvalTarget, fetchEvalTargets } from "@/lib/api"
import { EVAL_SETTINGS_TOOLS, type validateEvalSettingsSearch } from "@/lib/eval-settings-search"
import { EVAL_TARGET_QUERY_KEY } from "@/lib/evals-dashboard"
import { EVALS_READINESS_OPERATIONS, evalsReadinessReasonText } from "@/lib/evals-readiness"
import { evalTargetLabel } from "../../app/eval-view-model"
import { NavLink, PageBody, TopBar, useGo } from "../common"
import { JudgeProfiles, JudgeSettings } from "./settings-judges"
import { CorpusSettings } from "./settings-portability"
import { ScenarioSettings } from "./settings-scenarios"
import { FIELD_CLASS, LoadState, Section, useEvalAccess } from "./shared"
import { EvalExecutionProfileSummary } from "./technical-details"

export function EvalSettingsViewPage() {
  const { canReadDetails } = useEvalAccess()
  return canReadDetails ? (
    <SettingsTools />
  ) : (
    <div className="flex min-h-0 min-w-0 flex-1 flex-col">
      <SettingsHeader />
      <PageBody width="max-w-3xl">
        <p className="mt-4 text-sm text-muted-foreground" role="status">
          Your role can read eval suites and comparisons. Execution configuration and advanced tools
          require developer access.
        </p>
      </PageBody>
    </div>
  )
}

function SettingsHeader() {
  return (
    <TopBar>
      <span className="flex min-w-0 items-center gap-1">
        <NavLink to="/evals" className="shrink-0 text-muted-foreground hover:text-foreground">
          Evals
        </NavLink>
        <ChevronRight className="size-3.5 shrink-0 text-muted-foreground" aria-hidden="true" />
        <span>Settings</span>
      </span>
    </TopBar>
  )
}

function SettingsTools() {
  const search = useSearch({ strict: false }) as ReturnType<typeof validateEvalSettingsSearch>
  const { canWrite, readiness } = useEvalAccess()
  const go = useGo()
  const targets = useQuery({
    queryKey: EVAL_TARGET_QUERY_KEY,
    queryFn: ({ signal }) => fetchEvalTargets(signal),
  })
  const [initialTarget, setInitialTarget] = useState<string>()
  useEffect(() => {
    if (initialTarget === undefined && targets.data)
      setInitialTarget(targets.data.default_target_key ?? "")
  }, [initialTarget, targets.data])
  const selected = search.target ?? initialTarget ?? targets.data?.default_target_key ?? ""
  const liveTarget = targets.data?.items.find((item) => item.target_key === selected)
  const retainedTarget = useRef<EvalTarget | undefined>(undefined)
  useEffect(() => {
    if (liveTarget) retainedTarget.current = liveTarget
  }, [liveTarget])
  const scope = JSON.stringify([search.tool, selected])
  const [owner, setOwner] = useState<{ scope: string; state: CalibrationState }>()
  const state = owner?.scope === scope ? owner.state : { dirty: false, locked: false }
  // A catalog refresh cannot discard a private draft or an unresolved operation.
  const target =
    liveTarget ??
    (retainedTarget.current?.target_key === selected ? retainedTarget.current : undefined)
  const onStateChange = useCallback(
    (next: CalibrationState) => setOwner({ scope, state: next }),
    [scope],
  )
  const blocker = useBlocker({
    shouldBlockFn: () => state.dirty || state.locked,
    enableBeforeUnload: state.dirty || state.locked,
    withResolver: true,
  })
  const catalogWritable = canWrite && readiness.catalog_write.state === "ready"
  return (
    <div className="flex min-h-0 min-w-0 flex-1 flex-col">
      <SettingsHeader />
      <PageBody width={search.tool === "scenarios" ? "max-w-6xl" : "max-w-3xl"}>
        <nav
          aria-label="Eval settings tools"
          className="mt-5 flex flex-wrap gap-1 border-b border-border pb-3"
        >
          {EVAL_SETTINGS_TOOLS.map(([tool, label]) => (
            <NavLink
              key={tool}
              to="/evals/settings"
              search={{ tool, target: search.target }}
              aria-current={search.tool === tool ? "page" : undefined}
              className={`rounded-lg px-3 py-2 text-sm hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring ${search.tool === tool ? "bg-muted font-medium" : "text-muted-foreground"}`}
            >
              {label}
            </NavLink>
          ))}
        </nav>
        <LoadState error={targets.error} pending={targets.isPending} retry={targets.refetch} />
        {targets.data?.items.length === 0 && (
          <p className="mt-5 text-sm text-muted-foreground">
            No specialists are configured for evals.
          </p>
        )}
        {!!targets.data?.items.length && (
          <label className="mt-5 flex min-w-0 flex-wrap items-center gap-2 text-sm">
            Specialist
            <select
              aria-label="Settings specialist"
              className={`${FIELD_CLASS} max-w-full flex-1`}
              value={selected}
              disabled={state.dirty || state.locked}
              onChange={(event) =>
                void go({
                  to: "/evals/settings",
                  search: { tool: search.tool, target: event.target.value },
                })
              }
            >
              {!target && <option value={selected}>Specialist unavailable</option>}
              {targets.data.items.map((item) => (
                <option key={item.target_key} value={item.target_key}>
                  {evalTargetLabel(item)}
                </option>
              ))}
            </select>
          </label>
        )}
        {targets.data && selected && !liveTarget && (
          <p role="alert" className="mt-3 text-sm">
            This specialist is not available. Choose a configured specialist.
          </p>
        )}
        {!catalogWritable && (
          <p className="mt-3 text-sm text-muted-foreground" role="status">
            {canWrite
              ? evalsReadinessReasonText(readiness.catalog_write)
              : "Writing evaluations is not available in this deployment."}
          </p>
        )}
        <div key={scope}>
          {search.tool === "profiles" && (
            <>
              {target && (
                <ProfileSettings
                  target={target}
                  refreshing={targets.isFetching}
                  refresh={() => void targets.refetch()}
                />
              )}
              <Section title="Available in this deployment">
                <ul className="space-y-2 text-sm">
                  {EVALS_READINESS_OPERATIONS.map(([key, label]) => (
                    <li key={key} className="flex gap-2">
                      {readiness[key].state === "ready" ? (
                        <Check
                          className="mt-0.5 size-4 shrink-0 text-success"
                          aria-label="Available"
                        />
                      ) : (
                        <Minus
                          className="mt-0.5 size-4 shrink-0 text-warning"
                          aria-label="Not available"
                        />
                      )}
                      <span>
                        {key === "fresh_launch" ? "Fresh trials" : label}
                        {readiness[key].state !== "ready" && (
                          <span className="block text-xs text-muted-foreground">
                            {evalsReadinessReasonText(readiness[key])}
                          </span>
                        )}
                      </span>
                    </li>
                  ))}
                </ul>
              </Section>
            </>
          )}
          {search.tool === "judges" && target && (
            <JudgeSettings
              target={target}
              disabled={!canWrite || (!liveTarget && !state.locked)}
              onStateChange={onStateChange}
            />
          )}
          {search.tool === "scenarios" && target && (
            <ScenarioSettings
              targetKey={target.target_key}
              disabled={!catalogWritable || (!liveTarget && !state.locked)}
              onStateChange={onStateChange}
            />
          )}
          {search.tool === "memory" && (
            <Section title="Memory reports">
              <p className="mb-4 text-sm text-muted-foreground">
                Compare paired memory variants using stored results. Upload the campaign request to
                build a report and download JSON or HTML.
              </p>
              <MemoryExperimentReportPanel
                disabled={!canWrite || readiness.reports.state !== "ready"}
                onStateChange={onStateChange}
              />
            </Section>
          )}
          {search.tool === "portability" && target && (
            <CorpusSettings
              targetKey={target.target_key}
              disabled={!catalogWritable || (!liveTarget && !state.locked)}
              onStateChange={onStateChange}
            />
          )}
        </div>
      </PageBody>
      <Dialog.Root
        open={blocker.status === "blocked"}
        onOpenChange={(open) => {
          if (!open) blocker.reset?.()
        }}
      >
        <Dialog.Portal>
          <Dialog.Backdrop className="fixed inset-0 z-50 bg-black/40" />
          <Dialog.Popup className="fixed top-1/2 left-1/2 z-50 w-[min(28rem,calc(100vw-2rem))] -translate-x-1/2 -translate-y-1/2 space-y-4 rounded-xl border border-border bg-background p-5 shadow-xl">
            <Dialog.Title className="font-semibold">Leave this tool?</Dialog.Title>
            <Dialog.Description className="text-sm text-muted-foreground">
              {state.locked
                ? "Finish or retry the current operation before leaving. Its exact recovery is retained here."
                : "Leaving discards the private evidence or unsaved changes in this tool."}
            </Dialog.Description>
            <div className="flex justify-end gap-2">
              <Button variant="outline" onClick={() => blocker.reset?.()}>
                Keep working
              </Button>
              <Button disabled={state.locked} onClick={() => blocker.proceed?.()}>
                Leave tool
              </Button>
            </div>
          </Dialog.Popup>
        </Dialog.Portal>
      </Dialog.Root>
    </div>
  )
}

function ProfileSettings({
  target,
  refreshing,
  refresh,
}: {
  target: EvalTarget
  refreshing: boolean
  refresh: () => void
}) {
  return (
    <>
      <Section title="What trials run with">
        <p className="mb-3 text-sm text-muted-foreground">
          Each trial starts a fresh session with the specialist as it is configured now.
        </p>
        <div className="rounded-xl border border-border px-3 py-2 text-sm break-words">
          <p className="font-medium">{evalTargetLabel(target)}</p>
          <p className="text-muted-foreground">
            {target.execution_profile
              ? `${target.execution_profile.candidate.model} via ${target.execution_profile.candidate.provider_name}${target.execution_profile.candidate.environment_name ? ` · ${target.execution_profile.candidate.environment_name}` : ""}`
              : "Execution profile unavailable"}
          </p>
        </div>
      </Section>
      <Section title="Judge">
        <JudgeProfiles target={target} />
      </Section>
      <Section title="Limits">
        <p className="text-sm">
          {target.execution_profile
            ? `Up to ${target.execution_profile.ceilings.max_cases} cases, ${target.execution_profile.ceilings.max_trials} trials and ${target.execution_profile.ceilings.max_concurrency} concurrent trials; ${target.execution_profile.ceilings.max_timeout_seconds}s timeout.`
            : "Limits unavailable without an execution profile."}
        </p>
      </Section>
      <details className="mt-5">
        <summary className="cursor-pointer text-sm">Execution profile details</summary>
        <div className="mt-3">
          <EvalExecutionProfileSummary target={target} refreshing={refreshing} refresh={refresh} />
        </div>
        <PayloadViewer value={target} />
      </details>
    </>
  )
}
