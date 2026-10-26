import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { useRef, useState } from "react"
import { EvalSuiteAuthoringAction } from "@/components/dashboard/eval-suite-authoring"
import { MemoryExperimentReportAction } from "@/components/dashboard/evals-memory"
import { PayloadViewer } from "@/components/dashboard/layout"
import { Button } from "@/components/ui/button"
import { fetchEvalTargets, importEvalCorpus } from "@/lib/api"
import { EVAL_TARGET_QUERY_KEY, preflightEvalCorpusFile } from "@/lib/evals-dashboard"
import { EVALS_READINESS_OPERATIONS, evalsReadinessReasonText } from "@/lib/evals-readiness"
import { nativeRunSelector, runViewPath } from "../../app/eval-view-model"
import { NavLink, PageBody, TopBar, useGo } from "../common"
import { FIELD_CLASS, LoadState, Section, useEvalAccess } from "./shared"
import { EvalExecutionProfileSummary, ScenarioCatalog } from "./technical-details"

export function EvalSettingsViewPage() {
  const { canWrite, canReadDetails, readiness } = useEvalAccess()
  const client = useQueryClient()
  const go = useGo()
  const [targetKey, setTargetKey] = useState("")
  const input = useRef<HTMLInputElement>(null)
  const targets = useQuery({
    queryKey: EVAL_TARGET_QUERY_KEY,
    queryFn: ({ signal }) => fetchEvalTargets(signal),
    enabled: canReadDetails,
  })
  const selected = targetKey || targets.data?.default_target_key
  const target = targets.data?.items.find((t) => t.target_key === selected)
  const action = useMutation({
    mutationFn: async (file: File) => {
      await preflightEvalCorpusFile(file)
      return importEvalCorpus(file)
    },
    onSuccess: () => client.invalidateQueries({ queryKey: ["evals"] }),
  })
  const scenario = useMutation({
    mutationFn: (task: {
      name: string
      run: (signal: AbortSignal) => Promise<string | undefined>
    }) => task.run(new AbortController().signal),
  })
  return (
    <div className="flex min-h-0 min-w-0 flex-1 flex-col">
      <TopBar>
        <NavLink to="/evals" className="mr-2 text-muted-foreground">
          Evals /
        </NavLink>
        Settings
      </TopBar>
      <PageBody width="max-w-3xl">
        {!canReadDetails && (
          <p className="mt-4 text-sm text-muted-foreground">
            Your role can read eval suites and comparisons. Execution configuration and advanced
            tools require developer access.
          </p>
        )}
        {canReadDetails && (
          <LoadState
            error={targets.error}
            pending={canReadDetails && targets.isPending}
            retry={targets.refetch}
          />
        )}
        {canReadDetails && targets.data && (
          <>
            <Section title="What trials run with">
              {targets.data.items.length === 0 && (
                <p className="text-sm text-muted-foreground">
                  No specialists are configured for evals.
                </p>
              )}
              <div className="overflow-hidden rounded-xl border border-border">
                {targets.data.items.map((t) => (
                  <div
                    key={t.target_key}
                    className="border-b border-border px-3 py-2 text-sm break-words last:border-0"
                  >
                    <p className="font-medium">{t.target_key}</p>
                    <p className="text-muted-foreground">
                      {t.execution_profile
                        ? `${t.execution_profile.candidate.model} via ${t.execution_profile.candidate.provider_name}${t.execution_profile.candidate.environment_name ? ` · ${t.execution_profile.candidate.environment_name}` : ""}`
                        : "Execution profile unavailable"}
                    </p>
                  </div>
                ))}
              </div>
            </Section>
            {targets.data.items.length > 0 && (
              <Section title="Specialist">
                <select
                  aria-label="Settings specialist"
                  className={`${FIELD_CLASS} w-full`}
                  value={selected}
                  onChange={(e) => setTargetKey(e.target.value)}
                >
                  {targets.data.items.map((t) => (
                    <option key={t.target_key} value={t.target_key}>
                      {t.target_key}
                    </option>
                  ))}
                </select>
              </Section>
            )}
            {target && (
              <>
                <Section title="Judge">
                  {target.judge_profiles?.length ? (
                    <div className="space-y-3">
                      {target.judge_profiles.map((p) => (
                        <div
                          key={p.revision}
                          className="rounded-xl border border-border p-3 text-sm break-words"
                        >
                          <p className="font-medium">{p.label || p.key}</p>
                          <p className="mt-1 text-muted-foreground">
                            {p.model} via {p.provider_name}
                          </p>
                          <p className="mt-1 text-xs text-muted-foreground">
                            {p.same_model_use === "allowed_and_labeled"
                              ? "Same model allowed and labelled"
                              : p.same_model_use === "forbidden"
                                ? "Same model forbidden"
                                : "Same-model policy unavailable"}
                          </p>
                          <details className="mt-3">
                            <summary className="cursor-pointer text-xs">Judge details</summary>
                            <PayloadViewer value={p} />
                          </details>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="text-sm text-muted-foreground">No judge profile is configured.</p>
                  )}
                </Section>
                <Section title="Limits">
                  <p className="text-sm">
                    {target.execution_profile
                      ? `Up to ${target.execution_profile.ceilings.max_cases} cases, ${target.execution_profile.ceilings.max_trials} trials and ${target.execution_profile.ceilings.max_concurrency} concurrent trials; ${target.execution_profile.ceilings.max_timeout_seconds}s timeout.`
                      : "Limits unavailable without an execution profile."}
                  </p>
                  <details className="mt-4">
                    <summary className="cursor-pointer text-sm">Execution profile details</summary>
                    <div className="mt-3">
                      <EvalExecutionProfileSummary
                        target={target}
                        refreshing={targets.isFetching}
                        refresh={() => void targets.refetch()}
                      />
                    </div>
                    <PayloadViewer value={target} />
                  </details>
                </Section>
              </>
            )}
          </>
        )}
        <Section title="Available in this deployment">
          <ul className="space-y-3">
            {EVALS_READINESS_OPERATIONS.map(([key, label]) => (
              <li key={key} className="text-sm">
                <span
                  className={readiness[key].state === "ready" ? "text-success" : "text-warning"}
                >
                  {readiness[key].state === "ready" ? "✓" : "–"}
                </span>{" "}
                {key === "fresh_launch" ? "Fresh trials" : label}
                {readiness[key].state !== "ready" && (
                  <details className="ml-4 mt-1 text-xs text-muted-foreground">
                    <summary className="cursor-pointer">Why unavailable</summary>
                    <p className="mt-1 break-words">{evalsReadinessReasonText(readiness[key])}</p>
                  </details>
                )}
              </li>
            ))}
          </ul>
        </Section>
        {canWrite && (
          <Section title="Advanced tools">
            <div className="flex flex-wrap gap-2">
              <Button
                variant="outline"
                size="sm"
                disabled={action.isPending || readiness.catalog_write.state !== "ready"}
                onClick={() => input.current?.click()}
              >
                {action.isPending ? "Importing…" : "Import corpus"}
              </Button>
              <input
                ref={input}
                type="file"
                accept="application/json,.json"
                className="hidden"
                aria-label="Import corpus file"
                onChange={(e) => {
                  const file = e.target.files?.[0]
                  e.target.value = ""
                  if (file) action.mutate(file)
                }}
              />
              <MemoryExperimentReportAction disabled={false} />
              {target && (
                <EvalSuiteAuthoringAction
                  targetKey={target.target_key}
                  disabled={readiness.catalog_write.state !== "ready"}
                  label="Suites, checks and judges"
                  onLaunched={async (ids) => {
                    if (ids[0]) await go(runViewPath(nativeRunSelector(ids[0])))
                  }}
                />
              )}
            </div>
            <LoadState
              error={action.error ?? scenario.error}
              pending={false}
              retry={() => {
                action.reset()
                scenario.reset()
              }}
            />
            {action.isSuccess && (
              <p role="status" className="mt-3 text-sm">
                Corpus imported.{" "}
                <NavLink to="/evals" className="underline">
                  Open suites
                </NavLink>
              </p>
            )}
            {target && (
              <details className="mt-5">
                <summary className="cursor-pointer text-sm">Production scenarios</summary>
                <div className="mt-3">
                  <ScenarioCatalog
                    key={target.target_key}
                    targetKey={target.target_key}
                    pendingAction={scenario.isPending ? scenario.variables.name : null}
                    mutateEnabled={canWrite}
                    runAction={async (name, run) => {
                      await scenario.mutateAsync({ name, run })
                    }}
                  />
                </div>
              </details>
            )}
          </Section>
        )}
      </PageBody>
    </div>
  )
}
