import { useQuery, useQueryClient } from "@tanstack/react-query"
import {
  AlertTriangle,
  ArrowDown,
  ArrowUp,
  CheckCircle2,
  FileCheck2,
  LoaderCircle,
  Play,
  Plus,
  Save,
  Trash2,
} from "lucide-react"
import { useCallback, useEffect, useId, useMemo, useRef, useState } from "react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import {
  ApiClientError,
  type EvalRun,
  type EvalScenario,
  type EvalScenarioDraft,
  type EvalScenarioPreview,
  type EvalTarget,
  fetchEnvironments,
  fetchEvalTargets,
  launchEvalScenario,
  materializeEvalScenarioArtifact,
  previewEvalScenario,
  saveEvalScenario,
} from "@/lib/api"
import { dashboardConfig } from "@/lib/config"
import { scenarioArtifactBindingsRequireMaterialization } from "@/lib/eval-suite-authoring"
import {
  EvalLaunchIdempotencyRegistry,
  evalErrorMessage,
  evalLaunchFailureIsDefinitive,
  evalLaunchNotice,
  scenarioEvalLaunchRequestIdentity,
  shortEvalIdentity,
} from "@/lib/evals-dashboard"
import type {
  ScenarioInputV2,
  ScenarioJsonPartV2,
  ScenarioLaunchSettingsV2,
  ScenarioSecretRequirementV2,
} from "@/lib/generated/server-api"

const SELECT_CLASS =
  "h-8 w-full rounded-lg border border-input bg-background px-2.5 text-sm outline-none focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50"
const LABEL_CLASS = "mb-1 block text-xs font-medium text-muted-foreground"

type ScenarioEvent = EvalScenarioDraft["events"][number]
type ScenarioInputEvent = Exclude<ScenarioEvent, { tool_name: string }>
type ScenarioMessage = ScenarioInputV2["messages"][number]
type ScenarioPart = ScenarioMessage["content"][number]
type InputEventKind = "initial" | "queued" | "resumed"

export type ScenarioAuthoringState = Readonly<{
  dirty: boolean
  pending: boolean
}>

type ScenarioSettingsDraft = {
  environmentName: string
  trials: string
  maxConcurrency: string
  timeoutSeconds: string
  maxSteps: string
  maxTotalTokens: string
  maxToolCalls: string
  maxElapsedSeconds: string
  maxEstimatedCost: string
  currency: string
  artifactReferences: Record<string, string>
}

const DEFAULT_SETTINGS: ScenarioSettingsDraft = Object.freeze({
  environmentName: "",
  trials: "1",
  maxConcurrency: "1",
  timeoutSeconds: "300",
  maxSteps: "",
  maxTotalTokens: "",
  maxToolCalls: "",
  maxElapsedSeconds: "",
  maxEstimatedCost: "",
  currency: "USD",
  artifactReferences: {},
})

const MAX_SAFE_RUNTIME_LIMIT = Number.MAX_SAFE_INTEGER

function clone<T>(value: T): T {
  return structuredClone(value)
}

function draftFromScenario(scenario: EvalScenario | EvalScenarioDraft): EvalScenarioDraft {
  if ("revision" in scenario) {
    const { revision: _revision, schema_version: _schemaVersion, ...draft } = scenario
    return clone(draft)
  }
  return clone(scenario)
}

function eventKind(event: ScenarioEvent): ScenarioEvent["kind"] {
  if ("tool_name" in event) return "approval_checkpoint"
  if (event.kind) return event.kind
  return event.sequence === 0 ? "initial" : "queued"
}

function partKind(part: ScenarioPart): "text" | "json" | "file" {
  if ("text" in part) return "text"
  if ("value" in part) return "json"
  return "file"
}

function normalizeEvents(events: ScenarioEvent[]): ScenarioEvent[] {
  return events.map((event, sequence) => ({ ...event, sequence }))
}

function nextPortableId(draft: EvalScenarioDraft, prefix: string): string {
  const ids = new Set(draft.events.map((event) => event.id))
  let index = draft.events.length + 1
  while (ids.has(`${prefix}-${index}`)) index += 1
  return `${prefix}-${index}`
}

function defaultInput(): ScenarioInputV2 {
  return { messages: [{ role: "user", content: [{ type: "text", text: "New input" }] }] }
}

function eventForKind(
  kind: Exclude<ScenarioEvent["kind"], "initial">,
  sequence: number,
  id: string,
): ScenarioEvent {
  if (kind === "approval_checkpoint") {
    return {
      kind,
      sequence,
      id,
      tool_name: "tool_name",
      occurrence: 1,
      resolution: "fresh_decision" as const,
    }
  }
  if (kind === "resumed") {
    return {
      kind,
      sequence,
      id,
      resume_kind: "user_input" as const,
      input: defaultInput(),
    }
  }
  return {
    kind: "queued" as const,
    sequence,
    id,
    delivery_mode: "next_turn" as const,
    input: defaultInput(),
  }
}

function defaultEvent(draft: EvalScenarioDraft, kind: Exclude<ScenarioEvent["kind"], "initial">) {
  const prefix =
    kind === "approval_checkpoint" ? "approval" : kind === "resumed" ? "resume" : "queued"
  return eventForKind(kind, draft.events.length, nextPortableId(draft, prefix))
}

function optionalPositiveInteger(value: string, maximum: number): number | null | undefined {
  if (value.trim() === "") return undefined
  const parsed = Number(value)
  if (!Number.isSafeInteger(parsed) || parsed < 1 || parsed > maximum) return null
  return parsed
}

function settingsContract(
  settings: ScenarioSettingsDraft,
  target: EvalTarget | undefined,
): ScenarioLaunchSettingsV2 | null {
  const trials = Number(settings.trials)
  const maxConcurrency = Number(settings.maxConcurrency)
  const timeoutSeconds = Number(settings.timeoutSeconds)
  const maxSteps = optionalPositiveInteger(settings.maxSteps, 256)
  const maxTotalTokens = optionalPositiveInteger(settings.maxTotalTokens, MAX_SAFE_RUNTIME_LIMIT)
  const maxToolCalls = optionalPositiveInteger(settings.maxToolCalls, MAX_SAFE_RUNTIME_LIMIT)
  const maxElapsedSeconds = optionalPositiveInteger(
    settings.maxElapsedSeconds,
    MAX_SAFE_RUNTIME_LIMIT,
  )
  if (
    !Number.isInteger(trials) ||
    trials < 1 ||
    trials > 100 ||
    !Number.isInteger(maxConcurrency) ||
    maxConcurrency < 1 ||
    maxConcurrency > 32 ||
    !Number.isInteger(timeoutSeconds) ||
    timeoutSeconds < 1 ||
    timeoutSeconds > 3_600 ||
    maxSteps === null ||
    maxTotalTokens === null ||
    maxToolCalls === null ||
    maxElapsedSeconds === null
  ) {
    return null
  }
  const hasLimits =
    maxTotalTokens !== undefined || maxToolCalls !== undefined || maxElapsedSeconds !== undefined
  const cost = settings.maxEstimatedCost.trim()
  let costBudget: ScenarioLaunchSettingsV2["cost_budget"]
  if (cost !== "") {
    const currency = settings.currency.trim().toUpperCase()
    if (
      !/^(?:0|[1-9]\d*)(?:\.\d+)?$/.test(cost) ||
      Number(cost) <= 0 ||
      target?.cost_budget_available !== true ||
      !target.cost_budget_currencies.includes(currency)
    ) {
      return null
    }
    costBudget = { max_estimated_cost: cost, currency }
  }
  const artifactReferences = Object.fromEntries(
    Object.entries(settings.artifactReferences)
      .map(([key, value]) => [key, value.trim()])
      .filter(([, value]) => value !== ""),
  )
  return {
    environment_name: settings.environmentName.trim() || null,
    trials,
    max_concurrency: maxConcurrency,
    timeout_seconds: timeoutSeconds,
    max_steps: maxSteps ?? null,
    limits: hasLimits
      ? {
          scope: "run",
          max_total_tokens: maxTotalTokens,
          max_tool_calls: maxToolCalls,
          max_elapsed_seconds: maxElapsedSeconds,
        }
      : null,
    cost_budget: costBudget ?? null,
    artifact_references: artifactReferences,
  }
}

export function ScenarioAuthoring({
  captured,
  disabled = false,
  saved = false,
  showLaunch = true,
  showLaunchSettings = true,
  onSaved,
  onAuthoringStateChange,
}: {
  captured: EvalScenario | EvalScenarioDraft
  disabled?: boolean
  saved?: boolean
  showLaunch?: boolean
  showLaunchSettings?: boolean
  onSaved?: (scenario: EvalScenario) => void
  onAuthoringStateChange?: (state: ScenarioAuthoringState) => void
}) {
  const queryClient = useQueryClient()
  const formId = useId()
  const [draft, setDraft] = useState(() => draftFromScenario(captured))
  const [settings, setSettings] = useState<ScenarioSettingsDraft>(() => ({
    ...DEFAULT_SETTINGS,
    artifactReferences: {},
  }))
  const [preview, setPreview] = useState<EvalScenarioPreview | null>(null)
  const [previewIdentity, setPreviewIdentity] = useState<string | null>(null)
  const [pending, setPending] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const capturedRevision = "revision" in captured ? captured.revision : null
  const [savedRevision, setSavedRevision] = useState<string | null>(saved ? capturedRevision : null)
  const [savedDraftIdentity, setSavedDraftIdentity] = useState<string | null>(() =>
    saved ? JSON.stringify(draftFromScenario(captured)) : null,
  )
  const [invalidJsonEditors, setInvalidJsonEditors] = useState(0)
  const controllerRef = useRef<AbortController | null>(null)
  const launchRegistryRef = useRef<EvalLaunchIdempotencyRegistry | null>(null)
  const environments = useQuery({
    queryKey: ["environments"],
    queryFn: fetchEnvironments,
    staleTime: 15_000,
    enabled: showLaunchSettings,
  })
  const targets = useQuery({
    queryKey: ["evals", "targets"],
    queryFn: ({ signal }) => fetchEvalTargets(signal),
    staleTime: Number.POSITIVE_INFINITY,
  })
  const selectedTarget = targets.data?.items.find(
    (target) => target.target_key === draft.target_key,
  )
  const currentIdentity = useMemo(() => JSON.stringify([draft, settings]), [draft, settings])
  const draftIdentity = useMemo(() => JSON.stringify(draft), [draft])
  const hasUnmaterializedArtifactBindings = scenarioArtifactBindingsRequireMaterialization(
    settings.artifactReferences,
    !showLaunchSettings,
  )
  const dirty =
    savedDraftIdentity === null ||
    draftIdentity !== savedDraftIdentity ||
    invalidJsonEditors > 0 ||
    hasUnmaterializedArtifactBindings
  const previewCurrent =
    preview !== null && previewIdentity === currentIdentity && invalidJsonEditors === 0
  const settingsValue = settingsContract(settings, selectedTarget)

  useEffect(() => {
    controllerRef.current?.abort()
    const nextDraft = draftFromScenario(captured)
    setDraft(nextDraft)
    setSettings({ ...DEFAULT_SETTINGS, artifactReferences: {} })
    setPreview(null)
    setPreviewIdentity(null)
    setError(null)
    setNotice(null)
    setSavedRevision(saved ? capturedRevision : null)
    setSavedDraftIdentity(saved ? JSON.stringify(nextDraft) : null)
  }, [captured, capturedRevision, saved])
  useEffect(() => {
    if (!selectedTarget) return
    setSettings((current) => {
      if (!selectedTarget.cost_budget_available) {
        return current.maxEstimatedCost === "" ? current : { ...current, maxEstimatedCost: "" }
      }
      if (selectedTarget.cost_budget_currencies.includes(current.currency)) return current
      const currency = selectedTarget.cost_budget_currencies.includes("USD")
        ? "USD"
        : (selectedTarget.cost_budget_currencies[0] ?? current.currency)
      return currency === current.currency ? current : { ...current, currency }
    })
  }, [selectedTarget])
  useEffect(() => {
    onAuthoringStateChange?.({ dirty, pending: pending !== null })
  }, [dirty, onAuthoringStateChange, pending])
  useEffect(() => () => controllerRef.current?.abort(), [])

  const run = async (name: string, action: (signal: AbortSignal) => Promise<void>) => {
    controllerRef.current?.abort()
    const controller = new AbortController()
    controllerRef.current = controller
    setPending(name)
    setError(null)
    setNotice(null)
    try {
      await action(controller.signal)
    } catch (actionError) {
      if (!controller.signal.aborted) {
        setError(evalErrorMessage(actionError, "The scenario operation failed."))
      }
    } finally {
      if (controllerRef.current === controller) {
        controllerRef.current = null
        setPending(null)
      }
    }
  }

  const checkReadiness = () => {
    if (invalidJsonEditors > 0) {
      setError(
        "Every structured JSON part must contain valid JSON before readiness can be checked.",
      )
      return
    }
    if (settingsValue === null) {
      setError(
        "Launch bounds must be positive supported values, and any cost budget must use current target pricing.",
      )
      return
    }
    const identity = currentIdentity
    void run("preview", async (signal) => {
      const next = await previewEvalScenario({ draft, settings: settingsValue }, signal)
      if (signal.aborted) return
      setPreview(next)
      setPreviewIdentity(identity)
    })
  }

  const reportJsonValidity = useCallback((valid: boolean) => {
    setInvalidJsonEditors((current) => Math.max(0, current + (valid ? -1 : 1)))
  }, [])

  const save = () => {
    if (
      !previewCurrent ||
      preview === null ||
      settingsValue === null ||
      hasUnmaterializedArtifactBindings
    ) {
      return
    }
    void run("save", async (signal) => {
      const saved = await saveEvalScenario(
        {
          expected_scenario_revision: preview.scenario.revision,
          scenario: preview.scenario,
          settings: settingsValue,
        },
        signal,
      )
      if (signal.aborted) return
      const normalizedDraft = draftFromScenario(saved.scenario)
      const normalizedIdentity = JSON.stringify([normalizedDraft, settings])
      setDraft(normalizedDraft)
      setPreview({ scenario: saved.scenario, preflight: saved.preflight })
      setPreviewIdentity(normalizedIdentity)
      setSavedRevision(saved.scenario.revision)
      setSavedDraftIdentity(JSON.stringify(normalizedDraft))
      setNotice(`Saved scenario ${shortEvalIdentity(saved.entry.revision)}.`)
      onSaved?.(saved.scenario)
      await queryClient.invalidateQueries({ queryKey: ["evals", "scenarios"] })
    })
  }

  const launch = () => {
    const binding = preview?.preflight.binding
    if (
      !previewCurrent ||
      preview === null ||
      !preview.preflight.ready ||
      binding === null ||
      binding === undefined ||
      settingsValue === null ||
      savedRevision !== preview.scenario.revision
    ) {
      return
    }
    const requestIdentity = scenarioEvalLaunchRequestIdentity(
      preview.scenario.revision,
      binding.revision,
    )
    void run("launch", async (signal) => {
      const registry =
        launchRegistryRef.current ??
        new EvalLaunchIdempotencyRegistry(window.sessionStorage, dashboardConfig.apiBaseUrl)
      launchRegistryRef.current = registry
      const idempotencyKey = registry.keyFor(requestIdentity)
      let launched: EvalRun
      try {
        launched = await launchEvalScenario(
          preview.scenario.revision,
          {
            expected_binding_revision: binding.revision,
            settings: settingsValue,
          },
          idempotencyKey,
          signal,
        )
      } catch (launchError) {
        if (
          launchError instanceof ApiClientError &&
          evalLaunchFailureIsDefinitive(launchError.status)
        ) {
          registry.resolve(requestIdentity)
        }
        throw launchError
      }
      if (signal.aborted) return
      queryClient.setQueryData(["evals", "run", launched.spec.run_id], launched)
      await queryClient.invalidateQueries({ queryKey: ["evals", "runs"] })
      if (signal.aborted) return
      registry.resolve(requestIdentity)
      setNotice(`${evalLaunchNotice(launched)} Follow it in the Runs tab.`)
    })
  }

  const prepareArtifact = (requirementId: string) => {
    if (!previewCurrent || preview === null || settingsValue === null) return
    void run(`artifact:${requirementId}`, async (signal) => {
      const prepared = await materializeEvalScenarioArtifact(
        requirementId,
        {
          expected_scenario_revision: preview.scenario.revision,
          scenario: preview.scenario,
          settings: settingsValue,
        },
        signal,
      )
      if (signal.aborted) return
      const nextDraft = draftFromScenario(prepared.materialization.scenario)
      const nextSettings = {
        ...settings,
        artifactReferences: Object.fromEntries(
          Object.entries(settings.artifactReferences).filter(([key]) => key !== requirementId),
        ),
      }
      const nextIdentity = JSON.stringify([nextDraft, nextSettings])
      setDraft(nextDraft)
      setSettings(nextSettings)
      setPreview({
        scenario: prepared.materialization.scenario,
        preflight: prepared.preflight,
      })
      setPreviewIdentity(nextIdentity)
      setNotice(`Prepared reusable fixture ${prepared.materialization.artifact_id}.`)
    })
  }

  const editEvent = (index: number, next: ScenarioEvent) => {
    setDraft((current) => {
      const events = [...current.events]
      events[index] = next
      return { ...current, events: normalizeEvents(events) }
    })
  }
  const moveEvent = (index: number, direction: -1 | 1) => {
    const destination = index + direction
    if (index === 0 || destination < 1 || destination >= draft.events.length) return
    setDraft((current) => {
      const events = [...current.events]
      const source = events[index]
      const target = events[destination]
      if (source === undefined || target === undefined) return current
      events[index] = target
      events[destination] = source
      return { ...current, events: normalizeEvents(events) }
    })
  }
  const removeEvent = (index: number) => {
    if (index === 0) return
    setDraft((current) => ({
      ...current,
      events: normalizeEvents(current.events.filter((_, eventIndex) => eventIndex !== index)),
    }))
  }

  return (
    <Card size="sm" data-testid="scenario-authoring">
      <CardHeader className="grid-cols-[1fr_auto]">
        <div>
          <CardTitle>Production scenario</CardTitle>
          <p className="mt-1 text-xs text-muted-foreground">
            Edit ordered external inputs, then verify current target authority before saving.
          </p>
        </div>
        <Badge variant={previewCurrent && preview.preflight.ready ? "secondary" : "outline"}>
          {previewCurrent && preview.preflight.ready
            ? "Ready"
            : previewCurrent
              ? "Needs attention"
              : "Not checked"}
        </Badge>
      </CardHeader>
      <CardContent className="space-y-4">
        <fieldset className="space-y-4" disabled={disabled || pending !== null}>
          <div className="grid gap-3 md:grid-cols-2">
            <label htmlFor={`${formId}-name`}>
              <span className={LABEL_CLASS}>Scenario name</span>
              <Input
                id={`${formId}-name`}
                value={draft.name}
                onChange={(event) =>
                  setDraft((current) => ({ ...current, name: event.target.value }))
                }
              />
            </label>
            <label htmlFor={`${formId}-id`}>
              <span className={LABEL_CLASS}>Scenario ID</span>
              <Input
                id={`${formId}-id`}
                value={draft.id}
                onChange={(event) =>
                  setDraft((current) => ({ ...current, id: event.target.value }))
                }
              />
            </label>
          </div>
          <label htmlFor={`${formId}-description`}>
            <span className={LABEL_CLASS}>Description</span>
            <Textarea
              id={`${formId}-description`}
              rows={2}
              value={draft.description ?? ""}
              onChange={(event) =>
                setDraft((current) => ({
                  ...current,
                  description: event.target.value || null,
                }))
              }
            />
          </label>

          <div className="space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div>
                <div className="text-sm font-medium">Ordered events</div>
                <div className="text-xs text-muted-foreground">
                  The first event creates a fresh session. Later events queue, resume, or require a
                  new approval.
                </div>
              </div>
              <div className="flex flex-wrap gap-1.5">
                <Button
                  type="button"
                  size="xs"
                  variant="outline"
                  onClick={() =>
                    setDraft((current) => ({
                      ...current,
                      events: [...current.events, defaultEvent(current, "queued")],
                    }))
                  }
                >
                  <Plus /> Queue input
                </Button>
                <Button
                  type="button"
                  size="xs"
                  variant="outline"
                  onClick={() =>
                    setDraft((current) => ({
                      ...current,
                      events: [...current.events, defaultEvent(current, "resumed")],
                    }))
                  }
                >
                  <Plus /> Resume input
                </Button>
                <Button
                  type="button"
                  size="xs"
                  variant="outline"
                  onClick={() =>
                    setDraft((current) => ({
                      ...current,
                      events: [...current.events, defaultEvent(current, "approval_checkpoint")],
                    }))
                  }
                >
                  <Plus /> Approval
                </Button>
              </div>
            </div>
            {draft.events.map((event, index) => (
              <ScenarioEventEditor
                key={`${capturedRevision ?? "draft"}:${event.id}`}
                event={event}
                index={index}
                eventCount={draft.events.length}
                artifactRequirementIds={(draft.artifact_requirements ?? []).map((item) => item.id)}
                reportJsonValidity={reportJsonValidity}
                edit={(next) => editEvent(index, next)}
                move={(direction) => moveEvent(index, direction)}
                remove={() => removeEvent(index)}
              />
            ))}
          </div>

          <ScenarioRequirementsEditor draft={draft} setDraft={setDraft} />

          <div className={showLaunchSettings ? "rounded-lg border border-border p-3" : "hidden"}>
            <div className="mb-3">
              <div className="text-sm font-medium">Current launch selections</div>
              <div className="text-xs text-muted-foreground">
                These are narrowed by the server-owned target profile; they are not execution
                authority.
              </div>
            </div>
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6">
              <label>
                <span className={LABEL_CLASS}>Target profile</span>
                <select
                  className={SELECT_CLASS}
                  value={draft.target_key}
                  onChange={(event) =>
                    setDraft((current) => ({ ...current, target_key: event.target.value }))
                  }
                >
                  {!targets.data?.items.some(
                    (target) => target.target_key === draft.target_key,
                  ) && <option value={draft.target_key}>{draft.target_key}</option>}
                  {(targets.data?.items ?? []).map((target) => (
                    <option key={target.target_key} value={target.target_key}>
                      {target.label}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                <span className={LABEL_CLASS}>Environment</span>
                <select
                  className={SELECT_CLASS}
                  value={settings.environmentName}
                  onChange={(event) =>
                    setSettings((current) => ({
                      ...current,
                      environmentName: event.target.value,
                    }))
                  }
                >
                  <option value="">Target default</option>
                  {(environments.data?.environments ?? []).map((environment) => (
                    <option key={environment.name} value={environment.name}>
                      {environment.name}
                    </option>
                  ))}
                </select>
              </label>
              <NumberSetting
                label="Trials"
                value={settings.trials}
                maximum={selectedTarget?.max_trials ?? 100}
                edit={(trials) => setSettings((current) => ({ ...current, trials }))}
              />
              <NumberSetting
                label="Concurrency"
                value={settings.maxConcurrency}
                maximum={selectedTarget?.max_concurrency ?? 32}
                edit={(maxConcurrency) =>
                  setSettings((current) => ({ ...current, maxConcurrency }))
                }
              />
              <NumberSetting
                label="Timeout seconds"
                value={settings.timeoutSeconds}
                maximum={selectedTarget?.max_timeout_seconds ?? 3_600}
                edit={(timeoutSeconds) =>
                  setSettings((current) => ({ ...current, timeoutSeconds }))
                }
              />
              <NumberSetting
                label="Max steps"
                value={settings.maxSteps}
                maximum={selectedTarget?.max_steps ?? 256}
                optional
                edit={(maxSteps) => setSettings((current) => ({ ...current, maxSteps }))}
              />
            </div>
            <div className="mt-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              <NumberSetting
                label="Max total tokens per trial"
                value={settings.maxTotalTokens}
                maximum={MAX_SAFE_RUNTIME_LIMIT}
                optional
                edit={(maxTotalTokens) =>
                  setSettings((current) => ({ ...current, maxTotalTokens }))
                }
              />
              <NumberSetting
                label="Max tool calls per trial"
                value={settings.maxToolCalls}
                maximum={MAX_SAFE_RUNTIME_LIMIT}
                optional
                edit={(maxToolCalls) => setSettings((current) => ({ ...current, maxToolCalls }))}
              />
              <NumberSetting
                label="Max run time per trial"
                value={settings.maxElapsedSeconds}
                maximum={MAX_SAFE_RUNTIME_LIMIT}
                optional
                edit={(maxElapsedSeconds) =>
                  setSettings((current) => ({ ...current, maxElapsedSeconds }))
                }
              />
              <label>
                <span className={LABEL_CLASS}>Max estimated cost per trial</span>
                <div className="grid grid-cols-[minmax(0,1fr)_5rem] gap-2">
                  <Input
                    inputMode="decimal"
                    placeholder={selectedTarget?.cost_budget_available ? "Optional" : "No pricing"}
                    value={settings.maxEstimatedCost}
                    disabled={selectedTarget?.cost_budget_available !== true}
                    onChange={(event) =>
                      setSettings((current) => ({
                        ...current,
                        maxEstimatedCost: event.target.value,
                      }))
                    }
                  />
                  <select
                    aria-label="Scenario cost budget currency"
                    className={SELECT_CLASS}
                    value={settings.currency}
                    disabled={selectedTarget?.cost_budget_available !== true}
                    onChange={(event) =>
                      setSettings((current) => ({
                        ...current,
                        currency: event.target.value.toUpperCase(),
                      }))
                    }
                  >
                    {(selectedTarget?.cost_budget_currencies ?? []).map((currency) => (
                      <option key={currency} value={currency}>
                        {currency}
                      </option>
                    ))}
                  </select>
                </div>
              </label>
            </div>
          </div>

          {(draft.artifact_requirements ?? []).length > 0 && (
            <div className="space-y-2 rounded-lg border border-border p-3">
              <div className="text-sm font-medium">Artifact bindings</div>
              {!showLaunchSettings && (
                <p
                  className="text-xs text-muted-foreground"
                  data-testid="scenario-artifact-materialization-guidance"
                >
                  Prepare each artifact override as a reusable fixture before saving this scenario
                  into the suite.
                </p>
              )}
              {(draft.artifact_requirements ?? []).map((requirement) => {
                const diagnostic = preview?.preflight.diagnostics?.find(
                  (item) => item.requirement_id === requirement.id,
                )
                return (
                  <div key={requirement.id} className="grid gap-2 md:grid-cols-[1fr_1fr_auto]">
                    <div className="min-w-0">
                      <div className="truncate text-sm font-medium">{requirement.filename}</div>
                      <div className="truncate font-mono text-xs text-muted-foreground">
                        {requirement.id} · {requirement.content_sha256.slice(0, 12)}…
                      </div>
                    </div>
                    <Input
                      aria-label={`Artifact reference for ${requirement.id}`}
                      placeholder={requirement.reference ?? "Select retained artifact ID"}
                      value={settings.artifactReferences[requirement.id] ?? ""}
                      onChange={(event) =>
                        setSettings((current) => ({
                          ...current,
                          artifactReferences: {
                            ...current.artifactReferences,
                            [requirement.id]: event.target.value,
                          },
                        }))
                      }
                    />
                    <Button
                      type="button"
                      size="sm"
                      variant="outline"
                      disabled={
                        !previewCurrent ||
                        pending !== null ||
                        (requirement.reference == null &&
                          !(settings.artifactReferences[requirement.id] ?? "").trim())
                      }
                      onClick={() => prepareArtifact(requirement.id)}
                    >
                      {pending === `artifact:${requirement.id}` ? (
                        <LoaderCircle className="animate-spin" />
                      ) : (
                        <FileCheck2 />
                      )}
                      Prepare fixture
                    </Button>
                    {diagnostic && (
                      <div className="text-xs text-amber-700 md:col-span-3 dark:text-amber-300">
                        {diagnostic.message} {diagnostic.remediation}
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          )}
        </fieldset>

        {previewCurrent && <ScenarioPreflightSummary preview={preview} />}
        {error && (
          <div
            className="rounded-lg border border-destructive/30 bg-destructive/5 p-3 text-sm text-destructive"
            role="alert"
          >
            {error}
          </div>
        )}
        {notice && (
          <div
            className="rounded-lg border border-emerald-500/30 bg-emerald-500/5 p-3 text-sm text-emerald-700 dark:text-emerald-300"
            role="status"
          >
            {notice}
          </div>
        )}
        <div className="flex flex-wrap justify-end gap-2">
          <Button
            type="button"
            variant="outline"
            disabled={disabled || pending !== null}
            onClick={checkReadiness}
          >
            {pending === "preview" ? <LoaderCircle className="animate-spin" /> : <CheckCircle2 />}
            {pending === "preview" ? "Checking..." : "Check readiness"}
          </Button>
          <Button
            type="button"
            disabled={
              disabled || pending !== null || !previewCurrent || hasUnmaterializedArtifactBindings
            }
            onClick={save}
          >
            {pending === "save" ? <LoaderCircle className="animate-spin" /> : <Save />}
            {pending === "save" ? "Saving..." : "Save scenario"}
          </Button>
          {showLaunch && (
            <Button
              type="button"
              disabled={
                disabled ||
                pending !== null ||
                !previewCurrent ||
                !preview?.preflight.ready ||
                preview.preflight.binding == null ||
                savedRevision !== preview.scenario.revision
              }
              onClick={launch}
            >
              {pending === "launch" ? <LoaderCircle className="animate-spin" /> : <Play />}
              {pending === "launch" ? "Starting..." : "Run scenario"}
            </Button>
          )}
        </div>
      </CardContent>
    </Card>
  )
}

function ScenarioEventEditor({
  event,
  index,
  eventCount,
  artifactRequirementIds,
  reportJsonValidity,
  edit,
  move,
  remove,
}: {
  event: ScenarioEvent
  index: number
  eventCount: number
  artifactRequirementIds: string[]
  reportJsonValidity: (valid: boolean) => void
  edit: (event: ScenarioEvent) => void
  move: (direction: -1 | 1) => void
  remove: () => void
}) {
  const editorId = useId()
  const kind = eventKind(event)
  return (
    <div className="rounded-lg border border-border p-3" data-testid={`scenario-event-${index}`}>
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <Badge variant="outline">{index + 1}</Badge>
        <select
          className={`${SELECT_CLASS} w-44`}
          value={kind}
          disabled={index === 0}
          onChange={(change) => {
            const nextKind = change.target.value as Exclude<ScenarioEvent["kind"], "initial">
            edit(eventForKind(nextKind, event.sequence, event.id))
          }}
        >
          {index === 0 && <option value="initial">Initial input</option>}
          {index !== 0 && <option value="queued">Queued input</option>}
          {index !== 0 && <option value="resumed">Resumed input</option>}
          {index !== 0 && <option value="approval_checkpoint">Approval checkpoint</option>}
        </select>
        <span className="min-w-40 flex-1 truncate font-mono text-xs text-muted-foreground">
          {event.id}
        </span>
        {index > 0 && (
          <>
            <Button
              type="button"
              size="icon-xs"
              variant="ghost"
              aria-label={`Move event ${index + 1} up`}
              disabled={index === 1}
              onClick={() => move(-1)}
            >
              <ArrowUp />
            </Button>
            <Button
              type="button"
              size="icon-xs"
              variant="ghost"
              aria-label={`Move event ${index + 1} down`}
              disabled={index === eventCount - 1}
              onClick={() => move(1)}
            >
              <ArrowDown />
            </Button>
            <Button
              type="button"
              size="icon-xs"
              variant="ghost"
              aria-label={`Remove event ${index + 1}`}
              onClick={remove}
            >
              <Trash2 />
            </Button>
          </>
        )}
      </div>
      {kind === "approval_checkpoint" && "tool_name" in event ? (
        <div className="grid gap-3 sm:grid-cols-2">
          <label htmlFor={`${editorId}-tool`}>
            <span className={LABEL_CLASS}>Current tool name</span>
            <Input
              id={`${editorId}-tool`}
              value={event.tool_name}
              onChange={(change) => edit({ ...event, tool_name: change.target.value })}
            />
          </label>
          <label htmlFor={`${editorId}-occurrence`}>
            <span className={LABEL_CLASS}>Occurrence</span>
            <Input
              id={`${editorId}-occurrence`}
              type="number"
              min={1}
              value={event.occurrence ?? 1}
              onChange={(change) => edit({ ...event, occurrence: Number(change.target.value) })}
            />
          </label>
          <p className="text-xs text-muted-foreground sm:col-span-2">
            The rerun must pause for a fresh decision. No historical approval is reused.
          </p>
        </div>
      ) : "input" in event ? (
        <ScenarioInputEditor
          event={event}
          kind={kind as InputEventKind}
          artifactRequirementIds={artifactRequirementIds}
          reportJsonValidity={reportJsonValidity}
          edit={edit}
        />
      ) : null}
    </div>
  )
}

function ScenarioInputEditor({
  event,
  kind,
  artifactRequirementIds,
  reportJsonValidity,
  edit,
}: {
  event: ScenarioInputEvent
  kind: InputEventKind
  artifactRequirementIds: string[]
  reportJsonValidity: (valid: boolean) => void
  edit: (event: ScenarioEvent) => void
}) {
  const editInput = (input: ScenarioInputV2) => edit({ ...event, input })
  return (
    <div className="space-y-3">
      {kind === "queued" && "delivery_mode" in event && (
        <label className="block max-w-56">
          <span className={LABEL_CLASS}>Delivery</span>
          <select
            className={SELECT_CLASS}
            value={event.delivery_mode}
            onChange={(change) =>
              edit({ ...event, delivery_mode: change.target.value as "next_turn" | "on_idle" })
            }
          >
            <option value="next_turn">Next turn</option>
            <option value="on_idle">When idle</option>
          </select>
        </label>
      )}
      {kind === "resumed" && "resume_kind" in event && (
        <label className="block max-w-56">
          <span className={LABEL_CLASS}>Resume kind</span>
          <select
            className={SELECT_CLASS}
            value={event.resume_kind ?? "user_input"}
            onChange={(change) =>
              edit({
                ...event,
                resume_kind: change.target.value as "user_input" | "manual_recovery",
              })
            }
          >
            <option value="user_input">User input</option>
            <option value="manual_recovery">Session resume</option>
          </select>
        </label>
      )}
      {event.input.messages.map((message, messageIndex) => (
        // biome-ignore lint/suspicious/noArrayIndexKey: ordered scenario messages have no portable UI identity.
        <div key={messageIndex} className="rounded-md bg-muted/30 p-2.5">
          <div className="mb-2 flex items-center justify-between gap-2">
            <span className="text-xs font-medium">User message {messageIndex + 1}</span>
            <Button
              type="button"
              size="icon-xs"
              variant="ghost"
              aria-label={`Remove message ${messageIndex + 1}`}
              disabled={event.input.messages.length === 1}
              onClick={() =>
                editInput({
                  messages: event.input.messages.filter((_, index) => index !== messageIndex),
                })
              }
            >
              <Trash2 />
            </Button>
          </div>
          <div className="space-y-2">
            {message.content.map((part, partIndex) => (
              <ScenarioPartEditor
                // biome-ignore lint/suspicious/noArrayIndexKey: ordered scenario parts have no portable UI identity.
                key={partIndex}
                part={part}
                artifactRequirementIds={artifactRequirementIds}
                reportJsonValidity={reportJsonValidity}
                edit={(nextPart) => {
                  const messages = clone(event.input.messages)
                  const currentMessage = messages[messageIndex]
                  if (currentMessage === undefined) return
                  currentMessage.content[partIndex] = nextPart
                  editInput({ messages })
                }}
                remove={() => {
                  const messages = clone(event.input.messages)
                  const currentMessage = messages[messageIndex]
                  if (currentMessage === undefined) return
                  currentMessage.content = currentMessage.content.filter(
                    (_, index) => index !== partIndex,
                  )
                  editInput({ messages })
                }}
                removable={message.content.length > 1}
              />
            ))}
          </div>
          <Button
            type="button"
            size="xs"
            variant="ghost"
            className="mt-2"
            onClick={() => {
              const messages = clone(event.input.messages)
              const currentMessage = messages[messageIndex]
              if (currentMessage === undefined) return
              currentMessage.content.push({ type: "text", text: "New part" })
              editInput({ messages })
            }}
          >
            <Plus /> Add part
          </Button>
        </div>
      ))}
      <Button
        type="button"
        size="xs"
        variant="outline"
        onClick={() =>
          editInput({
            messages: [
              ...event.input.messages,
              { role: "user", content: [{ type: "text", text: "New message" }] },
            ],
          })
        }
      >
        <Plus /> Add message
      </Button>
    </div>
  )
}

function ScenarioPartEditor({
  part,
  artifactRequirementIds,
  reportJsonValidity,
  edit,
  remove,
  removable,
}: {
  part: ScenarioPart
  artifactRequirementIds: string[]
  reportJsonValidity: (valid: boolean) => void
  edit: (part: ScenarioPart) => void
  remove: () => void
  removable: boolean
}) {
  const kind = partKind(part)
  return (
    <div className="grid gap-2 sm:grid-cols-[8rem_minmax(0,1fr)_auto]">
      <select
        className={SELECT_CLASS}
        value={kind}
        onChange={(change) => {
          if (change.target.value === "json") edit({ type: "json", value: {} })
          else if (change.target.value === "file" && artifactRequirementIds[0] !== undefined) {
            edit({ type: "file", artifact_requirement_id: artifactRequirementIds[0] })
          } else edit({ type: "text", text: "New text" })
        }}
      >
        <option value="text">Text</option>
        <option value="json">JSON</option>
        <option value="file" disabled={artifactRequirementIds.length === 0}>
          File{artifactRequirementIds.length === 0 ? " (no retained requirement)" : ""}
        </option>
      </select>
      {kind === "text" && "text" in part ? (
        <Textarea
          rows={2}
          value={part.text}
          onChange={(change) => edit({ ...part, text: change.target.value })}
        />
      ) : kind === "json" && "value" in part ? (
        <JsonPartEditor part={part} edit={edit} reportValidity={reportJsonValidity} />
      ) : "artifact_requirement_id" in part ? (
        <select
          className={SELECT_CLASS}
          value={part.artifact_requirement_id}
          onChange={(change) => edit({ ...part, artifact_requirement_id: change.target.value })}
        >
          {artifactRequirementIds.length === 0 && <option value="file">No requirement yet</option>}
          {artifactRequirementIds.map((id) => (
            <option key={id} value={id}>
              {id}
            </option>
          ))}
        </select>
      ) : null}
      <Button
        type="button"
        size="icon-xs"
        variant="ghost"
        aria-label="Remove input part"
        disabled={!removable}
        onClick={remove}
      >
        <Trash2 />
      </Button>
    </div>
  )
}

function JsonPartEditor({
  part,
  edit,
  reportValidity,
}: {
  part: ScenarioJsonPartV2
  edit: (part: ScenarioPart) => void
  reportValidity: (valid: boolean) => void
}) {
  const serialized = JSON.stringify(part.value, null, 2)
  const [text, setText] = useState(() => serialized)
  const [invalid, setInvalid] = useState(false)
  const invalidRef = useRef(false)
  const lastEmittedRef = useRef<string | null>(null)
  const updateValidity = useCallback(
    (valid: boolean) => {
      const nextInvalid = !valid
      if (invalidRef.current !== nextInvalid) {
        invalidRef.current = nextInvalid
        reportValidity(valid)
      }
      setInvalid(nextInvalid)
    },
    [reportValidity],
  )
  useEffect(() => {
    if (lastEmittedRef.current === serialized) {
      lastEmittedRef.current = null
      return
    }
    setText(serialized)
    updateValidity(true)
  }, [serialized, updateValidity])
  useEffect(
    () => () => {
      if (invalidRef.current) reportValidity(true)
    },
    [reportValidity],
  )
  return (
    <Textarea
      rows={3}
      value={text}
      aria-invalid={invalid}
      onChange={(change) => {
        const next = change.target.value
        setText(next)
        try {
          const value = JSON.parse(next)
          lastEmittedRef.current = JSON.stringify(value, null, 2)
          edit({ type: "json", value })
          updateValidity(true)
        } catch {
          updateValidity(false)
        }
      }}
    />
  )
}

function ScenarioRequirementsEditor({
  draft,
  setDraft,
}: {
  draft: EvalScenarioDraft
  setDraft: React.Dispatch<React.SetStateAction<EvalScenarioDraft>>
}) {
  const secrets = draft.secret_requirements ?? []
  const addSecret = () => {
    let index = secrets.length + 1
    const ids = new Set(secrets.map((item) => item.id))
    while (ids.has(`secret-${index}`)) index += 1
    setDraft((current) => ({
      ...current,
      secret_requirements: [
        ...(current.secret_requirements ?? []),
        {
          id: `secret-${index}`,
          usage: "other",
          purpose: "Describe the server-side test secret requirement.",
        },
      ],
    }))
  }
  const editSecret = (index: number, next: ScenarioSecretRequirementV2) =>
    setDraft((current) => {
      const secretRequirements = [...(current.secret_requirements ?? [])]
      secretRequirements[index] = next
      return { ...current, secret_requirements: secretRequirements }
    })
  return (
    <div className="space-y-3 rounded-lg border border-border p-3">
      <div className="flex items-center justify-between gap-2">
        <div>
          <div className="text-sm font-medium">Named secret requirements</div>
          <div className="text-xs text-muted-foreground">
            Names and purpose only. Secret values and vault handles never enter the scenario.
          </div>
        </div>
        <Button type="button" size="xs" variant="outline" onClick={addSecret}>
          <Plus /> Add requirement
        </Button>
      </div>
      {secrets.length === 0 ? (
        <div className="text-xs text-muted-foreground">No explicit secret requirements.</div>
      ) : (
        secrets.map((secret, index) => (
          // biome-ignore lint/suspicious/noArrayIndexKey: requirement rows have no separate UI identity and cannot be reordered.
          <div key={index} className="grid gap-2 md:grid-cols-[1fr_10rem_2fr_auto]">
            <Input
              aria-label={`Secret requirement ${index + 1} ID`}
              value={secret.id}
              onChange={(change) => editSecret(index, { ...secret, id: change.target.value })}
            />
            <select
              className={SELECT_CLASS}
              value={secret.usage}
              onChange={(change) =>
                editSecret(index, {
                  ...secret,
                  usage: change.target.value as ScenarioSecretRequirementV2["usage"],
                })
              }
            >
              {(["provider", "tool", "environment", "artifact", "other"] as const).map((usage) => (
                <option key={usage} value={usage}>
                  {usage}
                </option>
              ))}
            </select>
            <Input
              aria-label={`Secret requirement ${index + 1} purpose`}
              value={secret.purpose}
              onChange={(change) => editSecret(index, { ...secret, purpose: change.target.value })}
            />
            <Button
              type="button"
              size="icon-xs"
              variant="ghost"
              aria-label={`Remove secret requirement ${index + 1}`}
              onClick={() =>
                setDraft((current) => ({
                  ...current,
                  secret_requirements: (current.secret_requirements ?? []).filter(
                    (_, secretIndex) => secretIndex !== index,
                  ),
                }))
              }
            >
              <Trash2 />
            </Button>
          </div>
        ))
      )}
    </div>
  )
}

function ScenarioPreflightSummary({ preview }: { preview: EvalScenarioPreview }) {
  return (
    <div
      className={
        preview.preflight.ready
          ? "rounded-lg border border-emerald-500/30 bg-emerald-500/5 p-3 text-sm"
          : "rounded-lg border border-amber-500/30 bg-amber-500/5 p-3 text-sm"
      }
      data-testid="scenario-preflight"
    >
      <div className="flex items-center gap-2 font-medium">
        {preview.preflight.ready ? (
          <CheckCircle2 className="h-4 w-4 text-emerald-600" />
        ) : (
          <AlertTriangle className="h-4 w-4 text-amber-600" />
        )}
        {preview.preflight.ready
          ? "Current launch requirements are ready"
          : "Current launch requirements need attention"}
      </div>
      {preview.preflight.binding && (
        <div className="mt-1 text-xs text-muted-foreground">
          Release {preview.preflight.binding.application_release_id} · agent{" "}
          {preview.preflight.binding.agent_name} · {preview.preflight.binding.trials} trial
          {preview.preflight.binding.trials === 1 ? "" : "s"}
        </div>
      )}
      {(preview.preflight.diagnostics ?? []).length > 0 && (
        <div className="mt-2 space-y-2">
          {(preview.preflight.diagnostics ?? []).map((diagnostic) => (
            <div
              key={`${diagnostic.code}:${diagnostic.event_id ?? ""}:${diagnostic.requirement_id ?? ""}`}
              className="text-xs"
            >
              <span className="font-medium">{diagnostic.message}</span> {diagnostic.remediation}
            </div>
          ))}
        </div>
      )}
      <div className="mt-2 font-mono text-[11px] text-muted-foreground">
        {shortEvalIdentity(preview.scenario.revision)}
      </div>
    </div>
  )
}

function NumberSetting({
  label,
  value,
  edit,
  maximum,
  optional = false,
}: {
  label: string
  value: string
  edit: (value: string) => void
  maximum: number
  optional?: boolean
}) {
  const inputId = useId()
  return (
    <label htmlFor={inputId}>
      <span className={LABEL_CLASS}>{label}</span>
      <Input
        id={inputId}
        type="number"
        min={1}
        max={maximum}
        step={1}
        placeholder={optional ? "Target default" : undefined}
        value={value}
        onChange={(event) => edit(event.target.value)}
      />
    </label>
  )
}
