import { useQuery, useQueryClient } from "@tanstack/react-query"
import { CheckCircle2, Copy, FlaskConical, LoaderCircle, Plus, Save, Trash2 } from "lucide-react"
import { type ReactNode, useCallback, useEffect, useMemo, useRef, useState } from "react"
import { ExpectedBehaviorEditor } from "@/components/dashboard/evaluation-promotion"
import {
  ScenarioAuthoring,
  type ScenarioAuthoringState,
} from "@/components/dashboard/scenario-authoring"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetFooter,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet"
import { Textarea } from "@/components/ui/textarea"
import {
  ApiClientError,
  type EvalAuthoredSuite,
  type EvalAuthoredSuiteDraft,
  type EvalAuthoredSuitePreview,
  type EvalAuthoredSuiteRunLaunched,
  type EvalAuthoredSuiteRunPreview,
  type EvalScenario,
  fetchEvalAuthoredSuite,
  fetchEvalAuthoredSuites,
  fetchEvalScenario,
  launchEvalAuthoredSuiteRun,
  previewEvalAuthoredSuite,
  previewEvalAuthoredSuiteRun,
  saveEvalAuthoredSuite,
} from "@/lib/api"
import { dashboardConfig } from "@/lib/config"
import {
  authoredSuiteRunPreviewIdentity,
  blankEvalScenarioDraft,
  duplicateEvalSuiteCase,
  EVAL_SUITE_MAX_CASES,
  evalSuiteDraftFromDocument,
  newEvalSuiteDraft,
  newSimpleEvalCase,
  validateEvalSuiteDraft,
} from "@/lib/eval-suite-authoring"
import {
  authoredSuiteEvalLaunchRequestIdentity,
  EVAL_TARGET_QUERY_KEY,
  EvalLaunchIdempotencyRegistry,
  evalErrorMessage,
  evalLaunchFailureIsDefinitive,
  evalTargetCatalogMayBeStale,
  shortEvalIdentity,
} from "@/lib/evals-dashboard"
import type { PromotionAssertion } from "@/lib/evaluation-promotion"
import type { EvalCaseDraftV1, EvalScenarioDraftV2 } from "@/lib/generated/server-api"

const FIELD_LABEL = "mb-1 block text-xs font-medium text-muted-foreground"

export function EvalSuiteAuthoringAction({
  targetKey,
  disabled,
  onLaunched,
}: {
  targetKey: string
  disabled: boolean
  onLaunched: (runIds: string[]) => Promise<void>
}) {
  const queryClient = useQueryClient()
  const [open, setOpen] = useState(false)
  const [draft, setDraft] = useState<EvalAuthoredSuiteDraft>(() => newEvalSuiteDraft(targetKey))
  const caseRowSequenceRef = useRef(0)
  const allocateCaseRowKey = () => {
    caseRowSequenceRef.current += 1
    return `case-row-${caseRowSequenceRef.current}`
  }
  const allocateCaseRowKeys = (count: number) => Array.from({ length: count }, allocateCaseRowKey)
  const [caseRowKeys, setCaseRowKeys] = useState(() => allocateCaseRowKeys(draft.cases.length))
  const [activeCaseKey, setActiveCaseKey] = useState(caseRowKeys[0] ?? "")
  const [selectedCaseKeys, setSelectedCaseKeys] = useState<Set<string>>(() => new Set(caseRowKeys))
  const [scenarioDrafts, setScenarioDrafts] = useState<Record<string, EvalScenarioDraftV2>>({})
  const [savedScenarios, setSavedScenarios] = useState<Record<string, EvalScenario>>({})
  const [scenarioAuthoringState, setScenarioAuthoringState] = useState<
    (ScenarioAuthoringState & { caseKey: string }) | null
  >(null)
  const [scenarioEditorEpoch, setScenarioEditorEpoch] = useState(0)
  const [suitePreview, setSuitePreview] = useState<EvalAuthoredSuitePreview | null>(null)
  const [suitePreviewIdentity, setSuitePreviewIdentity] = useState<string | null>(null)
  const [savedRevision, setSavedRevision] = useState<string | null>(null)
  const [runPreview, setRunPreview] = useState<EvalAuthoredSuiteRunPreview | null>(null)
  const [runPreviewIdentity, setRunPreviewIdentity] = useState<string | null>(null)
  const [pending, setPending] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const controllerRef = useRef<AbortController | null>(null)
  const launchRegistryRef = useRef<EvalLaunchIdempotencyRegistry | null>(null)
  const authoredSuites = useQuery({
    queryKey: ["evals", "authored-suites", targetKey],
    queryFn: ({ signal }) => fetchEvalAuthoredSuites({ target_key: targetKey, limit: 50 }, signal),
    enabled: open,
  })
  const activeCaseIndex = Math.max(0, caseRowKeys.indexOf(activeCaseKey))
  const activeCase = draft.cases[activeCaseIndex] ?? draft.cases[0]
  const activeCaseRowKey = caseRowKeys[activeCaseIndex]
  const activeScenarioAuthoringState =
    scenarioAuthoringState?.caseKey === activeCaseRowKey ? scenarioAuthoringState : null
  const activeScenarioDirty = activeScenarioAuthoringState?.dirty === true
  const activeScenarioPending = activeScenarioAuthoringState?.pending === true
  const unsavedScenarioReference = Object.keys(scenarioDrafts).length > 0
  const activeScenarioRevision =
    activeCase && isScenarioCase(activeCase) ? activeCase.stimulus.scenario_revision : null
  const loadedScenario = useQuery({
    queryKey: ["evals", "scenario", activeScenarioRevision],
    queryFn: ({ signal }) => fetchEvalScenario(activeScenarioRevision ?? "", signal),
    enabled:
      open &&
      activeScenarioRevision !== null &&
      (activeCaseRowKey === undefined || savedScenarios[activeCaseRowKey] === undefined),
  })
  const currentIdentity = useMemo(() => JSON.stringify(draft), [draft])
  const previewCurrent =
    suitePreview !== null &&
    suitePreviewIdentity === currentIdentity &&
    !unsavedScenarioReference &&
    !activeScenarioDirty
  const launchSelection = useMemo(() => {
    const selected = draft.cases
      .filter((_, index) => {
        const caseKey = caseRowKeys[index]
        return caseKey !== undefined && selectedCaseKeys.has(caseKey)
      })
      .map((item) => item.id)
      .sort()
    return selected.length === draft.cases.length ? {} : { case_ids: selected }
  }, [caseRowKeys, draft.cases, selectedCaseKeys])
  const currentRunPreviewIdentity = useMemo(
    () =>
      authoredSuiteRunPreviewIdentity(
        savedRevision,
        suitePreview?.suite.revision ?? null,
        launchSelection,
      ),
    [savedRevision, suitePreview?.suite.revision, launchSelection],
  )
  const runPreviewCurrent =
    runPreview !== null &&
    runPreviewIdentity === currentRunPreviewIdentity &&
    previewCurrent &&
    savedRevision !== null &&
    savedRevision === suitePreview?.suite.revision
  const authoringPending = pending !== null || activeScenarioPending
  const authoringLocked = disabled || authoringPending
  const scenarioHasUnsavedChanges = unsavedScenarioReference || activeScenarioDirty
  const scenarioTransitionLocked = authoringPending || scenarioHasUnsavedChanges

  useEffect(() => () => controllerRef.current?.abort(), [])

  const editDraft = (edit: (next: EvalAuthoredSuiteDraft) => void) => {
    setDraft((current) => {
      const next = structuredClone(current)
      edit(next)
      return next
    })
  }

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
        setError(evalErrorMessage(actionError, "The authored evaluation operation failed."))
      }
    } finally {
      if (controllerRef.current === controller) {
        controllerRef.current = null
        setPending(null)
      }
    }
  }

  const resetToNewSuite = () => {
    if (authoringLocked || scenarioHasUnsavedChanges) return
    const next = newEvalSuiteDraft(targetKey)
    const nextCaseKeys = allocateCaseRowKeys(next.cases.length)
    setDraft(next)
    setCaseRowKeys(nextCaseKeys)
    setActiveCaseKey(nextCaseKeys[0] ?? "")
    setSelectedCaseKeys(new Set(nextCaseKeys))
    setScenarioDrafts({})
    setSavedScenarios({})
    setScenarioAuthoringState(null)
    setScenarioEditorEpoch((current) => current + 1)
    setSuitePreview(null)
    setSuitePreviewIdentity(null)
    setSavedRevision(null)
    setRunPreview(null)
    setRunPreviewIdentity(null)
    setError(null)
    setNotice(null)
  }

  const loadSuite = (revision: string) => {
    if (scenarioTransitionLocked) return
    void run("load", async (signal) => {
      const document = await fetchEvalAuthoredSuite(revision, signal)
      if (signal.aborted) return
      loadDocument(document)
      setNotice(`Loaded immutable suite ${shortEvalIdentity(document.revision)}.`)
    })
  }

  const loadDocument = (document: EvalAuthoredSuite) => {
    const next = evalSuiteDraftFromDocument(document)
    const nextCaseKeys = allocateCaseRowKeys(next.cases.length)
    setDraft(next)
    setCaseRowKeys(nextCaseKeys)
    setActiveCaseKey(nextCaseKeys[0] ?? "")
    setSelectedCaseKeys(new Set(nextCaseKeys))
    setScenarioDrafts({})
    setSavedScenarios({})
    setScenarioAuthoringState(null)
    setScenarioEditorEpoch((current) => current + 1)
    setSuitePreview(null)
    setSuitePreviewIdentity(null)
    setSavedRevision(document.revision)
    setRunPreview(null)
    setRunPreviewIdentity(null)
  }

  const addCase = () => {
    if (
      authoringLocked ||
      scenarioHasUnsavedChanges ||
      draft.cases.length >= EVAL_SUITE_MAX_CASES
    ) {
      return
    }
    const nextCase = newSimpleEvalCase(draft.cases)
    const nextCaseKey = allocateCaseRowKey()
    editDraft((next) => next.cases.push(nextCase))
    setCaseRowKeys((current) => [...current, nextCaseKey])
    setActiveCaseKey(nextCaseKey)
    setSelectedCaseKeys((current) => new Set(current).add(nextCaseKey))
  }

  const duplicateActiveCase = () => {
    if (
      authoringLocked ||
      scenarioHasUnsavedChanges ||
      draft.cases.length >= EVAL_SUITE_MAX_CASES
    ) {
      return
    }
    if (!activeCase || activeCaseRowKey === undefined) return
    const duplicate = duplicateEvalSuiteCase(draft.cases, activeCaseIndex)
    const duplicateCaseKey = allocateCaseRowKey()
    editDraft((next) => next.cases.push(duplicate))
    setCaseRowKeys((current) => [...current, duplicateCaseKey])
    setActiveCaseKey(duplicateCaseKey)
    setSelectedCaseKeys((current) => new Set(current).add(duplicateCaseKey))
  }

  const removeActiveCase = () => {
    if (
      authoringLocked ||
      scenarioHasUnsavedChanges ||
      !activeCase ||
      activeCaseRowKey === undefined ||
      draft.cases.length === 1
    ) {
      return
    }
    const remainingCaseKeys = caseRowKeys.filter((_, index) => index !== activeCaseIndex)
    editDraft((next) => {
      next.cases.splice(activeCaseIndex, 1)
    })
    setCaseRowKeys(remainingCaseKeys)
    setActiveCaseKey(remainingCaseKeys[0] ?? "")
    setSelectedCaseKeys((current) => {
      const next = new Set(current)
      next.delete(activeCaseRowKey)
      return next
    })
    setScenarioDrafts((current) => withoutKey(current, activeCaseRowKey))
    setSavedScenarios((current) => withoutKey(current, activeCaseRowKey))
  }

  const switchStimulus = (kind: "simple_input" | "scenario") => {
    if (
      authoringLocked ||
      scenarioHasUnsavedChanges ||
      !activeCase ||
      activeCaseRowKey === undefined
    ) {
      return
    }
    setScenarioAuthoringState(null)
    if (kind === "simple_input") {
      const simple = newSimpleEvalCase([], activeCase.id)
      editDraft((next) => {
        const current = next.cases[activeCaseIndex]
        if (current) current.stimulus = simple.stimulus
      })
      setScenarioDrafts((current) => withoutKey(current, activeCaseRowKey))
      setSavedScenarios((current) => withoutKey(current, activeCaseRowKey))
      return
    }
    setScenarioDrafts((current) => ({
      ...current,
      [activeCaseRowKey]: blankEvalScenarioDraft(targetKey, activeCase),
    }))
  }

  const previewSuite = () => {
    if (authoringLocked) return
    if (unsavedScenarioReference) {
      setError("Save every new scenario before checking the suite.")
      return
    }
    if (activeScenarioDirty) {
      setError("Save the visible scenario edits before checking the suite.")
      return
    }
    const validation = validateEvalSuiteDraft(draft)
    if (!validation.ok) {
      setError(validation.error)
      return
    }
    const identity = currentIdentity
    void run("preview", async (signal) => {
      const next = await previewEvalAuthoredSuite({ draft: validation.draft }, signal)
      if (signal.aborted) return
      setSuitePreview(next)
      setSuitePreviewIdentity(identity)
    })
  }

  const saveSuite = () => {
    if (authoringLocked || !previewCurrent || !suitePreview?.ready) return
    void run("save", async (signal) => {
      const saved = await saveEvalAuthoredSuite(
        {
          expected_suite_revision: suitePreview.suite.revision,
          suite: suitePreview.suite,
        },
        signal,
      )
      if (signal.aborted) return
      const normalized = evalSuiteDraftFromDocument(saved.suite)
      const caseKeyById = new Map(
        draft.cases.flatMap((evalCase, index) => {
          const caseKey = caseRowKeys[index]
          return caseKey === undefined ? [] : ([[evalCase.id, caseKey]] as const)
        }),
      )
      const normalizedCaseKeys = normalized.cases.map(
        (evalCase) => caseKeyById.get(evalCase.id) ?? allocateCaseRowKey(),
      )
      setDraft(normalized)
      setCaseRowKeys(normalizedCaseKeys)
      setSuitePreview({
        suite: saved.suite,
        full_selection: saved.full_selection,
        ready: true,
        diagnostics: [],
      })
      setSuitePreviewIdentity(JSON.stringify(normalized))
      setSavedRevision(saved.suite.revision)
      setSelectedCaseKeys((current) => {
        const retainedKeys = new Set(normalizedCaseKeys)
        return new Set([...current].filter((caseKey) => retainedKeys.has(caseKey)))
      })
      setNotice(`Saved immutable suite ${shortEvalIdentity(saved.suite.revision)}.`)
      await queryClient.invalidateQueries({ queryKey: ["evals", "authored-suites"] })
    })
  }

  const previewRun = () => {
    if (authoringLocked) return
    if (!previewCurrent || savedRevision !== suitePreview?.suite.revision) {
      setError("Check and save the current suite revision before launch readiness.")
      return
    }
    if (selectedCaseKeys.size === 0) {
      setError("Select at least one case to run.")
      return
    }
    const identity = currentRunPreviewIdentity
    void run("run-preview", async (signal) => {
      const next = await previewEvalAuthoredSuiteRun(savedRevision, launchSelection, signal)
      if (signal.aborted) return
      setRunPreview(next)
      setRunPreviewIdentity(identity)
    })
  }

  const launch = () => {
    if (
      authoringLocked ||
      !runPreviewCurrent ||
      !runPreview?.ready ||
      savedRevision === null ||
      savedRevision !== suitePreview?.suite.revision
    ) {
      return
    }
    const expectedExecutionProfiles = runPreview.launches.flatMap((item) =>
      item.execution_profile_revision
        ? [
            {
              case_ids: item.case_ids,
              execution_profile_revision: item.execution_profile_revision,
            },
          ]
        : [],
    )
    if (expectedExecutionProfiles.length !== runPreview.launches.length) return
    const requestIdentity = authoredSuiteEvalLaunchRequestIdentity(
      savedRevision,
      runPreview.selection.revision,
      expectedExecutionProfiles,
    )
    void run("launch", async (signal) => {
      const registry =
        launchRegistryRef.current ??
        new EvalLaunchIdempotencyRegistry(window.sessionStorage, dashboardConfig.apiBaseUrl)
      launchRegistryRef.current = registry
      const idempotencyKey = registry.keyFor(requestIdentity)
      let launched: EvalAuthoredSuiteRunLaunched
      try {
        launched = await launchEvalAuthoredSuiteRun(
          savedRevision,
          {
            ...launchSelection,
            expected_execution_profiles: expectedExecutionProfiles,
          },
          idempotencyKey,
          signal,
        )
      } catch (launchError) {
        if (evalTargetCatalogMayBeStale(launchError)) {
          await queryClient.invalidateQueries({ queryKey: EVAL_TARGET_QUERY_KEY })
        }
        if (
          launchError instanceof ApiClientError &&
          evalLaunchFailureIsDefinitive(launchError.status)
        ) {
          registry.resolve(requestIdentity)
        }
        throw launchError
      }
      if (signal.aborted) return
      const runIds = launched.runs.map((item) => item.run.spec.run_id)
      for (const item of launched.runs) {
        queryClient.setQueryData(["evals", "run", item.run.spec.run_id], item.run)
      }
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["evals", "runs"] }),
        queryClient.invalidateQueries({ queryKey: ["evals", "corpora"] }),
      ])
      if (signal.aborted) return
      registry.resolve(requestIdentity)
      setNotice(
        `Started ${runIds.length} durable eval ${runIds.length === 1 ? "run" : "runs"} for ${launched.selection.cases.length} selected ${launched.selection.cases.length === 1 ? "case" : "cases"}.`,
      )
      await onLaunched(runIds)
    })
  }

  const retainSavedScenario = (caseKey: string, scenario: EvalScenario) => {
    const caseIndex = caseRowKeys.indexOf(caseKey)
    const caseId = caseIndex < 0 ? null : draft.cases[caseIndex]?.id
    if (caseIndex < 0 || caseId === null || caseId === undefined) return
    setSavedScenarios((current) => ({
      ...current,
      [caseKey]: scenario,
    }))
    setScenarioDrafts((current) => withoutKey(current, caseKey))
    editDraft((next) => {
      const current = next.cases[caseIndex]
      if (current) {
        current.stimulus = {
          kind: "scenario",
          scenario_id: scenario.id,
          scenario_revision: scenario.revision,
        }
      }
    })
    setNotice(`Saved scenario ${shortEvalIdentity(scenario.revision)} for ${caseId}.`)
  }

  const selectCaseForLaunch = (caseKey: string, selected: boolean) => {
    if (authoringLocked) return
    setSelectedCaseKeys((current) => {
      const next = new Set(current)
      if (selected) next.add(caseKey)
      else next.delete(caseKey)
      return next
    })
  }

  const retainScenarioAuthoringState = useCallback(
    (caseKey: string, state: ScenarioAuthoringState) => {
      setScenarioAuthoringState((current) => {
        if (!state.dirty && !state.pending) {
          return current?.caseKey === caseKey ? null : current
        }
        if (
          current?.caseKey === caseKey &&
          current.dirty === state.dirty &&
          current.pending === state.pending
        ) {
          return current
        }
        return { caseKey, ...state }
      })
    },
    [],
  )
  const reportActiveScenarioAuthoringState = useCallback(
    (state: ScenarioAuthoringState) => {
      if (activeCaseRowKey !== undefined) {
        retainScenarioAuthoringState(activeCaseRowKey, state)
      }
    },
    [activeCaseRowKey, retainScenarioAuthoringState],
  )

  const scenarioForActiveCase = activeCaseRowKey
    ? (savedScenarios[activeCaseRowKey] ?? loadedScenario.data)
    : undefined
  const scenarioDraftForActiveCase = activeCaseRowKey ? scenarioDrafts[activeCaseRowKey] : undefined

  const discardActiveScenarioEdits = () => {
    if (
      authoringPending ||
      !activeCase ||
      activeCaseRowKey === undefined ||
      !scenarioHasUnsavedChanges
    ) {
      return
    }
    if (scenarioDraftForActiveCase !== undefined) {
      setScenarioDrafts((current) => withoutKey(current, activeCaseRowKey))
      setNotice("Discarded the unsaved scenario draft and restored the simple input.")
    } else if (isScenarioCase(activeCase)) {
      setScenarioEditorEpoch((current) => current + 1)
      setNotice("Discarded the unsaved scenario edits and restored the saved revision.")
    } else {
      return
    }
    setScenarioAuthoringState(null)
    setError(null)
  }

  return (
    <>
      <Button
        type="button"
        data-testid="new-evaluation"
        disabled={disabled}
        onClick={() => setOpen(true)}
      >
        <Plus /> New evaluation
      </Button>
      <Sheet
        open={open}
        onOpenChange={(next) => {
          if (!next && scenarioTransitionLocked) return
          setOpen(next)
          if (!next) {
            controllerRef.current?.abort()
            setScenarioAuthoringState(null)
          }
        }}
      >
        <SheetContent
          className="w-[min(96vw,88rem)] overflow-hidden sm:max-w-none"
          data-testid="eval-suite-authoring-sheet"
        >
          <SheetHeader>
            <SheetTitle>New evaluation</SheetTitle>
            <SheetDescription>
              Build a reusable deterministic suite, save its immutable revision, then run all cases
              or an explicit subset against {targetKey}.
            </SheetDescription>
          </SheetHeader>

          <div className="grid min-h-0 flex-1 gap-4 overflow-hidden px-4 pb-2 xl:grid-cols-[18rem_minmax(0,1fr)]">
            <aside className="min-h-0 space-y-3 overflow-y-auto rounded-lg border border-border p-3">
              <div className="flex items-center justify-between gap-2">
                <div>
                  <div className="text-sm font-medium">Suites</div>
                  <div className="text-xs text-muted-foreground">
                    Create new or reuse a revision
                  </div>
                </div>
                <Button
                  type="button"
                  size="xs"
                  variant="outline"
                  disabled={authoringLocked || scenarioHasUnsavedChanges}
                  onClick={resetToNewSuite}
                >
                  <Plus /> New
                </Button>
              </div>
              {authoredSuites.isLoading ? (
                <div className="text-xs text-muted-foreground">Loading saved suites...</div>
              ) : authoredSuites.isError ? (
                <div className="text-xs text-destructive">Saved suites could not be loaded.</div>
              ) : authoredSuites.data?.items.length ? (
                <div className="space-y-1.5" data-testid="authored-suite-catalog">
                  {authoredSuites.data.items.map((item) => (
                    <button
                      key={item.revision}
                      type="button"
                      className="w-full rounded-md border border-border px-2.5 py-2 text-left hover:bg-muted/50"
                      disabled={scenarioTransitionLocked}
                      onClick={() => loadSuite(item.revision)}
                    >
                      <div className="truncate text-sm font-medium">{item.name}</div>
                      <div className="mt-1 text-xs text-muted-foreground">
                        {item.case_count} cases · {shortEvalIdentity(item.revision)}
                      </div>
                    </button>
                  ))}
                </div>
              ) : (
                <div className="rounded-md border border-dashed border-border p-3 text-xs text-muted-foreground">
                  No authored suites yet.
                </div>
              )}

              <div className="border-t border-border pt-3">
                <div className="mb-2 flex items-center justify-between gap-2">
                  <div>
                    <div className="text-sm font-medium">Cases</div>
                    <div className="text-xs text-muted-foreground">Check cases to run</div>
                  </div>
                  <Button
                    type="button"
                    size="icon-xs"
                    variant="outline"
                    disabled={
                      authoringLocked ||
                      scenarioHasUnsavedChanges ||
                      draft.cases.length >= EVAL_SUITE_MAX_CASES
                    }
                    onClick={addCase}
                  >
                    <Plus />
                    <span className="sr-only">Add case</span>
                  </Button>
                </div>
                <div className="space-y-1.5" data-testid="authored-suite-cases">
                  {draft.cases.map((evalCase, index) => {
                    const caseKey = caseRowKeys[index]
                    if (caseKey === undefined) return null
                    return (
                      <div
                        key={caseKey}
                        data-state={activeCaseRowKey === caseKey ? "selected" : undefined}
                        className="flex items-center gap-2 rounded-md border border-border px-2 py-1.5 data-[state=selected]:border-primary/50 data-[state=selected]:bg-primary/5"
                      >
                        <input
                          type="checkbox"
                          checked={selectedCaseKeys.has(caseKey)}
                          disabled={authoringLocked}
                          aria-label={`Select ${evalCase.name} for launch`}
                          onChange={(event) => selectCaseForLaunch(caseKey, event.target.checked)}
                        />
                        <button
                          type="button"
                          className="min-w-0 flex-1 text-left"
                          disabled={scenarioTransitionLocked}
                          onClick={() => {
                            setScenarioAuthoringState(null)
                            setActiveCaseKey(caseKey)
                          }}
                        >
                          <div className="truncate text-sm font-medium">{evalCase.name}</div>
                          <div className="truncate font-mono text-[11px] text-muted-foreground">
                            {evalCase.id}
                          </div>
                        </button>
                        <Badge variant="outline">
                          {isScenarioCase(evalCase) || scenarioDrafts[caseKey]
                            ? "Scenario"
                            : "Simple"}
                        </Badge>
                      </div>
                    )
                  })}
                </div>
              </div>
            </aside>

            <div className="min-h-0 overflow-y-auto pr-1">
              <fieldset className="space-y-4" disabled={authoringLocked}>
                <Card size="sm">
                  <CardHeader className="grid-cols-[1fr_auto]">
                    <div>
                      <CardTitle>Suite</CardTitle>
                      <p className="mt-1 text-xs text-muted-foreground">
                        Stable identity and one-trial execution settings.
                      </p>
                    </div>
                    {savedRevision && (
                      <Badge variant="outline">Saved {shortEvalIdentity(savedRevision)}</Badge>
                    )}
                  </CardHeader>
                  <CardContent className="grid gap-3 sm:grid-cols-2">
                    <Field label="Suite ID" id="authored-suite-id">
                      <Input
                        id="authored-suite-id"
                        data-testid="authored-suite-id"
                        value={draft.id}
                        onChange={(event) => editDraft((next) => (next.id = event.target.value))}
                      />
                    </Field>
                    <Field label="Suite name" id="authored-suite-name">
                      <Input
                        id="authored-suite-name"
                        data-testid="authored-suite-name"
                        value={draft.name}
                        onChange={(event) => editDraft((next) => (next.name = event.target.value))}
                      />
                    </Field>
                    <Field label="Description" id="authored-suite-description" wide>
                      <Textarea
                        id="authored-suite-description"
                        rows={2}
                        value={draft.description ?? ""}
                        onChange={(event) =>
                          editDraft((next) => (next.description = event.target.value || null))
                        }
                      />
                    </Field>
                    <Field label="Timeout per case" id="authored-suite-timeout">
                      <Input
                        id="authored-suite-timeout"
                        type="number"
                        min={1}
                        max={3_600}
                        value={draft.trial_request?.timeout_seconds ?? 300}
                        onChange={(event) =>
                          editDraft((next) => {
                            next.trial_request = {
                              trials: 1,
                              timeout_seconds: Number(event.target.value),
                            }
                          })
                        }
                      />
                    </Field>
                    <div className="self-end rounded-md border border-border bg-muted/30 px-3 py-2 text-xs text-muted-foreground">
                      1 trial · concurrency 1. Larger execution profiles arrive in the next V3
                      slice.
                    </div>
                  </CardContent>
                </Card>

                {activeCase && (
                  <>
                    <Card size="sm" data-testid="authored-case-editor">
                      <CardHeader className="grid-cols-[1fr_auto]">
                        <div>
                          <CardTitle>Case</CardTitle>
                          <p className="mt-1 text-xs text-muted-foreground">
                            Revise this portable case ID without mutating prior saved revisions.
                          </p>
                        </div>
                        <div className="flex gap-1.5">
                          <Button
                            type="button"
                            size="xs"
                            variant="outline"
                            disabled={
                              authoringLocked ||
                              scenarioHasUnsavedChanges ||
                              draft.cases.length >= EVAL_SUITE_MAX_CASES
                            }
                            onClick={duplicateActiveCase}
                          >
                            <Copy /> Duplicate
                          </Button>
                          <Button
                            type="button"
                            size="icon-xs"
                            variant="ghost"
                            aria-label={`Remove case ${activeCase.id}`}
                            disabled={
                              authoringLocked ||
                              scenarioHasUnsavedChanges ||
                              draft.cases.length === 1
                            }
                            onClick={removeActiveCase}
                          >
                            <Trash2 />
                          </Button>
                        </div>
                      </CardHeader>
                      <CardContent className="grid gap-3 sm:grid-cols-2">
                        <Field label="Case ID" id="authored-case-id">
                          <Input
                            id="authored-case-id"
                            data-testid="authored-case-id"
                            value={activeCase.id}
                            disabled={scenarioHasUnsavedChanges}
                            onChange={(event) => {
                              const nextId = event.target.value
                              editDraft((next) => {
                                const current = next.cases[activeCaseIndex]
                                if (current) current.id = nextId
                              })
                            }}
                          />
                        </Field>
                        <Field label="Case name" id="authored-case-name">
                          <Input
                            id="authored-case-name"
                            data-testid="authored-case-name"
                            value={activeCase.name}
                            onChange={(event) =>
                              editDraft((next) => {
                                const current = next.cases[activeCaseIndex]
                                if (current) current.name = event.target.value
                              })
                            }
                          />
                        </Field>
                        <Field label="Description" id="authored-case-description" wide>
                          <Textarea
                            id="authored-case-description"
                            rows={2}
                            value={activeCase.description ?? ""}
                            onChange={(event) =>
                              editDraft((next) => {
                                const current = next.cases[activeCaseIndex]
                                if (current) current.description = event.target.value || null
                              })
                            }
                          />
                        </Field>
                        <div className="sm:col-span-2">
                          <span className={FIELD_LABEL}>Scenario type</span>
                          <div className="flex gap-2">
                            <Button
                              type="button"
                              size="sm"
                              variant={
                                !isScenarioCase(activeCase) && !scenarioDraftForActiveCase
                                  ? "default"
                                  : "outline"
                              }
                              disabled={scenarioHasUnsavedChanges}
                              onClick={() => switchStimulus("simple_input")}
                            >
                              Simple input
                            </Button>
                            <Button
                              type="button"
                              size="sm"
                              variant={
                                isScenarioCase(activeCase) || scenarioDraftForActiveCase
                                  ? "default"
                                  : "outline"
                              }
                              disabled={scenarioHasUnsavedChanges}
                              onClick={() => switchStimulus("scenario")}
                            >
                              Multi-stage scenario
                            </Button>
                          </div>
                        </div>
                      </CardContent>
                    </Card>

                    {!isScenarioCase(activeCase) && !scenarioDraftForActiveCase ? (
                      <SimpleInputEditor
                        evalCase={activeCase}
                        edit={(updated) =>
                          editDraft((next) => {
                            if (next.cases[activeCaseIndex]) {
                              next.cases[activeCaseIndex] = updated
                            }
                          })
                        }
                      />
                    ) : scenarioDraftForActiveCase ? (
                      <ScenarioAuthoring
                        key={`${activeCaseRowKey}:new-scenario:${scenarioEditorEpoch}`}
                        captured={scenarioDraftForActiveCase}
                        showLaunch={false}
                        showLaunchSettings={false}
                        onSaved={(scenario) =>
                          activeCaseRowKey && retainSavedScenario(activeCaseRowKey, scenario)
                        }
                        onAuthoringStateChange={reportActiveScenarioAuthoringState}
                      />
                    ) : scenarioForActiveCase ? (
                      <ScenarioAuthoring
                        key={`${activeCaseRowKey}:${scenarioForActiveCase.revision}:${scenarioEditorEpoch}`}
                        captured={scenarioForActiveCase}
                        saved
                        showLaunch={false}
                        showLaunchSettings={false}
                        onSaved={(scenario) =>
                          activeCaseRowKey && retainSavedScenario(activeCaseRowKey, scenario)
                        }
                        onAuthoringStateChange={reportActiveScenarioAuthoringState}
                      />
                    ) : loadedScenario.isError ? (
                      <div className="rounded-lg border border-destructive/30 p-4 text-sm text-destructive">
                        The exact saved scenario could not be loaded.
                      </div>
                    ) : (
                      <div className="rounded-lg border border-border p-4 text-sm text-muted-foreground">
                        Loading the exact scenario revision...
                      </div>
                    )}

                    <ExpectedBehaviorEditor
                      assertions={activeCase.assertions as PromotionAssertion[]}
                      evidence={undefined}
                      onChange={(assertions) =>
                        editDraft((next) => {
                          const current = next.cases[activeCaseIndex]
                          if (current) current.assertions = assertions
                        })
                      }
                    />
                  </>
                )}
              </fieldset>

              {scenarioHasUnsavedChanges && (
                <div
                  className="mt-4 flex flex-wrap items-center justify-between gap-3 rounded-lg border border-amber-500/30 bg-amber-500/5 p-3 text-sm text-amber-700 dark:text-amber-300"
                  role="status"
                  data-testid="authored-suite-scenario-lock"
                >
                  <span>
                    Save or discard the visible scenario before leaving it or checking, saving, or
                    launching this suite.
                  </span>
                  <Button
                    type="button"
                    size="xs"
                    variant="outline"
                    disabled={authoringPending}
                    onClick={discardActiveScenarioEdits}
                  >
                    Discard scenario edits
                  </Button>
                </div>
              )}

              {suitePreview && previewCurrent && (
                <ReadinessSummary
                  title={suitePreview.ready ? "Suite is ready to save" : "Suite needs attention"}
                  ready={suitePreview.ready}
                  diagnostics={suitePreview.diagnostics ?? []}
                  identity={suitePreview.suite.revision}
                />
              )}
              {runPreview && runPreviewCurrent && (
                <ReadinessSummary
                  title={
                    runPreview.ready
                      ? `${runPreview.launches.length} durable ${runPreview.launches.length === 1 ? "run" : "runs"} ready`
                      : "Launch needs attention"
                  }
                  ready={runPreview.ready}
                  diagnostics={runPreview.diagnostics ?? []}
                  identity={runPreview.selection.revision}
                />
              )}
              {error && (
                <div
                  className="mt-4 rounded-lg border border-destructive/30 bg-destructive/5 p-3 text-sm text-destructive"
                  role="alert"
                  data-testid="authored-suite-error"
                >
                  {error}
                </div>
              )}
              {notice && (
                <div
                  className="mt-4 rounded-lg border border-emerald-500/30 bg-emerald-500/5 p-3 text-sm text-emerald-700 dark:text-emerald-300"
                  role="status"
                >
                  {notice}
                </div>
              )}
            </div>
          </div>

          <SheetFooter className="border-t border-border">
            <div className="flex flex-wrap justify-end gap-2">
              <Button
                type="button"
                variant="outline"
                data-testid="authored-suite-preview"
                disabled={authoringLocked || unsavedScenarioReference || activeScenarioDirty}
                onClick={previewSuite}
              >
                {pending === "preview" ? (
                  <LoaderCircle className="animate-spin" />
                ) : (
                  <CheckCircle2 />
                )}
                {pending === "preview" ? "Checking..." : "Check suite"}
              </Button>
              <Button
                type="button"
                data-testid="authored-suite-save"
                disabled={authoringLocked || !previewCurrent || !suitePreview?.ready}
                onClick={saveSuite}
              >
                {pending === "save" ? <LoaderCircle className="animate-spin" /> : <Save />}
                {pending === "save" ? "Saving..." : "Save revision"}
              </Button>
              <Button
                type="button"
                variant="outline"
                data-testid="authored-suite-run-preview"
                disabled={
                  authoringLocked ||
                  !previewCurrent ||
                  savedRevision !== suitePreview?.suite.revision ||
                  selectedCaseKeys.size === 0
                }
                onClick={previewRun}
              >
                <CheckCircle2 /> Check launch
              </Button>
              <Button
                type="button"
                data-testid="authored-suite-launch"
                disabled={authoringLocked || !runPreviewCurrent || !runPreview?.ready}
                onClick={launch}
              >
                {pending === "launch" ? (
                  <LoaderCircle className="animate-spin" />
                ) : (
                  <FlaskConical />
                )}
                {pending === "launch" ? "Starting..." : "Run selected"}
              </Button>
            </div>
          </SheetFooter>
        </SheetContent>
      </Sheet>
    </>
  )
}

function SimpleInputEditor({
  evalCase,
  edit,
}: {
  evalCase: EvalCaseDraftV1
  edit: (evalCase: EvalCaseDraftV1) => void
}) {
  if (!("input" in evalCase.stimulus)) return null
  const messages = evalCase.stimulus.input.messages
  const updateMessages = (nextMessages: typeof messages) => {
    edit({
      ...evalCase,
      stimulus: { kind: "simple_input", input: { messages: nextMessages } },
    })
  }
  return (
    <Card size="sm" data-testid="authored-simple-input">
      <CardHeader className="grid-cols-[1fr_auto]">
        <div>
          <CardTitle>Input</CardTitle>
          <p className="mt-1 text-xs text-muted-foreground">
            Each message is supplied as ordered user input to one fresh agent session.
          </p>
        </div>
        <Button
          type="button"
          size="xs"
          variant="outline"
          disabled={messages.length >= 16}
          onClick={() =>
            updateMessages([...messages, { role: "user", text: "Add another user message." }])
          }
        >
          <Plus /> Add message
        </Button>
      </CardHeader>
      <CardContent className="space-y-3">
        {messages.map((message, index) => (
          // biome-ignore lint/suspicious/noArrayIndexKey: message rows hold no local state.
          <div key={`${index}:${messages.length}`} className="flex items-start gap-2">
            <Textarea
              rows={3}
              aria-label={`Case input message ${index + 1}`}
              value={message.text}
              onChange={(event) => {
                const next = [...messages]
                next[index] = { role: "user", text: event.target.value }
                updateMessages(next)
              }}
            />
            <Button
              type="button"
              size="icon-xs"
              variant="ghost"
              aria-label={`Remove case input message ${index + 1}`}
              disabled={messages.length === 1}
              onClick={() => updateMessages(messages.filter((_, candidate) => candidate !== index))}
            >
              <Trash2 />
            </Button>
          </div>
        ))}
      </CardContent>
    </Card>
  )
}

function ReadinessSummary({
  title,
  ready,
  diagnostics,
  identity,
}: {
  title: string
  ready: boolean
  diagnostics: Array<{ case_id?: string | null; message: string }>
  identity: string
}) {
  return (
    <div
      className={`mt-4 rounded-lg border p-3 text-sm ${
        ready ? "border-emerald-500/30 bg-emerald-500/5" : "border-amber-500/30 bg-amber-500/5"
      }`}
      data-testid="authored-suite-readiness"
    >
      <div className="font-medium">{title}</div>
      {diagnostics.map((diagnostic, index) => (
        // biome-ignore lint/suspicious/noArrayIndexKey: diagnostics are immutable preview rows.
        <div key={`${diagnostic.case_id ?? "suite"}:${index}`} className="mt-1 text-xs">
          {diagnostic.case_id ? `${diagnostic.case_id}: ` : ""}
          {diagnostic.message}
        </div>
      ))}
      <div className="mt-2 font-mono text-[11px] text-muted-foreground">
        {shortEvalIdentity(identity)}
      </div>
    </div>
  )
}

function Field({
  label,
  id,
  wide = false,
  children,
}: {
  label: string
  id: string
  wide?: boolean
  children: ReactNode
}) {
  return (
    <label htmlFor={id} className={wide ? "sm:col-span-2" : undefined}>
      <span className={FIELD_LABEL}>{label}</span>
      {children}
    </label>
  )
}

function isScenarioCase(evalCase: EvalCaseDraftV1): evalCase is EvalCaseDraftV1 & {
  stimulus: { kind?: "scenario"; scenario_id: string; scenario_revision: string }
} {
  return "scenario_revision" in evalCase.stimulus
}

function withoutKey<T>(source: Record<string, T>, key: string): Record<string, T> {
  const next = { ...source }
  delete next[key]
  return next
}
