import { useQuery, useQueryClient } from "@tanstack/react-query"
import { CheckCircle2, FlaskConical, LoaderCircle, Plus, Save } from "lucide-react"
import {
  useCallback,
  useEffect,
  useEffectEvent,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react"
import { EvalSuiteCase } from "@/components/dashboard/eval-suite-case"
import { EvalSuiteChecks } from "@/components/dashboard/eval-suite-checks"
import { SimpleInputEditor } from "@/components/dashboard/eval-suite-input"
import { EvalSuiteRunReview, ReadinessSummary } from "@/components/dashboard/eval-suite-run-review"
import { EvalSuiteSettings } from "@/components/dashboard/eval-suite-settings"
import {
  ScenarioAuthoring,
  type ScenarioAuthoringState,
} from "@/components/dashboard/scenario-authoring"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  ApiClientError,
  type EvalAuthoredSuite,
  type EvalAuthoredSuiteDraft,
  type EvalAuthoredSuitePreview,
  type EvalAuthoredSuiteRunLaunched,
  type EvalAuthoredSuiteRunPreview,
  type EvalAuthoredSuiteSave,
  type EvalAuthoredSuiteSaved,
  type EvalScenario,
  fetchEvalAuthoredSuite,
  fetchEvalScenario,
  fetchEvalTargets,
  launchEvalAuthoredSuiteRun,
  previewEvalAuthoredSuite,
  previewEvalAuthoredSuiteRun,
  saveEvalAuthoredSuite,
} from "@/lib/api"
import { dashboardConfig } from "@/lib/config"
import { newMemoryUseJudgeAssertion } from "@/lib/eval-judge-authoring"
import {
  authoredSuiteCostBudgetContract,
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
  clearSuiteLaunchAttempt,
  readSuiteLaunchAttempt,
  type SuiteLaunchAttempt,
  saveSuiteLaunchAttempt,
} from "@/lib/eval-suite-launch-attempt"
import {
  createEvalIdempotencyKey,
  EVAL_TARGET_QUERY_KEY,
  evalErrorMessage,
  evalLaunchFailureIsDefinitive,
  evalTargetCatalogMayBeStale,
  shortEvalIdentity,
} from "@/lib/evals-dashboard"
import { createPromotionAssertion } from "@/lib/evaluation-promotion"
import type {
  EvalCaseDraftV2,
  EvalScenarioDraftV2,
  JudgeProfileIdentityV1,
  StructuredModelJudgeAssertionDraftV1,
} from "@/lib/generated/server-api"

export function EvalSuiteEditor({
  targetKey,
  disabled,
  onLaunched,
  initialRevision,
  active = true,
  onDirtyChange,
  onScopeLockedChange,
  onNavigationLockedChange,
  onSaved,
  initialCaseId,
  initialRunReview = false,
}: {
  targetKey: string
  initialRevision?: string
  disabled: boolean
  active?: boolean
  onDirtyChange?: (dirty: boolean) => void
  onScopeLockedChange?: (locked: boolean) => void
  onNavigationLockedChange?: (locked: boolean) => void
  onSaved?: (suite: EvalAuthoredSuite) => void
  initialCaseId?: string
  initialRunReview?: boolean
  onLaunched: (runIds: string[]) => Promise<void>
}) {
  const initialized = useRef(!initialRevision)
  const queryClient = useQueryClient()
  const [draft, setDraft] = useState<EvalAuthoredSuiteDraft>(() => newEvalSuiteDraft(targetKey))
  const [cleanIdentity, setCleanIdentity] = useState(() => JSON.stringify(draft))
  const activeRef = useRef(active && !disabled)
  activeRef.current = active && !disabled
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
  const [calibrationState, setCalibrationState] = useState({ dirty: false, locked: false })
  const calibrationLocked = calibrationState.dirty || calibrationState.locked
  const [scenarioEditorEpoch, setScenarioEditorEpoch] = useState(0)
  const [suitePreview, setSuitePreview] = useState<EvalAuthoredSuitePreview | null>(null)
  const [suitePreviewIdentity, setSuitePreviewIdentity] = useState<string | null>(null)
  const [savedRevision, setSavedRevision] = useState<string | null>(null)
  const [saveIntent, setSaveIntent] = useState<EvalAuthoredSuiteSave | null>(null)
  const initialSuite = useQuery({
    queryKey: ["evals", "authored-suite", initialRevision],
    queryFn: ({ signal }) => fetchEvalAuthoredSuite(initialRevision ?? "", signal),
    enabled: active && !disabled && initialRevision !== undefined && !initialized.current,
    retry: false,
  })
  const [runReviewOpen, setRunReviewOpen] = useState(initialRunReview)
  const [runPreview, setRunPreview] = useState<EvalAuthoredSuiteRunPreview | null>(null)
  const [runPreviewIdentity, setRunPreviewIdentity] = useState<string | null>(null)
  const [launchMaximumCost, setLaunchMaximumCost] = useState("")
  const [launchCostCurrency, setLaunchCostCurrency] = useState("USD")
  const [pending, setPending] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const controllerRef = useRef<AbortController | null>(null)
  const [launchAttempt, setLaunchAttempt] = useState<SuiteLaunchAttempt | null>(null)
  const [launchRecoveryError, setLaunchRecoveryError] = useState<string | null>(null)
  const [launchRecoveryRevision, setLaunchRecoveryRevision] = useState<string | null>(null)
  useEffect(() => {
    if (!savedRevision) return
    try {
      setLaunchAttempt(
        readSuiteLaunchAttempt(window.sessionStorage, dashboardConfig.apiBaseUrl, savedRevision),
      )
      setLaunchRecoveryError(null)
    } catch (error) {
      setLaunchRecoveryError(evalErrorMessage(error, "Launch recovery is unavailable."))
    }
    setLaunchRecoveryRevision(savedRevision)
  }, [savedRevision])
  const targets = useQuery({
    queryKey: EVAL_TARGET_QUERY_KEY,
    queryFn: ({ signal }) => fetchEvalTargets(signal),
    enabled: active && !disabled,
  })
  const target = targets.data?.items.find((item) => item.target_key === targetKey)
  const defaultProfile = (profiles: JudgeProfileIdentityV1[] = []) => {
    const relation = (profile: JudgeProfileIdentityV1) =>
      target?.judge_profile_routes?.find(
        (route) =>
          route.judge_profile_key === profile.key &&
          route.judge_profile_revision === profile.revision,
      )?.candidate_route_relation
    return (
      profiles.find((profile) => relation(profile) === "independent_model") ??
      profiles.find(
        (profile) =>
          relation(profile) === "same_model" && profile.same_model_use === "allowed_and_labeled",
      )
    )
  }
  const defaultJudgeProfile = defaultProfile(target?.judge_profiles)
  const defaultMemoryJudgeProfile = defaultProfile(
    target?.judge_profiles?.filter(
      (profile) =>
        profile.allowed_evidence.includes("final_output") &&
        profile.allowed_evidence.includes("public_reference"),
    ),
  )
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
      active &&
      !disabled &&
      activeScenarioRevision !== null &&
      (activeCaseRowKey === undefined || savedScenarios[activeCaseRowKey] === undefined),
  })
  const currentIdentity = useMemo(() => JSON.stringify(draft), [draft])
  const previewCurrent =
    suitePreview !== null &&
    suitePreviewIdentity === currentIdentity &&
    !unsavedScenarioReference &&
    !activeScenarioDirty
  const hasMemoryStructuralAssertion =
    activeCase?.assertions.some((assertion) => assertion.kind === "memory_attribution") ?? false
  const memorySemanticJudge = activeCase?.assertions.find(
    (assertion): assertion is StructuredModelJudgeAssertionDraftV1 =>
      assertion.kind === "structured_model_judge" && assertion.rubric.id === "memory-use",
  )
  const hasMemorySemanticJudge = memorySemanticJudge !== undefined
  const memorySemanticJudgeNeedsReference =
    memorySemanticJudge?.reference?.kind === "public_reference" &&
    memorySemanticJudge.reference.expected_answer == null &&
    (memorySemanticJudge.reference.expected_facts ?? []).every((fact) => fact.trim().length === 0)
  const reviewedActiveCase =
    suitePreview && previewCurrent
      ? suitePreview.suite.cases.find((item) => item.id === activeCase?.id)
      : undefined
  const defaultCalibrationMessages =
    activeCase && "input" in activeCase.stimulus ? (activeCase.stimulus.input.messages ?? []) : []
  const defaultCalibrationTask =
    defaultCalibrationMessages.length > 0
      ? defaultCalibrationMessages.map((message) => message.text).join("\n")
      : (activeCase?.name ?? "Evaluate the saved scenario.")
  const launchCostBudget = useMemo(
    () => authoredSuiteCostBudgetContract(launchMaximumCost, launchCostCurrency, target),
    [launchCostCurrency, launchMaximumCost, target],
  )
  const launchCostBudgetInvalid = launchCostBudget === null
  const launchSelection = useMemo(() => {
    const selected = draft.cases
      .filter((_, index) => {
        const caseKey = caseRowKeys[index]
        return caseKey !== undefined && selectedCaseKeys.has(caseKey)
      })
      .map((item) => item.id)
      .sort()
    const selection = selected.length === draft.cases.length ? {} : { case_ids: selected }
    return launchCostBudget === undefined
      ? selection
      : { ...selection, cost_budget: launchCostBudget }
  }, [caseRowKeys, draft.cases, launchCostBudget, selectedCaseKeys])
  const savedDraftCurrent =
    savedRevision !== null &&
    currentIdentity === cleanIdentity &&
    !unsavedScenarioReference &&
    !activeScenarioDirty
  const currentRunPreviewIdentity = useMemo(
    () =>
      authoredSuiteRunPreviewIdentity(
        savedRevision,
        savedDraftCurrent ? savedRevision : (suitePreview?.suite.revision ?? null),
        launchSelection,
      ),
    [savedRevision, savedDraftCurrent, suitePreview?.suite.revision, launchSelection],
  )
  const runPreviewCurrent =
    runPreview !== null && runPreviewIdentity === currentRunPreviewIdentity && savedDraftCurrent
  const authoringPending = pending !== null || activeScenarioPending
  const authoringLocked =
    disabled ||
    !active ||
    authoringPending ||
    launchAttempt !== null ||
    launchRecoveryError !== null ||
    (savedRevision !== null && launchRecoveryRevision !== savedRevision) ||
    saveIntent !== null ||
    (initialRevision !== undefined && !initialized.current)
  const scenarioHasUnsavedChanges = unsavedScenarioReference || activeScenarioDirty
  const dirty =
    currentIdentity !== cleanIdentity ||
    scenarioHasUnsavedChanges ||
    saveIntent !== null ||
    calibrationLocked
  useLayoutEffect(() => {
    if (active) onDirtyChange?.(dirty)
  }, [active, dirty, onDirtyChange])
  const scenarioTransitionLocked = authoringLocked || scenarioHasUnsavedChanges || calibrationLocked
  const scopeLocked =
    authoringPending || scenarioHasUnsavedChanges || saveIntent !== null || calibrationLocked
  useLayoutEffect(() => {
    if (active) onNavigationLockedChange?.(calibrationState.locked)
  }, [active, calibrationState.locked, onNavigationLockedChange])
  useLayoutEffect(() => {
    if (active) onScopeLockedChange?.(scopeLocked)
  }, [active, scopeLocked, onScopeLockedChange])

  useEffect(() => () => controllerRef.current?.abort(), [])
  useEffect(() => {
    if (!active || disabled) {
      controllerRef.current?.abort()
    }
  }, [active, disabled])

  useEffect(() => {
    const currencies = target?.cost_budget_currencies ?? []
    if (currencies.length > 0 && !currencies.includes(launchCostCurrency)) {
      setLaunchCostCurrency(currencies[0] ?? "USD")
    }
  }, [launchCostCurrency, target?.cost_budget_currencies])

  const editDraft = (edit: (next: EvalAuthoredSuiteDraft) => void) => {
    if (calibrationLocked) return
    initialized.current = true
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

  const loadDocument = (document: EvalAuthoredSuite) => {
    initialized.current = true
    const next = evalSuiteDraftFromDocument(document)
    const nextCaseKeys = allocateCaseRowKeys(next.cases.length)
    setDraft(next)
    setCleanIdentity(JSON.stringify(next))
    setCaseRowKeys(nextCaseKeys)
    const requestedIndex = initialCaseId
      ? next.cases.findIndex((item) => item.id === initialCaseId)
      : 0
    setActiveCaseKey(nextCaseKeys[Math.max(0, requestedIndex)] ?? "")
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

  const installInitialSuite = useEffectEvent((document: EvalAuthoredSuite) => {
    if (initialized.current) return
    if (document.revision !== initialRevision || document.target_key !== targetKey) {
      setError("The loaded suite does not match this editor. Return to the suite and try again.")
      return
    }
    loadDocument(document)
  })
  useEffect(() => {
    if (active && !disabled && initialSuite.data) installInitialSuite(initialSuite.data)
  }, [active, disabled, initialSuite.data])

  const addCase = () => {
    if (
      authoringLocked ||
      calibrationLocked ||
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
      calibrationLocked ||
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
      calibrationLocked ||
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
      calibrationLocked ||
      scenarioHasUnsavedChanges ||
      !activeCase ||
      activeCaseRowKey === undefined
    ) {
      return
    }
    const currentKind = isScenarioCase(activeCase) ? "scenario" : "simple_input"
    if (kind === currentKind) return
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
    if (authoringLocked || calibrationLocked) return
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
      if (signal.aborted || !activeRef.current) return
      setSuitePreview(next)
      setSuitePreviewIdentity(identity)
    })
  }

  const saveSuite = () => {
    if (
      disabled ||
      !active ||
      authoringPending ||
      calibrationLocked ||
      !previewCurrent ||
      !suitePreview?.ready
    )
      return
    // A retry owns the original child and parent/null tuple, including acknowledgement loss.
    const request =
      saveIntent ??
      structuredClone({
        expected_suite_revision: suitePreview.suite.revision,
        suite: suitePreview.suite,
        parent_suite_revision: savedRevision,
      })
    setSaveIntent(request)
    void run("save", async (signal) => {
      let saved: EvalAuthoredSuiteSaved
      try {
        saved = await saveEvalAuthoredSuite(request, signal)
      } catch (saveError) {
        if (
          !signal.aborted &&
          saveError instanceof ApiClientError &&
          evalLaunchFailureIsDefinitive(saveError.status)
        ) {
          setSaveIntent(null)
        }
        throw saveError
      }
      if (signal.aborted || !activeRef.current) return
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
      setCleanIdentity(JSON.stringify(normalized))
      setCaseRowKeys(normalizedCaseKeys)
      setSuitePreview({
        suite: saved.suite,
        full_selection: saved.full_selection,
        ready: true,
        diagnostics: [],
      })
      setSuitePreviewIdentity(JSON.stringify(normalized))
      setSavedRevision(saved.suite.revision)
      setSaveIntent(null)
      setSelectedCaseKeys((current) => {
        const retainedKeys = new Set(normalizedCaseKeys)
        return new Set([...current].filter((caseKey) => retainedKeys.has(caseKey)))
      })
      onSaved?.(saved.suite)
      setNotice(`Saved immutable suite ${shortEvalIdentity(saved.suite.revision)}.`)
      await queryClient.invalidateQueries({ queryKey: ["evals"] })
    })
  }

  const previewRun = () => {
    if (authoringLocked || calibrationLocked) return
    if (launchCostBudgetInvalid) {
      setError("Enter a positive cost ceiling in one currency published by this target.")
      return
    }
    if (!savedDraftCurrent || savedRevision === null) {
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
      if (signal.aborted || !activeRef.current) return
      setRunPreview(next)
      setRunPreviewIdentity(identity)
    })
  }

  const launch = () => {
    if (
      authoringLocked ||
      calibrationLocked ||
      !runPreviewCurrent ||
      !runPreview?.ready ||
      !runPreview.exposure ||
      launchCostBudgetInvalid ||
      savedRevision === null ||
      !savedDraftCurrent
    ) {
      return
    }
    const reviewedExposure = runPreview.exposure
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
    let attempt: SuiteLaunchAttempt
    try {
      attempt = {
        key: createEvalIdempotencyKey(),
        revision: savedRevision,
        request: structuredClone({
          ...launchSelection,
          expected_execution_profiles: expectedExecutionProfiles,
          expected_exposure_revision: reviewedExposure.revision,
        }),
      }
      saveSuiteLaunchAttempt(window.sessionStorage, dashboardConfig.apiBaseUrl, attempt)
      setLaunchAttempt(attempt)
    } catch (error) {
      setError(evalErrorMessage(error, "The exact launch request could not be retained."))
      return
    }
    dispatchLaunch(attempt)
  }

  const settleLaunch = (attempt: SuiteLaunchAttempt) => {
    if (clearSuiteLaunchAttempt(window.sessionStorage, dashboardConfig.apiBaseUrl, attempt)) {
      setLaunchAttempt((current) => (current?.key === attempt.key ? null : current))
    }
  }
  const dispatchLaunch = (attempt: SuiteLaunchAttempt) => {
    if (disabled || !active || pending) return
    void run("launch", async (signal) => {
      let launched: EvalAuthoredSuiteRunLaunched
      try {
        launched = await launchEvalAuthoredSuiteRun(
          attempt.revision,
          attempt.request,
          attempt.key,
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
          settleLaunch(attempt)
        }
        throw launchError
      }
      if (signal.aborted || !activeRef.current) return
      const runIds = launched.runs.map((item) => item.run.spec.run_id)
      for (const item of launched.runs) {
        queryClient.setQueryData(["evals", "run", item.run.spec.run_id], item.run)
      }
      await queryClient.invalidateQueries({ queryKey: ["evals"] })
      if (signal.aborted || !activeRef.current) return
      settleLaunch(attempt)
      setNotice(
        `Started ${runIds.length} durable eval ${runIds.length === 1 ? "run" : "runs"} for ${launched.selection.cases.length} selected cases.`,
      )
      await onLaunched(runIds)
    })
  }

  const retainSavedScenario = (caseKey: string, scenario: EvalScenario) => {
    if (!activeRef.current || scenario.target_key !== targetKey) return
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
    if (authoringLocked || calibrationLocked) return
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
      authoringLocked ||
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
    <section
      className={`@container/eval-sheet min-h-0 min-w-0 flex-1 flex-col ${active ? "flex" : "hidden"}`}
      aria-hidden={!active}
      data-testid={active ? "eval-suite-editor" : undefined}
    >
      <header className="border-b border-border p-4">
        <h1 className="text-xl font-semibold">
          {savedRevision ? "Edit suite" : initialRevision ? "Loading suite" : "New suite"}
        </h1>
        <p className="break-words text-sm text-muted-foreground">
          {targetKey} · Save a revision, then review and run it.
        </p>
      </header>
      {initialRevision && !initialized.current && (
        <div className="p-4" role={initialSuite.isError ? "alert" : "status"}>
          {initialSuite.isError ? (
            <>
              <p>The suite could not be loaded.</p>
              <Button variant="outline" onClick={() => void initialSuite.refetch()}>
                Retry loading suite
              </Button>
            </>
          ) : (
            "Loading suite…"
          )}
        </div>
      )}
      {launchRecoveryError && (
        <p role="alert" className="p-4 text-sm">
          {launchRecoveryError}
        </p>
      )}
      {launchAttempt && (
        <section
          aria-label="Unconfirmed launch"
          className="space-y-2 border-b border-border p-4 text-sm"
        >
          <p>
            The previous launch outcome is unconfirmed. Retry its exact reviewed request to recover
            the original run, even if current settings changed.
          </p>
          <p>
            {launchAttempt.request.case_ids?.join(", ") ?? "All cases"} ·{" "}
            {launchAttempt.request.cost_budget
              ? `${launchAttempt.request.cost_budget.max_estimated_cost} ${launchAttempt.request.cost_budget.currency} per trial`
              : "No cost ceiling"}
          </p>
          <div className="flex flex-wrap gap-2">
            <Button
              disabled={disabled || !active || pending !== null}
              onClick={() => dispatchLaunch(launchAttempt)}
            >
              Retry exact launch
            </Button>
            <Button
              variant="outline"
              disabled={disabled || !active || pending !== null}
              onClick={() => {
                if (
                  window.confirm(
                    "The previous run may already be executing. Abandon its recovery and review a separate new launch?",
                  )
                ) {
                  settleLaunch(launchAttempt)
                  setRunPreview(null)
                  setRunPreviewIdentity(null)
                }
              }}
            >
              Abandon launch recovery
            </Button>
          </div>
        </section>
      )}
      {saveIntent && pending !== "save" && (
        <p className="p-4 text-sm" role="status">
          The save outcome is not confirmed. Retry the exact save before editing this revision.
        </p>
      )}
      <div className="flex min-h-0 min-w-0 flex-1 flex-col gap-4 overflow-y-auto px-4 pb-2 @min-[56rem]/eval-sheet:grid @min-[56rem]/eval-sheet:grid-cols-[18rem_minmax(0,1fr)] @min-[56rem]/eval-sheet:overflow-hidden">
        <aside className="max-h-64 min-h-0 min-w-0 shrink-0 space-y-3 overflow-y-auto rounded-lg border border-border p-3 @min-[56rem]/eval-sheet:max-h-none">
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
                  calibrationLocked ||
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
                      disabled={authoringLocked || calibrationLocked}
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
                      {isScenarioCase(evalCase) || scenarioDrafts[caseKey] ? "Scenario" : "Simple"}
                    </Badge>
                  </div>
                )
              })}
            </div>
          </div>
        </aside>

        <div className="@container/eval-form min-w-0 shrink-0 pr-1 @min-[56rem]/eval-sheet:min-h-0 @min-[56rem]/eval-sheet:overflow-y-auto">
          <fieldset className="flex min-w-0 flex-col gap-4" disabled={authoringLocked}>
            <fieldset disabled={calibrationLocked}>
              <EvalSuiteSettings
                draft={draft}
                target={target}
                saved={savedRevision !== null}
                edit={editDraft}
              />
            </fieldset>

            {activeCase && (
              <EvalSuiteCase
                key={activeCaseRowKey}
                evalCase={activeCase}
                disabled={authoringLocked}
                editsLocked={calibrationLocked}
                scenarioDirty={scenarioHasUnsavedChanges}
                scenario={isScenarioCase(activeCase) || scenarioDraftForActiveCase !== undefined}
                onChange={(updated) =>
                  editDraft((next) => {
                    next.cases[activeCaseIndex] = updated
                  })
                }
                onStimulusChange={switchStimulus}
                onDuplicate={duplicateActiveCase}
                onRemove={removeActiveCase}
                canDuplicate={
                  !authoringLocked &&
                  !calibrationLocked &&
                  !scenarioHasUnsavedChanges &&
                  draft.cases.length < EVAL_SUITE_MAX_CASES
                }
                canRemove={
                  !authoringLocked &&
                  !calibrationLocked &&
                  !scenarioHasUnsavedChanges &&
                  draft.cases.length > 1
                }
                input={
                  !isScenarioCase(activeCase) && !scenarioDraftForActiveCase ? (
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
                  )
                }
                checks={
                  <>
                    <EvalSuiteChecks
                      key={activeCaseRowKey}
                      assertions={activeCase.assertions}
                      target={target}
                      defaultJudgeProfile={defaultJudgeProfile}
                      reviewedAssertions={reviewedActiveCase?.assertions ?? []}
                      defaultTask={defaultCalibrationTask}
                      disabled={authoringLocked || calibrationLocked}
                      calibrationDisabled={authoringLocked}
                      onCalibrationStateChange={setCalibrationState}
                      onChange={(assertions) =>
                        editDraft((next) => {
                          const current = next.cases[activeCaseIndex]
                          if (current) current.assertions = assertions
                        })
                      }
                    />

                    <div
                      className="space-y-3 rounded-lg border border-border bg-muted/20 p-3"
                      data-testid="eval-memory-authoring"
                    >
                      <div className="flex flex-wrap items-start justify-between gap-3">
                        <div className="max-w-3xl">
                          <div className="text-sm font-medium">Evaluate memory</div>
                          <div className="mt-1 text-xs text-muted-foreground">
                            Structural evidence proves admission and provider exposure. A trusted
                            reference lets the judge score correct use. Neither alone proves that
                            memory caused the result; use a paired memory experiment for that.
                          </div>
                        </div>
                        <div className="flex flex-wrap gap-2">
                          <Button
                            type="button"
                            size="sm"
                            variant="outline"
                            disabled={
                              authoringLocked ||
                              calibrationLocked ||
                              hasMemoryStructuralAssertion ||
                              activeCase.assertions.length >= 64
                            }
                            onClick={() =>
                              editDraft((next) => {
                                const current = next.cases[activeCaseIndex]
                                if (current) {
                                  current.assertions.push(
                                    createPromotionAssertion(
                                      "memory_attribution",
                                      current.assertions,
                                    ),
                                  )
                                }
                              })
                            }
                          >
                            <Plus />
                            {hasMemoryStructuralAssertion
                              ? "Memory exposure added"
                              : "Require memory exposure"}
                          </Button>
                          <Button
                            type="button"
                            size="sm"
                            variant="outline"
                            disabled={
                              authoringLocked ||
                              calibrationLocked ||
                              defaultMemoryJudgeProfile === undefined ||
                              hasMemorySemanticJudge ||
                              activeCase.assertions.length >= 64
                            }
                            onClick={() => {
                              const profile = defaultMemoryJudgeProfile
                              if (!profile) return
                              editDraft((next) => {
                                const current = next.cases[activeCaseIndex]
                                if (current) {
                                  current.assertions.push(
                                    newMemoryUseJudgeAssertion(
                                      profile,
                                      current.assertions.map((item) => item.id),
                                    ),
                                  )
                                }
                              })
                            }}
                          >
                            <Plus />
                            {hasMemorySemanticJudge
                              ? "Memory judge added"
                              : "Add reference-backed judge"}
                          </Button>
                        </div>
                      </div>
                      {defaultMemoryJudgeProfile === undefined && (
                        <div className="text-xs text-muted-foreground">
                          This target does not publish a trusted judge profile that permits final
                          output and public-reference evidence, so semantic memory scoring is
                          unavailable. Structural checks still work.
                        </div>
                      )}
                      {memorySemanticJudgeNeedsReference && (
                        <div className="text-xs text-amber-700 dark:text-amber-300">
                          Replace the blank expected fact in the memory-use judge with trusted
                          reference truth before checking or saving the suite.
                        </div>
                      )}
                    </div>
                  </>
                }
              />
            )}
          </fieldset>

          {runReviewOpen && (
            <EvalSuiteRunReview
              preview={runPreview}
              current={runPreviewCurrent}
              maximumCost={launchMaximumCost}
              currency={launchCostCurrency}
              budgetInvalid={launchCostBudgetInvalid}
              disabled={authoringLocked}
              canCheck={
                !authoringLocked &&
                !calibrationLocked &&
                savedDraftCurrent &&
                selectedCaseKeys.size > 0 &&
                !launchCostBudgetInvalid
              }
              canLaunch={
                !authoringLocked &&
                !calibrationLocked &&
                runPreviewCurrent &&
                runPreview?.ready === true &&
                Boolean(runPreview.exposure) &&
                !launchCostBudgetInvalid
              }
              pending={pending}
              target={target}
              onMaximumCostChange={setLaunchMaximumCost}
              onCurrencyChange={setLaunchCostCurrency}
              onPreview={previewRun}
              onLaunch={launch}
            />
          )}

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

      <footer className="border-t border-border p-4">
        <div className="flex flex-wrap justify-end gap-2">
          <Button
            type="button"
            variant="outline"
            data-testid="authored-suite-preview"
            disabled={
              authoringLocked ||
              calibrationLocked ||
              unsavedScenarioReference ||
              activeScenarioDirty
            }
            onClick={previewSuite}
          >
            {pending === "preview" ? <LoaderCircle className="animate-spin" /> : <CheckCircle2 />}
            {pending === "preview" ? "Checking..." : "Check suite"}
          </Button>
          <Button
            type="button"
            data-testid="authored-suite-save"
            disabled={
              disabled ||
              !active ||
              authoringPending ||
              calibrationLocked ||
              !previewCurrent ||
              !suitePreview?.ready
            }
            onClick={saveSuite}
          >
            {pending === "save" ? <LoaderCircle className="animate-spin" /> : <Save />}
            {pending === "save" ? "Saving..." : saveIntent ? "Retry exact save" : "Save revision"}
          </Button>
          <Button
            type="button"
            variant="outline"
            disabled={authoringLocked || calibrationLocked || !savedDraftCurrent}
            onClick={() => setRunReviewOpen((current) => !current)}
            aria-expanded={runReviewOpen}
          >
            <FlaskConical /> {runReviewOpen ? "Hide run review" : "Run saved revision"}
          </Button>
        </div>
      </footer>
    </section>
  )
}

function isScenarioCase(evalCase: EvalCaseDraftV2): evalCase is EvalCaseDraftV2 & {
  stimulus: { kind?: "scenario"; scenario_id: string; scenario_revision: string }
} {
  return "scenario_revision" in evalCase.stimulus
}

function withoutKey<T>(source: Record<string, T>, key: string): Record<string, T> {
  const next = { ...source }
  delete next[key]
  return next
}
