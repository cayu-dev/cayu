import { useInfiniteQuery, useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { useRouter } from "@tanstack/react-router"
import { useEffect, useId, useRef, useState } from "react"
import {
  EvaluationPromotionAction,
  ExpectedBehaviorEditor,
} from "@/components/dashboard/evaluation-promotion"
import { PayloadViewer } from "@/components/dashboard/layout"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet"
import { ApiClientError, previewCapturedEvaluation } from "@/lib/api"
import { dashboardConfig } from "@/lib/config"
import {
  clearEvalAdditionAttempt,
  type EvalAdditionAttempt,
  readEvalAdditionAttempt,
  saveEvalAdditionAttempt,
} from "@/lib/eval-addition-attempt"
import { createEvalIdempotencyKey, shortEvalIdentity } from "@/lib/evals-dashboard"
import { evalsReadinessReasonText } from "@/lib/evals-readiness"
import type { PromotionAssertion } from "@/lib/evaluation-promotion"
import type {
  EvalAdditionPlan,
  EvalAdditionRequest,
  EvalSuiteViewSummary,
} from "@/lib/generated/server-api"
import { cn } from "@/lib/utils"
import { previewEvalAddition, publishEvalAddition } from "../../api/eval-additions"
import { fetchSuiteViews } from "../../api/eval-views"
import {
  resultSelector,
  runViewPath,
  suiteSelector,
  suiteViewPath,
} from "../../app/eval-view-model"
import { useGo } from "../common"
import { FIELD_CLASS, LoadState, useEvalAccess } from "../evals/shared"

const modes = [
  {
    id: "case",
    label: "Add as a case",
    text: "Keep its captured input as a runnable case in a new suite revision.",
  },
  {
    id: "score",
    label: "Score this session",
    text: "Check retained evidence without rerunning it. Record one captured case in this suite’s history.",
  },
] as const

export function SessionEvalAdditionDialog({
  sessionId,
  status,
  open,
  onOpenChange,
}: {
  sessionId: string
  status: string
  open: boolean
  onOpenChange: (open: boolean) => void
}) {
  const { canWrite, readiness } = useEvalAccess()
  const [capturedOpen, setCapturedOpen] = useState(false)
  if (!canWrite) return null
  return (
    <Sheet
      open={open}
      onOpenChange={(next, event) => {
        if (!next && capturedOpen) event.cancel()
        else onOpenChange(next)
      }}
    >
      <SheetContent className="w-full! gap-0 sm:max-w-2xl!" data-testid="eval-addition-sheet">
        <SheetHeader className="border-b border-border pr-12">
          <SheetTitle>Add to evals</SheetTitle>
          <SheetDescription>
            Choose a suite and review the exact addition before saving.
          </SheetDescription>
        </SheetHeader>
        {open && (
          <AdditionForm
            key={sessionId}
            sessionId={sessionId}
            status={status}
            onClose={() => onOpenChange(false)}
            onCapturedOpenChange={setCapturedOpen}
            unavailable={
              !["completed", "failed"].includes(status)
                ? "The source session is currently unavailable for new additions."
                : readiness.session_additions.state === "ready"
                  ? null
                  : evalsReadinessReasonText(readiness.session_additions)
            }
          />
        )}
      </SheetContent>
    </Sheet>
  )
}

function AdditionForm({
  sessionId,
  status,
  onClose,
  onCapturedOpenChange,
  unavailable,
}: {
  sessionId: string
  status: string
  onClose: () => void
  onCapturedOpenChange: (open: boolean) => void
  unavailable: string | null
}) {
  const go = useGo()
  const router = useRouter()
  const navigationOrigin = useRef(router.state.location.href)
  const client = useQueryClient()
  const [mode, setMode] = useState<"case" | "score">("case")
  const [destination, setDestination] = useState("")
  const [newName, setNewName] = useState("")
  const [newId, setNewId] = useState("")
  const [caseName, setCaseName] = useState("")
  const [caseId, setCaseId] = useState("")
  const [assertions, setAssertions] = useState<PromotionAssertion[]>([])
  const [plan, setPlan] = useState<{ request: EvalAdditionRequest; plan: EvalAdditionPlan } | null>(
    null,
  )
  const [pending, setPending] = useState<EvalAdditionAttempt | null>(null)
  const [storageError, setStorageError] = useState<Error | null>(null)
  const [recoveryInitialized, setRecoveryInitialized] = useState(false)
  const initialized = useRef(false)
  const fieldId = useId()
  const alive = useRef(true)
  const previewController = useRef<AbortController | null>(null)
  useEffect(() => {
    alive.current = true
    return () => {
      alive.current = false
      previewController.current?.abort()
    }
  }, [])
  useEffect(() => {
    try {
      setPending(
        readEvalAdditionAttempt(window.sessionStorage, dashboardConfig.apiBaseUrl, sessionId),
      )
    } catch (error) {
      setStorageError(
        error instanceof Error ? error : new Error("Browser recovery storage is unavailable."),
      )
    } finally {
      setRecoveryInitialized(true)
    }
  }, [sessionId])
  const capture = useQuery({
    queryKey: ["evals", "addition-capture", sessionId],
    queryFn: ({ signal }) => previewCapturedEvaluation(sessionId, undefined, signal),
    enabled: recoveryInitialized && !pending && !storageError && !unavailable,
    gcTime: 0,
  })
  useEffect(() => {
    if (!capture.data || initialized.current) return
    initialized.current = true
    const candidate = capture.data.candidate
    setCaseId(candidate.case.id)
    setCaseName(candidate.case.name)
    setNewId(candidate.suite.id)
    setNewName(candidate.suite.name)
    setAssertions(candidate.case.assertions)
  }, [capture.data])
  const suites = useInfiniteQuery({
    queryKey: ["evals", "addition-suites", capture.data?.candidate.target_key],
    queryFn: ({ pageParam, signal }) =>
      fetchSuiteViews(
        { target_key: capture.data?.candidate.target_key, cursor: pageParam },
        signal,
      ),
    initialPageParam: undefined as string | undefined,
    getNextPageParam: (last) => (last.has_more ? (last.next_cursor ?? undefined) : undefined),
    enabled: !!capture.data && !pending,
  })
  const choices =
    suites.data?.pages
      .flatMap((page) => page.items)
      .filter(
        (suite): suite is EvalSuiteViewSummary =>
          suite.source === "authored" || suite.source === "corpus",
      ) ?? []
  const chosen = destination || choices[0]?.id || "new"
  const request: EvalAdditionRequest = {
    mode,
    destination:
      chosen === "new"
        ? { kind: "new", suite_id: newId, name: newName }
        : { kind: "existing", selector: chosen },
    case_id: caseId,
    case_name: caseName,
    assertions,
  }
  const material = JSON.stringify(request)
  const currentPlan = plan && JSON.stringify(plan.request) === material ? plan.plan : null
  const preview = useMutation({
    mutationFn: (reviewedRequest: EvalAdditionRequest) => {
      previewController.current?.abort()
      previewController.current = new AbortController()
      return previewEvalAddition(sessionId, reviewedRequest, previewController.current.signal)
    },
    onSuccess: (value, reviewedRequest) =>
      setPlan({ request: structuredClone(reviewedRequest), plan: value }),
  })
  const publish = useMutation({
    mutationFn: async ({
      attempt,
      recovering,
    }: {
      attempt: EvalAdditionAttempt
      recovering: boolean
    }) => {
      if (storageError) throw storageError
      if (!recovering) {
        saveEvalAdditionAttempt(
          window.sessionStorage,
          dashboardConfig.apiBaseUrl,
          sessionId,
          attempt,
        )
        setPending(attempt)
      }
      return publishEvalAddition(sessionId, attempt.request, attempt.key)
    },
    onSuccess: async (receipt, { attempt }) => {
      clearEvalAdditionAttempt(
        window.sessionStorage,
        dashboardConfig.apiBaseUrl,
        sessionId,
        attempt,
      )
      if (alive.current) setPending(null)
      await client.invalidateQueries({ queryKey: ["evals"] })
      if (!alive.current || router.state.location.href !== navigationOrigin.current) return
      onClose()
      const destination = suiteSelector(receipt.plan.destination)
      if (receipt.plan.captured_result_revision)
        await go(runViewPath(resultSelector(receipt.plan.captured_result_revision)), {
          placement: destination,
        })
      else await go(suiteViewPath(destination))
    },
    onError: (error, { attempt, recovering }) => {
      // Only a first-dispatch rejection settles this browser's attempt. A retry
      // can race an earlier request still preparing its publication.
      if (
        !recovering &&
        error instanceof ApiClientError &&
        [400, 404, 409, 413, 422].includes(error.status)
      ) {
        clearEvalAdditionAttempt(
          window.sessionStorage,
          dashboardConfig.apiBaseUrl,
          sessionId,
          attempt,
        )
        if (alive.current) {
          setPending(null)
          setPlan(null)
        }
      }
    },
  })
  const reviewed = pending?.plan ?? currentPlan
  const busy = preview.isPending || publish.isPending
  const valid =
    caseName.trim() &&
    /^[a-z][a-z0-9._-]{0,127}$/.test(caseId) &&
    assertions.length > 0 &&
    (chosen !== "new" || (newName.trim() && /^[a-z][a-z0-9._-]{0,127}$/.test(newId)))
  const conversionUnavailable =
    mode === "case" && capture.data?.runnable_conversion.available === false
  return (
    <div className="min-h-0 flex-1 overflow-y-auto p-4" aria-busy={busy}>
      <div className="space-y-4">
        {unavailable && (
          <p role="status" className="text-sm text-muted-foreground">
            {unavailable}
          </p>
        )}
        {pending && (
          <p role="status" className="rounded-lg border border-border p-3 text-sm">
            An addition is awaiting confirmation. Retry the saved request to recover its result;
            this uses the same operation key.
          </p>
        )}
        {!pending && !unavailable && (
          <>
            <LoadState error={capture.error} pending={capture.isLoading} retry={capture.refetch} />
            {capture.data && (
              <fieldset disabled={busy || !!storageError} className="space-y-4">
                <legend className="sr-only">How to add this session</legend>
                <div className="space-y-2">
                  {modes.map((item) => (
                    <label
                      key={item.id}
                      className={cn(
                        "flex cursor-pointer gap-3 rounded-lg border p-3 text-sm",
                        mode === item.id ? "border-foreground" : "border-border hover:bg-muted/50",
                      )}
                    >
                      <input
                        type="radio"
                        name="eval-addition-mode"
                        value={item.id}
                        checked={mode === item.id}
                        onChange={() => setMode(item.id)}
                        className="mt-0.5 accent-foreground"
                      />
                      <span>
                        <span className="font-medium">{item.label}</span>
                        <span className="mt-0.5 block text-xs text-muted-foreground">
                          {item.text}
                        </span>
                      </span>
                    </label>
                  ))}
                </div>
                <LoadState error={suites.error} pending={suites.isPending} retry={suites.refetch} />
                <label className="block space-y-1 text-sm">
                  <span>Suite</span>
                  <select
                    aria-label="Suite"
                    className={cn(FIELD_CLASS, "w-full")}
                    value={chosen}
                    onChange={(event) => setDestination(event.target.value)}
                  >
                    {choices.map((suite) => (
                      <option key={suite.id} value={suite.id}>
                        {suite.name} · {suite.source} · {suite.case_count} cases ·{" "}
                        {shortEvalIdentity(suite.identity.revision)}
                      </option>
                    ))}
                    <option value="new">New suite…</option>
                  </select>
                </label>
                {suites.hasNextPage && (
                  <Button
                    size="sm"
                    variant="outline"
                    disabled={suites.isFetchingNextPage}
                    onClick={() => void suites.fetchNextPage()}
                  >
                    Load more suites
                  </Button>
                )}
                {!suites.isPending && !suites.error && !choices.length && (
                  <p className="text-xs text-muted-foreground">
                    No suites for this session’s target yet. Create one below.
                  </p>
                )}
                {chosen === "new" && (
                  <label htmlFor={`${fieldId}-suite-name`} className="block space-y-1 text-sm">
                    <span>New suite name</span>
                    <Input
                      id={`${fieldId}-suite-name`}
                      value={newName}
                      maxLength={256}
                      onChange={(event) => setNewName(event.target.value)}
                    />
                  </label>
                )}
                <label htmlFor={`${fieldId}-case-name`} className="block space-y-1 text-sm">
                  <span>Case name</span>
                  <Input
                    id={`${fieldId}-case-name`}
                    value={caseName}
                    maxLength={256}
                    onChange={(event) => setCaseName(event.target.value)}
                  />
                </label>
                <details className="rounded-lg border border-border p-3">
                  <summary className="cursor-pointer text-sm">
                    Case identifiers and expectations
                  </summary>
                  <div className="mt-3 space-y-3">
                    <label htmlFor={`${fieldId}-case-id`} className="block space-y-1 text-sm">
                      <span>Case ID</span>
                      <Input
                        id={`${fieldId}-case-id`}
                        value={caseId}
                        maxLength={128}
                        onChange={(event) => setCaseId(event.target.value)}
                      />
                    </label>
                    {chosen === "new" && (
                      <label htmlFor={`${fieldId}-suite-id`} className="block space-y-1 text-sm">
                        <span>New suite ID</span>
                        <Input
                          id={`${fieldId}-suite-id`}
                          value={newId}
                          maxLength={128}
                          onChange={(event) => setNewId(event.target.value)}
                        />
                      </label>
                    )}
                    <ExpectedBehaviorEditor
                      assertions={assertions}
                      evidence={capture.data.candidate.evidence}
                      evidencePolicy={capture.data.candidate.evidence_policy}
                      onChange={setAssertions}
                    />
                  </div>
                </details>
              </fieldset>
            )}
            {conversionUnavailable && (
              <p role="status" className="text-sm text-muted-foreground">
                This session’s input cannot become a runnable case (
                {capture.data?.runnable_conversion.reason_code}). You can still score its retained
                evidence.
              </p>
            )}
          </>
        )}
        {reviewed && (
          <section
            className="space-y-2 rounded-lg border border-border p-3"
            aria-label="Reviewed addition"
          >
            <p className="text-sm font-medium">{reviewed.case_name}</p>
            <p className="text-sm text-muted-foreground">
              {reviewed.mode === "score"
                ? reviewed.membership_changed
                  ? "One captured case will be scored in a new captured suite."
                  : "One captured case will be scored. Existing runnable membership stays unchanged."
                : reviewed.membership_changed
                  ? `Save a new revision with ${reviewed.destination_case_count} ${reviewed.destination_case_count === 1 ? "case" : "cases"}.`
                  : "This exact case is already present. Membership stays unchanged."}
            </p>
            <details>
              <summary className="cursor-pointer text-xs">Review exact case</summary>
              <PayloadViewer value={reviewed.case_definition} />
            </details>
          </section>
        )}
        {(storageError || preview.error || publish.error) && (
          <p
            role="alert"
            className="rounded-lg border border-destructive/30 p-3 text-sm break-words text-destructive"
          >
            {(storageError ?? publish.error ?? preview.error)?.message}
          </p>
        )}
        {pending && publish.error instanceof ApiClientError && publish.error.status === 409 && (
          <div className="space-y-2 rounded-lg border border-border p-3 text-sm">
            <p>
              The saved request was rejected. An earlier request may still complete. Reviewing again
              creates a separate operation.
            </p>
            <Button
              size="sm"
              variant="outline"
              onClick={() => {
                clearEvalAdditionAttempt(
                  window.sessionStorage,
                  dashboardConfig.apiBaseUrl,
                  sessionId,
                  pending,
                )
                setPending(null)
                setPlan(null)
                publish.reset()
                preview.reset()
              }}
            >
              Review a new addition
            </Button>
          </div>
        )}
        <div className="flex flex-wrap justify-end gap-2">
          <Button variant="ghost" size="sm" onClick={onClose}>
            Close
          </Button>
          {!pending && (
            <Button
              variant="outline"
              size="sm"
              disabled={
                busy ||
                !valid ||
                !capture.data ||
                !!unavailable ||
                !!storageError ||
                conversionUnavailable
              }
              onClick={() => preview.mutate(structuredClone(request))}
            >
              {preview.isPending ? "Reviewing…" : "Review addition"}
            </Button>
          )}
          <Button
            size="sm"
            disabled={busy || !reviewed || !!storageError || !!unavailable}
            onClick={() => {
              if (!pending && !currentPlan) return
              const attempt = pending ?? {
                key: createEvalIdempotencyKey(),
                request: {
                  ...structuredClone(request),
                  expected_plan_revision: currentPlan!.revision,
                  expected_source_revision: currentPlan!.source_revision,
                },
                plan: structuredClone(currentPlan!),
              }
              publish.mutate({ attempt, recovering: pending !== null })
            }}
          >
            {publish.isPending
              ? "Saving…"
              : pending
                ? "Retry saved addition"
                : reviewed?.mode === "score"
                  ? "Score and save"
                  : "Add case"}
          </Button>
        </div>
        <details className="border-t border-border pt-3">
          <summary className="cursor-pointer text-sm text-muted-foreground">
            Captured evidence, export and fresh execution
          </summary>
          <div className="mt-3">
            <EvaluationPromotionAction
              sessionId={sessionId}
              status={status}
              label="Open captured evaluation details"
              onOpenChange={onCapturedOpenChange}
            />
          </div>
        </details>
      </div>
    </div>
  )
}
