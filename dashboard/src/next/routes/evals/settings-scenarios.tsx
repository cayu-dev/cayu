import { useQuery } from "@tanstack/react-query"
import { Download } from "lucide-react"
import { useCallback, useEffect, useRef, useState } from "react"
import {
  ScenarioAuthoring,
  type ScenarioAuthoringState,
} from "@/components/dashboard/scenario-authoring"
import type { CalibrationState } from "@/components/dashboard/structured-judge-editor"
import { Button } from "@/components/ui/button"
import {
  downloadEvalScenario,
  type EvalScenario,
  type EvalScenarioDraft,
  fetchEvalScenario,
  fetchEvalScenarios,
} from "@/lib/api"
import { blankEvalScenarioDraft, newSimpleEvalCase } from "@/lib/eval-suite-authoring"
import { evalErrorMessage, shortEvalIdentity } from "@/lib/evals-dashboard"
import { formatDateTime } from "@/lib/format"
import { nativeRunSelector, runViewRoute } from "../../app/eval-view-model"
import { NavLink } from "../common"
import { LoadState, Pages, Section } from "./shared"
import { downloadBlob } from "./technical-details"

export function ScenarioSettings({
  targetKey,
  disabled,
  onStateChange,
}: {
  targetKey: string
  disabled: boolean
  onStateChange: (state: CalibrationState) => void
}) {
  const [cursor, setCursor] = useState<string>()
  const [revision, setRevision] = useState<string>()
  const [reset, setReset] = useState(0)
  const [state, setState] = useState<ScenarioAuthoringState>({
    dirty: false,
    pending: false,
    edited: false,
  })
  const [download, setDownload] = useState<string>()
  const [error, setError] = useState<string>()
  const controller = useRef<AbortController | null>(null)
  useEffect(() => () => controller.current?.abort(), [])
  const reportState = useCallback((next: ScenarioAuthoringState) => setState(next), [])
  useEffect(() => {
    onStateChange({ dirty: state.dirty, locked: state.pending || !!download })
  }, [state, download, onStateChange])
  const locked = state.dirty || state.pending || !!download
  const scenarios = useQuery({
    queryKey: ["evals", "scenarios", targetKey, cursor],
    queryFn: ({ signal }) =>
      fetchEvalScenarios({ target_key: targetKey, limit: 10, cursor }, signal),
  })
  const scenario = useQuery({
    queryKey: ["evals", "scenario", revision],
    queryFn: ({ signal }) => fetchEvalScenario(revision!, signal),
    enabled: !!revision && revision !== "new",
  })
  const create = () => {
    setState({ dirty: false, pending: false, edited: false })
    setRevision("new")
    setReset((value) => value + 1)
  }
  const saveDownload = async (value: string) => {
    if (download) return
    const operation = new AbortController()
    controller.current = operation
    setDownload(value)
    setError(undefined)
    try {
      const file = await downloadEvalScenario(value, operation.signal)
      if (!operation.signal.aborted) downloadBlob(file.blob, file.filename)
    } catch (failure) {
      if (!operation.signal.aborted)
        setError(evalErrorMessage(failure, "Could not download the scenario."))
    } finally {
      if (controller.current === operation) {
        controller.current = null
        setDownload(undefined)
      }
    }
  }
  return (
    <Section title="Production scenarios">
      <p className="mb-4 text-sm text-muted-foreground">
        Build a multi-step input or reuse one captured from a session. Each saved revision can run
        on its own or become a suite case.
      </p>
      <div className="mb-4 flex flex-wrap gap-2">
        <Button variant="outline" size="sm" disabled={disabled || locked} onClick={create}>
          New scenario
        </Button>
        {state.dirty && (
          <Button
            variant="outline"
            size="sm"
            disabled={state.pending}
            onClick={() => {
              setState({ dirty: false, pending: false, edited: false })
              if (revision === "new") setRevision(undefined)
              setReset((value) => value + 1)
            }}
          >
            Discard scenario edits
          </Button>
        )}
      </div>
      {error && (
        <p role="alert" className="mb-3 text-sm text-destructive">
          {error}
        </p>
      )}
      <div data-testid="scenario-catalog" className="grid min-w-0 gap-5 @container/scenarios">
        <div className="min-w-0">
          <LoadState
            error={scenarios.error}
            pending={scenarios.isPending}
            retry={scenarios.refetch}
          />
          <ul
            aria-label="Saved scenarios"
            className="divide-y divide-border rounded-xl border border-border"
          >
            {scenarios.data?.items.map((entry) => (
              <li key={entry.revision} className="flex min-w-0 items-start gap-3 p-3 text-sm">
                <button
                  type="button"
                  disabled={locked}
                  aria-pressed={revision === entry.revision}
                  className="min-w-0 flex-1 rounded text-left break-words hover:text-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:opacity-60"
                  onClick={() => {
                    setRevision(entry.revision)
                    setReset((value) => value + 1)
                  }}
                >
                  <span
                    className={`block ${revision === entry.revision ? "font-semibold text-primary" : "font-medium"}`}
                  >
                    {entry.name}
                  </span>
                  <span className="mt-1 block text-xs text-muted-foreground">
                    {entry.event_count} events · {entry.artifact_requirement_count} files ·{" "}
                    {entry.secret_requirement_count} secrets
                  </span>
                  <span className="mt-1 block text-xs text-muted-foreground">
                    {formatDateTime(entry.created_at)} · {shortEvalIdentity(entry.revision)}
                  </span>
                </button>
                <Button
                  variant="ghost"
                  size="icon-xs"
                  aria-label={`Download scenario ${entry.name}`}
                  disabled={!!download}
                  onClick={() => void saveDownload(entry.revision)}
                >
                  <Download />
                </Button>
              </li>
            ))}
          </ul>
          {scenarios.data?.items.length === 0 && (
            <p className="mt-3 text-sm text-muted-foreground">
              No scenarios on this page. Create one here, or use Add to evals on a completed
              multi-step session.
            </p>
          )}
          <fieldset disabled={locked}>
            <Pages
              label="Scenario catalog"
              cursor={cursor}
              hasMore={!!scenarios.data?.has_more}
              busy={scenarios.isFetching}
              onFirst={() => setCursor(undefined)}
              onNext={() => setCursor(scenarios.data?.next_cursor ?? undefined)}
            />
          </fieldset>
        </div>
        {revision && revision !== "new" && (
          <LoadState error={scenario.error} pending={scenario.isPending} retry={scenario.refetch} />
        )}
        {revision === "new" ? (
          <OwnedScenario
            key={`new:${reset}`}
            captured={blankEvalScenarioDraft(targetKey, newSimpleEvalCase([]))}
            saved={false}
            disabled={disabled}
            onStateChange={reportState}
          />
        ) : (
          scenario.data?.target_key === targetKey && (
            <OwnedScenario
              key={`${revision}:${reset}`}
              captured={scenario.data}
              saved
              disabled={disabled}
              onStateChange={reportState}
            />
          )
        )}
      </div>
    </Section>
  )
}

function OwnedScenario({
  captured,
  saved,
  disabled,
  onStateChange,
}: {
  captured: EvalScenario | EvalScenarioDraft
  saved: boolean
  disabled: boolean
  onStateChange: (state: ScenarioAuthoringState) => void
}) {
  // Immutable selection owns this editor until an explicit selection/discard remount.
  const [initial] = useState(captured)
  const [launched, setLaunched] = useState<string>()
  return (
    <div className="min-w-0 space-y-3">
      {launched && (
        <NavLink {...runViewRoute(nativeRunSelector(launched))} className="text-sm underline">
          Open scenario run
        </NavLink>
      )}
      <ScenarioAuthoring
        captured={initial}
        saved={saved}
        disabled={disabled}
        onAuthoringStateChange={onStateChange}
        onLaunched={setLaunched}
      />
    </div>
  )
}
