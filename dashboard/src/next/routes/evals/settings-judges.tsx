import { useQuery } from "@tanstack/react-query"
import { useCallback, useState } from "react"
import { PayloadViewer } from "@/components/dashboard/layout"
import {
  type CalibrationState,
  JudgeCalibrationPanel,
} from "@/components/dashboard/structured-judge-editor"
import { type EvalTarget, fetchEvalAuthoredSuite, fetchEvalAuthoredSuites } from "@/lib/api"
import { shortEvalIdentity } from "@/lib/evals-dashboard"
import type { StructuredModelJudgeAssertionSpec } from "@/lib/generated/server-api"
import { NavLink } from "../common"
import { FIELD_CLASS, LoadState, Pages, Section } from "./shared"

export function JudgeSettings({
  target,
  disabled,
  onStateChange,
}: {
  target: EvalTarget
  disabled: boolean
  onStateChange: (state: CalibrationState) => void
}) {
  const [cursor, setCursor] = useState<string>()
  const [revision, setRevision] = useState("")
  const [caseId, setCaseId] = useState("")
  const [checkId, setCheckId] = useState("")
  const [state, setState] = useState<CalibrationState>({ dirty: false, locked: false })
  const reportState = useCallback(
    (next: CalibrationState) => {
      setState(next)
      onStateChange(next)
    },
    [onStateChange],
  )
  const suites = useQuery({
    queryKey: ["evals", "calibration-suites", target.target_key, cursor],
    queryFn: ({ signal }) =>
      fetchEvalAuthoredSuites({ target_key: target.target_key, limit: 10, cursor }, signal),
  })
  const suite = useQuery({
    queryKey: ["evals", "calibration-suite", revision],
    queryFn: ({ signal }) => fetchEvalAuthoredSuite(revision, signal),
    enabled: !!revision,
  })
  const document = suite.data?.target_key === target.target_key ? suite.data : undefined
  const selectedCase = document?.cases.find((item) => item.id === caseId)
  const checks =
    selectedCase?.assertions.filter(
      (item): item is StructuredModelJudgeAssertionSpec => item.kind === "structured_model_judge",
    ) ?? []
  const check = checks.find((item) => item.id === checkId)
  const scopeLocked = state.dirty || state.locked
  return (
    <>
      <Section title="Judges and references">
        <p className="mb-3 text-sm text-muted-foreground">
          Judge profiles and private references are configured in the agent. Edit rubrics and public
          references in a suite’s checks.
        </p>
        <JudgeProfiles target={target} details />
        <NavLink
          to="/evals/suites/new"
          search={{ target: target.target_key }}
          className="mt-3 inline-block text-sm underline"
        >
          Create a suite with judge checks
        </NavLink>
      </Section>
      <Section title="Judge calibration">
        <p className="mb-4 text-sm text-muted-foreground">
          Choose an immutable suite revision, case and structured check. Compare repeated judge
          scores with human scores on fixed evidence, without running the specialist.
        </p>
        <LoadState error={suites.error} pending={suites.isPending} retry={suites.refetch} />
        <fieldset disabled={scopeLocked} className="space-y-3">
          <label className="block text-sm">
            Saved suite revision
            <select
              aria-label="Calibration suite"
              className={`${FIELD_CLASS} mt-1 w-full`}
              value={revision}
              onChange={(event) => {
                setRevision(event.target.value)
                setCaseId("")
                setCheckId("")
              }}
            >
              <option value="">Choose a saved suite</option>
              {suites.data?.items.map((item) => (
                <option key={item.revision} value={item.revision}>
                  {item.name} · {shortEvalIdentity(item.revision)}
                </option>
              ))}
            </select>
          </label>
          <Pages
            label="Calibration suites"
            hasMore={!!suites.data?.has_more}
            cursor={cursor}
            busy={suites.isFetching}
            onFirst={() => {
              setCursor(undefined)
              setRevision("")
              setCaseId("")
              setCheckId("")
            }}
            onNext={() => {
              setCursor(suites.data?.next_cursor ?? undefined)
              setRevision("")
              setCaseId("")
              setCheckId("")
            }}
          />
          {suites.data?.items.length === 0 && (
            <p className="text-sm text-muted-foreground">
              No saved authored suites on this page. Save a suite with a structured judge check
              first.
            </p>
          )}
          {document && (
            <label className="block text-sm">
              Case
              <select
                aria-label="Calibration case"
                className={`${FIELD_CLASS} mt-1 w-full`}
                value={caseId}
                onChange={(event) => {
                  setCaseId(event.target.value)
                  setCheckId("")
                }}
              >
                <option value="">Choose a case</option>
                {document.cases.map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.name || item.id}
                  </option>
                ))}
              </select>
            </label>
          )}
          {selectedCase && (
            <label className="block text-sm">
              Structured check
              <select
                aria-label="Calibration check"
                className={`${FIELD_CLASS} mt-1 w-full`}
                value={checkId}
                onChange={(event) => setCheckId(event.target.value)}
              >
                <option value="">Choose a structured judge check</option>
                {checks.map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.description || item.id}
                  </option>
                ))}
              </select>
              {checks.length === 0 && (
                <span className="mt-2 block text-xs text-muted-foreground">
                  This case has no structured judge checks. Add one in the suite editor.
                </span>
              )}
            </label>
          )}
        </fieldset>
        {revision && (
          <LoadState error={suite.error} pending={suite.isPending} retry={suite.refetch} />
        )}
        {scopeLocked && (
          <p className="mt-3 text-sm text-muted-foreground">
            Finish or discard this calibration before choosing another check.
          </p>
        )}
        {check && selectedCase && (
          <div className="mt-4" key={JSON.stringify([revision, caseId, checkId])}>
            <JudgeCalibrationPanel
              targetKey={target.target_key}
              assertion={check}
              defaultTask={selectedCase.name || selectedCase.id}
              disabled={disabled}
              onStateChange={reportState}
            />
          </div>
        )}
      </Section>
    </>
  )
}

export function JudgeProfiles({
  target,
  details = false,
}: {
  target: EvalTarget
  details?: boolean
}) {
  return target.judge_profiles?.length ? (
    <div className="space-y-2">
      {target.judge_profiles.map((profile) => (
        <div
          key={profile.revision}
          className="rounded-xl border border-border p-3 text-sm break-words"
        >
          <p className="font-medium">{profile.label || profile.key}</p>
          <p className="text-muted-foreground">
            {profile.model} via {profile.provider_name}
          </p>
          <p className="mt-1 text-xs text-muted-foreground">
            {profile.same_model_use === "allowed_and_labeled"
              ? "Same model allowed and labelled"
              : profile.same_model_use === "forbidden"
                ? "Same model forbidden"
                : "Same-model policy unavailable"}
          </p>
          {details && (
            <details className="mt-2">
              <summary className="cursor-pointer text-xs">Judge details</summary>
              <PayloadViewer value={profile} />
            </details>
          )}
        </div>
      ))}
    </div>
  ) : (
    <p className="text-sm text-muted-foreground">No judge profile is configured.</p>
  )
}
