import type {
  EvalAdditionCommitRequest,
  EvalAdditionPlan,
  EvalAdditionReceipt,
  EvalAdditionRequest,
  EvalAdditionSourceLink,
} from "@/lib/generated/server-api"
import { queryString, requestJson } from "../../lib/api.ts"

export const previewEvalAddition = (
  session: string,
  request: EvalAdditionRequest,
  signal?: AbortSignal,
) =>
  requestJson<EvalAdditionPlan>(
    `/evals/sessions/${encodeURIComponent(session)}/additions/preview`,
    {
      method: "POST",
      body: JSON.stringify(request),
      signal,
    },
  )
export const publishEvalAddition = (
  session: string,
  request: EvalAdditionCommitRequest,
  key: string,
) =>
  requestJson<EvalAdditionReceipt>(`/evals/sessions/${encodeURIComponent(session)}/additions`, {
    method: "POST",
    body: JSON.stringify(request),
    headers: { "Idempotency-Key": key },
  })
export const fetchEvalCaseSource = (
  suite: string,
  caseId: string,
  resultRevision?: string,
  signal?: AbortSignal,
) =>
  requestJson<EvalAdditionSourceLink>(
    `/evals/suite-views/${encodeURIComponent(suite)}/cases/${encodeURIComponent(caseId)}/source${queryString({ result_revision: resultRevision })}`,
    { signal },
  )
