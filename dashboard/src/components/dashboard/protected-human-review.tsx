import { useEffect, useRef, useState } from "react"
import { ApiClientError, throwResponseError } from "../../lib/api"
import { apiUrl } from "../../lib/config"
import { roleForbiddenMessage } from "../../lib/dashboard-access"
import type {
  HumanReviewLongFieldView,
  HumanReviewView,
  ToolApprovalBody,
  UserInputResolveBody,
} from "../../lib/generated/server-api"
import { Button } from "../ui/button"
import { Card, CardContent } from "../ui/card"
import { Input } from "../ui/input"

export type ProtectedReviewDecision = ToolApprovalBody | UserInputResolveBody

// Decode the server's html.escape encoding once, then let React render text.
function reviewText(value: string): string {
  const entities: Record<string, string> = {
    "&amp;": "&",
    "&lt;": "<",
    "&gt;": ">",
    "&quot;": '"',
    "&#x27;": "'",
  }
  return value.replace(/&(?:amp|lt|gt|quot|#x27);/g, (entity) => entities[entity] ?? entity)
}

export function ProtectedHumanReview({
  sessionId,
  purpose,
  kind,
  disabled,
  unavailableReason,
  onDecision,
}: {
  sessionId: string
  purpose: string
  kind: "user_input" | "tool_approval"
  disabled: boolean
  unavailableReason: string | null
  onDecision: (body: ProtectedReviewDecision) => Promise<void>
}) {
  // Protected material stays here, outside shared query caches and persistence.
  const [view, setView] = useState<HumanReviewView | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [answer, setAnswer] = useState("")
  const [reason, setReason] = useState("")
  // Complete text of long fields, by key, for the view on screen only.
  const [longText, setLongText] = useState<Record<string, string>>({})
  const [reading, setReading] = useState<string | null>(null)
  const request = useRef<AbortController | null>(null)
  const longRequest = useRef<AbortController | null>(null)
  const deciding = useRef(false)
  useEffect(
    () => () => {
      request.current?.abort()
      longRequest.current?.abort()
    },
    [],
  )

  const forgetLongText = () => {
    longRequest.current?.abort()
    setLongText({})
    setReading(null)
  }

  const readLongField = async (key: string) => {
    if (!view?.reference || reading || busy) return
    const controller = new AbortController()
    longRequest.current = controller
    setReading(key)
    setError(null)
    try {
      const response = await fetch(
        apiUrl(`/sessions/${encodeURIComponent(view.session_id)}/human-review/long-field`),
        {
          method: "POST",
          credentials: "include",
          cache: "no-store",
          headers: { Accept: "application/json", "Content-Type": "application/json" },
          body: JSON.stringify({ review_reference: view.reference, key }),
          signal: controller.signal,
        },
      )
      if (!response.ok) await throwResponseError(response)
      const field: HumanReviewLongFieldView = await response.json()
      if (field.key !== key || field.session_id !== view.session_id) {
        throw new Error("The field does not match this review.")
      }
      if (field.status !== "permitted" || field.text == null) {
        throw new Error("The complete text is withheld or unavailable.")
      }
      const text = reviewText(field.text)
      if (!controller.signal.aborted) setLongText((current) => ({ ...current, [key]: text }))
    } catch (failure) {
      if (!controller.signal.aborted)
        setError(
          `${failure instanceof Error ? failure.message : "Reading the field failed."} Refresh the review and try again.`,
        )
    } finally {
      if (!controller.signal.aborted) setReading(null)
    }
  }

  const inspect = async () => {
    request.current?.abort()
    const controller = new AbortController()
    request.current = controller
    forgetLongText()
    setView(null)
    setAnswer("")
    setReason("")
    setError(null)
    setBusy(true)
    try {
      const response = await fetch(
        apiUrl(
          `/sessions/${encodeURIComponent(sessionId)}/human-review?purpose=${encodeURIComponent(purpose)}`,
        ),
        {
          credentials: "include",
          cache: "no-store",
          signal: controller.signal,
        },
      )
      if (!response.ok) await throwResponseError(response)
      const next: HumanReviewView = await response.json()
      if (
        next.session_id !== sessionId ||
        (next.kind != null && next.kind !== kind) ||
        (next.reference && next.reference.context.purpose !== purpose)
      ) {
        throw new Error("The review does not match this pending interaction.")
      }
      if (!controller.signal.aborted) setView(next)
    } catch (error) {
      if (!controller.signal.aborted)
        setError(
          (error instanceof ApiClientError
            ? roleForbiddenMessage(error.status, error.detail)
            : null) ??
            "Review access was denied or is unavailable. Refresh the review or contact the application owner.",
        )
    } finally {
      if (!controller.signal.aborted) setBusy(false)
    }
  }

  const decide = async (decision: "approve" | "deny" | "answer") => {
    if (!view?.reference || !view.interaction_id || busy || disabled || deciding.current) return
    if (decision !== "deny" && view.status !== "permitted") return
    deciding.current = true
    setBusy(true)
    setError(null)
    try {
      if (decision === "answer") {
        await onDecision({
          session_id: view.session_id,
          input_id: view.interaction_id,
          answer,
          review_reference: view.reference,
        })
      } else {
        if (!view.tool_round_id || !view.tool_call_id)
          throw new Error("Review decision identities are missing.")
        await onDecision({
          session_id: view.session_id,
          approval_id: view.interaction_id,
          tool_round_id: view.tool_round_id,
          tool_call_id: view.tool_call_id,
          decision,
          reason: reason.trim() || null,
          review_reference: view.reference,
        })
      }
      forgetLongText()
      setView(null)
      setAnswer("")
      setReason("")
    } catch (failure) {
      forgetLongText()
      setView(null)
      setError(
        `${failure instanceof Error ? failure.message : "Decision failed."} Refresh the review and inspect it before deciding again. An unconfirmed decision may still have completed.`,
      )
    } finally {
      deciding.current = false
      setBusy(false)
    }
  }

  const bound = view?.reference && view.interaction_id && view.kind === kind
  const blocked = disabled || busy || !bound
  return (
    <Card>
      <CardContent className="space-y-4 p-4">
        <p className="font-medium">Protected human review</p>
        <Button variant="outline" disabled={busy || disabled} onClick={() => void inspect()}>
          Refresh review
        </Button>
        {!view && <p>Inspect the current protected review before deciding.</p>}
        {unavailableReason && <p>{unavailableReason}</p>}
        {view && (
          <>
            <p>
              {view.status}: {view.guidance}
            </p>
            {view.status === "permitted" && (
              <dl>
                {view.fields?.map((field, index) => (
                  // The server permits duplicate labels; this is a static snapshot.
                  // biome-ignore lint/suspicious/noArrayIndexKey: fields have no unique IDs
                  <div key={index}>
                    <dt>{reviewText(field.label)}</dt>
                    <dd className="whitespace-pre-wrap">{reviewText(field.text)}</dd>
                  </div>
                ))}
              </dl>
            )}
            {view.status === "permitted" && (view.long_fields?.length ?? 0) > 0 && (
              <div className="space-y-2">
                <p className="text-xs text-muted-foreground">
                  Some values above are shown only in part. The decision covers them whole.
                </p>
                {view.long_fields?.map((field) =>
                  // Keys are policy text; "__proto__" must not read as loaded.
                  !Object.hasOwn(longText, field.key) ? (
                    <Button
                      key={field.key}
                      size="sm"
                      variant="outline"
                      disabled={busy || reading !== null}
                      onClick={() => void readLongField(field.key)}
                    >
                      Show all {field.characters.toLocaleString()} characters of{" "}
                      {reviewText(field.label)}
                    </Button>
                  ) : (
                    <div key={field.key} className="space-y-1">
                      <p className="font-medium">{reviewText(field.label)}, complete</p>
                      <pre className="max-h-96 overflow-auto whitespace-pre-wrap break-words rounded-lg bg-muted px-3 py-2 text-xs">
                        {longText[field.key]}
                      </pre>
                    </div>
                  ),
                )}
              </div>
            )}
            <p>
              Whole-round scope: all listed calls remain subject to their eligibility and execution
              checks.
            </p>
            <ul>
              {view.calls?.map((call) => (
                <li key={call.tool_call_id}>
                  {reviewText(call.tool_name)} ({reviewText(call.tool_call_id)}): {call.on_grant}
                </li>
              ))}
            </ul>
            {kind === "user_input" ? (
              <>
                <Input
                  aria-label="Answer"
                  value={answer}
                  onChange={(event) => setAnswer(event.target.value)}
                  disabled={blocked || view.status !== "permitted"}
                />
                <Button
                  disabled={blocked || view.status !== "permitted" || !answer.trim()}
                  onClick={() => void decide("answer")}
                >
                  Submit Answer
                </Button>
              </>
            ) : (
              <>
                <Input
                  aria-label="Decision reason"
                  value={reason}
                  onChange={(event) => setReason(event.target.value)}
                  disabled={blocked}
                />
                <Button
                  disabled={blocked || !view.tool_round_id || !view.tool_call_id}
                  onClick={() => void decide("deny")}
                >
                  Deny
                </Button>
                <Button
                  disabled={
                    blocked ||
                    view.status !== "permitted" ||
                    !view.tool_round_id ||
                    !view.tool_call_id
                  }
                  onClick={() => void decide("approve")}
                >
                  Approve
                </Button>
              </>
            )}
          </>
        )}
        {error && (
          <p role="alert" className="text-destructive">
            {error}
          </p>
        )}
      </CardContent>
    </Card>
  )
}
