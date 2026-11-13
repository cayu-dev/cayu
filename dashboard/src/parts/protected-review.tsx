/**
 * Protected human review as a part: the review a person reads before deciding an approval
 * or a question when the Agent has a human review policy, and the decision it allows,
 * carrying the review's reference. It fills the Inbox host's `ProtectedReview` slot, and an
 * Agent's own pages render it for their own approval cards.
 *
 * It reads only through the `ReviewApi` it's given and keeps protected material in its own
 * state: no query keys, no shared caches, nothing persisted. The host settles the decision
 * (`onDecision`) through its own settle, so the reference reaches the server unchanged.
 */

import { Button, Card, CardContent, Input } from "@kit/ui"
import { useEffect, useRef, useState } from "react"
import type { ProtectedReviewDecision } from "./host"
import type { HumanReviewLongFieldView, HumanReviewReference, HumanReviewView } from "./types"

/** The two reads protected review needs: client.js's, the dashboard's `/api`, or an app's. */
export interface ReviewApi {
  /** The current review of the session's pending approval or question, for `purpose`. */
  inspect(sessionId: string, purpose: string, signal?: AbortSignal): Promise<HumanReviewView>
  /** One field the review at `reference` shows only in part, whole. */
  longField(
    sessionId: string,
    reference: HumanReviewReference,
    key: string,
    signal?: AbortSignal,
  ): Promise<HumanReviewLongFieldView>
}

const ROLE_FORBIDDEN = "Your role can read and decide, but this action requires full access."

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

const roleForbidden = (error: unknown) => {
  const { status, code } = (error ?? {}) as { status?: unknown; code?: unknown }
  return status === 403 && code === "role_forbidden"
}

export function ProtectedReviewPart({
  api,
  purpose,
  sessionId,
  kind,
  disabled,
  unavailableReason = null,
  onDecision,
}: {
  api: ReviewApi
  /**
   * The purpose reviews are requested for (`human_review_purpose` from `/api/app`). Null
   * when the policy names none: the part then says so and offers no decision.
   */
  purpose: string | null
  sessionId: string
  kind: "user_input" | "tool_approval"
  disabled: boolean
  unavailableReason?: string | null
  /** Settle with this decision; it carries the reviewed reference. */
  onDecision: (decision: ProtectedReviewDecision) => Promise<void>
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
      const field = await api.longField(view.session_id, view.reference, key, controller.signal)
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
    if (purpose === null) return
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
      const next = await api.inspect(sessionId, purpose, controller.signal)
      // Never decide with a review for another interaction or another purpose.
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
          roleForbidden(error)
            ? ROLE_FORBIDDEN
            : "Review access was denied or is unavailable. Refresh the review or contact the application owner.",
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
        await onDecision({ decision, answer, review_reference: view.reference })
      } else {
        if (!view.tool_round_id || !view.tool_call_id)
          throw new Error("Review decision identities are missing.")
        await onDecision({
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
        {purpose === null ? (
          <p>
            Deciding this needs protected review, and this Agent's review policy names no purpose to
            request it with. Decide it in the Developer dashboard.
          </p>
        ) : (
          <Button variant="outline" disabled={busy || disabled} onClick={() => void inspect()}>
            Refresh review
          </Button>
        )}
        {!view && purpose !== null && <p>Inspect the current protected review before deciding.</p>}
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
