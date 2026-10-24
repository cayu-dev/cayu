import { createMutationBrowserIO } from "./mutation-browser.ts"
import { createMutationId, type MutationTransportIO } from "./mutation-transport.ts"

export type MessageAcceptance = {
  phase: "accepted" | "rejected" | "unknown"
  queueId: string | null
  message: string | null
}

/** Enqueue acknowledges one durable message, not a terminal run. Retries reuse its exact body. */
export class SessionMessageAcceptance {
  private readonly io: Pick<MutationTransportIO, "attempt">
  private readonly request
  private uncertain = false
  private inFlight = false
  private accepted: MessageAcceptance | null = null

  constructor(
    sessionId: string,
    content: string,
    idempotencyKey: string,
    io: Pick<MutationTransportIO, "attempt"> = createMutationBrowserIO(),
  ) {
    this.io = io
    this.request = Object.freeze({
      path: `/sessions/${encodeURIComponent(sessionId)}/messages`,
      sessionId,
      mutationId: createMutationId(),
      lastEventId: null,
      body: Object.freeze({ idempotency_key: idempotencyKey, content, delivery_mode: "next_turn" }),
    })
  }

  async submit(signal: AbortSignal): Promise<MessageAcceptance> {
    if (this.accepted) return this.accepted
    if (this.inFlight) throw new Error("Message acceptance is already being observed.")
    this.inFlight = true
    let queueId: string | null = null
    try {
      const result = await this.io.attempt(
        this.request,
        {
          onAccepted: () => undefined,
          onEvent: (event) => {
            if (
              event.type === "session.message.queued" &&
              event.session_id === this.request.sessionId &&
              typeof event.payload.queue_id === "string" &&
              event.payload.queue_id.length > 0
            )
              queueId = event.payload.queue_id
          },
        },
        signal,
      )
      if (queueId !== null) {
        this.accepted = { phase: "accepted", queueId, message: null }
        return this.accepted
      }
      if (result.kind === "http_failed" && result.status < 500 && !this.uncertain) {
        return { phase: "rejected", queueId: null, message: result.error.message }
      }
    } catch {
      // An unobserved outcome is reported as unknown below.
    } finally {
      this.inFlight = false
    }
    this.uncertain = true
    return {
      phase: "unknown",
      queueId: null,
      message:
        "Message acceptance is unconfirmed. Retry this exact message to recover its receipt.",
    }
  }
}
