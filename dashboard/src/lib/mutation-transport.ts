import type {
  ApiEventRecord,
  ListSessionEventsResponse,
  SessionStateResponse,
  SseErrorEnvelope,
  SseEventEnvelope,
} from "./generated/server-api"
import {
  MUTATION_OUT_OF_ORDER_RECORD_LIMIT,
  MutationEventAccumulator,
  type MutationEventAccumulatorSnapshot,
} from "./mutation-event-accumulator.ts"
import {
  assertMutationSseEvent,
  expectedMutationSseId,
  isMutationSseEventId,
  isMutationSseMarkerComponent,
  MutationStreamProtocolError,
} from "./mutation-stream-protocol.ts"

export const MUTATION_RECONNECT_INITIAL_DELAY_MS = 250
export const MUTATION_RECONNECT_BACKOFF_FACTOR = 2
export const MUTATION_RECONNECT_MAX_DELAY_MS = 10_000
export const MUTATION_RECONNECT_JITTER_RATIO = 0.2
export const MUTATION_RECONNECT_MAX_ATTEMPTS = 8
export const MUTATION_RECONNECT_MAX_ELAPSED_MS = 30_000
export const MUTATION_CONNECTION_TIMEOUT_MS = 15_000
export const MUTATION_CONFIRMATION_TIMEOUT_MS = 30_000
export const MUTATION_RECONCILIATION_REQUEST_TIMEOUT_MS = 15_000
export const MUTATION_RECONCILIATION_PAGE_LIMIT = 100
export const MUTATION_RECONCILIATION_MAX_PAGES = 10
export const MUTATION_FALLBACK_POLL_INTERVAL_MS = 2_000
export const MUTATION_FALLBACK_MAX_CONSECUTIVE_FAILURES = 8

const TERMINAL_PHASES = new Set<MutationTransportPhase>([
  "terminal",
  "request_failed",
  "runtime_failed",
  "transport_failed",
  "cancelled",
])
const TERMINAL_EVENT_TYPES = new Set(["session.completed", "session.failed", "session.interrupted"])
const MUTATION_ACCEPTANCE_EVENT_TYPE = "server.mutation.accepted"
const REPLAY_SAFE_ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._~-]{0,127}$/

export type MutationTransportPhase =
  | "connecting"
  | "streaming"
  | "reconnecting"
  | "polling_fallback"
  | "paused"
  | "terminal"
  | "request_failed"
  | "runtime_failed"
  | "transport_failed"
  | "cancelled"

export type MutationTransportFailureKind =
  | "http"
  | "runtime"
  | "observer"
  | "protocol"
  | "network"
  | "reconciliation"

export type MutationTransportFailure = {
  kind: MutationTransportFailureKind
  message: string
  error: unknown
  envelope: SseErrorEnvelope | null
}

export type MutationReplayBaseline = {
  marker: string
  sequence: number
}

export type MutationTransportRequest = {
  path: string
  body: Record<string, unknown>
  sessionId: string
  mutationId: string
  baseline: MutationReplayBaseline
}

export type MutationTransportSnapshot = MutationEventAccumulatorSnapshot & {
  phase: MutationTransportPhase
  sessionId: string
  latestSseId: string | null
  replayMarker: string
  reconnectAttempt: number
  failure: MutationTransportFailure | null
}

export type MutationAttemptCallbacks = {
  onAccepted: () => void
  onEvent: (event: SseEventEnvelope, sseId: string) => void
}

export type MutationAttemptRequest = {
  path: string
  body: Record<string, unknown>
  sessionId: string
  mutationId: string
  lastEventId: string | null
}

export type MutationAttemptResult =
  | { kind: "closed"; accepted: boolean }
  | { kind: "aborted"; accepted: boolean }
  | { kind: "http_failed"; accepted: false; status: number; error: Error }
  | { kind: "runtime_failed"; accepted: true; error: SseErrorEnvelope }
  | { kind: "observer_failed"; accepted: true; error: SseErrorEnvelope }
  | {
      kind: "protocol_failed"
      accepted: boolean
      error: MutationStreamProtocolError
    }
  | {
      kind: "network_failed"
      accepted: boolean
      error: unknown
    }

export type MutationTransportPolicy = {
  reconnectInitialDelayMs: number
  reconnectBackoffFactor: number
  reconnectMaxDelayMs: number
  reconnectJitterRatio: number
  reconnectMaxAttempts: number
  reconnectMaxElapsedMs: number
  connectionTimeoutMs: number
  confirmationTimeoutMs: number
  reconciliationRequestTimeoutMs: number
  reconciliationPageLimit: number
  reconciliationMaxPages: number
  fallbackPollIntervalMs: number
  fallbackMaxConsecutiveFailures: number
}

export type MutationVisibilitySource = {
  isVisible: () => boolean
  subscribe: (listener: () => void) => () => void
}

export type MutationTransportIO = {
  attempt: (
    request: MutationAttemptRequest,
    callbacks: MutationAttemptCallbacks,
    signal: AbortSignal,
  ) => Promise<MutationAttemptResult>
  readEvents: (
    sessionId: string,
    afterSequence: number,
    limit: number,
    signal: AbortSignal,
  ) => Promise<ListSessionEventsResponse>
  readState: (sessionId: string, signal: AbortSignal) => Promise<SessionStateResponse>
  delay: (milliseconds: number, signal: AbortSignal) => Promise<void>
  watchdog: (milliseconds: number, signal: AbortSignal) => Promise<void>
  now: () => number
  random: () => number
}

export type MutationTransportOptions = {
  request: MutationTransportRequest
  onChange?: (snapshot: MutationTransportSnapshot) => void
  io: MutationTransportIO
  policy?: Partial<MutationTransportPolicy>
  visibility?: MutationVisibilitySource
}

type MutationReconciliationOutcome = {
  terminal: boolean
  caughtUp: boolean
  terminalBoundaryMissing: boolean
}

type DurableMutationAcceptance = {
  acceptedEventId: string
  acceptedEventType: string
  mutationKind: string
}

const DEFAULT_POLICY: MutationTransportPolicy = {
  reconnectInitialDelayMs: MUTATION_RECONNECT_INITIAL_DELAY_MS,
  reconnectBackoffFactor: MUTATION_RECONNECT_BACKOFF_FACTOR,
  reconnectMaxDelayMs: MUTATION_RECONNECT_MAX_DELAY_MS,
  reconnectJitterRatio: MUTATION_RECONNECT_JITTER_RATIO,
  reconnectMaxAttempts: MUTATION_RECONNECT_MAX_ATTEMPTS,
  reconnectMaxElapsedMs: MUTATION_RECONNECT_MAX_ELAPSED_MS,
  connectionTimeoutMs: MUTATION_CONNECTION_TIMEOUT_MS,
  confirmationTimeoutMs: MUTATION_CONFIRMATION_TIMEOUT_MS,
  reconciliationRequestTimeoutMs: MUTATION_RECONCILIATION_REQUEST_TIMEOUT_MS,
  reconciliationPageLimit: MUTATION_RECONCILIATION_PAGE_LIMIT,
  reconciliationMaxPages: MUTATION_RECONCILIATION_MAX_PAGES,
  fallbackPollIntervalMs: MUTATION_FALLBACK_POLL_INTERVAL_MS,
  fallbackMaxConsecutiveFailures: MUTATION_FALLBACK_MAX_CONSECUTIVE_FAILURES,
}

const ALWAYS_VISIBLE: MutationVisibilitySource = {
  isVisible: () => true,
  subscribe: () => () => undefined,
}

function isAbortError(error: unknown): boolean {
  return error instanceof Error && error.name === "AbortError"
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

function mutationAbortError(): DOMException {
  return new DOMException("The mutation transport operation was aborted.", "AbortError")
}

class MutationTransportTimeoutError extends Error {
  constructor(message: string) {
    super(message)
    this.name = "MutationTransportTimeoutError"
  }
}

function validatePolicy(policy: MutationTransportPolicy): void {
  const positiveIntegers: Array<[string, number]> = [
    ["reconnectInitialDelayMs", policy.reconnectInitialDelayMs],
    ["reconnectMaxDelayMs", policy.reconnectMaxDelayMs],
    ["reconnectMaxAttempts", policy.reconnectMaxAttempts],
    ["reconnectMaxElapsedMs", policy.reconnectMaxElapsedMs],
    ["connectionTimeoutMs", policy.connectionTimeoutMs],
    ["confirmationTimeoutMs", policy.confirmationTimeoutMs],
    ["reconciliationRequestTimeoutMs", policy.reconciliationRequestTimeoutMs],
    ["reconciliationPageLimit", policy.reconciliationPageLimit],
    ["reconciliationMaxPages", policy.reconciliationMaxPages],
    ["fallbackPollIntervalMs", policy.fallbackPollIntervalMs],
    ["fallbackMaxConsecutiveFailures", policy.fallbackMaxConsecutiveFailures],
  ]
  for (const [name, value] of positiveIntegers) {
    if (!Number.isSafeInteger(value) || value <= 0) {
      throw new RangeError(`${name} must be a positive safe integer.`)
    }
  }
  if (!Number.isFinite(policy.reconnectBackoffFactor) || policy.reconnectBackoffFactor < 1) {
    throw new RangeError("reconnectBackoffFactor must be at least 1.")
  }
  if (
    !Number.isFinite(policy.reconnectJitterRatio) ||
    policy.reconnectJitterRatio < 0 ||
    policy.reconnectJitterRatio > 1
  ) {
    throw new RangeError("reconnectJitterRatio must be between 0 and 1.")
  }
  if (policy.reconciliationPageLimit > MUTATION_OUT_OF_ORDER_RECORD_LIMIT) {
    throw new RangeError(
      `reconciliationPageLimit cannot exceed ${MUTATION_OUT_OF_ORDER_RECORD_LIMIT}.`,
    )
  }
}

function validateRequest(request: MutationTransportRequest): void {
  if (
    !request.path.startsWith("/") ||
    request.path.startsWith("//") ||
    request.path !== request.path.trim() ||
    [...request.path].some((character) => {
      const codePoint = character.codePointAt(0)
      return codePoint !== undefined && (codePoint < 0x20 || codePoint === 0x7f)
    })
  ) {
    throw new TypeError("Mutation transport path must be a clean root-relative path.")
  }
  if (!isMutationSseMarkerComponent(request.sessionId)) {
    throw new TypeError("Mutation transport sessionId is not a valid replay component.")
  }
  if (!REPLAY_SAFE_ID_PATTERN.test(request.mutationId)) {
    throw new TypeError("Mutation transport mutationId is not a valid replay component.")
  }
  const expectedPrefix = `${request.sessionId}:`
  if (!request.baseline.marker.startsWith(expectedPrefix)) {
    throw new TypeError("Mutation replay baseline does not belong to the request session.")
  }
  const eventId = request.baseline.marker.slice(expectedPrefix.length)
  if (eventId.length > 0 && !isMutationSseEventId(eventId)) {
    throw new TypeError("Mutation replay baseline contains an invalid event id.")
  }
  if (!Number.isSafeInteger(request.baseline.sequence) || request.baseline.sequence < 0) {
    throw new RangeError("Mutation replay baseline sequence must be non-negative.")
  }
  if ((eventId.length === 0) !== (request.baseline.sequence === 0)) {
    throw new TypeError(
      "Mutation replay start markers require sequence zero and durable event markers require a positive sequence.",
    )
  }
}

function attemptFailure(
  kind: MutationTransportFailureKind,
  error: unknown,
  envelope: SseErrorEnvelope | null = null,
): MutationTransportFailure {
  return { kind, message: envelope?.error ?? errorMessage(error), error, envelope }
}

function isDefinitiveMutationHttpRejection(status: number): boolean {
  // Most 4xx responses prove that the command was rejected before execution.
  // Request timeout and the common proxy-specific client-closed status do not:
  // an intermediary may emit either after forwarding the mutation upstream.
  return status >= 400 && status < 500 && status !== 408 && status !== 499
}

function terminalEventTypeForStatus(status: string): string | null {
  if (status === "completed") return "session.completed"
  if (status === "failed") return "session.failed"
  if (status === "interrupted") return "session.interrupted"
  return null
}

export function mutationReconnectDelay(
  attempt: number,
  policy: MutationTransportPolicy,
  random: number,
): number {
  if (!Number.isSafeInteger(attempt) || attempt < 1) {
    throw new RangeError("Reconnect attempt must be a positive safe integer.")
  }
  const exponential =
    policy.reconnectInitialDelayMs * policy.reconnectBackoffFactor ** (attempt - 1)
  const base = Math.min(policy.reconnectMaxDelayMs, exponential)
  if (!Number.isFinite(random)) throw new RangeError("Reconnect random value must be finite.")
  const boundedRandom = Math.min(1, Math.max(0, random))
  const jitter = 1 + (boundedRandom * 2 - 1) * policy.reconnectJitterRatio
  return Math.min(policy.reconnectMaxDelayMs, Math.max(0, Math.round(base * jitter)))
}

export function createRunSessionId(randomUuid: () => string = () => crypto.randomUUID()): string {
  const sessionId = `session-${randomUuid()}`
  if (!REPLAY_SAFE_ID_PATTERN.test(sessionId)) {
    throw new TypeError("Generated run session id does not satisfy the server contract.")
  }
  return sessionId
}

export function createMutationId(randomUuid: () => string = () => crypto.randomUUID()): string {
  const mutationId = `mutation-${randomUuid()}`
  if (!REPLAY_SAFE_ID_PATTERN.test(mutationId)) {
    throw new TypeError("Generated mutation id does not satisfy the server contract.")
  }
  return mutationId
}

export function newRunReplayBaseline(sessionId: string): MutationReplayBaseline {
  return { marker: `${sessionId}:`, sequence: 0 }
}

export class MutationTransportController {
  private readonly request: MutationTransportRequest
  private readonly accumulator = new MutationEventAccumulator()
  private readonly onChange: (snapshot: MutationTransportSnapshot) => void
  private readonly io: MutationTransportIO
  private readonly policy: MutationTransportPolicy
  private readonly visibility: MutationVisibilitySource
  private phase: MutationTransportPhase = "connecting"
  private latestSseId: string | null = null
  private replayMarker: string
  private pendingSseBoundaryEventId: string | null = null
  private reconnectAttempt = 0
  private recoveryStartedAt: number | null = null
  private visibilityHiddenAt: number | null = null
  private failure: MutationTransportFailure | null = null
  private operationAccepted = false
  private operationEpochConfirmed = false
  private operationBoundaryEventId: string | null = null
  private operationBoundaryEventType: string | null = null
  private operationTerminalFloorSequence: number | null = null
  private readonly operationTerminalEventTypes = new Set<string>()
  private replayOnly = false
  private recovering = false
  private started = false
  private disposed = false
  private activeAbort: AbortController | null = null
  private attemptGeneration = 0
  private activeAttemptGeneration: number | null = null
  private unsubscribeVisibility: (() => void) | null = null
  private visibilityWaiters = new Set<() => void>()
  private startPromise: Promise<MutationTransportSnapshot> | null = null

  constructor(options: MutationTransportOptions) {
    validateRequest(options.request)
    this.request = {
      ...options.request,
      body: structuredClone(options.request.body),
      baseline: { ...options.request.baseline },
    }
    this.onChange = options.onChange ?? (() => undefined)
    this.io = options.io
    this.policy = { ...DEFAULT_POLICY, ...options.policy }
    this.visibility = options.visibility ?? ALWAYS_VISIBLE
    validatePolicy(this.policy)
    this.replayMarker = this.request.baseline.marker
    this.accumulator.advanceDurableSequence(this.request.baseline.sequence)
  }

  snapshot(): MutationTransportSnapshot {
    return {
      phase: this.phase,
      sessionId: this.request.sessionId,
      latestSseId: this.latestSseId,
      replayMarker: this.replayMarker,
      reconnectAttempt: this.reconnectAttempt,
      failure: this.failure,
      ...this.accumulator.snapshot(),
    }
  }

  start(signal?: AbortSignal): Promise<MutationTransportSnapshot> {
    if (this.startPromise !== null) return this.startPromise
    if (this.started) throw new Error("Mutation transport cannot be restarted.")
    this.started = true
    if (signal?.aborted) {
      this.disposed = true
      this.phase = "cancelled"
      this.emit()
      this.startPromise = Promise.resolve(this.snapshot())
      return this.startPromise
    }
    const abort = () => this.dispose()
    signal?.addEventListener("abort", abort, { once: true })
    this.unsubscribeVisibility = this.visibility.subscribe(() => this.visibilityChanged())
    this.startPromise = this.run().finally(() => {
      signal?.removeEventListener("abort", abort)
      this.unsubscribeVisibility?.()
      this.unsubscribeVisibility = null
    })
    return this.startPromise
  }

  dispose(): void {
    if (this.disposed || TERMINAL_PHASES.has(this.phase)) return
    this.disposed = true
    this.activeAttemptGeneration = null
    this.activeAbort?.abort()
    this.releaseVisibilityWaiters()
    this.transition("cancelled")
  }

  private async run(): Promise<MutationTransportSnapshot> {
    this.emit()
    while (!this.disposed) {
      if (!(await this.waitUntilVisible())) return this.snapshot()

      const replayAttempt = this.operationAccepted || this.replayOnly
      this.transition(replayAttempt ? "reconnecting" : "connecting")
      const lastEventId = replayAttempt ? this.replayMarker : null
      const wasRecovering = replayAttempt
      const attemptGeneration = this.attemptGeneration + 1
      this.attemptGeneration = attemptGeneration
      this.activeAttemptGeneration = attemptGeneration
      let acceptedThisAttempt = false
      let terminalBoundaryObserved = false
      let result: MutationAttemptResult
      try {
        result = await this.runAttemptWithWatchdog(
          {
            path: this.request.path,
            body: this.request.body,
            sessionId: this.request.sessionId,
            mutationId: this.request.mutationId,
            lastEventId,
          },
          {
            onAccepted: () => {
              if (this.disposed || this.activeAttemptGeneration !== attemptGeneration) {
                return
              }
              acceptedThisAttempt = true
              if (!replayAttempt) {
                this.acceptOperation()
              }
              if (
                this.operationAccepted &&
                (!replayAttempt || this.operationEpochConfirmed) &&
                this.visibility.isVisible()
              ) {
                // Opening a replacement observer resolves the active connection
                // failure. Keep its bounded retry episode intact until novel
                // durable progress arrives so connect/drop loops cannot reset the
                // retry budget indefinitely.
                this.failure = null
                this.transition("streaming")
              }
            },
            onEvent: (event, sseId) => {
              if (this.disposed || this.activeAttemptGeneration !== attemptGeneration) {
                return
              }
              assertMutationSseEvent(event)
              if (
                event.session_id !== this.request.sessionId ||
                sseId !== expectedMutationSseId(event)
              ) {
                throw new MutationStreamProtocolError(
                  "Mutation attempt returned an event with an invalid durable identity.",
                )
              }
              acceptedThisAttempt = true
              const acceptance = this.durableAcceptance(event)
              if (!replayAttempt && !this.operationEpochConfirmed) {
                this.acceptOperation()
                this.confirmOperationEpoch(
                  {
                    acceptedEventId: event.id,
                    acceptedEventType: event.type,
                    mutationKind: "initial_stream",
                  },
                  null,
                )
              }
              if (acceptance !== null) {
                if (!this.operationAccepted) this.acceptOperation()
                this.confirmOperationEpoch(acceptance, null)
              }
              const eventAdded = this.accumulator.addSseEvent(event)
              if (eventAdded) {
                this.latestSseId = sseId
                this.replayMarker = sseId
                this.pendingSseBoundaryEventId = event.id
              }

              // Session-wide replay may contain unrelated concurrent activity.
              // It becomes live mutation progress only after the exact durable
              // mutation marker has been observed.
              if (this.operationEpochConfirmed) {
                if (eventAdded) {
                  if (TERMINAL_EVENT_TYPES.has(event.type)) {
                    this.operationTerminalEventTypes.add(event.type)
                  }
                  this.failure = null
                  if (wasRecovering || this.recovering) {
                    this.reconnectAttempt = 0
                    this.recoveryStartedAt = null
                    this.recovering = false
                  }
                  if (this.visibility.isVisible()) this.transition("streaming")
                  if (
                    TERMINAL_EVENT_TYPES.has(event.type) ||
                    (acceptance !== null && TERMINAL_EVENT_TYPES.has(acceptance.acceptedEventType))
                  ) {
                    // The browser has the durable terminal boundary. Stop waiting
                    // for a server/network close and reconcile state immediately.
                    terminalBoundaryObserved = true
                    this.activeAbort?.abort()
                  }
                }
              }
            },
          },
          () => acceptedThisAttempt && !this.operationEpochConfirmed,
        )
      } catch (error) {
        const accepted = acceptedThisAttempt || this.operationAccepted
        result = isAbortError(error)
          ? { kind: "aborted", accepted }
          : error instanceof MutationStreamProtocolError
            ? { kind: "protocol_failed", accepted, error }
            : { kind: "network_failed", accepted, error }
      } finally {
        if (this.activeAttemptGeneration === attemptGeneration) {
          this.activeAttemptGeneration = null
        }
      }
      if (this.disposed) return this.snapshot()
      result = this.validateAttemptResult(result)

      if (
        result.kind === "http_failed" &&
        !replayAttempt &&
        isDefinitiveMutationHttpRejection(result.status)
      ) {
        return this.fail("request_failed", attemptFailure("http", result.error))
      }
      if (result.kind === "runtime_failed" && !terminalBoundaryObserved) {
        return this.fail("runtime_failed", attemptFailure("runtime", result.error, result.error))
      }
      if (!replayAttempt) {
        if (result.accepted) {
          this.acceptOperation()
        } else {
          // The initial mutation POST is never issued a second time. Every
          // subsequent request carries the durable baseline and can only replay.
          this.replayOnly = true
          if (this.recoveryStartedAt === null) this.recoveryStartedAt = this.io.now()
        }
      }
      if (!this.visibility.isVisible()) {
        if (terminalBoundaryObserved) return this.pollFallback()
        continue
      }

      let observerRequiresFallback = false
      if (!terminalBoundaryObserved) {
        if (result.kind === "http_failed") {
          this.failure = attemptFailure("http", result.error)
        } else if (result.kind === "observer_failed") {
          this.failure = attemptFailure("observer", result.error, result.error)
          observerRequiresFallback = !result.error.retryable
        } else if (result.kind === "protocol_failed") {
          this.failure = attemptFailure("protocol", result.error)
        } else if (result.kind === "network_failed") {
          this.failure = attemptFailure("network", result.error)
        } else if (result.kind === "closed") {
          this.failure = attemptFailure(
            "network",
            new Error("Mutation stream closed before a durable terminal boundary."),
          )
        } else if (result.kind === "aborted") {
          this.failure = attemptFailure(
            "network",
            new Error("Mutation stream was aborted before a durable terminal boundary."),
          )
        }
      }

      if (await this.reconcileAfterStream()) return this.snapshot()
      if (terminalBoundaryObserved) return this.pollFallback()
      if (observerRequiresFallback) return this.pollFallback()
      if (!(await this.scheduleReconnect())) return this.pollFallback()
    }
    return this.snapshot()
  }

  private async scheduleReconnect(): Promise<boolean> {
    if (this.recoveryStartedAt === null) this.recoveryStartedAt = this.io.now()
    const elapsed = Math.max(0, this.io.now() - this.recoveryStartedAt)
    if (
      this.reconnectAttempt >= this.policy.reconnectMaxAttempts ||
      elapsed >= this.policy.reconnectMaxElapsedMs
    ) {
      return false
    }
    const nextAttempt = this.reconnectAttempt + 1
    const delay = mutationReconnectDelay(nextAttempt, this.policy, this.io.random())
    if (elapsed + delay > this.policy.reconnectMaxElapsedMs) return false
    this.recovering = true
    if (this.visibility.isVisible()) this.transition("reconnecting")
    const slept = await this.sleepVisible(delay)
    if (!slept) return !this.disposed
    this.reconnectAttempt = nextAttempt
    return true
  }

  private async runAttemptWithWatchdog(
    request: MutationAttemptRequest,
    callbacks: MutationAttemptCallbacks,
    needsDurableConfirmation: () => boolean,
  ): Promise<MutationAttemptResult> {
    const attemptController = new AbortController()
    this.activeAbort = attemptController
    if (this.disposed || !this.visibility.isVisible()) attemptController.abort()

    let watchdogController: AbortController | null = null
    let watchdogGeneration = 0
    let watchdogError: unknown = null
    let rejectWatchdog: (error: unknown) => void = () => undefined
    const watchdogFailure = new Promise<never>((_resolve, reject) => {
      rejectWatchdog = reject
    })

    const disarmWatchdog = () => {
      watchdogGeneration += 1
      watchdogController?.abort()
      watchdogController = null
    }
    const armWatchdog = (milliseconds: number, message: string) => {
      disarmWatchdog()
      const generation = watchdogGeneration
      const controller = new AbortController()
      watchdogController = controller
      void this.io
        .watchdog(milliseconds, controller.signal)
        .then(() => {
          if (controller.signal.aborted || generation !== watchdogGeneration) return
          watchdogError = new MutationTransportTimeoutError(message)
          rejectWatchdog(watchdogError)
          attemptController.abort()
        })
        .catch((error: unknown) => {
          if (controller.signal.aborted || generation !== watchdogGeneration) return
          watchdogError = error
          rejectWatchdog(error)
          attemptController.abort()
        })
    }

    let rejectAbort: (error: unknown) => void = () => undefined
    const aborted = new Promise<never>((_resolve, reject) => {
      rejectAbort = reject
    })
    const abortListener = () => rejectAbort(mutationAbortError())
    attemptController.signal.addEventListener("abort", abortListener, { once: true })
    if (attemptController.signal.aborted) abortListener()

    const remainingRecoveryMs = () => {
      if (this.recoveryStartedAt === null) return this.policy.reconnectMaxElapsedMs
      return Math.max(
        1,
        this.policy.reconnectMaxElapsedMs - Math.max(0, this.io.now() - this.recoveryStartedAt),
      )
    }
    armWatchdog(
      Math.min(this.policy.connectionTimeoutMs, remainingRecoveryMs()),
      "Mutation stream connection timed out before the server accepted the request.",
    )

    const attempt = Promise.resolve().then(() =>
      this.io.attempt(
        request,
        {
          onAccepted: () => {
            callbacks.onAccepted()
            if (needsDurableConfirmation()) {
              armWatchdog(
                Math.min(this.policy.confirmationTimeoutMs, remainingRecoveryMs()),
                "Mutation stream did not confirm the submitted mutation before its deadline.",
              )
            } else {
              disarmWatchdog()
            }
          },
          onEvent: (event, sseId) => {
            callbacks.onEvent(event, sseId)
            if (!needsDurableConfirmation()) disarmWatchdog()
          },
        },
        attemptController.signal,
      ),
    )

    try {
      return await Promise.race([attempt, watchdogFailure, aborted])
    } catch (error) {
      if (isAbortError(error) && watchdogError !== null) throw watchdogError
      throw error
    } finally {
      disarmWatchdog()
      attemptController.signal.removeEventListener("abort", abortListener)
      if (this.activeAbort === attemptController) this.activeAbort = null
    }
  }

  private async reconcileAfterStream(): Promise<boolean> {
    try {
      return (await this.reconcile()).terminal
    } catch (error) {
      if (this.disposed || isAbortError(error)) return false
      this.failure = attemptFailure("reconciliation", error)
      this.emit()
      return false
    }
  }

  private async reconcile(): Promise<MutationReconciliationOutcome> {
    let cursor = this.accumulator.snapshot().latestSequence
    let caughtUp = true
    for (let pageNumber = 0; pageNumber < this.policy.reconciliationMaxPages; pageNumber += 1) {
      const page = await this.runActive(
        (signal) =>
          this.io.readEvents(
            this.request.sessionId,
            cursor,
            this.policy.reconciliationPageLimit,
            signal,
          ),
        {
          milliseconds: this.policy.reconciliationRequestTimeoutMs,
          message: "Mutation event reconciliation timed out.",
        },
      )
      const previousCursor = cursor
      this.validateReconciliationPage(page, cursor)
      this.observeDurableOperationEpoch(page.events)
      const reconciliation = this.accumulator.reconcile(page.events, this.pendingSseBoundaryEventId)
      if (reconciliation.observedBoundaryFound) {
        this.pendingSseBoundaryEventId = null
      }
      cursor = Math.max(
        cursor,
        page.scan_through_sequence ?? 0,
        ...page.events.map((event) => event.sequence),
      )
      this.accumulator.advanceDurableSequence(cursor)
      if (this.pendingSseBoundaryEventId === null && page.events.length > 0) {
        const latest = page.events.reduce((left, right) =>
          right.sequence > left.sequence ? right : left,
        )
        this.replayMarker = `${this.request.sessionId}:${latest.id}`
      }
      this.emit()
      caughtUp = !page.has_more
      if (caughtUp) break
      if (cursor <= previousCursor) {
        throw new MutationStreamProtocolError(
          "Event reconciliation did not advance its durable cursor.",
        )
      }
    }

    const state = await this.runActive(
      (signal) => this.io.readState(this.request.sessionId, signal),
      {
        milliseconds: this.policy.reconciliationRequestTimeoutMs,
        message: "Mutation state reconciliation timed out.",
      },
    )
    if (state.session_id !== this.request.sessionId) {
      throw new MutationStreamProtocolError("State reconciliation returned a different session.")
    }
    if (
      state.status !== "pending" &&
      state.status !== "running" &&
      state.status !== "interrupting" &&
      state.status !== "completed" &&
      state.status !== "failed" &&
      state.status !== "interrupted"
    ) {
      throw new MutationStreamProtocolError("State reconciliation returned an invalid status.")
    }
    const terminalEventType = terminalEventTypeForStatus(state.status)
    if (
      this.operationEpochConfirmed &&
      terminalEventType !== null &&
      this.operationTerminalEventTypes.has(terminalEventType)
    ) {
      this.failure = null
      this.transition("terminal")
      return { terminal: true, caughtUp, terminalBoundaryMissing: false }
    }
    const terminalBoundaryMissing =
      state.status === "completed" || state.status === "failed" || state.status === "interrupted"
    return { terminal: false, caughtUp, terminalBoundaryMissing }
  }

  private async pollFallback(): Promise<MutationTransportSnapshot> {
    let consecutiveFailures = 0
    this.reconnectAttempt = Math.min(this.reconnectAttempt, this.policy.reconnectMaxAttempts)
    while (!this.disposed) {
      if (!(await this.waitUntilVisible())) return this.snapshot()
      this.transition("polling_fallback")
      let shouldDelay = true
      try {
        const reconciliation = await this.reconcile()
        if (reconciliation.terminal) return this.snapshot()
        shouldDelay = reconciliation.caughtUp
        let incomplete: MutationTransportFailure | null = null
        if (reconciliation.caughtUp && !this.operationAccepted) {
          incomplete = attemptFailure(
            "network",
            new Error("Mutation acceptance was not observed at its durable replay boundary."),
          )
        } else if (reconciliation.caughtUp && reconciliation.terminalBoundaryMissing) {
          incomplete = attemptFailure(
            "protocol",
            new MutationStreamProtocolError(
              "Terminal session state is missing its post-baseline durable terminal event.",
            ),
          )
        }
        if (incomplete === null) {
          consecutiveFailures = 0
          if (this.failure !== null) {
            this.failure = null
            this.emit()
          }
        } else {
          consecutiveFailures += 1
          this.failure = incomplete
          this.emit()
          if (consecutiveFailures >= this.policy.fallbackMaxConsecutiveFailures) {
            return this.fail("transport_failed", incomplete)
          }
        }
      } catch (error) {
        if (this.disposed) return this.snapshot()
        if (isAbortError(error) && !this.visibility.isVisible()) continue
        consecutiveFailures += 1
        this.failure = attemptFailure("reconciliation", error)
        this.emit()
        if (consecutiveFailures >= this.policy.fallbackMaxConsecutiveFailures) {
          return this.fail("transport_failed", this.failure)
        }
      }
      if (shouldDelay) await this.sleepVisible(this.policy.fallbackPollIntervalMs)
    }
    return this.snapshot()
  }

  private async sleepVisible(milliseconds: number): Promise<boolean> {
    if (!this.visibility.isVisible()) return false
    try {
      await this.runActive((signal) => this.io.delay(milliseconds, signal))
      return true
    } catch (error) {
      if (this.disposed || isAbortError(error)) return false
      throw error
    }
  }

  private async waitUntilVisible(): Promise<boolean> {
    while (!this.disposed && !this.visibility.isVisible()) {
      this.transition("paused")
      await new Promise<void>((resolve) => this.visibilityWaiters.add(resolve))
    }
    return !this.disposed
  }

  private async runActive<T>(
    operation: (signal: AbortSignal) => Promise<T>,
    timeout?: { milliseconds: number; message: string },
  ): Promise<T> {
    const controller = new AbortController()
    this.activeAbort = controller
    if (this.disposed || !this.visibility.isVisible()) controller.abort()

    let timeoutController: AbortController | null = null
    let timeoutError: unknown = null
    let rejectTimeout: (error: unknown) => void = () => undefined
    const timeoutFailure = new Promise<never>((_resolve, reject) => {
      rejectTimeout = reject
    })
    if (timeout !== undefined) {
      timeoutController = new AbortController()
      const activeTimeoutController = timeoutController
      void this.io
        .watchdog(timeout.milliseconds, activeTimeoutController.signal)
        .then(() => {
          if (activeTimeoutController.signal.aborted) return
          timeoutError = new MutationTransportTimeoutError(timeout.message)
          rejectTimeout(timeoutError)
          controller.abort()
        })
        .catch((error: unknown) => {
          if (activeTimeoutController.signal.aborted) return
          timeoutError = error
          rejectTimeout(error)
          controller.abort()
        })
    }

    let rejectAbort: (error: unknown) => void = () => undefined
    const aborted = new Promise<never>((_resolve, reject) => {
      rejectAbort = reject
    })
    const abortListener = () => rejectAbort(mutationAbortError())
    controller.signal.addEventListener("abort", abortListener, { once: true })
    if (controller.signal.aborted) abortListener()

    const activeOperation = Promise.resolve().then(() => operation(controller.signal))
    try {
      const result = await Promise.race([activeOperation, timeoutFailure, aborted])
      if (controller.signal.aborted) {
        if (timeoutError !== null) throw timeoutError
        throw mutationAbortError()
      }
      return result
    } catch (error) {
      if (isAbortError(error) && timeoutError !== null) throw timeoutError
      throw error
    } finally {
      timeoutController?.abort()
      controller.signal.removeEventListener("abort", abortListener)
      if (this.activeAbort === controller) this.activeAbort = null
    }
  }

  private visibilityChanged(): void {
    if (this.disposed || TERMINAL_PHASES.has(this.phase)) return
    if (!this.visibility.isVisible()) {
      if (this.visibilityHiddenAt === null) this.visibilityHiddenAt = this.io.now()
      this.activeAbort?.abort()
      this.transition("paused")
      return
    }
    if (this.visibilityHiddenAt !== null) {
      if (this.recoveryStartedAt !== null) {
        const pausedFrom = Math.max(this.visibilityHiddenAt, this.recoveryStartedAt)
        this.recoveryStartedAt += Math.max(0, this.io.now() - pausedFrom)
      }
      this.visibilityHiddenAt = null
    }
    this.releaseVisibilityWaiters()
  }

  private validateReconciliationPage(page: ListSessionEventsResponse, afterSequence: number): void {
    if (page.session_id !== this.request.sessionId) {
      throw new MutationStreamProtocolError("Event reconciliation returned a different session.")
    }
    if (page.order_by !== "sequence_asc") {
      throw new MutationStreamProtocolError("Event reconciliation returned an invalid order.")
    }
    if (typeof page.has_more !== "boolean") {
      throw new MutationStreamProtocolError(
        "Event reconciliation returned an invalid continuation flag.",
      )
    }
    if (page.events.length > this.policy.reconciliationPageLimit) {
      throw new MutationStreamProtocolError("Event reconciliation exceeded its requested limit.")
    }
    if (
      page.scan_through_sequence !== null &&
      (!Number.isSafeInteger(page.scan_through_sequence) || page.scan_through_sequence < 0)
    ) {
      throw new MutationStreamProtocolError("Event reconciliation returned an invalid scan cursor.")
    }
    for (const event of page.events) {
      assertMutationSseEvent(event)
      if (event.session_id !== this.request.sessionId) {
        throw new MutationStreamProtocolError(
          "Event reconciliation contained a record from a different session.",
        )
      }
      if (!Number.isSafeInteger(event.sequence) || event.sequence <= afterSequence) {
        throw new MutationStreamProtocolError(
          "Event reconciliation returned a non-advancing event sequence.",
        )
      }
    }
    const maximumEventSequence = Math.max(
      afterSequence,
      ...page.events.map((event) => event.sequence),
    )
    if (page.scan_through_sequence !== null && page.scan_through_sequence < maximumEventSequence) {
      throw new MutationStreamProtocolError(
        "Event reconciliation scan cursor precedes a returned event.",
      )
    }
  }

  private releaseVisibilityWaiters(): void {
    const waiters = [...this.visibilityWaiters]
    this.visibilityWaiters.clear()
    for (const resolve of waiters) resolve()
  }

  private durableAcceptance(event: SseEventEnvelope): DurableMutationAcceptance | null {
    if (
      event.type !== MUTATION_ACCEPTANCE_EVENT_TYPE ||
      event.payload.mutation_id !== this.request.mutationId
    ) {
      return null
    }
    const acceptedEventId = event.payload.accepted_event_id
    const acceptedEventType = event.payload.accepted_event_type
    const mutationKind = event.payload.mutation_kind
    if (
      !isMutationSseEventId(acceptedEventId) ||
      typeof acceptedEventType !== "string" ||
      acceptedEventType.length === 0 ||
      typeof mutationKind !== "string" ||
      mutationKind.length === 0
    ) {
      throw new MutationStreamProtocolError(
        "Mutation acceptance event contains malformed durable identity metadata.",
      )
    }
    return {
      acceptedEventId,
      acceptedEventType,
      mutationKind,
    }
  }

  private confirmOperationEpoch(
    acceptance: DurableMutationAcceptance,
    terminalFloorSequence: number | null,
  ): void {
    if (this.operationEpochConfirmed) {
      if (
        this.operationBoundaryEventId !== acceptance.acceptedEventId ||
        this.operationBoundaryEventType !== acceptance.acceptedEventType
      ) {
        throw new MutationStreamProtocolError(
          "Mutation acceptance event conflicts with the confirmed operation boundary.",
        )
      }
    } else {
      this.operationEpochConfirmed = true
      this.operationBoundaryEventId = acceptance.acceptedEventId
      this.operationBoundaryEventType = acceptance.acceptedEventType
    }
    if (terminalFloorSequence !== null) {
      this.operationTerminalFloorSequence =
        this.operationTerminalFloorSequence === null
          ? terminalFloorSequence
          : Math.min(this.operationTerminalFloorSequence, terminalFloorSequence)
    }
    if (TERMINAL_EVENT_TYPES.has(acceptance.acceptedEventType)) {
      this.operationTerminalEventTypes.add(acceptance.acceptedEventType)
    }
  }

  private observeDurableOperationEpoch(records: readonly ApiEventRecord[]): void {
    const ordered = [...records].sort((left, right) => left.sequence - right.sequence)
    const acceptances = ordered.flatMap((record) => {
      const acceptance = this.durableAcceptance(record)
      return acceptance === null ? [] : [{ acceptance, markerSequence: record.sequence }]
    })

    for (const { acceptance, markerSequence } of acceptances) {
      const acceptedRecord = ordered.find((record) => record.id === acceptance.acceptedEventId)
      if (acceptedRecord !== undefined && acceptedRecord.type !== acceptance.acceptedEventType) {
        throw new MutationStreamProtocolError(
          "Mutation acceptance event references a durable event with a different type.",
        )
      }
      if (acceptedRecord !== undefined && acceptedRecord.sequence >= markerSequence) {
        throw new MutationStreamProtocolError(
          "Mutation acceptance event must follow its referenced durable event.",
        )
      }
      const wasConfirmed = this.operationEpochConfirmed
      this.confirmOperationEpoch(acceptance, acceptedRecord?.sequence ?? markerSequence)
      if (!wasConfirmed) {
        this.acceptOperation(true)
      }
    }

    if (
      this.operationEpochConfirmed &&
      this.operationBoundaryEventId !== null &&
      this.operationTerminalFloorSequence === null
    ) {
      const boundary = ordered.find((record) => record.id === this.operationBoundaryEventId)
      if (boundary !== undefined) {
        if (boundary.type !== this.operationBoundaryEventType) {
          throw new MutationStreamProtocolError(
            "Durable operation boundary has a different event type than the live stream.",
          )
        }
        this.operationTerminalFloorSequence = boundary.sequence
      }
    }

    if (!this.operationEpochConfirmed || this.operationTerminalFloorSequence === null) return
    for (const record of ordered) {
      if (
        record.sequence >= this.operationTerminalFloorSequence &&
        TERMINAL_EVENT_TYPES.has(record.type)
      ) {
        this.operationTerminalEventTypes.add(record.type)
      }
    }
  }

  private validateAttemptResult(result: MutationAttemptResult): MutationAttemptResult {
    if (
      result.kind === "http_failed" &&
      (!Number.isSafeInteger(result.status) || result.status < 300 || result.status > 599)
    ) {
      return {
        kind: "protocol_failed",
        accepted: false,
        error: new MutationStreamProtocolError(
          "Mutation HTTP failure returned an invalid response status.",
        ),
      }
    }
    if (
      (result.kind === "runtime_failed" || result.kind === "observer_failed") &&
      result.error.session_id !== this.request.sessionId
    ) {
      return {
        kind: "protocol_failed",
        accepted: result.accepted,
        error: new MutationStreamProtocolError(
          "Mutation stream error belongs to a different or unspecified session.",
        ),
      }
    }
    return result
  }

  private acceptOperation(resetRecovery = false): void {
    this.operationAccepted = true
    this.replayOnly = false
    if (!resetRecovery) return
    this.reconnectAttempt = 0
    this.recoveryStartedAt = null
    this.recovering = false
  }

  private fail(
    phase: "request_failed" | "runtime_failed" | "transport_failed",
    failure: MutationTransportFailure,
  ): MutationTransportSnapshot {
    this.failure = failure
    this.transition(phase)
    return this.snapshot()
  }

  private transition(phase: MutationTransportPhase): void {
    if (this.disposed && phase !== "cancelled") return
    this.phase = phase
    this.emit()
  }

  private emit(): void {
    if (this.disposed && this.phase !== "cancelled") return
    this.onChange(this.snapshot())
  }
}
