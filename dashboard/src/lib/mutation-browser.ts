import type { EventSourceMessage, FetchEventSourceInit } from "@microsoft/fetch-event-source"
import { ApiClientError, fetchSessionEvents, fetchSessionState, throwResponseError } from "./api.ts"
import { apiUrl } from "./config.ts"
import type {
  InterruptSessionBody,
  ResumeBody,
  RunBody,
  ToolApprovalBody,
  ToolApprovalRecoveryBody,
  ToolRoundRecoveryBody,
  UserInputRecoveryBody,
  UserInputResolveBody,
} from "./generated/server-api"
import {
  MutationStreamProtocolError,
  parseMutationSseError,
  parseMutationSseEvent,
} from "./mutation-stream-protocol.ts"
import {
  createMutationId,
  createRunSessionId,
  MUTATION_RECONCILIATION_REQUEST_TIMEOUT_MS,
  type MutationAttemptResult,
  type MutationReplayBaseline,
  MutationTransportController,
  type MutationTransportIO,
  type MutationTransportOptions,
  type MutationTransportPolicy,
  type MutationTransportRequest,
  type MutationTransportSnapshot,
  type MutationVisibilitySource,
  newRunReplayBaseline,
} from "./mutation-transport.ts"

export const MUTATION_SSE_FRAME_MAX_BYTES = 2 * 1024 * 1024
export const MUTATION_SSE_DATA_MAX_BYTES = MUTATION_SSE_FRAME_MAX_BYTES

type FetchEventSource = (input: RequestInfo, init: FetchEventSourceInit) => Promise<void>

export type MutationBrowserDependencies = {
  fetchEventSource?: FetchEventSource
  fetchEventSourceLoader?: () => Promise<FetchEventSource>
  fetchImpl?: typeof fetch
  readSessionEvents?: typeof fetchSessionEvents
  readSessionState?: typeof fetchSessionState
  now?: () => number
  random?: () => number
}

export type MutationExecutionOptions = {
  signal?: AbortSignal
  onChange?: (snapshot: MutationTransportSnapshot) => void
  onController?: (controller: MutationTransportController) => void
  policy?: Partial<MutationTransportPolicy>
  dependencies?: MutationBrowserDependencies
  visibility?: MutationVisibilitySource
}

class MutationAttemptStopped extends Error {
  readonly result: MutationAttemptResult

  constructor(result: MutationAttemptResult) {
    super("Mutation stream attempt reached a typed terminal frame.")
    this.name = "MutationAttemptStopped"
    this.result = result
  }
}

function isAbortError(error: unknown): boolean {
  return error instanceof Error && error.name === "AbortError"
}

function abortError(): DOMException {
  return new DOMException("The mutation browser operation was aborted.", "AbortError")
}

function abortableTimer(milliseconds: number, signal: AbortSignal): Promise<void> {
  if (signal.aborted) return Promise.reject(abortError())
  return new Promise<void>((resolve, reject) => {
    const timer = globalThis.setTimeout(() => {
      signal.removeEventListener("abort", abort)
      resolve()
    }, milliseconds)
    const abort = () => {
      globalThis.clearTimeout(timer)
      reject(abortError())
    }
    signal.addEventListener("abort", abort, { once: true })
  })
}

async function defaultFetchEventSourceLoader(): Promise<FetchEventSource> {
  const module = await import("@microsoft/fetch-event-source")
  return module.fetchEventSource
}

function eventStreamContentType(response: Response): boolean {
  const contentType = response.headers.get("content-type")
  if (contentType === null) return false
  const mediaType = contentType.split(";", 1)[0]?.trim().toLowerCase()
  return mediaType === "text/event-stream"
}

function assertFrameSize(message: EventSourceMessage): void {
  const bytes = new TextEncoder().encode(message.data).byteLength
  if (bytes > MUTATION_SSE_DATA_MAX_BYTES) {
    throw new MutationStreamProtocolError(
      `Mutation SSE data exceeded ${MUTATION_SSE_DATA_MAX_BYTES} bytes.`,
    )
  }
}

function boundedMutationSseBody(body: ReadableStream<Uint8Array>): ReadableStream<Uint8Array> {
  let currentLineBytes = 0
  let currentFrameBytes = 0
  let discardLineFeedAfterCarriageReturn = false

  return body.pipeThrough(
    new TransformStream<Uint8Array, Uint8Array>({
      transform(chunk, controller) {
        for (const byte of chunk) {
          if (discardLineFeedAfterCarriageReturn) {
            discardLineFeedAfterCarriageReturn = false
            if (byte === 0x0a) continue
          }

          if (byte === 0x0d || byte === 0x0a) {
            if (currentLineBytes === 0) currentFrameBytes = 0
            currentLineBytes = 0
            discardLineFeedAfterCarriageReturn = byte === 0x0d
            continue
          }

          currentLineBytes += 1
          currentFrameBytes += 1
          if (currentFrameBytes > MUTATION_SSE_FRAME_MAX_BYTES) {
            throw new MutationStreamProtocolError(
              `Mutation SSE frame exceeded ${MUTATION_SSE_FRAME_MAX_BYTES} bytes.`,
            )
          }
        }
        controller.enqueue(chunk)
      },
    }),
  )
}

function boundMutationSseResponse(response: Response): Response {
  if (!response.ok || response.body === null || !eventStreamContentType(response)) return response
  return new Response(boundedMutationSseBody(response.body), {
    status: response.status,
    statusText: response.statusText,
    headers: response.headers,
  })
}

export function browserMutationVisibilitySource(): MutationVisibilitySource {
  if (typeof document === "undefined") {
    return { isVisible: () => true, subscribe: () => () => undefined }
  }
  return {
    isVisible: () => document.visibilityState !== "hidden",
    subscribe: (listener) => {
      document.addEventListener("visibilitychange", listener)
      return () => document.removeEventListener("visibilitychange", listener)
    },
  }
}

export function createMutationBrowserIO(
  dependencies: MutationBrowserDependencies = {},
): MutationTransportIO {
  const loadFetchEventSource =
    dependencies.fetchEventSource === undefined
      ? (dependencies.fetchEventSourceLoader ?? defaultFetchEventSourceLoader)
      : async () => dependencies.fetchEventSource as FetchEventSource
  const fetchImpl = dependencies.fetchImpl ?? globalThis.fetch.bind(globalThis)
  const readSessionEvents = dependencies.readSessionEvents ?? fetchSessionEvents
  const readSessionState = dependencies.readSessionState ?? fetchSessionState
  const boundedFetch: typeof fetch = async (input, init) =>
    boundMutationSseResponse(await fetchImpl(input, init))

  return {
    async attempt(request, callbacks, signal) {
      let accepted = false
      try {
        const fetchEventSource = await loadFetchEventSource()
        if (signal.aborted) return { kind: "aborted", accepted }
        const headers: Record<string, string> = {
          Accept: "text/event-stream",
          "Cayu-Mutation-ID": request.mutationId,
          "Content-Type": "application/json",
        }
        if (request.lastEventId !== null) headers["Last-Event-ID"] = request.lastEventId

        await fetchEventSource(apiUrl(request.path), {
          method: "POST",
          headers,
          credentials: "same-origin",
          body: JSON.stringify(request.body),
          signal,
          fetch: boundedFetch,
          // The transport controller owns visibility suspension and reconnection.
          // Disabling the library's hidden-document behavior prevents a bare
          // mutation POST from being repeated behind the controller's back.
          openWhenHidden: true,
          async onopen(response) {
            if (!response.ok) await throwResponseError(response)
            if (!eventStreamContentType(response)) {
              throw new MutationStreamProtocolError(
                "Mutation response did not use the text/event-stream content type.",
              )
            }
            accepted = true
            callbacks.onAccepted()
          },
          onmessage(message) {
            if (message.data.length === 0) return
            assertFrameSize(message)
            if (message.event === "error") {
              const error = parseMutationSseError(message.data)
              throw new MutationAttemptStopped(
                error.kind === "runtime"
                  ? { kind: "runtime_failed", accepted: true, error }
                  : { kind: "observer_failed", accepted: true, error },
              )
            }
            const event = parseMutationSseEvent(message.data)
            callbacks.onEvent(event, message.id)
          },
          onerror(error) {
            // A throw opts out of fetch-event-source's implicit retry loop. The
            // controller alone decides whether and how to reconnect.
            throw error
          },
        })
        if (signal.aborted) return { kind: "aborted", accepted }
        return { kind: "closed", accepted }
      } catch (error) {
        if (error instanceof MutationAttemptStopped) return error.result
        if (isAbortError(error) || signal.aborted) return { kind: "aborted", accepted }
        if (error instanceof ApiClientError && !accepted) {
          return { kind: "http_failed", accepted: false, status: error.status, error }
        }
        if (error instanceof MutationStreamProtocolError) {
          return { kind: "protocol_failed", accepted, error }
        }
        return { kind: "network_failed", accepted, error }
      }
    },
    readEvents: (sessionId, afterSequence, limit, signal) =>
      readSessionEvents(
        sessionId,
        { after_sequence: afterSequence, order_by: "sequence_asc", limit },
        signal,
      ),
    readState: (sessionId, signal) => readSessionState(sessionId, signal),
    delay: abortableTimer,
    watchdog: abortableTimer,
    now: dependencies.now ?? Date.now,
    random: dependencies.random ?? Math.random,
  }
}

class MutationReplayBaselineTimeoutError extends Error {
  constructor() {
    super("Timed out while reading the durable mutation replay baseline.")
    this.name = "MutationReplayBaselineTimeoutError"
  }
}

type MutationReplayBaselineOptions = {
  signal?: AbortSignal
  timeoutMs?: number
  readEvents?: typeof fetchSessionEvents
}

export async function fetchMutationReplayBaseline(
  sessionId: string,
  options: MutationReplayBaselineOptions = {},
): Promise<MutationReplayBaseline> {
  const timeoutMs = options.timeoutMs ?? MUTATION_RECONCILIATION_REQUEST_TIMEOUT_MS
  if (!Number.isSafeInteger(timeoutMs) || timeoutMs <= 0) {
    throw new RangeError("Mutation replay baseline timeout must be a positive safe integer.")
  }
  if (options.signal?.aborted) throw abortError()

  const controller = new AbortController()
  let timer: ReturnType<typeof globalThis.setTimeout> | undefined
  let abortListener: (() => void) | undefined
  const timeout = new Promise<never>((_resolve, reject) => {
    timer = globalThis.setTimeout(() => {
      controller.abort()
      reject(new MutationReplayBaselineTimeoutError())
    }, timeoutMs)
  })
  const externalAbort = new Promise<never>((_resolve, reject) => {
    if (options.signal === undefined) return
    abortListener = () => {
      controller.abort()
      reject(abortError())
    }
    options.signal.addEventListener("abort", abortListener, { once: true })
  })

  try {
    const readEvents = options.readEvents ?? fetchSessionEvents
    const page = await Promise.race([
      readEvents(sessionId, { order_by: "sequence_desc", limit: 1 }, controller.signal),
      timeout,
      externalAbort,
    ])
    const latest = page.events[0]
    return latest === undefined
      ? newRunReplayBaseline(sessionId)
      : { marker: `${sessionId}:${latest.id}`, sequence: latest.sequence }
  } finally {
    if (timer !== undefined) globalThis.clearTimeout(timer)
    if (abortListener !== undefined) {
      options.signal?.removeEventListener("abort", abortListener)
    }
    controller.abort()
  }
}

function createBrowserMutationController(
  request: MutationTransportRequest,
  options: MutationExecutionOptions,
): MutationTransportController {
  const controllerOptions: MutationTransportOptions = {
    request,
    io: createMutationBrowserIO(options.dependencies),
    visibility: options.visibility ?? browserMutationVisibilitySource(),
    ...(options.onChange === undefined ? {} : { onChange: options.onChange }),
    ...(options.policy === undefined ? {} : { policy: options.policy }),
  }
  return new MutationTransportController(controllerOptions)
}

async function executeMutation(
  request: MutationTransportRequest,
  options: MutationExecutionOptions,
): Promise<MutationTransportSnapshot> {
  const controller = createBrowserMutationController(request, options)
  options.onController?.(controller)
  return controller.start(options.signal)
}

async function executeExistingSessionMutation(
  path: string,
  body: Record<string, unknown>,
  sessionId: string,
  options: MutationExecutionOptions,
): Promise<MutationTransportSnapshot> {
  const baseline = await fetchMutationReplayBaseline(sessionId, {
    ...(options.signal === undefined ? {} : { signal: options.signal }),
    timeoutMs:
      options.policy?.reconciliationRequestTimeoutMs ?? MUTATION_RECONCILIATION_REQUEST_TIMEOUT_MS,
    readEvents: options.dependencies?.readSessionEvents ?? fetchSessionEvents,
  })
  return executeMutation(
    {
      path,
      body,
      sessionId,
      mutationId: createMutationId(),
      baseline,
    },
    options,
  )
}

export async function executeRunMutation(
  body: RunBody,
  options: MutationExecutionOptions = {},
): Promise<MutationTransportSnapshot> {
  const sessionId = createRunSessionId()
  return executeMutation(
    {
      path: "/run",
      body: { ...body, session_id: sessionId },
      sessionId,
      mutationId: createMutationId(),
      baseline: newRunReplayBaseline(sessionId),
    },
    options,
  )
}

export function executeResumeMutation(
  body: ResumeBody,
  options: MutationExecutionOptions = {},
): Promise<MutationTransportSnapshot> {
  return executeExistingSessionMutation("/resume", body, body.session_id, options)
}

export function executeInterruptMutation(
  sessionId: string,
  body: InterruptSessionBody,
  options: MutationExecutionOptions = {},
): Promise<MutationTransportSnapshot> {
  return executeExistingSessionMutation(
    `/sessions/${encodeURIComponent(sessionId)}/interrupt`,
    body,
    sessionId,
    options,
  )
}

export function executeResolveToolApprovalMutation(
  body: ToolApprovalBody,
  options: MutationExecutionOptions = {},
): Promise<MutationTransportSnapshot> {
  return executeExistingSessionMutation("/tool-approvals/resolve", body, body.session_id, options)
}

export function executeRecoverToolApprovalMutation(
  body: ToolApprovalRecoveryBody,
  options: MutationExecutionOptions = {},
): Promise<MutationTransportSnapshot> {
  return executeExistingSessionMutation("/tool-approvals/recover", body, body.session_id, options)
}

export function executeRecoverToolRoundMutation(
  body: ToolRoundRecoveryBody,
  options: MutationExecutionOptions = {},
): Promise<MutationTransportSnapshot> {
  return executeExistingSessionMutation("/tool-rounds/recover", body, body.session_id, options)
}

export function executeResolveUserInputMutation(
  body: UserInputResolveBody,
  options: MutationExecutionOptions = {},
): Promise<MutationTransportSnapshot> {
  return executeExistingSessionMutation("/user-input/resolve", body, body.session_id, options)
}

export function executeRecoverUserInputMutation(
  body: UserInputRecoveryBody,
  options: MutationExecutionOptions = {},
): Promise<MutationTransportSnapshot> {
  return executeExistingSessionMutation("/user-input/recover", body, body.session_id, options)
}
