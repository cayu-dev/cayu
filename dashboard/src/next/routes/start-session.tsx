/**
 * Starting a session to try the agent. "Run a specialist" (the default) runs one specialist
 * directly with a prompt. "Create a task" runs the app the real way: the app's own worker
 * picks the task up and runs its orchestration. It needs the app to declare task types.
 */
import { useMutation, useQueryClient } from "@tanstack/react-query"
import { useRouter } from "@tanstack/react-router"
import { Play, X } from "lucide-react"
import { useEffect, useRef, useState, useSyncExternalStore } from "react"
import { MutationTransportStatus } from "@/components/dashboard/mutation-transport-status"
import { useViewer } from "@/components/dashboard/viewer"
import { Button } from "@/components/ui/button"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { ROLE_FORBIDDEN_REASON } from "@/lib/dashboard-access"
import type {
  MutationTransportController,
  MutationTransportSnapshot,
} from "@/lib/mutation-transport"
import { reconcileRunAgentSelection } from "@/lib/run-agent-selection"
import { cn } from "@/lib/utils"
import type { Api, TaskType } from "../api/types"
import { useSpecialists, useTaskTypes } from "../app/queries"
import { useApi, useServices } from "../app/services"
import { useSessionActionGuard } from "../app/session-action-guard"
import { TaskOperation } from "../app/task-operation"
import { ErrorState, sessionRoute, taskRoute, useGo } from "./common"

const taskOperations = new WeakMap<Api, TaskOperation>()

function useStartUnavailableReason(mode?: "task" | "specialist"): string | null {
  const viewer = useViewer()
  const { prototype, executionEnabled, taskCreationEnabled } = useServices()
  if (prototype) return null
  if (viewer?.full_access !== true) return ROLE_FORBIDDEN_REASON
  if (mode === "task")
    return taskCreationEnabled === true ? null : "Task creation is unavailable for this deployment."
  return executionEnabled === true ? null : "Starting sessions is unavailable for this deployment."
}

/** The button that opens the form, for a page's top bar. */
export function StartSessionButton({
  open,
  onToggle,
  pending = false,
}: {
  open: boolean
  onToggle: () => void
  pending?: boolean
}) {
  const unavailable = useStartUnavailableReason()
  if (unavailable) return <span className="text-xs text-muted-foreground">{unavailable}</span>
  return (
    <Button variant="outline" size="sm" onClick={onToggle} aria-expanded={open} disabled={pending}>
      <Play />
      Start a session
    </Button>
  )
}

type Mode = "task" | "specialist"
type Field = {
  name: string
  type: string
  description: string
  required: boolean
  options: string[] | null
}

/** The simple fields a task's input schema asks for; anything else is entered as JSON. */
function schemaFields(schema: Record<string, unknown>): Field[] | null {
  const properties = schema.properties
  if (!properties || typeof properties !== "object") return null
  const required = new Set(Array.isArray(schema.required) ? (schema.required as string[]) : [])
  const fields: Field[] = []
  for (const [name, raw] of Object.entries(properties as Record<string, Record<string, unknown>>)) {
    if (!raw || typeof raw !== "object" || typeof raw.type !== "string") return null
    const type = raw.type
    const items = raw.items as Record<string, unknown> | undefined
    if (
      !["string", "integer", "number"].includes(type) &&
      !(type === "array" && items?.type === "string")
    )
      return null
    fields.push({
      name,
      type,
      description: typeof raw.description === "string" ? raw.description : "",
      required: required.has(name),
      options:
        Array.isArray(raw.enum) && raw.enum.every((value) => typeof value === "string")
          ? (raw.enum as string[])
          : null,
    })
  }
  return fields
}

function inputFrom(fields: Field[], values: Record<string, string>): Record<string, unknown> {
  const input: Record<string, unknown> = {}
  for (const f of fields) {
    const v = (values[f.name] ?? "").trim()
    if (!v) continue
    if (f.type === "integer" || f.type === "number") input[f.name] = Number(v)
    else if (f.type === "array")
      input[f.name] = v
        .split(",")
        .map((x) => x.trim())
        .filter(Boolean)
    else input[f.name] = v
  }
  return input
}

function StartSessionForm({ onClose, initialMode, onPendingChange }: StartSessionProps) {
  const api = useApi()
  const router = useRouter()
  const origin = router.state.location.pathname
  const taskOperation =
    taskOperations.get(api) ?? new TaskOperation((body, id) => api.createTask(body, id))
  taskOperations.set(api, taskOperation)
  const taskSnapshot = useSyncExternalStore(taskOperation.subscribe, taskOperation.snapshot)
  const { prototype, taskCreationEnabled } = useServices()
  const mounted = useRef(true)
  const abort = useRef<AbortController | null>(null)
  const transport = useRef<MutationTransportController | null>(null)
  const [snapshot, setSnapshot] = useState<MutationTransportSnapshot | null>(null)
  const [retrying, setRetrying] = useState(false)
  const [observationError, setObservationError] = useState<unknown>(null)
  useEffect(() => {
    mounted.current = true
    return () => {
      mounted.current = false
      abort.current?.abort()
      transport.current?.dispose()
    }
  }, [])
  const go = useGo()
  const queryClient = useQueryClient()
  const specialists = useSpecialists()
  const taskTypes = useTaskTypes()
  const types = taskTypes.data ?? []
  const [mode, setMode] = useState<Mode | null>(initialMode ?? null)
  const current: Mode = mode === "task" ? "task" : "specialist"
  const [agent, setAgent] = useState("")
  const [prompt, setPrompt] = useState("")
  const [typeName, setTypeName] = useState("")
  const [values, setValues] = useState<Record<string, string>>({})
  const [json, setJson] = useState("{}")
  useEffect(() => {
    setAgent(reconcileRunAgentSelection(agent, specialists.data?.agents ?? []))
  }, [agent, specialists.data])
  useEffect(() => {
    if (types.length && !types.some((t) => t.name === typeName)) setTypeName(types[0]?.name ?? "")
  }, [types, typeName])
  const taskType: TaskType | undefined = types.find((t) => t.name === typeName)
  const fields = taskType ? schemaFields(taskType.input_schema) : null
  const missing = fields?.filter((f) => f.required && !(values[f.name] ?? "").trim()) ?? []

  // A new session opens; a new task opens its task page, where its session shows up.
  const done = (path: ReturnType<typeof sessionRoute | typeof taskRoute>) => {
    if (!mounted.current || router.state.location.pathname !== origin) return
    void queryClient.invalidateQueries({ queryKey: ["sessions"] })
    void queryClient.invalidateQueries({ queryKey: ["session-list"] })
    void queryClient.invalidateQueries({ queryKey: ["tasks"] })
    void guard.complete(() => go(path))
  }
  const start = useMutation({
    mutationFn: async () => {
      abort.current?.abort()
      transport.current?.dispose()
      transport.current = null
      setSnapshot(null)
      setObservationError(null)
      if (current === "specialist") {
        const controller = new AbortController()
        abort.current = controller
        const result = await api.start(
          { agent, prompt: prompt.trim() },
          {
            signal: controller.signal,
            onController: (owner) => {
              transport.current = owner
            },
            onChange: (next) => {
              if (!controller.signal.aborted) setSnapshot(next)
            },
          },
        )
        if (controller.signal.aborted) return null
        if (result.snapshot) setSnapshot(result.snapshot)
        if (result.snapshot && !["terminal", "runtime_failed"].includes(result.snapshot.phase))
          return null
        if (!result.session_id) throw new Error("The runtime did not return a session identity.")
        return sessionRoute(result.session_id)
      }
      const input = fields
        ? inputFrom(fields, values)
        : (JSON.parse(json) as Record<string, unknown>)
      if (typeof input !== "object" || input === null || Array.isArray(input))
        throw new Error("Task input must be a JSON object.")
      await taskOperation.submit({ type: typeName, input })
      if (!mounted.current) return null
      if (taskOperation.state.phase === "succeeded") {
        const path = taskRoute(taskOperation.state.taskId!)
        taskOperation.clear()
        return path
      }
      return null
    },
    onSuccess: (path) => {
      if (path) done(path)
    },
  })
  const pending =
    current === "specialist" &&
    (start.isPending ||
      (snapshot !== null &&
        !["terminal", "runtime_failed", "request_failed"].includes(snapshot.phase)))
  const guard = useSessionActionGuard(pending)
  useEffect(() => {
    onPendingChange?.(pending)
    return () => onPendingChange?.(false)
  }, [pending, onPendingChange])
  const outcomeBlocksSubmission = snapshot !== null && snapshot.phase !== "request_failed"
  const ready =
    current === "specialist" ? !!(prompt.trim() && agent) : !!taskType && missing.length === 0
  const field =
    "h-8 w-full rounded-lg border border-input bg-background px-2.5 text-sm outline-none focus:border-ring"

  if (
    current === "task" &&
    (taskSnapshot.phase === "pending" ||
      taskSnapshot.phase === "unknown" ||
      taskSnapshot.phase === "succeeded")
  )
    return (
      <div className="mb-6 space-y-3 rounded-xl border p-4" role="status">
        <p>
          {taskSnapshot.phase === "pending"
            ? "Creating task…"
            : taskSnapshot.phase === "unknown"
              ? "The creation outcome is unknown. Retry the same operation to recover its result; no second task will be created."
              : "Task created."}
        </p>
        {taskSnapshot.error != null && <ErrorState error={taskSnapshot.error} />}
        {taskSnapshot.phase === "unknown" && (
          <Button onClick={() => void taskOperation.submit()}>Retry task creation</Button>
        )}
        {taskSnapshot.phase === "succeeded" && (
          <Button
            onClick={() => {
              const id = taskSnapshot.taskId!
              taskOperation.clear()
              done(taskRoute(id))
            }}
          >
            Open created task
          </Button>
        )}
        <Button
          variant="outline"
          disabled={taskSnapshot.phase === "pending"}
          onClick={() => setMode("specialist")}
        >
          Run a specialist
        </Button>
        {taskSnapshot.phase === "succeeded" && (
          <Button variant="ghost" onClick={() => taskOperation.clear()}>
            Dismiss task result
          </Button>
        )}
        <Button variant="ghost" onClick={onClose}>
          Close
        </Button>
      </div>
    )
  return (
    <form
      className="mb-8 space-y-4 rounded-xl border border-border bg-card p-4"
      onSubmit={(e) => {
        e.preventDefault()
        if (ready && !start.isPending && !outcomeBlocksSubmission) start.mutate()
      }}
    >
      <div className="flex items-start justify-between gap-3">
        <div
          className="flex rounded-lg border border-border p-0.5"
          role="tablist"
          aria-label="How to start"
        >
          {(
            [
              ["specialist", "Run a specialist"],
              ["task", "Create a task"],
            ] as const
          ).map(([value, label]) => {
            const unavailable =
              value === "task" &&
              (types.length === 0 || (!prototype && taskCreationEnabled !== true))
            const tab = (
              <button
                key={value}
                type="button"
                role="tab"
                aria-selected={current === value}
                aria-disabled={unavailable || undefined}
                aria-describedby={unavailable ? "start-no-task-types" : undefined}
                disabled={pending}
                onClick={() => !unavailable && !pending && setMode(value)}
                className={cn(
                  "rounded-md px-2.5 py-1 text-[13px]",
                  current === value
                    ? "bg-muted font-medium"
                    : unavailable
                      ? "cursor-not-allowed text-muted-foreground/50"
                      : "text-muted-foreground hover:text-foreground",
                )}
              >
                {label}
              </button>
            )
            if (!unavailable) return tab
            // aria-disabled rather than disabled, so the tab still takes hover and focus
            // and the tooltip can say why.
            return (
              <Tooltip key={value}>
                <TooltipTrigger render={tab} />
                <TooltipContent id="start-no-task-types">
                  This agent doesn't declare any task types. Declare the tasks your worker handles,
                  with an input schema, to create them from here.
                </TooltipContent>
              </Tooltip>
            )
          })}
        </div>
        <Button
          type="button"
          variant="ghost"
          size="icon-sm"
          aria-label="Close"
          onClick={() => guard.confirmClose(onClose)}
        >
          <X />
        </Button>
      </div>

      {current === "task" ? (
        <>
          <p className="text-xs text-muted-foreground">
            Creates a task for the agent's own worker, which runs it the way production does.
          </p>
          <div className="grid gap-3 sm:grid-cols-[10rem_1fr] sm:items-center">
            <label htmlFor="start-task-type" className="text-sm text-muted-foreground">
              Task type
            </label>
            <select
              id="start-task-type"
              value={typeName}
              onChange={(e) => {
                setTypeName(e.target.value)
                setValues({})
              }}
              className={field}
            >
              {types.map((t) => (
                <option key={t.name} value={t.name}>
                  {t.name}
                </option>
              ))}
            </select>
            {taskType && (
              <p className="text-xs text-muted-foreground sm:col-start-2">
                {taskType.description}
                {taskType.agent_name ? ` Handled by ${taskType.agent_name}.` : ""}
              </p>
            )}
            {fields?.map((f) => (
              <div key={f.name} className="contents">
                <label htmlFor={`start-field-${f.name}`} className="text-sm">
                  <span className="font-mono text-[13px]">{f.name}</span>
                  {f.required ? "" : <span className="text-muted-foreground"> (optional)</span>}
                </label>
                {f.options ? (
                  <select
                    id={`start-field-${f.name}`}
                    value={values[f.name] ?? ""}
                    onChange={(e) => setValues({ ...values, [f.name]: e.target.value })}
                    className={field}
                  >
                    <option value="">{f.description || "Choose"}</option>
                    {f.options.map((o) => (
                      <option key={o} value={o}>
                        {o}
                      </option>
                    ))}
                  </select>
                ) : (
                  <input
                    id={`start-field-${f.name}`}
                    inputMode={f.type === "integer" || f.type === "number" ? "numeric" : undefined}
                    value={values[f.name] ?? ""}
                    onChange={(e) => setValues({ ...values, [f.name]: e.target.value })}
                    placeholder={f.description}
                    className={field}
                  />
                )}
              </div>
            ))}
          </div>
          {taskType && !fields && (
            <>
              <label htmlFor="start-task-json" className="text-sm text-muted-foreground">
                Input (JSON)
              </label>
              <textarea
                id="start-task-json"
                rows={5}
                value={json}
                onChange={(e) => setJson(e.target.value)}
                className="block w-full rounded-lg border border-input bg-background px-3 py-2 font-mono text-xs outline-none focus:border-ring"
              />
            </>
          )}
        </>
      ) : (
        <>
          <p className="text-xs text-muted-foreground">
            Runs one specialist directly with a prompt. The agent's own orchestration isn't
            involved.
          </p>
          <div className="flex flex-wrap items-center gap-2">
            <label htmlFor="start-specialist" className="text-sm text-muted-foreground">
              Specialist
            </label>
            <select
              id="start-specialist"
              value={agent}
              onChange={(e) => setAgent(e.target.value)}
              className="h-8 rounded-lg border border-input bg-background px-2 text-sm"
            >
              <option value="">Choose a specialist</option>
              {specialists.data?.agents.map((a) => (
                <option key={a.name} value={a.name}>
                  {a.name}
                </option>
              ))}
            </select>
          </div>
          {specialists.isSuccess && specialists.data.agents.length === 0 && (
            <p role="status">No specialists are registered.</p>
          )}
          {specialists.error && <ErrorState error={specialists.error} />}
          <label htmlFor="start-prompt" className="sr-only">
            Prompt
          </label>
          <textarea
            id="start-prompt"
            rows={3}
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="The prompt, as the agent would hand it to the specialist"
            className="block w-full resize-y rounded-lg border border-input bg-background px-3 py-2 text-sm outline-none focus:border-ring"
          />
        </>
      )}

      {guard.dialog}
      {pending && (
        <p role="status">
          This action is unconfirmed. Keep this form open to recover its original result.
        </p>
      )}
      {start.error && <ErrorState error={start.error} />}
      {taskSnapshot.phase === "rejected" && taskSnapshot.error != null && (
        <ErrorState error={taskSnapshot.error} />
      )}
      {taskTypes.error && <ErrorState error={taskTypes.error} />}
      {current === "task" && taskTypes.isSuccess && types.length === 0 && (
        <p role="status">
          This agent does not declare task types. Declare the tasks your worker handles to create
          them here.
        </p>
      )}
      {observationError !== null && <ErrorState error={observationError} />}
      <MutationTransportStatus
        snapshot={snapshot}
        retryingObservation={retrying}
        onRetryObservation={
          transport.current
            ? () => {
                if (retrying) return
                setRetrying(true)
                void transport.current
                  ?.reobserve(abort.current?.signal)
                  .then((next) => {
                    if (!mounted.current || router.state.location.pathname !== origin) return
                    setSnapshot(next)
                    if (["terminal", "runtime_failed"].includes(next.phase) && next.sessionId)
                      done(sessionRoute(next.sessionId))
                  })
                  .catch(setObservationError)
                  .finally(() => setRetrying(false))
              }
            : undefined
        }
      />
      <div className="flex justify-end">
        <Button
          type="submit"
          size="sm"
          disabled={!ready || start.isPending || outcomeBlocksSubmission}
        >
          {current === "task" ? "Create task" : "Start"}
        </Button>
      </div>
    </form>
  )
}

type StartSessionProps = {
  onClose: () => void
  initialMode?: Mode
  onPendingChange?: (pending: boolean) => void
}

export function StartSession(props: StartSessionProps) {
  useEffect(() => {
    const previous = document.activeElement
    return () => {
      if (
        previous instanceof HTMLElement &&
        previous !== document.body &&
        previous.isConnected &&
        previous.getClientRects().length
      )
        previous.focus()
    }
  }, [])
  const unavailable = useStartUnavailableReason(props.initialMode)
  if (unavailable) return <p role="status">{unavailable}</p>
  return <StartSessionForm {...props} />
}
