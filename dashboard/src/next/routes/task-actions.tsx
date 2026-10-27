import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { useEffect, useRef, useState, useSyncExternalStore } from "react"
import { useViewer } from "@/components/dashboard/viewer"
import { Button } from "@/components/ui/button"
import { ApiClientError } from "@/lib/api"
import type { TaskReplaySource } from "@/lib/generated/server-api"
import { cancelDeclaredTask, readTaskReplay, replayDeclaredTask } from "../api/tasks"
import type { Api, TaskInfo } from "../api/types"
import { useServices } from "../app/services"
import { FINISHED_STATUSES } from "../app/task-model"
import { TaskOperation } from "../app/task-operation"
import { TASK_POLL_MS } from "../app/task-polling"
import { ErrorState, taskRoute, useGo } from "./common"

const replayOwners = new WeakMap<Api, Map<string, TaskOperation>>()

/** Replay always uses the runtime's exact source incarnation; it never copies visible input. */
export function TaskActions({ task }: { task: TaskInfo }) {
  const viewer = useViewer(),
    services = useServices(),
    go = useGo(),
    cache = useQueryClient()
  const writable =
    !!services.prototype || (viewer?.full_access === true && services.taskMutationsEnabled === true)
  const creatable =
    !services.prototype && viewer?.full_access === true && services.taskCreationEnabled === true
  const owners = replayOwners.get(services.api) ?? new Map<string, TaskOperation>()
  replayOwners.set(services.api, owners)
  const replay =
    owners.get(task.id) ??
    new TaskOperation((body, operationId) =>
      replayDeclaredTask(task.id, {
        type: body.type,
        operation_id: operationId,
        source: body.input as TaskReplaySource,
      }),
    )
  owners.set(task.id, replay)
  const replayState = useSyncExternalStore(replay.subscribe, replay.snapshot)
  const mounted = useRef(true)
  useEffect(() => {
    mounted.current = true
    return () => {
      mounted.current = false
    }
  }, [])
  const replayStarting = useRef(false)
  const replayTask = async () => {
    if (replayStarting.current) return
    replayStarting.current = true
    try {
      if (replay.state.phase === "unknown") {
        await replay.submit()
      } else {
        // Fresh authority belongs only to a new operation. Unknown retries retain
        // the original source even if this read would now produce another tuple.
        const observed = await availability.refetch()
        if (
          !mounted.current ||
          observed.error ||
          !observed.data?.available ||
          !observed.data.source
        )
          return
        await replay.submit({ type: task.type, input: { ...observed.data.source } })
      }
      if (mounted.current && replay.state.phase === "succeeded") {
        const id = replay.state.taskId!
        replay.clear()
        void cache.invalidateQueries({ queryKey: ["tasks"] })
        go(taskRoute(id))
      }
    } finally {
      replayStarting.current = false
    }
  }
  const availability = useQuery({
    queryKey: ["tasks", task.id, "replay"],
    enabled: creatable,
    queryFn: ({ signal }) => readTaskReplay(task.id, signal),
    refetchInterval: TASK_POLL_MS,
  })
  const [operation, setOperation] = useState<{
    id: string
    kind: "cancel"
    uncertain: boolean
    task: TaskInfo
  } | null>(null)
  const [confirm, setConfirm] = useState(false)
  const action = useMutation({
    mutationFn: async (kind: "cancel") => {
      const selected = operation ?? {
        id: crypto.randomUUID(),
        kind,
        uncertain: false,
        task,
      }
      setOperation(selected)
      try {
        await cancelDeclaredTask(selected.task, selected.id)
      } catch (error) {
        const definitive =
          !selected.uncertain &&
          error instanceof ApiClientError &&
          [400, 401, 403, 404, 409, 413, 422].includes(error.status)
        setOperation(definitive ? null : { ...selected, uncertain: true })
        if (definitive) await cache.invalidateQueries({ queryKey: ["tasks", task.id], exact: true })
        throw error
      }
    },
    onSuccess: () => {
      setOperation(null)
      setConfirm(false)
      void cache.invalidateQueries({ queryKey: ["tasks"] })
    },
  })
  const refresh = useMutation({
    mutationFn: () => services.api.task(task.id),
    onSuccess: (current) => {
      cache.setQueryData(["tasks", task.id], current)
      action.reset()
      setOperation(null)
      setConfirm(false)
    },
  })
  const unknown = operation?.uncertain === true
  if (!writable) return null
  return (
    <div className="my-4 flex flex-wrap items-center gap-2">
      {!FINISHED_STATUSES.has(task.status) && !services.prototype && (
        <Button
          variant="outline"
          size="sm"
          disabled={action.isPending || unknown}
          onClick={() => setConfirm(true)}
        >
          Cancel task
        </Button>
      )}
      {creatable && (
        <Button
          variant="outline"
          size="sm"
          disabled={
            action.isPending ||
            unknown ||
            ["pending", "unknown", "succeeded"].includes(replayState.phase) ||
            availability.isFetching ||
            availability.data?.available !== true
          }
          onClick={() => void replayTask()}
        >
          Run again
        </Button>
      )}
      {creatable && availability.data?.available === false && (
        <p className="text-xs text-muted-foreground">
          Run again unavailable:{" "}
          {availability.data.unavailable_reason?.replaceAll("_", " ") ??
            "no authorized replay source"}
          .
        </p>
      )}
      {availability.error && <ErrorState error={availability.error} />}
      {replayState.error != null && <ErrorState error={replayState.error} />}
      {replayState.phase === "unknown" && (
        <div role="status">
          <p>The replay outcome is unknown. Recover the same operation before starting another.</p>
          <Button onClick={() => void replayTask()}>Retry same replay</Button>
        </div>
      )}
      {replayState.phase === "succeeded" && (
        <Button
          onClick={() => {
            const id = replayState.taskId!
            replay.clear()
            go(taskRoute(id))
          }}
        >
          Open replayed task
        </Button>
      )}

      {confirm && (
        <fieldset aria-label="Confirm cancellation" className="w-full rounded-lg border p-3">
          <p>
            Cancel this task? A running worker retains ownership until it stops; dispatched effects
            may already have occurred.
          </p>
          <Button disabled={action.isPending || unknown} onClick={() => action.mutate("cancel")}>
            Confirm cancel task
          </Button>
          <Button variant="ghost" onClick={() => setConfirm(false)}>
            Keep task
          </Button>
        </fieldset>
      )}
      {action.error && <ErrorState error={action.error} />}
      {refresh.error && <ErrorState error={refresh.error} />}
      {unknown && operation && (
        <div role="status">
          <p>The outcome is unknown. Inspect the current task before another action.</p>
          {operation.task.schedule ? (
            <Button disabled={action.isPending} onClick={() => action.mutate(operation.kind)}>
              Retry same operation
            </Button>
          ) : (
            <Button disabled={refresh.isPending} onClick={() => refresh.mutate()}>
              Refresh task status
            </Button>
          )}
        </div>
      )}
    </div>
  )
}
