/**
 * Starting a session to try the agent. "Run a specialist" (the default) runs one specialist
 * directly with a prompt. "Create a task" runs the app the real way: the app's own worker
 * picks the task up and runs its orchestration. It needs the app to declare task types.
 */
import { useMutation, useQueryClient } from "@tanstack/react-query"
import { Play, X } from "lucide-react"
import { useEffect, useState } from "react"
import { Button } from "@/components/ui/button"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { cn } from "@/lib/utils"
import type { TaskType } from "../api/types"
import { useSpecialists, useTaskTypes } from "../app/queries"
import { useApi } from "../app/services"
import { ErrorState, sessionPath, taskPath, useGo } from "./common"

/** The button that opens the form, for a page's top bar. */
export function StartSessionButton({ open, onToggle }: { open: boolean; onToggle: () => void }) {
  return (
    <Button variant="outline" size="sm" onClick={onToggle} aria-expanded={open}>
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
    const type = typeof raw.type === "string" ? raw.type : "string"
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
      options: Array.isArray(raw.enum) ? (raw.enum as string[]) : null,
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

export function StartSession({
  onClose,
  initialMode,
}: {
  onClose: () => void
  initialMode?: Mode
}) {
  const api = useApi()
  const go = useGo()
  const queryClient = useQueryClient()
  const specialists = useSpecialists()
  const taskTypes = useTaskTypes()
  const types = taskTypes.data ?? []
  const [mode, setMode] = useState<Mode | null>(initialMode ?? null)
  const current: Mode = mode === "task" && types.length ? "task" : "specialist"
  const [agent, setAgent] = useState("")
  const [prompt, setPrompt] = useState("")
  const [typeName, setTypeName] = useState("")
  const [values, setValues] = useState<Record<string, string>>({})
  const [json, setJson] = useState("{}")
  useEffect(() => {
    const names = specialists.data?.agents.map((a) => a.name) ?? []
    if (names.length && !names.includes(agent)) setAgent(names[0] ?? "")
  }, [agent, specialists.data])
  useEffect(() => {
    if (types.length && !types.some((t) => t.name === typeName)) setTypeName(types[0]?.name ?? "")
  }, [types, typeName])
  const taskType: TaskType | undefined = types.find((t) => t.name === typeName)
  const fields = taskType ? schemaFields(taskType.input_schema) : null
  const missing = fields?.filter((f) => f.required && !(values[f.name] ?? "").trim()) ?? []

  // A new session opens; a new task opens its task page, where its session shows up.
  const done = (path: string) => {
    void queryClient.invalidateQueries({ queryKey: ["sessions"] })
    void queryClient.invalidateQueries({ queryKey: ["tasks"] })
    go(path)
  }
  const start = useMutation({
    mutationFn: async () => {
      if (current === "specialist")
        return sessionPath((await api.start({ agent, prompt: prompt.trim() })).session_id)
      const input = fields
        ? inputFrom(fields, values)
        : (JSON.parse(json) as Record<string, unknown>)
      return taskPath((await api.createTask({ type: typeName, input })).task_id)
    },
    onSuccess: done,
  })
  const ready =
    current === "specialist" ? !!(prompt.trim() && agent) : !!taskType && missing.length === 0
  const field =
    "h-8 w-full rounded-lg border border-input bg-background px-2.5 text-sm outline-none focus:border-ring"

  return (
    <form
      className="mb-8 space-y-4 rounded-xl border border-border bg-card p-4"
      onSubmit={(e) => {
        e.preventDefault()
        if (ready) start.mutate()
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
            const unavailable = value === "task" && types.length === 0
            const tab = (
              <button
                key={value}
                type="button"
                role="tab"
                aria-selected={current === value}
                aria-disabled={unavailable || undefined}
                aria-describedby={unavailable ? "start-no-task-types" : undefined}
                onClick={() => !unavailable && setMode(value)}
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
        <Button type="button" variant="ghost" size="icon-sm" aria-label="Close" onClick={onClose}>
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
              {specialists.data?.agents.map((a) => (
                <option key={a.name} value={a.name}>
                  {a.name}
                </option>
              ))}
            </select>
          </div>
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

      {start.error && <ErrorState error={start.error} />}
      <div className="flex justify-end">
        <Button type="submit" size="sm" disabled={!ready || start.isPending}>
          {current === "task" ? "Create task" : "Start"}
        </Button>
      </div>
    </form>
  )
}
