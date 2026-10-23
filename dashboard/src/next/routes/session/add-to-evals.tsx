/**
 * Add a finished session to evals: score it as it is (a captured evaluation, nothing reruns),
 * or keep its first message as a case that reruns with every suite run.
 */
import { useMutation, useQueryClient } from "@tanstack/react-query"
import { X } from "lucide-react"
import { useEffect, useState } from "react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import type { AddToEvals } from "../../api/types"
import { useEvalSuites } from "../../app/queries"
import { useApi } from "../../app/services"
import { ErrorState, useGo } from "../common"
import { runPath, suitePath } from "../evals-page"

const MODES: { id: AddToEvals["mode"]; label: string; text: string }[] = [
  {
    id: "rerun",
    label: "Add as a case",
    text: "Its first message becomes a case. Every run of the suite sends it to the agent again and checks the answer.",
  },
  {
    id: "score",
    label: "Score this session",
    text: "Checks this session's answer as it is, without rerunning it. Useful to record a good or bad outcome.",
  },
]

export function AddToEvalsDialog({
  sessionId,
  specialist,
  onClose,
}: {
  sessionId: string
  specialist: string
  onClose: () => void
}) {
  const api = useApi()
  const go = useGo()
  const queryClient = useQueryClient()
  const suites = useEvalSuites()
  const mine = (suites.data ?? []).filter((s) => s.specialist === specialist)
  const [mode, setMode] = useState<AddToEvals["mode"]>("rerun")
  const [suiteId, setSuiteId] = useState("")
  const [newName, setNewName] = useState("")
  const chosen = suiteId || mine[0]?.id || "new"
  const add = useMutation({
    mutationFn: () =>
      api.addToEvals({
        session_id: sessionId,
        mode,
        suite:
          chosen === "new" ? { name: newName.trim() || `${specialist} cases` } : { id: chosen },
      }),
    onSuccess: ({ suite_id, run_id }) => {
      void queryClient.invalidateQueries({ queryKey: ["evals"] })
      go(run_id ? runPath(run_id) : suitePath(suite_id))
    },
  })
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose()
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [onClose])
  const field =
    "h-8 w-full rounded-lg border border-input bg-background px-2.5 text-sm outline-none focus:border-ring"

  return (
    <div className="fixed inset-0 z-40 flex items-start justify-center p-4 pt-[12vh]">
      <button
        type="button"
        aria-label="Close"
        className="absolute inset-0 bg-black/20"
        onClick={onClose}
      />
      <form
        aria-label="Add to evals"
        className="relative w-full max-w-md space-y-4 rounded-xl border border-border bg-background p-4 shadow-xl"
        onSubmit={(e) => {
          e.preventDefault()
          add.mutate()
        }}
      >
        <div className="flex items-start justify-between gap-3">
          <p className="font-medium">Add to evals</p>
          <Button type="button" variant="ghost" size="icon-sm" aria-label="Close" onClick={onClose}>
            <X />
          </Button>
        </div>
        <fieldset className="space-y-2">
          <legend className="sr-only">How to add it</legend>
          {MODES.map((m) => (
            <label
              key={m.id}
              className={cn(
                "flex cursor-pointer gap-3 rounded-lg border p-3 text-sm",
                mode === m.id ? "border-foreground" : "border-border hover:bg-muted/50",
              )}
            >
              <input
                type="radio"
                name="add-to-evals-mode"
                value={m.id}
                checked={mode === m.id}
                onChange={() => setMode(m.id)}
                className="mt-0.5 accent-foreground"
              />
              <span>
                <span className="font-medium">{m.label}</span>
                <span className="mt-0.5 block text-xs text-muted-foreground">{m.text}</span>
              </span>
            </label>
          ))}
        </fieldset>
        <label className="block space-y-1 text-sm">
          <span className="text-muted-foreground">Suite</span>
          <select
            id="add-to-evals-suite"
            value={chosen}
            onChange={(e) => setSuiteId(e.target.value)}
            className={field}
          >
            {mine.map((s) => (
              <option key={s.id} value={s.id}>
                {s.name} ({s.case_count} cases)
              </option>
            ))}
            <option value="new">New suite…</option>
          </select>
        </label>
        {chosen === "new" && (
          <label className="block space-y-1 text-sm">
            <span className="text-muted-foreground">New suite name</span>
            <input
              id="add-to-evals-name"
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              placeholder={`${specialist} cases`}
              className={field}
            />
          </label>
        )}
        {add.error && <ErrorState error={add.error} />}
        <div className="flex justify-end gap-2">
          <Button type="button" variant="ghost" size="sm" onClick={onClose}>
            Cancel
          </Button>
          <Button type="submit" size="sm" disabled={add.isPending}>
            {add.isPending ? "Adding…" : mode === "score" ? "Score and save" : "Add case"}
          </Button>
        </div>
      </form>
    </div>
  )
}
