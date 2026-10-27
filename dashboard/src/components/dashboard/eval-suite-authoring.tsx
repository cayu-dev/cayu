import { Plus } from "lucide-react"
import { Button } from "@/components/ui/button"
import { suiteSelector, suiteViewPath } from "@/next/app/eval-view-model"
import { useGo } from "@/next/routes/common"

/** All authoring entrances share the full-page owner. */
export function EvalSuiteAuthoringAction({
  targetKey,
  initialRevision,
  label = "New suite",
  disabled,
  reviewRun = false,
}: {
  targetKey: string
  initialRevision?: string
  label?: string
  disabled: boolean
  reviewRun?: boolean
}) {
  const go = useGo()
  const path = initialRevision
    ? `${suiteViewPath(suiteSelector({ source: "authored", target_key: targetKey, revision: initialRevision, suite_id: null }))}/edit`
    : "/evals/suites/new"
  return (
    <Button
      data-testid="new-evaluation"
      size="sm"
      variant="outline"
      disabled={disabled}
      onClick={() =>
        void go(path, initialRevision ? { run: reviewRun || undefined } : { target: targetKey })
      }
    >
      <Plus /> {label}
    </Button>
  )
}
