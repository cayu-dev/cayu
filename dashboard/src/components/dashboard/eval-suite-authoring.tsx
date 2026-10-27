import { Plus } from "lucide-react"
import { Button } from "@/components/ui/button"
import { suiteSelector } from "@/next/app/eval-view-model"
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
  return (
    <Button
      data-testid="new-evaluation"
      size="sm"
      variant="outline"
      disabled={disabled}
      onClick={() =>
        void (initialRevision
          ? go({
              to: "/evals/suites/$suiteId/edit",
              params: {
                suiteId: suiteSelector({
                  source: "authored",
                  target_key: targetKey,
                  revision: initialRevision,
                  suite_id: null,
                }),
              },
              search: { run: reviewRun || undefined },
            })
          : go({ to: "/evals/suites/new", search: { target: targetKey } }))
      }
    >
      <Plus /> {label}
    </Button>
  )
}
