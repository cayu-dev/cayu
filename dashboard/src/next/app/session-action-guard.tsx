import { Dialog } from "@base-ui/react/dialog"
import { useBlocker } from "@tanstack/react-router"
import { useRef, useState } from "react"
import { Button } from "@/components/ui/button"

/** Keep the mounted exact-operation owner until settlement or explicit departure. */
export function useSessionActionGuard(locked: boolean) {
  const completing = useRef(false)
  const [close, setClose] = useState<(() => void) | null>(null)
  const blocker = useBlocker({
    shouldBlockFn: ({ current, next }) =>
      !completing.current && locked && current.pathname !== next.pathname,
    enableBeforeUnload: locked,
    withResolver: true,
  })
  const reset = () => {
    setClose(null)
    blocker.reset?.()
  }
  const complete = async (navigate: () => unknown) => {
    completing.current = true
    try {
      await navigate()
    } finally {
      completing.current = false
    }
  }
  const confirmClose = (action: () => void) => {
    if (locked) setClose(() => action)
    else action()
  }
  const dialog = (
    <Dialog.Root
      open={close !== null || blocker.status === "blocked"}
      onOpenChange={(open) => !open && reset()}
    >
      <Dialog.Portal>
        <Dialog.Backdrop className="fixed inset-0 z-50 bg-black/40" />
        <Dialog.Popup className="fixed top-1/2 left-1/2 z-50 w-[min(28rem,calc(100vw-2rem))] -translate-x-1/2 -translate-y-1/2 space-y-4 rounded-xl border border-border bg-background p-5 shadow-xl">
          <Dialog.Title className="font-semibold">An action is still unconfirmed</Dialog.Title>
          <Dialog.Description className="text-sm text-muted-foreground">
            Stay here to recover the original action. Leaving loses this page's local recovery
            handle and does not cancel work already accepted by the server. Inspect the resulting
            session or task before starting new work.
          </Dialog.Description>
          <div className="flex flex-wrap justify-end gap-2">
            <Button variant="outline" onClick={reset}>
              Stay and recover
            </Button>
            <Button
              onClick={() => {
                if (close) {
                  const action = close
                  setClose(null)
                  action()
                } else blocker.proceed?.()
              }}
            >
              Leave without local recovery
            </Button>
          </div>
        </Dialog.Popup>
      </Dialog.Portal>
    </Dialog.Root>
  )
  return { confirmClose, complete, dialog }
}
