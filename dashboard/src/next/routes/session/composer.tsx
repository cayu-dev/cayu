import { ArrowUp, Square } from "lucide-react"
import { useState } from "react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

/**
 * A one-line bar for messaging the session; it grows while focused or holding text.
 * Enter sends, Shift+Enter adds a line. Shows Stop while the session runs.
 */
export function Composer({
  id,
  placeholder,
  onSend,
  running = false,
  onStop,
  disabled = false,
}: {
  id: string
  placeholder: string
  onSend: (text: string) => void
  running?: boolean
  onStop?: () => void
  disabled?: boolean
}) {
  const [text, setText] = useState("")
  const [focused, setFocused] = useState(false)
  const open = focused || text.length > 0
  const send = () => {
    const value = text.trim()
    if (!value || disabled) return
    onSend(value)
    setText("")
  }
  return (
    <div
      className={cn(
        "flex gap-2 rounded-xl border border-border bg-card px-3 py-1.5 focus-within:border-ring",
        open ? "items-end" : "items-center",
      )}
    >
      <label htmlFor={id} className="sr-only">
        {placeholder}
      </label>
      <textarea
        id={id}
        rows={open ? 3 : 1}
        value={text}
        placeholder={placeholder}
        onFocus={() => setFocused(true)}
        onBlur={() => setFocused(false)}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
            e.preventDefault()
            send()
          }
        }}
        className="block max-h-40 min-w-0 flex-1 resize-none bg-transparent py-1 text-sm outline-none placeholder:text-muted-foreground"
      />
      {running && onStop ? (
        <Button size="icon-sm" className="rounded-full" aria-label="Stop" onClick={onStop}>
          <Square className="size-3 fill-current" />
        </Button>
      ) : (
        <Button
          size="icon-sm"
          className="rounded-full"
          aria-label="Send"
          disabled={!text.trim() || disabled}
          onClick={send}
        >
          <ArrowUp />
        </Button>
      )}
    </div>
  )
}
