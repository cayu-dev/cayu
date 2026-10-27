import { Plus, Trash2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Textarea } from "@/components/ui/textarea"
import type { EvalCaseDraftV2 } from "@/lib/generated/server-api"

export function SimpleInputEditor({
  evalCase,
  edit,
}: {
  evalCase: EvalCaseDraftV2
  edit: (evalCase: EvalCaseDraftV2) => void
}) {
  if (!("input" in evalCase.stimulus)) return null
  const messages = evalCase.stimulus.input.messages
  if (messages === undefined) {
    return (
      <Card size="sm" data-testid="authored-simple-input">
        <CardContent className="text-sm text-muted-foreground">
          This case uses runtime-owned opaque external input and cannot be edited in the Developer
          dashboard suite authoring.
        </CardContent>
      </Card>
    )
  }
  const updateMessages = (nextMessages: typeof messages) => {
    edit({
      ...evalCase,
      stimulus: { kind: "simple_input", input: { messages: nextMessages } },
    })
  }
  return (
    <Card size="sm" data-testid="authored-simple-input">
      <CardHeader className="grid-cols-[1fr_auto]">
        <div>
          <CardTitle>Input</CardTitle>
          <p className="mt-1 text-xs text-muted-foreground">
            Each message is supplied as ordered user input to one fresh agent session.
          </p>
        </div>
        <Button
          type="button"
          size="xs"
          variant="outline"
          disabled={messages.length >= 16}
          onClick={() =>
            updateMessages([...messages, { role: "user", text: "Add another user message." }])
          }
        >
          <Plus /> Add message
        </Button>
      </CardHeader>
      <CardContent className="space-y-3">
        {messages.map((message, index) => (
          // biome-ignore lint/suspicious/noArrayIndexKey: message rows hold no local state.
          <div key={`${index}:${messages.length}`} className="flex items-start gap-2">
            <Textarea
              rows={3}
              aria-label={`Case input message ${index + 1}`}
              value={message.text}
              onChange={(event) => {
                const next = [...messages]
                next[index] = { role: "user", text: event.target.value }
                updateMessages(next)
              }}
            />
            <Button
              type="button"
              size="icon-xs"
              variant="ghost"
              aria-label={`Remove case input message ${index + 1}`}
              disabled={messages.length === 1}
              onClick={() => updateMessages(messages.filter((_, candidate) => candidate !== index))}
            >
              <Trash2 />
            </Button>
          </div>
        ))}
      </CardContent>
    </Card>
  )
}
