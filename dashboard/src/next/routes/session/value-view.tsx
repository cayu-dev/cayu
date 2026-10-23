/**
 * Renders any structured output without knowing its schema: objects as labelled rows,
 * lists of objects as cards (a "title", "name" or "summary" field becomes the heading),
 * and a JSON view for the exact value.
 */
import { useState } from "react"
import { cn } from "@/lib/utils"

const HEADING_KEYS = ["title", "name", "summary", "label"]
const isRecord = (v: unknown): v is Record<string, unknown> =>
  typeof v === "object" && v !== null && !Array.isArray(v)
const humanize = (key: string) => key.replaceAll("_", " ").replace(/^\w/, (c) => c.toUpperCase())

function Scalar({ value }: { value: unknown }) {
  if (value === null || value === undefined || value === "")
    return <span className="text-muted-foreground">–</span>
  if (typeof value === "boolean") return <span>{value ? "Yes" : "No"}</span>
  if (typeof value === "number")
    return <span className="tabular-nums">{value.toLocaleString()}</span>
  return <span className="whitespace-pre-wrap break-words">{String(value)}</span>
}

function Value({ value, depth }: { value: unknown; depth: number }) {
  if (Array.isArray(value)) {
    if (value.length === 0) return <span className="text-muted-foreground">None</span>
    if (value.every((v) => !isRecord(v) && !Array.isArray(v))) {
      return (
        <ul className="list-disc space-y-0.5 pl-4">
          {value.map((v, i) => (
            // biome-ignore lint/suspicious/noArrayIndexKey: output values have no identity
            <li key={i}>
              <Scalar value={v} />
            </li>
          ))}
        </ul>
      )
    }
    return (
      <div className="space-y-2">
        {value.map((v, i) => (
          // biome-ignore lint/suspicious/noArrayIndexKey: output values have no identity
          <Card key={i} value={v} depth={depth + 1} />
        ))}
      </div>
    )
  }
  if (isRecord(value)) return <Rows value={value} depth={depth + 1} />
  return <Scalar value={value} />
}

function Rows({
  value,
  depth,
  skip,
}: {
  value: Record<string, unknown>
  depth: number
  skip?: string
}) {
  const entries = Object.entries(value).filter(([k]) => k !== skip)
  return (
    <dl
      className={cn(
        "grid gap-x-4 gap-y-1.5 text-sm sm:grid-cols-[minmax(6rem,auto)_1fr]",
        depth > 1 && "text-[13px]",
      )}
    >
      {entries.map(([k, v]) => (
        <div key={k} className="contents">
          <dt className="text-muted-foreground">{humanize(k)}</dt>
          <dd className="min-w-0">
            <Value value={v} depth={depth} />
          </dd>
        </div>
      ))}
    </dl>
  )
}

function Card({ value, depth }: { value: unknown; depth: number }) {
  if (!isRecord(value)) return <Value value={value} depth={depth} />
  const headingKey = HEADING_KEYS.find((k) => typeof value[k] === "string")
  return (
    <div className="rounded-xl border border-border bg-card p-3.5">
      {headingKey && (
        <p className="mb-2 text-sm font-medium leading-snug">{String(value[headingKey])}</p>
      )}
      <Rows value={value} depth={depth} skip={headingKey} />
    </div>
  )
}

/** `label` names the value above it; pass null where a section heading already does. */
export function ValueView({
  value,
  label = "Structured output",
}: {
  value: unknown
  label?: string | null
}) {
  const [json, setJson] = useState(false)
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-muted-foreground">{label}</span>
        <button
          type="button"
          className="text-xs text-muted-foreground hover:text-foreground"
          onClick={() => setJson(!json)}
        >
          {json ? "Show formatted" : "Show JSON"}
        </button>
      </div>
      {json ? (
        <pre className="max-h-96 overflow-auto rounded-lg bg-muted p-3 font-mono text-xs leading-relaxed">
          {JSON.stringify(value, null, 2)}
        </pre>
      ) : isRecord(value) ? (
        <Rows value={value} depth={0} />
      ) : (
        <Value value={value} depth={0} />
      )}
    </div>
  )
}
