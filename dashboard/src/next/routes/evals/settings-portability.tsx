import { useQuery, useQueryClient } from "@tanstack/react-query"
import { Download, Upload } from "lucide-react"
import { useEffect, useRef, useState } from "react"
import type { CalibrationState } from "@/components/dashboard/structured-judge-editor"
import { Button } from "@/components/ui/button"
import { downloadEvalCorpus, fetchEvalCorpora, importEvalCorpus } from "@/lib/api"
import { evalErrorMessage, preflightEvalCorpusFile, shortEvalIdentity } from "@/lib/evals-dashboard"
import { formatBytes, formatDateTime } from "@/lib/format"
import { NavLink } from "../common"
import { LoadState, Pages, Section } from "./shared"
import { downloadBlob } from "./technical-details"

export function CorpusSettings({
  targetKey,
  disabled,
  onStateChange,
}: {
  targetKey: string
  disabled: boolean
  onStateChange: (state: CalibrationState) => void
}) {
  const client = useQueryClient()
  const [cursor, setCursor] = useState<string>()
  const [pending, setPending] = useState(false)
  const [error, setError] = useState<string>()
  const [notice, setNotice] = useState<string>()
  const input = useRef<HTMLInputElement>(null)
  const controller = useRef<AbortController | null>(null)
  useEffect(() => () => controller.current?.abort(), [])
  useEffect(() => onStateChange({ dirty: false, locked: pending }), [pending, onStateChange])
  const corpora = useQuery({
    queryKey: ["evals", "corpora", targetKey, cursor],
    queryFn: ({ signal }) => fetchEvalCorpora({ target_key: targetKey, limit: 10, cursor }, signal),
  })
  const run = async (operation: (signal: AbortSignal) => Promise<void>) => {
    if (pending) return
    const current = new AbortController()
    controller.current = current
    setPending(true)
    setError(undefined)
    setNotice(undefined)
    try {
      await operation(current.signal)
    } catch (failure) {
      if (!current.signal.aborted)
        setError(evalErrorMessage(failure, "The corpus operation failed."))
    } finally {
      if (controller.current === current) {
        controller.current = null
        setPending(false)
      }
    }
  }
  const importFile = (file: File) =>
    void run(async (signal) => {
      await preflightEvalCorpusFile(file)
      if (signal.aborted) return
      const entry = await importEvalCorpus(file, signal)
      if (signal.aborted) return
      setNotice(`Corpus imported for ${entry.target_key}.`)
      setCursor(undefined)
      await client.invalidateQueries({ queryKey: ["evals"] })
    })
  return (
    <Section title="Import / export">
      <p className="mb-4 text-sm text-muted-foreground">
        Move suites between agents or into CI as a portable corpus. Downloads preserve the exact
        stored revision.
      </p>
      <Button
        variant="outline"
        size="sm"
        disabled={disabled || pending}
        onClick={() => input.current?.click()}
      >
        <Upload />
        {pending ? "Working…" : "Import corpus"}
      </Button>
      <input
        ref={input}
        type="file"
        accept="application/json,.json"
        className="sr-only"
        aria-label="Import corpus file"
        disabled={disabled || pending}
        onChange={(event) => {
          const file = event.target.files?.[0]
          event.target.value = ""
          if (file && !disabled) importFile(file)
        }}
      />
      <p className="mt-2 text-xs text-muted-foreground">
        JSON corpus, up to 8 MiB. Importing adds an immutable revision; it does not run evaluations.
      </p>
      {error && (
        <p role="alert" className="mt-3 text-sm text-destructive break-words">
          {error}
        </p>
      )}
      {notice && (
        <p role="status" className="mt-3 text-sm">
          {notice}{" "}
          <NavLink to="/evals" className="underline">
            Open suites
          </NavLink>
        </p>
      )}
      <h3 className="mt-6 mb-3 text-sm font-medium">Saved corpora</h3>
      <LoadState error={corpora.error} pending={corpora.isPending} retry={corpora.refetch} />
      <ul className="divide-y divide-border rounded-xl border border-border">
        {corpora.data?.items.map((entry) => (
          <li
            key={entry.revision}
            className="flex min-w-0 flex-wrap items-center gap-3 p-3 text-sm"
          >
            <div className="min-w-0 flex-1 break-words">
              <p>
                {entry.suite_count} suites · {entry.case_count} cases
              </p>
              <p className="mt-1 text-xs text-muted-foreground">
                {shortEvalIdentity(entry.revision)} · {formatDateTime(entry.created_at)} ·{" "}
                {formatBytes(entry.document_bytes)}
              </p>
            </div>
            <Button
              size="sm"
              variant="outline"
              disabled={pending}
              aria-label={`Download corpus ${shortEvalIdentity(entry.revision)}`}
              onClick={() =>
                void run(async (signal) => {
                  const file = await downloadEvalCorpus(entry.revision, signal)
                  if (!signal.aborted) downloadBlob(file.blob, file.filename)
                })
              }
            >
              <Download />
              Download
            </Button>
          </li>
        ))}
      </ul>
      {corpora.data?.items.length === 0 && (
        <p className="mt-3 text-sm text-muted-foreground">No saved corpora on this page.</p>
      )}
      <fieldset disabled={pending}>
        <Pages
          label="Corpora"
          cursor={cursor}
          hasMore={!!corpora.data?.has_more}
          busy={corpora.isFetching}
          onFirst={() => setCursor(undefined)}
          onNext={() => setCursor(corpora.data?.next_cursor ?? undefined)}
        />
      </fieldset>
    </Section>
  )
}
