import { useEffect, useState } from "react"
import { apiUrl } from "../../lib/config"
import { useDashboardCapability } from "./server-contract"

type Recording = {
  recording_id: string
  browser_id: string
  status: string
  reason: string
  can_download: boolean
  video_sha256: string | null
  started_at_ms: number
  expires_at_ms: number
  gaps: { start_ms: number; end_ms: number; reason: string }[]
  segments: { sequence: number }[]
}

export function BrowserRecordings({ sessionId }: { sessionId: string }) {
  const { enabled } = useDashboardCapability({ kind: "surface", surface: "browser_recordings" })
  const [recordings, setRecordings] = useState<Recording[]>([])
  const [selected, setSelected] = useState("")
  const [error, setError] = useState("")
  useEffect(() => {
    const abort = new AbortController()
    setRecordings([])
    setSelected("")
    setError("")
    if (!enabled) return () => abort.abort()
    let timer: ReturnType<typeof setTimeout> | undefined
    let unavailable = false
    async function refresh() {
      try {
        const response = await fetch(
          apiUrl(`/browser-recordings/sessions/${encodeURIComponent(sessionId)}`),
          {
            credentials: "same-origin",
            cache: "no-store",
            signal: abort.signal,
          },
        )
        if (response.status === 404) {
          unavailable = true
          if (!abort.signal.aborted) setRecordings([])
          return
        }
        if (!response.ok) throw new Error("Browser recordings are unavailable.")
        const records: unknown = await response.json()
        if (!Array.isArray(records) || records.length > 1024)
          throw new Error("Invalid recording response.")
        for (const record of records) {
          if (
            !record ||
            typeof record !== "object" ||
            !/^[0-9a-f]{64}$/.test(record.recording_id) ||
            !Array.isArray(record.gaps) ||
            !Array.isArray(record.segments) ||
            typeof record.reason !== "string" ||
            typeof record.browser_id !== "string" ||
            !Number.isFinite(record.started_at_ms) ||
            !Number.isFinite(record.expires_at_ms)
          ) {
            throw new Error("Invalid recording response.")
          }
        }
        if (!abort.signal.aborted) {
          setRecordings(records as Recording[])
          setError("")
        }
      } catch {
        if (!abort.signal.aborted) setError("Browser recordings are unavailable.")
      } finally {
        if (!abort.signal.aborted && !unavailable) timer = setTimeout(refresh, 5000)
      }
    }
    void refresh()
    return () => {
      abort.abort()
      clearTimeout(timer)
    }
  }, [sessionId, enabled])
  if (!enabled) return null
  if (!recordings.length && !error) return null
  const recording = recordings.find((item) => item.recording_id === selected)
  const playable = recording?.video_sha256 && ["complete", "partial"].includes(recording.status)
  const source = recording
    ? apiUrl(`/browser-recordings/${recording.recording_id}/media`)
    : undefined
  return (
    <section
      className="space-y-3 rounded-lg border border-border p-4"
      aria-label="Browser recordings"
    >
      <div className="flex items-center justify-between gap-3">
        <h2 className="font-medium">Browser recordings</h2>
      </div>
      {error && (
        <p role="alert" className="text-sm text-destructive">
          {error}
        </p>
      )}
      <p className="text-sm text-muted-foreground">
        Application-authorized samples of the active page. Capture gaps are omitted from playback.
      </p>
      <label className="flex items-center gap-3 text-sm">
        Recording
        <select
          aria-label="Select browser recording"
          className="rounded border border-input bg-background p-2"
          value={selected}
          onChange={(event) => setSelected(event.target.value)}
        >
          <option value="">Select a recording</option>
          {recordings.map((item, index) => (
            <option key={item.recording_id} value={item.recording_id}>
              Recording {index + 1} · {new Date(item.started_at_ms).toLocaleString()} ·{" "}
              {item.status}
            </option>
          ))}
        </select>
      </label>
      {recording && (
        <div className="space-y-2 text-sm">
          <p>
            {recording.status} · {recording.reason.replaceAll("_", " ")} ·{" "}
            {recording.segments.length} samples
          </p>
          <p className="text-muted-foreground">
            Available until {new Date(recording.expires_at_ms).toLocaleString()}.
          </p>
          {playable && source && (
            <>
              {/* biome-ignore lint/a11y/useMediaCaption: Silent browser recording; gaps are described below. */}
              <video
                key={recording.recording_id}
                className="max-h-[36rem] w-full rounded bg-black"
                controls
                preload="none"
                src={source}
                controlsList="nodownload"
              />
              {recording.can_download && (
                <a className="inline-block underline" href={`${source}?download=true`}>
                  Download video
                </a>
              )}
            </>
          )}
          {recording.gaps.length > 0 && (
            <details>
              <summary>{recording.gaps.length} capture gaps</summary>
              <ul className="mt-2 max-h-40 list-disc overflow-auto pl-5">
                {recording.gaps.map((gap) => (
                  <li key={`${gap.start_ms}-${gap.end_ms}`}>
                    {(gap.start_ms / 1000).toFixed(1)}–{(gap.end_ms / 1000).toFixed(1)} seconds ·{" "}
                    {gap.reason.replaceAll("_", " ")}
                  </li>
                ))}
              </ul>
            </details>
          )}
        </div>
      )}
    </section>
  )
}
