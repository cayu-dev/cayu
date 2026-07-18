import { useQueryClient } from "@tanstack/react-query"
import { Braces, LoaderCircle, Pencil, Tags } from "lucide-react"
import { useEffect, useMemo, useRef, useState } from "react"
import { type SessionDetail, updateSessionLabels, updateSessionMetadata } from "../../lib/api.ts"
import {
  applyConfirmedSessionEdit,
  formatSessionEditDraft,
  parseSessionLabelsDraft,
  parseSessionMetadataDraft,
  sessionEditErrorMessage,
  splitSessionMetadata,
} from "../../lib/session-editing.ts"
import { Badge } from "../ui/badge"
import { Button } from "../ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "../ui/card"
import { Textarea } from "../ui/textarea"
import { PayloadViewer } from "./layout"

type EditorState = {
  draft: string
  error: string | null
}

export function SessionAnnotations({ session }: { session: SessionDetail }) {
  const queryClient = useQueryClient()
  const [labelsEditor, setLabelsEditor] = useState<EditorState | null>(null)
  const [metadataEditor, setMetadataEditor] = useState<EditorState | null>(null)
  const [labelsSaving, setLabelsSaving] = useState(false)
  const [metadataSaving, setMetadataSaving] = useState(false)
  const [announcement, setAnnouncement] = useState("")
  const saveInFlightRef = useRef(false)
  const labelsEditButtonRef = useRef<HTMLButtonElement>(null)
  const metadataEditButtonRef = useRef<HTMLButtonElement>(null)
  const labelsWasEditingRef = useRef(false)
  const metadataWasEditingRef = useRef(false)
  const { userMetadata, runtimeMetadata } = useMemo(
    () => splitSessionMetadata(session.metadata),
    [session.metadata],
  )

  useEffect(() => {
    if (labelsEditor !== null) {
      labelsWasEditingRef.current = true
    } else if (labelsWasEditingRef.current) {
      labelsWasEditingRef.current = false
      labelsEditButtonRef.current?.focus()
    }
  }, [labelsEditor])
  useEffect(() => {
    if (metadataEditor !== null) {
      metadataWasEditingRef.current = true
    } else if (metadataWasEditingRef.current) {
      metadataWasEditingRef.current = false
      metadataEditButtonRef.current?.focus()
    }
  }, [metadataEditor])

  const savePending = labelsSaving || metadataSaving

  const beginLabelsEdit = () => {
    setAnnouncement("")
    setLabelsEditor({ draft: formatSessionEditDraft(session.labels), error: null })
  }
  const beginMetadataEdit = () => {
    setAnnouncement("")
    setMetadataEditor({ draft: formatSessionEditDraft(userMetadata), error: null })
  }
  const saveLabels = async () => {
    if (labelsEditor === null || savePending || saveInFlightRef.current) return
    const parsed = parseSessionLabelsDraft(labelsEditor.draft)
    if (!parsed.ok) {
      setLabelsEditor((current) =>
        current === null ? null : { ...current, error: parsed.message },
      )
      return
    }
    setLabelsEditor((current) => (current === null ? null : { ...current, error: null }))
    saveInFlightRef.current = true
    setLabelsSaving(true)
    try {
      const updated = await updateSessionLabels(session.id, { labels: parsed.value })
      applyConfirmedSessionEdit(queryClient, updated)
      setLabelsEditor(null)
      setAnnouncement("Session labels saved.")
    } catch (error) {
      setLabelsEditor((current) =>
        current === null ? null : { ...current, error: sessionEditErrorMessage(error, "labels") },
      )
    } finally {
      setLabelsSaving(false)
      saveInFlightRef.current = false
    }
  }
  const saveMetadata = async () => {
    if (metadataEditor === null || savePending || saveInFlightRef.current) return
    const parsed = parseSessionMetadataDraft(metadataEditor.draft)
    if (!parsed.ok) {
      setMetadataEditor((current) =>
        current === null ? null : { ...current, error: parsed.message },
      )
      return
    }
    setMetadataEditor((current) => (current === null ? null : { ...current, error: null }))
    saveInFlightRef.current = true
    setMetadataSaving(true)
    try {
      const updated = await updateSessionMetadata(session.id, { metadata: parsed.value })
      applyConfirmedSessionEdit(queryClient, updated)
      setMetadataEditor(null)
      setAnnouncement("Session metadata saved.")
    } catch (error) {
      setMetadataEditor((current) =>
        current === null ? null : { ...current, error: sessionEditErrorMessage(error, "metadata") },
      )
    } finally {
      setMetadataSaving(false)
      saveInFlightRef.current = false
    }
  }
  const sortedLabels = Object.entries(session.labels).sort(([left], [right]) =>
    left.localeCompare(right),
  )

  return (
    <Card className="gap-0 py-0" data-testid="session-annotations">
      <CardHeader className="border-b border-border py-4">
        <CardTitle className="text-base">Labels and metadata</CardTitle>
        <p className="mt-1 text-xs text-muted-foreground">
          Operator-editable session context. Runtime history and lifecycle state are unchanged.
        </p>
        <p className="sr-only" role="status" aria-live="polite">
          {announcement}
        </p>
      </CardHeader>
      <CardContent className="grid grid-cols-1 p-0 xl:grid-cols-2">
        <section className="min-w-0 border-b border-border p-4 xl:border-r xl:border-b-0">
          <div className="flex items-start justify-between gap-3">
            <div className="min-w-0">
              <h2
                className="flex items-center gap-2 text-sm font-semibold"
                id="session-labels-title"
              >
                <Tags className="h-4 w-4 text-primary" />
                Labels
              </h2>
              <p className="mt-1 text-xs text-muted-foreground">
                Indexed key/value labels used by session discovery.
              </p>
            </div>
            {labelsEditor === null && (
              <Button
                ref={labelsEditButtonRef}
                variant="outline"
                size="sm"
                onClick={beginLabelsEdit}
              >
                <Pencil className="mr-1.5 h-3.5 w-3.5" />
                Edit labels
              </Button>
            )}
          </div>

          {labelsEditor === null ? (
            sortedLabels.length === 0 ? (
              <p className="mt-4 text-sm text-muted-foreground">No labels.</p>
            ) : (
              <div className="mt-4 flex flex-wrap gap-2">
                {sortedLabels.map(([key, value]) => (
                  <Badge key={key} variant="outline" className="max-w-full gap-1 font-mono">
                    <span className="truncate font-semibold">{key}</span>
                    <span aria-hidden="true">=</span>
                    <span className="truncate text-muted-foreground">{value}</span>
                  </Badge>
                ))}
              </div>
            )
          ) : (
            <div className="mt-4 space-y-3">
              <div>
                <label htmlFor="session-labels-editor" className="sr-only">
                  Session labels JSON
                </label>
                <Textarea
                  id="session-labels-editor"
                  value={labelsEditor.draft}
                  onChange={(event) => setLabelsEditor({ draft: event.target.value, error: null })}
                  onKeyDown={(event) => {
                    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
                      event.preventDefault()
                      void saveLabels()
                    } else if (event.key === "Escape" && !labelsSaving) {
                      setLabelsEditor(null)
                    }
                  }}
                  rows={10}
                  spellCheck={false}
                  autoFocus
                  disabled={labelsSaving}
                  aria-invalid={labelsEditor.error !== null}
                  aria-describedby={
                    labelsEditor.error === null
                      ? "session-labels-editor-help"
                      : "session-labels-editor-help session-labels-editor-error"
                  }
                  className="min-h-44 max-h-80 font-mono text-xs"
                />
                <p id="session-labels-editor-help" className="mt-2 text-xs text-muted-foreground">
                  Saving replaces the complete label map. Use <code>{"{}"}</code> to remove all
                  labels. Press Ctrl/Command+Enter to save or Escape to cancel.
                </p>
              </div>
              {labelsEditor.error !== null && (
                <p
                  id="session-labels-editor-error"
                  role="alert"
                  className="text-sm text-destructive"
                >
                  {labelsEditor.error}
                </p>
              )}
              <div className="flex justify-end gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  aria-label="Cancel label editing"
                  disabled={labelsSaving}
                  onClick={() => setLabelsEditor(null)}
                >
                  Cancel
                </Button>
                <Button size="sm" disabled={savePending} onClick={() => void saveLabels()}>
                  {labelsSaving && <LoaderCircle className="mr-1.5 h-3.5 w-3.5 animate-spin" />}
                  {labelsSaving ? "Saving labels..." : "Save labels"}
                </Button>
              </div>
            </div>
          )}
        </section>

        <section className="min-w-0 p-4">
          <div className="flex items-start justify-between gap-3">
            <div className="min-w-0">
              <h2
                className="flex items-center gap-2 text-sm font-semibold"
                id="session-metadata-title"
              >
                <Braces className="h-4 w-4 text-primary" />
                Metadata
              </h2>
              <p className="mt-1 text-xs text-muted-foreground">
                Complete user-authored JSON metadata for this session.
              </p>
            </div>
            {metadataEditor === null && (
              <Button
                ref={metadataEditButtonRef}
                variant="outline"
                size="sm"
                onClick={beginMetadataEdit}
              >
                <Pencil className="mr-1.5 h-3.5 w-3.5" />
                Edit metadata
              </Button>
            )}
          </div>

          {metadataEditor === null ? (
            Object.keys(userMetadata).length === 0 ? (
              <p className="mt-4 text-sm text-muted-foreground">No user metadata.</p>
            ) : (
              <PayloadViewer value={userMetadata} className="mt-4" maxHeight="max-h-80" />
            )
          ) : (
            <div className="mt-4 space-y-3">
              <div>
                <label htmlFor="session-metadata-editor" className="sr-only">
                  Session metadata JSON
                </label>
                <Textarea
                  id="session-metadata-editor"
                  value={metadataEditor.draft}
                  onChange={(event) =>
                    setMetadataEditor({ draft: event.target.value, error: null })
                  }
                  onKeyDown={(event) => {
                    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
                      event.preventDefault()
                      void saveMetadata()
                    } else if (event.key === "Escape" && !metadataSaving) {
                      setMetadataEditor(null)
                    }
                  }}
                  rows={10}
                  spellCheck={false}
                  autoFocus
                  disabled={metadataSaving}
                  aria-invalid={metadataEditor.error !== null}
                  aria-describedby={
                    metadataEditor.error === null
                      ? "session-metadata-editor-help"
                      : "session-metadata-editor-help session-metadata-editor-error"
                  }
                  className="min-h-44 max-h-80 font-mono text-xs"
                />
                <p id="session-metadata-editor-help" className="mt-2 text-xs text-muted-foreground">
                  Saving replaces all user-authored metadata. Cayu-managed entries remain read-only.
                  Use <code>{"{}"}</code> to clear user metadata. Press Ctrl/Command+Enter to save
                  or Escape to cancel.
                </p>
              </div>
              {metadataEditor.error !== null && (
                <p
                  id="session-metadata-editor-error"
                  role="alert"
                  className="text-sm text-destructive"
                >
                  {metadataEditor.error}
                </p>
              )}
              <div className="flex justify-end gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  aria-label="Cancel metadata editing"
                  disabled={metadataSaving}
                  onClick={() => setMetadataEditor(null)}
                >
                  Cancel
                </Button>
                <Button size="sm" disabled={savePending} onClick={() => void saveMetadata()}>
                  {metadataSaving && <LoaderCircle className="mr-1.5 h-3.5 w-3.5 animate-spin" />}
                  {metadataSaving ? "Saving metadata..." : "Save metadata"}
                </Button>
              </div>
            </div>
          )}

          {Object.keys(runtimeMetadata).length > 0 && (
            <div className="mt-4 border-t border-border pt-4">
              <h3 className="text-xs font-semibold">Runtime-managed metadata</h3>
              <p className="mt-1 text-xs text-muted-foreground">
                Read-only Cayu state used by runtime policy and workflow coordination.
              </p>
              <PayloadViewer value={runtimeMetadata} className="mt-3" maxHeight="max-h-56" />
            </div>
          )}
        </section>
      </CardContent>
    </Card>
  )
}
