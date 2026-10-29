/** The preview drawer for one artifact, opened by `?artifact=<id>` on the Artifacts page. */
import { useQuery } from "@tanstack/react-query"
import { Download, ExternalLink } from "lucide-react"
import { buttonVariants } from "@/components/ui/button"
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet"
import {
  type ArtifactRead,
  type ArtifactSummary,
  artifactContentUrl,
  fetchArtifact,
} from "@/lib/api"
import { formatBytes } from "@/lib/format"
import type { SessionOverview } from "../../api/types"
import { useApi, useNow, useServices } from "../../app/services"
import { when } from "../../app/task-model"
import { ErrorState, NavLink, StateBadge, sessionRoute } from "../common"
import { environmentRoute } from "../environments/environment-model"
import {
  imageUrl,
  jsonText,
  kindOf,
  opensInline,
  type PreviewKind,
  sessionTitle,
} from "./artifact-model"
import { DebugDetails } from "./debug"

/** Enough for a text file or a chart; bigger files say the preview is cut short. */
const PREVIEW_BYTES = 128 * 1024

export function ArtifactDrawer({
  artifactId,
  storeId,
  known,
  onClose,
}: {
  artifactId: string
  /** The store from the link; artifact IDs are only unique within a store. */
  storeId: string | undefined
  /** The artifact's summary when the list already holds it (it names the store to read). */
  known: ArtifactSummary | undefined
  onClose: () => void
}) {
  const api = useApi()
  const { overviewQueriesEnabled = true } = useServices()
  const now = useNow()
  // A file with no preview (an archive, a PDF) only needs its summary, which the list has.
  const read = useQuery({
    enabled: !known || kindOf(known).preview !== "none",
    queryKey: ["artifact-preview", artifactId, known?.artifact_store_id ?? storeId ?? null],
    queryFn: () =>
      fetchArtifact(artifactId, {
        artifact_store_id: known?.artifact_store_id ?? storeId,
        max_bytes: PREVIEW_BYTES,
      }),
  })
  const artifact = known ?? read.data?.artifact
  const sessionId = artifact?.session_id
  const overview = useQuery({
    queryKey: ["overview", sessionId],
    enabled: overviewQueriesEnabled && !!sessionId,
    queryFn: async (): Promise<SessionOverview | null> =>
      (await api.overviews([sessionId!]))[0] ?? null,
  })
  const kind = artifact ? kindOf(artifact) : null
  const content = artifact
    ? (disposition: "attachment" | "inline") =>
        artifactContentUrl(artifact.id, {
          artifact_store_id: artifact.artifact_store_id,
          disposition,
        })
    : null

  return (
    <Sheet
      open
      onOpenChange={(open) => {
        if (!open) onClose()
      }}
    >
      <SheetContent className="overflow-y-auto data-[side=right]:w-[min(100vw,44rem)] data-[side=right]:sm:max-w-none">
        <SheetHeader className="pr-12">
          <SheetTitle className="break-words">{artifact?.filename ?? "File"}</SheetTitle>
          <SheetDescription>
            {artifact && kind
              ? `${kind.label} · ${formatBytes(artifact.size_bytes)} · saved ${when(Date.parse(artifact.created_at), now())}`
              : read.isPending
                ? "Loading…"
                : ""}
          </SheetDescription>
        </SheetHeader>
        <div className="space-y-5 px-4 pb-6">
          {read.error && !artifact && <ErrorState error={read.error} />}
          {artifact && content && (
            <div className="flex flex-wrap gap-2">
              <a
                href={content("attachment")}
                download={artifact.filename}
                className={buttonVariants({ size: "sm" })}
              >
                <Download aria-hidden="true" />
                Download
              </a>
              {opensInline(artifact) && (
                <a
                  href={content("inline")}
                  target="_blank"
                  rel="noreferrer"
                  className={buttonVariants({ variant: "outline", size: "sm" })}
                >
                  <ExternalLink aria-hidden="true" />
                  Open raw
                </a>
              )}
            </div>
          )}

          {artifact && (
            <dl className="rounded-xl border border-border px-3 py-1 text-sm">
              <div className="grid grid-cols-[minmax(0,7rem)_minmax(0,1fr)] gap-3 border-b border-border py-2">
                <dt className="text-muted-foreground">
                  {artifact.session_id ? "Session" : "Saved for"}
                </dt>
                <dd className="min-w-0">
                  {artifact.session_id ? (
                    <span className="flex min-w-0 flex-wrap items-center gap-2">
                      <NavLink
                        {...sessionRoute(artifact.session_id)}
                        className="min-w-0 [overflow-wrap:anywhere] underline-offset-4 hover:underline"
                      >
                        {overview.isFetching
                          ? "Loading…"
                          : sessionTitle(overviewQueriesEnabled ? overview.data : null)}
                      </NavLink>
                      {overviewQueriesEnabled && overview.data && (
                        <StateBadge overview={overview.data} />
                      )}
                    </span>
                  ) : (
                    "Every session in the environment (not one session)"
                  )}
                </dd>
              </div>
              <div className="grid grid-cols-[minmax(0,7rem)_minmax(0,1fr)] gap-3 border-b border-border py-2">
                <dt className="text-muted-foreground">Specialist</dt>
                <dd>{artifact.agent_name ?? "–"}</dd>
              </div>
              <div className="grid grid-cols-[minmax(0,7rem)_minmax(0,1fr)] gap-3 py-2">
                <dt className="text-muted-foreground">Environment</dt>
                <dd>
                  {artifact.environment_name ? (
                    <NavLink
                      {...environmentRoute(artifact.environment_name)}
                      className="underline-offset-4 hover:underline"
                    >
                      {artifact.environment_name}
                    </NavLink>
                  ) : (
                    "–"
                  )}
                </dd>
              </div>
            </dl>
          )}

          {artifact && kind && (
            <section>
              <h3 className="mb-2 text-sm font-medium">Preview</h3>
              {kind.preview === "none" ? (
                <NoPreview />
              ) : read.isPending ? (
                <p role="status" className="text-sm text-muted-foreground">
                  Loading preview…
                </p>
              ) : read.error ? (
                <ErrorState error={read.error} />
              ) : read.data ? (
                <Preview read={read.data} preview={kind.preview} />
              ) : null}
            </section>
          )}

          {artifact && (
            <DebugDetails
              rows={[
                ["Artifact ID", artifact.id],
                ["Store ID", artifact.artifact_store_id],
                ["Scope", artifact.scope],
                ["Content type", artifact.content_type],
                ["Session ID", artifact.session_id],
                ["Size", `${artifact.size_bytes} bytes`],
              ]}
              json={[["Metadata", artifact.metadata]]}
            />
          )}
        </div>
      </SheetContent>
    </Sheet>
  )
}

function Preview({ read, preview }: { read: ArtifactRead; preview: PreviewKind }) {
  const cut = read.truncated && (
    <p className="mt-2 text-xs text-muted-foreground">
      Showing the first {formatBytes(read.preview_base64.length * 0.75)} of{" "}
      {formatBytes(read.total_bytes)}. Download the file to see all of it.
    </p>
  )
  const box = "max-h-[28rem] overflow-auto rounded-lg border border-border bg-muted/40 p-3"
  if (preview === "image") {
    const src = imageUrl(read)
    if (src)
      return (
        <div className="rounded-lg border border-border bg-muted/40 p-3">
          <img src={src} alt={read.artifact.filename} className="mx-auto max-h-96 max-w-full" />
        </div>
      )
    return (
      <p className="rounded-lg border border-dashed border-border px-3 py-6 text-center text-sm text-muted-foreground">
        This image is too large to preview. Download it to see it.
      </p>
    )
  }
  if ((preview === "json" || preview === "text") && read.text_preview !== null)
    return (
      <>
        <pre className={`${box} whitespace-pre-wrap break-words font-mono text-xs leading-5`}>
          {preview === "json" ? jsonText(read) : read.text_preview}
        </pre>
        {cut}
      </>
    )
  return <NoPreview />
}

function NoPreview() {
  return (
    <p className="rounded-lg border border-dashed border-border px-3 py-6 text-center text-sm text-muted-foreground">
      No preview for this kind of file. Download it to open it.
    </p>
  )
}
