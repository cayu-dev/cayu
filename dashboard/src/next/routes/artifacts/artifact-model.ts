/** What kind of file an artifact is, how to preview it, and how to name its session. */
import {
  FileArchive,
  File as FileIcon,
  FileImage,
  FileJson,
  FileSpreadsheet,
  FileText,
} from "lucide-react"
import type { ArtifactRead, ArtifactSummary } from "@/lib/api"
import type { SessionOverview } from "../../api/types"

export type PreviewKind = "image" | "json" | "text" | "none"

export type ArtifactKind = { label: string; icon: typeof FileText; preview: PreviewKind }

const IMAGE_TYPES = new Set(["image/gif", "image/jpeg", "image/png", "image/webp"])
/** Types the browser can show in a new tab without running anything from the file. */
const INLINE_TYPES = new Set([
  ...IMAGE_TYPES,
  "application/json",
  "application/pdf",
  "text/csv",
  "text/markdown",
  "text/plain",
])

export const contentTypeOf = (artifact: ArtifactSummary) =>
  artifact.content_type.split(";", 1)[0]?.trim().toLowerCase() ?? artifact.content_type

export function kindOf(artifact: ArtifactSummary): ArtifactKind {
  const type = contentTypeOf(artifact)
  if (IMAGE_TYPES.has(type))
    return { label: `${type.slice(6).toUpperCase()} image`, icon: FileImage, preview: "image" }
  if (type.startsWith("image/")) return { label: "Image", icon: FileImage, preview: "none" }
  if (type === "application/json" || type.endsWith("+json"))
    return { label: "JSON", icon: FileJson, preview: "json" }
  if (type === "text/markdown") return { label: "Markdown", icon: FileText, preview: "text" }
  if (type === "text/csv") return { label: "CSV", icon: FileSpreadsheet, preview: "text" }
  if (type.startsWith("text/")) return { label: "Text", icon: FileText, preview: "text" }
  if (type === "application/pdf") return { label: "PDF", icon: FileText, preview: "none" }
  if (/zip|tar|gzip|x-7z|compressed/.test(type))
    return { label: "Archive", icon: FileArchive, preview: "none" }
  return { label: "File", icon: FileIcon, preview: "none" }
}

export const opensInline = (artifact: ArtifactSummary) => INLINE_TYPES.has(contentTypeOf(artifact))

/** A data URL for an image preview, only when the whole image was read. */
export function imageUrl(read: ArtifactRead): string | null {
  const type = contentTypeOf(read.artifact)
  if (!IMAGE_TYPES.has(type) || read.truncated) return null
  return `data:${type};base64,${read.preview_base64}`
}

/** Pretty-print JSON when the preview holds all of it; otherwise show it as read. */
export function jsonText(read: ArtifactRead): string {
  const raw = read.text_preview ?? ""
  if (read.truncated) return raw
  try {
    return JSON.stringify(JSON.parse(raw), null, 2)
  } catch {
    return raw
  }
}

export const sessionTitle = (overview: SessionOverview | null | undefined) =>
  overview?.title?.trim() || "Untitled session"
