/** Native navigation bypasses fetch. Give unchanged artifact links local seed bytes. */
import { findArtifact } from "./core"
import type { LegacyContext } from "./server"

export function installArtifactLinks(context: LegacyContext): () => void {
  const urls = new Set<string>()
  const rewrite = (anchor: HTMLAnchorElement) => {
    const url = new URL(anchor.href, window.location.href)
    if (url.origin !== window.location.origin) return
    const match = /^\/api\/artifacts\/([^/]+)\/content$/.exec(url.pathname)
    if (!match) return
    const { summary, content } = findArtifact(
      { params: { artifactId: decodeURIComponent(match[1]!) }, query: url.searchParams },
      context,
    )
    // Text files open as plain text, so the browser displays a copied URL or an Open raw
    // tab instead of downloading by MIME type. Images and archives keep their own type.
    const bytes = content()
    const type = typeof bytes === "string" ? "text/plain;charset=utf-8" : summary.content_type
    const blobUrl = URL.createObjectURL(new Blob([bytes], { type }))
    urls.add(blobUrl)
    anchor.href = blobUrl
    if (url.searchParams.get("disposition") === "attachment") anchor.download = summary.filename
  }
  const scan = (node: Node) => {
    if (node instanceof HTMLAnchorElement) rewrite(node)
    if (node instanceof Element)
      node.querySelectorAll<HTMLAnchorElement>("a[href]").forEach((a) => {
        rewrite(a)
      })
  }
  // Mutation callbacks run before the next user interaction, so context-menu,
  // modified clicks and copied link addresses all get the same local URL.
  const observer = new MutationObserver((changes) => {
    for (const change of changes) {
      if (change.type === "attributes") scan(change.target)
      else change.addedNodes.forEach(scan)
    }
  })
  observer.observe(document.documentElement, {
    childList: true,
    subtree: true,
    attributes: true,
    attributeFilter: ["href"],
  })
  scan(document.documentElement)
  // Keep URLs alive across route/seed changes: an already opened raw tab owns its
  // snapshot. Release them only when the prototype document or HMR owner goes away.
  const dispose = () => {
    observer.disconnect()
    urls.forEach((url) => {
      URL.revokeObjectURL(url)
    })
    urls.clear()
  }
  return dispose
}
