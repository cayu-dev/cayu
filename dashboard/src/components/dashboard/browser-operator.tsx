import { useEffect, useId, useRef, useState } from "react"
import {
  type BrowserCheckpointConsent,
  type BrowserDescriptor,
  type BrowserInputKind,
  type BrowserPage,
  type BrowserPageLocation,
  createBrowserControlClient,
} from "../../lib/browser-control-client"
import { startBrowserPageObserver } from "../../lib/browser-page-observer"
import { encodePrivateBrowserText, startPrivateBrowserInput } from "../../lib/browser-private-input"
import { startPrivateBrowserViewer } from "../../lib/browser-private-viewer"
import { apiUrl } from "../../lib/config"
import { Button } from "../ui/button"

export function BrowserOperator({ sessionId }: { sessionId: string }) {
  const consentId = useId()
  const [checkpointConsent, setCheckpointConsent] = useState<BrowserCheckpointConsent>("undecided")
  const canvas = useRef<HTMLCanvasElement>(null)
  const privateText = useRef<HTMLInputElement>(null)
  const inputOwner = useRef<ReturnType<typeof startPrivateBrowserInput> | null>(null)
  const client = useRef<ReturnType<typeof createBrowserControlClient> | null>(null)
  const viewer = useRef<ReturnType<typeof startPrivateBrowserViewer> | null>(null)
  const generation = useRef(0)
  const running = useRef(false)
  const [busy, setBusy] = useState(false)
  const [browsers, setBrowsers] = useState<BrowserDescriptor[]>([])
  const [selected, setSelected] = useState<BrowserDescriptor | null>(null)
  const [pages, setPages] = useState<BrowserPage[]>([])
  const [locations, setLocations] = useState<BrowserPageLocation[]>([])
  const [locationStatus, setLocationStatus] = useState("Locations require page discovery.")
  const [inputPageId, setInputPageId] = useState("")
  const [activePageId, setActivePageId] = useState<string | null>(null)
  const [status, setStatus] = useState("Live viewing requires explicit application authorization.")

  function clear() {
    const target = canvas.current
    if (target) {
      // Resetting the backing store also discards the previous frame pixels.
      target.width = 0
      target.height = 0
    }
  }

  useEffect(() => {
    return () => {
      generation.current++
      if (privateText.current) privateText.current.value = ""
      inputOwner.current?.stop()
      inputOwner.current = null
      viewer.current?.stop()
      viewer.current = null
      client.current?.dispose()
      client.current = null
    }
  }, [])

  useEffect(() => {
    const owner = client.current
    if (!owner || !selected) return
    setLocationStatus("Observed origins refresh while idle for up to five minutes.")
    const observer = startBrowserPageObserver(
      () => owner.pages(selected),
      (result) => setLocations(result.locations),
      () => {
        setLocations([])
        setLocationStatus("Live locations are unavailable. Refresh browser discovery.")
      },
      () => !running.current && document.visibilityState === "visible",
    )
    return () => observer.stop()
  }, [selected])

  function stop() {
    generation.current++
    if (privateText.current) privateText.current.value = ""
    inputOwner.current?.stop()
    inputOwner.current = null
    viewer.current?.stop()
    viewer.current = null
    client.current?.dispose()
    client.current = null
    clear()
    setBrowsers([])
    setPages([])
    setLocations([])
    setSelected(null)
    setCheckpointConsent("undecided")
    setStatus("Viewer closed. Closing a viewer does not hand control back to the agent.")
  }

  async function perform(
    action: (owner: NonNullable<typeof client.current>, current: () => boolean) => Promise<void>,
  ) {
    if (running.current) return
    running.current = true
    setBusy(true)
    const epoch = generation.current
    const current = () => epoch === generation.current
    try {
      if (!client.current) {
        client.current = createBrowserControlClient(
          new URL(apiUrl("browser-control"), window.location.href),
          sessionId,
        )
      }
      await action(client.current, current)
    } catch {
      if (current()) {
        // Keep continuity for readback after an acknowledgement loss. Replacing
        // it here would discard ownership of an already accepted takeover. Keep
        // the viewer alive too: an accepted sensitive-entry transition may still
        // need its purge acknowledgement. Explicit close remains available.
        setStatus(
          "Browser request did not settle locally. Refresh control state before another action; do not repeat input.",
        )
      }
    } finally {
      running.current = false
      if (current()) setBusy(false)
    }
  }

  async function refresh(
    owner: NonNullable<typeof client.current>,
    current: () => boolean,
    previous: BrowserDescriptor,
  ) {
    const discovered = await owner.discover()
    if (!current()) return
    setBrowsers(discovered)
    const updated = discovered.find(
      (browser) => JSON.stringify(browser.identity) === JSON.stringify(previous.identity),
    )
    setSelected(updated ?? null)
    setPages([])
    setLocations([])
    setStatus(
      updated
        ? `Control state: ${updated.state}. ${updated.sensitive_entry ? "Sensitive entry is settled; capture is restricted." : updated.sensitive_entry_pending ? "Waiting for native capture and viewer purge settlement." : "Refresh page discovery before viewing or requesting takeover."}`
        : "The selected browser allocation is no longer available.",
    )
  }

  function submitText() {
    if (running.current || !selected || !privateText.current) return
    const page = pages.find((candidate) => candidate.page_id === inputPageId)
    if (!page || page.page_id !== activePageId) return
    let payload: Uint8Array<ArrayBuffer>
    try {
      payload = encodePrivateBrowserText(privateText.current.value)
    } catch {
      setStatus("Private input must contain 1–4096 valid Unicode characters without NUL.")
      return
    } finally {
      privateText.current.value = ""
    }
    sendPayload(payload, "text")
  }

  function sendPayload(payload: Uint8Array<ArrayBuffer>, kind: BrowserInputKind) {
    const page = pages.find((candidate) => candidate.page_id === inputPageId)
    if (running.current || !selected || !page || page.page_id !== activePageId) {
      payload.fill(0)
      return
    }
    if (privateText.current) privateText.current.value = ""
    void perform(async (owner, current) => {
      try {
        const grant = await owner.inputTicket(selected, page, kind)
        if (!current()) {
          grant.ticket = ""
          return
        }
        const socket = new WebSocket(owner.inputUrl(), "cayu.browser-input.v1")
        inputOwner.current = startPrivateBrowserInput(socket, grant.ticket, payload, grant.expected)
        grant.ticket = ""
        await inputOwner.current.settled
        await refresh(owner, current, selected)
      } finally {
        payload.fill(0)
        inputOwner.current = null
      }
    })
  }

  return (
    <section
      aria-label="Private browser operator"
      className="rounded-lg border border-border p-4 space-y-3"
    >
      <h2 className="font-semibold">Private browser view</h2>
      <p className="text-sm text-muted-foreground" role="status">
        {status}
      </p>
      <div className="flex flex-wrap gap-2">
        <Button
          disabled={busy}
          variant="outline"
          onClick={() =>
            void perform(async (owner, current) => {
              await viewer.current?.retire()
              if (!current()) return
              viewer.current = null
              clear()
              const discovered = await owner.discover()
              if (!current()) return
              setBrowsers(discovered)
              setSelected(null)
              setPages([])
              setLocations([])
              setStatus(
                discovered.length
                  ? "Select an authorized browser and page."
                  : "No authorized live browsers.",
              )
            })
          }
        >
          Discover browsers
        </Button>
        <Button
          variant="outline"
          disabled={busy}
          onClick={() =>
            void perform(async () => {
              await viewer.current?.retire()
              stop()
              setBusy(false)
            })
          }
        >
          Close private view
        </Button>
      </div>
      <div className="flex flex-wrap gap-2">
        {browsers.map((browser, index) => (
          <Button
            key={browser.identity.browser_session_id}
            disabled={busy}
            variant="outline"
            onClick={() =>
              void perform(async (owner, current) => {
                await viewer.current?.retire()
                if (!current()) return
                viewer.current = null
                clear()
                const discovered = await owner.pages(browser)
                if (!current()) return
                setSelected(browser)
                setCheckpointConsent("undecided")
                setPages(discovered.pages)
                setLocations(discovered.locations)
                setActivePageId(discovered.active_page_id)
                setInputPageId(discovered.active_page_id ?? "")
                setStatus("Select a page to view. Viewing does not grant input authority.")
              })
            }
          >
            Browser {index + 1} · {browser.state}
          </Button>
        ))}
        {selected &&
          pages.map((page, index) => (
            <Button
              key={page.page_id}
              disabled={busy}
              variant="outline"
              onClick={() =>
                void perform(async (owner, current) => {
                  await viewer.current?.retire()
                  if (!current()) return
                  viewer.current = null
                  clear()
                  let ticket = await owner.viewerTicket(selected, page)
                  if (!current()) {
                    ticket = ""
                    return
                  }
                  const socket = new WebSocket(owner.viewerUrl(), "cayu.browser-view.v1")
                  viewer.current = startPrivateBrowserViewer(
                    socket,
                    ticket,
                    (bitmap) => {
                      if (!current()) return
                      const target = canvas.current
                      if (!target) return
                      target.width = bitmap.width
                      target.height = bitmap.height
                      target.getContext("2d")?.drawImage(bitmap, 0, 0)
                    },
                    clear,
                  )
                  ticket = ""
                  setStatus(
                    "Private view requested; frames remain transient. A blank canvas may mean capture is paused or access has ended.",
                  )
                })
              }
            >
              View page {index + 1}
            </Button>
          ))}
      </div>
      {selected && (
        <div className="space-y-2">
          <dl className="space-y-1 text-sm" aria-live="polite" aria-atomic="true">
            <div>
              <dt className="font-medium">Browser session</dt>
              <dd className="break-all font-mono" data-testid="selected-browser-identity">
                {selected.identity.browser_session_id}
              </dd>
            </div>
            <div>
              <dt className="font-medium">Environment</dt>
              <dd className="break-all" data-testid="selected-browser-environment">
                {selected.identity.environment_name}
              </dd>
            </div>
            <div>
              <dt className="font-medium">Allocation fingerprint</dt>
              <dd className="break-all font-mono" data-testid="selected-browser-allocation">
                {selected.identity.allocation_fingerprint}
              </dd>
            </div>
          </dl>
          <div className="text-sm" role="status" aria-live="polite" aria-atomic="true">
            <p>Application purpose: {selected.identity.operator_purpose.code}.</p>
            <p>Application-declared expected origins:</p>
            <ul className="list-inside list-disc">
              {selected.identity.operator_purpose.expected_origins.map((origin) => (
                <li key={origin}>{origin}</li>
              ))}
            </ul>
            <p className="text-muted-foreground">
              This declaration does not bypass destination policy or prove that the current page
              matches an expected origin.
            </p>
          </div>
          <div className="space-y-1 text-sm" role="status" aria-live="polite" aria-atomic="true">
            <p>{locationStatus}</p>
            {locations.map((location, index) => (
              <p key={location.page.page_id}>
                Page {index + 1} observed origin: {location.origin ?? "Unavailable or withheld"}.
              </p>
            ))}
            <p className="text-muted-foreground">
              Observed origins describe the last page discovery, not application instructions or
              proof of successful login.
            </p>
          </div>
          <p
            className="text-sm text-muted-foreground"
            role="status"
            aria-live="polite"
            aria-atomic="true"
          >
            Control: {selected.state}
            {selected.owned_request?.lease_until_ms != null &&
              ` · Current lease ends ${new Date(selected.owned_request.lease_until_ms).toLocaleTimeString()}`}
          </p>
          {selected.owned_request && (
            <p className="text-sm" role="status">
              Recorded checkpoint decision: {selected.owned_request.checkpoint_consent}.
            </p>
          )}
          <div className="space-y-1">
            <p className="text-sm">
              {selected.identity.profile_checkpoint_policy === "unavailable"
                ? "Profile checkpointing is unavailable for this browser."
                : selected.identity.profile_checkpoint_policy === "disabled"
                  ? "An application profile is configured, but checkpointing is disabled."
                  : selected.identity.profile_checkpoint_policy === "on_close"
                    ? "Application profile policy: checkpoint on close, subject to consent."
                    : "Application profile policy: checkpoint after terminal operations, subject to consent."}
            </p>
            <label htmlFor={consentId} className="block text-sm">
              Profile checkpoint consent for the next takeover
            </label>
            <select
              id={consentId}
              value={checkpointConsent}
              disabled={busy || selected.state !== "agent_controlled"}
              aria-describedby={`${consentId}-help`}
              className="rounded border bg-background p-2 text-sm"
              onChange={(event) => {
                const value = event.target.value
                if (value === "allow" || value === "deny" || value === "undecided")
                  setCheckpointConsent(value)
              }}
            >
              <option value="undecided">Not decided — do not save new state</option>
              <option value="deny">Do not save authenticated state</option>
              <option
                value="allow"
                disabled={
                  selected.identity.profile_checkpoint_policy === "unavailable" ||
                  selected.identity.profile_checkpoint_policy === "disabled"
                }
              >
                Allow configured profile checkpointing
              </option>
            </select>
            <p id={`${consentId}-help`} className="text-sm text-muted-foreground">
              Allow permits saving only through an application-configured profile after handback and
              a fresh observation. It does not enable profile storage or prove login succeeded. This
              decision is bound to the next takeover request.
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            <Button
              variant="outline"
              disabled={
                busy ||
                !selected.owned_request ||
                selected.state !== "operator_controlled" ||
                selected.owned_request.pending_lease_until_ms !== null ||
                selected.sensitive_entry_pending ||
                selected.owned_request.pending_input_sequence !== null
              }
              onClick={() =>
                void perform(async (owner, current) => {
                  await owner.renew(selected)
                  await refresh(owner, current, selected)
                })
              }
            >
              Renew current lease
            </Button>
            <Button
              variant="outline"
              disabled={busy}
              onClick={() => void perform((owner, current) => refresh(owner, current, selected))}
            >
              Refresh control state
            </Button>
            <Button
              variant="outline"
              disabled={busy || selected.state !== "agent_controlled" || pages.length === 0}
              onClick={() =>
                void perform(async (owner, current) => {
                  await owner.takeover(selected, pages, checkpointConsent)
                  await refresh(owner, current, selected)
                })
              }
            >
              Request exclusive takeover
            </Button>
            <Button
              variant="outline"
              disabled={
                busy ||
                !selected.owned_request ||
                selected.state !== "operator_controlled" ||
                selected.sensitive_entry ||
                selected.owned_request.pending_lease_until_ms !== null ||
                selected.sensitive_entry_pending
              }
              onClick={() =>
                void perform(async (owner, current) => {
                  // Do not close the viewer first: it must positively acknowledge
                  // purging outstanding frames before input can be admitted.
                  await owner.sensitiveEntry(selected)
                  await refresh(owner, current, selected)
                })
              }
            >
              Prepare sensitive entry
            </Button>
            <Button
              variant="outline"
              disabled={
                busy ||
                !selected.owned_request ||
                selected.state !== "operator_controlled" ||
                selected.sensitive_entry_pending ||
                selected.owned_request.pending_lease_until_ms !== null ||
                selected.owned_request.pending_input_sequence !== null
              }
              onClick={() =>
                void perform(async (owner, current) => {
                  await owner.handback(selected)
                  await refresh(owner, current, selected)
                })
              }
            >
              Return control to agent
            </Button>
          </div>
        </div>
      )}
      <canvas
        ref={canvas}
        width={0}
        height={0}
        aria-label="Private live browser frame"
        className="max-w-full"
      />
      {selected?.owned_request &&
        selected.state === "operator_controlled" &&
        selected.sensitive_entry &&
        !selected.sensitive_entry_pending && (
          <fieldset
            disabled={
              busy ||
              selected.owned_request.pending_input_sequence !== null ||
              selected.owned_request.pending_lease_until_ms !== null
            }
            className="space-y-2"
          >
            <legend className="text-sm font-medium">Private text input</legend>
            <div className="flex flex-wrap gap-2">
              {(
                [
                  ["tab", "Next field"],
                  ["backtab", "Previous field"],
                  ["enter", "Enter"],
                  ["escape", "Escape"],
                  ["backspace", "Backspace"],
                ] as const
              ).map(([kind, label]) => (
                <Button
                  key={kind}
                  variant="outline"
                  disabled={
                    !pages.some(
                      (page) => page.page_id === inputPageId && page.page_id === activePageId,
                    )
                  }
                  onClick={() => sendPayload(new TextEncoder().encode(kind), kind)}
                >
                  {label}
                </Button>
              ))}
            </div>
            <p className="text-sm text-muted-foreground">
              Text goes once to the selected page’s focused field. It is cleared locally before
              requesting transport. Refresh state after uncertainty; do not resend.
            </p>
            <label className="block text-sm">
              Target page
              <select
                value={inputPageId}
                onChange={(event) => setInputPageId(event.target.value)}
                className="ml-2 rounded border border-border bg-background p-2"
              >
                <option value="">Select a page</option>
                {pages.map((page, index) => (
                  <option
                    key={page.page_id}
                    value={page.page_id}
                    disabled={page.page_id !== activePageId}
                  >
                    Page {index + 1}
                  </option>
                ))}
              </select>
            </label>
            <label className="block text-sm">
              Private value
              <input
                ref={privateText}
                type="password"
                autoComplete="off"
                spellCheck={false}
                maxLength={8192}
                className="ml-2 rounded border border-border bg-background p-2"
              />
            </label>
            <Button
              variant="outline"
              disabled={
                !pages.some((page) => page.page_id === inputPageId && page.page_id === activePageId)
              }
              onClick={submitText}
            >
              Send private text once
            </Button>
          </fieldset>
        )}
    </section>
  )
}
