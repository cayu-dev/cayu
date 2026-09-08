export type BrowserInputReceipt = {
  revision: number
  control_epoch: number
  settled_input_sequence: number
}

function failed() {
  return new Error("Private input did not settle. Inspect control state; do not resend it.")
}

export function encodePrivateBrowserText(value: string): Uint8Array<ArrayBuffer> {
  if (
    !value ||
    Array.from(value).length > 4096 ||
    Array.from(value).some((character) => {
      const point = character.codePointAt(0) ?? 0
      return point === 0 || (point >= 0xd800 && point <= 0xdfff)
    })
  )
    throw failed()
  return new TextEncoder().encode(value)
}

// Takes ownership of payload. It is erased after one send or any terminal path.
// Cancellation closes local waiting; it never claims that native input aborted.
export function startPrivateBrowserInput(
  socket: WebSocket,
  ticket: string,
  payload: Uint8Array<ArrayBuffer>,
  expected: BrowserInputReceipt,
) {
  const authority = {
    revision: expected.revision,
    control_epoch: expected.control_epoch,
    settled_input_sequence: expected.settled_input_sequence,
  }
  let phase: "opening" | "ready" | "settlement" | "closed" = "opening"
  let resolve: () => void
  let reject: (error: Error) => void
  const settled = new Promise<void>((yes, no) => {
    resolve = yes
    reject = no
  })
  const timer = setTimeout(() => finish(false), 30_000)

  function finish(success: boolean) {
    if (phase === "closed") return
    phase = "closed"
    ticket = ""
    payload.fill(0)
    clearTimeout(timer)
    socket.onopen = null
    socket.onmessage = null
    socket.onerror = null
    socket.onclose = null
    try {
      socket.close()
    } catch {
      /* No transport exception may reflect input. */
    }
    if (success) resolve()
    else reject(failed())
  }

  if (
    !(payload.buffer instanceof ArrayBuffer) ||
    payload.length < 1 ||
    payload.length > 16384 ||
    !Object.values(authority).every((value) => Number.isSafeInteger(value) && value >= 1)
  ) {
    finish(false)
    return { settled, stop: () => finish(false) }
  }

  socket.onopen = () => {
    if (phase !== "opening") {
      finish(false)
      return
    }
    phase = "ready"
    try {
      socket.send(ticket)
      ticket = ""
    } catch {
      finish(false)
    }
  }
  socket.onerror = () => finish(false)
  socket.onclose = () => finish(false)
  socket.onmessage = (event: MessageEvent) => {
    try {
      if (phase === "ready" && event.data === "ready") {
        // Advance before send: duplicate ready can never cause a second input.
        phase = "settlement"
        socket.send(payload)
        payload.fill(0)
        return
      }
      if (phase !== "settlement" || typeof event.data !== "string" || event.data.length > 1024) {
        finish(false)
        return
      }
      const receipt: unknown = JSON.parse(event.data)
      if (!receipt || typeof receipt !== "object" || Array.isArray(receipt)) {
        finish(false)
        return
      }
      const value = receipt as Record<string, unknown>
      finish(
        Object.keys(value).length === 4 &&
          value.state === "settled" &&
          value.revision === authority.revision &&
          value.control_epoch === authority.control_epoch &&
          value.settled_input_sequence === authority.settled_input_sequence,
      )
    } catch {
      finish(false)
    }
  }
  return { settled, stop: () => finish(false) }
}
