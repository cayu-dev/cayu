// Transient frames stay out of query caches, object URLs, storage and telemetry.
export function startPrivateBrowserViewer(
  socket: WebSocket,
  ticket: string,
  draw: (bitmap: ImageBitmap) => void,
  clear: () => void,
  decode: (blob: Blob) => Promise<ImageBitmap> = createImageBitmap,
) {
  let closed = false
  let generation = 0
  let outstanding = false
  let ready = false
  let readyReceived = false
  let retiring = false
  let purgeAcknowledged = false
  let retired = false
  let retirement: Promise<void> | undefined
  let retirementTimer: ReturnType<typeof setTimeout> | undefined
  let resolveRetirement: (() => void) | undefined
  let rejectRetirement: ((error: Error) => void) | undefined
  let timer: ReturnType<typeof setTimeout> | undefined
  const decoding = new Set<Promise<void>>()

  function stop() {
    if (closed) return
    closed = true
    generation++
    ticket = ""
    clearTimeout(timer)
    clearTimeout(retirementTimer)
    clear()
    if (!retired) rejectRetirement?.(new Error("Private viewer retirement did not settle."))
    socket.close()
  }

  function retire(): Promise<void> {
    if (retired) return Promise.resolve()
    if (retirement) return retirement
    if (closed) return Promise.reject(new Error("Private viewer retirement did not settle."))
    retiring = true
    clearTimeout(timer)
    clear()
    retirement = new Promise<void>((resolve, reject) => {
      resolveRetirement = resolve
      rejectRetirement = reject
      retirementTimer = setTimeout(stop, 5000)
    })
    if (ready) socket.send("retire")
    return retirement
  }

  function requestFrame() {
    if (closed || retiring || !ready || outstanding) return
    outstanding = true
    socket.send("frame")
  }

  socket.binaryType = "arraybuffer"
  socket.onopen = () => {
    if (closed) return
    socket.send(ticket)
    ticket = ""
  }
  socket.onerror = stop
  socket.onclose = stop
  socket.onmessage = (event: MessageEvent) => {
    if (closed) return
    if (typeof event.data === "string") {
      if (event.data === "ready" && !readyReceived) {
        readyReceived = true
        ready = true
        if (retiring) socket.send("retire")
        else requestFrame()
        return
      }
      if (event.data === "retired" && purgeAcknowledged) {
        retired = true
        resolveRetirement?.()
        stop()
        return
      }
      if (/^purge:[a-f0-9]{32}$/.test(event.data) && ready) {
        const token = event.data.slice(6)
        generation++
        ready = false
        outstanding = false
        clearTimeout(timer)
        clear()
        // A decoding bitmap can outlive the message callback. Join its disposal
        // before promising that no old frame remains renderable.
        void Promise.allSettled([...decoding]).then(() => {
          clear()
          if (!closed) {
            purgeAcknowledged = true
            socket.send(`purged:${token}`)
          }
        })
        return
      }
      stop()
      return
    }
    if (
      !ready ||
      !outstanding ||
      !(event.data instanceof ArrayBuffer) ||
      event.data.byteLength < 24 ||
      event.data.byteLength > 2 * 1024 * 1024
    ) {
      stop()
      return
    }
    const bytes = new Uint8Array(event.data)
    const header = [137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82]
    const view = new DataView(event.data)
    const width = view.getUint32(16)
    const height = view.getUint32(20)
    if (
      header.some((value, index) => bytes[index] !== value) ||
      width < 1 ||
      width > 1920 ||
      height < 1 ||
      height > 1080
    ) {
      stop()
      return
    }
    outstanding = false
    const frameGeneration = generation
    const blob = new Blob([event.data], { type: "image/png" })
    const operation = Promise.resolve()
      .then(() => decode(blob))
      .then((bitmap) => {
        try {
          if (bitmap.width !== width || bitmap.height !== height) {
            stop()
            return
          }
          if (!closed && !retiring && ready && frameGeneration === generation) draw(bitmap)
        } finally {
          bitmap.close()
        }
      })
      .catch(stop)
      .finally(() => {
        decoding.delete(operation)
        if (!closed && !retiring && ready && frameGeneration === generation) {
          timer = setTimeout(requestFrame, 550)
        }
      })
    decoding.add(operation)
  }
  return { stop, retire }
}
