/** Time as the parts show it, in the viewer's own time zone. */

const viewerTimeZone = () => Intl.DateTimeFormat().resolvedOptions().timeZone
const dayKey = (ms: number, timeZone = viewerTimeZone()) =>
  new Date(ms).toLocaleDateString("en-CA", { timeZone })

export function relativeTime(ms: number, now: number, timeZone = viewerTimeZone()): string {
  const s = Math.max(0, (now - ms) / 1000)
  if (s < 60) return "now"
  if (s < 3600) return `${Math.round(s / 60)}m`
  if (dayKey(ms, timeZone) === dayKey(now, timeZone)) return `${Math.round(s / 3600)}h`
  if (dayKey(ms, timeZone) === dayKey(now - 864e5, timeZone)) return "Yesterday"
  return new Date(ms).toLocaleDateString("en-US", { timeZone, month: "short", day: "numeric" })
}
