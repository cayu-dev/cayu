/** This browser tab's storage, or null where storage is missing or blocked. */
export function tabStorage(): Storage | null {
  try {
    return window.sessionStorage
  } catch {
    return null
  }
}
