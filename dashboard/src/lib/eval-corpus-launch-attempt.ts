/** The corpus page sends only these public admission fields, never candidate evidence. */
export type CorpusLaunchAttempt = {
  key: string
  request: {
    corpus_revision: string
    suite_id: string
    max_concurrency: number
    expected_execution_profile_revision: string
  }
}
type AttemptStorage = Pick<Storage, "getItem" | "setItem" | "removeItem">
const storageKey = (api: string, suite: string) =>
  `cayu.corpus-launch.v1:${JSON.stringify([api, suite])}`
const isRevision = (value: unknown): value is string =>
  typeof value === "string" && /^sha256:[0-9a-f]{64}$/.test(value)

export function saveCorpusLaunchAttempt(
  storage: AttemptStorage,
  api: string,
  suite: string,
  attempt: CorpusLaunchAttempt,
) {
  const key = storageKey(api, suite)
  const text = JSON.stringify(attempt)
  const existing = storage.getItem(key)
  if (existing !== null && existing !== text)
    throw new Error("Resolve the previous launch before starting another one.")
  storage.setItem(key, text)
}

export function readCorpusLaunchAttempt(
  storage: AttemptStorage,
  api: string,
  suite: string,
): CorpusLaunchAttempt | null {
  const text = storage.getItem(storageKey(api, suite))
  if (text === null) return null
  try {
    if (text.length > 8192) throw new Error("oversized")
    const value = JSON.parse(text)
    const request = value.request
    if (
      !value ||
      typeof value.key !== "string" ||
      !/^cayu-dashboard-[0-9a-f-]{36}$/.test(value.key) ||
      !request ||
      Object.keys(request).sort().join(",") !==
        "corpus_revision,expected_execution_profile_revision,max_concurrency,suite_id" ||
      !isRevision(request.corpus_revision) ||
      !isRevision(request.expected_execution_profile_revision) ||
      typeof request.suite_id !== "string" ||
      !request.suite_id.trim() ||
      request.suite_id.length > 256 ||
      !Number.isSafeInteger(request.max_concurrency) ||
      request.max_concurrency < 1
    )
      throw new Error("invalid")
    return value as CorpusLaunchAttempt
  } catch {
    throw new Error(
      "The unresolved launch cannot be read. Keep this tab open and contact your operator before starting another run.",
    )
  }
}

/** A late completion cannot erase a different operation. */
export function clearCorpusLaunchAttempt(
  storage: AttemptStorage,
  api: string,
  suite: string,
  attempt: CorpusLaunchAttempt,
): boolean {
  const key = storageKey(api, suite)
  if (storage.getItem(key) !== JSON.stringify(attempt)) return false
  storage.removeItem(key)
  return true
}
