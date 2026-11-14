import { requestJson } from "@/lib/api"

/** JSON over the dashboard's shared request path (credentials, ETags, error envelopes). */
export function json<T>(path: string, init: RequestInit = {}): Promise<T> {
  return requestJson<T>(path, init)
}
