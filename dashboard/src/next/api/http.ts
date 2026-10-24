import { throwResponseError } from "@/lib/api"
import { apiUrl } from "@/lib/config"

export async function json<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(apiUrl(path), {
    ...init,
    headers: { "Content-Type": "application/json", ...init.headers },
  })
  if (!response.ok) await throwResponseError(response)
  return (await response.json()) as T
}
