import * as React from "react";
import { MutationCache, QueryCache, QueryClient, queryOptions } from "@tanstack/react-query";
import type { ActionResult } from "./patterns/action";
import type { ContentState } from "./patterns/async-content";

/*
 * Data access for an app's screens: one fetch wrapper with typed failures, the
 * QueryClient defaults, query state as table/content props, and the session.
 */

/** How a request failed. Screens branch on this, never on status codes. */
export type ApiErrorKind =
  | "network"
  | "server"
  | "unavailable"
  | "aborted"
  | "unauthorized"
  | "forbidden"
  | "not-found"
  | "conflict"
  | "rejected"
  | "validation"
  | "invalid-response"
  | "unknown-outcome";

export class ApiError extends Error {
  readonly kind: ApiErrorKind;
  readonly status?: number;
  readonly detail?: string;
  constructor(kind: ApiErrorKind, message: string, options: { status?: number; detail?: string; cause?: unknown } = {}) {
    super(message, { cause: options.cause });
    this.name = "ApiError";
    this.kind = kind;
    this.status = options.status;
    this.detail = options.detail;
  }
}

export const isApiError = (error: unknown, kind?: ApiErrorKind): error is ApiError =>
  error instanceof ApiError && (kind === undefined || error.kind === kind);

/** Failures before a definitive answer; a read may try again. */
export const isRetryableRead = (error: unknown) => isApiError(error) && ["network", "server", "unavailable"].includes(error.kind);

/** Anything with zod's `safeParse`. */
export type ResponseSchema<T> = { safeParse(value: unknown): { success: true; data: T } | { success: false; error: unknown } };

export type RequestOptions<T> = {
  method?: "GET" | "POST" | "PUT" | "PATCH" | "DELETE";
  /** Search params; empty values are left out. */
  query?: Record<string, string | number | boolean | null | undefined>;
  /** Sent as JSON. */
  body?: unknown;
  signal?: AbortSignal;
  /** Validates the response; omit it when the body isn't read. */
  schema?: ResponseSchema<T>;
  /** A read sent with another method, such as a query `POST`: its failures are ordinary
   * read failures, retried as reads are, never an unknown write outcome. */
  read?: boolean;
};

export type Requester = <T>(path: string, options?: RequestOptions<T>) => Promise<T>;

async function failure(response: Response, write: boolean): Promise<ApiError> {
  const status = response.status;
  let detail: string | undefined;
  try {
    const body = (await response.json()) as { detail?: unknown };
    if (typeof body?.detail === "string" && body.detail) detail = body.detail;
  } catch {
    // No JSON body; the default message stands.
  }
  const answer = (kind: ApiErrorKind, fallback: string) => new ApiError(kind, detail ?? fallback, { status, detail });
  if (status === 401) return new ApiError("unauthorized", "Your sign-in expired.", { status, detail });
  if (status === 403) return answer("forbidden", "You don't have access to this.");
  if (status === 404) return answer("not-found", "Not found.");
  if (status === 409) return answer("conflict", "This changed since it was loaded. Reload it, then try again.");
  if (status === 422) return answer("validation", "The request needs changes.");
  if (status >= 500 && write) return new ApiError("unknown-outcome", "The server failed while handling the request.", { status, detail });
  if (status === 503) return answer("unavailable", "The service is unavailable right now. Try again shortly.");
  if (status >= 500) return new ApiError("server", "The server couldn't load this. Try again.", { status, detail });
  return answer("rejected", "The server rejected the request.");
}

/**
 * The app's fetch wrapper: base URL, search params, JSON bodies, response
 * validation and typed failures. A write whose result is unknown is never a plain
 * failure: it may have been applied.
 */
export function createRequester({
  baseUrl,
  fetch: fetchImpl = globalThis.fetch.bind(globalThis),
  credentials = "same-origin",
  headers,
}: {
  baseUrl: string;
  fetch?: typeof fetch;
  credentials?: RequestCredentials;
  headers?: Record<string, string>;
}): Requester {
  const base = baseUrl.endsWith("/") ? baseUrl : `${baseUrl}/`;
  return async function request<T>(path: string, { method = "GET", query, body, signal, schema, read }: RequestOptions<T> = {}): Promise<T> {
    const url = new URL(path.replace(/^\/+/, ""), base);
    for (const [key, value] of Object.entries(query ?? {})) {
      if (value !== undefined && value !== null && value !== "") url.searchParams.set(key, String(value));
    }
    const write = method !== "GET" && !read;
    let response: Response;
    try {
      response = await fetchImpl(url, {
        method,
        signal,
        credentials,
        headers: body === undefined ? headers : { ...headers, "content-type": "application/json" },
        body: body === undefined ? undefined : JSON.stringify(body),
      });
    } catch (cause) {
      if (write) throw new ApiError("unknown-outcome", "The request was sent but no response arrived.", { cause });
      if (signal?.aborted) throw new ApiError("aborted", "The request was cancelled.", { cause });
      throw new ApiError("network", "Couldn't reach the server. Check your connection.", { cause });
    }
    if (!response.ok) throw await failure(response, write);
    if (response.status === 204 || !schema) return undefined as T;
    const parsed = schema.safeParse(await response.json().catch(() => undefined));
    if (parsed.success) return parsed.data;
    if (write) throw new ApiError("unknown-outcome", "The change may have been applied, but the response couldn't be read.", { status: response.status, cause: parsed.error });
    throw new ApiError("invalid-response", "The server response didn't match the expected shape.", { status: response.status, cause: parsed.error });
  };
}

const QUIET: ReadonlySet<ApiErrorKind> = new Set(["aborted", "forbidden", "not-found", "conflict", "validation"]);

/**
 * The app's QueryClient: reads retry twice and only before a definitive answer,
 * writes never retry, 30s stale time, and any 401 calls `onUnauthorized`.
 */
export function createQueryClient({
  onUnauthorized,
  reportError = (error: unknown) => console.error(error),
  staleTime = 30_000,
}: {
  onUnauthorized?: () => void;
  reportError?: (error: unknown, context: { source: "query" | "mutation" }) => void;
  staleTime?: number;
} = {}) {
  const onError = (source: "query" | "mutation") => (error: unknown) => {
    if (isApiError(error, "unauthorized")) return onUnauthorized?.();
    if (isApiError(error) && QUIET.has(error.kind)) return;
    reportError(error, { source });
  };
  return new QueryClient({
    queryCache: new QueryCache({ onError: onError("query") }),
    mutationCache: new MutationCache({ onError: onError("mutation") }),
    defaultOptions: {
      queries: { staleTime, retry: (count, error) => isRetryableRead(error) && count < 2 },
      mutations: { retry: 0 },
    },
  });
}

/** The parts of a `useQuery` result the content helpers read. */
export type QueryView = {
  isPending: boolean;
  isError: boolean;
  isFetching: boolean;
  data: unknown;
  error: { message: string } | null;
  refetch: () => Promise<{ isError: boolean }>;
};

/** A query as `AsyncContent` props: loading, a failed first load, a failed refresh, and Try again. */
export function queryContentProps(query: QueryView, failure: string) {
  const state: ContentState = query.isPending ? "loading" : query.isError ? (query.data !== undefined ? "refresh-error" : "error") : "ready";
  const onRetry = async (): Promise<ActionResult> => {
    const result = await query.refetch();
    return result.isError ? { status: "error", retryable: true, message: failure } : { status: "success" };
  };
  return { state, message: query.error?.message, onRetry };
}

/** A query as `DataTable` props; when ready, the table decides between rows, empty and no results. */
export function queryTableProps(query: QueryView, failure: string) {
  const { state, message, onRetry } = queryContentProps(query, failure);
  return { contentState: state === "ready" ? undefined : state, contentMessage: message, busy: query.isFetching, onRetry };
}

/** Whether the sign-in still holds. Any 401 calls `expire()`; the layout then asks to sign in again. */
export function createSessionStore() {
  let expired = false;
  const listeners = new Set<() => void>();
  const notify = () => listeners.forEach((listener) => listener());
  const subscribe = (listener: () => void) => {
    listeners.add(listener);
    return () => {
      listeners.delete(listener);
    };
  };
  return {
    expire() {
      if (expired) return;
      expired = true;
      notify();
    },
    restore() {
      if (!expired) return;
      expired = false;
      notify();
    },
    isExpired: () => expired,
    useExpired: () =>
      React.useSyncExternalStore(
        subscribe,
        () => expired,
        () => false,
      ),
  };
}
export type SessionStore = ReturnType<typeof createSessionStore>;

export const whoamiKey = ["session", "me"] as const;

/** Who is signed in, read once per page load. `read` is the app's own endpoint function. */
export function whoamiQuery<T>(read: (signal: AbortSignal) => Promise<T>) {
  return queryOptions({ queryKey: whoamiKey, queryFn: ({ signal }) => read(signal), staleTime: Infinity });
}

/** A typed context for one app service, with a hook that fails loudly outside its provider. */
export function createServiceContext<T>(hookName: string) {
  const Context = React.createContext<T | null>(null);
  function useService(): T {
    const service = React.useContext(Context);
    if (service === null) throw new Error(`${hookName} must be used inside its Provider`);
    return service;
  }
  return { Provider: Context.Provider, useService };
}
