export type SessionIndexErrorKind =
  | "validation"
  | "authentication"
  | "authorization"
  | "network"
  | "server"
  | "request"
  | "unexpected"

export type SessionIndexErrorState = {
  kind: SessionIndexErrorKind
  title: string
  detail: string
  retryable: boolean
}

function errorStatus(error: unknown): number | undefined {
  if (error === null || typeof error !== "object" || !("status" in error)) return undefined
  const status = (error as { status?: unknown }).status
  return typeof status === "number" && Number.isInteger(status) ? status : undefined
}

function errorMessage(error: unknown): string | undefined {
  if (!(error instanceof Error)) return undefined
  const message = error.message.trim()
  return message === "" ? undefined : message
}

export function sessionIndexErrorState(error: unknown): SessionIndexErrorState {
  const status = errorStatus(error)
  const message = errorMessage(error)

  if (status === 400 || status === 422) {
    return {
      kind: "validation",
      title: "The session filters are invalid.",
      detail: message ?? "Review the filter values and try again.",
      retryable: false,
    }
  }
  if (status === 401) {
    return {
      kind: "authentication",
      title: "Authentication is required to view sessions.",
      detail: "Sign in again or refresh your credentials.",
      retryable: false,
    }
  }
  if (status === 403) {
    return {
      kind: "authorization",
      title: "You do not have permission to view these sessions.",
      detail: "Ask an administrator for access to the control plane.",
      retryable: false,
    }
  }
  if (status !== undefined && status >= 500) {
    return {
      kind: "server",
      title: "The session service is temporarily unavailable.",
      detail: "Try the request again shortly.",
      retryable: true,
    }
  }
  if (status !== undefined) {
    return {
      kind: "request",
      title: "The session request could not be completed.",
      detail: message ?? `The server returned HTTP ${status}.`,
      retryable: false,
    }
  }
  if (error instanceof TypeError) {
    return {
      kind: "network",
      title: "The Cayu server could not be reached.",
      detail: "Check the connection and try again.",
      retryable: true,
    }
  }
  return {
    kind: "unexpected",
    title: "The session page could not be loaded.",
    detail: message ?? "An unexpected response was received.",
    retryable: true,
  }
}
