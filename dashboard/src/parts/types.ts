/**
 * The types the parts read and send: Cayu's inbox records as the server's contract names
 * them (src/lib/generated), and the two request shapes the dashboard defines on top.
 */
import type {
  ApiPendingAction,
  HumanReviewLongFieldView,
  HumanReviewReference,
  HumanReviewView,
  InboxCounts,
  InboxItem,
  InboxItemState,
  InboxPage,
  InboxSettleRequest,
  PendingKnowledgeDetailResponse,
  ToolApprovalBody,
  UserInputResolveBody,
} from "../lib/generated/server-api"

export type {
  ApiPendingAction,
  HumanReviewLongFieldView,
  HumanReviewReference,
  HumanReviewView,
  InboxCounts,
  InboxItem,
  InboxItemState,
  InboxPage,
  InboxSettleRequest,
  PendingKnowledgeDetailResponse,
  ToolApprovalBody,
  UserInputResolveBody,
}

/**
 * Public server-side Inbox filters; query and cursor share the same access scope.
 */
export type InboxQuery = {
  q?: string
  limit?: number
  cursor?: string | null
  session_id?: string
  agent_name?: string
  kind?: InboxItem["kind"]
}

/**
 * Settling a manual recovery with the evidence of what happened: whether the tool call
 * that never reported a result really completed or failed, the externally verified result
 * or failure reason, and, when the call was a question, the person's answer again.
 */
export type RecoveryRequest = {
  /** The inbox item's `attention_id`, naming which pending recovery this settles. */
  attention_id?: string | null
  outcome: "completed" | "failed"
  message: string
  answer?: string | null
}
