from __future__ import annotations

import asyncio
import base64
import heapq
import math
import secrets
from abc import ABC, abstractmethod
from bisect import bisect_left, bisect_right
from collections import deque
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from itertools import islice
from pathlib import Path
from types import FunctionType, MethodType
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast, overload
from uuid import uuid4

from cayu._resource_store_surface import model_store_surface
from cayu.budgets.base import SessionBudgetInspection as SessionBudgetInspection
from cayu.collaboration.peer_content import (
    PeerAppendKey,
    PeerContentAppendRequest,
    PeerContentConflict,
    PeerContentExposureReceipt,
    PeerContentExposureRequest,
    PeerContentReceipt,
    PeerContentUnavailable,
)
from cayu.sessions import _checkpoint_mutations as checkpoint_mutation_rules
from cayu.sessions import _checkpoint_preservation as checkpoint_preservation
from cayu.sessions import _completion_finalization as completion_finalization
from cayu.sessions import _execution_profile_rules as session_profile_rules
from cayu.sessions import _interaction_receipts as interaction_receipts
from cayu.sessions import _interaction_transitions as interaction_rules
from cayu.sessions import _invocation_scope as invocation_scope
from cayu.sessions import _model_failover_rules as model_failover_rules
from cayu.sessions import _model_stage_publication as model_stage_publication
from cayu.sessions import _model_stage_rules as model_stage_rules
from cayu.sessions import _operation_records as operation_record_rules
from cayu.sessions import _runtime_publication as publication_contracts
from cayu.sessions import _runtime_publication_rules as publication_rules
from cayu.sessions import _terminal_evidence as terminal_event_evidence
from cayu.sessions import _tool_round_publication_rules as tool_publication_rules
from cayu.sessions import _user_input_publication as user_input_publication_rules
from cayu.sessions import authority as session_authority_rules
from cayu.sessions import creation_fence
from cayu.sessions import event_delivery as event_delivery_rules
from cayu.sessions import event_delivery as session_event_rules
from cayu.sessions import event_delivery as side_effect_health
from cayu.sessions import event_queries as event_query_rules
from cayu.sessions import forks as session_fork_rules
from cayu.sessions import inspection as session_inspection_rules
from cayu.sessions import messaging as session_messaging
from cayu.sessions import queries as session_query_rules
from cayu.sessions import records as session_record_rules
from cayu.sessions import recovery as session_recovery_rules
from cayu.sessions import requests as session_request_contracts
from cayu.sessions import terminal_evidence as terminal_session_evidence
from cayu.sessions import transcript_input as session_transcript_input
from cayu.sessions import transcript_queries as session_transcript_queries
from cayu.sessions import usage as session_usage_rules
from cayu.sessions._checkpoint_mutations import (
    RUNTIME_PUBLICATION_MAX_CHECKPOINT_OPERATIONS as RUNTIME_PUBLICATION_MAX_CHECKPOINT_OPERATIONS,
)
from cayu.sessions._checkpoint_mutations import (
    RuntimePublicationCheckpointOperation as RuntimePublicationCheckpointOperation,
)
from cayu.sessions._checkpoint_mutations import (
    RuntimePublicationMutation as RuntimePublicationMutation,
)
from cayu.sessions._checkpoint_mutations import (
    SessionRuntimePublicationConflict as SessionRuntimePublicationConflict,
)
from cayu.sessions._checkpoint_mutations import (
    apply_runtime_publication_checkpoint_mutation as apply_runtime_publication_checkpoint_mutation,
)
from cayu.sessions._checkpoint_mutations import (
    runtime_publication_checkpoint_mutation as runtime_publication_checkpoint_mutation,
)
from cayu.sessions._checkpoint_mutations import (
    runtime_publication_checkpoint_value_digest as runtime_publication_checkpoint_value_digest,
)
from cayu.sessions._completion_finalization import (
    PENDING_COMPLETION_FINALIZATION_CHECKPOINT_KEY as PENDING_COMPLETION_FINALIZATION_CHECKPOINT_KEY,
)
from cayu.sessions._execution_profile_rules import SessionModelTransition as SessionModelTransition
from cayu.sessions._execution_profile_rules import (
    execution_profile_adoption_request_fingerprint as execution_profile_adoption_request_fingerprint,
)
from cayu.sessions._execution_profile_rules import (
    session_model_projection_cursor as session_model_projection_cursor,
)
from cayu.sessions._external_wait_memory import MemoryExternalWaitMixin
from cayu.sessions._interaction_transitions import (
    QUEUED_INTERACTION_PROFILE_HANDOFF_PAYLOAD_KEY as QUEUED_INTERACTION_PROFILE_HANDOFF_PAYLOAD_KEY,
)
from cayu.sessions._interaction_transitions import (
    InteractionTransitionReceiptResult as InteractionTransitionReceiptResult,
)
from cayu.sessions._interaction_transitions import (
    InteractionTransitionResult as InteractionTransitionResult,
)
from cayu.sessions._interaction_transitions import (
    InteractionTransitionSpec as InteractionTransitionSpec,
)
from cayu.sessions._interaction_transitions import (
    QueuedInteractionProfileHandoff as QueuedInteractionProfileHandoff,
)
from cayu.sessions._interaction_transitions import (
    SessionInvocationAdmission as SessionInvocationAdmission,
)
from cayu.sessions._interaction_transitions import (
    copy_interaction_transition_spec as copy_interaction_transition_spec,
)
from cayu.sessions._interaction_transitions import (
    queued_interaction_profile_handoff_evidence as queued_interaction_profile_handoff_evidence,
)
from cayu.sessions._invocation_scope import INHERIT_INTERACTION as INHERIT_INTERACTION
from cayu.sessions._invocation_scope import (
    UNASSOCIATED_RUNTIME_EVENT_TYPES as UNASSOCIATED_RUNTIME_EVENT_TYPES,
)
from cayu.sessions._invocation_scope import InteractionAttribution as InteractionAttribution
from cayu.sessions._invocation_scope import (
    attribute_event_to_current_interaction as attribute_event_to_current_interaction,
)
from cayu.sessions._invocation_scope import (
    attribute_events_to_current_interaction as attribute_events_to_current_interaction,
)
from cayu.sessions._invocation_scope import (
    resolve_interaction_attribution as resolve_interaction_attribution,
)
from cayu.sessions._model_completion_publication import (
    ModelCompletionStageDisposition as ModelCompletionStageDisposition,
)
from cayu.sessions._model_completion_publication import (
    ModelCompletionStageSettlementRequest as ModelCompletionStageSettlementRequest,
)
from cayu.sessions._model_failover import (
    MODEL_TARGET_PROJECTION_METADATA_KEY as MODEL_TARGET_PROJECTION_METADATA_KEY,
)
from cayu.sessions._model_failover import (
    MODEL_TARGET_PROJECTION_RECORD_TYPE as MODEL_TARGET_PROJECTION_RECORD_TYPE,
)
from cayu.sessions._model_failover import (
    MODEL_TARGET_PROJECTION_SCHEMA_VERSION as MODEL_TARGET_PROJECTION_SCHEMA_VERSION,
)
from cayu.sessions._model_stage import (
    MODEL_COMPLETION_ACTIVE_STAGE_RECORD_TYPE as MODEL_COMPLETION_ACTIVE_STAGE_RECORD_TYPE,
)
from cayu.sessions._model_stage import (
    MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY as MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY,
)
from cayu.sessions._model_stage import (
    MODEL_COMPLETION_PROVIDER_START_INTENT_MAX_BYTES as MODEL_COMPLETION_PROVIDER_START_INTENT_MAX_BYTES,
)
from cayu.sessions._model_stage import (
    MODEL_COMPLETION_RECOVERY_CONTEXT_MAX_BYTES as MODEL_COMPLETION_RECOVERY_CONTEXT_MAX_BYTES,
)
from cayu.sessions._model_stage import (
    MODEL_COMPLETION_STAGE_ABANDONMENT_RECORD_TYPE as MODEL_COMPLETION_STAGE_ABANDONMENT_RECORD_TYPE,
)
from cayu.sessions._model_stage import (
    MODEL_COMPLETION_STAGE_DISPATCH_OPERATION_KEY_PREFIX as MODEL_COMPLETION_STAGE_DISPATCH_OPERATION_KEY_PREFIX,
)
from cayu.sessions._model_stage import (
    MODEL_COMPLETION_STAGE_DISPATCH_RECORD_TYPE as MODEL_COMPLETION_STAGE_DISPATCH_RECORD_TYPE,
)
from cayu.sessions._model_stage import (
    MODEL_COMPLETION_STAGE_PREPARATION_RECORD_TYPE as MODEL_COMPLETION_STAGE_PREPARATION_RECORD_TYPE,
)
from cayu.sessions._model_stage import (
    MODEL_COMPLETION_STAGE_RECORD_TYPE as MODEL_COMPLETION_STAGE_RECORD_TYPE,
)
from cayu.sessions._model_stage import (
    MODEL_COMPLETION_STAGE_SCHEMA_VERSION as MODEL_COMPLETION_STAGE_SCHEMA_VERSION,
)
from cayu.sessions._model_stage import (
    MODEL_COMPLETION_STAGE_SETTLEMENT_RECORD_TYPE as MODEL_COMPLETION_STAGE_SETTLEMENT_RECORD_TYPE,
)
from cayu.sessions._model_stage import (
    MODEL_COMPLETION_STAGE_TERMINAL_RECORD_TYPE as MODEL_COMPLETION_STAGE_TERMINAL_RECORD_TYPE,
)
from cayu.sessions._model_stage import (
    MODEL_COMPLETION_WINNER_RECORD_TYPE as MODEL_COMPLETION_WINNER_RECORD_TYPE,
)
from cayu.sessions._model_stage import ActiveModelCompletionStage as ActiveModelCompletionStage
from cayu.sessions._model_stage import (
    ModelCompletionManualRecoveryRequest as ModelCompletionManualRecoveryRequest,
)
from cayu.sessions._model_stage import (
    ModelCompletionManualRecoveryResult as ModelCompletionManualRecoveryResult,
)
from cayu.sessions._model_stage import ModelCompletionPurpose as ModelCompletionPurpose
from cayu.sessions._model_stage import ModelCompletionStage as ModelCompletionStage
from cayu.sessions._model_stage import (
    ModelCompletionStageAbandonment as ModelCompletionStageAbandonment,
)
from cayu.sessions._model_stage import (
    ModelCompletionStageAbandonmentResult as ModelCompletionStageAbandonmentResult,
)
from cayu.sessions._model_stage import ModelCompletionStageDispatch as ModelCompletionStageDispatch
from cayu.sessions._model_stage import (
    ModelCompletionStageRecoveryFence as ModelCompletionStageRecoveryFence,
)
from cayu.sessions._model_stage import ModelCompletionStageRequest as ModelCompletionStageRequest
from cayu.sessions._model_stage import ModelCompletionStageResult as ModelCompletionStageResult
from cayu.sessions._model_stage import (
    ModelCompletionStageSettlement as ModelCompletionStageSettlement,
)
from cayu.sessions._model_stage import (
    SessionModelCompletionDispatchAlreadyAuthorized as SessionModelCompletionDispatchAlreadyAuthorized,
)
from cayu.sessions._model_stage import (
    SessionModelCompletionStageConflict as SessionModelCompletionStageConflict,
)
from cayu.sessions._model_stage import (
    SessionModelCompletionStageIncomplete as SessionModelCompletionStageIncomplete,
)
from cayu.sessions._model_stage import (
    copy_model_completion_manual_recovery_request as copy_model_completion_manual_recovery_request,
)
from cayu.sessions._model_stage_rules import (
    model_completion_stage_settlement_request as model_completion_stage_settlement_request,
)
from cayu.sessions._operation_records import (
    INTERACTION_TRANSITION_OPERATION_KEY_PREFIX as INTERACTION_TRANSITION_OPERATION_KEY_PREFIX,
)
from cayu.sessions._operation_records import (
    INVOCATION_TERMINAL_EVENT_OPERATION_KEY_PREFIX as INVOCATION_TERMINAL_EVENT_OPERATION_KEY_PREFIX,
)
from cayu.sessions._operation_records import (
    MODEL_COMPLETION_STAGE_OPERATION_KEY_PREFIX as MODEL_COMPLETION_STAGE_OPERATION_KEY_PREFIX,
)
from cayu.sessions._operation_records import (
    RUNTIME_PUBLICATION_OPERATION_KEY_PREFIX as RUNTIME_PUBLICATION_OPERATION_KEY_PREFIX,
)
from cayu.sessions._operation_records import (
    ZERO_WORK_INTERRUPTION_OPERATION_KEY as ZERO_WORK_INTERRUPTION_OPERATION_KEY,
)
from cayu.sessions._operation_records import (
    ModelCompletionStageRelease as ModelCompletionStageRelease,
)
from cayu.sessions._operation_records import (
    RuntimePublicationOperationRecordMutation as RuntimePublicationOperationRecordMutation,
)
from cayu.sessions._operation_records import (
    SessionOperationPublication as SessionOperationPublication,
)
from cayu.sessions._operation_records import (
    runtime_publication_operation_record_value_digest as runtime_publication_operation_record_value_digest,
)
from cayu.sessions._runtime_publication import (
    RUNTIME_PUBLICATION_MAX_EVENT_BINDINGS as RUNTIME_PUBLICATION_MAX_EVENT_BINDINGS,
)
from cayu.sessions._runtime_publication import (
    RUNTIME_PUBLICATION_MAX_TRANSCRIPT_MESSAGES as RUNTIME_PUBLICATION_MAX_TRANSCRIPT_MESSAGES,
)
from cayu.sessions._runtime_publication import (
    RUNTIME_PUBLICATION_RECORD_TYPE as RUNTIME_PUBLICATION_RECORD_TYPE,
)
from cayu.sessions._runtime_publication import (
    RUNTIME_PUBLICATION_SCHEMA_VERSION as RUNTIME_PUBLICATION_SCHEMA_VERSION,
)
from cayu.sessions._runtime_publication import (
    RuntimePublicationEventReference as RuntimePublicationEventReference,
)
from cayu.sessions._runtime_publication import RuntimePublicationKind as RuntimePublicationKind
from cayu.sessions._runtime_publication import (
    RuntimePublicationReceipt as RuntimePublicationReceipt,
)
from cayu.sessions._runtime_publication import (
    RuntimePublicationRequest as RuntimePublicationRequest,
)
from cayu.sessions._runtime_publication import RuntimePublicationResult as RuntimePublicationResult
from cayu.sessions._runtime_publication import (
    runtime_publication_event_reference as runtime_publication_event_reference,
)
from cayu.sessions._runtime_publication_rules import (
    runtime_publication_request_digest as runtime_publication_request_digest,
)
from cayu.sessions.authority import (
    CHECKPOINT_ROOT_FIELD_SCALAR_MAX_CHARS as CHECKPOINT_ROOT_FIELD_SCALAR_MAX_CHARS,
)
from cayu.sessions.authority import CheckpointRootFieldGuard as CheckpointRootFieldGuard
from cayu.sessions.authority import CheckpointRootFieldProjection as CheckpointRootFieldProjection
from cayu.sessions.authority import (
    CheckpointRootFieldProjectionValidator as CheckpointRootFieldProjectionValidator,
)
from cayu.sessions.authority import (
    checkpoint_root_field_projection as checkpoint_root_field_projection,
)
from cayu.sessions.authority import (
    checkpoint_root_field_projection_from_storage as checkpoint_root_field_projection_from_storage,
)
from cayu.sessions.creation import (
    run_request_with_prepared_work_attempt_creation as run_request_with_prepared_work_attempt_creation,
)
from cayu.sessions.creation import (
    session_invocation_for_run_request as session_invocation_for_run_request,
)
from cayu.sessions.creation import session_metadata_for_creation as session_metadata_for_creation
from cayu.sessions.creation_claims import (
    RUNTIME_SESSION_CREATE_CLAIM_REFERENCE_MAX_KEY_ID_CHARS as RUNTIME_SESSION_CREATE_CLAIM_REFERENCE_MAX_KEY_ID_CHARS,
)
from cayu.sessions.creation_claims import (
    RUNTIME_SESSION_CREATE_CLAIM_REFERENCE_MAX_OPERATION_ID_CHARS as RUNTIME_SESSION_CREATE_CLAIM_REFERENCE_MAX_OPERATION_ID_CHARS,
)
from cayu.sessions.creation_claims import (
    RUNTIME_SESSION_CREATE_CLAIM_REFERENCE_RECORD_TYPE as RUNTIME_SESSION_CREATE_CLAIM_REFERENCE_RECORD_TYPE,
)
from cayu.sessions.creation_claims import (
    RUNTIME_SESSION_CREATE_CLAIM_REFERENCE_SCHEMA_VERSION as RUNTIME_SESSION_CREATE_CLAIM_REFERENCE_SCHEMA_VERSION,
)
from cayu.sessions.creation_claims import (
    SESSION_CREATE_CLAIM_METADATA_KEY as SESSION_CREATE_CLAIM_METADATA_KEY,
)
from cayu.sessions.creation_claims import (
    SESSION_CREATE_CLAIM_RECORD_TYPE as SESSION_CREATE_CLAIM_RECORD_TYPE,
)
from cayu.sessions.creation_claims import (
    SESSION_CREATE_CLAIM_SCHEMA_VERSION as SESSION_CREATE_CLAIM_SCHEMA_VERSION,
)
from cayu.sessions.creation_claims import (
    RuntimeSessionCreateClaimAuthentication as RuntimeSessionCreateClaimAuthentication,
)
from cayu.sessions.creation_claims import (
    RuntimeSessionCreateClaimAuthenticationDisposition as RuntimeSessionCreateClaimAuthenticationDisposition,
)
from cayu.sessions.creation_claims import (
    RuntimeSessionCreateClaimReference as RuntimeSessionCreateClaimReference,
)
from cayu.sessions.creation_claims import (
    RuntimeSessionCreateClaimReferenceKey as RuntimeSessionCreateClaimReferenceKey,
)
from cayu.sessions.creation_claims import (
    apply_runtime_session_create_claim as apply_runtime_session_create_claim,
)
from cayu.sessions.creation_claims import (
    authenticate_runtime_session_create_claim_reference as authenticate_runtime_session_create_claim_reference,
)
from cayu.sessions.creation_claims import (
    bind_runtime_session_create_claim as bind_runtime_session_create_claim,
)
from cayu.sessions.creation_claims import (
    run_request_with_runtime_session_create_claim as run_request_with_runtime_session_create_claim,
)
from cayu.sessions.creation_claims import (
    run_request_with_runtime_session_create_claim_reference as run_request_with_runtime_session_create_claim_reference,
)
from cayu.sessions.creation_claims import (
    runtime_session_create_claim_reference as runtime_session_create_claim_reference,
)
from cayu.sessions.creation_claims import (
    session_has_runtime_create_claim as session_has_runtime_create_claim,
)
from cayu.sessions.creation_claims import (
    session_invocation_matches_run_request as session_invocation_matches_run_request,
)
from cayu.sessions.creation_claims import (
    session_matches_reconstructed_runtime_create_claim as session_matches_reconstructed_runtime_create_claim,
)
from cayu.sessions.creation_claims import (
    session_matches_runtime_create_claim as session_matches_runtime_create_claim,
)
from cayu.sessions.creation_claims import (
    strip_runtime_session_create_claim_before_redaction as strip_runtime_session_create_claim_before_redaction,
)
from cayu.sessions.event_delivery import (
    PERSISTED_EVENT_SIDE_EFFECT_ERROR_MAX_BYTES as PERSISTED_EVENT_SIDE_EFFECT_ERROR_MAX_BYTES,
)
from cayu.sessions.event_delivery import (
    PersistedEventSideEffectClaim as PersistedEventSideEffectClaim,
)
from cayu.sessions.event_delivery import (
    PersistedEventSideEffectClaimLost as PersistedEventSideEffectClaimLost,
)
from cayu.sessions.event_delivery import (
    PersistedEventSideEffectDelivery as PersistedEventSideEffectDelivery,
)
from cayu.sessions.event_delivery import (
    PersistedEventSideEffectHealth,
    PersistedEventSideEffectPage,
    PersistedEventSideEffectQuery,
)
from cayu.sessions.event_delivery import (
    PersistedEventSideEffectStatus as PersistedEventSideEffectStatus,
)
from cayu.sessions.event_delivery import (
    portable_persisted_event_side_effect_error as portable_persisted_event_side_effect_error,
)
from cayu.sessions.event_delivery import (
    restore_persisted_event_authority as restore_persisted_event_authority,
)
from cayu.sessions.event_delivery import (
    validate_persisted_event_side_effect_error as validate_persisted_event_side_effect_error,
)
from cayu.sessions.event_queries import EventOrder as EventOrder
from cayu.sessions.event_queries import EventQuery as EventQuery
from cayu.sessions.event_queries import EventQueryResultTooLarge as EventQueryResultTooLarge
from cayu.sessions.event_queries import copy_event_query as copy_event_query
from cayu.sessions.execution import SessionExecutionState
from cayu.sessions.forks import (
    FORK_EXECUTION_PROFILE_EXACT_SOURCE_SCHEMA_VERSION as FORK_EXECUTION_PROFILE_EXACT_SOURCE_SCHEMA_VERSION,
)
from cayu.sessions.forks import (
    FORK_EXECUTION_PROFILE_METADATA_KEY as FORK_EXECUTION_PROFILE_METADATA_KEY,
)
from cayu.sessions.forks import (
    FORK_EXECUTION_PROFILE_ORDINARY_SCHEMA_VERSION as FORK_EXECUTION_PROFILE_ORDINARY_SCHEMA_VERSION,
)
from cayu.sessions.forks import (
    FORK_EXECUTION_PROFILE_RECORD_TYPE as FORK_EXECUTION_PROFILE_RECORD_TYPE,
)
from cayu.sessions.forks import (
    FORK_SOURCE_SNAPSHOT_METADATA_KEY as FORK_SOURCE_SNAPSHOT_METADATA_KEY,
)
from cayu.sessions.forks import (
    PROMPT_ANATOMY_TRANSITION_METADATA_KEY as PROMPT_ANATOMY_TRANSITION_METADATA_KEY,
)
from cayu.sessions.forks import (
    PROMPT_ANATOMY_TRANSITION_RECORD_TYPE as PROMPT_ANATOMY_TRANSITION_RECORD_TYPE,
)
from cayu.sessions.forks import (
    PROMPT_ANATOMY_TRANSITION_SCHEMA_VERSION as PROMPT_ANATOMY_TRANSITION_SCHEMA_VERSION,
)
from cayu.sessions.forks import (
    ForkExecutionProfileDecisionRecord as ForkExecutionProfileDecisionRecord,
)
from cayu.sessions.forks import ForkExecutionProfileSelection as ForkExecutionProfileSelection
from cayu.sessions.forks import ForkExecutionProfileSource as ForkExecutionProfileSource
from cayu.sessions.forks import ForkSourceSnapshot as ForkSourceSnapshot
from cayu.sessions.forks import ForkSystemPromptPolicy as ForkSystemPromptPolicy
from cayu.sessions.forks import ForkSystemPromptReplacement as ForkSystemPromptReplacement
from cayu.sessions.forks import ProfiledSessionForkResult as ProfiledSessionForkResult
from cayu.sessions.forks import PromptAnatomyTransitionReceipt as PromptAnatomyTransitionReceipt
from cayu.sessions.forks import (
    SessionForkActiveModelStageConflict as SessionForkActiveModelStageConflict,
)
from cayu.sessions.forks import (
    SessionForkEnvironmentAllocationOwner as SessionForkEnvironmentAllocationOwner,
)
from cayu.sessions.forks import SessionForkProfileRelationship as SessionForkProfileRelationship
from cayu.sessions.forks import SessionForkSourceNotFound as SessionForkSourceNotFound
from cayu.sessions.forks import (
    apply_fork_system_prompt_replacement as apply_fork_system_prompt_replacement,
)
from cayu.sessions.forks import (
    copy_profiled_session_fork_result as copy_profiled_session_fork_result,
)
from cayu.sessions.forks import (
    effective_fork_source_execution_profile as effective_fork_source_execution_profile,
)
from cayu.sessions.forks import fork_session_invocation as fork_session_invocation
from cayu.sessions.forks import fork_source_state_sha256 as fork_source_state_sha256
from cayu.sessions.forks import (
    session_fork_profile_relationship as session_fork_profile_relationship,
)
from cayu.sessions.forks import (
    session_prompt_anatomy_transition as session_prompt_anatomy_transition,
)
from cayu.sessions.forks import validate_profiled_fork_evidence as validate_profiled_fork_evidence
from cayu.sessions.inspection import (
    SESSION_INSPECTION_LABEL_LIMIT as SESSION_INSPECTION_LABEL_LIMIT,
)
from cayu.sessions.inspection import SerializedRecordSummary as SerializedRecordSummary
from cayu.sessions.inspection import SessionInspectionIdentity as SessionInspectionIdentity
from cayu.sessions.inspection import SessionInspectionSummary as SessionInspectionSummary
from cayu.sessions.inspection import SessionInspectionUsageSummary as SessionInspectionUsageSummary
from cayu.sessions.lineage import (
    SESSION_LINEAGE_DEFAULT_CHILD_LIMIT as SESSION_LINEAGE_DEFAULT_CHILD_LIMIT,
)
from cayu.sessions.lineage import SESSION_LINEAGE_MAX_CHILD_LIMIT as SESSION_LINEAGE_MAX_CHILD_LIMIT
from cayu.sessions.lineage import (
    SESSION_LINEAGE_MAX_CURSOR_BYTES as SESSION_LINEAGE_MAX_CURSOR_BYTES,
)
from cayu.sessions.lineage import (
    SESSION_LINEAGE_MAX_EVENT_ID_BYTES as SESSION_LINEAGE_MAX_EVENT_ID_BYTES,
)
from cayu.sessions.lineage import (
    SESSION_LINEAGE_MAX_IDENTIFIER_BYTES as SESSION_LINEAGE_MAX_IDENTIFIER_BYTES,
)
from cayu.sessions.lineage import (
    SESSION_LINEAGE_MAX_ORIGIN_EVENTS as SESSION_LINEAGE_MAX_ORIGIN_EVENTS,
)
from cayu.sessions.lineage import (
    SESSION_LINEAGE_MAX_TIMESTAMP_BYTES as SESSION_LINEAGE_MAX_TIMESTAMP_BYTES,
)
from cayu.sessions.lineage import SessionLineageNode as SessionLineageNode
from cayu.sessions.lineage import SessionLineageOrigin as SessionLineageOrigin
from cayu.sessions.lineage import SessionLineageQuery as SessionLineageQuery
from cayu.sessions.lineage import SessionLineageResult as SessionLineageResult
from cayu.sessions.lineage import copy_session_lineage_query as copy_session_lineage_query
from cayu.sessions.lineage import decode_session_lineage_cursor as decode_session_lineage_cursor
from cayu.sessions.lineage import encode_session_lineage_cursor as encode_session_lineage_cursor
from cayu.sessions.mcp_manifest_history import (
    McpManifestBaseline as McpManifestBaseline,
)
from cayu.sessions.mcp_manifest_history import (
    McpManifestBaselineLoadResult as McpManifestBaselineLoadResult,
)
from cayu.sessions.mcp_manifest_history import (
    McpManifestHistoryConflict as McpManifestHistoryConflict,
)
from cayu.sessions.mcp_manifest_history import (
    McpManifestPublicationResult as McpManifestPublicationResult,
)
from cayu.sessions.mcp_manifest_history import (
    _mcp_manifest_session_ref,
    _require_sha256_identifier,
    _validate_mcp_manifest_history_keys,
    _validate_mcp_manifest_publication_state,
)
from cayu.sessions.messaging import (
    SESSION_MESSAGE_CONTENT_MAX_BYTES as SESSION_MESSAGE_CONTENT_MAX_BYTES,
)
from cayu.sessions.messaging import (
    SESSION_MESSAGE_DELIVERY_BATCH_LIMIT as SESSION_MESSAGE_DELIVERY_BATCH_LIMIT,
)
from cayu.sessions.messaging import (
    SESSION_MESSAGE_QUEUE_STORAGE_VALUE_MAX_BYTES as SESSION_MESSAGE_QUEUE_STORAGE_VALUE_MAX_BYTES,
)
from cayu.sessions.messaging import EnqueueSessionMessageRequest as EnqueueSessionMessageRequest
from cayu.sessions.messaging import EnqueueSessionMessageResult as EnqueueSessionMessageResult
from cayu.sessions.messaging import SessionMessageActionResult as SessionMessageActionResult
from cayu.sessions.messaging import SessionMessageDeliveryBatch as SessionMessageDeliveryBatch
from cayu.sessions.messaging import SessionMessageDeliveryMode as SessionMessageDeliveryMode
from cayu.sessions.messaging import SessionMessageInspection as SessionMessageInspection
from cayu.sessions.messaging import SessionMessageInspectionRecord as SessionMessageInspectionRecord
from cayu.sessions.messaging import SessionQueuedMessage as SessionQueuedMessage
from cayu.sessions.messaging import SessionQueuedMessagesPending as SessionQueuedMessagesPending
from cayu.sessions.messaging import (
    copy_enqueue_session_message_request as copy_enqueue_session_message_request,
)
from cayu.sessions.messaging import enqueue_session_message_input as enqueue_session_message_input
from cayu.sessions.messaging import queued_session_message_input as queued_session_message_input
from cayu.sessions.pending_action_contracts import (
    DEFAULT_PENDING_ACTION_RESULT_MAX_BYTES as DEFAULT_PENDING_ACTION_RESULT_MAX_BYTES,
)
from cayu.sessions.pending_action_contracts import (
    MAX_PENDING_ACTION_LEDGER_EVENTS_PER_CALL as MAX_PENDING_ACTION_LEDGER_EVENTS_PER_CALL,
)
from cayu.sessions.pending_action_contracts import (
    MAX_PENDING_ACTION_RESULT_BYTES as MAX_PENDING_ACTION_RESULT_BYTES,
)
from cayu.sessions.pending_action_contracts import (
    MAX_PENDING_ACTION_TOOL_CALLS as MAX_PENDING_ACTION_TOOL_CALLS,
)
from cayu.sessions.pending_action_contracts import (
    PENDING_ACTION_BARRIER_EVENT_TYPE_VALUES as PENDING_ACTION_BARRIER_EVENT_TYPE_VALUES,
)
from cayu.sessions.pending_action_contracts import (
    PENDING_ACTION_EVENT_TYPE_VALUES as PENDING_ACTION_EVENT_TYPE_VALUES,
)
from cayu.sessions.pending_action_contracts import (
    DelegatedActionReference as DelegatedActionReference,
)
from cayu.sessions.pending_action_contracts import PendingActionIssue as PendingActionIssue
from cayu.sessions.pending_action_contracts import PendingActionIssueCode as PendingActionIssueCode
from cayu.sessions.pending_action_contracts import (
    PendingActionListResult as PendingActionListResult,
)
from cayu.sessions.pending_action_contracts import PendingActionQuery as PendingActionQuery
from cayu.sessions.pending_action_contracts import PendingActionRecord as PendingActionRecord
from cayu.sessions.pending_action_contracts import (
    PendingActionResultTooLarge as PendingActionResultTooLarge,
)
from cayu.sessions.pending_action_contracts import (
    enforce_pending_action_result_size as enforce_pending_action_result_size,
)
from cayu.sessions.queries import MAX_AGGREGATE_LABEL_FILTERS as MAX_AGGREGATE_LABEL_FILTERS
from cayu.sessions.queries import (
    MAX_AGGREGATE_LABEL_SELECTOR_VALUES as MAX_AGGREGATE_LABEL_SELECTOR_VALUES,
)
from cayu.sessions.queries import MAX_AGGREGATE_LABEL_SELECTORS as MAX_AGGREGATE_LABEL_SELECTORS
from cayu.sessions.queries import MAX_SESSION_LIST_CURSOR_BYTES as MAX_SESSION_LIST_CURSOR_BYTES
from cayu.sessions.queries import LabelSelectorOperator as LabelSelectorOperator
from cayu.sessions.queries import LabelSelectorRequirement as LabelSelectorRequirement
from cayu.sessions.queries import SessionAggregateFilter as SessionAggregateFilter
from cayu.sessions.queries import SessionDebugState as SessionDebugState
from cayu.sessions.queries import SessionListResult as SessionListResult
from cayu.sessions.queries import SessionOrder as SessionOrder
from cayu.sessions.queries import SessionQuery as SessionQuery
from cayu.sessions.queries import SessionStatusCounts as SessionStatusCounts
from cayu.sessions.queries import (
    copy_label_selector_requirements as copy_label_selector_requirements,
)
from cayu.sessions.queries import copy_session_aggregate_filter as copy_session_aggregate_filter
from cayu.sessions.queries import copy_session_query as copy_session_query
from cayu.sessions.queries import decode_session_cursor as decode_session_cursor
from cayu.sessions.queries import encode_session_cursor as encode_session_cursor
from cayu.sessions.queries import session_next_cursor as session_next_cursor
from cayu.sessions.queries import session_order_is_descending as session_order_is_descending
from cayu.sessions.queries import (
    session_query_from_aggregate_filter as session_query_from_aggregate_filter,
)
from cayu.sessions.queries import session_sort_column as session_sort_column
from cayu.sessions.records import MAX_SESSION_ID_BYTES as MAX_SESSION_ID_BYTES
from cayu.sessions.records import (
    RUNTIME_BUILD_PROVENANCE_METADATA_KEY as RUNTIME_BUILD_PROVENANCE_METADATA_KEY,
)
from cayu.sessions.records import SESSION_RUNTIME_METADATA_KEYS as SESSION_RUNTIME_METADATA_KEYS
from cayu.sessions.records import SESSION_RUNTIME_METADATA_PREFIX as SESSION_RUNTIME_METADATA_PREFIX
from cayu.sessions.records import CheckpointTransform as CheckpointTransform
from cayu.sessions.records import EventRecord as EventRecord
from cayu.sessions.records import PendingActionKind as PendingActionKind
from cayu.sessions.records import PendingActionSession as PendingActionSession
from cayu.sessions.records import RunnerObservedEventIdentity as RunnerObservedEventIdentity
from cayu.sessions.records import Session as Session
from cayu.sessions.records import SessionIdentity as SessionIdentity
from cayu.sessions.records import (
    SessionInvocationSnapshot as SessionInvocationSnapshot,
)
from cayu.sessions.records import SessionRuntimeIdentity as SessionRuntimeIdentity
from cayu.sessions.records import (
    SessionStateSnapshot as SessionStateSnapshot,
)
from cayu.sessions.records import SessionStatus as SessionStatus
from cayu.sessions.records import SessionStatusConflict as SessionStatusConflict
from cayu.sessions.records import StoreTimeCheckpointTransform as StoreTimeCheckpointTransform
from cayu.sessions.records import TranscriptRecord as TranscriptRecord
from cayu.sessions.records import copy_session as copy_session
from cayu.sessions.records import copy_session_identity as copy_session_identity
from cayu.sessions.records import copy_session_runtime_identity as copy_session_runtime_identity
from cayu.sessions.records import copy_session_user_metadata as copy_session_user_metadata
from cayu.sessions.records import (
    is_runtime_owned_session_metadata_key as is_runtime_owned_session_metadata_key,
)
from cayu.sessions.records import replace_session_user_metadata as replace_session_user_metadata
from cayu.sessions.records import (
    runtime_build_provenance_from_session_metadata as runtime_build_provenance_from_session_metadata,
)
from cayu.sessions.records import session_user_metadata as session_user_metadata
from cayu.sessions.recovery import (
    MAX_INCOMPLETE_SESSIONS_RECOVERY_CURSOR_BYTES as MAX_INCOMPLETE_SESSIONS_RECOVERY_CURSOR_BYTES,
)
from cayu.sessions.recovery import (
    IncompleteSessionRecoveryAction as IncompleteSessionRecoveryAction,
)
from cayu.sessions.recovery import (
    IncompleteSessionRecoveryRequest as IncompleteSessionRecoveryRequest,
)
from cayu.sessions.recovery import (
    IncompleteSessionRecoveryResult as IncompleteSessionRecoveryResult,
)
from cayu.sessions.recovery import IncompleteSessionsRecoveryPage as IncompleteSessionsRecoveryPage
from cayu.sessions.recovery import (
    IncompleteSessionsRecoveryRequest as IncompleteSessionsRecoveryRequest,
)
from cayu.sessions.recovery import (
    copy_incomplete_session_recovery_request as copy_incomplete_session_recovery_request,
)
from cayu.sessions.recovery import (
    copy_incomplete_sessions_recovery_request as copy_incomplete_sessions_recovery_request,
)
from cayu.sessions.requests import (
    COMPACTION_INSTRUCTIONS_MAX_CHARS as COMPACTION_INSTRUCTIONS_MAX_CHARS,
)
from cayu.sessions.requests import CompactSessionRequest as CompactSessionRequest
from cayu.sessions.requests import ForkSessionRequest as ForkSessionRequest
from cayu.sessions.requests import InterruptSessionRequest as InterruptSessionRequest
from cayu.sessions.requests import ResumeRequest as ResumeRequest
from cayu.sessions.requests import RunRequest as RunRequest
from cayu.sessions.requests import copy_compact_session_request as copy_compact_session_request
from cayu.sessions.requests import copy_fork_session_request as copy_fork_session_request
from cayu.sessions.requests import copy_interrupt_session_request as copy_interrupt_session_request
from cayu.sessions.requests import copy_resume_request as copy_resume_request
from cayu.sessions.requests import copy_run_request as copy_run_request
from cayu.sessions.requests import (
    deferred_interaction_input_for_run_request as deferred_interaction_input_for_run_request,
)
from cayu.sessions.requests import (
    run_request_authority_is_runtime_generated as run_request_authority_is_runtime_generated,
)
from cayu.sessions.requests import (
    run_request_with_prepared_session_authority as run_request_with_prepared_session_authority,
)
from cayu.sessions.requests import (
    run_request_with_runtime_generated_authority as run_request_with_runtime_generated_authority,
)
from cayu.sessions.requests import (
    run_request_with_runtime_initial_transcript_authority as run_request_with_runtime_initial_transcript_authority,
)
from cayu.sessions.requests import (
    run_request_with_runtime_invocation as run_request_with_runtime_invocation,
)
from cayu.sessions.requests import (
    run_request_with_runtime_session_instance_authority as run_request_with_runtime_session_instance_authority,
)
from cayu.sessions.requests import (
    run_request_with_task_invocation as run_request_with_task_invocation,
)
from cayu.sessions.requests import (
    runtime_prepared_session_authority as runtime_prepared_session_authority,
)
from cayu.sessions.requests import (
    session_input_contract_evidence as session_input_contract_evidence,
)
from cayu.sessions.requests import (
    session_instance_id_for_run_request as session_instance_id_for_run_request,
)
from cayu.sessions.summaries import EventSummary as EventSummary
from cayu.sessions.summaries import SessionOperationalSnapshot as SessionOperationalSnapshot
from cayu.sessions.summaries import SessionOutcome as SessionOutcome
from cayu.sessions.summaries import event_summary_from_records as event_summary_from_records
from cayu.sessions.summaries import session_outcome as session_outcome
from cayu.sessions.summaries import session_outcome_from_records as session_outcome_from_records
from cayu.sessions.terminal_evidence import (
    TERMINAL_SESSION_EVIDENCE_DEFAULT_MAX_EVENTS as TERMINAL_SESSION_EVIDENCE_DEFAULT_MAX_EVENTS,
)
from cayu.sessions.terminal_evidence import (
    TERMINAL_SESSION_EVIDENCE_DEFAULT_MAX_RECORD_BYTES as TERMINAL_SESSION_EVIDENCE_DEFAULT_MAX_RECORD_BYTES,
)
from cayu.sessions.terminal_evidence import (
    TERMINAL_SESSION_EVIDENCE_DEFAULT_MAX_TOTAL_BYTES as TERMINAL_SESSION_EVIDENCE_DEFAULT_MAX_TOTAL_BYTES,
)
from cayu.sessions.terminal_evidence import (
    TERMINAL_SESSION_EVIDENCE_DEFAULT_MAX_TRANSCRIPT_RECORDS as TERMINAL_SESSION_EVIDENCE_DEFAULT_MAX_TRANSCRIPT_RECORDS,
)
from cayu.sessions.terminal_evidence import (
    TERMINAL_SESSION_EVIDENCE_HARD_MAX_EVENTS as TERMINAL_SESSION_EVIDENCE_HARD_MAX_EVENTS,
)
from cayu.sessions.terminal_evidence import (
    TERMINAL_SESSION_EVIDENCE_HARD_MAX_RECORD_BYTES as TERMINAL_SESSION_EVIDENCE_HARD_MAX_RECORD_BYTES,
)
from cayu.sessions.terminal_evidence import (
    TERMINAL_SESSION_EVIDENCE_HARD_MAX_TOTAL_BYTES as TERMINAL_SESSION_EVIDENCE_HARD_MAX_TOTAL_BYTES,
)
from cayu.sessions.terminal_evidence import (
    TERMINAL_SESSION_EVIDENCE_HARD_MAX_TRANSCRIPT_RECORDS as TERMINAL_SESSION_EVIDENCE_HARD_MAX_TRANSCRIPT_RECORDS,
)
from cayu.sessions.terminal_evidence import TerminalPublicationMarker as TerminalPublicationMarker
from cayu.sessions.terminal_evidence import TerminalSessionEvidence as TerminalSessionEvidence
from cayu.sessions.terminal_evidence import (
    TerminalSessionEvidenceBoundary as TerminalSessionEvidenceBoundary,
)
from cayu.sessions.terminal_evidence import (
    TerminalSessionEvidenceError as TerminalSessionEvidenceError,
)
from cayu.sessions.terminal_evidence import (
    TerminalSessionEvidenceErrorCode as TerminalSessionEvidenceErrorCode,
)
from cayu.sessions.terminal_evidence import (
    TerminalSessionEvidenceLimits as TerminalSessionEvidenceLimits,
)
from cayu.sessions.terminal_evidence import (
    copy_terminal_session_evidence as copy_terminal_session_evidence,
)
from cayu.sessions.topology import (
    SESSION_TOPOLOGY_DEFAULT_CHILD_LIMIT as SESSION_TOPOLOGY_DEFAULT_CHILD_LIMIT,
)
from cayu.sessions.topology import (
    SESSION_TOPOLOGY_MAX_ANCESTOR_DEPTH as SESSION_TOPOLOGY_MAX_ANCESTOR_DEPTH,
)
from cayu.sessions.topology import (
    SESSION_TOPOLOGY_MAX_CHILD_LIMIT as SESSION_TOPOLOGY_MAX_CHILD_LIMIT,
)
from cayu.sessions.topology import (
    SESSION_TOPOLOGY_MAX_CURSOR_BYTES as SESSION_TOPOLOGY_MAX_CURSOR_BYTES,
)
from cayu.sessions.topology import (
    SESSION_TOPOLOGY_MAX_EXPANDED_PARENTS as SESSION_TOPOLOGY_MAX_EXPANDED_PARENTS,
)
from cayu.sessions.topology import (
    SESSION_TOPOLOGY_MAX_IDENTIFIER_BYTES as SESSION_TOPOLOGY_MAX_IDENTIFIER_BYTES,
)
from cayu.sessions.topology import SESSION_TOPOLOGY_MAX_NODES as SESSION_TOPOLOGY_MAX_NODES
from cayu.sessions.topology import SessionTopologyBranch as SessionTopologyBranch
from cayu.sessions.topology import SessionTopologyCycle as SessionTopologyCycle
from cayu.sessions.topology import SessionTopologyDepthExceeded as SessionTopologyDepthExceeded
from cayu.sessions.topology import SessionTopologyNode as SessionTopologyNode
from cayu.sessions.topology import SessionTopologyQuery as SessionTopologyQuery
from cayu.sessions.topology import SessionTopologyStoreResult as SessionTopologyStoreResult
from cayu.sessions.topology import build_session_topology_result as build_session_topology_result
from cayu.sessions.topology import decode_session_topology_cursor as decode_session_topology_cursor
from cayu.sessions.topology import encode_session_topology_cursor as encode_session_topology_cursor
from cayu.sessions.transcript_input import (
    SESSION_STARTED_INPUT_CONTRACT_PAYLOAD_KEY as SESSION_STARTED_INPUT_CONTRACT_PAYLOAD_KEY,
)
from cayu.sessions.transcript_input import DeferredInteractionInput as DeferredInteractionInput
from cayu.sessions.transcript_input import (
    SessionInputContractEvidence as SessionInputContractEvidence,
)
from cayu.sessions.transcript_input import copy_transcript_messages as copy_transcript_messages
from cayu.sessions.transcript_input import (
    deferred_interaction_input_from_storage_payload as deferred_interaction_input_from_storage_payload,
)
from cayu.sessions.transcript_input import (
    deferred_interaction_input_storage_payload as deferred_interaction_input_storage_payload,
)
from cayu.sessions.transcript_input import (
    parse_session_input_contract_evidence as parse_session_input_contract_evidence,
)
from cayu.sessions.transcript_input import (
    require_deferred_initial_transcript_replacement as require_deferred_initial_transcript_replacement,
)
from cayu.sessions.transcript_input import (
    session_input_messages_sha256 as session_input_messages_sha256,
)
from cayu.sessions.transcript_input import (
    session_messages_input_contract_evidence as session_messages_input_contract_evidence,
)
from cayu.sessions.transcript_input import (
    system_prompt_messages_sha256 as system_prompt_messages_sha256,
)
from cayu.sessions.transcript_queries import (
    LATEST_TRANSCRIPT_TEXT_MAX_CHARS as LATEST_TRANSCRIPT_TEXT_MAX_CHARS,
)
from cayu.sessions.transcript_queries import (
    LATEST_TRANSCRIPT_TEXT_MAX_PARTS as LATEST_TRANSCRIPT_TEXT_MAX_PARTS,
)
from cayu.sessions.transcript_queries import (
    LATEST_TRANSCRIPT_TEXT_MAX_SOURCE_BYTES as LATEST_TRANSCRIPT_TEXT_MAX_SOURCE_BYTES,
)
from cayu.sessions.transcript_queries import (
    TRANSCRIPT_SEARCH_DEFAULT_LIMIT as TRANSCRIPT_SEARCH_DEFAULT_LIMIT,
)
from cayu.sessions.transcript_queries import (
    TRANSCRIPT_SEARCH_DEFAULT_MAX_BYTES as TRANSCRIPT_SEARCH_DEFAULT_MAX_BYTES,
)
from cayu.sessions.transcript_queries import (
    TRANSCRIPT_SEARCH_DEFAULT_SCAN_LIMIT as TRANSCRIPT_SEARCH_DEFAULT_SCAN_LIMIT,
)
from cayu.sessions.transcript_queries import (
    TRANSCRIPT_SEARCH_INDEX_VERSION as TRANSCRIPT_SEARCH_INDEX_VERSION,
)
from cayu.sessions.transcript_queries import (
    TRANSCRIPT_SEARCH_MAX_BYTES as TRANSCRIPT_SEARCH_MAX_BYTES,
)
from cayu.sessions.transcript_queries import (
    TRANSCRIPT_SEARCH_MAX_CURSOR_BYTES as TRANSCRIPT_SEARCH_MAX_CURSOR_BYTES,
)
from cayu.sessions.transcript_queries import (
    TRANSCRIPT_SEARCH_MAX_LIMIT as TRANSCRIPT_SEARCH_MAX_LIMIT,
)
from cayu.sessions.transcript_queries import (
    TRANSCRIPT_SEARCH_MAX_QUERY_BYTES as TRANSCRIPT_SEARCH_MAX_QUERY_BYTES,
)
from cayu.sessions.transcript_queries import (
    TRANSCRIPT_SEARCH_MAX_SCAN_LIMIT as TRANSCRIPT_SEARCH_MAX_SCAN_LIMIT,
)
from cayu.sessions.transcript_queries import (
    TRANSCRIPT_SEARCH_MAX_SESSION_IDS as TRANSCRIPT_SEARCH_MAX_SESSION_IDS,
)
from cayu.sessions.transcript_queries import (
    TRANSCRIPT_SEARCH_MIN_MAX_BYTES as TRANSCRIPT_SEARCH_MIN_MAX_BYTES,
)
from cayu.sessions.transcript_queries import (
    TRANSCRIPT_SEARCH_TOKENIZER_VERSION as TRANSCRIPT_SEARCH_TOKENIZER_VERSION,
)
from cayu.sessions.transcript_queries import ForkTranscriptValidator as ForkTranscriptValidator
from cayu.sessions.transcript_queries import TranscriptPage as TranscriptPage
from cayu.sessions.transcript_queries import TranscriptQuery as TranscriptQuery
from cayu.sessions.transcript_queries import TranscriptSearchHit as TranscriptSearchHit
from cayu.sessions.transcript_queries import TranscriptSearchQuery as TranscriptSearchQuery
from cayu.sessions.transcript_queries import TranscriptSearchResult as TranscriptSearchResult
from cayu.sessions.transcript_queries import TranscriptSnapshot as TranscriptSnapshot
from cayu.sessions.transcript_queries import (
    TranscriptTextReadLimitExceeded as TranscriptTextReadLimitExceeded,
)
from cayu.sessions.transcript_queries import copy_transcript_query as copy_transcript_query
from cayu.sessions.transcript_queries import (
    copy_transcript_search_hit as copy_transcript_search_hit,
)
from cayu.sessions.transcript_queries import (
    copy_transcript_search_query as copy_transcript_search_query,
)
from cayu.sessions.transcript_queries import (
    decode_transcript_search_cursor as decode_transcript_search_cursor,
)
from cayu.sessions.transcript_queries import (
    encode_transcript_search_cursor as encode_transcript_search_cursor,
)
from cayu.sessions.transcript_queries import filter_transcript_records as filter_transcript_records
from cayu.sessions.transcript_queries import (
    fork_source_transcript_sha256 as fork_source_transcript_sha256,
)
from cayu.sessions.transcript_queries import (
    fork_transcript_is_accepted as fork_transcript_is_accepted,
)
from cayu.sessions.transcript_queries import (
    transcript_search_document as transcript_search_document,
)
from cayu.sessions.transcript_queries import (
    transcript_search_document_from_text as transcript_search_document_from_text,
)
from cayu.sessions.transcript_queries import (
    transcript_search_document_score as transcript_search_document_score,
)
from cayu.sessions.transcript_queries import (
    transcript_search_hit_from_message as transcript_search_hit_from_message,
)
from cayu.sessions.transcript_queries import (
    transcript_search_position_after_cursor as transcript_search_position_after_cursor,
)
from cayu.sessions.transcript_queries import (
    transcript_search_query_document as transcript_search_query_document,
)
from cayu.sessions.transcript_queries import (
    transcript_search_query_fingerprint as transcript_search_query_fingerprint,
)
from cayu.sessions.transcript_queries import (
    transcript_search_query_tokens as transcript_search_query_tokens,
)
from cayu.sessions.transcript_queries import transcript_search_score as transcript_search_score
from cayu.sessions.transcript_queries import (
    transcript_search_session_token as transcript_search_session_token,
)
from cayu.sessions.transcript_queries import transcript_search_text as transcript_search_text
from cayu.sessions.usage import MAX_USAGE_ROLLUP_WINDOW as MAX_USAGE_ROLLUP_WINDOW
from cayu.sessions.usage import UsageRollupQuery as UsageRollupQuery
from cayu.sessions.usage import copy_usage_rollup_query as copy_usage_rollup_query
from cayu.storage._creation_fence import MemoryCreationFenceMixin
from cayu.storage._session_execution import MemorySessionExecutionMixin

if TYPE_CHECKING:
    from cayu.collaboration._contracts import ExactLookup
    from cayu.collaboration.participants import ParticipantRef as _ParticipantRef
    from cayu.runtime._model_failover_stage import ModelFailoverStageAdmission
    from cayu.runtime._zero_work_interruption import (
        ZeroWorkInterruptionPublication,
        ZeroWorkInterruptionRequest,
    )
    from cayu.runtime.evidence_spool import EvidenceSpool
    from cayu.sessions._access_rules import _SessionAccessBounds
    from cayu.sessions._context_view_source import (
        CompletedTurnSnapshot,
        ContextViewPublicationSource,
    )
    from cayu.sessions._external_wait_transition import ExternalWaitMutation
    from cayu.sessions._invocation_lifecycle import (
        AdmitInvocationCommand,
        CreateInvocationCommand,
        InvocationLifecycleCommand,
        InvocationMutationResult,
        InvocationReleaseResult,
        RebindInvocationCommand,
        RejectInvocationCommand,
        ReleaseInvocationCommand,
        SettleInvocationCommand,
    )
    from cayu.sessions._recipient_continuation import RecipientContinuationSelection
    from cayu.sessions._session_continuation import (
        ContinuationConsumption,
        ContinuationLatch,
        ContinuationNamespace,
        ContinuationPreparation,
        ContinuationRecord,
        ContinuationRetirement,
        ContinuationTicket,
    )
    from cayu.sessions._temporary_continuation import (
        TemporaryServiceAdmission,
        TemporaryServicePreparation,
        TemporaryServiceRecord,
    )
    from cayu.sessions.context_views import (
        ContextViewLifecycleEvent as _ContextViewLifecycleEvent,
    )
    from cayu.sessions.context_views import (
        ContextViewManifest as _ContextViewManifest,
    )
    from cayu.sessions.context_views import (
        ContextViewOwnershipRequest as _ContextViewOwnershipRequest,
    )
    from cayu.sessions.context_views import (
        ContextViewReadback as _ContextViewReadback,
    )
    from cayu.sessions.context_views import (
        ContextViewSelectionReceipt as _ContextViewSelectionReceipt,
    )
    from cayu.sessions.context_views import (
        ContextViewSelectionRequest as _ContextViewSelectionRequest,
    )
    from cayu.sessions.context_views import (
        ParticipantSessionBinding,
        ParticipantSessionCreationReceipt,
        ParticipantSessionCreationRequest,
    )
    from cayu.sessions.exports import SessionExportLimits, SessionExportSnapshot
    from cayu.sessions.external_waits import (
        ExternalWaitPruneResult,
        ExternalWaitRecord,
        ExternalWaitRetirement,
        ExternalWaitRetirementRequest,
        ExternalWaitScope,
    )
    from cayu.storage.retention import (
        RetentionAuditEntry,
        RetentionAuditRecord,
        RetentionMode,
        RetentionProgressCallback,
        RetentionProtection,
        RetentionReport,
        SessionRetentionPolicy,
    )

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    StrictInt,
    field_validator,
    model_validator,
)

from cayu._clock import utc_clock, utc_duration_cutoff
from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
    JsonUtf8SizeCounter,
    canonical_bounded_durable_json_bytes,
    compact_json_utf8_size,
    copy_durable_json_object,
    copy_label_map,
)
from cayu._validation import (
    require_durable_clean_nonblank as require_clean_nonblank,
)
from cayu._validation import (
    require_durable_nonblank as require_nonblank,
)
from cayu.approvals.tools import (
    ResolutionActor,
)
from cayu.budgets.aggregates import (
    _IN_MEMORY_AGGREGATE_CANCELLATION_INTERVAL,
    EXACT_AGGREGATE,
    UsageRollupStoreResult,
    _cooperate_with_in_memory_aggregate_cancellation,
    project_aggregate_usage_inspection_event,
)
from cayu.budgets.base import (
    BudgetReservationIdentityConflict,
    is_budget_inspection_event,
    is_budget_model_attempt_terminal_event,
    project_budget_inspection_event,
    project_budget_model_attempt_inspection_event,
    session_budget_inspection,
)
from cayu.budgets.pricing import PriceBook
from cayu.events import (
    EVENT_ID_MAX_CHARS,
    Event,
    EventType,
    copy_event,
    event_with_runtime_generated_id,
    event_with_runtime_payload_authority,
)
from cayu.execution_profiles import (
    ExecutionProfileDecision,
    ExecutionProfileIdentity,
    ExecutionProfileRejectionResult,
)
from cayu.execution_units import (
    ToolRoundIdentity,
    copy_tool_round_identity,
)
from cayu.memory.evidence import (
    MAX_RECALL_RECEIPT_ITEMS,
    ContextExposure,
    ContextExposurePage,
    ContextExposureTransitionRequest,
    RecallEvidenceConflict,
    RecallEvidenceQuery,
    RecallItemExposure,
    RecallReceipt,
    RecallReceiptPage,
    append_context_exposure_transition,
    context_exposure_creation_matches,
    context_exposure_transition_replays,
    copy_context_exposure,
    copy_recall_item_exposure,
    copy_recall_receipt,
    decode_recall_evidence_cursor,
    encode_recall_evidence_cursor,
    memory_evidence_document_bytes,
    recall_item_exposure_matches_receipt_item,
    require_memory_evidence_id,
    require_memory_evidence_session_id,
    validate_context_exposure_carried_receipt_scope,
    validate_context_exposure_receipt_scope,
    validate_new_context_exposure,
)
from cayu.messages import (
    Message,
    MessageRole,
    copy_message,
    detach_message,
)
from cayu.runtime._child_session_notifications import (
    CHILD_SESSION_ADMISSION_OCCURRENCE_TYPE,
    ChildSessionLifecycleEntry,
    ChildSessionLifecycleOccurrence,
    ChildSessionLifecycleOccurrenceSource,
    ChildSessionLifecyclePage,
    ChildSessionLifecycleQuery,
    ChildSessionLifecycleState,
    ChildSessionNotificationConsumption,
    ChildSessionNotificationFreshness,
    ChildSessionRelationship,
    child_session_notification_stage_binding,
    child_session_notification_storage_key,
)
from cayu.runtime._cost_accounting import CostAccountingSnapshot
from cayu.runtime._usage_accounting import UsageAccountingSnapshot
from cayu.runtime.public_authority import (
    PublicAuthorityAliasCodec,
    PublicAuthorityAliasKeyring,
    parse_public_authority_alias,
)
from cayu.runtime.service_manifest import RuntimeStoreDurability
from cayu.sessions._execution_profile_checkpoint import (
    ActiveInvocationExecutionProfile,
    active_invocation_execution_profile_from_checkpoint,
    active_invocation_execution_profile_is_released,
    checkpoint_with_active_invocation_execution_profile,
)
from cayu.sessions._invocation_terminal_decision import (
    InvocationTerminalDecision,
)
from cayu.sessions._model_failover import (
    ModelFailoverPolicy as ModelFailoverPolicy,
)
from cayu.sessions._model_failover import (
    ModelTarget as ModelTarget,
)
from cayu.sessions._model_failover import (
    copy_model_failover_policy as copy_model_failover_policy,
)
from cayu.sessions._provider_operation_cancellation_claim import (
    active_provider_operation_cancellation_claim_from_checkpoint,
)
from cayu.sessions._terminal_evidence import (
    SESSION_RUN_OPERATION_ID_PAYLOAD_KEY,
    TERMINAL_EVIDENCE_EVENT_TYPES,
    TERMINAL_EVIDENCE_QUERY_LIMIT,
    classify_current_terminal_evidence,
)
from cayu.sessions.authority import (
    CheckpointValueAuthority,
    SessionRunFenced,
    checkpoint_value_authority,
)
from cayu.sessions.checkpoints import (
    CHECKPOINT_SCHEMA_VERSION_KEY,
    CURRENT_CHECKPOINT_SCHEMA_VERSION,
)
from cayu.sessions.invocation import SessionExecutionSource as SessionExecutionSource
from cayu.sessions.invocation import (
    SessionInvocationBinding,
)
from cayu.sessions.messaging import (
    SessionMessageActionRequest,
    SessionMessageConditions,
    SessionMessageConflict,
    SessionMessageQuery,
    SessionMessageQueueStatus,
    SessionMessageSource,
    copy_session_message_action,
    session_message_checkpoint_sha256,
    session_message_rejection,
)
from cayu.tools.exposure import (
    ToolCapabilityCeiling,
)
from cayu.tools.grants import (
    TARGETED_TOOL_GRANT_INSPECTION_MAX_RECORDS,
    TARGETED_TOOL_GRANT_MAX_REQUESTS,
    TARGETED_TOOL_REFERENCE_FIELD_NAME,
    TargetedToolGrantIssueOutcome,
    TargetedToolGrantIssueResult,
    TargetedToolGrantReconstructionResult,
    TargetedToolGrantRecord,
    TargetedToolGrantStateSnapshot,
    TargetedToolUseBinding,
    TargetedToolUseDisposition,
    TargetedToolUseRejectionReason,
    TargetedToolUseRequest,
    TargetedToolUseResult,
    copy_targeted_tool_grant_record,
    targeted_tool_grant_event,
    targeted_tool_grant_reconstruction_rejection_reason,
    targeted_tool_grant_with_active_reference,
    targeted_tool_unresolved_rejection_event,
    targeted_tool_use_binding,
    targeted_tool_use_rejection_event,
    targeted_tool_use_rejection_reason,
    targeted_tool_use_scope_rejection_reason,
    validate_targeted_tool_grant_batch_evidence,
    validate_targeted_tool_grant_issuance_evidence,
    validate_targeted_tool_grant_lifecycle_event,
    validate_targeted_tool_grant_reference,
    validate_targeted_tool_grant_revocation_evidence,
    validate_targeted_tool_grant_revocation_reason,
    validate_targeted_tool_unresolved_rejection_evidence,
    validate_targeted_tool_use_rejection_evidence,
)
from cayu.workflows.base import WORKFLOW_ATTEMPT_EVENT_TYPE

# Unexported capability used only by CayuApp's recipient boundary. Direct
# store callers cannot manufacture recipient provenance by shaping a request.
_RECIPIENT_PROVENANCE_CAPABILITY = object()


class _SessionCommitGuardCancelled(RuntimeError):
    """An owned commit guard stopped without caller-owned cancellation."""


@dataclass(frozen=True, slots=True)
class _OwnedOffThreadSessionCommitGuard:
    callback: Callable[[], None]

    def __call__(self) -> None:
        try:
            self.callback()
        except asyncio.CancelledError as error:
            raise _SessionCommitGuardCancelled(
                "Session commit guard was unexpectedly cancelled."
            ) from error


async def _run_session_commit_guard_owned(
    commit_guard: Callable[[], None],
) -> BaseException | None:
    """Run an explicitly off-thread commit guard without abandoning its mutation."""

    if type(commit_guard) is not _OwnedOffThreadSessionCommitGuard:
        commit_guard()
        return None

    worker = asyncio.create_task(asyncio.to_thread(commit_guard))
    caller_signal: BaseException | None = None
    while not worker.done():
        try:
            await asyncio.shield(worker)
        except asyncio.CancelledError as cancellation:
            if caller_signal is None:
                caller_signal = cancellation
            elif isinstance(caller_signal, asyncio.CancelledError):
                caller_signal.add_note(
                    "Additional cancellation arrived while the session commit guard settled."
                )
            else:
                caller_signal = BaseExceptionGroup(
                    "Session commit guard received multiple control signals.",
                    [caller_signal, cancellation],
                )
        except BaseException as signal:
            if worker.done():
                break
            caller_signal = (
                signal
                if caller_signal is None
                else BaseExceptionGroup(
                    "Session commit guard received multiple control signals.",
                    [caller_signal, signal],
                )
            )
    try:
        worker.result()
    except BaseException as guard_error:
        if not isinstance(guard_error, (Exception, asyncio.CancelledError)):
            if caller_signal is not None:
                raise guard_error from caller_signal
            raise
        if caller_signal is not None:
            raise caller_signal from guard_error
        raise
    return caller_signal


class SessionExecutionInProgress(SessionStatusConflict):
    """Continuation cannot establish that the current execution was abandoned."""

    def __init__(self, session_id: str, status: SessionStatus) -> None:
        self.session_id = session_id
        self.status = status
        super().__init__(
            f"Session {session_id!r} is {status.value}: execution is still in progress "
            "or its process stopped before releasing ownership. Retry after its execution "
            "lease expires; continuation recovers an abandoned owner automatically. "
            "For sessions without execution ownership, confirm that the previous executor "
            "has stopped before calling app.recover_incomplete_session("
            "IncompleteSessionRecoveryRequest(session_id=...))."
        )


_INTERRUPTION_CASCADE_ATTEMPT_ID_MAX_CHARS = 128
_INTERRUPTION_CASCADE_CLAIM_ID_MAX_CHARS = 128
_INTERRUPTION_CASCADE_TIMESTAMP_MAX_CHARS = 64
_INTERRUPTION_CASCADE_GENERATION_MAX_CHARS = 32
_INTERRUPTION_CASCADE_MISSING = object()


def _project_interruption_cascade_marker_fields(
    marker_type: str | None,
    field_types: Mapping[str, str | None],
    field_values: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Build the bounded marker view used by control-plane status polling."""

    if marker_type is None or marker_type == "null":
        return None
    if marker_type != "object":
        return {}

    projected: dict[str, Any] = {}

    def required_string(field: str, max_chars: int) -> None:
        value = field_values.get(field)
        if (
            field_types.get(field) == "string"
            and type(value) is str
            and value.strip()
            and len(value) <= max_chars
        ):
            projected[field] = value
        else:
            projected[field] = None

    def optional_string(field: str, max_chars: int) -> None:
        field_type = field_types.get(field)
        if field_type is None or field_type == "null":
            return
        value = field_values.get(field)
        if type(value) is str and field_type == "string" and len(value) <= max_chars:
            projected[field] = value
        else:
            projected[field] = 0

    required_string("attempt_id", _INTERRUPTION_CASCADE_ATTEMPT_ID_MAX_CHARS)
    projected["interrupt_payload"] = (
        {} if field_types.get("interrupt_payload") == "object" else None
    )

    generation_type = field_types.get("generation")
    if generation_type is not None:
        generation = field_values.get("generation")
        try:
            if (
                generation_type not in {"integer", "number"}
                or type(generation) is not str
                or len(generation) > _INTERRUPTION_CASCADE_GENERATION_MAX_CHARS
                or not generation.lstrip("-").isdigit()
            ):
                raise ValueError
            projected["generation"] = int(generation)
        except ValueError:
            projected["generation"] = None

    failure_type = field_types.get("failure_recorded")
    if failure_type is not None:
        failure = field_values.get("failure_recorded")
        if failure_type == "boolean" and type(failure) is bool:
            projected["failure_recorded"] = failure
        else:
            projected["failure_recorded"] = None

    optional_string("claim_id", _INTERRUPTION_CASCADE_CLAIM_ID_MAX_CHARS)
    optional_string("claim_expires_at", _INTERRUPTION_CASCADE_TIMESTAMP_MAX_CHARS)
    optional_string("created_at", _INTERRUPTION_CASCADE_TIMESTAMP_MAX_CHARS)
    return projected


def _project_interruption_cascade_marker(marker: Any) -> dict[str, Any] | None:
    if marker is _INTERRUPTION_CASCADE_MISSING or marker is None:
        return None
    if type(marker) is not dict:
        return {}

    field_types: dict[str, str | None] = {}
    field_values: dict[str, Any] = {}
    for field in (
        "attempt_id",
        "interrupt_payload",
        "generation",
        "failure_recorded",
        "claim_id",
        "claim_expires_at",
        "created_at",
    ):
        if field not in marker:
            field_types[field] = None
            continue
        value = marker[field]
        if value is None:
            field_types[field] = "null"
        elif type(value) is str:
            field_types[field] = "string"
            max_chars = {
                "attempt_id": _INTERRUPTION_CASCADE_ATTEMPT_ID_MAX_CHARS,
                "claim_id": _INTERRUPTION_CASCADE_CLAIM_ID_MAX_CHARS,
                "claim_expires_at": _INTERRUPTION_CASCADE_TIMESTAMP_MAX_CHARS,
                "created_at": _INTERRUPTION_CASCADE_TIMESTAMP_MAX_CHARS,
            }.get(field, _INTERRUPTION_CASCADE_GENERATION_MAX_CHARS)
            field_values[field] = value[: max_chars + 1]
        elif type(value) is bool:
            field_types[field] = "boolean"
            field_values[field] = value
        elif type(value) is int:
            field_types[field] = "integer"
            field_values[field] = str(value)[: _INTERRUPTION_CASCADE_GENERATION_MAX_CHARS + 1]
        elif type(value) is float:
            field_types[field] = "number"
            field_values[field] = str(value)[: _INTERRUPTION_CASCADE_GENERATION_MAX_CHARS + 1]
        elif type(value) is dict:
            field_types[field] = "object"
        elif type(value) is list:
            field_types[field] = "array"
        else:
            field_types[field] = "other"
    return _project_interruption_cascade_marker_fields("object", field_types, field_values)


def _validate_inactive_for_seconds(value: int | None) -> int | None:
    if value is None:
        return None
    if type(value) is not int:
        raise TypeError("inactive_for_seconds must be an integer.")
    if value < 0 or value > MAX_DURABLE_JSON_INTEGER:
        raise ValueError(f"inactive_for_seconds must be between 0 and {MAX_DURABLE_JSON_INTEGER}.")
    return value


SessionOperationInitializer = Callable[
    [Session],
    Mapping[str, dict[str, Any]],
]


ForkCheckpointAuthorityDecoder = Callable[
    [dict[str, Any] | None],
    dict[str, Any] | None,
]
FORK_TRANSCRIPT_VALIDATION_ERROR = (
    "Fork transcript failed atomic copy validation because it changed after "
    "preflight or contains a workload secret and cannot be copied without "
    "changing durable conversation history."
)


def transform_fork_checkpoint(
    source_session: Session,
    source_checkpoint: dict[str, Any] | None,
    transform: CheckpointTransform,
) -> dict[str, Any] | None:
    """Apply a fork transform without retaining its raw input on failure."""

    try:
        transformed = transform(source_session, source_checkpoint)
        if transformed is None:
            return None
        # Forked checkpoints never inherit the source session's private
        # completion-result publication ownership. Apply the same rule when a
        # raw store caller returns the source checkpoint unchanged or attempts
        # to manufacture the reserved root.
        return checkpoint_preservation._replace_checkpoint_preserving_completion_result_event_publications(
            None,
            transformed,
            preserve_session_exports=False,
            preserve_session_continuations=False,
            session_id=source_session.id,
        )
    except BaseException:
        if source_checkpoint is not None:
            source_checkpoint.clear()
        raise


# Must hold everything an admitted RunRequest carries into recovery: up to
# 256 KiB of request metadata plus budget limits and run settings.


SessionOperationTransform = Callable[
    [Session, dict[str, Any] | None, dict[str, Any] | None],
    SessionOperationPublication,
]
StoreTimeSessionOperationTransform = Callable[
    [Session, dict[str, Any] | None, dict[str, Any] | None, datetime],
    SessionOperationPublication,
]


class QueuedDispatchTerminalReceipt(BaseModel):
    """Exact session-side handoff awaiting queue-task acknowledgement."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    session_id: str
    queue_task_id: str
    operation_id: str
    terminal_event_id: str

    @field_validator(
        "session_id",
        "queue_task_id",
        "operation_id",
        "terminal_event_id",
    )
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        value = require_clean_nonblank(value, info.field_name)
        if info.field_name == "terminal_event_id" and len(value) > EVENT_ID_MAX_CHARS:
            raise ValueError("terminal_event_id is too long.")
        return value


class QueuedDispatchTerminalReceiptQuery(BaseModel):
    """Bounded keyset query for live queued-dispatch handoffs."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    after_session_id: str | None = None
    after_operation_id: str | None = None
    limit: StrictInt = Field(default=100, ge=1, le=1000)

    @field_validator("after_session_id", "after_operation_id")
    @classmethod
    def validate_optional_identity(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)

    @model_validator(mode="after")
    def require_complete_cursor(self) -> QueuedDispatchTerminalReceiptQuery:
        if (self.after_session_id is None) != (self.after_operation_id is None):
            raise ValueError("Queued dispatch receipt cursor fields must be supplied together.")
        return self


def _public_authority_alias_store_key(
    public_alias: str,
    *,
    field_name: str,
    private_value: str | None,
    scope_session_id: str | None,
) -> tuple[str, str, str]:
    parsed = parse_public_authority_alias(public_alias)
    if parsed is None or parsed.field_name != field_name:
        raise ValueError("Public authority alias is malformed or field-mismatched.")
    if field_name in {"session_id", "causal_budget_id"}:
        if scope_session_id is not None:
            raise ValueError(f"{field_name} aliases must not have a session scope.")
        scope_key = ""
    elif field_name in {"interaction_id", TARGETED_TOOL_REFERENCE_FIELD_NAME}:
        if scope_session_id is None:
            raise ValueError(f"{field_name} aliases require a private session scope.")
        scope_key = require_nonblank(scope_session_id, "scope_session_id")
    else:
        raise ValueError(
            "field_name must be session_id, causal_budget_id, interaction_id, or tool_ref."
        )
    if private_value is not None:
        require_nonblank(private_value, "private_value")
    return field_name, scope_key, public_alias


def _authenticated_public_authority_alias_private_value(
    codec: PublicAuthorityAliasCodec | None,
    public_alias: str,
    private_value: str | None,
    *,
    field_name: str,
    scope_session_id: str | None,
) -> str | None:
    """Return a registry value only while its signing key remains accepted."""

    if private_value is None or codec is None:
        return None
    try:
        authenticated = codec.matches(
            public_alias,
            private_value,
            field_name=field_name,
            session_id=scope_session_id,
        )
    except (TypeError, ValueError):
        raise ValueError(
            "Public authority alias registry contains invalid private authority."
        ) from None
    return private_value if authenticated else None


_SESSION_EXPORT_OWNER_METHODS = (
    "__getattr__",
    "load",
    "load_checkpoint",
    "load_session_operation",
    "load_transcript_window",
    "create",
    "create_fork",
    "create_fork_with_transcript_validation",
    "create_profiled_fork",
    "_create_fork",
    "checkpoint",
    "transform_checkpoint",
    "transform_checkpoint_with_store_time",
    "transition_status_and_checkpoint",
    "admit_session_invocation",
    "admit_execution_profile_resume",
    "fence_run_and_transform_checkpoint",
    "replace_initial_transcript_messages",
    "append_transcript_messages_and_transform_checkpoint",
    "append_event",
    "append_events",
    "publish_checkpoint_and_events",
    "publish_checkpoint_and_events_with_store_time",
    "_publish_checkpoint_and_events",
    "publish_session_operation",
    "publish_session_operation_guarded",
    "publish_session_operation_guarded_with_store_time",
    "_publish_session_operation",
    "publish_runtime_publication",
    "_publish_runtime_publication_atomic",
    "_prepare_checkpoint_store_unlocked",
    "_apply_checkpoint_store_unlocked",
    "_store_checkpoint_unlocked",
    "_upsert_checkpoint",
    "_prepare_event_append_unlocked",
    "_append_events_unlocked",
    "_append_events_with_cursor",
    "_load_checkpoint_unlocked",
    "_load_checkpoint",
    "_load_unlocked",
    "_load_for_update",
    "_publish_completion_result_event_publication",
    "delete_session",
    "validate_session_closure_admission",
    "claim_session_closure_progress",
    "_require_session_erasure_quiescence_unlocked",
    "_require_session_erasure_quiescence",
    "_run_write",
    "_run_read",
    "_connection",
)


def _session_export_methods_owned(store: object) -> bool:
    """An override must explicitly re-attest the complete export owner boundary."""
    return _session_owner_methods_owned(
        store, "session_export_version", _SESSION_EXPORT_OWNER_METHODS
    )


def _session_continuation_methods_owned(store: object) -> bool:
    return _session_owner_methods_owned(
        store,
        "session_continuation_version",
        (
            *_SESSION_EXPORT_OWNER_METHODS,
            "_publish_continuation_operation",
            "_publish_temporary_continuation_service",
            "_publish_temporary_service_target",
            "_prepare_temporary_side_service",
            "_load_temporary_service_target",
            "_load_temporary_continuation_service",
            "_read_temporary_continuation_outcome",
            "_reconcile_temporary_continuation_service",
            "_initialize_continuation_namespace",
            "prepare_continuation_ticket",
            "load_continuation_ticket",
            "lookup_continuation_ticket",
            "mark_continuation_waiting",
            "_recover_continuation_writer",
            "latch_continuation",
            "consume_continuation",
            "_consume_continuation",
            "_claim_continuation_admission",
            "_release_continuation_admission_claim",
            "_finalize_continuation_admission",
            "retire_continuation",
            "apply_invocation_lifecycle_command",
        ),
    )


def _session_owner_methods_owned(
    store: object, version_field: str, methods: tuple[str, ...]
) -> bool:
    mro = type(store).__mro__
    capability_owner = next(
        (index for index, owner in enumerate(mro) if version_field in vars(owner)),
        len(mro),
    )
    if capability_owner == len(mro):
        return False
    instance_fields = vars(store)
    if version_field in instance_fields:
        return False
    for method in methods:
        method_owner = next(
            (index for index, owner in enumerate(mro) if method in vars(owner)), len(mro)
        )
        # SQLite's _connection is an owned resource field, whereas PostgreSQL's
        # is a method. Only the latter can be shadowed as a method override.
        if method in instance_fields and not (method == "_connection" and method_owner == len(mro)):
            return False
        if method_owner < capability_owner:
            return False
    return True


def _continuation_writer_was_released(
    session: Session,
    checkpoint: dict[str, Any] | None,
) -> bool:
    """Allow exactly one post-release epoch without accepting a new writer."""

    active = active_invocation_execution_profile_from_checkpoint(checkpoint)
    return active is not None and active_invocation_execution_profile_is_released(
        active,
        session_id=session.id,
        run_epoch=session.run_epoch,
    )


class SessionStore(ABC):
    """Persistent store for sessions and append-only events.

    Implementations own the run-epoch contract. A successful transition to
    ``RUNNING`` claims a new epoch for the current execution context. Runtime
    progress writes made by that context must reject a stale epoch with
    ``SessionRunFenced``. Releasing the claim revokes the durable epoch before
    clearing task-local ownership so inherited child contexts cannot write late.

    Implementations also own the clock used for inactivity, recovery-claim,
    epoch-takeover, and activity-stamp decisions. Callers pass durations. The
    store samples, compares, and stamps time inside one atomic boundary; a
    caller or worker clock is never durable ownership authority.
    """

    # Scoped access is an explicit backend protocol, not a filtering wrapper.
    session_access_version: ClassVar[int | None] = None

    async def _access_create_session(self, bounds, request, identity):
        raise NotImplementedError("Atomic scoped session creation is unsupported.")

    async def _access_update_metadata(self, bounds, session_id, metadata):
        raise NotImplementedError("Atomic scoped metadata mutation is unsupported.")

    async def _access_delete_session(self, bounds, session_id):
        raise NotImplementedError("Atomic scoped session deletion is unsupported.")

    async def _access_read_records(self, bounds, session_id, kind, offset, limit, max_bytes):
        raise NotImplementedError("Atomic scoped record reads are unsupported.")

    async def _access_load_session(self, bounds: _SessionAccessBounds, session_id: str) -> Session:
        raise NotImplementedError("Atomic scoped session access is unsupported.")

    async def _access_list_sessions(
        self, bounds: _SessionAccessBounds, query: SessionQuery
    ) -> SessionListResult:
        raise NotImplementedError("Atomic scoped session access is unsupported.")

    async def _access_update_labels(
        self, bounds: _SessionAccessBounds, session_id: str, labels: dict[str, str]
    ) -> Session:
        raise NotImplementedError("Atomic scoped session access is unsupported.")

    # Custom stores must opt into optional capabilities explicitly. Conservative
    # defaults keep discovery truthful for inherited methods that fail closed.
    session_export_version: ClassVar[int] = 0
    session_continuation_version: ClassVar[int] = 0
    external_wait_version: ClassVar[int] = 0
    _producer_attachment_version: ClassVar[int] = 0
    supports_usage_aggregates: ClassVar[bool] = False
    supports_private_argument_continuity: ClassVar[bool] = False
    supports_mcp_manifest_history: ClassVar[bool] = False
    supports_session_topology: ClassVar[bool] = False
    supports_session_lineage: ClassVar[bool] = False
    child_session_notification_version: ClassVar[int | None] = None
    supports_public_authority_aliases: ClassVar[bool] = False
    supports_targeted_tool_grants: ClassVar[bool] = False
    supports_incremental_terminal_evidence: ClassVar[bool] = False
    supports_terminal_session_evidence: ClassVar[bool] = False
    supports_runner_owned_interrupted_evidence: ClassVar[bool] = False
    supports_execution_profile_admission: ClassVar[bool] = False
    supports_active_invocation_execution_profiles: ClassVar[bool] = False
    # Versioned proof for the frozen invocation-lifecycle command boundary.
    # Unlike the historical booleans, an explicit integer lets the runtime
    # reject partial, legacy, and unknown-future store implementations before
    # delivering a command they cannot interpret atomically.
    invocation_lifecycle_command_version: ClassVar[int | None] = None
    terminal_interaction_publication_version: ClassVar[int | None] = None
    durable_model_terminalization_version: ClassVar[int | None] = None
    queued_interaction_profile_handoff_version: ClassVar[int | None] = None
    session_message_lifecycle_version: ClassVar[int | None] = None
    session_steering_version: ClassVar[int | None] = None
    supports_pending_session_initial_checkpoint: ClassVar[bool] = False
    supports_profiled_forks: ClassVar[bool] = False
    supports_atomic_session_operation_initialization: ClassVar[bool] = False
    supports_atomic_model_completion_stage_release: ClassVar[bool] = False
    # Session closure erasure must leave a durable, replayable receipt even
    # after the session row has been removed.  Stores opt in only when they can
    # persist that receipt as part of their deletion/recovery boundary.
    supports_session_closure_receipts: ClassVar[bool] = False
    supports_session_closure_detachment: ClassVar[bool] = False
    supports_session_closure_recursive_deletion: ClassVar[bool] = False
    supports_session_closure_progress: ClassVar[bool] = False
    # Storage retention is an explicit operator capability, never implicit.
    supports_storage_retention: ClassVar[bool] = False
    model_completion_recovery_fence_version: ClassVar[int] = 0
    model_failover_stage_version: ClassVar[int] = 0
    supports_completion_result_event_publication_reservations: ClassVar[bool] = False
    supports_transcript_search: ClassVar[bool] = False
    supports_recall_evidence: ClassVar[bool] = False
    supports_owned_off_thread_session_commit_guards: ClassVar[bool] = False
    service_durability: RuntimeStoreDurability = RuntimeStoreDurability.UNVERIFIED

    participant_session_binding_version: ClassVar[int | None] = None
    recipient_continuation_selection_version: ClassVar[int | None] = None
    context_view_selection_fence_version: ClassVar[int | None] = None
    context_view_version: ClassVar[int | None] = None
    # Versioned proof for authenticated peer-content append/readback.  Stores
    # must implement the exact append-key CAS before advertising this value.
    peer_content_version: ClassVar[int | None] = None

    async def _prepare_session_creation_target(self, target, *, authority):
        raise NotImplementedError("This SessionStore does not support creation decisions.")

    async def _register_session_creation_target(self, target, *, authority):
        raise NotImplementedError("This SessionStore does not support creation decisions.")

    async def _exclude_session_creation_target(self, target, *, authority):
        raise NotImplementedError("This SessionStore does not support creation decisions.")

    async def _acknowledge_session_creation_settlement(self, target, *, authority):
        raise NotImplementedError("This SessionStore does not support creation settlement.")

    async def read_session_creation_decision(self, target):
        from cayu.collaboration._contracts import ExactUnavailable

        return ExactUnavailable()

    async def list_pending_session_creations(self, owner, *, cursor=None, limit=32):
        raise NotImplementedError("This SessionStore does not support creation discovery.")

    async def create_participant_owned_session(
        self,
        creation_request: ParticipantSessionCreationRequest,
        *,
        resolved_request: RunRequest,
        identity: SessionIdentity,
        binding_factory: Callable[
            [Session], tuple[ParticipantSessionBinding, ParticipantSessionCreationReceipt]
        ],
        recipient_provenance: object | None = None,
        recipient_selection: _ContextViewSelectionReceipt | None = None,
        creation_target=None,
        recipient_receipt_validator: Callable[[Session, ParticipantSessionCreationReceipt], object]
        | None = None,
    ) -> tuple[Session, ParticipantSessionCreationReceipt]:
        raise NotImplementedError(
            "This SessionStore does not support participant-owned session creation."
        )

    async def load_participant_session_binding(
        self, session_id: str
    ) -> ParticipantSessionBinding | None:
        raise NotImplementedError("This SessionStore does not support participant bindings.")

    async def _scan_participant_session_bindings(self, participant, *, after=None, limit=32):
        """Owner-internal inventory; callers must authenticate current participant access."""
        raise NotImplementedError("This SessionStore does not qualify participant discovery.")

    async def load_participant_session_creation_receipt(
        self, session_id: str
    ) -> ParticipantSessionCreationReceipt | None:
        raise NotImplementedError("This SessionStore does not support participant receipts.")

    async def lookup_participant_session_creation(
        self, creation_request: ParticipantSessionCreationRequest
    ) -> tuple[Session, ParticipantSessionCreationReceipt] | None:
        raise NotImplementedError("This SessionStore does not support creation-key lookup.")

    async def capture_context_view_publication_source(
        self, session_id: str
    ) -> ContextViewPublicationSource:
        """Capture a bounded completed-turn snapshot under one native read boundary."""
        raise NotImplementedError("This SessionStore cannot capture context-view publication.")

    async def _capture_completed_turn_snapshot(self, session_id: str) -> CompletedTurnSnapshot:
        raise NotImplementedError("This SessionStore cannot capture a current completed turn.")

    async def capture_recipient_continuation(
        self, session_id: str
    ) -> RecipientContinuationSelection:
        """Select exact identity without acquiring a writer or execution authority.

        Wrappers must independently qualify this coherent native read. Inheriting
        an implementation is not evidence that wrapper-specific state is covered.
        """
        from cayu.sessions._recipient_continuation import (
            require_continuation_selection_store,
            select_continuation,
        )

        require_continuation_selection_store(self)
        session_id = require_clean_nonblank(session_id, "session_id")
        return select_continuation(await self._capture_completed_turn_snapshot(session_id))

    async def publish_context_view(
        self, manifest: _ContextViewManifest, *, publication_key: str
    ) -> _ContextViewManifest:
        raise NotImplementedError("This SessionStore does not support context-view publication.")

    async def lookup_context_view_selection(
        self, selection_key: str
    ) -> _ContextViewSelectionReceipt | None:
        raise NotImplementedError(
            "This SessionStore does not support context-view selection lookup."
        )

    async def lookup_context_view_publication(
        self, publication_key: str
    ) -> _ContextViewManifest | None:
        raise NotImplementedError(
            "This SessionStore does not support context-view publication lookup."
        )

    async def select_context_view(
        self, request: _ContextViewSelectionRequest
    ) -> _ContextViewSelectionReceipt:
        raise NotImplementedError("This SessionStore does not support context-view selection.")

    async def _exclude_context_view_selection(self, request, *, authority):
        """Atomically decide exclusion or return positive prior selection evidence."""
        raise NotImplementedError("This SessionStore cannot exclude context-view selection.")

    async def _reserve_context_view_selection_control(self, request, *, authority):
        """Reserve exact mandatory exclusion capacity before acquiring a view."""
        raise NotImplementedError("This SessionStore cannot reserve selection cleanup capacity.")

    async def read_context_view_selection_decision(self, request):
        """Exact receiving readback, not current disclosure or ownership authority."""
        raise NotImplementedError("This SessionStore cannot read selection decisions.")

    async def _prepare_context_view_selection_target(self, target, *, authority):
        return await self._context_selection_target_control(target, authority=authority)

    async def _register_context_view_selection_target(self, target, *, authority):
        return await self._context_selection_target_control(
            target, authority=authority, register=True
        )

    async def _exclude_context_view_selection_target(self, target, *, authority):
        return await self._context_selection_target_control(
            target, authority=authority, exclude=True
        )

    async def _context_selection_target_control(
        self, target, *, authority, register=False, exclude=False
    ):
        from cayu.sessions._context_selection_fence import (
            ContextViewSelectionTarget,
            require_authority,
            require_selection_fence_store,
        )

        require_authority(authority)
        require_selection_fence_store(self)
        target = ContextViewSelectionTarget.model_validate(target)
        return await self._context_selection_control(
            target.request, authority=authority, exclude=exclude, target=target, register=register
        )

    async def _context_selection_control(
        self, request, *, authority, exclude, target=None, register=False
    ):
        raise NotImplementedError("This SessionStore cannot own selection responsibility.")

    async def _select_context_view_target(self, target, *, authority):
        from cayu.sessions._context_selection_fence import (
            ContextViewSelectionTarget,
            require_authority,
            require_selection_fence_store,
        )

        require_authority(authority)
        require_selection_fence_store(self)
        target = ContextViewSelectionTarget.model_validate(target)
        return await self._select_context_view(target.request, target=target)

    async def _select_context_view(self, request, *, target=None):
        raise NotImplementedError("This SessionStore cannot consume selection responsibility.")

    async def _read_context_view_retention(self, target):
        raise NotImplementedError("This SessionStore cannot authenticate retention settlement.")

    async def _adopt_context_view_target(self, target, request, *, authority):
        from cayu.sessions._context_selection_fence import (
            ContextViewSelectionTarget,
            require_authority,
            require_selection_fence_store,
        )

        require_authority(authority)
        require_selection_fence_store(self)
        target = ContextViewSelectionTarget.model_validate(target)
        return await self._transition_context_view_ownership(request, target=target)

    async def _transition_context_view_ownership(self, request, *, target=None):
        raise NotImplementedError("This SessionStore cannot consume retention responsibility.")

    async def read_context_view(
        self, view_id: str, *, source_session_id: str
    ) -> _ContextViewReadback:
        raise NotImplementedError("This SessionStore does not support context-view readback.")

    async def transition_context_view_ownership(
        self, request: _ContextViewOwnershipRequest
    ) -> _ContextViewSelectionReceipt:
        raise NotImplementedError(
            "This SessionStore does not support context-view ownership transitions."
        )

    async def validate_context_view_source_closure(self, session_id: str) -> None:
        """Refuse source deletion while an active context-view pin needs it."""

        raise NotImplementedError(
            "This SessionStore does not support context-view closure admission."
        )

    async def validate_context_view_compaction(
        self, session_id: str, expected_transcript_cursor: int
    ) -> None:
        """Prove retained view material survives a source-session compaction."""

        raise NotImplementedError(
            "This SessionStore does not support context-view compaction admission."
        )

    async def read_context_view_lifecycle_events(
        self, view_id: str, *, limit: int = 256, owner_participant: _ParticipantRef | None = None
    ) -> tuple[_ContextViewLifecycleEvent, ...]:
        """Read bounded evidence; apply an exact participant filter before limiting."""
        raise NotImplementedError(
            "This SessionStore does not support context-view lifecycle events."
        )

    def durable_state_paths(self) -> tuple[Path, ...]:
        """Return local files that own this store's durable state.

        In-memory and remote stores return no paths. Filesystem-confined
        features must fail closed when the store cannot provide this evidence;
        custom local stores can opt in by overriding this method.
        """

        return ()

    @property
    def public_authority_alias_codec(self) -> PublicAuthorityAliasCodec | None:
        """Return the immutable codec bound to this store, when supported."""

        return None

    async def _terminalize_zero_work_interruption(
        self,
        request: ZeroWorkInterruptionRequest,
    ) -> ZeroWorkInterruptionPublication | None:
        """Atomically prove and terminalize zero work; unsupported stores decline."""
        return None

    async def register_public_authority_alias(
        self,
        public_alias: str,
        *,
        field_name: str,
        private_value: str,
        scope_session_id: str | None = None,
    ) -> None:
        """Persist one public-to-private authority mapping before exposure.

        This optional capability is deliberately explicit. A runtime must not
        publish a one-way public alias unless its store can resolve that alias
        after restart and from another worker.
        """

        raise NotImplementedError(
            "This SessionStore does not support durable public authority aliases."
        )

    async def resolve_public_authority_alias(
        self,
        public_alias: str,
        *,
        field_name: str,
        scope_session_id: str | None = None,
    ) -> str | None:
        """Resolve one exact indexed alias within its authority scope."""

        raise NotImplementedError(
            "This SessionStore does not support durable public authority aliases."
        )

    async def public_authority_private_value_exists(
        self,
        private_value: str,
        *,
        field_name: str,
        scope_session_id: str | None = None,
    ) -> bool:
        """Return positive indexed evidence that private authority exists."""

        raise NotImplementedError(
            "This SessionStore does not support durable public authority aliases."
        )

    async def issue_targeted_tool_grants(
        self,
        session_id: str,
        *,
        expected_run_epoch: int,
        records: tuple[TargetedToolGrantRecord, ...],
        events: tuple[Event, ...],
    ) -> TargetedToolGrantIssueResult:
        """Atomically issue or rejoin one exact interaction-scoped grant batch."""

        raise NotImplementedError(
            "This SessionStore does not support durable targeted tool grants."
        )

    async def list_targeted_tool_grants(
        self,
        session_id: str,
        *,
        interaction_id: str | None = None,
        limit: int = TARGETED_TOOL_GRANT_INSPECTION_MAX_RECORDS,
    ) -> tuple[TargetedToolGrantRecord, ...]:
        """Load a bounded session or interaction grant view for inspection."""

        raise NotImplementedError(
            "This SessionStore does not support durable targeted tool grants."
        )

    async def load_targeted_tool_grant_state(
        self,
        session_id: str,
    ) -> TargetedToolGrantStateSnapshot:
        """Load complete typed grant state for a trusted durable export."""

        raise NotImplementedError(
            "This SessionStore does not support durable targeted tool grants."
        )

    async def bind_targeted_tool_grant_use(
        self,
        request: TargetedToolUseRequest,
        *,
        observed_at: datetime,
    ) -> TargetedToolUseResult:
        """Atomically consume one budget unit or rejoin its exact prior binding."""

        raise NotImplementedError(
            "This SessionStore does not support durable targeted tool grants."
        )

    async def revoke_targeted_tool_grant(
        self,
        tool_ref: str,
        *,
        session_id: str,
        expected_run_epoch: int,
        reason: str,
        revoked_at: datetime,
    ) -> TargetedToolGrantRecord | None:
        """Atomically revoke one exact session-scoped reference, idempotently."""

        raise NotImplementedError(
            "This SessionStore does not support durable targeted tool grants."
        )

    async def reconstruct_targeted_tool_grants(
        self,
        session_id: str,
        *,
        expected_run_epoch: int,
        interaction_id: str,
        generation_id: str,
        agent_name: str,
        task_id: str | None,
        environment_name: str | None,
        principal: str | None,
        tenant: str | None,
        catalogue_revision: str,
        descriptors_by_id: Mapping[str, tuple[str, str, str]],
        capability_ceiling_names: frozenset[str],
        observed_at: datetime,
    ) -> TargetedToolGrantReconstructionResult:
        """Validate durable grants against current bounded registration evidence."""

        raise NotImplementedError(
            "This SessionStore does not support durable targeted tool grants."
        )

    @abstractmethod
    async def create(
        self,
        request: RunRequest,
        *,
        identity: SessionIdentity,
        interaction_started_event: Event | None = None,
        interaction_source_messages: list[Message] | None = None,
        checkpoint_transform: CheckpointTransform | None = None,
        result_checkpoint_transform: CheckpointTransform | None = None,
        operation_initializer: SessionOperationInitializer | None = None,
        participant_binding_factory: Callable[
            [Session], tuple[ParticipantSessionBinding, ParticipantSessionCreationReceipt]
        ]
        | None = None,
        participant_request_commitment: str | None = None,
        participant_provenance: object | None = None,
        recipient_selection: _ContextViewSelectionReceipt | None = None,
        creation_target=None,
        recipient_receipt_validator: Callable[[Session, ParticipantSessionCreationReceipt], object]
        | None = None,
    ) -> Session:
        """Create a session, optionally admitting its first interaction atomically.

        Implementations must mint the final session ID, load any requested parent,
        and call ``session_invocation_for_run_request`` inside the same create
        boundary. The returned immutable invocation record is persisted with the
        session and preserved by all later updates.

        With admission, create the session directly in ``RUNNING``, claim its
        first run epoch, and persist the start event plus deferred source batch
        in the same transaction. Both optional arguments are supplied together.
        ``checkpoint_transform`` projects initial authority inside the same
        transaction. Without admission, stores that advertise
        ``supports_pending_session_initial_checkpoint`` invoke it with a
        ``PENDING`` session and no current checkpoint. Stores that do not
        advertise that capability must reject such a request before creating
        the session.

        ``operation_initializer`` receives the final detached session and
        synchronously returns bounded operation records that must commit
        atomically with it. The callback runs inside the creation boundary and
        must be deterministic and side-effect-free. Stores must reject the
        callback or any record before exposing a partial session.
        """

    @abstractmethod
    async def create_fork(
        self,
        *,
        source_session_id: str,
        fork: Session,
        source_statuses: set[SessionStatus],
        transcript_cursor: int | None,
        checkpoint_transform: CheckpointTransform | None,
        system_prompt_replacement: ForkSystemPromptReplacement | None = None,
        expected_source_run_epoch: int,
        operation_initializer: SessionOperationInitializer | None = None,
    ) -> Session:
        """Create a forked session with copied transcript/checkpoint state.

        A prompt replacement, when supplied by the runtime, atomically removes
        inherited system messages and installs its one authoritative system
        message while preserving non-system message attribution.
        """

    async def create_fork_with_transcript_validation(
        self,
        *,
        source_session_id: str,
        fork: Session,
        source_statuses: set[SessionStatus],
        transcript_cursor: int | None,
        checkpoint_transform: CheckpointTransform | None,
        system_prompt_replacement: ForkSystemPromptReplacement | None = None,
        expected_source_run_epoch: int,
        transcript_validator: ForkTranscriptValidator,
        operation_initializer: SessionOperationInitializer | None = None,
    ) -> Session:
        """Create a fork after validating the final transcript inside the copy boundary.

        Custom stores must opt into this safety capability explicitly. The
        conservative default prevents the runtime from silently falling back to
        a time-of-check/time-of-use transcript validation. When prompt
        replacement is requested, the validator receives the atomically
        transformed child transcript rather than the source transcript.
        """

        raise NotImplementedError(
            "This SessionStore does not support atomic fork transcript validation."
        )

    async def create_profiled_fork(
        self,
        *,
        source_session_id: str,
        fork: Session,
        source_statuses: set[SessionStatus],
        transcript_cursor: int | None,
        checkpoint_transform: CheckpointTransform | None,
        system_prompt_replacement: ForkSystemPromptReplacement | None = None,
        expected_source_run_epoch: int,
        relationship: SessionForkProfileRelationship,
        events: list[Event],
        transcript_validator: ForkTranscriptValidator | None = None,
        checkpoint_authority_decoder: ForkCheckpointAuthorityDecoder | None = None,
        operation_initializer: SessionOperationInitializer | None = None,
    ) -> ProfiledSessionForkResult:
        """Atomically create a child and its immutable profile evidence."""

        raise NotImplementedError(
            "This SessionStore does not support atomic profiled session forks."
        )

    @abstractmethod
    async def load(self, session_id: str) -> Session | None:
        """Load a session by id."""

    supports_session_execution = False

    async def _inspect_session_execution_owner(self, session_id: str) -> SessionExecutionState:
        """Internal liveness admission; custom stores without it remain conservative."""
        return await self.inspect_session_execution(session_id)

    async def inspect_session_execution(self, session_id: str) -> SessionExecutionState:
        """Read observational liveness; custom stores without leases return unknown."""
        session = await self.load(session_id)
        if session is None:
            raise KeyError(f"Session not found: {session_id}")
        return SessionExecutionState(
            session_id=session.id,
            run_epoch=session.run_epoch,
            state="terminal"
            if session.status
            in {SessionStatus.COMPLETED, SessionStatus.FAILED, SessionStatus.INTERRUPTED}
            else "unknown",
        )

    @abstractmethod
    async def load_state(self, session_id: str) -> SessionStateSnapshot | None:
        """Load bounded mutable state without labels or unbounded metadata."""

    @abstractmethod
    async def load_invocation_snapshot(
        self,
        session_id: str,
    ) -> SessionInvocationSnapshot | None:
        """Load bounded immutable provenance without labels or metadata."""

    @abstractmethod
    async def inspect_identity(self, session_id: str) -> SessionInspectionIdentity:
        """Load bounded session identity without materializing session metadata."""

    @abstractmethod
    async def update_status(self, session_id: str, status: SessionStatus) -> Session:
        """Update session status and return the updated session."""

    @abstractmethod
    async def transition_status(
        self,
        session_id: str,
        *,
        from_statuses: set[SessionStatus],
        to_status: SessionStatus,
    ) -> Session:
        """Atomically transition status, claiming a new epoch when entering RUNNING.

        Entering ``RUNNING`` must reject an incomplete-recovery claim whose
        lease has expired at the store's authoritative transaction time.
        """

    @abstractmethod
    async def transition_status_and_checkpoint(
        self,
        session_id: str,
        *,
        from_statuses: set[SessionStatus],
        to_status: SessionStatus,
        checkpoint_transform: CheckpointTransform | None = None,
        store_time_checkpoint_transform: StoreTimeCheckpointTransform | None = None,
        result_checkpoint_transform: CheckpointTransform | None = None,
        interaction_started_event: Event | None = None,
        interaction_source_messages: list[Message] | None = None,
        continued_interaction_id: str | None = None,
        defer_interaction_source: bool = False,
        model_transition: SessionModelTransition | None = None,
        execution_profile: ExecutionProfileIdentity | None = None,
        execution_profile_decision: ExecutionProfileDecision | None = None,
        adopted_runtime_identity: SessionRuntimeIdentity | None = None,
        tool_capability_ceiling: ToolCapabilityCeiling | None = None,
        expected_latest_interaction_event_id: str | None = None,
        require_no_active_model_completion_dispatch: bool = False,
        temporary_service_admission: TemporaryServiceAdmission | None = None,
    ) -> Session:
        """Atomically persist status, checkpoint, interaction, target, and profile admission.

        Entering ``RUNNING`` must reject an incomplete-recovery claim whose
        lease has expired at the store's authoritative transaction time.
        """

    async def admit_execution_profile_resume(
        self,
        session_id: str,
        *,
        from_statuses: set[SessionStatus],
        to_status: SessionStatus,
        checkpoint_transform: CheckpointTransform | None = None,
        store_time_checkpoint_transform: StoreTimeCheckpointTransform | None = None,
        result_checkpoint_transform: CheckpointTransform | None = None,
        execution_profile: ExecutionProfileIdentity,
        tool_capability_ceiling: ToolCapabilityCeiling | None = None,
        execution_profile_decision: ExecutionProfileDecision | None = None,
        adopted_runtime_identity: SessionRuntimeIdentity | None = None,
        interaction_started_event: Event | None = None,
        interaction_source_messages: list[Message] | None = None,
        continued_interaction_id: str | None = None,
        defer_interaction_source: bool = False,
        model_transition: SessionModelTransition | None = None,
        temporary_service_admission: TemporaryServiceAdmission | None = None,
    ) -> Session:
        """Atomically compare a profile through the normal resume transition."""

        if not self.supports_execution_profile_admission:
            raise NotImplementedError(
                "This SessionStore does not support atomic execution-profile admission."
            )
        transition_kwargs: dict[str, Any] = {
            "from_statuses": from_statuses,
            "to_status": to_status,
            "checkpoint_transform": checkpoint_transform,
            "result_checkpoint_transform": result_checkpoint_transform,
            "interaction_started_event": interaction_started_event,
            "interaction_source_messages": interaction_source_messages,
            "continued_interaction_id": continued_interaction_id,
            "defer_interaction_source": defer_interaction_source,
            "model_transition": model_transition,
            "execution_profile": execution_profile,
            "execution_profile_decision": execution_profile_decision,
            "tool_capability_ceiling": tool_capability_ceiling,
        }
        if adopted_runtime_identity is not None:
            transition_kwargs["adopted_runtime_identity"] = adopted_runtime_identity
        if temporary_service_admission is not None:
            transition_kwargs["temporary_service_admission"] = temporary_service_admission
        if store_time_checkpoint_transform is not None:
            transition_kwargs["store_time_checkpoint_transform"] = store_time_checkpoint_transform
        return await self.transition_status_and_checkpoint(session_id, **transition_kwargs)

    async def admit_session_invocation(
        self,
        session_id: str,
        *,
        admission: SessionInvocationAdmission,
    ) -> Session:
        """Atomically claim one valid profiled invocation and its interaction."""

        if not self._supports_active_invocation_execution_profile_protocol():
            raise NotImplementedError(
                "This SessionStore does not support atomic active-invocation execution profiles."
            )
        if not self.supports_execution_profile_admission:
            raise NotImplementedError(
                "This SessionStore does not support atomic execution-profile admission."
            )
        if type(admission) is not SessionInvocationAdmission:
            raise TypeError("admission must be a SessionInvocationAdmission.")
        if not checkpoint_preservation._INVOCATION_LIFECYCLE_AUTHORITY_MUTATION_ALLOWED.get():
            raise SessionRunFenced(
                "Invocation admission must use the typed lifecycle command boundary."
            )
        session_id = require_clean_nonblank(session_id, "session_id")
        active_profile = admission.expected_active_invocation_profile
        if active_profile is not None and active_profile.session_id != session_id:
            raise ValueError("Active invocation profile belongs to another session.")
        interaction_id = (
            admission.continued_interaction_id
            if admission.interaction_started_event is None
            else admission.interaction_started_event.interaction_id
        )
        if interaction_id is None:
            raise ValueError("Invocation admission has no interaction identity.")

        def bind_active_invocation_profile(
            current_session: Session,
            checkpoint: dict[str, Any] | None,
            now: datetime | None = None,
        ) -> dict[str, Any]:
            current_active_profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
            if current_active_profile is not None and not (
                active_invocation_execution_profile_is_released(
                    current_active_profile,
                    session_id=current_session.id,
                    run_epoch=current_session.run_epoch,
                )
            ):
                raise SessionRunFenced(
                    "The previous invocation still owns terminal hooks or trailing cleanup."
                )
            if admission.store_time_checkpoint_transform is not None:
                assert now is not None
                updated = admission.store_time_checkpoint_transform(
                    current_session, checkpoint, now
                )
            else:
                assert admission.checkpoint_transform is not None
                updated = admission.checkpoint_transform(current_session, checkpoint)
            return checkpoint_with_active_invocation_execution_profile(
                updated,
                session_id=current_session.id,
                interaction_id=interaction_id,
                run_epoch=current_session.run_epoch + 1,
                profile=admission.execution_profile,
                expected=active_profile,
            )

        transition_kwargs: dict[str, Any] = {
            "from_statuses": set(admission.from_statuses),
            "to_status": SessionStatus.RUNNING,
            "checkpoint_transform": bind_active_invocation_profile,
            "interaction_started_event": admission.interaction_started_event,
            "interaction_source_messages": list(admission.interaction_source_messages),
            "continued_interaction_id": admission.continued_interaction_id,
            "defer_interaction_source": admission.defer_interaction_source,
            "model_transition": admission.model_transition,
            "execution_profile_decision": admission.execution_profile_decision,
            "tool_capability_ceiling": admission.tool_capability_ceiling,
        }
        if admission.store_time_checkpoint_transform is not None:
            del transition_kwargs["checkpoint_transform"]
            transition_kwargs["store_time_checkpoint_transform"] = bind_active_invocation_profile
        if admission.adopted_runtime_identity is not None:
            transition_kwargs["adopted_runtime_identity"] = admission.adopted_runtime_identity
        if active_profile is not None and admission.interaction_started_event is not None:
            transition_kwargs["execution_profile"] = admission.execution_profile
        if admission.result_checkpoint_transform is not None:
            transition_kwargs["result_checkpoint_transform"] = admission.result_checkpoint_transform
        if admission.temporary_service_admission is not None:
            transition_kwargs["temporary_service_admission"] = admission.temporary_service_admission
        if active_profile is None:
            return await self.admit_execution_profile_resume(
                session_id,
                execution_profile=admission.execution_profile,
                **transition_kwargs,
            )
        return await self.transition_status_and_checkpoint(
            session_id,
            **transition_kwargs,
        )

    def _supports_invocation_lifecycle_command_protocol(self) -> bool:
        """Authenticate one complete version-1 command implementation."""

        method_names = (
            "apply_invocation_lifecycle_command",
            "load",
            "load_checkpoint",
            "transform_checkpoint",
            "update_labels",
            "update_metadata",
            "load_interaction_transition_receipt",
            "load_invocation_settlement_transition",
            "_load_interaction_transition_receipt_by_event_id",
            "create",
            "admit_session_invocation",
            "admit_execution_profile_resume",
            "transition_status_and_checkpoint",
            "fence_run_and_transform_checkpoint",
            "reject_active_invocation_execution_profile",
            "reject_execution_profile_resume",
            "publish_interaction_transition",
            "settle_session_invocation",
            "release_session_invocation",
        )
        store_type = type(self)
        mro = type.__getattribute__(store_type, "__mro__")
        concrete_declarations = type.__getattribute__(store_type, "__dict__")
        if concrete_declarations.get("invocation_lifecycle_command_version") != 1:
            return False

        getattribute_owner = next(
            (
                owner
                for owner in mro
                if "__getattribute__" in type.__getattribute__(owner, "__dict__")
            ),
            None,
        )
        if getattribute_owner is None or (
            type.__getattribute__(getattribute_owner, "__dict__")["__getattribute__"]
            is not object.__getattribute__
        ):
            return False
        try:
            instance_state = object.__getattribute__(self, "__dict__")
        except AttributeError:
            instance_state = None
        protected_names = (*method_names, "invocation_lifecycle_command_version")
        if isinstance(instance_state, dict) and any(
            name in instance_state for name in protected_names
        ):
            return False

        for method_name in method_names:
            owner = next(
                (
                    candidate
                    for candidate in mro
                    if method_name in type.__getattribute__(candidate, "__dict__")
                ),
                None,
            )
            if owner is None:
                return False
            declarations = type.__getattribute__(owner, "__dict__")
            implementation = declarations[method_name]
            if type(implementation) is not FunctionType:
                return False
            if owner is not SessionStore and (
                declarations.get("invocation_lifecycle_command_version") != 1
            ):
                return False
            resolved = object.__getattribute__(self, method_name)
            if (
                type(resolved) is not MethodType
                or resolved.__self__ is not self
                or resolved.__func__ is not implementation
            ):
                return False
        return True

    @overload
    async def apply_invocation_lifecycle_command(
        self,
        command: CreateInvocationCommand | AdmitInvocationCommand | RebindInvocationCommand,
    ) -> InvocationMutationResult: ...

    @overload
    async def apply_invocation_lifecycle_command(
        self,
        command: RejectInvocationCommand,
    ) -> ExecutionProfileRejectionResult: ...

    @overload
    async def apply_invocation_lifecycle_command(
        self,
        command: SettleInvocationCommand,
    ) -> InteractionTransitionResult: ...

    @overload
    async def apply_invocation_lifecycle_command(
        self,
        command: ReleaseInvocationCommand,
    ) -> InvocationReleaseResult: ...

    async def apply_invocation_lifecycle_command(
        self,
        command: object,
    ) -> object:
        """Apply one command only after authenticating the store protocol."""

        if not self._supports_invocation_lifecycle_command_protocol():
            raise NotImplementedError(
                "This SessionStore does not support invocation lifecycle command version 1."
            )
        from cayu.runtime._invocation_lifecycle import (
            apply_invocation_lifecycle_command,
        )

        return await apply_invocation_lifecycle_command(
            self,
            cast("InvocationLifecycleCommand", command),
        )

    async def settle_session_invocation(self, command: Any) -> InteractionTransitionResult:
        """Atomically settle an interaction under exact active authority."""

        del command
        raise NotImplementedError(
            "This SessionStore does not support invocation settlement commands."
        )

    async def release_session_invocation(self, command: Any) -> Any:
        """Atomically revoke one exact active invocation epoch."""

        del command
        raise NotImplementedError("This SessionStore does not support invocation release commands.")

    def _supports_active_invocation_execution_profile_protocol(self) -> bool:
        """Return whether the advertised capability covers the active store methods."""

        capability_owner_index = next(
            index
            for index, owner in enumerate(type(self).__mro__)
            if "supports_active_invocation_execution_profiles" in owner.__dict__
        )
        rejection_owner_index = next(
            index
            for index, owner in enumerate(type(self).__mro__)
            if "reject_execution_profile_resume" in owner.__dict__
        )
        return (
            self.supports_active_invocation_execution_profiles is True
            and capability_owner_index <= rejection_owner_index
        )

    def _supports_terminal_interaction_publication_protocol(self) -> bool:
        """Authenticate the optional paired terminal-publication extension."""

        method_names = (
            "transition_status_and_checkpoint",
            "publish_checkpoint_and_events",
            "publish_interaction_transition",
            "_mark_model_completion_stage_dispatched_atomic",
        )
        store_type = type(self)
        mro = type.__getattribute__(store_type, "__mro__")
        concrete_declarations = type.__getattribute__(store_type, "__dict__")
        if concrete_declarations.get("terminal_interaction_publication_version") != 1:
            return False

        getattribute_owner = next(
            (
                owner
                for owner in mro
                if "__getattribute__" in type.__getattribute__(owner, "__dict__")
            ),
            None,
        )
        if getattribute_owner is None or (
            type.__getattribute__(getattribute_owner, "__dict__")["__getattribute__"]
            is not object.__getattribute__
        ):
            return False
        try:
            instance_state = object.__getattribute__(self, "__dict__")
        except AttributeError:
            instance_state = None
        protected_names = (*method_names, "terminal_interaction_publication_version")
        if isinstance(instance_state, dict) and any(
            name in instance_state for name in protected_names
        ):
            return False

        for method_name in method_names:
            owner = next(
                (
                    candidate
                    for candidate in mro
                    if method_name in type.__getattribute__(candidate, "__dict__")
                ),
                None,
            )
            if owner is None:
                return False
            declarations = type.__getattribute__(owner, "__dict__")
            implementation = declarations[method_name]
            if type(implementation) is not FunctionType:
                return False
            if owner is not SessionStore and (
                declarations.get("terminal_interaction_publication_version") != 1
            ):
                return False
            resolved = object.__getattribute__(self, method_name)
            if (
                type(resolved) is not MethodType
                or resolved.__self__ is not self
                or resolved.__func__ is not implementation
            ):
                return False
        return True

    async def reject_execution_profile_resume(
        self,
        session_id: str,
        *,
        expected_session_instance_id: str | None = None,
        expected_statuses: set[SessionStatus],
        expected_run_epoch: int,
        expected_profile: ExecutionProfileIdentity,
        candidate_profile: ExecutionProfileIdentity,
        event: Event,
        decision: ExecutionProfileDecision | None = None,
        expected_active_invocation_profile_authority: CheckpointValueAuthority | None = None,
    ) -> ExecutionProfileRejectionResult:
        """Atomically reject a changed resume profile without admitting work."""

        raise NotImplementedError(
            "This SessionStore does not support atomic execution-profile rejection."
        )

    async def reject_active_invocation_execution_profile(
        self,
        session_id: str,
        *,
        expected_session_instance_id: str | None = None,
        expected_statuses: set[SessionStatus],
        expected_run_epoch: int,
        expected_active_invocation_profile: ActiveInvocationExecutionProfile,
        candidate_profile: ExecutionProfileIdentity,
        event: Event,
        decision: ExecutionProfileDecision | None = None,
    ) -> ExecutionProfileRejectionResult:
        """Atomically compare checkpoint authority and reject changed continuation."""

        if not self._supports_active_invocation_execution_profile_protocol():
            raise NotImplementedError(
                "This SessionStore does not support atomic active-invocation "
                "execution-profile rejection."
            )
        expected_authority = checkpoint_value_authority(
            expected_active_invocation_profile.model_dump(mode="json"),
            "expected_active_invocation_execution_profile",
        )
        return await self.reject_execution_profile_resume(
            session_id,
            expected_session_instance_id=expected_session_instance_id,
            expected_statuses=expected_statuses,
            expected_run_epoch=expected_run_epoch,
            expected_profile=expected_active_invocation_profile.profile,
            candidate_profile=candidate_profile,
            event=event,
            decision=decision,
            expected_active_invocation_profile_authority=expected_authority,
        )

    @abstractmethod
    async def transition_status_if_no_queued_messages(
        self,
        session_id: str,
        *,
        from_statuses: set[SessionStatus],
        to_status: SessionStatus,
        checkpoint_mutation: dict[str, Any] | None = None,
    ) -> Session:
        """Atomically terminalize only when no durable queued input remains.

        A completion-finalization checkpoint mutation keeps RUNNING at the
        current epoch while installing its existing owned finalization marker.
        It does not admit a new invocation or rewrite an interaction receipt.
        """

    @abstractmethod
    async def publish_interaction_transition(
        self,
        session_id: str,
        *,
        event: Event,
        from_statuses: set[SessionStatus],
        to_status: SessionStatus,
        only_if_no_queued_messages: bool = False,
        model_completion_stage_settlement: ModelCompletionStageSettlementRequest | None = None,
        checkpoint_mutation: dict[str, Any] | None = None,
        terminal_event: Event | None = None,
        terminal_decision: InvocationTerminalDecision | None = None,
        expected_session_instance_id: str | None = None,
        expected_active_invocation_profile: ActiveInvocationExecutionProfile | None = None,
        expected_invocation_authority_state: Literal["active", "released"] = "active",
        expected_recovery_claim_id: str | None = None,
        terminalization_only: bool = False,
        terminalization_plan_ownership: Any = None,
    ) -> InteractionTransitionResult:
        """Atomically publish one interaction state and its session transition.

        A completion guarded by ``only_if_no_queued_messages`` always publishes
        the interaction event, but leaves the session status unchanged when
        queued input remains. Repeating the exact complete transition
        reconstructs the committed result without re-evaluating that queue
        boundary. Reusing its event identity with different transition data
        fails closed. Implementations validate a runtime-supplied recovery claim
        against store-owned time after acquiring the mutation lock and before
        first publication, and reject an unclaimed first publication while a
        store-time-live recovery claim exists. Exact receipt replay remains
        valid after the lease has settled. When terminal evidence is supplied,
        the interaction event, session terminal event, terminal status, and
        exact winning decision commit in one transaction.
        """

    async def load_interaction_transition_receipt(
        self,
        session_id: str,
        *,
        transition: InteractionTransitionSpec,
        expected_recovery_claim_id: str | None = None,
    ) -> InteractionTransitionReceiptResult | None:
        """Load immutable evidence for one exact interaction transition.

        Stores that support acknowledgement-loss recovery must implement this
        read-only boundary. The default keeps existing custom stores source
        compatible while making unsupported cancellation reconciliation fail
        closed instead of issuing another transition publication. Claim-bound
        receipts require the exact original claim and reject a superseding
        durable claimant, while ordinary lease expiry does not invalidate
        exact acknowledgement-loss readback.
        """

        del session_id, transition, expected_recovery_claim_id
        raise NotImplementedError(
            "This session store does not support interaction-transition receipt lookup."
        )

    async def load_historical_interaction_settlement(
        self,
        session_id: str,
        *,
        expected_session_instance_id: str,
        expected_event: Event,
        expected_profile: ExecutionProfileIdentity,
    ) -> InteractionTransitionReceiptResult:
        """Read exact prior settlement evidence without granting run authority.

        Unlike invocation release lookup, this permits a later current epoch.
        It cannot release a fence or authorize a historical invocation: callers
        must separately acquire their own execution or mutation authority.
        """
        session_id = require_clean_nonblank(session_id, "session_id")
        expected_session_instance_id = SessionInvocationBinding.validate_session_instance_id(
            expected_session_instance_id
        )
        event = copy_event(expected_event)
        profile = ExecutionProfileIdentity.model_validate(expected_profile.model_dump(mode="json"))
        if event.session_id != session_id or event.interaction_id is None:
            raise ValueError("Historical settlement requires an exact session interaction.")
        record = await self._load_historical_interaction_settlement_record(session_id, event.id)
        if record is None:
            raise RuntimeError("Historical interaction settlement has no immutable receipt.")
        receipt = interaction_receipts._load_interaction_transition_receipt(record)
        current = await self.load(session_id)
        active = receipt.invocation_active_profile
        if (
            current is None
            or current.instance_id != expected_session_instance_id
            or receipt.session.id != session_id
            or receipt.session.instance_id != expected_session_instance_id
            or receipt.invocation_session_instance_id != expected_session_instance_id
            or receipt.event != event
            or active is None
            or active.session_id != session_id
            or active.interaction_id != event.interaction_id
            or active.profile != profile
            or active.run_epoch != receipt.session.run_epoch
            or active.run_epoch >= current.run_epoch
        ):
            raise SessionRunFenced("Historical interaction settlement has conflicting lineage.")
        return InteractionTransitionReceiptResult(
            session=receipt.session,
            transition=interaction_receipts._interaction_transition_spec_from_receipt(receipt),
            status_changed=receipt.status_changed,
        )

    async def _load_historical_interaction_settlement_record(
        self, session_id: str, event_id: str
    ) -> dict[str, Any] | None:
        raise NotImplementedError("This store does not support historical settlement lookup.")

    async def load_invocation_settlement_transition(
        self,
        session_id: str,
        *,
        expected_session_instance_id: str,
        expected_active_invocation_profile: ActiveInvocationExecutionProfile,
    ) -> InteractionTransitionSpec | None:
        """Resolve exact terminal settlement for one active invocation.

        Event lookup is bounded and supplies only the receipt identity. The
        backend authenticates the complete stored receipt and its invocation
        tuple before returning the transition. Release compares the same
        receipt again inside its mutation transaction.
        """

        session_id = require_clean_nonblank(session_id, "session_id")
        expected_session_instance_id = SessionInvocationBinding.validate_session_instance_id(
            expected_session_instance_id
        )
        if type(expected_active_invocation_profile) is not ActiveInvocationExecutionProfile:
            raise TypeError(
                "expected_active_invocation_profile must be an ActiveInvocationExecutionProfile."
            )
        if expected_active_invocation_profile.session_id != session_id:
            raise ValueError("Active invocation settlement belongs to another session.")
        admission_records = await self.query_events(
            EventQuery(
                session_id=session_id,
                interaction_id=expected_active_invocation_profile.interaction_id,
                event_type=EventType.INTERACTION_STARTED,
                order_by=EventOrder.SEQUENCE_DESC,
                limit=1,
            )
        )
        if not admission_records:
            raise RuntimeError(
                "Active invocation authority has no durable interaction admission event."
            )
        records = await self.query_events(
            EventQuery(
                session_id=session_id,
                event_types=(
                    EventType.INTERACTION_PAUSED,
                    EventType.INTERACTION_COMPLETED,
                    EventType.INTERACTION_FAILED,
                    EventType.INTERACTION_INTERRUPTED,
                ),
                after_sequence=admission_records[0].sequence,
                order_by=EventOrder.SEQUENCE_DESC,
                limit=1,
            )
        )
        if not records:
            return None
        receipt = await self._load_interaction_transition_receipt_by_event_id(
            session_id,
            event_id=records[0].event.id,
            expected_session_instance_id=expected_session_instance_id,
            expected_active_invocation_profile=expected_active_invocation_profile,
        )
        if receipt is None:
            raise RuntimeError(
                "Terminal interaction evidence exists without its immutable receipt."
            )
        return receipt.transition

    async def _load_interaction_transition_receipt_by_event_id(
        self,
        session_id: str,
        *,
        event_id: str,
        expected_session_instance_id: str,
        expected_active_invocation_profile: ActiveInvocationExecutionProfile,
    ) -> InteractionTransitionReceiptResult | None:
        """Load one store-authenticated receipt without caller-supplied fields."""

        del (
            session_id,
            event_id,
            expected_session_instance_id,
            expected_active_invocation_profile,
        )
        raise NotImplementedError(
            "This session store does not support invocation settlement receipt lookup."
        )

    @abstractmethod
    async def fence_stalled_run(
        self,
        session_id: str,
        *,
        statuses: set[SessionStatus],
        inactive_for_seconds: int,
    ) -> Session | None:
        """Atomically evict a stale run and claim its newly incremented epoch.

        Returns ``None`` when the session is no longer in one of ``statuses`` or
        has not been inactive for ``inactive_for_seconds`` at store time. An
        existing incomplete-recovery claim must be reconciled through the exact
        checkpoint-authorized takeover path instead.
        """
        raise NotImplementedError

    @abstractmethod
    async def reserve_stalled_run_recovery(
        self,
        session_id: str,
        *,
        statuses: set[SessionStatus],
        inactive_for_seconds: int | None,
        checkpoint_transform: StoreTimeCheckpointTransform,
    ) -> Session | None:
        """Reserve recovery at store time without yet advancing the run epoch.

        The reservation transform and eligibility check are atomic. The method
        intentionally leaves session activity unchanged so a subsequent typed
        invocation rebind can authenticate and fence the exact reserved state.
        """
        raise NotImplementedError

    @abstractmethod
    async def fence_run_and_transform_checkpoint(
        self,
        session_id: str,
        *,
        statuses: set[SessionStatus],
        checkpoint_transform: CheckpointTransform,
        result_checkpoint_transform: CheckpointTransform | None = None,
    ) -> Session:
        """Atomically transform a checkpoint and claim a newly incremented epoch.

        The transform must return the replacement checkpoint. An exception or
        ``None`` result aborts both the checkpoint update and the epoch fence.
        This operation is for ownership that must persist a checkpoint lease
        and fence the prior run as one transaction. Recovery that needs only an
        inactivity predicate should continue to use :meth:`fence_stalled_run`.
        An incomplete-recovery claim present in the source checkpoint must
        still be live at the store's authoritative transaction time.
        """
        raise NotImplementedError

    @abstractmethod
    async def release_run_fence(self, session_id: str) -> None:
        """Revoke this task's epoch after all trailing writes finish."""
        invocation_scope._deactivate_session_run_fence(
            require_clean_nonblank(session_id, "session_id")
        )

    @abstractmethod
    async def claim_budget_reservation_identity(
        self,
        *,
        reservation_id: str,
        publication_session_id: str,
        publication_id: str,
    ) -> None:
        """Durably bind one ledger-wide reservation id to an exact session event.

        Claims are idempotent only for the same session/event pair and must not
        be removed when a session is deleted. ``append_events`` atomically marks
        the matching claim published when it stores ``budget.reserved``. The
        claim is a session-scoped mutation: the publication session must exist
        and an active run fence must still own its current epoch.
        """

    async def append_event(self, session_id: str, event: Event) -> None:
        """Append one event to a session."""
        await self.append_events(session_id, [event])

    @abstractmethod
    async def append_events(self, session_id: str, events: list[Event]) -> None:
        """Append events to a session in one durable batch."""

    async def append_workflow_step_started(
        self,
        session_id: str,
        event: Event,
        *,
        workflow_name: str,
        attempt_id: str,
    ) -> bool:
        """Atomically append a step reservation only for the current attempt.

        The attempt-fence check and event insert share one store transaction.
        ``False`` means either a newer attempt won or this exact reservation was
        already published.
        """

        raise NotImplementedError(
            f"{type(self).__name__} does not support atomic workflow step reservation."
        )

    async def append_tool_effect_conflict(self, request: object) -> Event:
        """Append exact late-dispatch evidence without granting run authority.

        Implementations compare the retained winner and session incarnation in
        the append transaction. This narrow operation accepts only an internal
        ToolEffectConflictAudit, constructs its own event, preserves liveness
        and all execution state, and returns original evidence on exact replay.
        A stale run context is permitted only for this evidence-only operation.
        """
        raise NotImplementedError("SessionStore does not support tool effect conflict audit.")

    async def load_mcp_manifest_baselines(
        self,
        history_keys: tuple[str, ...],
    ) -> McpManifestBaselineLoadResult:
        """Load authoritative baselines for the requested manifest-history keys."""

        raise NotImplementedError(
            f"{type(self).__name__} must implement durable MCP manifest history."
        )

    async def compare_and_publish_mcp_manifest_checks(
        self,
        session_id: str,
        *,
        expected_generations: dict[str, int | None],
        baseline_updates: dict[str, McpManifestBaseline],
        events: list[Event],
    ) -> McpManifestPublicationResult:
        """Atomically publish decisions and return the exact resulting requested baselines.

        Every accepted ``first_seen`` or ``changed`` event must install the
        matching authoritative baseline in the same publication. Every
        expected history key must have exactly one coherent outcome;
        ``unchanged`` must match its current baseline, and blocked outcomes
        must not update one.
        """

        raise NotImplementedError(
            f"{type(self).__name__} must implement atomic MCP manifest publication."
        )

    @abstractmethod
    async def claim_persisted_event_side_effect(
        self,
        *,
        session_id: str | None = None,
        event_id: str | None = None,
        lease_seconds: float = 300.0,
    ) -> PersistedEventSideEffectClaim | None:
        """Claim the oldest eligible persisted event side-effect handoff."""

    @abstractmethod
    async def get_persisted_event_side_effect_delivery(
        self,
        *,
        session_id: str,
        event_id: str,
    ) -> PersistedEventSideEffectDelivery | None:
        """Load one persisted event side-effect handoff by event identity."""

    async def claim_first_persisted_event_side_effect(
        self, expected: PersistedEventSideEffectDelivery
    ) -> PersistedEventSideEffectClaim | None:
        """Claim only the exact unattempted snapshot under the ordinary claim lock.

        Never take over a lease or retry a failed delivery. Return None when
        unsupported or when any expected field changed; callers must reread.
        """
        event_delivery_rules._copy_pending_first_event_delivery(expected)
        return None

    async def retire_failed_first_event_delivery(
        self, expected: PersistedEventSideEffectDelivery
    ) -> PersistedEventSideEffectDelivery | None:
        """Atomically dead-letter an exact failed first delivery, preserving its error.

        Compare the complete snapshot under the claim lock. A competing retry
        wins or retirement wins, never both. Return None on conflict or when
        unsupported; that absence is not proof of callback settlement.
        """
        event_delivery_rules._copy_failed_first_delivery_retirement(expected)
        return None

    @abstractmethod
    async def mark_persisted_event_side_effect_delivered(
        self,
        claim: PersistedEventSideEffectClaim,
    ) -> PersistedEventSideEffectDelivery:
        """Mark one claimed event's configured side effects delivered."""

    @abstractmethod
    async def mark_persisted_event_side_effect_failed(
        self,
        claim: PersistedEventSideEffectClaim,
        *,
        error: str,
        max_attempts: int,
        retry_delay_seconds: float,
    ) -> PersistedEventSideEffectDelivery:
        """Release one failed claim for retry or dead-letter it."""

    @abstractmethod
    async def list_persisted_event_side_effect_deliveries(
        self,
        *,
        statuses: set[PersistedEventSideEffectStatus] | None = None,
        claimable_only: bool = False,
        after_sequence: int | None = None,
        limit: int = 100,
    ) -> list[PersistedEventSideEffectDelivery]:
        """Inspect persisted side-effect delivery state in event order.

        ``claimable_only`` excludes terminal deliveries and live leases.
        """

    async def get_persisted_event_side_effect_health(self) -> PersistedEventSideEffectHealth:
        """Read exact aggregate delivery health without loading source events."""
        raise NotImplementedError("This store does not support event side-effect health.")

    async def query_persisted_event_side_effect_deliveries(
        self,
        query: PersistedEventSideEffectQuery,
    ) -> PersistedEventSideEffectPage:
        """Read a bounded operational page; inspection never reserves a claim."""
        raise NotImplementedError("This store does not support event side-effect inspection.")

    async def defer_persisted_event_side_effect(
        self,
        claim: PersistedEventSideEffectClaim,
    ) -> PersistedEventSideEffectDelivery:
        """Release an exact claim without delivery or spending a failure attempt.

        Used when an internal continuation cannot yet acquire execution ownership.
        The persisted event remains eligible for bounded recovery; its old claim
        must no longer be capable of acknowledging a replacement delivery.
        """
        raise NotImplementedError("This SessionStore does not support side-effect deferral.")

    async def renew_persisted_event_side_effect(
        self,
        claim: PersistedEventSideEffectClaim,
        *,
        lease_seconds: float = 300.0,
    ) -> PersistedEventSideEffectDelivery:
        """Extend an exact, still-live claim using the store's ownership clock.

        Expired ownership cannot be revived, even before another worker claims
        the event. Renewal preserves the claim identity and attempt count.
        """
        raise NotImplementedError("This SessionStore does not support side-effect renewal.")

    @abstractmethod
    async def enqueue_session_message(
        self,
        request: EnqueueSessionMessageRequest,
        *,
        expected_authorized_target_instance_id: str | None = None,
    ) -> EnqueueSessionMessageResult:
        """Atomically accept queued input with its causal event, or replay it.

        When supplied by the authorization owner, the target incarnation must
        match under the transaction lock before either replay or admission.
        This fence is not a persisted message freshness condition.
        """

    async def append_peer_content(
        self,
        request: PeerContentAppendRequest,
        *,
        qualify_target: Callable[[Session], None] | None = None,
        pending_transcript_cursor: int | None = None,
    ) -> PeerContentReceipt:
        """Atomically append or replay one authenticated peer occurrence.

        The default is fail-closed: a custom store cannot accidentally expose
        peer content by inheriting a user-message implementation.
        Before queue insertion, native owners call the runtime's synchronous,
        side-effect-free qualifier with a detached transaction-owned target.
        Missing qualification permits only pending/excluded decisions or replay.
        pending_transcript_cursor is a receiving-owner fence for servicing an
        already retained exact pending operation. It never changes caller intent.
        """
        raise NotImplementedError("This SessionStore does not support peer content.")

    async def read_peer_content(
        self,
        append_key: PeerAppendKey,
    ) -> PeerContentReceipt | None:
        """Return the bounded append receipt for an exact key, if present."""
        raise NotImplementedError("This SessionStore does not support peer content.")

    async def exclude_peer_content(
        self,
        request: PeerContentAppendRequest,
        *,
        reason: str,
    ) -> PeerContentReceipt:
        """Record a definitive exclusion without pretending it was appended."""
        raise NotImplementedError("This SessionStore does not support peer content.")

    async def read_peer_content_attempt(
        self, request: PeerContentAppendRequest
    ) -> PeerContentReceipt | None:
        """Reconcile the complete expected operation, including excluded predecessors."""
        raise NotImplementedError("This SessionStore does not support exact peer reconciliation.")

    async def list_pending_peer_content(self, *, after_operation_key=None, limit=32):
        """Discover bounded peer responsibility independently of creation settlement."""
        raise NotImplementedError("This SessionStore does not support peer discovery.")

    async def retry_pending_peer_content(
        self,
        session_id: str,
        *,
        expected_session_instance_id: str,
        expected_run_epoch: int,
        expected_transcript_cursor: int,
        admit=None,
    ) -> tuple[PeerContentReceipt, ...]:
        """Retry retained peer appends only when the existing host reaches a boundary."""
        raise NotImplementedError("This SessionStore does not support peer delivery servicing.")

    async def record_peer_content_exposure(
        self, request: PeerContentExposureRequest
    ) -> PeerContentExposureReceipt:
        raise NotImplementedError("This SessionStore does not support peer exposure receipts.")

    async def begin_peer_content_exposure(
        self, request: PeerContentExposureRequest
    ) -> PeerContentExposureReceipt:
        """Durably reserve an exposure before provider serialization."""
        raise NotImplementedError("This SessionStore does not support peer exposure receipts.")

    async def read_peer_content_exposure(
        self, append_key: PeerAppendKey, exposure_id: str
    ) -> PeerContentExposureReceipt | None:
        raise NotImplementedError("This SessionStore does not support peer exposure receipts.")

    async def inspect_session_messages(
        self,
        query: SessionMessageQuery,
        *,
        expected_authorized_session_instance_id: str | None = None,
    ) -> SessionMessageInspection:
        """Read one bounded delivery-priority page, retaining unreadable row identities.

        Check the optional authorization-owner incarnation before reading queue
        content, including on the first page without a cursor.
        """
        raise NotImplementedError("Session-message inspection is not supported by this store.")

    async def apply_session_message_action(
        self, request: SessionMessageActionRequest
    ) -> SessionMessageActionResult:
        """Atomically withdraw/quarantine an exact undelivered row and publish its event."""
        raise NotImplementedError(
            "Session-message terminal actions are not supported by this store."
        )

    async def snapshot_session_message_source(
        self,
        session_id: str,
        *,
        include_transcript_digest: bool = False,
        include_checkpoint_digest: bool = False,
        expected_authorized_session_instance_id: str | None = None,
    ) -> SessionMessageSource:
        """Capture source identity/cursor and requested digests under one store snapshot.

        Check the optional authorization-owner incarnation before accessing
        transcript or checkpoint content, within that same snapshot.
        """
        raise NotImplementedError(
            "Session-message source snapshots are not supported by this store."
        )

    @abstractmethod
    async def deliver_queued_session_messages(
        self,
        session_id: str,
        *,
        include_on_idle: bool,
        delivery_id: str | None = None,
        eligible_through: int | None = None,
        limit: int = SESSION_MESSAGE_DELIVERY_BATCH_LIMIT,
        interaction_id: str | None = None,
        interaction_started_event: Event | None = None,
        profile_handoff: QueuedInteractionProfileHandoff | None = None,
        reject_only: bool = False,
    ) -> SessionMessageDeliveryBatch:
        """Atomically append and mark one bounded queue batch delivered.

        Runtime callers supply a stable ``delivery_id`` for acknowledgement-loss
        reconstruction. Omitting it creates a one-shot identity for direct store
        callers and returns that identity in the result.

        ``reject_only`` retires stale/expired records without delivering valid
        input, for a runtime that has exhausted its model-step authority. Valid
        records remain pending; ``has_more`` then describes remaining rejection
        work inside the fixed eligibility boundary. The mode is part of the
        exact delivery identity and cannot change on replay.
        """

    async def repair_queued_interaction_profile_handoff(
        self,
        session_id: str,
        *,
        interaction_started_event: Event,
        profile_handoff: QueuedInteractionProfileHandoff,
    ) -> ActiveInvocationExecutionProfile:
        """Repair one historical split delivery from exact provider-stage evidence.

        This compatibility seam is intentionally fail-closed for custom stores.
        New deliveries must bind the target profile in
        :meth:`deliver_queued_session_messages` instead.
        """

        raise NotImplementedError(
            "This SessionStore does not support historical queued handoff repair."
        )

    @abstractmethod
    async def publish_checkpoint_and_events(
        self,
        session_id: str,
        *,
        checkpoint_transform: CheckpointTransform,
        events: list[Event],
        expected_statuses: set[SessionStatus] | None = None,
        expected_run_epoch: int | None = None,
        expected_transcript_cursor: int | None = None,
    ) -> Session:
        """Atomically transform a checkpoint and append its causal event batch."""

    async def publish_checkpoint_and_events_with_store_time(
        self,
        session_id: str,
        *,
        idempotency_key: str,
        checkpoint_transform: StoreTimeCheckpointTransform,
        commit_time_guard: Callable[[datetime], None],
        events: list[Event],
        expected_statuses: set[SessionStatus] | None = None,
        expected_run_epoch: int | None = None,
        expected_transcript_cursor: int | None = None,
    ) -> Session:
        """Publish checkpoint state using the backend's transactional clock."""

        if checkpoint_transform is None:
            raise TypeError("checkpoint_transform is required.")
        if commit_time_guard is None:
            raise TypeError("commit_time_guard is required.")

        def operation_transform(
            session: Session,
            checkpoint: dict[str, Any] | None,
            _record: dict[str, Any] | None,
            store_now: datetime,
        ) -> SessionOperationPublication:
            transformed = checkpoint_transform(session, checkpoint, store_now)
            if transformed is None:
                raise ValueError("Checkpoint transform must return a checkpoint.")
            return SessionOperationPublication(checkpoint=transformed)

        return await self.publish_session_operation_guarded_with_store_time(
            session_id,
            idempotency_key=idempotency_key,
            operation_transform=operation_transform,
            commit_guard=lambda: None,
            commit_time_guard=commit_time_guard,
            events=events,
            expected_statuses=expected_statuses,
            expected_run_epoch=expected_run_epoch,
            expected_transcript_cursor=expected_transcript_cursor,
        )

    async def _publish_completion_result_event_publication(
        self,
        session_id: str,
        *,
        checkpoint_transform: StoreTimeCheckpointTransform,
        events: list[Event],
    ) -> Session:
        """Mutate the private publication root with authoritative store time."""

        raise NotImplementedError(
            "This SessionStore does not own completion-result event publication."
        )

    @abstractmethod
    async def load_session_operation(
        self,
        session_id: str,
        idempotency_key: str,
        *,
        checkpoint_root_guard: CheckpointRootFieldGuard | None = None,
    ) -> dict[str, Any] | None:
        """Load one terminal durable operation record by caller idempotency key.

        A supplied root guard runs in the same read snapshot before the
        checkpoint-coupled operation record is interpreted.
        """

    @abstractmethod
    async def publish_session_operation(
        self,
        session_id: str,
        *,
        idempotency_key: str,
        operation_transform: SessionOperationTransform,
        events: list[Event],
        expected_statuses: set[SessionStatus] | None = None,
        expected_run_epoch: int | None = None,
        expected_transcript_cursor: int | None = None,
    ) -> Session:
        """Atomically publish a checkpoint, events, and terminal operation records."""

    def _supports_session_export_protocol(self) -> bool:
        """Qualify exact owner methods, not an inherited version flag alone."""
        return (
            type(self.session_export_version) is int
            and self.session_export_version == 1
            and _session_export_methods_owned(self)
        )

    def _supports_session_continuation_protocol(self) -> bool:
        return (
            type(self.session_continuation_version) is int
            and self.session_continuation_version == 1
            and _session_continuation_methods_owned(self)
        )

    def supports_external_waits(self) -> bool:
        return (
            type(self.external_wait_version) is int
            and self.external_wait_version == 1
            and _session_owner_methods_owned(
                self,
                "external_wait_version",
                (
                    "_mutate_external_wait",
                    "_read_external_wait",
                    "_list_external_waits",
                    "_retire_external_wait_scope",
                    "_read_external_wait_retirement",
                    "_prune_external_wait_scope",
                    "create",
                    "apply_invocation_lifecycle_command",
                    "_require_external_creation_unlocked",
                    "_require_external_creation",
                    "_require_external_wait_admission_unlocked",
                    "_require_external_wait_admission",
                    "transition_status",
                    "transition_status_and_checkpoint",
                    "_connection",
                    "delete_session",
                    "validate_session_closure_admission",
                    "_require_session_erasure_quiescence_unlocked",
                    "_require_session_erasure_quiescence",
                    "_require_external_wait_quiescence_unlocked",
                ),
            )
        )

    async def _read_external_wait_retirement(
        self, scope: ExternalWaitScope
    ) -> ExternalWaitRetirement | None:
        raise NotImplementedError("Store does not qualify external waits.")

    async def _retire_external_wait_scope(
        self, request: ExternalWaitRetirementRequest
    ) -> ExternalWaitRetirement:
        raise NotImplementedError("Store does not qualify external waits.")

    async def _prune_external_wait_scope(
        self, retirement: ExternalWaitRetirement, *, limit: int
    ) -> ExternalWaitPruneResult:
        raise NotImplementedError("Store does not qualify external waits.")

    async def _mutate_external_wait(self, command: ExternalWaitMutation) -> ExternalWaitRecord:
        raise NotImplementedError("Store does not qualify external waits.")

    async def _read_external_wait(
        self, scope: ExternalWaitScope, correlation_key: str
    ) -> ExternalWaitRecord | None:
        raise NotImplementedError("Store does not qualify external waits.")

    async def _list_external_waits(
        self, scope: ExternalWaitScope, *, source: str, after: str, limit: int
    ) -> tuple[ExternalWaitRecord, ...]:
        raise NotImplementedError("Store does not qualify external waits.")

    def _supports_producer_attachment_protocol(self) -> bool:
        return (
            type(self._producer_attachment_version) is int
            and self._producer_attachment_version == 1
            and self._supports_session_steering_protocol() is True
            and _session_owner_methods_owned(
                self,
                "_producer_attachment_version",
                (
                    *_SESSION_EXPORT_OWNER_METHODS,
                    "_attach_native_producer",
                    "_read_native_producer_attachment",
                    "_admit_native_producer",
                    "_exclude_native_producer",
                    "_read_native_producer_exclusion",
                    "_acknowledge_native_producer_cleanup",
                    "_retain_native_producer_output",
                    "_read_retained_native_producer_output",
                    "_read_native_producer_release",
                    "_read_native_producer_progress",
                    "_complete_native_producer_cleanup",
                    "_read_completed_native_producer_cleanup",
                    "_retire_native_producer_cleanup",
                    "query_events",
                    "load_historical_interaction_settlement",
                    "_load_historical_interaction_settlement_record",
                    "_load_interaction_transition_receipt_by_event_id",
                    "apply_invocation_lifecycle_command",
                ),
            )
        )

    async def _attach_native_producer(self, registration):
        """Registered runtime handoff, not a caller-provided publication callback."""
        from cayu.runtime._producer_output_store import attach_native_producer

        if not self._supports_producer_attachment_protocol():
            raise NotImplementedError("This store does not qualify native producer attachment.")
        return await attach_native_producer(self, registration)

    async def _read_native_producer_attachment(self, command):
        raise NotImplementedError("Atomic native producer attachment readback is not qualified.")

    async def _admit_native_producer(self, registration, command):
        from cayu.runtime._producer_output_store import admit_native_producer

        if not self._supports_producer_attachment_protocol():
            raise NotImplementedError("This store does not qualify native producer admission.")
        return await admit_native_producer(self, registration, command)

    async def _exclude_native_producer(self, registration, control):
        """Cleanup-only registered handoff; never accept a public receipt as authority."""
        from cayu.runtime._producer_output_store import exclude_native_producer

        if not self._supports_producer_attachment_protocol():
            raise NotImplementedError("This store does not qualify native producer exclusion.")
        return await exclude_native_producer(self, registration, control)

    async def _read_native_producer_exclusion(self, registration, control):
        from cayu.runtime._producer_output_store import read_native_exclusion

        if not self._supports_producer_attachment_protocol():
            raise NotImplementedError("This store does not qualify native producer exclusion.")
        return await read_native_exclusion(self, registration, control)

    async def _retain_native_producer_output(self, session_id, *, invocation, stage_id):
        from cayu.runtime._producer_output_store import retain_native_output

        return await retain_native_output(
            self, session_id, invocation=invocation, stage_id=stage_id
        )

    async def _read_retained_native_producer_output(self, command):
        from cayu.runtime._producer_output_store import read_retained_native_output

        return await read_retained_native_output(self, command)

    async def _read_native_producer_release(self, command):
        raise NotImplementedError("Atomic native producer release readback is not qualified.")

    async def _read_native_producer_progress(self, command, *, kind):
        raise NotImplementedError("Atomic native producer progress readback is not qualified.")

    async def _complete_native_producer_cleanup(self, registration, *, authority):
        raise NotImplementedError("Durable native producer cleanup is not qualified.")

    async def _read_completed_native_producer_cleanup(self, registration):
        raise NotImplementedError("Durable native producer cleanup readback is not qualified.")

    async def _retire_native_producer_cleanup(self, retirement, *, authority, limit):
        raise NotImplementedError("Native producer cleanup retirement is not qualified.")

    async def _acknowledge_native_producer_cleanup(self, registration, control, cleanup):
        from cayu.runtime._producer_output_store import acknowledge_native_cleanup

        if not self._supports_producer_attachment_protocol():
            raise NotImplementedError("Native producer cleanup is not qualified.")
        return await acknowledge_native_cleanup(self, registration, control, cleanup)

    def _supports_owned_off_thread_session_commit_guard_protocol(self) -> bool:
        """Return whether this exact guarded-publication override owns its capability."""

        capability_owner_index = next(
            index
            for index, owner in enumerate(type(self).__mro__)
            if "supports_owned_off_thread_session_commit_guards" in owner.__dict__
        )
        publication_owner_index = next(
            index
            for index, owner in enumerate(type(self).__mro__)
            if "publish_session_operation_guarded" in owner.__dict__
        )
        return (
            self.supports_owned_off_thread_session_commit_guards is True
            and capability_owner_index <= publication_owner_index
        )

    def _supports_atomic_model_completion_stage_release_protocol(self) -> bool:
        """Return whether this exact publication override owns stage release semantics."""

        capability_owner_index = next(
            index
            for index, owner in enumerate(type(self).__mro__)
            if "supports_atomic_model_completion_stage_release" in owner.__dict__
        )
        publication_owner_index = next(
            index
            for index, owner in enumerate(type(self).__mro__)
            if "publish_session_operation" in owner.__dict__
        )
        return (
            self.supports_atomic_model_completion_stage_release is True
            and capability_owner_index <= publication_owner_index
        )

    def _supports_session_steering_protocol(self) -> bool:
        """Require one owner for acceptance, completion, and new-work exclusion."""

        mro = type(self).__mro__
        capability_owner = next(
            index for index, owner in enumerate(mro) if "session_steering_version" in owner.__dict__
        )
        return (
            type(self.session_steering_version) is int
            and self.session_steering_version == 1
            and all(
                capability_owner
                <= next(
                    (index for index, owner in enumerate(mro) if method in owner.__dict__),
                    len(mro),
                )
                for method in (
                    "publish_session_operation",
                    "_publish_session_operation",
                    "prepare_model_completion_stage",
                    "_prepare_model_completion_stage_atomic",
                    "deliver_queued_session_messages",
                    "publish_interaction_transition",
                    "settle_session_invocation",
                )
            )
        )

    def _supports_completion_result_event_publication_reservation_protocol(self) -> bool:
        """Return whether this exact store owns the complete reservation contract."""

        mro = type(self).__mro__
        capability_owner_index = next(
            index
            for index, owner in enumerate(mro)
            if "supports_completion_result_event_publication_reservations" in owner.__dict__
        )
        required_owner_indexes = tuple(
            next(index for index, owner in enumerate(mro) if method_name in owner.__dict__)
            for method_name in (
                "create",
                "create_fork",
                "create_fork_with_transcript_validation",
                "create_profiled_fork",
                "transition_status_and_checkpoint",
                "admit_execution_profile_resume",
                "admit_session_invocation",
                "fence_run_and_transform_checkpoint",
                "checkpoint",
                "transform_checkpoint",
                "delete_session",
                "publish_checkpoint_and_events",
                "publish_session_operation",
                "publish_session_operation_guarded",
                "replace_initial_transcript_messages",
                "append_transcript_messages_and_transform_checkpoint",
                "_publish_completion_result_event_publication",
            )
        )
        return self.supports_completion_result_event_publication_reservations is True and all(
            capability_owner_index <= index for index in required_owner_indexes
        )

    async def publish_session_operation_guarded(
        self,
        session_id: str,
        *,
        idempotency_key: str,
        operation_transform: SessionOperationTransform,
        commit_guard: Callable[[], None],
        events: list[Event],
        expected_statuses: set[SessionStatus] | None = None,
        expected_run_epoch: int | None = None,
        expected_transcript_cursor: int | None = None,
    ) -> Session:
        """Publish with a guard evaluated at the backend's commit boundary."""

        raise NotImplementedError(
            f"{type(self).__name__} must implement atomic guarded operation publication."
        )

    async def publish_session_operation_guarded_with_store_time(
        self,
        session_id: str,
        *,
        idempotency_key: str,
        operation_transform: StoreTimeSessionOperationTransform,
        commit_guard: Callable[[], None],
        commit_time_guard: Callable[[datetime], None],
        events: list[Event],
        expected_statuses: set[SessionStatus] | None = None,
        expected_run_epoch: int | None = None,
        expected_transcript_cursor: int | None = None,
        context_view_compaction_cursor: int | None = None,
    ) -> Session:
        """Publish an operation using time sampled at the commit transaction."""

        raise NotImplementedError(
            f"{type(self).__name__} must implement store-time guarded operation publication."
        )

    async def _publish_continuation_operation(
        self,
        session_id: str,
        *,
        idempotency_key: str,
        operation_transform: SessionOperationTransform,
        events: list[Event],
    ) -> Session:
        """Enclose only a session-owned transform, never a receiving callback."""
        from cayu.sessions._session_continuation_scope import publication_scope
        from cayu.sessions._session_continuation_store import index_publication

        def indexed_transform(session, checkpoint, current):
            publication = operation_transform(session, checkpoint, current)
            return index_publication(session, checkpoint, current, publication, key=idempotency_key)

        with (
            publication_scope(idempotency_key),
            checkpoint_preservation._invocation_lifecycle_authority_read_scope(),
        ):
            return await self.publish_session_operation(
                session_id,
                idempotency_key=idempotency_key,
                operation_transform=indexed_transform,
                events=events,
            )

    async def _publish_temporary_continuation_service(
        self,
        *,
        previous: TemporaryServiceRecord | None,
        proposed: TemporaryServiceRecord,
    ) -> TemporaryServiceRecord:
        """Publish a receiving owner's exact child under the native session lock.

        This private mutation is not an application execution entrance. The
        registered runtime owner must authenticate permits and receiving evidence
        before calling it; it cannot wrap foreign callbacks in publication scope.
        """
        from cayu.collaboration._preparation import prepare_contract
        from cayu.sessions._session_continuation import (
            ContinuationUnavailable,
            continuation_operation_key,
        )
        from cayu.sessions._session_continuation_scope import service_publication_scope
        from cayu.sessions._temporary_continuation import (
            TemporaryServiceRecord,
            temporary_service_key,
        )
        from cayu.sessions._temporary_continuation_store import (
            publish_service_record,
            require_service_deadline,
        )
        from cayu.vaults.redaction import SecretRedactor

        proposed = prepare_contract(TemporaryServiceRecord, proposed, redactor=SecretRedactor())
        if previous is not None:
            previous = prepare_contract(TemporaryServiceRecord, previous, redactor=SecretRedactor())
        ticket = proposed.intent.ticket
        key = continuation_operation_key(ticket)

        def transform(session, checkpoint, current, now):
            return publish_service_record(session, checkpoint, current, previous, proposed, now)

        def check_commit_time(now):
            if previous is None:
                require_service_deadline(proposed, now)

        with (
            service_publication_scope(key, temporary_service_key(proposed.intent.operation)),
            checkpoint_preservation._invocation_lifecycle_authority_read_scope(),
        ):
            await self.publish_session_operation_guarded_with_store_time(
                ticket.session_id,
                idempotency_key=key,
                operation_transform=transform,
                commit_guard=lambda: None,
                commit_time_guard=check_commit_time,
                events=[],
            )
            retained = await self.load_session_operation(
                ticket.session_id, temporary_service_key(proposed.intent.operation)
            )
        if retained is None:
            raise ContinuationUnavailable("Temporary service publication is not durably readable.")
        observed = prepare_contract(TemporaryServiceRecord, retained, redactor=SecretRedactor())
        if observed != proposed and not (
            proposed.state in {"returned", "excluded"}
            and observed == proposed.model_copy(update={"settlement_acknowledged": True})
        ):
            raise ContinuationUnavailable("Temporary service readback differs from publication.")
        return observed

    async def _prepare_temporary_side_service(self, preparation):
        """Native stores must reserve both existing-session preparations atomically."""
        from cayu.sessions._session_continuation import ContinuationUnavailable

        raise ContinuationUnavailable("Session store lacks joint side-service preparation.")

    async def _publish_temporary_service_target(self, *, previous, proposed):
        """Private registered-owner compare under the target's native transaction."""
        from cayu.collaboration._preparation import prepare_contract
        from cayu.sessions._session_continuation import (
            CONTINUATION_NAMESPACE_KEY,
            ContinuationUnavailable,
        )
        from cayu.sessions._session_continuation_scope import service_publication_scope
        from cayu.sessions._temporary_service_target import (
            TemporaryServiceTarget,
            publish_target_record,
            target_service_key,
        )
        from cayu.vaults.redaction import SecretRedactor

        proposed = prepare_contract(TemporaryServiceTarget, proposed, redactor=SecretRedactor())
        if previous is not None:
            previous = prepare_contract(TemporaryServiceTarget, previous, redactor=SecretRedactor())
        if previous == proposed:
            retained = await self._load_temporary_service_target(proposed.service.admission)
            if retained != proposed and not (
                proposed.service.state in {"returned", "excluded"}
                and retained == proposed.model_copy(update={"source_acknowledged": True})
            ):
                raise ContinuationUnavailable("Side-session target replay changed its exact state.")
            return retained
        target = proposed.service.intent.target
        key = target_service_key(proposed.service.intent.operation)

        def transform(session, checkpoint, current, now):
            return publish_target_record(session, checkpoint, current, previous, proposed, now)

        with (
            service_publication_scope(key, CONTINUATION_NAMESPACE_KEY),
            checkpoint_preservation._invocation_lifecycle_authority_read_scope(),
        ):
            await self.publish_session_operation_guarded_with_store_time(
                target.object_id,
                idempotency_key=key,
                operation_transform=transform,
                commit_guard=lambda: None,
                commit_time_guard=lambda now: None,
                events=[],
            )
            raw = await self.load_session_operation(target.object_id, key)
        if raw is None:
            raise ContinuationUnavailable("Side-session target publication is unreadable.")
        retained = prepare_contract(TemporaryServiceTarget, raw, redactor=SecretRedactor())
        if retained != proposed and not (
            proposed.service.state in {"returned", "excluded"}
            and retained == proposed.model_copy(update={"source_acknowledged": True})
        ):
            raise ContinuationUnavailable("Side-session target acknowledgement changed.")
        return retained

    async def _load_temporary_service_target(self, expected):
        """Read an exact indexed target, retaining ambiguity across concurrent changes."""
        from cayu.collaboration._preparation import prepare_contract
        from cayu.sessions._session_continuation import (
            ContinuationConflict,
            ContinuationUnavailable,
        )
        from cayu.sessions._session_continuation_scope import publication_scope
        from cayu.sessions._session_continuation_store import ROOT_KEY, ContinuationRoot
        from cayu.sessions._temporary_continuation import (
            TemporaryServiceAdmission,
            TemporaryServicePreparation,
        )
        from cayu.sessions._temporary_service_target import (
            TemporaryServiceTarget,
            target_reference,
            target_service_key,
        )
        from cayu.vaults.redaction import SecretRedactor

        expected = prepare_contract(
            TemporaryServiceAdmission
            if isinstance(expected, TemporaryServiceAdmission)
            else TemporaryServicePreparation,
            expected,
            redactor=SecretRedactor(),
        )
        intent = expected.dispatch.intent
        key = target_service_key(intent.operation)
        with (
            publication_scope(key),
            checkpoint_preservation._invocation_lifecycle_authority_read_scope(),
        ):
            session = await self.load(intent.target.object_id)
            checkpoint = await self.load_checkpoint(intent.target.object_id)
            raw = await self.load_session_operation(intent.target.object_id, key)
        if session is None or session.instance_id != intent.target.incarnation:
            raise ContinuationUnavailable("Side-session receiving incarnation is unavailable.")
        root_raw = None if checkpoint is None else checkpoint.get(ROOT_KEY)
        root = None if root_raw is None else ContinuationRoot.model_validate(root_raw)
        if root is not None and (
            root.namespace.session_id != session.id
            or root.namespace.session_instance_id != session.instance_id
            or root.namespace.owner != intent.target.owner
        ):
            raise ContinuationConflict("Side-session target index belongs to another owner.")
        reference = (
            None
            if root is None
            else next((item for item in root.target_services if item.key == key), None)
        )
        if raw is None and reference is None:
            return None
        if raw is None or reference is None:
            raise ContinuationUnavailable("Side-session target has incomplete native evidence.")
        retained = TemporaryServiceTarget.model_validate(raw)
        if target_reference(retained) != reference:
            raise ContinuationUnavailable("Side-session target changed during indexed readback.")
        observed = retained.service.admission
        if isinstance(observed, TemporaryServiceAdmission) and not isinstance(
            expected, TemporaryServiceAdmission
        ):
            observed = observed.preparation
        if observed != expected:
            raise ContinuationConflict("Side-session target conflicts with the expected operation.")
        return retained

    async def _load_temporary_continuation_service(
        self, expected: TemporaryServiceAdmission | TemporaryServicePreparation
    ) -> TemporaryServiceRecord | None:
        """Reconstruct one exact child, never treating a missing indexed row as absence."""
        from cayu.collaboration._preparation import prepare_contract
        from cayu.sessions._session_continuation import (
            ContinuationConflict,
            ContinuationUnavailable,
            require_ticket_identity,
        )
        from cayu.sessions._temporary_continuation import (
            TemporaryServiceAdmission,
            TemporaryServicePreparation,
            TemporaryServiceRecord,
            reference_for_service,
            temporary_service_key,
        )
        from cayu.vaults.redaction import SecretRedactor

        expected = (
            prepare_contract(TemporaryServiceAdmission, expected, redactor=SecretRedactor())
            if isinstance(expected, TemporaryServiceAdmission)
            else prepare_contract(TemporaryServicePreparation, expected, redactor=SecretRedactor())
        )
        ticket = expected.dispatch.intent.ticket
        retained = await self.load_continuation_ticket(
            ticket.session_id,
            session_instance_id=ticket.session_instance_id,
            registration_key=ticket.registration_key,
        )
        if retained is None:
            raise ContinuationUnavailable("Temporary service has no retained source ticket.")
        require_ticket_identity(ticket, retained.ticket)
        key = temporary_service_key(expected.dispatch.intent.operation)
        reference = next((item for item in retained.services if item.key == key), None)
        raw = await self.load_session_operation(ticket.session_id, key)
        if reference is None:
            if raw is not None:
                raise ContinuationUnavailable("Temporary service child has lost its source index.")
            return None
        if raw is None:
            raise ContinuationUnavailable("Temporary service index has lost its child evidence.")
        observed = prepare_contract(TemporaryServiceRecord, raw, redactor=SecretRedactor())
        if reference_for_service(observed, retained.services) != reference:
            # A concurrent transition may have advanced between these reads.
            # Neither the stale index nor the new child proves an exact match.
            raise ContinuationUnavailable("Temporary service changed during reconstruction.")
        observed_expected = (
            observed.admission.preparation
            if isinstance(observed.admission, TemporaryServiceAdmission)
            and not isinstance(expected, TemporaryServiceAdmission)
            else observed.admission
        )
        if observed_expected != expected:
            raise ContinuationConflict(
                "Temporary service readback conflicts with expected admission."
            )
        return observed

    async def _read_temporary_continuation_outcome(
        self, expected: TemporaryServiceAdmission
    ) -> TemporaryServiceRecord | None:
        """Read exact native admission/release; absence never proves exclusion."""
        from cayu.collaboration._preparation import prepare_contract
        from cayu.sessions._session_continuation import ContinuationUnavailable
        from cayu.sessions._temporary_continuation import TemporaryServiceAdmission
        from cayu.sessions._temporary_continuation_store import native_service_outcome
        from cayu.vaults.redaction import SecretRedactor

        expected = prepare_contract(TemporaryServiceAdmission, expected, redactor=SecretRedactor())
        target = expected.dispatch.intent.target
        session = await self.load(target.object_id)
        if session is None or session.instance_id != target.incarnation:
            raise ContinuationUnavailable("Temporary service receiving incarnation is unavailable.")
        with checkpoint_preservation._invocation_lifecycle_authority_read_scope():
            checkpoint = await self.load_checkpoint(session.id)
        return native_service_outcome(expected, checkpoint)

    async def _reconcile_temporary_continuation_service(
        self, expected: TemporaryServiceAdmission
    ) -> TemporaryServiceRecord:
        """Settle the source only from native receiving evidence, retaining ambiguity."""
        from cayu.sessions._session_continuation import ContinuationUnavailable

        previous = await self._load_temporary_continuation_service(expected)
        if previous is None:
            raise ContinuationUnavailable(
                "Temporary service has no source responsibility to reconcile."
            )
        target_record = None
        if expected.dispatch.intent.mode == "side_session":
            if previous.state in {"returned", "excluded"}:
                target = expected.dispatch.intent.target
                target_session = await self.load(target.object_id)
                if target_session is None or target_session.instance_id != target.incarnation:
                    # The exact source record already contains positive receiving
                    # settlement. Target absence grants nothing: it only means
                    # there is no old incarnation left to acknowledge/clean up.
                    # Never make terminal replay depend on an unrelated reuse of
                    # the target's public ID.
                    return previous
            target_record = await self._load_temporary_service_target(expected)
            if target_record is None:
                raise ContinuationUnavailable("Side-session receiving fence is unavailable.")
        if previous.state in {"returned", "excluded"}:
            if target_record is not None:
                from cayu.sessions._temporary_service_target import acknowledge_side_target

                await self._publish_temporary_service_target(
                    previous=target_record,
                    proposed=acknowledge_side_target(target_record, previous),
                )
            return previous
        observed = await self._read_temporary_continuation_outcome(expected)
        if observed is None:
            if previous.state == "admitted":
                raise ContinuationUnavailable(
                    "Temporary service native admission evidence is unavailable."
                )
            return previous
        if target_record is not None and observed.state == "returned":
            from cayu.sessions._temporary_service_target import return_side_target

            target_record = await self._publish_temporary_service_target(
                previous=target_record,
                proposed=return_side_target(target_record, observed),
            )
        if observed == previous:
            return previous
        settled = await self._publish_temporary_continuation_service(
            previous=previous, proposed=observed
        )
        if target_record is not None and settled.state == "returned":
            from cayu.sessions._temporary_service_target import acknowledge_side_target

            await self._publish_temporary_service_target(
                previous=target_record,
                proposed=acknowledge_side_target(target_record, settled),
            )
        return settled

    async def _initialize_continuation_namespace(
        self, namespace: ContinuationNamespace
    ) -> ContinuationNamespace:
        """Retain one immutable application/session owner at the fixed bootstrap key."""
        from cayu.collaboration._preparation import prepare_contract
        from cayu.sessions._session_continuation import (
            CONTINUATION_NAMESPACE_KEY,
            ContinuationConflict,
            ContinuationNamespace,
            ContinuationUnavailable,
        )
        from cayu.vaults.redaction import SecretRedactor

        namespace = prepare_contract(ContinuationNamespace, namespace, redactor=SecretRedactor())
        from cayu.sessions._session_continuation_scope import (
            require_namespace_preparation,
            require_preparation_writer,
        )

        require_namespace_preparation(namespace)

        def transform(session, checkpoint, current):
            if (
                session.id != namespace.session_id
                or session.instance_id != namespace.session_instance_id
            ):
                raise SessionRunFenced(
                    "Continuation namespace belongs to another session instance."
                )
            if current is not None and ContinuationNamespace.model_validate(current) != namespace:
                raise ContinuationConflict("Session continuation owner binding already exists.")
            if current is None:
                require_preparation_writer(session, checkpoint)
            return SessionOperationPublication(
                checkpoint={} if checkpoint is None else checkpoint,
                operation_records={CONTINUATION_NAMESPACE_KEY: namespace.model_dump(mode="json")},
            )

        await self._publish_continuation_operation(
            namespace.session_id,
            idempotency_key=CONTINUATION_NAMESPACE_KEY,
            operation_transform=transform,
            events=[],
        )
        retained = await self.load_session_operation(
            namespace.session_id, CONTINUATION_NAMESPACE_KEY
        )
        if retained is None or ContinuationNamespace.model_validate(retained) != namespace:
            raise ContinuationUnavailable("Continuation namespace is not durably readable.")
        return namespace

    async def prepare_continuation_ticket(
        self, command: ContinuationPreparation
    ) -> ContinuationRecord:
        """Atomically retain one exact continuation ticket in ``ARMING`` state."""

        from cayu.collaboration._contracts import CollaborationConflict
        from cayu.collaboration._preparation import prepare_contract, require_exact_contract
        from cayu.sessions._session_continuation import (
            ContinuationConflict,
            ContinuationPreparation,
            ContinuationRecord,
            ContinuationUnavailable,
            continuation_namespace_id,
            continuation_operation_key,
            require_writer_generation,
        )
        from cayu.vaults.redaction import SecretRedactor

        command = prepare_contract(ContinuationPreparation, command, redactor=SecretRedactor())
        from cayu.sessions._session_continuation_scope import (
            require_preparation,
            require_preparation_writer,
        )

        require_preparation(command)
        from cayu.sessions._session_continuation_store import require_reserved_capacity

        require_reserved_capacity(command)
        ticket = command.intent
        session = await self.load(ticket.session_id)
        if session is None:
            raise KeyError(f"Session not found: {ticket.session_id}")
        if (
            session.instance_id != ticket.session_instance_id
            or ticket.namespace.session_id != session.id
            or ticket.namespace.session_instance_id != session.instance_id
            or ticket.namespace.namespace_id
            != continuation_namespace_id(session.id, session.instance_id, ticket.owner)
            or ticket.state != "ARMING"
            or ticket.revision != 1
        ):
            raise SessionRunFenced("Continuation ticket lacks the current session authority.")
        await self._initialize_continuation_namespace(ticket.namespace)
        key = continuation_operation_key(ticket)

        def transform(
            current_session: Session,
            checkpoint: dict[str, Any] | None,
            current: dict[str, Any] | None,
        ) -> SessionOperationPublication:
            if current_session.instance_id != ticket.session_instance_id:
                raise SessionRunFenced("Continuation ticket belongs to a retired session instance.")
            if current is None:
                require_preparation_writer(current_session, checkpoint)
                if current_session.run_epoch != ticket.writer_generation:
                    raise SessionRunFenced(
                        "Continuation ticket lacks the current writer generation."
                    )
                require_writer_generation(ticket, current_session.run_epoch)
                record = ContinuationRecord(
                    namespace=ticket.namespace, preparation=command, ticket=ticket
                )
            else:
                record = ContinuationRecord.model_validate(current)
                try:
                    require_exact_contract(command, record.preparation, redactor=SecretRedactor())
                except CollaborationConflict:
                    raise ContinuationConflict("Continuation preparation conflicts.") from None
            return SessionOperationPublication(
                checkpoint={} if checkpoint is None else checkpoint,
                operation_records={key: record.model_dump(mode="json")},
            )

        await self._publish_continuation_operation(
            ticket.session_id,
            idempotency_key=key,
            operation_transform=transform,
            events=[],
        )
        record = await self.load_continuation_ticket(
            ticket.session_id,
            registration_key=ticket.registration_key,
            session_instance_id=ticket.session_instance_id,
        )
        if record is None:
            raise ContinuationUnavailable("Continuation preparation was not durably readable.")
        return record

    async def lookup_continuation_ticket(
        self, expected: ContinuationPreparation
    ) -> ExactLookup[ContinuationRecord]:
        """Return exact retained preparation evidence, never infer an outcome."""

        from cayu.collaboration._contracts import (
            CollaborationConflict,
            ExactConflict,
            ExactMatch,
            ExactNotFound,
            ExactUnavailable,
        )
        from cayu.collaboration._preparation import prepare_contract, require_exact_contract
        from cayu.sessions._session_continuation import (
            ContinuationPreparation,
            ContinuationRecord,
        )
        from cayu.vaults.redaction import SecretRedactor

        expected = prepare_contract(ContinuationPreparation, expected, redactor=SecretRedactor())
        try:
            record = await self.load_continuation_ticket(
                expected.intent.session_id,
                registration_key=expected.operation.caller_key,
                session_instance_id=expected.intent.session_instance_id,
            )
        except (PermissionError, SessionRunFenced):
            raise
        except Exception:
            # A failed/malformed authoritative read is an explicit lookup
            # outcome, never proof of absence. Cancellation/fatal signals are
            # deliberately outside this ordinary-dependency boundary.
            return ExactUnavailable()
        if record is None:
            return ExactNotFound()
        try:
            require_exact_contract(expected, record.preparation, redactor=SecretRedactor())
        except CollaborationConflict:
            return ExactConflict()
        return ExactMatch[ContinuationRecord](receipt=record)

    async def load_continuation_ticket(
        self,
        session_id: str,
        *,
        registration_key: str,
        session_instance_id: str,
    ) -> ContinuationRecord | None:
        """Read one continuation aggregate without treating absence as proof."""

        from cayu.sessions._session_continuation import (
            CONTINUATION_NAMESPACE_KEY,
            ContinuationNamespace,
            ContinuationRecord,
            continuation_operation_key_for_registration,
        )

        session = await self.load(session_id)
        if session is None:
            raise KeyError(f"Session not found: {session_id}")
        if session.instance_id != session_instance_id:
            raise SessionRunFenced("Continuation readback belongs to a retired session instance.")
        raw_namespace = await self.load_session_operation(session_id, CONTINUATION_NAMESPACE_KEY)
        if raw_namespace is None:
            return None
        namespace = ContinuationNamespace.model_validate(raw_namespace)
        if (
            namespace.session_id != session.id
            or namespace.session_instance_id != session_instance_id
        ):
            raise SessionRunFenced("Continuation namespace belongs to another session instance.")
        key = continuation_operation_key_for_registration(
            session_id=session.id,
            session_instance_id=session_instance_id,
            namespace_id=namespace.namespace_id,
            registration_key=registration_key,
        )
        raw = await self.load_session_operation(session.id, key)
        if raw is None:
            return None
        record = ContinuationRecord.model_validate(raw)
        from cayu.sessions._session_continuation_store import require_history

        require_history(record)
        if (
            record.namespace != namespace
            or record.ticket.session_id != session.id
            or record.ticket.session_instance_id != session_instance_id
            or record.ticket.registration_key != registration_key
        ):
            raise SessionRunFenced("Continuation readback belongs to another session authority.")
        return record

    async def _recover_continuation_writer(
        self, ticket: ContinuationTicket, *, park: bool = False
    ) -> ContinuationRecord:
        """Retain genuine recovery succession without replacing ticket identity."""
        from cayu.collaboration._preparation import prepare_contract
        from cayu.runtime._continuation_recovery import require_recovery_writer
        from cayu.sessions._session_continuation import (
            ContinuationConflict,
            ContinuationRecord,
            ContinuationTicket,
            ContinuationUnavailable,
            continuation_operation_key,
        )
        from cayu.sessions._session_continuation_scope import publication_scope
        from cayu.sessions._session_continuation_store import index_publication
        from cayu.vaults.redaction import SecretRedactor

        ticket = prepare_contract(ContinuationTicket, ticket, redactor=SecretRedactor())
        if type(park) is not bool:
            raise TypeError("Continuation recovery action must be explicit.")
        key = continuation_operation_key(ticket)

        commit_authority: tuple[Session, dict[str, Any] | None, ContinuationRecord] | None = None

        def transform(current_session, checkpoint, current, now):
            nonlocal commit_authority
            if current is None:
                raise ContinuationConflict("Continuation recovery lacks its durable preparation.")
            record = ContinuationRecord.model_validate(current)
            writer = require_recovery_writer(
                ticket, current_session, checkpoint, record, now=now, require_attached_writer=park
            )
            commit_authority = (current_session, checkpoint, record)
            if park and record.recovery_writer != writer:
                raise ContinuationConflict(
                    "Continuation parking lacks retained recovery succession."
                )
            updated = record.model_copy(update={"recovery_writer": writer})
            if park and record.ticket.state == "ARMING":
                updated = updated.model_copy(
                    update={
                        "ticket": record.ticket.model_copy(
                            update={"state": "WAITING", "revision": record.ticket.revision + 1}
                        )
                    }
                )
            publication = SessionOperationPublication(
                checkpoint={} if checkpoint is None else checkpoint,
                operation_records={key: updated.model_dump(mode="json")},
            )
            return index_publication(current_session, checkpoint, current, publication, key=key)

        def guard_time(now):
            if commit_authority is None:
                raise ContinuationConflict("Continuation recovery did not validate its writer.")
            current_session, checkpoint, record = commit_authority
            require_recovery_writer(
                ticket, current_session, checkpoint, record, now=now, require_attached_writer=park
            )

        with (
            publication_scope(key),
            checkpoint_preservation._invocation_lifecycle_authority_read_scope(),
        ):
            await self.publish_session_operation_guarded_with_store_time(
                ticket.session_id,
                idempotency_key=key,
                operation_transform=transform,
                commit_guard=lambda: None,
                commit_time_guard=guard_time,
                events=[],
            )
        result = await self.load_continuation_ticket(
            ticket.session_id,
            registration_key=ticket.registration_key,
            session_instance_id=ticket.session_instance_id,
        )
        if result is None:
            raise ContinuationUnavailable("Continuation recovery succession is not readable.")
        return result

    async def mark_continuation_waiting(self, ticket: ContinuationTicket) -> ContinuationRecord:
        """Publish the ARMING-to-WAITING transition under the ticket's CAS."""

        from cayu.collaboration._preparation import prepare_contract
        from cayu.sessions._session_continuation import (
            ContinuationConflict,
            ContinuationRecord,
            ContinuationTicket,
            ContinuationUnavailable,
            continuation_operation_key,
            require_ticket_identity,
            require_writer_generation,
        )
        from cayu.sessions._session_continuation_scope import require_park
        from cayu.vaults.redaction import SecretRedactor

        ticket = prepare_contract(ContinuationTicket, ticket, redactor=SecretRedactor())
        require_park(ticket)
        key = continuation_operation_key(ticket)

        def transform(
            current_session: Session,
            checkpoint: dict[str, Any] | None,
            current: dict[str, Any] | None,
        ) -> SessionOperationPublication:
            if current_session.instance_id != ticket.session_instance_id:
                raise SessionRunFenced("Continuation ticket belongs to a retired session instance.")
            require_writer_generation(
                ticket,
                current_session.run_epoch,
                allow_released_next_generation=_continuation_writer_was_released(
                    current_session, checkpoint
                ),
            )
            if current is None:
                raise ContinuationConflict("Continuation ticket is not durably armed.")
            record = ContinuationRecord.model_validate(current)
            require_ticket_identity(ticket, record.ticket)
            if record.ticket.state == "WAITING":
                updated = record
            elif record.ticket.state == "ARMING" and record.ticket.revision == ticket.revision:
                updated_ticket = record.ticket.model_copy(
                    update={"state": "WAITING", "revision": record.ticket.revision + 1}
                )
                updated = record.model_copy(update={"ticket": updated_ticket})
            else:
                raise ContinuationConflict("Continuation ticket cannot enter WAITING state.")
            return SessionOperationPublication(
                checkpoint={} if checkpoint is None else checkpoint,
                operation_records={key: updated.model_dump(mode="json")},
            )

        await self._publish_continuation_operation(
            ticket.session_id,
            idempotency_key=key,
            operation_transform=transform,
            events=[],
        )
        record = await self.load_continuation_ticket(
            ticket.session_id,
            registration_key=ticket.registration_key,
            session_instance_id=ticket.session_instance_id,
        )
        if record is None:
            raise ContinuationUnavailable("Continuation wait transition was not durably readable.")
        return record

    async def latch_continuation(
        self,
        latch: ContinuationLatch,
    ) -> ContinuationRecord:
        """Publish one immutable readiness latch, including an ARMING latch."""

        from cayu.sessions._session_continuation import (
            ContinuationConflict,
            ContinuationRecord,
            ContinuationUnavailable,
            continuation_operation_key,
            require_latch_identity,
            require_record_writer_generation,
            require_ticket_identity,
        )
        from cayu.sessions._session_continuation_scope import require_authenticated_latch

        require_authenticated_latch(latch)

        ticket = latch.ticket
        key = continuation_operation_key(ticket)

        # Reconcile an already committed latch before re-entering the receiving
        # owner.  A lost acknowledgement must not turn an exact durable replay
        # into a fresh foreign read (or make replay depend on a source that has
        # since expired).  A missing latch still requires positive receiver
        # authentication below.
        existing = await self.load_continuation_ticket(
            ticket.session_id,
            registration_key=ticket.registration_key,
            session_instance_id=ticket.session_instance_id,
        )
        if existing is not None and existing.latch is not None:
            require_ticket_identity(existing.latch.ticket, ticket)
            require_latch_identity(existing.latch, latch)
            return existing

        def transform(
            current_session: Session,
            checkpoint: dict[str, Any] | None,
            current: dict[str, Any] | None,
        ) -> SessionOperationPublication:
            if current_session.instance_id != ticket.session_instance_id:
                raise SessionRunFenced("Continuation latch belongs to a retired session instance.")
            if current is None:
                raise ContinuationConflict("Continuation ticket is not durably armed.")
            record = ContinuationRecord.model_validate(current)
            require_record_writer_generation(
                record,
                current_session.run_epoch,
                allow_released_next_generation=_continuation_writer_was_released(
                    current_session, checkpoint
                ),
            )
            require_ticket_identity(ticket, record.ticket)
            if record.latch is not None:
                require_latch_identity(record.latch, latch)
                updated = record
            elif record.ticket.state in {"CONSUMED", "RETIRED"}:
                raise ContinuationConflict("A terminal continuation cannot accept a latch.")
            else:
                updated = record.model_copy(update={"latch": latch})
            return SessionOperationPublication(
                checkpoint={} if checkpoint is None else checkpoint,
                operation_records={key: updated.model_dump(mode="json")},
            )

        await self._publish_continuation_operation(
            ticket.session_id,
            idempotency_key=key,
            operation_transform=transform,
            events=[],
        )
        record = await self.load_continuation_ticket(
            ticket.session_id,
            registration_key=ticket.registration_key,
            session_instance_id=ticket.session_instance_id,
        )
        if record is None:
            raise ContinuationUnavailable("Continuation latch was not durably readable.")
        return record

    async def consume_continuation(
        self,
        consumption: ContinuationConsumption,
    ) -> ContinuationRecord:
        """Atomically elect one inline or queued continuation and replay it."""

        from cayu.sessions._session_continuation_scope import require_consumption

        require_consumption(consumption)

        return await self._consume_continuation(consumption, admission_boundary=False)

    async def _claim_continuation_admission(
        self, consumption: ContinuationConsumption
    ) -> tuple[ContinuationRecord, bool]:
        """Fence retirement before dispatching the typed admission mutation."""

        from cayu.collaboration._preparation import prepare_contract
        from cayu.sessions._session_continuation import (
            ContinuationConflict,
            ContinuationConsumption,
            ContinuationRecord,
            ContinuationUnavailable,
            continuation_operation_key,
            require_latch_identity,
            require_ticket_identity,
        )
        from cayu.sessions._session_continuation_scope import require_consumption
        from cayu.vaults.redaction import SecretRedactor

        consumption = prepare_contract(
            ContinuationConsumption, consumption, redactor=SecretRedactor()
        )
        require_consumption(consumption)
        if consumption.receipt_stage != "prepared":
            raise ContinuationConflict(
                "Admission claims require a prepared responsibility receipt."
            )
        ticket = consumption.ticket
        key = continuation_operation_key(ticket)
        acquired = False
        claim_id = uuid4().hex

        def transform(
            current_session: Session,
            checkpoint: dict[str, Any] | None,
            current: dict[str, Any] | None,
        ) -> SessionOperationPublication:
            nonlocal acquired
            if current_session.instance_id != ticket.session_instance_id:
                raise SessionRunFenced(
                    "Continuation admission claim belongs to a retired session instance."
                )
            if current is None:
                raise ContinuationConflict("Continuation ticket is unavailable.")
            record = ContinuationRecord.model_validate(current)
            if record.consumption is None:
                raise ContinuationConflict("Continuation responsibility is not prepared.")
            existing = record.consumption
            require_ticket_identity(existing.ticket, ticket)
            require_latch_identity(existing.latch, consumption.latch)
            if existing.receipt_stage != "prepared":
                raise ContinuationConflict("Continuation responsibility is no longer prepared.")
            base_update = {
                "admission_claimed": False,
                "admission_claim_id": None,
            }
            comparable_existing = existing.model_copy(update=base_update)
            comparable_consumption = consumption.model_copy(update=base_update)
            if comparable_existing.model_dump(mode="json") != comparable_consumption.model_dump(
                mode="json"
            ):
                raise ContinuationConflict("Continuation admission differs from its receipt.")
            if existing.admission_claimed:
                updated = record
            else:
                acquired = True
                updated = record.model_copy(
                    update={
                        "consumption": existing.model_copy(
                            update={
                                "admission_claimed": True,
                                "admission_claim_id": claim_id,
                            }
                        )
                    }
                )
            return SessionOperationPublication(
                checkpoint={} if checkpoint is None else checkpoint,
                operation_records={key: updated.model_dump(mode="json")},
            )

        await self._publish_continuation_operation(
            ticket.session_id,
            idempotency_key=key,
            operation_transform=transform,
            events=[],
        )
        record = await self.load_continuation_ticket(
            ticket.session_id,
            registration_key=ticket.registration_key,
            session_instance_id=ticket.session_instance_id,
        )
        if record is None or record.consumption is None:
            raise ContinuationUnavailable("Continuation admission claim was not durably readable.")
        return record, acquired

    async def _release_continuation_admission_claim(
        self, consumption: ContinuationConsumption
    ) -> ContinuationRecord:
        """Fence an uncommitted claim, or exclude it from positive supersession evidence."""

        from cayu.collaboration._preparation import prepare_contract
        from cayu.sessions._session_continuation import (
            ContinuationConflict,
            ContinuationConsumption,
            ContinuationRecord,
            ContinuationUnavailable,
            continuation_operation_key,
            require_latch_identity,
            require_ticket_identity,
        )
        from cayu.sessions._session_continuation_scope import require_consumption
        from cayu.vaults.redaction import SecretRedactor

        consumption = prepare_contract(
            ContinuationConsumption, consumption, redactor=SecretRedactor()
        )
        require_consumption(consumption)
        if consumption.receipt_stage != "prepared":
            raise ContinuationConflict("Admission claim release requires a prepared receipt.")
        ticket = consumption.ticket
        key = continuation_operation_key(ticket)

        def transform(
            current_session: Session,
            checkpoint: dict[str, Any] | None,
            current: dict[str, Any] | None,
        ) -> SessionOperationPublication:
            if current_session.instance_id != ticket.session_instance_id:
                raise SessionRunFenced(
                    "Continuation admission claim belongs to a retired session instance."
                )
            if current is None:
                raise ContinuationConflict("Continuation ticket is unavailable.")
            record = ContinuationRecord.model_validate(current)
            if record.consumption is None:
                raise ContinuationConflict("Continuation responsibility is not claimed.")
            existing = record.consumption
            require_ticket_identity(existing.ticket, ticket)
            require_latch_identity(existing.latch, consumption.latch)
            if (
                existing.receipt_stage != "prepared"
                or existing.admission_claimed != consumption.admission_claimed
                or existing.admission_claim_id != consumption.admission_claim_id
                or existing.model_copy(
                    update={
                        "admission_claimed": False,
                        "admission_claim_id": None,
                    }
                ).model_dump(mode="json")
                != consumption.model_copy(
                    update={
                        "admission_claimed": False,
                        "admission_claim_id": None,
                    }
                ).model_dump(mode="json")
            ):
                raise ContinuationConflict("Continuation admission claim differs from its receipt.")
            from cayu.sessions._invocation_lifecycle import (
                reconcile_invocation_admission_from_state,
                superseding_invocation_admission_digest_from_state,
            )

            superseding_digest = superseding_invocation_admission_digest_from_state(
                current_session,
                checkpoint,
                session_id=ticket.session_id,
                session_instance_id=ticket.session_instance_id,
                expected_run_epoch=consumption.admission_expected_run_epoch,
                command_sha256=consumption.admission_command_digest,
            )
            if superseding_digest is not None:
                from cayu.sessions._session_continuation import ContinuationRetirement

                retired_ticket = record.ticket.model_copy(
                    update={
                        "state": "RETIRED",
                        "revision": record.ticket.revision + 1,
                    }
                )
                updated = record.model_copy(
                    update={
                        "ticket": retired_ticket,
                        "consumption": existing.model_copy(
                            update={
                                "ticket": retired_ticket,
                                "receipt_stage": "excluded",
                                "admission_claimed": False,
                                "admission_claim_id": None,
                            }
                        ),
                        "retirement": ContinuationRetirement(
                            ticket=retired_ticket,
                            control_id="admission-superseded:" + superseding_digest,
                            reason="superseded",
                            retired_at=current_session.updated_at.isoformat(),
                        ),
                    }
                )
                return SessionOperationPublication(
                    checkpoint={} if checkpoint is None else checkpoint,
                    operation_records={key: updated.model_dump(mode="json")},
                )

            if not existing.admission_claimed:
                raise ContinuationConflict(
                    "Unclaimed continuation exclusion requires positive supersession evidence."
                )
            if (
                reconcile_invocation_admission_from_state(
                    current_session,
                    checkpoint,
                    session_id=ticket.session_id,
                    session_instance_id=ticket.session_instance_id,
                    expected_run_epoch=consumption.admission_expected_run_epoch,
                    command_sha256=consumption.admission_command_digest,
                    profile_sha256=consumption.profile_digest,
                )
                is not None
            ):
                raise ContinuationConflict("Committed continuation admission cannot be released.")
            updated = record.model_copy(
                update={
                    "consumption": existing.model_copy(
                        update={
                            "admission_claimed": False,
                            "admission_claim_id": None,
                        }
                    )
                }
            )
            return SessionOperationPublication(
                checkpoint={} if checkpoint is None else checkpoint,
                operation_records={key: updated.model_dump(mode="json")},
            )

        await self._publish_continuation_operation(
            ticket.session_id,
            idempotency_key=key,
            operation_transform=transform,
            events=[],
        )
        record = await self.load_continuation_ticket(
            ticket.session_id,
            registration_key=ticket.registration_key,
            session_instance_id=ticket.session_instance_id,
        )
        if record is None or record.consumption is None:
            raise ContinuationUnavailable("Continuation claim release was not durably readable.")
        return record

    async def _finalize_continuation_admission(
        self,
        consumption: ContinuationConsumption,
        command: AdmitInvocationCommand | None = None,
    ) -> ContinuationRecord:
        """Commit the admitted stage only from the typed lifecycle boundary."""

        from cayu.sessions._session_continuation_scope import require_consumption

        require_consumption(consumption)

        return await self._consume_continuation(
            consumption, admission_boundary=True, admission_command=command
        )

    async def _consume_continuation(
        self,
        consumption: ContinuationConsumption,
        *,
        admission_boundary: bool,
        admission_command: AdmitInvocationCommand | None = None,
    ) -> ContinuationRecord:
        """Apply the store-side CAS for public preparation or typed admission."""

        from cayu.sessions._session_continuation import (
            ContinuationConflict,
            ContinuationRecord,
            ContinuationUnavailable,
            continuation_admission_digest,
            continuation_operation_key,
            require_latch_identity,
            require_record_writer_generation,
            require_ticket_identity,
        )

        ticket = consumption.ticket
        key = continuation_operation_key(ticket)
        if consumption.receipt_stage == "admitted" and not admission_boundary:
            raise ContinuationConflict(
                "Admitted continuation receipts require the typed lifecycle boundary."
            )

        def transform(
            current_session: Session,
            checkpoint: dict[str, Any] | None,
            current: dict[str, Any] | None,
        ) -> SessionOperationPublication:
            if current_session.instance_id != ticket.session_instance_id:
                raise SessionRunFenced(
                    "Continuation consumption belongs to a retired session instance."
                )
            if current is None:
                raise ContinuationConflict("Continuation ticket is unavailable.")
            if admission_boundary:
                from cayu.sessions._invocation_lifecycle import (
                    AdmitInvocationCommand,
                    reconcile_invocation_admission_from_state,
                    replay_invocation_lifecycle_command_from_state,
                )

                if admission_command is not None and (
                    type(admission_command) is not AdmitInvocationCommand
                    or continuation_admission_digest(admission_command)
                    != consumption.admission_command_digest
                    or admission_command.session_id != ticket.session_id
                    or admission_command.expected_session_instance_id != ticket.session_instance_id
                    or replay_invocation_lifecycle_command_from_state(
                        current_session, checkpoint, admission_command
                    )
                    is None
                ):
                    raise ContinuationConflict(
                        "Continuation lacks exact durable admission evidence."
                    )
                if (
                    reconcile_invocation_admission_from_state(
                        current_session,
                        checkpoint,
                        session_id=ticket.session_id,
                        session_instance_id=ticket.session_instance_id,
                        expected_run_epoch=consumption.admission_expected_run_epoch,
                        command_sha256=consumption.admission_command_digest,
                        profile_sha256=consumption.profile_digest,
                    )
                    is None
                ):
                    raise ContinuationConflict(
                        "Continuation lacks exact durable admission evidence."
                    )
            record = ContinuationRecord.model_validate(current)
            if record.consumption is not None:
                existing = record.consumption
                require_ticket_identity(existing.ticket, consumption.ticket)
                require_latch_identity(existing.latch, consumption.latch)
                replay = existing.model_copy(
                    update={"ticket": consumption.ticket, "latch": consumption.latch}
                )
                if replay == consumption or (
                    existing.receipt_stage in {"prepared", "admitted"}
                    and consumption.receipt_stage == "prepared"
                    and replay.model_copy(
                        update={
                            "receipt_stage": "prepared",
                            "admission_claimed": False,
                            "admission_claim_id": None,
                        }
                    )
                    == consumption.model_copy(
                        update={
                            "admission_claimed": False,
                            "admission_claim_id": None,
                        }
                    )
                ):
                    updated = record
                elif (
                    existing.receipt_stage == "prepared"
                    and consumption.receipt_stage == "admitted"
                    and existing.admission_claimed
                    and existing.admission_claim_id == consumption.admission_claim_id
                    and existing.model_copy(
                        update={
                            "ticket": consumption.ticket,
                            "latch": consumption.latch,
                            "receipt_stage": "admitted",
                            "admission_claimed": True,
                        }
                    ).model_dump(mode="json")
                    == consumption.model_dump(mode="json")
                ):
                    consumed_ticket = record.ticket.model_copy(
                        update={"state": "CONSUMED", "revision": record.ticket.revision + 1}
                    )
                    updated_consumption = consumption.model_copy(
                        update={
                            "ticket": consumed_ticket,
                            "latch": record.latch,
                            "admission_claimed": True,
                            "admission_claim_id": consumption.admission_claim_id,
                        }
                    )
                    updated = record.model_copy(
                        update={"ticket": consumed_ticket, "consumption": updated_consumption}
                    )
                else:
                    raise ContinuationConflict("Continuation was consumed differently.")
            else:
                if current_session.run_epoch != consumption.admission_expected_run_epoch:
                    raise ContinuationConflict(
                        "Continuation admission belongs to a stale writer generation."
                    )
                require_ticket_identity(ticket, record.ticket)
                # The new admission command describes the current writer, not
                # the wait's originating authority. A previously latched result
                # must not let an overtaken wait start a fresh consumption.
                # Existing preparations above reconcile their exact handoff.
                require_record_writer_generation(
                    record,
                    current_session.run_epoch,
                    allow_released_next_generation=_continuation_writer_was_released(
                        current_session, checkpoint
                    ),
                )
                if record.latch is None or record.ticket.state != "WAITING":
                    raise ContinuationConflict("Continuation is not ready for consumption.")
                if consumption.receipt_stage != "prepared":
                    raise ContinuationConflict(
                        "Continuation admission requires a prepared responsibility receipt."
                    )
                if record.ticket.revision != ticket.revision:
                    raise ContinuationConflict("Continuation ticket revision is stale.")
                require_ticket_identity(record.latch.ticket, record.ticket)
                require_latch_identity(record.latch, consumption.latch)
                updated = record.model_copy(
                    update={"consumption": consumption.model_copy(update={"latch": record.latch})}
                )
            return SessionOperationPublication(
                checkpoint={} if checkpoint is None else checkpoint,
                operation_records={key: updated.model_dump(mode="json")},
            )

        await self._publish_continuation_operation(
            ticket.session_id,
            idempotency_key=key,
            operation_transform=transform,
            events=[],
        )
        record = await self.load_continuation_ticket(
            ticket.session_id,
            registration_key=ticket.registration_key,
            session_instance_id=ticket.session_instance_id,
        )
        if record is None:
            raise ContinuationUnavailable("Continuation consumption was not durably readable.")
        return record

    async def retire_continuation(self, retirement: ContinuationRetirement) -> ContinuationRecord:
        """Atomically retire a ticket, losing only to an already committed consume."""

        from cayu.collaboration._preparation import prepare_contract
        from cayu.sessions._session_continuation import (
            ContinuationConflict,
            ContinuationRecord,
            ContinuationReleasedRetirement,
            ContinuationRetirement,
            ContinuationUnavailable,
            continuation_operation_key,
            continuation_writer_frontier,
            require_record_writer_generation,
            require_ticket_identity,
        )
        from cayu.sessions._session_continuation_scope import require_retirement
        from cayu.vaults.redaction import SecretRedactor

        retirement = prepare_contract(ContinuationRetirement, retirement, redactor=SecretRedactor())
        retirement_authority = require_retirement(retirement)
        ticket = retirement.ticket
        key = continuation_operation_key(ticket)

        def transform(
            current_session: Session,
            checkpoint: dict[str, Any] | None,
            current: dict[str, Any] | None,
        ) -> SessionOperationPublication:
            if current_session.instance_id != ticket.session_instance_id:
                raise SessionRunFenced(
                    "Continuation retirement belongs to a retired session instance."
                )
            if current is None:
                raise ContinuationConflict("Continuation ticket is unavailable.")
            record = ContinuationRecord.model_validate(current)
            if record.ticket.state == "SERVICING":
                raise ContinuationConflict("Temporary service must settle before retirement.")
            released_evidence = record.released_retirement
            if isinstance(retirement_authority, ContinuationReleasedRetirement):
                from cayu.sessions._session_continuation_store import (
                    require_released_wait_invocation,
                )

                if released_evidence is None:
                    if record.retirement is not None:
                        raise ContinuationConflict(
                            "Retirement has no retained released-execution proof."
                        )
                    released_evidence = require_released_wait_invocation(
                        ticket,
                        checkpoint,
                        permit_operation=retirement_authority.permit_operation,
                        permit_commitment=retirement_authority.permit_commitment,
                        record=record,
                    )
                elif (
                    released_evidence.permit_operation != retirement_authority.permit_operation
                    or released_evidence.permit_commitment != retirement_authority.permit_commitment
                ):
                    raise ContinuationConflict("Retirement replay changed its released execution.")
            if record.retirement is not None:
                existing = record.retirement
                require_ticket_identity(existing.ticket, retirement.ticket)
                replay = existing.model_copy(update={"ticket": retirement.ticket})
                if replay == retirement:
                    updated = record
                else:
                    raise ContinuationConflict("Continuation was retired differently.")
            else:
                if isinstance(retirement_authority, ContinuationReleasedRetirement):
                    if (
                        record.ticket.state not in {"ARMING", "WAITING"}
                        or record.consumption is not None
                        or any(
                            item.state not in {"returned", "excluded"} for item in record.services
                        )
                        or retirement_authority.settled_services
                        != tuple((item.key, item.record_sha256) for item in record.services)
                        or not _continuation_writer_was_released(current_session, checkpoint)
                    ):
                        raise ContinuationConflict(
                            "Released cleanup requires a quiescent wait and settled services."
                        )
                    require_record_writer_generation(
                        record, current_session.run_epoch, allow_released_next_generation=True
                    )
                elif retirement_authority:
                    # A release alone advances by one. A later epoch in this
                    # exact incarnation proves the old wait has been overtaken.
                    # Never settle prepared/claimed receiving work by inference.
                    writer, released = continuation_writer_frontier(record)
                    if (
                        retirement.reason != "superseded"
                        or current_session.run_epoch <= writer + (not released)
                        or record.consumption is not None
                    ):
                        raise ContinuationConflict(
                            "Supersession requires an overtaken, unconsumed continuation."
                        )
                else:
                    require_record_writer_generation(
                        record,
                        current_session.run_epoch,
                        allow_released_next_generation=_continuation_writer_was_released(
                            current_session, checkpoint
                        ),
                    )
                require_ticket_identity(ticket, record.ticket)
                if record.ticket.revision != ticket.revision:
                    raise ContinuationConflict("Continuation ticket revision is stale.")
                if record.ticket.state in {"CONSUMED", "RETIRED"}:
                    raise ContinuationConflict("Continuation is already terminal.")
                retired_ticket = record.ticket.model_copy(
                    update={"state": "RETIRED", "revision": record.ticket.revision + 1}
                )
                updated_retirement = retirement.model_copy(update={"ticket": retired_ticket})
                if record.consumption is not None:
                    if (
                        record.consumption.receipt_stage != "prepared"
                        or record.consumption.admission_claimed
                    ):
                        raise ContinuationConflict(
                            "Continuation admission is in progress and must settle before retirement."
                        )
                    excluded_consumption = record.consumption.model_copy(
                        update={"ticket": retired_ticket, "receipt_stage": "excluded"}
                    )
                    updated = record.model_copy(
                        update={
                            "ticket": retired_ticket,
                            "consumption": excluded_consumption,
                            "retirement": updated_retirement,
                        }
                    )
                else:
                    updated = record.model_copy(
                        update={
                            "ticket": retired_ticket,
                            "retirement": updated_retirement,
                            "released_retirement": released_evidence,
                        }
                    )
            return SessionOperationPublication(
                checkpoint={} if checkpoint is None else checkpoint,
                operation_records={key: updated.model_dump(mode="json")},
            )

        await self._publish_continuation_operation(
            ticket.session_id,
            idempotency_key=key,
            operation_transform=transform,
            events=[],
        )
        record = await self.load_continuation_ticket(
            ticket.session_id,
            registration_key=ticket.registration_key,
            session_instance_id=ticket.session_instance_id,
        )
        if record is None:
            raise ContinuationUnavailable("Continuation retirement was not durably readable.")
        return record

    async def load_runtime_publication_receipt(
        self,
        session_id: str,
        publication_id: str,
    ) -> RuntimePublicationReceipt | None:
        """Load and verify one store-owned runtime publication receipt."""

        session_id = require_clean_nonblank(session_id, "session_id")
        publication_id = require_clean_nonblank(publication_id, "publication_id")
        storage_key = publication_contracts._runtime_publication_storage_key(publication_id)
        record = await self._load_runtime_publication_receipt_record(
            session_id,
            storage_key,
            publication_id,
        )
        if record is None:
            return None
        return publication_rules._reconstruct_runtime_publication_receipt(
            record,
            storage_key=storage_key,
            session_id=session_id,
            publication_id=publication_id,
        )

    async def publish_runtime_publication(
        self,
        session_id: str,
        *,
        request: RuntimePublicationRequest,
        expected_statuses: set[SessionStatus] | None = None,
        expected_run_epoch: int | None = None,
        expected_transcript_cursor: int | None = None,
    ) -> RuntimePublicationResult:
        """Atomically publish or exactly replay one logical runtime boundary."""

        prepared = publication_rules._prepare_runtime_publication(
            session_id,
            request,
            expected_statuses=expected_statuses,
            expected_run_epoch=expected_run_epoch,
            expected_transcript_cursor=expected_transcript_cursor,
        )
        return await self._publish_runtime_publication_atomic(prepared)

    async def _load_runtime_publication_receipt_record(
        self,
        session_id: str,
        storage_key: str,
        publication_id: str,
    ) -> dict[str, Any] | None:
        """Backend receipt/material lookup hook; fail closed for custom stores.

        Implementations must verify the committed transcript slice, appended
        event batch, and exact content of every referenced event in the same
        locked or transactional snapshot used to load the receipt.
        """

        raise NotImplementedError(
            f"{type(self).__name__} does not support durable runtime publications."
        )

    async def _publish_runtime_publication_atomic(
        self,
        prepared: publication_rules._PreparedRuntimePublication,
    ) -> RuntimePublicationResult:
        """Backend atomic publication hook; intentionally non-abstract for compatibility."""

        raise NotImplementedError(
            f"{type(self).__name__} does not support durable runtime publications."
        )

    async def load_model_completion_stage(
        self,
        session_id: str,
        stage_id: str,
    ) -> ModelCompletionStage | None:
        """Load and verify append-only pre-dispatch and terminal completion evidence."""

        session_id, stage_id, preparation_key, terminal_key = (
            model_stage_rules._model_completion_stage_storage_identity(session_id, stage_id)
        )
        preparation_record, terminal_record = await self._load_model_completion_stage_records(
            session_id,
            preparation_key,
            terminal_key,
        )
        return model_stage_rules._reconstruct_model_completion_stage(
            preparation_record,
            terminal_record,
            session_id=session_id,
            stage_id=stage_id,
            preparation_storage_key=preparation_key,
            terminal_storage_key=terminal_key,
        )

    async def load_model_completion_stage_settlement(
        self,
        session_id: str,
        stage_id: str,
    ) -> ModelCompletionStageSettlement | None:
        """Load verified terminal non-success evidence for one exact stage."""

        session_id, stage_id, _, _ = model_stage_rules._model_completion_stage_storage_identity(
            session_id,
            stage_id,
        )
        storage_key = model_stage_rules._model_completion_stage_settlement_storage_key(stage_id)
        record = await self._load_model_completion_stage_settlement_record(
            session_id,
            storage_key,
        )
        if record is None:
            return None
        return model_stage_rules._reconstruct_model_completion_stage_settlement(
            record,
            session_id=session_id,
            stage_id=stage_id,
            storage_key=storage_key,
        )

    async def load_model_completion_stage_abandonment(
        self,
        session_id: str,
        stage_id: str,
    ) -> ModelCompletionStageAbandonment | None:
        """Read positive no-dispatch evidence after the preparation was removed.

        This is a query, not permission to dispatch. Re-preparation compares the
        complete expected predecessor and this receipt in its own transaction.
        The shared operation-record lookup keeps all supported backends under
        their existing read owner; no second route cache is needed after restart.
        """

        session_id, stage_id, _, _ = model_stage_rules._model_completion_stage_storage_identity(
            session_id, stage_id
        )
        storage_key = model_stage_rules._model_completion_stage_abandonment_storage_key(stage_id)
        record = await self._load_model_completion_stage_settlement_record(session_id, storage_key)
        if record is None:
            return None
        return model_stage_rules._reconstruct_model_completion_stage_abandonment(
            record, session_id=session_id, stage_id=stage_id, storage_key=storage_key
        )

    async def load_active_model_completion_stage(
        self,
        session_id: str,
    ) -> ActiveModelCompletionStage | None:
        """Load a verified snapshot of the dispatch currently eligible to publish."""

        session_id = require_clean_nonblank(session_id, "session_id")
        (
            active_record,
            preparation_record,
            terminal_record,
        ) = await self._load_active_model_completion_stage_records(session_id)
        return model_stage_rules._reconstruct_active_model_completion_stage(
            active_record,
            preparation_record,
            terminal_record,
            session_id=session_id,
        )

    async def load_model_completion_stage_dispatch(
        self,
        session_id: str,
        stage_id: str,
    ) -> ModelCompletionStageDispatch | None:
        """Load exact evidence that a prepared stage crossed the dispatch fence."""

        session_id = require_clean_nonblank(session_id, "session_id")
        storage_key = model_stage_rules._model_completion_stage_dispatch_storage_key(stage_id)
        record = await self._load_model_completion_stage_dispatch_record(
            session_id,
            storage_key,
        )
        if record is None:
            return None
        return model_stage_rules._reconstruct_model_completion_stage_dispatch(
            record,
            session_id=session_id,
            stage_id=stage_id,
            storage_key=storage_key,
        )

    async def mark_model_completion_stage_dispatched(
        self,
        session_id: str,
        *,
        stage: ModelCompletionStage,
        consume_child_session_notifications: bool = True,
    ) -> ModelCompletionStageDispatch:
        """Durably cross the last local fence before entering provider-controlled code."""

        if type(stage) is not ModelCompletionStage:
            raise TypeError("stage must be a ModelCompletionStage.")
        if type(consume_child_session_notifications) is not bool:
            raise TypeError("consume_child_session_notifications must be a boolean.")
        copied_stage = stage.model_copy(deep=True)
        session_id = require_clean_nonblank(session_id, "session_id")
        if copied_stage.session_id != session_id:
            raise SessionModelCompletionStageConflict(
                "Model-completion dispatch belongs to a different session."
            )
        try:
            return await self._mark_model_completion_stage_dispatched_atomic(
                session_id,
                stage=copied_stage,
                consume_child_session_notifications=consume_child_session_notifications,
            )
        except Exception:
            reconciled = await self.load_model_completion_stage_dispatch(
                session_id,
                copied_stage.stage_id,
            )
            if reconciled is None:
                raise
            active = await self.load_active_model_completion_stage(session_id)
            if active is None or active.stage != copied_stage:
                raise
            model_stage_rules._validate_model_completion_stage_dispatch(reconciled, copied_stage)
            if (
                consume_child_session_notifications
                and child_session_notification_stage_binding(copied_stage.intent) is not None
            ):
                return await self._mark_model_completion_stage_dispatched_atomic(
                    session_id,
                    stage=copied_stage,
                    consume_child_session_notifications=True,
                )
            return reconciled

    async def prepare_model_completion_stage(
        self,
        session_id: str,
        *,
        request: ModelCompletionStageRequest,
        expected_statuses: set[SessionStatus],
        expected_run_epoch: int,
        expected_transcript_cursor: int,
    ) -> ModelCompletionStageResult:
        """Insert or replay a stage; dispatch only when the result explicitly authorizes it."""

        prepared = model_stage_rules._prepare_model_completion_stage(
            session_id,
            request,
            expected_statuses=expected_statuses,
            expected_run_epoch=expected_run_epoch,
            expected_transcript_cursor=expected_transcript_cursor,
        )
        return await self._prepare_model_completion_stage_atomic(prepared)

    def _supports_model_failover_stage_protocol(self) -> bool:
        """Overriding a mutation owner requires renewed atomic-route attestation."""

        mro = type(self).__mro__
        capability_owner = next(
            index
            for index, owner in enumerate(mro)
            if "model_failover_stage_version" in owner.__dict__
        )
        return (
            type(self.model_failover_stage_version) is int
            and self.model_failover_stage_version == 1
            and all(
                capability_owner
                <= next((index for index, owner in enumerate(mro) if hook in owner.__dict__), -1)
                for hook in (
                    "_prepare_model_completion_stage_atomic",
                    "_mark_model_completion_stage_dispatched_atomic",
                    "_promote_model_completion_stage_atomic",
                    "create_profiled_fork",
                    "_create_fork",
                    "admit_session_invocation",
                    "admit_execution_profile_resume",
                    "transition_status_and_checkpoint",
                )
            )
        )

    def _require_model_failover_stage_protocol(self) -> None:
        """Reject routed admission unless every native mutation owner attests it."""

        if not self._supports_model_failover_stage_protocol():
            raise NotImplementedError("Store does not attest atomic model failover stages.")

    async def _prepare_model_completion_stage_with_failover(
        self,
        session_id: str,
        *,
        request: ModelCompletionStageRequest,
        expected_statuses: set[SessionStatus],
        expected_run_epoch: int,
        expected_transcript_cursor: int,
        admission: ModelFailoverStageAdmission,
    ) -> ModelCompletionStageResult:
        """Private live-authority entrance; raw public intents cannot select a route."""

        self._require_model_failover_stage_protocol()
        prepared = model_stage_rules._prepare_model_completion_stage(
            session_id,
            request,
            expected_statuses=expected_statuses,
            expected_run_epoch=expected_run_epoch,
            expected_transcript_cursor=expected_transcript_cursor,
            failover_admission=admission,
        )
        return await self._prepare_model_completion_stage_atomic(prepared)

    def _supports_model_completion_recovery_fence_protocol(self) -> bool:
        """Require an explicit attestation from the concrete atomic-hook owner."""
        mro = type(self).__mro__
        capability_owner = next(
            index
            for index, owner in enumerate(mro)
            if "model_completion_recovery_fence_version" in owner.__dict__
        )
        mutation_owner = next(
            index
            for index, owner in enumerate(mro)
            if "_complete_model_completion_stage_atomic" in owner.__dict__
        )
        return (
            type(self.model_completion_recovery_fence_version) is int
            and self.model_completion_recovery_fence_version == 1
            and capability_owner <= mutation_owner
        )

    async def complete_recovered_model_completion_stage(
        self,
        session_id: str,
        *,
        stage_id: str,
        publication: RuntimePublicationRequest,
        recovery_fence: ModelCompletionStageRecoveryFence,
    ) -> ModelCompletionStageResult:
        """Insert synthetic auxiliary terminal evidence under an atomic recovery fence.

        Backends overriding the atomic hook must explicitly attest this protocol.
        Ordinary provider completion continues through complete_model_completion_stage.
        """
        if not self._supports_model_completion_recovery_fence_protocol():
            raise NotImplementedError("Store does not attest atomic model recovery fences.")
        if type(recovery_fence) is not ModelCompletionStageRecoveryFence:
            raise TypeError("Recovery completion requires a ModelCompletionStageRecoveryFence.")
        fence = ModelCompletionStageRecoveryFence.model_validate(
            {
                name: getattr(recovery_fence, name)
                for name in ModelCompletionStageRecoveryFence.model_fields
            }
        )
        if (
            fence.expected_active_profile.session_id != session_id
            or fence.expected_active_profile.run_epoch != fence.expected_run_epoch
        ):
            raise ValueError("Recovery fence conflicts with its session or epoch.")
        prepared = model_stage_rules._prepare_model_completion_stage_terminal(
            session_id,
            stage_id=stage_id,
            publication=publication,
            recovery_fence=fence,
        )
        if prepared.publication.kind != "auxiliary-inference":
            raise ValueError("Recovery-generated completion requires auxiliary publication.")
        return await self._complete_model_completion_stage_atomic(prepared)

    async def complete_model_completion_stage(
        self,
        session_id: str,
        *,
        stage_id: str,
        publication: RuntimePublicationRequest,
    ) -> ModelCompletionStageResult:
        """Append or exactly replay immutable terminal provider-attempt material.

        For auxiliary inference, terminal material records the explicit attempt
        outcome and accounting; it does not imply a successful model response.
        """

        prepared = model_stage_rules._prepare_model_completion_stage_terminal(
            session_id,
            stage_id=stage_id,
            publication=publication,
        )
        return await self._complete_model_completion_stage_atomic(prepared)

    async def abandon_model_completion_stage(
        self,
        session_id: str,
        *,
        stage_id: str,
        preparation_digest: str,
        expected_run_epoch: int,
        stage_source_run_epoch: int | None = None,
    ) -> ModelCompletionStageAbandonmentResult:
        """Atomically abandon one exact active stage before provider dispatch.

        An exact replay reconstructs the durable abandonment tombstone. A later
        preparation of the same request receives a new preparation digest, so a
        delayed retry of an older abandonment can never clear it.
        """

        prepared = model_stage_rules._prepare_model_completion_stage_abandonment(
            session_id,
            stage_id=stage_id,
            preparation_digest=preparation_digest,
            expected_run_epoch=expected_run_epoch,
            stage_source_run_epoch=stage_source_run_epoch,
        )
        return await self._abandon_model_completion_stage_atomic(prepared)

    async def promote_model_completion_stage(
        self,
        session_id: str,
        *,
        stage_id: str,
        expected_run_epoch: int,
    ) -> RuntimePublicationResult:
        """Atomically verify and publish a terminal stage without a load/publish race."""

        expected_run_epoch = session_authority_rules._validate_required_runtime_fence(
            expected_run_epoch,
            "expected_run_epoch",
        )
        session_id, stage_id, preparation_key, terminal_key = (
            model_stage_rules._model_completion_stage_storage_identity(session_id, stage_id)
        )
        return await self._promote_model_completion_stage_atomic(
            session_id=session_id,
            stage_id=stage_id,
            preparation_storage_key=preparation_key,
            terminal_storage_key=terminal_key,
            expected_run_epoch=expected_run_epoch,
        )

    async def _load_model_completion_stage_records(
        self,
        session_id: str,
        preparation_storage_key: str,
        terminal_storage_key: str,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """Backend stage lookup hook; intentionally fail closed for custom stores."""

        raise NotImplementedError(
            f"{type(self).__name__} does not support durable model-completion stages."
        )

    async def _load_model_completion_stage_settlement_record(
        self,
        session_id: str,
        settlement_storage_key: str,
    ) -> dict[str, Any] | None:
        """Backend settlement lookup hook; fail closed for custom stores."""

        raise NotImplementedError(
            f"{type(self).__name__} does not support durable model-completion settlements."
        )

    async def _load_model_completion_stage_dispatch_record(
        self,
        session_id: str,
        dispatch_storage_key: str,
    ) -> dict[str, Any] | None:
        """Backend dispatch lookup hook; fail closed for custom stores."""

        raise NotImplementedError(
            f"{type(self).__name__} does not support durable model-completion dispatches."
        )

    async def _load_active_model_completion_stage_records(
        self,
        session_id: str,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
        """Backend snapshot hook for the active marker and its immutable stage."""

        raise NotImplementedError(
            f"{type(self).__name__} does not support durable model-completion stages."
        )

    async def _mark_model_completion_stage_dispatched_atomic(
        self,
        session_id: str,
        *,
        stage: ModelCompletionStage,
        consume_child_session_notifications: bool,
    ) -> ModelCompletionStageDispatch:
        """Backend hook that validates active authority and inserts one dispatch receipt."""

        raise NotImplementedError(
            f"{type(self).__name__} does not support atomic model-completion dispatch."
        )

    async def _prepare_model_completion_stage_atomic(
        self,
        prepared: model_stage_rules._PreparedModelCompletionStage,
    ) -> ModelCompletionStageResult:
        """Backend hook for an insert-only, fenced preparation record."""

        raise NotImplementedError(
            f"{type(self).__name__} does not support durable model-completion stages."
        )

    async def _complete_model_completion_stage_atomic(
        self,
        prepared: model_stage_rules._PreparedModelCompletionStageTerminal,
    ) -> ModelCompletionStageResult:
        """Backend hook for an insert-only terminal provider-response record."""

        raise NotImplementedError(
            f"{type(self).__name__} does not support durable model-completion stages."
        )

    async def _abandon_model_completion_stage_atomic(
        self,
        prepared: model_stage_rules._PreparedModelCompletionStageAbandonment,
    ) -> ModelCompletionStageAbandonmentResult:
        """Backend hook for an exact, fenced pre-dispatch abandonment."""

        raise NotImplementedError(
            f"{type(self).__name__} does not support durable model-completion stages."
        )

    async def _promote_model_completion_stage_atomic(
        self,
        *,
        session_id: str,
        stage_id: str,
        preparation_storage_key: str,
        terminal_storage_key: str,
        expected_run_epoch: int,
    ) -> RuntimePublicationResult:
        """Backend hook that verifies the stage and publishes under one store lock."""

        raise NotImplementedError(
            f"{type(self).__name__} does not support durable model-completion stages."
        )

    async def load_session_export_snapshot(
        self,
        session_id: str,
        *,
        limits: SessionExportLimits | None = None,
    ) -> SessionExportSnapshot | None:
        """Read all export-owned components at one bounded per-session snapshot.

        Return None if the session has been deleted. No sequential-read fallback
        is safe for custom stores: implement one lock or database snapshot.
        """
        raise NotImplementedError("This SessionStore does not support consistent exports.")

    @abstractmethod
    async def load_events(self, session_id: str) -> list[Event]:
        """Load all events for a session."""

    async def load_user_input_supersession_events(
        self,
        session_id: str,
        input_id: str,
    ) -> list[Event]:
        """Load exact interruption evidence for one user-input pause.

        Built-in persistent stores override this with their fixed-size
        pending-action lookup indexes. The fallback pages complete durable
        history for correctness without retaining an unbounded result.
        """

        from cayu.approvals.user_input import USER_INPUT_SUPERSESSION_INTENT_KEY

        session_id = require_clean_nonblank(session_id, "session_id")
        input_id = require_clean_nonblank(input_id, "input_id")
        selected: list[Event] = []
        after_sequence = 0
        while True:
            records = await self.query_events(
                EventQuery(
                    session_id=session_id,
                    event_type=EventType.SESSION_INTERRUPTED,
                    after_sequence=after_sequence,
                    limit=5000,
                    order_by=EventOrder.SEQUENCE_ASC,
                )
            )
            if not records:
                break
            for record in records:
                marker = record.event.payload.get(USER_INPUT_SUPERSESSION_INTENT_KEY)
                if type(marker) is not dict or marker.get("input_id") != input_id:
                    continue
                selected.append(copy_event(record.event))
                if len(selected) == 2:
                    return selected
            after_sequence = records[-1].sequence
            if len(records) < 5000:
                break
        return selected

    async def load_tool_round_lifecycle_events(
        self,
        session_id: str,
        tool_call_ids: list[str] | tuple[str, ...],
    ) -> list[Event]:
        """Load exact started/terminal candidates for bounded tool-call identities.

        Built-in stores override this with their fixed-size pending-action
        indexes. The paginated fallback preserves correctness for custom stores
        without silently truncating lifecycle evidence.
        """

        session_id = require_clean_nonblank(session_id, "session_id")
        copied_ids = tool_publication_rules._validate_tool_round_call_ids(
            tool_call_ids,
            "tool_call_ids",
        )
        selected_ids = set(copied_ids)
        lifecycle_events: list[Event] = []
        after_sequence = 0
        while True:
            records = await self.query_events(
                EventQuery(
                    session_id=session_id,
                    event_types=tuple(tool_publication_rules._TOOL_ROUND_LIFECYCLE_EVENT_TYPES),
                    after_sequence=after_sequence,
                    limit=5000,
                    order_by=EventOrder.SEQUENCE_ASC,
                )
            )
            if not records:
                break
            for record in records:
                if record.event.payload.get("tool_call_id") not in selected_ids:
                    continue
                lifecycle_events.append(copy_event(record.event))
                if len(lifecycle_events) > publication_contracts._tool_round_lifecycle_event_limit(
                    copied_ids
                ):
                    raise ValueError("Tool-round lifecycle evidence exceeds the publication limit.")
            after_sequence = records[-1].sequence
            if len(records) < 5000:
                break
        return lifecycle_events

    async def load_tool_round_lifecycle_events_for_round(
        self,
        session_id: str,
        tool_call_ids: list[str] | tuple[str, ...],
        *,
        tool_round_identity: ToolRoundIdentity,
    ) -> list[Event]:
        """Load bounded lifecycle evidence for one exact logical tool round.

        Built-in stores provide indexed implementations. The default
        implementation pages over durable lifecycle events so reused provider
        call ids cannot hide current-round evidence behind the publication
        binding limit.
        """

        session_id = require_clean_nonblank(session_id, "session_id")
        copied_ids = tool_publication_rules._validate_tool_round_call_ids(
            tool_call_ids,
            "tool_call_ids",
        )
        tool_round_identity = copy_tool_round_identity(tool_round_identity)
        selected_ids = set(copied_ids)
        lifecycle_events: list[Event] = []
        after_sequence = 0
        while True:
            records = await self.query_events(
                EventQuery(
                    session_id=session_id,
                    event_types=tuple(tool_publication_rules._TOOL_ROUND_LIFECYCLE_EVENT_TYPES),
                    after_sequence=after_sequence,
                    limit=5000,
                    order_by=EventOrder.SEQUENCE_ASC,
                )
            )
            if not records:
                break
            for record in records:
                event = record.event
                if (
                    event.payload.get("tool_call_id") not in selected_ids
                    or not tool_publication_rules._tool_round_event_matches_scope_or_is_ambiguous(
                        event,
                        tool_round_identity,
                    )
                ):
                    continue
                lifecycle_events.append(copy_event(event))
                if len(lifecycle_events) > publication_contracts._tool_round_lifecycle_event_limit(
                    copied_ids
                ):
                    raise ValueError("Tool-round lifecycle evidence exceeds the publication limit.")
            after_sequence = records[-1].sequence
            if len(records) < 5000:
                break
        return lifecycle_events

    @abstractmethod
    async def query_events(self, query: EventQuery | None = None) -> list[EventRecord]:
        """Query stored events with durable sequence cursors."""

    async def read_usage_accounting(
        self, query: EventQuery, *, by_session: bool = False, by_identity: bool = False
    ) -> UsageAccountingSnapshot:
        """Reduce exact usage in one stable store snapshot with 256-row pages.

        The result retains totals, identity lists and optional per-session rows,
        never historical events. ``after_sequence`` is a single shared watermark
        across every usage-bearing event type. Unsupported stores fail closed.
        """
        raise NotImplementedError("This SessionStore does not support bounded usage accounting.")

    async def read_cost_accounting(
        self,
        query: EventQuery,
        pricing: PriceBook,
        *,
        currency: str = "USD",
        details: bool = False,
        by_session: bool = False,
        additional_events: tuple[Event, ...] = (),
        max_detail_bytes: int | None = None,
        previous: CostAccountingSnapshot | None = None,
    ) -> CostAccountingSnapshot:
        """Price a stable snapshot with bounded working memory plus requested output.

        Built-ins group hosted evidence and completions by session and attempt;
        totals-only reads retain no historical line items. Unsupported stores fail closed.
        """
        raise NotImplementedError("This SessionStore does not support bounded cost accounting.")

    async def event_exists(self, query: EventQuery) -> bool:
        """Test matching durable evidence without hydrating event payloads."""
        raise NotImplementedError("This SessionStore does not support event existence queries.")

    async def query_events_bounded(
        self,
        query: EventQuery,
        *,
        max_bytes: int,
    ) -> list[EventRecord]:
        """Query events only when the backend can enforce bytes before hydration."""

        raise NotImplementedError(
            f"{type(self).__name__} does not implement byte-bounded event queries."
        )

    async def load_bounded(self, session_id: str, *, max_bytes: int) -> Session | None:
        """Load a full session only after bounding transport bytes in one snapshot."""
        raise NotImplementedError("This SessionStore does not support byte-bounded session loads.")

    async def export_terminal_session_evidence(
        self,
        session_id: str,
        *,
        spool: EvidenceSpool,
    ) -> None:
        """Export a stable terminal snapshot to bounded indexed backing.

        Optional capability. Unsupported stores fail before reading payloads;
        callers must explicitly choose the separate bounded eager fallback.
        """
        raise NotImplementedError(
            "This SessionStore does not support incremental terminal evidence export."
        )

    async def load_terminal_session_evidence(
        self,
        session_id: str,
        *,
        limits: TerminalSessionEvidenceLimits | None = None,
    ) -> TerminalSessionEvidence:
        """Read one exact bounded snapshot through the current terminal event.

        Implementations must resolve terminal status, publication authority,
        the event cutoff, and the complete attributed transcript in one stable
        read. Unsupported, incomplete, contradictory, or oversized evidence
        fails with ``TerminalSessionEvidenceError``; it is never truncated.
        """

        del session_id, limits
        raise NotImplementedError(
            "This SessionStore does not support terminal-session evidence reads."
        )

    async def load_runner_owned_interrupted_evidence(
        self,
        session_id: str,
        *,
        observed_events: tuple[RunnerObservedEventIdentity, ...] | None = None,
        expected_parent_session_id: str | None = None,
        limits: TerminalSessionEvidenceLimits | None = None,
    ) -> TerminalSessionEvidence:
        """Read one exact interrupted snapshot from a runner-owned execution tree.

        A root read must compare the complete durable event-log identity with
        ``observed_events``. A descendant read instead proves its direct parent
        through ``expected_parent_session_id``; the caller must have established
        that parent as part of the same freshly executed tree. Exactly one proof
        is required. Implementations must apply record/count/byte limits before
        hydrating evidence, all inside one stable store snapshot. This optional
        operation does not broaden ordinary historical terminal-evidence reads.
        """

        del session_id, observed_events, expected_parent_session_id, limits
        raise NotImplementedError(
            "This SessionStore does not support runner-owned interrupted evidence reads."
        )

    async def query_latest_interaction_events(
        self,
        session_id: str,
        *,
        before_sequence: int | None = None,
        limit: int = 100,
    ) -> list[EventRecord]:
        """Return each interaction's latest lifecycle event in descending order.

        Built-in stores implement this as a native keyset query. The explicit
        capability error prevents endpoints from silently degrading to an
        unbounded event-log scan for out-of-tree stores.
        """

        raise NotImplementedError(
            "This SessionStore does not support native interaction pagination."
        )

    @abstractmethod
    async def summarize_events(self, session_id: str) -> EventSummary:
        """Summarize stored events for one session without loading every event."""

    @abstractmethod
    async def summarize_outcome(self, session_id: str) -> SessionOutcome:
        """Derive the current session outcome from durable events."""

    async def inspect_summary(
        self,
        session_id: str,
        *,
        checkpoint_root_guard: CheckpointRootFieldGuard | None = None,
    ) -> SessionInspectionSummary:
        """Return one exact, content-free diagnostic summary.

        The default implementation deliberately composes the public paginated
        query contracts so out-of-tree stores gain the same semantics without a
        backend-specific schema dependency. Implementations may replace it with
        an equivalent bounded native aggregate.
        """

        session_id = require_clean_nonblank(session_id, "session_id")
        identity = await self.inspect_identity(session_id)

        event_count = 0
        event_total_bytes = 0
        event_largest_bytes = 0
        usage_accumulator = session_inspection_rules._SessionInspectionUsageAccumulator()
        budget_events: list[Event] = []
        budget_model_attempt_terminal_events: list[Event] = []
        retained_event_bytes = 0
        queued_message_count = 0
        delivered_message_count = 0
        rejected_message_count = 0
        operation_event_count = 0
        after_sequence = 0
        while True:
            records = await self.query_events(
                EventQuery(
                    session_id=session_id,
                    after_sequence=after_sequence,
                    limit=session_inspection_rules._SESSION_INSPECTION_PAGE_SIZE,
                    order_by=EventOrder.SEQUENCE_ASC,
                )
            )
            if not records:
                break
            event_count += len(records)
            if event_count > session_inspection_rules._SESSION_INSPECTION_MAX_RECORDS:
                raise ValueError(
                    "Session inspection exceeds the "
                    f"{session_inspection_rules._SESSION_INSPECTION_MAX_RECORDS}-event safety limit."
                )
            for record in records:
                payload_bytes = compact_json_utf8_size(record.event.payload)
                event_total_bytes += payload_bytes
                event_largest_bytes = max(event_largest_bytes, payload_bytes)
                event = record.event
                if is_budget_model_attempt_terminal_event(event):
                    model_attempt_event = project_budget_model_attempt_inspection_event(event)
                    retained_event_bytes = (
                        session_inspection_rules._retain_session_inspection_event(
                            retained_event_bytes,
                            model_attempt_event,
                        )
                    )
                    budget_model_attempt_terminal_events.append(model_attempt_event)
                if event.type in {
                    EventType.MODEL_COMPLETED,
                    EventType.MODEL_HOSTED_TOOL_CALL,
                    EventType.TOOL_CALL_STARTED,
                }:
                    usage_event, usage_metrics = project_aggregate_usage_inspection_event(event)
                    retained_event_bytes = (
                        session_inspection_rules._retain_session_inspection_event(
                            retained_event_bytes,
                            usage_event,
                        )
                    )
                    usage_accumulator.add(event.type, usage_metrics)
                if is_budget_inspection_event(event):
                    budget_event = project_budget_inspection_event(event)
                    retained_event_bytes = (
                        session_inspection_rules._retain_session_inspection_event(
                            retained_event_bytes,
                            budget_event,
                        )
                    )
                    budget_events.append(budget_event)
                queued_message_count += event.type == EventType.SESSION_MESSAGE_QUEUED
                delivered_message_count += event.type == EventType.SESSION_MESSAGE_DELIVERED
                rejected_message_count += event.type in {
                    EventType.SESSION_MESSAGE_WITHDRAWN,
                    EventType.SESSION_MESSAGE_QUARANTINED,
                    EventType.SESSION_MESSAGE_STALE,
                    EventType.SESSION_MESSAGE_EXPIRED,
                }
                operation_event_count += event.type == EventType.SERVER_MUTATION_ACCEPTED
            after_sequence = records[-1].sequence
            if len(records) < session_inspection_rules._SESSION_INSPECTION_PAGE_SIZE:
                break

        transcript_count = 0
        transcript_total_bytes = 0
        transcript_largest_bytes = 0
        offset = 0
        while True:
            page = await self.query_transcript(
                TranscriptQuery(
                    session_id=session_id,
                    offset=offset,
                    limit=session_inspection_rules._SESSION_INSPECTION_PAGE_SIZE,
                )
            )
            if (
                offset == 0
                and page.total_records > session_inspection_rules._SESSION_INSPECTION_MAX_RECORDS
            ):
                raise ValueError(
                    "Session inspection exceeds the "
                    f"{session_inspection_rules._SESSION_INSPECTION_MAX_RECORDS}-message safety limit."
                )
            for record in page.records:
                message_bytes = compact_json_utf8_size(record.message.model_dump(mode="json"))
                transcript_count += 1
                transcript_total_bytes += message_bytes
                transcript_largest_bytes = max(transcript_largest_bytes, message_bytes)
            offset += session_inspection_rules._SESSION_INSPECTION_PAGE_SIZE
            if offset >= page.total_records:
                break

        pending = await self.query_pending_actions(
            PendingActionQuery(session_id=session_id, limit=200),
            checkpoint_root_guard=checkpoint_root_guard,
        )
        usage, model_calls_with_usage = usage_accumulator.result(session_id)
        budget = session_budget_inspection(
            budget_events,
            model_attempt_terminal_events=budget_model_attempt_terminal_events,
        )
        return SessionInspectionSummary(
            session=identity,
            transcript=SerializedRecordSummary(
                record_count=transcript_count,
                total_bytes=transcript_total_bytes,
                largest_record_bytes=transcript_largest_bytes,
            ),
            events=SerializedRecordSummary(
                record_count=event_count,
                total_bytes=event_total_bytes,
                largest_record_bytes=event_largest_bytes,
            ),
            usage=usage,
            model_calls=usage.model_steps,
            model_calls_with_usage=model_calls_with_usage,
            tool_calls=usage.tool_calls,
            pending_action_count=len(pending.actions),
            pending_action_kinds=tuple(
                sorted({action.kind for action in pending.actions}, key=str)
            ),
            pending_action_issue_count=len(pending.issues),
            queued_message_count=queued_message_count,
            delivered_message_count=delivered_message_count,
            outstanding_message_count=max(
                queued_message_count - delivered_message_count - rejected_message_count,
                0,
            ),
            operation_event_count=operation_event_count,
            terminal_failure_state=(
                "failed"
                if identity.status is SessionStatus.FAILED
                else "interrupted"
                if identity.status is SessionStatus.INTERRUPTED
                else "none"
            ),
            budget=budget,
        )

    @abstractmethod
    async def list_sessions(self, query: SessionQuery | None = None) -> SessionListResult:
        """List sessions (filtered/sorted/paginated) with a keyset cursor and total count."""

    async def query_session_topology(
        self,
        query: SessionTopologyQuery,
    ) -> SessionTopologyStoreResult:
        """Load bounded ancestry and batched direct-child branches.

        This optional store-native read prevents the control plane from issuing
        one list query per rendered node. Custom stores opt in explicitly through
        ``supports_session_topology``.
        """

        raise NotImplementedError(
            "This SessionStore does not support bounded session topology queries."
        )

    async def query_session_lineage(
        self,
        query: SessionLineageQuery,
    ) -> SessionLineageResult:
        """Load one minimal, byte-bounded direct-child lineage page.

        This optional store-native read is the durable admission boundary for
        historical trajectory capture. Implementations must project only the
        bounded structural fields represented by ``SessionLineageNode`` and must
        not hydrate session display fields or event payloads.
        """

        raise NotImplementedError(
            "This SessionStore does not support bounded session lineage queries."
        )

    async def query_child_session_lifecycle(
        self,
        query: ChildSessionLifecycleQuery,
    ) -> ChildSessionLifecyclePage:
        """Read bounded canonical direct-child state and notification freshness.

        Implementations advertising ``child_session_notification_version == 1``
        must resolve session state, its exact lifecycle occurrence, and any
        dispatch-fenced terminal consumption in one stable snapshot.
        """

        del query
        raise NotImplementedError(
            "This SessionStore does not support child-session lifecycle projections."
        )

    async def aggregate_operational_snapshot(
        self,
        filters: SessionAggregateFilter | None = None,
    ) -> SessionOperationalSnapshot:
        """Count current session states in one store-local read snapshot.

        Default raises ``NotImplementedError`` so adding this control-plane read
        model does not make existing out-of-tree stores uninstantiable.
        """
        raise NotImplementedError(
            "This SessionStore does not support operational aggregate snapshots."
        )

    async def aggregate_usage(self, query: UsageRollupQuery) -> UsageRollupStoreResult:
        """Aggregate bounded event-time activity and usage without loading histories.

        Default raises ``NotImplementedError`` so adding this control-plane read
        model does not make existing out-of-tree stores uninstantiable.
        """
        raise NotImplementedError("This SessionStore does not support usage aggregates.")

    @abstractmethod
    async def list_sessions_with_pending_interruption_cascade(
        self,
        query: SessionQuery | None = None,
    ) -> SessionListResult:
        """List only sessions carrying a pending interruption cascade marker.

        Implementations should filter at the storage layer. This is the
        restart-discovery path, so scanning all historical sessions and loading
        their checkpoints individually is not an acceptable implementation.
        """

    async def list_queued_dispatch_terminal_receipts(
        self,
        query: QueuedDispatchTerminalReceiptQuery | None = None,
    ) -> list[QueuedDispatchTerminalReceipt]:
        """List bounded live queue/session handoffs for restart reconciliation."""

        raise NotImplementedError(
            "This SessionStore does not support queued dispatch receipt discovery."
        )

    @abstractmethod
    async def query_pending_actions(
        self,
        query: PendingActionQuery | None = None,
        *,
        checkpoint_root_guard: CheckpointRootFieldGuard | None = None,
    ) -> PendingActionListResult:
        """List current checkpoint-backed actions without scanning session histories.

        Implementations must bound candidate and event reads, apply filtering at
        the storage layer where possible, and avoid per-session history queries.
        A supplied root guard runs on bounded opaque evidence in the same read
        snapshot before any pending state is interpreted.
        """

    async def validate_session_closure_admission(self, session_id: str) -> None:
        """Check every native active-operation/deletion guard without mutation.

        Read the guards in one store snapshot. This preflight is not an
        ownership claim and does not replace final atomic deletion checks.
        """
        raise NotImplementedError("This SessionStore does not support closure admission.")

    async def load_session_closure_records(
        self, session_id: str, *, max_records: int, max_bytes: int
    ) -> dict[str, Any]:
        """Return a bounded native closure snapshot with per-class counts/bytes.

        Read all included classes under one store snapshot. Exceeding a bound
        must return no partial records. Unsupported stores must fail explicitly.
        """
        raise NotImplementedError("This SessionStore does not support closure record snapshots.")

    async def delete_session(
        self,
        session_id: str,
        *,
        closure_receipt: dict[str, Any] | None = None,
    ) -> None:
        """Delete a session and cascade to its events, transcript, and checkpoint.

        Raises ``ValueError`` if the session is in-flight (``RUNNING`` or
        ``INTERRUPTING`` — interrupt it first), has incomplete current-run terminal
        evidence, still owns a durable subagent child, or if one of its accepted
        budget reservations has no fully delivered terminal settlement audit event.
        Durable subagent children must be settled and deleted before their parent so
        deletion cannot erase queue-admission authority while child work still exists.
        When ``closure_receipt`` is supplied, reject every remaining child edge
        in the deletion transaction, including non-subagent children. Closure
        callers must detach or delete those edges explicitly before finalization.
        Idempotent: deleting a session that does not exist is a no-op.

        Default raises ``NotImplementedError`` so out-of-tree stores keep working.
        """
        raise NotImplementedError("This SessionStore does not support delete_session.")

    async def apply_retention_policy(
        self,
        policy: SessionRetentionPolicy,
        *,
        protected_session_ids: Collection[str] = (),
        references: Mapping[RetentionProtection, Collection[str]] | None = None,
        progress: RetentionProgressCallback | None = None,
    ) -> RetentionReport:
        """Dry-run or apply a session retention policy; an operator-only action.

        Only sessions whose status the policy selects and whose last update is
        older than ``policy.older_than`` are considered. A store never compacts
        or deletes a session it cannot prove unreferenced: live tasks and
        leases, pending actions and clarifications, dependent context views,
        snapshot pins, eval and knowledge references, undelivered events and
        lineage members that must be kept all protect a session, and a lineage
        is pruned whole or not at all. ``protected_session_ids`` and
        ``references`` (session ids other stores reference, keyed by the
        protection they imply) add references the store cannot see. A dry run reports exactly what an apply
        with the same inputs and store state acts on. An apply changes one
        lineage per write transaction, re-checking it there, and releases the
        store between lineages; ``progress`` is awaited after planning and after
        each lineage. Every apply writes a durable audit record.

        Default raises ``NotImplementedError``; retention is off unless a store
        implements it and a caller applies a policy.
        """
        raise NotImplementedError("This SessionStore does not support storage retention.")

    async def inspect_session_retention(
        self,
        session_ids: Collection[str],
        *,
        mode: RetentionMode | None = None,
        include_store_guards: bool = True,
    ) -> dict[str, tuple[RetentionProtection, ...]]:
        """Report the direct retention protections of existing sessions.

        ``mode`` defaults to delete. Lineage and age selection are not applied. Other retention targets use
        this to keep resources that belong to protected sessions.
        """
        raise NotImplementedError("This SessionStore does not support storage retention.")

    async def retention_artifact_references(self) -> frozenset[str]:
        """Artifact ids that this store's transcripts, operations or evidence name."""
        raise NotImplementedError("This SessionStore does not support storage retention.")

    async def list_retention_audits(
        self,
        *,
        limit: int = 20,
        item_id: str | None = None,
        store_kind: str | None = None,
    ) -> tuple[RetentionAuditRecord, ...]:
        """List retention audit records, newest first, optionally filtered."""
        raise NotImplementedError("This SessionStore does not support storage retention.")

    async def load_retention_audit(self, audit_id: str) -> RetentionAuditRecord | None:
        """Load one retention audit record with every entry it wrote."""
        raise NotImplementedError("This SessionStore does not support storage retention.")

    async def begin_retention_audit(
        self,
        *,
        store_kind: str,
        mode: RetentionMode,
        policy: Mapping[str, Any],
        started_at: datetime,
    ) -> str:
        """Start a durable audit record for another store's retention apply."""
        raise NotImplementedError("This SessionStore does not support storage retention.")

    async def record_retention_entry(self, audit_id: str, entry: RetentionAuditEntry) -> None:
        raise NotImplementedError("This SessionStore does not support storage retention.")

    async def complete_retention_audit(
        self, audit_id: str, *, completed_at: datetime, summary: Mapping[str, Any]
    ) -> None:
        raise NotImplementedError("This SessionStore does not support storage retention.")

    async def detach_session_children(
        self,
        parent_session_id: str,
        child_session_ids: tuple[str, ...],
        *,
        closure_receipt: dict[str, Any],
    ) -> tuple[dict[str, Any], ...]:
        """Atomically detach children and retain auditable lineage tombstones.

        Replay binds the root/plan, original parent and exact unordered child
        set. A conflicting replay or invalid child must mutate no batch member.
        """
        raise NotImplementedError("This SessionStore does not support child detachment.")

    async def load_session_closure_tombstones(
        self, root_session_id: str, plan_id: str
    ) -> tuple[dict[str, Any], ...]:
        """Load durable lineage tombstones for one closure operation."""
        raise NotImplementedError("This SessionStore does not support closure tombstones.")

    async def load_session_closure_receipt(
        self, session_id: str, plan_id: str
    ) -> dict[str, Any] | None:
        """Load an exact durable closure receipt, if this store supports them."""
        raise NotImplementedError("This SessionStore does not support session closure receipts.")

    async def load_session_closure_progress(
        self, session_id: str, plan_id: str
    ) -> dict[str, Any] | None:
        raise NotImplementedError("This SessionStore does not support closure progress.")

    async def save_session_closure_progress(self, progress: dict[str, Any]) -> None:
        raise NotImplementedError("This SessionStore does not support closure progress.")

    async def claim_session_closure_progress(self, progress: dict[str, Any]) -> None:
        """Atomically validate and reserve a recursive plan's current lineage.

        Verify the complete recursive edge set, not just the listed edges,
        including plans with no descendants. Reject unexplained additions before
        publishing ownership. Creation must serialize with this admission and
        reject destinations retired by retained closure claims or receipts.
        Validate every target's native inventory against progress max_records
        and max_bytes in that transaction, before publishing any claim. Replay
        must preserve these admitted limits as part of the exact plan.
        Retain ownership across failures and restart. Detachment and competing
        deletion must reject claimed sessions; only exact closure deletion may
        consume them. No lease expiry may release a still-live cleanup owner.
        """
        raise NotImplementedError("This SessionStore does not support closure lineage claims.")

    async def update_labels(self, session_id: str, labels: dict[str, str]) -> Session:
        """Replace a session's labels (full replacement, not a merge) and return it.

        Annotation writes update ``updated_at`` but do not refresh the runtime
        recovery signal in ``last_activity_at``.

        Raises ``KeyError`` if the session does not exist. Default raises
        ``NotImplementedError`` so out-of-tree stores keep working.
        """
        raise NotImplementedError("This SessionStore does not support update_labels.")

    async def update_metadata(self, session_id: str, metadata: dict[str, Any]) -> Session:
        """Replace a session's user metadata and return the updated session.

        This is a full replacement of user-authored metadata, not a merge.
        Runtime-owned entries are retained atomically and cannot be supplied by
        callers. Annotation writes update ``updated_at`` but do not refresh the
        runtime recovery signal in ``last_activity_at``.

        Raises ``KeyError`` if the session does not exist. Default raises
        ``NotImplementedError`` so out-of-tree stores keep working.
        """
        raise NotImplementedError("This SessionStore does not support update_metadata.")

    @abstractmethod
    async def append_transcript_messages(
        self,
        session_id: str,
        messages: list[Message],
        *,
        interaction_id: InteractionAttribution = INHERIT_INTERACTION,
    ) -> None:
        """Append provider-neutral transcript messages to a session.

        Stored state must not alias caller-passed messages: implementations
        must detach nested JSON payloads (serialize, or use
        `cayu.messages.detach_message`) so a producer mutating a message
        after append cannot rewrite history.
        """

    @abstractmethod
    async def replace_initial_transcript_messages(
        self,
        session_id: str,
        expected_messages: list[Message],
        replacement_messages: list[Message],
        *,
        interaction_id: InteractionAttribution = INHERIT_INTERACTION,
        checkpoint_transform: CheckpointTransform | None = None,
        runtime_suffix_count: int = 0,
    ) -> None:
        """Atomically materialize deferred input as the ordered initial transcript.

        The replacement must contain ``expected_messages`` after any leading
        bootstrap/system messages and before ``runtime_suffix_count`` runtime-owned
        messages. The bootstrap prefix is stored with null interaction attribution;
        the admitted source and runtime suffix are attributed to ``interaction_id``.
        When deferred input retains a runtime-authenticated complete initial
        projection, the replacement must equal that projection exactly. No partial
        or externally visible pre-finalization transcript is permitted.
        When supplied, ``checkpoint_transform`` runs inside the same write
        transaction before the initial-transcript authority marker is read.
        """

    @abstractmethod
    async def load_deferred_interaction_input(
        self,
        session_id: str,
    ) -> DeferredInteractionInput | None:
        """Load the private admitted source batch, if one still exists."""

    @abstractmethod
    async def materialize_deferred_interaction_input(
        self,
        session_id: str,
        *,
        interaction_id: InteractionAttribution = INHERIT_INTERACTION,
    ) -> bool:
        """Materialize deferred source input without a bootstrap prefix.

        Returns ``True`` when a pending admission was materialized and ``False``
        when no matching deferred input exists.  Recovery and setup-failure paths
        use this before making the owning interaction or session terminal.
        """

    @abstractmethod
    async def append_transcript_messages_and_transform_checkpoint(
        self,
        session_id: str,
        messages: list[Message],
        checkpoint_transform: CheckpointTransform,
        *,
        interaction_id: InteractionAttribution = INHERIT_INTERACTION,
    ) -> None:
        """Append transcript messages and atomically transform the current checkpoint.

        The transform must be synchronous and thread-safe; a store may execute
        it on a worker thread while holding its transactional write boundary.
        """

    @abstractmethod
    async def load_transcript(self, session_id: str) -> list[Message]:
        """Load provider-neutral transcript messages for a session.

        Returned messages must not alias stored state: callers own the result
        and may mutate nested payloads without corrupting the transcript.
        """

    @abstractmethod
    async def load_transcript_snapshot(self, session_id: str) -> TranscriptSnapshot:
        """Load retained records and the permanent cursor in one read snapshot.

        Unlike ``len(load_transcript(...))``, ``cursor`` does not move backwards
        when a backend applies physical transcript retention.
        """

    async def load_transcript_cursor(self, session_id: str) -> int:
        """Load the permanent append cursor without requiring retained rows."""

        return (await self.load_transcript_snapshot(session_id)).cursor

    async def load_transcript_window(
        self,
        session_id: str,
        *,
        start_index: int,
        limit: int,
    ) -> TranscriptSnapshot:
        """Load a bounded absolute-index window and its permanent append cursor.

        ``start_index`` is inclusive. If its message was removed, return the
        first surviving records at or after it; at or beyond the append cursor,
        return no records. The returned cursor is the full logical extent, not
        the end of this window. Continue from the last returned index plus one.
        An empty window does not prove that earlier payloads never existed.

        Stores may override this method to avoid loading the complete retained
        transcript. The default keeps custom stores correct through their atomic
        ``load_transcript_snapshot`` implementation.
        """

        session_id = require_clean_nonblank(session_id, "session_id")
        if type(start_index) is not int:
            raise TypeError("start_index must be an integer.")
        if not 0 <= start_index <= MAX_DURABLE_JSON_INTEGER:
            raise ValueError("start_index exceeds the durable integer limit.")
        if type(limit) is not int:
            raise TypeError("limit must be an integer.")
        if not 1 <= limit <= 5000:
            raise ValueError("limit must be between 1 and 5000.")
        snapshot = await self.load_transcript_snapshot(session_id)
        start = bisect_left(
            [record.index for record in snapshot.records],
            start_index,
        )
        return TranscriptSnapshot(
            records=[
                record.model_copy(deep=True) for record in snapshot.records[start : start + limit]
            ],
            cursor=snapshot.cursor,
        )

    async def load_latest_transcript_message(
        self,
        session_id: str,
        *,
        role: MessageRole,
    ) -> TranscriptRecord | None:
        """Load one latest role-matched record without requiring full history.

        Stores must override this method with a reverse-indexed read. There is
        deliberately no OFFSET-based fallback because its work grows with the
        transcript and would make callers' bounded-read guarantees dishonest.
        """

        session_id = require_clean_nonblank(session_id, "session_id")
        if not isinstance(role, MessageRole):
            raise TypeError("role must be a MessageRole.")
        raise NotImplementedError(
            f"{type(self).__name__} does not support reverse-indexed transcript reads."
        )

    async def load_latest_transcript_text(
        self,
        session_id: str,
        *,
        role: MessageRole,
        max_chars: int,
    ) -> tuple[str, bool] | None:
        """Load a bounded text projection without hydrating the source message."""

        session_id = require_clean_nonblank(session_id, "session_id")
        if not isinstance(role, MessageRole):
            raise TypeError("role must be a MessageRole.")
        if type(max_chars) is not int:
            raise TypeError("max_chars must be an integer.")
        if not 1 <= max_chars <= LATEST_TRANSCRIPT_TEXT_MAX_CHARS:
            raise ValueError(f"max_chars must be between 1 and {LATEST_TRANSCRIPT_TEXT_MAX_CHARS}.")
        raise NotImplementedError(
            f"{type(self).__name__} does not support bounded transcript-text reads."
        )

    @abstractmethod
    async def query_transcript(self, query: TranscriptQuery) -> TranscriptPage:
        """Query provider-neutral transcript messages with stable message indexes.

        Offset and total_records refer to retained role/interaction-matched
        rows, not logical extent. Thinking exclusion is applied after pagination;
        advance by the raw page span, not the number of projected records.
        Same result-ownership isolation contract as `load_transcript`.
        """

    async def search_transcript(
        self,
        query: TranscriptSearchQuery,
    ) -> TranscriptSearchResult:
        """Search explicitly selected transcript narratives within hard bounds.

        This optional capability fails closed for custom stores until they opt in.
        Implementations must enforce the supplied session set inside the query and
        must never interpret an empty or omitted set as global access.
        """

        raise NotImplementedError("This SessionStore does not support transcript search.")

    async def create_recall_receipt(self, receipt: RecallReceipt) -> RecallReceipt:
        """Idempotently persist one immutable, session-scoped recall receipt."""

        raise NotImplementedError("This SessionStore does not support recall evidence.")

    async def load_recall_receipt(
        self,
        session_id: str,
        receipt_id: str,
    ) -> RecallReceipt | None:
        """Load one exact receipt without widening its session scope."""

        raise NotImplementedError("This SessionStore does not support recall evidence.")

    async def list_recall_receipts(
        self,
        query: RecallEvidenceQuery,
    ) -> RecallReceiptPage:
        """List a deterministic, byte-bounded page of scoped recall receipts."""

        raise NotImplementedError("This SessionStore does not support recall evidence.")

    async def create_context_exposure(
        self,
        exposure: ContextExposure,
        item_exposures: tuple[RecallItemExposure, ...] = (),
    ) -> ContextExposure:
        """Atomically persist one planned composition and its exact recalled items."""

        raise NotImplementedError("This SessionStore does not support recall evidence.")

    async def load_context_exposure(
        self,
        session_id: str,
        exposure_id: str,
    ) -> ContextExposure | None:
        """Load one exact provider-attempt lifecycle within its session scope."""

        raise NotImplementedError("This SessionStore does not support recall evidence.")

    async def load_recall_item_exposures(
        self,
        session_id: str,
        exposure_id: str,
    ) -> tuple[RecallItemExposure, ...]:
        """Load the complete bounded item set atomically linked to one exposure."""

        raise NotImplementedError("This SessionStore does not support recall evidence.")

    async def list_context_exposures(
        self,
        query: RecallEvidenceQuery,
    ) -> ContextExposurePage:
        """List a deterministic, byte-bounded page of scoped exposure lifecycles."""

        raise NotImplementedError("This SessionStore does not support recall evidence.")

    async def transition_context_exposure(
        self,
        session_id: str,
        exposure_id: str,
        request: ContextExposureTransitionRequest,
    ) -> ContextExposure:
        """Apply one idempotent lifecycle transition behind an exact CAS fence."""

        raise NotImplementedError("This SessionStore does not support recall evidence.")

    @abstractmethod
    async def checkpoint(self, session_id: str, state: dict[str, Any]) -> None:
        """Persist a checkpoint for resume/replay."""

    @abstractmethod
    async def transform_checkpoint(
        self,
        session_id: str,
        checkpoint_transform: CheckpointTransform,
    ) -> None:
        """Atomically transform the latest checkpoint for a session.

        The transform runs while the store owns its session/checkpoint write
        boundary and must be synchronous and thread-safe because a store may
        execute it on a worker thread. Returning ``None`` leaves the checkpoint
        unchanged.
        """

    @abstractmethod
    async def transform_checkpoint_with_store_time(
        self,
        session_id: str,
        checkpoint_transform: StoreTimeCheckpointTransform,
    ) -> None:
        """Transform a checkpoint with time sampled by the store transaction.

        The same sampled value must govern any lease comparison made by the
        transform and the resulting checkpoint/activity timestamps.
        """

    @abstractmethod
    async def load_checkpoint(self, session_id: str) -> dict[str, Any] | None:
        """Load the latest checkpoint for a session."""

    async def load_execution_snapshot_checkpoint(self, session_id: str) -> dict[str, Any] | None:
        """Operator inspection of snapshot metadata only, without other checkpoint roots.

        Custom stores may override this fallback with a native bounded projection.
        """
        checkpoint = await self.load_checkpoint(session_id)
        return (
            None
            if checkpoint is None
            else {
                key: value
                for key, value in checkpoint.items()
                if key in {"checkpoint_schema_version", "execution_snapshots"}
            }
        )

    @abstractmethod
    async def load_interruption_cascade_marker(
        self,
        session_id: str,
        *,
        checkpoint_root_guard: CheckpointRootFieldGuard | None = None,
    ) -> dict[str, Any] | None:
        """Load a bounded structural projection of the durable cascade marker.

        A supplied root guard runs on bounded opaque evidence in the same read
        snapshot before the marker is interpreted.
        """


@dataclass(frozen=True)
class _PreparedInMemoryEvent:
    record: EventRecord
    event_type: str
    projected_record: EventRecord | None
    retained_lookup_keys: tuple[str, ...]
    history_lookup_keys: tuple[str, ...]
    delivery: PersistedEventSideEffectDelivery | None
    invocation_terminal_receipt: tuple[str, dict[str, Any]] | None


@dataclass(frozen=True)
class _PreparedInMemoryEventAppend:
    events: tuple[_PreparedInMemoryEvent, ...]
    next_sequence: int
    budget_reservation_publications: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class _PreparedInMemoryCheckpointStore:
    checkpoint: dict[str, Any]
    has_pending_action: bool
    rebuilt_records: dict[str, dict[str, EventRecord]] | None = None
    rebuilt_event_history: dict[str, list[EventRecord]] | None = None
    rebuilt_index_scope: tuple[frozenset[str], str | None, str | None, str | None] | None = None


@dataclass(frozen=True)
class _InMemoryMessageDeliveryRecord:
    session_id: str
    include_on_idle: bool
    requested_eligible_through: int | None
    limit: int
    interaction_id: str | None
    interaction_started_event: Event | None
    profile_handoff: QueuedInteractionProfileHandoff | None
    batch: SessionMessageDeliveryBatch
    reject_only: bool = False


def _validate_session_closure_detach_replay(
    tombstones: tuple[dict[str, Any], ...],
    *,
    root_id: str,
    plan_id: str,
    parent_session_id: str,
    child_session_ids: tuple[str, ...],
) -> None:
    if (
        len(tombstones) != len(child_session_ids)
        or {item.get("child_session_id") for item in tombstones} != set(child_session_ids)
        or any(
            item.get("root_session_id") != root_id
            or item.get("plan_id") != plan_id
            or item.get("original_parent_session_id") != parent_session_id
            for item in tombstones
        )
    ):
        raise ValueError("Closure detachment replay identity conflict.")


def _closure_progress_targets(progress: dict[str, Any]) -> set[str]:
    return {progress["root_session_id"], *(item["session_id"] for item in progress["descendants"])}


def _check_closure_lineage_owner(
    progress: dict[str, Any], session_ids: Iterable[str], receipt: dict[str, Any] | None = None
) -> None:
    """Only the exact recursive owner may delete a claimed session."""
    from cayu.runtime.session_closure import session_closure_target_plan_id

    for session_id in set(session_ids) & _closure_progress_targets(progress):
        root_id, plan_id = progress["root_session_id"], progress["plan_id"]
        expected_plan = (
            plan_id
            if session_id == root_id
            else session_closure_target_plan_id(plan_id, session_id)
        )
        if receipt is None or receipt.get("plan_id") != expected_plan:
            raise ValueError("Session lineage is owned by an unfinished recursive closure.")
        if session_id != root_id and (
            receipt.get("root_session_id") != root_id
            or receipt.get("target_session_id") != session_id
            or receipt.get("operation") != "recursive"
        ):
            raise ValueError("Recursive closure deletion owner conflicts.")


def _validate_closure_progress_update(existing: dict[str, Any], proposed: dict[str, Any]) -> None:
    for key in (
        "schema_version",
        "root_session_id",
        "plan_id",
        "policy_digest",
        "max_records",
        "max_bytes",
        "phase",
        "descendants",
    ):
        if existing.get(key) != proposed.get(key):
            raise ValueError("Closure progress identity conflict.")
    if not set(existing["completed"]) <= set(proposed["completed"]):
        raise ValueError("Closure progress cannot forget completed descendants.")


class _InMemoryContextViewSelectionTransaction:
    """Rollback selection-side expiry evidence when validation rejects a request."""

    def __init__(self, store: InMemorySessionStore) -> None:
        self._store = store
        self._receipts: dict[str, Any] | None = None
        self._commitments: dict[str, str] | None = None
        self._events: dict[str, Any] | None = None

    async def __aenter__(self) -> _InMemoryContextViewSelectionTransaction:
        await self._store._lock.acquire()
        self._receipts = self._store._context_selection_receipts.copy()
        self._commitments = self._store._context_selection_request_commitments.copy()
        self._events = self._store._context_lifecycle_events.copy()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        if exc_type is not None:
            assert self._receipts is not None
            assert self._commitments is not None
            assert self._events is not None
            self._store._context_selection_receipts = self._receipts
            self._store._context_selection_request_commitments = self._commitments
            self._store._context_lifecycle_events = self._events
        self._store._lock.release()


def runtime_session_mutation(operation):
    from functools import wraps

    @wraps(operation)
    async def guarded(self, *args, **kwargs):
        from cayu.sessions._access_rules import runtime_session_mutation as wrap

        return await wrap(operation)(self, *args, **kwargs)

    return guarded


def require_resource_session(session, action="read"):
    from cayu.sessions._access_rules import require_resource_session as require

    require(session, action)


def runtime_session_query(operation):
    from functools import wraps

    @wraps(operation)
    async def guarded(self, *args, **kwargs):
        from cayu.sessions._access_rules import runtime_session_query as wrap

        return await wrap(operation)(self, *args, **kwargs)

    return guarded


@model_store_surface("sessions")
class InMemorySessionStore(
    MemoryExternalWaitMixin, MemorySessionExecutionMixin, MemoryCreationFenceMixin, SessionStore
):
    """In-process session store for tests, local development, and examples."""

    session_access_version: ClassVar[int | None] = 1

    async def _access_create_session(self, bounds, request, identity):
        from cayu.sessions._access_rules import _creation_bounds

        token = _creation_bounds.set(bounds)
        try:
            return await self.create(request, identity=identity)
        finally:
            _creation_bounds.reset(token)

    async def _access_update_metadata(self, bounds, session_id, metadata):
        return await self.update_metadata(session_id, metadata, _access_bounds=bounds)

    async def _access_delete_session(self, bounds, session_id):
        await self.delete_session(session_id, _access_bounds=bounds)

    async def _access_read_records(self, bounds, session_id, kind, offset, limit, max_bytes):
        from cayu.storage._session_access_records import memory_read

        return await memory_read(self, bounds, session_id, kind, offset, limit, max_bytes)

    async def _access_list_sessions(
        self, bounds: _SessionAccessBounds, query: SessionQuery
    ) -> SessionListResult:
        return await self._list_sessions(
            query, pending_interruption_cascade_only=False, access_bounds=bounds
        )

    async def _access_update_labels(
        self, bounds: _SessionAccessBounds, session_id: str, labels: dict[str, str]
    ) -> Session:
        return await self.update_labels(session_id, labels, _access_bounds=bounds)

    async def _access_load_session(self, bounds: _SessionAccessBounds, session_id: str) -> Session:
        session_id = require_clean_nonblank(session_id, "session_id")
        async with self._lock:
            return bounds.require_read(self._sessions.get(session_id)).model_copy(deep=True)

    peer_content_version: ClassVar[int | None] = 1

    session_export_version: ClassVar[int] = 1
    session_continuation_version: ClassVar[int] = 1
    external_wait_version: ClassVar[int] = 1
    _producer_attachment_version: ClassVar[int] = 1
    supports_session_closure_receipts: ClassVar[bool] = True
    supports_session_closure_detachment: ClassVar[bool] = True
    supports_session_closure_recursive_deletion: ClassVar[bool] = True
    supports_session_closure_progress: ClassVar[bool] = True

    async def load_session_closure_records(
        self, session_id: str, *, max_records: int, max_bytes: int
    ) -> dict[str, Any]:
        async with self._lock:
            return self._load_session_closure_records_unlocked(
                session_id, max_records=max_records, max_bytes=max_bytes
            )

    def _load_session_closure_records_unlocked(
        self, session_id: str, *, max_records: int, max_bytes: int
    ) -> dict[str, Any]:
        from cayu.runtime._session_closure_records import ClosureRecordsBuilder

        session_id = require_clean_nonblank(session_id, "session_id")
        builder = ClosureRecordsBuilder(max_records=max_records, max_bytes=max_bytes)
        session = self._sessions.get(session_id)
        builder.add_class(
            "session",
            ()
            if session is None
            else (
                {
                    name: getattr(session, name)
                    for name in type(session).model_fields
                    if name not in {"labels", "metadata"}
                },
            ),
        )
        builder.add_class(
            "labels",
            ()
            if session is None
            else ({"key": key, "value": value} for key, value in session.labels.items()),
        )
        builder.add_class("metadata", () if session is None else (session.metadata,))
        builder.add_class(
            "recall_receipts",
            (
                self._recall_receipts[receipt_id]
                for _, receipt_id in self._recall_receipt_page_keys_by_session.get(session_id, ())
            ),
        )
        builder.add_class(
            "context_exposures",
            (
                self._context_exposures[exposure_id]
                for _, exposure_id in self._context_exposure_page_keys_by_session.get(
                    session_id, ()
                )
            ),
        )
        builder.add_class(
            "recall_item_exposures",
            (
                item
                for _, exposure_id in self._context_exposure_page_keys_by_session.get(
                    session_id, ()
                )
                for item in self._recall_item_exposures[exposure_id]
            ),
        )
        builder.add_class("events", self._session_event_records.get(session_id, ()))
        builder.add_class(
            "transcript",
            (
                {
                    "transcript_index": index,
                    "message": message,
                    "interaction_id": self._transcript_interaction_ids[session_id][index],
                }
                for index, message in enumerate(self._transcripts.get(session_id, ()))
            ),
        )
        checkpoint = self._checkpoints.get(session_id)
        builder.add_class("checkpoint", () if checkpoint is None else (checkpoint,))
        builder.add_class(
            "queued_messages",
            (
                {
                    "message": message,
                    "terminal": self._session_message_terminal_receipts.get(
                        (session_id, message.queue_id)
                    ),
                }
                for message in self._queued_session_messages_by_idempotency.get(
                    session_id, {}
                ).values()
            ),
        )
        builder.add_class(
            "session_operations",
            (
                {"idempotency_key": key, "record": value}
                for key, value in self._session_operation_records.get(session_id, {}).items()
            ),
        )
        builder.add_class(
            "event_side_effect_deliveries",
            (
                delivery
                for record in self._session_event_records.get(session_id, ())
                if (
                    delivery := self._persisted_event_side_effect_deliveries.get(
                        (session_id, record.event.id)
                    )
                )
                is not None
            ),
        )
        builder.add_class(
            "queue_deliveries",
            (
                {
                    "delivery_id": key,
                    "session_id": value.session_id,
                    "include_on_idle": value.include_on_idle,
                    "requested_eligible_through": value.requested_eligible_through,
                    "eligible_through": value.batch.eligible_through,
                    "batch_limit": value.limit,
                    "has_more": value.batch.has_more,
                    "interaction_id": value.interaction_id,
                    "interaction_started_event": value.interaction_started_event,
                    "queue_ids": [message.queue_id for message in value.batch.messages],
                    "events": value.batch.events,
                    "reject_only": value.reject_only,
                }
                for key, value in self._session_message_delivery_records.items()
                if value.session_id == session_id
            ),
        )
        deferred = self._deferred_interaction_inputs.get(session_id)
        builder.add_class("deferred_interaction_inputs", () if deferred is None else (deferred,))
        builder.add_class(
            "targeted_tool_grants",
            (
                targeted_tool_grant_with_active_reference(
                    self._targeted_tool_grants[grant_id], self.public_authority_alias_codec
                )
                for grant_id in self._targeted_tool_grant_ids_by_session.get(session_id, ())
            ),
        )
        builder.add_class(
            "targeted_tool_grant_uses",
            (
                use
                for grant_id in self._targeted_tool_grant_ids_by_session.get(session_id, ())
                for use in self._targeted_tool_uses.get(grant_id, {}).values()
            ),
        )
        return builder.finish()

    async def load_session_closure_receipt(
        self, session_id: str, plan_id: str
    ) -> dict[str, Any] | None:
        require_clean_nonblank(session_id, "session_id")
        require_clean_nonblank(plan_id, "plan_id")
        async with self._lock:
            receipt = self._session_closure_receipts.get((session_id, plan_id))
            return None if receipt is None else deepcopy(receipt)

    async def load_session_closure_progress(self, session_id: str, plan_id: str):
        async with self._lock:
            value = self._session_closure_progress.get((session_id, plan_id))
            return None if value is None else deepcopy(value)

    async def save_session_closure_progress(self, progress: dict[str, Any]) -> None:
        root_id = progress.get("root_session_id")
        plan_id = progress.get("plan_id")
        if type(root_id) is not str or type(plan_id) is not str:
            raise ValueError("Closure progress is missing identity.")
        async with self._lock:
            existing = self._session_closure_progress.get((root_id, plan_id))
            if existing is not None:
                _validate_closure_progress_update(existing, progress)
            self._session_closure_progress[(root_id, plan_id)] = deepcopy(progress)

    async def claim_session_closure_progress(self, progress: dict[str, Any]) -> None:
        progress = deepcopy(progress)
        async with self._lock:
            key = (progress["root_session_id"], progress["plan_id"])
            existing = self._session_closure_progress.get(key)
            if existing is not None:
                _validate_closure_progress_update(existing, progress)
                return
            targets = _closure_progress_targets(progress)
            for owner in self._session_closure_progress.values():
                _check_closure_lineage_owner(owner, targets)
            if progress["root_session_id"] not in self._sessions:
                raise ValueError("Closure root disappeared before lineage admission.")
            for item in progress["descendants"]:
                child = self._sessions.get(item["session_id"])
                if child is None or child.parent_session_id != item["parent_session_id"]:
                    raise ValueError("Child lineage changed before closure admission.")
            if progress["phase"] in {"recursive", "reject"}:
                expected = {
                    (item["session_id"], item["parent_session_id"])
                    for item in progress["descendants"]
                }
                observed = set()
                for parent_id in targets:
                    for _, child_id in self._child_session_keys_by_parent.get(parent_id, ()):
                        edge = (child_id, parent_id)
                        if edge not in expected:
                            raise ValueError("Child lineage changed before closure admission.")
                        observed.add(edge)
                if observed != expected:
                    raise ValueError("Child lineage changed before closure admission.")
            for target_id in targets:
                self._require_session_erasure_quiescence_unlocked(self._sessions[target_id])
                self._load_session_closure_records_unlocked(
                    target_id, max_records=progress["max_records"], max_bytes=progress["max_bytes"]
                )
            self._session_closure_progress[key] = progress

    def _require_available_closure_identity_unlocked(self, session_id: str) -> None:
        for owner in self._session_closure_progress.values():
            _check_closure_lineage_owner(owner, (session_id,))
        if any(target == session_id for target, _plan in self._session_closure_receipts):
            raise ValueError("Session identity was retired by closure.")

    async def load_session_closure_tombstones(
        self, root_session_id: str, plan_id: str
    ) -> tuple[dict[str, Any], ...]:
        require_clean_nonblank(root_session_id, "root_session_id")
        require_clean_nonblank(plan_id, "plan_id")
        async with self._lock:
            return tuple(
                deepcopy(value)
                for (root_id, stored_plan, _child_id), value in sorted(
                    self._session_closure_tombstones.items()
                )
                if root_id == root_session_id and stored_plan == plan_id
            )

    async def detach_session_children(
        self,
        parent_session_id: str,
        child_session_ids: tuple[str, ...],
        *,
        closure_receipt: dict[str, Any],
    ) -> tuple[dict[str, Any], ...]:
        parent_session_id = require_clean_nonblank(parent_session_id, "parent_session_id")
        child_session_ids = tuple(
            require_clean_nonblank(child_id, "child_session_id") for child_id in child_session_ids
        )
        if len(set(child_session_ids)) != len(child_session_ids):
            raise ValueError("Detached child session IDs must be unique.")
        root_id = closure_receipt.get("root_session_id")
        plan_id = closure_receipt.get("plan_id")
        if type(root_id) is not str or type(plan_id) is not str:
            raise ValueError("Closure detachment receipt is missing identity.")
        async with self._lock:
            existing = tuple(
                deepcopy(value)
                for (stored_root, stored_plan, _child_id), value in sorted(
                    self._session_closure_tombstones.items()
                )
                if stored_root == root_id and stored_plan == plan_id
            )
            if existing:
                _validate_session_closure_detach_replay(
                    existing,
                    root_id=root_id,
                    plan_id=plan_id,
                    parent_session_id=parent_session_id,
                    child_session_ids=child_session_ids,
                )
                return existing
            for owner in self._session_closure_progress.values():
                _check_closure_lineage_owner(owner, child_session_ids)
            parent = self._sessions.get(parent_session_id)
            if parent is None:
                raise KeyError(parent_session_id)
            tombstones: list[dict[str, Any]] = []
            prepared = []
            detached_at = self._ownership_clock()
            for child_id in child_session_ids:
                child = self._sessions.get(child_id)
                if child is None or child.parent_session_id != parent_session_id:
                    raise ValueError("Child lineage changed before detachment.")
                tombstone = {
                    "root_session_id": root_id,
                    "plan_id": plan_id,
                    "child_session_id": child_id,
                    "original_parent_session_id": parent_session_id,
                    "detached_at": detached_at.isoformat(),
                }
                detached = child.model_copy(
                    update={"parent_session_id": None, "updated_at": detached_at}
                )
                prepared.append((child_id, child, detached, deepcopy(tombstone)))
                tombstones.append(tombstone)
            for child_id, child, detached, tombstone in prepared:
                self._remove_session_parent_index_unlocked(child)
                self._sessions[child_id] = detached
                self._refresh_child_lifecycle_candidate_unlocked(detached)
                self._session_closure_tombstones[(root_id, plan_id, child_id)] = tombstone
            return tuple(tombstones)

    supports_usage_aggregates: ClassVar[bool] = True
    supports_private_argument_continuity: ClassVar[bool] = True
    supports_mcp_manifest_history: ClassVar[bool] = True
    supports_session_topology: ClassVar[bool] = True
    supports_session_lineage: ClassVar[bool] = True
    child_session_notification_version: ClassVar[int | None] = 1
    supports_public_authority_aliases: ClassVar[bool] = True
    supports_targeted_tool_grants: ClassVar[bool] = True
    supports_terminal_session_evidence: ClassVar[bool] = True
    supports_runner_owned_interrupted_evidence: ClassVar[bool] = True
    supports_execution_profile_admission: ClassVar[bool] = True
    model_failover_stage_version: ClassVar[int] = 1
    supports_active_invocation_execution_profiles: ClassVar[bool] = True
    invocation_lifecycle_command_version: ClassVar[int | None] = 1
    terminal_interaction_publication_version: ClassVar[int | None] = 1
    durable_model_terminalization_version: ClassVar[int | None] = 1
    queued_interaction_profile_handoff_version: ClassVar[int | None] = 1
    session_message_lifecycle_version: ClassVar[int | None] = 1
    supports_pending_session_initial_checkpoint: ClassVar[bool] = True
    supports_profiled_forks: ClassVar[bool] = True
    supports_atomic_session_operation_initialization: ClassVar[bool] = True
    supports_atomic_model_completion_stage_release: ClassVar[bool] = True
    model_completion_recovery_fence_version: ClassVar[int] = 1
    session_steering_version: ClassVar[int | None] = 1
    supports_completion_result_event_publication_reservations: ClassVar[bool] = True
    supports_transcript_search: ClassVar[bool] = True
    supports_recall_evidence: ClassVar[bool] = True
    supports_owned_off_thread_session_commit_guards: ClassVar[bool] = True
    participant_session_binding_version: ClassVar[int | None] = 1
    recipient_continuation_selection_version: ClassVar[int | None] = 1
    context_view_selection_fence_version: ClassVar[int | None] = 1
    context_view_version: ClassVar[int | None] = 1
    service_durability: RuntimeStoreDurability = RuntimeStoreDurability.DEVELOPMENT

    def __init__(
        self,
        *,
        public_authority_alias_codec: PublicAuthorityAliasCodec | None = None,
        ownership_clock: Callable[[], datetime] | None = None,
    ) -> None:
        from cayu.runtime._cost_accounting_refresh import CostAccountingAuthority
        from cayu.runtime._usage_accounting import SessionUsageCache

        self._cost_accounting_authority = CostAccountingAuthority()
        self._session_usage_cache = SessionUsageCache()
        self._accounting_generation = 0
        if public_authority_alias_codec is not None and not isinstance(
            public_authority_alias_codec,
            PublicAuthorityAliasCodec,
        ):
            raise TypeError("public_authority_alias_codec must be a PublicAuthorityAliasCodec.")
        if public_authority_alias_codec is None:
            # In-memory stores have no restart or multi-worker key-distribution
            # boundary. Give each store lifetime a cryptographically random key;
            # persistent stores require explicit deployment provisioning.
            encoded_key = (
                base64.urlsafe_b64encode(secrets.token_bytes(32)).decode("ascii").rstrip("=")
            )
            public_authority_alias_codec = PublicAuthorityAliasCodec(
                PublicAuthorityAliasKeyring(
                    active_key_id="memory",
                    keys={"memory": SecretStr(encoded_key)},
                )
            )
        self._public_authority_alias_codec = public_authority_alias_codec
        self._ownership_clock = utc_clock(ownership_clock)
        self._lock = asyncio.Lock()
        self._participant_creation_lock = asyncio.Lock()
        self._session_creation_decisions = {}
        self._execution_owners = {}
        self._producer_cleanup_receipts = {}
        self._producer_cleanup_index = {}
        self._producer_cleanup_retirements = {}
        self._sessions: dict[str, Session] = {}
        self._participant_session_bindings: dict[str, ParticipantSessionBinding] = {}
        self._participant_session_receipts: dict[str, ParticipantSessionCreationReceipt] = {}
        self._participant_session_ids_by_key: dict[str, str] = {}
        self._participant_session_request_commitments: dict[str, str | None] = {}
        self._context_views: dict[str, Any] = {}
        self._context_publication_keys: dict[str, str] = {}
        self._context_selection_receipts: dict[str, Any] = {}
        self._context_selection_request_commitments: dict[str, str] = {}
        self._context_selection_exclusions: dict[str, Any] = {}
        self._context_ownership_operation_commitments: dict[str, str] = {}
        self._context_ownership_operation_receipts: dict[str, Any] = {}
        self._context_lifecycle_events: dict[str, Any] = {}
        self._peer_content_receipts: dict[str, Any] = {}
        self._peer_content_deleted_targets: set[tuple[str, str]] = set()
        self._peer_content_operation_commitments: dict[str, str] = {}
        self._peer_content_operations: dict[str, str] = {}
        self._peer_content_requests: dict[str, PeerContentAppendRequest] = {}
        self._peer_content_attempts: dict[str, tuple[str, str]] = {}
        self._peer_content_exposures: dict[str, tuple[str, PeerContentExposureReceipt]] = {}
        self._session_closure_receipts: dict[tuple[str, str], dict[str, Any]] = {}
        self._session_closure_progress: dict[tuple[str, str], dict[str, Any]] = {}
        self._session_closure_tombstones: dict[tuple[str, str, str], dict[str, Any]] = {}
        self._public_authority_aliases: dict[tuple[str, str, str], str] = {}
        self._targeted_tool_grants: dict[str, TargetedToolGrantRecord] = {}
        self._targeted_tool_grant_ids_by_session: dict[str, list[str]] = {}
        self._targeted_tool_grant_ids_by_interaction: dict[tuple[str, str], list[str]] = {}
        self._targeted_tool_grant_ids_by_request: dict[tuple[str, str, str], str] = {}
        self._targeted_tool_grant_ids_by_tool: dict[tuple[str, str, str], str] = {}
        self._targeted_tool_grant_refs: dict[str, tuple[str, str]] = {}
        self._targeted_tool_uses: dict[str, dict[str, TargetedToolUseBinding]] = {}
        self._targeted_tool_use_by_invocation: dict[
            tuple[str, str, str], TargetedToolUseBinding
        ] = {}
        self._targeted_tool_use_by_outer_call: dict[
            tuple[str, str, str], TargetedToolUseBinding
        ] = {}
        # Stable direct-child keys maintained with session lifecycle writes. Topology
        # pages can therefore seek one parent branch without scanning the complete
        # in-memory session registry.
        self._child_session_keys_by_parent: dict[str, list[tuple[datetime, str]]] = {}
        # Canonical priority indexes keep child-context inspection bounded. Each
        # child has exactly one key derived from its session status, latest
        # lifecycle occurrence, and exact durable notification consumption.
        self._child_lifecycle_candidate_keys_by_parent: dict[
            str,
            dict[int, list[tuple[datetime, str]]],
        ] = {}
        self._child_lifecycle_candidate_key_by_child: dict[
            str,
            tuple[str, int, tuple[datetime, str]],
        ] = {}
        self._child_lifecycle_event_records: dict[
            str,
            dict[EventType, EventRecord],
        ] = {}
        self._events: dict[str, list[Event]] = {}
        self._event_records: list[EventRecord] = []
        # Secondary indexes over ``_event_records`` (same EventRecord objects, kept in
        # sequence-ascending order) so per-session summaries and type/session-scoped
        # queries — including the per-step budget read path — stop scanning the global
        # event list. Keyed by session id and by ``str(event.type)`` respectively.
        self._session_event_records: dict[str, list[EventRecord]] = {}
        self._interaction_event_records: dict[
            str,
            dict[str, list[EventRecord]],
        ] = {}
        self._latest_interaction_event_records: dict[str, dict[str, EventRecord]] = {}
        self._latest_interaction_event_records_by_sequence: dict[
            str,
            list[EventRecord],
        ] = {}
        self._pending_action_event_records: dict[
            str,
            dict[str, dict[str, EventRecord]],
        ] = {}
        # The identity scope represented by each bounded pending-action index.
        # A checkpoint can reuse a tool-call id for a later execution, so lookup
        # keys alone are insufficient to decide whether rebuilding is required.
        self._pending_action_index_scopes: dict[
            str,
            tuple[frozenset[str], str | None, str | None, str | None],
        ] = {}
        # Unlike the bounded pending-action projection above, runtime publication
        # validation must observe *every* lifecycle event for the active tool
        # round so a caller cannot omit a contradictory second terminal event.
        # This history is identifier-scoped and released with the pending marker.
        self._pending_action_event_history: dict[
            str,
            dict[str, list[EventRecord]],
        ] = {}
        # Supersession lookup remains available after the pause checkpoint and
        # its general pending-action history are released.
        self._user_input_supersession_event_records: dict[
            tuple[str, str],
            list[EventRecord],
        ] = {}
        self._pending_action_latest_barrier_records: dict[str, EventRecord] = {}
        self._event_records_by_id: dict[tuple[str, str], EventRecord] = {}
        self._type_event_records: dict[str, list[EventRecord]] = {}
        from cayu.runtime._cost_accounting_index import CostEventIndex

        self._cost_event_index = CostEventIndex()
        self._workflow_step_event_records: dict[
            tuple[str, str, str, str, str],
            list[EventRecord],
        ] = {}
        self._workflow_fenced_step_event_records: dict[
            tuple[str, str, str, str],
            list[EventRecord],
        ] = {}
        self._workflow_attempt_event_records: dict[
            tuple[str, str],
            list[EventRecord],
        ] = {}
        self._workflow_active_attempt_ids: dict[tuple[str, str], str] = {}
        self._event_ids: dict[str, set[str]] = {}
        # Store-wide ownership registry. It intentionally outlives session
        # deletion: reservation ids are ledger reconciliation keys, not
        # session-local event attributes.
        self._budget_reservation_identities: dict[str, tuple[str, str, bool]] = {}
        self._persisted_event_side_effect_deliveries: dict[
            tuple[str, str], PersistedEventSideEffectDelivery
        ] = {}
        self._next_event_sequence = 1
        self._transcripts: dict[str, list[Message]] = {}
        self._transcript_interaction_ids: dict[str, list[str | None]] = {}
        self._latest_transcript_indexes_by_role: dict[str, dict[MessageRole, int]] = {}
        self._transcript_indices_by_interaction: dict[str, dict[str | None, list[int]]] = {}
        self._transcript_search_documents: dict[str, list[tuple[MessageRole, str]]] = {}
        self._transcript_search_postings: dict[tuple[str, MessageRole], dict[str, list[int]]] = {}
        self._deferred_interaction_inputs: dict[str, DeferredInteractionInput] = {}
        self._checkpoints: dict[str, dict[str, Any]] = {}
        self._recall_receipts: dict[str, RecallReceipt] = {}
        self._recall_receipt_ids_by_session: dict[str, set[str]] = {}
        self._recall_receipt_page_keys_by_session: dict[str, list[tuple[datetime, str]]] = {}
        self._recall_receipt_page_keys_by_interaction: dict[
            tuple[str, str], list[tuple[datetime, str]]
        ] = {}
        self._recall_receipt_page_keys_by_step: dict[
            tuple[str, str], list[tuple[datetime, str]]
        ] = {}
        self._recall_receipt_page_keys_by_interaction_step: dict[
            tuple[str, str, str], list[tuple[datetime, str]]
        ] = {}
        self._context_exposures: dict[str, ContextExposure] = {}
        self._context_exposure_ids_by_session: dict[str, set[str]] = {}
        self._context_exposure_page_keys_by_session: dict[str, list[tuple[datetime, str]]] = {}
        self._context_exposure_page_keys_by_interaction: dict[
            tuple[str, str], list[tuple[datetime, str]]
        ] = {}
        self._context_exposure_page_keys_by_step: dict[
            tuple[str, str], list[tuple[datetime, str]]
        ] = {}
        self._context_exposure_page_keys_by_interaction_step: dict[
            tuple[str, str, str], list[tuple[datetime, str]]
        ] = {}
        self._recall_item_exposures: dict[str, tuple[RecallItemExposure, ...]] = {}
        self._context_exposure_by_model_attempt: dict[tuple[str, str], str] = {}
        self._context_exposure_by_provider_attempt: dict[tuple[str, str], str] = {}
        # Live queued-dispatch handoffs are indexed incrementally so restart
        # reconciliation is bounded by the requested page, not by historical
        # sessions or events.
        self._queued_dispatch_terminal_receipts: dict[
            tuple[str, str], QueuedDispatchTerminalReceipt
        ] = {}
        self._queued_dispatch_terminal_receipt_keys: list[tuple[str, str]] = []
        self._queued_dispatch_terminal_receipt_keys_by_session: dict[
            str, tuple[tuple[str, str], ...]
        ] = {}
        self._session_operation_records: dict[str, dict[str, dict[str, Any]]] = {}
        self._mcp_manifest_baselines: dict[str, McpManifestBaseline] = {}
        self._pending_action_session_ids: set[str] = set()
        # Retain every accepted message for idempotent replay, but keep queued
        # delivery state in per-session/mode deques. Hot delivery and terminal
        # fencing therefore depend on the pending set, not historical deliveries.
        self._queued_session_messages_by_idempotency: dict[
            str, dict[str, SessionQueuedMessage]
        ] = {}
        self._pending_session_messages: dict[
            tuple[str, SessionMessageDeliveryMode], deque[SessionQueuedMessage]
        ] = {}
        self._session_message_delivery_records: dict[str, _InMemoryMessageDeliveryRecord] = {}
        self._session_message_terminal_receipts: dict[tuple[str, str], dict[str, Any]] = {}
        # Event-owned lookup index, not a copy of acceptance provenance. Retain
        # at most two IDs: a second canonical acceptance makes quarantine ambiguous.
        self._session_message_acceptance_event_ids: dict[tuple[str, str], set[str]] = {}
        self._next_session_message_ordering_key = 1

    @property
    def public_authority_alias_codec(self) -> PublicAuthorityAliasCodec:
        return self._public_authority_alias_codec

    def _index_session_parent_unlocked(self, session: Session) -> None:
        parent_session_id = session.parent_session_id
        if parent_session_id is None:
            return
        keys = self._child_session_keys_by_parent.setdefault(parent_session_id, [])
        key = (session.created_at, session.id)
        index = bisect_left(keys, key)
        if index < len(keys) and keys[index] == key:
            raise RuntimeError("Duplicate in-memory session topology index entry.")
        keys.insert(index, key)

    def _remove_child_lifecycle_candidate_unlocked(self, child_session_id: str) -> None:
        indexed = self._child_lifecycle_candidate_key_by_child.pop(child_session_id, None)
        if indexed is None:
            return
        parent_session_id, priority, key = indexed
        priorities = self._child_lifecycle_candidate_keys_by_parent.get(parent_session_id)
        if priorities is None:
            raise RuntimeError("Missing in-memory child lifecycle candidate branch.")
        keys = priorities.get(priority)
        if keys is None:
            raise RuntimeError("Missing in-memory child lifecycle candidate priority.")
        index = bisect_left(keys, key)
        if index >= len(keys) or keys[index] != key:
            raise RuntimeError("Missing in-memory child lifecycle candidate key.")
        keys.pop(index)
        if not keys:
            del priorities[priority]
        if not priorities:
            del self._child_lifecycle_candidate_keys_by_parent[parent_session_id]

    def _refresh_child_lifecycle_candidate_unlocked(self, child: Session) -> None:
        self._remove_child_lifecycle_candidate_unlocked(child.id)
        parent_session_id = child.parent_session_id
        if parent_session_id is None:
            return
        occurrence_source = _child_session_lifecycle_occurrence(
            child,
            self._child_lifecycle_event_records.get(child.id, {}),
        )
        priority = 1
        sort_at = child.created_at
        if occurrence_source is not None:
            _relationship, occurrence = occurrence_source
            sort_at = occurrence.occurred_at
            if _child_session_lifecycle_state(child.status).terminal:
                consumption_key = child_session_notification_storage_key(
                    child.instance_id,
                    occurrence.source_id,
                )
                parent_operations = self._session_operation_records.get(parent_session_id, {})
                priority = 2 if consumption_key in parent_operations else 0
        key = (sort_at, child.id)
        priorities = self._child_lifecycle_candidate_keys_by_parent.setdefault(
            parent_session_id,
            {},
        )
        keys = priorities.setdefault(priority, [])
        index = bisect_left(keys, key)
        if index < len(keys) and keys[index] == key:
            raise RuntimeError("Duplicate in-memory child lifecycle candidate key.")
        keys.insert(index, key)
        self._child_lifecycle_candidate_key_by_child[child.id] = (
            parent_session_id,
            priority,
            key,
        )

    def _remove_session_parent_index_unlocked(self, session: Session) -> None:
        parent_session_id = session.parent_session_id
        if parent_session_id is None:
            return
        keys = self._child_session_keys_by_parent.get(parent_session_id)
        if keys is None:
            raise RuntimeError("Missing in-memory session topology index branch.")
        key = (session.created_at, session.id)
        index = bisect_left(keys, key)
        if index >= len(keys) or keys[index] != key:
            raise RuntimeError("Missing in-memory session topology index entry.")
        keys.pop(index)
        if not keys:
            del self._child_session_keys_by_parent[parent_session_id]
        self._remove_child_lifecycle_candidate_unlocked(session.id)

    def _prepare_checkpoint_store_unlocked(
        self,
        session_id: str,
        checkpoint: dict[str, Any],
        *,
        additional_event_records: tuple[EventRecord, ...] = (),
    ) -> _PreparedInMemoryCheckpointStore:
        from cayu.sessions.pending_actions import (
            _pending_action_checkpoint_index_state,
            checkpoint_has_pending_action_candidate,
            pending_action_event_has_unknown_round_call,
            pending_action_event_is_from_different_tool_round,
            pending_action_event_lookup_id,
            pending_action_event_retains_history,
            pending_action_lookup_key,
            project_pending_action_event_record,
            retain_pending_action_index_record,
        )

        if not checkpoint_has_pending_action_candidate(checkpoint):
            return _PreparedInMemoryCheckpointStore(
                checkpoint=checkpoint,
                has_pending_action=False,
            )

        lookup_ids, pending_round = _pending_action_checkpoint_index_state(checkpoint)
        lookup_keys = frozenset(pending_action_lookup_key(identifier) for identifier in lookup_ids)
        index_scope = (
            lookup_keys,
            None if pending_round is None else pending_round.model_step_id,
            None if pending_round is None else pending_round.model_attempt_id,
            None if pending_round is None else pending_round.tool_round_id,
        )
        rebuilt: dict[str, dict[str, EventRecord]] | None = None
        rebuilt_history: dict[str, list[EventRecord]] | None = None
        if self._pending_action_index_scopes.get(session_id) != index_scope:
            # A public checkpoint transform may legitimately reintroduce a
            # previously cleared durable action or reuse a provider call id for
            # a later execution. SQL stores retain the source events, so rebuild
            # both identity-scoped projections from complete durable evidence.
            rebuilt = {}
            rebuilt_history = {}
            round_lookup_key = (
                None
                if pending_round is None
                else pending_action_lookup_key(pending_round.tool_round_id)
            )
            records = (
                *self._session_event_records.get(session_id, ()),
                *additional_event_records,
            )
            for record in records:
                event_type = str(record.event.type)
                if (
                    event_type not in PENDING_ACTION_EVENT_TYPE_VALUES
                    or event_type in PENDING_ACTION_BARRIER_EVENT_TYPE_VALUES
                ):
                    continue
                retain_for_scope = (
                    pending_round is not None
                    and pending_action_event_has_unknown_round_call(
                        record.event,
                        pending_round,
                    )
                )
                lookup_id = pending_action_event_lookup_id(record.event)
                if lookup_id is None and not retain_for_scope:
                    continue
                projected_record = project_pending_action_event_record(record)
                if lookup_id is not None:
                    lookup_key = pending_action_lookup_key(lookup_id)
                    if lookup_key in lookup_keys:
                        rebuilt_history.setdefault(lookup_key, []).append(record)
                        if not (
                            pending_action_event_retains_history(event_type)
                            and (
                                pending_round is None
                                or pending_action_event_is_from_different_tool_round(
                                    record.event,
                                    pending_round,
                                )
                            )
                        ):
                            retain_pending_action_index_record(
                                rebuilt.setdefault(lookup_key, {}),
                                projected_record,
                            )
                if retain_for_scope:
                    assert round_lookup_key is not None
                    retain_pending_action_index_record(
                        rebuilt.setdefault(round_lookup_key, {}),
                        projected_record,
                    )
                    rebuilt_history.setdefault(round_lookup_key, []).append(record)
        return _PreparedInMemoryCheckpointStore(
            checkpoint=checkpoint,
            has_pending_action=True,
            rebuilt_records=rebuilt,
            rebuilt_event_history=rebuilt_history,
            rebuilt_index_scope=index_scope if rebuilt is not None else None,
        )

    def _apply_checkpoint_store_unlocked(
        self,
        session_id: str,
        prepared: _PreparedInMemoryCheckpointStore,
    ) -> None:
        self._refresh_queued_dispatch_terminal_receipts_unlocked(
            session_id,
            prepared.checkpoint,
        )
        self._checkpoints[session_id] = prepared.checkpoint
        if prepared.has_pending_action:
            if prepared.rebuilt_records is not None:
                self._pending_action_event_records[session_id] = prepared.rebuilt_records
                assert prepared.rebuilt_event_history is not None
                self._pending_action_event_history[session_id] = prepared.rebuilt_event_history
                assert prepared.rebuilt_index_scope is not None
                self._pending_action_index_scopes[session_id] = prepared.rebuilt_index_scope
            self._pending_action_session_ids.add(session_id)
            return

        self._pending_action_session_ids.discard(session_id)
        # Once the checkpoint no longer names a pending action, identifier-
        # scoped event history cannot contribute to a future action. Keep the
        # latest lifecycle barrier, but release the potentially growing maps.
        removed = self._pending_action_event_records.pop(session_id, None)
        self._pending_action_event_history.pop(session_id, None)
        if removed or session_id in self._pending_action_index_scopes:
            self._pending_action_index_scopes[session_id] = (
                frozenset(),
                None,
                None,
                None,
            )

    def _refresh_queued_dispatch_terminal_receipts_unlocked(
        self,
        session_id: str,
        checkpoint: dict[str, Any] | None,
    ) -> None:
        previous_keys = self._queued_dispatch_terminal_receipt_keys_by_session.get(session_id, ())
        previous_key_set = set(previous_keys)
        receipts = _queued_dispatch_terminal_receipts_for_session(
            session_id,
            checkpoint,
            durable_event_ids=self._event_ids.get(session_id, set()),
        )
        next_receipts: dict[tuple[str, str], QueuedDispatchTerminalReceipt] = {}
        for receipt in receipts:
            key = (receipt.session_id, receipt.operation_id)
            if key in next_receipts:
                raise RuntimeError("Duplicate in-memory queued dispatch receipt identity.")
            if key in self._queued_dispatch_terminal_receipts and key not in previous_key_set:
                raise RuntimeError("Duplicate in-memory queued dispatch receipt identity.")
            next_receipts[key] = receipt.model_copy(deep=True)

        previous_indices: list[tuple[int, tuple[str, str]]] = []
        for key in previous_keys:
            index = bisect_left(self._queued_dispatch_terminal_receipt_keys, key)
            if (
                index >= len(self._queued_dispatch_terminal_receipt_keys)
                or self._queued_dispatch_terminal_receipt_keys[index] != key
                or key not in self._queued_dispatch_terminal_receipts
            ):
                raise RuntimeError("In-memory queued dispatch receipt index is inconsistent.")
            previous_indices.append((index, key))

        self._queued_dispatch_terminal_receipt_keys_by_session.pop(session_id, None)
        for index, key in reversed(previous_indices):
            self._queued_dispatch_terminal_receipt_keys.pop(index)
            del self._queued_dispatch_terminal_receipts[key]

        next_keys: list[tuple[str, str]] = []
        for key in sorted(next_receipts):
            index = bisect_left(self._queued_dispatch_terminal_receipt_keys, key)
            self._queued_dispatch_terminal_receipt_keys.insert(index, key)
            self._queued_dispatch_terminal_receipts[key] = next_receipts[key]
            next_keys.append(key)
        if next_keys:
            self._queued_dispatch_terminal_receipt_keys_by_session[session_id] = tuple(next_keys)

    def _store_checkpoint_unlocked(self, session_id: str, checkpoint: dict[str, Any]) -> None:
        prepared = self._prepare_checkpoint_store_unlocked(session_id, checkpoint)
        self._apply_checkpoint_store_unlocked(session_id, prepared)

    def _extend_transcript_attribution_unlocked(
        self,
        session_id: str,
        interaction_ids: list[str | None],
    ) -> None:
        """Append attribution and maintain a lazily built absolute-index projection."""

        stored = self._transcript_interaction_ids[session_id]
        start = len(stored)
        stored.extend(interaction_ids)
        messages = self._transcripts[session_id]
        if len(messages) != len(stored):
            raise RuntimeError("In-memory transcript attribution is inconsistent.")
        latest_by_role = self._latest_transcript_indexes_by_role.setdefault(session_id, {})
        for index in range(start, len(messages)):
            latest_by_role[messages[index].role] = index
        projection = self._transcript_indices_by_interaction.get(session_id)
        if projection is None:
            return
        for offset, interaction_id in enumerate(interaction_ids):
            projection.setdefault(interaction_id, []).append(start + offset)

    def _replace_latest_transcript_role_indexes_unlocked(self, session_id: str) -> None:
        latest_by_role: dict[MessageRole, int] = {}
        for index, message in enumerate(self._transcripts.get(session_id, ())):
            latest_by_role[message.role] = index
        self._latest_transcript_indexes_by_role[session_id] = latest_by_role

    def _transcript_interaction_projection_unlocked(
        self,
        session_id: str,
    ) -> dict[str | None, list[int]]:
        projection = self._transcript_indices_by_interaction.get(session_id)
        if projection is not None:
            return projection
        projection = {}
        for index, interaction_id in enumerate(
            self._transcript_interaction_ids.get(session_id, ())
        ):
            projection.setdefault(interaction_id, []).append(index)
        self._transcript_indices_by_interaction[session_id] = projection
        return projection

    def _remove_transcript_search_session_unlocked(self, session_id: str) -> None:
        documents = self._transcript_search_documents.pop(session_id, ())
        posting_keys = {(token, role) for role, document in documents for token in document.split()}
        for posting_key in posting_keys:
            sessions = self._transcript_search_postings.get(posting_key)
            if sessions is None:
                raise RuntimeError("In-memory transcript search index is inconsistent.")
            sessions.pop(session_id, None)
            if not sessions:
                del self._transcript_search_postings[posting_key]

    def _extend_transcript_search_unlocked(
        self,
        session_id: str,
        messages: Sequence[Message],
    ) -> None:
        documents = self._transcript_search_documents.setdefault(session_id, [])
        expected_start = len(self._transcripts.get(session_id, ())) - len(messages)
        if len(documents) != expected_start:
            raise RuntimeError("In-memory transcript search index is inconsistent.")
        for offset, message in enumerate(messages):
            document = (
                transcript_search_document(message)
                if message.role in {MessageRole.USER, MessageRole.ASSISTANT}
                else ""
            )
            transcript_index = expected_start + offset
            documents.append((message.role, document))
            for token in set(document.split()):
                self._transcript_search_postings.setdefault((token, message.role), {}).setdefault(
                    session_id, []
                ).append(transcript_index)

    def _replace_transcript_search_session_unlocked(
        self,
        session_id: str,
        messages: Sequence[Message],
    ) -> None:
        self._remove_transcript_search_session_unlocked(session_id)
        self._transcript_search_documents[session_id] = []
        self._extend_transcript_search_unlocked(session_id, messages)

    async def _terminalize_zero_work_interruption(
        self,
        request: ZeroWorkInterruptionRequest,
    ) -> ZeroWorkInterruptionPublication | None:
        """Atomically prove and terminalize zero work; unsupported stores decline."""
        from cayu.storage._zero_work_interruption import memory_terminalize

        return await memory_terminalize(self, request)

    async def register_public_authority_alias(
        self,
        public_alias: str,
        *,
        field_name: str,
        private_value: str,
        scope_session_id: str | None = None,
    ) -> None:
        key = _public_authority_alias_store_key(
            public_alias,
            field_name=field_name,
            private_value=private_value,
            scope_session_id=scope_session_id,
        )
        codec = self.public_authority_alias_codec
        if codec is None or not codec.matches(
            public_alias,
            private_value,
            field_name=field_name,
            session_id=scope_session_id,
        ):
            raise ValueError("Public authority alias lacks valid store-configured provenance.")
        async with self._lock:
            self._register_public_authority_alias_unlocked(key, private_value)

    def _register_public_authority_alias_unlocked(
        self,
        key: tuple[str, str, str],
        private_value: str,
    ) -> None:
        existing = self._public_authority_aliases.get(key)
        if existing is not None and existing != private_value:
            raise ValueError("Public authority alias conflicts with existing private authority.")
        self._public_authority_aliases[key] = private_value

    def _register_private_authority_alias_unlocked(
        self,
        private_value: str,
        *,
        field_name: str,
        scope_session_id: str | None = None,
    ) -> None:
        codec = self.public_authority_alias_codec
        if codec is None:
            return
        for public_alias in codec.aliases(
            private_value,
            field_name=field_name,
            session_id=scope_session_id,
        ):
            key = _public_authority_alias_store_key(
                public_alias,
                field_name=field_name,
                private_value=private_value,
                scope_session_id=scope_session_id,
            )
            self._register_public_authority_alias_unlocked(key, private_value)

    async def resolve_public_authority_alias(
        self,
        public_alias: str,
        *,
        field_name: str,
        scope_session_id: str | None = None,
    ) -> str | None:
        key = _public_authority_alias_store_key(
            public_alias,
            field_name=field_name,
            private_value=None,
            scope_session_id=scope_session_id,
        )
        async with self._lock:
            return _authenticated_public_authority_alias_private_value(
                self.public_authority_alias_codec,
                public_alias,
                self._public_authority_aliases.get(key),
                field_name=field_name,
                scope_session_id=scope_session_id,
            )

    async def public_authority_private_value_exists(
        self,
        private_value: str,
        *,
        field_name: str,
        scope_session_id: str | None = None,
    ) -> bool:
        codec = self.public_authority_alias_codec
        if codec is None:
            raise RuntimeError("Public authority alias codec is unavailable.")
        probe = codec.encode(
            private_value,
            field_name=field_name,
            session_id=scope_session_id,
        )
        field_name, scope_key, _probe = _public_authority_alias_store_key(
            probe,
            field_name=field_name,
            private_value=private_value,
            scope_session_id=scope_session_id,
        )
        async with self._lock:
            return any(
                key_field == field_name and key_scope == scope_key and stored == private_value
                for (key_field, key_scope, _alias), stored in self._public_authority_aliases.items()
            )

    async def issue_targeted_tool_grants(
        self,
        session_id: str,
        *,
        expected_run_epoch: int,
        records: tuple[TargetedToolGrantRecord, ...],
        events: tuple[Event, ...],
    ) -> TargetedToolGrantIssueResult:
        session_id = require_clean_nonblank(session_id, "session_id")
        if type(expected_run_epoch) is not int or expected_run_epoch < 0:
            raise ValueError("expected_run_epoch must be a non-negative integer.")
        if type(records) is not tuple or type(events) is not tuple:
            raise TypeError("records and events must be tuples.")
        if len(records) > TARGETED_TOOL_GRANT_MAX_REQUESTS:
            raise ValueError("Targeted grant issuance exceeds the bounded request count.")
        copied_records = tuple(copy_targeted_tool_grant_record(record) for record in records)
        copied_events = tuple(copy_event(event) for event in events)
        if len(copied_records) != len(copied_events):
            raise ValueError("Each targeted grant record requires one issuance event.")
        if len({record.request_id for record in copied_records}) != len(copied_records):
            raise ValueError("Targeted grant records must have unique request identities.")
        if len({record.tool_id for record in copied_records}) != len(copied_records):
            raise ValueError("Targeted grant records must have unique tool identities.")
        interaction_ids = {record.interaction_id for record in copied_records}
        if len(interaction_ids) > 1:
            raise ValueError("Targeted grant records must share one interaction scope.")
        codec = self.public_authority_alias_codec
        if copied_records and codec is None:
            raise RuntimeError("Targeted grants require a public authority alias codec.")
        for record, event in zip(copied_records, copied_events, strict=True):
            if record.session_id != session_id:
                raise ValueError("Targeted grant scope is inconsistent.")
            assert codec is not None
            validate_targeted_tool_grant_reference(record, codec)
            validate_targeted_tool_grant_issuance_evidence(record, event)

        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            if session.run_epoch != expected_run_epoch:
                raise SessionRunFenced(
                    f"Session source run epoch is stale: expected {expected_run_epoch}, "
                    f"current {session.run_epoch}."
                )
            if session.status is not SessionStatus.RUNNING:
                raise SessionStatusConflict("Targeted grants require a running session.")
            if interaction_ids:
                latest_interactions = self._latest_interaction_event_records_by_sequence.get(
                    session_id,
                    (),
                )
                latest = latest_interactions[-1] if latest_interactions else None
                if (
                    latest is None
                    or latest.event.interaction_id != next(iter(interaction_ids))
                    or latest.event.type
                    in {
                        EventType.INTERACTION_COMPLETED,
                        EventType.INTERACTION_FAILED,
                        EventType.INTERACTION_INTERRUPTED,
                    }
                ):
                    raise ValueError("Targeted grants require the current open interaction.")
                interaction_started = next(
                    (
                        event_record.event
                        for event_record in latest_interactions
                        if event_record.event.type is EventType.INTERACTION_STARTED
                        and event_record.event.interaction_id == next(iter(interaction_ids))
                    ),
                    None,
                )
                if interaction_started is None:
                    raise RuntimeError("Targeted grant issuance lost interaction admission.")
                validate_targeted_tool_grant_batch_evidence(
                    copied_records,
                    interaction_started,
                )
            for record in copied_records:
                if (
                    record.agent_name != session.agent_name
                    or record.environment_name != session.environment_name
                    or record.principal != session.invocation.origin.subject
                    or record.tenant != session.invocation.origin.tenant
                ):
                    raise ValueError("Targeted grant scope conflicts with session authority.")

            resolved_records: list[TargetedToolGrantRecord] = []
            outcomes: list[TargetedToolGrantIssueOutcome] = []
            resolved_events: list[Event] = []
            new_records: list[tuple[TargetedToolGrantRecord, Event]] = []
            events_to_append: list[Event] = []
            for record, event in zip(copied_records, copied_events, strict=True):
                request_key = (session_id, record.interaction_id, record.request_id)
                tool_key = (session_id, record.interaction_id, record.tool_id)
                existing_id = self._targeted_tool_grant_ids_by_request.get(request_key)
                existing = (
                    None if existing_id is None else self._targeted_tool_grants.get(existing_id)
                )
                if existing is not None:
                    if existing.grant_id != record.grant_id:
                        raise ValueError(
                            "Targeted grant request identity conflicts with durable authority."
                        )
                    if len(self._targeted_tool_uses.get(existing.grant_id, ())) != (
                        existing.used_calls
                    ):
                        raise ValueError("Targeted grant call counter conflicts with durable uses.")
                    existing_event_record = self._event_records_by_id.get((session_id, event.id))
                    if existing_event_record is None:
                        raise RuntimeError("Targeted grant lost its durable issuance evidence.")
                    validate_targeted_tool_grant_issuance_evidence(
                        existing,
                        existing_event_record.event,
                    )
                    reused_event = targeted_tool_grant_event(
                        existing,
                        event_type=EventType.TARGETED_TOOL_GRANT_REUSED,
                        timestamp=event.timestamp,
                        outcome=TargetedToolGrantIssueOutcome.REUSED.value,
                        event_id_suffix="reused",
                    )
                    existing_reuse = self._event_records_by_id.get((session_id, reused_event.id))
                    codec = self.public_authority_alias_codec
                    if codec is None:  # pragma: no cover - capability invariant
                        raise RuntimeError("Targeted grants require a public alias codec.")
                    resolved_records.append(
                        targeted_tool_grant_with_active_reference(existing, codec)
                    )
                    outcomes.append(TargetedToolGrantIssueOutcome.REUSED)
                    if existing_reuse is None:
                        events_to_append.append(reused_event)
                        resolved_events.append(reused_event)
                    else:
                        resolved_events.append(copy_event(existing_reuse.event))
                    continue
                conflicting_tool_grant_id = self._targeted_tool_grant_ids_by_tool.get(tool_key)
                if conflicting_tool_grant_id is not None:
                    raise ValueError(
                        "Targeted grant tool identity conflicts with durable authority."
                    )
                if record.grant_id in self._targeted_tool_grants:
                    raise ValueError("Targeted grant identity collides with existing authority.")
                if (session_id, event.id) in self._event_records_by_id:
                    raise ValueError("Targeted grant event identity is already in use.")
                new_records.append((record, event))
                resolved_records.append(record)
                outcomes.append(TargetedToolGrantIssueOutcome.ISSUED)
                resolved_events.append(event)
                events_to_append.append(event)

            if new_records:
                interaction_id = new_records[0][0].interaction_id
                existing_count = len(
                    self._targeted_tool_grant_ids_by_interaction.get(
                        (session_id, interaction_id),
                        (),
                    )
                )
                if existing_count + len(new_records) > TARGETED_TOOL_GRANT_MAX_REQUESTS:
                    raise ValueError("Targeted grant interaction exceeds its bounded count.")
            if events_to_append:
                prepared_append = self._prepare_event_append_unlocked(
                    session,
                    events_to_append,
                )
            alias_plans: dict[
                str,
                tuple[tuple[str, str, str], str, str],
            ] = {}
            for record, _event in new_records:
                assert codec is not None
                for public_alias in codec.aliases(
                    record.grant_id,
                    field_name=TARGETED_TOOL_REFERENCE_FIELD_NAME,
                    session_id=session_id,
                ):
                    alias_key = _public_authority_alias_store_key(
                        public_alias,
                        field_name=TARGETED_TOOL_REFERENCE_FIELD_NAME,
                        private_value=record.grant_id,
                        scope_session_id=session_id,
                    )
                    existing_private = self._public_authority_aliases.get(alias_key)
                    if existing_private is not None and existing_private != record.grant_id:
                        raise ValueError("Targeted tool reference collides with authority.")
                    existing_plan = alias_plans.get(public_alias)
                    if existing_plan is not None and existing_plan[1] != record.grant_id:
                        raise ValueError("Targeted tool reference batch contains a collision.")
                    existing_ref = self._targeted_tool_grant_refs.get(public_alias)
                    if existing_ref is not None and existing_ref != (
                        session_id,
                        record.grant_id,
                    ):
                        raise ValueError("Targeted tool reference collides with authority.")
                    alias_plans[public_alias] = (
                        alias_key,
                        record.grant_id,
                        session_id,
                    )
            if new_records:
                for record, _event in new_records:
                    for public_alias in codec.aliases(
                        record.grant_id,
                        field_name=TARGETED_TOOL_REFERENCE_FIELD_NAME,
                        session_id=session_id,
                    ):
                        alias_key, private_value, scope_key = alias_plans[public_alias]
                        self._public_authority_aliases[alias_key] = private_value
                        self._targeted_tool_grant_refs[public_alias] = (
                            scope_key,
                            private_value,
                        )
                    self._targeted_tool_grants[record.grant_id] = record
                    self._targeted_tool_grant_ids_by_session.setdefault(session_id, []).append(
                        record.grant_id
                    )
                    self._targeted_tool_grant_ids_by_interaction.setdefault(
                        (session_id, record.interaction_id),
                        [],
                    ).append(record.grant_id)
                    self._targeted_tool_grant_ids_by_request[
                        (session_id, record.interaction_id, record.request_id)
                    ] = record.grant_id
                    self._targeted_tool_grant_ids_by_tool[
                        (session_id, record.interaction_id, record.tool_id)
                    ] = record.grant_id
            if events_to_append:
                self._sessions[session_id] = self._apply_event_append_unlocked(
                    session,
                    prepared_append,
                )
            return TargetedToolGrantIssueResult(
                records=tuple(resolved_records),
                outcomes=tuple(outcomes),
                events=tuple(resolved_events),
            )

    async def list_targeted_tool_grants(
        self,
        session_id: str,
        *,
        interaction_id: str | None = None,
        limit: int = TARGETED_TOOL_GRANT_INSPECTION_MAX_RECORDS,
    ) -> tuple[TargetedToolGrantRecord, ...]:
        session_id = require_clean_nonblank(session_id, "session_id")
        if interaction_id is not None:
            interaction_id = require_clean_nonblank(interaction_id, "interaction_id")
        if type(limit) is not int or not 1 <= limit <= TARGETED_TOOL_GRANT_INSPECTION_MAX_RECORDS:
            raise ValueError(
                f"limit must be between 1 and {TARGETED_TOOL_GRANT_INSPECTION_MAX_RECORDS}."
            )
        async with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session not found: {session_id}")
            codec = self.public_authority_alias_codec
            if codec is None:  # pragma: no cover - in-memory capability invariant
                raise RuntimeError("Targeted grants require a public authority alias codec.")
            grant_ids = (
                self._targeted_tool_grant_ids_by_session.get(session_id, ())
                if interaction_id is None
                else self._targeted_tool_grant_ids_by_interaction.get(
                    (session_id, interaction_id),
                    (),
                )
            )
            selected = (self._targeted_tool_grants[grant_id] for grant_id in grant_ids)
            bounded = heapq.nsmallest(
                limit + 1,
                selected,
                key=lambda candidate: (candidate.issued_at, candidate.grant_id),
            )
            if len(bounded) > limit:
                raise ValueError("Targeted grant inspection exceeds its bounded result limit.")
            if any(
                len(self._targeted_tool_uses.get(record.grant_id, ())) != record.used_calls
                for record in bounded
            ):
                raise ValueError("Targeted grant call counter conflicts with durable uses.")
            return tuple(
                targeted_tool_grant_with_active_reference(record, codec) for record in bounded
            )

    async def load_targeted_tool_grant_state(
        self,
        session_id: str,
    ) -> TargetedToolGrantStateSnapshot:
        session_id = require_clean_nonblank(session_id, "session_id")
        async with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session not found: {session_id}")
            codec = self.public_authority_alias_codec
            if codec is None:  # pragma: no cover - in-memory capability invariant
                raise RuntimeError("Targeted grants require a public authority alias codec.")
            records = tuple(
                targeted_tool_grant_with_active_reference(record, codec)
                for record in sorted(
                    (
                        self._targeted_tool_grants[grant_id]
                        for grant_id in self._targeted_tool_grant_ids_by_session.get(
                            session_id,
                            (),
                        )
                    ),
                    key=lambda record: (record.issued_at, record.grant_id),
                )
            )
            grant_ids = {record.grant_id for record in records}
            uses = tuple(
                TargetedToolUseBinding.model_validate(binding.model_dump(mode="python"))
                for binding in sorted(
                    (
                        binding
                        for grant_id in grant_ids
                        for binding in self._targeted_tool_uses.get(grant_id, {}).values()
                    ),
                    key=lambda binding: (binding.bound_at, binding.use_id),
                )
            )
            return TargetedToolGrantStateSnapshot(records=records, uses=uses)

    def _resolve_targeted_tool_ref_unlocked(
        self,
        tool_ref: str,
    ) -> tuple[str, str] | None:
        try:
            parsed = parse_public_authority_alias(tool_ref)
        except (TypeError, ValueError):
            return None
        if parsed is None or parsed.field_name != TARGETED_TOOL_REFERENCE_FIELD_NAME:
            return None
        resolved = self._targeted_tool_grant_refs.get(tool_ref)
        if resolved is None:
            return None
        scope_session_id, grant_id = resolved
        codec = self.public_authority_alias_codec
        if codec is None or not codec.matches(
            tool_ref,
            grant_id,
            field_name=TARGETED_TOOL_REFERENCE_FIELD_NAME,
            session_id=scope_session_id,
        ):
            return None
        return resolved

    async def bind_targeted_tool_grant_use(
        self,
        request: TargetedToolUseRequest,
        *,
        observed_at: datetime,
    ) -> TargetedToolUseResult:
        if type(request) is not TargetedToolUseRequest:
            raise TypeError("request must be a TargetedToolUseRequest.")
        request = TargetedToolUseRequest.model_validate(request.model_dump(mode="python"))
        if observed_at.tzinfo is None or observed_at.utcoffset() is None:
            raise ValueError("observed_at must be timezone-aware.")
        observed_at = observed_at.astimezone(UTC)
        async with self._lock:
            session = self._sessions.get(request.session_id)
            if session is None:
                raise KeyError(f"Session not found: {request.session_id}")
            if session.run_epoch != request.expected_run_epoch:
                raise SessionRunFenced(
                    f"Session source run epoch is stale: expected {request.expected_run_epoch}, "
                    f"current {session.run_epoch}."
                )
            if session.status is not SessionStatus.RUNNING:
                raise SessionStatusConflict("Targeted tool use requires a running session.")

            resolved = self._resolve_targeted_tool_ref_unlocked(request.tool_ref)
            if resolved is None:
                try:
                    parsed = parse_public_authority_alias(request.tool_ref)
                    well_formed = (
                        parsed is not None
                        and parsed.field_name == TARGETED_TOOL_REFERENCE_FIELD_NAME
                    )
                except (TypeError, ValueError):
                    well_formed = False
                reason = (
                    TargetedToolUseRejectionReason.UNKNOWN
                    if well_formed
                    else TargetedToolUseRejectionReason.MALFORMED
                )
                event = targeted_tool_unresolved_rejection_event(
                    request,
                    reason=reason,
                    timestamp=observed_at,
                    agent_name=session.agent_name,
                    environment_name=session.environment_name,
                )
                existing = self._event_records_by_id.get((request.session_id, event.id))
                if existing is None:
                    prepared = self._prepare_event_append_unlocked(session, [event])
                    session = self._apply_event_append_unlocked(session, prepared)
                    self._sessions[request.session_id] = session
                else:
                    event = copy_event(existing.event)
                validate_targeted_tool_unresolved_rejection_evidence(
                    request,
                    reason=reason,
                    event=event,
                    agent_name=session.agent_name,
                    environment_name=session.environment_name,
                )
                return TargetedToolUseResult(
                    disposition=TargetedToolUseDisposition.REJECTED,
                    reason=reason,
                    event=event,
                )
            scope_session_id, grant_id = resolved
            if scope_session_id != request.session_id:
                event = targeted_tool_unresolved_rejection_event(
                    request,
                    reason=TargetedToolUseRejectionReason.CROSS_SESSION,
                    timestamp=observed_at,
                    agent_name=session.agent_name,
                    environment_name=session.environment_name,
                )
                existing = self._event_records_by_id.get((request.session_id, event.id))
                if existing is None:
                    prepared = self._prepare_event_append_unlocked(session, [event])
                    session = self._apply_event_append_unlocked(session, prepared)
                    self._sessions[request.session_id] = session
                else:
                    event = copy_event(existing.event)
                validate_targeted_tool_unresolved_rejection_evidence(
                    request,
                    reason=TargetedToolUseRejectionReason.CROSS_SESSION,
                    event=event,
                    agent_name=session.agent_name,
                    environment_name=session.environment_name,
                )
                return TargetedToolUseResult(
                    disposition=TargetedToolUseDisposition.REJECTED,
                    reason=TargetedToolUseRejectionReason.CROSS_SESSION,
                    event=event,
                )
            record = self._targeted_tool_grants.get(grant_id)
            if record is None:
                raise RuntimeError("Targeted tool reference lost its durable grant record.")
            if len(self._targeted_tool_uses.get(grant_id, ())) != record.used_calls:
                raise ValueError("Targeted grant call counter conflicts with durable uses.")
            current_session: Session = session

            def append_once(event: Event) -> Event:
                nonlocal current_session
                existing = self._event_records_by_id.get((request.session_id, event.id))
                if existing is not None:
                    return copy_event(existing.event)
                prepared = self._prepare_event_append_unlocked(current_session, [event])
                current_session = self._apply_event_append_unlocked(current_session, prepared)
                self._sessions[request.session_id] = current_session
                return event

            def rejected(reason: TargetedToolUseRejectionReason) -> TargetedToolUseResult:
                if reason is TargetedToolUseRejectionReason.EXPIRED:
                    expiry_event = append_once(
                        targeted_tool_grant_event(
                            record,
                            event_type=EventType.TARGETED_TOOL_GRANT_EXPIRED,
                            timestamp=observed_at,
                            outcome="expired",
                            event_id_suffix="expired",
                            rejection_reason=reason,
                        )
                    )
                    validate_targeted_tool_grant_lifecycle_event(
                        record,
                        expiry_event,
                        event_type=EventType.TARGETED_TOOL_GRANT_EXPIRED,
                        outcome="expired",
                        event_id_suffix="expired",
                        rejection_reason=reason,
                        require_current_call_count=False,
                    )
                event = append_once(
                    targeted_tool_use_rejection_event(
                        record,
                        request,
                        reason=reason,
                        timestamp=observed_at,
                    )
                )
                validate_targeted_tool_use_rejection_evidence(
                    record,
                    request,
                    reason=reason,
                    event=event,
                )
                return TargetedToolUseResult(
                    disposition=TargetedToolUseDisposition.REJECTED,
                    reason=reason,
                    grant=record,
                    event=event,
                )

            from cayu.sessions.interactions import INTERACTION_TERMINAL_EVENT_TYPES

            if any(
                event_record.event.type in INTERACTION_TERMINAL_EVENT_TYPES
                for event_record in self._interaction_event_records.get(
                    request.session_id,
                    {},
                ).get(record.interaction_id, ())
            ):
                return rejected(TargetedToolUseRejectionReason.EXPIRED)
            uses = self._targeted_tool_uses.setdefault(grant_id, {})
            related_by_id = {
                binding.use_id: binding
                for binding in (
                    self._targeted_tool_use_by_invocation.get(
                        (request.session_id, request.interaction_id, request.invocation_id)
                    ),
                    self._targeted_tool_use_by_outer_call.get(
                        (request.session_id, request.interaction_id, request.outer_tool_call_id)
                    ),
                )
                if binding is not None
            }
            related = tuple(related_by_id.values())
            if related:
                scope_rejection = targeted_tool_use_scope_rejection_reason(record, request)
                if scope_rejection is not None:
                    return rejected(scope_rejection)
                candidate = targeted_tool_use_binding(
                    grant_id,
                    request,
                    bound_at=related[0].bound_at,
                )
                if len(related) == 1 and related[0].model_dump(mode="json") == candidate.model_dump(
                    mode="json"
                ):
                    event_record = self._event_records_by_id.get(
                        (request.session_id, f"{grant_id}:use:{candidate.use_id}")
                    )
                    if event_record is None:
                        raise RuntimeError("Targeted tool use lost its durable event evidence.")
                    validate_targeted_tool_grant_lifecycle_event(
                        record,
                        event_record.event,
                        event_type=EventType.TARGETED_TOOL_REFERENCE_CONSUMED,
                        outcome=TargetedToolUseDisposition.BOUND.value,
                        event_id_suffix=f"use:{candidate.use_id}",
                        binding=related[0],
                        require_current_call_count=False,
                    )
                    rejoined_event = append_once(
                        targeted_tool_grant_event(
                            record,
                            event_type=EventType.TARGETED_TOOL_REFERENCE_REJOINED,
                            timestamp=observed_at,
                            outcome=TargetedToolUseDisposition.REJOINED.value,
                            event_id_suffix=f"rejoined:{candidate.use_id}",
                            binding=related[0],
                        )
                    )
                    return TargetedToolUseResult(
                        disposition=TargetedToolUseDisposition.REJOINED,
                        grant=record,
                        binding=related[0],
                        event=rejoined_event,
                    )
                return rejected(TargetedToolUseRejectionReason.ALTERED_REPLAY)

            reason = targeted_tool_use_rejection_reason(
                record,
                request,
                observed_at=observed_at,
            )
            if reason is not None:
                return rejected(reason)
            binding = targeted_tool_use_binding(grant_id, request, bound_at=observed_at)
            updated = TargetedToolGrantRecord.model_validate(
                record.model_copy(update={"used_calls": record.used_calls + 1}).model_dump(
                    mode="python"
                )
            )
            event = targeted_tool_grant_event(
                updated,
                event_type=EventType.TARGETED_TOOL_REFERENCE_CONSUMED,
                timestamp=observed_at,
                outcome=TargetedToolUseDisposition.BOUND.value,
                event_id_suffix=f"use:{binding.use_id}",
                binding=binding,
            )
            prepared_append = self._prepare_event_append_unlocked(session, [event])
            uses[binding.use_id] = binding
            self._targeted_tool_use_by_invocation[
                (binding.session_id, binding.interaction_id, binding.invocation_id)
            ] = binding
            self._targeted_tool_use_by_outer_call[
                (binding.session_id, binding.interaction_id, binding.outer_tool_call_id)
            ] = binding
            self._targeted_tool_grants[grant_id] = updated
            self._sessions[request.session_id] = self._apply_event_append_unlocked(
                session,
                prepared_append,
            )
            return TargetedToolUseResult(
                disposition=TargetedToolUseDisposition.BOUND,
                grant=updated,
                binding=binding,
                event=event,
            )

    async def revoke_targeted_tool_grant(
        self,
        tool_ref: str,
        *,
        session_id: str,
        expected_run_epoch: int,
        reason: str,
        revoked_at: datetime,
    ) -> TargetedToolGrantRecord | None:
        session_id = require_clean_nonblank(session_id, "session_id")
        if type(expected_run_epoch) is not int or expected_run_epoch < 0:
            raise ValueError("expected_run_epoch must be a non-negative integer.")
        reason = validate_targeted_tool_grant_revocation_reason(reason)
        if revoked_at.tzinfo is None or revoked_at.utcoffset() is None:
            raise ValueError("revoked_at must be timezone-aware.")
        revoked_at = revoked_at.astimezone(UTC)
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            if session.run_epoch != expected_run_epoch:
                raise SessionRunFenced(
                    f"Session source run epoch is stale: expected {expected_run_epoch}, "
                    f"current {session.run_epoch}."
                )
            resolved = self._resolve_targeted_tool_ref_unlocked(tool_ref)
            if resolved is None or resolved[0] != session_id:
                return None
            record = self._targeted_tool_grants.get(resolved[1])
            if record is None:
                raise RuntimeError("Targeted tool reference lost its durable grant record.")
            if len(self._targeted_tool_uses.get(record.grant_id, ())) != record.used_calls:
                raise ValueError("Targeted grant call counter conflicts with durable uses.")
            if record.revoked_at is not None:
                if record.revocation_reason != reason:
                    raise ValueError("Targeted grant was revoked with a different reason.")
                event_record = self._event_records_by_id.get(
                    (session_id, f"{record.grant_id}:revoked")
                )
                if event_record is None:
                    raise RuntimeError("Targeted grant revocation lost its durable event evidence.")
                validate_targeted_tool_grant_revocation_evidence(
                    record,
                    event_record.event,
                )
                return copy_targeted_tool_grant_record(record)
            for owner in self._session_closure_progress.values():
                _check_closure_lineage_owner(owner, (session_id,))
            uses = self._targeted_tool_uses.get(record.grant_id, {})
            if any(binding.bound_at > revoked_at for binding in uses.values()):
                raise ValueError("revoked_at cannot precede a bound targeted tool use.")
            updated = TargetedToolGrantRecord.model_validate(
                record.model_copy(
                    update={"revoked_at": revoked_at, "revocation_reason": reason}
                ).model_dump(mode="python")
            )
            event = targeted_tool_grant_event(
                updated,
                event_type=EventType.TARGETED_TOOL_GRANT_REVOKED,
                timestamp=revoked_at,
                outcome="revoked",
                event_id_suffix="revoked",
            )
            prepared_append = self._prepare_event_append_unlocked(session, [event])
            self._targeted_tool_grants[record.grant_id] = updated
            self._sessions[session_id] = self._apply_event_append_unlocked(
                session,
                prepared_append,
            )
            return copy_targeted_tool_grant_record(updated)

    async def reconstruct_targeted_tool_grants(
        self,
        session_id: str,
        *,
        expected_run_epoch: int,
        interaction_id: str,
        generation_id: str,
        agent_name: str,
        task_id: str | None,
        environment_name: str | None,
        principal: str | None,
        tenant: str | None,
        catalogue_revision: str,
        descriptors_by_id: Mapping[str, tuple[str, str, str]],
        capability_ceiling_names: frozenset[str],
        observed_at: datetime,
    ) -> TargetedToolGrantReconstructionResult:
        session_id = require_clean_nonblank(session_id, "session_id")
        interaction_id = require_clean_nonblank(interaction_id, "interaction_id")
        if type(expected_run_epoch) is not int or expected_run_epoch < 0:
            raise ValueError("expected_run_epoch must be a non-negative integer.")
        if observed_at.tzinfo is None or observed_at.utcoffset() is None:
            raise ValueError("observed_at must be timezone-aware.")
        observed_at = observed_at.astimezone(UTC)
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            if session.run_epoch != expected_run_epoch:
                raise SessionRunFenced(
                    f"Session source run epoch is stale: expected {expected_run_epoch}, "
                    f"current {session.run_epoch}."
                )
            if session.status is not SessionStatus.RUNNING:
                raise SessionStatusConflict("Grant reconstruction requires a running session.")
            records = sorted(
                (
                    self._targeted_tool_grants[grant_id]
                    for grant_id in self._targeted_tool_grant_ids_by_interaction.get(
                        (session_id, interaction_id),
                        (),
                    )
                ),
                key=lambda candidate: (candidate.issued_at, candidate.grant_id),
            )
            if len(records) > TARGETED_TOOL_GRANT_MAX_REQUESTS:
                raise ValueError("Targeted grant interaction exceeds its bounded count.")
            if any(
                len(self._targeted_tool_uses.get(record.grant_id, ())) != record.used_calls
                for record in records
            ):
                raise ValueError("Targeted grant call counter conflicts with durable uses.")

            current_session: Session = session

            def append_once(event: Event) -> Event:
                nonlocal current_session
                existing = self._event_records_by_id.get((session_id, event.id))
                if existing is not None:
                    return copy_event(existing.event)
                prepared = self._prepare_event_append_unlocked(current_session, [event])
                current_session = self._apply_event_append_unlocked(current_session, prepared)
                self._sessions[session_id] = current_session
                return event

            from cayu.sessions.interactions import INTERACTION_TERMINAL_EVENT_TYPES

            interaction_records = self._interaction_event_records.get(session_id, {}).get(
                interaction_id,
                (),
            )
            interaction_started = next(
                (
                    event_record.event
                    for event_record in interaction_records
                    if event_record.event.type is EventType.INTERACTION_STARTED
                ),
                None,
            )
            if interaction_started is None:
                raise RuntimeError("Targeted grant reconstruction lost interaction admission.")
            validate_targeted_tool_grant_batch_evidence(tuple(records), interaction_started)
            interaction_ended = any(
                event_record.event.type in INTERACTION_TERMINAL_EVENT_TYPES
                for event_record in interaction_records
            )
            valid: list[TargetedToolGrantRecord] = []
            rejected: list[tuple[str, TargetedToolUseRejectionReason]] = []
            events: list[Event] = []
            for record in records:
                reason = targeted_tool_grant_reconstruction_rejection_reason(
                    record,
                    generation_id=generation_id,
                    agent_name=agent_name,
                    task_id=task_id,
                    environment_name=environment_name,
                    principal=principal,
                    tenant=tenant,
                    catalogue_revision=catalogue_revision,
                    descriptors_by_id=descriptors_by_id,
                    capability_ceiling_names=capability_ceiling_names,
                    observed_at=observed_at,
                    interaction_ended=interaction_ended,
                )
                if reason is None:
                    valid.append(record)
                    event = targeted_tool_grant_event(
                        record,
                        event_type=EventType.TARGETED_TOOL_GRANT_RECONSTRUCTED,
                        timestamp=observed_at,
                        outcome="reconstructed",
                        event_id_suffix="reconstructed",
                    )
                else:
                    rejected.append((record.grant_id, reason))
                    if reason is TargetedToolUseRejectionReason.EXPIRED:
                        persisted_expiry = append_once(
                            targeted_tool_grant_event(
                                record,
                                event_type=EventType.TARGETED_TOOL_GRANT_EXPIRED,
                                timestamp=observed_at,
                                outcome="expired",
                                event_id_suffix="expired",
                                rejection_reason=reason,
                            )
                        )
                        validate_targeted_tool_grant_lifecycle_event(
                            record,
                            persisted_expiry,
                            event_type=EventType.TARGETED_TOOL_GRANT_EXPIRED,
                            outcome="expired",
                            event_id_suffix="expired",
                            rejection_reason=reason,
                            require_current_call_count=False,
                        )
                    event = targeted_tool_grant_event(
                        record,
                        event_type=EventType.TARGETED_TOOL_GRANT_RECONSTRUCTED,
                        timestamp=observed_at,
                        outcome="rejected",
                        event_id_suffix=f"reconstruction-rejected:{reason.value}",
                        rejection_reason=reason,
                    )
                persisted = append_once(event)
                validate_targeted_tool_grant_lifecycle_event(
                    record,
                    persisted,
                    event_type=EventType.TARGETED_TOOL_GRANT_RECONSTRUCTED,
                    outcome="reconstructed" if reason is None else "rejected",
                    event_id_suffix=(
                        "reconstructed"
                        if reason is None
                        else f"reconstruction-rejected:{reason.value}"
                    ),
                    rejection_reason=reason,
                    require_current_call_count=False,
                )
                events.append(persisted)
            return TargetedToolGrantReconstructionResult(
                valid=tuple(valid),
                rejected=tuple(rejected),
                events=tuple(events),
            )

    async def create(
        self,
        request: RunRequest,
        *,
        identity: SessionIdentity,
        interaction_started_event: Event | None = None,
        interaction_source_messages: list[Message] | None = None,
        checkpoint_transform: CheckpointTransform | None = None,
        result_checkpoint_transform: CheckpointTransform | None = None,
        operation_initializer: SessionOperationInitializer | None = None,
        participant_binding_factory: Callable[
            [Session], tuple[ParticipantSessionBinding, ParticipantSessionCreationReceipt]
        ]
        | None = None,
        participant_request_commitment: str | None = None,
        participant_provenance: object | None = None,
        recipient_selection: _ContextViewSelectionReceipt | None = None,
        creation_target=None,
        recipient_receipt_validator: Callable[[Session, ParticipantSessionCreationReceipt], object]
        | None = None,
    ) -> Session:
        if type(request) is not RunRequest:
            raise TypeError("Session creation requires a RunRequest.")
        if participant_binding_factory is not None and not callable(participant_binding_factory):
            raise TypeError("participant_binding_factory must be callable.")
        if participant_binding_factory is not None:
            from cayu.sessions.context_views import (
                ParticipantSessionBinding,
                ParticipantSessionCreationReceipt,
            )
        request = copy_run_request(request)
        identity = copy_session_identity(identity)
        if result_checkpoint_transform is not None and not callable(result_checkpoint_transform):
            raise TypeError("result_checkpoint_transform must be callable.")
        async with self._lock:
            session_id = request.session_id or str(uuid4())
            self._require_external_creation_unlocked(request)
            if (
                participant_provenance is _RECIPIENT_PROVENANCE_CAPABILITY
                and creation_target is None
            ):
                raise PermissionError("Recipient creation requires its exact durable target.")
            if creation_target is not None:
                creation_target = creation_fence.snapshot_target(creation_target)
                creation_fence.require_pending(
                    creation_target,
                    self._session_creation_decisions.get(creation_target.key),
                    request_commitment=participant_request_commitment,
                    requested_session_id=request.session_id,
                )
            self._require_available_closure_identity_unlocked(session_id)
            admission = interaction_rules._copy_optional_interaction_admission(
                session_id,
                interaction_started_event,
                interaction_source_messages,
                defer_transcript=True,
            )
            if session_id in self._sessions:
                raise ValueError(f"Session already exists: {session_id}")
            if request.parent_session_id == session_id:
                raise ValueError("Session cannot be its own parent.")
            if (
                request.parent_session_id is not None
                and request.parent_session_id not in self._sessions
            ):
                raise ValueError(f"Parent session not found: {request.parent_session_id}")
            parent_session = (
                None
                if request.parent_session_id is None
                else self._sessions[request.parent_session_id]
            )
            if parent_session is not None:
                for owner in self._session_closure_progress.values():
                    _check_closure_lineage_owner(owner, (parent_session.id,))
            now = self._ownership_clock()
            session = Session(
                id=session_id,
                instance_id=session_instance_id_for_run_request(
                    request,
                    session_id=session_id,
                ),
                agent_name=request.agent_name,
                provider_name=identity.provider_name,
                model=identity.model,
                parent_session_id=request.parent_session_id,
                causal_budget_id=request.causal_budget_id or request.task_id or session_id,
                runtime_name=identity.runtime_name,
                runtime_version=identity.runtime_version,
                environment_name=request.environment_name,
                status=(SessionStatus.RUNNING if admission is not None else SessionStatus.PENDING),
                created_at=now,
                updated_at=now,
                last_activity_at=now,
                invocation=session_invocation_for_run_request(
                    request,
                    session_id=session_id,
                    parent_session=parent_session,
                ),
                labels=request.labels,
                metadata=session_metadata_for_creation(
                    request.metadata,
                    identity=identity,
                    tool_capability_ceiling=request.tool_capability_ceiling,
                    execution_deadline=request.execution_deadline,
                    parent_session=parent_session,
                    prepared_request=request,
                ),
                run_epoch=1 if admission is not None else 0,
            )
            initial_operation_records = _prepare_initial_session_operation_records(
                session,
                operation_initializer,
            )
            prepared_admission_events: _PreparedInMemoryEventAppend | None = None
            prepared_checkpoint: _PreparedInMemoryCheckpointStore | None = None
            pending_checkpoint: _PreparedInMemoryCheckpointStore | None = None
            if admission is None and checkpoint_transform is not None:
                transformed = checkpoint_transform(session.model_copy(deep=True), None)
                if transformed is not None:
                    copied_checkpoint = checkpoint_preservation._replace_checkpoint_preserving_completion_result_event_publications(
                        None,
                        copy_durable_json_object(transformed, "checkpoint"),
                        session_id=session.id,
                    )
                    pending_checkpoint = self._prepare_checkpoint_store_unlocked(
                        session.id,
                        copied_checkpoint,
                    )
            if admission is not None:
                started_event, source_messages = admission
                interaction_id = started_event.interaction_id
                if interaction_id is None:
                    raise AssertionError("Interaction admission lost its identity.")
                prepared_admission_events = self._prepare_event_append_unlocked(
                    session,
                    [started_event],
                )
                session = session.model_copy(update={"last_activity_at": now})
                prepared_checkpoint = self._prepare_checkpoint_store_unlocked(
                    session.id,
                    _initial_transcript_pending_checkpoint(
                        session,
                        interaction_id,
                        checkpoint_transform=checkpoint_transform,
                    ),
                    additional_event_records=tuple(
                        event.record for event in prepared_admission_events.events
                    ),
                )
            if result_checkpoint_transform is not None:
                current_checkpoint = (
                    None if prepared_checkpoint is None else prepared_checkpoint.checkpoint
                )
                transformed = result_checkpoint_transform(
                    session.model_copy(deep=True),
                    checkpoint_preservation._copy_checkpoint_for_transform(
                        current_checkpoint,
                        session_id=session.id,
                    ),
                )
                if transformed is None:
                    raise ValueError("Result checkpoint transform must return a checkpoint.")
                prepared_checkpoint = self._prepare_checkpoint_store_unlocked(
                    session.id,
                    checkpoint_preservation._checkpoint_transform_result_preserving_completion_result_event_publications(
                        current_checkpoint,
                        transformed,
                        session_id=session.id,
                    ),
                    additional_event_records=(
                        ()
                        if prepared_admission_events is None
                        else tuple(event.record for event in prepared_admission_events.events)
                    ),
                )
            elif pending_checkpoint is not None:
                prepared_checkpoint = pending_checkpoint

            participant_binding = None
            participant_receipt = None
            recipient_messages = []
            if participant_binding_factory is not None:
                participant_binding, participant_receipt = participant_binding_factory(
                    session.model_copy(deep=True)
                )
                if type(participant_binding) is not ParticipantSessionBinding:
                    raise TypeError("Participant binding factory returned an invalid binding.")
                if type(participant_receipt) is not ParticipantSessionCreationReceipt:
                    raise TypeError("Participant binding factory returned an invalid receipt.")
                if participant_binding.session_id != session.id:
                    raise ValueError("Participant binding session identity conflicts.")
                if participant_binding.session_instance_id != session.instance_id:
                    raise ValueError("Participant binding incarnation conflicts.")
                if participant_receipt.binding != participant_binding:
                    raise ValueError("Participant receipt binding conflicts.")
                if participant_binding.creation_key in self._participant_session_ids_by_key:
                    raise ValueError("Participant creation key was concurrently claimed.")
                if type(participant_request_commitment) is not str:
                    raise ValueError("Participant request commitment is required.")
                if (
                    participant_receipt.recipient_metadata_json is not None
                    and participant_provenance is not _RECIPIENT_PROVENANCE_CAPABILITY
                ):
                    raise PermissionError(
                        "Recipient provenance requires the trusted application boundary."
                    )
                if participant_receipt.recipient_metadata_json is not None:
                    recipient_messages = copy_transcript_messages(request.messages)
                if recipient_selection is not None:
                    stored_selection = self._context_selection_receipts.get(
                        recipient_selection.selection_key
                    )
                    if (
                        stored_selection is None
                        or stored_selection != recipient_selection
                        or stored_selection.state not in {"adopted", "transferred"}
                    ):
                        raise PermissionError(
                            "Recipient context-view ownership changed before child creation."
                        )
                if recipient_receipt_validator is not None:
                    recipient_receipt_validator(session, participant_receipt)

            creation_decision = None
            if creation_target is not None:
                if participant_binding is None:
                    raise PermissionError("Creation targets require participant ownership.")
                assert participant_receipt is not None
                creation_fence.validate_binding(
                    creation_target,
                    participant_binding,
                    requested_session_id=participant_receipt.requested_session_id,
                )
                creation_decision = creation_fence.created(
                    creation_target, session, participant_receipt
                )

            # Every caller-controlled transform and validation step is complete.
            # Publish the prepared state while retaining the in-memory store lock.
            self._register_private_authority_alias_unlocked(
                session.id,
                field_name="session_id",
            )
            self._sessions[session.id] = session
            if creation_decision is not None:
                self._session_creation_decisions[creation_decision.target.key] = creation_decision
            self._index_session_parent_unlocked(session)
            self._events[session.id] = []
            self._event_ids[session.id] = set()
            self._session_event_records[session.id] = []
            self._child_lifecycle_event_records[session.id] = {}
            self._interaction_event_records[session.id] = {}
            self._latest_interaction_event_records[session.id] = {}
            self._latest_interaction_event_records_by_sequence[session.id] = []
            self._pending_action_event_records[session.id] = {}
            self._session_operation_records[session.id] = initial_operation_records
            if participant_binding is not None:
                self._participant_session_bindings[session.id] = participant_binding.model_copy(
                    deep=True
                )
                assert participant_receipt is not None
                self._participant_session_receipts[session.id] = participant_receipt.model_copy(
                    deep=True
                )
                self._participant_session_ids_by_key[participant_binding.creation_key] = session.id
                self._participant_session_request_commitments[participant_binding.creation_key] = (
                    participant_request_commitment
                )
            self._transcripts[session.id] = recipient_messages
            self._transcript_interaction_ids[session.id] = [None] * len(recipient_messages)
            self._latest_transcript_indexes_by_role[session.id] = {}
            self._transcript_search_documents[session.id] = []
            if recipient_messages:
                self._replace_latest_transcript_role_indexes_unlocked(session.id)
                self._replace_transcript_search_session_unlocked(session.id, recipient_messages)
            self._refresh_child_lifecycle_candidate_unlocked(session)
            if prepared_admission_events is not None:
                assert admission is not None
                started_event, source_messages = admission
                interaction_id = started_event.interaction_id
                assert interaction_id is not None
                self._sessions[session.id] = self._apply_event_append_unlocked(
                    session,
                    prepared_admission_events,
                    activity_at=now,
                )
                self._deferred_interaction_inputs[session.id] = (
                    deferred_interaction_input_for_run_request(
                        request,
                        session_id=session.id,
                        interaction_id=interaction_id,
                        source_messages=source_messages,
                    )
                )
            if prepared_checkpoint is not None:
                self._apply_checkpoint_store_unlocked(session.id, prepared_checkpoint)
            session = self._sessions[session.id]
            if admission is not None:
                invocation_scope._activate_session_run_fence(session)
            return session.model_copy(deep=True)

    async def create_participant_owned_session(
        self,
        creation_request: ParticipantSessionCreationRequest,
        *,
        resolved_request: RunRequest,
        identity: SessionIdentity,
        binding_factory: Callable[
            [Session], tuple[ParticipantSessionBinding, ParticipantSessionCreationReceipt]
        ],
        recipient_provenance: object | None = None,
        recipient_selection: _ContextViewSelectionReceipt | None = None,
        creation_target=None,
        recipient_receipt_validator=None,
    ) -> tuple[Session, ParticipantSessionCreationReceipt]:
        from cayu.sessions.context_views import (
            ParticipantSessionCreationRequest,
        )

        if type(creation_request) is not ParticipantSessionCreationRequest:
            raise TypeError("Participant creation requires a typed creation request.")
        if (
            creation_request.metadata_json is not None
            and recipient_provenance is not _RECIPIENT_PROVENANCE_CAPABILITY
        ):
            raise PermissionError("Recipient provenance requires the trusted application boundary.")
        if type(resolved_request) is not RunRequest:
            raise TypeError("Participant creation requires a resolved RunRequest.")
        if not callable(binding_factory):
            raise TypeError("binding_factory must be callable.")
        async with self._participant_creation_lock:
            existing_id = self._participant_session_ids_by_key.get(creation_request.creation_key)
            if existing_id is not None:
                session = self._sessions.get(existing_id)
                receipt = self._participant_session_receipts.get(existing_id)
                if session is None or receipt is None:
                    raise RuntimeError("Participant creation index is inconsistent.")
                if (
                    self._participant_session_request_commitments.get(creation_request.creation_key)
                    != creation_request.request_commitment
                ):
                    raise ValueError("Participant creation key conflicts with the request.")
                if (session.provider_name, session.model) != (
                    identity.provider_name,
                    identity.model,
                ):
                    raise ValueError(
                        "Participant creation key conflicts with the execution target."
                    )
                expected_binding, expected_receipt = binding_factory(session.model_copy(deep=True))
                if expected_binding != self._participant_session_bindings.get(existing_id):
                    raise ValueError("Participant creation key conflicts with its binding.")
                if expected_receipt != receipt:
                    raise ValueError("Participant creation key conflicts with its receipt.")
                if creation_target is not None:
                    from cayu.storage._creation_fence import validate_replay

                    await validate_replay(self, creation_target, session)
                return session.model_copy(deep=True), receipt.model_copy(deep=True)
            session = await self.create(
                resolved_request,
                identity=identity,
                participant_binding_factory=binding_factory,
                participant_request_commitment=creation_request.request_commitment,
                participant_provenance=recipient_provenance,
                recipient_selection=recipient_selection,
                creation_target=creation_target,
                recipient_receipt_validator=recipient_receipt_validator,
            )
            receipt = self._participant_session_receipts.get(session.id)
            if receipt is None:
                raise RuntimeError("Participant creation did not persist its receipt.")
            return session, receipt.model_copy(deep=True)

    async def lookup_participant_session_creation(
        self, creation_request: ParticipantSessionCreationRequest
    ) -> tuple[Session, ParticipantSessionCreationReceipt] | None:
        from cayu.sessions.context_views import ParticipantSessionCreationRequest

        if type(creation_request) is not ParticipantSessionCreationRequest:
            raise TypeError("Participant creation requires a typed creation request.")
        async with self._participant_creation_lock:
            session_id = self._participant_session_ids_by_key.get(creation_request.creation_key)
            if session_id is None:
                return None
            session = self._sessions.get(session_id)
            receipt = self._participant_session_receipts.get(session_id)
            if session is None or receipt is None:
                raise RuntimeError("Participant creation index is inconsistent.")
            if self._participant_session_request_commitments.get(creation_request.creation_key) != (
                creation_request.request_commitment
            ):
                raise ValueError("Participant creation key conflicts with the request.")
            return session.model_copy(deep=True), receipt.model_copy(deep=True)

    async def load_participant_session_binding(
        self, session_id: str
    ) -> ParticipantSessionBinding | None:
        async with self._lock:
            binding = self._participant_session_bindings.get(session_id)
            return None if binding is None else binding.model_copy(deep=True)

    async def _scan_participant_session_bindings(self, participant, *, after=None, limit=32):
        from cayu.sessions._participant_discovery import prepare_scan, reference

        query = prepare_scan(participant, after, limit)
        async with self._lock:
            bindings = sorted(
                (
                    item
                    for item in self._participant_session_bindings.values()
                    if item.participant == query.participant
                    and (query.after is None or item.creation_key > query.after)
                ),
                key=lambda item: item.creation_key,
            )[: query.limit]
            result = []
            for binding in bindings:
                session = self._sessions.get(binding.session_id)
                receipt = self._participant_session_receipts.get(binding.session_id)
                if (
                    session is None
                    or session.instance_id != binding.session_instance_id
                    or receipt is None
                    or receipt.binding != binding
                    or self._participant_session_ids_by_key.get(binding.creation_key) != session.id
                ):
                    raise ValueError("Participant inventory lacks coherent native creation.")
                result.append(reference(receipt, query))
            return tuple(result)

    async def load_participant_session_creation_receipt(
        self, session_id: str
    ) -> ParticipantSessionCreationReceipt | None:
        async with self._lock:
            receipt = self._participant_session_receipts.get(session_id)
            return None if receipt is None else receipt.model_copy(deep=True)

    async def capture_context_view_publication_source(
        self, session_id: str
    ) -> ContextViewPublicationSource:
        return (await self._capture_completed_turn_snapshot(session_id)).publication

    async def _capture_completed_turn_snapshot(self, session_id: str) -> CompletedTurnSnapshot:
        from cayu.sessions._context_view_source import (
            CompletedTurnSnapshot,
            capture_source,
            closed_round_publication_id,
            completed_boundary,
            publication_frontier,
        )

        async with self._lock:
            session = self._sessions.get(session_id)
            binding = self._participant_session_bindings.get(session_id)
            checkpoint = self._checkpoints.get(session_id)
            pointer = completed_boundary(session, binding, checkpoint)
            assert session is not None and binding is not None
            tool_receipt = None
            if pointer.tool_round_id is not None:
                publication_id = f"tool-round:{pointer.tool_round_id}"
                key = publication_contracts._runtime_publication_storage_key(publication_id)
                record = self._session_operation_records.get(session_id, {}).get(key)
                if record is None:
                    from itertools import islice

                    closures = tuple(
                        islice(
                            (
                                item.event
                                for item in reversed(
                                    self._session_event_records.get(session_id, ())
                                )
                                if item.event.type is EventType.SESSION_CHECKPOINTED
                                and item.event.payload.get("tool_round_id") == pointer.tool_round_id
                                and (
                                    item.event.payload.get("cleared") is True
                                    or item.event.payload.get("transition") == "answered"
                                )
                            ),
                            2,
                        )
                    )
                    publication_id = closed_round_publication_id(pointer, closures)
                    key = publication_contracts._runtime_publication_storage_key(publication_id)
                    record = self._session_operation_records.get(session_id, {}).get(key)
                if record is not None:
                    tool_receipt = publication_rules._reconstruct_runtime_publication_receipt(
                        record,
                        storage_key=key,
                        session_id=session_id,
                        publication_id=publication_id,
                    )
                publication_frontier(pointer, tool_receipt)
                assert tool_receipt is not None
                self._validate_runtime_publication_material_unlocked(tool_receipt)
            end = publication_frontier(pointer, tool_receipt)
            event = self._event_records_by_id.get((session_id, pointer.completion_event_id))
            messages = self._transcripts.get(session_id, [])
            interactions = self._transcript_interaction_ids.get(session_id, [])
            records = tuple(
                TranscriptRecord(
                    index=index,
                    interaction_id=interactions[index],
                    message=detach_message(messages[index]),
                )
                for index in range(
                    pointer.source_transcript_cursor,
                    min(end, len(messages)),
                )
            )
            publication = capture_source(
                session,
                binding,
                checkpoint,
                pointer,
                None if event is None else event.event,
                records,
                tool_receipt,
            )
            return CompletedTurnSnapshot(
                publication=publication,
                current_session=copy_session(session),
                checkpoint=deepcopy(checkpoint),
                has_queued_input=any(
                    (session_id, mode) in self._pending_session_messages
                    for mode in SessionMessageDeliveryMode
                ),
                has_closure_owner=any(
                    session_id in _closure_progress_targets(owner)
                    for owner in self._session_closure_progress.values()
                ),
                has_active_model_stage=MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY
                in self._session_operation_records.get(session_id, {}),
                current_transcript_cursor=len(messages),
            )

    async def publish_context_view(
        self, manifest: _ContextViewManifest, *, publication_key: str
    ) -> _ContextViewManifest:
        from cayu.sessions.context_views import (
            CONTEXT_VIEW_MAX_PUBLICATIONS_PER_OWNER,
        )
        from cayu.sessions.context_views import (
            ContextViewManifest as ContextViewManifestType,
        )

        if type(manifest) is not ContextViewManifestType or type(publication_key) is not str:
            raise TypeError("Context-view publication requires typed manifest and key.")
        if len(publication_key.encode("utf-8")) > 512:
            raise ValueError("publication_key must be at most 512 UTF-8 bytes.")
        publication_key = require_clean_nonblank(publication_key, "publication_key")
        async with self._lock:
            existing_id = self._context_publication_keys.get(publication_key)
            if existing_id is not None:
                existing = self._context_views[existing_id]
                if existing != manifest:
                    raise ValueError("Context-view publication key conflicts with its manifest.")
                return existing.model_copy(deep=True)
            if manifest.view_id in self._context_views:
                raise ValueError("Context-view ID is already bound to another manifest.")
            source = self._sessions.get(manifest.source_session_id)
            if source is None or source.instance_id != manifest.source_session_instance_id:
                raise LookupError("The source session incarnation is unavailable.")
            publication_count = sum(
                view.source_owner == manifest.source_owner for view in self._context_views.values()
            )
            if publication_count >= CONTEXT_VIEW_MAX_PUBLICATIONS_PER_OWNER:
                raise OverflowError("Context-view publication quota exceeded for the owner.")
            self._context_views[manifest.view_id] = manifest.model_copy(deep=True)
            self._context_publication_keys[publication_key] = manifest.view_id
            return manifest.model_copy(deep=True)

    async def lookup_context_view_publication(
        self, publication_key: str
    ) -> _ContextViewManifest | None:
        from cayu.sessions.context_views import ContextViewManifest as ContextViewManifestType

        if type(publication_key) is not str or not publication_key.strip():
            raise ValueError("Context-view publication keys must be non-blank strings.")
        if len(publication_key.encode("utf-8")) > 512:
            raise ValueError("publication_key must be at most 512 UTF-8 bytes.")
        async with self._lock:
            view_id = self._context_publication_keys.get(publication_key)
            if view_id is None:
                return None
            manifest = self._context_views.get(view_id)
            if manifest is None or type(manifest) is not ContextViewManifestType:
                raise RuntimeError("Context-view publication index is inconsistent.")
            return manifest.model_copy(deep=True)

    def _require_context_view_lifecycle_capacity(
        self, view_id: str, *, additional_slots: int = 0
    ) -> None:
        from cayu.sessions.context_views import validate_context_view_lifecycle_capacity

        events = sum(event.view_id == view_id for event in self._context_lifecycle_events.values())
        unsettled = sum(
            receipt.view.view_id == view_id
            and receipt.state in {"selected", "adopted", "transferred"}
            for receipt in self._context_selection_receipts.values()
        )
        validate_context_view_lifecycle_capacity(
            events, unsettled, additional_slots=additional_slots
        )

    async def select_context_view(
        self, request: _ContextViewSelectionRequest
    ) -> _ContextViewSelectionReceipt:
        return await self._select_context_view(request)

    async def _select_context_view(self, request, *, target=None):
        from cayu.sessions.context_views import (
            CONTEXT_VIEW_EXPIRY_BATCH_SIZE,
            CONTEXT_VIEW_MAX_SELECTIONS_PER_OWNER,
            context_view_manifest_bytes,
        )
        from cayu.sessions.context_views import (
            ContextViewSelectionReceipt as ContextViewSelectionReceiptType,
        )
        from cayu.sessions.context_views import (
            ContextViewSelectionRequest as ContextViewSelectionRequestType,
        )

        if type(request) is not ContextViewSelectionRequestType:
            raise TypeError("Context-view selection requires a typed request.")
        request = ContextViewSelectionRequestType.model_validate(request)
        from cayu.sessions.context_views import json_commitment

        request_commitment = json_commitment(
            canonical_bounded_durable_json_bytes(
                request.model_dump(mode="json"),
                "context view selection request",
                max_bytes=256 * 1024,
                max_nodes=8192,
                max_nesting=64,
            ).decode("utf-8"),
            "context view selection request",
        )
        async with _InMemoryContextViewSelectionTransaction(self):
            from cayu.sessions._context_selection_fence import require_not_excluded

            require_not_excluded(
                request,
                self._context_selection_exclusions.get(request.selection_key),
                target=target,
            )
            now_ms = int(self._ownership_clock().timestamp() * 1000)
            expired_count = 0
            for selection_key, receipt in sorted(
                self._context_selection_receipts.items(),
                key=lambda item: item[0] != request.selection_key,
            ):
                if expired_count >= CONTEXT_VIEW_EXPIRY_BATCH_SIZE:
                    break
                if (
                    receipt.owner == request.source_owner
                    and receipt.state == "selected"
                    and receipt.expires_at_ms <= now_ms
                ):
                    self._require_context_view_lifecycle_capacity(receipt.view.view_id)
                    expired = receipt.model_copy(update={"state": "expired"}, deep=True)
                    self._context_selection_receipts[selection_key] = expired
                    from cayu.sessions.context_views import ContextViewLifecycleEvent

                    operation_key = f"expiry:{selection_key}:{receipt.ownership_revision}"
                    event = ContextViewLifecycleEvent(
                        event_id="sha256:"
                        + sha256(f"context-view-event:{operation_key}".encode()).hexdigest(),
                        operation_key=operation_key,
                        selection_key=selection_key,
                        view_id=receipt.view.view_id,
                        state="expired",
                        owner=expired.owner,
                        owner_participant=expired.owner_participant,
                        pin_commitment=expired.pin_commitment,
                        ownership_revision=expired.ownership_revision,
                    )
                    self._context_lifecycle_events[event.event_id] = event
                    expired_count += 1
            existing = self._context_selection_receipts.get(request.selection_key)
            if existing is not None:
                if self._context_selection_request_commitments.get(request.selection_key) != (
                    request_commitment
                ):
                    raise ValueError("Context-view selection key conflicts with its request.")
                if existing.state == "selected" and existing.expires_at_ms <= now_ms:
                    self._require_context_view_lifecycle_capacity(existing.view.view_id)
                    existing = existing.model_copy(update={"state": "expired"}, deep=True)
                    self._context_selection_receipts[request.selection_key] = existing
                    from cayu.sessions.context_views import ContextViewLifecycleEvent

                    operation_key = f"expiry:{request.selection_key}:{existing.ownership_revision}"
                    event = ContextViewLifecycleEvent(
                        event_id="sha256:"
                        + sha256(f"context-view-event:{operation_key}".encode()).hexdigest(),
                        operation_key=operation_key,
                        selection_key=request.selection_key,
                        view_id=existing.view.view_id,
                        state="expired",
                        owner=existing.owner,
                        owner_participant=existing.owner_participant,
                        pin_commitment=existing.pin_commitment,
                        ownership_revision=existing.ownership_revision,
                    )
                    self._context_lifecycle_events[event.event_id] = event
                return existing.model_copy(deep=True)
            source = self._sessions.get(request.source_session_id)
            if source is None or source.instance_id != request.source_session_instance_id:
                raise LookupError("Context-view source session incarnation is unavailable.")
            from cayu.sessions._context_selection_fence import require_selection_source

            require_selection_source(
                target,
                self._participant_session_bindings.get(request.source_session_id),
                now_ms=now_ms,
            )
            candidates = [
                view
                for view in self._context_views.values()
                if view.source_owner == request.source_owner
                and view.source_session_id == request.source_session_id
                and view.source_session_instance_id == request.source_session_instance_id
                and view.projection_schema == request.projection_schema
                and view.extension_set_commitment == request.extension_set_commitment
                and (request.selector != "exact" or view.view_id == request.exact_view_id)
                and (
                    request.minimum_transcript_cursor is None
                    or view.transcript_cursor >= request.minimum_transcript_cursor
                )
            ]
            if not candidates:
                raise LookupError("No eligible context view is available.")
            if len(candidates) > request.limits.max_views:
                raise OverflowError("Context-view count exceeds the requested limit.")
            selection_count = sum(
                receipt.owner == request.source_owner
                for receipt in self._context_selection_receipts.values()
            )
            if selection_count >= CONTEXT_VIEW_MAX_SELECTIONS_PER_OWNER:
                raise OverflowError("Context-view selection quota exceeded for the owner.")
            active_pins = sum(
                1
                for receipt in self._context_selection_receipts.values()
                if receipt.owner == request.source_owner
                and receipt.state in {"selected", "adopted", "transferred"}
                and (receipt.state != "selected" or receipt.expires_at_ms > now_ms)
            )
            if active_pins >= request.limits.max_pins:
                raise OverflowError("Context-view pin count exceeds the requested limit.")
            selected = max(candidates, key=lambda view: (view.transcript_cursor, view.view_id))
            from cayu.sessions._context_selection_fence import require_selected_participant

            require_selected_participant(target, selected)
            self._require_context_view_lifecycle_capacity(selected.view_id, additional_slots=1)
            selected_bytes = context_view_manifest_bytes(selected)
            if selected_bytes > request.limits.max_view_bytes:
                raise OverflowError("Context-view manifest exceeds the requested byte limit.")
            retained_views = {
                receipt.view.view_id: receipt.view
                for receipt in self._context_selection_receipts.values()
                if receipt.owner == request.source_owner
                and receipt.state in {"selected", "adopted", "transferred"}
                and (receipt.state != "selected" or receipt.expires_at_ms > now_ms)
            }
            retained_views[selected.view_id] = selected
            retained_bytes = sum(
                context_view_manifest_bytes(view) for view in retained_views.values()
            )
            if retained_bytes > request.limits.max_retained_bytes:
                raise OverflowError("Context-view retained bytes exceed the requested limit.")
            pin_commitment = (
                "sha256:"
                + sha256(
                    f"context-pin:{request.selection_key}:{selected.view_id}".encode()
                ).hexdigest()
            )
            receipt = ContextViewSelectionReceiptType(
                selection_key=request.selection_key,
                view=selected,
                owner=request.source_owner,
                state="selected",
                pin_commitment=pin_commitment,
                owner_participant=selected.participant,
                expires_at_ms=int(
                    self._ownership_clock().timestamp() * 1000
                    + request.limits.max_lifetime_seconds * 1000
                ),
            )
            self._context_selection_receipts[request.selection_key] = receipt
            self._context_selection_request_commitments[request.selection_key] = request_commitment
            return receipt.model_copy(deep=True)

    async def _exclude_context_view_selection(self, request, *, authority):
        return await self._context_selection_control(request, authority=authority, exclude=True)

    async def _reserve_context_view_selection_control(self, request, *, authority):
        return await self._context_selection_control(request, authority=authority, exclude=False)

    async def _context_selection_control(
        self, request, *, authority, exclude, target=None, register=False
    ):
        from cayu.sessions._context_selection_fence import (
            CONTEXT_SELECTION_MAX_CONTROLS_PER_OWNER,
            ContextViewSelectionDecision,
            control_transition,
            require_authority,
            snapshot_request,
        )

        require_authority(authority)
        request = snapshot_request(request)
        async with self._lock:
            existing = self._context_selection_decision_unlocked(request)
            decision = control_transition(
                request, existing, exclude=exclude, target=target, register=register
            )
            if decision == existing:
                return decision
            if (
                existing is None
                and sum(
                    value.request.source_owner == request.source_owner
                    for value in self._context_selection_exclusions.values()
                )
                >= CONTEXT_SELECTION_MAX_CONTROLS_PER_OWNER
            ):
                raise OverflowError("Context-view selection control quota exceeded.")
            self._context_selection_exclusions[request.selection_key] = decision
            return ContextViewSelectionDecision.model_validate(decision)

    def _context_selection_decision_unlocked(self, request):
        from cayu.sessions._context_selection_fence import (
            ContextViewSelectionConflict,
            ContextViewSelectionDecision,
            selected_decision,
        )

        excluded = self._context_selection_exclusions.get(request.selection_key)
        receipt = self._context_selection_receipts.get(request.selection_key)
        if excluded is not None:
            excluded = ContextViewSelectionDecision.model_validate(excluded)
            if (
                excluded.request.selection_key != request.selection_key
                or (receipt is not None and excluded.state != "reserved")
                or excluded.state not in {"reserved", "excluded"}
            ):
                raise ValueError("Context-view exclusion indexes conflict.")
            if excluded.request != request:
                raise ContextViewSelectionConflict("Context-view exclusion evidence conflicts.")
            if receipt is None:
                return excluded
        if receipt is None:
            return None
        return selected_decision(
            request,
            receipt,
            self._context_selection_request_commitments.get(request.selection_key),
            excluded,
        )

    async def _read_context_view_retention(self, target):
        from cayu.collaboration._contracts import (
            ExactConflict,
            ExactMatch,
            ExactNotFound,
            ExactUnavailable,
        )
        from cayu.sessions._context_selection_fence import (
            ContextViewRetentionEvidence,
            ContextViewSelectionConflict,
            ContextViewSelectionTarget,
            retention_evidence,
        )

        target = ContextViewSelectionTarget.model_validate(target)
        async with self._lock:
            try:
                decision = self._context_selection_decision_unlocked(target.request)
                if decision is not None and decision.target != target:
                    return ExactConflict()
                result = retention_evidence(
                    target,
                    decision,
                    self._context_selection_receipts.get(target.request.selection_key),
                )
            except ContextViewSelectionConflict:
                return ExactConflict()
            except (TypeError, ValueError):
                return ExactUnavailable()
            return (
                ExactNotFound()
                if result is None
                else ExactMatch[ContextViewRetentionEvidence](receipt=result)
            )

    async def read_context_view_selection_decision(self, request):
        from cayu.collaboration._contracts import (
            ExactConflict,
            ExactMatch,
            ExactNotFound,
            ExactUnavailable,
        )
        from cayu.sessions._context_selection_fence import (
            ContextViewSelectionConflict,
            ContextViewSelectionDecision,
            snapshot_request,
        )

        request = snapshot_request(request)
        async with self._lock:
            try:
                decision = self._context_selection_decision_unlocked(request)
            except ContextViewSelectionConflict:
                return ExactConflict()
            except (TypeError, ValueError):
                return ExactUnavailable()
            return (
                ExactNotFound()
                if decision is None
                else ExactMatch[ContextViewSelectionDecision](receipt=decision)
            )

    async def lookup_context_view_selection(self, selection_key: str):
        from cayu.sessions.context_views import ContextViewSelectionReceipt

        async with self._lock:
            receipt = self._context_selection_receipts.get(selection_key)
            if receipt is None:
                return None
            return ContextViewSelectionReceipt.model_validate(receipt.model_dump(mode="json"))

    async def transition_context_view_ownership(
        self, request: _ContextViewOwnershipRequest
    ) -> _ContextViewSelectionReceipt:
        return await self._transition_context_view_ownership(request)

    async def _transition_context_view_ownership(self, request, *, target=None):
        from cayu.sessions.context_views import (
            ContextViewOwnershipRequest as ContextViewOwnershipRequestType,
        )
        from cayu.sessions.context_views import (
            json_commitment,
        )

        if type(request) is not ContextViewOwnershipRequestType:
            raise TypeError("Context-view ownership requires a typed request.")
        request = ContextViewOwnershipRequestType.model_validate(request)
        request_commitment = json_commitment(
            canonical_bounded_durable_json_bytes(
                request.model_dump(mode="json"),
                "context view ownership request",
                max_bytes=256 * 1024,
                max_nodes=8192,
                max_nesting=64,
            ).decode("utf-8"),
            "context view ownership request",
        )
        async with self._lock:
            from cayu.sessions._context_selection_fence import require_selection_adoption

            control = self._context_selection_exclusions.get(request.selection_key)
            require_selection_adoption(
                request,
                self._context_selection_decision_unlocked(control.request)
                if control is not None
                else None,
                target=target,
            )
            prior_commitment = self._context_ownership_operation_commitments.get(
                request.operation_key
            )
            if prior_commitment is not None:
                if prior_commitment != request_commitment:
                    raise ValueError("Ownership operation key conflicts with its request.")
                receipt = self._context_ownership_operation_receipts[request.operation_key]
                return receipt.model_copy(deep=True)
            receipt = self._context_selection_receipts.get(request.selection_key)
            if receipt is None or receipt.view.view_id != request.view_id:
                raise LookupError("Context-view selection is unavailable.")
            if receipt.pin_commitment != request.pin_commitment:
                raise ValueError("Context-view pin commitment conflicts with the request.")
            if receipt.state != request.expected_state:
                raise ValueError("Context-view ownership state is stale or already settled.")
            if receipt.ownership_revision != request.expected_revision:
                raise ValueError("Context-view ownership revision is stale.")
            if receipt.owner != request.current_owner:
                raise PermissionError("The current owner does not control this context-view pin.")
            if (
                request.current_participant is not None
                and receipt.owner_participant is not None
                and receipt.owner_participant != request.current_participant
            ):
                raise PermissionError(
                    "The current participant does not control this context-view pin."
                )
            now_ms = int(self._ownership_clock().timestamp() * 1000)
            from cayu.sessions._context_selection_fence import require_adoption_deadline

            require_adoption_deadline(target, now_ms)
            self._require_context_view_lifecycle_capacity(receipt.view.view_id)
            if receipt.state == "selected" and receipt.expires_at_ms <= now_ms:
                expired = receipt.model_copy(update={"state": "expired"}, deep=True)
                self._context_selection_receipts[request.selection_key] = expired
                from cayu.sessions.context_views import ContextViewLifecycleEvent

                operation_key = f"expiry:{request.selection_key}:{receipt.ownership_revision}"
                self._context_lifecycle_events[
                    "sha256:" + sha256(f"context-view-event:{operation_key}".encode()).hexdigest()
                ] = ContextViewLifecycleEvent(
                    event_id="sha256:"
                    + sha256(f"context-view-event:{operation_key}".encode()).hexdigest(),
                    operation_key=operation_key,
                    selection_key=request.selection_key,
                    view_id=receipt.view.view_id,
                    state="expired",
                    owner=expired.owner,
                    owner_participant=expired.owner_participant,
                    pin_commitment=expired.pin_commitment,
                    ownership_revision=expired.ownership_revision,
                )
                raise LookupError("Context-view selection has expired.")
            if request.operation == "adopt":
                assert request.destination_owner is not None
                next_state = "adopted"
                next_owner = request.destination_owner
            elif request.operation == "transfer":
                assert request.destination_owner is not None
                next_state = "transferred"
                next_owner = request.destination_owner
            else:
                next_state = "released"
                next_owner = receipt.owner
            next_participant = receipt.owner_participant
            if request.operation != "release":
                if request.destination_participant is None:
                    if request.destination_owner != receipt.owner:
                        raise PermissionError(
                            "A transfer to another owner requires destination participant evidence."
                        )
                else:
                    next_participant = request.destination_participant
            updated = receipt.model_copy(
                update={
                    "owner": next_owner,
                    "state": next_state,
                    "ownership_revision": receipt.ownership_revision + 1,
                    "owner_participant": next_participant,
                },
                deep=True,
            )
            self._require_context_view_lifecycle_capacity(
                updated.view.view_id, additional_slots=int(next_state != "released")
            )
            self._context_selection_receipts[request.selection_key] = updated
            self._context_ownership_operation_commitments[request.operation_key] = (
                request_commitment
            )
            self._context_ownership_operation_receipts[request.operation_key] = updated
            from cayu.sessions.context_views import ContextViewLifecycleEvent

            event = ContextViewLifecycleEvent(
                event_id="sha256:"
                + sha256(f"context-view-event:{request.operation_key}".encode()).hexdigest(),
                operation_key=request.operation_key,
                selection_key=request.selection_key,
                view_id=request.view_id,
                state=next_state,
                owner=updated.owner,
                owner_participant=updated.owner_participant,
                pin_commitment=updated.pin_commitment,
                ownership_revision=updated.ownership_revision,
            )
            self._context_lifecycle_events[event.event_id] = event
            return updated.model_copy(deep=True)

    async def validate_context_view_source_closure(self, session_id: str) -> None:
        async with self._lock:
            now_ms = int(self._ownership_clock().timestamp() * 1000)
            for receipt in self._context_selection_receipts.values():
                if (
                    receipt.view.source_session_id == session_id
                    and receipt.state in {"selected", "adopted", "transferred"}
                    and (receipt.state != "selected" or receipt.expires_at_ms > now_ms)
                ):
                    raise ValueError("Session has an active context-view retention pin.")

    async def validate_context_view_compaction(
        self, session_id: str, expected_transcript_cursor: int
    ) -> None:
        if type(expected_transcript_cursor) is not int or expected_transcript_cursor < 0:
            raise ValueError("Context-view compaction requires a non-negative transcript cursor.")
        async with self._lock:
            self._validate_context_view_compaction_unlocked(session_id, expected_transcript_cursor)

    def _validate_context_view_compaction_unlocked(
        self, session_id: str, expected_transcript_cursor: int
    ) -> None:
        from cayu.sessions.context_views import require_independent_context_view_material

        now_ms = int(self._ownership_clock().timestamp() * 1000)
        for receipt in self._context_selection_receipts.values():
            if (
                receipt.view.source_session_id == session_id
                and receipt.state in {"selected", "adopted", "transferred"}
                and (receipt.state != "selected" or receipt.expires_at_ms > now_ms)
                and receipt.view.transcript_cursor <= expected_transcript_cursor
            ):
                manifest = self._context_views.get(receipt.view.view_id)
                if manifest is None or manifest != receipt.view:
                    raise ValueError("Pinned context-view material is unavailable for compaction.")
                require_independent_context_view_material(manifest)

    async def read_context_view_lifecycle_events(
        self, view_id: str, *, limit: int = 256, owner_participant: _ParticipantRef | None = None
    ) -> tuple[_ContextViewLifecycleEvent, ...]:
        if type(limit) is not int or not 1 <= limit <= 256:
            raise ValueError("Context-view lifecycle event limits must be between 1 and 256.")
        if owner_participant is not None:
            from cayu.collaboration.participants import ParticipantRef

            owner_participant = ParticipantRef.model_validate(owner_participant)
        async with self._lock:
            events = [
                event
                for event in self._context_lifecycle_events.values()
                if event.view_id == view_id
                and (owner_participant is None or event.owner_participant == owner_participant)
            ]
            events.sort(key=lambda event: (event.ownership_revision, event.event_id))
            return tuple(event.model_copy(deep=True) for event in events[:limit])

    async def read_context_view(
        self, view_id: str, *, source_session_id: str
    ) -> _ContextViewReadback:
        from cayu.sessions.context_views import ContextViewReadback as ContextViewReadbackType

        async with self._lock:
            view = self._context_views.get(view_id)
            if view is None or view.source_session_id != source_session_id:
                raise LookupError("Context view is unavailable.")
            return ContextViewReadbackType(view=view.model_copy(deep=True))

    async def create_fork(
        self,
        *,
        source_session_id: str,
        fork: Session,
        source_statuses: set[SessionStatus],
        transcript_cursor: int | None,
        checkpoint_transform: CheckpointTransform | None,
        system_prompt_replacement: ForkSystemPromptReplacement | None = None,
        expected_source_run_epoch: int,
        operation_initializer: SessionOperationInitializer | None = None,
    ) -> Session:
        return await self._create_fork(
            source_session_id=source_session_id,
            fork=fork,
            source_statuses=source_statuses,
            transcript_cursor=transcript_cursor,
            checkpoint_transform=checkpoint_transform,
            system_prompt_replacement=system_prompt_replacement,
            expected_source_run_epoch=expected_source_run_epoch,
            transcript_validator=None,
            operation_initializer=operation_initializer,
        )

    async def create_fork_with_transcript_validation(
        self,
        *,
        source_session_id: str,
        fork: Session,
        source_statuses: set[SessionStatus],
        transcript_cursor: int | None,
        checkpoint_transform: CheckpointTransform | None,
        system_prompt_replacement: ForkSystemPromptReplacement | None = None,
        expected_source_run_epoch: int,
        transcript_validator: ForkTranscriptValidator,
        operation_initializer: SessionOperationInitializer | None = None,
    ) -> Session:
        return await self._create_fork(
            source_session_id=source_session_id,
            fork=fork,
            source_statuses=source_statuses,
            transcript_cursor=transcript_cursor,
            checkpoint_transform=checkpoint_transform,
            system_prompt_replacement=system_prompt_replacement,
            expected_source_run_epoch=expected_source_run_epoch,
            transcript_validator=transcript_validator,
            operation_initializer=operation_initializer,
        )

    async def create_profiled_fork(
        self,
        *,
        source_session_id: str,
        fork: Session,
        source_statuses: set[SessionStatus],
        transcript_cursor: int | None,
        checkpoint_transform: CheckpointTransform | None,
        system_prompt_replacement: ForkSystemPromptReplacement | None = None,
        expected_source_run_epoch: int,
        relationship: SessionForkProfileRelationship,
        events: list[Event],
        transcript_validator: ForkTranscriptValidator | None = None,
        checkpoint_authority_decoder: ForkCheckpointAuthorityDecoder | None = None,
        operation_initializer: SessionOperationInitializer | None = None,
    ) -> ProfiledSessionForkResult:
        relationship, copied_events = session_fork_rules._copy_profiled_fork_authority(
            fork=fork,
            relationship=relationship,
            events=events,
        )
        created = await self._create_fork(
            source_session_id=source_session_id,
            fork=fork,
            source_statuses=source_statuses,
            transcript_cursor=transcript_cursor,
            checkpoint_transform=checkpoint_transform,
            system_prompt_replacement=system_prompt_replacement,
            expected_source_run_epoch=expected_source_run_epoch,
            transcript_validator=transcript_validator,
            profile_relationship=relationship,
            events=copied_events,
            checkpoint_authority_decoder=checkpoint_authority_decoder,
            operation_initializer=operation_initializer,
        )
        return ProfiledSessionForkResult(session=created, events=tuple(copied_events))

    async def _create_fork(
        self,
        *,
        source_session_id: str,
        fork: Session,
        source_statuses: set[SessionStatus],
        transcript_cursor: int | None,
        checkpoint_transform: CheckpointTransform | None,
        system_prompt_replacement: ForkSystemPromptReplacement | None,
        expected_source_run_epoch: int,
        transcript_validator: ForkTranscriptValidator | None,
        profile_relationship: SessionForkProfileRelationship | None = None,
        events: list[Event] | None = None,
        checkpoint_authority_decoder: ForkCheckpointAuthorityDecoder | None = None,
        operation_initializer: SessionOperationInitializer | None = None,
    ) -> Session:
        source_session_id, fork, allowed_statuses, transcript_cursor = (
            session_fork_rules._prepare_session_fork_request(
                source_session_id=source_session_id,
                fork=fork,
                source_statuses=source_statuses,
                transcript_cursor=transcript_cursor,
            )
        )
        fork = fork.model_copy(update={"instance_id": str(uuid4())})
        async with self._lock:
            self._require_available_closure_identity_unlocked(fork.id)
            for owner in self._session_closure_progress.values():
                _check_closure_lineage_owner(owner, (source_session_id,))
            source_session = session_fork_rules._validate_session_fork_source(
                source_session=self._sessions.get(source_session_id),
                source_session_id=source_session_id,
                fork=fork,
                allowed_statuses=allowed_statuses,
                expected_source_run_epoch=expected_source_run_epoch,
                profile_relationship=profile_relationship,
            )
            if MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY in self._session_operation_records.get(
                source_session_id, {}
            ):
                raise SessionForkActiveModelStageConflict(
                    "Cannot fork a session while a model-completion stage is active."
                )
            if fork.id in self._sessions:
                raise ValueError(f"Session already exists: {fork.id}")

            stored_checkpoint = self._checkpoints.get(source_session_id)
            source_checkpoint_present = stored_checkpoint is not None
            if profile_relationship is not None:
                profile_checkpoint = None
                profile_failure: ValueError | None = None
                try:
                    profile_checkpoint = (
                        stored_checkpoint
                        if checkpoint_authority_decoder is None
                        else checkpoint_authority_decoder(
                            None if stored_checkpoint is None else deepcopy(stored_checkpoint)
                        )
                    )
                    session_fork_rules._validate_profiled_fork_authority(
                        source_session=source_session,
                        source_checkpoint=profile_checkpoint,
                        fork=fork,
                        relationship=profile_relationship,
                        events=() if events is None else events,
                        transcript_cursor=transcript_cursor,
                        checkpoint_transform=checkpoint_transform,
                        system_prompt_replacement=system_prompt_replacement,
                        transcript_validator=transcript_validator,
                    )
                except Exception as exc:
                    profile_failure = session_fork_rules._profiled_fork_authority_validation_error(
                        exc
                    )
                    stored_checkpoint = None
                finally:
                    profile_checkpoint = None
                if profile_failure is not None:
                    raise profile_failure from None

            source_transcript = self._transcripts.get(source_session_id, [])
            source_transcript_length = len(source_transcript)
            # Fork and source transcripts may share message objects: internal
            # transcript lists never escape except through the detaching
            # load/append/query boundaries, so the cheap share stays invisible.
            if transcript_cursor is None:
                copied_transcript = [copy_message(message) for message in source_transcript]
            else:
                if transcript_cursor > source_transcript_length:
                    source_transcript = []
                    raise ValueError("transcript_cursor is greater than source transcript length.")
                copied_transcript = [
                    copy_message(message) for message in source_transcript[:transcript_cursor]
                ]
            source_interaction_ids = self._transcript_interaction_ids.get(source_session_id, [])
            copied_interaction_ids = list(source_interaction_ids[: len(copied_transcript)])
            source_transcript_snapshot = (
                None
                if transcript_validator is None
                else TranscriptSnapshot(
                    records=[
                        TranscriptRecord(
                            index=index,
                            interaction_id=copied_interaction_ids[index],
                            message=message,
                        )
                        for index, message in enumerate(copied_transcript)
                    ],
                    cursor=source_transcript_length,
                )
            )
            copied_transcript, copied_interaction_ids = apply_fork_system_prompt_replacement(
                copied_transcript,
                copied_interaction_ids,
                system_prompt_replacement,
            )
            # A cursor may intentionally exclude legacy secret-bearing suffix
            # entries. The source store owns those records; this frame does not.
            source_transcript = []
            if not fork_transcript_is_accepted(
                copied_transcript,
                source_transcript_snapshot,
                transcript_validator,
            ):
                copied_transcript.clear()
                copied_transcript = []
                source_transcript_snapshot = None
                raise ValueError(FORK_TRANSCRIPT_VALIDATION_ERROR) from None
            source_transcript_snapshot = None
            copied_checkpoint = None
            if checkpoint_transform is not None:
                checkpoint_input = (
                    None if stored_checkpoint is None else deepcopy(stored_checkpoint)
                )
                stored_checkpoint = None
                copied_checkpoint = transform_fork_checkpoint(
                    source_session.model_copy(deep=True),
                    checkpoint_input,
                    checkpoint_transform,
                )
                checkpoint_input = None
                if copied_checkpoint is not None:
                    copied_checkpoint = copy_durable_json_object(
                        copied_checkpoint,
                        "checkpoint",
                    )
            if profile_relationship is not None:
                copied_checkpoint = session_fork_rules._prepare_profiled_fork_checkpoint_result(
                    supports_model_failover=self._supports_model_failover_stage_protocol(),
                    fork=fork,
                    transcript_cursor=len(copied_transcript),
                    relationship=profile_relationship,
                    source_checkpoint_present=source_checkpoint_present,
                    copied_checkpoint=copied_checkpoint,
                )
            initial_operation_records = _prepare_initial_session_operation_records(
                fork,
                operation_initializer,
            )

            self._register_private_authority_alias_unlocked(
                fork.id,
                field_name="session_id",
            )
            self._sessions[fork.id] = fork.model_copy(deep=True)
            self._index_session_parent_unlocked(fork)
            self._events[fork.id] = []
            self._event_ids[fork.id] = set()
            self._session_event_records[fork.id] = []
            self._child_lifecycle_event_records[fork.id] = {}
            self._interaction_event_records[fork.id] = {}
            self._latest_interaction_event_records[fork.id] = {}
            self._latest_interaction_event_records_by_sequence[fork.id] = []
            self._pending_action_event_records[fork.id] = {}
            self._session_operation_records[fork.id] = initial_operation_records
            self._transcripts[fork.id] = copied_transcript
            self._replace_latest_transcript_role_indexes_unlocked(fork.id)
            self._replace_transcript_search_session_unlocked(fork.id, copied_transcript)
            # Historical attribution remains tied to the source session's
            # interactions; new child work receives new child interaction IDs.
            self._transcript_interaction_ids[fork.id] = copied_interaction_ids
            for interaction_id in set(self._transcript_interaction_ids[fork.id]):
                if interaction_id is None:
                    continue
                self._register_private_authority_alias_unlocked(
                    interaction_id,
                    field_name="interaction_id",
                    scope_session_id=fork.id,
                )
            if copied_checkpoint is not None:
                self._store_checkpoint_unlocked(fork.id, copied_checkpoint)
            if events:
                self._sessions[fork.id] = self._append_events_unlocked(
                    self._sessions[fork.id],
                    events,
                )
            else:
                self._refresh_child_lifecycle_candidate_unlocked(self._sessions[fork.id])
            return self._sessions[fork.id].model_copy(deep=True)

    async def load(self, session_id: str) -> Session | None:
        from cayu.resource_access import current_data_bounds

        access_bounds = await current_data_bounds()
        if access_bounds is not None:
            from cayu._resource_access_errors import ResourceAccessDenied

            try:
                return await self._access_load_session(access_bounds, session_id)
            except ResourceAccessDenied:
                return None
        session_id = require_clean_nonblank(session_id, "session_id")
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            return session.model_copy(deep=True)

    async def load_state(self, session_id: str) -> SessionStateSnapshot | None:
        session_id = require_clean_nonblank(session_id, "session_id")
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            return SessionStateSnapshot(
                id=session.id,
                status=session.status,
                updated_at=session.updated_at,
                last_activity_at=session.last_activity_at,
            )

    async def load_invocation_snapshot(
        self,
        session_id: str,
    ) -> SessionInvocationSnapshot | None:
        session_id = require_clean_nonblank(session_id, "session_id")
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            return SessionInvocationSnapshot(
                id=session.id,
                session_instance_id=session.instance_id,
                status=session.status,
                invocation=session.invocation,
            )

    async def inspect_identity(self, session_id: str) -> SessionInspectionIdentity:
        session_id = require_clean_nonblank(session_id, "session_id")
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(session_id)
            labels, label_count, labels_truncated = (
                session_inspection_rules._bounded_session_inspection_labels(session.labels)
            )
            return SessionInspectionIdentity(
                id=session.id,
                agent_name=session.agent_name,
                provider_name=session.provider_name,
                model=session.model,
                parent_session_id=session.parent_session_id,
                causal_budget_id=session.causal_budget_id,
                runtime_name=session.runtime_name,
                runtime_version=session.runtime_version,
                runtime_build_provenance=session.runtime_build_provenance,
                environment_name=session.environment_name,
                status=session.status,
                created_at=session.created_at,
                updated_at=session.updated_at,
                last_activity_at=session.last_activity_at,
                run_epoch=session.run_epoch,
                labels=labels,
                label_count=label_count,
                labels_truncated=labels_truncated,
            )

    async def update_status(self, session_id: str, status: SessionStatus) -> Session:
        session_id = require_clean_nonblank(session_id, "session_id")
        if not isinstance(status, SessionStatus):
            raise ValueError("Session status must be a SessionStatus.")
        # Delegate to the guarded transition machine with every source status
        # allowed: preserves the unconditional (any -> status) setter contract
        # while sharing one write path and its not-found guard.
        return await self.transition_status(
            session_id,
            from_statuses=set(SessionStatus),
            to_status=status,
        )

    def _require_session_erasure_quiescence_unlocked(self, session: Session) -> None:
        """Shared admission for closure and final deletion; no mutations."""
        self._require_external_wait_quiescence_unlocked(session)
        from cayu.collaboration import _session_export_store as session_exports
        from cayu.runtime._session_closure_records import require_terminal_protected_effect
        from cayu.sessions import _session_continuation_store as continuations

        session_id = session.id
        continuations.require_erasure_quiescence(
            session=session,
            checkpoint=self._checkpoints.get(session_id),
            records=self._session_operation_records.get(session_id, {}),
        )
        session_exports.require_erasure_quiescence(
            session=session,
            checkpoint=self._checkpoints.get(session_id),
            export_records={
                key: value
                for key, value in self._session_operation_records.get(session_id, {}).items()
                if key.startswith(session_exports.OPERATION_PREFIX)
            },
        )
        from cayu.runtime._producer_output_store import require_erasure_quiescence
        from cayu.sessions._producer_checkpoint import OPERATION_PREFIX

        require_erasure_quiescence(
            session=session,
            checkpoint=self._checkpoints.get(session_id),
            records={
                key: value
                for key, value in self._session_operation_records.get(session_id, {}).items()
                if key.startswith(OPERATION_PREFIX)
            },
        )
        for key, raw in self._session_operation_records.get(session_id, {}).items():
            if key.startswith("tool-effect:"):
                require_terminal_protected_effect(session_id, session.instance_id, key, raw)
        if session.status in DELETE_BLOCKED_SESSION_STATUSES:
            raise ValueError("Session closure requires a non-running target.")
        if any(
            delivery.session_id == session_id
            and delivery.status is PersistedEventSideEffectStatus.LEASED
            for delivery in self._persisted_event_side_effect_deliveries.values()
        ):
            raise ValueError("Session closure requires settled event side-effect deliveries.")
        checkpoint = self._checkpoints.get(session_id)
        deletion_now = self._ownership_clock()
        active_recovery_claim_id = (
            session_recovery_rules._active_unexpired_incomplete_recovery_claim_id(
                checkpoint,
                now=deletion_now,
            )
        )
        if active_recovery_claim_id is not None:
            raise ValueError(
                "Cannot delete a session while incomplete-session recovery claim "
                f"{active_recovery_claim_id} is active: {session_id}"
            )
        run_operation = terminal_event_evidence._session_run_operation_from_checkpoint(checkpoint)
        if run_operation is not None:
            raise ValueError(
                "Cannot delete a session while terminal publication "
                f"{run_operation.operation_id} is incomplete: {session_id}"
            )
        queued_terminal_receipts = _queued_dispatch_terminal_receipts_from_checkpoint(checkpoint)
        if queued_terminal_receipts:
            raise ValueError(
                "Cannot delete a session while queued dispatch terminal "
                f"acknowledgement is incomplete: {session_id}"
            )
        evidence_events: list[Event] = []
        for record in reversed(self._session_event_records.get(session_id, [])):
            if record.event.type not in _TERMINAL_PUBLICATION_EVIDENCE_EVENT_TYPES:
                continue
            evidence_events.append(record.event)
            if len(evidence_events) == _TERMINAL_PUBLICATION_EVIDENCE_QUERY_LIMIT:
                break
        terminal_publication_block = _terminal_publication_delete_block_reason(
            session=session,
            checkpoint=checkpoint,
            evidence_events=evidence_events,
        )
        if terminal_publication_block is not None:
            raise ValueError(
                f"Cannot delete a session while {terminal_publication_block}: {session_id}"
            )
        active_operation_id = _active_unexpired_session_operation_id(
            checkpoint,
            now=deletion_now,
        )
        if active_operation_id is not None:
            raise ValueError(
                "Cannot delete a session while durable operation "
                f"{active_operation_id} is active: {session_id}"
            )
        completion_result_publication_block = (
            checkpoint_preservation._completion_result_event_publication_delete_block_reason(
                checkpoint,
                now=deletion_now,
            )
        )
        if completion_result_publication_block is not None:
            raise ValueError(
                f"Cannot delete a session while {completion_result_publication_block}: {session_id}"
            )
        if MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY in self._session_operation_records.get(
            session_id, {}
        ):
            raise ValueError(
                f"Cannot delete a session while a model-completion stage is active: {session_id}"
            )
        owned_reservation_ids = {
            reservation_id
            for reservation_id, ownership in self._budget_reservation_identities.items()
            if ownership[0] == session_id
        }
        for reservation_id in owned_reservation_ids:
            terminal_events = [
                event
                for event in self._events.get(session_id, [])
                if event.type
                in {
                    EventType.BUDGET_RECONCILED,
                    EventType.BUDGET_RESERVATION_RELEASED,
                }
                and event.payload.get("reservation_id") == reservation_id
            ]
            delivery = (
                None
                if len(terminal_events) != 1
                else self._persisted_event_side_effect_deliveries.get(
                    (session_id, terminal_events[0].id)
                )
            )
            if delivery is None or delivery.status is not PersistedEventSideEffectStatus.DELIVERED:
                raise ValueError(
                    "Cannot delete a session while a budget settlement audit "
                    f"event is pending: {session_id}"
                )

    async def validate_session_closure_admission(self, session_id: str) -> None:
        session_id = require_clean_nonblank(session_id, "session_id")
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise ValueError("Closure target is unavailable.")
            self._require_session_erasure_quiescence_unlocked(session)

    @runtime_session_mutation
    async def delete_session(
        self,
        session_id: str,
        *,
        closure_receipt: dict[str, Any] | None = None,
        _access_bounds: _SessionAccessBounds | None = None,
    ) -> None:
        # Participant creation/replay acquires this lock before the store lock.
        # Deletion follows the same order so a committed session cannot lose
        # its receipt between the atomic create and readback.
        await self._participant_creation_lock.acquire()
        try:
            await self._delete_session_unserialized(
                session_id,
                closure_receipt=closure_receipt,
                _access_bounds=_access_bounds,
            )
        finally:
            self._participant_creation_lock.release()

    async def _delete_session_unserialized(
        self,
        session_id: str,
        *,
        closure_receipt: dict[str, Any] | None = None,
        _access_bounds: _SessionAccessBounds | None = None,
    ) -> None:
        session_id = require_clean_nonblank(session_id, "session_id")
        async with self._lock:
            session = self._sessions.get(session_id)
            if _access_bounds is not None:
                _access_bounds.require_action(session, "delete")
            if session is None:
                return  # idempotent: deleting a missing session is a no-op
            for owner in self._session_closure_progress.values():
                _check_closure_lineage_owner(owner, (session_id,), closure_receipt)
            if closure_receipt is not None and closure_receipt.get("operation") == "recursive":
                expected_parent = closure_receipt.get("original_parent_session_id")
                if type(expected_parent) is not str or session.parent_session_id != expected_parent:
                    raise ValueError("Recursive closure child parent identity conflict.")
            if session.status in DELETE_BLOCKED_SESSION_STATUSES:
                raise ValueError(
                    f"Cannot delete a session while it is {session.status}; "
                    f"interrupt it first: {session_id}"
                )
            if self.context_view_version is not None:
                now_ms = int(self._ownership_clock().timestamp() * 1000)
                for receipt in self._context_selection_receipts.values():
                    if (
                        receipt.view.source_session_id == session_id
                        and receipt.state in {"selected", "adopted", "transferred"}
                        and (receipt.state != "selected" or receipt.expires_at_ms > now_ms)
                    ):
                        raise ValueError("Session has an active context-view retention pin.")
            for _, child_id in self._child_session_keys_by_parent.get(session_id, []):
                child = self._sessions.get(child_id)
                if child is None or child.parent_session_id != session_id:
                    raise RuntimeError("Inconsistent in-memory session topology index.")
                for owner in self._session_closure_progress.values():
                    _check_closure_lineage_owner(owner, (child_id,))
                if closure_receipt is not None or _access_bounds is not None:
                    raise ValueError("Closure deletion requires no remaining child edges.")
                if _is_durable_subagent_child(child):
                    raise ValueError(_durable_subagent_parent_delete_block_reason(child.id))
            self._require_session_erasure_quiescence_unlocked(session)
            self._remove_session_parent_index_unlocked(session)
            targeted_grant_ids = {
                grant_id
                for grant_id, record in self._targeted_tool_grants.items()
                if record.session_id == session_id
            }
            self._targeted_tool_grants = {
                grant_id: record
                for grant_id, record in self._targeted_tool_grants.items()
                if grant_id not in targeted_grant_ids
            }
            self._targeted_tool_grant_ids_by_session.pop(session_id, None)
            self._targeted_tool_grant_ids_by_interaction = {
                key: grant_ids
                for key, grant_ids in self._targeted_tool_grant_ids_by_interaction.items()
                if key[0] != session_id
            }
            self._targeted_tool_grant_ids_by_request = {
                key: grant_id
                for key, grant_id in self._targeted_tool_grant_ids_by_request.items()
                if key[0] != session_id and grant_id not in targeted_grant_ids
            }
            self._targeted_tool_grant_ids_by_tool = {
                key: grant_id
                for key, grant_id in self._targeted_tool_grant_ids_by_tool.items()
                if key[0] != session_id and grant_id not in targeted_grant_ids
            }
            self._targeted_tool_grant_refs = {
                public_alias: target
                for public_alias, target in self._targeted_tool_grant_refs.items()
                if target[0] != session_id and target[1] not in targeted_grant_ids
            }
            for grant_id in targeted_grant_ids:
                self._targeted_tool_uses.pop(grant_id, None)
            self._targeted_tool_use_by_invocation = {
                key: binding
                for key, binding in self._targeted_tool_use_by_invocation.items()
                if key[0] != session_id and binding.grant_id not in targeted_grant_ids
            }
            self._targeted_tool_use_by_outer_call = {
                key: binding
                for key, binding in self._targeted_tool_use_by_outer_call.items()
                if key[0] != session_id and binding.grant_id not in targeted_grant_ids
            }
            self._sessions.pop(session_id, None)
            self._execution_owners.pop(session_id, None)
            if any(
                receipt.status == "appended"
                and receipt.target_session_id == session.id
                and receipt.target_session_instance_id == session.instance_id
                for receipt in self._peer_content_receipts.values()
            ):
                self._peer_content_deleted_targets.add((session.id, session.instance_id))
            binding = self._participant_session_bindings.pop(session_id, None)
            self._participant_session_receipts.pop(session_id, None)
            if binding is not None:
                self._participant_session_ids_by_key.pop(binding.creation_key, None)
                self._participant_session_request_commitments.pop(binding.creation_key, None)
            self._events.pop(session_id, None)
            self._event_ids.pop(session_id, None)
            self._persisted_event_side_effect_deliveries = {
                key: delivery
                for key, delivery in self._persisted_event_side_effect_deliveries.items()
                if key[0] != session_id
            }
            self._session_event_records.pop(session_id, None)
            self._interaction_event_records.pop(session_id, None)
            self._latest_interaction_event_records.pop(session_id, None)
            self._latest_interaction_event_records_by_sequence.pop(session_id, None)
            self._pending_action_event_records.pop(session_id, None)
            self._pending_action_index_scopes.pop(session_id, None)
            self._pending_action_latest_barrier_records.pop(session_id, None)
            self._user_input_supersession_event_records = {
                key: records
                for key, records in self._user_input_supersession_event_records.items()
                if key[0] != session_id
            }
            self._remove_transcript_search_session_unlocked(session_id)
            self._transcripts.pop(session_id, None)
            self._transcript_interaction_ids.pop(session_id, None)
            self._latest_transcript_indexes_by_role.pop(session_id, None)
            self._transcript_indices_by_interaction.pop(session_id, None)
            self._child_lifecycle_event_records.pop(session_id, None)
            self._deferred_interaction_inputs.pop(session_id, None)
            self._refresh_queued_dispatch_terminal_receipts_unlocked(session_id, None)
            self._checkpoints.pop(session_id, None)
            receipt_ids = self._recall_receipt_ids_by_session.pop(session_id, set())
            exposure_ids = self._context_exposure_ids_by_session.pop(session_id, set())
            self._recall_receipt_page_keys_by_session.pop(session_id, None)
            self._context_exposure_page_keys_by_session.pop(session_id, None)
            for receipt_id in receipt_ids:
                receipt = self._recall_receipts.pop(receipt_id)
                self._recall_receipt_page_keys_by_interaction.pop(
                    (session_id, receipt.interaction_id),
                    None,
                )
                self._recall_receipt_page_keys_by_step.pop(
                    (session_id, receipt.model_step_id),
                    None,
                )
                self._recall_receipt_page_keys_by_interaction_step.pop(
                    (session_id, receipt.interaction_id, receipt.model_step_id),
                    None,
                )
            for exposure_id in exposure_ids:
                exposure = self._context_exposures.pop(exposure_id)
                self._context_exposure_page_keys_by_interaction.pop(
                    (session_id, exposure.interaction_id),
                    None,
                )
                self._context_exposure_page_keys_by_step.pop(
                    (session_id, exposure.model_step_id),
                    None,
                )
                self._context_exposure_page_keys_by_interaction_step.pop(
                    (session_id, exposure.interaction_id, exposure.model_step_id),
                    None,
                )
                self._recall_item_exposures.pop(exposure_id, None)
                self._context_exposure_by_model_attempt.pop(
                    (session_id, exposure.model_attempt_id),
                    None,
                )
                self._context_exposure_by_provider_attempt.pop(
                    (session_id, exposure.provider_attempt_id),
                    None,
                )
            self._session_operation_records.pop(session_id, None)
            self._pending_action_session_ids.discard(session_id)
            self._queued_session_messages_by_idempotency.pop(session_id, None)
            self._session_message_acceptance_event_ids = {
                key: ids
                for key, ids in self._session_message_acceptance_event_ids.items()
                if key[0] != session_id
            }
            self._session_message_terminal_receipts = {
                key: receipt
                for key, receipt in self._session_message_terminal_receipts.items()
                if key[0] != session_id
            }
            self._session_message_delivery_records = {
                delivery_id: record
                for delivery_id, record in self._session_message_delivery_records.items()
                if record.session_id != session_id
            }
            for delivery_mode in SessionMessageDeliveryMode:
                self._pending_session_messages.pop((session_id, delivery_mode), None)
            self._event_records = [
                record for record in self._event_records if record.event.session_id != session_id
            ]
            self._event_records_by_id = {
                key: record
                for key, record in self._event_records_by_id.items()
                if key[0] != session_id
            }
            self._accounting_generation += 1
            self._cost_event_index.remove_session(session_id)

            self._workflow_step_event_records = {
                key: records
                for key, records in self._workflow_step_event_records.items()
                if key[0] != session_id
            }
            self._workflow_fenced_step_event_records = {
                key: records
                for key, records in self._workflow_fenced_step_event_records.items()
                if key[0] != session_id
            }
            self._workflow_attempt_event_records = {
                key: records
                for key, records in self._workflow_attempt_event_records.items()
                if key[0] != session_id
            }
            self._workflow_active_attempt_ids = {
                key: attempt_id
                for key, attempt_id in self._workflow_active_attempt_ids.items()
                if key[0] != session_id
            }
            for event_type, records in list(self._type_event_records.items()):
                remaining = [record for record in records if record.event.session_id != session_id]
                if remaining:
                    self._type_event_records[event_type] = remaining
                else:
                    del self._type_event_records[event_type]
            # Mirror the SQL FK's ON DELETE SET NULL: children keep loading, parent ref
            # cleared. The branch index makes this proportional to the deleted
            # session's direct children rather than the complete registry.
            child_keys = self._child_session_keys_by_parent.pop(session_id, [])
            candidate_priorities = self._child_lifecycle_candidate_keys_by_parent.pop(
                session_id,
                {},
            )
            indexed_candidate_ids = {
                child_id for keys in candidate_priorities.values() for _sort_at, child_id in keys
            }
            if indexed_candidate_ids != {child_id for _created_at, child_id in child_keys}:
                raise RuntimeError("Inconsistent in-memory child lifecycle candidate branch.")
            for child_id in indexed_candidate_ids:
                self._child_lifecycle_candidate_key_by_child.pop(child_id, None)
            for _, child_id in child_keys:
                child = self._sessions.get(child_id)
                if child is None or child.parent_session_id != session_id:
                    raise RuntimeError("Inconsistent in-memory session topology index.")
                self._sessions[child_id] = child.model_copy(update={"parent_session_id": None})
            if closure_receipt is not None:
                receipt_plan_id = closure_receipt.get("plan_id")
                if type(receipt_plan_id) is not str:
                    raise ValueError("Session closure receipt is missing plan identity.")
                receipt_key = (session_id, receipt_plan_id)
                existing = self._session_closure_receipts.get(receipt_key)
                if existing is not None and existing != closure_receipt:
                    raise ValueError("Session closure receipt identity conflict.")
                self._session_closure_receipts[receipt_key] = deepcopy(closure_receipt)

    @runtime_session_mutation
    async def update_labels(
        self,
        session_id: str,
        labels: dict[str, str],
        *,
        _access_bounds: _SessionAccessBounds | None = None,
    ) -> Session:
        session_id = require_clean_nonblank(session_id, "session_id")
        new_labels = copy_label_map(labels, "labels", allow_reserved=False)
        async with self._lock:
            session = self._sessions.get(session_id)
            if _access_bounds is not None:
                _access_bounds.require_label_update(session, new_labels)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            invocation_scope._assert_session_run_epoch(session_id, session)
            now = self._ownership_clock()
            for owner in self._session_closure_progress.values():
                _check_closure_lineage_owner(owner, (session_id,))
            updated = session.model_copy(update={"labels": new_labels, "updated_at": now})
            from cayu.sessions._invocation_lifecycle import (
                require_invocation_lifecycle_release_capacity,
            )

            require_invocation_lifecycle_release_capacity(
                self._checkpoints.get(session_id),
                updated,
            )
            if _access_bounds is not None:
                audit = _access_bounds.label_audit(session, new_labels, now)
                if audit is not None:
                    key, record = audit
                    self._session_operation_records.setdefault(session_id, {})[key] = record
            self._sessions[session_id] = updated
            return updated.model_copy(deep=True)

    @runtime_session_mutation
    async def update_metadata(
        self,
        session_id: str,
        metadata: dict[str, Any],
        *,
        _access_bounds: _SessionAccessBounds | None = None,
    ) -> Session:
        session_id = require_clean_nonblank(session_id, "session_id")
        user_metadata = copy_session_user_metadata(metadata)
        async with self._lock:
            session = self._sessions.get(session_id)
            if _access_bounds is not None:
                _access_bounds.require_action(session, "modify")
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            invocation_scope._assert_session_run_epoch(session_id, session)
            for owner in self._session_closure_progress.values():
                _check_closure_lineage_owner(owner, (session_id,))
            new_metadata = replace_session_user_metadata(session.metadata, user_metadata)
            now = self._ownership_clock()
            updated = session.model_copy(update={"metadata": new_metadata, "updated_at": now})
            from cayu.sessions._invocation_lifecycle import (
                require_invocation_lifecycle_release_capacity,
            )

            require_invocation_lifecycle_release_capacity(
                self._checkpoints.get(session_id),
                updated,
            )
            self._sessions[session_id] = updated
            return updated.model_copy(deep=True)

    async def transition_status(
        self,
        session_id: str,
        *,
        from_statuses: set[SessionStatus],
        to_status: SessionStatus,
    ) -> Session:
        session_id = require_clean_nonblank(session_id, "session_id")
        allowed_statuses = session_record_rules._validate_status_set(from_statuses, "from_statuses")
        if not isinstance(to_status, SessionStatus):
            raise ValueError("to_status must be a SessionStatus.")
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            invocation_scope._assert_session_run_epoch(session_id, session)
            for owner in self._session_closure_progress.values():
                _check_closure_lineage_owner(owner, (session_id,))
            if session.status not in allowed_statuses:
                raise SessionStatusConflict(
                    f"Session status transition not allowed: {session.status} -> {to_status}"
                )
            if to_status is SessionStatus.RUNNING:
                self._require_external_wait_admission_unlocked(session)

            now = self._ownership_clock()
            if to_status is SessionStatus.RUNNING:
                session_recovery_rules._require_live_incomplete_recovery_claim_for_run_epoch_transfer(
                    self._checkpoints.get(session_id),
                    now=now,
                )
            updated = session.model_copy(
                update={
                    "status": to_status,
                    "updated_at": now,
                    "last_activity_at": now,
                    "run_epoch": session.run_epoch + (to_status == SessionStatus.RUNNING),
                }
            )
            self._sessions[session_id] = updated
            self._refresh_child_lifecycle_candidate_unlocked(updated)
            result = updated.model_copy(deep=True)
            if to_status == SessionStatus.RUNNING:
                invocation_scope._activate_session_run_fence(result)
            return result

    async def reject_execution_profile_resume(
        self,
        session_id: str,
        *,
        expected_session_instance_id: str | None = None,
        expected_statuses: set[SessionStatus],
        expected_run_epoch: int,
        expected_profile: ExecutionProfileIdentity,
        candidate_profile: ExecutionProfileIdentity,
        event: Event,
        decision: ExecutionProfileDecision | None = None,
        expected_active_invocation_profile_authority: CheckpointValueAuthority | None = None,
    ) -> ExecutionProfileRejectionResult:
        (
            session_id,
            statuses,
            expected_run_epoch,
            expected_profile,
            _candidate_profile,
            copied_event,
        ) = session_profile_rules._prepare_execution_profile_rejection(
            session_id,
            expected_statuses=expected_statuses,
            expected_run_epoch=expected_run_epoch,
            expected_profile=expected_profile,
            candidate_profile=candidate_profile,
            event=event,
            decision=decision,
        )
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            session_profile_rules._validate_execution_profile_rejection_session(
                session,
                checkpoint=self._checkpoints.get(session_id),
                expected_session_instance_id=expected_session_instance_id,
                expected_statuses=statuses,
                expected_run_epoch=expected_run_epoch,
                expected_profile=expected_profile,
                event=copied_event,
                expected_active_invocation_profile_authority=(
                    expected_active_invocation_profile_authority
                ),
            )
            for existing in self._events.get(session_id, []):
                if existing.id != copied_event.id:
                    continue
                if not session_profile_rules._execution_profile_rejection_events_equivalent(
                    existing,
                    copied_event,
                ):
                    raise ValueError(
                        f"Execution-profile rejection id was reused: {copied_event.id}"
                    )
                return ExecutionProfileRejectionResult(event=existing, replayed=True)
            for owner in self._session_closure_progress.values():
                _check_closure_lineage_owner(owner, (session_id,))
            self._sessions[session_id] = self._append_events_unlocked(
                session,
                [copied_event],
            )
            return ExecutionProfileRejectionResult(event=copied_event, replayed=False)

    async def transition_status_and_checkpoint(
        self,
        session_id: str,
        *,
        from_statuses: set[SessionStatus],
        to_status: SessionStatus,
        checkpoint_transform: CheckpointTransform | None = None,
        store_time_checkpoint_transform: StoreTimeCheckpointTransform | None = None,
        result_checkpoint_transform: CheckpointTransform | None = None,
        interaction_started_event: Event | None = None,
        interaction_source_messages: list[Message] | None = None,
        continued_interaction_id: str | None = None,
        defer_interaction_source: bool = False,
        model_transition: SessionModelTransition | None = None,
        execution_profile: ExecutionProfileIdentity | None = None,
        execution_profile_decision: ExecutionProfileDecision | None = None,
        adopted_runtime_identity: SessionRuntimeIdentity | None = None,
        tool_capability_ceiling: ToolCapabilityCeiling | None = None,
        expected_latest_interaction_event_id: str | None = None,
        require_no_active_model_completion_dispatch: bool = False,
        temporary_service_admission: TemporaryServiceAdmission | None = None,
    ) -> Session:
        from cayu.sessions._temporary_continuation_scope import prepare_temporary_transition

        temporary_service_admission = prepare_temporary_transition(temporary_service_admission)
        session_id = require_clean_nonblank(session_id, "session_id")
        allowed_statuses = session_record_rules._validate_status_set(from_statuses, "from_statuses")
        if not isinstance(to_status, SessionStatus):
            raise ValueError("to_status must be a SessionStatus.")
        if (checkpoint_transform is None) == (store_time_checkpoint_transform is None):
            raise TypeError("Exactly one checkpoint transform is required.")
        if result_checkpoint_transform is not None and not callable(result_checkpoint_transform):
            raise TypeError("result_checkpoint_transform must be callable.")
        admission = interaction_rules._copy_transition_interaction_admission(
            session_id,
            interaction_started_event,
            interaction_source_messages,
            continued_interaction_id=continued_interaction_id,
            defer_interaction_source=defer_interaction_source,
        )
        prepared_model_transition = session_profile_rules._copy_session_model_transition(
            session_id,
            model_transition,
            interaction_id=(None if admission is None else admission[1]),
            interaction_is_new=(admission is not None and admission[0] is not None),
        )
        prepared_execution_profile = session_profile_rules._copy_optional_execution_profile(
            execution_profile
        )
        prepared_execution_profile_decision = (
            session_profile_rules._copy_optional_execution_profile_decision(
                execution_profile_decision
            )
        )
        prepared_adopted_runtime_identity = (
            None
            if adopted_runtime_identity is None
            else copy_session_runtime_identity(adopted_runtime_identity)
        )
        prepared_tool_capability_ceiling = (
            session_request_contracts._copy_optional_tool_capability_ceiling(
                tool_capability_ceiling
            )
        )
        expected_latest_interaction_event_id = event_delivery_rules._copy_optional_event_id(
            expected_latest_interaction_event_id,
            "expected_latest_interaction_event_id",
        )
        if type(require_no_active_model_completion_dispatch) is not bool:
            raise TypeError("require_no_active_model_completion_dispatch must be a boolean.")
        if prepared_execution_profile_decision is not None and admission is None:
            raise ValueError("An execution-profile decision requires atomic interaction admission.")
        if admission is not None and to_status is not SessionStatus.RUNNING:
            raise ValueError("Interaction admission requires a transition to running.")
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            invocation_scope._assert_session_run_epoch(session_id, session)
            for owner in self._session_closure_progress.values():
                _check_closure_lineage_owner(owner, (session_id,))
            if session.status not in allowed_statuses:
                raise SessionStatusConflict(
                    f"Session status transition not allowed: {session.status} -> {to_status}"
                )
            if to_status is SessionStatus.RUNNING:
                self._require_external_wait_admission_unlocked(session)
            if expected_latest_interaction_event_id is not None:
                latest_interactions = self._latest_interaction_event_records_by_sequence.get(
                    session_id,
                    [],
                )
                latest_interaction = latest_interactions[-1] if latest_interactions else None
                if (
                    latest_interaction is None
                    or latest_interaction.event.id != expected_latest_interaction_event_id
                ):
                    raise SessionRunFenced(
                        "Session latest interaction changed before the status transition."
                    )
            if require_no_active_model_completion_dispatch:
                operation_records = self._session_operation_records.get(session_id, {})
                active_record = operation_records.get(MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY)
                if active_record is not None:
                    active_marker = (
                        model_stage_rules._reconstruct_active_model_completion_stage_record(
                            active_record,
                            session_id=session_id,
                        )
                    )
                    dispatch_key = model_stage_rules._model_completion_stage_dispatch_storage_key(
                        active_marker.stage_id
                    )
                    if operation_records.get(dispatch_key) is not None:
                        raise SessionModelCompletionDispatchAlreadyAuthorized(
                            "Provider dispatch was authorized before terminal decision election."
                        )
            transition_profile_metadata = (
                session_profile_rules._validate_execution_profile_admission(
                    session,
                    candidate_profile=prepared_execution_profile,
                    model_transition=prepared_model_transition,
                    decision=prepared_execution_profile_decision,
                )
            )

            if session.status == SessionStatus.PENDING and admission is not None and admission[3]:
                receipt = self._participant_session_receipts.get(session_id)
                if receipt is not None:
                    from cayu.sessions._participant_execution_identity import (
                        require_initial_execution_input,
                    )

                    require_initial_execution_input(
                        session, receipt, self._transcripts.get(session_id, []), admission[2]
                    )

            if prepared_model_transition is not None:
                session_profile_rules._validate_session_model_transition(
                    session,
                    self._transcripts.get(session_id, []),
                    len(self._transcripts.get(session_id, [])),
                    prepared_model_transition,
                )

            now = self._ownership_clock()
            if to_status is SessionStatus.RUNNING:
                session_recovery_rules._require_live_incomplete_recovery_claim_for_run_epoch_transfer(
                    self._checkpoints.get(session_id),
                    now=now,
                )
            session_updates: dict[str, Any] = {
                "status": to_status,
                "updated_at": now,
                "last_activity_at": now,
                "run_epoch": session.run_epoch + (to_status == SessionStatus.RUNNING),
            }
            transition_metadata = transition_profile_metadata
            if prepared_model_transition is not None:
                transition_metadata = (
                    session_profile_rules._session_metadata_after_model_transition(
                        session,
                        prepared_model_transition,
                        execution_profile_metadata=transition_profile_metadata,
                    )
                )
                session_updates.update(
                    provider_name=prepared_model_transition.target.provider_name,
                    model=prepared_model_transition.target.model,
                )
            transition_metadata = (
                session_profile_rules._session_metadata_after_runtime_identity_adoption(
                    session,
                    prepared_adopted_runtime_identity,
                    model_transition=prepared_model_transition,
                    execution_profile_metadata=transition_metadata,
                )
            )
            if prepared_adopted_runtime_identity is not None:
                session_updates.update(
                    runtime_name=prepared_adopted_runtime_identity.runtime_name,
                    runtime_version=prepared_adopted_runtime_identity.runtime_version,
                )
            transition_metadata = (
                session_profile_rules._session_metadata_after_tool_capability_ceiling_admission(
                    session,
                    prepared_tool_capability_ceiling,
                    transition_metadata=transition_metadata,
                    require_existing_ceiling=prepared_execution_profile is not None,
                )
            )

            current_checkpoint = self._checkpoints.get(session_id)
            checkpoint_copy = checkpoint_preservation._copy_checkpoint_for_transform(
                current_checkpoint,
                session_id=session_id,
            )
            if store_time_checkpoint_transform is not None:
                transformed_checkpoint = store_time_checkpoint_transform(
                    session.model_copy(deep=True),
                    checkpoint_copy,
                    now,
                )
            else:
                assert checkpoint_transform is not None
                transformed_checkpoint = checkpoint_transform(
                    session.model_copy(deep=True),
                    checkpoint_copy,
                )
            if transformed_checkpoint is not None:
                transformed_checkpoint = checkpoint_preservation._checkpoint_transform_result_preserving_completion_result_event_publications(
                    current_checkpoint,
                    transformed_checkpoint,
                    session_id=session_id,
                )

            if admission is not None:
                _, interaction_id, _, defer_source = admission
                existing_deferred = self._deferred_interaction_inputs.get(session_id)
                if existing_deferred is not None and (
                    not defer_source or existing_deferred.interaction_id != interaction_id
                ):
                    raise RuntimeError("Session already has deferred interaction input.")
            if transition_metadata is not None:
                session_updates["metadata"] = transition_metadata
            updated = session.model_copy(update=session_updates)
            transformed_checkpoint = (
                model_failover_rules._model_failover_checkpoint_after_profile_admission(
                    source_session=session,
                    admitted_session=updated,
                    source_checkpoint=current_checkpoint,
                    admitted_checkpoint=transformed_checkpoint,
                    candidate_profile=prepared_execution_profile,
                    records=self._session_operation_records.get(session_id, {}),
                    transcript_cursor=len(self._transcripts.get(session_id, [])),
                    supports_model_failover=self._supports_model_failover_stage_protocol(),
                )
            )
            if result_checkpoint_transform is not None:
                result_checkpoint = result_checkpoint_transform(
                    updated.model_copy(deep=True),
                    checkpoint_preservation._copy_checkpoint_for_transform(
                        transformed_checkpoint,
                        session_id=session_id,
                    ),
                )
                if result_checkpoint is None:
                    raise ValueError("Result checkpoint transform must return a checkpoint.")
                transformed_checkpoint = checkpoint_preservation._checkpoint_transform_result_preserving_completion_result_event_publications(
                    transformed_checkpoint,
                    result_checkpoint,
                    session_id=session_id,
                )
            temporary_records: dict[str, dict[str, Any]] = {}
            if temporary_service_admission is not None:
                from cayu.sessions._session_continuation import continuation_operation_key
                from cayu.sessions._temporary_continuation import temporary_service_key
                from cayu.sessions._temporary_continuation_store import (
                    compose_temporary_service_admission,
                )

                parent_key = continuation_operation_key(
                    temporary_service_admission.dispatch.intent.ticket
                )
                from cayu.sessions._temporary_service_target import target_service_key

                intent = temporary_service_admission.dispatch.intent
                child_key = (
                    temporary_service_key(intent.operation)
                    if intent.mode == "same_session"
                    else target_service_key(intent.operation)
                )
                publication = compose_temporary_service_admission(
                    source_session=session,
                    source_checkpoint=current_checkpoint,
                    parent_record=self._session_operation_records.get(session_id, {}).get(
                        parent_key
                    ),
                    child_record=self._session_operation_records.get(session_id, {}).get(child_key),
                    admitted_session=updated,
                    admitted_checkpoint=transformed_checkpoint,
                    admission=temporary_service_admission,
                    now=now,
                )
                transformed_checkpoint = publication.checkpoint
                temporary_records = publication.operation_records
            prepared_events: _PreparedInMemoryEventAppend | None = None
            admission_events = []
            if prepared_execution_profile_decision is not None:
                admission_events.append(prepared_execution_profile_decision.event)
            if prepared_model_transition is not None:
                admission_events.append(prepared_model_transition.event)
            if admission is not None and admission[0] is not None:
                admission_events.append(admission[0])
            if admission_events:
                prepared_events = self._prepare_event_append_unlocked(
                    updated,
                    admission_events,
                )
            prepared_checkpoint = (
                None
                if transformed_checkpoint is None
                else self._prepare_checkpoint_store_unlocked(
                    session_id,
                    transformed_checkpoint,
                    additional_event_records=(
                        ()
                        if prepared_events is None
                        else tuple(event.record for event in prepared_events.events)
                    ),
                )
            )

            self._sessions[session_id] = updated
            self._refresh_child_lifecycle_candidate_unlocked(updated)
            if prepared_checkpoint is not None:
                self._apply_checkpoint_store_unlocked(session_id, prepared_checkpoint)
            if temporary_records:
                self._session_operation_records.setdefault(session_id, {}).update(temporary_records)
            if admission is not None:
                _, interaction_id, source_messages, defer_source = admission
                if prepared_events is not None:
                    self._sessions[session_id] = self._apply_event_append_unlocked(
                        updated,
                        prepared_events,
                        activity_at=now,
                    )
                if defer_source:
                    self._deferred_interaction_inputs[session_id] = DeferredInteractionInput(
                        interaction_id=interaction_id,
                        source_messages=source_messages,
                    )
                else:
                    self._transcripts[session_id].extend(source_messages)
                    self._extend_transcript_search_unlocked(session_id, source_messages)
                    self._extend_transcript_attribution_unlocked(
                        session_id, [interaction_id] * len(source_messages)
                    )
                updated = self._sessions[session_id]
            result = updated.model_copy(deep=True)
            if to_status == SessionStatus.RUNNING:
                invocation_scope._activate_session_run_fence(result)
            return result

    async def transition_status_if_no_queued_messages(
        self,
        session_id: str,
        *,
        from_statuses: set[SessionStatus],
        to_status: SessionStatus,
        checkpoint_mutation: dict[str, Any] | None = None,
    ) -> Session:
        session_id = require_clean_nonblank(session_id, "session_id")
        allowed_statuses = session_record_rules._validate_status_set(from_statuses, "from_statuses")
        if not isinstance(to_status, SessionStatus):
            raise ValueError("to_status must be a SessionStatus.")
        mutation = _prepare_queue_completion_checkpoint_mutation(checkpoint_mutation, to_status)
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            invocation_scope._assert_session_run_epoch(session_id, session)
            if session.status not in allowed_statuses:
                raise SessionStatusConflict(
                    f"Session status transition not allowed: {session.status} -> {to_status}"
                )
            if any(
                (session_id, delivery_mode) in self._pending_session_messages
                for delivery_mode in SessionMessageDeliveryMode
            ):
                raise SessionQueuedMessagesPending(
                    f"Session has durable queued messages: {session_id}"
                )
            for owner in self._session_closure_progress.values():
                _check_closure_lineage_owner(owner, (session_id,))
            now = self._ownership_clock()
            checkpoint = (
                None
                if mutation is None
                else _apply_queue_completion_checkpoint_mutation(
                    session, mutation, self._checkpoints.get(session_id)
                )
            )
            updated = session.model_copy(
                update={
                    "status": to_status,
                    "updated_at": now,
                    "last_activity_at": now,
                    "run_epoch": session.run_epoch
                    + (to_status == SessionStatus.RUNNING and mutation is None),
                }
            )
            self._sessions[session_id] = updated
            if checkpoint is not None:
                self._store_checkpoint_unlocked(session_id, checkpoint)
            self._refresh_child_lifecycle_candidate_unlocked(updated)
            result = updated.model_copy(deep=True)
            if to_status == SessionStatus.RUNNING:
                invocation_scope._activate_session_run_fence(result)
            return result

    async def publish_interaction_transition(
        self,
        session_id: str,
        *,
        event: Event,
        from_statuses: set[SessionStatus],
        to_status: SessionStatus,
        only_if_no_queued_messages: bool = False,
        model_completion_stage_settlement: ModelCompletionStageSettlementRequest | None = None,
        checkpoint_mutation: dict[str, Any] | None = None,
        terminal_event: Event | None = None,
        terminal_decision: InvocationTerminalDecision | None = None,
        expected_session_instance_id: str | None = None,
        expected_active_invocation_profile: ActiveInvocationExecutionProfile | None = None,
        expected_invocation_authority_state: Literal["active", "released"] = "active",
        expected_recovery_claim_id: str | None = None,
        terminalization_only: bool = False,
        terminalization_plan_ownership: Any = None,
    ) -> InteractionTransitionResult:
        expected_recovery_claim_id = (
            interaction_receipts._validate_interaction_transition_recovery_claim_id(
                expected_recovery_claim_id
            )
        )
        expected_invocation_authority_state = (
            interaction_receipts._validate_interaction_transition_invocation_authority_parameters(
                expected_session_instance_id=expected_session_instance_id,
                expected_active_invocation_profile=expected_active_invocation_profile,
                expected_invocation_authority_state=expected_invocation_authority_state,
            )
        )
        session_id, transition = interaction_rules._prepare_interaction_transition(
            session_id,
            event=event,
            from_statuses=from_statuses,
            to_status=to_status,
            only_if_no_queued_messages=only_if_no_queued_messages,
            model_completion_stage_settlement=model_completion_stage_settlement,
            checkpoint_mutation=checkpoint_mutation,
            terminal_event=terminal_event,
            terminal_decision=terminal_decision,
        )
        copied_event = transition.event
        allowed_statuses = set(transition.from_statuses)
        target_status = transition.to_status
        conditional = transition.only_if_no_queued_messages
        settlement_request = transition.model_completion_stage_settlement
        checkpoint_mutation_request = transition.checkpoint_mutation
        copied_terminal_event = transition.terminal_event
        terminal_decision = transition.terminal_decision
        receipt_storage_key = interaction_receipts._interaction_transition_storage_key(
            copied_event.id
        )
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            if expected_active_invocation_profile is None:
                invocation_scope._assert_session_run_epoch(session_id, session)
            operation_records = self._session_operation_records.setdefault(session_id, {})
            receipt_record = operation_records.get(receipt_storage_key)
            existing = self._event_records_by_id.get((session_id, copied_event.id))
            existing_terminal = (
                None
                if copied_terminal_event is None
                else self._event_records_by_id.get((session_id, copied_terminal_event.id))
            )
            if receipt_record is not None:
                receipt = interaction_receipts._reconstruct_interaction_transition_receipt(
                    receipt_record,
                    transition=transition,
                )
                interaction_receipts._validate_interaction_transition_receipt_authority(
                    receipt,
                    current_session=session,
                    current_checkpoint=self._checkpoints.get(session_id),
                    expected_session_instance_id=expected_session_instance_id,
                    expected_active_invocation_profile=expected_active_invocation_profile,
                    expected_invocation_authority_state=(expected_invocation_authority_state),
                    expected_recovery_claim_id=expected_recovery_claim_id,
                )
                if existing is not None and existing.event != receipt.event:
                    raise RuntimeError(
                        "Interaction transition receipt conflicts with retained event history."
                    )
                if copied_terminal_event is not None and (
                    receipt.terminal_event != copied_terminal_event
                    or (
                        existing_terminal is not None
                        and existing_terminal.event != receipt.terminal_event
                    )
                ):
                    raise RuntimeError(
                        "Interaction transition receipt conflicts with its terminal session event."
                    )
                return InteractionTransitionResult(
                    session=receipt.session,
                    event=receipt.event,
                    terminal_event=receipt.terminal_event,
                    status_changed=receipt.status_changed,
                    replayed=True,
                )
            if existing is not None:
                raise RuntimeError(
                    "Interaction transition event exists without its immutable receipt."
                )
            if existing_terminal is not None:
                raise RuntimeError("Terminal session event exists without its interaction receipt.")
            for owner in self._session_closure_progress.values():
                _check_closure_lineage_owner(owner, (session_id,))
            current_checkpoint = self._checkpoints.get(session_id)
            if terminalization_only:
                from cayu.runtime._durable_model_terminalization import (
                    require_terminalization_checkpoint,
                )
                from cayu.sessions._model_stage_rules import require_terminalization_plan_owner

                require_terminalization_checkpoint(session, current_checkpoint)

                require_terminalization_plan_owner(
                    current_checkpoint, terminalization_plan_ownership, self._ownership_clock()
                )
                if (
                    self._child_session_keys_by_parent.get(session_id)
                    or self._deferred_interaction_inputs.get(session_id)
                    or any(
                        (session_id, mode) in self._pending_session_messages
                        for mode in SessionMessageDeliveryMode
                    )
                ):
                    raise SessionRunFenced("Model terminalization has dependent work.")
            settled_checkpoint = (
                interaction_receipts._checkpoint_after_exact_invocation_terminal_decision(
                    current_checkpoint,
                    session=session,
                    expected=terminal_decision,
                )
            )
            active_recovery_claim_id = (
                session_recovery_rules._active_unexpired_incomplete_recovery_claim_id(
                    current_checkpoint,
                    now=self._ownership_clock(),
                )
            )
            if expected_recovery_claim_id is None:
                if active_recovery_claim_id is not None:
                    raise SessionRunFenced(
                        "Interaction transition is owned by another terminal recovery claim."
                    )
            elif active_recovery_claim_id != expected_recovery_claim_id:
                raise SessionRunFenced(
                    "Interaction transition lost its exact terminal recovery claim."
                )
            if expected_active_invocation_profile is not None:
                from cayu.sessions._invocation_lifecycle import (
                    require_invocation_command_authority,
                    require_released_invocation_command_authority,
                )

                assert expected_session_instance_id is not None
                if expected_invocation_authority_state == "released":
                    require_released_invocation_command_authority(
                        session,
                        current_checkpoint,
                        session_id=session_id,
                        session_instance_id=expected_session_instance_id,
                        active_profile=expected_active_invocation_profile,
                        events=tuple(
                            event
                            for event in (copied_event, copied_terminal_event)
                            if event is not None
                        ),
                    )
                else:
                    require_invocation_command_authority(
                        session,
                        current_checkpoint,
                        session_id=session_id,
                        session_instance_id=expected_session_instance_id,
                        run_epochs=frozenset({expected_active_invocation_profile.run_epoch}),
                        active_profile=expected_active_invocation_profile,
                        events=tuple(
                            event
                            for event in (copied_event, copied_terminal_event)
                            if event is not None
                        ),
                    )
            if session.status not in allowed_statuses:
                raise SessionStatusConflict(
                    f"Session status transition not allowed: {session.status} -> {target_status}"
                )
            queued = conditional and any(
                (session_id, delivery_mode) in self._pending_session_messages
                for delivery_mode in SessionMessageDeliveryMode
            )
            from cayu.runtime._session_steering import (
                interaction_completion_steering_key,
                prepare_interaction_completion_steering_record,
            )

            steering_key = interaction_completion_steering_key(
                session, current_checkpoint, copied_event
            )
            completion_record = None
            if steering_key is not None:
                completion_record = prepare_interaction_completion_steering_record(
                    session,
                    current_checkpoint,
                    copied_event,
                    operation_records.get(steering_key),
                    keeps_running=queued or target_status is SessionStatus.RUNNING,
                )
            now = self._ownership_clock()
            settlement_record = None
            settlement_storage_key = None
            if settlement_request is not None:
                active = model_stage_rules._reconstruct_active_model_completion_stage_from_records(
                    operation_records,
                    session_id=session_id,
                )
                if active is None:
                    raise SessionModelCompletionStageConflict(
                        "The interaction transition has no active model-completion stage to settle."
                    )
                stage = active.stage
                settlement_storage_key = (
                    model_stage_rules._model_completion_stage_settlement_storage_key(stage.stage_id)
                )
                model_stage_rules._validate_model_completion_stage_for_settlement(
                    session=session,
                    stage=stage,
                    active=active,
                    request=settlement_request,
                    settlement_record=operation_records.get(settlement_storage_key),
                    winner_exists=(
                        operation_records.get(
                            model_stage_rules._model_completion_stage_winner_storage_key(
                                stage.logical_step_id
                            )
                        )
                        is not None
                    ),
                    receipt_exists=(
                        operation_records.get(
                            publication_contracts._runtime_publication_storage_key(
                                stage.logical_step_id
                            )
                        )
                        is not None
                    ),
                )
                settlement_record = model_stage_rules._model_completion_stage_settlement_record(
                    stage,
                    request=settlement_request,
                    settled_at=now,
                )
            prepared_event_append = self._prepare_event_append_unlocked(
                session,
                [event for event in (copied_event, copied_terminal_event) if event is not None],
            )
            updated_checkpoint = self._checkpoints.get(session_id)
            if not queued and checkpoint_mutation_request is not None:
                updated_checkpoint = (
                    checkpoint_mutation_rules._apply_runtime_publication_checkpoint_mutation(
                        RuntimePublicationMutation.model_validate(checkpoint_mutation_request),
                        updated_checkpoint,
                    )
                )
                if updated_checkpoint is None:
                    raise AssertionError("Interaction checkpoint mutation deleted its checkpoint.")
            updated = session.model_copy(update={"last_activity_at": now})
            if not queued:
                updated = updated.model_copy(
                    update={
                        "status": target_status,
                        "updated_at": now,
                        "last_activity_at": now,
                    }
                )
            receipt_record = interaction_receipts._interaction_transition_receipt_record(
                session=updated,
                event=copied_event,
                from_statuses=allowed_statuses,
                to_status=target_status,
                only_if_no_queued_messages=conditional,
                model_completion_stage_settlement=settlement_request,
                checkpoint_mutation=checkpoint_mutation_request,
                terminal_event=copied_terminal_event,
                terminal_decision=terminal_decision,
                status_changed=not queued,
                invocation_session_instance_id=expected_session_instance_id,
                invocation_active_profile=expected_active_invocation_profile,
                invocation_authority_state=expected_invocation_authority_state,
                recovery_claim_id=expected_recovery_claim_id,
            )
            applied = self._apply_event_append_unlocked(
                session,
                prepared_event_append,
                activity_at=now,
            )
            if not queued:
                applied = applied.model_copy(
                    update={
                        "status": target_status,
                        "updated_at": now,
                        "last_activity_at": now,
                    }
                )
            if applied != updated:
                raise RuntimeError("Prepared interaction transition result changed during commit.")
            self._sessions[session_id] = updated
            if not queued and checkpoint_mutation_request is not None:
                assert updated_checkpoint is not None
                self._checkpoints[session_id] = updated_checkpoint
            if terminal_decision is not None:
                assert settled_checkpoint is not None
                self._store_checkpoint_unlocked(session_id, settled_checkpoint)
            self._refresh_child_lifecycle_candidate_unlocked(updated)
            if settlement_record is not None and settlement_storage_key is not None:
                operation_records[settlement_storage_key] = settlement_record
                del operation_records[MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY]
            operation_records[receipt_storage_key] = receipt_record
            if completion_record is not None:
                assert steering_key is not None
                operation_records[steering_key] = completion_record
            return InteractionTransitionResult(
                session=updated,
                event=copied_event,
                terminal_event=copied_terminal_event,
                status_changed=not queued,
            )

    async def settle_session_invocation(self, command: Any) -> InteractionTransitionResult:
        from cayu.sessions._invocation_lifecycle import (
            SettleInvocationCommand,
            copy_invocation_lifecycle_command,
        )

        copied = copy_invocation_lifecycle_command(command)
        if type(copied) is not SettleInvocationCommand:
            raise TypeError("command must be a SettleInvocationCommand.")
        transition = copied.transition
        kwargs: dict[str, Any] = {
            "event": transition.event,
            "from_statuses": set(transition.from_statuses),
            "to_status": transition.to_status,
            "only_if_no_queued_messages": transition.only_if_no_queued_messages,
            "model_completion_stage_settlement": transition.model_completion_stage_settlement,
            "expected_session_instance_id": copied.expected_session_instance_id,
            "expected_active_invocation_profile": copied.expected_active_profile,
            "expected_invocation_authority_state": copied.expected_authority_state,
        }
        if transition.checkpoint_mutation is not None:
            kwargs["checkpoint_mutation"] = transition.checkpoint_mutation
        if transition.terminal_event is not None:
            kwargs["terminal_event"] = transition.terminal_event
            kwargs["terminal_decision"] = transition.terminal_decision
        if copied.recovery_claim_id is not None:
            kwargs["expected_recovery_claim_id"] = copied.recovery_claim_id
        if copied.terminalization_only:
            kwargs["terminalization_only"] = True
            kwargs["terminalization_plan_ownership"] = copied.terminalization_plan_ownership
        return await self.publish_interaction_transition(copied.session_id, **kwargs)

    async def load_interaction_transition_receipt(
        self,
        session_id: str,
        *,
        transition: InteractionTransitionSpec,
        expected_recovery_claim_id: str | None = None,
    ) -> InteractionTransitionReceiptResult | None:
        session_id, copied_transition = (
            interaction_rules._prepare_interaction_transition_receipt_lookup(
                session_id,
                transition=transition,
            )
        )
        copied_event = copied_transition.event
        receipt_storage_key = interaction_receipts._interaction_transition_storage_key(
            copied_event.id
        )
        async with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session not found: {session_id}")
            receipt_record = self._session_operation_records.get(session_id, {}).get(
                receipt_storage_key
            )
            existing = self._event_records_by_id.get((session_id, copied_event.id))
            if receipt_record is None:
                if existing is not None:
                    raise RuntimeError(
                        "Interaction transition event exists without its immutable receipt."
                    )
                return None
            receipt = interaction_receipts._reconstruct_interaction_transition_receipt(
                receipt_record,
                transition=copied_transition,
            )
            interaction_receipts._validate_interaction_transition_receipt_recovery_authority(
                receipt,
                current_checkpoint=self._checkpoints.get(session_id),
                expected_recovery_claim_id=expected_recovery_claim_id,
            )
            if existing is not None and existing.event != receipt.event:
                raise RuntimeError(
                    "Interaction transition receipt conflicts with retained event history."
                )
            if receipt.terminal_event is not None:
                retained_terminal = self._event_records_by_id.get(
                    (session_id, receipt.terminal_event.id)
                )
                if (
                    retained_terminal is not None
                    and retained_terminal.event != receipt.terminal_event
                ):
                    raise RuntimeError(
                        "Interaction transition receipt conflicts with retained terminal history."
                    )
            return InteractionTransitionReceiptResult(
                session=receipt.session,
                transition=interaction_receipts._interaction_transition_spec_from_receipt(receipt),
                status_changed=receipt.status_changed,
            )

    async def _load_interaction_transition_receipt_by_event_id(
        self,
        session_id: str,
        *,
        event_id: str,
        expected_session_instance_id: str,
        expected_active_invocation_profile: ActiveInvocationExecutionProfile,
    ) -> InteractionTransitionReceiptResult | None:
        session_id = require_clean_nonblank(session_id, "session_id")
        event_id = require_clean_nonblank(event_id, "event_id")
        receipt_storage_key = interaction_receipts._interaction_transition_storage_key(event_id)
        async with self._lock:
            current_session = self._sessions.get(session_id)
            if current_session is None:
                raise KeyError(f"Session not found: {session_id}")
            record = self._session_operation_records.get(session_id, {}).get(receipt_storage_key)
            existing = self._event_records_by_id.get((session_id, event_id))
            if record is None:
                if existing is not None:
                    raise RuntimeError(
                        "Interaction transition event exists without its immutable receipt."
                    )
                return None
            receipt = interaction_receipts._load_interaction_transition_receipt(record)
            interaction_receipts._validate_invocation_release_settlement_receipt_authority(
                receipt,
                current_session=current_session,
                expected_session_instance_id=expected_session_instance_id,
                expected_active_invocation_profile=expected_active_invocation_profile,
            )
            if receipt.event.id != event_id:
                raise RuntimeError("Interaction transition receipt has a conflicting event ID.")
            if existing is not None and existing.event != receipt.event:
                raise RuntimeError(
                    "Interaction transition receipt conflicts with retained event history."
                )
            return InteractionTransitionReceiptResult(
                session=receipt.session,
                transition=interaction_receipts._interaction_transition_spec_from_receipt(receipt),
                status_changed=receipt.status_changed,
            )

    async def fence_stalled_run(
        self,
        session_id: str,
        *,
        statuses: set[SessionStatus],
        inactive_for_seconds: int,
    ) -> Session | None:
        session_id = require_clean_nonblank(session_id, "session_id")
        allowed_statuses = session_record_rules._validate_status_set(statuses, "statuses")
        validated_inactive_for_seconds = _validate_inactive_for_seconds(inactive_for_seconds)
        assert validated_inactive_for_seconds is not None
        async with self._lock:
            now = self._ownership_clock()
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            if self._has_live_execution_owner_unlocked(session, now):
                return None
            current_checkpoint = self._checkpoints.get(session_id)
            if (
                active_provider_operation_cancellation_claim_from_checkpoint(
                    current_checkpoint,
                    now=now,
                )
                is not None
                or session_recovery_rules._incomplete_recovery_claim_from_checkpoint(
                    current_checkpoint
                )
                is not None
            ):
                return None
            inactive_before = utc_duration_cutoff(now, validated_inactive_for_seconds)
            if (
                inactive_before is None
                or session.status not in allowed_statuses
                or session.last_activity_at > inactive_before
            ):
                return None
            fenced = session.model_copy(
                update={
                    "run_epoch": session.run_epoch + 1,
                    "last_activity_at": now,
                }
            )
            self._sessions[session_id] = fenced
            result = fenced.model_copy(deep=True)
            invocation_scope._activate_session_run_fence(result)
            return result

    async def reserve_stalled_run_recovery(
        self,
        session_id: str,
        *,
        statuses: set[SessionStatus],
        inactive_for_seconds: int | None,
        checkpoint_transform: StoreTimeCheckpointTransform,
    ) -> Session | None:
        session_id = require_clean_nonblank(session_id, "session_id")
        allowed_statuses = session_record_rules._validate_status_set(statuses, "statuses")
        inactive_for_seconds = _validate_inactive_for_seconds(inactive_for_seconds)
        if checkpoint_transform is None:
            raise TypeError("checkpoint_transform is required.")
        async with self._lock:
            now = self._ownership_clock()
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            if inactive_for_seconds is not None and self._has_live_execution_owner_unlocked(
                session, now
            ):
                return None
            current = self._checkpoints.get(session_id)
            inactive_before = (
                None
                if inactive_for_seconds is None
                else utc_duration_cutoff(now, inactive_for_seconds)
            )
            if (
                session.status not in allowed_statuses
                or (
                    inactive_for_seconds is not None
                    and (inactive_before is None or session.last_activity_at > inactive_before)
                )
                or active_provider_operation_cancellation_claim_from_checkpoint(
                    current,
                    now=now,
                )
                is not None
            ):
                return None
            for owner in self._session_closure_progress.values():
                _check_closure_lineage_owner(owner, (session_id,))
            transformed = checkpoint_transform(
                session.model_copy(deep=True),
                checkpoint_preservation._copy_checkpoint_for_transform(
                    current, session_id=session_id
                ),
                now,
            )
            if transformed is None:
                return None
            self._store_checkpoint_unlocked(
                session_id,
                checkpoint_preservation._checkpoint_transform_result_preserving_completion_result_event_publications(
                    current,
                    transformed,
                    session_id=session_id,
                ),
            )
            return session.model_copy(deep=True)

    async def fence_run_and_transform_checkpoint(
        self,
        session_id: str,
        *,
        statuses: set[SessionStatus],
        checkpoint_transform: CheckpointTransform,
        result_checkpoint_transform: CheckpointTransform | None = None,
    ) -> Session:
        session_id = require_clean_nonblank(session_id, "session_id")
        allowed_statuses = session_record_rules._validate_status_set(statuses, "statuses")
        if checkpoint_transform is None:
            raise TypeError("checkpoint_transform is required.")
        if result_checkpoint_transform is not None and not callable(result_checkpoint_transform):
            raise TypeError("result_checkpoint_transform must be callable.")
        async with self._lock:
            now = self._ownership_clock()
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            if session.status not in allowed_statuses:
                raise SessionStatusConflict(f"Session status cannot be fenced: {session.status}")
            for owner in self._session_closure_progress.values():
                _check_closure_lineage_owner(owner, (session_id,))
            current = self._checkpoints.get(session_id)
            session_recovery_rules._require_live_incomplete_recovery_claim_for_run_epoch_transfer(
                current,
                now=now,
            )
            if (
                active_provider_operation_cancellation_claim_from_checkpoint(
                    current,
                    now=now,
                )
                is not None
            ):
                raise SessionStatusConflict(
                    "Provider-operation cancellation still owns the session run epoch."
                )
            transformed = checkpoint_transform(
                session.model_copy(deep=True),
                checkpoint_preservation._copy_checkpoint_for_transform(
                    current, session_id=session_id
                ),
            )
            if transformed is None:
                raise ValueError("Fenced checkpoint transform must return a checkpoint.")
            transformed = checkpoint_preservation._checkpoint_transform_result_preserving_completion_result_event_publications(
                current,
                transformed,
                session_id=session_id,
            )
            fenced = session.model_copy(
                update={
                    "run_epoch": session.run_epoch + 1,
                    "last_activity_at": now,
                }
            )
            if result_checkpoint_transform is not None:
                result_checkpoint = result_checkpoint_transform(
                    fenced.model_copy(deep=True),
                    checkpoint_preservation._copy_checkpoint_for_transform(
                        transformed,
                        session_id=session_id,
                    ),
                )
                if result_checkpoint is None:
                    raise ValueError("Result checkpoint transform must return a checkpoint.")
                transformed = checkpoint_preservation._checkpoint_transform_result_preserving_completion_result_event_publications(
                    transformed,
                    result_checkpoint,
                    session_id=session_id,
                )
            self._store_checkpoint_unlocked(session_id, transformed)
            self._sessions[session_id] = fenced
            result = fenced.model_copy(deep=True)
            invocation_scope._activate_session_run_fence(result)
            return result

    async def release_run_fence(self, session_id: str) -> None:
        session_id = require_clean_nonblank(session_id, "session_id")
        expected_run_epoch = invocation_scope._current_session_run_epoch(session_id)
        if expected_run_epoch is None:
            invocation_scope._deactivate_session_interaction(session_id)
            return
        try:
            async with self._lock:
                session = self._sessions.get(session_id)
                if session is not None and session.run_epoch == expected_run_epoch:
                    self._sessions[session_id] = session.model_copy(
                        update={"run_epoch": session.run_epoch + 1}
                    )
        finally:
            invocation_scope._deactivate_session_run_fence(session_id)
            invocation_scope._deactivate_session_interaction(session_id)

    async def release_session_invocation(self, command: Any) -> Any:
        from cayu.runtime._invocation_lifecycle import (
            checkpoint_with_invocation_lifecycle_receipt,
        )
        from cayu.sessions._invocation_lifecycle import (
            InvocationReleaseResult,
            ReleaseInvocationCommand,
            _invocation_lifecycle_receipt_ledger_from_checkpoint,
            copy_invocation_lifecycle_command,
            invocation_release_replay_from_state,
            require_invocation_command_authority,
            require_invocation_release_store_authority,
        )

        copied = copy_invocation_lifecycle_command(command)
        if type(copied) is not ReleaseInvocationCommand:
            raise TypeError("command must be a ReleaseInvocationCommand.")
        require_invocation_release_store_authority(copied)
        released = False
        try:
            async with self._lock:
                session = self._sessions.get(copied.session_id)
                if session is None:
                    raise KeyError(f"Session not found: {copied.session_id}")
                checkpoint = self._checkpoints.get(copied.session_id)
                ledger = _invocation_lifecycle_receipt_ledger_from_checkpoint(checkpoint)
                replay = invocation_release_replay_from_state(
                    session,
                    checkpoint,
                    copied,
                    _ledger=ledger,
                )
                if replay is not None:
                    released = True
                    return replay
                if copied.terminal_session_event is not None:
                    terminal_receipt = self._session_operation_records.get(
                        copied.session_id,
                        {},
                    ).get(
                        interaction_receipts._invocation_terminal_event_storage_key(
                            copied.terminal_session_event.id
                        )
                    )
                    retained_terminal_event = self._event_records_by_id.get(
                        (copied.session_id, copied.terminal_session_event.id)
                    )
                    interaction_receipts._require_invocation_release_terminal_session_event(
                        terminal_receipt,
                        (
                            None
                            if retained_terminal_event is None
                            else retained_terminal_event.event
                        ),
                        current_session=session,
                        expected_event=copied.terminal_session_event,
                        expected_session_instance_id=copied.expected_session_instance_id,
                        expected_active_invocation_profile=copied.expected_active_profile,
                    )
                elif copied.settlement_transition is None:
                    assert copied.recovery_claim_id is not None
                    interaction_receipts._require_invocation_release_recovery_claim(
                        checkpoint,
                        current_session=session,
                        recovery_claim_id=copied.recovery_claim_id,
                    )
                else:
                    settlement_record = self._session_operation_records.get(
                        copied.session_id,
                        {},
                    ).get(
                        interaction_receipts._interaction_transition_storage_key(
                            copied.settlement_transition.event.id
                        )
                    )
                    if settlement_record is None:
                        raise SessionRunFenced(
                            "Invocation release lacks exact durable terminal settlement."
                        )
                    interaction_receipts._require_invocation_release_settlement_record(
                        settlement_record,
                        current_session=session,
                        transition=copied.settlement_transition,
                        expected_session_instance_id=copied.expected_session_instance_id,
                        expected_active_invocation_profile=copied.expected_active_profile,
                    )
                require_invocation_command_authority(
                    session,
                    checkpoint,
                    session_id=copied.session_id,
                    session_instance_id=copied.expected_session_instance_id,
                    run_epochs=frozenset({copied.expected_run_epoch}),
                    active_profile=copied.expected_active_profile,
                )
                session = session.model_copy(update={"run_epoch": copied.expected_run_epoch + 1})
                updated_checkpoint = checkpoint_with_invocation_lifecycle_receipt(
                    checkpoint,
                    copied,
                    active_profile=copied.expected_active_profile,
                    result_session=session,
                    _ledger=ledger,
                )
                self._store_checkpoint_unlocked(copied.session_id, updated_checkpoint)
                self._sessions[copied.session_id] = session
                result = InvocationReleaseResult(
                    session=session,
                    active_profile=copied.expected_active_profile,
                    replayed=False,
                )
                released = True
                return result
        finally:
            if released and (
                invocation_scope._current_session_run_epoch(copied.session_id)
                == copied.expected_run_epoch
            ):
                invocation_scope._deactivate_session_run_fence(copied.session_id)
                invocation_scope._deactivate_session_interaction(copied.session_id)

    async def append_event(self, session_id: str, event: Event) -> None:
        await self.append_events(session_id, [event])

    async def claim_budget_reservation_identity(
        self,
        *,
        reservation_id: str,
        publication_session_id: str,
        publication_id: str,
    ) -> None:
        reservation_id = require_clean_nonblank(reservation_id, "reservation_id")
        publication_session_id = require_clean_nonblank(
            publication_session_id,
            "publication_session_id",
        )
        publication_id = require_clean_nonblank(publication_id, "publication_id")
        async with self._lock:
            session = self._sessions.get(publication_session_id)
            if session is None:
                raise KeyError(f"Session not found: {publication_session_id}")
            invocation_scope._assert_session_run_epoch(publication_session_id, session)
            existing = self._budget_reservation_identities.get(reservation_id)
            if existing is None:
                for owner in self._session_closure_progress.values():
                    _check_closure_lineage_owner(owner, (publication_session_id,))
                self._budget_reservation_identities[reservation_id] = (
                    publication_session_id,
                    publication_id,
                    False,
                )
                return
            if existing[:2] != (publication_session_id, publication_id):
                raise BudgetReservationIdentityConflict(
                    "Budget ledger reused a reservation identity."
                )

    def _prepare_event_append_unlocked(
        self,
        session: Session,
        events: Iterable[Event],
        *,
        events_are_detached: bool = False,
    ) -> _PreparedInMemoryEventAppend:
        from cayu.sessions.pending_actions import (
            pending_action_event_has_unknown_round_call,
            pending_action_event_is_from_different_tool_round,
            pending_action_event_lookup_id,
            pending_action_event_retains_history,
            pending_action_evidence_round_from_checkpoint,
            pending_action_lookup_key,
            project_pending_action_event_record,
        )

        session_id = session.id
        existing_ids = self._event_ids.get(session_id, set())
        event_batch = tuple(events)
        batch_ids: set[str] = set()
        budget_reservation_publications: dict[str, str] = {}
        for event in event_batch:
            if event.id in existing_ids or event.id in batch_ids:
                raise ValueError(f"Event already exists for session {session_id}: {event.id}")
            batch_ids.add(event.id)
            if event.type != EventType.BUDGET_RESERVED:
                continue
            raw_reservation_id = event.payload.get("reservation_id")
            if type(raw_reservation_id) is not str:
                continue
            reservation_id = raw_reservation_id
            existing = self._budget_reservation_identities.get(reservation_id)
            if existing is not None and (
                existing[:2] != (event.session_id, event.id) or existing[2]
            ):
                raise BudgetReservationIdentityConflict(
                    "Budget ledger reused a reservation identity."
                )
            previous_publication_id = budget_reservation_publications.setdefault(
                reservation_id,
                event.id,
            )
            if previous_publication_id != event.id:
                raise BudgetReservationIdentityConflict(
                    "Budget ledger reused a reservation identity."
                )

        try:
            pending_round = pending_action_evidence_round_from_checkpoint(
                self._checkpoints.get(session_id)
            )
        except (TypeError, ValueError):
            pending_round = None
        index_scope = self._pending_action_index_scopes.get(session_id)
        active_lookup_keys = frozenset() if index_scope is None else index_scope[0]
        round_lookup_key = (
            None
            if pending_round is None
            else pending_action_lookup_key(pending_round.tool_round_id)
        )
        prepared: list[_PreparedInMemoryEvent] = []
        next_sequence = self._next_event_sequence
        for event in event_batch:
            stored_event = event if events_are_detached else event.model_copy(deep=True)
            record = EventRecord(sequence=next_sequence, event=stored_event)
            self._cost_event_index.validate(record)
            event_type = str(stored_event.type)
            projected_record: EventRecord | None = None
            retention_keys: list[str] = []
            history_keys: list[str] = []
            if event_type in PENDING_ACTION_EVENT_TYPE_VALUES:
                retain_for_scope = (
                    pending_round is not None
                    and pending_action_event_has_unknown_round_call(
                        stored_event,
                        pending_round,
                    )
                )
                lookup_id = pending_action_event_lookup_id(stored_event)
                if lookup_id is not None:
                    lookup_key = pending_action_lookup_key(lookup_id)
                    if lookup_key in active_lookup_keys:
                        history_keys.append(lookup_key)
                        if not (
                            pending_action_event_retains_history(event_type)
                            and (
                                pending_round is None
                                or pending_action_event_is_from_different_tool_round(
                                    stored_event,
                                    pending_round,
                                )
                            )
                        ):
                            retention_keys.append(lookup_key)
                if retain_for_scope:
                    assert round_lookup_key is not None
                    retention_keys.append(round_lookup_key)
                    history_keys.append(round_lookup_key)
                if (
                    event_type in PENDING_ACTION_BARRIER_EVENT_TYPE_VALUES
                    or retention_keys
                    or history_keys
                ):
                    projected_record = project_pending_action_event_record(record)
            delivery = (
                None
                if event_type == str(EventType.RUNTIME_SINK_FAILED)
                else PersistedEventSideEffectDelivery(
                    session_id=session_id,
                    event_id=stored_event.id,
                    event_sequence=record.sequence,
                    status=PersistedEventSideEffectStatus.PENDING,
                )
            )
            prepared.append(
                _PreparedInMemoryEvent(
                    record=record,
                    event_type=event_type,
                    projected_record=projected_record,
                    retained_lookup_keys=tuple(dict.fromkeys(retention_keys)),
                    history_lookup_keys=tuple(dict.fromkeys(history_keys)),
                    delivery=delivery,
                    invocation_terminal_receipt=(
                        interaction_receipts._invocation_terminal_event_receipt_record(
                            session=session,
                            checkpoint=self._checkpoints.get(session_id),
                            event=stored_event,
                        )
                    ),
                )
            )
            next_sequence += 1
        return _PreparedInMemoryEventAppend(
            events=tuple(prepared),
            next_sequence=next_sequence,
            budget_reservation_publications=tuple(budget_reservation_publications.items()),
        )

    def _apply_event_append_unlocked(
        self,
        session: Session,
        prepared: _PreparedInMemoryEventAppend,
        *,
        activity_at: datetime | None = None,
    ) -> Session:
        from cayu.approvals.user_input import USER_INPUT_SUPERSESSION_INTENT_KEY
        from cayu.sessions.pending_actions import (
            pending_action_lookup_key,
            retain_pending_action_index_record,
        )

        session_id = session.id
        existing_ids = self._event_ids[session_id]
        session_records = self._session_event_records.setdefault(session_id, [])
        from cayu.sessions.interactions import INTERACTION_LIFECYCLE_EVENT_TYPES

        lifecycle_types = {str(event_type) for event_type in INTERACTION_LIFECYCLE_EVENT_TYPES}
        child_lifecycle_changed = False
        for prepared_event in prepared.events:
            record = prepared_event.record
            event_type = prepared_event.event_type
            projected_record = prepared_event.projected_record
            retained_lookup_keys = prepared_event.retained_lookup_keys
            history_lookup_keys = prepared_event.history_lookup_keys
            stored_event = record.event
            self._events[session_id].append(stored_event)
            self._event_records.append(record)
            self._event_records_by_id[(session_id, stored_event.id)] = record
            if stored_event.type is EventType.SESSION_MESSAGE_QUEUED:
                queue_id = stored_event.payload.get("queue_id")
                if type(queue_id) is str and len(queue_id) <= 512:
                    ids = self._session_message_acceptance_event_ids.setdefault(
                        (session_id, queue_id), set()
                    )
                    if len(ids) < 2:
                        ids.add(stored_event.id)
            self._cost_event_index.append(record)
            session_records.append(record)
            if session.parent_session_id is not None and stored_event.type in {
                EventType.SESSION_STARTED,
                EventType.SESSION_RESUMED,
                EventType.SESSION_FORKED,
                EventType.SESSION_COMPLETED,
                EventType.SESSION_FAILED,
                EventType.SESSION_INTERRUPTED,
            }:
                lifecycle_event_type = EventType(stored_event.type)
                self._child_lifecycle_event_records.setdefault(session_id, {})[
                    lifecycle_event_type
                ] = record
                child_lifecycle_changed = True
            if stored_event.type is EventType.SESSION_INTERRUPTED:
                supersession = stored_event.payload.get(USER_INPUT_SUPERSESSION_INTENT_KEY)
                superseded_input_id = (
                    None if type(supersession) is not dict else supersession.get("input_id")
                )
                if type(superseded_input_id) is str and superseded_input_id.strip():
                    lookup_key = pending_action_lookup_key(superseded_input_id)
                    indexed_supersessions = self._user_input_supersession_event_records.setdefault(
                        (session_id, lookup_key),
                        [],
                    )
                    if len(indexed_supersessions) < 2:
                        indexed_supersessions.append(record)
            interaction_id = stored_event.interaction_id
            if interaction_id is not None:
                self._register_private_authority_alias_unlocked(
                    interaction_id,
                    field_name="interaction_id",
                    scope_session_id=session_id,
                )
                self._interaction_event_records.setdefault(session_id, {}).setdefault(
                    interaction_id,
                    [],
                ).append(record)
            if stored_event.type == EventType.TURN_COMPLETED:
                nested_interaction_ids = stored_event.payload.get("interaction_ids")
                if type(nested_interaction_ids) is list:
                    for nested_interaction_id in dict.fromkeys(
                        value
                        for value in nested_interaction_ids
                        if type(value) is str and value.strip()
                    ):
                        self._register_private_authority_alias_unlocked(
                            nested_interaction_id,
                            field_name="interaction_id",
                            scope_session_id=session_id,
                        )
            if interaction_id is not None and event_type in lifecycle_types:
                latest_by_id = self._latest_interaction_event_records.setdefault(session_id, {})
                latest_by_sequence = self._latest_interaction_event_records_by_sequence.setdefault(
                    session_id, []
                )
                previous = latest_by_id.get(interaction_id)
                if previous is not None:
                    previous_index = bisect_left(
                        latest_by_sequence,
                        previous.sequence,
                        key=lambda candidate: candidate.sequence,
                    )
                    if (
                        previous_index >= len(latest_by_sequence)
                        or latest_by_sequence[previous_index].sequence != previous.sequence
                    ):
                        raise RuntimeError(
                            "In-memory interaction latest-event indexes are inconsistent."
                        )
                    latest_by_sequence.pop(previous_index)
                latest_by_id[interaction_id] = record
                latest_by_sequence.append(record)
            if projected_record is not None:
                if event_type in PENDING_ACTION_BARRIER_EVENT_TYPE_VALUES:
                    self._pending_action_latest_barrier_records[session_id] = projected_record
                else:
                    by_lookup_id = self._pending_action_event_records.setdefault(session_id, {})
                    event_history = self._pending_action_event_history.setdefault(
                        session_id,
                        {},
                    )
                    for retention_key in retained_lookup_keys:
                        retain_pending_action_index_record(
                            by_lookup_id.setdefault(retention_key, {}),
                            projected_record,
                        )
                    for history_key in history_lookup_keys:
                        event_history.setdefault(history_key, []).append(record)
            self._type_event_records.setdefault(event_type, []).append(record)
            workflow_name = stored_event.workflow_name
            workflow_attempt_id = stored_event.payload.get("attempt_id")
            workflow_step_id = stored_event.payload.get("step_id")
            if (
                stored_event.type == WORKFLOW_ATTEMPT_EVENT_TYPE
                and isinstance(workflow_name, str)
                and workflow_name
                and isinstance(workflow_attempt_id, str)
                and workflow_attempt_id
            ):
                self._workflow_attempt_event_records.setdefault(
                    (session_id, workflow_name),
                    [],
                ).append(record)
                self._workflow_active_attempt_ids[(session_id, workflow_name)] = workflow_attempt_id
            if (
                isinstance(workflow_name, str)
                and workflow_name
                and isinstance(workflow_attempt_id, str)
                and workflow_attempt_id
                and isinstance(workflow_step_id, str)
                and workflow_step_id
            ):
                self._workflow_step_event_records.setdefault(
                    (
                        session_id,
                        workflow_name,
                        workflow_attempt_id,
                        workflow_step_id,
                        event_type,
                    ),
                    [],
                ).append(record)
                if (
                    self._workflow_active_attempt_ids.get((session_id, workflow_name))
                    == workflow_attempt_id
                ):
                    self._workflow_fenced_step_event_records.setdefault(
                        (
                            session_id,
                            workflow_name,
                            workflow_step_id,
                            event_type,
                        ),
                        [],
                    ).append(record)
            existing_ids.add(stored_event.id)
            if prepared_event.invocation_terminal_receipt is not None:
                receipt_key, receipt_record = prepared_event.invocation_terminal_receipt
                self._session_operation_records[session_id][receipt_key] = receipt_record
            if prepared_event.delivery is not None:
                self._persisted_event_side_effect_deliveries[(session_id, stored_event.id)] = (
                    prepared_event.delivery
                )
        self._budget_reservation_identities.update(
            {
                reservation_id: (session_id, publication_id, True)
                for reservation_id, publication_id in (prepared.budget_reservation_publications)
            }
        )
        self._next_event_sequence = prepared.next_sequence
        self._refresh_queued_dispatch_terminal_receipts_unlocked(
            session_id,
            self._checkpoints.get(session_id),
        )
        if not prepared.events:
            return session
        updated = session.model_copy(
            update={"last_activity_at": activity_at or self._ownership_clock()}
        )
        if child_lifecycle_changed:
            self._refresh_child_lifecycle_candidate_unlocked(updated)
        return updated

    def _append_events_unlocked(
        self,
        session: Session,
        events: list[Event],
    ) -> Session:
        prepared = self._prepare_event_append_unlocked(session, events)
        return self._apply_event_append_unlocked(session, prepared)

    async def append_events(self, session_id: str, events: list[Event]) -> None:
        session_id, copied_events = session_event_rules._copy_session_event_batch(
            session_id, events
        )

        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            invocation_scope._assert_session_run_epoch(session_id, session)
            if copied_events:
                for owner in self._session_closure_progress.values():
                    _check_closure_lineage_owner(owner, (session_id,))
            self._sessions[session_id] = self._append_events_unlocked(session, copied_events)

    async def append_tool_effect_conflict(self, request: object) -> Event:
        from cayu.runtime._tool_effect_conflicts import (
            copy_tool_effect_conflict_audit,
            reconcile_tool_effect_conflict_event,
        )

        audit = copy_tool_effect_conflict_audit(request)
        session_id = audit.executing.intent.session_id
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError("Tool effect audit session is unavailable.")
            current = self._session_operation_records.get(session_id, {}).get(audit.storage_key)
            event = audit.prepare_event(session, current, now=self._ownership_clock())
            existing = self._event_records_by_id.get((session_id, event.id))
            if existing is not None:
                return reconcile_tool_effect_conflict_event(event, existing.event)
            for owner in self._session_closure_progress.values():
                _check_closure_lineage_owner(owner, (session_id,))
            prepared = self._prepare_event_append_unlocked(session, [event])
            self._sessions[session_id] = self._apply_event_append_unlocked(
                session, prepared, activity_at=session.last_activity_at
            )
            return copy_event(event)

    async def append_workflow_step_started(
        self,
        session_id: str,
        event: Event,
        *,
        workflow_name: str,
        attempt_id: str,
    ) -> bool:
        session_id, copied_event, workflow_name, attempt_id = _copy_workflow_step_reservation(
            session_id,
            event,
            workflow_name=workflow_name,
            attempt_id=attempt_id,
        )
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            invocation_scope._assert_session_run_epoch(session_id, session)
            if self._workflow_active_attempt_ids.get((session_id, workflow_name)) != attempt_id:
                return False
            if (session_id, copied_event.id) in self._event_records_by_id:
                return False
            for owner in self._session_closure_progress.values():
                _check_closure_lineage_owner(owner, (session_id,))
            self._sessions[session_id] = self._append_events_unlocked(session, [copied_event])
            return True

    async def load_mcp_manifest_baselines(
        self,
        history_keys: tuple[str, ...],
    ) -> McpManifestBaselineLoadResult:
        keys = _validate_mcp_manifest_history_keys(history_keys)
        async with self._lock:
            return McpManifestBaselineLoadResult(
                baselines={
                    key: self._mcp_manifest_baselines[key].model_copy(deep=True)
                    for key in keys
                    if key in self._mcp_manifest_baselines
                },
            )

    async def compare_and_publish_mcp_manifest_checks(
        self,
        session_id: str,
        *,
        expected_generations: dict[str, int | None],
        baseline_updates: dict[str, McpManifestBaseline],
        events: list[Event],
    ) -> McpManifestPublicationResult:
        (
            session_id,
            expected,
            updates,
            copied_events,
        ) = _copy_mcp_manifest_publication(
            session_id,
            expected_generations=expected_generations,
            baseline_updates=baseline_updates,
            events=events,
        )

        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            invocation_scope._assert_session_run_epoch(session_id, session)
            current = {
                key: self._mcp_manifest_baselines[key]
                for key in expected
                if key in self._mcp_manifest_baselines
            }
            if any(
                expected_generation
                != (None if (baseline := current.get(key)) is None else baseline.generation)
                for key, expected_generation in expected.items()
            ):
                return McpManifestPublicationResult(
                    published=False,
                    baselines={
                        key: baseline.model_copy(deep=True) for key, baseline in current.items()
                    },
                )
            _validate_mcp_manifest_publication_state(
                expected_generations=expected,
                current_baselines=current,
                baseline_updates=updates,
                events=copied_events,
            )
            for owner in self._session_closure_progress.values():
                _check_closure_lineage_owner(owner, (session_id,))
            self._sessions[session_id] = self._append_events_unlocked(session, copied_events)
            for key, baseline in updates.items():
                self._mcp_manifest_baselines[key] = baseline.model_copy(deep=True)
            return McpManifestPublicationResult(
                published=True,
                baselines={
                    key: self._mcp_manifest_baselines[key].model_copy(deep=True)
                    for key in expected
                    if key in self._mcp_manifest_baselines
                },
            )

    async def claim_first_persisted_event_side_effect(
        self, expected: PersistedEventSideEffectDelivery
    ) -> PersistedEventSideEffectClaim | None:
        expected = event_delivery_rules._copy_pending_first_event_delivery(expected)
        return await self._claim_persisted_event_side_effect(
            session_id=expected.session_id, event_id=expected.event_id, expected=expected
        )

    async def claim_persisted_event_side_effect(
        self,
        *,
        session_id: str | None = None,
        event_id: str | None = None,
        lease_seconds: float = 300.0,
    ) -> PersistedEventSideEffectClaim | None:
        return await self._claim_persisted_event_side_effect(
            session_id=session_id, event_id=event_id, lease_seconds=lease_seconds
        )

    async def _claim_persisted_event_side_effect(
        self,
        *,
        session_id: str | None = None,
        event_id: str | None = None,
        lease_seconds: float = 300.0,
        expected: PersistedEventSideEffectDelivery | None = None,
    ) -> PersistedEventSideEffectClaim | None:
        if session_id is not None:
            session_id = require_clean_nonblank(session_id, "session_id")
        if event_id is not None:
            event_id = require_clean_nonblank(event_id, "event_id")
        if (session_id is None) != (event_id is None):
            raise ValueError("session_id and event_id must be supplied together.")
        if type(lease_seconds) not in {int, float} or lease_seconds <= 0:
            raise ValueError("lease_seconds must be greater than 0.")
        async with self._lock:
            now = self._ownership_clock()
            deliveries = sorted(
                self._persisted_event_side_effect_deliveries.values(),
                key=lambda delivery: delivery.event_sequence,
            )
            closure_targets = {
                target
                for owner in self._session_closure_progress.values()
                for target in _closure_progress_targets(owner)
            }
            for delivery in deliveries:
                if expected is not None and delivery != expected:
                    continue
                if delivery.session_id in closure_targets:
                    continue
                if session_id is not None and (
                    delivery.session_id != session_id or delivery.event_id != event_id
                ):
                    continue
                if delivery.status in {
                    PersistedEventSideEffectStatus.DELIVERED,
                    PersistedEventSideEffectStatus.DEAD_LETTERED,
                }:
                    continue
                if (
                    delivery.status is PersistedEventSideEffectStatus.LEASED
                    and delivery.lease_expires_at is not None
                    and delivery.lease_expires_at > now
                ):
                    continue
                if (
                    delivery.status is PersistedEventSideEffectStatus.FAILED
                    and delivery.next_attempt_at is not None
                    and delivery.next_attempt_at > now
                ):
                    continue
                record = self._event_records_by_id.get((delivery.session_id, delivery.event_id))
                if record is None:
                    raise RuntimeError("Persisted side-effect delivery lost its source event.")
                claim = PersistedEventSideEffectClaim(
                    session_id=delivery.session_id,
                    event_id=delivery.event_id,
                    event_sequence=delivery.event_sequence,
                    event=record.event,
                    attempt=delivery.attempts + 1,
                    lease_expires_at=now + timedelta(seconds=float(lease_seconds)),
                )
                self._persisted_event_side_effect_deliveries[
                    (delivery.session_id, delivery.event_id)
                ] = delivery.model_copy(
                    update={
                        "status": PersistedEventSideEffectStatus.LEASED,
                        "attempts": claim.attempt,
                        "claim_id": claim.claim_id,
                        "lease_expires_at": claim.lease_expires_at,
                        "next_attempt_at": None,
                        "last_error": None,
                        "updated_at": now,
                    },
                    deep=True,
                )
                return claim
            return None

    async def get_persisted_event_side_effect_delivery(
        self,
        *,
        session_id: str,
        event_id: str,
    ) -> PersistedEventSideEffectDelivery | None:
        session_id = require_clean_nonblank(session_id, "session_id")
        event_id = require_clean_nonblank(event_id, "event_id")
        async with self._lock:
            delivery = self._persisted_event_side_effect_deliveries.get((session_id, event_id))
            return None if delivery is None else delivery.model_copy(deep=True)

    async def retire_failed_first_event_delivery(
        self, expected: PersistedEventSideEffectDelivery
    ) -> PersistedEventSideEffectDelivery | None:
        expected = event_delivery_rules._copy_failed_first_delivery_retirement(expected)
        key = (expected.session_id, expected.event_id)
        async with self._lock:
            if self._persisted_event_side_effect_deliveries.get(key) != expected:
                return None
            retired = expected.model_copy(
                update={
                    "status": PersistedEventSideEffectStatus.DEAD_LETTERED,
                    "next_attempt_at": None,
                    "updated_at": self._ownership_clock(),
                }
            )
            self._persisted_event_side_effect_deliveries[key] = retired
            return retired.model_copy(deep=True)

    async def mark_persisted_event_side_effect_delivered(
        self,
        claim: PersistedEventSideEffectClaim,
    ) -> PersistedEventSideEffectDelivery:
        claim = PersistedEventSideEffectClaim.model_validate(claim)
        async with self._lock:
            delivery = self._matching_persisted_event_side_effect_claim_unlocked(claim)
            updated = delivery.model_copy(
                update={
                    "status": PersistedEventSideEffectStatus.DELIVERED,
                    "claim_id": None,
                    "lease_expires_at": None,
                    "next_attempt_at": None,
                    "last_error": None,
                    "updated_at": self._ownership_clock(),
                },
                deep=True,
            )
            self._persisted_event_side_effect_deliveries[(claim.session_id, claim.event_id)] = (
                updated
            )
            return updated.model_copy(deep=True)

    async def mark_persisted_event_side_effect_failed(
        self,
        claim: PersistedEventSideEffectClaim,
        *,
        error: str,
        max_attempts: int,
        retry_delay_seconds: float,
    ) -> PersistedEventSideEffectDelivery:
        claim = PersistedEventSideEffectClaim.model_validate(claim)
        error = validate_persisted_event_side_effect_error(error)
        if type(max_attempts) is not int or max_attempts < 1:
            raise ValueError("max_attempts must be an integer greater than or equal to 1.")
        if (
            type(retry_delay_seconds) not in {int, float}
            or not math.isfinite(retry_delay_seconds)
            or retry_delay_seconds < 0
        ):
            raise ValueError("retry_delay_seconds must be a finite non-negative number.")
        dead_lettered = claim.attempt >= max_attempts
        async with self._lock:
            now = self._ownership_clock()
            delivery = self._matching_persisted_event_side_effect_claim_unlocked(claim)
            updated = delivery.model_copy(
                update={
                    "status": (
                        PersistedEventSideEffectStatus.DEAD_LETTERED
                        if dead_lettered
                        else PersistedEventSideEffectStatus.FAILED
                    ),
                    "claim_id": None,
                    "lease_expires_at": None,
                    "next_attempt_at": (
                        None
                        if dead_lettered
                        else now + timedelta(seconds=float(retry_delay_seconds))
                    ),
                    "last_error": error,
                    "updated_at": now,
                },
                deep=True,
            )
            self._persisted_event_side_effect_deliveries[(claim.session_id, claim.event_id)] = (
                updated
            )
            return updated.model_copy(deep=True)

    async def defer_persisted_event_side_effect(
        self,
        claim: PersistedEventSideEffectClaim,
    ) -> PersistedEventSideEffectDelivery:
        claim = PersistedEventSideEffectClaim.model_validate(claim)
        async with self._lock:
            delivery = self._matching_persisted_event_side_effect_claim_unlocked(claim)
            updated = delivery.model_copy(
                update={
                    "status": PersistedEventSideEffectStatus.PENDING,
                    "attempts": delivery.attempts - 1,
                    "claim_id": None,
                    "lease_expires_at": None,
                    "next_attempt_at": None,
                    "last_error": None,
                    "updated_at": self._ownership_clock(),
                },
                deep=True,
            )
            self._persisted_event_side_effect_deliveries[(claim.session_id, claim.event_id)] = (
                updated
            )
            return updated.model_copy(deep=True)

    async def get_persisted_event_side_effect_health(self) -> PersistedEventSideEffectHealth:
        async with self._lock:
            now = self._ownership_clock().astimezone(UTC)
            rows = list(self._persisted_event_side_effect_deliveries.values())
            counts = dict.fromkeys(side_effect_health.CONDITIONS, 0)
            timestamps: dict[str, datetime | None] = dict.fromkeys(
                [
                    "oldest_claimable_at",
                    "oldest_pending_at",
                    "oldest_failed_at",
                    "oldest_dead_letter_at",
                    "earliest_live_lease_expires_at",
                ]
            )
            max_attempts = 0

            def oldest(key: str, value: datetime) -> None:
                previous = timestamps[key]
                timestamps[key] = value if previous is None else min(previous, value)

            for row in rows:
                if row.status == "leased":
                    counts[
                        "leased_expired"
                        if side_effect_health.claimable(row, now)
                        else "leased_live"
                    ] += 1
                    if row.lease_expires_at is not None and row.lease_expires_at > now:
                        oldest("earliest_live_lease_expires_at", row.lease_expires_at)
                elif row.status == "failed":
                    counts["failed_retryable"] += 1
                    counts["failed_deferred"] += int(
                        row.next_attempt_at is not None and row.next_attempt_at > now
                    )
                    counts["repeatedly_failing"] += int(row.attempts > 1)
                    oldest("oldest_failed_at", row.updated_at)
                else:
                    counts[str(row.status)] += 1
                if row.status == "pending":
                    oldest("oldest_pending_at", row.updated_at)
                if row.status == "dead_lettered":
                    oldest("oldest_dead_letter_at", row.updated_at)
                if row.status != "delivered":
                    counts["outstanding_total"] += 1
                    max_attempts = max(max_attempts, row.attempts)
                if row.status in {"pending", "failed", "leased"}:
                    live = (
                        row.status == "leased"
                        and row.lease_expires_at is not None
                        and row.lease_expires_at > now
                    )
                    threshold = side_effect_health.PERSISTED_EVENT_SIDE_EFFECT_MAX_ATTEMPTS - (
                        0 if live else 1
                    )
                    counts["final_attempt_boundary"] += int(row.attempts >= threshold)
                if side_effect_health.claimable(row, now):
                    counts["claimable_total"] += 1
                    eligible = row.updated_at
                    if row.status == "leased" and row.lease_expires_at is not None:
                        eligible = row.lease_expires_at
                    elif row.status == "failed" and row.next_attempt_at is not None:
                        eligible = max(eligible, row.next_attempt_at)
                    oldest("oldest_claimable_at", eligible)
            return side_effect_health.finish_health(
                {**counts, **timestamps, "max_outstanding_attempts": max_attempts},
                now,
            )

    async def query_persisted_event_side_effect_deliveries(
        self,
        query: PersistedEventSideEffectQuery,
    ) -> PersistedEventSideEffectPage:
        query = PersistedEventSideEffectQuery.model_validate(query)
        key = side_effect_health.cursor_key(query)
        async with self._lock:
            now = self._ownership_clock().astimezone(UTC)
            rows = sorted(
                (
                    row
                    for row in self._persisted_event_side_effect_deliveries.values()
                    if (key is None or (row.session_id, row.event_id) > key)
                    and (query.statuses is None or row.status in query.statuses)
                    and (not query.outstanding_only or row.status != "delivered")
                    and (not query.claimable_only or side_effect_health.claimable(row, now))
                ),
                key=lambda row: (row.session_id, row.event_id),
            )[: query.limit + 1]
            return side_effect_health.page(rows, query, now)

    async def list_persisted_event_side_effect_deliveries(
        self,
        *,
        statuses: set[PersistedEventSideEffectStatus] | None = None,
        claimable_only: bool = False,
        after_sequence: int | None = None,
        limit: int = 100,
    ) -> list[PersistedEventSideEffectDelivery]:
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000.")
        if type(claimable_only) is not bool:
            raise TypeError("claimable_only must be a bool.")
        if after_sequence is not None and (type(after_sequence) is not int or after_sequence < 0):
            raise ValueError("after_sequence must be a non-negative integer.")
        selected_statuses = (
            None
            if statuses is None
            else {PersistedEventSideEffectStatus(status) for status in statuses}
        )
        async with self._lock:
            now = self._ownership_clock()
            deliveries = sorted(
                self._persisted_event_side_effect_deliveries.values(),
                key=lambda delivery: delivery.event_sequence,
            )
            return [
                delivery.model_copy(deep=True)
                for delivery in deliveries
                if after_sequence is None or delivery.event_sequence > after_sequence
                if selected_statuses is None or delivery.status in selected_statuses
                if not claimable_only
                or (
                    delivery.status
                    in {
                        PersistedEventSideEffectStatus.PENDING,
                        PersistedEventSideEffectStatus.FAILED,
                    }
                    and (
                        delivery.status is not PersistedEventSideEffectStatus.FAILED
                        or delivery.next_attempt_at is None
                        or delivery.next_attempt_at <= now
                    )
                )
                or (
                    delivery.status is PersistedEventSideEffectStatus.LEASED
                    and delivery.lease_expires_at is not None
                    and delivery.lease_expires_at <= now
                )
            ][:limit]

    async def renew_persisted_event_side_effect(
        self,
        claim: PersistedEventSideEffectClaim,
        *,
        lease_seconds: float = 300.0,
    ) -> PersistedEventSideEffectDelivery:
        claim = PersistedEventSideEffectClaim.model_validate(claim)
        if type(lease_seconds) not in {int, float} or not 0 < lease_seconds <= 86_400:
            raise ValueError("lease_seconds must be positive and at most 86400.")
        async with self._lock:
            if (
                claim.session_id,
                claim.event_id,
            ) not in self._persisted_event_side_effect_deliveries:
                raise PersistedEventSideEffectClaimLost(
                    "Persisted event side-effect claim is no longer active."
                )
            delivery = self._matching_persisted_event_side_effect_claim_unlocked(claim)
            now = self._ownership_clock()
            if delivery.lease_expires_at is None or delivery.lease_expires_at <= now:
                raise PersistedEventSideEffectClaimLost(
                    "Persisted event side-effect claim is no longer active."
                )
            renewed = delivery.model_copy(
                update={
                    "lease_expires_at": max(
                        delivery.lease_expires_at, now + timedelta(seconds=float(lease_seconds))
                    ),
                    "updated_at": now,
                },
                deep=True,
            )
            self._persisted_event_side_effect_deliveries[(claim.session_id, claim.event_id)] = (
                renewed
            )
            return renewed.model_copy(deep=True)

    def _matching_persisted_event_side_effect_claim_unlocked(
        self,
        claim: PersistedEventSideEffectClaim,
    ) -> PersistedEventSideEffectDelivery:
        delivery = self._persisted_event_side_effect_deliveries.get(
            (claim.session_id, claim.event_id)
        )
        if delivery is None:
            raise ValueError("Persisted event side-effect delivery was not found.")
        if (
            delivery.status is not PersistedEventSideEffectStatus.LEASED
            or delivery.claim_id != claim.claim_id
            or delivery.attempts != claim.attempt
        ):
            raise PersistedEventSideEffectClaimLost(
                "Persisted event side-effect claim is no longer active."
            )
        return delivery

    def _session_message_source_unlocked(
        self,
        session_id: str,
        *,
        include_transcript_digest: bool,
        include_checkpoint_digest: bool,
    ) -> SessionMessageSource:
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError("Session-message source not found.")
        messages = self._transcripts.get(session_id, [])
        transcript = None
        if include_transcript_digest:
            transcript = TranscriptSnapshot(
                records=[
                    TranscriptRecord(index=index, message=detach_message(message))
                    for index, message in enumerate(messages)
                ],
                cursor=len(messages),
            )
        return SessionMessageSource(
            session_id=session.id,
            session_instance_id=session.instance_id,
            run_epoch=session.run_epoch,
            transcript_cursor=len(messages),
            transcript_sha256=(
                None if transcript is None else fork_source_transcript_sha256(transcript)
            ),
            checkpoint_sha256=(
                session_message_checkpoint_sha256(self._checkpoints.get(session_id))
                if include_checkpoint_digest
                else None
            ),
        )

    def _session_message_accepted_event_unlocked(
        self,
        raw: dict[str, Any],
        *,
        quarantine: bool = False,
    ) -> Event | None:
        if quarantine:
            ids = self._session_message_acceptance_event_ids.get(
                (raw["session_id"], raw["queue_id"]), set()
            )
            if len(ids) != 1:
                raise SessionMessageConflict()
            record = self._event_records_by_id.get((raw["session_id"], next(iter(ids))))
            return None if record is None else record.event
        event_id = raw["accepted_event_id"]
        if type(event_id) is not str or len(event_id) > 512:
            raise SessionMessageConflict()
        record = self._event_records_by_id.get((raw["session_id"], event_id))
        return None if record is None else record.event

    def _raw_session_message_unlocked(self, message: SessionQueuedMessage) -> dict[str, Any]:
        raw = message.model_dump(mode="json", warnings=False)
        raw["terminal_json"] = self._session_message_terminal_receipts.get(
            (message.session_id, message.queue_id)
        )
        return raw

    def _inspect_session_message_unlocked(
        self, message: SessionQueuedMessage
    ) -> SessionMessageInspectionRecord:
        from cayu.runtime._session_message_queue import inspect_record

        raw = self._raw_session_message_unlocked(message)
        return inspect_record(
            raw,
            lambda: SessionQueuedMessage.model_validate(
                {key: value for key, value in raw.items() if key != "terminal_json"}
            ),
        )

    @runtime_session_query
    async def inspect_session_messages(
        self,
        query: SessionMessageQuery,
        *,
        expected_authorized_session_instance_id: str | None = None,
    ) -> SessionMessageInspection:
        from cayu.runtime._session_message_queue import (
            copy_inspection_query,
            inspection_boundary,
            inspection_next_cursor,
            inspection_priority,
            require_authorized_session_instance,
        )

        query = copy_inspection_query(query)
        async with self._lock:
            session = self._sessions.get(query.session_id)
            require_resource_session(session, "read")
            if session is None:
                raise KeyError("Session-message target not found.")
            require_authorized_session_instance(session, expected_authorized_session_instance_id)
            candidates = self._queued_session_messages_by_idempotency.get(session.id, {}).values()
            boundary = inspection_boundary(
                session,
                query.cursor,
                max((message.ordering_key for message in candidates), default=0)
                if query.cursor is None
                else 0,
            )
            messages = heapq.nsmallest(
                query.limit + 1,
                (
                    message
                    for message in candidates
                    if message.ordering_key <= boundary.through_ordering_key
                    and (inspection_priority(message.delivery_mode), message.ordering_key)
                    > (boundary.after_priority, boundary.after_ordering_key)
                ),
                key=lambda message: (
                    inspection_priority(message.delivery_mode),
                    message.ordering_key,
                ),
            )
            records = tuple(
                self._inspect_session_message_unlocked(message)
                for message in messages[: query.limit]
            )
            return SessionMessageInspection(
                session_id=session.id,
                session_instance_id=session.instance_id,
                records=records,
                next_cursor=(
                    inspection_next_cursor(
                        boundary,
                        inspection_priority(messages[query.limit - 1].delivery_mode),
                        records[-1].ordering_key,
                    )
                    if len(messages) > query.limit
                    else None
                ),
            )

    @runtime_session_query
    async def apply_session_message_action(
        self, request: SessionMessageActionRequest
    ) -> SessionMessageActionResult:
        from cayu.runtime._session_message_queue import (
            raw_revision,
            replay_action,
            terminal_event,
            terminal_receipt,
        )

        request = copy_session_message_action(request)
        async with self._lock:
            session = self._sessions.get(request.session_id)
            require_resource_session(session, "modify")
            if session is None or session.instance_id != request.session_instance_id:
                raise SessionMessageConflict()
            messages = self._queued_session_messages_by_idempotency.get(session.id, {})
            message = next(
                (item for item in messages.values() if item.queue_id == request.queue_id), None
            )
            if message is None:
                raise SessionMessageConflict()
            raw = self._raw_session_message_unlocked(message)
            accepted_event = self._session_message_accepted_event_unlocked(
                raw,
                quarantine=request.action == "quarantine",
            )
            replay = replay_action(raw, request, accepted_event=accepted_event)
            if replay is not None:
                return SessionMessageActionResult(
                    record=self._inspect_session_message_unlocked(message),
                    event=replay,
                    replayed=True,
                )
            for owner in self._session_closure_progress.values():
                _check_closure_lineage_owner(owner, (session.id,))
            if (
                raw_revision(raw) != request.expected_revision
                or message.status != SessionMessageQueueStatus.QUEUED
                or message.delivered_event_id is not None
                or any(
                    delivered.queue_id == message.queue_id
                    for delivery in self._session_message_delivery_records.values()
                    if delivery.session_id == session.id
                    for delivered in delivery.batch.messages
                )
            ):
                raise SessionMessageConflict()
            if (
                request.action == "withdraw"
                and self._inspect_session_message_unlocked(message).validity != "valid"
            ):
                raise SessionMessageConflict()
            status = (
                SessionMessageQueueStatus.WITHDRAWN
                if request.action == "withdraw"
                else SessionMessageQueueStatus.QUARANTINED
            )
            event = terminal_event(
                session,
                raw,
                status,
                self._ownership_clock(),
                actor=request.requested_by,
                accepted_event=accepted_event,
            )
            updated = message.model_copy(update={"status": status}, deep=True)
            updated_session = self._append_events_unlocked(session, [event])
            self._session_message_terminal_receipts[(session.id, message.queue_id)] = (
                terminal_receipt(status, event, request)
            )
            messages[message.idempotency_key] = updated
            for mode in SessionMessageDeliveryMode:
                key = (session.id, mode)
                pending = self._pending_session_messages.get(key)
                if pending is not None:
                    remaining = deque(item for item in pending if item.queue_id != message.queue_id)
                    if remaining:
                        self._pending_session_messages[key] = remaining
                    else:
                        del self._pending_session_messages[key]
            self._sessions[session.id] = updated_session
            return SessionMessageActionResult(
                record=self._inspect_session_message_unlocked(updated), event=event
            )

    @runtime_session_query
    async def snapshot_session_message_source(
        self,
        session_id: str,
        *,
        include_transcript_digest: bool = False,
        include_checkpoint_digest: bool = False,
        expected_authorized_session_instance_id: str | None = None,
    ) -> SessionMessageSource:
        from cayu.runtime._session_message_queue import require_authorized_session_instance

        session_id = require_clean_nonblank(session_id, "session_id")
        if (
            type(include_transcript_digest) is not bool
            or type(include_checkpoint_digest) is not bool
        ):
            raise TypeError("Source snapshot digest flags must be bools.")
        async with self._lock:
            session = self._sessions.get(session_id)
            require_resource_session(session, "read")
            if session is None:
                raise KeyError("Session-message source not found.")
            require_authorized_session_instance(session, expected_authorized_session_instance_id)
            return self._session_message_source_unlocked(
                session_id,
                include_transcript_digest=include_transcript_digest,
                include_checkpoint_digest=include_checkpoint_digest,
            )

    @runtime_session_query
    async def append_peer_content(
        self,
        request: PeerContentAppendRequest,
        *,
        qualify_target: Callable[[Session], None] | None = None,
        pending_transcript_cursor: int | None = None,
    ) -> PeerContentReceipt:
        from cayu.collaboration.peer_content import PeerContentReceipt

        if type(request) is not PeerContentAppendRequest:
            raise TypeError("Peer append requires a PeerContentAppendRequest.")
        request = PeerContentAppendRequest.model_validate(request)
        from cayu.collaboration.peer_content import resolve_peer_target

        key = request.append_key.model_dump_json()
        commitment = request.model_dump_json()
        async with self._lock:
            creation = request.append_key.creation_target
            target_id, target_instance, creation_excluded = resolve_peer_target(
                request.append_key,
                None if creation is None else self._session_creation_decisions.get(creation.key),
            )
            from cayu.sessions._access_rules import _query_bounds

            if _query_bounds.get() is not None:
                require_resource_session(
                    self._sessions.get(request.occurrence.sender_session_id), "read"
                )
                require_resource_session(
                    None if target_id is None else self._sessions.get(target_id), "modify"
                )
            from cayu.storage._peer_attempts import historical_replay, replay_or_advance

            historical = historical_replay(
                request, self._peer_content_attempts.get(request.operation_key)
            )
            if historical is not None:
                return historical
            existing = self._peer_content_receipts.get(key)
            if existing is not None:
                replay = replay_or_advance(request, self._peer_content_requests[key], existing)
                if replay is not None:
                    return replay
            elif request.replaces_operation_key is not None:
                raise PeerContentConflict()
            from cayu.storage._peer_attempts import receiving_cursor

            target_cursor = receiving_cursor(request, existing, pending_transcript_cursor)
            target = None if target_id is None else self._sessions.get(target_id)
            parked_target = False
            if target is not None and target.status in {
                SessionStatus.COMPLETED,
                SessionStatus.FAILED,
                SessionStatus.INTERRUPTED,
            }:
                from cayu.storage._peer_attempts import (
                    parked_delivery_key,
                    permits_parked_delivery_append,
                )

                checkpoint = self._checkpoints.get(target.id)
                wait_key = parked_delivery_key(
                    checkpoint, session_id=target.id, instance_id=target.instance_id
                )
                if wait_key is not None:
                    parked_target = permits_parked_delivery_append(
                        request,
                        checkpoint,
                        self._session_operation_records.get(target.id, {}).get(wait_key),
                        session_id=target.id,
                        instance_id=target.instance_id,
                        run_epoch=target.run_epoch,
                    )
            if target is not None and target.status not in {
                SessionStatus.COMPLETED,
                SessionStatus.FAILED,
                SessionStatus.INTERRUPTED,
            }:
                from cayu.runtime._session_message_queue import require_open_admission

                require_open_admission(target.status, self._checkpoints.get(target.id))
            target_binding = (
                None if target_id is None else self._participant_session_bindings.get(target_id)
            )
            target_binding_valid = target_binding is not None and (
                target_binding.session_instance_id == target_instance
                and target_binding.participant.participant_id == request.append_key.consumer_id
                and target_binding.participant.incarnation
                == request.append_key.consumer_participant_incarnation
            )
            source_binding = self._participant_session_bindings.get(
                request.occurrence.sender_session_id
            )
            source_valid = (
                source_binding is not None
                and source_binding.session_instance_id
                == request.occurrence.sender_session_instance_id
                and source_binding.participant.participant_id
                == request.occurrence.sender_participant_id
                and source_binding.participant.incarnation
                == request.occurrence.sender_participant_incarnation
            )
            expired = (
                int(self._ownership_clock().timestamp() * 1000)
                >= request.attempt_key.deadline_at_ms
            )
            from cayu.storage._peer_attempts import qualify, require_capacity

            outstanding = 0
            for other_key, other in self._peer_content_receipts.items():
                if other_key == key or (
                    other.append_key.consumer_id != request.append_key.consumer_id
                    or other.append_key.consumer_participant_incarnation
                    != request.append_key.consumer_participant_incarnation
                ):
                    continue
                if other.status == "pending":
                    outstanding += 1
                elif other.status == "appended":
                    if (other.target_session_id, other.target_session_instance_id) in (
                        self._peer_content_deleted_targets
                    ):
                        continue
                    queued = self._queued_session_messages_by_idempotency.get(
                        other.target_session_id, {}
                    ).get(other.operation_key)
                    if queued is None or queued.status == SessionMessageQueueStatus.QUEUED:
                        outstanding += 1
            if expired:
                receipt = PeerContentReceipt(
                    operation_key=request.operation_key,
                    append_key=request.append_key,
                    attempt_generation=request.attempt_key.attempt_generation,
                    status="excluded",
                    reason="delivery_deadline_expired",
                )
            elif (
                creation_excluded
                or not source_valid
                or (
                    target is not None
                    and not parked_target
                    and target.status
                    in {
                        SessionStatus.COMPLETED,
                        SessionStatus.FAILED,
                        SessionStatus.INTERRUPTED,
                    }
                )
            ):
                receipt = PeerContentReceipt(
                    operation_key=request.operation_key,
                    append_key=request.append_key,
                    attempt_generation=request.attempt_key.attempt_generation,
                    status="excluded",
                    reason="source_or_target_unavailable",
                )
            elif target is None:
                receipt = PeerContentReceipt(
                    operation_key=request.operation_key,
                    append_key=request.append_key,
                    attempt_generation=request.attempt_key.attempt_generation,
                    status="pending",
                    reason="target_not_created",
                )
            elif target_binding is None:
                receipt = PeerContentReceipt(
                    operation_key=request.operation_key,
                    append_key=request.append_key,
                    attempt_generation=request.attempt_key.attempt_generation,
                    status="pending",
                    reason="target_binding_pending",
                )
            elif not target_binding_valid or target.instance_id != target_instance:
                receipt = PeerContentReceipt(
                    operation_key=request.operation_key,
                    append_key=request.append_key,
                    attempt_generation=request.attempt_key.attempt_generation,
                    status="excluded",
                    reason="target_binding_mismatch",
                )
            elif (
                isinstance(self._checkpoints.get(target.id), dict)
                and tool_publication_rules._PENDING_TOOL_ROUND_CHECKPOINT_KEY
                in self._checkpoints[target.id]
            ):
                receipt = PeerContentReceipt(
                    operation_key=request.operation_key,
                    append_key=request.append_key,
                    attempt_generation=request.attempt_key.attempt_generation,
                    status="pending",
                    reason="target_busy",
                )
            elif (
                target.run_epoch != request.attempt_key.target_run_epoch
                or len(self._transcripts.get(target.id, ())) != target_cursor
            ):
                receipt = PeerContentReceipt(
                    operation_key=request.operation_key,
                    append_key=request.append_key,
                    attempt_generation=request.attempt_key.attempt_generation,
                    status="pending",
                    reason="target_cursor_changed",
                )
            else:
                require_capacity(outstanding)
                qualify(qualify_target, target)
                peer_message = Message(
                    role=MessageRole.ASSISTANT,
                    content=(
                        request.occurrence.to_message_part(
                            append_key=request.append_key,
                            projection_id=request.append_key.projection_id,
                            operation_key=request.operation_key,
                        ),
                    ),
                )
                from cayu.collaboration.peer_content import peer_queue_id

                assert target_id is not None and target_instance is not None
                queue_id = peer_queue_id(request.append_key, target_id, target_instance)
                if any(
                    item.queue_id == queue_id or item.idempotency_key == request.operation_key
                    for item in self._queued_session_messages_by_idempotency.get(
                        target_id, {}
                    ).values()
                ):
                    raise PeerContentConflict("Peer queue identity already has another authority.")
                delivery_mode = (
                    SessionMessageDeliveryMode.ON_IDLE
                    if request.wake_policy == "ordinary_continuation"
                    else SessionMessageDeliveryMode.NEXT_TURN
                )
                accepted_at = self._ownership_clock()
                ordering_key = self._next_session_message_ordering_key
                accepted_event = Event(
                    type=EventType.SESSION_MESSAGE_QUEUED,
                    session_id=target.id,
                    agent_name=target.agent_name,
                    environment_name=target.environment_name,
                    timestamp=accepted_at,
                    payload={
                        **session_messaging._queued_session_message_event_payload(
                            queue_id=queue_id,
                            delivery_mode=delivery_mode,
                            ordering_key=ordering_key,
                            actor=None,
                            run_epoch=target.run_epoch,
                            transcript_cursor=len(self._transcripts.get(target.id, [])),
                        ),
                        "peer_occurrence_id": request.occurrence.occurrence_id,
                        "peer_provenance_sha256": request.occurrence.provenance_sha256,
                    },
                )
                queued_message = SessionQueuedMessage(
                    queue_id=queue_id,
                    session_id=target.id,
                    idempotency_key=request.operation_key,
                    content=request.occurrence.payload.text,
                    message=peer_message,
                    delivery_mode=delivery_mode,
                    status=SessionMessageQueueStatus.QUEUED,
                    ordering_key=ordering_key,
                    accepted_run_epoch=target.run_epoch,
                    accepted_transcript_cursor=len(self._transcripts.get(target.id, [])),
                    accepted_event_id=accepted_event.id,
                    accepted_at=accepted_at,
                    conditions=SessionMessageConditions(),
                )
                self._sessions[target.id] = self._append_events_unlocked(target, [accepted_event])
                self._queued_session_messages_by_idempotency.setdefault(target.id, {})[
                    queued_message.idempotency_key
                ] = queued_message
                self._pending_session_messages.setdefault(
                    (target.id, delivery_mode), deque()
                ).append(queued_message)
                self._next_session_message_ordering_key += 1
                receipt = PeerContentReceipt(
                    operation_key=request.operation_key,
                    append_key=request.append_key,
                    attempt_generation=request.attempt_key.attempt_generation,
                    status="appended",
                    target_session_id=target_id,
                    target_session_instance_id=target_instance,
                    occurrence=request.occurrence,
                    queue_id=queue_id,
                )
            if receipt.status == "pending":
                require_capacity(outstanding)
            self._peer_content_receipts[key] = receipt
            self._peer_content_operation_commitments[key] = commitment
            self._peer_content_operations[request.operation_key] = commitment
            self._peer_content_requests[key] = request.model_copy(deep=True)
            self._peer_content_attempts[request.operation_key] = (
                request.model_dump_json(),
                receipt.model_dump_json(),
            )
            return receipt.model_copy(deep=True)

    async def read_peer_content_attempt(
        self, request: PeerContentAppendRequest
    ) -> PeerContentReceipt | None:
        from cayu.storage._peer_attempts import exact_read

        request = PeerContentAppendRequest.model_validate(request)
        async with self._lock:
            return exact_read(request, self._peer_content_attempts.get(request.operation_key))

    async def list_pending_peer_content(self, *, after_operation_key=None, limit=32):
        """Trusted receiving-owner discovery, independent of creation settlement."""
        from cayu.collaboration.peer_content import validate_peer_discovery

        validate_peer_discovery(after_operation_key, limit)
        async with self._lock:
            keys = sorted(
                self._peer_content_requests,
                key=lambda key: self._peer_content_requests[key].operation_key,
            )
            return tuple(
                self._peer_content_requests[key].model_copy(deep=True)
                for key in keys
                if self._peer_content_receipts[key].status == "pending"
                and self._peer_content_requests[key].operation_key > (after_operation_key or "")
            )[:limit]

    async def read_peer_content(self, append_key: PeerAppendKey) -> PeerContentReceipt | None:
        if type(append_key) is not PeerAppendKey:
            raise TypeError("append_key must be a PeerAppendKey.")
        append_key = PeerAppendKey.model_validate(append_key)
        async with self._lock:
            value = self._peer_content_receipts.get(append_key.model_dump_json())
            return None if value is None else value.model_copy(deep=True)

    async def record_peer_content_exposure(
        self, request: PeerContentExposureRequest
    ) -> PeerContentExposureReceipt:
        if type(request) is not PeerContentExposureRequest:
            raise TypeError("Exposure requires a PeerContentExposureRequest.")
        key = request.exposure_id
        commitment = request.model_dump_json()
        async with self._lock:
            existing = self._peer_content_exposures.get(key)
            if existing is None:
                existing = next(
                    (
                        value
                        for value in self._peer_content_exposures.values()
                        if value[1].operation_key == request.operation_key
                    ),
                    None,
                )
            if existing is not None:
                expected = (
                    request.identity_commitment()
                    if existing[1].outcome == "pending"
                    else commitment
                )
                actual = existing[0]
                if actual != expected:
                    raise PeerContentConflict()
                if existing[1].outcome == "pending":
                    receipt = PeerContentExposureReceipt(
                        operation_key=request.operation_key,
                        append_key=request.append_key,
                        exposure_id=request.exposure_id,
                        model_attempt_id=request.model_attempt_id,
                        outcome=request.outcome,
                        reason=request.reason,
                    )
                    self._peer_content_exposures[key] = (commitment, receipt)
                    return receipt
                return existing[1].model_copy(update={"replayed": True})
            append = self._peer_content_receipts.get(request.append_key.model_dump_json())
            if append is None or append.status != "appended":
                raise PeerContentUnavailable("Peer content was not durably appended.")
            receipt = PeerContentExposureReceipt(
                operation_key=request.operation_key,
                append_key=request.append_key,
                exposure_id=request.exposure_id,
                model_attempt_id=request.model_attempt_id,
                outcome=request.outcome,
                reason=request.reason,
            )
            self._peer_content_exposures[key] = (commitment, receipt)
            return receipt.model_copy(deep=True)

    async def begin_peer_content_exposure(
        self, request: PeerContentExposureRequest
    ) -> PeerContentExposureReceipt:
        if type(request) is not PeerContentExposureRequest:
            raise TypeError("Exposure requires a PeerContentExposureRequest.")
        identity = request.identity_commitment()
        key = request.exposure_id
        async with self._lock:
            existing = self._peer_content_exposures.get(key)
            if existing is None:
                existing = next(
                    (
                        value
                        for value in self._peer_content_exposures.values()
                        if value[1].operation_key == request.operation_key
                    ),
                    None,
                )
            if existing is not None:
                if existing[1].outcome == "pending" and existing[0] == identity:
                    return existing[1].model_copy(update={"replayed": True})
                if existing[1].outcome != "pending" and existing[0] == request.model_dump_json():
                    return existing[1].model_copy(update={"replayed": True})
                raise PeerContentConflict()
            append = self._peer_content_receipts.get(request.append_key.model_dump_json())
            if append is None or append.status != "appended":
                raise PeerContentUnavailable("Peer content was not durably appended.")
            receipt = PeerContentExposureReceipt(
                operation_key=request.operation_key,
                append_key=request.append_key,
                exposure_id=request.exposure_id,
                model_attempt_id=request.model_attempt_id,
                outcome="pending",
            )
            self._peer_content_exposures[key] = (identity, receipt)
            return receipt.model_copy(deep=True)

    async def read_peer_content_exposure(
        self, append_key: PeerAppendKey, exposure_id: str
    ) -> PeerContentExposureReceipt | None:
        if type(append_key) is not PeerAppendKey or type(exposure_id) is not str:
            raise TypeError("Invalid exposure lookup.")
        append_key = PeerAppendKey.model_validate(append_key)
        async with self._lock:
            value = self._peer_content_exposures.get(exposure_id)
            if value is None or value[1].append_key != append_key:
                return None
            return value[1].model_copy(deep=True)

    async def exclude_peer_content(
        self,
        request: PeerContentAppendRequest,
        *,
        reason: str,
    ) -> PeerContentReceipt:
        from cayu.collaboration.peer_content import PeerContentReceipt

        if type(request) is not PeerContentAppendRequest:
            raise TypeError("Peer exclusion requires a PeerContentAppendRequest.")
        request = PeerContentAppendRequest.model_validate(request)
        reason = require_clean_nonblank(reason, "reason")
        from cayu.collaboration.peer_content import resolve_peer_target

        key = request.append_key.model_dump_json()
        async with self._lock:
            creation = request.append_key.creation_target
            resolve_peer_target(
                request.append_key,
                None if creation is None else self._session_creation_decisions.get(creation.key),
            )
            commitment = request.model_dump_json()
            from cayu.storage._peer_attempts import historical_replay

            replay = historical_replay(
                request, self._peer_content_attempts.get(request.operation_key)
            )
            if replay is not None:
                if replay.status == "excluded" and replay.reason != reason:
                    raise PeerContentConflict()
                return replay
            prior_operation = self._peer_content_operations.get(request.operation_key)
            if prior_operation is not None and prior_operation != commitment:
                raise PeerContentConflict()
            existing = self._peer_content_receipts.get(key)
            if existing is not None:
                if self._peer_content_operation_commitments[key] != commitment:
                    raise PeerContentConflict()
                if existing.status == "excluded" and existing.reason != reason:
                    raise PeerContentConflict()
                if existing.status != "pending":
                    return existing.model_copy(deep=True).model_copy(update={"replayed": True})
            receipt = PeerContentReceipt(
                operation_key=request.operation_key,
                append_key=request.append_key,
                attempt_generation=request.attempt_key.attempt_generation,
                status="excluded",
                reason=reason,
            )
            self._peer_content_receipts[key] = receipt
            self._peer_content_operation_commitments[key] = commitment
            self._peer_content_operations[request.operation_key] = commitment
            self._peer_content_requests[key] = request.model_copy(deep=True)
            self._peer_content_attempts[request.operation_key] = (
                request.model_dump_json(),
                receipt.model_dump_json(),
            )
            return receipt.model_copy(deep=True)

    async def retry_pending_peer_content(
        self,
        session_id: str,
        *,
        expected_session_instance_id: str,
        expected_run_epoch: int,
        expected_transcript_cursor: int,
        admit=None,
    ) -> tuple[PeerContentReceipt, ...]:
        if type(session_id) is not str or type(expected_session_instance_id) is not str:
            raise TypeError("Peer retry identity values must be strings.")
        from cayu.collaboration.peer_content import resolve_peer_target

        def target_matches(request):
            creation = request.append_key.creation_target
            target_id, instance, _ = resolve_peer_target(
                request.append_key,
                None if creation is None else self._session_creation_decisions.get(creation.key),
            )
            return target_id == session_id and instance == expected_session_instance_id

        async with self._lock:
            requests = [
                request.model_copy(deep=True)
                for key, request in self._peer_content_requests.items()
                if target_matches(request)
                and self._peer_content_receipts.get(key) is not None
                and self._peer_content_receipts[key].status == "pending"
            ][:SESSION_MESSAGE_DELIVERY_BATCH_LIMIT]
        results: list[PeerContentReceipt] = []
        for request in requests:
            if (
                request.append_key.creation_target is None
                and request.append_key.target_session_instance_id != expected_session_instance_id
            ):
                continue
            if request.attempt_key.target_run_epoch != expected_run_epoch:
                continue
            if admit is None:
                raise PeerContentUnavailable(
                    "Pending delivery requires fresh export authorization."
                )
            results.append(
                await admit(request, pending_transcript_cursor=expected_transcript_cursor)
            )
        return tuple(results)

    @runtime_session_query
    async def enqueue_session_message(
        self,
        request: EnqueueSessionMessageRequest,
        *,
        expected_authorized_target_instance_id: str | None = None,
    ) -> EnqueueSessionMessageResult:
        request = copy_enqueue_session_message_request(request)
        async with self._lock:
            session = self._sessions.get(request.session_id)
            require_resource_session(session, "modify")
            if session is None:
                raise KeyError(f"Session not found: {request.session_id}")
            if expected_authorized_target_instance_id is not None and (
                type(expected_authorized_target_instance_id) is not str
                or session.instance_id != expected_authorized_target_instance_id
            ):
                raise SessionMessageConflict()
            if request.conditions.source is not None:
                source_session = self._sessions.get(request.conditions.source.session_id)
                require_resource_session(source_session, "read")
                if (
                    source_session is None
                    or source_session.instance_id != request.conditions.source.session_instance_id
                ):
                    raise SessionMessageConflict()
            messages_by_idempotency = self._queued_session_messages_by_idempotency.get(
                request.session_id
            )
            existing = (
                None
                if messages_by_idempotency is None
                else messages_by_idempotency.get(request.idempotency_key)
            )
            if existing is not None:
                session_messaging._validate_equivalent_queued_session_message(existing, request)
                record = self._event_records_by_id.get(
                    (request.session_id, existing.accepted_event_id)
                )
                if record is None:
                    raise RuntimeError(
                        "Queued session message is missing its durable acceptance event."
                    )
                return EnqueueSessionMessageResult(
                    message=existing,
                    event=record.event,
                    replayed=True,
                )
            for owner in self._session_closure_progress.values():
                _check_closure_lineage_owner(owner, (request.session_id,))
            from cayu.runtime._session_message_queue import require_open_admission

            require_open_admission(session.status, self._checkpoints.get(session.id))
            accepted_at = self._ownership_clock()
            queue_id = str(uuid4())
            source = request.conditions.source
            if source is not None and source != self._session_message_source_unlocked(
                source.session_id,
                include_transcript_digest=source.transcript_sha256 is not None,
                include_checkpoint_digest=source.checkpoint_sha256 is not None,
            ):
                raise SessionMessageConflict()
            ordering_key = self._next_session_message_ordering_key
            accepted_transcript_cursor = len(self._transcripts.get(session.id, []))
            accepted_message = enqueue_session_message_input(request)
            from cayu.runtime._session_message_queue import source_event_payload

            accepted_event = event_with_runtime_payload_authority(
                Event(
                    type=EventType.SESSION_MESSAGE_QUEUED,
                    session_id=session.id,
                    agent_name=session.agent_name,
                    environment_name=session.environment_name,
                    timestamp=accepted_at,
                    payload={
                        **session_messaging._queued_session_message_event_payload(
                            queue_id=queue_id,
                            delivery_mode=request.delivery_mode,
                            ordering_key=ordering_key,
                            actor=request.requested_by,
                            run_epoch=session.run_epoch,
                            transcript_cursor=accepted_transcript_cursor,
                        ),
                        **source_event_payload(request.conditions.source),
                        SESSION_STARTED_INPUT_CONTRACT_PAYLOAD_KEY: (
                            session_messages_input_contract_evidence(
                                (accepted_message,),
                                message_start_index=accepted_transcript_cursor,
                                redactions_applied=request._input_redactions_applied,
                                structured_output_requested=False,
                            )
                        ),
                    },
                ),
                SESSION_STARTED_INPUT_CONTRACT_PAYLOAD_KEY,
            )
            queued_message = SessionQueuedMessage(
                queue_id=queue_id,
                session_id=session.id,
                idempotency_key=request.idempotency_key,
                content=request.content,
                message=request.message,
                delivery_mode=request.delivery_mode,
                status=SessionMessageQueueStatus.QUEUED,
                ordering_key=ordering_key,
                accepted_run_epoch=session.run_epoch,
                accepted_transcript_cursor=accepted_transcript_cursor,
                accepted_event_id=accepted_event.id,
                accepted_at=accepted_at,
                requested_by=_audit_resolution_actor(request.requested_by),
                conditions=request.conditions,
            )
            self._sessions[session.id] = self._append_events_unlocked(
                session,
                [accepted_event],
            )
            self._queued_session_messages_by_idempotency.setdefault(session.id, {})[
                queued_message.idempotency_key
            ] = queued_message
            pending_key = (session.id, queued_message.delivery_mode)
            self._pending_session_messages.setdefault(pending_key, deque()).append(queued_message)
            self._next_session_message_ordering_key += 1
            return EnqueueSessionMessageResult(
                message=queued_message,
                event=accepted_event,
            )

    async def deliver_queued_session_messages(
        self,
        session_id: str,
        *,
        include_on_idle: bool,
        delivery_id: str | None = None,
        eligible_through: int | None = None,
        limit: int = SESSION_MESSAGE_DELIVERY_BATCH_LIMIT,
        interaction_id: str | None = None,
        interaction_started_event: Event | None = None,
        profile_handoff: QueuedInteractionProfileHandoff | None = None,
        reject_only: bool = False,
    ) -> SessionMessageDeliveryBatch:
        session_id = require_clean_nonblank(session_id, "session_id")
        delivery_id = (
            str(uuid4())
            if delivery_id is None
            else require_clean_nonblank(delivery_id, "delivery_id")
        )
        if type(include_on_idle) is not bool:
            raise TypeError("include_on_idle must be a bool.")
        if type(reject_only) is not bool:
            raise TypeError("reject_only must be a bool.")
        if interaction_id is not None:
            interaction_id = require_clean_nonblank(interaction_id, "interaction_id")
        interaction_started_event = interaction_rules._copy_queued_interaction_started_event(
            session_id,
            interaction_id,
            interaction_started_event,
        )
        profile_handoff = interaction_rules._copy_queued_interaction_profile_handoff(
            session_id,
            delivery_id,
            interaction_id,
            interaction_started_event,
            profile_handoff,
        )
        eligible_through = _validate_message_delivery_eligible_through(eligible_through)
        if type(limit) is not int or not 1 <= limit <= SESSION_MESSAGE_DELIVERY_BATCH_LIMIT:
            raise ValueError(f"limit must be between 1 and {SESSION_MESSAGE_DELIVERY_BATCH_LIMIT}.")
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            invocation_scope._assert_session_run_epoch(session_id, session)
            existing_delivery = self._session_message_delivery_records.get(delivery_id)
            if existing_delivery is not None:
                _validate_equivalent_message_delivery(
                    existing_delivery,
                    session_id=session_id,
                    include_on_idle=include_on_idle,
                    eligible_through=eligible_through,
                    limit=limit,
                    interaction_id=interaction_id,
                    interaction_started_event=interaction_started_event,
                    profile_handoff=profile_handoff,
                    reject_only=reject_only,
                )
                replayed_batch = existing_delivery.batch.model_copy(
                    update={"replayed": True},
                    deep=True,
                )
                if replayed_batch.messages and profile_handoff is not None:
                    operation_records = self._session_operation_records.get(session_id, {})
                    receipt_record = operation_records.get(
                        interaction_receipts._interaction_transition_storage_key(
                            profile_handoff.predecessor_settlement_event_id
                        )
                    )
                    if receipt_record is None:
                        raise SessionRunFenced(
                            "Queued interaction handoff lost its predecessor settlement receipt."
                        )
                    active_model_stage: ActiveModelCompletionStage | None = None
                    stage_dispatch: ModelCompletionStageDispatch | None = None
                    active_record = operation_records.get(MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY)
                    if active_record is not None:
                        marker = (
                            model_stage_rules._reconstruct_active_model_completion_stage_record(
                                active_record,
                                session_id=session_id,
                            )
                        )
                        _, _, preparation_key, terminal_key = (
                            model_stage_rules._model_completion_stage_storage_identity(
                                session_id,
                                marker.stage_id,
                            )
                        )
                        active_model_stage = (
                            model_stage_rules._reconstruct_active_model_completion_stage(
                                active_record,
                                operation_records.get(preparation_key),
                                operation_records.get(terminal_key),
                                session_id=session_id,
                            )
                        )
                        dispatch_record = operation_records.get(
                            model_stage_rules._model_completion_stage_dispatch_storage_key(
                                marker.stage_id
                            )
                        )
                        if dispatch_record is not None:
                            stage_dispatch = model_stage_rules._reconstruct_model_completion_stage_dispatch(
                                dispatch_record,
                                session_id=session_id,
                                stage_id=marker.stage_id,
                                storage_key=model_stage_rules._model_completion_stage_dispatch_storage_key(
                                    marker.stage_id
                                ),
                            )
                    repaired_checkpoint = (
                        interaction_receipts._checkpoint_after_queued_interaction_profile_handoff(
                            session,
                            self._checkpoints.get(session_id),
                            profile_handoff,
                            settlement_record=receipt_record,
                            replayed_delivery=True,
                            active_model_stage=active_model_stage,
                            stage_dispatch=stage_dispatch,
                        )
                    )
                    self._store_checkpoint_unlocked(session_id, repaired_checkpoint)
                return replayed_batch
            if session.status != SessionStatus.RUNNING:
                raise SessionStatusConflict(
                    "Queued session messages may be delivered only while running."
                )
            boundary = (
                self._next_session_message_ordering_key - 1
                if eligible_through is None
                else eligible_through
            )

            selected_key: tuple[str, SessionMessageDeliveryMode] | None = None
            selected: list[SessionQueuedMessage] = []
            delivered_at = self._ownership_clock()
            transcript_cursor = len(self._transcripts.get(session_id, []))

            def validated_message(message: SessionQueuedMessage) -> SessionQueuedMessage:
                # Reconstruct the complete record, as persistent stores do. Passing
                # the existing model to model_validate would trust mutated fields;
                # serializing it first could emit rejected values in warnings.
                return SessionQueuedMessage.model_validate(
                    {name: getattr(message, name) for name in SessionQueuedMessage.model_fields}
                )

            def rejected_now(message: SessionQueuedMessage) -> bool:
                message = validated_message(message)
                return (
                    session_message_rejection(
                        message.conditions,
                        session_instance_id=session.instance_id,
                        run_epoch=session.run_epoch,
                        transcript_cursor=transcript_cursor,
                        now=delivered_at,
                    )
                    is not None
                )

            delivery_modes: tuple[SessionMessageDeliveryMode, ...] = (
                SessionMessageDeliveryMode.NEXT_TURN,
            )
            if include_on_idle:
                delivery_modes += (SessionMessageDeliveryMode.ON_IDLE,)
            for delivery_mode in delivery_modes:
                pending_key = (session_id, delivery_mode)
                pending = self._pending_session_messages.get(pending_key)
                if pending is None or pending[0].ordering_key > boundary:
                    continue
                selected_key = pending_key
                for message in pending:
                    if len(selected) >= limit or message.ordering_key > boundary:
                        break
                    message = validated_message(message)
                    if reject_only and not rejected_now(message):
                        continue
                    selected.append(message)
                if selected:
                    break
            if not selected:
                batch = SessionMessageDeliveryBatch(
                    delivery_id=delivery_id,
                    interaction_id=interaction_id,
                    eligible_through=boundary,
                    has_more=False,
                )
                self._session_message_delivery_records[delivery_id] = (
                    _InMemoryMessageDeliveryRecord(
                        session_id=session_id,
                        include_on_idle=include_on_idle,
                        requested_eligible_through=eligible_through,
                        limit=limit,
                        interaction_id=interaction_id,
                        interaction_started_event=interaction_started_event,
                        profile_handoff=profile_handoff,
                        batch=batch,
                        reject_only=reject_only,
                    )
                )
                return batch
            if selected_key is None:
                raise RuntimeError("Queued message selection lost its pending index.")

            from cayu.runtime._session_message_queue import (
                source_audit_payload,
                terminal_event,
                terminal_receipt,
            )

            selected_all = selected
            selected_ids = {message.queue_id for message in selected_all}

            def has_eligible(delivery_mode: SessionMessageDeliveryMode) -> bool:
                pending = self._pending_session_messages.get((session_id, delivery_mode), ())
                return any(
                    message.queue_id not in selected_ids
                    and message.ordering_key <= boundary
                    and (not reject_only or rejected_now(message))
                    for message in pending
                )

            # Lookahead can reject malformed rows. Finish it before publishing
            # any events or changing the pending index, including reject-only drains.
            has_more = has_eligible(SessionMessageDeliveryMode.NEXT_TURN) or (
                include_on_idle and has_eligible(SessionMessageDeliveryMode.ON_IDLE)
            )
            accepted_events = {
                message.queue_id: self._session_message_accepted_event_unlocked(
                    self._raw_session_message_unlocked(message)
                )
                for message in selected_all
            }
            selected = []
            rejected_messages: list[SessionQueuedMessage] = []
            rejection_events: list[Event] = []
            rejected_receipts: dict[tuple[str, str], dict[str, Any]] = {}
            for queued_message in selected_all:
                rejection = session_message_rejection(
                    queued_message.conditions,
                    session_instance_id=session.instance_id,
                    run_epoch=session.run_epoch,
                    transcript_cursor=transcript_cursor + len(selected),
                    now=delivered_at,
                )
                if rejection is None:
                    selected.append(queued_message)
                    continue
                event = terminal_event(
                    session,
                    self._raw_session_message_unlocked(queued_message),
                    rejection,
                    delivered_at,
                    accepted_event=accepted_events[queued_message.queue_id],
                    actor=queued_message.requested_by,
                    interaction_id=interaction_id,
                )
                rejection_events.append(event)
                rejected_messages.append(
                    queued_message.model_copy(update={"status": rejection}, deep=True)
                )
                rejected_receipts[(session_id, queued_message.queue_id)] = terminal_receipt(
                    rejection, event
                )

            rebound_checkpoint: dict[str, Any] | None = None
            if profile_handoff is not None and selected:
                self._reject_new_work_after_steering_unlocked(
                    session, allow_completed_interaction=True
                )
                receipt_record = self._session_operation_records.get(session_id, {}).get(
                    interaction_receipts._interaction_transition_storage_key(
                        profile_handoff.predecessor_settlement_event_id
                    )
                )
                if receipt_record is None:
                    raise SessionRunFenced(
                        "Queued interaction handoff lost its predecessor settlement receipt."
                    )
                rebound_checkpoint = (
                    interaction_receipts._checkpoint_after_queued_interaction_profile_handoff(
                        session,
                        self._checkpoints.get(session_id),
                        profile_handoff,
                        settlement_record=receipt_record,
                        replayed_delivery=False,
                    )
                )

            updated_messages: list[SessionQueuedMessage] = []
            delivery_events: list[Event] = []
            transcript_messages: list[Message] = []
            for offset, queued_message in enumerate(selected, start=1):
                delivered_cursor = transcript_cursor + offset
                delivered_message = queued_session_message_input(queued_message)
                delivery_event = event_with_runtime_payload_authority(
                    Event(
                        type=EventType.SESSION_MESSAGE_DELIVERED,
                        session_id=session.id,
                        interaction_id=interaction_id,
                        agent_name=session.agent_name,
                        environment_name=session.environment_name,
                        timestamp=delivered_at,
                        payload={
                            **session_messaging._queued_session_message_event_payload(
                                queue_id=queued_message.queue_id,
                                delivery_mode=queued_message.delivery_mode,
                                ordering_key=queued_message.ordering_key,
                                actor=queued_message.requested_by,
                                run_epoch=session.run_epoch,
                                transcript_cursor=delivered_cursor,
                            ),
                            **source_audit_payload(
                                self._raw_session_message_unlocked(queued_message),
                                accepted_events[queued_message.queue_id],
                            ),
                            "accepted_run_epoch": queued_message.accepted_run_epoch,
                            "accepted_transcript_cursor": (
                                queued_message.accepted_transcript_cursor
                            ),
                            SESSION_STARTED_INPUT_CONTRACT_PAYLOAD_KEY: (
                                session_messages_input_contract_evidence(
                                    (delivered_message,),
                                    message_start_index=delivered_cursor - 1,
                                    redactions_applied=False,
                                    structured_output_requested=False,
                                )
                            ),
                        },
                    ),
                    SESSION_STARTED_INPUT_CONTRACT_PAYLOAD_KEY,
                )
                updated_messages.append(
                    queued_message.model_copy(
                        update={
                            "status": SessionMessageQueueStatus.DELIVERED,
                            "delivered_run_epoch": session.run_epoch,
                            "delivered_transcript_cursor": delivered_cursor,
                            "delivered_event_id": delivery_event.id,
                            "delivered_at": delivered_at,
                        },
                        deep=True,
                    )
                )
                delivery_events.append(delivery_event)
                transcript_messages.append(delivered_message)

            persisted_events = [
                *(
                    [interaction_started_event]
                    if interaction_started_event is not None and selected
                    else []
                ),
                *sorted(
                    [*delivery_events, *rejection_events],
                    key=lambda event: event.payload["ordering_key"],
                ),
            ]
            batch = SessionMessageDeliveryBatch(
                messages=tuple(updated_messages),
                events=tuple(persisted_events),
                delivery_id=delivery_id,
                interaction_id=interaction_id,
                eligible_through=boundary,
                has_more=has_more,
                active_invocation_profile=(
                    None
                    if profile_handoff is None or not selected
                    else profile_handoff.target_active_profile
                ),
            )
            delivery_record = _InMemoryMessageDeliveryRecord(
                session_id=session_id,
                include_on_idle=include_on_idle,
                requested_eligible_through=eligible_through,
                limit=limit,
                interaction_id=interaction_id,
                interaction_started_event=interaction_started_event,
                profile_handoff=profile_handoff,
                batch=batch,
                reject_only=reject_only,
            )
            selected_queue = deque(
                message
                for message in self._pending_session_messages[selected_key]
                if message.queue_id not in selected_ids
            )
            updated_session = self._append_events_unlocked(session, persisted_events)
            self._transcripts.setdefault(session_id, []).extend(transcript_messages)
            self._extend_transcript_search_unlocked(session_id, transcript_messages)
            self._transcript_interaction_ids.setdefault(session_id, [])
            self._extend_transcript_attribution_unlocked(
                session_id, [interaction_id] * len(transcript_messages)
            )
            self._pending_session_messages[selected_key] = selected_queue
            if not selected_queue:
                del self._pending_session_messages[selected_key]
            messages_by_idempotency = self._queued_session_messages_by_idempotency[session_id]
            for updated_message in [*updated_messages, *rejected_messages]:
                messages_by_idempotency[updated_message.idempotency_key] = updated_message
            self._session_message_terminal_receipts.update(rejected_receipts)
            self._sessions[session_id] = updated_session
            if rebound_checkpoint is not None:
                self._store_checkpoint_unlocked(session_id, rebound_checkpoint)
            self._session_message_delivery_records[delivery_id] = delivery_record
            return batch

    async def repair_queued_interaction_profile_handoff(
        self,
        session_id: str,
        *,
        interaction_started_event: Event,
        profile_handoff: QueuedInteractionProfileHandoff,
    ) -> ActiveInvocationExecutionProfile:
        session_id = require_clean_nonblank(session_id, "session_id")
        interaction_started_event, profile_handoff = (
            interaction_rules._copy_historical_queued_interaction_profile_handoff(
                session_id,
                interaction_started_event,
                profile_handoff,
            )
        )
        target = profile_handoff.target_active_profile
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            invocation_scope._assert_session_run_epoch(session_id, session)
            delivery = self._session_message_delivery_records.get(target.interaction_id)
            if (
                delivery is None
                or delivery.session_id != session_id
                or delivery.interaction_id != target.interaction_id
                or delivery.interaction_started_event != interaction_started_event
                or not delivery.batch.messages
                or (
                    delivery.profile_handoff is not None
                    and delivery.profile_handoff != profile_handoff
                )
            ):
                raise SessionRunFenced(
                    "Historical queued interaction handoff lacks its exact delivery receipt."
                )
            operation_records = self._session_operation_records.get(session_id, {})
            settlement_record = operation_records.get(
                interaction_receipts._interaction_transition_storage_key(
                    profile_handoff.predecessor_settlement_event_id
                )
            )
            if settlement_record is None:
                raise SessionRunFenced(
                    "Historical queued interaction handoff lost its predecessor settlement."
                )
            active_model_stage, stage_dispatch = (
                interaction_receipts._historical_queued_handoff_stage_from_records(
                    session_id,
                    operation_records,
                )
            )
            repaired_checkpoint = (
                interaction_receipts._checkpoint_after_queued_interaction_profile_handoff(
                    session,
                    self._checkpoints.get(session_id),
                    profile_handoff,
                    settlement_record=settlement_record,
                    replayed_delivery=True,
                    active_model_stage=active_model_stage,
                    stage_dispatch=stage_dispatch,
                )
            )
            self._store_checkpoint_unlocked(session_id, repaired_checkpoint)
            return target.model_copy(deep=True)

    async def publish_checkpoint_and_events(
        self,
        session_id: str,
        *,
        checkpoint_transform: CheckpointTransform,
        events: list[Event],
        expected_statuses: set[SessionStatus] | None = None,
        expected_run_epoch: int | None = None,
        expected_transcript_cursor: int | None = None,
    ) -> Session:
        return await self._publish_checkpoint_and_events(
            session_id,
            checkpoint_transform=checkpoint_transform,
            events=events,
            expected_statuses=expected_statuses,
            expected_run_epoch=expected_run_epoch,
            expected_transcript_cursor=expected_transcript_cursor,
            preserve_completion_result_publications=True,
        )

    async def _publish_completion_result_event_publication(
        self,
        session_id: str,
        *,
        checkpoint_transform: StoreTimeCheckpointTransform,
        events: list[Event],
    ) -> Session:
        return await self._publish_checkpoint_and_events(
            session_id,
            checkpoint_transform=None,
            store_time_checkpoint_transform=checkpoint_transform,
            events=events,
            expected_statuses=None,
            expected_run_epoch=None,
            expected_transcript_cursor=None,
            preserve_completion_result_publications=False,
        )

    async def _publish_checkpoint_and_events(
        self,
        session_id: str,
        *,
        checkpoint_transform: CheckpointTransform | None,
        store_time_checkpoint_transform: StoreTimeCheckpointTransform | None = None,
        events: list[Event],
        expected_statuses: set[SessionStatus] | None,
        expected_run_epoch: int | None,
        expected_transcript_cursor: int | None,
        preserve_completion_result_publications: bool,
    ) -> Session:
        session_id, copied_events = session_event_rules._copy_session_event_batch(
            session_id, events
        )
        if (checkpoint_transform is None) == (store_time_checkpoint_transform is None):
            raise TypeError("Exactly one checkpoint transform is required.")
        allowed_statuses = (
            None
            if expected_statuses is None
            else session_record_rules._validate_status_set(expected_statuses, "expected_statuses")
        )
        async with self._lock:
            store_now = self._ownership_clock()
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            invocation_scope._assert_session_run_epoch(session_id, session)
            for owner in self._session_closure_progress.values():
                _check_closure_lineage_owner(owner, (session_id,))
            if allowed_statuses is not None and session.status not in allowed_statuses:
                raise SessionStatusConflict(
                    f"Session status is not eligible for checkpoint publication: {session.status}"
                )
            if expected_run_epoch is not None and session.run_epoch != expected_run_epoch:
                raise SessionRunFenced(
                    f"Session source run epoch is stale: expected {expected_run_epoch}, "
                    f"current {session.run_epoch}."
                )
            current_cursor = len(self._transcripts.get(session_id, []))
            if (
                expected_transcript_cursor is not None
                and current_cursor != expected_transcript_cursor
            ):
                raise ValueError(
                    "Session source transcript cursor is stale: expected "
                    f"{expected_transcript_cursor}, current {current_cursor}."
                )
            current = self._checkpoints.get(session_id)
            callback_session = session.model_copy(deep=True)
            callback_checkpoint = checkpoint_preservation._copy_checkpoint_for_transform(
                current,
                session_id=session_id,
            )
            if checkpoint_transform is not None:
                transformed = checkpoint_transform(
                    callback_session,
                    callback_checkpoint,
                )
            else:
                assert store_time_checkpoint_transform is not None
                transformed = store_time_checkpoint_transform(
                    callback_session,
                    callback_checkpoint,
                    store_now,
                )
            if transformed is None:
                raise ValueError("Checkpoint transform must return a checkpoint.")
            copied_checkpoint = copy_durable_json_object(transformed, "checkpoint")
            copied_checkpoint = checkpoint_preservation._replace_checkpoint_preserving_completion_result_event_publications(
                current,
                copied_checkpoint,
                preserve_completion_result_publications=(preserve_completion_result_publications),
                session_id=session_id,
            )
            updated = self._append_events_unlocked(session, copied_events)
            if not copied_events:
                updated = updated.model_copy(update={"last_activity_at": store_now})
            self._store_checkpoint_unlocked(session_id, copied_checkpoint)
            self._sessions[session_id] = updated
            return updated.model_copy(deep=True)

    async def load_session_operation(
        self,
        session_id: str,
        idempotency_key: str,
        *,
        checkpoint_root_guard: CheckpointRootFieldGuard | None = None,
    ) -> dict[str, Any] | None:
        from cayu.resource_access import current_data_bounds

        access_bounds = await current_data_bounds()
        session_id = require_clean_nonblank(session_id, "session_id")
        idempotency_key = operation_record_rules._reject_reserved_runtime_publication_key(
            idempotency_key,
            "idempotency_key",
            browser_control_read=True,
        )
        _require_session_export_target(session_id, idempotency_key)
        async with self._lock:
            if access_bounds is not None:
                access_bounds.require_read(self._sessions.get(session_id))
            if session_id not in self._sessions:
                raise KeyError(f"Session not found: {session_id}")
            if checkpoint_root_guard is not None:
                checkpoint_root_guard.validate(
                    session_id,
                    checkpoint_root_field_projection(
                        self._checkpoints.get(session_id),
                        checkpoint_root_guard.key,
                    ),
                )
            record = self._session_operation_records.get(session_id, {}).get(idempotency_key)
            return None if record is None else copy_durable_json_object(record, "session_operation")

    async def _load_historical_interaction_settlement_record(
        self, session_id: str, event_id: str
    ) -> dict[str, Any] | None:
        key = interaction_receipts._interaction_transition_storage_key(event_id)
        async with self._lock:
            record = self._session_operation_records.get(session_id, {}).get(key)
            return None if record is None else copy_durable_json_object(record, "settlement")

    async def _load_runtime_publication_receipt_record(
        self,
        session_id: str,
        storage_key: str,
        publication_id: str,
    ) -> dict[str, Any] | None:
        async with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session not found: {session_id}")
            record = self._session_operation_records.get(session_id, {}).get(storage_key)
            if record is None:
                return None
            copied = deepcopy(record)
            receipt = publication_rules._reconstruct_runtime_publication_receipt(
                copied,
                storage_key=storage_key,
                session_id=session_id,
                publication_id=publication_id,
            )
            self._validate_runtime_publication_material_unlocked(receipt)
            return copied

    def _validate_runtime_publication_material_unlocked(
        self,
        receipt: RuntimePublicationReceipt,
    ) -> None:
        transcript = self._transcripts.get(receipt.session_id, [])
        transcript_segment = transcript[
            receipt.transcript_start_cursor : receipt.transcript_end_cursor
        ]
        transcript_interaction_ids = self._transcript_interaction_ids.get(
            receipt.session_id,
            [],
        )[receipt.transcript_start_cursor : receipt.transcript_end_cursor]
        referenced_event_ids = publication_contracts._runtime_publication_referenced_event_ids(
            receipt.referenced_events
        )
        requested_event_ids = set((*receipt.appended_event_ids, *referenced_event_ids))
        events_by_id = {}
        for event_id in requested_event_ids:
            record = self._event_records_by_id.get((receipt.session_id, event_id))
            if record is not None:
                events_by_id[event_id] = record.event
        publication_rules._validate_runtime_publication_durable_material(
            receipt,
            transcript_messages=transcript_segment,
            transcript_interaction_ids=transcript_interaction_ids,
            appended_events=(
                events_by_id[event_id]
                for event_id in receipt.appended_event_ids
                if event_id in events_by_id
            ),
            durable_referenced_events=(
                events_by_id[event_id]
                for event_id in referenced_event_ids
                if event_id in events_by_id
            ),
        )

    async def publish_session_operation(
        self,
        session_id: str,
        *,
        idempotency_key: str,
        operation_transform: SessionOperationTransform,
        events: list[Event],
        expected_statuses: set[SessionStatus] | None = None,
        expected_run_epoch: int | None = None,
        expected_transcript_cursor: int | None = None,
    ) -> Session:
        return await self._publish_session_operation(
            session_id,
            idempotency_key=idempotency_key,
            operation_transform=operation_transform,
            store_time_operation_transform=None,
            commit_guard=None,
            commit_time_guard=None,
            events=events,
            expected_statuses=expected_statuses,
            expected_run_epoch=expected_run_epoch,
            expected_transcript_cursor=expected_transcript_cursor,
        )

    async def publish_session_operation_guarded(
        self,
        session_id: str,
        *,
        idempotency_key: str,
        operation_transform: SessionOperationTransform,
        commit_guard: Callable[[], None],
        events: list[Event],
        expected_statuses: set[SessionStatus] | None = None,
        expected_run_epoch: int | None = None,
        expected_transcript_cursor: int | None = None,
    ) -> Session:
        return await self._publish_session_operation(
            session_id,
            idempotency_key=idempotency_key,
            operation_transform=operation_transform,
            store_time_operation_transform=None,
            commit_guard=commit_guard,
            commit_time_guard=None,
            events=events,
            expected_statuses=expected_statuses,
            expected_run_epoch=expected_run_epoch,
            expected_transcript_cursor=expected_transcript_cursor,
        )

    async def publish_session_operation_guarded_with_store_time(
        self,
        session_id: str,
        *,
        idempotency_key: str,
        operation_transform: StoreTimeSessionOperationTransform,
        commit_guard: Callable[[], None],
        commit_time_guard: Callable[[datetime], None],
        events: list[Event],
        expected_statuses: set[SessionStatus] | None = None,
        expected_run_epoch: int | None = None,
        expected_transcript_cursor: int | None = None,
        context_view_compaction_cursor: int | None = None,
    ) -> Session:
        return await self._publish_session_operation(
            session_id,
            idempotency_key=idempotency_key,
            operation_transform=None,
            store_time_operation_transform=operation_transform,
            commit_guard=commit_guard,
            commit_time_guard=commit_time_guard,
            events=events,
            expected_statuses=expected_statuses,
            expected_run_epoch=expected_run_epoch,
            expected_transcript_cursor=expected_transcript_cursor,
            context_view_compaction_cursor=context_view_compaction_cursor,
        )

    async def _prepare_temporary_side_service(self, preparation):
        from cayu.sessions._side_service_preparation import (
            SidePreparationSnapshot,
            plan_preparation,
            preparation_record_keys,
            prepare_selection,
        )

        prepared = prepare_selection(preparation)
        keys = preparation_record_keys(prepared)
        async with self._lock:
            snapshots = {}
            for session_id in sorted(keys):
                session = self._sessions.get(session_id)
                if session is None:
                    raise KeyError("Side-service session is unavailable.")
                invocation_scope._assert_session_run_epoch(session_id, session)
                for owner in self._session_closure_progress.values():
                    _check_closure_lineage_owner(owner, (session_id,))
                snapshots[session_id] = SidePreparationSnapshot(
                    session=session,
                    checkpoint=self._checkpoints.get(session_id),
                    records={
                        key: value
                        for key in keys[session_id]
                        if (value := self._session_operation_records.get(session_id, {}).get(key))
                        is not None
                    },
                )
            now = self._ownership_clock()
            plans = plan_preparation(prepared, snapshots, now)
            sessions = {
                session_id: snapshots[session_id].session.model_copy(
                    update={"updated_at": now, "last_activity_at": now}
                )
                for session_id in plans
            }
            operations = {
                session_id: self._session_operation_records.get(session_id, {})
                | plan.operation_records
                for session_id, plan in plans.items()
            }
            # Planning verifies that only the continuation root changes. Other
            # checkpoint owners' material and derived indexes remain unchanged.
            for session_id, plan in plans.items():
                self._checkpoints[session_id] = plan.checkpoint
                self._session_operation_records[session_id] = operations[session_id]
                self._sessions[session_id] = sessions[session_id]

    async def _publish_session_operation(
        self,
        session_id: str,
        *,
        idempotency_key: str,
        operation_transform: SessionOperationTransform | None,
        store_time_operation_transform: StoreTimeSessionOperationTransform | None,
        commit_guard: Callable[[], None] | None,
        commit_time_guard: Callable[[datetime], None] | None,
        events: list[Event],
        expected_statuses: set[SessionStatus] | None,
        expected_run_epoch: int | None,
        expected_transcript_cursor: int | None,
        context_view_compaction_cursor: int | None = None,
    ) -> Session:
        session_id, copied_events = session_event_rules._copy_session_event_batch(
            session_id, events
        )
        idempotency_key = operation_record_rules._reject_reserved_runtime_publication_key(
            idempotency_key,
            "idempotency_key",
        )
        if (operation_transform is None) == (store_time_operation_transform is None):
            raise TypeError("Exactly one session operation transform is required.")
        _require_session_export_target(session_id, idempotency_key)
        allowed_statuses = (
            None
            if expected_statuses is None
            else session_record_rules._validate_status_set(expected_statuses, "expected_statuses")
        )
        async with self._lock:
            store_now = self._ownership_clock()
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            invocation_scope._assert_session_run_epoch(session_id, session)
            for owner in self._session_closure_progress.values():
                _check_closure_lineage_owner(owner, (session_id,))
            if allowed_statuses is not None and session.status not in allowed_statuses:
                raise SessionStatusConflict(
                    f"Session status is not eligible for checkpoint publication: {session.status}"
                )
            if expected_run_epoch is not None and session.run_epoch != expected_run_epoch:
                raise SessionRunFenced(
                    f"Session source run epoch is stale: expected {expected_run_epoch}, "
                    f"current {session.run_epoch}."
                )
            current_cursor = len(self._transcripts.get(session_id, []))
            if context_view_compaction_cursor is not None:
                self._validate_context_view_compaction_unlocked(
                    session_id, context_view_compaction_cursor
                )
            if (
                expected_transcript_cursor is not None
                and current_cursor != expected_transcript_cursor
            ):
                raise ValueError(
                    "Session source transcript cursor is stale: expected "
                    f"{expected_transcript_cursor}, current {current_cursor}."
                )
            current_checkpoint = self._checkpoints.get(session_id)
            current_record = self._session_operation_records.get(session_id, {}).get(
                idempotency_key
            )
            from cayu.collaboration import _session_export_store as session_exports
            from cayu.sessions import _producer_checkpoint as producers
            from cayu.sessions import _session_continuation_store as continuations

            callback_session = session.model_copy(deep=True)
            callback_checkpoint = (
                None if current_checkpoint is None else deepcopy(current_checkpoint)
            )
            # Keep the raw operation callback's existing view of sibling state;
            # only the newly private export root changes visibility here.
            if callback_checkpoint is not None and not session_exports.checkpoint_visible(
                session_id=session_id
            ):
                callback_checkpoint.pop(session_exports.ROOT_KEY, None)
            if callback_checkpoint is not None and not continuations.checkpoint_visible():
                callback_checkpoint.pop(continuations.ROOT_KEY, None)
            if callback_checkpoint is not None and not producers.checkpoint_visible(
                session_id=session_id
            ):
                callback_checkpoint.pop(producers.ROOT_KEY, None)
            callback_record = (
                None
                if current_record is None
                else copy_durable_json_object(current_record, "session_operation")
            )
            if operation_transform is not None:
                publication = operation_transform(
                    callback_session,
                    callback_checkpoint,
                    callback_record,
                )
            else:
                assert store_time_operation_transform is not None
                publication = store_time_operation_transform(
                    callback_session,
                    callback_checkpoint,
                    callback_record,
                    store_now,
                )
            if type(publication) is not SessionOperationPublication:
                raise TypeError(
                    "Session operation transform must return a SessionOperationPublication."
                )
            copied_checkpoint = copy_durable_json_object(
                publication.checkpoint,
                "checkpoint",
            )
            copied_checkpoint = checkpoint_preservation._checkpoint_transform_result_preserving_completion_result_event_publications(
                current_checkpoint,
                copied_checkpoint,
                session_id=session_id,
            )
            copied_records = copy_durable_json_object(
                publication.operation_records,
                "operation_records",
            )
            operation_record_rules._validate_session_operation_record_keys(copied_records)

            indices = session_exports.selected_transcript_indices(session_id=session_id)
            if (
                indices
                or session_exports.checkpoint_visible(session_id=session_id)
                or any(key.startswith(session_exports.OPERATION_PREFIX) for key in copied_records)
            ):
                messages = self._transcripts.get(session_id, [])
                interaction_ids = self._transcript_interaction_ids.get(session_id, [])
                session_exports.validate_publication(
                    session=session,
                    current_checkpoint=current_checkpoint,
                    proposed_checkpoint=copied_checkpoint,
                    operation_records=copied_records,
                    selected_transcript_rows=tuple(
                        TranscriptRecord(
                            index=index,
                            interaction_id=interaction_ids[index],
                            message=detach_message(messages[index]),
                        )
                        for index in sorted(set(indices))
                        if 0 <= index < len(messages)
                    ),
                    events=copied_events,
                )
            release = publication.model_completion_stage_release
            if release is not None:
                operation_records = self._session_operation_records.get(session_id, {})
                _, _, preparation_key, terminal_key = (
                    model_stage_rules._model_completion_stage_storage_identity(
                        session_id, release.stage_id
                    )
                )
                model_stage_rules._validate_model_completion_stage_release(
                    session=session,
                    active_record=operation_records.get(MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY),
                    preparation_record=operation_records.get(preparation_key),
                    terminal_record=operation_records.get(terminal_key),
                    release=release,
                )
            commit_guard_signal = (
                None
                if commit_guard is None
                else await _run_session_commit_guard_owned(commit_guard)
            )
            commit_at = self._ownership_clock()
            if commit_time_guard is not None:
                commit_time_guard(commit_at)
            updated = self._append_events_unlocked(session, copied_events)
            if not copied_events:
                updated = updated.model_copy(update={"last_activity_at": commit_at})
            self._store_checkpoint_unlocked(session_id, copied_checkpoint)
            operation_records = self._session_operation_records.setdefault(session_id, {})
            operation_records.update(copied_records)
            if release is not None:
                del operation_records[MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY]
            self._sessions[session_id] = updated
            if commit_guard_signal is not None:
                raise commit_guard_signal
            return updated.model_copy(deep=True)

    async def _load_model_completion_stage_records(
        self,
        session_id: str,
        preparation_storage_key: str,
        terminal_storage_key: str,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        async with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session not found: {session_id}")
            records = self._session_operation_records[session_id]
            preparation = records.get(preparation_storage_key)
            terminal = records.get(terminal_storage_key)
            return (
                None if preparation is None else deepcopy(preparation),
                None if terminal is None else deepcopy(terminal),
            )

    async def _load_model_completion_stage_settlement_record(
        self,
        session_id: str,
        settlement_storage_key: str,
    ) -> dict[str, Any] | None:
        async with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session not found: {session_id}")
            record = self._session_operation_records[session_id].get(settlement_storage_key)
            return None if record is None else deepcopy(record)

    async def _load_model_completion_stage_dispatch_record(
        self,
        session_id: str,
        dispatch_storage_key: str,
    ) -> dict[str, Any] | None:
        async with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session not found: {session_id}")
            record = self._session_operation_records[session_id].get(dispatch_storage_key)
            return None if record is None else deepcopy(record)

    async def _load_active_model_completion_stage_records(
        self,
        session_id: str,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
        async with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session not found: {session_id}")
            records = self._session_operation_records[session_id]
            active_record = records.get(MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY)
            if active_record is None:
                return None, None, None
            marker = model_stage_rules._reconstruct_active_model_completion_stage_record(
                active_record,
                session_id=session_id,
            )
            _, _, preparation_key, terminal_key = (
                model_stage_rules._model_completion_stage_storage_identity(
                    session_id,
                    marker.stage_id,
                )
            )
            preparation = records.get(preparation_key)
            terminal = records.get(terminal_key)
            return (
                deepcopy(active_record),
                None if preparation is None else deepcopy(preparation),
                None if terminal is None else deepcopy(terminal),
            )

    async def _mark_model_completion_stage_dispatched_atomic(
        self,
        session_id: str,
        *,
        stage: ModelCompletionStage,
        consume_child_session_notifications: bool,
    ) -> ModelCompletionStageDispatch:
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            records = self._session_operation_records[session_id]
            _, _, preparation_key, terminal_key = (
                model_stage_rules._model_completion_stage_storage_identity(
                    session_id,
                    stage.stage_id,
                )
            )
            settlement_key = model_stage_rules._model_completion_stage_settlement_storage_key(
                stage.stage_id
            )
            dispatch_key = model_stage_rules._model_completion_stage_dispatch_storage_key(
                stage.stage_id
            )
            model_stage_rules._validate_model_completion_stage_for_dispatch(
                session=session,
                checkpoint=self._checkpoints.get(session_id),
                current_transcript_cursor=len(self._transcripts.get(session_id, [])),
                stage=stage,
                active_record=records.get(MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY),
                preparation_record=records.get(preparation_key),
                terminal_record=records.get(terminal_key),
                settlement_record=records.get(settlement_key),
            )
            dispatch_record = records.get(dispatch_key)
            dispatch_is_new = dispatch_record is None
            if dispatch_is_new:
                published_at = publication_rules._next_runtime_publication_timestamp(session)
                dispatch_record = model_stage_rules._model_completion_stage_dispatch_record(
                    stage,
                    dispatched_at=published_at,
                )
            else:
                published_at = publication_rules._next_runtime_publication_timestamp(session)
            assert dispatch_record is not None
            dispatch = model_stage_rules._reconstruct_model_completion_stage_dispatch(
                dispatch_record,
                session_id=session_id,
                stage_id=stage.stage_id,
                storage_key=dispatch_key,
            )
            model_stage_rules._validate_model_completion_stage_dispatch(dispatch, stage)
            binding = child_session_notification_stage_binding(stage.intent)
            prepared_consumptions: list[tuple[str, dict[str, Any], str, str]] = []
            if binding is not None:
                for claim in binding.claims:
                    child = self._sessions.get(claim.child_session_id)
                    if child is None:
                        raise SessionModelCompletionStageConflict(
                            "Child-session notification occurrence is no longer canonical."
                        )
                    occurrence_source = _child_session_lifecycle_occurrence(
                        child,
                        self._child_lifecycle_event_records.get(child.id, {}),
                    )
                    if occurrence_source is None:
                        raise SessionModelCompletionStageConflict(
                            "Child-session notification occurrence is no longer canonical."
                        )
                    _relationship, occurrence = occurrence_source
                    consumption = _child_session_notification_consumption_record(
                        parent=session,
                        child=child,
                        occurrence=occurrence,
                        stage=stage,
                        consumed_at=published_at,
                    )
                    storage_key = child_session_notification_storage_key(
                        child.instance_id,
                        occurrence.source_id,
                    )
                    durable = records.get(storage_key)
                    material = consumption.model_dump(mode="json")
                    if durable is not None and not _child_session_notification_consumption_replays(
                        durable,
                        consumption,
                    ):
                        raise SessionModelCompletionStageConflict(
                            "Child-session terminal notification was consumed by another stage."
                        )
                    prepared_consumptions.append(
                        (storage_key, material, child.id, occurrence.source_id)
                    )
            if dispatch_is_new:
                records[dispatch_key] = dispatch_record
            if consume_child_session_notifications:
                for storage_key, material, _child_id, _occurrence_id in prepared_consumptions:
                    records.setdefault(storage_key, material)
            for _storage_key, _material, child_id, _occurrence_id in (
                prepared_consumptions if consume_child_session_notifications else ()
            ):
                child = self._sessions.get(child_id)
                if child is None:
                    raise RuntimeError("Consumed child-session notification disappeared.")
                self._refresh_child_lifecycle_candidate_unlocked(child)
            self._sessions[session_id] = session.model_copy(
                update={"updated_at": published_at, "last_activity_at": published_at},
                deep=True,
            )
            return dispatch

    def _reject_new_work_after_steering_unlocked(
        self, session: Session, *, allow_completed_interaction: bool = False
    ) -> None:
        from cayu.runtime._session_steering import (
            reject_new_work_after_steering,
            steering_operation_key_from_checkpoint,
        )

        checkpoint = self._checkpoints.get(session.id)
        key = steering_operation_key_from_checkpoint(session, checkpoint)
        if key is not None:
            reject_new_work_after_steering(
                session,
                checkpoint,
                self._session_operation_records[session.id].get(key),
                allow_completed_interaction=allow_completed_interaction,
            )

    async def _prepare_model_completion_stage_atomic(
        self,
        prepared: model_stage_rules._PreparedModelCompletionStage,
    ) -> ModelCompletionStageResult:
        session_id = prepared.session_id
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            records = self._session_operation_records[session_id]
            stage = model_stage_rules._reconstruct_model_completion_stage(
                records.get(prepared.preparation_storage_key),
                records.get(prepared.terminal_storage_key),
                session_id=session_id,
                stage_id=prepared.request.stage_id,
                preparation_storage_key=prepared.preparation_storage_key,
                terminal_storage_key=prepared.terminal_storage_key,
            )
            model_stage_rules._validate_model_completion_stage_repreparation(
                records.get(prepared.abandonment_storage_key),
                prepared,
                source_status=session.status if stage is None else stage.source_status,
            )
            if stage is not None:
                model_stage_rules._validate_model_completion_stage_preparation_replay(
                    stage, prepared
                )
                active = model_stage_rules._reconstruct_active_model_completion_stage_from_records(
                    records,
                    session_id=session_id,
                )
                model_stage_rules._validate_model_completion_preparation_replay_state(
                    stage,
                    active=active,
                    winner_exists=records.get(prepared.winner_storage_key) is not None,
                    receipt_exists=(records.get(prepared.publication_storage_key) is not None),
                )
                model_failover_rules._model_failover_preparation_checkpoint(
                    prepared,
                    session=session,
                    checkpoint=self._checkpoints.get(session_id),
                    current_transcript_cursor=len(self._transcripts.get(session_id, [])),
                    active=active,
                    records=records,
                    replayed=True,
                )
                expected_selection = model_failover_rules._model_failover_selection_event(
                    prepared, session=session, prepared_at=stage.prepared_at
                )
                if expected_selection is not None:
                    event_record = self._event_records_by_id.get(
                        (session_id, expected_selection.id)
                    )
                    model_failover_rules._validate_model_failover_selection_replay(
                        expected_selection, None if event_record is None else event_record.event
                    )
                return ModelCompletionStageResult(
                    stage=stage,
                    replayed=True,
                    dispatch_authorized=False,
                )

            active = model_stage_rules._reconstruct_active_model_completion_stage_from_records(
                records,
                session_id=session_id,
            )
            model_stage_rules._validate_model_completion_active_marker_for_preparation(
                active,
                prepared,
                source_status=session.status,
            )
            retry_settlement_request = model_stage_rules._model_completion_retry_settlement_request(
                active,
                prepared,
            )
            if (
                records.get(prepared.winner_storage_key) is not None
                or records.get(prepared.publication_storage_key) is not None
            ):
                raise SessionModelCompletionStageConflict(
                    "The logical model step already has durable publication state."
                )

            invocation_scope._assert_session_run_epoch(session_id, session)
            if session.status not in prepared.expected_statuses:
                raise SessionStatusConflict(
                    "Session status is not eligible for model-completion preparation: "
                    f"{session.status}"
                )
            if session.run_epoch != prepared.expected_run_epoch:
                raise SessionRunFenced(
                    "Session source run epoch is stale: expected "
                    f"{prepared.expected_run_epoch}, current {session.run_epoch}."
                )
            current_cursor = len(self._transcripts.get(session_id, []))
            if current_cursor != prepared.expected_transcript_cursor:
                raise ValueError(
                    "Session source transcript cursor is stale: expected "
                    f"{prepared.expected_transcript_cursor}, current {current_cursor}."
                )

            if active is None:
                self._reject_new_work_after_steering_unlocked(session)
            route_checkpoint = model_failover_rules._model_failover_preparation_checkpoint(
                prepared,
                session=session,
                checkpoint=self._checkpoints.get(session_id),
                current_transcript_cursor=current_cursor,
                active=active,
                records=records,
                replayed=False,
            )
            prepared_at = publication_rules._next_runtime_publication_timestamp(session)
            preparation_record = model_stage_rules._model_completion_stage_preparation_record(
                prepared,
                source_session=session,
                prepared_at=prepared_at,
            )
            selection_event = model_failover_rules._model_failover_selection_event(
                prepared, session=session, prepared_at=prepared_at
            )
            prepared_selection = (
                None
                if selection_event is None
                else self._prepare_event_append_unlocked(session, (selection_event,))
            )
            stage = model_stage_rules._reconstruct_model_completion_stage(
                preparation_record,
                None,
                session_id=session_id,
                stage_id=prepared.request.stage_id,
                preparation_storage_key=prepared.preparation_storage_key,
                terminal_storage_key=prepared.terminal_storage_key,
            )
            assert stage is not None
            active_record = model_stage_rules._active_model_completion_stage_record(
                stage,
                activated_at=prepared_at,
            )
            retry_settlement_storage_key = None
            retry_settlement_record = None
            if retry_settlement_request is not None:
                assert active is not None
                retry_settlement_storage_key = (
                    model_stage_rules._model_completion_stage_settlement_storage_key(
                        active.stage.stage_id
                    )
                )
                model_stage_rules._validate_model_completion_stage_for_settlement(
                    session=session,
                    stage=active.stage,
                    active=active,
                    request=retry_settlement_request,
                    settlement_record=records.get(retry_settlement_storage_key),
                    winner_exists=(
                        records.get(
                            model_stage_rules._model_completion_stage_winner_storage_key(
                                active.stage.logical_step_id
                            )
                        )
                        is not None
                    ),
                    receipt_exists=(
                        records.get(
                            publication_contracts._runtime_publication_storage_key(
                                active.stage.logical_step_id
                            )
                        )
                        is not None
                    ),
                )
                retry_settlement_record = (
                    model_stage_rules._model_completion_stage_settlement_record(
                        active.stage,
                        request=retry_settlement_request,
                        settled_at=prepared_at,
                    )
                )
            updated_session = session.model_copy(
                update={"updated_at": prepared_at, "last_activity_at": prepared_at},
                deep=True,
            )
            if retry_settlement_storage_key is not None and retry_settlement_record is not None:
                records[retry_settlement_storage_key] = retry_settlement_record
            records[prepared.preparation_storage_key] = preparation_record
            records[MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY] = active_record
            if route_checkpoint is not None:
                # Only the route root changed; pending-action/queue projections
                # remain identical and need no separate mutation owner.
                self._checkpoints[session_id] = route_checkpoint
            if prepared_selection is not None:
                updated_session = self._apply_event_append_unlocked(
                    updated_session, prepared_selection, activity_at=prepared_at
                )
            self._sessions[session_id] = updated_session
            return ModelCompletionStageResult(
                stage=stage,
                replayed=False,
                dispatch_authorized=True,
                prepared_events=() if selection_event is None else (selection_event,),
            )

    async def _complete_model_completion_stage_atomic(
        self,
        prepared: model_stage_rules._PreparedModelCompletionStageTerminal,
    ) -> ModelCompletionStageResult:
        session_id = prepared.session_id
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            records = self._session_operation_records[session_id]
            stage = model_stage_rules._reconstruct_model_completion_stage(
                records.get(prepared.preparation_storage_key),
                records.get(prepared.terminal_storage_key),
                session_id=session_id,
                stage_id=prepared.stage_id,
                preparation_storage_key=prepared.preparation_storage_key,
                terminal_storage_key=prepared.terminal_storage_key,
            )
            if stage is None:
                raise KeyError(f"Model-completion stage not found: {prepared.stage_id}")
            model_stage_rules._reject_settled_model_completion_stage(
                records.get(prepared.settlement_storage_key),
                session_id=session_id,
                stage_id=prepared.stage_id,
                settlement_storage_key=prepared.settlement_storage_key,
            )
            if stage.state == "completed":
                model_stage_rules._validate_model_completion_stage_terminal_replay(stage, prepared)
                return ModelCompletionStageResult(
                    stage=stage,
                    replayed=True,
                    dispatch_authorized=False,
                )
            if prepared.recovery_fence is not None:
                model_stage_rules._validate_model_completion_stage_recovery_fence(
                    prepared.recovery_fence,
                    session=session,
                    checkpoint=self._checkpoints.get(session_id),
                    stage=stage,
                    active_record=records.get(MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY),
                    dispatch_record=records.get(
                        model_stage_rules._model_completion_stage_dispatch_storage_key(
                            stage.stage_id
                        )
                    ),
                    now=self._ownership_clock(),
                )
            model_stage_publication._validate_model_completion_stage_publication(
                prepared.publication,
                session_id=session_id,
                stage=stage,
            )
            if not session_authority_rules._runtime_publication_json_equal(
                prepared.publication.intent, stage.intent
            ):
                raise SessionModelCompletionStageConflict(
                    "The terminal model completion intent conflicts with its preparation."
                )
            completed_at = publication_rules._next_runtime_publication_timestamp(session)
            terminal_record = model_stage_rules._model_completion_stage_terminal_record(
                prepared,
                stage=stage,
                completed_at=completed_at,
            )
            completed_stage = model_stage_rules._reconstruct_model_completion_stage(
                records[prepared.preparation_storage_key],
                terminal_record,
                session_id=session_id,
                stage_id=prepared.stage_id,
                preparation_storage_key=prepared.preparation_storage_key,
                terminal_storage_key=prepared.terminal_storage_key,
            )
            assert completed_stage is not None
            advances_last_activity = (
                model_stage_rules._model_completion_terminal_advances_last_activity(
                    records.get(MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY),
                    stage=stage,
                    current_run_epoch=session.run_epoch,
                )
            )
            records[prepared.terminal_storage_key] = terminal_record
            self._sessions[session_id] = session.model_copy(
                update={
                    "updated_at": completed_at,
                    "last_activity_at": (
                        completed_at if advances_last_activity else session.last_activity_at
                    ),
                },
                deep=True,
            )
            return ModelCompletionStageResult(
                stage=completed_stage,
                replayed=False,
                dispatch_authorized=False,
            )

    async def _abandon_model_completion_stage_atomic(
        self,
        prepared: model_stage_rules._PreparedModelCompletionStageAbandonment,
    ) -> ModelCompletionStageAbandonmentResult:
        session_id = prepared.session_id
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            records = self._session_operation_records[session_id]
            stage = model_stage_rules._reconstruct_model_completion_stage(
                records.get(prepared.preparation_storage_key),
                records.get(prepared.terminal_storage_key),
                session_id=session_id,
                stage_id=prepared.stage_id,
                preparation_storage_key=prepared.preparation_storage_key,
                terminal_storage_key=prepared.terminal_storage_key,
            )
            if stage is None:
                active_record = records.get(MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY)
                if active_record is not None:
                    active_marker = (
                        model_stage_rules._reconstruct_active_model_completion_stage_record(
                            active_record,
                            session_id=session_id,
                        )
                    )
                    if active_marker.stage_id == prepared.stage_id:
                        raise SessionModelCompletionStageConflict(
                            "The active model-completion marker references a missing stage."
                        )
                replayed = model_stage_rules._replay_model_completion_stage_abandonment(
                    prepared,
                    records.get(prepared.abandonment_storage_key),
                )
                logical_step_id = replayed.abandonment.logical_step_id
                if (
                    records.get(
                        model_stage_rules._model_completion_stage_winner_storage_key(
                            logical_step_id
                        )
                    )
                    is not None
                    or records.get(
                        publication_contracts._runtime_publication_storage_key(logical_step_id)
                    )
                    is not None
                ):
                    raise SessionModelCompletionStageConflict(
                        "An abandoned model-completion stage has durable publication state."
                    )
                return replayed

            active = model_stage_rules._reconstruct_active_model_completion_stage_from_records(
                records,
                session_id=session_id,
            )
            winner_storage_key = model_stage_rules._model_completion_stage_winner_storage_key(
                stage.logical_step_id
            )
            publication_storage_key = publication_contracts._runtime_publication_storage_key(
                stage.logical_step_id
            )
            model_stage_rules._validate_model_completion_stage_for_abandonment(
                session=session,
                stage=stage,
                active=active,
                prepared=prepared,
                abandonment_record=records.get(prepared.abandonment_storage_key),
                dispatch_exists=(
                    records.get(
                        model_stage_rules._model_completion_stage_dispatch_storage_key(
                            stage.stage_id
                        )
                    )
                    is not None
                ),
                winner_exists=records.get(winner_storage_key) is not None,
                receipt_exists=records.get(publication_storage_key) is not None,
            )
            assert active is not None
            abandoned_at = publication_rules._next_runtime_publication_timestamp(session)
            abandonment_record = model_stage_rules._model_completion_stage_abandonment_record(
                stage,
                active=active,
                abandoned_at=abandoned_at,
            )
            abandonment = model_stage_rules._reconstruct_model_completion_stage_abandonment(
                abandonment_record,
                session_id=session_id,
                stage_id=prepared.stage_id,
                storage_key=prepared.abandonment_storage_key,
            )

            records[prepared.abandonment_storage_key] = abandonment_record
            del records[prepared.preparation_storage_key]
            del records[MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY]
            self._sessions[session_id] = session.model_copy(
                update={"updated_at": abandoned_at, "last_activity_at": abandoned_at},
                deep=True,
            )
            return ModelCompletionStageAbandonmentResult(
                abandonment=abandonment,
                replayed=False,
            )

    async def _promote_model_completion_stage_atomic(
        self,
        *,
        session_id: str,
        stage_id: str,
        preparation_storage_key: str,
        terminal_storage_key: str,
        expected_run_epoch: int,
    ) -> RuntimePublicationResult:
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            records = self._session_operation_records[session_id]
            stage = model_stage_rules._reconstruct_model_completion_stage(
                records.get(preparation_storage_key),
                records.get(terminal_storage_key),
                session_id=session_id,
                stage_id=stage_id,
                preparation_storage_key=preparation_storage_key,
                terminal_storage_key=terminal_storage_key,
            )
            if stage is None:
                raise KeyError(f"Model-completion stage not found: {stage_id}")

            publication_storage_key = publication_contracts._runtime_publication_storage_key(
                stage.logical_step_id
            )
            winner_storage_key = model_stage_rules._model_completion_stage_winner_storage_key(
                stage.logical_step_id
            )
            receipt_record = records.get(publication_storage_key)
            if receipt_record is not None:
                active = model_stage_rules._reconstruct_active_model_completion_stage_from_records(
                    records,
                    session_id=session_id,
                )
                model_stage_rules._validate_model_completion_promotion_replay_active_marker(
                    active, stage
                )
                receipt = publication_rules._reconstruct_runtime_publication_receipt(
                    deepcopy(receipt_record),
                    storage_key=publication_storage_key,
                    session_id=session_id,
                    publication_id=stage.logical_step_id,
                )
                self._validate_runtime_publication_material_unlocked(receipt)
                return model_stage_rules._replay_promoted_model_completion_stage(
                    session=session,
                    stage=stage,
                    receipt_record=deepcopy(receipt_record),
                    winner_record=deepcopy(records.get(winner_storage_key)),
                )
            if records.get(winner_storage_key) is not None:
                raise SessionModelCompletionStageConflict(
                    "A model-completion winner exists without its runtime publication receipt."
                )
            active = model_stage_rules._reconstruct_active_model_completion_stage_from_records(
                records,
                session_id=session_id,
            )
            model_stage_rules._validate_model_completion_active_marker_for_promotion(active, stage)
            prepared = model_stage_rules._prepare_model_completion_stage_promotion(
                stage,
                expected_run_epoch=expected_run_epoch,
            )
            result = self._publish_runtime_publication_unlocked(
                prepared,
                session=session,
                model_completion_stage=stage,
            )
            return result

    async def _publish_runtime_publication_atomic(
        self,
        prepared: publication_rules._PreparedRuntimePublication,
    ) -> RuntimePublicationResult:
        async with self._lock:
            session = self._sessions.get(prepared.session_id)
            if session is None:
                raise KeyError(f"Session not found: {prepared.session_id}")
            return self._publish_runtime_publication_unlocked(
                prepared,
                session=session,
            )

    def _publish_runtime_publication_unlocked(
        self,
        prepared: publication_rules._PreparedRuntimePublication,
        *,
        session: Session,
        model_completion_stage: ModelCompletionStage | None = None,
    ) -> RuntimePublicationResult:
        checkpoint_codec = prepared.checkpoint_codec
        checkpoint_decode = None if checkpoint_codec is None else checkpoint_codec.decode
        session_id = prepared.session_id
        request = prepared.request
        operation_records = self._session_operation_records[session_id]
        existing_record = operation_records.get(prepared.storage_key)
        if existing_record is not None:
            receipt = publication_rules._reconstruct_runtime_publication_receipt(
                deepcopy(existing_record),
                storage_key=prepared.storage_key,
                session_id=session_id,
                publication_id=request.publication_id,
                request_digest=prepared.request_digest,
            )
            publication_rules._validate_runtime_publication_replay_receipt(receipt, prepared)
            self._validate_runtime_publication_material_unlocked(receipt)
            return RuntimePublicationResult(
                session=session.model_copy(deep=True),
                receipt=receipt,
                replayed=True,
            )

        for owner in self._session_closure_progress.values():
            _check_closure_lineage_owner(owner, (session_id,))

        mutated_operation_records = (
            operation_records
            if not request.operation_record_mutations
            else operation_record_rules._apply_runtime_publication_operation_record_mutations(
                request.operation_record_mutations,
                operation_records,
            )
        )

        private_record = None
        if request.argument_continuity is not None:
            from cayu.sessions._argument_continuity import STORAGE_KEY, append_record

            private_record = append_record(
                operation_records.get(STORAGE_KEY),
                continuity=request.argument_continuity,
                request_digest=prepared.request_digest,
                session=session,
                messages=request.transcript_messages,
            )

        invocation_scope._assert_session_run_epoch(session_id, session)
        if (
            prepared.expected_statuses is not None
            and session.status not in prepared.expected_statuses
        ):
            raise SessionStatusConflict(
                f"Session status is not eligible for runtime publication: {session.status}"
            )
        if (
            prepared.expected_run_epoch is not None
            and session.run_epoch != prepared.expected_run_epoch
        ):
            raise SessionRunFenced(
                "Session source run epoch is stale: expected "
                f"{prepared.expected_run_epoch}, current {session.run_epoch}."
            )
        transcript_start_cursor = len(self._transcripts.get(session_id, []))
        if (
            prepared.expected_transcript_cursor is not None
            and transcript_start_cursor != prepared.expected_transcript_cursor
        ):
            raise ValueError(
                "Session source transcript cursor is stale: expected "
                f"{prepared.expected_transcript_cursor}, "
                f"current {transcript_start_cursor}."
            )

        appended_event_ids = {event.id for event in request.events}
        referenced_event_ids = set(
            publication_contracts._runtime_publication_referenced_event_ids(
                request.referenced_events
            )
        )
        if appended_event_ids & referenced_event_ids:
            raise ValueError("Appended and referenced runtime publication events overlap.")
        durable_events_by_id = {}
        for event_id in referenced_event_ids:
            record = self._event_records_by_id.get((session_id, event_id))
            if record is not None:
                durable_events_by_id[event_id] = record.event
        publication_contracts._validate_runtime_publication_event_references(
            request.referenced_events,
            durable_events_by_id,
            interaction_id=request.interaction_id,
        )
        stored_checkpoint = self._checkpoints.get(session_id)
        current_checkpoint = (
            stored_checkpoint
            if checkpoint_decode is None
            else checkpoint_decode(session, deepcopy(stored_checkpoint))
        )
        user_input_publication_rules._validate_user_input_checkpoint_mutation(
            request,
            current_checkpoint,
            session_id=session_id,
            session_instance_id=session.instance_id,
            current_run_epoch=session.run_epoch,
            durable_events_by_id=durable_events_by_id,
        )
        tool_publication_rules._validate_tool_round_checkpoint_mutation(request, current_checkpoint)
        durable_tool_events: list[Event] = []
        tool_round_identity = tool_publication_rules._tool_lifecycle_publication_identity(request)
        if tool_round_identity is not None:
            from cayu.sessions.pending_actions import pending_action_lookup_key

            execution_identity, tool_call_ids = tool_round_identity
            event_history = self._pending_action_event_history.get(session_id, {})
            for tool_call_id in tool_call_ids:
                lookup_key = pending_action_lookup_key(tool_call_id)
                durable_tool_events.extend(
                    record.event
                    for record in event_history.get(lookup_key, ())
                    if record.event.type in tool_publication_rules._TOOL_ROUND_LIFECYCLE_EVENT_TYPES
                    and tool_publication_rules._tool_round_event_matches_scope_or_is_ambiguous(
                        record.event,
                        execution_identity,
                    )
                )
            if len(durable_tool_events) > publication_contracts._tool_round_lifecycle_event_limit(
                tool_call_ids
            ):
                raise ValueError("Tool-round lifecycle evidence exceeds the publication limit.")
        tool_publication_rules._validate_tool_round_publication(
            request,
            durable_events_by_id,
            durable_tool_events=durable_tool_events,
        )
        existing_event_ids = appended_event_ids & self._event_ids[session_id]
        if existing_event_ids:
            raise ValueError(
                f"Event already exists for session {session_id}: {min(existing_event_ids)}"
            )

        if checkpoint_codec is None:
            checkpoint = checkpoint_mutation_rules._apply_runtime_publication_checkpoint_mutation(
                request.mutation,
                current_checkpoint,
            )
            stored_target_checkpoint = checkpoint
        else:
            checkpoint = checkpoint_codec.apply_mutation(
                session,
                deepcopy(stored_checkpoint),
                request.mutation,
            )
            stored_target_checkpoint = checkpoint_codec.encode(session, checkpoint)
        if prepared.storage_key in operation_records:
            raise SessionRuntimePublicationConflict(
                "Runtime publication receipt appeared while the session lock was held."
            )
        prepared_events = self._prepare_event_append_unlocked(
            session,
            request.events,
            events_are_detached=True,
        )
        prepared_checkpoint = (
            None
            if stored_target_checkpoint is None or not request.mutation.operations
            else self._prepare_checkpoint_store_unlocked(
                session_id,
                stored_target_checkpoint,
                additional_event_records=tuple(event.record for event in prepared_events.events),
            )
        )

        published_at = publication_rules._next_runtime_publication_timestamp(session)
        receipt = publication_rules._build_runtime_publication_receipt(
            prepared,
            source_session=session,
            checkpoint=stored_target_checkpoint,
            transcript_start_cursor=transcript_start_cursor,
            published_at=published_at,
        )
        receipt_record = publication_rules._runtime_publication_receipt_record(receipt)
        winner_record = (
            None
            if model_completion_stage is None
            else model_stage_rules._model_completion_stage_winner_record(
                model_completion_stage,
                receipt=receipt,
            )
        )

        updated = self._apply_event_append_unlocked(
            session,
            prepared_events,
            activity_at=published_at,
        ).model_copy(
            update={
                "updated_at": published_at,
                "last_activity_at": published_at,
            }
        )
        self._transcripts[session_id].extend(request.transcript_messages)
        self._extend_transcript_search_unlocked(session_id, request.transcript_messages)
        self._extend_transcript_attribution_unlocked(
            session_id, [request.interaction_id] * len(request.transcript_messages)
        )
        if prepared_checkpoint is not None:
            self._apply_checkpoint_store_unlocked(session_id, prepared_checkpoint)
        if private_record is not None:
            mutated_operation_records[STORAGE_KEY] = private_record
        mutated_operation_records[prepared.storage_key] = receipt_record
        if model_completion_stage is not None:
            assert winner_record is not None
            winner_storage_key = model_stage_rules._model_completion_stage_winner_storage_key(
                model_completion_stage.logical_step_id
            )
            mutated_operation_records[winner_storage_key] = winner_record
            mutated_operation_records.pop(MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY)
        self._session_operation_records[session_id] = mutated_operation_records
        self._sessions[session_id] = updated
        return RuntimePublicationResult(
            session=updated.model_copy(deep=True),
            receipt=receipt,
            replayed=False,
        )

    async def load_session_export_snapshot(
        self,
        session_id: str,
        *,
        limits: SessionExportLimits | None = None,
    ) -> SessionExportSnapshot | None:
        from cayu.sessions.exports import SessionExportBuilder

        session_id = require_clean_nonblank(session_id, "session_id")
        builder = SessionExportBuilder(limits)
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            for record in self._session_event_records.get(session_id, ()):
                builder.event(record)
            for index, message in enumerate(self._transcripts.get(session_id, ())):
                builder.message(
                    TranscriptRecord(
                        index=index,
                        message=message,
                        interaction_id=self._transcript_interaction_ids[session_id][index],
                    )
                )
            codec = self.public_authority_alias_codec
            records = []
            uses = []
            for grant_id in self._targeted_tool_grant_ids_by_session.get(session_id, ()):
                if codec is None:
                    raise RuntimeError(
                        "Exporting targeted grants requires an authority alias codec."
                    )
                record = targeted_tool_grant_with_active_reference(
                    self._targeted_tool_grants[grant_id],
                    codec,
                )
                builder.charge(record.model_dump(mode="json"))
                records.append(record)
                for binding in self._targeted_tool_uses.get(grant_id, {}).values():
                    builder.charge(binding.model_dump(mode="json"))
                    uses.append(binding.model_copy(deep=True))
            records.sort(key=lambda record: (record.issued_at, record.grant_id))
            uses.sort(key=lambda binding: (binding.bound_at, binding.use_id))
            deferred = self._deferred_interaction_inputs.get(session_id)
            checkpoint = self._checkpoints.get(session_id)
            return builder.finish(
                session=session,
                transcript_cursor=len(self._transcripts.get(session_id, ())),
                checkpoint=None
                if checkpoint is None
                else copy_durable_json_object(checkpoint, "checkpoint"),
                deferred=None if deferred is None else deferred.model_copy(deep=True),
                grants_charged=True,
                grants=TargetedToolGrantStateSnapshot(records=tuple(records), uses=tuple(uses)),
            )

    async def _complete_native_producer_cleanup(self, registration, *, authority):
        from cayu.storage._producer_cleanup import memory_cleanup

        if not self._supports_producer_attachment_protocol():
            raise NotImplementedError("Native producer cleanup is not qualified.")
        return await memory_cleanup(self, registration, authority=authority, commit=True)

    async def _retire_native_producer_cleanup(self, retirement, *, authority, limit):
        from cayu.storage._producer_retirement import memory_retirement

        if not self._supports_producer_attachment_protocol():
            raise NotImplementedError("Native producer retirement is not qualified.")
        return await memory_retirement(self, retirement, authority=authority, limit=limit)

    async def _read_completed_native_producer_cleanup(self, registration):
        from cayu.storage._producer_cleanup import memory_cleanup

        if not self._supports_producer_attachment_protocol():
            raise NotImplementedError("Native producer cleanup readback is not qualified.")
        return await memory_cleanup(self, registration)

    async def _read_native_producer_release(self, command):
        from cayu.storage._producer_observation import memory_observation

        if not self._supports_producer_attachment_protocol():
            raise NotImplementedError("Native producer release readback is not qualified.")
        return await memory_observation(self, command)

    async def _read_native_producer_attachment(self, command):
        from cayu.storage._producer_observation import memory_observation

        if not self._supports_producer_attachment_protocol():
            raise NotImplementedError("Native producer attachment readback is not qualified.")
        return await memory_observation(self, command, attachment_only=True)

    async def _read_native_producer_progress(self, command, *, kind):
        from cayu.storage._producer_observation import memory_observation

        if not self._supports_producer_attachment_protocol():
            raise NotImplementedError("Native producer progress readback is not qualified.")
        return await memory_observation(self, command, kind=kind)

    async def load_events(self, session_id: str) -> list[Event]:
        from cayu.resource_access import current_data_bounds

        access_bounds = await current_data_bounds()
        session_id = require_clean_nonblank(session_id, "session_id")
        async with self._lock:
            if access_bounds is not None:
                access_bounds.require_read(self._sessions.get(session_id))
            if session_id not in self._sessions:
                raise KeyError(f"Session not found: {session_id}")
            return [event.model_copy(deep=True) for event in self._events.get(session_id, [])]

    async def load_user_input_supersession_events(
        self,
        session_id: str,
        input_id: str,
    ) -> list[Event]:
        from cayu.sessions.pending_actions import pending_action_lookup_key

        session_id = require_clean_nonblank(session_id, "session_id")
        input_id = require_clean_nonblank(input_id, "input_id")
        lookup_key = pending_action_lookup_key(input_id)
        async with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session not found: {session_id}")
            records = self._user_input_supersession_event_records.get(
                (session_id, lookup_key),
                (),
            )
            return [copy_event(record.event) for record in records[:2]]

    async def load_tool_round_lifecycle_events(
        self,
        session_id: str,
        tool_call_ids: list[str] | tuple[str, ...],
    ) -> list[Event]:
        from cayu.sessions.pending_actions import pending_action_lookup_key

        session_id = require_clean_nonblank(session_id, "session_id")
        copied_ids = tool_publication_rules._validate_tool_round_call_ids(
            tool_call_ids, "tool_call_ids"
        )
        async with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session not found: {session_id}")
            history = self._pending_action_event_history.get(session_id, {})
            records = [
                record
                for tool_call_id in copied_ids
                for record in history.get(pending_action_lookup_key(tool_call_id), ())
                if record.event.type in tool_publication_rules._TOOL_ROUND_LIFECYCLE_EVENT_TYPES
            ]
            records.sort(key=lambda record: record.sequence)
            if len(records) > publication_contracts._tool_round_lifecycle_event_limit(copied_ids):
                raise ValueError("Tool-round lifecycle evidence exceeds the publication limit.")
            return [copy_event(record.event) for record in records]

    async def load_tool_round_lifecycle_events_for_round(
        self,
        session_id: str,
        tool_call_ids: list[str] | tuple[str, ...],
        *,
        tool_round_identity: ToolRoundIdentity,
    ) -> list[Event]:
        from cayu.sessions.pending_actions import pending_action_lookup_key

        session_id = require_clean_nonblank(session_id, "session_id")
        copied_ids = tool_publication_rules._validate_tool_round_call_ids(
            tool_call_ids, "tool_call_ids"
        )
        tool_round_identity = copy_tool_round_identity(tool_round_identity)
        async with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session not found: {session_id}")
            history = self._pending_action_event_history.get(session_id, {})
            records = [
                record
                for tool_call_id in copied_ids
                for record in history.get(pending_action_lookup_key(tool_call_id), ())
                if record.event.type in tool_publication_rules._TOOL_ROUND_LIFECYCLE_EVENT_TYPES
                and tool_publication_rules._tool_round_event_matches_scope_or_is_ambiguous(
                    record.event,
                    tool_round_identity,
                )
            ]
            records.sort(key=lambda record: record.sequence)
            if len(records) > publication_contracts._tool_round_lifecycle_event_limit(copied_ids):
                raise ValueError("Tool-round lifecycle evidence exceeds the publication limit.")
            return [copy_event(record.event) for record in records]

    @runtime_session_query
    async def query_events(self, query: EventQuery | None = None) -> list[EventRecord]:
        query = copy_event_query(query)
        async with self._lock:
            return self._query_events_unlocked(query, max_bytes=None)

    @runtime_session_query
    async def event_exists(self, query: EventQuery) -> bool:
        query = copy_event_query(query)
        event_types = frozenset(
            (str(query.event_type),)
            if query.event_type is not None
            else map(str, query.event_types)
        )
        excluded_types = frozenset(map(str, query.exclude_event_types))
        async with self._lock:
            records = (
                self._session_event_records.get(query.session_id, [])
                if query.session_id is not None
                else self._event_records
            )
            start = bisect_right(records, query.after_sequence or 0, key=lambda row: row.sequence)
            for index in range(start, len(records)):
                record = records[index]
                if query.before_sequence is not None and record.sequence >= query.before_sequence:
                    break
                if event_query_rules._event_record_matches(
                    record, query, event_types, excluded_types
                ) and _event_record_matches_session(record, query, self._sessions):
                    return True
            return False

    @runtime_session_query
    async def read_usage_accounting(
        self, query: EventQuery, *, by_session: bool = False, by_identity: bool = False
    ) -> UsageAccountingSnapshot:
        from cayu.runtime._usage_accounting import (
            USAGE_ACCOUNTING_PAGE_SIZE,
            SessionUsageCache,
            UsageAccountingReducer,
            usage_accounting_query,
        )

        query = usage_accounting_query(query)
        cached_session_id = SessionUsageCache.session_scope(
            query, by_session=by_session, by_identity=by_identity
        )
        async with self._lock:
            generation = self._accounting_generation
            prior = (
                None
                if cached_session_id is None
                else self._session_usage_cache.resume_after(cached_session_id, generation)
            )
            if prior is not None:
                query = copy_event_query(query, update={"after_sequence": prior.scanned_through})
            reducer = UsageAccountingReducer(query, by_session=by_session, by_identity=by_identity)
            event_types = frozenset(str(kind) for kind in query.event_types)
            # Reuse durable indexes without merging complete per-type/session lists.
            candidates = (
                self._session_event_records.get(query.session_id, [])
                if query.session_id is not None
                else self._event_records
            )
            start = bisect_right(
                candidates, query.after_sequence or 0, key=lambda row: row.sequence
            )
            page: list[EventRecord] = []
            for index in range(start, len(candidates)):
                record = candidates[index]
                if query.before_sequence is not None and record.sequence >= query.before_sequence:
                    break
                if not event_query_rules._event_record_matches(
                    record, query, event_types, frozenset()
                ):
                    continue
                if not _event_record_matches_session(record, query, self._sessions):
                    continue
                page.append(record)
                if len(page) == USAGE_ACCOUNTING_PAGE_SIZE:
                    reducer.add_page(page)
                    page.clear()
                    await asyncio.sleep(0)
            if page:
                reducer.add_page(page)
            if cached_session_id is not None:
                return self._session_usage_cache.settle(
                    cached_session_id,
                    generation,
                    prior,
                    reducer.snapshot(),
                    boundary=candidates[-1].sequence if candidates else 0,
                )
        return reducer.snapshot().model_copy(update={"generation": generation})

    @runtime_session_query
    async def read_cost_accounting(
        self,
        query: EventQuery,
        pricing: PriceBook,
        *,
        currency: str = "USD",
        details: bool = False,
        by_session: bool = False,
        additional_events: tuple[Event, ...] = (),
        max_detail_bytes: int | None = None,
        previous: CostAccountingSnapshot | None = None,
    ) -> CostAccountingSnapshot:
        from cayu.runtime._cost_accounting import (
            COST_ACCOUNTING_PAGE_SIZE,
            cost_accounting_query,
            cost_pending_events,
        )
        from cayu.runtime._cost_accounting_refresh import CostAccountingRead

        query = cost_accounting_query(query)
        pending = cost_pending_events(query, additional_events)
        event_types = frozenset(str(kind) for kind in query.event_types)
        async with self._lock:
            if query.causal_budget_id is not None:
                pending = tuple(
                    event
                    for event in pending
                    if (session := self._sessions.get(event.session_id)) is not None
                    and session.causal_budget_id == query.causal_budget_id
                )
            pending = tuple(
                event
                for event in pending
                if (record := self._event_records_by_id.get((event.session_id, event.id))) is None
                or not event_query_rules._event_record_matches(
                    record, query, event_types, frozenset()
                )
            )
            boundary_session_id = query.session_id or (
                query.session_ids[0] if len(query.session_ids) == 1 else None
            )
            boundary_records = (
                self._session_event_records.get(boundary_session_id, [])
                if boundary_session_id is not None
                else self._event_records
            )
            boundary_end = (
                bisect_left(boundary_records, query.before_sequence, key=lambda row: row.sequence)
                if query.before_sequence is not None
                else len(boundary_records)
            )
            through_sequence = boundary_records[boundary_end - 1].sequence if boundary_end else 0
            reducer = CostAccountingRead(
                query,
                pricing,
                currency=currency,
                details=details,
                max_detail_bytes=max_detail_bytes,
                previous=previous,
                generation=self._accounting_generation,
                through_sequence=through_sequence,
                authority=self._cost_accounting_authority,
                by_session=by_session,
                additional_events=pending,
            )
            scanned = 0
            for key, records in self._cost_event_index.groups(reducer):
                if query.session_id is not None and key[0] != query.session_id:
                    continue
                if query.session_ids and key[0] not in query.session_ids:
                    continue
                if (
                    query.causal_budget_id is not None
                    and self._sessions[key[0]].causal_budget_id != query.causal_budget_id
                ):
                    continue
                for hosted_phase in (True, False):
                    for record in records:
                        scanned += 1
                        if scanned % COST_ACCOUNTING_PAGE_SIZE == 0:
                            await asyncio.sleep(0)
                        if (record.event.type == EventType.MODEL_HOSTED_TOOL_CALL) != hosted_phase:
                            continue
                        if event_query_rules._event_record_matches(
                            record, reducer.source_query, event_types, frozenset()
                        ) and _event_record_matches_session(record, query, self._sessions):
                            reducer.add(record.sequence, record.event)
            return reducer.snapshot()

    @runtime_session_query
    async def query_events_bounded(
        self,
        query: EventQuery,
        *,
        max_bytes: int,
    ) -> list[EventRecord]:
        query = copy_event_query(query)
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError("max_bytes must be a positive integer.")
        async with self._lock:
            return self._query_events_unlocked(query, max_bytes=max_bytes)

    async def load_terminal_session_evidence(
        self,
        session_id: str,
        *,
        limits: TerminalSessionEvidenceLimits | None = None,
    ) -> TerminalSessionEvidence:
        return await self._load_terminal_session_evidence(
            session_id,
            limits=limits,
            observed_interrupted_events=None,
            expected_interrupted_parent_session_id=None,
            require_interrupted_proof=False,
        )

    async def load_runner_owned_interrupted_evidence(
        self,
        session_id: str,
        *,
        observed_events: tuple[RunnerObservedEventIdentity, ...] | None = None,
        expected_parent_session_id: str | None = None,
        limits: TerminalSessionEvidenceLimits | None = None,
    ) -> TerminalSessionEvidence:
        return await self._load_terminal_session_evidence(
            session_id,
            limits=limits,
            observed_interrupted_events=observed_events,
            expected_interrupted_parent_session_id=expected_parent_session_id,
            require_interrupted_proof=True,
        )

    async def _load_terminal_session_evidence(
        self,
        session_id: str,
        *,
        limits: TerminalSessionEvidenceLimits | None,
        observed_interrupted_events: tuple[RunnerObservedEventIdentity, ...] | None,
        expected_interrupted_parent_session_id: str | None,
        require_interrupted_proof: bool,
    ) -> TerminalSessionEvidence:
        session_id = require_clean_nonblank(session_id, "session_id")
        limits = terminal_session_evidence._copy_terminal_session_evidence_limits(limits)
        observed, expected_parent_session_id = (
            terminal_session_evidence._copy_runner_owned_interruption_proof(
                session_id,
                observed_events=observed_interrupted_events,
                expected_parent_session_id=expected_interrupted_parent_session_id,
                limits=limits,
                required=require_interrupted_proof,
            )
        )
        allow_interrupted = observed is not None or expected_parent_session_id is not None
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise TerminalSessionEvidenceError(
                    TerminalSessionEvidenceErrorCode.SESSION_NOT_FOUND
                )
            terminal_session_evidence._terminal_session_evidence_expected_event_type(
                session.status,
                allow_interrupted=allow_interrupted,
            )
            if allow_interrupted and session.status != SessionStatus.INTERRUPTED:
                raise TerminalSessionEvidenceError(
                    TerminalSessionEvidenceErrorCode.EVIDENCE_INCONSISTENT
                )
            if (
                expected_parent_session_id is not None
                and session.parent_session_id != expected_parent_session_id
            ):
                raise TerminalSessionEvidenceError(
                    TerminalSessionEvidenceErrorCode.EVIDENCE_INCONSISTENT
                )
            checkpoint = self._checkpoints.get(session_id)
            marker = terminal_session_evidence._terminal_session_evidence_marker_from_checkpoint(
                checkpoint
            )
            session_records = self._session_event_records.get(session_id, [])
            if observed is not None:
                if len(session_records) != len(observed):
                    raise TerminalSessionEvidenceError(
                        TerminalSessionEvidenceErrorCode.EVIDENCE_INCONSISTENT
                    )
                terminal_session_evidence._validate_runner_observed_event_identity_snapshot(
                    observed,
                    tuple(
                        RunnerObservedEventIdentity(
                            session_id=record.event.session_id,
                            sequence=record.sequence,
                            event_type=record.event.type,
                        )
                        for record in session_records
                    ),
                )
            newest_evidence_records = tuple(
                islice(
                    (
                        record
                        for record in reversed(session_records)
                        if record.event.type in TERMINAL_EVIDENCE_EVENT_TYPES
                    ),
                    TERMINAL_EVIDENCE_QUERY_LIMIT,
                )
            )
            terminal_record = terminal_session_evidence._classify_terminal_session_evidence_records(
                session_id=session.id,
                status=session.status,
                run_epoch=session.run_epoch,
                marker=marker,
                newest_evidence_records=newest_evidence_records,
                initial_transcript_pending=(
                    checkpoint is not None
                    and INITIAL_TRANSCRIPT_PENDING_CHECKPOINT_KEY in checkpoint
                ),
                pending_session_interrupt=(
                    checkpoint is not None and "pending_session_interrupt" in checkpoint
                ),
                allow_interrupted=allow_interrupted,
            )
            event_stop = bisect_right(
                session_records,
                terminal_record.sequence,
                key=lambda record: record.sequence,
            )
            if event_stop > limits.max_events:
                raise TerminalSessionEvidenceError(
                    TerminalSessionEvidenceErrorCode.EVENT_LIMIT_EXCEEDED,
                    limit=limits.max_events,
                    observed=event_stop,
                )
            events = session_records[:event_stop]

            messages = self._transcripts.get(session_id, [])
            interaction_ids = self._transcript_interaction_ids.get(session_id, [])
            if len(messages) != len(interaction_ids):
                raise TerminalSessionEvidenceError(
                    TerminalSessionEvidenceErrorCode.EVIDENCE_INCONSISTENT
                )
            if len(messages) > limits.max_transcript_records:
                raise TerminalSessionEvidenceError(
                    TerminalSessionEvidenceErrorCode.TRANSCRIPT_LIMIT_EXCEEDED,
                    limit=limits.max_transcript_records,
                    observed=len(messages),
                )
            # The store already owns detached, validated messages. Construct
            # lightweight record views so oversized evidence is byte-checked
            # before any message payload is copied for the returned model.
            transcript_records = tuple(
                TranscriptRecord.model_construct(
                    index=index,
                    interaction_id=interaction_ids[index],
                    message=message,
                )
                for index, message in enumerate(messages)
            )
            return terminal_session_evidence._assemble_terminal_session_evidence(
                session=session,
                marker=marker,
                terminal_record=terminal_record,
                events=events,
                transcript=transcript_records,
                limits=limits,
                allow_interrupted=allow_interrupted,
            )

    def _query_events_unlocked(
        self,
        query: EventQuery,
        *,
        max_bytes: int | None,
    ) -> list[EventRecord]:
        # `event_type` and `event_types` are mutually exclusive, so they
        # collapse into one filter set (empty = no type filter).
        if query.event_type is not None:
            event_types = frozenset((str(query.event_type),))
        else:
            event_types = frozenset(str(event_type) for event_type in query.event_types)
        excluded_event_types = frozenset(
            str(event_type) for event_type in query.exclude_event_types
        )
        candidates = self._query_candidate_records(query, event_types)
        start = (
            bisect_right(candidates, query.after_sequence, key=lambda record: record.sequence)
            if query.after_sequence is not None
            else 0
        )
        stop = (
            bisect_left(candidates, query.before_sequence, key=lambda record: record.sequence)
            if query.before_sequence is not None
            else len(candidates)
        )
        indexes = (
            range(stop - 1, start - 1, -1)
            if query.order_by == EventOrder.SEQUENCE_DESC
            else range(start, stop)
        )
        size_counter = None if max_bytes is None else JsonUtf8SizeCounter(max_bytes)
        records: list[EventRecord] = []
        for index in indexes:
            record = candidates[index]
            if not event_query_rules._event_record_matches(
                record,
                query,
                event_types,
                excluded_event_types,
            ):
                continue
            if not _event_record_matches_session(record, query, self._sessions):
                continue
            if size_counter is not None and not size_counter.value(record):
                assert max_bytes is not None
                raise EventQueryResultTooLarge(max_bytes)
            records.append(
                EventRecord(
                    sequence=record.sequence,
                    event=record.event,
                )
            )
            if len(records) == query.limit:
                break
        return records

    async def query_latest_interaction_events(
        self,
        session_id: str,
        *,
        before_sequence: int | None = None,
        limit: int = 100,
    ) -> list[EventRecord]:
        session_id = require_clean_nonblank(session_id, "session_id")
        before_sequence, limit = _validate_interaction_page(before_sequence, limit)
        async with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session not found: {session_id}")
            records = self._latest_interaction_event_records_by_sequence.get(session_id, [])
            stop = (
                bisect_left(
                    records,
                    before_sequence,
                    key=lambda record: record.sequence,
                )
                if before_sequence is not None
                else len(records)
            )
            start = max(0, stop - limit)
            return [record.model_copy(deep=True) for record in reversed(records[start:stop])]

    def _query_candidate_records(
        self,
        query: EventQuery,
        event_types: frozenset[str],
    ) -> list[EventRecord]:
        """Pick the narrowest index that still covers a query's candidate rows.

        All returned lists stay sequence-ascending so downstream ordering and
        ``after_sequence`` paging behave exactly as a full scan would.
        """
        if query.event_id is not None and query.session_id is not None:
            record = self._event_records_by_id.get((query.session_id, query.event_id))
            return [] if record is None else [record]
        if (
            query.session_id is not None
            and query.workflow_name is not None
            and event_types == frozenset((str(WORKFLOW_ATTEMPT_EVENT_TYPE),))
        ):
            return self._workflow_attempt_event_records.get(
                (query.session_id, query.workflow_name),
                [],
            )
        if query.workflow_step_id is not None:
            assert query.session_id is not None
            assert query.workflow_name is not None
            if len(event_types) == 1:
                event_type = next(iter(event_types))
                if query.workflow_attempt_fenced:
                    return self._workflow_fenced_step_event_records.get(
                        (
                            query.session_id,
                            query.workflow_name,
                            query.workflow_step_id,
                            event_type,
                        ),
                        [],
                    )
                if query.workflow_attempt_id is not None:
                    return self._workflow_step_event_records.get(
                        (
                            query.session_id,
                            query.workflow_name,
                            query.workflow_attempt_id,
                            query.workflow_step_id,
                            event_type,
                        ),
                        [],
                    )
                raise RuntimeError("Workflow-step queries require an attempt scope.")
            merged: list[EventRecord] = []
            for event_type in event_types:
                if query.workflow_attempt_fenced:
                    merged.extend(
                        self._workflow_fenced_step_event_records.get(
                            (
                                query.session_id,
                                query.workflow_name,
                                query.workflow_step_id,
                                event_type,
                            ),
                            [],
                        )
                    )
                elif query.workflow_attempt_id is not None:
                    merged.extend(
                        self._workflow_step_event_records.get(
                            (
                                query.session_id,
                                query.workflow_name,
                                query.workflow_attempt_id,
                                query.workflow_step_id,
                                event_type,
                            ),
                            [],
                        )
                    )
                else:
                    raise RuntimeError("Workflow-step queries require an attempt scope.")
            merged.sort(key=lambda record: record.sequence)
            return merged
        if query.session_id is not None:
            if query.interaction_id is not None:
                return self._interaction_event_records.get(query.session_id, {}).get(
                    query.interaction_id,
                    [],
                )
            return self._session_event_records.get(query.session_id, [])
        if query.session_ids:
            merged: list[EventRecord] = []
            for session_id in query.session_ids:
                if query.interaction_id is None:
                    merged.extend(self._session_event_records.get(session_id, []))
                else:
                    merged.extend(
                        self._interaction_event_records.get(session_id, {}).get(
                            query.interaction_id,
                            [],
                        )
                    )
            merged.sort(key=lambda record: record.sequence)
            return merged
        if event_types:
            if len(event_types) == 1:
                return self._type_event_records.get(next(iter(event_types)), [])
            merged = []
            for event_type in event_types:
                merged.extend(self._type_event_records.get(event_type, []))
            merged.sort(key=lambda record: record.sequence)
            return merged
        return self._event_records

    async def summarize_events(self, session_id: str) -> EventSummary:
        from cayu.resource_access import current_data_bounds

        access_bounds = await current_data_bounds()
        session_id = require_clean_nonblank(session_id, "session_id")
        async with self._lock:
            if access_bounds is not None:
                access_bounds.require_read(self._sessions.get(session_id))
            if session_id not in self._sessions:
                raise KeyError(f"Session not found: {session_id}")

            records = self._session_event_records.get(session_id, [])
            return event_summary_from_records(session_id, records)

    async def summarize_outcome(self, session_id: str) -> SessionOutcome:
        from cayu.resource_access import current_data_bounds

        access_bounds = await current_data_bounds()
        session_id = require_clean_nonblank(session_id, "session_id")
        async with self._lock:
            if access_bounds is not None:
                access_bounds.require_read(self._sessions.get(session_id))
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")

            records = self._session_event_records.get(session_id, [])
            return session_outcome_from_records(session, records)

    async def list_sessions(self, query: SessionQuery | None = None) -> SessionListResult:
        from cayu.resource_access import current_data_bounds

        access_bounds = await current_data_bounds()
        if access_bounds is not None:
            return await self._access_list_sessions(access_bounds, copy_session_query(query))
        return await self._list_sessions(query, pending_interruption_cascade_only=False)

    async def query_session_topology(
        self,
        query: SessionTopologyQuery,
    ) -> SessionTopologyStoreResult:
        if type(query) is not SessionTopologyQuery:
            raise TypeError("Session topology queries must be SessionTopologyQuery instances.")
        query = query.model_copy(deep=True)
        async with self._lock:
            focus_session = self._sessions.get(query.focus_session_id)
            if focus_session is None:
                raise KeyError(f"Session not found: {query.focus_session_id}")

            ancestor_sessions: list[Session] = []
            seen_ancestor_ids = {focus_session.id}
            parent_session_id = focus_session.parent_session_id
            while parent_session_id is not None:
                if parent_session_id in seen_ancestor_ids:
                    raise SessionTopologyCycle(
                        f"Session topology contains a parent cycle at {parent_session_id}."
                    )
                if len(ancestor_sessions) >= query.ancestor_depth_limit:
                    raise SessionTopologyDepthExceeded(
                        f"Session topology exceeds the {query.ancestor_depth_limit}-ancestor limit."
                    )
                parent = self._sessions.get(parent_session_id)
                if parent is None:
                    raise ValueError(
                        f"Session topology references missing parent {parent_session_id}."
                    )
                ancestor_sessions.append(parent)
                seen_ancestor_ids.add(parent.id)
                parent_session_id = parent.parent_session_id
            ancestor_sessions.reverse()

            expanded_sessions: list[Session] = []
            for parent_id in query.expanded_parent_ids:
                parent = self._sessions.get(parent_id)
                if parent is None:
                    raise KeyError(f"Session not found: {parent_id}")
                expanded_sessions.append(parent)

            branch_candidates: list[list[Session]] = []
            for parent in expanded_sessions:
                child_keys = self._child_session_keys_by_parent.get(parent.id, [])
                start_index = 0
                cursor = query.child_cursors.get(parent.id)
                if cursor is not None:
                    cursor_created_at, cursor_id = decode_session_topology_cursor(
                        cursor,
                        parent_session_id=parent.id,
                    )
                    start_index = bisect_right(child_keys, (cursor_created_at, cursor_id))
                page_keys = child_keys[start_index : start_index + query.child_limit + 1]
                candidates: list[Session] = []
                for _, child_id in page_keys:
                    child = self._sessions.get(child_id)
                    if child is None or child.parent_session_id != parent.id:
                        raise RuntimeError("Inconsistent in-memory session topology index.")
                    candidates.append(child)
                branch_candidates.append(candidates)

            result = build_session_topology_result(
                focus=SessionTopologyNode.from_session(focus_session),
                ancestors=(
                    SessionTopologyNode.from_session(session) for session in ancestor_sessions
                ),
                expanded_parents=(
                    SessionTopologyNode.from_session(session) for session in expanded_sessions
                ),
                branch_candidates=(
                    (SessionTopologyNode.from_session(session) for session in candidates)
                    for candidates in branch_candidates
                ),
                child_limit=query.child_limit,
            )
            return result

    async def query_session_lineage(
        self,
        query: SessionLineageQuery,
    ) -> SessionLineageResult:
        query = copy_session_lineage_query(query)
        async with self._lock:
            if query.parent_session_id not in self._sessions:
                raise KeyError(f"Session not found: {query.parent_session_id}")
            child_keys = self._child_session_keys_by_parent.get(query.parent_session_id, [])
            start_index = 0
            if query.cursor is not None:
                cursor_created_at, cursor_id = decode_session_lineage_cursor(
                    query.cursor,
                    parent_session_id=query.parent_session_id,
                )
                start_index = bisect_right(child_keys, (cursor_created_at, cursor_id))
            page_keys = child_keys[start_index : start_index + query.limit + 1]
            retained_keys = page_keys[: query.limit]
            children: list[SessionLineageNode] = []
            for created_at, child_id in retained_keys:
                child = self._sessions.get(child_id)
                if child is None or child.parent_session_id != query.parent_session_id:
                    raise RuntimeError("Inconsistent in-memory session lineage index.")
                origins = tuple(
                    islice(
                        (
                            SessionLineageOrigin(
                                sequence=record.sequence,
                                event_id=record.event.id,
                                event_type=EventType(record.event.type),
                            )
                            for record in self._session_event_records.get(child_id, [])
                            if record.event.type
                            in {EventType.SESSION_STARTED, EventType.SESSION_FORKED}
                        ),
                        SESSION_LINEAGE_MAX_ORIGIN_EVENTS,
                    )
                )
                children.append(
                    SessionLineageNode(
                        id=child.id,
                        parent_session_id=query.parent_session_id,
                        created_at=created_at,
                        origin_events=origins,
                    )
                )
            has_more = len(page_keys) > len(retained_keys)
            return SessionLineageResult(
                parent_session_id=query.parent_session_id,
                children=tuple(children),
                next_cursor=(
                    encode_session_lineage_cursor(query.parent_session_id, children[-1])
                    if has_more and children
                    else None
                ),
                has_more=has_more,
            )

    async def query_child_session_lifecycle(
        self,
        query: ChildSessionLifecycleQuery,
    ) -> ChildSessionLifecyclePage:
        query = ChildSessionLifecycleQuery.model_validate(query)
        async with self._lock:
            parent = self._sessions.get(query.parent_session_id)
            if parent is None:
                raise KeyError(f"Session not found: {query.parent_session_id}")
            child_keys = self._child_session_keys_by_parent.get(parent.id, ())
            candidate_limit = query.max_children_inspected + 1
            candidate_ids: list[str] = []
            priorities = self._child_lifecycle_candidate_keys_by_parent.get(parent.id, {})
            for priority in (0, 1, 2):
                for _sort_at, child_id in priorities.get(priority, ()):
                    if len(candidate_ids) >= candidate_limit:
                        break
                    candidate_ids.append(child_id)
                if len(candidate_ids) >= candidate_limit:
                    break
            if sum(len(keys) for keys in priorities.values()) != len(child_keys):
                raise RuntimeError("In-memory child lifecycle candidate index is inconsistent.")
            retained_ids = candidate_ids[: query.max_children_inspected]
            has_more = len(child_keys) > len(retained_ids)
            entries: list[ChildSessionLifecycleEntry] = []
            unavailable_count = 0
            operation_records = self._session_operation_records[parent.id]
            for child_id in retained_ids:
                child = self._sessions.get(child_id)
                if child is None:
                    raise RuntimeError("In-memory child-session index is inconsistent.")
                records_by_type = self._child_lifecycle_event_records.get(child_id, {})
                occurrence_source = _child_session_lifecycle_occurrence(
                    child,
                    records_by_type,
                )
                consumption_record = None
                if occurrence_source is not None:
                    _relationship, occurrence = occurrence_source
                    consumption_record = operation_records.get(
                        child_session_notification_storage_key(
                            child.instance_id,
                            occurrence.source_id,
                        )
                    )
                entry = _child_session_lifecycle_entry(
                    parent=parent,
                    child=child,
                    records_by_type=records_by_type,
                    consumption_record=consumption_record,
                )
                if entry is None:
                    unavailable_count += 1
                else:
                    entries.append(entry)
            entries.sort(key=_child_session_lifecycle_entry_sort_key)
            return ChildSessionLifecyclePage(
                parent_session_id=parent.id,
                parent_session_instance_id=parent.instance_id,
                entries=tuple(entries),
                inspected_child_count=len(retained_ids),
                unavailable_child_count=unavailable_count,
                has_more=has_more,
            )

    async def aggregate_operational_snapshot(
        self,
        filters: SessionAggregateFilter | None = None,
    ) -> SessionOperationalSnapshot:
        filters = copy_session_aggregate_filter(filters)
        session_query = session_query_from_aggregate_filter(filters)
        async with self._lock:
            as_of = self._ownership_clock()
            counts = {status: 0 for status in SessionStatus}
            total_count = 0
            for item_index, session in enumerate(self._sessions.values(), start=1):
                if session_query_rules._session_matches(session, session_query):
                    counts[session.status] += 1
                    total_count += 1
                if item_index % _IN_MEMORY_AGGREGATE_CANCELLATION_INTERVAL == 0:
                    await _cooperate_with_in_memory_aggregate_cancellation()
            return SessionOperationalSnapshot(
                as_of=as_of,
                total_count=total_count,
                counts_by_status=SessionStatusCounts.model_validate(counts),
                accuracy=EXACT_AGGREGATE.model_copy(),
            )

    async def aggregate_usage(self, query: UsageRollupQuery) -> UsageRollupStoreResult:
        query = copy_usage_rollup_query(query)
        session_query = session_query_from_aggregate_filter(query.sessions)
        async with self._lock:
            as_of = self._ownership_clock()
            matching_session_count = 0
            active_session_count = 0
            for session in self._sessions.values():
                if not session_query_rules._session_matches(session, session_query):
                    continue
                matching_session_count += 1
                active_session_count += session.status in {
                    SessionStatus.PENDING,
                    SessionStatus.RUNNING,
                    SessionStatus.INTERRUPTING,
                }

            def matching_session_records() -> Iterable[
                tuple[str, SessionStatus, Iterable[EventRecord]]
            ]:
                for session_id, session in self._sessions.items():
                    if session_query_rules._session_matches(session, session_query):
                        yield (
                            session_id,
                            session.status,
                            self._session_event_records.get(session_id, ()),
                        )

            return session_usage_rules._usage_rollup_from_session_records(
                session_records=matching_session_records,
                query=query,
                as_of=as_of,
                matching_session_count=matching_session_count,
                active_session_count=active_session_count,
            )

    async def list_sessions_with_pending_interruption_cascade(
        self,
        query: SessionQuery | None = None,
    ) -> SessionListResult:
        return await self._list_sessions(query, pending_interruption_cascade_only=True)

    async def list_queued_dispatch_terminal_receipts(
        self,
        query: QueuedDispatchTerminalReceiptQuery | None = None,
    ) -> list[QueuedDispatchTerminalReceipt]:
        if query is None:
            query = QueuedDispatchTerminalReceiptQuery()
        elif type(query) is not QueuedDispatchTerminalReceiptQuery:
            raise TypeError(
                "Queued dispatch receipt queries must be "
                "QueuedDispatchTerminalReceiptQuery instances."
            )
        else:
            query = QueuedDispatchTerminalReceiptQuery(
                after_session_id=query.after_session_id,
                after_operation_id=query.after_operation_id,
                limit=query.limit,
            )
        cursor: tuple[str, str] | None = None
        if query.after_session_id is not None:
            assert query.after_operation_id is not None
            cursor = (query.after_session_id, query.after_operation_id)
        async with self._lock:
            start = (
                0
                if cursor is None
                else bisect_right(self._queued_dispatch_terminal_receipt_keys, cursor)
            )
            keys = self._queued_dispatch_terminal_receipt_keys[start : start + query.limit]
            return [
                self._queued_dispatch_terminal_receipts[key].model_copy(deep=True) for key in keys
            ]

    async def query_pending_actions(
        self,
        query: PendingActionQuery | None = None,
        *,
        checkpoint_root_guard: CheckpointRootFieldGuard | None = None,
    ) -> PendingActionListResult:
        from cayu.sessions.pending_actions import (
            PENDING_ACTION_CHECKPOINT_KEYS,
            pending_action_checkpoint_tool_call_count,
            pending_action_event_projection_bytes,
            pending_action_from_records,
            pending_action_matches_query,
            pending_action_source_is_invalid,
            project_pending_action_checkpoint,
            select_pending_action_indexed_records,
        )

        if query is None:
            query = PendingActionQuery()
        elif type(query) is not PendingActionQuery:
            raise TypeError("Pending-action queries must be PendingActionQuery instances.")
        else:
            query = query.model_copy(deep=True)

        candidate_limit = min(query.limit * 4, 800) + 1
        async with self._lock:
            if query.session_id is None:
                candidate_ids = self._pending_action_session_ids
            elif query.session_id in self._pending_action_session_ids:
                candidate_ids = {query.session_id}
            else:
                candidate_ids = set()
            candidates = [
                session
                for session_id in candidate_ids
                if (session := self._sessions.get(session_id)) is not None
                and session.status in query.statuses
                and (query.agent_name is None or session.agent_name == query.agent_name)
                and (
                    query.environment_name is None
                    or session.environment_name == query.environment_name
                )
            ]
            candidates = session_query_rules._sort_sessions(
                candidates, SessionOrder.UPDATED_AT_DESC
            )
            if query.cursor is not None:
                cursor_dt, cursor_id = decode_session_cursor(query.cursor)
                candidates = [
                    session
                    for session in candidates
                    if session_query_rules._session_after_cursor(
                        session,
                        SessionOrder.UPDATED_AT_DESC,
                        cursor_dt,
                        cursor_id,
                    )
                ]

            candidate_window = candidates[:candidate_limit]
            has_more_candidates = len(candidate_window) == candidate_limit
            inspected_candidates = candidate_window[: candidate_limit - 1]
            actions: list[PendingActionRecord] = []
            issues: list[PendingActionIssue] = []
            materialized_source_bytes = 0
            inspected = 0
            more_matching = False
            last_inspected_session: PendingActionSession | None = None
            for session in inspected_candidates:
                previous_last_inspected_session = last_inspected_session
                projected_session = PendingActionSession.from_session(session)
                checkpoint = self._checkpoints.get(session.id)
                if checkpoint_root_guard is not None:
                    checkpoint_root_guard.validate(
                        session.id,
                        checkpoint_root_field_projection(
                            checkpoint,
                            checkpoint_root_guard.key,
                        ),
                    )
                pending_checkpoint_source = (
                    None
                    if checkpoint is None
                    else {
                        key: checkpoint[key]
                        for key in PENDING_ACTION_CHECKPOINT_KEYS
                        if checkpoint.get(key) is not None
                    }
                )
                source_too_complex = (
                    pending_action_checkpoint_tool_call_count(pending_checkpoint_source)
                    > MAX_PENDING_ACTION_TOOL_CALLS
                )
                candidate_size = JsonUtf8SizeCounter(query.max_result_bytes)
                source_fits = (
                    not source_too_complex
                    and candidate_size.value(projected_session)
                    and candidate_size.value(pending_checkpoint_source)
                )
                projected_checkpoint = None
                records: list[EventRecord] = []
                if source_fits:
                    projected_checkpoint = project_pending_action_checkpoint(checkpoint)
                    records, ledger_too_complex = select_pending_action_indexed_records(
                        projected_checkpoint,
                        self._pending_action_event_records.get(session.id, {}),
                        self._pending_action_latest_barrier_records.get(session.id),
                    )
                    source_fits = (
                        not ledger_too_complex
                        and all(
                            (projection_bytes := pending_action_event_projection_bytes(record))
                            is None
                            or projection_bytes <= query.max_result_bytes
                            for record in records
                        )
                        and all(candidate_size.value(record) for record in records)
                    )
                else:
                    ledger_too_complex = False
                candidate_source_bytes = query.max_result_bytes - candidate_size.remaining
                if source_fits and (
                    materialized_source_bytes + candidate_source_bytes > query.max_result_bytes
                ):
                    more_matching = True
                    break

                inspected += 1
                last_inspected_session = projected_session
                if not source_fits:
                    if len(actions) + len(issues) == query.limit:
                        more_matching = True
                        inspected -= 1
                        last_inspected_session = previous_last_inspected_session
                        break
                    issues.append(
                        PendingActionIssue.ledger_too_complex(
                            projected_session,
                            max_events_per_call=MAX_PENDING_ACTION_LEDGER_EVENTS_PER_CALL,
                        )
                        if ledger_too_complex
                        else PendingActionIssue.source_too_complex(
                            projected_session,
                            max_tool_calls=MAX_PENDING_ACTION_TOOL_CALLS,
                        )
                        if source_too_complex
                        else PendingActionIssue.source_too_large(
                            projected_session,
                            max_bytes=query.max_result_bytes,
                        )
                    )
                    continue

                materialized_source_bytes += candidate_source_bytes
                action = pending_action_from_records(
                    projected_session,
                    records,
                    projected_checkpoint,
                )
                if pending_action_source_is_invalid(
                    projected_session,
                    projected_checkpoint,
                    action,
                    records,
                ):
                    if len(actions) + len(issues) == query.limit:
                        more_matching = True
                        inspected -= 1
                        last_inspected_session = previous_last_inspected_session
                        break
                    issues.append(PendingActionIssue.source_invalid(projected_session))
                    continue
                if action is None:
                    continue
                if query.kind is not None and action.kind != query.kind:
                    continue
                if not pending_action_matches_query(action, query.q):
                    continue
                if len(actions) + len(issues) == query.limit:
                    more_matching = True
                    inspected -= 1
                    last_inspected_session = previous_last_inspected_session
                    break
                actions.append(action)

            has_more = more_matching or has_more_candidates
            cursor_session = last_inspected_session
            next_cursor = (
                encode_session_cursor(cursor_session, SessionOrder.UPDATED_AT_DESC)
                if has_more and cursor_session is not None
                else None
            )
            return enforce_pending_action_result_size(
                PendingActionListResult(
                    actions=actions,
                    issues=issues,
                    next_cursor=next_cursor,
                    has_more=has_more,
                    # A cursor query sees only the suffix after that cursor, so a
                    # final page's local length is not a global total. The first
                    # page can report an exact total only when it exhausted the
                    # candidate set without hitting either bound.
                    total_count=(
                        len(actions) + len(issues)
                        if query.cursor is None and not has_more
                        else None
                    ),
                    inspected_candidate_count=inspected,
                ),
                max_bytes=query.max_result_bytes,
            )

    async def _list_sessions(
        self,
        query: SessionQuery | None,
        *,
        pending_interruption_cascade_only: bool,
        access_bounds: _SessionAccessBounds | None = None,
    ) -> SessionListResult:
        query = copy_session_query(query)
        async with self._lock:
            inactive_before = (
                query.last_activity_before
                if query.inactive_for_seconds is None
                else utc_duration_cutoff(
                    self._ownership_clock(),
                    query.inactive_for_seconds,
                )
            )
            duration_matches_no_timestamp = (
                query.inactive_for_seconds is not None and inactive_before is None
            )
            base_query = query.model_copy(
                update={
                    "debug_state": None,
                    "last_activity_before": inactive_before,
                    "inactive_for_seconds": None,
                }
            )
            candidates = (
                (
                    self._sessions[session_id]
                    for session_id, checkpoint in self._checkpoints.items()
                    if "pending_interruption_cascade" in checkpoint and session_id in self._sessions
                )
                if pending_interruption_cascade_only
                else self._sessions.values()
            )
            matching = (
                []
                if duration_matches_no_timestamp
                else [
                    session
                    for session in candidates
                    if (access_bounds is None or access_bounds.matches(session.labels))
                    and session_query_rules._session_matches(session, base_query)
                    and _session_matches_debug_state(
                        session,
                        self._session_event_records.get(session.id, []),
                        query.debug_state,
                    )
                ]
            )
            total = len(matching) if query.include_total_count else None
            ordered = session_query_rules._sort_sessions(matching, query.order_by)
            if query.cursor is not None:
                cursor_dt, cursor_id = decode_session_cursor(query.cursor)
                ordered = [
                    session
                    for session in ordered
                    if session_query_rules._session_after_cursor(
                        session, query.order_by, cursor_dt, cursor_id
                    )
                ]
                window = ordered[: query.limit + 1]
            else:
                window = ordered[query.offset : query.offset + query.limit + 1]
            page = window[: query.limit]
            next_cursor = session_next_cursor(page, len(window) > query.limit, query.order_by)
            return SessionListResult(
                sessions=[session.model_copy(deep=True) for session in page],
                next_cursor=next_cursor,
                total_count=total,
            )

    async def append_transcript_messages(
        self,
        session_id: str,
        messages: list[Message],
        *,
        interaction_id: InteractionAttribution = INHERIT_INTERACTION,
    ) -> None:
        session_id = require_clean_nonblank(session_id, "session_id")
        interaction_id = resolve_interaction_attribution(session_id, interaction_id)
        copied_messages = copy_transcript_messages(messages)
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            invocation_scope._assert_session_run_epoch(session_id, session)
            if not copied_messages:
                return
            for owner in self._session_closure_progress.values():
                _check_closure_lineage_owner(owner, (session_id,))
            if interaction_id is not None:
                self._register_private_authority_alias_unlocked(
                    interaction_id,
                    field_name="interaction_id",
                    scope_session_id=session_id,
                )
            self._transcripts[session_id].extend(copied_messages)
            self._extend_transcript_search_unlocked(session_id, copied_messages)
            self._extend_transcript_attribution_unlocked(
                session_id, [interaction_id] * len(copied_messages)
            )
            self._sessions[session_id] = session.model_copy(
                update={"last_activity_at": self._ownership_clock()}
            )

    async def transform_checkpoint_with_store_time(
        self,
        session_id: str,
        checkpoint_transform: StoreTimeCheckpointTransform,
    ) -> None:
        session_id = require_clean_nonblank(session_id, "session_id")
        if checkpoint_transform is None:
            raise TypeError("checkpoint_transform is required.")
        async with self._lock:
            now = self._ownership_clock()
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            invocation_scope._assert_session_run_epoch(session_id, session)
            for owner in self._session_closure_progress.values():
                _check_closure_lineage_owner(owner, (session_id,))
            current = self._checkpoints.get(session_id)
            transformed = checkpoint_transform(
                session.model_copy(deep=True),
                checkpoint_preservation._copy_checkpoint_for_transform(
                    current, session_id=session_id
                ),
                now,
            )
            if transformed is None:
                return
            self._store_checkpoint_unlocked(
                session_id,
                checkpoint_preservation._replace_checkpoint_preserving_completion_result_event_publications(
                    current,
                    copy_durable_json_object(transformed, "checkpoint"),
                    session_id=session_id,
                ),
            )
            self._sessions[session_id] = session.model_copy(update={"last_activity_at": now})

    async def replace_initial_transcript_messages(
        self,
        session_id: str,
        expected_messages: list[Message],
        replacement_messages: list[Message],
        *,
        interaction_id: InteractionAttribution = INHERIT_INTERACTION,
        checkpoint_transform: CheckpointTransform | None = None,
        runtime_suffix_count: int = 0,
    ) -> None:
        session_id = require_clean_nonblank(session_id, "session_id")
        interaction_id = resolve_interaction_attribution(session_id, interaction_id)
        if interaction_id is None:
            raise ValueError("Initial transcript publication requires an interaction identity.")
        expected = copy_transcript_messages(expected_messages)
        replacement = copy_transcript_messages(replacement_messages)
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            invocation_scope._assert_session_run_epoch(session_id, session)
            deferred = self._deferred_interaction_inputs.get(session_id)
            if deferred is None or deferred.interaction_id != interaction_id:
                raise RuntimeError("Deferred interaction input changed before finalization.")
            for owner in self._session_closure_progress.values():
                _check_closure_lineage_owner(owner, (session_id,))
            require_deferred_initial_transcript_replacement(
                deferred,
                expected_messages=expected,
                replacement_messages=replacement,
            )
            self._register_private_authority_alias_unlocked(
                interaction_id,
                field_name="interaction_id",
                scope_session_id=session_id,
            )
            if self._transcripts.get(session_id):
                from cayu.sessions._participant_execution_identity import (
                    require_initial_execution_input,
                )

                require_initial_execution_input(
                    session,
                    self._participant_session_receipts.get(session_id),
                    self._transcripts[session_id],
                    expected,
                )
            prefix_count = session_transcript_input._initial_transcript_prefix_count(
                expected,
                replacement,
                runtime_suffix_count=runtime_suffix_count,
            )
            current_checkpoint = self._checkpoints.get(session_id)
            if checkpoint_transform is not None:
                transformed = checkpoint_transform(
                    session.model_copy(deep=True),
                    checkpoint_preservation._copy_checkpoint_for_transform(
                        current_checkpoint,
                        session_id=session_id,
                    ),
                )
                if transformed is not None:
                    current_checkpoint = checkpoint_preservation._checkpoint_transform_result_preserving_completion_result_event_publications(
                        current_checkpoint,
                        transformed,
                        session_id=session_id,
                    )
            checkpoint = _checkpoint_after_initial_transcript_publication(
                current_checkpoint,
                interaction_id=interaction_id,
            )
            self._transcripts[session_id] = replacement
            self._replace_latest_transcript_role_indexes_unlocked(session_id)
            self._replace_transcript_search_session_unlocked(session_id, replacement)
            self._transcript_interaction_ids[session_id] = [None] * prefix_count + [
                interaction_id
            ] * (len(expected) + runtime_suffix_count)
            self._transcript_indices_by_interaction.pop(session_id, None)
            self._deferred_interaction_inputs.pop(session_id, None)
            if checkpoint is None:
                self._refresh_queued_dispatch_terminal_receipts_unlocked(session_id, None)
                self._checkpoints.pop(session_id, None)
            else:
                self._store_checkpoint_unlocked(session_id, checkpoint)
            self._sessions[session_id] = session.model_copy(
                update={"last_activity_at": self._ownership_clock()}
            )

    async def materialize_deferred_interaction_input(
        self,
        session_id: str,
        *,
        interaction_id: InteractionAttribution = INHERIT_INTERACTION,
    ) -> bool:
        session_id = require_clean_nonblank(session_id, "session_id")
        interaction_id = resolve_interaction_attribution(session_id, interaction_id)
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            invocation_scope._assert_session_run_epoch(session_id, session)
            deferred = self._deferred_interaction_inputs.get(session_id)
            if deferred is None:
                return False
            if deferred.interaction_id != interaction_id:
                raise RuntimeError("Deferred interaction input belongs to another interaction.")
            for owner in self._session_closure_progress.values():
                _check_closure_lineage_owner(owner, (session_id,))
            messages = deferred.source_messages
            self._transcripts[session_id].extend(messages)
            self._extend_transcript_search_unlocked(session_id, messages)
            self._extend_transcript_attribution_unlocked(
                session_id,
                [interaction_id] * len(messages),
            )
            self._deferred_interaction_inputs.pop(session_id, None)
            self._sessions[session_id] = session.model_copy(
                update={"last_activity_at": self._ownership_clock()}
            )
            return True

    async def load_deferred_interaction_input(
        self,
        session_id: str,
    ) -> DeferredInteractionInput | None:
        session_id = require_clean_nonblank(session_id, "session_id")
        async with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session not found: {session_id}")
            deferred = self._deferred_interaction_inputs.get(session_id)
            if deferred is None:
                return None
            return DeferredInteractionInput.model_validate(
                deferred.model_dump(mode="python", warnings=False)
            )

    async def append_transcript_messages_and_transform_checkpoint(
        self,
        session_id: str,
        messages: list[Message],
        checkpoint_transform: CheckpointTransform,
        *,
        interaction_id: InteractionAttribution = INHERIT_INTERACTION,
    ) -> None:
        session_id = require_clean_nonblank(session_id, "session_id")
        interaction_id = resolve_interaction_attribution(session_id, interaction_id)
        copied_messages = copy_transcript_messages(messages)
        if checkpoint_transform is None:
            raise TypeError("checkpoint_transform is required.")
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            invocation_scope._assert_session_run_epoch(session_id, session)
            for owner in self._session_closure_progress.values():
                _check_closure_lineage_owner(owner, (session_id,))
            current = self._checkpoints.get(session_id)
            transformed = checkpoint_transform(
                session.model_copy(deep=True),
                checkpoint_preservation._copy_checkpoint_for_transform(
                    current, session_id=session_id
                ),
            )
            if transformed is None:
                raise ValueError("Checkpoint transform must return a checkpoint.")
            copied_checkpoint = checkpoint_preservation._checkpoint_transform_result_preserving_completion_result_event_publications(
                current,
                transformed,
                session_id=session_id,
            )
            if copied_messages:
                if interaction_id is not None:
                    self._register_private_authority_alias_unlocked(
                        interaction_id,
                        field_name="interaction_id",
                        scope_session_id=session_id,
                    )
                self._transcripts[session_id].extend(copied_messages)
                self._extend_transcript_search_unlocked(session_id, copied_messages)
                self._extend_transcript_attribution_unlocked(
                    session_id, [interaction_id] * len(copied_messages)
                )
            self._store_checkpoint_unlocked(session_id, copied_checkpoint)
            self._sessions[session_id] = session.model_copy(
                update={"last_activity_at": self._ownership_clock()}
            )

    async def load_transcript(self, session_id: str) -> list[Message]:
        from cayu.resource_access import current_data_bounds

        access_bounds = await current_data_bounds()
        session_id = require_clean_nonblank(session_id, "session_id")
        async with self._lock:
            if access_bounds is not None:
                access_bounds.require_read(self._sessions.get(session_id))
            if session_id not in self._sessions:
                raise KeyError(f"Session not found: {session_id}")
            return [detach_message(message) for message in self._transcripts.get(session_id, [])]

    async def load_transcript_snapshot(self, session_id: str) -> TranscriptSnapshot:
        from cayu.resource_access import current_data_bounds

        access_bounds = await current_data_bounds()
        session_id = require_clean_nonblank(session_id, "session_id")
        async with self._lock:
            if access_bounds is not None:
                access_bounds.require_read(self._sessions.get(session_id))
            if session_id not in self._sessions:
                raise KeyError(f"Session not found: {session_id}")
            messages = self._transcripts.get(session_id, [])
            interaction_ids = self._transcript_interaction_ids.get(session_id, [])
            return TranscriptSnapshot(
                records=[
                    TranscriptRecord(
                        index=index,
                        interaction_id=interaction_ids[index],
                        message=detach_message(message),
                    )
                    for index, message in enumerate(messages)
                ],
                cursor=len(messages),
            )

    async def load_transcript_cursor(self, session_id: str) -> int:
        session_id = require_clean_nonblank(session_id, "session_id")
        async with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session not found: {session_id}")
            return len(self._transcripts.get(session_id, []))

    async def load_latest_transcript_message(
        self,
        session_id: str,
        *,
        role: MessageRole,
    ) -> TranscriptRecord | None:
        session_id = require_clean_nonblank(session_id, "session_id")
        if not isinstance(role, MessageRole):
            raise TypeError("role must be a MessageRole.")
        async with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session not found: {session_id}")
            messages = self._transcripts.get(session_id, [])
            interaction_ids = self._transcript_interaction_ids.get(session_id, [])
            index = self._latest_transcript_indexes_by_role.get(session_id, {}).get(role)
            if index is None:
                return None
            return TranscriptRecord(
                index=index,
                interaction_id=interaction_ids[index],
                message=detach_message(messages[index]),
            )

    async def load_latest_transcript_text(
        self,
        session_id: str,
        *,
        role: MessageRole,
        max_chars: int,
    ) -> tuple[str, bool] | None:
        from cayu.resource_access import current_data_bounds

        access_bounds = await current_data_bounds()
        session_id = require_clean_nonblank(session_id, "session_id")
        if not isinstance(role, MessageRole):
            raise TypeError("role must be a MessageRole.")
        if type(max_chars) is not int:
            raise TypeError("max_chars must be an integer.")
        if not 1 <= max_chars <= LATEST_TRANSCRIPT_TEXT_MAX_CHARS:
            raise ValueError(f"max_chars must be between 1 and {LATEST_TRANSCRIPT_TEXT_MAX_CHARS}.")
        async with self._lock:
            if access_bounds is not None:
                access_bounds.require_read(self._sessions.get(session_id))
            if session_id not in self._sessions:
                raise KeyError(f"Session not found: {session_id}")
            index = self._latest_transcript_indexes_by_role.get(session_id, {}).get(role)
            if index is None:
                return None
            return session_transcript_queries._bounded_transcript_message_text(
                self._transcripts[session_id][index],
                max_chars=max_chars,
            )

    async def query_transcript(self, query: TranscriptQuery) -> TranscriptPage:
        from cayu.resource_access import current_data_bounds

        access_bounds = await current_data_bounds()
        query = copy_transcript_query(query)
        async with self._lock:
            if access_bounds is not None:
                access_bounds.require_read(self._sessions.get(query.session_id))
            if query.session_id not in self._sessions:
                raise KeyError(f"Session not found: {query.session_id}")

            messages = self._transcripts.get(query.session_id, [])
            interaction_ids = self._transcript_interaction_ids.get(query.session_id, [])
            if query.interaction_id is None:
                candidate_indices: Sequence[int] = range(len(messages))
            else:
                candidate_indices = self._transcript_interaction_projection_unlocked(
                    query.session_id
                ).get(query.interaction_id, ())
            if query.role is not None:
                candidate_indices = [
                    index for index in candidate_indices if messages[index].role == query.role
                ]
            total_records = len(candidate_indices)
            page_indices = candidate_indices[query.offset : query.offset + query.limit]
            # Per filter_transcript_records' contract, the include_thinking=False
            # path already isolates — detach only pass-through records.
            records = [
                TranscriptRecord(
                    index=index,
                    interaction_id=interaction_ids[index],
                    message=(
                        detach_message(messages[index])
                        if query.include_thinking
                        else messages[index]
                    ),
                )
                for index in page_indices
            ]
            return TranscriptPage(
                records=filter_transcript_records(records, include_thinking=query.include_thinking),
                total_records=total_records,
            )

    async def search_transcript(
        self,
        query: TranscriptSearchQuery,
    ) -> TranscriptSearchResult:
        query = copy_transcript_search_query(query)
        cursor = decode_transcript_search_cursor(query)
        query_document = transcript_search_query_document(query.text)
        query_tokens = tuple(query_document.split())
        role_set = set(query.roles)
        candidate_heap: list[tuple[int, str, int, Message, str | None]] = []
        overflow = False

        async with self._lock:
            selected = frozenset(
                session_id for session_id in query.session_ids if session_id in self._sessions
            )
            posting_sources: list[tuple[str, list[int]]] = []
            posting_heap: list[tuple[str, int, int, int]] = []
            for token in query_tokens:
                for role in query.roles:
                    posting_sessions = self._transcript_search_postings.get((token, role), {})
                    for session_id in sorted(selected):
                        indexes = posting_sessions.get(session_id)
                        if not indexes:
                            continue
                        source_index = len(posting_sources)
                        posting_sources.append((session_id, indexes))
                        before_index = query.before_transcript_indexes.get(session_id)
                        position = (
                            len(indexes) - 1
                            if before_index is None
                            else bisect_left(indexes, before_index) - 1
                        )
                        if position < 0:
                            continue
                        heapq.heappush(
                            posting_heap,
                            (session_id, -indexes[position], source_index, position),
                        )

            popped = 0
            next_yield = 256
            while posting_heap:
                session_id, negative_index, source_index, position = heapq.heappop(posting_heap)
                candidate_key = (session_id, negative_index)
                same_candidate = [(source_index, position)]
                while posting_heap and posting_heap[0][:2] == candidate_key:
                    _, _, duplicate_source, duplicate_position = heapq.heappop(posting_heap)
                    same_candidate.append((duplicate_source, duplicate_position))
                for matched_source, matched_position in same_candidate:
                    source_session_id, indexes = posting_sources[matched_source]
                    next_position = matched_position - 1
                    if next_position >= 0:
                        heapq.heappush(
                            posting_heap,
                            (
                                source_session_id,
                                -indexes[next_position],
                                matched_source,
                                next_position,
                            ),
                        )
                popped += len(same_candidate)
                if popped >= next_yield:
                    next_yield = popped + 256
                    await asyncio.sleep(0)

                transcript_index = -negative_index
                messages = self._transcripts.get(session_id, ())
                interaction_ids = self._transcript_interaction_ids.get(session_id, ())
                documents = self._transcript_search_documents.get(session_id, ())
                if not (
                    transcript_index < len(messages)
                    and len(messages) == len(interaction_ids) == len(documents)
                ):
                    raise RuntimeError("In-memory transcript search index is inconsistent.")
                message = messages[transcript_index]
                if message.role not in role_set:
                    continue
                indexed_role, document = documents[transcript_index]
                if indexed_role != message.role:
                    raise RuntimeError("In-memory transcript search index is inconsistent.")
                score = transcript_search_document_score(document, query_document)
                if score <= 0:  # pragma: no cover - postings establish a token match
                    raise RuntimeError("In-memory transcript search posting is inconsistent.")
                if len(candidate_heap) >= query.max_records_scanned:
                    overflow = True
                    break
                heapq.heappush(
                    candidate_heap,
                    (
                        -score,
                        session_id,
                        -transcript_index,
                        message,
                        interaction_ids[transcript_index],
                    ),
                )

            if overflow:
                return TranscriptSearchResult(
                    query=query,
                    matched_records_examined=query.max_records_scanned,
                    truncated=True,
                    coverage_complete=False,
                )

            matched_records_examined = len(candidate_heap)
            hits: list[TranscriptSearchHit] = []
            remaining_bytes = query.max_bytes
            truncated = False
            continuation_available = False
            ranked_examined = 0
            while candidate_heap:
                (
                    negative_score,
                    session_id,
                    negative_transcript_index,
                    message,
                    interaction_id,
                ) = heapq.heappop(candidate_heap)
                score = -negative_score
                transcript_index = -negative_transcript_index
                ranked_examined += 1
                if ranked_examined % 256 == 0:
                    await asyncio.sleep(0)
                if cursor is not None and not transcript_search_position_after_cursor(
                    raw_score=score,
                    session_id=session_id,
                    transcript_index=transcript_index,
                    cursor=cursor,
                ):
                    continue
                if len(hits) >= query.limit:
                    truncated = True
                    continuation_available = True
                    break
                hit = transcript_search_hit_from_message(
                    session_id=session_id,
                    transcript_index=transcript_index,
                    interaction_id=interaction_id,
                    message=message,
                    max_text_bytes=remaining_bytes,
                    raw_score=float(score),
                )
                if hit is None:
                    truncated = True
                    continuation_available = bool(hits)
                    break
                hits.append(hit)
                remaining_bytes -= len(hit.text.encode("utf-8"))
                if not hit.text_complete:
                    truncated = True
                    continuation_available = bool(candidate_heap)
                    break
                if remaining_bytes == 0:
                    truncated = bool(candidate_heap)
                    continuation_available = truncated
                    break

        next_cursor = (
            encode_transcript_search_cursor(
                query,
                raw_score=int(hits[-1].raw_score or 0),
                session_id=hits[-1].session_id,
                transcript_index=hits[-1].transcript_index,
            )
            if continuation_available and hits
            else None
        )
        return TranscriptSearchResult(
            query=query,
            hits=tuple(hits),
            matched_records_examined=matched_records_examined,
            truncated=truncated,
            coverage_complete=True,
            next_cursor=next_cursor,
        )

    async def create_recall_receipt(self, receipt: RecallReceipt) -> RecallReceipt:
        copied = copy_recall_receipt(receipt)
        async with self._lock:
            current = self._recall_receipts.get(copied.receipt_id)
            if current is not None:
                if memory_evidence_document_bytes(
                    current, "stored recall receipt"
                ) != memory_evidence_document_bytes(copied, "recall receipt"):
                    raise RecallEvidenceConflict("Recall receipt", copied.receipt_id)
                return copy_recall_receipt(current)
            if copied.session_id not in self._sessions:
                raise KeyError(f"Session not found: {copied.session_id}")
            for owner in self._session_closure_progress.values():
                _check_closure_lineage_owner(owner, (copied.session_id,))
            self._recall_receipts[copied.receipt_id] = copied
            self._recall_receipt_ids_by_session.setdefault(copied.session_id, set()).add(
                copied.receipt_id
            )
            page_key = copied.created_at, copied.receipt_id
            for page_keys in (
                self._recall_receipt_page_keys_by_session.setdefault(copied.session_id, []),
                self._recall_receipt_page_keys_by_interaction.setdefault(
                    (copied.session_id, copied.interaction_id), []
                ),
                self._recall_receipt_page_keys_by_step.setdefault(
                    (copied.session_id, copied.model_step_id), []
                ),
                self._recall_receipt_page_keys_by_interaction_step.setdefault(
                    (copied.session_id, copied.interaction_id, copied.model_step_id), []
                ),
            ):
                page_keys.insert(bisect_right(page_keys, page_key), page_key)
            return copy_recall_receipt(copied)

    async def load_recall_receipt(
        self,
        session_id: str,
        receipt_id: str,
    ) -> RecallReceipt | None:
        session_id = require_memory_evidence_session_id(session_id)
        receipt_id = require_memory_evidence_id(receipt_id, "receipt_id")
        async with self._lock:
            receipt = self._recall_receipts.get(receipt_id)
            if receipt is None or receipt.session_id != session_id:
                return None
            return copy_recall_receipt(receipt)

    async def list_recall_receipts(
        self,
        query: RecallEvidenceQuery,
    ) -> RecallReceiptPage:
        copied_query = RecallEvidenceQuery.model_validate(query.model_dump(mode="python"))
        query_fingerprint = copied_query.fingerprint("receipt")
        after = (
            None
            if copied_query.cursor is None
            else decode_recall_evidence_cursor(
                copied_query.cursor,
                record_kind="receipt",
                query_fingerprint=query_fingerprint,
            )
        )
        async with self._lock:
            if copied_query.interaction_id is not None and copied_query.model_step_id is not None:
                page_keys = self._recall_receipt_page_keys_by_interaction_step.get(
                    (
                        copied_query.session_id,
                        copied_query.interaction_id,
                        copied_query.model_step_id,
                    ),
                    (),
                )
            elif copied_query.interaction_id is not None:
                page_keys = self._recall_receipt_page_keys_by_interaction.get(
                    (copied_query.session_id, copied_query.interaction_id),
                    (),
                )
            elif copied_query.model_step_id is not None:
                page_keys = self._recall_receipt_page_keys_by_step.get(
                    (copied_query.session_id, copied_query.model_step_id),
                    (),
                )
            else:
                page_keys = self._recall_receipt_page_keys_by_session.get(
                    copied_query.session_id,
                    (),
                )
            start = 0 if after is None else bisect_right(page_keys, after)
            candidates = tuple(
                self._recall_receipts[receipt_id]
                for _, receipt_id in islice(
                    page_keys,
                    start,
                    start + copied_query.limit + 1,
                )
            )
            retained: list[RecallReceipt] = []
            retained_bytes = 2
            for receipt in candidates:
                document_bytes = len(
                    memory_evidence_document_bytes(receipt, "recall receipt page item")
                )
                separator_bytes = 1 if retained else 0
                if len(retained) >= copied_query.limit or (
                    retained_bytes + separator_bytes + document_bytes > copied_query.max_bytes
                ):
                    break
                retained.append(receipt)
                retained_bytes += separator_bytes + document_bytes
            truncated = len(retained) < len(candidates)
            next_cursor = (
                encode_recall_evidence_cursor(
                    record_kind="receipt",
                    query_fingerprint=query_fingerprint,
                    created_at=retained[-1].created_at,
                    record_id=retained[-1].receipt_id,
                )
                if truncated and retained
                else None
            )
            return RecallReceiptPage(
                items=tuple(retained),
                next_cursor=next_cursor,
                truncated=truncated,
            )

    async def create_context_exposure(
        self,
        exposure: ContextExposure,
        item_exposures: tuple[RecallItemExposure, ...] = (),
    ) -> ContextExposure:
        copied = copy_context_exposure(exposure)
        copied_items = tuple(copy_recall_item_exposure(item) for item in item_exposures)
        validate_new_context_exposure(copied, copied_items)

        async with self._lock:
            current = self._context_exposures.get(copied.exposure_id)
            if current is not None:
                current_items = self._recall_item_exposures[copied.exposure_id]
                if not context_exposure_creation_matches(current, copied) or tuple(
                    memory_evidence_document_bytes(item, "stored recall item exposure")
                    for item in current_items
                ) != tuple(
                    memory_evidence_document_bytes(item, "recall item exposure")
                    for item in copied_items
                ):
                    raise RecallEvidenceConflict("Context exposure", copied.exposure_id)
                return copy_context_exposure(current)
            if copied.session_id not in self._sessions:
                raise KeyError(f"Session not found: {copied.session_id}")
            for owner in self._session_closure_progress.values():
                _check_closure_lineage_owner(owner, (copied.session_id,))

            attempt_key = copied.session_id, copied.model_attempt_id
            conflicting_exposure_id = self._context_exposure_by_model_attempt.get(attempt_key)
            if conflicting_exposure_id is not None:
                raise RecallEvidenceConflict("Model-attempt exposure", conflicting_exposure_id)
            provider_attempt_key = copied.session_id, copied.provider_attempt_id
            conflicting_exposure_id = self._context_exposure_by_provider_attempt.get(
                provider_attempt_key
            )
            if conflicting_exposure_id is not None:
                raise RecallEvidenceConflict(
                    "Provider-attempt exposure",
                    conflicting_exposure_id,
                )

            receipts: dict[str, RecallReceipt] = {}
            for receipt_id in copied.receipt_ids:
                receipt = self._recall_receipts.get(receipt_id)
                if receipt is None:
                    raise KeyError(f"Recall receipt not found: {receipt_id}")
                validate_context_exposure_receipt_scope(copied, receipt)
                receipts[receipt_id] = receipt
            for receipt_id in copied.carried_receipt_ids:
                receipt = self._recall_receipts.get(receipt_id)
                if receipt is None:
                    raise KeyError(f"Recall receipt not found: {receipt_id}")
                validate_context_exposure_carried_receipt_scope(copied, receipt)
            for item in copied_items:
                if not recall_item_exposure_matches_receipt_item(
                    item,
                    receipts[item.receipt_id],
                ):
                    raise ValueError(
                        "Recall item exposure differs from its immutable receipt item."
                    )

            self._context_exposures[copied.exposure_id] = copied
            self._context_exposure_ids_by_session.setdefault(copied.session_id, set()).add(
                copied.exposure_id
            )
            self._recall_item_exposures[copied.exposure_id] = copied_items
            self._context_exposure_by_model_attempt[attempt_key] = copied.exposure_id
            self._context_exposure_by_provider_attempt[provider_attempt_key] = copied.exposure_id
            page_key = copied.created_at, copied.exposure_id
            for page_keys in (
                self._context_exposure_page_keys_by_session.setdefault(copied.session_id, []),
                self._context_exposure_page_keys_by_interaction.setdefault(
                    (copied.session_id, copied.interaction_id), []
                ),
                self._context_exposure_page_keys_by_step.setdefault(
                    (copied.session_id, copied.model_step_id), []
                ),
                self._context_exposure_page_keys_by_interaction_step.setdefault(
                    (copied.session_id, copied.interaction_id, copied.model_step_id), []
                ),
            ):
                page_keys.insert(bisect_right(page_keys, page_key), page_key)
            return copy_context_exposure(copied)

    async def load_context_exposure(
        self,
        session_id: str,
        exposure_id: str,
    ) -> ContextExposure | None:
        session_id = require_memory_evidence_session_id(session_id)
        exposure_id = require_memory_evidence_id(exposure_id, "exposure_id")
        async with self._lock:
            exposure = self._context_exposures.get(exposure_id)
            if exposure is None or exposure.session_id != session_id:
                return None
            return copy_context_exposure(exposure)

    async def load_recall_item_exposures(
        self,
        session_id: str,
        exposure_id: str,
    ) -> tuple[RecallItemExposure, ...]:
        session_id = require_memory_evidence_session_id(session_id)
        exposure_id = require_memory_evidence_id(exposure_id, "exposure_id")
        async with self._lock:
            exposure = self._context_exposures.get(exposure_id)
            if exposure is None or exposure.session_id != session_id:
                return ()
            if len(self._recall_item_exposures[exposure_id]) > MAX_RECALL_RECEIPT_ITEMS:
                raise ValueError("Stored recall item exposures exceed their count bound.")
            return tuple(
                copy_recall_item_exposure(item) for item in self._recall_item_exposures[exposure_id]
            )

    async def list_context_exposures(
        self,
        query: RecallEvidenceQuery,
    ) -> ContextExposurePage:
        copied_query = RecallEvidenceQuery.model_validate(query.model_dump(mode="python"))
        query_fingerprint = copied_query.fingerprint("exposure")
        after = (
            None
            if copied_query.cursor is None
            else decode_recall_evidence_cursor(
                copied_query.cursor,
                record_kind="exposure",
                query_fingerprint=query_fingerprint,
            )
        )
        async with self._lock:
            if copied_query.interaction_id is not None and copied_query.model_step_id is not None:
                page_keys = self._context_exposure_page_keys_by_interaction_step.get(
                    (
                        copied_query.session_id,
                        copied_query.interaction_id,
                        copied_query.model_step_id,
                    ),
                    (),
                )
            elif copied_query.interaction_id is not None:
                page_keys = self._context_exposure_page_keys_by_interaction.get(
                    (copied_query.session_id, copied_query.interaction_id),
                    (),
                )
            elif copied_query.model_step_id is not None:
                page_keys = self._context_exposure_page_keys_by_step.get(
                    (copied_query.session_id, copied_query.model_step_id),
                    (),
                )
            else:
                page_keys = self._context_exposure_page_keys_by_session.get(
                    copied_query.session_id,
                    (),
                )
            start = 0 if after is None else bisect_right(page_keys, after)
            candidates = tuple(
                self._context_exposures[exposure_id]
                for _, exposure_id in islice(
                    page_keys,
                    start,
                    start + copied_query.limit + 1,
                )
            )
            retained: list[ContextExposure] = []
            retained_bytes = 2
            for exposure in candidates:
                document_bytes = len(
                    memory_evidence_document_bytes(exposure, "context exposure page item")
                )
                separator_bytes = 1 if retained else 0
                if len(retained) >= copied_query.limit or (
                    retained_bytes + separator_bytes + document_bytes > copied_query.max_bytes
                ):
                    break
                retained.append(exposure)
                retained_bytes += separator_bytes + document_bytes
            truncated = len(retained) < len(candidates)
            next_cursor = (
                encode_recall_evidence_cursor(
                    record_kind="exposure",
                    query_fingerprint=query_fingerprint,
                    created_at=retained[-1].created_at,
                    record_id=retained[-1].exposure_id,
                )
                if truncated and retained
                else None
            )
            return ContextExposurePage(
                items=tuple(retained),
                next_cursor=next_cursor,
                truncated=truncated,
            )

    async def transition_context_exposure(
        self,
        session_id: str,
        exposure_id: str,
        request: ContextExposureTransitionRequest,
    ) -> ContextExposure:
        session_id = require_memory_evidence_session_id(session_id)
        exposure_id = require_memory_evidence_id(exposure_id, "exposure_id")
        copied_request = ContextExposureTransitionRequest.model_validate(
            request.model_dump(mode="python")
        )
        async with self._lock:
            exposure = self._context_exposures.get(exposure_id)
            if exposure is None or exposure.session_id != session_id:
                raise KeyError(f"Context exposure not found: {exposure_id}")
            replay = next(
                (
                    transition
                    for transition in exposure.transitions
                    if transition.transition_id == copied_request.transition_id
                ),
                None,
            )
            if replay is not None:
                if not context_exposure_transition_replays(exposure, copied_request):
                    raise RecallEvidenceConflict(
                        "Context exposure transition",
                        copied_request.transition_id,
                    )
                return copy_context_exposure(exposure)
            for owner in self._session_closure_progress.values():
                _check_closure_lineage_owner(owner, (session_id,))
            updated = append_context_exposure_transition(exposure, copied_request)
            self._context_exposures[exposure_id] = updated
            return copy_context_exposure(updated)

    async def checkpoint(self, session_id: str, state: dict[str, Any]) -> None:
        session_id = require_clean_nonblank(session_id, "session_id")
        if not isinstance(state, dict):
            raise ValueError("Checkpoint state must be a dictionary.")
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            invocation_scope._assert_session_run_epoch(session_id, session)
            for owner in self._session_closure_progress.values():
                _check_closure_lineage_owner(owner, (session_id,))
            self._store_checkpoint_unlocked(
                session_id,
                checkpoint_preservation._replace_checkpoint_preserving_completion_result_event_publications(
                    self._checkpoints.get(session_id),
                    state,
                    session_id=session_id,
                ),
            )
            self._sessions[session_id] = session.model_copy(
                update={"last_activity_at": self._ownership_clock()}
            )

    async def transform_checkpoint(
        self,
        session_id: str,
        checkpoint_transform: CheckpointTransform,
    ) -> None:
        session_id = require_clean_nonblank(session_id, "session_id")
        if checkpoint_transform is None:
            raise TypeError("checkpoint_transform is required.")
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            invocation_scope._assert_session_run_epoch(session_id, session)
            for owner in self._session_closure_progress.values():
                _check_closure_lineage_owner(owner, (session_id,))
            current = self._checkpoints.get(session_id)
            transformed = checkpoint_transform(
                session.model_copy(deep=True),
                checkpoint_preservation._copy_checkpoint_for_transform(
                    current, session_id=session_id
                ),
            )
            if transformed is None:
                return
            self._store_checkpoint_unlocked(
                session_id,
                checkpoint_preservation._replace_checkpoint_preserving_completion_result_event_publications(
                    current,
                    copy_durable_json_object(transformed, "checkpoint"),
                    session_id=session_id,
                ),
            )
            self._sessions[session_id] = session.model_copy(
                update={"last_activity_at": self._ownership_clock()}
            )

    async def load_checkpoint(self, session_id: str) -> dict[str, Any] | None:
        from cayu.resource_access import current_data_bounds

        access_bounds = await current_data_bounds()
        session_id = require_clean_nonblank(session_id, "session_id")
        async with self._lock:
            if access_bounds is not None:
                access_bounds.require_action(self._sessions.get(session_id), "inspect_state")
            checkpoint = self._checkpoints.get(session_id)
            if checkpoint is None:
                return None
            return deepcopy(checkpoint)

    async def load_execution_snapshot_checkpoint(self, session_id: str) -> dict[str, Any] | None:
        session_id = require_clean_nonblank(session_id, "session_id")
        async with self._lock:
            checkpoint = self._checkpoints.get(session_id)
            return (
                None
                if checkpoint is None
                else deepcopy(
                    {
                        key: value
                        for key, value in checkpoint.items()
                        if key in {"checkpoint_schema_version", "execution_snapshots"}
                    }
                )
            )

    async def load_interruption_cascade_marker(
        self,
        session_id: str,
        *,
        checkpoint_root_guard: CheckpointRootFieldGuard | None = None,
    ) -> dict[str, Any] | None:
        session_id = require_clean_nonblank(session_id, "session_id")
        async with self._lock:
            checkpoint = self._checkpoints.get(session_id)
            if checkpoint_root_guard is not None:
                checkpoint_root_guard.validate(
                    session_id,
                    checkpoint_root_field_projection(
                        checkpoint,
                        checkpoint_root_guard.key,
                    ),
                )
            marker = (
                _INTERRUPTION_CASCADE_MISSING
                if checkpoint is None or "pending_interruption_cascade" not in checkpoint
                else checkpoint["pending_interruption_cascade"]
            )
            return _project_interruption_cascade_marker(marker)


def _audit_resolution_actor(actor: ResolutionActor | None) -> ResolutionActor | None:
    if actor is None:
        return None
    return ResolutionActor(
        subject=actor.subject,
        tenant=actor.tenant,
        source=actor.source,
    )


def _require_session_export_target(session_id: str, key: str) -> None:
    """Bind private key access to its target before returning callback input."""
    from cayu.collaboration._session_export_store import OPERATION_PREFIX, checkpoint_visible
    from cayu.collaboration.exports import SessionExportConflict

    if key.startswith(OPERATION_PREFIX) and not checkpoint_visible(session_id=session_id):
        raise SessionExportConflict()


def _prepare_initial_session_operation_records(
    session: Session,
    initializer: SessionOperationInitializer | None,
) -> dict[str, dict[str, Any]]:
    """Run one initializer against a detached session and validate its records."""

    if initializer is None:
        return {}
    raw_records = initializer(session.model_copy(deep=True))
    if not isinstance(raw_records, Mapping):
        raise TypeError("Session operation initializer must return a mapping.")
    if len(raw_records) > RUNTIME_PUBLICATION_MAX_CHECKPOINT_OPERATIONS:
        raise ValueError(
            "Initial session operation records cannot exceed "
            f"{RUNTIME_PUBLICATION_MAX_CHECKPOINT_OPERATIONS} entries."
        )
    copied = copy_durable_json_object(
        dict(raw_records),
        "initial_session_operation_records",
    )
    from cayu.collaboration._session_export_store import OPERATION_PREFIX

    if any(key.startswith(OPERATION_PREFIX) for key in copied):
        raise ValueError("Session export records require their publication owner.")
    operation_record_rules._validate_session_operation_record_keys(copied)
    for record in copied.values():
        if type(record) is not dict:
            raise ValueError("Initial session operation records must be objects.")
    return copied


def _prepare_queue_completion_checkpoint_mutation(
    value: dict[str, Any] | None,
    to_status: SessionStatus,
) -> RuntimePublicationMutation | None:
    if value is None:
        return None
    if to_status is not SessionStatus.RUNNING:
        raise ValueError("Queue completion checkpointing must retain running status.")
    copied = copy_durable_json_object(value, "checkpoint_mutation")
    completion_finalization._validate_completion_finalization_mutation_shape(copied)
    mutation = RuntimePublicationMutation.model_validate(copied)
    completion_finalization.pending_completion_finalization_from_checkpoint(
        {PENDING_COMPLETION_FINALIZATION_CHECKPOINT_KEY: mutation.operations[0].value}
    )
    return mutation


def _apply_queue_completion_checkpoint_mutation(
    session: Session,
    mutation: RuntimePublicationMutation,
    checkpoint: dict[str, Any] | None,
) -> dict[str, Any]:
    """Check durable owner authority under the session lock, then apply the marker CAS."""

    active = active_invocation_execution_profile_from_checkpoint(checkpoint)
    marker = mutation.operations[0].value
    if (
        session.status is not SessionStatus.RUNNING
        or active is None
        or active.session_id != session.id
        or active.run_epoch != session.run_epoch
        or marker["execution_profile_fingerprint"] != active.profile.fingerprint
        or marker["environment_name"] != session.environment_name
    ):
        raise SessionRunFenced("Queue completion finalization lost its active owner authority.")
    updated = checkpoint_mutation_rules._apply_runtime_publication_checkpoint_mutation(
        mutation, checkpoint
    )
    assert updated is not None
    return updated


_CHILD_LIFECYCLE_EVENT_TYPE_BY_TERMINAL_STATUS = {
    SessionStatus.COMPLETED: EventType.SESSION_COMPLETED,
    SessionStatus.FAILED: EventType.SESSION_FAILED,
    SessionStatus.INTERRUPTED: EventType.SESSION_INTERRUPTED,
}


def _child_session_lifecycle_state(status: SessionStatus) -> ChildSessionLifecycleState:
    if status is SessionStatus.PENDING:
        return ChildSessionLifecycleState.ADMITTED
    if status in {SessionStatus.RUNNING, SessionStatus.INTERRUPTING}:
        return ChildSessionLifecycleState.RUNNING
    return {
        SessionStatus.COMPLETED: ChildSessionLifecycleState.COMPLETED,
        SessionStatus.FAILED: ChildSessionLifecycleState.FAILED,
        SessionStatus.INTERRUPTED: ChildSessionLifecycleState.INTERRUPTED,
    }[status]


def _child_session_lifecycle_occurrence(
    child: Session,
    records_by_type: Mapping[EventType, EventRecord],
) -> tuple[ChildSessionRelationship, ChildSessionLifecycleOccurrence] | None:
    fork_record = records_by_type.get(EventType.SESSION_FORKED)
    relationship = (
        ChildSessionRelationship.SESSION_FORK
        if fork_record is not None
        else ChildSessionRelationship.DIRECT_CHILD
    )
    started_record = records_by_type.get(EventType.SESSION_STARTED)
    resumed_record = records_by_type.get(EventType.SESSION_RESUMED)
    activation_record = max(
        (record for record in (started_record, resumed_record, fork_record) if record is not None),
        key=lambda record: record.sequence,
        default=None,
    )
    terminal_record = max(
        (
            record
            for event_type in _CHILD_LIFECYCLE_EVENT_TYPE_BY_TERMINAL_STATUS.values()
            if (record := records_by_type.get(event_type)) is not None
        ),
        key=lambda record: record.sequence,
        default=None,
    )
    latest_lifecycle_record = max(
        (record for record in (activation_record, terminal_record) if record is not None),
        key=lambda record: record.sequence,
        default=None,
    )
    terminal_type = _CHILD_LIFECYCLE_EVENT_TYPE_BY_TERMINAL_STATUS.get(child.status)
    if terminal_type is not None:
        record = latest_lifecycle_record
        if record is None or record.event.type is not terminal_type:
            return None
    elif child.status in {SessionStatus.RUNNING, SessionStatus.INTERRUPTING}:
        if latest_lifecycle_record is None or latest_lifecycle_record is not activation_record:
            return None
        record = latest_lifecycle_record
    else:
        if started_record is not None or resumed_record is not None or terminal_record is not None:
            return None
        record = fork_record
    if record is None:
        if child.status is SessionStatus.PENDING:
            return relationship, ChildSessionLifecycleOccurrence(
                source=ChildSessionLifecycleOccurrenceSource.SESSION,
                source_id=child.instance_id,
                source_type=CHILD_SESSION_ADMISSION_OCCURRENCE_TYPE,
                occurred_at=child.created_at,
            )
        return None
    return relationship, ChildSessionLifecycleOccurrence(
        source=ChildSessionLifecycleOccurrenceSource.EVENT,
        source_id=record.event.id,
        source_sequence=record.sequence,
        source_type=str(record.event.type),
        occurred_at=record.event.timestamp,
    )


def _child_session_notification_consumption(
    record: Mapping[str, Any] | None,
    *,
    parent: Session,
    child: Session,
    occurrence: ChildSessionLifecycleOccurrence,
) -> ChildSessionNotificationConsumption | None:
    if record is None:
        return None
    try:
        consumption = ChildSessionNotificationConsumption.model_validate(record)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Child-session notification consumption is malformed.") from exc
    if (
        consumption.parent_session_id != parent.id
        or consumption.parent_session_instance_id != parent.instance_id
        or consumption.child_session_id != child.id
        or consumption.child_session_instance_id != child.instance_id
        or consumption.occurrence != occurrence
    ):
        raise RuntimeError("Child-session notification consumption changed its authority.")
    return consumption


def _child_session_lifecycle_entry(
    *,
    parent: Session,
    child: Session,
    records_by_type: Mapping[EventType, EventRecord],
    consumption_record: Mapping[str, Any] | None,
) -> ChildSessionLifecycleEntry | None:
    if child.parent_session_id != parent.id:
        raise RuntimeError("Child-session lifecycle candidate changed its parent authority.")
    source = _child_session_lifecycle_occurrence(child, records_by_type)
    if source is None:
        return None
    relationship, occurrence = source
    state = _child_session_lifecycle_state(child.status)
    consumption = (
        _child_session_notification_consumption(
            consumption_record,
            parent=parent,
            child=child,
            occurrence=occurrence,
        )
        if state.terminal
        else None
    )
    return ChildSessionLifecycleEntry(
        parent_session_id=parent.id,
        parent_session_instance_id=parent.instance_id,
        child_session_id=child.id,
        child_session_instance_id=child.instance_id,
        relationship=relationship,
        state=state,
        occurrence=occurrence,
        freshness=(
            ChildSessionNotificationFreshness.CURRENT
            if not state.terminal
            else ChildSessionNotificationFreshness.FRESH
            if consumption is None
            else ChildSessionNotificationFreshness.CONSUMED
        ),
        consumed_by_stage_id=None if consumption is None else consumption.stage_id,
        created_at=child.created_at,
        updated_at=child.updated_at,
    )


def _child_session_lifecycle_entry_sort_key(
    entry: ChildSessionLifecycleEntry,
) -> tuple[int, datetime, str]:
    priority = (
        0
        if entry.freshness is ChildSessionNotificationFreshness.FRESH
        else 1
        if entry.freshness is ChildSessionNotificationFreshness.CURRENT
        else 2
    )
    return priority, entry.occurrence.occurred_at, entry.child_session_id


def _child_session_notification_consumption_record(
    *,
    parent: Session,
    child: Session,
    occurrence: ChildSessionLifecycleOccurrence,
    stage: ModelCompletionStage,
    consumed_at: datetime,
) -> ChildSessionNotificationConsumption:
    binding = child_session_notification_stage_binding(stage.intent)
    if binding is None:
        raise ValueError("Model stage has no child-session notification binding.")
    if (
        binding.parent_session_id != parent.id
        or binding.parent_session_instance_id != parent.instance_id
        or stage.session_id != parent.id
    ):
        raise SessionModelCompletionStageConflict(
            "Child-session notification binding changed its parent authority."
        )
    claim = next(
        (
            candidate
            for candidate in binding.claims
            if candidate.child_session_id == child.id
            and candidate.occurrence.source_id == occurrence.source_id
        ),
        None,
    )
    if (
        claim is None
        or claim.child_session_instance_id != child.instance_id
        or claim.occurrence != occurrence
        or child.parent_session_id != parent.id
        or not _child_session_lifecycle_state(child.status).terminal
        or occurrence.source is not ChildSessionLifecycleOccurrenceSource.EVENT
        or occurrence.source_type
        != str(_CHILD_LIFECYCLE_EVENT_TYPE_BY_TERMINAL_STATUS.get(child.status))
    ):
        raise SessionModelCompletionStageConflict(
            "Child-session notification claim no longer matches canonical terminal state."
        )
    model_step_id = stage.intent.get("model_step_id")
    model_attempt_id = stage.intent.get("model_attempt_id")
    if type(model_step_id) is not str or type(model_attempt_id) is not str:
        raise SessionModelCompletionStageConflict(
            "Child-session notification binding lost its model-attempt identity."
        )
    return ChildSessionNotificationConsumption(
        parent_session_id=parent.id,
        parent_session_instance_id=parent.instance_id,
        child_session_id=child.id,
        child_session_instance_id=child.instance_id,
        occurrence=occurrence,
        stage_id=stage.stage_id,
        model_step_id=model_step_id,
        model_attempt_id=model_attempt_id,
        source_run_epoch=stage.source_run_epoch,
        consumed_at=consumed_at,
    )


def _child_session_notification_consumption_replays(
    existing: Mapping[str, Any],
    requested: ChildSessionNotificationConsumption,
) -> bool:
    """Allow transport retries only within the same logical model step."""

    try:
        durable = ChildSessionNotificationConsumption.model_validate(existing)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Child-session notification consumption is malformed.") from exc
    return (
        durable.parent_session_id == requested.parent_session_id
        and durable.parent_session_instance_id == requested.parent_session_instance_id
        and durable.child_session_id == requested.child_session_id
        and durable.child_session_instance_id == requested.child_session_instance_id
        and durable.occurrence == requested.occurrence
        and durable.model_step_id == requested.model_step_id
        and durable.source_run_epoch == requested.source_run_epoch
    )


def _copy_workflow_step_reservation(
    session_id: str,
    event: Event,
    *,
    workflow_name: str,
    attempt_id: str,
) -> tuple[str, Event, str, str]:
    session_id, copied_events = session_event_rules._copy_session_event_batch(session_id, [event])
    copied_event = copied_events[0]
    workflow_name = require_clean_nonblank(workflow_name, "workflow_name")
    attempt_id = require_clean_nonblank(attempt_id, "attempt_id")
    if copied_event.type != EventType.WORKFLOW_STEP_STARTED:
        raise ValueError("Workflow step reservation requires a workflow.step.started event.")
    if copied_event.workflow_name != workflow_name:
        raise ValueError("Workflow step reservation event has the wrong workflow_name.")
    if copied_event.payload.get("attempt_id") != attempt_id:
        raise ValueError("Workflow step reservation event has the wrong attempt_id.")
    step_id = copied_event.payload.get("step_id")
    if not isinstance(step_id, str) or not step_id:
        raise ValueError("Workflow step reservation requires a non-empty step_id.")
    return session_id, copied_event, workflow_name, attempt_id


def _copy_mcp_manifest_publication(
    session_id: str,
    *,
    expected_generations: dict[str, int | None],
    baseline_updates: dict[str, McpManifestBaseline],
    events: list[Event],
) -> tuple[
    str,
    dict[str, int | None],
    dict[str, McpManifestBaseline],
    list[Event],
]:
    if type(expected_generations) is not dict or not expected_generations:
        raise ValueError("expected_generations must be a non-empty dict.")
    expected: dict[str, int | None] = {}
    for key, generation in expected_generations.items():
        key = _require_sha256_identifier(key, "expected_generations key")
        if generation is not None and (type(generation) is not int or generation < 1):
            raise ValueError("Expected MCP manifest generations must be positive integers or None.")
        expected[key] = generation

    if type(baseline_updates) is not dict:
        raise TypeError("baseline_updates must be a dict.")
    updates: dict[str, McpManifestBaseline] = {}
    for key, baseline in baseline_updates.items():
        key = _require_sha256_identifier(key, "baseline_updates key")
        if key not in expected:
            raise ValueError("MCP baseline update has no matching generation expectation.")
        if type(baseline) is not McpManifestBaseline:
            raise TypeError("baseline_updates must contain McpManifestBaseline values.")
        copied = McpManifestBaseline.model_validate(baseline.model_dump(mode="python"))
        if copied.history_key != key:
            raise ValueError("MCP baseline history_key does not match its update key.")
        expected_generation = expected[key]
        next_generation = 1 if expected_generation is None else expected_generation + 1
        if copied.generation != next_generation:
            raise ValueError("MCP baseline update generation does not follow its expectation.")
        updates[key] = copied

    session_id, copied_events = session_event_rules._copy_session_event_batch(session_id, events)
    if not copied_events:
        raise ValueError("MCP manifest publication requires at least one event.")
    for event in copied_events:
        if event.type not in {
            EventType.MCP_MANIFEST_CHECKED,
            EventType.MCP_MANIFEST_BLOCKED,
        }:
            raise ValueError("MCP manifest publication contains an unrelated event.")
        history_key = event.payload.get("history_key")
        if not isinstance(history_key, str) or history_key not in expected:
            raise ValueError("MCP manifest event has no matching history expectation.")
        if (
            event.type == EventType.MCP_MANIFEST_CHECKED
            and event.payload.get("status") in {"first_seen", "changed"}
            and event.payload.get("outcome") == "accepted"
        ):
            baseline = updates.get(history_key)
            if baseline is None or baseline.accepted_event_id != event.id:
                raise ValueError(
                    "Every accepted MCP manifest transition must have exactly one "
                    "matching baseline update."
                )
    events_by_id = {event.id: event for event in copied_events}
    for history_key, baseline in updates.items():
        accepted_event = events_by_id.get(baseline.accepted_event_id)
        if (
            accepted_event is None
            or accepted_event.type != EventType.MCP_MANIFEST_CHECKED
            or _mcp_manifest_session_ref(accepted_event.session_id) != baseline.accepted_session_ref
            or accepted_event.timestamp != baseline.accepted_at
            or accepted_event.payload.get("history_key") != history_key
            or accepted_event.payload.get("manifest_identity") != baseline.manifest_identity
            or accepted_event.payload.get("manifest_hash") != baseline.manifest_hash
            or accepted_event.payload.get("source_manifest_hash") != baseline.source_manifest_hash
            or accepted_event.payload.get("server_hash") != baseline.server_hash
            or accepted_event.payload.get("status") not in {"first_seen", "changed"}
            or accepted_event.payload.get("outcome") != "accepted"
        ):
            raise ValueError(
                "MCP baseline update must match exactly one accepted manifest-check event."
            )
    return session_id, expected, updates, copied_events


def _validate_equivalent_message_delivery(
    existing: _InMemoryMessageDeliveryRecord,
    *,
    session_id: str,
    include_on_idle: bool,
    eligible_through: int | None,
    limit: int,
    interaction_id: str | None,
    interaction_started_event: Event | None,
    profile_handoff: QueuedInteractionProfileHandoff | None,
    reject_only: bool = False,
) -> None:
    if (
        existing.session_id != session_id
        or existing.include_on_idle != include_on_idle
        or existing.requested_eligible_through != eligible_through
        or existing.limit != limit
        or existing.interaction_id != interaction_id
        or existing.interaction_started_event != interaction_started_event
        or existing.profile_handoff != profile_handoff
        or existing.reject_only != reject_only
    ):
        raise ValueError("delivery_id was already used for a different queue delivery.")


def _validate_message_delivery_eligible_through(value: int | None) -> int | None:
    if value is None:
        return None
    if type(value) is not int or not 0 <= value <= MAX_DURABLE_JSON_INTEGER:
        raise ValueError(
            f"eligible_through must be an integer between 0 and {MAX_DURABLE_JSON_INTEGER}."
        )
    return value


def _validate_interaction_page(
    before_sequence: int | None,
    limit: int,
) -> tuple[int | None, int]:
    if before_sequence is not None and (type(before_sequence) is not int or before_sequence < 1):
        raise ValueError("before_sequence must be a positive integer.")
    if type(limit) is not int or limit < 1 or limit > 5000:
        raise ValueError("limit must be between 1 and 5000.")
    return before_sequence, limit


def _session_matches_debug_state(
    session: Session,
    records: list[EventRecord],
    debug_state: SessionDebugState | None,
) -> bool:
    if debug_state is None:
        return True
    has_tool_debug_event = any(_is_tool_debug_event(record.event.type) for record in records)
    has_session_failure = session.status == SessionStatus.FAILED
    has_interruption = session.status == SessionStatus.INTERRUPTED
    if debug_state == SessionDebugState.TOOL_ISSUE:
        return has_tool_debug_event
    if debug_state == SessionDebugState.SESSION_FAILURE:
        return has_session_failure
    if debug_state == SessionDebugState.INTERRUPTION:
        return has_interruption
    if debug_state == SessionDebugState.NEEDS_ATTENTION:
        return has_session_failure or has_interruption or has_tool_debug_event
    return False


def _is_tool_debug_event(event_type: EventType | str) -> bool:
    return event_type in {EventType.TOOL_CALL_FAILED, EventType.TOOL_CALL_BLOCKED}


# Sessions that must be interrupted before they can be deleted (in-flight work).
DELETE_BLOCKED_SESSION_STATUSES = frozenset({SessionStatus.RUNNING, SessionStatus.INTERRUPTING})
INITIAL_TRANSCRIPT_PENDING_CHECKPOINT_KEY = "initial_transcript_pending"
# Storage implementations import these aliases while collecting the same
# bounded evidence consumed by the shared runtime classifier.
_TERMINAL_PUBLICATION_EVIDENCE_EVENT_TYPES = TERMINAL_EVIDENCE_EVENT_TYPES
_TERMINAL_PUBLICATION_EVIDENCE_QUERY_LIMIT = TERMINAL_EVIDENCE_QUERY_LIMIT
_TERMINAL_PENDING_ACTION_CHECKPOINT_KEYS = frozenset(
    {
        "pending_tool_approval",
        "pending_user_input",
        tool_publication_rules._PENDING_TOOL_ROUND_CHECKPOINT_KEY,
    }
)


def _is_durable_subagent_child(session: Session) -> bool:
    """Return positive runtime metadata evidence for a task-backed subagent child."""

    if type(session) is not Session:
        raise TypeError("Durable subagent child classification requires an exact session.")
    subagent = session.metadata.get("subagent")
    return type(subagent) is dict and subagent.get("mode") == "durable"


def _durable_subagent_parent_delete_block_reason(child_session_id: str) -> str:
    child_session_id = require_clean_nonblank(child_session_id, "child_session_id")
    return (
        "Cannot delete a session while durable subagent child authority still exists; "
        f"settle and delete the child first: {child_session_id}"
    )


def _terminal_publication_delete_block_reason(
    *,
    session: Session,
    checkpoint: dict[str, Any] | None,
    evidence_events: Sequence[Event],
) -> str | None:
    """Return why deletion would erase repairable current-run terminal evidence.

    ``evidence_events`` must contain the newest lifecycle/terminal events in
    descending session-sequence order, bounded by
    ``_TERMINAL_PUBLICATION_EVIDENCE_QUERY_LIMIT``.
    """

    active_profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
    if active_profile is not None and not active_invocation_execution_profile_is_released(
        active_profile, session_id=session.id, run_epoch=session.run_epoch
    ):
        return "invocation terminal hooks or trailing cleanup still own the run fence"

    expected_event_type = terminal_session_evidence._TERMINAL_PUBLICATION_EVENT_TYPE_BY_STATUS.get(
        session.status
    )
    if expected_event_type is None:
        return None
    if len(evidence_events) > _TERMINAL_PUBLICATION_EVIDENCE_QUERY_LIMIT:
        raise ValueError("Terminal publication deletion evidence exceeds its bounded query.")

    if checkpoint is not None and "pending_session_interrupt" in checkpoint:
        return "pending interruption terminal publication is incomplete"

    run_operation = terminal_event_evidence._session_run_operation_from_checkpoint(checkpoint)
    classification = classify_current_terminal_evidence(
        evidence_events=evidence_events,
        expected_event_type=expected_event_type,
        run_operation_id=(None if run_operation is None else run_operation.operation_id),
        interruption_request_id=None,
    )
    terminal_events = classification.events
    if classification.run_operation_conflict:
        return "terminal publication evidence contradicts the pending run operation"

    if any(event.type != expected_event_type for event in terminal_events):
        return "terminal publication evidence contradicts the durable session status"
    if len(terminal_events) > 1:
        return "terminal publication contains duplicate current-run evidence"

    pending_action = checkpoint is not None and any(
        key in checkpoint for key in _TERMINAL_PENDING_ACTION_CHECKPOINT_KEYS
    )
    terminal_event_required = (
        run_operation is not None
        or pending_action
        or classification.latest_lifecycle_event_type != EventType.SESSION_FORKED
    )
    if terminal_event_required and not terminal_events:
        return "terminal publication evidence is incomplete"
    return None


def _initial_transcript_pending_checkpoint(
    session: Session,
    interaction_id: str,
    *,
    checkpoint_transform: CheckpointTransform | None = None,
) -> dict[str, Any]:
    """Create the authority marker held until initial transcript publication."""

    checkpoint = {
        CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION,
        INITIAL_TRANSCRIPT_PENDING_CHECKPOINT_KEY: {
            "version": 1,
            "interaction_id": require_clean_nonblank(interaction_id, "interaction_id"),
        },
    }
    if checkpoint_transform is None:
        return checkpoint
    transformed = checkpoint_transform(session, checkpoint)
    if type(transformed) is not dict:
        raise TypeError("Initial transcript checkpoint transform must return an object.")
    return (
        checkpoint_preservation._replace_checkpoint_preserving_completion_result_event_publications(
            checkpoint,
            copy_durable_json_object(transformed, "checkpoint"),
            session_id=session.id,
        )
    )


def _initial_transcript_pending_interaction_id(
    checkpoint: dict[str, Any] | None,
) -> str | None:
    if checkpoint is None or INITIAL_TRANSCRIPT_PENDING_CHECKPOINT_KEY not in checkpoint:
        return None
    marker = checkpoint[INITIAL_TRANSCRIPT_PENDING_CHECKPOINT_KEY]
    if type(marker) is not dict or marker.get("version") != 1:
        raise ValueError("Initial transcript authority marker is invalid.")
    interaction_id = marker.get("interaction_id")
    if type(interaction_id) is not str:
        raise ValueError("Initial transcript authority marker is invalid.")
    return require_clean_nonblank(interaction_id, "interaction_id")


def _checkpoint_after_initial_transcript_publication(
    checkpoint: dict[str, Any] | None,
    *,
    interaction_id: str,
) -> dict[str, Any] | None:
    pending_interaction_id = _initial_transcript_pending_interaction_id(checkpoint)
    if pending_interaction_id is None:
        # Deferred admission is also used after session creation, when there is
        # no environment-derived bootstrap authority to protect.
        return checkpoint
    if pending_interaction_id != interaction_id:
        raise RuntimeError("Initial transcript authority changed before finalization.")
    assert checkpoint is not None
    updated = copy_durable_json_object(checkpoint, "checkpoint")
    updated.pop(INITIAL_TRANSCRIPT_PENDING_CHECKPOINT_KEY)
    return updated or None


_SESSION_RUN_OPERATION_ID_PAYLOAD_KEY = SESSION_RUN_OPERATION_ID_PAYLOAD_KEY
_QUEUED_DISPATCH_TERMINAL_RECEIPTS_CHECKPOINT_KEY = "queued_dispatch_terminal_receipts"


@dataclass(frozen=True)
class _QueuedDispatchTerminalReceiptIdentity:
    queue_task_id: str
    terminal_event_id: str
    run_epoch: int


def _queued_dispatch_terminal_receipts_from_checkpoint(
    checkpoint: dict[str, Any] | None,
) -> dict[str, _QueuedDispatchTerminalReceiptIdentity]:
    """Parse terminal evidence retained until its queue task is acknowledged."""

    if checkpoint is None or _QUEUED_DISPATCH_TERMINAL_RECEIPTS_CHECKPOINT_KEY not in checkpoint:
        return {}
    marker = checkpoint[_QUEUED_DISPATCH_TERMINAL_RECEIPTS_CHECKPOINT_KEY]
    if type(marker) is not dict or set(marker) != {"version", "receipts"}:
        raise ValueError("Queued dispatch terminal receipt checkpoint is invalid.")
    if marker.get("version") != 1 or type(marker.get("receipts")) is not dict:
        raise ValueError("Queued dispatch terminal receipt checkpoint is invalid.")
    receipts: dict[str, _QueuedDispatchTerminalReceiptIdentity] = {}
    for operation_id, receipt in marker["receipts"].items():
        operation_id = require_clean_nonblank(
            operation_id,
            "queued_dispatch_terminal_receipt.operation_id",
        )
        if type(receipt) is not dict or set(receipt) != {
            "queue_task_id",
            "terminal_event_id",
            "run_epoch",
        }:
            raise ValueError("Queued dispatch terminal receipt is invalid.")
        queue_task_id = require_clean_nonblank(
            receipt.get("queue_task_id"),
            "queued_dispatch_terminal_receipt.queue_task_id",
        )
        terminal_event_id = require_clean_nonblank(
            receipt.get("terminal_event_id"),
            "queued_dispatch_terminal_receipt.terminal_event_id",
        )
        if len(terminal_event_id) > EVENT_ID_MAX_CHARS:
            raise ValueError("Queued dispatch terminal receipt event identity is too long.")
        run_epoch = receipt.get("run_epoch")
        if type(run_epoch) is not int or run_epoch < 1 or run_epoch > MAX_DURABLE_JSON_INTEGER:
            raise ValueError("Queued dispatch terminal receipt run_epoch is invalid.")
        receipts[operation_id] = _QueuedDispatchTerminalReceiptIdentity(
            queue_task_id=queue_task_id,
            terminal_event_id=terminal_event_id,
            run_epoch=run_epoch,
        )
    return receipts


def _queued_dispatch_terminal_receipts_for_session(
    session_id: str,
    checkpoint: dict[str, Any] | None,
    *,
    durable_event_ids: set[str],
) -> list[QueuedDispatchTerminalReceipt]:
    """Project all exact live queue/session handoffs from one checkpoint."""

    session_id = require_clean_nonblank(session_id, "session_id")
    projected: dict[str, QueuedDispatchTerminalReceipt] = {}
    run_operation = terminal_event_evidence._session_run_operation_from_checkpoint(checkpoint)
    if (
        run_operation is not None
        and run_operation.queue_task_id is not None
        and run_operation.terminal_event_id is not None
        and run_operation.terminal_event_id in durable_event_ids
    ):
        projected[run_operation.operation_id] = QueuedDispatchTerminalReceipt(
            session_id=session_id,
            queue_task_id=run_operation.queue_task_id,
            operation_id=run_operation.operation_id,
            terminal_event_id=run_operation.terminal_event_id,
        )
    for operation_id, receipt in _queued_dispatch_terminal_receipts_from_checkpoint(
        checkpoint
    ).items():
        candidate = QueuedDispatchTerminalReceipt(
            session_id=session_id,
            queue_task_id=receipt.queue_task_id,
            operation_id=operation_id,
            terminal_event_id=receipt.terminal_event_id,
        )
        existing = projected.get(operation_id)
        if existing is not None and existing != candidate:
            raise RuntimeError("Queued dispatch terminal handoff identity conflicts.")
        projected[operation_id] = candidate
    return list(projected.values())


def _checkpoint_after_session_run_operation_cleanup(
    checkpoint: dict[str, Any] | None,
    *,
    operation: terminal_event_evidence._SessionRunOperation,
    retain_terminal_receipt: bool,
) -> dict[str, Any]:
    """Clear run ownership and optionally transfer terminal retention to the queue."""

    updated = {} if checkpoint is None else copy_durable_json_object(checkpoint, "checkpoint")
    updated.pop(terminal_event_evidence._SESSION_RUN_OPERATION_CHECKPOINT_KEY, None)
    if not retain_terminal_receipt or operation.terminal_event_id is None:
        return updated
    receipts = _queued_dispatch_terminal_receipts_from_checkpoint(checkpoint)
    if operation.queue_task_id is None:
        return updated
    receipt = _QueuedDispatchTerminalReceiptIdentity(
        queue_task_id=operation.queue_task_id,
        terminal_event_id=operation.terminal_event_id,
        run_epoch=operation.run_epoch,
    )
    existing = receipts.get(operation.operation_id)
    if existing not in {None, receipt}:
        raise RuntimeError("Queued dispatch terminal receipt identity conflicts.")
    receipts[operation.operation_id] = receipt
    updated[_QUEUED_DISPATCH_TERMINAL_RECEIPTS_CHECKPOINT_KEY] = {
        "version": 1,
        "receipts": {
            operation_id: {
                "queue_task_id": stored_receipt.queue_task_id,
                "terminal_event_id": stored_receipt.terminal_event_id,
                "run_epoch": stored_receipt.run_epoch,
            }
            for operation_id, stored_receipt in receipts.items()
        },
    }
    return updated


def _checkpoint_after_queued_dispatch_acknowledgement(
    checkpoint: dict[str, Any] | None,
    *,
    queue_task_id: str,
    operation_id: str,
    terminal_event_id: str,
) -> dict[str, Any] | None:
    """Release exact queue-owned terminal retention after task terminalization."""

    queue_task_id = require_clean_nonblank(queue_task_id, "dispatch queue_task_id")
    operation_id = require_clean_nonblank(operation_id, "dispatch operation_id")
    terminal_event_id = require_clean_nonblank(
        terminal_event_id,
        "dispatch terminal_event_id",
    )
    current_operation = terminal_event_evidence._session_run_operation_from_checkpoint(checkpoint)
    receipts = _queued_dispatch_terminal_receipts_from_checkpoint(checkpoint)
    receipt = receipts.get(operation_id)
    if receipt is not None and (
        receipt.queue_task_id != queue_task_id or receipt.terminal_event_id != terminal_event_id
    ):
        raise RuntimeError("Queued dispatch terminal receipt identity conflicts.")
    if current_operation is not None and current_operation.operation_id == operation_id:
        if (
            current_operation.queue_task_id != queue_task_id
            or current_operation.terminal_event_id != terminal_event_id
        ):
            raise RuntimeError("Queued dispatch run operation identity conflicts.")
    elif receipt is None:
        return None

    updated = {} if checkpoint is None else copy_durable_json_object(checkpoint, "checkpoint")
    if current_operation is not None and current_operation.operation_id == operation_id:
        updated.pop(terminal_event_evidence._SESSION_RUN_OPERATION_CHECKPOINT_KEY, None)
    if receipt is not None:
        receipts.pop(operation_id)
        if receipts:
            updated[_QUEUED_DISPATCH_TERMINAL_RECEIPTS_CHECKPOINT_KEY] = {
                "version": 1,
                "receipts": {
                    stored_operation_id: {
                        "queue_task_id": stored_receipt.queue_task_id,
                        "terminal_event_id": stored_receipt.terminal_event_id,
                        "run_epoch": stored_receipt.run_epoch,
                    }
                    for stored_operation_id, stored_receipt in receipts.items()
                },
            }
        else:
            updated.pop(_QUEUED_DISPATCH_TERMINAL_RECEIPTS_CHECKPOINT_KEY, None)
    # ``transform_checkpoint`` reserves ``None`` for a read-only callback, so an
    # empty object is the required mutation that clears the final receipt.
    return updated


def _checkpoint_with_session_run_operation(
    *,
    checkpoint: dict[str, Any] | None,
    current_session: Session,
    operation_id: str,
    terminal_event_id: str | None = None,
    queue_task_id: str | None = None,
) -> dict[str, Any]:
    """Record the next running epoch in the same transaction as its status claim."""
    operation_id = require_clean_nonblank(operation_id, "session run operation_id")
    if terminal_event_id is not None:
        terminal_event_id = require_clean_nonblank(
            terminal_event_id,
            "session run terminal_event_id",
        )
        if len(terminal_event_id) > EVENT_ID_MAX_CHARS:
            raise ValueError("Session run terminal_event_id is too long.")
    if queue_task_id is not None:
        queue_task_id = require_clean_nonblank(
            queue_task_id,
            "session run queue_task_id",
        )
        if terminal_event_id is None:
            raise ValueError("Session run queue_task_id requires a terminal_event_id.")
    existing_operation = terminal_event_evidence._session_run_operation_from_checkpoint(checkpoint)
    if existing_operation is not None:
        raise RuntimeError(
            "Session has incomplete terminal evidence for its previous run. "
            "Recover the session before starting another run."
        )
    if queue_task_id is not None:
        receipt = _queued_dispatch_terminal_receipts_from_checkpoint(checkpoint).get(operation_id)
        if receipt is not None:
            if (
                receipt.queue_task_id != queue_task_id
                or receipt.terminal_event_id != terminal_event_id
            ):
                raise RuntimeError("Queued dispatch terminal receipt identity conflicts.")
            raise SessionRunFenced(
                "Queued dispatch terminal evidence is awaiting queue acknowledgement."
            )
    next_run_epoch = current_session.run_epoch + 1
    if next_run_epoch > MAX_DURABLE_JSON_INTEGER:
        raise ValueError("Session run operation run_epoch exceeds the durable integer limit.")
    updated = {} if checkpoint is None else copy_durable_json_object(checkpoint, "checkpoint")
    marker: dict[str, Any] = {
        "version": 1,
        "operation_id": operation_id,
        "run_epoch": next_run_epoch,
    }
    if terminal_event_id is not None:
        marker["terminal_event_id"] = terminal_event_id
    if queue_task_id is not None:
        marker["queue_task_id"] = queue_task_id
    updated[terminal_event_evidence._SESSION_RUN_OPERATION_CHECKPOINT_KEY] = marker
    return updated


def _event_with_session_run_operation(
    event: Event,
    operation: terminal_event_evidence._SessionRunOperation,
) -> Event:
    """Bind terminal evidence to the durable run operation that produced it."""
    payload = copy_durable_json_object(event.payload, "event payload")
    existing_operation_id = payload.get(_SESSION_RUN_OPERATION_ID_PAYLOAD_KEY)
    if existing_operation_id not in {None, operation.operation_id}:
        raise ValueError("Terminal event carries a conflicting session run operation identity.")
    payload[_SESSION_RUN_OPERATION_ID_PAYLOAD_KEY] = operation.operation_id
    updates: dict[str, Any] = {"payload": payload}
    if operation.terminal_event_id is not None:
        updates["id"] = operation.terminal_event_id
    bound = event.model_copy(update=updates, deep=True)
    if operation.terminal_event_id is not None:
        bound = event_with_runtime_generated_id(bound)
    return event_with_runtime_payload_authority(
        bound,
        _SESSION_RUN_OPERATION_ID_PAYLOAD_KEY,
    )


def _active_unexpired_session_operation_id(
    checkpoint: dict[str, Any] | None,
    *,
    now: datetime,
) -> str | None:
    """Return the active durable operation while its claim lease is valid."""

    if checkpoint is None:
        return None
    stored = checkpoint.get("session_operations")
    if stored is None:
        return None
    if type(stored) is not dict:
        raise ValueError("Session operation checkpoint must be an object.")
    active_operation_id = stored.get("active_operation_id")
    if active_operation_id is None:
        return None
    active_operation_id = require_clean_nonblank(
        active_operation_id,
        "active_operation_id",
    )
    records = stored.get("records")
    if type(records) is not dict:
        raise ValueError("Session operation checkpoint records must be an object.")
    active_record = next(
        (
            record
            for record in records.values()
            if type(record) is dict and record.get("operation_id") == active_operation_id
        ),
        None,
    )
    if active_record is None:
        raise ValueError("Active durable session operation record is missing.")
    claim_expires_at = active_record.get("claim_expires_at")
    if type(claim_expires_at) is not str:
        return active_operation_id
    try:
        expiry = datetime.fromisoformat(claim_expires_at)
    except ValueError:
        return active_operation_id
    if expiry.tzinfo is None or expiry.utcoffset() is None:
        return active_operation_id
    return active_operation_id if expiry.astimezone(UTC) > now.astimezone(UTC) else None


def _event_record_matches_session(
    record: EventRecord,
    query: EventQuery,
    sessions: dict[str, Session],
) -> bool:
    from cayu.sessions._access_rules import _query_bounds

    bounds = _query_bounds.get()
    if bounds is not None:
        owner = sessions.get(record.event.session_id)
        if owner is None or not bounds.matches(owner.labels):
            return False
    if query.causal_budget_id is None:
        return True
    session = sessions.get(record.event.session_id)
    return session is not None and session.causal_budget_id == query.causal_budget_id
