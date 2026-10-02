"""Immutable native cleanup identity shared by the protected index and readback."""

from typing import Literal

from pydantic import model_validator

from cayu.collaboration._contracts import (
    ContractValue,
    Generation,
    Identifier,
    ObjectRef,
    OperationRef,
)
from cayu.collaboration.prepared_admission import NativeCommitment


class NativeProducerCleanupReceipt(ContractValue):
    mode: Literal["invocation", "exclusion"] = "invocation"
    registration: OperationRef
    receiver: ObjectRef
    registration_commitment: NativeCommitment
    source_cleanup_commitment: NativeCommitment
    session_id: Identifier
    session_instance_id: Identifier
    interaction_id: Identifier | None
    run_epoch: Generation | None
    native_release_commitment: NativeCommitment

    @model_validator(mode="after")
    def exact_kind(self):
        if any(
            (value is not None) != (self.mode == "invocation")
            for value in (self.interaction_id, self.run_epoch)
        ):
            raise ValueError("Native cleanup cannot confuse exclusion with an invocation.")
        return self
