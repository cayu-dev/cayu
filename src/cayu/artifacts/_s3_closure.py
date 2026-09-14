"""Conditional S3 state for publication reservations and irreversible closure."""

from __future__ import annotations

import json
from hashlib import sha256

from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
    canonical_durable_json_bytes,
    durable_json_object_from_pairs,
    require_durable_clean_nonblank,
)
from cayu.artifacts._closure import (
    ArtifactClosureClaim,
    copy_artifact_closure_claim,
    decode_artifact_closure_claim,
    encode_artifact_closure_claim,
)

_STATE_MAX_BYTES = 32 * 1024 * 1024
_MAX_ACTIVE = 1000


class S3ArtifactClosureGate:
    """No expiry: an uncertain writer must not lose its reservation to a clock."""

    def __init__(self, client, *, bucket, prefix, store_id, session_id, encryption):
        # The publication gate accepts ordinary artifact owners; the narrower
        # administrative claim bounds are enforced when constructing a claim.
        ArtifactClosureClaim(store_id, "validation", "0" * 64, ())
        require_durable_clean_nonblank(session_id, "artifact session identity")
        self.client = client
        self.bucket = bucket
        self.store_id = store_id
        self.session_id = session_id
        self.key = (
            (prefix + "/" if prefix else "")
            + "_closure/"
            + sha256(session_id.encode()).hexdigest()
            + ".json"
        )
        self.encryption = dict(encryption)

    def _empty(self):
        return {
            "schema_version": 1,
            "store_id": self.store_id,
            "session_id": self.session_id,
            "revision": 0,
            "active": {},
            "settled": {},
            "claim": None,
        }

    def _validate(self, state):
        if (
            type(state) is not dict
            or set(state) != set(self._empty())
            or type(state["schema_version"]) is not int
            or state["schema_version"] != 1
            or state["store_id"] != self.store_id
            or state["session_id"] != self.session_id
            or type(state["revision"]) is not int
            or not 0 <= state["revision"] <= MAX_DURABLE_JSON_INTEGER
            or type(state["active"]) is not dict
            or type(state["settled"]) is not dict
            or len(state["active"]) + len(state["settled"]) > _MAX_ACTIVE
            or state["active"].keys() & state["settled"].keys()
        ):
            raise ValueError("Invalid S3 artifact closure state.")
        for token, intent in state["active"].items():
            self._validate_intent(token, intent)
        for token, intent in state["settled"].items():
            self._validate_intent(token, intent)
        if state["claim"] is not None:
            if state["active"] or state["settled"]:
                raise ValueError("S3 artifact closure conflicts with active writes.")
            claim = decode_artifact_closure_claim(
                canonical_durable_json_bytes(
                    state["claim"], "S3 artifact closure claim", max_bytes=_STATE_MAX_BYTES
                )
            )
            if claim.store_id != self.store_id or claim.session_id != self.session_id:
                raise ValueError("S3 artifact closure authority conflicts.")
        return state

    @staticmethod
    def _validate_intent(token, intent):
        if (
            type(token) is not str
            or len(token) != 32
            or any(c not in "0123456789abcdef" for c in token)
        ):
            raise ValueError("Invalid S3 artifact write reservation.")
        if type(intent) is not dict or set(intent) != {"artifact_id", "request_sha256"}:
            raise ValueError("Invalid S3 artifact write intent.")
        artifact_id, digest = intent["artifact_id"], intent["request_sha256"]
        if (
            type(artifact_id) is not str
            or len(artifact_id) != 36
            or not artifact_id.startswith("art_")
            or any(c not in "0123456789abcdef" for c in artifact_id[4:])
        ):
            raise ValueError("Invalid S3 artifact reservation identity.")
        if (
            type(digest) is not str
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ValueError("Invalid S3 artifact reservation digest.")

    def read(self):
        from cayu.artifacts.aws_s3 import _aws_error_code

        try:
            response = self.client.get_object(Bucket=self.bucket, Key=self.key)
        except Exception as error:
            if _aws_error_code(error) in {"404", "NoSuchKey", "NotFound"}:
                return self._empty(), None
            raise
        body = response["Body"]
        failure = None
        try:
            length = response.get("ContentLength")
            if type(length) is not int or not 0 <= length <= _STATE_MAX_BYTES:
                raise ValueError("S3 artifact closure state exceeds its byte bound.")
            data = body.read(_STATE_MAX_BYTES + 1)
            if type(data) is not bytes or len(data) != length:
                raise ValueError("S3 artifact closure read is incomplete.")
            etag = response.get("ETag")
            if (
                type(etag) is not str
                or not 0 < len(etag) <= 256
                or any(not 32 <= ord(c) < 127 for c in etag)
            ):
                raise ValueError("S3 artifact closure has no conditional-write authority.")
            state = json.loads(
                data,
                object_pairs_hook=lambda pairs: durable_json_object_from_pairs(
                    pairs, "S3 artifact closure"
                ),
            )
            self._validate(state)
        except BaseException as error:
            failure = error
        try:
            body.close()
        except BaseException as cleanup_error:
            if failure is not None and cleanup_error is not failure:
                raise BaseExceptionGroup(
                    "S3 artifact closure read and cleanup failed.", [failure, cleanup_error]
                ) from None
            raise
        if failure is not None:
            raise failure
        return state, etag

    def _transition(self, mutate):
        from cayu.artifacts.aws_s3 import _aws_error_code

        for _ in range(16):
            state, etag = self.read()
            if not mutate(state):
                return
            if state["revision"] == MAX_DURABLE_JSON_INTEGER:
                raise ValueError("S3 artifact closure revision is exhausted.")
            state["revision"] += 1
            self._validate(state)
            encoded = canonical_durable_json_bytes(
                state, "S3 artifact closure", max_bytes=_STATE_MAX_BYTES
            )
            try:
                self.client.put_object(
                    Bucket=self.bucket,
                    Key=self.key,
                    Body=encoded,
                    ContentType="application/json",
                    **({"IfNoneMatch": "*"} if etag is None else {"IfMatch": etag}),
                    **self.encryption,
                )
                return
            except Exception as error:
                if _aws_error_code(error) in {
                    "PreconditionFailed",
                    "ConditionalRequestConflict",
                    "409",
                    "412",
                }:
                    continue
                # A lost acknowledgement is accepted only after exact readback.
                try:
                    observed, _ = self.read()
                except Exception as read_error:
                    raise ExceptionGroup(
                        "S3 artifact closure publication could not be reconciled.",
                        [error, read_error],
                    ) from None
                if observed == state:
                    return
                raise
        raise ValueError("S3 artifact closure contention exceeded its retry bound.")

    def reserve(self, token, artifact_id, request_sha256):
        intent = {"artifact_id": artifact_id, "request_sha256": request_sha256}
        self._validate_intent(token, intent)

        def mutate(state):
            if state["claim"] is not None:
                raise ValueError("Artifact session is fenced for closure.")
            if token in state["settled"]:
                raise ValueError("S3 artifact write reservation already settled.")
            if token in state["active"]:
                if state["active"][token] != intent:
                    raise ValueError("S3 artifact write reservation conflicts.")
                return False
            # Completed settlements no longer own external work. Retire them
            # atomically with fresh admission, never expire active reservations.
            state["settled"].clear()
            if len(state["active"]) >= _MAX_ACTIVE:
                raise ValueError("S3 artifact write reservations exceed their bound.")
            state["active"][token] = intent
            return True

        self._transition(mutate)

    def release(self, token, artifact_id, request_sha256):
        """Publish positive quiescence before removing its durable retry owner.

        Only the publishing owner calls this after its business calls settle;
        readers must never infer this proof from object presence or elapsed time.
        """
        intent = {"artifact_id": artifact_id, "request_sha256": request_sha256}
        self._validate_intent(token, intent)

        def mutate(state):
            if token in state["settled"]:
                if state["settled"][token] != intent:
                    raise ValueError("S3 artifact write settlement conflicts.")
                return False
            if token not in state["active"]:
                return False
            if state["active"][token] != intent:
                raise ValueError("S3 artifact write settlement conflicts.")
            del state["active"][token]
            state["settled"][token] = intent
            return True

        self._transition(mutate)
        self.retire_settled()

    def retire_settled(self):
        """Recover only durable owner-produced quiescence evidence after restart."""

        def mutate(state):
            if not state["settled"]:
                return False
            state["settled"].clear()
            return True

        self._transition(mutate)

    def seal(self, claim, *, expected_revision):
        claim = copy_artifact_closure_claim(claim)
        if (
            type(expected_revision) is not int
            or not 0 <= expected_revision <= MAX_DURABLE_JSON_INTEGER
        ):
            raise ValueError("Invalid S3 artifact closure inventory revision.")
        if claim.store_id != self.store_id or claim.session_id != self.session_id:
            raise ValueError("S3 artifact closure authority conflicts.")
        encoded_claim = json.loads(encode_artifact_closure_claim(claim))

        def mutate(state):
            if state["claim"] is not None:
                if state["claim"] != encoded_claim:
                    raise ValueError("S3 artifact closure claim conflicts.")
                return False
            if state["active"] or state["settled"]:
                raise ValueError("S3 artifact publication has not quiesced.")
            if state["revision"] != expected_revision:
                raise ValueError("S3 artifact inventory changed before closure.")
            state["claim"] = encoded_claim
            return True

        self._transition(mutate)
