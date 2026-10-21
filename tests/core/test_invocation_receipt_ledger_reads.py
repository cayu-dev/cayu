"""Work-count and isolation guards for invocation receipt ledger reads."""

from __future__ import annotations

import asyncio
from copy import deepcopy

import pytest

import cayu.sessions._invocation_lifecycle as lifecycle
from cayu import AgentSpec, CayuApp, Message, ResumeRequest, RunRequest
from cayu.evals import ScriptedModelProvider
from cayu.providers import ModelStreamEvent
from cayu.sessions import InMemorySessionStore
from cayu.sessions.checkpoints import INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY
from cayu.storage.sqlite import SQLiteSessionStore


def _reply(request):
    del request
    return [ModelStreamEvent.text_delta("ok"), ModelStreamEvent.completed()]


async def _converse(store, turns: int, *, after_turn=None) -> dict:
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(ScriptedModelProvider(response_factory=_reply), default=True)
    app.register_agent(AgentSpec(name="assistant", model="scripted-model"))
    try:
        for turn in range(turns):
            messages = [Message.text("user", f"turn {turn}")]
            stream = (
                app.run(RunRequest(agent_name="assistant", session_id="s", messages=messages))
                if turn == 0
                else app.resume(ResumeRequest(session_id="s", messages=messages))
            )
            async for _ in stream:
                pass
            if after_turn is not None:
                after_turn(turn)
        checkpoint = await store.load_checkpoint("s")
    finally:
        if isinstance(store, SQLiteSessionStore):
            await store.close()
    assert checkpoint is not None
    return checkpoint


def _store(kind: str, tmp_path):
    return (
        InMemorySessionStore() if kind == "memory" else SQLiteSessionStore(tmp_path / "s.sqlite3")
    )


@pytest.mark.parametrize("kind", ["memory", "sqlite"])
def test_resume_validates_only_the_ledgers_it_writes(monkeypatch, tmp_path, kind):
    validated: list[int] = []
    per_turn: list[list[int]] = []
    validate = lifecycle._validate_receipt_ledger

    def count(raw):
        validated.append(len(raw["receipts"]))
        return validate(raw)

    def close_turn(turn: int) -> None:
        del turn
        per_turn.append(list(validated))
        validated.clear()

    monkeypatch.setattr(lifecycle, "_validate_receipt_ledger", count)
    asyncio.run(_converse(_store(kind, tmp_path), 10, after_turn=close_turn))

    # Each resumed turn adds an admission and a release receipt. Every other
    # read during the turn (dispatch budget checks, replay checks, secret
    # validation) sees one of those two ledgers and must not re-validate it,
    # so the work per turn stays two full validations as the history grows.
    resumed = per_turn[1:]
    assert all(len(turn) <= 2 for turn in resumed), per_turn
    assert resumed[-1][-1] > resumed[0][-1]


def test_cached_ledger_reads_are_private_copies(tmp_path):
    checkpoint = asyncio.run(_converse(InMemorySessionStore(), 3))
    ledger = lifecycle._invocation_lifecycle_receipt_ledger_from_checkpoint(checkpoint)
    identity = ledger.receipts[-1].command_identity
    receipt = lifecycle._invocation_lifecycle_receipt_from_checkpoint(
        checkpoint, command_identity=identity
    )
    assert receipt is not None
    ledger.receipts[-1].result_session.metadata["probe"] = "ledger"
    receipt.result_session.metadata["probe"] = "receipt"

    again = lifecycle._invocation_lifecycle_receipt_ledger_from_checkpoint(checkpoint)
    single = lifecycle._invocation_lifecycle_receipt_from_checkpoint(
        checkpoint, command_identity=identity
    )
    assert single is not None
    assert "probe" not in again.receipts[-1].result_session.metadata
    assert "probe" not in single.result_session.metadata
    assert again.model_dump(mode="json") == checkpoint[INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY]


def test_cache_key_distinguishes_values_that_share_a_canonical_encoding(monkeypatch):
    checkpoint = asyncio.run(_converse(InMemorySessionStore(), 2))
    lifecycle._invocation_lifecycle_receipt_ledger_from_checkpoint(checkpoint)
    validated = []
    validate = lifecycle._validate_receipt_ledger

    def count(raw):
        validated.append(raw)
        return validate(raw)

    monkeypatch.setattr(lifecycle, "_validate_receipt_ledger", count)
    lifecycle._invocation_lifecycle_receipt_ledger_from_checkpoint(deepcopy(checkpoint))
    assert validated == []

    # The digest-bearing canonical encoding spells 2.0 as 2, but run_epoch is a
    # strict integer. The float spelling must be validated, and rejected.
    respelled = deepcopy(checkpoint)
    receipt = respelled[INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY]["receipts"][0]
    receipt["active_profile"]["run_epoch"] = float(receipt["active_profile"]["run_epoch"])
    with pytest.raises(RuntimeError, match="malformed"):
        lifecycle._invocation_lifecycle_receipt_ledger_from_checkpoint(respelled)
    assert len(validated) == 1


def test_cached_ledger_does_not_admit_a_tampered_or_oversized_copy(monkeypatch):
    checkpoint = asyncio.run(_converse(InMemorySessionStore(), 2))
    lifecycle._invocation_lifecycle_receipt_ledger_from_checkpoint(checkpoint)

    tampered = deepcopy(checkpoint)
    tampered[INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY]["receipts"][0]["session_id"] = "other"
    with pytest.raises(RuntimeError, match="malformed"):
        lifecycle._invocation_lifecycle_receipt_ledger_from_checkpoint(tampered)

    # Bounds are enforced before the cache is consulted, so a ledger that once
    # validated is rejected when the limit no longer admits it.
    monkeypatch.setattr(lifecycle, "INVOCATION_LIFECYCLE_RECEIPT_LEDGER_MAX_BYTES", 1024)
    with pytest.raises(RuntimeError, match="malformed"):
        lifecycle._invocation_lifecycle_receipt_ledger_from_checkpoint(checkpoint)
