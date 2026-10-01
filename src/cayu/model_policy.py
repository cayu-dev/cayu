"""Explicit application-default policy integration, independent of inference auth.

Construct one controller per local agent. Supply a management channel only when
automatic adoption/reporting is desired. Start/stop through CayuApp lifecycle
methods; the generated entrypoint and server own that lifecycle automatically.
"""

import os
from pathlib import Path

from cayu.runtime._model_policy import ModelPolicy, ModelPolicyController, PolicyChannel
from cayu.runtime._policy_http import (
    HttpPolicyChannel,
    PolicyResponseError,
    PolicyTransportUnavailable,
)
from cayu.runtime._policy_storage import InMemoryModelPolicyStore, ModelPolicyStore
from cayu.runtime._policy_wire import PolicyContractError, decode, require


def model_policy_enabled() -> bool:
    """Only an explicit configuration path opts the application in."""
    return bool(os.environ.get("CAYU_MODEL_POLICY_CONFIG"))


def configured_model_policy(store: ModelPolicyStore | None) -> ModelPolicy | None:
    """Load a bounded deployment-owned configuration; never infer from API keys."""
    path = os.environ.get("CAYU_MODEL_POLICY_CONFIG")
    if not path:
        return None
    require(store is not None)
    assert store is not None
    with Path(path).open("rb") as stream:
        config = decode(stream.read(65537))
    require(set(config) == {"bindings"} and type(config["bindings"]) is list)
    require(1 <= len(config["bindings"]) <= 64)
    controllers = []
    for binding in config["bindings"]:
        require(
            type(binding) is dict
            and set(binding)
            == {
                "agent_name",
                "provider_name",
                "origin",
                "scope",
                "incarnation_id",
                "incarnation_epoch",
                "credential_env",
            }
        )
        channel = HttpPolicyChannel(
            origin=binding["origin"],
            scope=binding["scope"],
            incarnation=(binding["incarnation_id"], binding["incarnation_epoch"]),
            credential_env=binding["credential_env"],
        )
        controllers.append(
            ModelPolicyController(
                store=store,
                agent_name=binding["agent_name"],
                provider_name=binding["provider_name"],
                scope=channel.scope,
                incarnation=channel.incarnation,
                channel=channel,
            )
        )
    return ModelPolicy(controllers, close_channels=True)


__all__ = [
    "HttpPolicyChannel",
    "InMemoryModelPolicyStore",
    "ModelPolicy",
    "ModelPolicyController",
    "ModelPolicyStore",
    "PolicyChannel",
    "PolicyContractError",
    "PolicyResponseError",
    "PolicyTransportUnavailable",
    "configured_model_policy",
    "model_policy_enabled",
]
