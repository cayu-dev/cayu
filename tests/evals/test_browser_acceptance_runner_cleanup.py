"""Campaign ownership crosses the real adapter and DockerRunner allocation paths."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from examples.browser_acceptance import local_authenticated as example
from tests.egress.test_docker_adapter import _credentialless_broker

from cayu.egress.adapter import VirtualEgressRunnerRequest
from cayu.egress.docker_adapter import GUEST_CA_PATH


@pytest.mark.parametrize("lost_ack", [False, True])
@pytest.mark.parametrize("remove_fails", [False, True])
def test_real_runner_allocation_is_owned_through_host_cleanup(
    tmp_path, monkeypatch, lost_ack, remove_fails
):
    async def scenario():
        resources = {}
        calls = []
        runner_id = "a" * 64

        async def egress(argv):
            if argv[:2] == ["network", "create"]:
                resources[argv[-1]] = "network"
            elif argv[:1] == ["run"]:
                resources[argv[argv.index("--name") + 1]] = "sidecar"
            return 0, ""

        async def runner_transport(docker, argv, **kwargs):
            if argv[0] == "run":
                records = [
                    json.loads(line)
                    for line in (tmp_path / "resource-intents.jsonl").read_text().splitlines()
                ]
                assert records[-1]["kind"] == "runner" and "created" not in records[-1]
                resources["campaign-runner"] = "runner"
                if lost_ack:
                    raise RuntimeError("creation acknowledgement lost")
            return SimpleNamespace(exit_code=0, stdout=runner_id, stderr="", timed_out=False)

        monkeypatch.setattr("cayu.runners.docker._run_docker", runner_transport)
        monkeypatch.setattr("cayu.runners.docker.shutil.which", lambda name: "/docker")
        adapter = example._campaign_docker_adapter(
            tmp_path,
            docker_exec=example._recording_docker_exec(tmp_path, egress),
            proxy_host="127.0.0.1",
        )
        binding = await adapter.prepare(
            session_id="campaign", grants=[], broker=_credentialless_broker()
        )
        ca = tmp_path / "ca.pem"
        ca.write_bytes(binding.ca_cert_pem)
        request = VirtualEgressRunnerRequest(
            name="campaign-runner",
            runner_kind="docker",
            image="test-image",
            binding=binding,
            env_overlay={},
            ca_cert_host_path=str(ca),
            guest_ca_path=GUEST_CA_PATH,
            setup_commands=(),
            egress_destinations=(),
            session_id="campaign",
        )

        def docker(*args):
            calls.append(args)
            if args[:2] == ("network", "ls"):
                return "\n".join(key for key, kind in resources.items() if kind == "network")
            if args[0] == "ps":
                if "{{.ID}} {{.Names}}" in args:
                    return runner_id + " campaign-runner" if "campaign-runner" in resources else ""
                if "{{.ID}}" in args:
                    return runner_id if "campaign-runner" in resources else ""
                return "\n".join(key for key, kind in resources.items() if kind == "sidecar")
            if args[:2] in (("rm", "-f"), ("network", "rm")):
                key = "campaign-runner" if args[-1] == runner_id else args[-1]
                if remove_fails and key in resources:
                    raise RuntimeError("removal unavailable")
                resources.pop(key, None)
            return ""

        try:
            if lost_ack:
                with pytest.raises(RuntimeError, match="acknowledgement lost"):
                    await adapter.create_runner(request)
            else:
                await adapter.create_runner(request)
            assert sorted(resources.values()) == ["network", "runner", "sidecar"]
            if lost_ack or remove_fails:
                with pytest.raises((RuntimeError, ExceptionGroup)):
                    example._retire_host(docker, "controller", tmp_path, None)
                inventory = json.loads((tmp_path / "cleanup-resources.json").read_text())
                assert "campaign-runner" in inventory["containers"]
                assert inventory["runner_ids"] == [runner_id]
                assert binding.network in inventory["networks"]
                assert binding.sidecar in inventory["containers"]
                if lost_ack:
                    assert inventory["unsettled_creations"][0]["name"] == "campaign-runner"
            else:
                example._retire_host(docker, "controller", tmp_path, None)
            assert calls.index(("rm", "-f", runner_id)) < calls.index(
                ("network", "rm", binding.network)
            )
            if not remove_fails:
                assert resources == {}
                if not lost_ack:
                    # Ordinary completed teardown is replayable with absent resources.
                    example._retire_host(docker, "controller", tmp_path, None)
        finally:
            await binding.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("network_matches", [False, True])
def test_runner_removal_requires_exact_network_and_full_identity(tmp_path, network_matches):
    record = {"kind": "runner", "name": "runner", "session": "campaign", "network": "owned-network"}
    (tmp_path / "resource-intents.jsonl").write_text(
        json.dumps(record) + "\n" + json.dumps({**record, "created": True})
    )
    calls = []

    def docker(*args):
        calls.append(args)
        if "{{.ID}} {{.Names}}" in args:
            return ("bad-id" if network_matches else "a" * 64) + " runner"
        if args[:2] == ("network", "ls"):
            return "owned-network"
        if "{{.ID}}" in args:
            return "bad-id" if network_matches else "b" * 64
        return ""

    with pytest.raises(RuntimeError, match="identity is unverified"):
        example._retire_host(docker, "controller", tmp_path, None)
    assert [args for args in calls if args[:2] == ("rm", "-f")] == [("rm", "-f", "controller")]
    retained = json.loads((tmp_path / "cleanup-resources.json").read_text())
    assert retained["containers"] == ["runner"]
