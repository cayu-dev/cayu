"""Emitted deployment contract; not native Compose or Docker qualification."""

import json
import tomllib

import pytest

from tests.qualification.repository_maintenance_application import maintenance_project_files


@pytest.fixture(scope="module")
def files():
    return maintenance_project_files(database="postgres")


def test_emission_roles_match_registered_owners(files):
    services = json.loads(files["compose.yaml"])["services"]
    workers = tomllib.loads(files["pyproject.toml"])["tool"]["cayu"]["workers"]
    assert set(services) == {"api", "postgres", *workers}
    for role in workers:
        assert services[role]["command"] == [
            "cayu",
            "worker",
            role,
            "--shutdown-grace-seconds",
            "30",
        ]
    api = services["api"]
    assert api["command"] == [
        "python",
        "-m",
        "uvicorn",
        "operations.maintenance_api:build_api",
        "--factory",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--timeout-graceful-shutdown",
        "30",
    ]
    assert api["ports"] == ["127.0.0.1:8000:8000"]
    assert "/internal/cayu/api/health" in api["healthcheck"]["test"][-1]
    assert all("migrate" not in service.get("command", []) for service in services.values())


@pytest.mark.parametrize(
    "role", ["api", "coding", "git_preparation", "git_delivery", "github_delivery"]
)
def test_host_roles_separate_credentials_and_persist_owners(files, role):
    services = json.loads(files["compose.yaml"])["services"]
    service = services[role]
    assert service["image"] == services["api"]["image"]
    assert ":?" in service["image"]
    assert service["pull_policy"] == "never" and service["restart"] == "no"
    assert service["user"] == "10001:10001" and service["init"] is True
    assert service["read_only"] is True and "privileged" not in service
    assert service["cap_drop"] == ["ALL"]
    assert service["security_opt"] == ["no-new-privileges:true"]
    assert service["stop_grace_period"] == "45s"
    assert service["depends_on"] == {"postgres": {"condition": "service_healthy"}}
    assert len(service["group_add"]) == 1 and ":?" in service["group_add"][0]
    expected = ["common"]
    if role == "api":
        expected.append("api")
    if role in {"git_preparation", "git_delivery"}:
        expected.append("git")
    if role == "github_delivery":
        expected.append("github")
    assert service["env_file"] == [
        {"path": f"./deployment/{name}.env", "format": "raw", "required": True} for name in expected
    ]
    mounts = {mount["target"]: mount for mount in service["volumes"]}
    assert set(mounts) == {
        "/repository",
        "/app/.cayu/runtime",
        "/var/run/docker.sock",
        *(["/delivery"] if "git" in expected else []),
    }
    assert mounts["/repository"]["read_only"] is (role != "coding")
    for target, mount in mounts.items():
        assert mount["type"] == "bind" and mount["bind"] == {"create_host_path": False}
        assert mount["source"] == target or ":?" in mount["source"]
    assert service["environment"]["CAYU_WORKSPACE_ROOT"] == "/repository"


def test_database_is_private_persistent_and_explicit(files):
    document = json.loads(files["compose.yaml"])
    db = document["services"]["postgres"]
    assert "ports" not in db and "command" not in db
    assert db["volumes"] == ["postgres-data:/var/lib/postgresql/data"]
    assert document["volumes"] == {"postgres-data": {}}
    assert db["restart"] == "no" and db["pull_policy"] == "never"
    assert "pg_isready" in db["healthcheck"]["test"][-1]
    assert "$$POSTGRES_USER" in db["healthcheck"]["test"][-1]


def test_image_is_offline_and_copies_only_emitted_application(files):
    dockerfile = files["Dockerfile.application"]
    assert "--no-index --find-links=/wheels --require-hashes" in dockerfile
    assert "FROM ${CAYU_HOST_TOOLS_IMAGE}" in dockerfile
    for line in dockerfile.splitlines():
        if not line.startswith("COPY "):
            continue
        sources = line.split()[1:-1]
        for source in sources:
            assert source not in {".", "./"}
            assert source in {"deployment/wheels/", "deployment/requirements.lock"} or (
                source in files or any(name.startswith(source) for name in files)
            )
            assert ".env" not in source and ".cayu" not in source
    ignore = files["Dockerfile.application.dockerignore"]
    assert ignore.startswith("**\n")
    assert "!deployment/\ndeployment/**\n" in ignore
    assert ignore.index("deployment/**") < ignore.index("!deployment/requirements.lock")
    for rule in ("deployment/*.env", "deployment/backups/**", "**/.env*", "**/*.key"):
        assert rule in ignore.splitlines()
    assert files["Dockerfile.host-tools.dockerignore"] == "**\n"
    assert "deployment/*.env" in files[".gitignore"]
    assert "not periodic database readiness" in files["deployment/README.md"]


def test_sqlite_emission_does_not_advertise_postgres_deployment():
    files = maintenance_project_files(database="sqlite")
    assert "compose.yaml" not in files and "Dockerfile.application" not in files
