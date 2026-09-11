"""Static deployment assets for the generated PostgreSQL consumer, not a supervisor."""

import json
from pathlib import Path

_PACKAGES = (
    "agents",
    "configuration",
    "domain",
    "environments",
    "evals",
    "integrations",
    "knowledge",
    "memory",
    "observability",
    "operations",
    "policies",
    "prompts",
    "tests",
    "tools",
    "workflows",
)


def _bind(source, target, *, read_only=False):
    return {
        "type": "bind",
        "source": source,
        "target": target,
        "read_only": read_only,
        "bind": {"create_host_path": False},
    }


def _env(name):
    return {"path": f"./deployment/{name}.env", "format": "raw", "required": True}


def compose_assets() -> dict[str, str]:
    services: dict[str, dict[str, object]] = {}
    for role in ("api", "coding", "git_preparation", "git_delivery", "github_delivery"):
        mounts = [
            _bind(
                "${CAYU_MAINTENANCE_SOURCE:?Set the admitted absolute source path}",
                "/repository",
                read_only=role != "coding",
            ),
            _bind(
                "${CAYU_MAINTENANCE_STATE:?Set the persistent runtime directory}",
                "/app/.cayu/runtime",
            ),
            _bind("/var/run/docker.sock", "/var/run/docker.sock"),
        ]
        env_files = [_env("common")]
        if role == "api":
            env_files.append(_env("api"))
        if role in {"git_preparation", "git_delivery"}:
            env_files.append(_env("git"))
            mounts.append(
                _bind(
                    "${CAYU_MAINTENANCE_BROKER:?Set the persistent broker directory}", "/delivery"
                )
            )
        if role == "github_delivery":
            env_files.append(_env("github"))
        service: dict[str, object] = {
            "image": "${CAYU_MAINTENANCE_APP_IMAGE:?Set the reviewed immutable application image}",
            "pull_policy": "never",
            "init": True,
            "restart": "no",
            "user": "10001:10001",
            "group_add": ["${CAYU_MAINTENANCE_DOCKER_GID:?Set the Docker socket group ID}"],
            "working_dir": "/app",
            "read_only": True,
            "cap_drop": ["ALL"],
            "security_opt": ["no-new-privileges:true"],
            "tmpfs": ["/tmp:rw,nosuid,nodev,size=268435456,mode=1777"],
            "pids_limit": 256,
            "mem_limit": "1g",
            "stop_grace_period": "45s",
            "depends_on": {"postgres": {"condition": "service_healthy"}},
            "env_file": env_files,
            "environment": {
                "CAYU_WORKSPACE_ROOT": "/repository",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONUNBUFFERED": "1",
            },
            "volumes": mounts,
            "command": ["cayu", "worker", role, "--shutdown-grace-seconds", "30"],
        }
        if role == "api":
            service.update(
                command=[
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
                ],
                ports=["127.0.0.1:8000:8000"],
                healthcheck={
                    "test": [
                        "CMD",
                        "python",
                        "-c",
                        "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/internal/cayu/api/health', timeout=3).close()",
                    ],
                    "interval": "15s",
                    "timeout": "5s",
                    "retries": 3,
                    "start_period": "30s",
                },
            )
        services[role] = service
    services["postgres"] = {
        "image": "${CAYU_MAINTENANCE_POSTGRES_IMAGE:?Set the reviewed pgvector PostgreSQL 16 image digest}",
        "pull_policy": "never",
        "restart": "no",
        "env_file": [_env("postgres")],
        "volumes": ["postgres-data:/var/lib/postgresql/data"],
        "stop_grace_period": "60s",
        "healthcheck": {
            "test": ["CMD-SHELL", 'pg_isready -U "$$POSTGRES_USER" -d "$$POSTGRES_DB"'],
            "interval": "5s",
            "timeout": "5s",
            "retries": 12,
            "start_period": "30s",
        },
    }
    dockerfile = (
        """ARG CAYU_HOST_TOOLS_IMAGE
FROM ${CAYU_HOST_TOOLS_IMAGE}
USER root
WORKDIR /app
COPY deployment/wheels/ /wheels/
COPY deployment/requirements.lock /opt/maintenance-requirements.lock
RUN python -m pip install --no-index --find-links=/wheels --require-hashes -r /opt/maintenance-requirements.lock
COPY app.py pyproject.toml docker-coding-image.json ./
"""
        + "".join(f"COPY {name}/ ./{name}/\n" for name in _PACKAGES)
        + """RUN mkdir -p /app/.cayu/runtime /repository /delivery && chown -R 10001:10001 /app/.cayu /repository /delivery
USER 10001:10001
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
CMD ["python", "-m", "uvicorn", "operations.maintenance_api:build_api", "--factory", "--host", "0.0.0.0", "--port", "8000"]
"""
    )
    tools = """ARG PYTHON_IMAGE
ARG DOCKER_CLI_IMAGE
FROM ${DOCKER_CLI_IMAGE} AS docker_cli
FROM ${PYTHON_IMAGE}
ARG GIT_PACKAGE_VERSION
ARG RIPGREP_PACKAGE_VERSION
RUN test -n "$GIT_PACKAGE_VERSION" && test -n "$RIPGREP_PACKAGE_VERSION" && apt-get update && apt-get install -y --no-install-recommends "git=$GIT_PACKAGE_VERSION" "ripgrep=$RIPGREP_PACKAGE_VERSION" && rm -rf /var/lib/apt/lists/*
COPY --from=docker_cli /usr/local/bin/docker /usr/local/bin/docker
RUN groupadd --gid 10001 cayu && useradd --uid 10001 --gid 10001 --create-home cayu
RUN python --version && git --version && rg --version && docker --version
"""
    allowed = [
        "app.py",
        "pyproject.toml",
        "docker-coding-image.json",
        "deployment/requirements.lock",
        "deployment/wheels/",
        "deployment/wheels/**",
        *[entry for name in _PACKAGES for entry in (f"{name}/", f"{name}/**")],
    ]
    ignore = (
        "**\n!deployment/\ndeployment/**\n"
        + "".join(f"!{name}\n" for name in allowed)
        + """**/.env*
deployment/*.env
deployment/backups/**
**/*.key
**/*.sqlite*
**/__pycache__/
**/*.pyc
"""
    )
    return {
        "compose.yaml": json.dumps(
            {"services": services, "volumes": {"postgres-data": {}}}, indent=2
        )
        + "\n",
        "Dockerfile.application": dockerfile,
        "Dockerfile.host-tools": tools,
        "Dockerfile.host-tools.dockerignore": "**\n",
        "Dockerfile.application.dockerignore": ignore,
        "deployment/README.md": Path(__file__)
        .with_name("repository_maintenance_deployment.md")
        .read_text(),
    }
