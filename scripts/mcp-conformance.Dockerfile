FROM node:22-bookworm-slim AS conformance
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates python3 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/conformance
COPY scripts/mcp_conformance_build.py /tmp/mcp_conformance_build.py
RUN git init && git remote add origin https://github.com/modelcontextprotocol/conformance.git \
    && git fetch --depth 1 origin 7169291ec0b68eb370fddcd9947313ab0d5e4156 \
    && git checkout --detach FETCH_HEAD \
    && python3 /tmp/mcp_conformance_build.py --upstream /opt/conformance

FROM python:3.14-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
COPY --from=conformance /usr/local/bin/node /usr/local/bin/node
COPY --from=conformance /opt/conformance /opt/conformance
RUN pip install --no-cache-dir uv==0.10.0
WORKDIR /repo
ENV UV_PROJECT_ENVIRONMENT=/opt/venv PYTHONPATH=/repo/src PYTHONDONTWRITEBYTECODE=1
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --extra dev --extra server --no-install-project --no-cache
COPY src ./src
COPY tests ./tests
COPY scripts/mcp_conformance_client.py scripts/run_mcp_conformance.py scripts/mcp_conformance_build.py ./scripts/
ENTRYPOINT ["/opt/venv/bin/python", "scripts/run_mcp_conformance.py", "--upstream", "/opt/conformance"]
CMD ["--output", "/results/run"]
