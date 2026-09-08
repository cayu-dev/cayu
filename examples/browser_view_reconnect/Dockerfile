FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends docker-cli ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/cayu
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY examples/aws/lambda_microvm_sidecar ./examples/aws/lambda_microvm_sidecar
RUN python -m pip install --no-cache-dir '.[server,browser]'
CMD ["sleep", "infinity"]
