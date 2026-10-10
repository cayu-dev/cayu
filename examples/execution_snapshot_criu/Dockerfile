# Disposable trusted-controller probe only; never an agent runtime image.
FROM python:3.13-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential pkg-config protobuf-compiler protobuf-c-compiler libprotobuf-c-dev libprotobuf-dev git uuid-dev \
    libnl-3-dev libcap-dev libaio-dev libnet-dev libnftables-dev libgnutls28-dev \
    curl ca-certificates criu python3-yaml

RUN curl -fsSL https://github.com/checkpoint-restore/criu/archive/refs/tags/v4.2.tar.gz -o /tmp/criu.tar.gz \
    && echo "0c6e51af878e63df7391e6dffbbe5f0ced429bc9f1e5a603020bfd2503065c39  /tmp/criu.tar.gz" | sha256sum -c - \
    && tar -xzf /tmp/criu.tar.gz -C /tmp \
    && PATH=/usr/bin:/bin:/usr/sbin:/sbin make -C /tmp/criu-4.2 -j2 criu \
    && install -m 0755 /tmp/criu-4.2/criu/criu /usr/local/sbin/criu \
    && rm -rf /tmp/criu-4.2 /tmp/criu.tar.gz /var/lib/apt/lists/*
