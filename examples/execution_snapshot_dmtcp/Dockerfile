FROM python:3.13-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates && \
    curl -fsSL https://github.com/dmtcp/dmtcp/archive/refs/tags/v4.2.0.tar.gz -o /tmp/dmtcp.tar.gz && \
    echo '043410566fd7c09f21e0ec485cf72e9481722629ea877650142b7505e0d7f2d2  /tmp/dmtcp.tar.gz' | sha256sum -c - && \
    tar -xzf /tmp/dmtcp.tar.gz -C /tmp && cd /tmp/dmtcp-4.2.0 && \
    CFLAGS=-O2 CXXFLAGS=-O2 ./configure --prefix=/opt/cayu-dmtcp && \
    make -j2 && make install && rm -rf /tmp/dmtcp* /var/lib/apt/lists/*
COPY src/cayu/environments/_snapshot_barrier.c /tmp/cayu-snapshot-barrier.c
RUN mkdir -p /opt/cayu-dmtcp/include/dmtcp && \
    cp /opt/cayu-dmtcp/include/version.h /opt/cayu-dmtcp/include/dmtcp/version.h && \
    gcc -shared -fPIC -O2 -I/opt/cayu-dmtcp/include /tmp/cayu-snapshot-barrier.c \
    -o /opt/cayu-dmtcp/lib/dmtcp/libcayu_snapshot.so && \
    rm /tmp/cayu-snapshot-barrier.c && mkdir /workspace && chown 1000:1000 /workspace
USER 1000:1000
WORKDIR /workspace
CMD ["python3", "-c", "import time; time.sleep(86400)"]
