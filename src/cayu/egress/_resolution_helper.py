from __future__ import annotations

import json
import socket
import sys

_MAX_RESOLVER_ADDRESSES = 64
_MAX_RESOLVER_OUTPUT_BYTES = 16 * 1024


def main() -> int:
    if len(sys.argv) != 3:
        return 2
    host = sys.argv[1]
    try:
        port = int(sys.argv[2])
    except ValueError:
        return 2
    if not host or not 1 <= port <= 65535:
        return 2
    try:
        records = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError:
        return 3
    addresses = list(dict.fromkeys(str(record[4][0]) for record in records))
    if len(addresses) > _MAX_RESOLVER_ADDRESSES:
        return 4
    payload = json.dumps(addresses, separators=(",", ":")).encode("utf-8")
    if len(payload) > _MAX_RESOLVER_OUTPUT_BYTES:
        return 4
    sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
