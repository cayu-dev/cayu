"""Install host process limits before executing the exact admitted Git binary."""

from __future__ import annotations

import os
import sys


def main(arguments: list[str]) -> int:
    if os.name != "posix" or len(arguments) < 3:
        return 125
    import resource

    file_bytes = int(arguments[0])
    if file_bytes < 1024:
        return 125
    # Limits are inherited by Git's transport/index-pack children. The wall
    # deadline and process-group settlement remain owned by LocalRunner.
    _soft, hard = resource.getrlimit(resource.RLIMIT_FSIZE)
    if hard != resource.RLIM_INFINITY:
        file_bytes = min(file_bytes, hard)
    resource.setrlimit(resource.RLIMIT_FSIZE, (file_bytes, file_bytes))
    os.execv(arguments[1], arguments[1:])
    return 125


if __name__ == "__main__":
    try:
        code = main(sys.argv[1:])
    except (OSError, ValueError):
        code = 125
    raise SystemExit(code)
