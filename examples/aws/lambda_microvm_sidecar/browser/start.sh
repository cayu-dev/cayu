#!/usr/bin/env bash
# Browser image entrypoint. MicroVMs restore from a memory snapshot taken after
# the ready hook while their root filesystem is fetched lazily, so a first
# Chromium start could take close to a minute. Reading the browser's files
# before the sidecar starts leaves them in the snapshot's page cache; a cold
# start then measured a few seconds. (A warm-up launch of Chromium before the
# snapshot measured slower, so none is done.)
set -uo pipefail

find /opt/cayu-browser /ms-playwright /usr/lib/python3.11/site-packages/playwright -type f \
    -exec cat {} + > /dev/null

exec bash /opt/cayu/lambda_microvm_sidecar/entrypoint.sh
