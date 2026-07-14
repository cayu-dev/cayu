#!/usr/bin/env bash
set -u

sidecar_pid=0
watchdog_pid=0

forward_signal() {
    if ((sidecar_pid > 0)); then kill "-$1" "$sidecar_pid" 2>/dev/null || true; fi
    if ((watchdog_pid > 0)); then kill "-$1" "$watchdog_pid" 2>/dev/null || true; fi
}

trap 'forward_signal TERM' TERM
trap 'forward_signal INT' INT
trap 'forward_signal HUP' HUP
trap 'forward_signal QUIT' QUIT

/usr/bin/amazon-efs-mount-watchdog &
watchdog_pid=$!
python3.11 -m uvicorn lambda_microvm_sidecar.app:app --host 0.0.0.0 --port 8080 &
sidecar_pid=$!

while kill -0 "$sidecar_pid" 2>/dev/null; do wait -n || true; done
forward_signal TERM
wait "$sidecar_pid"
