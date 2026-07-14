#!/usr/bin/env bash
set -u

child_pid=0

forward_signal() {
    if ((child_pid > 0)); then
        kill "-$1" "$child_pid" 2>/dev/null || true
    fi
}

trap 'forward_signal TERM' TERM
trap 'forward_signal INT' INT
trap 'forward_signal HUP' HUP
trap 'forward_signal QUIT' QUIT

python3.11 -m uvicorn lambda_microvm_sidecar.app:app --host 0.0.0.0 --port 8080 &
child_pid=$!

# As PID 1, Bash adopts command descendants whose direct parent exits. Waiting
# for any child, rather than only Uvicorn, prevents those descendants becoming
# permanent zombies during long-lived MicroVM sessions.
while kill -0 "$child_pid" 2>/dev/null; do
    wait -n || true
done

wait "$child_pid"
exit $?
