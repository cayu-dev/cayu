#!/usr/bin/env bash
set -euo pipefail

configure_agent_network_boundary() {
    ip netns add "$CAYU_MICROVM_AGENT_NETNS"
    ip link add cayu-root type veth peer name cayu-agent
    ip addr add 192.0.2.1/30 dev cayu-root
    ip link set cayu-root up
    ip link set cayu-agent netns "$CAYU_MICROVM_AGENT_NETNS"
    ip netns exec "$CAYU_MICROVM_AGENT_NETNS" ip addr add 192.0.2.2/30 dev cayu-agent
    ip netns exec "$CAYU_MICROVM_AGENT_NETNS" ip link set lo up
    ip netns exec "$CAYU_MICROVM_AGENT_NETNS" ip link set cayu-agent up
    iptables -w -I INPUT 1 -i cayu-root -j REJECT
    iptables -w -I INPUT 1 -i cayu-root -p tcp --dport 18080 -j ACCEPT
}

configure_agent_network_boundary

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
