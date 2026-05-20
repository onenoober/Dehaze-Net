#!/usr/bin/env bash
set -u

ROOT="/root/workspace/tailscale-ssh"
START_SCRIPT="$ROOT/start.sh"
TS_SOCKET="/var/run/tailscale/tailscaled.sock"
SSHD_PORT="2223"
SERVE_PORT="2222"

log() {
  printf '[runyun-ts-watch] %s\n' "$*"
}

tailscale_ready() {
  tailscale --socket="$TS_SOCKET" status >/dev/null 2>&1
}

sshd_ready() {
  ss -lnt 2>/dev/null | awk '{print $4}' | grep -Eq "(^|:)$SSHD_PORT$"
}

serve_ready() {
  tailscale --socket="$TS_SOCKET" serve status 2>/dev/null | grep -q "127.0.0.1:$SSHD_PORT"
}

recover() {
  log "running recovery script"
  /bin/bash "$START_SCRIPT"
}

main() {
  mkdir -p "$ROOT/logs" "$ROOT/state"

  while true; do
    if ! tailscale_ready || ! sshd_ready || ! serve_ready; then
      recover || log "recovery failed with exit code $?"
    else
      log "ok: tailscale, serve:$SERVE_PORT, sshd:$SSHD_PORT"
    fi
    sleep 60
  done
}

main "$@"
