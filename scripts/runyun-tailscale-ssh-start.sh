#!/usr/bin/env bash
set -u

TS_SOCKET="/var/run/tailscale/tailscaled.sock"
TS_STATE="/root/workspace/tailscale-ssh/state/tailscaled.state"
TS_LOG="/root/workspace/tailscale-ssh/logs/tailscaled.log"
TS_HOSTNAME="runyun"

SSH_DIR="/root/workspace/tailscale-ssh"
SSHD_CONFIG="$SSH_DIR/sshd_config"
SSHD_LOG="$SSH_DIR/sshd.log"
SSHD_PORT="2223"
SERVE_PORT="2222"

log() {
  printf '[runyun-ts] %s\n' "$*"
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    log "missing required command: $1"
    exit 1
  fi
}

ensure_dirs() {
  mkdir -p /var/run/tailscale "$SSH_DIR" "$SSH_DIR/state" "$SSH_DIR/logs"
}

tailscaled_running() {
  tailscale --socket="$TS_SOCKET" status >/dev/null 2>&1
}

start_tailscaled() {
  if tailscaled_running; then
    log "tailscaled is already running"
    return
  fi

  log "starting tailscaled in userspace-networking mode"
  pkill tailscaled 2>/dev/null || true
  rm -f "$TS_SOCKET"

  nohup tailscaled \
    --state="$TS_STATE" \
    --socket="$TS_SOCKET" \
    --tun=userspace-networking \
    --socks5-server=127.0.0.1:1055 \
    --outbound-http-proxy-listen=127.0.0.1:1055 \
    >> "$TS_LOG" 2>&1 &

  sleep 3

  if ! tailscaled_running; then
    log "tailscaled did not become ready; last log lines:"
    tail -80 "$TS_LOG" 2>/dev/null || true
    exit 1
  fi
}

ensure_tailscale_up() {
  log "ensuring Tailscale node is up"
  tailscale --socket="$TS_SOCKET" up --hostname="$TS_HOSTNAME" --accept-dns=false || exit 1
}

port_listening() {
  ss -lnt 2>/dev/null | awk '{print $4}' | grep -Eq "(^|:)$1$"
}

start_sshd() {
  if [ ! -f "$SSHD_CONFIG" ]; then
    log "missing sshd config: $SSHD_CONFIG"
    exit 1
  fi

  if port_listening "$SSHD_PORT"; then
    log "local sshd is already listening on 127.0.0.1:$SSHD_PORT"
    return
  fi

  log "starting local sshd on 127.0.0.1:$SSHD_PORT"
  rm -f "$SSH_DIR/sshd.pid"
  /usr/sbin/sshd -t -f "$SSHD_CONFIG" || exit 1
  /usr/sbin/sshd -f "$SSHD_CONFIG" -E "$SSHD_LOG" || exit 1

  sleep 1
  if ! port_listening "$SSHD_PORT"; then
    log "local sshd did not start; last log lines:"
    tail -80 "$SSHD_LOG" 2>/dev/null || true
    exit 1
  fi
}

start_serve() {
  log "publishing tailnet port $SERVE_PORT to local sshd port $SSHD_PORT"
  tailscale --socket="$TS_SOCKET" serve --bg --tcp "$SERVE_PORT" "$SSHD_PORT" || exit 1
}

show_status() {
  log "Tailscale status:"
  tailscale --socket="$TS_SOCKET" status || true
  log "Serve status:"
  tailscale --socket="$TS_SOCKET" serve status || true
  log "Listening ports:"
  ss -lntp 2>/dev/null | grep -E ":(22|$SSHD_PORT) " || true
  return 0
}

main() {
  require_cmd tailscale
  require_cmd tailscaled
  require_cmd ss
  ensure_dirs
  start_tailscaled
  ensure_tailscale_up
  start_sshd
  start_serve
  show_status
}

main "$@"
exit 0
