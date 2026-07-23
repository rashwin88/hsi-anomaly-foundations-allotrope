#!/usr/bin/env bash
# Open an SSH tunnel from your laptop to the remote stack.
#
# Forwards:
#   localhost:3000   →   remote 127.0.0.1:3000   (frontend)
#   localhost:8000   →   remote 127.0.0.1:8000   (api, for curl/debug)
#
# Then in your browser: http://localhost:3000
#
# Usage:
#   ./scripts/remote_tunnel.sh user@host [-p PORT] [-i KEY]
#
# Example:
#   ./scripts/remote_tunnel.sh root@116.127.115.43 -p 12345 -i ~/.ssh/id_ed25519
#
# Leave the terminal window open while you're working. Ctrl-C closes the
# tunnel. The remote services keep running.
#
# If your local 3000 or 8000 is already in use, override with env vars:
#   LOCAL_FRONTEND=3001 LOCAL_API=8001 ./scripts/remote_tunnel.sh root@... -p ...
#
# For auto-reconnect on flaky networks, install autossh and set
# USE_AUTOSSH=1 in your env. The script will pick autossh up automatically.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 user@host [-p PORT] [-i KEY] [extra ssh args]"
  exit 1
fi

DEST="$1"; shift

LOCAL_FRONTEND="${LOCAL_FRONTEND:-3000}"
LOCAL_API="${LOCAL_API:-8000}"

# Prefer autossh when present and requested. autossh re-establishes the
# tunnel automatically if it drops (laptop sleep, network blip).
SSH_BIN="ssh"
if [[ "${USE_AUTOSSH:-0}" == "1" ]] && command -v autossh >/dev/null 2>&1; then
  SSH_BIN="autossh -M 0"
fi

echo "▶ Opening SSH tunnel to $DEST"
echo "    frontend  →  http://localhost:${LOCAL_FRONTEND}"
echo "    api       →  http://localhost:${LOCAL_API}"
echo
echo "  Leave this window open. Ctrl-C to close the tunnel."
echo

exec $SSH_BIN -N \
  -o "ServerAliveInterval=30" \
  -o "ServerAliveCountMax=3" \
  -o "ExitOnForwardFailure=yes" \
  -L "${LOCAL_FRONTEND}:127.0.0.1:3000" \
  -L "${LOCAL_API}:127.0.0.1:8000" \
  "$@" \
  "$DEST"
