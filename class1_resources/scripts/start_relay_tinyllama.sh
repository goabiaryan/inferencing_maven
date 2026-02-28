set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELAY_DIR="$(cd "$SCRIPT_DIR/.." && pwd)/RelayServe"
cd "$RELAY_DIR"
export RELAYSERVE_ROOT="$RELAY_DIR"
export RELAYSERVE_BACKENDS=http://localhost:8081
exec relayserve
