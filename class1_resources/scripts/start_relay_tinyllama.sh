#!/usr/bin/env bash
# Start RelayServe pointed at TinyLlama (llama-server on 8081).
# Run this from class1_resources. Ensure TinyLlama is already running (spawn_backends.py).

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELAY_DIR="$(cd "$SCRIPT_DIR/.." && pwd)/RelayServe"
cd "$RELAY_DIR"
export RELAYSERVE_BACKENDS=http://localhost:8081
exec relayserve
