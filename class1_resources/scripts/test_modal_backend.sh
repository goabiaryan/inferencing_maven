set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLASS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RELAY_URL="${RELAY_URL:-http://localhost:8080}"

echo "RelayServe URL: $RELAY_URL"
echo ""
read -r -p "Enter prompt (then press Enter): " PROMPT
if [ -z "$PROMPT" ]; then
  PROMPT="Explain KV cache in one sentence."
  echo "(using default: $PROMPT)"
fi
echo ""
echo "Which backends do you want to run?"
echo "  1) Local llama.cpp only"
echo "  2) Modal llama.cpp only"
echo "  3) Both: local + modal (llama.cpp)"
echo "  4) Both + Modal vLLM (cold start only)"
echo "  5) All: local, modal, and Modal vLLM (cold + warm)"
read -r -p "Choice (1–5): " CHOICE
CHOICE="${CHOICE:-3}"
if [ "$CHOICE" != "1" ] && [ "$CHOICE" != "2" ] && [ "$CHOICE" != "3" ] && [ "$CHOICE" != "4" ] && [ "$CHOICE" != "5" ]; then
  echo "Invalid choice. Using 3 (both local + modal)."
  CHOICE="3"
fi
echo ""

PAYLOAD_FILE="/tmp/relay_payload_$$.json"
trap 'rm -f "$PAYLOAD_FILE"' EXIT
write_payload() {
  local model="$1"
  local content="${2:-$PROMPT}"
  printf '%s' "$content" | python3 -c "
import json, sys
p = sys.stdin.read().strip()
payload = {\"model\": \"$model\", \"messages\": [{\"role\": \"user\", \"content\": p}], \"stream\": False}
with open(\"$PAYLOAD_FILE\", \"w\") as f: json.dump(payload, f)
"
}

LOCAL_MS="—"
MODAL_MS="—"
MODAL_VLLM_COLD_MS="—"
MODAL_VLLM_MS="—"

CONFIG="$CLASS_ROOT/RelayServe/config.yaml"
if [ "$CHOICE" = "2" ] || [ "$CHOICE" = "3" ] || [ "$CHOICE" = "4" ] || [ "$CHOICE" = "5" ]; then
  if [ -f "$CONFIG" ]; then
    if grep -q "url: https://YOUR_MODAL_URL" "$CONFIG" 2>/dev/null; then
      echo "ERROR: Modal is required. Set backends.modal.url in RelayServe/config.yaml to your Modal web URL."
      echo "  Deploy first: bash scripts/deploy_modal.sh"
      echo "  Then paste the printed URL into config.yaml under backends.modal.url"
      exit 1
    fi
  fi
fi

echo "Testing with prompt: $PROMPT"
echo ""

# Local backend (options 1, 3, 4, 5)
if [ "$CHOICE" = "1" ] || [ "$CHOICE" = "3" ] || [ "$CHOICE" = "4" ] || [ "$CHOICE" = "5" ]; then
echo "Request to local backend (model: local)..."
write_payload "local"
START=$(python3 -c "import time; print(time.perf_counter())")
curl -s -D /tmp/relay_hdr_local.txt -o /tmp/relay_local.txt -X POST "$RELAY_URL/v1/chat/pretty" \
  -H "Content-Type: application/json" \
  -d @"$PAYLOAD_FILE"
END=$(python3 -c "import time; print(time.perf_counter())")
LOCAL_MS=$(python3 -c "print(int(($END - $START) * 1000))")
if ! grep -q "Relay Response" /tmp/relay_local.txt 2>/dev/null; then
  echo "WARNING: Local request may have failed. Check RelayServe and local llama.cpp on 8081."
fi
echo "Local latency: ${LOCAL_MS}ms"
echo "--- Response headers ---"
cat /tmp/relay_hdr_local.txt
echo "--- Response (pretty) ---"
cat /tmp/relay_local.txt
echo ""
fi

# Modal backend (options 2, 3, 4, 5)
if [ "$CHOICE" = "2" ] || [ "$CHOICE" = "3" ] || [ "$CHOICE" = "4" ] || [ "$CHOICE" = "5" ]; then
echo "Request to Modal backend (model: modal)..."
write_payload "modal"
START=$(python3 -c "import time; print(time.perf_counter())")
HTTP_CODE=$(curl -s -D /tmp/relay_hdr_modal.txt -o /tmp/relay_modal.txt -w "%{http_code}" -X POST "$RELAY_URL/v1/chat/pretty" \
  -H "Content-Type: application/json" \
  -d @"$PAYLOAD_FILE")
END=$(python3 -c "import time; print(time.perf_counter())")
MODAL_MS=$(python3 -c "print(int(($END - $START) * 1000))")
if [ "$HTTP_CODE" != "200" ] || ! grep -q "Relay Response" /tmp/relay_modal.txt 2>/dev/null; then
  echo "ERROR: Modal request failed (HTTP $HTTP_CODE). Ensure Modal is deployed and config.yaml has the correct backends.modal.url."
  exit 1
fi
echo "Modal latency: ${MODAL_MS}ms"
echo "--- Response headers ---"
cat /tmp/relay_hdr_modal.txt
echo "--- Response (pretty) ---"
cat /tmp/relay_modal.txt
echo ""
fi

# Modal vLLM — cold only (option 4), cold + warm (option 5)
if [ "$CHOICE" = "4" ] || [ "$CHOICE" = "5" ]; then
echo "Request to Modal vLLM (model: modal_vllm) [cold]..."
write_payload "modal_vllm" "Hi"
START=$(python3 -c "import time; print(time.perf_counter())")
HTTP_VLLM_MODAL=$(curl -s -D /tmp/relay_hdr_modal_vllm.txt -o /tmp/relay_modal_vllm.txt -w "%{http_code}" -X POST "$RELAY_URL/v1/chat/pretty" \
  -H "Content-Type: application/json" \
  -d @"$PAYLOAD_FILE")
END=$(python3 -c "import time; print(time.perf_counter())")
MODAL_VLLM_COLD_MS=$(python3 -c "print(int(($END - $START) * 1000))")
if [ "$HTTP_VLLM_MODAL" != "200" ] || ! grep -q "Relay Response" /tmp/relay_modal_vllm.txt 2>/dev/null; then
  echo "WARNING: modal_vllm failed (HTTP $HTTP_VLLM_MODAL). Deploy vLLM on Modal and set backends.modal_vllm.url in config."
  MODAL_VLLM_COLD_MS="—"
  MODAL_VLLM_MS="—"
else
  echo "Modal vLLM (cold): ${MODAL_VLLM_COLD_MS}ms"
  echo "--- Response headers (cold) ---"
  cat /tmp/relay_hdr_modal_vllm.txt
  echo "--- Response (pretty, cold) ---"
  cat /tmp/relay_modal_vllm.txt
  if [ "$CHOICE" = "5" ]; then
    echo "Request to Modal vLLM (model: modal_vllm) [warm]..."
    write_payload "modal_vllm"
    START=$(python3 -c "import time; print(time.perf_counter())")
    curl -s -D /tmp/relay_hdr_modal_vllm.txt -o /tmp/relay_modal_vllm.txt -X POST "$RELAY_URL/v1/chat/pretty" \
      -H "Content-Type: application/json" \
      -d @"$PAYLOAD_FILE"
    END=$(python3 -c "import time; print(time.perf_counter())")
    MODAL_VLLM_MS=$(python3 -c "print(int(($END - $START) * 1000))")
    echo "Modal vLLM (warm): ${MODAL_VLLM_MS}ms"
    echo "--- Response headers (warm) ---"
    cat /tmp/relay_hdr_modal_vllm.txt
    echo "--- Response (pretty, warm) ---"
    cat /tmp/relay_modal_vllm.txt
  fi
fi
echo ""
fi

echo "--- Summary ---"
echo "Local (llama.cpp): ${LOCAL_MS}ms"
echo "Modal (llama.cpp): ${MODAL_MS}ms"
echo "Modal vLLM cold: ${MODAL_VLLM_COLD_MS}ms  warm: ${MODAL_VLLM_MS}ms"
echo "Done."
