## Project layout

- `relayserve`: CLI entrypoint
- `relayserve/internal/device`: device registry + strength scoring
- `relayserve/internal/profile`: probe stubs + profiling hooks
- `relayserve/internal/runner`: per-device runner selection
- `relayserve/internal/scheduler`: request scheduler and phase info
- `relayserve/internal/queue`: in-memory request queue
- `relayserve/internal/shard`: sharding plan stub
- `relayserve/internal/kv`: KV cache manager stub
- `relayserve/internal/metrics`: metrics collection stub
- `relayserve/internal/server`: HTTP server

Defaults:
- HTTP server: `:8080`
- Endpoints:
  - `GET /healthz`
  - `GET /v1/models`
  - `POST /v1/chat/completions`
  - `GET /metrics`
  - `GET /debug/shard`
  - `POST /v1/chat/pretty` (colorized text response)
- Backends: set `RELAYSERVE_BACKENDS` to comma-separated llama.cpp servers

## Environment

- `RELAYSERVE_PORT` (default `8080`)
- `RELAYSERVE_MODEL_ID` (default `relay-gguf`)
- `RELAYSERVE_BACKENDS` (comma-separated, e.g. `http://localhost:8081,http://localhost:8082`)
- `RELAYSERVE_BATCH_SIZE` (default `4`)
- `RELAYSERVE_BATCH_WAIT_MS` (default `10`)
- `RELAYSERVE_METRICS_MAX_ITEMS` (default `1000`)
- `RELAYSERVE_TOTAL_LAYERS` (default `32`)
- `RELAYSERVE_PRETTY_JSON` (set `1` for readable JSON responses)
- `RELAYSERVE_PRETTY_DEFAULT` (default `1`, set `0` for JSON by default)