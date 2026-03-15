# Single-GPU Inference Exercise

Hands-on for **"Optimizing Inference on a Single GPU"**. Run entirely in Cursor.

**→ Step-by-step:** See **`RUN_GUIDE.md`** in this directory for a one-by-one run order (setup → Tokasaurus → vLLM → SGLang → compare).

**→ Run on Modal (macOS / no local GPU):** From this directory run `pip install modal && modal token set`, then `modal run run_single_gpu_modal.py` to run all three engines on a cloud T4 GPU. See RUN_GUIDE.md “Option A: Run on Modal”.

## Quick start

1. **Environment**
   - Clone and install Tokasaurus: `cd tokasaurus && pip install -e .`
   - **API keys:** Copy `secrets.example.env` to `secrets.env` and fill in your values (HF_TOKEN, Modal, Lambda Labs, etc.). The notebook and script load them via `load_secrets()`. Never commit `secrets.env`.
   - Or set in shell: `export HF_TOKEN=your_token`

2. **Open the notebook**
   - `run_single_gpu_inference_nb.ipynb` in this directory

3. **Run one server at a time** (in a terminal), then run the notebook cells.
   - **Command line:** See notebook §1.4 for minimal commands (engine chooses kernel/settings).
   - **Options files (recommended for custom flags):** Copy `*_serve_options.example.txt` to `*_serve_options.txt` (e.g. `vllm_serve_options.txt`), edit the file, then run:
     ```bash
     python launch_server.py --engine vllm --model Qwen/Qwen2.5-1.5B-Instruct --options-file vllm_serve_options.txt
     python launch_server.py --engine sglang --model Qwen/Qwen2.5-1.5B-Instruct --options-file sglang_serve_options.txt
     python launch_server.py --engine toka --model Qwen/Qwen2.5-1.5B-Instruct --options-file toka_serve_options.txt
     ```
     Omit `--options-file` to use only the model (defaults). Options files are in this directory; don’t commit your `*_serve_options.txt` (only the `.example.txt` files are in git).
   - If you use a different port in the options file, set `VLLM_PORT` or `SGLANG_PORT` in the notebook config so the client connects to the right server.

4. **Benchmarks**
   - **Best practices:** Warm-up (10 requests), then 5–10 repeated runs per scenario; report **mean / p50 / p95** for TTFT (prefill), decode time, decode tokens/sec, overall tokens/sec, total time, peak VRAM (nvidia-smi), avg GPU util %. Greedy decoding (temperature=0) for reproducibility. With streaming, **prefill** and **decode** are reported separately.
   - Shared prompt set: **short** (prefill), **long** (KV stress), **multi-turn chat** (prefix caching).
   - Same OpenAI-compatible client for all engines.
   - **Saved results:** Each run writes a timestamped `.md` to `benchmark_results/` with the config (flags/settings) and results table. Use `--no-save` (script) to skip saving; use `--output-dir` to change the directory.

5. **Bonus & flag experiments**
   - **Tokasaurus:** `use_hydragen=true`/`false`, `hydragen_min_group_size=32`/`129`/`256`
   - **vLLM:** `--gpu-memory-utilization 0.7`/`0.85`/`0.95`, `--enforce-eager`, `--max-num-seqs 128`/`256`
   - **SGLang:** `--mem-fraction-static 0.7`/`0.85`, `--disable-radix-cache`, `--speculative-algorithm EAGLE` (if supported)
   - See notebook §7 and the `*_serve_options.example.txt` files for copy-paste values.

## Files

- `run_single_gpu_inference_nb.ipynb` — main exercise (setup, client, monitoring, results table)
- `run_single_gpu_inference.py` — standalone script: `python run_single_gpu_inference.py --engine toka|vllm|sglang [--warmup 10] [--n-runs 5] [--no-gpu] [--wait] [--no-save] [--output-dir PATH]` (warm-up + repeated runs → mean/p50/p95 table; saves to `benchmark_results/benchmark_<timestamp>.md` unless `--no-save`)
- `launch_server.py` — launch a server using options from a file: `python launch_server.py --engine vllm|sglang|toka --model <model> [--options-file ...]`
- `vllm_serve_options.example.txt`, `sglang_serve_options.example.txt`, `toka_serve_options.example.txt` — copy to `*_serve_options.txt` and edit to set flags in a file instead of the command line
- `secrets.example.env` — template for API keys (copy to `secrets.env`; see above)
- `load_secrets.py` — loads `secrets.env` into `os.environ` for notebooks/scripts
- `run_single_gpu_modal.py` — Modal app: runs the benchmark on a cloud GPU; saves results to `benchmark_results/benchmark_<timestamp>.md` with config and flags (Modal 1.0+: `image.add_local_dir`, no `Mount`)

## Cloud (Modal / Lambda Labs)

- **Modal:** Run the full exercise without a local GPU: `modal run run_single_gpu_modal.py` (see RUN_GUIDE.md Option A). Use `--engine toka|vllm|sglang` to run a single engine. Requires **Modal 1.0+** (uses `Image.add_local_dir` for class3_runs; `modal.Mount` is no longer used).
- **Other clouds:** Use the same notebook; set `API_HOST` to the remote host and ensure the server is bound to `0.0.0.0` and the port is open.
