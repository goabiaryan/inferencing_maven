# Class3 Runs — Step-by-Step Run Guide

Do everything from the **class3_runs** directory unless noted.

---

## Overview: Two ways to run (choose one)

| Path | What you run | When to use |
|------|--------------|-------------|
| **Path 1 — Modal script** | `modal run run_single_gpu_modal.py` | You want benchmarks (Toka, vLLM, SGLang) to run **entirely on Modal**. No notebook, no local server. One command; results saved to `benchmark_results/`. |
| **Path 2 — Notebook** | Start servers yourself (local or Modal), then open `run_single_gpu_inference_nb.ipynb` | You want to run the **hands-on notebook** and control servers step by step (e.g. start vLLM in a terminal, then run notebook cells). You do **not** need to run the Modal script first. |

**Important:** You do **not** need to run `run_single_gpu_modal.py` before running the notebook. They are separate: the Modal script runs benchmarks on the cloud; the notebook expects you to start a server (locally or via a deployed endpoint) and then run the notebook against it.

---

## vLLM Chat + CrewAI demo (separate from the benchmark)

For the **vLLM Chat + CrewAI** notebook (`vllm_chat_and_crewai_demo.ipynb`):

1. **Deploy the vLLM server once:**
   ```bash
   modal deploy vllm_modal_serve.py
   ```
2. Copy the printed URL (e.g. `https://you--vllm-chat-demo-serve.modal.run`).
3. In the **vLLM Chat notebook**: set **OPENAI_BASE_URL** to that URL + `/v1` (in the notebook’s “paste endpoint” cell or in Modal Notebook → Settings → Secrets).
4. Run the notebook (Step 1 → chat → CrewAI). No need to run `run_single_gpu_modal.py` for this.

---

## Option A: Run benchmarks on Modal (no notebook)

**Use this when:** You want to run the Tokasaurus / vLLM / SGLang benchmarks on Modal in one go. You do **not** run the notebook for this path.

If you're on **macOS** or don't have a Linux machine with an NVIDIA GPU, run the full exercise on **Modal** (Linux + T4 GPU in the cloud). No local Triton, vLLM, or Tokasaurus install needed.

1. **Install Modal and log in** (one-time):
   ```bash
   pip install modal
   modal token set
   ```
   Use the browser or paste a token from https://modal.com/settings.


2. **Run a single engine** (faster for iteration):
   ```bash
   modal run run_single_gpu_modal.py --engine toka
   modal run run_single_gpu_modal.py --engine vllm
   modal run run_single_gpu_modal.py --engine sglang
   ```

3. **From class3_runs, run the benchmark on Modal**:
   ```bash
   cd /path/to/class1_resources/class3_runs
   modal run run_single_gpu_modal.py
   ```
   This runs **all three engines** (Tokasaurus, vLLM, SGLang) in parallel and prints the combined results table. First run may take several minutes (image build + model download + Tokasaurus install).
   **Results** are saved to `benchmark_results/benchmark_<timestamp>.md` with the config (engines, model, warmup, n_runs, max_tokens, GPU) and the full results table.
   
4. **Optional arguments** (same as the local script):
   ```bash
   modal run run_single_gpu_modal.py --model Qwen/Qwen2.5-1.5B-Instruct --warmup 10 --n-runs 5 --max-tokens 128
   ```

The default model **Qwen/Qwen2.5-1.5B-Instruct** is non-gated; no HF token needed. To use a gated model on Modal, you’d need to pass secrets (e.g. via Modal secrets); see Modal docs.

---

## Option B: Run the notebook (local Linux + NVIDIA GPU)

**Use this when:** You want to run **run_single_gpu_inference_nb.ipynb** and control servers yourself. You do **not** run `run_single_gpu_modal.py` for this path. You start one server at a time (Tokasaurus, then vLLM, then SGLang) in a terminal and run the notebook cells against that server.

---

## Step 1: Go to class3_runs

```bash
cd /path/to/class1_resources/class3_runs
```

(Use your actual path to the repo.)

---

## Step 2: Install Tokasaurus (if not already done)

```bash
cd tokasaurus
pip install -e .
cd ..
```

You should be back in **class3_runs**. If you get errors, fix the environment (Python 3.10+, etc.) then retry.

**If you see `ModuleNotFoundError: No module named 'triton'`:** Triton (required by flashinfer) is **Linux-only** — there are no macOS wheels, so `pip install triton` will fail on a Mac. **Tokasaurus cannot run locally on macOS.** Run the Tokasaurus part of the exercise on a **Linux machine with an NVIDIA GPU** (e.g. Modal, Lambda Labs, a cloud VM, or a Linux workstation). On that Linux environment, `pip install triton` (or installing tokasaurus with `pip install -e .`) will pull in Triton. To do the full exercise (Tokasaurus + vLLM + SGLang) on one GPU, use a single Linux GPU instance for all three.

---

## Step 3: (Optional) API keys

The default model **Qwen/Qwen2.5-1.5B-Instruct** is **non-gated** — no Hugging Face login or token needed. You can skip this step.

If you switch to a gated model (e.g. Llama) or use Modal/Lambda keys:

- Copy `secrets.example.env` to `secrets.env` (or `mysecrets.env`) and set `HF_TOKEN` and any other keys. The notebook and script load them via `load_secrets()`.
- Or in the shell: `export HF_TOKEN=your_token`

---

## Step 4: (Optional) Pre-download the model

The default model **Qwen/Qwen2.5-1.5B-Instruct** is non-gated. It will download automatically when you start the server; no token needed. To pre-download:

```bash
cd tokasaurus
toka-download model=Qwen/Qwen2.5-1.5B-Instruct
cd ..
```

If you use a **gated** model (e.g. Llama), set `HF_TOKEN` in `class3_runs/mysecrets.env` or `export HF_TOKEN=...` first, accept the model terms on the Hub, then run `toka-download model=meta-llama/Llama-3.2-1B-Instruct`.

---

## Step 5: Prepare options files (optional but useful)

To use custom server flags without long command lines:

```bash
cp vllm_serve_options.example.txt vllm_serve_options.txt
cp sglang_serve_options.example.txt sglang_serve_options.txt
cp toka_serve_options.example.txt toka_serve_options.txt
```

Edit each `*_serve_options.txt` if you want (ports, memory, Hydragen, etc.). The defaults are fine for a first run.

---

## Step 6: Open the notebook

In Cursor (or Jupyter), open:

**`run_single_gpu_inference_nb.ipynb`**

**You do not need to run `run_single_gpu_modal.py` first.** This notebook talks to a server that you start yourself (Step 7 onwards). The Modal script is a separate path that runs benchmarks entirely on Modal without this notebook.

Run the cells in **§1 Setup** (1.1 and 1.2) so that:

- Imports and `load_secrets()` run.
- `MODEL`, `API_HOST`, ports, and benchmark settings are set.

Check that there’s no error and, if you use Llama, that the HF token warning is gone after loading secrets.

---

## Step 7: Start the first server (Tokasaurus)

Use a **separate terminal** (leave the notebook running).

**Option A — With options file (from class3_runs):**

```bash
cd /path/to/class3_runs
python launch_server.py --engine toka --model Qwen/Qwen2.5-1.5B-Instruct --options-file toka_serve_options.txt
```

**Option B — Command line only:**

```bash
cd class3_runs/tokasaurus
toka model=Qwen/Qwen2.5-1.5B-Instruct
```

Wait until the server prints that it’s ready (e.g. “System startup time” or similar). Leave this terminal open.

---

## Step 8: Run the Tokasaurus benchmark in the notebook

In the notebook:

1. In **§5** (benchmark loop), ensure the cell is set to run **Tokasaurus** (default):
   - `BASE_URL = f"http://{API_HOST}:{TOKASAURUS_PORT}/v1"` (port 10210).
   - The block that calls `run_benchmark(..., "Tokasaurus", stream=False, ...)` is uncommented.
2. Run that cell. It will warm up, then run 5 (or your `N_RUNS`) timed runs per scenario.
3. Run the **§6 Results table** cell. You should see a table with mean/p50/p95 for TTFT, tokens/sec, total time, VRAM, GPU util for short, long, and multi_turn_chat.

If you get connection errors, check that the server is still running and that `API_HOST` / `TOKASAURUS_PORT` in the notebook match the server (e.g. `localhost`, 10210).

---

## Step 9: Stop Tokasaurus and start vLLM

In the terminal where Tokasaurus is running, stop it: **Ctrl+C**.

Then start vLLM from **class3_runs**:

**With options file:**

```bash
python launch_server.py --engine vllm --model Qwen/Qwen2.5-1.5B-Instruct --options-file vllm_serve_options.txt
```

**Without options file:**

```bash
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-1.5B-Instruct --host 0.0.0.0 --port 8000
```

Wait until vLLM is ready (e.g. “Uvicorn running on ...”).

---

## Step 10: Run the vLLM benchmark in the notebook

1. In the **§5** cell, comment out the Tokasaurus block and uncomment the vLLM block, e.g.:
   ```python
   # BASE_URL = f"http://{API_HOST}:{TOKASAURUS_PORT}/v1"
   BASE_URL_VLLM = f"http://{API_HOST}:{VLLM_PORT}/v1"
   # ... (comment toka, uncomment vLLM run_benchmark and all_rows.extend)
   ```
2. Set `BASE_URL = BASE_URL_VLLM` (or use `BASE_URL_VLLM` in the `run_benchmark` call).
3. Run the benchmark cell, then the **§6 Results** cell again. You should see vLLM rows added (or a table with only vLLM if you cleared `all_rows`).

---

## Step 11: Stop vLLM and start SGLang

In the vLLM terminal: **Ctrl+C**.

From **class3_runs**:

**With options file:**

```bash
python launch_server.py --engine sglang --model Qwen/Qwen2.5-1.5B-Instruct --options-file sglang_serve_options.txt
```

**Without options file:**

```bash
python -m sglang.launch_server --model-path Qwen/Qwen2.5-1.5B-Instruct --host 0.0.0.0 --port 30000
```

Wait until SGLang is ready.

---

## Step 12: Run the SGLang benchmark in the notebook

1. In **§5**, switch to SGLang: use `BASE_URL_SGLANG = f"http://{API_HOST}:{SGLANG_PORT}/v1"` and the SGLang `run_benchmark` block (comment vLLM, uncomment SGLang).
2. Run the benchmark cell and the **§6 Results** cell. You should now have results for all three engines (if you didn’t clear `all_rows`).

---

## Step 13: Compare and optional experiments

- Look at the final table: compare **tokens_per_sec_mean**, **ttft_mean** (where available), **peak_vram_mb_mean**, **avg_gpu_util_pct_mean** across engines and scenarios.
- **§7 Flag experimentation:** Change options in `toka_serve_options.txt`, `vllm_serve_options.txt`, or `sglang_serve_options.txt` (e.g. Hydragen, gpu-memory-utilization, radix cache), restart that server, and re-run the benchmark for that engine to see the effect.
- **§8 Speculative decoding:** Start vLLM or SGLang with speculative decoding enabled (see notebook §8 for exact flags), then run `run_spec_demo(...)` and compare tokens/sec with the non-spec run.

---

## Step 14: (Optional) Use the standalone script instead of the notebook

From **class3_runs**, with one server running (e.g. Tokasaurus):

```bash
python run_single_gpu_inference.py --engine toka --model Qwen/Qwen2.5-1.5B-Instruct --warmup 10 --n-runs 5 --wait
```

`--wait` waits for the server to be ready. Use `--engine vllm` or `--engine sglang` when the corresponding server is running. Results print to the terminal (mean/p50/p95 table).

---

## Quick reference

**Path 1 (Modal only):** `modal run run_single_gpu_modal.py` — no notebook. Results in `benchmark_results/`.

**Path 2 (Notebook):** Follow steps below. You do **not** run the Modal script first; you start servers yourself and run the notebook.

| Step | What you do |
|------|-------------|
| 1 | `cd class3_runs` |
| 2 | `cd tokasaurus && pip install -e . && cd ..` |
| 3 | Copy `secrets.example.env` → `secrets.env`, set `HF_TOKEN` (or `export HF_TOKEN=...`) |
| 4 | (Optional) Download model: `toka-download model=...` or huggingface_hub |
| 5 | (Optional) `cp *_serve_options.example.txt *_serve_options.txt` and edit |
| 6 | Open `run_single_gpu_inference_nb.ipynb`, run §1.1 and §1.2 |
| 7 | Terminal: start Tokasaurus (`launch_server.py --engine toka` or `toka model=...`) |
| 8 | Notebook: run §5 (Tokasaurus) and §6 (results) |
| 9 | Ctrl+C server; start vLLM (`launch_server.py --engine vllm` or `vllm` command) |
| 10 | Notebook: run §5 (vLLM) and §6 |
| 11 | Ctrl+C server; start SGLang (`launch_server.py --engine sglang` or `sglang` command) |
| 12 | Notebook: run §5 (SGLang) and §6 |
| 13 | Compare table; try §7 / §8 experiments |
| 14 | (Optional) `python run_single_gpu_inference.py --engine toka|vllm|sglang --wait` |

If something fails, check: (1) server is running and port matches notebook config, (2) `HF_TOKEN` set for Llama, (3) you’re in **class3_runs** when running `launch_server.py` or `run_single_gpu_inference.py`.
