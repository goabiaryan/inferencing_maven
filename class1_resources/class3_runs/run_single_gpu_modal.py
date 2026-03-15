"""
Run the Single-GPU Inference Exercise on Modal (Linux + NVIDIA GPU).

From class3_runs:
  modal run run_single_gpu_modal.py                    # run all 3 engines
  modal run run_single_gpu_modal.py --engine toka      # Tokasaurus only
  modal run run_single_gpu_modal.py --engine vllm      # vLLM only
  modal run run_single_gpu_modal.py --engine sglang   # SGLang only

Requires: pip install modal && modal token set
"""
from __future__ import annotations

import concurrent.futures
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import modal

# Add class3_runs to the image so we have tokasaurus and run_single_gpu_inference.py.
# Run this script from class3_runs:  modal run run_single_gpu_modal.py
CLASS3_RUNS = Path(__file__).resolve().parent
MOUNT_PATH = "/class3_runs"
BENCHMARK_DIR = CLASS3_RUNS / "benchmark_results"

# Separate images per engine to avoid pip resolving vllm+sglang together (backtracks to
# source-only vllm that needs CUDA_HOME during CPU image build). Each image has only what that engine needs.
def _base_with_mount(img: modal.Image) -> modal.Image:
    return img.add_local_dir(str(CLASS3_RUNS), remote_path=MOUNT_PATH)

# Tokasaurus: Triton + Flashinfer JIT need CUDA toolkit (nvcc). Use NVIDIA CUDA devel image; git for clone/install.
image_toka = _base_with_mount(
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git")
    .pip_install("openai", "pandas", "numpy")
)

# vLLM: prebuilt wheels, no nvcc needed. Minimal image is enough.
image_vllm = _base_with_mount(
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm", "openai", "pandas", "numpy")
)

# SGLang: Flashinfer CUDA graph capture needs nvcc; sgl_kernel needs libnuma. Use NVIDIA CUDA devel + libnuma1.
image_sglang = _base_with_mount(
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("libnuma1")
    .pip_install("sglang", "openai", "pandas", "numpy")
)

app = modal.App("single-gpu-inference-bench")

# Default model (non-gated, no HF token needed)
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
# GPU: "A10G" (24GB, faster) or "A100" (40/80GB), "T4" (16GB, slower). Bigger = faster load + inference.
GPU_TYPE = "A10G"
TOKA_PORT = 10210
VLLM_PORT = 8000
SGLANG_PORT = 30000


def _wait_for_server(base_url: str, model: str, max_retries: int = 60, sleep_s: float = 2.0) -> bool:
    try:
        from openai import OpenAI
        client = OpenAI(base_url=base_url, api_key="fake-key", max_retries=0, timeout=5)
        for i in range(max_retries):
            try:
                client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": "hi"}],
                    max_tokens=2,
                )
                return True
            except Exception:
                if (i + 1) % 15 == 0 and i > 0:
                    print(f"  Still waiting... ({i + 1}/{max_retries})", flush=True)
                time.sleep(sleep_s)
        return False
    except ImportError:
        return False


def _run_engine_benchmark(
    engine: str,
    model: str,
    warmup: int,
    n_runs: int,
    max_tokens: int,
    toka_installed: bool,
) -> list[dict]:
    """Start server for engine, run benchmark, stop server. toka_installed: skip pip install if True."""
    if engine == "toka":
        if not toka_installed:
            print("Installing Tokasaurus (pip install -e)... may take 3–5 min.", flush=True)
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-e", f"{MOUNT_PATH}/tokasaurus"],
                check=True,
            )
            print("Tokasaurus installed. Starting server...", flush=True)
        toka_env = {**os.environ}
        if not toka_env.get("CUDA_HOME"):
            try:
                out = subprocess.run(
                    ["which", "nvcc"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if out.returncode == 0 and out.stdout.strip():
                    nvcc = out.stdout.strip()
                    cuda_home = str(Path(nvcc).resolve().parent.parent)
                    if (Path(cuda_home) / "bin" / "nvcc").exists():
                        toka_env["CUDA_HOME"] = cuda_home
                        print(f"Set CUDA_HOME={cuda_home} from nvcc", flush=True)
            except Exception:
                pass
            if not toka_env.get("CUDA_HOME"):
                for path in ("/usr/local/cuda", "/usr/lib/cuda"):
                    if os.path.isdir(path):
                        toka_env["CUDA_HOME"] = path
                        break
        # If still no CUDA_HOME (e.g. Modal image has driver but no toolkit), disable cudagraphs
        # so flashinfer doesn't JIT-compile (avoids CUDA_HOME); inference still works, slightly slower.
        toka_cmd = [sys.executable, "-m", "tokasaurus.entry", f"model={model}"]
        if not toka_env.get("CUDA_HOME"):
            toka_cmd.append("use_cudagraphs=False")
            print("CUDA_HOME not found; starting Tokasaurus with use_cudagraphs=False", flush=True)
        proc = subprocess.Popen(
            toka_cmd,
            cwd=f"{MOUNT_PATH}/tokasaurus",
            env=toka_env,
        )
        port, engine_name, stream = TOKA_PORT, "Tokasaurus", False
    elif engine == "vllm":
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--model",
                model,
                "--host",
                "127.0.0.1",
                "--port",
                str(VLLM_PORT),
                "--dtype",
                "bfloat16",
                "--max-model-len",
                "8192",
            ],
            env={**os.environ},
        )
        port, engine_name, stream = VLLM_PORT, "vLLM", True
    else:
        # SGLang: set CUDA_HOME so Flashinfer can JIT for CUDA graphs (image has nvcc from CUDA devel base).
        sglang_env = {**os.environ}
        if not sglang_env.get("CUDA_HOME"):
            try:
                out = subprocess.run(
                    ["which", "nvcc"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if out.returncode == 0 and out.stdout.strip():
                    nvcc = out.stdout.strip()
                    cuda_home = str(Path(nvcc).resolve().parent.parent)
                    if (Path(cuda_home) / "bin" / "nvcc").exists():
                        sglang_env["CUDA_HOME"] = cuda_home
            except Exception:
                pass
            if not sglang_env.get("CUDA_HOME"):
                for path in ("/usr/local/cuda", "/usr/lib/cuda"):
                    if os.path.isdir(path):
                        sglang_env["CUDA_HOME"] = path
                        break
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "sglang.launch_server",
                "--model-path",
                model,
                "--host",
                "127.0.0.1",
                "--port",
                str(SGLANG_PORT),
                "--tp",
                "1",
            ],
            env=sglang_env,
        )
        port, engine_name, stream = SGLANG_PORT, "SGLang", True

    base_url = f"http://127.0.0.1:{port}/v1"
    # First run: model download + load can take 5–10 min for any engine.
    wait_retries = 300  # 10 min (300 * 2s) for all engines
    print(f"Waiting for {engine_name} at {base_url} (up to ~{wait_retries * 2 // 60} min)...", flush=True)
    if not _wait_for_server(base_url, model, max_retries=wait_retries, sleep_s=2.0):
        proc.terminate()
        raise RuntimeError(f"{engine_name} server did not become ready. Check Modal App Logs for server output.")
    print(f"{engine_name} ready. Running benchmark (warmup + {n_runs} runs per scenario)...", flush=True)

    try:
        from run_single_gpu_inference import run_benchmark

        return run_benchmark(
            base_url,
            model,
            engine_name,
            stream=stream,
            with_gpu=True,
            warmup_requests=warmup,
            n_runs=n_runs,
            max_tokens=max_tokens,
            temperature=0.0,
        )
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


@app.function(image=image_toka, gpu=GPU_TYPE, timeout=2400)  # 40 min: pip install + model download + server + benchmark
def run_benchmark_toka(
    model: str = DEFAULT_MODEL,
    warmup: int = 10,
    n_runs: int = 5,
    max_tokens: int = 128,
) -> list[dict]:
    """Run benchmark for Tokasaurus only."""
    sys.path.insert(0, MOUNT_PATH)
    os.chdir(MOUNT_PATH)
    return _run_engine_benchmark("toka", model, warmup, n_runs, max_tokens, toka_installed=False)


@app.function(image=image_vllm, gpu=GPU_TYPE, timeout=1800)  # 30 min: model download + server + benchmark
def run_benchmark_vllm(
    model: str = DEFAULT_MODEL,
    warmup: int = 10,
    n_runs: int = 5,
    max_tokens: int = 128,
) -> list[dict]:
    """Run benchmark for vLLM only."""
    sys.path.insert(0, MOUNT_PATH)
    os.chdir(MOUNT_PATH)
    return _run_engine_benchmark("vllm", model, warmup, n_runs, max_tokens, toka_installed=True)


@app.function(image=image_sglang, gpu=GPU_TYPE, timeout=1800)  # 30 min: model download + server + benchmark
def run_benchmark_sglang(
    model: str = DEFAULT_MODEL,
    warmup: int = 10,
    n_runs: int = 5,
    max_tokens: int = 128,
) -> list[dict]:
    """Run benchmark for SGLang only."""
    sys.path.insert(0, MOUNT_PATH)
    os.chdir(MOUNT_PATH)
    return _run_engine_benchmark("sglang", model, warmup, n_runs, max_tokens, toka_installed=True)


def _run_one_engine_remote(
    engine: str,
    model: str,
    warmup: int,
    n_runs: int,
    max_tokens: int,
) -> list[dict]:
    """Dispatch to the correct per-engine function (each has its own image)."""
    if engine == "toka":
        return run_benchmark_toka.remote(model=model, warmup=warmup, n_runs=n_runs, max_tokens=max_tokens)
    if engine == "vllm":
        return run_benchmark_vllm.remote(model=model, warmup=warmup, n_runs=n_runs, max_tokens=max_tokens)
    return run_benchmark_sglang.remote(model=model, warmup=warmup, n_runs=n_runs, max_tokens=max_tokens)


def _save_benchmark_md(rows: list[dict], config: dict, output_dir: Path) -> Path:
    """Write benchmark results and flags to a timestamped .md file. Returns path to saved file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%SZ")
    path = output_dir / f"benchmark_{ts}.md"

    lines = [
        "# Single-GPU inference benchmark results",
        "",
        f"**Run:** {ts}",
        "",
        "## Config (flags / settings)",
        "",
        "| Option | Value |",
        "|--------|-------|",
    ]
    for k, v in config.items():
        lines.append(f"| {k} | {v} |")
    lines.extend(["", "## Results", ""])
    lines.append("*(Prefill/decode show \"-\" for non-streaming engines. All engines report tokens/sec, total time, VRAM, and GPU util.)*")
    lines.append("")

    if not rows:
        lines.append("(no rows)")
    else:
        def _cell(v):
            return "-" if v is None else str(v)
        # Use union of keys from all rows so no column is missing (e.g. when mixing engines).
        all_keys = set()
        for r in rows:
            all_keys.update(k for k in r.keys() if k != "scenario")
        # Prefer column order: engine, streaming, then main metrics, then prefill/decode details.
        preferred_order = [
            "engine", "streaming",
            "tokens_per_sec_mean", "tokens_per_sec_p50", "tokens_per_sec_p95",
            "total_time_s_mean", "total_time_s_p95",
            "peak_vram_mb_mean", "avg_gpu_util_pct_mean",
            "ttft_mean", "ttft_p50", "ttft_p95",
            "prefill_s_mean", "prefill_s_p50", "prefill_s_p95",
            "decode_s_mean", "decode_s_p50", "decode_s_p95",
            "decode_tok_s_mean", "decode_tok_s_p50", "decode_tok_s_p95",
        ]
        keys = [k for k in preferred_order if k in all_keys]
        keys += sorted(all_keys - set(keys))
        engine_order = ["Tokasaurus", "vLLM", "SGLang"]
        scenario_order = ["short", "long", "multi_turn_chat"]
        by_scenario: dict[str, list[dict]] = {}
        for r in rows:
            sc = r.get("scenario", "")
            by_scenario.setdefault(sc, []).append(r)
        def _esc(s: str) -> str:
            return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        for scenario in scenario_order:
            if scenario not in by_scenario:
                continue
            scenario_rows = by_scenario[scenario]
            scenario_rows.sort(key=lambda r: (engine_order.index(r.get("engine", "")) if r.get("engine") in engine_order else 99))
            # Transposed: rows = metrics, columns = engines (narrow table, no hidden columns)
            engines = [r.get("engine", "") for r in scenario_rows]
            lines.append(f"### {scenario.replace('_', ' ').title()}")
            lines.append("")
            lines.append("<table>")
            lines.append("<thead><tr><th>Metric</th>" + "".join(f"<th>{_esc(e)}</th>" for e in engines) + "</tr></thead>")
            lines.append("<tbody>")
            for k in keys:
                cells = [_esc(_cell(r.get(k))) for r in scenario_rows]
                lines.append("<tr><td>" + _esc(k) + "</td>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
            lines.append("</tbody></table>")
            lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


@app.local_entrypoint()
def main(
    engine: str = "",
    model: str = DEFAULT_MODEL,
    warmup: int = 10,
    n_runs: int = 5,
    max_tokens: int = 128,
) -> None:
    """Entrypoint: run one engine or all three and print the results table."""
    failed: list[tuple[str, BaseException]] = []
    if engine and engine not in ("toka", "vllm", "sglang"):
        print("--engine must be toka, vllm, or sglang (or omit to run all three)")
        sys.exit(1)

    if engine:
        rows = _run_one_engine_remote(engine, model, warmup, n_runs, max_tokens)
    else:
        # Run all three engines in parallel (each in its own container).
        # Collect per-engine so a single failure doesn't discard others; save partial results.
        kw = dict(model=model, warmup=warmup, n_runs=n_runs, max_tokens=max_tokens)
        rows = []
        failed = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
            fut_toka = ex.submit(run_benchmark_toka.remote, **kw)
            fut_vllm = ex.submit(run_benchmark_vllm.remote, **kw)
            fut_sglang = ex.submit(run_benchmark_sglang.remote, **kw)
            for name, fut in [("Tokasaurus", fut_toka), ("vLLM", fut_vllm), ("SGLang", fut_sglang)]:
                try:
                    rows.extend(fut.result())
                except Exception as e:
                    failed.append((name, e))
                    print(f"{name} failed: {e}", flush=True)
        if failed:
            print(f"Completed: {[r.get('engine') for r in rows]}; failed: {[f[0] for f in failed]}", flush=True)
        # Keep consistent order: toka, vllm, sglang
        order = {"Tokasaurus": 0, "vLLM": 1, "SGLang": 2}
        rows.sort(key=lambda r: (order.get(r.get("engine", ""), 99), r.get("scenario", "")))

    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        df = df.fillna("-")  # Tokasaurus (non-streaming) has no prefill/decode; show "-" instead of None
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", None)
        print(df.to_string())
    except ImportError:
        for row in rows:
            print(row)

    # Save results to a timestamped .md file with config/flags (even if only some engines completed)
    engines_desc = engine if engine else ("toka, vllm, sglang (parallel)" if not failed else f"completed: {list(dict.fromkeys(r.get('engine') for r in rows))}; failed: {[f[0] for f in failed]}")
    config = {
        "engines": engines_desc,
        "model": model,
        "warmup": str(warmup),
        "n_runs": str(n_runs),
        "max_tokens": str(max_tokens),
        "gpu (Modal)": GPU_TYPE,
        "platform": "Modal",
    }
    out_path = _save_benchmark_md(rows, config, BENCHMARK_DIR)
    print(f"\nResults saved to: {out_path}")
