#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# Optional: load secrets from secrets.env
try:
    from load_secrets import load_secrets
    load_secrets()
except ImportError:
    pass

from openai import OpenAI

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# -----------------------------------------------------------------------------
# Config (override via env or CLI)
# -----------------------------------------------------------------------------
MODEL = os.environ.get("INFERENCE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
API_HOST = os.environ.get("API_HOST", "localhost")
TOKASAURUS_PORT = int(os.environ.get("TOKASAURUS_PORT", "10210"))
VLLM_PORT = int(os.environ.get("VLLM_PORT", "8000"))
SGLANG_PORT = int(os.environ.get("SGLANG_PORT", "30000"))
DEFAULT_WARMUP = 10
DEFAULT_N_RUNS = 5
DEFAULT_MAX_TOKENS = 128
GPU_POLL_INTERVAL_S = 0.5

BENCHMARK_DIR = Path(__file__).resolve().parent / "benchmark_results"


def save_benchmark_md(rows: list[dict], config: dict, output_dir: Path) -> Path:
    """Write benchmark results and config/flags to a timestamped .md file. Returns path to saved file."""
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
        all_keys = set()
        for r in rows:
            all_keys.update(k for k in r.keys() if k != "scenario")
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


@dataclass
class InferenceResult:
    ttft_s: float | None  # time to first token (prefill stage)
    tokens_per_sec: float  # overall output throughput
    total_time_s: float
    prompt_tokens: int
    completion_tokens: int
    raw_content: str = ""
    prefill_s: float | None = None  # time for prompt processing until first token
    decode_s: float | None = None  # time from first token to last token
    decode_tokens_per_sec: float | None = None  # completion_tokens / decode_s


@dataclass
class GPUSnapshot:
    used_mb: float
    util_pct: float


# -----------------------------------------------------------------------------
# Prompt scenarios
# -----------------------------------------------------------------------------
SHORT_PROMPT = [{"role": "user", "content": "What is 2+2? Answer in one word."}]
LONG_PROMPT = [{
    "role": "user",
    "content": "Summarize the following text in 2 sentences. " + "The quick brown fox jumps over the lazy dog. " * 80,
}]
MULTI_TURN_CHAT = [
    {"role": "system", "content": "You are a helpful assistant. Be concise."},
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."},
    {"role": "user", "content": "What is one famous landmark there?"},
]
CREATIVE_PROMPT = [{"role": "user", "content": "Write a short poem about coding in exactly 4 lines."}]


def run_inference(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int = 128,
    stream: bool = True,
    temperature: float = 0.0,
    api_key: str = "fake-key",
) -> InferenceResult:
    client = OpenAI(base_url=base_url, api_key=api_key, max_retries=0, timeout=60)
    if stream:
        ttft_s: float | None = None
        first_chunk_time: float | None = None
        last_chunk_time: float | None = None
        chunks: list[str] = []
        start = time.perf_counter()
        try:
            stream_obj = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            )
            for chunk in stream_obj:
                t = time.perf_counter()
                if first_chunk_time is None and chunk.choices and chunk.choices[0].delta.content:
                    first_chunk_time = t
                    ttft_s = first_chunk_time - start
                if chunk.choices and chunk.choices[0].delta.content:
                    last_chunk_time = t
                    chunks.append(chunk.choices[0].delta.content)
        except Exception:
            pass
        if first_chunk_time is not None and chunks:
            total_time_s = (last_chunk_time or time.perf_counter()) - start
            content = "".join(chunks)
            completion_tokens = len(chunks)
            tokens_per_sec = completion_tokens / total_time_s if total_time_s > 0 else 0
            # Prefill = time until first token; decode = time from first to last token
            prefill_s = ttft_s
            decode_s = (last_chunk_time - first_chunk_time) if last_chunk_time is not None else None
            decode_tokens_per_sec = (
                completion_tokens / decode_s if (decode_s is not None and decode_s > 0) else None
            )
            return InferenceResult(
                ttft_s=ttft_s,
                tokens_per_sec=tokens_per_sec,
                total_time_s=total_time_s,
                prompt_tokens=0,
                completion_tokens=completion_tokens,
                raw_content=content,
                prefill_s=prefill_s,
                decode_s=decode_s,
                decode_tokens_per_sec=decode_tokens_per_sec,
            )
    start = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=False,
    )
    total_time_s = time.perf_counter() - start
    content = resp.choices[0].message.content or ""
    usage = getattr(resp, "usage", None)
    prompt_tokens = usage.prompt_tokens if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0
    tokens_per_sec = completion_tokens / total_time_s if total_time_s > 0 else 0
    return InferenceResult(
        ttft_s=None,
        tokens_per_sec=tokens_per_sec,
        total_time_s=total_time_s,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        raw_content=content,
        prefill_s=None,
        decode_s=None,
        decode_tokens_per_sec=None,
    )


def _poll_nvidia_smi(interval_s: float, stop_event: threading.Event, out_list: list) -> None:
    while not stop_event.is_set():
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if out.returncode == 0 and out.stdout.strip():
                line = out.stdout.strip().split("\n")[0]
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2:
                    used_mb = float(parts[0].split()[0])
                    util = parts[1].strip().replace(" %", "")
                    util_pct = float(util) if util.isdigit() else 0.0
                    out_list.append(GPUSnapshot(used_mb=used_mb, util_pct=util_pct))
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
            pass
        stop_event.wait(interval_s)


def measure_gpu_during_inference(
    inference_fn: Any,
    *args: Any,
    poll_interval_s: float = 0.5,
    **kwargs: Any,
) -> tuple[InferenceResult, float | None, float | None]:
    kwargs_no_poll = {k: v for k, v in kwargs.items() if k != "poll_interval_s"}
    stop_event = threading.Event()
    snapshots: list[GPUSnapshot] = []
    thread = threading.Thread(target=_poll_nvidia_smi, args=(poll_interval_s, stop_event, snapshots))
    thread.start()
    try:
        result = inference_fn(*args, **kwargs_no_poll)
    finally:
        stop_event.set()
        thread.join(timeout=2)
    peak_vram_mb = max(s.used_mb for s in snapshots) if snapshots else None
    avg_util_pct = (sum(s.util_pct for s in snapshots) / len(snapshots)) if snapshots else None
    return result, peak_vram_mb, avg_util_pct


def wait_for_server(base_url: str, model: str, max_retries: int = 30, sleep_s: float = 2.0) -> bool:
    client = OpenAI(base_url=base_url, api_key="fake-key", max_retries=0, timeout=5)
    for i in range(max_retries):
        try:
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=2,
            )
            print(f"Server at {base_url} is ready.")
            return True
        except Exception as e:
            print(f"Waiting for server... ({i + 1}/{max_retries}) {e}")
            time.sleep(sleep_s)
    return False


def _agg_stats(values: list[float]) -> dict[str, float]:
    a = np.array([v for v in values if v is not None])
    if len(a) == 0:
        return {"mean": float("nan"), "p50": float("nan"), "p95": float("nan")}
    return {
        "mean": float(np.mean(a)),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
    }


def run_benchmark(
    base_url: str,
    model: str,
    engine_name: str,
    stream: bool = False,
    with_gpu: bool = True,
    warmup_requests: int = DEFAULT_WARMUP,
    n_runs: int = DEFAULT_N_RUNS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 0.0,
    poll_interval_s: float = GPU_POLL_INTERVAL_S,
) -> list[dict[str, Any]]:
    """Warm-up, then N runs per scenario; aggregate mean/p50/p95."""
    scenarios = [
        ("short", SHORT_PROMPT),
        ("long", LONG_PROMPT),
        ("multi_turn_chat", MULTI_TURN_CHAT),
    ]
    rows: list[dict[str, Any]] = []
    for scenario_name, messages in scenarios:
        for _ in range(warmup_requests):
            run_inference(base_url, model, messages, max_tokens=max_tokens, stream=stream, temperature=temperature)
        ttft_list: list[float] = []
        tps_list: list[float] = []
        total_list: list[float] = []
        prefill_list: list[float] = []
        decode_s_list: list[float] = []
        decode_tps_list: list[float] = []
        vram_list: list[float] = []
        util_list: list[float] = []
        for _ in range(n_runs):
            if with_gpu:
                result, peak_vram_mb, avg_util_pct = measure_gpu_during_inference(
                    run_inference,
                    base_url,
                    model,
                    messages,
                    max_tokens=max_tokens,
                    stream=stream,
                    temperature=temperature,
                    poll_interval_s=poll_interval_s,
                )
            else:
                result = run_inference(
                    base_url, model, messages, max_tokens=max_tokens, stream=stream, temperature=temperature
                )
                peak_vram_mb, avg_util_pct = None, None
            if result.ttft_s is not None:
                ttft_list.append(result.ttft_s)
            if result.prefill_s is not None:
                prefill_list.append(result.prefill_s)
            if result.decode_s is not None:
                decode_s_list.append(result.decode_s)
            if result.decode_tokens_per_sec is not None:
                decode_tps_list.append(result.decode_tokens_per_sec)
            tps_list.append(result.tokens_per_sec)
            total_list.append(result.total_time_s)
            if peak_vram_mb is not None:
                vram_list.append(peak_vram_mb)
            if avg_util_pct is not None:
                util_list.append(avg_util_pct)
        ttft_s = _agg_stats(ttft_list)
        tps_s = _agg_stats(tps_list)
        total_s = _agg_stats(total_list)
        prefill_s = _agg_stats(prefill_list)
        decode_s = _agg_stats(decode_s_list)
        decode_tps_s = _agg_stats(decode_tps_list)
        vram_s = _agg_stats(vram_list)
        util_s = _agg_stats(util_list)
        row: dict[str, Any] = {
            "engine": engine_name,
            "streaming": stream,
            "scenario": scenario_name,
            "ttft_mean": round(ttft_s["mean"], 4) if ttft_list else None,
            "ttft_p50": round(ttft_s["p50"], 4) if ttft_list else None,
            "ttft_p95": round(ttft_s["p95"], 4) if ttft_list else None,
            "prefill_s_mean": round(prefill_s["mean"], 4) if prefill_list else None,
            "prefill_s_p50": round(prefill_s["p50"], 4) if prefill_list else None,
            "prefill_s_p95": round(prefill_s["p95"], 4) if prefill_list else None,
            "decode_s_mean": round(decode_s["mean"], 4) if decode_s_list else None,
            "decode_s_p50": round(decode_s["p50"], 4) if decode_s_list else None,
            "decode_s_p95": round(decode_s["p95"], 4) if decode_s_list else None,
            "decode_tok_s_mean": round(decode_tps_s["mean"], 2) if decode_tps_list else None,
            "decode_tok_s_p50": round(decode_tps_s["p50"], 2) if decode_tps_list else None,
            "decode_tok_s_p95": round(decode_tps_s["p95"], 2) if decode_tps_list else None,
            "tokens_per_sec_mean": round(tps_s["mean"], 2),
            "tokens_per_sec_p50": round(tps_s["p50"], 2),
            "tokens_per_sec_p95": round(tps_s["p95"], 2),
            "total_time_s_mean": round(total_s["mean"], 3),
            "total_time_s_p95": round(total_s["p95"], 3),
            "peak_vram_mb_mean": round(vram_s["mean"], 1) if vram_list else None,
            "avg_gpu_util_pct_mean": round(util_s["mean"], 1) if util_list else None,
        }
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Single-GPU inference benchmark: warm-up + N runs, report mean/p50/p95 (Tokasaurus, vLLM, SGLang)"
    )
    parser.add_argument(
        "--engine",
        choices=["toka", "vllm", "sglang"],
        default="toka",
        help="Which server to benchmark (default: toka)",
    )
    parser.add_argument("--no-gpu", action="store_true", help="Disable nvidia-smi GPU monitoring")
    parser.add_argument("--model", default=MODEL, help=f"Model name (default: {MODEL})")
    parser.add_argument("--host", default=API_HOST, help=f"API host (default: {API_HOST})")
    parser.add_argument("--wait", action="store_true", help="Wait for server to be ready before benchmarking")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP, help=f"Warm-up requests per scenario (default: {DEFAULT_WARMUP})")
    parser.add_argument("--n-runs", type=int, default=DEFAULT_N_RUNS, help=f"Timed runs per scenario for mean/p50/p95 (default: {DEFAULT_N_RUNS})")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help=f"Max completion tokens (default: {DEFAULT_MAX_TOKENS})")
    parser.add_argument("--no-save", action="store_true", help="Do not save results to benchmark_results/*.md")
    parser.add_argument("--output-dir", type=Path, default=BENCHMARK_DIR, help=f"Directory for saved .md (default: {BENCHMARK_DIR})")
    args = parser.parse_args()

    if args.engine == "toka":
        port = TOKASAURUS_PORT
        stream = False
    elif args.engine == "vllm":
        port = VLLM_PORT
        stream = True
    else:
        port = SGLANG_PORT
        stream = True

    base_url = f"http://{args.host}:{port}/v1"
    engine_name = {"toka": "Tokasaurus", "vllm": "vLLM", "sglang": "SGLang"}[args.engine]

    if args.wait and not wait_for_server(base_url, args.model):
        sys.exit(1)

    print(f"Running benchmark: {engine_name} at {base_url} (warmup={args.warmup}, n_runs={args.n_runs}, stream={stream}, with_gpu={not args.no_gpu})")
    try:
        rows = run_benchmark(
            base_url,
            args.model,
            engine_name,
            stream=stream,
            with_gpu=not args.no_gpu,
            warmup_requests=args.warmup,
            n_runs=args.n_runs,
            max_tokens=args.max_tokens,
            temperature=0.0,
            poll_interval_s=GPU_POLL_INTERVAL_S,
        )
    except Exception as e:
        print(f"Benchmark failed: {e}")
        sys.exit(1)

    if HAS_PANDAS:
        df = pd.DataFrame(rows)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", None)
        print(df.to_string())
    else:
        for row in rows:
            print(row)

    if not args.no_save:
        config = {
            "engine": engine_name,
            "model": args.model,
            "host": args.host,
            "warmup": str(args.warmup),
            "n_runs": str(args.n_runs),
            "max_tokens": str(args.max_tokens),
            "with_gpu": str(not args.no_gpu),
            "platform": "local",
        }
        out_path = save_benchmark_md(rows, config, args.output_dir)
        print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
