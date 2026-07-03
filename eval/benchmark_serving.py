"""
benchmark_serving.py
--------------------
Measures serving PERFORMANCE (not answer quality) for one engine:
  - single-request latency  (one prompt, time to full answer)
  - batch throughput        (many prompts at once, tokens/second)

Both Ollama and vLLM expose an OpenAI-compatible API, so the SAME code
benchmarks either — we just point base_url at the right server and pass
which engine we're testing. Run it once per engine (with that engine
running), and it saves results for the dashboard to compare.

Usage:
    python eval/benchmark_serving.py ollama
    python eval/benchmark_serving.py vllm
"""

import sys
import json
import time
from pathlib import Path

from openai import OpenAI

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Same model on both engines = a fair test. Only the address + model-id
# string differ between the two servers.
ENGINES = {
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "model": "qwen2.5:1.5b",
    },
    "vllm": {
        "base_url": "http://localhost:8000/v1",
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
    },
}

# A fixed set of prompts so both engines do identical work.
PROMPTS = [
    "Explain what a transformer is in one sentence.",
    "What is attention in deep learning?",
    "Define a neural network briefly.",
    "What is machine translation?",
    "Explain gradient descent simply.",
    "What is a GPU used for in AI?",
    "Define overfitting in one sentence.",
    "What is a language model?",
] * 2   # 16 prompts


def make_client(cfg):
    return OpenAI(base_url=cfg["base_url"], api_key="not-needed")


def one_call(client, model, prompt):
    """Single request; return (seconds, tokens_generated)."""
    t0 = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=64,
        temperature=0.7,
    )
    elapsed = time.time() - t0
    tokens = resp.usage.completion_tokens
    return elapsed, tokens


def benchmark(engine: str) -> dict:
    cfg = ENGINES[engine]
    client = make_client(cfg)
    model = cfg["model"]

    print(f"\n=== Benchmarking {engine} ({model}) ===")

    # ── Metric 1: single-request latency (average of a few) ──
    print("Measuring single-request latency...")
    lat_times, lat_tokens = [], []
    for p in PROMPTS[:5]:
        s, t = one_call(client, model, p)
        lat_times.append(s)
        lat_tokens.append(t)
    avg_latency = sum(lat_times) / len(lat_times)
    single_tps = sum(lat_tokens) / sum(lat_times)

    # ── Metric 2: batch throughput (all prompts, concurrent) ──
    # We fire all prompts and measure total wall-clock + total tokens.
    # For a true concurrent test we use threads so requests overlap —
    # this is where vLLM's batching should pull ahead of Ollama.
    print(f"Measuring batch throughput ({len(PROMPTS)} concurrent)...")
    import concurrent.futures

    t0 = time.time()
    total_tokens = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(PROMPTS)) as ex:
        futures = [ex.submit(one_call, client, model, p) for p in PROMPTS]
        for f in concurrent.futures.as_completed(futures):
            _, tok = f.result()
            total_tokens += tok
    batch_elapsed = time.time() - t0
    batch_tps = total_tokens / batch_elapsed

    result = {
        "engine": engine,
        "model": model,
        "avg_single_latency_s": round(avg_latency, 3),
        "single_tokens_per_s": round(single_tps, 1),
        "batch_prompts": len(PROMPTS),
        "batch_total_s": round(batch_elapsed, 2),
        "batch_tokens_per_s": round(batch_tps, 1),
    }

    out_path = RESULTS_DIR / f"serving_{engine}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n--- {engine} results ---")
    for k, v in result.items():
        print(f"  {k}: {v}")
    print(f"saved -> {out_path}")
    return result


if __name__ == "__main__":
    engine = sys.argv[1] if len(sys.argv) > 1 else "vllm"
    if engine not in ENGINES:
        print(f"Unknown engine '{engine}'. Use: ollama or vllm")
        sys.exit(1)
    benchmark(engine)
