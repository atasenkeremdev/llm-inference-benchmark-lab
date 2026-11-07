#!/usr/bin/env python3
"""Run MLX-LM benchmarks and export the latest measured results to Prometheus.

Usage:
    python mlx_exporter.py serve --host <your-mac-lan-ip>
    python mlx_exporter.py benchmark -p 1024 -g 128

Historical trials live in benchmark_results.jsonl. A Prometheus scrape returns
the newest completed run for each model/prompt/output combination; it does NOT
run inference on each scrape.
"""

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

from prometheus_client import CollectorRegistry, start_http_server
from prometheus_client.core import GaugeMetricFamily


MODEL = "Qwen/Qwen3-4B-MLX-4bit"
RESULTS = Path(__file__).resolve().parent / "benchmark_results.jsonl"
CONCURRENCY_RESULTS = Path(__file__).resolve().parent / "concurrency_results.jsonl"
LABELS = ["model", "prompt_tokens", "generation_tokens"]


class BenchmarkCollector:
    """Expose latest persisted benchmark measurements on every scrape."""

    def collect(self):
        definitions = {
            "prefill": (
                "mlx_prefill_tokens_per_second",
                "Measured prompt processing throughput (tokens/second)",
            ),
            "generation": (
                "mlx_generation_tokens_per_second",
                "Measured output generation throughput (tokens/second)",
            ),
            "memory": (
                "mlx_peak_memory_gb",
                "Measured MLX peak memory (GB, as reported by mlx_lm.benchmark)",
            ),
            "timestamp": (
                "mlx_benchmark_last_run_timestamp_seconds",
                "Unix time of last completed benchmark",
            ),
            "prefill_duration": (
                "mlx_prefill_duration_seconds",
                "Estimated prefill duration from prompt tokens / prompt throughput",
            ),
            "decode_duration": (
                "mlx_decode_duration_seconds",
                "Estimated decode duration from generated tokens / generation throughput",
            ),
            "total_latency": (
                "mlx_total_latency_seconds",
                "Measured average total_time of benchmark trials (seconds)",
            ),
            "estimated_ttft": (
                "mlx_estimated_ttft_seconds",
                "Estimated TTFT: prefill duration plus one average decode token; not measured TTFT",
            ),
        }
        metrics = {
            key: GaugeMetricFamily(name, description, labels=LABELS)
            for key, (name, description) in definitions.items()
        }

        latest = {}
        if RESULTS.exists():
            with RESULTS.open(encoding="utf-8") as source:
                for line in source:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    key = (
                        record["model"],
                        int(record["prompt_tokens"]),
                        int(record["generation_tokens"]),
                    )
                    latest[key] = record

        for record in latest.values():
            labels = [
                record["model"],
                str(record["prompt_tokens"]),
                str(record["generation_tokens"]),
            ]
            for key, metric in metrics.items():
                # Older JSONL rows do not have the newly added latency fields.
                if key in record:
                    metric.add_metric(labels, record[key])

        yield from metrics.values()


class ConcurrencyCollector:
    """Expose the most recent non-streaming load test for each concurrency."""

    def collect(self):
        definitions = {
            "p50_latency_seconds": ("mlx_concurrency_p50_latency_seconds", "P50 request latency in seconds"),
            "p95_latency_seconds": ("mlx_concurrency_p95_latency_seconds", "P95 request latency in seconds"),
            "output_tokens_per_second": ("mlx_concurrency_output_tokens_per_second", "Aggregate output token throughput per wall second"),
            "requests_per_second": ("mlx_concurrency_requests_per_second", "Successful requests per wall second"),
            "successes": ("mlx_concurrency_successes", "Number of successful requests"),
            "requests": ("mlx_concurrency_requests", "Number of attempted requests"),
            "wall_seconds": ("mlx_concurrency_wall_seconds", "Whole test wall duration"),
            "timestamp": ("mlx_concurrency_last_run_timestamp_seconds", "Timestamp of completed load test"),
        }
        families = {
            k: GaugeMetricFamily(name, help_text, labels=["model", "concurrency"])
            for k, (name, help_text) in definitions.items()
        }
        latest = {}
        if CONCURRENCY_RESULTS.exists():
            with CONCURRENCY_RESULTS.open(encoding="utf-8") as source:
                for line in source:
                    if line.strip():
                        record = json.loads(line)
                        latest[(record["model"], int(record["concurrency"]))] = record
        for record in latest.values():
            labels = [record["model"], str(record["concurrency"])]
            for key, family in families.items():
                if record.get(key) is not None:
                    family.add_metric(labels, record[key])
        yield from families.values()


def run_benchmark(prompt_tokens: int, generation_tokens: int):
    if prompt_tokens <= 0 or generation_tokens <= 0:
        raise ValueError("Token counts must be positive")

    command = [
        "mlx_lm.benchmark",
        "--model", MODEL,
        "-p", str(prompt_tokens),
        "-g", str(generation_tokens),
    ]
    print("Running:", " ".join(command), flush=True)
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout, end="", flush=True)
    if result.returncode != 0:
        print(result.stderr, end="", flush=True)
        raise RuntimeError(f"Benchmark failed with exit code {result.returncode}")

    averages = re.search(
        r"Averages:\s*prompt_tps=([\d.]+),\s*"
        r"generation_tps=([\d.]+),\s*peak_memory=([\d.]+)",
        result.stdout,
    )
    trial_times = [
        float(value)
        for value in re.findall(
            r"Trial\s+\d+:.*?total_time=([\d.]+)", result.stdout
        )
    ]
    if averages is None or not trial_times:
        raise RuntimeError("Cannot parse benchmark averages or trial durations")

    prefill_tps = float(averages.group(1))
    generation_tps = float(averages.group(2))
    if prefill_tps <= 0 or generation_tps <= 0:
        raise RuntimeError("Invalid measured throughput")

    prefill_seconds = prompt_tokens / prefill_tps
    decode_seconds = generation_tokens / generation_tps
    record = {
        "model": MODEL,
        "prompt_tokens": prompt_tokens,
        "generation_tokens": generation_tokens,
        "prefill": prefill_tps,
        "generation": generation_tps,
        "memory": float(averages.group(3)),
        "timestamp": time.time(),
        "prefill_duration": prefill_seconds,
        "decode_duration": decode_seconds,
        "total_latency": sum(trial_times) / len(trial_times),
        "estimated_ttft": prefill_seconds + 1.0 / generation_tps,
    }

    with RESULTS.open("a", encoding="utf-8") as destination:
        destination.write(json.dumps(record, ensure_ascii=False) + "\n")
    print("Benchmark saved successfully.")
    print(
        "NOTE: TTFT is estimated, not observed from a streaming token event."
    )


def serve(host: str, port: int):
    registry = CollectorRegistry()
    registry.register(BenchmarkCollector())
    registry.register(ConcurrencyCollector())
    start_http_server(port, addr=host, registry=registry)
    print(f"MLX exporter listening on http://{host}:{port}/metrics", flush=True)
    while True:
        time.sleep(60)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["serve", "benchmark"])
    parser.add_argument("-p", type=int, default=2048, help="Prompt token count")
    parser.add_argument("-g", type=int, default=128, help="Generation token count")
    parser.add_argument("--host", default="127.0.0.1", help="Exporter bind IP; set your LAN IP for remote Prometheus")
    parser.add_argument("--port", type=int, default=9108, help="Exporter port")
    args = parser.parse_args()
    if args.action == "serve":
        serve(args.host, args.port)
    else:
        run_benchmark(args.p, args.g)


if __name__ == "__main__":
    main()
