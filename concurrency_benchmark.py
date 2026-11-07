#!/usr/bin/env python3
"""Small, dependency-free concurrency baseline for the local MLX-LM API.

Reports request latency, aggregate output throughput and HTTP errors.
Non-streaming responses cannot reveal true time-to-first-token (TTFT).
"""

import argparse
import concurrent.futures
import json
import statistics
import time
import urllib.error
import urllib.request


def percentile(values, percentage):
    if not values:
        return float("nan")
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentage / 100
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def request_one(index, url, model, max_tokens, timeout):
    payload = json.dumps({
        "model": model,
        "messages": [{
            "role": "user",
            "content": (
                "Explain the differences between Kubernetes Deployments, "
                "StatefulSets and DaemonSets in five short bullet points. "
                f"Test case {index}. /no_think"
            ),
        }],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = json.load(response)
        elapsed = time.perf_counter() - start
        usage = body.get("usage", {})
        return {
            "success": True,
            "latency": elapsed,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "finish_reason": body["choices"][0].get("finish_reason"),
        }
    except (urllib.error.URLError, ValueError, KeyError, TimeoutError) as error:
        return {"success": False, "latency": time.perf_counter() - start, "error": str(error)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080/v1/chat/completions")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-MLX-4bit")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 5])
    parser.add_argument("--requests", type=int, default=10, help="Total requests at each concurrency level")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--output", default="concurrency_results.jsonl")
    args = parser.parse_args()
    if args.requests < 1 or any(c < 1 for c in args.concurrency):
        parser.error("--requests and all concurrency levels must be positive")

    print("Concurrency | Success/total | P50 latency | P95 latency | Output tok/s | Req/s")
    print("------------|---------------|-------------|-------------|--------------|------")
    with open(args.output, "a", encoding="utf-8") as output:
        for concurrency in args.concurrency:
            start = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = [
                    pool.submit(request_one, i, args.url, args.model, args.max_tokens, args.timeout)
                    for i in range(args.requests)
                ]
                responses = [future.result() for future in futures]
            wall = time.perf_counter() - start
            successes = [r for r in responses if r["success"]]
            latencies = [r["latency"] for r in successes]
            total_tokens = sum(r["completion_tokens"] for r in successes)
            summary = {
                "timestamp": time.time(),
                "model": args.model,
                "concurrency": concurrency,
                "requests": args.requests,
                "successes": len(successes),
                "wall_seconds": wall,
                "p50_latency_seconds": percentile(latencies, 50),
                "p95_latency_seconds": percentile(latencies, 95),
                "mean_latency_seconds": statistics.mean(latencies) if latencies else None,
                "output_tokens_per_second": total_tokens / wall,
                "requests_per_second": len(successes) / wall,
                "total_output_tokens": total_tokens,
                "errors": [r["error"] for r in responses if not r["success"]],
            }
            output.write(json.dumps(summary) + "\n")
            output.flush()
            print(
                f"{concurrency:>11} | {len(successes):>3}/{args.requests:<9} | "
                f"{summary['p50_latency_seconds']:>10.2f}s | "
                f"{summary['p95_latency_seconds']:>10.2f}s | "
                f"{summary['output_tokens_per_second']:>11.2f} | "
                f"{summary['requests_per_second']:>5.2f}"
            )
            if summary["errors"]:
                print("  Errors:", summary["errors"][:3])
    print(f"Saved summaries to {args.output}")
    print("NOTE: These are non-streaming end-to-end latency measurements, NOT TTFT.")


if __name__ == "__main__":
    main()
