import os
import time
import asyncio
import httpx
import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = os.environ.get("MODEL_NAME", "mlsysops-cms-model")
STAGE = os.environ.get("MODEL_STAGE", "staging").lower()
APP_URL = os.environ.get("APP_URL", "http://mlsysops-cms-app.mlsysops-cms-staging.svc.cluster.local:8000")
CONCURRENCY = int(os.environ.get("CONCURRENCY", 50))
DURATION = int(os.environ.get("DURATION", 30))  # seconds
LATENCY_P95_THRESHOLD = float(os.environ.get("P95_LATENCY_THRESHOLD", 0.5))  # seconds
SUCCESS_RATE_THRESHOLD = float(os.environ.get("SUCCESS_RATE_THRESHOLD", 0.99))
TARGET_THROUGHPUT = int(os.environ.get("THROUGHPUT_THRESHOLD", 20))  # req/sec

EXAMPLES = [
    "you are amazing",
    "you are disgusting",
    "everyone loves this",
    "what an idiot",
    "you're a kind person",
    "i hate you",
    "i love this",
    "this is terrible",
    "you're brilliant",
    "how dare you",
]

client = MlflowClient()

def get_staging_version():
    vinfo = client.get_model_version_by_alias(MODEL_NAME, STAGE)
    return vinfo.version

async def wait_for_model_ready():
    expected_version = get_staging_version()
    print(f"[INFO] Waiting for /healthz to return version {expected_version}...")
    while True:
        try:
            async with httpx.AsyncClient(timeout=3) as client:
                resp = await client.get(f"{APP_URL}/healthz")
                if resp.status_code == 200 and str(resp.json().get("model_version")) == str(expected_version):
                    print("[READY] Model is now live in staging")
                    return expected_version
        except Exception as e:
            print(f"[WARN] Health check failed: {e}")
        await asyncio.sleep(5)

async def worker(stats, stop_event):
    async with httpx.AsyncClient(timeout=10) as client:
        while not stop_event.is_set():
            text = {"text": EXAMPLES[time.time_ns() % len(EXAMPLES)]}
            start = time.perf_counter()
            try:
                resp = await client.post(f"{APP_URL}/predict", json=text)
                latency = time.perf_counter() - start
                stats["latencies"].append(latency)
                stats["successes"] += int(resp.status_code == 200)
                stats["total"] += 1

                if resp.status_code != 200:
                    print(f"[DEBUG] Failed response: {resp.status_code} | {resp.text}")
                    
            except Exception:
                stats["failures"] += 1
                stats["total"] += 1

async def run_load_test():
    print(f"[INFO] Starting load test: {CONCURRENCY} users for {DURATION}s")
    stats = {"latencies": [], "successes": 0, "failures": 0, "total": 0}
    stop_event = asyncio.Event()
    workers = [worker(stats, stop_event) for _ in range(CONCURRENCY)]
    task_group = asyncio.gather(*workers)
    await asyncio.sleep(DURATION)
    stop_event.set()
    await task_group
    return stats

def evaluate(stats):
    latencies = sorted(stats["latencies"])
    p95 = latencies[int(len(latencies) * 0.95)] if latencies else float("inf")
    rps = stats["total"] / DURATION
    success_rate = stats["successes"] / stats["total"] if stats["total"] > 0 else 0.0

    print(f"\n--- Load Test Summary ---")
    print(f"Total requests: {stats['total']}")
    print(f"Successes: {stats['successes']}, Failures: {stats['failures']}")
    print(f"Throughput: {rps:.2f} req/s")
    print(f"P95 latency: {p95:.3f}s")
    print(f"Success rate: {success_rate:.2%}")

    passed = (p95 <= LATENCY_P95_THRESHOLD and
              success_rate >= SUCCESS_RATE_THRESHOLD and
              rps >= TARGET_THROUGHPUT)
    print("Result:", "PASS" if passed else "FAIL")

    mlflow.log_metric("loadtest_throughput_rps", rps)
    mlflow.log_metric("loadtest_p95_latency", p95)
    mlflow.log_metric("loadtest_success_rate", success_rate)
    return passed

def promote_to_canary(version):
    print(f"[PROMOTE] Promoting version {version} to 'canary'")
    client.transition_model_version_stage(MODEL_NAME, version, stage="Canary", archive_existing_versions=False)
    client.set_registered_model_alias(MODEL_NAME, "canary", version)

if __name__ == "__main__":
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:8000"))
    with mlflow.start_run(run_name="load-test"):
        version = asyncio.run(wait_for_model_ready())
        stats = asyncio.run(run_load_test())
        if evaluate(stats):
            promote_to_canary(version)
