import requests
import time
import statistics

BASE_URL = "http://localhost:8000"
NUM_REQUESTS = 50

test_query = {"query": "concurrent socket cache benchmarking", "top_k": 3}
latencies = []

print(f"Starting benchmark: {NUM_REQUESTS} requests...")

for i in range(NUM_REQUESTS):
    start = time.perf_counter()
    response = requests.post(f"{BASE_URL}/search", json=test_query)
    elapsed = (time.perf_counter() - start) * 1000

    if response.status_code == 200:
        latencies.append(elapsed)
    else:
        print(f"Request {i} failed with status {response.status_code}")

print("\n--- Benchmark Results ---")
print(f"Completed Requests: {len(latencies)}/{NUM_REQUESTS}")
if latencies:
    print(f"Min Latency:       {min(latencies):.2f} ms")
    print(f"Mean Latency:      {statistics.mean(latencies):.2f} ms")
    print(f"P50 (Median):      {statistics.median(latencies):.2f} ms")
    print(f"P95 Latency:      {statistics.quantiles(latencies, n=20)[18]:.2f} ms")
    print(f"Max Latency:      {max(latencies):.2f} ms")
