import time
import requests
import statistics

BASE_URL = "http://127.0.0.1:8000"

def run_benchmark():
    query_payload = {"query": "deep learning neural networks", "top_k": 2}
    
    # 1. Warm-up / Cache Miss Request
    start = time.perf_counter()
    r = requests.post(f"{BASE_URL}/search", json=query_payload)
    miss_latency = (time.perf_counter() - start) * 1000
    miss_data = r.json()
    
    print(f"Cold Request (Cache Miss): {miss_latency:.2f} ms | Internal Engine Latency: {miss_data.get('latency_ms', 0):.2f} ms")
    
    # 2. 50 Repeated Warm Requests (Cache Hits)
    latencies = []
    engine_latencies = []
    
    for _ in range(50):
        t0 = time.perf_counter()
        res = requests.post(f"{BASE_URL}/search", json=query_payload)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)
        engine_latencies.append(res.json().get("latency_ms", 0))

    print("\n--- Cache Hit Benchmark (50 Requests) ---")
    print(f"Engine Internal Latency (P50):  {statistics.median(engine_latencies):.3f} ms")
    print(f"Engine Internal Latency (Min):  {min(engine_latencies):.3f} ms")
    print(f"End-to-End HTTP Latency (P50):  {statistics.median(latencies):.2f} ms")
    print(f"End-to-End HTTP Latency (Min):  {min(latencies):.2f} ms")
    print(f"Speedup Factor:                 {miss_data.get('latency_ms', 1) / max(statistics.median(engine_latencies), 0.001):.1f}x")

if __name__ == "__main__":
    run_benchmark()
