import time
import requests
import statistics

BASE_URL = "http://127.0.0.1:8000"

def run_benchmark():
    # Use a unique query with a timestamp to guarantee a true Cache Miss
    unique_query = f"reinforcement learning robotics system {int(time.time())}"
    query_payload = {"query": unique_query, "top_k": 2}
    
    # 1. True Cache Miss (Model Vector Inference + Similarity Search)
    t0 = time.perf_counter()
    r = requests.post(f"{BASE_URL}/search", json=query_payload)
    t1 = time.perf_counter()
    miss_data = r.json()
    miss_internal_ms = miss_data.get("latency_ms", 0.0)
    miss_e2e_ms = (t1 - t0) * 1000
    
    print("=" * 55)
    print("           L1 CACHE ACCELERATION BENCHMARK           ")
    print("=" * 55)
    print(f"Cold Query: \"{unique_query}\"")
    print(f"Cache Hit Flag:           {miss_data.get('cache_hit')}")
    print(f"Cold Engine Latency:      {miss_internal_ms:.3f} ms")
    print(f"Cold Total HTTP Latency:  {miss_e2e_ms:.2f} ms")
    print("-" * 55)

    # 2. 50 Repeated Requests for the EXACT same query (Pure Cache Hits)
    engine_latencies = []
    e2e_latencies = []
    
    for _ in range(50):
        start = time.perf_counter()
        res = requests.post(f"{BASE_URL}/search", json=query_payload)
        end = time.perf_counter()
        
        data = res.json()
        engine_latencies.append(data.get("latency_ms", 0.0))
        e2e_latencies.append((end - start) * 1000)

    median_engine = statistics.median(engine_latencies)
    min_engine = min(engine_latencies)
    median_e2e = statistics.median(e2e_latencies)
    min_e2e = min(e2e_latencies)
    speedup = miss_internal_ms / max(median_engine, 0.001)

    print("--- 50 Repeated Warm Queries (C++ L1 Hits) ---")
    print(f"Engine Latency P50 (Hit): {median_engine:.3f} ms")
    print(f"Engine Latency Min (Hit): {min_engine:.3f} ms")
    print(f"Total HTTP Latency P50:   {median_e2e:.2f} ms")
    print(f"Total HTTP Latency Min:   {min_e2e:.2f} ms")
    print("-" * 55)
    print(f"Cache Acceleration:       {speedup:.1f}x faster")
    print("=" * 55)

if __name__ == "__main__":
    run_benchmark()
