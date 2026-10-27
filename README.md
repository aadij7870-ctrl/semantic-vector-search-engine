# Semantic Vector Search Engine with C++ L1 Cache

A high-performance semantic vector search engine built with FastAPI, Sentence-Transformers, and NumPy, integrated with an external multi-threaded C++ LRU in-memory cache server over TCP sockets.

## Architecture
- **L1 Cache Layer**: Multi-threaded C++ in-memory TCP socket cache (`127.0.0.1:8080`) storing Base64-serialized vector responses.
- **L2 Vector Engine**: Sentence-Transformers (`all-MiniLM-L6-v2`) dense vector embeddings with cosine similarity scoring.
- **Protocol**: Raw TCP socket streaming with newline-delimited commands (`GET`, `SET`, `PING`).
- **Resilience**: Graceful fallback to full semantic inference if the C++ cache is offline.

## Performance Benchmark

Benchmarked using 50 repeated queries across cold vector compute vs. warm in-memory cache hits:

| Metric | Cache Miss (Cold) | Cache Hit (Warm L1) | Acceleration |
|---|---|---|---|
| **Engine Processing Latency (P50)** | **33.58 ms** | **0.43 ms** | **78.2x** |
| **Engine Minimum Latency** | 33.58 ms | 0.27 ms | 124x |
| **End-to-End HTTP Latency (P50)** | ~2.9s (incl. connection warm-up) | 3.20 ms | — |

## Running the System

```bash
# Terminal 1: Launch C++ Cache Server (Project 1)
cd "/mnt/c/Users/ACER/OneDrive/Desktop/Project 1"
./cache_server

# Terminal 2: Launch FastAPI Vector Search Engine (Project 2)
cd "/mnt/c/Users/ACER/OneDrive/Desktop/project 2"
python3 -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# Terminal 3: Run Acceleration Benchmark
python3 benchmark.py
