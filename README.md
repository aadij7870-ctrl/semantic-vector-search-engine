# Semantic Vector Search Engine

A lightweight, persistent semantic vector search engine built with FastAPI, Sentence-Transformers, and NumPy.

## Features
- **Dense Vector Search**: Powered by all-MiniLM-L6-v2 dense embeddings and cosine similarity.
- **Sub-25ms Latency**: Fast dot-product vector querying over normalized document spaces.
- **State Persistence**: Serializes index and vector arrays to disk across service restarts.
- **RESTful API**: Document batch indexing and semantic querying endpoints via FastAPI and Pydantic validation.

## Performance Benchmark
- **Median Latency (P50)**: ~16.2 ms
- **Mean Latency**: ~22.3 ms
- **Success Rate**: 100% under concurrent loads

## Setup & Running
```bash
python3 -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
python3 benchmark.py


