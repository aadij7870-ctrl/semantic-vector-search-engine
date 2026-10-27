import time
import socket
import json
import base64
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel

from vector_engine.model import EmbeddingModel
from vector_engine.index import VectorIndex

app = FastAPI(title="Semantic Vector Search Engine with C++ L1 Cache", version="1.2.0")

model = EmbeddingModel()
index = VectorIndex()

CACHE_HOST = "127.0.0.1"
CACHE_PORT = 8080

def query_cpp_cache(command: str) -> Optional[str]:
    """Communicate directly with Project 1 C++ cache over TCP socket."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.1)
            s.connect((CACHE_HOST, CACHE_PORT))
            s.sendall((command.strip() + "\n").encode())
            
            # Read complete response until newline delimiter
            response = ""
            while True:
                chunk = s.recv(4096).decode()
                if not chunk:
                    break
                response += chunk
                if "\n" in chunk:
                    break
            return response.strip()
    except Exception:
        return None

class DocumentItem(BaseModel):
    id: str
    text: str
    metadata: Optional[Dict[str, Any]] = {}

class IndexRequest(BaseModel):
    documents: List[DocumentItem]

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

class SearchResponse(BaseModel):
    query: str
    latency_ms: float
    cache_hit: bool
    total_indexed: int
    results: List[Dict[str, Any]]

@app.get("/health")
def health_check():
    cache_status = query_cpp_cache("PING")
    return {
        "status": "healthy",
        "indexed_documents": index.count(),
        "cpp_cache_connected": cache_status is not None
    }

@app.post("/index", status_code=status.HTTP_201_CREATED)
def index_documents(payload: IndexRequest):
    if not payload.documents:
        raise HTTPException(status_code=400, detail="Document list cannot be empty.")
    
    docs = [doc.model_dump() for doc in payload.documents]
    texts = [doc["text"] for doc in docs]
    
    embeddings = model.encode(texts)
    index.add_documents(docs, embeddings)
    
    return {"status": "success", "added": len(docs), "total_indexed": index.count()}

@app.post("/search", response_model=SearchResponse)
def search(payload: SearchRequest):
    query_str = payload.query.strip()
    if not query_str:
        raise HTTPException(status_code=400, detail="Search query cannot be empty.")
    
    start_time = time.perf_counter()
    cache_key = f"query:{query_str}:{payload.top_k}".replace(" ", "_")

    # 1. Attempt L1 Cache Lookup in C++ Server
    cached_val = query_cpp_cache(f"GET {cache_key}")
    if cached_val and not cached_val.startswith("NOT_FOUND") and not cached_val.startswith("ERROR"):
        try:
            # Decode Base64 string back into JSON array
            decoded_json = base64.b64decode(cached_val.encode()).decode()
            results = json.loads(decoded_json)
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return SearchResponse(
                query=query_str,
                latency_ms=round(elapsed_ms, 3),
                cache_hit=True,
                total_indexed=index.count(),
                results=results
            )
        except Exception:
            pass

    # 2. Cache Miss: Compute Vector Embeddings & Similarity Search
    query_vector = model.encode(query_str)
    hits = index.search(query_vector=query_vector, top_k=payload.top_k)

    # 3. Serialize and write back to C++ Cache (Base64 encoded)
    try:
        raw_json = json.dumps(hits)
        encoded_val = base64.b64encode(raw_json.encode()).decode()
        query_cpp_cache(f"SET {cache_key} {encoded_val}")
    except Exception:
        pass

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    return SearchResponse(
        query=query_str,
        latency_ms=round(elapsed_ms, 3),
        cache_hit=False,
        total_indexed=index.count(),
        results=hits
    )
