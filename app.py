import time
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel

from vector_engine.model import EmbeddingModel
from vector_engine.index import VectorIndex

app = FastAPI(title="Semantic Vector Search Engine", version="1.0.0")

model = EmbeddingModel()
index = VectorIndex()

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
    total_indexed: int
    results: List[Dict[str, Any]]

@app.get("/health")
def health_check():
    return {"status": "healthy", "indexed_documents": index.count()}

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
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Search query cannot be empty.")
    
    start_time = time.perf_counter()
    query_vector = model.encode(payload.query)
    hits = index.search(query_vector=query_vector, top_k=payload.top_k)
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    return SearchResponse(
        query=payload.query,
        latency_ms=round(elapsed_ms, 3),
        total_indexed=index.count(),
        results=hits
    )
