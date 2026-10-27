python3 -c '
with open("README.md", "r") as f:
    content = f.read()

diagram = """## System Architecture

```mermaid
flowchart TD
    Client([HTTP Client / Query]) -->|POST /search| API[FastAPI Vector Engine]
    
    subgraph L1_Cache [C++ In-Memory L1 Cache Server]
        SocketServer[POSIX Socket Server :8080]
        MutexEngine[Thread-Safe Sharded Mutex Map]
        SocketServer <--> MutexEngine
    end
    
    API -->|1. Query Cache Hash| SocketServer
    SocketServer -->|Cache Hit: Return Cached Vectors| API
    
    subgraph Core_Engine [Vector Inference Engine]
        Transformer[SentenceTransformer Embeddings]
        FAISS[(FAISS Flat L2 Index)]
        Transformer --> FAISS
    end
    
    SocketServer -.->|Cache Miss| Core_Engine
    Core_Engine -->|Compute & Populate| SocketServer
    
    API -->|JSON Response P50: 0.43ms| Client
    