import socket, json, base64

sample_data = [{"id": "1", "text": "test document", "score": 0.99}]
encoded = base64.b64encode(json.dumps(sample_data).encode()).decode()
key = "test_search_key"

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect(("127.0.0.1", 8080))
    s.sendall(f"SET {key} {encoded}\n".encode())
    print("SET:", repr(s.recv(1024).decode()))

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect(("127.0.0.1", 8080))
    s.sendall(f"GET {key}\n".encode())
    res = s.recv(4096).decode()
    print("GET:", repr(res))
    try:
        decoded = base64.b64decode(res.strip()).decode()
        print("Decoded matches original:", json.loads(decoded) == sample_data)
    except Exception as e:
        print("Decode error:", e)
