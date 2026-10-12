# A fake OpenAI-compatible provider for measuring how OGX passes provider
# errors on. The path prefix picks the answer:
#   /r429/...  OpenRouter's live 429 body (2026-10-12, free shared pool)
#   /r401/...  OpenRouter's 401 for an unknown key
#   /r403/...  an OpenAI-style 403 that echoes part of a key and a URL
#   /ok/...    a normal answer, and 8-dim bag-of-words embeddings at /ok/v1/embeddings
import hashlib
import json
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BODIES = {
    "r429": (
        429,
        {
            "error": {
                "message": "Provider returned error",
                "code": 429,
                "metadata": {
                    "raw": "google/gemma-4-26b-a4b-it:free is temporarily rate-limited upstream. Please retry shortly, "
                    "or add your own key to accumulate your rate limits: https://openrouter.ai/settings/integrations",
                    "provider_name": "Google AI Studio",
                    "limit_source": "upstream_provider_shared_pool",
                    "is_byok": False,
                },
            },
            "user_id": "user_2xYz",
        },
    ),
    "r401": (401, {"error": {"message": "User not found.", "code": 401}}),
    "r403": (
        403,
        {
            "error": {
                "message": "Project proj_abc does not have access to model m. Key sk-proj-abcdefgh1234 "
                "see https://platform.example.com/docs and http://10.0.0.5:8080/x",
                "type": "invalid_request_error",
                "code": "model_not_found",
            }
        },
    ),
}


class H(BaseHTTPRequestHandler):
    """Answers by the first path segment (see the header)."""

    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        req = json.loads(self.rfile.read(n) or b"{}")
        key = self.path.strip("/").split("/")[0]
        if key in BODIES:
            status, body = BODIES[key]
            raw = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        if key == "rmid" and req.get("stream"):
            # OpenRouter's documented mid-stream error: HTTP 200, then a chunk with `error`
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            first = {
                "id": "c1",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": req.get("model"),
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hel"}, "finish_reason": None}],
            }
            err = {
                "id": "c1",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": req.get("model"),
                "provider": "Google AI Studio",
                "error": {
                    "code": 429,
                    "message": "Provider returned error",
                    "metadata": {"raw": "rate-limited upstream mid-stream", "provider_name": "Google AI Studio"},
                },
                "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": "error"}],
            }
            for c in (first, err):
                self.wfile.write(b"data: " + json.dumps(c).encode() + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")
            return
        if self.path.rstrip("/").endswith("/embeddings"):
            # Deterministic 8-dim bag-of-words vectors: similar texts, similar vectors.
            inputs = req.get("input")
            inputs = [inputs] if isinstance(inputs, str) else inputs
            data = []
            for i, text in enumerate(inputs):
                v = [0.0] * 8
                for w in str(text).lower().split():
                    v[int(hashlib.sha256(w.encode()).hexdigest(), 16) % 8] += 1.0
                n = math.sqrt(sum(x * x for x in v)) or 1.0
                data.append({"object": "embedding", "index": i, "embedding": [x / n for x in v]})
            raw = json.dumps(
                {
                    "object": "list",
                    "data": data,
                    "model": req.get("model"),
                    "usage": {"prompt_tokens": 1, "total_tokens": 1},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        # ok: one streamed chunk or one JSON answer
        msg = {"role": "assistant", "content": "hello"}
        if req.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            chunk = {
                "id": "c1",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": req.get("model"),
                "choices": [{"index": 0, "delta": msg, "finish_reason": None}],
            }
            done = {
                "id": "c1",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": req.get("model"),
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            for c in (chunk, done):
                self.wfile.write(b"data: " + json.dumps(c).encode() + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")
            return
        raw = json.dumps(
            {
                "id": "c1",
                "object": "chat.completion",
                "created": 1,
                "model": req.get("model"),
                "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


ThreadingHTTPServer(("127.0.0.1", 9999), H).serve_forever()
