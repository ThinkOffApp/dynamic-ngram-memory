"""Shared test pieces: the tokenizer, the reference store, a stub upstream and a proxy run in this process."""
import http.client
import json
import os
import shutil
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from notes_proxy import proxy
from notes_proxy.store import nm, seed

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
EXAMPLES = os.path.join(REPO, "tests", "examples")
L6 = os.path.join(EXAMPLES, "L6-seed1.json")
L6_OTHER = os.path.join(EXAMPLES, "L6-seed2.json")
NEMOTRON = os.path.expanduser("~/.cache/huggingface/hub/"
                              "models--TensorFold--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit/snapshots/d9d758fb83953437f7263256b0d96157e2a348b8/tokenizer.json")


def tokenizer_spec():
    """NOTES_PROXY_TEST_TOKENIZER (a tokenizer.json, or "fake"); else the Nemotron tokenizer when this Python has
    `tokenizers` and the file is here; else the stand-in."""
    spec = os.environ.get("NOTES_PROXY_TEST_TOKENIZER")
    if spec:
        return spec
    try:
        import tokenizers  # noqa: F401
    except ImportError:
        return "fake"
    return NEMOTRON if os.path.isfile(NEMOTRON) else "fake"


_TOK = {}


def tokenizer():
    spec = tokenizer_spec()
    if spec not in _TOK:
        _TOK[spec] = nm.load_tokenizer(spec)
    return _TOK[spec]


def reference(set_path=L6, phase="teach"):
    """(the set, the reference NoteStore for one phase, as notes.set_store builds it)."""
    S = nm.load_set(set_path)
    ph = next(p for p in S["phases"] if p["name"] == phase)
    ns, _ = nm.set_store(S, ph["store"], tokenizer())
    return S, ns


class TempDir:
    def __init__(self):
        self.path = tempfile.mkdtemp(prefix="notes-proxy-test-")

    def file(self, name):
        return os.path.join(self.path, name)

    def cleanup(self):
        shutil.rmtree(self.path, ignore_errors=True)


def seeded(tmp, name, set_path=L6, phase="teach", owner="user"):
    path = tmp.file(name)
    seed(path, owner, set_path, phase)
    return path


# ============================================================================================ the stub upstream
class Server(ThreadingHTTPServer):
    request_queue_size = 128             # many clients at once (socketserver's default is 5)


class Stub:
    """An OpenAI-shaped echo server. Every request is recorded. A chat reply holds the messages it received; a
    streamed one sends its first chunk, then waits for `release` (or 5 seconds) before the rest."""

    def __init__(self):
        self.requests = []
        self.release = threading.Event()
        self.released_by = None          # "event" or "timeout": how the streamed reply went on after chunk 1
        self.finished_at = None
        stub = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def _body(self):
                n = int(self.headers.get("Content-Length") or 0)
                return self.rfile.read(n) if n else b""

            def _send(self, status, obj):
                data = json.dumps(obj).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                stub.requests.append({"method": "GET", "path": self.path, "headers": dict(self.headers), "body": b""})
                if self.path == "/v1/models":
                    return self._send(200, {"object": "list", "data": [{"id": "stub-model", "object": "model"}]})
                self._send(404, {"error": {"message": "no such route"}})

            def do_POST(self):
                raw = self._body()
                stub.requests.append({"method": "POST", "path": self.path, "headers": dict(self.headers), "body": raw})
                body = json.loads(raw)
                msgs = body.get("messages")
                if self.path != "/v1/chat/completions":
                    return self._send(404, {"error": {"message": "no such route"}})
                if not body.get("stream"):
                    return self._send(200, {"id": "chatcmpl-stub", "object": "chat.completion", "model": "stub-model",
                                            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                                                "role": "assistant", "content": json.dumps(msgs)}}],
                                            "received_messages": msgs})
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()

                def chunk(obj):
                    b = (obj if isinstance(obj, bytes) else b"data: " + json.dumps(obj).encode()) + b"\n\n"
                    self.wfile.write(b"%x\r\n%s\r\n" % (len(b), b))
                    self.wfile.flush()

                piece = lambda c: {"id": "chatcmpl-stub", "object": "chat.completion.chunk", "model": "stub-model",
                                   "choices": [{"index": 0, "delta": {"content": c}, "finish_reason": None}]}
                chunk(dict(piece(json.dumps(msgs)), received_messages=msgs))
                stub.released_by = "event" if stub.release.wait(5.0) else "timeout"
                for c in ("one", "two", "three"):
                    chunk(piece(c))
                    time.sleep(0.05)
                chunk(b"data: [DONE]")
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
                stub.finished_at = time.monotonic()

        self.srv = Server(("127.0.0.1", 0), H)
        self.srv.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def last_body(self):
        return json.loads(self.requests[-1]["body"])

    def stop(self):
        self.srv.shutdown()
        self.srv.server_close()


# ============================================================================================ the proxy, in-process
class Running:
    """A proxy on a free port, serving in a thread. `handler` swaps the request handler (the controls). With `users`
    (a users file) it serves many users, and `store` is their store folder."""

    def __init__(self, upstream, store, owner="user", extra=(), handler=None, users=None):
        self.logs = []
        who = ["--users", users, "--store-dir", store] if users else ["--owner", owner, "--store", store]
        args = proxy.parser().parse_args(["--upstream", upstream] + who + ["--tokenizer", tokenizer_spec(),
                                                                           "--port", "0"] + list(extra))
        self.srv, self.notes = proxy.build(args, log=self.logs.append)
        if handler is not None:
            self.srv.RequestHandlerClass = handler
        self.port = self.srv.server_address[1]
        self.thread = threading.Thread(target=self.srv.serve_forever, daemon=True)
        self.thread.start()

    def call(self, method, path, body=None, headers=None, raw=None):
        """(status, headers, body bytes)."""
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        h = {"Content-Type": "application/json"} if data is not None else {}
        h.update(headers or {})
        c.request(method, path, body=data, headers=h)
        r = c.getresponse()
        out = (r.status, dict(r.getheaders()), r.read())
        c.close()
        return out

    def json(self, method, path, body=None, headers=None):
        st, h, b = self.call(method, path, body, headers)
        return st, json.loads(b) if b else None

    def stop(self):
        if self.srv is None:
            return
        self.srv.shutdown()
        self.srv.server_close()
        self.notes.close()
        self.srv = None
