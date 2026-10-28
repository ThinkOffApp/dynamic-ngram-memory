#!/usr/bin/env python3
"""A TensorFold-shaped child for Notes Proxy tests: health, tokenize, detokenize, chat, responses. No model.

Tokenize matches TensorFold: POST /v1/tokenize {"prompt"} -> {"tokens", "count"}. A "text"/"ids" body is ignored.
Detokenize: POST /v1/detokenize {"tokens"} -> {"prompt"} (chr of each id, the inverse of this child's tokenize).
POST routes require Authorization: Bearer $TENSORFOLD_API_KEY when that env is set (health stays public).
"""
import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _read(self):
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else b""

    def _send(self, status, obj, extra=None):
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def _auth_ok(self):
        want = os.environ.get("TENSORFOLD_API_KEY")
        if not want:
            return True
        got = self.headers.get("Authorization") or ""
        return got == f"Bearer {want}"

    def _record_auth(self, path):
        auths = self.headers.get_all("Authorization") or []
        last = os.path.join(self.server.model, "last_auth.json")
        try:
            with open(last, "w") as f:
                json.dump({"path": path, "authorization": auths}, f)
        except OSError:
            pass

    def do_GET(self):
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        if path in ("/health", "/"):
            return self._send(200, {"status": "ok", "model": "fake-tf", "pid": self.server.pid})
        return self._send(404, {"error": {"message": path}})

    def do_POST(self):
        path = self.path.split("?", 1)[0].rstrip("/")
        raw = self._read()
        try:
            body = json.loads(raw) if raw else {}
        except ValueError:
            body = {}
        self.server.posts.append({"path": path, "body": body, "auth": self.headers.get("Authorization")})
        self._record_auth(path)
        if not self._auth_ok():
            return self._send(401, {"error": {"message": "unauthorized", "type": "invalid_request_error", "code": 401}},
                              extra={"WWW-Authenticate": "Bearer"})
        if path == "/v1/tokenize":
            last = os.path.join(self.server.model, "last_tokenize.json")
            try:
                with open(last, "w") as f:
                    json.dump(body, f)
            except OSError:
                pass
            if os.path.isfile(os.path.join(self.server.model, "no_tokenize")):
                return self._send(404, {"error": {"message": "no tokenize"}})
            prompt = body["prompt"] if isinstance(body.get("prompt"), str) else ""
            tokens = [ord(c) for c in prompt]
            return self._send(200, {"tokens": tokens, "count": len(tokens)})
        if path == "/v1/detokenize":
            last = os.path.join(self.server.model, "last_detokenize.json")
            try:
                with open(last, "w") as f:
                    json.dump(body, f)
            except OSError:
                pass
            if os.path.isfile(os.path.join(self.server.model, "no_detokenize")):
                return self._send(404, {"error": {"message": "no detokenize"}})
            tokens = body.get("tokens")
            if not isinstance(tokens, list):
                return self._send(400, {"error": {"message": "tokens must be a list of integer token ids"}})
            parts = []
            for t in tokens:
                if not isinstance(t, int) or isinstance(t, bool) or not 0 <= t <= 0x10FFFF:
                    return self._send(400, {"error": {"message": "tokens token ids must be in range"}})
                parts.append("\ufffd" if 0xD800 <= t <= 0xDFFF else chr(t))
            return self._send(200, {"prompt": "".join(parts)})
        if path.endswith("/chat/completions"):
            return self._send(200, {"id": "chatcmpl-fake", "object": "chat.completion", "model": "fake-tf",
                                    "choices": [{"index": 0, "finish_reason": "stop",
                                                 "message": {"role": "assistant", "content": "ok"}}],
                                    "received_messages": body.get("messages"), "store": body.get("store")})
        if path.endswith("/responses"):
            return self._send(200, {"id": "resp-fake", "object": "response", "store": body.get("store")})
        return self._send(404, {"error": {"message": path}})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("serve")
    p.add_argument("model")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--no-update-check", action="store_true")
    p.add_argument("--no-drafts", action="store_true")
    p.add_argument("--no-thinking", action="store_true")
    p.add_argument("--context", default=None)
    p.add_argument("--backend", default=None)
    a = p.parse_args()
    fail = os.path.join(a.model, "fail_starts")
    if os.path.isfile(fail):
        try:
            n = int(open(fail).read().strip() or "0")
        except ValueError:
            n = 0
        if n > 0:
            with open(fail, "w") as f:
                f.write(str(n - 1))
            sys.exit(1)
    srv = ThreadingHTTPServer((a.host, a.port), H)
    srv.daemon_threads = True
    srv.pid = os.getpid()
    srv.posts = []
    srv.model = a.model
    srv.serve_forever()


if __name__ == "__main__":
    main()
