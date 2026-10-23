"""The Notes Proxy: TensorFold Notes in front of any OpenAI-compatible inference server.

For each POST /v1/chat/completions the proxy takes the last user message as the question, looks the owner's notes up
with the test suite's own lookup (tests/zengine/notes.py, NoteStore.retrieve: the n-grams, scoring, hop, cap fitting
and framing are that file's), puts the block in as the first system message, as the TensorFold server does, and
forwards the request to the upstream server. Streaming replies are passed on line by line as they arrive.

One owner per proxy: every request is answered from that owner's notes, so run one proxy per person. It binds to
127.0.0.1 by default. The notes are quoted as things the user said, never as instructions.

  python -m notes_proxy --upstream http://127.0.0.1:8080 --owner me --store ~/notes/me.jsonl \\
      --tokenizer /path/to/model/tokenizer.json [--port 8090] [--cap 512] [--write] [--abstain-line TEXT]
  python -m notes_proxy seed --set tests/examples/L6-seed1.json --phase teach --owner user --store STORE.jsonl

Standard library for the HTTP side; the model's tokenizer.json is read with the `tokenizers` package, as notes.py
reads it.
"""
import argparse
import hashlib
import http.client
import ipaddress
import json
import socket
import sys
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import gate
from .store import Notes, StoreError, nm, seed

NOTE_FIELDS = ("notes", "notes_exclude", "notes_past", "notes_cap", "notes_echo", "notes_write", "notes_language")
HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "proxy-connection", "te",
              "trailer", "trailers", "transfer-encoding", "upgrade"}
CHAT_BODY_LIMIT = 64 * 1024 * 1024
NOTES_BODY_LIMIT = 1024 * 1024


class Refused(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def _bool(body, key):
    v = body.get(key)
    if v is not None and not isinstance(v, bool):
        raise Refused(f"{key} must be a boolean")
    return v


def asked(body):
    """The request's notes fields, as the server reads them, except that notes are on unless "notes": false.
    Returns None when the request asks for no notes."""
    on = _bool(body, "notes")
    ask = {"exclude": (), "past": _bool(body, "notes_past"), "write": _bool(body, "notes_write"),
           "echo": bool(_bool(body, "notes_echo")), "cap": None}
    listed = "notes_exclude" in body
    if listed:
        x = body["notes_exclude"]
        if not isinstance(x, list) or not all(isinstance(i, int) and not isinstance(i, bool) and 0 <= i < 2 ** 32
                                              for i in x):
            raise Refused("notes_exclude must be a list of note ids")
        ask["exclude"] = tuple(x)
    if "notes_cap" in body:
        c = body["notes_cap"]
        if not isinstance(c, int) or isinstance(c, bool) or not 1 <= c < 2 ** 32:
            raise Refused("notes_cap must be a whole number of tokens, 1 or more")
        ask["cap"] = c
    if "notes_language" in body:
        if body["notes_language"] != "en":
            raise Refused("notes_language must be en: this proxy's past cue and question gate are English only")
    if on is False:
        if listed:
            raise Refused("notes is false but notes_exclude names notes to leave out")
        if ask["past"] is not None or "notes_cap" in body or ask["echo"]:
            raise Refused("notes_past, notes_cap and notes_echo need notes on")
        if ask["write"] is not None:
            raise Refused("notes_write needs notes on")
        return None
    return ask


def text_of(content):
    """A message's text as the TensorFold server's normalize reads it: a string as it is, text parts joined."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content
                       if isinstance(p, dict) and p.get("type") == "text" and isinstance(p.get("text"), str))
    return ""


def last_user(messages):
    """The last user message's text ("" when there is none)."""
    for m in reversed(messages):
        if isinstance(m, dict) and m.get("role") == "user":
            return text_of(m.get("content"))
    return ""


def with_block(messages, block, join=False):
    """The client's messages with the block first, as {"role": "system", "content": block}. With `join`, the client's
    leading system or developer messages are joined after it with a blank line into that one system message, as the
    TensorFold server's normalize joins them (for an upstream whose chat template takes one system message only)."""
    first = {"role": "system", "content": block}
    if not join:
        return [first] + list(messages)
    k, texts = 0, [block]
    while k < len(messages) and isinstance(messages[k], dict) and messages[k].get("role") in ("system", "developer") \
            and isinstance(messages[k].get("content"), (str, list)) and \
            (isinstance(messages[k]["content"], str) or all(isinstance(p, dict) and p.get("type") == "text"
                                                             for p in messages[k]["content"])):
        texts.append(text_of(messages[k]["content"]))
        k += 1
    return [{"role": "system", "content": "\n\n".join(texts)}] + list(messages[k:])


def splice(body, field):
    """A JSON object's bytes with one more field at its end, the upstream's bytes before it untouched. None when the
    body is not one JSON object."""
    try:
        if not isinstance(json.loads(body), dict):
            return None
    except ValueError:
        return None
    s = body.rstrip()
    inner = s[:-1].rstrip()
    sep = b"" if inner.endswith(b"{") else b","
    return inner + sep + b'"notes":' + json.dumps(field, ensure_ascii=False).encode() + b"}" + body[len(s):]


class Config:
    def __init__(self, upstream, owner, write=False, join_system=False, timeout=600.0, allow_origins=(),
                 local_only=True, log=None):
        u = urllib.parse.urlsplit(upstream)
        if u.scheme not in ("http", "https") or not u.hostname:
            raise ValueError("--upstream is an http:// or https:// base URL, such as http://127.0.0.1:8080")
        self.scheme, self.host = u.scheme, u.hostname
        self.port = u.port or (443 if u.scheme == "https" else 80)
        self.base = u.path.rstrip("/")
        self.owner = owner
        self.write = write
        self.join_system = join_system
        self.timeout = timeout
        self.allow_origins = set(allow_origins)
        self.local_only = local_only
        self.log = log or (lambda line: print(line, file=sys.stderr, flush=True))

    def connect(self):
        cls = http.client.HTTPSConnection if self.scheme == "https" else http.client.HTTPConnection
        return cls(self.host, self.port, timeout=self.timeout)

    def target(self, path):
        """The upstream path: the base URL's path, then the request's (".../v1" + "/v1/..." keeps one "/v1")."""
        if self.base.endswith("/v1") and (path == "/v1" or path.startswith("/v1/")):
            return self.base + path[3:]
        return self.base + path


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "notes-proxy/1"
    sys_version = ""

    # the server's own access log would print client addresses and paths only; ours is one line per request below
    def log_message(self, fmt, *args):
        pass

    @property
    def cfg(self):
        return self.server.cfg

    @property
    def notes(self):
        return self.server.notes

    # ------------------------------------------------------------------------------------------ plumbing
    def _cors(self):
        o = self.headers.get("Origin")
        return {"Access-Control-Allow-Origin": o, "Vary": "Origin"} if o and o in self.cfg.allow_origins else {}

    def send_json(self, status, obj, extra=None):
        data = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        for k, v in dict(self._cors(), **(extra or {})).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)
        self.wfile.flush()

    def send_error_json(self, status, message):
        kind = "invalid_request_error" if 400 <= status < 500 else "server_error"
        self.send_json(status, {"error": {"message": message, "type": kind, "code": status}})

    def read_body(self, limit):
        if "chunked" in (self.headers.get("Transfer-Encoding") or "").lower():
            self.close_connection = True
            raise Refused("send the body with a Content-Length", 411)
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self.close_connection = True
            raise Refused("bad Content-Length")
        if n < 0:
            self.close_connection = True
            raise Refused("bad Content-Length")
        if n > limit:
            self.close_connection = True
            raise Refused(f"request body exceeds {limit} bytes", 413)
        return self.rfile.read(n) if n else b""

    def json_body(self, limit):
        raw = self.read_body(limit)
        try:
            v = json.loads(raw) if raw else None
        except ValueError as e:
            raise Refused(f"the request body is not JSON: {e}")
        return raw, v

    def forward_headers(self, length):
        out = {}
        for k, v in self.headers.items():
            lk = k.lower()
            if lk in HOP_BY_HOP or lk in ("host", "content-length", "accept-encoding", "origin", "expect"):
                continue
            out[k] = v                        # Authorization goes on as the client sent it; it is never stored
        out["Host"] = self.cfg.host if self.cfg.port in (80, 443) else f"{self.cfg.host}:{self.cfg.port}"
        out["Accept-Encoding"] = "identity"
        if length is not None:
            out["Content-Length"] = str(length)
        return out

    def logline(self, status, t0, extra=""):
        self.cfg.log(f"notes-proxy: {self.command} {urllib.parse.urlsplit(self.path).path} {status} "
                     f"{(time.perf_counter() - t0) * 1000:.1f} ms{extra}")

    # ------------------------------------------------------------------------------------------ routing
    def handle_one(self):
        t0 = time.perf_counter()
        path = urllib.parse.urlsplit(self.path).path
        try:
            if self.cfg.local_only and not loopback_host(self.headers.get("Host")):
                # a page whose own name was pointed at 127.0.0.1 (DNS rebinding) is same-origin and sends no Origin;
                # its Host header still names the page's site, so only loopback names are served
                self.close_connection = True
                raise Refused("the Host header must name this machine (localhost, 127.0.0.1 or [::1]); start the "
                              "proxy with --allow-remote to serve other names", 403)
            origin = self.headers.get("Origin")
            if origin and origin not in self.cfg.allow_origins:
                # a web page's request (a browser always names its origin): refused, so no page can read or write notes
                self.close_connection = True
                raise Refused("requests from web pages are refused; start the proxy with --allow-origin to allow one",
                              403)
            if self.command == "OPTIONS" and origin:
                self.send_response(204)
                for k, v in self._cors().items():
                    self.send_header(k, v)
                self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
                self.send_header("Access-Control-Allow-Headers",
                                 self.headers.get("Access-Control-Request-Headers") or "Authorization, Content-Type")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return self.logline(204, t0)
            if path == "/v1/notes" or path.startswith("/v1/notes/"):
                return self.notes_api(path, t0)
            if path == "/v1/chat/completions" and self.command == "POST":
                return self.chat(t0)
            return self.passthrough(t0)
        except Refused as e:
            self.close_connection = True      # a refusal may leave a body unread: never read it as the next request
            self.send_error_json(e.status, str(e))
            self.logline(e.status, t0)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    do_GET = do_POST = do_PUT = do_DELETE = do_PATCH = do_OPTIONS = do_HEAD = handle_one

    # ------------------------------------------------------------------------------------------ /v1/notes
    def notes_api(self, path, t0):
        rest = path[len("/v1/notes"):].strip("/")
        nid = None
        if rest:
            if not rest.isdigit() or not rest.isascii():
                raise Refused("a note id is a whole number: /v1/notes/ID", 404)
            nid = int(rest)
        q = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
        flag = lambda name: (q.get(name) or ["0"])[-1].lower() in ("1", "true")
        m = self.command
        if m in ("GET", "HEAD"):
            if nid is not None:
                v = self.notes.get(nid)
                if v is None:
                    raise Refused("no note has this id", 404)
                self.send_json(200, v)
            else:
                self.send_json(200, {"object": "list", "owner": self.cfg.owner,
                                     "data": self.notes.listing(flag("history")), "history": flag("history")})
            return self.logline(200, t0)
        _raw, v = self.json_body(NOTES_BODY_LIMIT)
        if v is not None and not isinstance(v, dict):
            raise Refused("the request body must be a JSON object")
        v = v or {}
        if m == "DELETE":
            if nid is None:
                raise Refused("DELETE takes a note: /v1/notes/ID", 405)
            if v:
                raise Refused("an erase takes no body")
            try:
                done = self.notes.erase(nid)
            except StoreError as e:
                raise Refused(str(e), e.status)
            status = 500 if "error" in done else 200
            self.send_json(status, done)
            return self.logline(status, t0, f" erased [{nid}], removed {len(done['removed'])} writes")
        if m != "POST" or nid is not None:
            raise Refused("use GET or POST on /v1/notes, and GET or DELETE (erase) on /v1/notes/ID; this proxy does "
                          "not change or take back a note in place", 405)
        for k in v:
            if k not in ("text", "subject"):
                raise Refused(f"unknown field {k}; the fields here are text and subject")
        if "text" not in v:
            raise Refused("text is required: the note's sentence, as the user said it")
        if not isinstance(v["text"], str):
            raise Refused("text must be a string")
        if "subject" in v and not isinstance(v["subject"], str):
            raise Refused("subject must be a string")
        try:
            done = self.notes.write_text(v["text"], subject=v.get("subject"))
        except StoreError as e:
            raise Refused(str(e), e.status)
        status = 201 if done["written"] else 200
        self.send_json(status, done)
        self.logline(status, t0, f" write {done['verdict']} [{done['id']}]")

    # ------------------------------------------------------------------------------------------ chat
    def auto_write(self, ask, body):
        """The chat's own write, the server's rule: only the last message, only when it is the user's, only when it
        tells rather than asks; never the model's answers, system messages or earlier messages."""
        if not self.cfg.write:
            return {"skipped": "off: the proxy runs without --write"}, 0.0
        if ask["write"] is False:
            return {"skipped": "off: the request sends \"notes_write\": false"}, 0.0
        msgs = body.get("messages") or []
        if not msgs or not isinstance(msgs[-1], dict) or msgs[-1].get("role") != "user":
            return {"skipped": "the last message is not the user's: only the user's own words are written"}, 0.0
        t, why = gate.statement(last_user(msgs))
        if why:
            return {"skipped": why}, 0.0
        t0 = time.perf_counter()
        try:
            done = self.notes.write_text(t)
        except StoreError as e:
            return {"failed": str(e)}, (time.perf_counter() - t0) * 1e6
        return {"verdict": done["verdict"], "id": done["id"], "source": done["source"]}, \
            (time.perf_counter() - t0) * 1e6

    def chat(self, t0):
        _raw, body = self.json_body(CHAT_BODY_LIMIT)
        if not isinstance(body, dict):
            raise Refused("the request body must be a JSON object")
        ask = asked(body)
        fwd = {k: v for k, v in body.items() if k not in NOTE_FIELDS}
        used = None
        if ask is not None:
            msgs = body.get("messages")
            if not isinstance(msgs, list) or not msgs:
                raise Refused("messages must be a non-empty list")
            question = last_user(msgs)
            res = self.notes.lookup(question, cap=ask["cap"], past=ask["past"], exclude=ask["exclude"])
            if res["block"]:
                fwd["messages"] = with_block(msgs, res["block"], self.cfg.join_system)
            total_us = (time.perf_counter() - t0) * 1e6
            wrote, write_us = self.auto_write(ask, body)
            used = {"ids": res["ids"], "past": res["past"], "past_from": "request" if ask["past"] is not None
                    else "question", "language": "en", "cap": res["cap"], "tokens": res["tokens"],
                    "candidates": res["candidates"], "notes_in_store": self.notes.count(),
                    "lookup_us": res["lookup_us"], "total_us": round(total_us, 1),
                    "block_sha256": hashlib.sha256(res["block"].encode()).hexdigest() if res["block"] else None}
            if ask["echo"]:
                used["block"] = res["block"]
            used["write"] = wrote
            used["write_us"] = round(write_us, 1)
        data = json.dumps(fwd, ensure_ascii=False).encode()
        stream = fwd.get("stream") is True
        extra = ""
        if used is not None:
            w = used["write"]
            extra = (f" notes [{','.join(map(str, used['ids']))}] {used['tokens']} tokens of {used['cap']}, past "
                     f"{'yes' if used['past'] else 'no'}, {used['candidates']} candidates, lookup {used['lookup_us']}"
                     f" us; write " + (f"{w['verdict']} [{w['id']}]" if "verdict" in w else
                                       f"failed" if "failed" in w else f"skipped ({w['skipped']})"))
        self.relay("POST", data, used, stream, t0, extra)

    # ------------------------------------------------------------------------------------------ upstream
    def passthrough(self, t0):
        body = None
        if self.command in ("POST", "PUT", "PATCH", "DELETE") or self.headers.get("Content-Length"):
            body = self.read_body(CHAT_BODY_LIMIT)
        self.relay(self.command, body, None, False, t0, "")

    def relay(self, method, body, used, stream, t0, extra):
        conn = self.cfg.connect()
        target = self.cfg.target(self.path)
        try:
            conn.request(method, target, body=body, headers=self.forward_headers(None if body is None else len(body)))
            r = conn.getresponse()
        except (OSError, http.client.HTTPException) as e:
            conn.close()
            raise Refused(f"the upstream server at {self.cfg.host}:{self.cfg.port} could not be reached "
                          f"({type(e).__name__})", 502)
        try:
            ctype = (r.getheader("Content-Type") or "").lower()
            headers = [(k, v) for k, v in r.getheaders()
                       if k.lower() not in HOP_BY_HOP and k.lower() not in ("content-length", "server", "date")]
            notes_hdr = {}
            if used is not None:
                notes_hdr = {"X-Notes-Ids": ",".join(map(str, used["ids"])), "X-Notes-Tokens": str(used["tokens"]),
                             "X-Notes-Past": "true" if used["past"] else "false"}
            if used is not None and stream and "text/event-stream" in ctype:
                return self.relay_sse(r, headers, notes_hdr, used, t0, extra)
            if used is not None and method != "HEAD":
                data = r.read()
                if 200 <= r.status < 300 and "json" in ctype:
                    data = splice(data, used) or data
                self.send_response(r.status, r.reason)
                for k, v in headers + list(notes_hdr.items()) + list(self._cors().items()):
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                self.wfile.flush()
                return self.logline(r.status, t0, extra)
            return self.relay_raw(r, headers, method, t0, extra)
        finally:
            conn.close()

    def relay_raw(self, r, headers, method, t0, extra):
        """Any other reply, passed on as it comes: with its Content-Length when it has one, else chunked."""
        length = r.getheader("Content-Length")
        self.send_response(r.status, r.reason)
        for k, v in headers + list(self._cors().items()):
            self.send_header(k, v)
        chunked = False
        if method == "HEAD" or r.status in (204, 304) or 100 <= r.status < 200:
            if length is not None:
                self.send_header("Content-Length", length)
            self.end_headers()
            self.wfile.flush()
            return self.logline(r.status, t0, extra)
        if length is not None:
            self.send_header("Content-Length", length)
        else:
            chunked = True
            self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        try:
            while True:
                b = r.read1(65536)
                if not b:
                    break
                self.wfile.write(b"%x\r\n%s\r\n" % (len(b), b) if chunked else b)
                self.wfile.flush()
            if chunked:
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except (OSError, http.client.HTTPException) as e:
            self.close_connection = True
            extra += f" (upstream failed mid-reply: {type(e).__name__})"
        self.logline(r.status, t0, extra)

    def relay_sse(self, r, headers, notes_hdr, used, t0, extra):
        """A streamed reply, line by line as the upstream sends it. The notes object goes in as an SSE comment line
        (": notes {...}") just before "data: [DONE]" (or at the end when there is none): a comment is ignored by
        every SSE parser, so no client sees a chunk it does not expect."""
        self.send_response(r.status, r.reason)
        for k, v in headers + list(notes_hdr.items()) + list(self._cors().items()):
            self.send_header(k, v)
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        self.wfile.flush()
        note = b": notes " + json.dumps(used, ensure_ascii=False).encode() + b"\n\n"

        def put(b):
            self.wfile.write(b"%x\r\n%s\r\n" % (len(b), b))
            self.wfile.flush()

        sent = False
        try:
            while True:
                line = r.readline(1 << 20)
                if not line:
                    break
                if not sent and line.strip() == b"data: [DONE]":
                    put(note)
                    sent = True
                put(line)
            if not sent:
                put(b"\n" + note)
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True      # the client left: closing the upstream connection stops its work
        except (OSError, http.client.HTTPException) as e:
            self.close_connection = True      # the upstream failed mid-stream: the client sees the stream cut
            extra += f" (upstream failed mid-stream: {type(e).__name__})"
        self.logline(r.status, t0, extra + " (streamed)")


class Proxy(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, addr, cfg, notes):
        self.cfg = cfg
        self.notes = notes
        family = socket.AF_INET6 if ":" in addr[0] else socket.AF_INET
        self.address_family = family
        super().__init__(addr, Handler)


def is_loopback(host):
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def loopback_host(value):
    """Whether a Host header ("name", "name:port", "[v6]" or "[v6]:port") names a loopback address."""
    h = (value or "").strip().lower()
    if h.startswith("["):
        name = h[1:h.find("]")] if "]" in h else ""
    else:
        name = h.rsplit(":", 1)[0] if h.count(":") == 1 else h
    return bool(name) and is_loopback(name)


def build(args, log=None):
    """(the proxy server, its Notes) from parsed arguments; serve with .serve_forever()."""
    if not is_loopback(args.host) and not args.allow_remote:
        raise SystemExit(f"--host {args.host} is not a loopback address: anyone who can reach it could read and erase "
                         f"this owner's notes. Pass --allow-remote to do it anyway.")
    config = json.loads(args.notes_config) if args.notes_config else {}
    if not isinstance(config, dict):
        raise SystemExit("--notes-config is a JSON object of notes.py settings")
    config["cap"] = args.cap
    tok = nm.load_tokenizer(args.tokenizer)
    notes = Notes(args.store, args.owner, tok, config=config, abstain_line=args.abstain_line)
    cfg = Config(args.upstream, args.owner, write=args.write, join_system=args.join_system, timeout=args.timeout,
                 allow_origins=args.allow_origin or (), local_only=not args.allow_remote, log=log)
    try:
        return Proxy((args.host, args.port), cfg, notes), notes
    except BaseException:
        notes.close()
        raise


def parser():
    p = argparse.ArgumentParser(prog="python -m notes_proxy", description="TensorFold Notes in front of any "
                                "OpenAI-compatible server. One owner per proxy.")
    p.add_argument("--upstream", required=True, help="the inference server's base URL, e.g. http://127.0.0.1:8080")
    p.add_argument("--owner", required=True, help="whose notes this proxy serves (one owner per proxy)")
    p.add_argument("--store", required=True, help="the notes journal file (made if missing)")
    p.add_argument("--tokenizer", required=True,
                   help="the upstream model's tokenizer.json (or its folder): the 512-token cap is counted with it; "
                        "\"fake\" only for tests")
    p.add_argument("--cap", type=int, default=512, help="the block's token cap, framing included (default 512)")
    p.add_argument("--abstain-line", default=None, help="a line of text joined to the block's first line (off)")
    p.add_argument("--write", action="store_true", help="write notes from the user's own chat statements (off)")
    p.add_argument("--join-system", action="store_true",
                   help="join the block and the client's leading system messages into one system message")
    p.add_argument("--notes-config", default=None, help="other notes.py settings, as a JSON object")
    p.add_argument("--host", default="127.0.0.1", help="the address to bind (default 127.0.0.1)")
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--allow-remote", action="store_true", help="allow binding to a non-loopback address")
    p.add_argument("--allow-origin", action="append", help="a web origin allowed to call the proxy (repeatable)")
    p.add_argument("--timeout", type=float, default=600.0, help="seconds to wait for the upstream (default 600)")
    return p


def seed_main(argv):
    p = argparse.ArgumentParser(prog="python -m notes_proxy seed",
                                description="Make a new store from a test set's lessons for one phase.")
    p.add_argument("--set", required=True)
    p.add_argument("--phase", default="teach")
    p.add_argument("--owner", required=True)
    p.add_argument("--store", required=True)
    a = p.parse_args(argv)
    n = seed(a.store, a.owner, a.set, a.phase)
    print(f"notes-proxy: {n} lessons of {a.set} ({a.phase}) written to {a.store} for owner {a.owner}")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["seed"]:
        return seed_main(argv[1:])
    args = parser().parse_args(argv)
    try:
        srv, notes = build(args)
    except (StoreError, nm.NotesError, ValueError) as e:
        raise SystemExit(f"notes-proxy: {e}")
    h, pt = srv.server_address[:2]
    print(f"notes-proxy: owner {args.owner}, {notes.count()} notes, tokenizer {notes.store.tok.describe()}, cap "
          f"{args.cap}, writes from chat {'on' if args.write else 'off'}; listening on http://{h}:{pt}, forwarding "
          f"to {srv.cfg.scheme}://{srv.cfg.host}:{srv.cfg.port}{srv.cfg.base}", file=sys.stderr, flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
        notes.close()


if __name__ == "__main__":
    main()
