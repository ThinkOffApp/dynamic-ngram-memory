"""The Notes Proxy: TensorFold Notes in front of any OpenAI-compatible inference server.

For each POST /v1/chat/completions the proxy takes the last user message as the question, looks the owner's notes up
with the test suite's own lookup (tests/zengine/notes.py, NoteStore.retrieve: the n-grams, scoring, hop, cap fitting
and framing are that file's, and --framing firm only swaps the block's first paragraph), puts the block in as the
first system message, as the TensorFold server does, and forwards the request to the upstream server. Streaming
replies are passed on line by line as they arrive.

With --owner and --store it serves one owner: every request is answered from that owner's notes. With --users and
--store-dir it serves many: each request's key (Authorization: Bearer KEY) names its owner, a missing or unknown key
is a 401, and the key never goes on to the upstream (users.py). With many users the upstream's prompt cache is kept
apart per user, no client field or header names another user's state there, and only the model routes pass through.
It binds to 127.0.0.1 by default. The notes are quoted as
things the user said, never as instructions.

  python -m notes_proxy --upstream http://127.0.0.1:8080 --owner me --store ~/notes/me.jsonl \\
      --tokenizer /path/to/model/tokenizer.json [--port 8090] [--cap 512] [--write] [--abstain-line TEXT] \\
      [--framing quoted|firm]
  python -m notes_proxy --upstream http://127.0.0.1:8080 --users users.json --store-dir ~/notes \\
      --tokenizer /path/to/model/tokenizer.json [--upstream-key-env NAME] [--share-upstream-cache]
  python -m notes_proxy --tensorfold org/name --owner me --store ~/notes/me.jsonl
  python -m notes_proxy add-user --users users.json --owner alice --store-dir ~/notes [--reuse-notes]
  python -m notes_proxy remove-user --users users.json --owner alice (--keep-notes | --erase-notes) [--store-dir DIR]
  python -m notes_proxy seed --set tests/examples/L6-seed1.json --phase teach --owner user --store STORE.jsonl

Standard library for the HTTP side; the model's tokenizer.json is read with the `tokenizers` package, as notes.py
reads it.
"""
import argparse
import hashlib
import hmac
import http.client
import ipaddress
import json
import os
import secrets
import socket
import sys
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import gate, users
from . import tensorfold as tfchild
from .store import Notes, ProxyStore, StoreError, nm, seed

NOTE_FIELDS = ("notes", "notes_exclude", "notes_past", "notes_cap", "notes_echo", "notes_write", "notes_language")
HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "proxy-connection", "te",
              "trailer", "trailers", "transfer-encoding", "upgrade"}
CHAT_BODY_LIMIT = 64 * 1024 * 1024
NOTES_BODY_LIMIT = 1024 * 1024
# with many users, the routes passed on besides the chat route: the others (llama.cpp's /slots, /metrics, /props and
# the like) can show or change other users' prompts
PASS_MANY = {"/v1/models": ("GET", "HEAD"), "/v1/completions": ("POST",), "/v1/embeddings": ("POST",)}
TIMES = ("lookup_us", "total_us", "write_us")      # left out with many users: the shared tokenizer cache shows in them
# with many users, the client's fields that could name state on the upstream (the README's "What reaches the upstream")
DROP_MANY = ("id_slot", "slot_id", "request_id", "rid", "session_id")      # the upstream picks a slot and an id itself
MINE_MANY = ("user", "safety_identifier")                                   # replaced by the user's own value
REFUSE_MANY = ("conversation", "previous_response_id", "session_params", "kv_transfer_params")   # state it keeps
HEADERS_MANY = ("content-type", "accept", "user-agent")                     # the only client headers passed on


class Refused(Exception):
    def __init__(self, message, status=400, log="", headers=None):
        super().__init__(message)
        self.status = status
        self.log = log                 # the cause, for the operator's log line only
        self.headers = headers or {}


def holds(v, key):
    """Whether a decoded JSON value holds the key in any string or object key, at any depth."""
    if isinstance(v, str):
        return key in v
    if isinstance(v, dict):
        return any(holds(k, key) or holds(x, key) for k, x in v.items())
    if isinstance(v, list):
        return any(holds(x, key) for x in v)
    return False


KEY_IN_BODY = ("the request body holds your key: send the key only as Authorization: Bearer KEY, because the body "
               "goes on to the upstream or into your notes")


def scrub(obj):
    """Take the counts of reused prompt tokens out of a reply object (llama.cpp's timings and prompt_progress, and
    usage's cached_tokens): with many users they tell one user about another's prompts. True when it changed."""
    if not isinstance(obj, dict):
        return False
    hit = False
    for k in ("timings", "prompt_progress"):
        if k in obj:
            del obj[k]
            hit = True
    u = obj.get("usage")
    d = u.get("prompt_tokens_details") if isinstance(u, dict) else None
    if isinstance(d, dict) and "cached_tokens" in d:
        del d["cached_tokens"]
        hit = True
    return hit


def scrubbed(data):
    """A JSON reply's bytes, scrubbed; the upstream's own bytes when there is nothing to take out."""
    try:
        obj = json.loads(data)
    except ValueError:
        return data
    return json.dumps(obj, ensure_ascii=False).encode() if scrub(obj) else data


def scrubbed_line(line):
    """One SSE line, its data scrubbed; the line as it came when there is nothing to take out."""
    if not line.startswith(b"data:"):
        return line
    end = b"\r\n" if line.endswith(b"\r\n") else b"\n" if line.endswith(b"\n") else b""
    payload = line[5:len(line) - len(end)].strip()
    try:
        obj = json.loads(payload)
    except ValueError:
        return line
    return b"data: " + json.dumps(obj, ensure_ascii=False).encode() + end if scrub(obj) else line


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
        if not isinstance(x, list) or not all(isinstance(i, str) and 0 < len(i) <= 256 for i in x):
            raise Refused("notes_exclude must be a list of note ids, such as [\"w12\"], as GET /v1/notes lists them; "
                          "a number is a note's place in the block, which moves when a note is erased")
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
                 local_only=True, log=None, upstream_key=None, isolate_cache=False, engine=None):
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
        self.upstream_key = upstream_key   # --upstream-key-env's value: what the upstream gets as Authorization
        self.isolate_cache = isolate_cache  # many users without --share-upstream-cache
        self.engine = engine               # TensorFold child started with --tensorfold, or None
        self.log = log or (lambda line: print(line, file=sys.stderr, flush=True))

    def connect(self):
        if self.engine is not None:
            host, port = self.engine.endpoint()
            return http.client.HTTPConnection(host, port, timeout=self.timeout)
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
        """The request's owner's Notes, opened at the first route that needs them (never for a pass-through)."""
        if self._notes is None:
            self._notes = self.server.notes_of(self.owner)
        return self._notes

    @property
    def many(self):
        return self.server.users is not None

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

    def send_error_json(self, status, message, extra=None):
        kind = "invalid_request_error" if 400 <= status < 500 else "server_error"
        self.send_json(status, {"error": {"message": message, "type": kind, "code": status}}, extra)

    def isolated(self, body):
        """With many users, a request body for the upstream that names no other user's state there. A field that
        points at state the upstream keeps, an owner or a notes field is refused. Slot and request ids are dropped.
        The user ids are replaced by this user's own value. The cache is kept apart: llama.cpp reuses no cached prompt
        (cache_prompt false) and vLLM reuses only this user's (cache_salt), whatever the client sent for them. With
        one owner, the body as it is."""
        if not self.many:
            return body
        for k in body:
            if k in REFUSE_MANY:
                raise Refused(f"{k} is refused: it points at state the upstream keeps, and with many users this proxy "
                              f"cannot tell whose it is")
            if k == "owner":
                raise Refused("owner is refused: with many users the key alone names whose notes a request reads")
            if k == "notes" or k.startswith("notes_"):
                raise Refused(f"{k} is refused: with many users no notes field goes on to the upstream. Only "
                              f"/v1/chat/completions takes notes fields, and only the proxy's own")
        mine = self.server.salt(self.owner)
        out = {k: mine if k in MINE_MANY else v for k, v in body.items() if k not in DROP_MANY}
        if self.cfg.isolate_cache:
            out.update(cache_prompt=False, cache_salt=mine)
            if "prompt_cache_key" in out:
                out["prompt_cache_key"] = mine
        return out

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
        raw = self.rfile.read(n) if n else b""
        if self.key and self.key.encode() in raw:
            raise Refused(KEY_IN_BODY)
        return raw

    def json_body(self, limit):
        raw = self.read_body(limit)
        try:
            v = json.loads(raw) if raw else None
        except ValueError as e:
            raise Refused(f"the request body is not JSON: {e}")
        if self.key and holds(v, self.key):
            raise Refused(KEY_IN_BODY)              # \u escapes hide the key from the raw bytes, not from the values
        return raw, v

    def forward_headers(self, length):
        key = self.key
        out = {}
        for k, v in self.headers.items():
            lk = k.lower()
            if lk in HOP_BY_HOP or lk in ("host", "content-length", "accept-encoding", "origin", "expect"):
                continue
            if self.many and lk not in HEADERS_MANY:
                continue                      # the user's key, a cookie, another key or a session id: none goes on
            if lk == "authorization" and self.cfg.upstream_key is not None:
                continue                      # a key the upstream key replaces
            if key and key in v:
                continue                      # the user's key, in whatever other header the client also put it
            out[k] = v                        # with one owner and no upstream key, Authorization goes on as sent
        if self.cfg.upstream_key is not None:
            out["Authorization"] = f"Bearer {self.cfg.upstream_key}"
        if self.cfg.engine is not None:
            out.update(self.cfg.engine.auth_headers())   # the child's key; the client never needs it
            host, port = self.cfg.engine.endpoint()
            out["Host"] = host if port in (80, 443) else f"{host}:{port}"
        else:
            out["Host"] = self.cfg.host if self.cfg.port in (80, 443) else f"{self.cfg.host}:{self.cfg.port}"
        out["Accept-Encoding"] = "identity"
        if length is not None:
            out["Content-Length"] = str(length)
        return out

    def logline(self, status, t0, extra=""):
        who = f"[{self.owner}] " if self.many and self.owner else ""
        path = urllib.parse.urlsplit(self.path).path
        if self.key and self.key in urllib.parse.unquote(path):
            path = "(a path that holds the key)"
        self.cfg.log(f"notes-proxy: {who}{self.command} {path} {status} {(time.perf_counter() - t0) * 1000:.1f} "
                     f"ms{extra}")

    # ------------------------------------------------------------------------------------------ routing
    def handle_one(self):
        t0 = time.perf_counter()
        path = urllib.parse.urlsplit(self.path).path
        self.owner = self._notes = self.key = None   # a kept-alive connection names its owner anew at every request
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
            if path == "/health" and self.command in ("GET", "HEAD") and self.cfg.engine is not None:
                h = self.cfg.engine.health()
                status = 200 if h.get("status") == "ok" else 503
                extra = {"Retry-After": str(self.cfg.engine.retry_after())} if status == 503 else None
                self.send_json(status, h, extra)
                return self.logline(status, t0)
            self.owner = self.server.who(self.headers)     # with many users: the key's owner, or a 401 here
            if self.many:
                self.key = bearer(self.headers.get("Authorization"))
                if self.key in urllib.parse.unquote(self.path):
                    raise Refused("the URL holds your key: send the key only as Authorization: Bearer KEY, because "
                                  "the URL goes on to the upstream")
            if path == "/v1/notes" or path.startswith("/v1/notes/"):
                return self.notes_api(path, t0)
            if path == "/v1/chat/completions" and self.command == "POST":
                return self.chat(t0)
            return self.passthrough(t0)
        except tfchild.TensorFoldError as e:
            self.close_connection = True
            extra = {"Retry-After": str(self.cfg.engine.retry_after())} if self.cfg.engine is not None else None
            self.send_error_json(503, "the TensorFold engine is not ready", extra)
            self.logline(503, t0, f" ({e})")
        except Refused as e:
            self.close_connection = True      # a refusal may leave a body unread: never read it as the next request
            extra = dict(e.headers)
            if e.status == 401:
                extra.setdefault("WWW-Authenticate", "Bearer")
            self.send_error_json(e.status, str(e), extra or None)
            self.logline(e.status, t0, e.log)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    do_GET = do_POST = do_PUT = do_DELETE = do_PATCH = do_OPTIONS = do_HEAD = handle_one

    # ------------------------------------------------------------------------------------------ /v1/notes
    def notes_api(self, path, t0):
        rest = path[len("/v1/notes"):].strip("/")
        nid = None
        if rest:
            nid = urllib.parse.unquote(rest)
            if nid.isascii() and nid.isdigit():
                raise Refused("a note id is the id GET /v1/notes lists, such as w12, and it never changes; a number "
                              "is a note's place in the block, which moves when a note is erased, so it names no "
                              "note here", 400)
        q = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
        flag = lambda name: (q.get(name) or ["0"])[-1].lower() in ("1", "true")
        m = self.command
        if rest == "erase-status":
            if m not in ("GET", "HEAD"):
                raise Refused("use GET on /v1/notes/erase-status", 405)
            if self.cfg.engine is None:
                raise Refused("erase-status is for --tensorfold", 404)
            self.send_json(200, self.cfg.engine.erase_status())
            return self.logline(200, t0)
        if m in ("GET", "HEAD"):
            if nid is not None:
                v = self.notes.get(nid)
                if v is None:
                    raise Refused("no note has this id", 404)
                self.send_json(200, v)
            else:
                self.send_json(200, {"object": "list", "owner": self.owner,
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
            if self.cfg.engine is not None and "error" not in done:
                done = self.cfg.engine.after_erase(done)
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
        fwd = self.isolated({k: v for k, v in body.items() if k not in NOTE_FIELDS})
        used = None
        extra = ""
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
            used = {"ids": res["note_ids"], "numbers": res["ids"], "past": res["past"],
                    "past_from": "request" if ask["past"] is not None else "question", "language": "en",
                    "cap": res["cap"], "tokens": res["tokens"],
                    "candidates": res["candidates"], "notes_in_store": self.notes.count(),
                    "lookup_us": res["lookup_us"], "total_us": round(total_us, 1),
                    "block_sha256": hashlib.sha256(res["block"].encode()).hexdigest() if res["block"] else None}
            if ask["echo"]:
                used["block"] = res["block"]
            used["write"] = wrote
            used["write_us"] = round(write_us, 1)
            w = used["write"]
            extra = (f" notes [{','.join(map(str, used['ids']))}] {used['tokens']} tokens of {used['cap']}, past "
                     f"{'yes' if used['past'] else 'no'}, {used['candidates']} candidates, lookup {used['lookup_us']}"
                     f" us; write " + (f"{w['verdict']} [{w['id']}]" if "verdict" in w else
                                       f"failed" if "failed" in w else f"skipped ({w['skipped']})"))
            if self.many:
                for k in TIMES:          # the operator's log keeps them
                    del used[k]
        if self.cfg.engine is not None:
            fwd["store"] = False
        data = json.dumps(fwd, ensure_ascii=False).encode()
        stream = fwd.get("stream") is True
        self.relay("POST", data, used, stream, t0, extra)

    # ------------------------------------------------------------------------------------------ upstream
    def passthrough(self, t0):
        route = self.path.partition("?")[0]
        path = urllib.parse.urlsplit(self.path).path.rstrip("/")
        engine_responses = (self.cfg.engine is not None and self.command == "POST"
                            and path.endswith("/responses"))
        if self.many and self.command not in PASS_MANY.get(route, ()) and not engine_responses:
            raise Refused("with many users this proxy passes on only GET /v1/models, POST /v1/completions and POST "
                          "/v1/embeddings, and gives notes to POST /v1/chat/completions", 404)
        body = None
        if self.command in ("POST", "PUT", "PATCH", "DELETE") or self.headers.get("Content-Length"):
            body = self.read_body(CHAT_BODY_LIMIT)
        if self.many and (route in ("/v1/completions", "/v1/embeddings") or engine_responses):
            try:
                v = json.loads(body) if body else None
            except ValueError:
                v = None
            if not isinstance(v, dict):
                raise Refused("the request body must be a JSON object")
            if self.key and holds(v, self.key):
                raise Refused(KEY_IN_BODY)
            body = json.dumps(self.isolated(v), ensure_ascii=False).encode()
        if engine_responses and body:
            body = tfchild.inject_store_false(body)
        self.relay(self.command, body, None, False, t0, "")

    def relay(self, method, body, used, stream, t0, extra):
        engine = self.cfg.engine
        held = False
        if engine is not None:
            try:
                engine.begin()
                held = True
            except tfchild.TensorFoldError:
                raise Refused("the TensorFold engine is not ready", 503,
                              headers={"Retry-After": str(engine.retry_after())})
        try:
            conn = self.cfg.connect()
            target = self.cfg.target(self.path)
            try:
                conn.request(method, target, body=body,
                             headers=self.forward_headers(None if body is None else len(body)))
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
                sse, js = "text/event-stream" in ctype, "json" in ctype
                if sse and (used is not None and stream or self.many):
                    return self.relay_sse(r, headers, notes_hdr, used, t0, extra)
                if method != "HEAD" and (used is not None or self.many and js):
                    data = r.read()
                    if self.many and js:
                        data = scrubbed(data)
                    if used is not None and 200 <= r.status < 300 and js:
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
        finally:
            if held:
                engine.end()

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
        every SSE parser, so no client sees a chunk it does not expect. With many users each data line is scrubbed,
        and a reply without notes (used None) gets no notes line."""
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

        sent = used is None
        try:
            while True:
                line = r.readline(1 << 20)
                if not line:
                    break
                if not sent and line.strip() == b"data: [DONE]":
                    put(note)
                    sent = True
                put(scrubbed_line(line) if self.many else line)
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
    request_queue_size = 128             # socketserver's 5 resets the sixth waiting connection

    def __init__(self, addr, cfg, notes, users=None):
        self.cfg = cfg
        self.notes = notes                # one owner's Notes, or with a users file every owner's (users.Owners)
        self.users = users
        self.salt_key = secrets.token_bytes(32)    # each user's cache_salt comes from it, new at every start
        family = socket.AF_INET6 if ":" in addr[0] else socket.AF_INET
        self.address_family = family
        super().__init__(addr, Handler)

    def who(self, headers):
        """A request's owner: the one owner, or with a users file the owner of the key it sends (else a 401)."""
        if self.users is None:
            return self.cfg.owner
        owner = self.users.who(bearer(headers.get("Authorization")))
        if owner is None:
            raise Refused("the key is missing or unknown: send your own as Authorization: Bearer KEY", 401)
        return owner

    def notes_of(self, owner):
        """The owner's Notes. With many users a store that cannot be opened is a short reply to its owner, and its
        cause goes to the operator's log only (it names the server's folders)."""
        if self.users is None:
            return self.notes
        try:
            return self.notes.get(owner)
        except (StoreError, users.UsersError, OSError) as e:
            status = 503 if isinstance(e, OSError) else getattr(e, "status", 500)
            raise Refused("your notes cannot be opened now: the proxy's operator can see why in its log", status,
                          log=f" ({owner}'s store cannot be opened: {e})")

    def salt(self, owner):
        """The owner's cache_salt: the same for all their requests, another for every other owner, unknown to all."""
        return hmac.new(self.salt_key, owner.encode(), hashlib.sha256).hexdigest()


def bearer(value):
    """The KEY of an "Authorization: Bearer KEY" header, or None."""
    scheme, _, key = (value or "").strip().partition(" ")
    key = key.strip()
    return key if scheme.lower() == "bearer" and key else None


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


def usage_error(args):
    """The mistake in the choice of one owner or many users, or None. main() reports it as argparse reports a missing
    option on main (exit 2, with the usage line), and build() refuses it too."""
    many = args.users is not None or args.store_dir is not None
    if many and (args.users is None or args.store_dir is None or args.owner is not None or args.store is not None):
        return "many users take --users FILE and --store-dir DIR, and no --owner or --store"
    if not many and (args.owner is None or args.store is None):
        missing = [o for o, v in (("--owner", args.owner), ("--store", args.store)) if v is None]
        return "the following arguments are required: " + ", ".join(missing)
    if not many and args.share_upstream_cache:
        return "--share-upstream-cache is for many users (--users and --store-dir)"
    tf, up = getattr(args, "tensorfold", None), getattr(args, "upstream", None)
    if bool(tf) == bool(up):
        return "give --upstream URL, or --tensorfold MODEL, not both"
    if tf and getattr(args, "tokenizer", None):
        return "--tokenizer is for --upstream; with --tensorfold the child tokenizes"
    if not tf and not getattr(args, "tokenizer", None):
        return "the following arguments are required: --tokenizer"
    if tf and many and not args.share_upstream_cache:
        return "--tensorfold with many users shares TensorFold's prompt cache: pass --share-upstream-cache to allow it, or run one proxy per owner"
    return None


def build(args, log=None):
    """(the proxy server, its Notes or with a users file its users.Owners) from parsed arguments; serve with
    .serve_forever()."""
    many = args.users is not None or args.store_dir is not None
    err = usage_error(args)
    if err:
        if err.startswith("the following arguments") and "--store" in err:
            err = "give --owner NAME and --store FILE for one owner, or --users FILE and --store-dir DIR for many users"
        raise SystemExit(f"notes-proxy: {err}")
    if not is_loopback(args.host) and not args.allow_remote:
        if many:
            raise SystemExit(f"notes-proxy: --host {args.host} is not a loopback address: the keys and the notes would "
                             f"cross the network unencrypted. Pass --allow-remote to do it anyway.")
        key = f", and use the upstream key in ${args.upstream_key_env}" if args.upstream_key_env else ""
        raise SystemExit(f"--host {args.host} is not a loopback address: anyone who can reach it could read and erase "
                         f"this owner's notes{key}. Pass --allow-remote to do it anyway.")
    upstream_key = None
    if args.upstream_key_env:
        upstream_key = os.environ.get(args.upstream_key_env, "")
        if not upstream_key or not all("!" <= c <= "~" for c in upstream_key):
            raise SystemExit(f"notes-proxy: --upstream-key-env {args.upstream_key_env}: that environment variable "
                             f"holds no key (it is unset or empty, or has a space or a character a header cannot "
                             f"carry)")
    config = json.loads(args.notes_config) if args.notes_config else {}
    if not isinstance(config, dict):
        raise SystemExit("--notes-config is a JSON object of notes.py settings")
    config["cap"] = args.cap
    engine = notes = None
    try:
        if args.tensorfold:
            try:
                engine = tfchild.TensorFoldChild(args.tensorfold, extra=args.tensorfold_arg or (),
                                                 binary=args.tensorfold_bin, log=log,
                                                 quiet=args.tensorfold_quiet,
                                                 restart_max=args.tensorfold_restart_max,
                                                 timeout=args.timeout)
            except tfchild.TensorFoldError as e:
                raise SystemExit(f"notes-proxy: {e}")
            tok = tfchild.ChildTokenizer(engine)
            args.upstream = engine.url
        else:
            tok = nm.load_tokenizer(args.tokenizer)
        head = {"abstain_line": args.abstain_line, "framing": args.framing}
        ProxyStore(tok, config=config, **head)          # the settings are checked now, before any store is opened
        if many:
            who = users.Users(args.users)
            users.make_room(len(who))                   # every user's open files fit
            notes = users.Owners(args.store_dir, tok, config=config, **head)
        else:
            who = None
            notes = Notes(args.store, args.owner, tok, config=config, **head)
        cfg = Config(args.upstream, args.owner, write=args.write, join_system=args.join_system, timeout=args.timeout,
                     allow_origins=args.allow_origin or (), local_only=not args.allow_remote, log=log,
                     upstream_key=upstream_key, isolate_cache=many and not args.share_upstream_cache, engine=engine)
        return Proxy((args.host, args.port), cfg, notes, who), notes
    except BaseException:
        if notes is not None:
            notes.close()
        if engine is not None:
            engine.close()
        raise


def parser():
    p = argparse.ArgumentParser(prog="python -m notes_proxy", description="TensorFold Notes in front of any "
                                "OpenAI-compatible server. One owner (--owner, --store), or many users (--users, "
                                "--store-dir), each named by their key.")
    p.add_argument("--upstream", default=None, help="the inference server's base URL, e.g. http://127.0.0.1:8080")
    p.add_argument("--tensorfold", default=None, metavar="MODEL",
                   help="start TensorFold on this model (local dir or org/name) and use it as the upstream")
    p.add_argument("--tensorfold-bin", default="tensorfold", help="TensorFold executable (default: tensorfold on PATH)")
    p.add_argument("--tensorfold-arg", action="append", default=None,
                   help="one extra TensorFold serve flag from the allowlist (repeatable). Use equals so argparse "
                        "does not eat the flag: --tensorfold-arg=--no-drafts")
    p.add_argument("--tensorfold-quiet", type=float, default=tfchild.QUIET, metavar="S",
                   help="seconds of idle after the last erase before one TensorFold restart (default 10)")
    p.add_argument("--tensorfold-restart-max", type=float, default=tfchild.RESTART_MAX, metavar="S",
                   help="if traffic is still in flight this many seconds after the first pending erase, refuse new "
                        "requests so the restart can run (default 60)")
    p.add_argument("--owner", help="one owner: whose notes this proxy serves")
    p.add_argument("--store", help="one owner: the notes journal file (made if missing)")
    p.add_argument("--users", help="many users: the users file (see add-user). Each request's key names its owner")
    p.add_argument("--store-dir", help="many users: the folder of their journals, NAME.jsonl each (made if missing)")
    p.add_argument("--upstream-key-env", metavar="NAME",
                   help="send the upstream the key held in this environment variable (with many users and without "
                        "it, the upstream gets no Authorization at all)")
    p.add_argument("--share-upstream-cache", action="store_true",
                   help="many users: let the upstream reuse one user's cached prompt for another. Faster, but a user "
                        "can then test guesses about another user's notes (off: every request asks for no reuse)")
    p.add_argument("--tokenizer", default=None,
                   help="the upstream model's tokenizer.json (or its folder): the 512-token cap is counted with it; "
                        "\"fake\" only for tests. Not needed with --tensorfold (the child tokenizes)")
    p.add_argument("--cap", type=int, default=512, help="the block's token cap, framing included (default 512)")
    p.add_argument("--abstain-line", default=None, help="a line of text joined to the block's first line (off)")
    p.add_argument("--framing", choices=("quoted", "firm"), default="quoted",
                   help="the block's first paragraph: quoted, notes.py's own (the default), or firm, which also says a "
                        "note is right when it disagrees with what the model remembers")
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


def add_user_main(argv):
    p = argparse.ArgumentParser(prog="python -m notes_proxy add-user",
                                description="Add a user to a users file and print their new key, once.")
    p.add_argument("--users", required=True, help="the users file (made if missing)")
    p.add_argument("--owner", required=True, help="the user's name: 1 to 32 of a-z, 0-9, _ and -")
    p.add_argument("--store-dir", required=True,
                   help="the folder the proxy keeps the notes in: a name whose notes are still there is refused")
    p.add_argument("--reuse-notes", action="store_true",
                   help="give the new key the notes kept under this name (the same person, with a new key)")
    a = p.parse_args(argv)
    try:
        key = users.add_user(a.users, a.owner, store_dir=a.store_dir, reuse_notes=a.reuse_notes)
    except users.UsersError as e:
        raise SystemExit(f"notes-proxy: {e}")
    print(f"notes-proxy: added {a.owner} to {a.users}. Their key is on the next line. It is shown only this once: the "
          f"file keeps only its SHA-256. Restart the proxy to let them in.", file=sys.stderr, flush=True)
    print(key, flush=True)


def remove_user_main(argv):
    p = argparse.ArgumentParser(prog="python -m notes_proxy remove-user",
                                description="Take a user out of a users file. Say what happens to their notes.")
    p.add_argument("--users", required=True, help="the users file")
    p.add_argument("--owner", required=True, help="the user's name")
    what = p.add_mutually_exclusive_group(required=True)
    what.add_argument("--keep-notes", action="store_true", help="leave their journal in the store folder")
    what.add_argument("--erase-notes", action="store_true",
                      help="delete their journal too (needs --store-dir, their journal in it, and no proxy serving it)")
    p.add_argument("--store-dir", help="the store folder that holds their notes (with --keep-notes, to say what is "
                                       "kept)")
    a = p.parse_args(argv)
    if a.erase_notes and not a.store_dir:
        p.error("--erase-notes needs --store-dir")
    try:
        erased = users.remove_user(a.users, a.owner, "erase" if a.erase_notes else "keep",
                                   store_dir=a.store_dir if a.erase_notes else None)
    except users.UsersError as e:
        raise SystemExit(f"notes-proxy: {e}")
    if a.erase_notes:
        notes = f"their notes at {erased} are erased"
    elif a.store_dir:
        j = users.journal_path(a.store_dir, a.owner)
        notes = (f"their notes are kept, at {j}" if os.path.lexists(j) else
                 f"there are no notes of theirs in {os.path.dirname(j)} to keep")
    else:
        notes = f"their notes, if any, are kept in the store folder as {a.owner}.jsonl"
    then = "" if a.erase_notes else " Restart the proxy to shut them out."   # an erase ran with no proxy serving
    print(f"notes-proxy: removed {a.owner} from {a.users}, and {notes}.{then}", file=sys.stderr, flush=True)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["seed"]:
        return seed_main(argv[1:])
    if argv[:1] == ["add-user"]:
        return add_user_main(argv[1:])
    if argv[:1] == ["remove-user"]:
        return remove_user_main(argv[1:])
    p = parser()
    args = p.parse_args(argv)
    err = usage_error(args)
    if err:
        p.error(err)
    try:
        srv, notes = build(args)
    except (StoreError, nm.NotesError, ValueError, users.UsersError) as e:
        raise SystemExit(f"notes-proxy: {e}")
    h, pt = srv.server_address[:2]
    if srv.users is None:
        who, tok = f"owner {args.owner}, {notes.count()} notes", notes.store.tok
    else:
        n = len(srv.users)
        who, tok = f"{n} user{'' if n == 1 else 's'} ({args.users}), their notes in {notes.store_dir}", notes.tok
    key = f", upstream key from ${args.upstream_key_env}" if args.upstream_key_env else ""
    print(f"notes-proxy: {who}, tokenizer {tok.describe()}, cap {args.cap}, writes from chat "
          f"{'on' if args.write else 'off'}{key}; listening on http://{h}:{pt}, forwarding to "
          f"{srv.cfg.scheme}://{srv.cfg.host}:{srv.cfg.port}{srv.cfg.base}", file=sys.stderr, flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
        notes.close()
        if srv.cfg.engine is not None:
            srv.cfg.engine.close()


if __name__ == "__main__":
    main()
