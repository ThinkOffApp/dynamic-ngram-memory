"""Supervise a TensorFold child for the Notes Proxy.

The proxy starts TensorFold from a fixed flag allowlist, with a clean environment. The API key is passed in
TENSORFOLD_API_KEY, never on the command line. After an erase, the child is restarted once the proxy is idle, after a
quiet window (default 10 s), so several erases share one restart. If traffic never goes idle,
new requests are refused after a max wait (default 60 s) so the restart can run. Token counts go through the child's POST /v1/tokenize
({"prompt"} -> {"tokens", "count"}); piece text through POST /v1/detokenize ({"tokens"} -> {"prompt"}). A missing or
wrong tokenize reply is a refusal, never FakeTokenizer. SIGTERM/SIGHUP on the proxy, atexit, and a parent-death
watchdog all run close(), so the child is not left behind.
"""
from __future__ import annotations

import atexit
import http.client
import json
import os
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time

ALLOWED_FLAGS = {
    "--name": "value",
    "--alias": "value",
    "--context": "value",
    "--max-tokens": "value",
    "--temperature": "value",
    "--no-thinking": "flag",
    "--thinking": "flag",
    "--no-drafts": "flag",
    "--backend": "value",
    "--prompt-cache-gib": "value",
    "--no-update-check": "flag",
    "--parallel": "value",
    "--kv-dtype": "value",
}

# flags that would bind the child on the LAN, follow a remote master, or take extra network
FORBIDDEN = {
    "--host", "--port", "--vision-urls", "--snapshot-dir", "--master", "--master-port",
    "--tp", "--rank", "--spill-gib", "--ssd-experts",
}

MODEL_ID = __import__("re").compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")
QUIET = 10.0         # seconds of idle after the last erase before one restart
RESTART_MAX = 60.0   # then refuse new requests so a busy proxy still restarts
BACKOFF_START = 0.4
BACKOFF_CAP = 8.0
# sibling of the TensorFold child: argv[1] is the pipe fd, argv[2] the child's pid (session leader).
# EOF on the pipe means the proxy is gone (close(), exit, or SIGKILL); then SIGTERM the child's session.
_WATCH_PARENT = (
    "import os,signal,sys,time\n"
    "r,c=int(sys.argv[1]),int(sys.argv[2])\n"
    "try:\n"
    "    while os.read(r,1): pass\n"
    "except OSError: pass\n"
    "try: os.kill(c,0)\n"
    "except OSError: sys.exit(0)\n"
    "try: os.killpg(c,signal.SIGTERM)\n"
    "except OSError: sys.exit(0)\n"
    "for _ in range(20):\n"
    "    time.sleep(0.1)\n"
    "    try: os.kill(c,0)\n"
    "    except OSError: sys.exit(0)\n"
    "try: os.killpg(c,signal.SIGKILL)\n"
    "except OSError: pass\n"
)


def _parent_death_preexec():
    """Linux: the child gets SIGTERM when the proxy dies, even if the watchdog has not run yet."""
    try:
        import ctypes
        import ctypes.util
        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
        libc.prctl(1, int(signal.SIGTERM), 0, 0, 0)  # PR_SET_PDEATHSIG
    except Exception:
        pass


class TensorFoldError(Exception):
    """The child could not be started, reached, or restarted."""


def free_loopback_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def check_model(model):
    if os.path.isdir(model):
        return os.path.realpath(model)
    if MODEL_ID.match(model):
        return model
    raise TensorFoldError("the TensorFold model is a local directory or a Hugging Face repo id (org/name)")


def check_extra(extra):
    """`extra` is a list of argv tokens after the model. Only ALLOWED_FLAGS. Returns the kept list."""
    out, i = [], 0
    extra = list(extra or ())
    while i < len(extra):
        tok = extra[i]
        if tok in FORBIDDEN or tok.startswith("--host=") or tok.startswith("--port="):
            raise TensorFoldError(f"TensorFold flag {tok} is not on the proxy allowlist")
        name, val = tok, None
        if tok.startswith("--") and "=" in tok:
            name, val = tok.split("=", 1)
        kind = ALLOWED_FLAGS.get(name)
        if kind is None:
            raise TensorFoldError(f"TensorFold flag {name} is not on the proxy allowlist")
        if kind == "flag":
            if val is not None:
                raise TensorFoldError(f"{name} takes no value")
            out.append(name)
            i += 1
            continue
        if val is not None:
            out.extend([name, val])
            i += 1
            continue
        if i + 1 >= len(extra):
            raise TensorFoldError(f"{name} needs a value")
        nxt = extra[i + 1]
        if nxt in FORBIDDEN or nxt.startswith("--"):
            raise TensorFoldError(f"{name} needs a value")
        out.extend([name, nxt])
        i += 2
    return out


class ChildTokenizer:
    """Token counts from the child's POST /v1/tokenize; piece text from POST /v1/detokenize. Refuses FakeTokenizer."""

    def __init__(self, engine):
        self.engine = engine
        self.name = "tensorfold-child"
        self._piece = {}

    def encode(self, text):
        self.engine.begin()
        try:
            st, body = self.engine.request("POST", "/v1/tokenize", {"prompt": text})
        finally:
            self.engine.end()
        if st != 200 or not isinstance(body, dict) or not isinstance(body.get("tokens"), list):
            raise TensorFoldError("TensorFold /v1/tokenize did not return tokens; refusing FakeTokenizer")
        try:
            return [int(i) for i in body["tokens"]]
        except (TypeError, ValueError) as e:
            raise TensorFoldError("TensorFold /v1/tokenize returned a token that is not an int") from e

    def piece(self, i):
        """The child's text for token id `i`, cached. notes.py key() and wordy() need the piece, not the id."""
        try:
            i = int(i)
        except (TypeError, ValueError) as e:
            raise TensorFoldError("TensorFold piece() needs an int token id") from e
        if i in self._piece:
            return self._piece[i]
        self.engine.begin()
        try:
            st, body = self.engine.request("POST", "/v1/detokenize", {"tokens": [i]})
        finally:
            self.engine.end()
        if st != 200 or not isinstance(body, dict) or not isinstance(body.get("prompt"), str):
            raise TensorFoldError("TensorFold /v1/detokenize did not return prompt")
        self._piece[i] = body["prompt"]
        return self._piece[i]

    def describe(self):
        return {"name": self.name, "upstream": self.engine.url}


class TensorFoldChild:
    """One TensorFold `serve` process. Erases schedule one restart after a quiet idle window; spawn failures retry."""

    def __init__(self, model, extra=(), binary="tensorfold", timeout=120.0, log=None, quiet=QUIET,
                 restart_max=RESTART_MAX):
        self.model = check_model(model)
        self.extra = check_extra(extra)
        self.binary = binary if os.path.sep in binary else shutil.which(binary) or binary
        self.timeout = timeout
        if quiet < 0 or restart_max < 0:
            raise TensorFoldError("--tensorfold-quiet and --tensorfold-restart-max must be >= 0")
        self.quiet = quiet
        self.restart_max = restart_max
        self.log = log or (lambda line: print(line, flush=True))
        self.key = secrets.token_urlsafe(32)
        self._tmpdir = tempfile.mkdtemp(prefix="notes-proxy-tf-")
        self.port = None
        self.proc = None
        self.url = None
        self._env = None
        self._logf = None
        self._spawn = threading.Lock()          # stop/start: one restarter or close at a time
        self._gate = threading.Condition()      # inflight, pending, serving, seq
        self.inflight = 0
        self._serving = False
        self._down = False
        self._stopped = False
        self._restart_pending = False
        self._restart_running = False
        self._quiet_until = 0.0
        self._force_at = 0.0
        self._drain = False
        self._erase_seq = 0
        self._restarted_through = 0
        self._next_retry = 0.0
        self._retry_after = 1
        self._life_w = None
        self._watch = None
        try:
            self.start()
        except BaseException:
            self.close()
            raise
        atexit.register(self.close)

    def start(self):
        with self._spawn:
            self._spawn_locked()
        with self._gate:
            self._serving = True
            self._down = False
            self._gate.notify_all()

    def _spawn_locked(self):
        if self._stopped:
            raise TensorFoldError("TensorFold was stopped")
        self.port = free_loopback_port()
        cmd = [self.binary, "serve", self.model, "--host", "127.0.0.1", "--port", str(self.port),
               "--no-update-check", *self.extra]
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin:/usr/sbin:/sbin"),
            "HOME": os.environ.get("HOME", ""),
            "LANG": os.environ.get("LANG", "C"),
            "LC_ALL": os.environ.get("LC_ALL", "C"),
            "TENSORFOLD_NO_UPDATE_CHECK": "1",
            "TENSORFOLD_API_KEY": self.key,
        }
        for k in ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "HF_HUB_CACHE", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN",
                  "TMPDIR", "XDG_CACHE_HOME", "MLX_METAL_PATH"):
            if k in os.environ:
                env[k] = os.environ[k]
        self._env = env
        if self._logf is not None:
            try:
                self._logf.close()
            except OSError:
                pass
            self._logf = None
        self._logf = open(os.path.join(self._tmpdir, "child.log"), "wb")
        preexec = _parent_death_preexec if sys.platform.startswith("linux") else None
        self.proc = subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL, stdout=self._logf,
                                     stderr=subprocess.STDOUT, start_new_session=True, preexec_fn=preexec)
        self.url = f"http://127.0.0.1:{self.port}"
        self._watch_parent(self.proc.pid)
        self._note(f"notes-proxy: started TensorFold pid {self.proc.pid} on {self.url}")
        deadline = time.time() + self.timeout
        last = None
        while time.time() < deadline:
            if self._stopped:
                self._stop_locked()
                raise TensorFoldError("TensorFold was stopped")
            if self.proc.poll() is not None:
                raise TensorFoldError(f"TensorFold exited {self.proc.returncode} before it became ready")
            try:
                st, body = self._http(self.port, self.key, "GET", "/health", None, timeout=2.0)
                if st == 200:
                    self._note(f"notes-proxy: TensorFold ready ({body.get('model') if isinstance(body, dict) else st})")
                    return
                last = st
            except TensorFoldError as e:
                last = str(e)
            time.sleep(0.05)
        self._stop_locked()
        raise TensorFoldError(f"TensorFold did not answer /health within {self.timeout:.0f}s ({last})")

    def _stop_locked(self):
        p = self.proc
        self.proc = None
        if p is not None and p.poll() is None:
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except OSError:
                p.terminate()
            try:
                p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except OSError:
                    p.kill()
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
        self._stop_watchdog()

    def _watch_parent(self, child_pid):
        """A sibling that SIGTERMs the child's session when this process dies (pipe EOF), including SIGKILL."""
        self._stop_watchdog()
        try:
            r, w = os.pipe()
        except OSError:
            return
        try:
            os.set_inheritable(r, True)
            self._watch = subprocess.Popen(
                [sys.executable, "-c", _WATCH_PARENT, str(r), str(child_pid)],
                pass_fds=(r,), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, start_new_session=True)
            self._life_w = w
            w = None
        except (OSError, ValueError):
            self._watch = None
            self._life_w = None
        finally:
            try:
                os.close(r)
            except OSError:
                pass
            if w is not None:
                try:
                    os.close(w)
                except OSError:
                    pass

    def _stop_watchdog(self):
        # Kill the watcher first so its parent-death SIGTERM/SIGKILL cannot race a planned restart,
        # and so close() does not wait out the watcher's drain.
        p, self._watch = self._watch, None
        if p is not None and p.poll() is None:
            try:
                p.kill()
                p.wait(timeout=1)
            except (OSError, subprocess.TimeoutExpired):
                pass
        w, self._life_w = self._life_w, None
        if w is not None:
            try:
                os.close(w)
            except OSError:
                pass

    def close(self):
        with self._gate:
            self._stopped = True
            self._restart_pending = False
            self._serving = False
            self._drain = False
            self._gate.notify_all()
        with self._spawn:
            self._stop_locked()
            if self._logf is not None:
                try:
                    self._logf.close()
                except OSError:
                    pass
                self._logf = None
        try:
            shutil.rmtree(self._tmpdir, ignore_errors=True)
        except OSError:
            pass

    def begin(self):
        """Take a slot for an upstream call. Raises at once when the child is down or mid-restart; never waits."""
        with self._gate:
            if self._stopped or self._down or not self._serving or self._drain:
                raise TensorFoldError("TensorFold is not ready")
            self.inflight += 1

    def end(self):
        with self._gate:
            self.inflight = max(0, self.inflight - 1)
            if self.inflight == 0:
                self._gate.notify_all()

    def after_erase(self, reply):
        """Mark a restart. Always returns pending: the caller does not wait for the model to reload."""
        with self._gate:
            now = time.time()
            if not self._restart_pending:
                self._force_at = now + self.restart_max
            self._erase_seq += 1
            self._restart_pending = True
            self._quiet_until = now + self.quiet
            if not self._restart_running and not self._stopped:
                self._restart_running = True
                threading.Thread(target=self._restart_loop, daemon=True).start()
            self._gate.notify_all()
        reply = dict(reply)
        reply["state"] = "pending"
        reply["complete"] = False
        reply["engine"] = {"state": "pending", "restarted": False}
        remaining = list(reply.get("remaining") or [])
        if not any("Python" in r or "freed" in r for r in remaining):
            remaining.append("memory this process freed is not overwritten (Python strings)")
        reply["remaining"] = remaining
        return reply

    def _restart_loop(self):
        backoff = BACKOFF_START
        try:
            while True:
                try:
                    with self._gate:
                        while True:
                            if self._stopped or not self._restart_pending:
                                return
                            now = time.time()
                            if now >= self._force_at and self.inflight > 0 and not self._drain:
                                self._drain = True
                                self._note("notes-proxy: TensorFold restart waited long enough; pausing new requests")
                                self._gate.notify_all()
                            idle = self.inflight == 0 and now >= self._quiet_until
                            forced = self.inflight == 0 and now >= self._force_at
                            if idle or forced:
                                erase_seen = self._erase_seq
                                break
                            wait = 0.2
                            if self.inflight == 0:
                                wait = min(self._quiet_until, self._force_at) - now
                            elif not self._drain:
                                wait = min(0.2, max(0.05, self._force_at - now))
                            self._gate.wait(timeout=max(0.05, wait))
                    with self._spawn:
                        if self._stopped:
                            return
                        with self._gate:
                            self._serving = False
                            self._down = False
                            self._gate.notify_all()
                        self._stop_locked()
                        if self._stopped:
                            return
                        self._spawn_locked()
                    with self._gate:
                        self._serving = True
                        self._down = False
                        self._drain = False
                        self._restarted_through = erase_seen
                        backoff = BACKOFF_START
                        self._retry_after = 1
                        if self._erase_seq == erase_seen:
                            self._restart_pending = False
                        self._note("notes-proxy: TensorFold restarted after erase")
                        self._gate.notify_all()
                        if not self._restart_pending or self._stopped:
                            return
                except Exception as e:
                    self._note(f"notes-proxy: TensorFold restart failed: {e}")
                    try:
                        with self._spawn:
                            self._stop_locked()
                    except Exception:
                        pass
                    with self._gate:
                        if self._stopped or not self._restart_pending:
                            return
                        self._serving = False
                        self._down = True
                        self._retry_after = max(1, int(backoff + 0.999))
                        self._next_retry = time.time() + backoff
                        self._gate.notify_all()
                    deadline = time.time() + backoff
                    while time.time() < deadline:
                        with self._gate:
                            if self._stopped or not self._restart_pending:
                                return
                            self._gate.wait(timeout=max(0.05, deadline - time.time()))
                    backoff = min(backoff * 2, BACKOFF_CAP)
        finally:
            with self._gate:
                self._restart_running = False
                if self._restart_pending and not self._stopped:
                    self._restart_running = True
                    threading.Thread(target=self._restart_loop, daemon=True).start()
                self._gate.notify_all()

    def _note(self, line):
        try:
            self.log(line)
        except Exception:
            pass

    def health(self):
        with self._gate:
            if self._stopped:
                return {"status": "down", "engine": "stopped"}
            if self._down:
                return {"status": "down", "engine": "down"}
            if self._drain or not self._serving:
                return {"status": "restarting", "engine": "draining" if self._drain else "restarting"}
            return {"status": "ok", "engine": "ready"}

    def erase_status(self):
        with self._gate:
            complete = (self._serving and not self._down and not self._stopped
                        and self._restarted_through >= self._erase_seq)
            if self._down:
                state, engine = "down", "down"
            elif not complete:
                state = "pending"
                engine = "ready" if self._serving else "restarting"
            else:
                state, engine = "complete", "ready"
            return {"complete": complete, "state": state, "engine": engine,
                    "erase_seq": self._erase_seq, "restart_seq": self._restarted_through}

    def retry_after(self):
        with self._gate:
            if self._down:
                wait = int(self._next_retry - time.time() + 0.999)
                return max(1, wait, self._retry_after)
            return 1

    def endpoint(self):
        with self._gate:
            if self.port is None:
                raise TensorFoldError("TensorFold is not running")
            return "127.0.0.1", self.port

    def auth_headers(self):
        return {"Authorization": f"Bearer {self.key}"}

    def request(self, method, path, body=None, timeout=None):
        """HTTP to the child. Snapshots port and key, then does not hold the spawn lock (tokenize must not serialize)."""
        with self._gate:
            port, key, serving = self.port, self.key, self._serving
        if not serving or port is None:
            raise TensorFoldError("TensorFold is not running")
        return self._http(port, key, method, path, body, timeout=timeout)

    def _http(self, port, key, method, path, body, timeout=None):
        raw = None if body is None else json.dumps(body).encode()
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout or min(self.timeout, 10.0))
        try:
            headers = {"Authorization": f"Bearer {key}"}
            if raw is not None:
                headers["Content-Type"] = "application/json"
                headers["Content-Length"] = str(len(raw))
            conn.request(method, path, body=raw, headers=headers)
            r = conn.getresponse()
            data = r.read()
        except (OSError, http.client.HTTPException) as e:
            raise TensorFoldError(f"TensorFold {method} {path} failed ({type(e).__name__})") from e
        finally:
            conn.close()
        try:
            parsed = json.loads(data) if data else None
        except ValueError:
            parsed = None
        return r.status, parsed if parsed is not None else {}


def inject_store_false(raw):
    """Request bytes with store:false set. Unchanged when the body is not a JSON object."""
    try:
        obj = json.loads(raw) if raw else None
    except ValueError:
        return raw
    if not isinstance(obj, dict):
        return raw
    obj["store"] = False
    return json.dumps(obj, ensure_ascii=False).encode()
