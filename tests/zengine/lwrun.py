#!/usr/bin/env python3
"""Run one Living Weights test set (written by `lwlogic.py gen`) against a real TensorFold server, end to end.

  python3 zengine/lwrun.py SET.json --bin PATH --model-src SNAPSHOT_DIR --work DIR --port N
      [--phases baseline,teach,del-A,...] [--delete-mode reteach|zero|both] [--thinking off,on]
      [--max-tokens-off 160] [--max-tokens-on 1500] [--teach-probe slide|restart|both]
      [--resume] [--dry-run] [--judge-only] [--cleanup] [--fresh-empty-reteach]   (more: --help)
      [--reference in_context,replacement_memory] [--reference-of teach|all|PHASE,...]   (opt-in reference phases)
      [--reference notes,notes_pocket] [--notes-tokenizer PATH|fake] [--notes-cap 512]  (the note memory, Phase 1)
  python3 zengine/lwrun.py --sets SET.json [SET.json ...] --bin PATH --model-src SNAPSHOT_DIR --work DIR --port N ...
      the same for several sets, one after another, each in WORK/SET_ID; one summary at the end
      (WORK/sets-summary.json, and a table in WORK/sets.log)

Steps, in this order:
  baseline       serve SNAPSHOT_DIR read-only, without --slide (or a fresh copy: --baseline-from copy); probe.
  teach          fresh copy (macOS cp -cRL, Linux cp -RL), serve with --slide, teach every lesson as its own
                 POST /v1/slide/learn {"text": lesson}, in order, each waited for to [DONE] and to the end of the
                 save that follows it; record per lesson the wall time, recalled, message, the fact ids (SSE and
                 the diff of GET /v1/slide/graph), each fact's neurons and the server's own log line after the
                 save (save_log); save every graph; then probe.
  DEL.zero       copy of the taught folder; the deleted lessons' facts zeroed by philip-pentatonic's removal
                 method, the way our earlier zero_fact.py did it
                 (shared_experts.up_proj rows own+neuron: weight, scales, biases); serve without --slide; probe.
                 Records what zeroing cannot do here (zero_limits): a kept fact that a deleted one replaced (unless a
                 kept fact replaced that one in turn), and deleted facts with nothing left to zero. When the teach
                 step stored nothing (every lesson missed, every save 0 bytes, no pocket in the taught folder) the
                 step is "not applicable": finished with the reason on file, no copy made, nothing served.
  DEL.reteach    fresh copy, teach only the lessons the phase keeps (the "never taught" reference); probe.
                 An empty store is the original model: the baseline answers are reused, filed under the phase
                 (--fresh-empty-reteach, or no complete baseline on file: served like baseline).
  PHASE.in_context          (only with --reference) the untaught model, served like baseline, with the phase's
                 lessons written into the system prompt (in_context_prompt, lw-in-context/1); probe.
  PHASE.replacement_memory  (only with --reference) no model: a lookup that keeps the latest value per (entity,
                 attribute) and answers from it (memory_slots, memory_answer); its answers are written and judged
                 like a probe's. Both are references for the pocket, not ceilings (TEST-LOGIC.md section 8).
  PHASE.notes               (only with --reference notes) the untaught model, served like baseline, with only the
                 notes retrieved for each question in its system prompt (zengine/notes.py, lw-notes/1): the phase's
                 lessons written into a note store, each question's notes found by token n-grams under a token cap
                 (--notes-cap, framing included); no notes found, no system message. Per question the note ids, the
                 tokens and the lookup time go to WORK/phases/PHASE.notes/notes.jsonl; probe.
  PHASE.notes_pocket        (only with --reference notes_pocket) the same notes, given to the pocket: the folder the
                 phase's own step taught (teach; for a delete phase its reteach copy, or its zero copy without reteach),
                 served without --slide; probe. docs/NOTES-LAYER.md.
Probing: one fresh chat per question, temperature 0, thinking off (max_tokens 160) then on (1500), as
our earlier #678 probe script did. Answers go to WORK/phases/KEY/answers*.jsonl, one line each:
  {qid, phase, mode, thinking, question, content, reasoning_content, finish, ms, ...}
`phase` is always the set's own phase name, so `lwlogic.py judge` accepts it; zero and reteach answers of one
phase go to separate files (the judge refuses unknown phase names such as "del-A:zero").
After each step: python3 lwlogic.py judge SET answers.jsonl --json verdicts.json (lwlogic.py next to this file,
or --lwlogic PATH), report kept beside it.
At the end: combined reports per delete mode, zero-vs-reteach agreement, WORK/summary.json, and the
kept-stable inputs: WORK/kept-stable/MODE/teach.jsonl and WORK/kept-stable/MODE/PHASE.jsonl per delete mode, so
the judge can compare each delete phase with the same model's teach answers (manifest.json says which files).
With reference phases also WORK/reference/report.txt and summary.json `reference`: per group (CURRENT, WHO_HAS,
negation, dates, two-step when present, and the judge's groups) the pocket's, in_context's and replacement_memory's
pass counts and the pocket gap (in_context rate minus pocket rate), and for each CHANGE and RETRACT lesson whose
effect today's answers can show: corrected only, stale only, both or neither, per answer source and thinking mode.

Robustness: one server at a time on --port (checked free first); WORK/server.pid; servers stopped through their
Popen handle (terminate, then kill after --stop-timeout); atexit and SIGINT/SIGTERM/SIGHUP handlers; each server
also carries an alarm (--server-max-hours) and, on Linux, a parent-death signal, so a SIGKILLed runner cannot leave
it running for long. Copies are never deleted on macOS (Trash rule): their paths and sizes are printed at the end.
On Linux --cleanup removes the runner's own copies once judged (18.5 GB each), the taught folder once every zero
step is done or not applicable, and a zero copy at once when its zeroing was skipped (nothing was written to it).
Every write in place (zero mode) goes through open_for_inplace_write: a shard that is a symlink or a hard link
(st_nlink > 1) is refused, every shard of the step checked before its first byte.

Standard library only. macOS and Linux.
"""
import argparse
import array
import atexit
import datetime
import errno
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import signal
import socket
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
IS_MAC = sys.platform == "darwin"
IS_LINUX = sys.platform.startswith("linux")
GIB = 1 << 30


# ============================================================================================ basics
class RunError(Exception):
    """A step cannot go on; the message says why."""


class Interrupted(BaseException):
    """SIGINT, SIGTERM or SIGHUP: unwind, stop the server, write the summary."""


def utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


class Log:
    def __init__(self):
        self.path = None
        self.prefix = ""

    def __call__(self, msg):
        for i, part in enumerate(str(msg).split("\n")):
            line = f"{utc()} {self.prefix}{part}" if i == 0 else f"{' ' * 21}{self.prefix}{part}"
            print(line, flush=True)
            if self.path:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")


LOG = Log()


def read_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_bytes(path):
    """Logical bytes under `path`, symlinks followed (an HF snapshot holds links to its blobs)."""
    total = 0
    for root, _dirs, files in os.walk(path):
        for x in files:
            try:
                total += os.stat(os.path.join(root, x)).st_size
            except OSError:
                pass
    return total


def fingerprint(path):
    out = {}
    for root, _dirs, files in os.walk(path):
        for x in files:
            p = os.path.join(root, x)
            try:
                s = os.stat(p)
                out[os.path.relpath(p, path)] = [s.st_size, s.st_mtime_ns]
            except OSError as e:
                out[os.path.relpath(p, path)] = str(e)
    return out


def free_bytes(path):
    p = os.path.abspath(path)
    while not os.path.exists(p):
        p = os.path.dirname(p)
    return shutil.disk_usage(p).free


def gb(n):
    return f"{n / 1e9:.1f} GB"


def tail(path, n=40):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 65536))
            lines = f.read().decode("utf-8", "replace").splitlines()[-n:]
        return "server log tail:\n" + "\n".join("  | " + x for x in lines)
    except OSError as e:
        return f"(no server log: {e})"


# The line 1.0.8 logs once a save has put the learned neurons in the shards, for example
#   info: slide pocket: the learned neurons are in the model's shards (261 MB written in 18.4 s); the last fact's
#   held-out answers do not all come back with them as written
# It is the server's own check of the saved neurons. A save that writes 0 bytes logs no such line: in our runs, a
# first lesson that was missed. A missed lesson after earlier facts still writes the shards and logs it. Lines that
# quote the model ('[tensorfold] slide: "..." is learned as ...') never start with "info: slide", so they cannot match.
SAVE_LINE = re.compile(r"^info: slide\b.*\bshards\b")
# 1.0.8 says a pocket is full by the learner's error name: pocket.zig fitAll returns error.PocketFull when no layer the
# fact's key may use has a free neuron left; learner.zig reports @errorName(e) and slide.zig sends it as the lesson's
# message, with `"recalled": false` (READ, tag v1.0.8). Other wordings are matched too, in case a later version says it.
POCKET_FULL = re.compile(r"\bPocketFull\b|\bpocket (?:is )?full\b", re.I)
SAVE_NUMS = re.compile(r"\((?P<mb>[\d.]+) MB written in (?P<s>[\d.]+) s\)")
SAVE_BACK = re.compile(r"held-out answers (do not (?:all )?)?come back\b")  # 1.0.8's two endings; any other: no verdict


def log_size(path):
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def log_lines_from(path, offset):
    """The complete lines a server has written to `path` from byte `offset` on (a line still being written is left
    out)."""
    try:
        with open(path, "rb") as f:
            f.seek(offset)
            text = f.read().decode("utf-8", "replace")
    except OSError:
        return []
    return text.split("\n")[:-1]


def parse_save_line(line):
    out = {"line": line}
    m = SAVE_NUMS.search(line)
    if m:
        out["mb"], out["seconds"] = float(m["mb"]), float(m["s"])
    m = SAVE_BACK.search(line)
    if m:
        out["held_out_back"] = m.group(1) is None
    return out


def find_save_line(path, offset):
    """The last post-save line (SAVE_LINE) in the log after `offset`, parsed, or None."""
    hits = [x.rstrip("\r") for x in log_lines_from(path, offset) if SAVE_LINE.search(x)]
    return parse_save_line(hits[-1]) if hits else None


def http(method, url, body=None, timeout=60):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def port_busy(port):
    """True when something listens on 127.0.0.1:port, or the port cannot be bound."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(1.0)
    try:
        if s.connect_ex(("127.0.0.1", port)) == 0:
            return True
    finally:
        s.close()
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind(("127.0.0.1", port))
        return False
    except OSError:
        return True
    finally:
        s.close()


def pid_command(pid):
    """The command line of `pid` (ps), or None when no such process exists."""
    try:
        r = subprocess.run(["ps", "-p", str(int(pid)), "-o", "command="], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    return r.stdout.strip() or None


# ============================================================================================ servers
LIVE = []          # servers this run started and has not stopped
STOPPING = [False]  # a stop is under way: further signals are noted, not raised


def on_signal(signum, _frame):
    name = signal.Signals(signum).name
    if STOPPING[0]:
        LOG(f"{name} received while stopping a server: ignored, the stop goes on")
        return
    raise Interrupted(name)


def stop_all(why="runner exiting"):
    for s in list(LIVE):
        try:
            s.stop(why)
        except Exception as e:  # noqa: BLE001  (never let one stop failure skip another)
            print(f"lwrun: could not stop server pid {getattr(s, 'pid', '?')}: {e}", file=sys.stderr)


def child_setup(max_seconds):
    """Runs in the server's child process before exec: an alarm (kept across exec) and, on Linux, a parent-death
    signal, so a server outlives a SIGKILLed runner by --server-max-hours at most (Linux: not at all)."""
    libc = None
    if IS_LINUX:
        try:
            import ctypes
            import ctypes.util
            libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
        except Exception:  # noqa: BLE001
            libc = None
    term = int(signal.SIGTERM)

    def setup():
        if max_seconds > 0:
            signal.alarm(int(max_seconds))
        if libc is not None:
            libc.prctl(1, term, 0, 0, 0)  # PR_SET_PDEATHSIG

    return setup


class Server:
    def __init__(self, run, folder, slide, log_path, graph_path=None, label=""):
        self.run, self.folder, self.slide = run, folder, slide
        self.log_path, self.graph_path, self.label = log_path, graph_path, label
        self.proc = None
        self.pid = None
        self.logf = None
        self.base = f"http://127.0.0.1:{run.port}"
        self.timings = {}

    def cmd(self):
        c = [self.run.bin, "serve", self.folder, "--host", "127.0.0.1", "--port", str(self.run.port),
             "--no-update-check"]
        if self.slide:
            c += ["--slide", "--slide-graph", self.graph_path]
        return c + list(self.run.args.server_arg or [])

    def start(self):
        r = self.run
        if self.slide:
            r.guard_teachable(self.folder)
        r.check_pidfile()
        if port_busy(r.port):
            raise RunError(f"port {r.port} is in use (or cannot be bound): refusing to start a server there")
        env = dict(os.environ)
        if not r.args.keep_env:
            dropped = [k for k in env if k.startswith("TF_SLIDE_") or k == "TF_CUDA_SEGMENTS"]
            for k in dropped:
                env.pop(k)
            if dropped:
                LOG(f"server env: dropped {', '.join(sorted(dropped))} (release defaults; --keep-env keeps them)")
        cmd = self.cmd()
        LOG(f"server start [{self.label}]: {' '.join(shlex.quote(x) for x in cmd)}")
        self.logf = open(self.log_path, "ab")
        self.logf.write(f"\n==== {utc()} lwrun: {' '.join(cmd)}\n".encode())
        self.logf.flush()
        t0 = time.time()
        self.proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=self.logf, stderr=subprocess.STDOUT,
                                     env=env, start_new_session=True, close_fds=True,
                                     preexec_fn=child_setup(r.args.server_max_hours * 3600))
        self.pid = self.proc.pid
        LIVE.append(self)
        r.write_pidfile(self, "running")
        try:
            self.wait_ready(t0)
        except BaseException:
            self.stop("startup failed")
            raise
        return self

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def wait_ready(self, t0):
        a = self.run.args
        deadline = t0 + a.startup_timeout
        info = None
        while True:
            rc = self.proc.poll()
            if rc is not None:
                raise RunError(f"server exited with code {rc} during startup\n{tail(self.log_path)}")
            try:
                st, body = http("GET", self.base + "/health", timeout=5)
                if st == 200:
                    try:
                        info = json.loads(body)
                    except ValueError:
                        info = {"raw": body[:200].decode("utf-8", "replace")}
                    break
            except (urllib.error.URLError, OSError):
                pass
            if time.time() > deadline:
                raise RunError(f"server not ready after {a.startup_timeout} s\n{tail(self.log_path)}")
            time.sleep(2)
        self.timings["ready_s"] = round(time.time() - t0, 1)
        wdead = time.time() + a.warm_timeout
        while isinstance(info, dict) and info.get("warming") and time.time() < wdead:
            time.sleep(2)
            try:
                st, body = http("GET", self.base + "/health", timeout=5)
                info = json.loads(body) if st == 200 else info
            except (urllib.error.URLError, OSError, ValueError):
                pass
        self.timings["warm_s"] = round(time.time() - t0, 1)
        if not self.alive():
            raise RunError(f"server answered /health, then exited\n{tail(self.log_path)}")
        LOG(f"server ready [{self.label}] pid {self.pid}: health in {self.timings['ready_s']} s"
            + (f", still warming after {a.warm_timeout} s" if isinstance(info, dict) and info.get("warming") else
               f", warm at {self.timings['warm_s']} s"))

    def stop(self, why=""):
        p = self.proc
        if p is None:
            return
        STOPPING[0] = True
        try:
            if p.poll() is None:
                LOG(f"server stop [{self.label}] pid {p.pid}" + (f": {why}" if why else ""))
                t0 = time.time()
                p.terminate()
                try:
                    p.wait(timeout=self.run.args.stop_timeout)
                except subprocess.TimeoutExpired:
                    LOG(f"server pid {p.pid} still up {self.run.args.stop_timeout} s after SIGTERM: SIGKILL via its Popen handle")
                    p.kill()
                    try:
                        p.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        LOG(f"WARNING: server pid {p.pid} did not exit after SIGKILL")
                self.timings["stop_s"] = round(time.time() - t0, 1)
            LOG(f"server [{self.label}] pid {p.pid} exited with code {p.returncode}")
            self.proc = None
            if self in LIVE:
                LIVE.remove(self)
            if self.logf:
                self.logf.close()
                self.logf = None
            self.run.write_pidfile(self, "stopped")
            for _ in range(30):  # the port free again before the next server
                if not port_busy(self.run.port):
                    break
                time.sleep(1)
        finally:
            STOPPING[0] = False

    # -------------------------------------------------------------------------------- slide API
    def graph(self):
        st, body = http("GET", self.base + "/v1/slide/graph", timeout=60)
        if st != 200:
            raise RunError(f"GET /v1/slide/graph: HTTP {st}: {body[:300]!r}")
        return json.loads(body)

    def delete_fact(self, fid):
        """DELETE /v1/slide/fact/{id}, the per-fact delete Ash Hart says comes after 1.0.8: (HTTP status, reply text).
        1.0.8 has no such route, so a 404 (or 405, 501) there means "no delete endpoint", not a failed delete."""
        st, body = http("DELETE", f"{self.base}/v1/slide/fact/{int(fid)}", timeout=self.run.args.save_timeout)
        return st, body.decode("utf-8", "replace")[:500]

    def wait_save(self, since_ms):
        """The save the server runs after a learn stream's [DONE] (slide.zig: `defer saveAll` runs after
        sse.done), waited for through the graph's model.save; the server must not be stopped while it writes."""
        a = self.run.args
        t0 = time.time()
        deadline = t0 + a.save_timeout
        seen_model = False
        last = None
        while True:
            if not self.alive():
                raise RunError(f"server exited while saving\n{tail(self.log_path)}")
            g = self.graph()
            model = g.get("model")
            seen_model = seen_model or isinstance(model, dict)
            sv = model.get("save") if isinstance(model, dict) else None
            last = sv
            if isinstance(sv, dict) and sv.get("state") in ("saved", "failed") and int(sv.get("at") or 0) >= since_ms - 5000:
                out = dict(sv)
                out["waited_s"] = round(time.time() - t0, 1)
                return out
            if not seen_model and time.time() - t0 > 30:
                LOG(f"WARNING: the graph shows no model.save; waiting {a.save_grace} s instead")
                time.sleep(a.save_grace)
                return {"state": "unobserved", "waited_s": round(time.time() - t0, 1)}
            if time.time() > deadline:
                return {"state": "timeout", "last": last, "waited_s": round(time.time() - t0, 1)}
            time.sleep(2)

    def save_line(self, offset, save):
        """The server's own line after a save (SAVE_LINE), from what it logged since `offset`. Waited for up to
        --save-line-wait s when the save wrote bytes; a save of 0 bytes (a missed first lesson) logs none: no wait."""
        wrote = isinstance(save, dict) and save.get("state") == "saved" and int(save.get("bytes") or 0) > 0
        deadline = time.time() + (self.run.args.save_line_wait if wrote else 0)
        while True:
            found = find_save_line(self.log_path, offset)
            if found or time.time() >= deadline:
                return found
            time.sleep(0.5)


# ============================================================================================ learning
def post_sse(url, body, read_timeout, deadline_s, on_event):
    """POST, then read server-sent events until [DONE] or EOF: (status, events, raw text, got_done)."""
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
    t0 = time.time()
    try:
        resp = urllib.request.urlopen(req, timeout=read_timeout)
    except urllib.error.HTTPError as e:
        return e.code, [], e.read().decode("utf-8", "replace"), False
    raw, events, done = [], [], False
    with resp:
        kind, data = None, []
        while True:
            if time.time() - t0 > deadline_s:
                raise RunError(f"lesson still streaming after {deadline_s} s")
            line = resp.readline()
            if not line:
                break
            s = line.decode("utf-8", "replace")
            raw.append(s)
            s = s.rstrip("\r\n")
            if s == "":
                if data:
                    payload = "\n".join(data)
                    if payload.strip() == "[DONE]":
                        done = True
                        break
                    try:
                        obj = json.loads(payload)
                    except ValueError:
                        obj = {"raw": payload}
                    ev = {"event": kind or "message", "data": obj, "t": round(time.time() - t0, 1)}
                    events.append(ev)
                    on_event(ev)
                kind, data = None, []
            elif s.startswith("event:"):
                kind = s[6:].strip()
            elif s.startswith("data:"):
                data.append(s[6:] if s.startswith("data: ") else s[5:])
        return resp.status, events, "".join(raw), done


def learn_one(run, srv, lesson, sdir, g_before, neurons_seen):
    a = run.args
    tag = lesson["name"]
    LOG(f"learn {tag} (lesson {lesson['id']}, {lesson.get('op')}): {lesson['text']!r}")
    t0 = time.time()
    since_ms = int(t0 * 1000)
    log_from = log_size(srv.log_path)  # this lesson's part of the server log starts here
    busy = 0

    def show(ev):
        d = ev["data"]
        LOG(f"  {tag} +{ev['t']:.0f}s {ev['event']}: {json.dumps(d, ensure_ascii=False)[:200]}")

    while True:
        try:
            status, events, raw, done = post_sse(srv.base + "/v1/slide/learn", {"text": lesson["text"]},
                                                 a.learn_timeout, a.lesson_deadline, show)
        except (urllib.error.URLError, OSError) as e:
            raise RunError(f"learn {tag}: the stream broke: {e}\n{tail(srv.log_path)}")
        if status == 503 and busy < a.busy_retries:
            busy += 1
            LOG(f"  {tag}: 503 busy ({raw.strip()[:120]}); retry {busy}/{a.busy_retries} in 10 s")
            time.sleep(10)
            continue
        break
    sse_file = os.path.join(sdir, f"learn-{tag}.sse")
    with open(sse_file, "w", encoding="utf-8") as f:
        f.write(raw)
    if status != 200:
        raise RunError(f"learn {tag}: HTTP {status}: {raw.strip()[:300]}")
    facts, order, failed = {}, [], None
    for ev in events:
        d = ev["data"] if isinstance(ev["data"], dict) else {}
        if ev["event"] == "fact":
            facts[d.get("id")] = {"id": d.get("id"), "text": d.get("text")}
            order.append(d.get("id"))
        elif ev["event"] == "learned":
            f = facts.setdefault(d.get("id"), {"id": d.get("id")})
            f.update({"recalled": d.get("recalled"), "steps": d.get("steps"), "message": d.get("message")})
        elif ev["event"] == "failed":
            failed = d.get("message") or "failed"
    stream_s = round(time.time() - t0, 1)
    save = srv.wait_save(since_ms) if order else None
    save_log = srv.save_line(log_from, save) if order else None
    g_after = srv.graph()
    gfile = os.path.join(sdir, f"graph-{tag}.json")
    write_json(gfile, g_after)
    before = {f["id"]: f for f in g_before.get("facts", [])}
    after = {f["id"]: f for f in g_after.get("facts", [])}
    for fid, f in after.items():
        if f.get("neurons"):
            neurons_seen[str(fid)] = f["neurons"]
    new_ids = sorted(set(after) - set(before))
    changed = [{"id": i, "from": before[i].get("state"), "to": after[i].get("state"), "text": after[i].get("text"),
                "message": after[i].get("message")}
               for i in sorted(set(before) & set(after)) if before[i].get("state") != after[i].get("state")]
    fl = [facts[i] for i in order]
    for f in fl:
        g = after.get(f["id"], {})
        f["state"] = g.get("state")
        f["neurons"] = g.get("neurons") or []
        f["questions"] = g.get("questions") or []
        if g.get("loss") is not None:  # 1.0.8's graph gives each fact its last teach loss
            f["loss"] = g.get("loss")
    rec = {
        "lesson": lesson["id"], "name": tag, "op": lesson.get("op"), "text": lesson["text"],
        "started": datetime.datetime.fromtimestamp(t0, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seconds": round(time.time() - t0, 1), "stream_seconds": stream_s, "http": status, "done": done,
        "busy_retries": busy, "facts": fl,
        "recalled": bool(fl) and all(f.get("recalled") is True for f in fl),
        "recalled_any": any(f.get("recalled") is True for f in fl),
        "message": "; ".join(f["message"] for f in fl if f.get("message")) or failed or "",
        "failed": failed, "fact_ids_sse": order, "fact_ids_graph": new_ids,
        "ids_agree": sorted(x for x in order if x is not None) == new_ids,
        "changed_states": changed, "save": save, "save_log": save_log,
        "pocket_full": any(POCKET_FULL.search(x or "") for x in [failed] + [f.get("message") for f in fl]),
        "server_log": {"path": srv.log_path, "from": log_from, "to": log_size(srv.log_path)},
        "sse_file": sse_file, "graph_file": gfile,
    }
    LOG(f"learned {tag} in {rec['seconds']} s: facts {order}, recalled "
        f"{[f.get('recalled') for f in fl]}" + (f", message: {rec['message']}" if rec["message"] else "")
        + (f"; changed earlier facts: {[(c['id'], c['to']) for c in changed]}" if changed else "")
        + (f"; save {save.get('state')} ({save.get('seconds')} s, {save.get('bytes')} bytes)" if save else ""))
    if save_log:
        LOG(f"  {tag} server after the save: {save_log['line']}")
    elif save and save.get("state") == "saved" and int(save.get("bytes") or 0) > 0:
        LOG(f"  WARNING {tag}: the save wrote {save.get('bytes')} bytes but the server log shows no post-save line "
            f"within {run.args.save_line_wait} s (save_log is null)")
    if not rec["ids_agree"]:
        LOG(f"  WARNING {tag}: SSE fact ids {order} differ from the graph diff {new_ids}")
    if not done:
        raise RunError(f"learn {tag}: the stream ended without [DONE]\n{tail(srv.log_path)}")
    if save and save.get("state") != "saved":
        raise RunError(f"learn {tag}: the save after it is {save.get('state')}: {save}; the folder may not hold it")
    return rec, g_after


def teach(run, srv, lesson_ids, sdir, st, save_state):
    keep = set(lesson_ids)
    lessons = [l for l in run.S["lessons"] if l["id"] in keep]  # teaching order is the set's order
    g = srv.graph()
    write_json(os.path.join(sdir, "graph-0-before.json"), g)
    if g.get("facts"):
        raise RunError(f"the slide graph of a fresh copy is not empty ({len(g['facts'])} facts): stale graph file?")
    neurons_seen = {}
    st["lessons"] = []
    t0 = time.time()
    for l in lessons:
        rec, g = learn_one(run, srv, l, sdir, g, neurons_seen)
        st["lessons"].append(rec)
        save_state()
    write_json(os.path.join(sdir, "graph-final.json"), g)
    st["teach_seconds"] = round(time.time() - t0, 1)
    st["fact_states"] = {str(f["id"]): f.get("state") for f in g.get("facts", [])}
    st["fact_neurons"] = neurons_seen
    st["fact_lesson"] = {str(fid): rec["lesson"] for rec in st["lessons"] for fid in rec["fact_ids_sse"]}
    st["refused_lessons"] = [rec["lesson"] for rec in st["lessons"] if not rec["recalled"]]
    st["partly_refused_lessons"] = [rec["lesson"] for rec in st["lessons"] if rec["recalled_any"] and not rec["recalled"]]
    st["taught"] = True
    save_state()
    LOG(f"taught {len(lessons)} lessons in {st['teach_seconds']} s; refused (not every fact recalled): "
        f"{[run.lesson_name(i) for i in st['refused_lessons']] or 'none'}; final fact states "
        f"{st['fact_states']}")


# ============================================================================================ probing
def ask(base, text, thinking, max_tokens, timeout, system=None):
    """One fresh chat. `system` (the in_context reference phase only) goes first, as a system message."""
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": text}]
    body = {"messages": msgs, "temperature": 0, "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": thinking}}
    t = time.time()
    st, raw = http("POST", base + "/v1/chat/completions", body, timeout)
    ms = int((time.time() - t) * 1000)
    if st != 200:
        raise RunError(f"HTTP {st}: {raw[:300]!r}")
    d = json.loads(raw)
    ch = d["choices"][0]
    m = ch.get("message") or {}
    return {"content": m.get("content") or "", "reasoning_content": m.get("reasoning_content") or "",
            "finish": ch.get("finish_reason"), "ms": ms, "usage": d.get("usage")}


def answered_pairs(path):
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done.add((r["qid"], bool(r["thinking"])))
    return done


def probe(run, srv, akey, phase, mode, path, errpath, system=None, extra=None):
    """Ask every question not yet on file. `system`: one system prompt for every question (in_context), or one per
    question, {qid: prompt or None} (the notes phases: None sends no system message). `extra`: {qid: fields} added to
    each answer line (the notes phases: the note ids and tokens given)."""
    a = run.args
    per_q = isinstance(system, dict)
    sys_sha = hashlib.sha256(system.encode()).hexdigest() if system and not per_q else None
    done = answered_pairs(path)
    want = run.pairs()
    todo = [(q, th) for th in run.thinking for q in run.S["questions"] if (q["qid"], th) not in done]
    if done:
        LOG(f"probe {akey}: {len(done)} of {len(want)} answers already on file, {len(todo)} to ask")
    t0 = time.time()
    with open(path, "a", encoding="utf-8") as f:
        for q, th in todo:
            last = None
            for attempt in range(1, a.chat_retries + 2):
                if not srv.alive():
                    raise RunError(f"server died while probing\n{tail(srv.log_path)}")
                try:
                    r = ask(srv.base, q["text"], th, a.max_tokens_on if th else a.max_tokens_off, a.chat_timeout,
                            system.get(q["qid"]) if per_q else system)
                    break
                except (RunError, urllib.error.URLError, OSError, ValueError, KeyError, IndexError) as e:
                    last = f"{type(e).__name__}: {e}"
                    with open(errpath, "a", encoding="utf-8") as ef:
                        ef.write(json.dumps({"qid": q["qid"], "thinking": th, "attempt": attempt, "error": last,
                                             "at": utc()}) + "\n")
                    LOG(f"  {akey} {q['qid']} {'think' if th else 'plain'} attempt {attempt}: {last[:200]}")
                    time.sleep(5)
            else:
                LOG(f"  {akey} {q['qid']}: no answer after {a.chat_retries + 1} attempts; left out (the step stays incomplete)")
                continue
            rec = {"set": run.S["id"], "key": akey, "phase": phase, "mode": mode, "qid": q["qid"], "thinking": th,
                   "question": q["text"], "content": r["content"], "reasoning_content": r["reasoning_content"],
                   "finish": r["finish"], "ms": r["ms"], "usage": r["usage"],
                   "max_tokens": a.max_tokens_on if th else a.max_tokens_off, "at": utc()}
            if sys_sha:
                rec["system_sha256"] = sys_sha
            if per_q:
                sq = system.get(q["qid"])
                rec["system_sha256"] = hashlib.sha256(sq.encode()).hexdigest() if sq else None
            rec.update((extra or {}).get(q["qid"]) or {})
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            LOG(f"  {akey:18} {'think' if th else 'plain'} {q['qid']} {' '.join(r['content'].split())[:110]!r} "
                f"[{r['finish']}, {r['ms'] / 1000:.1f}s]")
    have = answered_pairs(path)
    return {"path": path, "n": len(have & want), "expected": len(want), "complete": want <= have,
            "seconds": round(time.time() - t0, 1)}


# ============================================================================================ writes in place
INPLACE_GUARD = True  # only the self-tests switch it off, for their negative controls


class SharedFileError(RunError):
    """A write in place was refused: the file is a symlink, or shares its bytes with another file (a hard link)."""


def shared_file_problem(path):
    """Why a write in place into `path` would also change another file, or None. A symlink's write lands in its
    target; a file with st_nlink > 1 is one set of bytes under several names, so a write changes them all. On gb10-a
    leftover run copies were deduplicated into hard links to the Hugging Face source: zeroing one in place would
    silently have changed the source model."""
    st = os.lstat(path)
    if os.path.islink(path):
        return (f"{path} is a symlink (st_nlink {st.st_nlink}) to {os.readlink(path)}: a write in place would change "
                "the file it points to")
    if st.st_nlink > 1:
        return (f"{path} has st_nlink {st.st_nlink}: it shares its bytes with another file (a hard link), so a write "
                "in place would change that file too")
    return None


def check_inplace_writable(paths):
    """Every file an operation is about to write in place, checked before its first byte: SharedFileError naming
    each symlink and hard link, so a refused operation leaves no file half written."""
    if not INPLACE_GUARD:
        return
    bad = [why for why in map(shared_file_problem, dict.fromkeys(paths)) if why]
    if bad:
        raise SharedFileError("refusing to write in place:\n  " + "\n  ".join(bad))


def open_for_inplace_write(path):
    """open(path, "r+b"), for every write in place into a model file; SharedFileError (the path, its st_nlink and
    why) when the file is a symlink or has more than one hard link. The link count is read from the opened file
    itself and a symlink is not followed (O_NOFOLLOW), so a link made after an earlier check is refused too."""
    if not INPLACE_GUARD:
        return open(path, "r+b")
    why = shared_file_problem(path)
    if why:
        raise SharedFileError(f"refusing to write in place: {why}")
    try:
        fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
    except OSError as e:
        if e.errno == errno.ELOOP:
            raise SharedFileError(f"refusing to write in place: {path} became a symlink") from e
        raise
    try:
        n = os.fstat(fd).st_nlink
        if n > 1:
            raise SharedFileError(f"refusing to write in place: {path} has st_nlink {n}: it shares its bytes with "
                                  "another file (a hard link), so a write in place would change that file too")
    except BaseException:
        os.close(fd)
        raise
    return os.fdopen(fd, "r+b")


# ============================================================================================ zeroing
DTYPE_SIZE = {"U32": 4, "BF16": 2, "F16": 2, "F32": 4, "U8": 1}
PART_DTYPE = {"weight": "U32", "scales": "BF16", "biases": "BF16"}  # pocket_shards.zig writes u32 codes, bf16 halves


class ZeroSkip(Exception):
    """The folder's layout is not the one zero_fact.py assumes: zero mode is skipped, not guessed."""


NOT_APPLICABLE = "not applicable"  # a step status: nothing for the step to do, so it is finished (not failed, not run)
NO_POCKET = "config.json has no tensorfold_learned_neurons: the folder holds no pocket"


def pocket_size(folder):
    """config.json's tensorfold_learned_neurons, the number of learned neurons the folder's pocket has room for; None
    when it names none, as in a folder no save ever wrote to."""
    P = (read_json(os.path.join(folder, "config.json"), {}) or {}).get("tensorfold_learned_neurons")
    return P if isinstance(P, int) and P > 0 else None


def nothing_stored(tst):
    """Why the teach step's records say it stored nothing at all, or None when it may have stored something: no fact
    learned or replaced, no fact with neurons, and no save that wrote a byte. On 1.0.8 a missed lesson saves 0 bytes
    when no earlier fact is there (TEST-LOGIC.md section 8), so a store whose every lesson was missed that way leaves
    the taught folder as it was copied, with no pocket."""
    if not tst.get("taught"):
        return None
    states = tst.get("fact_states") or {}
    if any(s in ("learned", "replaced") for s in states.values()):
        return None
    if any(n for n in (tst.get("fact_neurons") or {}).values()):
        return None
    if any(int((r.get("save") or {}).get("bytes") or 0) > 0 for r in tst.get("lessons") or []):
        return None
    facts = ", ".join(f"fact {f} {s}" for f, s in sorted(states.items(), key=lambda x: (len(x[0]), x[0])))
    return f"the teach step stored nothing ({facts or 'no fact'}; every save wrote 0 bytes)"


def zero_limits(tst, deleted):
    """What zeroing cannot do for one delete phase, read from the teach step's records (TEST-LOGIC.md section 8).
    1.0.8 zeroes a fact's rows when a newer fact replaces it (pocket.zig `retire`). So when a deleted lesson's fact
    had replaced a kept lesson's fact, zeroing leaves neither, where never teaching the deleted lesson keeps the old
    one: cannot_restore lists those kept facts. Not when a kept lesson taught later replaced the deleted fact in turn
    (old <- deleted <- kept): never taught, that kept fact replaces the old one itself, so it is gone both ways (the
    #678 set's rm-4 reteach did just that). already_zero lists the deleted facts that were replaced or missed: their
    rows hold nothing, so zeroing them changes no answer."""
    owner = {str(k): v for k, v in (tst.get("fact_lesson") or {}).items()}
    states = tst.get("fact_states") or {}
    deleted = set(deleted)
    by = {}  # fact id (str) -> (the fact that replaced it or None, that fact's lesson, its text), in teaching order
    for rec in tst.get("lessons") or []:
        for c in rec.get("changed_states") or []:
            if c.get("to") != "replaced":
                continue
            m = re.search(r"replaced by fact (\d+)", c.get("message") or "")
            by_fact = int(m.group(1)) if m else None
            by_lesson = owner.get(str(by_fact), rec["lesson"]) if by_fact is not None else rec["lesson"]
            by[str(c.get("id"))] = (by_fact, by_lesson, c.get("text"))

    def kept_successor(fact):
        """The first kept lesson down the chain of facts that replaced `fact`, or None."""
        seen = set()
        while fact is not None and str(fact) in by and str(fact) not in seen:
            seen.add(str(fact))
            fact, lesson, _text = by[str(fact)]
            if lesson not in deleted:
                return lesson
        return None

    lost = []
    for fid, (by_fact, by_lesson, text) in by.items():
        old = owner.get(fid)
        if by_lesson in deleted and old is not None and old not in deleted and kept_successor(by_fact) is None:
            lost.append({"fact": int(fid), "lesson": old, "text": text, "replaced_by_fact": by_fact,
                         "replaced_by_lesson": by_lesson})
    inert = sorted(int(f) for f, l in owner.items() if l in deleted and states.get(f) in ("replaced", "missed"))
    return {"cannot_restore": lost, "already_zero": inert}


def st_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def zero_facts(folder, targets, kept, ref=None, audit="touched"):
    """zero_fact.py's removal on 1.0.8's pocket: for each (layer, neuron) of a deleted lesson's facts, zero row
    rows - P + neuron of backbone.layers.L.mixer.shared_experts.up_proj.{weight,scales,biases}, in place
    (P = config.json tensorfold_learned_neurons; pocket_shards.zig puts learned neuron k at row own + k).
    down_proj is not touched: with up_proj's row zero the neuron's relu2 is zero. targets/kept: (fact, state,
    layer, neuron). Rows already zero are expected for facts 1.0.8 marked `replaced` (pocket.zig `retire` zeroes
    them); a `learned` fact whose rows are all zero means wrong indexing, and the zeroing is refused. So is a shard
    that is a symlink or a hard link (SharedFileError), for every shard to be written before the first byte."""
    cfg = read_json(os.path.join(folder, "config.json"), {})
    P = pocket_size(folder)
    wide = cfg.get("moe_shared_expert_intermediate_size")
    if P is None:
        raise ZeroSkip(NO_POCKET)
    if P != 64:
        raise ZeroSkip(f"tensorfold_learned_neurons is {P}, not the 64 of 1.0.8's pocket.zig")
    wm = read_json(os.path.join(folder, "model.safetensors.index.json"), {}).get("weight_map")
    if not wm:
        raise ZeroSkip("no model.safetensors.index.json weight_map")
    headers = {}

    def span(layer, neuron, part):
        name = f"backbone.layers.{layer}.mixer.shared_experts.up_proj.{part}"
        if name not in wm:
            raise ZeroSkip(f"{name} is not in the index")
        shard = os.path.join(folder, wm[name])
        if shard not in headers:
            headers[shard] = st_header(shard)
        h, base = headers[shard]
        m = h[name]
        dtype, shape, (s, e) = m["dtype"], m["shape"], m["data_offsets"]
        if dtype != PART_DTYPE[part] or len(shape) != 2 or shape[0] != wide:
            raise ZeroSkip(f"{name} is {dtype} {shape}, expected {PART_DTYPE[part]} [{wide}, ...]")
        if not 0 <= neuron < P:
            raise ZeroSkip(f"neuron {neuron} outside the pocket of {P}")
        width = shape[1] * DTYPE_SIZE[dtype]
        if e - s != shape[0] * width:
            raise ZeroSkip(f"{name}: data_offsets do not match its shape")
        return shard, base + s + (shape[0] - P + neuron) * width, width

    def row(shard, off, width):
        with open(shard, "rb") as f:
            f.seek(off)
            return f.read(width)

    tkeys = {(l, n) for _f, _s, l, n in targets}
    clash = [(f, l, n) for f, _s, l, n in kept if (l, n) in tkeys]
    if clash:
        raise RunError(f"kept facts share pocket neurons with deleted ones {clash}: zeroing would remove them too")
    plan, per_fact = [], {}
    for fid, state, layer, neuron in targets:
        for part in ("weight", "scales", "biases"):
            shard, off, width = span(layer, neuron, part)
            nz = any(row(shard, off, width))
            plan.append({"fact": fid, "state": state, "layer": layer, "neuron": neuron, "part": part,
                         "shard": os.path.basename(shard), "path": shard, "offset": off, "bytes": width,
                         "nonzero_before": nz})
            pf = per_fact.setdefault(str(fid), {"state": state, "rows": 0, "nonzero_before": 0})
            pf["rows"] += 1
            pf["nonzero_before"] += nz
    for fid, pf in per_fact.items():
        if pf["state"] == "learned" and pf["nonzero_before"] == 0:
            raise RunError(f"fact {fid} is learned but every one of its pocket rows is already zero: wrong indexing? refusing")

    def kept_nonzero():
        return sum(1 for _f, s, l, n in kept if s == "learned" and any(row(*span(l, n, "weight"))))

    kept_learned = sum(1 for _f, s, _l, _n in kept if s == "learned")
    kept_before = kept_nonzero()
    if kept_learned and kept_before == 0:
        raise RunError("no kept learned neuron is nonzero before zeroing: wrong indexing? refusing")
    check_inplace_writable(p["path"] for p in plan if p["nonzero_before"])  # every shard, before the first byte
    for p in plan:
        if p["nonzero_before"]:
            with open_for_inplace_write(p["path"]) as f:
                f.seek(p["offset"])
                f.write(b"\0" * p["bytes"])
                f.flush()
                os.fsync(f.fileno())
    for p in plan:
        if any(row(p["path"], p["offset"], p["bytes"])):
            raise RunError(f"row not zero after the write: {p}")
    kept_after = kept_nonzero()
    if kept_after != kept_before:
        raise RunError(f"kept neurons nonzero went from {kept_before} to {kept_after}: a kept fact was hit")
    rep = {"pocket": P, "shared_width": wide, "targets": len(targets), "rows_planned": len(plan),
           "rows_zeroed": sum(p["nonzero_before"] for p in plan),
           "rows_already_zero": sum(not p["nonzero_before"] for p in plan),
           "bytes_zeroed": sum(p["bytes"] for p in plan if p["nonzero_before"]),
           "per_fact": per_fact, "kept_neurons": len(kept), "kept_learned": kept_learned,
           "kept_nonzero_before": kept_before, "kept_nonzero_after": kept_after,
           "shards_touched": sorted({p["shard"] for p in plan if p["nonzero_before"]}),
           "rows": [{k: v for k, v in p.items() if k != "path"} for p in plan]}
    if ref and audit != "off":
        rep["audit"] = audit_copy(folder, ref, plan, full=(audit == "full"))
        if not rep["audit"]["ok"]:
            raise RunError(f"byte audit failed: {rep['audit']}")
    return rep


def audit_copy(folder, ref, plan, full=False):
    """The zeroed copy against the folder it was copied from: every differing byte inside a planned row; touched
    shards and small files compared byte for byte, other shards by size (all of them with --zero-audit full)."""
    allowed = {}
    for p in plan:
        if p["nonzero_before"]:
            allowed.setdefault(p["shard"], []).append((p["offset"], p["offset"] + p["bytes"]))
    names_a = sorted(os.listdir(ref))
    names_b = sorted(os.listdir(folder))
    out = {"ok": True, "files": len(names_a), "compared": [], "size_only": [], "bytes_differing": 0,
           "outside_rows": 0, "problems": []}
    if names_a != names_b:
        out["ok"] = False
        out["problems"].append(f"file lists differ: {sorted(set(names_a) ^ set(names_b))}")
    for name in names_a:
        pa, pb = os.path.join(ref, name), os.path.join(folder, name)
        if not os.path.isfile(pa) or not os.path.isfile(pb):
            continue
        if os.path.getsize(pa) != os.path.getsize(pb):
            out["ok"] = False
            out["problems"].append(f"{name}: size differs")
            continue
        if not (full or name in allowed or os.path.getsize(pa) < (64 << 20)):
            out["size_only"].append(name)
            continue
        spans = sorted(allowed.get(name, []))
        diff, outside = compare_files(pa, pb, spans)
        out["compared"].append(name)
        out["bytes_differing"] += diff
        out["outside_rows"] += outside
    if out["outside_rows"]:
        out["ok"] = False
        out["problems"].append(f"{out['outside_rows']} differing bytes lie outside the zeroed rows")
    return out


def compare_files(pa, pb, spans, chunk=16 << 20, block=1 << 16):
    diff = outside = 0
    with open(pa, "rb") as fa, open(pb, "rb") as fb:
        pos = 0
        while True:
            a = fa.read(chunk)
            b = fb.read(chunk)
            if not a and not b:
                break
            if a != b:
                for i in range(0, max(len(a), len(b)), block):
                    x, y = a[i:i + block], b[i:i + block]
                    if x == y:
                        continue
                    for j in range(max(len(x), len(y))):
                        if j >= len(x) or j >= len(y) or x[j] != y[j]:
                            diff += 1
                            at = pos + i + j
                            if not any(s <= at < e for s, e in spans):
                                outside += 1
            pos += len(a)
    return diff, outside


# ============================================================================================ pocket weight norms
# How big each fact's pocket neurons are, read from a saved folder (READ from our stored 1.0.8 run records: config.json
# tensorfold_learned_neurons 64; per MoE layer backbone.layers.L.mixer.shared_experts.{up,down}_proj.{weight,scales,
# biases} in MLX's affine 4-bit format, weight U32 with 8 codes per word, scales and biases BF16 per group of 64; up_proj
# is [3776, 336] (2688 inputs), down_proj [2688, 472] (3776 inputs); learned neuron k is up_proj row 3712 + k and
# down_proj column 3712 + k, and the 64 pocket columns of down_proj are one group, so they share its scale and bias).
# A value is scale * code + bias; an unused neuron's up_proj row is all zero. A fact's norm is the square root of the
# sum of squares over its neurons: up (its input weights), down (its output weights) and total (both).
def bf16_floats(raw):
    hs = array.array("H")
    hs.frombytes(raw)
    if sys.byteorder != "little":
        hs.byteswap()
    out = array.array("f")
    out.frombytes(array.array("I", (h << 16 for h in hs)).tobytes())  # a bfloat16 is the top half of a float32
    return out


def u32_words(raw):
    ws = array.array("I")
    ws.frombytes(raw)
    if sys.byteorder != "little":
        ws.byteswap()
    return ws


class PocketReader:
    """Reads one saved folder's pocket rows, read only. Raises ZeroSkip when the folder is not laid out as above."""

    def __init__(self, folder):
        self.folder = folder
        self.P = pocket_size(folder)
        if self.P is None:
            raise ZeroSkip(NO_POCKET)
        self.wm = (read_json(os.path.join(folder, "model.safetensors.index.json"), {}) or {}).get("weight_map")
        if not self.wm:
            raise ZeroSkip("no model.safetensors.index.json weight_map")
        self.headers = {}
        self.down_cache = {}

    def tensor(self, name):
        """(path, data offset, dtype, shape) of a tensor, or None when the index does not name it."""
        shard = self.wm.get(name)
        if shard is None:
            return None
        path = os.path.join(self.folder, shard)
        if path not in self.headers:
            self.headers[path] = st_header(path)
        h, base = self.headers[path]
        m = h[name]
        return path, base + m["data_offsets"][0], m["dtype"], m["shape"]

    def read(self, path, off, n):
        with open(path, "rb") as f:
            f.seek(off)
            b = f.read(n)
        if len(b) != n:
            raise ZeroSkip(f"{path}: short read at {off}")
        return b

    def parts(self, layer, proj):
        out = {}
        for part in ("weight", "scales", "biases"):
            t = self.tensor(f"backbone.layers.{layer}.mixer.shared_experts.{proj}.{part}")
            if t is None:
                return None
            path, off, dtype, shape = t
            if dtype != PART_DTYPE[part] or len(shape) != 2:
                raise ZeroSkip(f"{proj}.{part} of layer {layer} is {dtype} {shape}: not MLX affine 4-bit")
            out[part] = t
        rows, words = out["weight"][3]
        groups = out["scales"][3][1]
        if out["scales"][3][0] != rows or out["biases"][3] != out["scales"][3] or (words * 8) % groups:
            raise ZeroSkip(f"{proj} of layer {layer}: scales and biases do not fit the weight")
        return out, rows, words, groups

    def up_norm2(self, layer, neuron):
        got = self.parts(layer, "up_proj")
        if got is None:
            raise ZeroSkip(f"layer {layer} has no shared_experts.up_proj")
        t, rows, words, groups = got
        if not 0 <= neuron < self.P or rows < self.P:
            raise ZeroSkip(f"neuron {neuron} outside the pocket of {self.P}")
        r = rows - self.P + neuron
        w = u32_words(self.read(t["weight"][0], t["weight"][1] + r * words * 4, words * 4))
        s = bf16_floats(self.read(t["scales"][0], t["scales"][1] + r * groups * 2, groups * 2))
        b = bf16_floats(self.read(t["biases"][0], t["biases"][1] + r * groups * 2, groups * 2))
        per = words // groups  # words per group
        total = 0.0
        for g in range(groups):
            sq = sq2 = 0
            for word in w[g * per:(g + 1) * per]:
                for k in range(8):
                    q = (word >> (4 * k)) & 15
                    sq += q
                    sq2 += q * q
            gs = per * 8
            total += s[g] * s[g] * sq2 + 2 * s[g] * b[g] * sq + gs * b[g] * b[g]
        return max(total, 0.0)

    def down_norm2(self, layer, neuron):
        """None when the folder has no down_proj for the layer (the runner's mock model has none)."""
        got = self.parts(layer, "down_proj")
        if got is None:
            return None
        t, rows, words, groups = got
        width = words * 8
        c = width - self.P + neuron
        if not 0 <= neuron < self.P or c < 0:
            raise ZeroSkip(f"neuron {neuron} outside the pocket of {self.P}")
        key = layer
        if key not in self.down_cache:  # every row of one layer's down_proj, read once (5 MB on Nemotron)
            self.down_cache = {key: (u32_words(self.read(t["weight"][0], t["weight"][1], rows * words * 4)),
                                     bf16_floats(self.read(t["scales"][0], t["scales"][1], rows * groups * 2)),
                                     bf16_floats(self.read(t["biases"][0], t["biases"][1], rows * groups * 2)))}
        w, s, b = self.down_cache[key]
        g, word, shift = c // (width // groups), c // 8, 4 * (c % 8)
        total = 0.0
        for r in range(rows):
            v = s[r * groups + g] * ((w[r * words + word] >> shift) & 15) + b[r * groups + g]
            total += v * v
        return total

    def fact_norms(self, neurons):
        """{up, down, total, neurons, per_neuron} of one fact, from its [(layer, neuron)] list."""
        up = down = 0.0
        has_down = True
        per = []
        for layer, neuron in sorted(neurons, key=lambda x: (x[0], x[1])):
            u = self.up_norm2(layer, neuron)
            d = self.down_norm2(layer, neuron)
            has_down = has_down and d is not None
            up += u
            down += d or 0.0
            per.append([layer, neuron, round(u ** 0.5, 6), None if d is None else round(d ** 0.5, 6)])
        out = {"neurons": len(per), "up": round(up ** 0.5, 6),
               "down": round(down ** 0.5, 6) if has_down and per else None,
               "total": round((up + (down if has_down else 0.0)) ** 0.5, 6), "per_neuron": per}
        return out


def pocket_norms(folder, fact_neurons):
    """{fact id (str): norms} for {fact id: [{layer, neuron}, ...]} (a graph's or lwrun's fact_neurons), read from
    `folder`, read only. Facts without neurons are left out. Raises ZeroSkip when the folder is laid out otherwise."""
    rd = PocketReader(folder)
    out = {}
    for fid, ns in sorted(fact_neurons.items(), key=lambda x: int(x[0])):
        keys = sorted({(int(n["layer"]), int(n["neuron"])) for n in ns or []})
        if keys:
            out[str(fid)] = rd.fact_norms(keys)
    return out


def norm_outliers(values, z_cut=3.5, ratio_cut=10.0):
    """Outliers among facts' norms ({fact: number}): the modified z score 0.6745 (x - median) / MAD above z_cut
    (Iglewicz and Hoaglin), or more than ratio_cut times the median. Only large norms are flagged: Gupta et al. (2024)
    report single "disabling" edits whose weight change is hundreds of times the usual size."""
    xs = sorted(v for v in values.values() if v is not None)
    if not xs:
        return {"median": None, "mad": None, "z_cut": z_cut, "ratio_cut": ratio_cut, "facts": {}, "outliers": []}
    med = xs[len(xs) // 2] if len(xs) % 2 else (xs[len(xs) // 2 - 1] + xs[len(xs) // 2]) / 2
    dev = sorted(abs(x - med) for x in xs)
    mad = dev[len(dev) // 2] if len(dev) % 2 else (dev[len(dev) // 2 - 1] + dev[len(dev) // 2]) / 2
    mean_ad = sum(dev) / len(dev)
    facts = {}
    for f, v in values.items():
        if v is None:
            continue
        # more than half the norms equal (MAD 0): the mean absolute deviation stands in, as is usual; all equal: z 0
        z = (0.6745 * (v - med) / mad if mad > 0 else (v - med) / (1.253314 * mean_ad) if mean_ad > 0 else 0.0)
        ratio = v / med if med > 0 else None
        out = (z > z_cut and v > med) or (ratio is not None and ratio > ratio_cut)
        facts[str(f)] = {"norm": v, "z": round(z, 3), "ratio": None if ratio is None else round(ratio, 3),
                         "outlier": bool(out)}
    return {"median": med, "mad": mad, "mean_ad": mean_ad, "z_cut": z_cut, "ratio_cut": ratio_cut, "facts": facts,
            "outliers": sorted((f for f, x in facts.items() if x["outlier"]), key=int)}


# ============================================================================================ judging
def aggregate(v):
    out = {}
    for r in v.get("results", []):
        th = "on" if r.get("thinking") else "off"
        verdict = r.get("verdict") if r.get("verdict") in ("pass", "fail") else "unclear"
        for b in list(r.get("buckets", [])) + ["all"]:
            c = out.setdefault(r["phase"], {}).setdefault(b, {}).setdefault(th, {"pass": 0, "fail": 0, "unclear": 0, "total": 0})
            c[verdict] += 1
            c["total"] += 1
    return out


def run_judge(run, files, verdicts, report, label):
    cmd = [sys.executable, run.lwlogic, "judge", run.set_for_judge] + list(files) + ["--json", verdicts]
    LOG(f"judge [{label}]: {' '.join(shlex.quote(x) for x in cmd)}")
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    except (OSError, subprocess.SubprocessError) as e:
        LOG(f"judge [{label}] could not run: {e}")
        return {"rc": None, "error": str(e)}
    with open(report, "w", encoding="utf-8") as f:
        f.write(p.stdout)
        if p.stderr.strip():
            f.write("\n--- stderr ---\n" + p.stderr)
    out = {"rc": p.returncode, "report": report, "verdicts": verdicts, "seconds": round(time.time() - t0, 1),
           "headline": (p.stdout.strip().splitlines() or [""])[0]}
    if p.returncode != 0:
        LOG(f"judge [{label}] exited {p.returncode}: {p.stderr.strip()[-600:]}")
        return out
    for line in p.stderr.strip().splitlines()[:3]:
        LOG(f"judge [{label}] stderr: {line[:300]}")
    LOG(f"judge [{label}]: {out['headline']}; report {report}")
    try:  # the judge's JSON as lwlogic.py writes it today: results[].{phase, thinking, verdict, buckets}
        v = read_json(verdicts, {})
        out["scores"] = aggregate(v)
        out["consistency"] = v.get("consistency")
        if v.get("kept_stable"):  # the judge's score of delete phases against the same model's teach answers
            out["kept_stable"] = v["kept_stable"]
        for ph, bk in out["scores"].items():
            al = bk.get("all", {})
            LOG("  " + ph + ": " + ", ".join(f"thinking {k} {c['pass']}/{c['total']} pass ({c['unclear']} unclear)"
                                             for k, c in sorted(al.items(), key=lambda x: x[0] != "off")))
    except (ValueError, KeyError, TypeError, AttributeError) as e:
        out["scores_error"] = f"{type(e).__name__}: {e}"
        LOG(f"judge [{label}]: could not read scores from {verdicts}: {out['scores_error']} (the report is kept)")
    return out


# ============================================================================================ reference phases
# Two opt-in references for the pocket (--reference), each beside one or more phases of the set (--reference-of):
#   in_context          the untaught model, served like baseline, with the phase's lessons in the system prompt;
#   replacement_memory  no model: a lookup that keeps the latest value per (entity, attribute) and answers from it.
# Their answers carry the phase's own name, so the judge scores them against that phase's right answers. Neither is
# a ceiling: PersonaMem (2025) reports frontier models at about 50% on changing preferences told in context.
REF_KINDS = ("in_context", "replacement_memory")
# The note memory (docs/NOTES-LAYER.md, Phase 1: notes only): the retrieved notes in the system prompt of the untaught
# model (notes) and of the pocket (notes_pocket). Named one by one: "both" keeps meaning the two references above.
NOTE_KINDS = ("notes", "notes_pocket")
ALL_REF_KINDS = REF_KINDS + NOTE_KINDS
REF_FORMAT = "lw-reference/1"
IN_CONTEXT_FORMAT = "lw-in-context/1"
IN_CONTEXT_HEAD = "Earlier, the user told you the following, one message at a time, oldest first:"
IN_CONTEXT_TAIL = 'In these messages, "I", "me" and "my" mean the user.'
MEMORY_FORMAT = "lw-replacement-memory/1"
MEM_NONE = "(none)"          # a slot that holds "no value at all" ("I have no favourite colour anymore.")
DONT_KNOW = "I don't know."
# The report's groups, in this order: the four the plan names, two-step questions when a set has them, then the
# judge's own groups (its recall and reverse groups are CURRENT and WHO_HAS here) and all answers.
REF_GROUPS = ["CURRENT", "WHO_HAS", "negation", "dates", "two_step", "compound", "temporal", "either_or", "yes_no",
              "controls", "delete_affected", "delete_kept", "all"]
# Two-step question types, as level 6 names them (lwlogic.TWO_STEP, the same two since the level 6 merge): a CHAIN walks
# from a start through `hops` to a final attribute ("What colour are the eyes of the owner of the boat?"), a COMPOUND
# joins two yes/no parts with and, or, not. The judge's own groups for them are two_step (CHAIN) and compound.
TWO_STEP_TYPES = ("CHAIN", "COMPOUND")
OUTCOMES = ("corrected_only", "stale_only", "both", "neither")


def in_context_prompt(S, store):
    """The system prompt of the in_context reference (lw-in-context/1): one fixed line, then the store's lessons in
    teaching order, numbered, each in double quotes and word for word as it is taught, then one fixed line that says
    whose "I" they are. Nothing else: no date of telling (the pocket is never told one either), no rule of the rule
    book, no request to say "I don't know". The wording is documented in TEST-LOGIC.md section 8; change it only
    with a new format name, since answers on file are tied to it by sha256."""
    keep = set(store)
    lessons = [l for l in S["lessons"] if l["id"] in keep]  # teaching order is the set's order, as in teach()
    if not lessons:
        raise RunError("an empty store has no lessons for the in_context prompt (the baseline is that reference)")
    body = "\n".join(f'{i}. "{l["text"]}"' for i, l in enumerate(lessons, 1))
    return f"{IN_CONTEXT_HEAD}\n{body}\n{IN_CONTEXT_TAIL}"


def memory_slots(S, store):
    """The replacement memory (lw-replacement-memory/1): one slot per (entity, attribute), replayed over the store's
    lessons in teaching order. A slot holds the latest value a lesson gives for now, and nothing else: no history, no
    second value (also on likes, which then hold one food, as 1.0.8 did in RESULTS finding 10), no "not v".
      ASSERT v (present tense, "since D", "always"), CHANGE v      the slot is v
      DATED v0 until D, v1 since D                              the slot is v1
      ASSERT v "until D" (or between two dates)                 says nothing about now: the slot stays as it is
      RETRACT v                                                 the slot is emptied if it holds v; else it stays
      RETRACT with no value ("no favourite colour anymore")     the slot holds MEM_NONE
    Returns {(entity, attribute): value or MEM_NONE}; a key with no slot is "don't know"."""
    keep = set(store)
    slots = {}
    for l in S["lessons"]:
        if l["id"] not in keep:
            continue
        k, op, when = (l["e"], l["a"]), l["op"], l.get("when") or {}
        if op == "DATED":
            slots[k] = l["v1"]
        elif op == "RETRACT":
            if l.get("v") is None:
                slots[k] = MEM_NONE
            elif slots.get(k) == l["v"]:
                del slots[k]
        elif op == "ASSERT" and ("until" in when or "between" in when):
            continue
        else:
            slots[k] = l["v"]
    return slots


_LW = {}


def lw_module(path):
    """lwlogic.py loaded as a module: the reference phases need its templates (to word the memory's answers), its
    rule book (which values a CHANGE or RETRACT made stale) and the judge's own readers (corrected or stale)."""
    path = os.path.abspath(path)
    if path not in _LW:
        import importlib.util
        try:
            spec = importlib.util.spec_from_file_location("lwlogic_for_lwrun", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        except Exception as e:  # noqa: BLE001  (any failure to load is reported, not raised as itself)
            raise RunError(f"the reference phases need lwlogic.py as a module; {path} did not load: "
                           f"{type(e).__name__}: {e}")
        need = ("T", "ENTITIES", "NOW", "YES", "NO", "state_at", "judge_core", "extract_value", "extract_set")
        missing = [x for x in need if not hasattr(mod, x)]
        if missing:
            raise RunError(f"the reference phases need lwlogic.py's {', '.join(missing)}; {path} has none")
        _LW[path] = mod
    return _LW[path]


_NOTES = {}


def notes_module():
    """zengine/notes.py as a module: the note memory of the notes phases. It reads the rule book and the intake check
    from its own folder. Standard library only, but a real tokenizer needs the `tokenizers` package."""
    if "m" not in _NOTES:
        import importlib.util
        path = os.path.join(HERE, "notes.py")
        try:
            spec = importlib.util.spec_from_file_location("notes_for_lwrun", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        except Exception as e:  # noqa: BLE001  (any failure to load is reported, not raised as itself)
            raise RunError(f"the notes phases need zengine/notes.py; {path} did not load: {type(e).__name__}: {e}")
        _NOTES["m"] = mod
    return _NOTES["m"]


def pctl(xs, p):
    xs = sorted(xs)
    return xs[max(0, min(len(xs) - 1, -(-p * len(xs) // 100) - 1))] if xs else None


def to_user(lw, e):
    """lwlogic's fields of entity `e`, said to the user: "me" is "you", "my sister" is "your sister"."""
    f = dict(lw.ENTITIES[e])
    if e == "me":
        f.update(nom="you", poss="your", pron="you", be="are", have="have", do="do", s="")
    else:
        n = re.sub(r"^my ", "your ", e)
        f.update(nom=n, poss=n + "'s")
    return f


def say_to_user(x):
    """A value or entity said to the user (owner values and WHO_HAS answers): "me" is "you", "my X" is "your X"."""
    return "you" if x == "me" else re.sub(r"^my ", "your ", x)


def capital(s):
    return s[:1].upper() + s[1:]


def memory_answer(lw, S, slots, q):
    """The replacement memory's answer to question q, in plain words the judge reads (TEST-LOGIC.md section 8). It
    answers only about now (CURRENT, YES_NO, EITHER_OR, WHO_HAS), from the slots alone; it keeps no history and no
    general knowledge, so every other question gets "I don't know." A value is said with the attribute's first
    ASSERT wording, to the user ("Your favourite colour is green.")."""
    t, a = q["type"], q.get("a")
    if t not in ("CURRENT", "YES_NO", "EITHER_OR", "WHO_HAS") or not a:
        return DONT_KNOW

    def stated(e, v):
        if v == MEM_NONE:
            clear = lw.T[a].get("CLEAR") or []
            return capital(clear[0].format(**to_user(lw, e))) if clear else "There is none anymore."
        return capital(lw.T[a]["ASSERT"][0].format(**to_user(lw, e), v=say_to_user(v)))

    if t == "WHO_HAS":
        who = [e for e in S["entities"] if slots.get((e, a)) == q["v"]]
        return capital(" and ".join(say_to_user(e) for e in who) + ".") if who else "No one you told me about."
    slot = slots.get((q["e"], a))
    if slot is None:
        return DONT_KNOW
    if t == "CURRENT":
        return stated(q["e"], slot)
    if t == "YES_NO":
        return "Yes." if slot == q["v"] else "No."
    return stated(q["e"], slot) if slot in (q["x"], q["y"]) else "Neither."


def two_step(q):
    """A two-step question of level 6: type CHAIN or COMPOUND (TWO_STEP_TYPES). Only level 6 sets have them, so the
    two_step group appears only there."""
    return q.get("type") in TWO_STEP_TYPES


def answer_groups(groups, qid, buckets):
    """The report's groups of one answer: the question's own (question_groups) and the judge's groups from its
    verdict, without recall and reverse (they are CURRENT and WHO_HAS here), and all."""
    return groups.get(qid, []) + [b for b in buckets or [] if b not in ("recall", "reverse")] + ["all"]


def question_groups(S, phase_name):
    """{qid: [group, ...]} of the report's own groups for one phase (section 8). The judge's groups are added from
    each answer's verdict. Only questions about the user's facts (role probe) are grouped here:
      CURRENT    the value now (the judge's recall group)
      WHO_HAS    reverse questions (the judge's reverse group)
      negation   a question on a key that a RETRACT lesson in this phase's store touches ("not green anymore",
                 "no favourite colour anymore"); for WHO_HAS, a retraction of its value (or of every value) on its
                 attribute
      dates      the question names a date (AT, BEFORE)
      two_step   level 6's CHAIN and COMPOUND questions (two_step())"""
    ph = next(p for p in S["phases"] if p["name"] == phase_name)
    keep = set(ph["store"])
    retracts = [l for l in S["lessons"] if l["id"] in keep and l["op"] == "RETRACT"]
    out = {}
    for q in S["questions"]:
        g = []
        if q.get("role") == "probe":
            if q["type"] == "CURRENT":
                g.append("CURRENT")
            if q["type"] == "WHO_HAS":
                g.append("WHO_HAS")
            if any(l["a"] == q.get("a") and (l["e"] == q["e"] if q.get("e") else l.get("v") in (None, q.get("v")))
                   for l in retracts):
                g.append("negation")
            if q.get("date"):
                g.append("dates")
            if two_step(q):
                g.append("two_step")
        out[q["qid"]] = g
    return out


def change_lessons(lw, S, phase_name):
    """The CHANGE and RETRACT lessons of one phase's store, each with what it made stale, what holds just after it,
    and whether today's answers can show its effect. stale: for CHANGE v, the values that held now just before it
    (other than v); for RETRACT v, v (also when v did not hold: an answer that names it is stale, as "learned
    backwards" in plan item 1); for a RETRACT with no value, every value that held just before. now: the values that
    hold just after it, by the rule book (CHANGE v: v; RETRACT: the values left, often none). in_force: no later lesson
    in the store writes the same key again (on likes, which hold several values: a later CHANGE, or a lesson about v
    or a stale value). A lesson that is not in force, or made nothing stale, is listed with the reason and not
    classified."""
    sch = S["schema"]
    keep = set(next(p for p in S["phases"] if p["name"] == phase_name)["store"])
    store = [l for l in S["lessons"] if l["id"] in keep]
    out = []
    for i, l in enumerate(store):
        if l["op"] not in ("CHANGE", "RETRACT"):
            continue
        e, a, v = l["e"], l["a"], l.get("v")
        multi = sch[a]["multi"]
        before = lw.state_at(sch, store[:i], e, a, lw.NOW)
        after = lw.state_at(sch, store[:i + 1], e, a, lw.NOW)
        if l["op"] == "CHANGE":
            stale = [x for x in sch[a]["domain"] if before.get(x) == lw.YES and x != v]
        elif v is not None:
            stale = [v]
        else:
            stale = [x for x in sch[a]["domain"] if before.get(x) == lw.YES]
        touched = set(stale) | ({v} if v is not None else set())
        later = [m["name"] for m in store[i + 1:] if (m["e"], m["a"]) == (e, a) and (
            not multi or m["op"] == "CHANGE" or m.get("v") is None or m.get("v") in touched)]
        rec = {"lesson": l["id"], "name": l["name"], "op": l["op"], "text": l["text"], "e": e, "a": a,
               "new": v if l["op"] == "CHANGE" else (f"not {v}" if v is not None else "none"), "stale": stale,
               "now": [x for x in sch[a]["domain"] if after.get(x) == lw.YES],
               "in_force": not later, "classified": bool(stale) and not later,
               "questions": [q["qid"] for q in S["questions"] if q.get("role") == "probe" and q.get("e") == e
                             and q.get("a") == a and q["type"] in ("CURRENT", "YES_NO", "EITHER_OR")]}
        if later:
            rec["why"] = (f"a later lesson ({', '.join(later)}) writes this key again, so today's answers cannot show "
                          "this lesson's effect")
        elif not stale:
            rec["why"] = "no other value held just before it, so no answer can be stale"
        out.append(rec)
    return out


def read_answer(lw, S, phase_name, q, content):
    """How the judge reads one answer, whatever the right answer is: the values it names (not negated, not guessed),
    whether it says it does not know or that the value no longer holds (absent), its yes or no, its choice."""
    sch, a, t = S["schema"], q["a"], q["type"]
    if t == "CURRENT":
        g = lw.extract_set(content, sch, a) if sch[a]["multi"] else lw.extract_value(content, sch, a, "present")
        named = (g["yes"] if g["got"] == "SET" else [g["value"]] if g["got"] == "VALUE" else
                 g.get("values", []) if g["got"] == "AMBIGUOUS" else [])
        return {"got": g["got"], "named": list(named), "absent": g["got"] == "ABSENT"}
    g = lw.judge_core(S, phase_name, q, content)[1]  # YES_NO and EITHER_OR read the same whatever is expected
    return {"got": g.get("got"), "yn": g.get("value") if g.get("got") == "YN" else None,
            "choice": g.get("value") if g.get("got") == "CHOICE" else None}


def outcome_marks(lw, S, rec, q, r):
    """What one answer to a question on a CHANGE or RETRACT lesson's key shows: 'corrected' (what holds just after
    the lesson, rec["now"]), 'stale' (a value it made stale), both, or neither (TEST-LOGIC.md section 8).
      CURRENT    names a stale value: stale. Names a value that holds now (CHANGE v: v; RETRACT: a value left):
                 corrected. RETRACT that leaves nothing: "don't know" or "not anymore": corrected.
      YES_NO     yes to a stale value: stale. Yes to a value that holds now: corrected. RETRACT: no to the stale
                 value: corrected. (No to a stale value after a CHANGE is not the new value, so it shows nothing.)
      EITHER_OR  picks a stale value: stale. Picks a value that holds now: corrected."""
    marks, stale, now = set(), set(rec["stale"]), set(rec["now"])
    t = q["type"]
    if t == "CURRENT":
        named = set(r["named"])
        if named & stale:
            marks.add("stale")
        if named & now:
            marks.add("corrected")
        if rec["op"] == "RETRACT" and not now and r["absent"]:
            marks.add("corrected")
    elif t == "YES_NO":
        if r["yn"] == lw.YES and q["v"] in stale:
            marks.add("stale")
        if r["yn"] == lw.YES and q["v"] in now:
            marks.add("corrected")
        if r["yn"] == lw.NO and rec["op"] == "RETRACT" and q["v"] in stale:
            marks.add("corrected")
    elif t == "EITHER_OR":
        if r["choice"] in stale:
            marks.add("stale")
        if r["choice"] in now:
            marks.add("corrected")
    return marks


def lesson_outcomes(lw, S, phase_name, lessons, answers, reader=read_answer, marks=outcome_marks):
    """For one answer source ({(qid, thinking): content}) and each classified lesson of change_lessons(): per thinking
    mode the outcome, corrected only, stale only, both or neither, over the lesson key's CURRENT, YES_NO and
    EITHER_OR answers, with the questions that showed each side. `reader` is read_answer and `marks` outcome_marks
    (the self-test passes broken ones as its negative controls)."""
    qs = {q["qid"]: q for q in S["questions"]}
    modes = sorted({th for _qid, th in answers})
    out = {}
    for rec in lessons:
        if not rec["classified"]:
            continue
        per = {}
        for th in modes:
            seen = {"corrected": [], "stale": []}
            asked = 0
            for qid in rec["questions"]:
                if (qid, th) not in answers:
                    continue
                asked += 1
                for m in marks(lw, S, rec, qs[qid], reader(lw, S, phase_name, qs[qid], answers[(qid, th)])):
                    seen[m].append(qid)
            c, s = bool(seen["corrected"]), bool(seen["stale"])
            if asked:
                per["on" if th else "off"] = {"outcome": "both" if c and s else "corrected_only" if c else
                                              "stale_only" if s else "neither", "answers": asked, **seen}
        out[rec["name"]] = per
    return out


def rate(t):
    return t[0] / t[1] if t and t[1] else None


def fmt_gap(x):
    return "-" if x is None else f"{x:+.2f}"


# ============================================================================================ the run
class Step:
    def __init__(self, key, phase, kind, store=(), deleted=()):
        self.key, self.phase, self.kind = key, phase, kind
        self.store, self.deleted = list(store), list(deleted)
        self.flavors = []  # (akey, file suffix, how, mode): how = slide | plain | plain-src | computed

    def needs_copy(self):
        return self.kind in ("teach", "reteach", "zero") or (
            self.kind in ("baseline", "in_context", "notes") and self.copy_baseline)

    copy_baseline = False
    system = None      # the in_context step's system prompt; a notes step's, one per question ({qid: block or None})
    pocket_from = None  # a notes_pocket step: the step whose taught folder it serves (teach, PHASE.reteach, PHASE.zero)
    notes = None        # a notes step's note store and retrievals, worked out once (Run.notes_plan)


class Run:
    def __init__(self, a):
        self.args = a
        self.set_path = os.path.abspath(a.set)
        self.S = read_json(self.set_path)
        if not isinstance(self.S, dict) or self.S.get("format") != "lw-test-set/1":
            raise RunError(f"{a.set}: not an lwlogic test set (format lw-test-set/1)")
        if self.S.get("l7"):
            raise RunError(f"{a.set}: a level 7 set (fill to capacity): run it with zengine/capacity.py, which asks the "
                           "model at every checkpoint while it teaches; here its checkpoints would be read as deletes")
        self.work = os.path.abspath(a.work)
        self.models = os.path.abspath(a.models_dir or os.path.join(self.work, "models"))
        self.pidfile = os.path.join(self.work, "server.pid")
        self.port = a.port
        self.bin = os.path.abspath(a.bin)
        self.src = os.path.abspath(a.model_src)
        self.lwlogic = os.path.abspath(a.lwlogic)
        self.set_for_judge = os.path.join(self.work, "set.json")
        self.thinking = []
        for x in a.thinking.split(","):
            x = x.strip().lower()
            if x not in ("off", "on"):
                raise RunError(f"--thinking takes off and/or on, not {x!r}")
            self.thinking.append(x == "on")
        self.ref_kinds, self.ref_of = self.reference_args()
        self._notes = None
        if any(k in NOTE_KINDS for k in self.ref_kinds):
            self.notes_setup()
        elif any(getattr(a, x, None) is not None for x in ("notes_tokenizer", "notes_cap", "notes_config")):
            raise RunError("--notes-tokenizer, --notes-cap and --notes-config need --reference notes or notes_pocket")
        self.steps = self.plan(found=a.judge_only)
        self.summary = None
        self.bin_sha = None  # sha256 of --bin, set when the run starts; kept in every probed step's state
        self.copies_path = os.path.join(self.work, "copies.json")

    # ---------------------------------------------------------------------------------- planning
    def lesson_name(self, i):
        return next((l["name"] for l in self.S["lessons"] if l["id"] == i), f"#{i}")

    def pairs(self):
        return {(q["qid"], th) for q in self.S["questions"] for th in self.thinking}

    def reference_args(self):
        """--reference KINDS and --reference-of PHASES, checked: (kinds, phase names). Without --reference: ([], [])."""
        a = self.args
        if not getattr(a, "reference", None):
            if getattr(a, "reference_of", None) is not None:
                raise RunError("--reference-of needs --reference (in_context, replacement_memory or both); without it "
                               "no reference phase would run")
            return [], []
        kinds = []
        for x in a.reference.split(","):
            x = x.strip()
            for k in (REF_KINDS if x in ("both", "all") else [x]):
                if k not in ALL_REF_KINDS:
                    raise RunError(f"--reference takes in_context, replacement_memory, notes and notes_pocket (both: the "
                                   f"first two), not {x!r}")
                if k not in kinds:
                    kinds.append(k)
        names = [p["name"] for p in self.S["phases"]]
        stores = {p["name"]: p["store"] for p in self.S["phases"]}
        of = getattr(a, "reference_of", None)
        want = [x.strip() for x in (of if of is not None else "teach").split(",") if x.strip()]
        if not want:
            raise RunError(f"--reference-of {of!r} names no phase (teach, a comma list of phases, or all)")
        if want == ["all"]:
            return kinds, [n for n in names if n != "baseline" and stores[n]]
        for x in want:
            if x not in names:
                raise RunError(f"--reference-of: {x!r} is not a phase of {self.S['id']} ({', '.join(names)}; or 'all')")
            if not stores[x]:
                raise RunError(f"--reference-of: phase {x} keeps no lesson, so it has nothing to put in context "
                               "(the baseline is that reference)")
        return kinds, list(dict.fromkeys(want))

    def reference_found(self):
        """Reference steps an earlier call left in WORK (phases/PHASE.KIND), so that finish() and --judge-only see
        them whatever this call's --reference says."""
        out = []
        for p in self.S["phases"]:
            for k in ALL_REF_KINDS:
                if p["store"] and os.path.isdir(os.path.join(self.work, "phases", f"{p['name']}.{k}")):
                    out.append((p["name"], k))
        return out

    def reference_steps(self, found=False):
        want = [(ph, k) for k in self.ref_kinds for ph in self.ref_of]
        if found:
            want += [x for x in self.reference_found() if x not in want]
        phases = {p["name"]: p for p in self.S["phases"]}
        order = {p["name"]: i for i, p in enumerate(self.S["phases"])}
        steps = []
        for ph, kind in sorted(want, key=lambda x: (ALL_REF_KINDS.index(x[1]), order[x[0]])):
            p = phases[ph]
            s = Step(f"{ph}.{kind}", ph, kind, store=p["store"], deleted=p["deleted"])
            if kind == "in_context":
                s.copy_baseline = self.args.baseline_from == "copy"
                s.system = in_context_prompt(self.S, p["store"])
                s.flavors = [(s.key, "", "plain" if s.copy_baseline else "plain-src", "in_context")]
            elif kind == "notes":
                s.copy_baseline = self.args.baseline_from == "copy"
                s.flavors = [(s.key, "", "plain" if s.copy_baseline else "plain-src", "notes")]
            elif kind == "notes_pocket":
                s.pocket_from = "teach" if ph == "teach" else (
                    f"{ph}.zero" if self.args.delete_mode == "zero" else f"{ph}.reteach")
                s.flavors = [(s.key, "", "plain", "notes_pocket")]
            else:
                s.flavors = [(s.key, "", "computed", "replacement_memory")]
            steps.append(s)
        return steps

    def notes_setup(self):
        """(notes module, tokenizer, config) of the notes phases, loaded once, so a missing tokenizer or a bad flag is
        refused before anything runs. The tokenizer is --notes-tokenizer, or the source model's own tokenizer.json."""
        if self._notes is None:
            a = self.args
            nm = notes_module()
            spec = a.notes_tokenizer or os.path.join(self.src, "tokenizer.json")
            try:
                tok = nm.load_tokenizer(spec)
            except nm.NotesError as e:
                raise RunError(f"the notes phases need the serving model's tokenizer (--notes-tokenizer): {e}")
            try:
                cfg = json.loads(a.notes_config) if a.notes_config else {}
            except ValueError as e:
                raise RunError(f"--notes-config is not JSON: {e}")
            if not isinstance(cfg, dict) or [k for k in cfg if k not in nm.DEFAULTS]:
                raise RunError(f"--notes-config takes an object with keys of {sorted(nm.DEFAULTS)}, not {a.notes_config}")
            if a.notes_cap is not None:
                cfg["cap"] = a.notes_cap
            if not isinstance(cfg.get("cap", 1), int) or cfg.get("cap", 1) < 1:
                raise RunError(f"--notes-cap is a positive number of tokens, not {cfg.get('cap')!r}")
            try:
                nm.NoteStore(tok, config=cfg)
            except nm.NotesError as e:
                raise RunError(f"--notes-config: {e}")
            self._notes = (nm, tok, cfg)
        return self._notes

    def notes_plan(self, step):
        """The notes of a notes step: its phase's lessons written into a note store in teaching order, and each
        question's retrieval (zengine/notes.py). Deterministic for one tokenizer and config, so a resumed step asks
        with the same blocks. Sets step.system to {qid: block or None}."""
        if step.notes is None:
            nm, tok, cfg = self.notes_setup()
            ns, writes = nm.set_store(self.S, step.store, tok, cfg)
            ret = {q["qid"]: ns.retrieve("user", q["text"]) for q in self.S["questions"]}
            step.notes = {"ret": ret, "notes": ns.notes_of("user"), "writes": writes, "describe": ns.describe(),
                          "stats": ns.stats("user")}
            step.system = {qid: (r["block"] or None) for qid, r in ret.items()}
        return step.notes

    def notes_extra(self, step):
        """The fields each answer line of a notes step carries: the note ids given and their tokens."""
        ret = self.notes_plan(step)["ret"]
        return {qid: {"notes": r["ids"], "notes_tokens": r["tokens"]} for qid, r in ret.items()}

    def write_notes(self, step, st, save):
        """WORK/phases/KEY/notes.jsonl, one line per question: the note ids retrieved (current and earlier), the tokens
        injected (framing included), the cap, the lookup time in microseconds and the block itself;
        notes-store.json: the notes and what each write did; and a summary in the step's state."""
        nm, _tok, cfg = self.notes_setup()
        plan = self.notes_plan(step)
        d = self.sdir(step)
        path = os.path.join(d, "notes.jsonl")
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            for q in self.S["questions"]:
                r = plan["ret"][q["qid"]]
                f.write(json.dumps({"qid": q["qid"], "question": q["text"], "past": r["past"], "notes": r["ids"],
                                    "current": r["current"], "earlier": r["history"], "tokens": r["tokens"],
                                    "cap": r["cap"], "lookup_us": r["lookup_us"], "candidates": r["candidates"],
                                    "block_sha256": hashlib.sha256(r["block"].encode()).hexdigest() if r["block"]
                                    else None, "block": r["block"]}, ensure_ascii=False) + "\n")
        os.replace(path + ".tmp", path)
        store_path = os.path.join(d, "notes-store.json")
        write_json(store_path, {"format": nm.NOTES_FORMAT, "phase": step.phase, "store": step.store,
                                "store_how": plan["describe"],
                                "notes": [dict(n, tokens=len(n["tokens"])) for n in plan["notes"]],
                                "writes": plan["writes"]})
        ret = plan["ret"].values()
        st["notes"] = {"format": nm.NOTES_FORMAT, "tokenizer": plan["describe"]["tokenizer"],
                       "config": plan["describe"]["config"], "notes": plan["stats"]["notes"],
                       "current": plan["stats"]["current"], "earlier": plan["stats"]["history"],
                       "kept_out": sum(1 for w in plan["writes"] if not w["note"]), "questions": len(ret),
                       "nothing_retrieved": sum(1 for r in ret if not r["ids"]),
                       "tokens_p50": pctl([r["tokens"] for r in ret], 50), "tokens_p95": pctl([r["tokens"] for r in ret], 95),
                       "tokens_max": max(r["tokens"] for r in ret), "lookup_us_p50": pctl([r["lookup_us"] for r in ret], 50),
                       "lookup_us_p95": pctl([r["lookup_us"] for r in ret], 95), "log": os.path.relpath(path, self.work),
                       "store_file": os.path.relpath(store_path, self.work)}
        save()
        LOG(f"step {step.key}: {plan['stats']['notes']} notes ({plan['stats']['current']} current) from "
            f"{len(step.store)} lessons; per question {st['notes']['tokens_p50']} tokens (p50, cap {cfg.get('cap', nm.DEFAULTS['cap'])}"
            f", framing included), lookup {st['notes']['lookup_us_p50']} us (p50); nothing retrieved for "
            f"{st['notes']['nothing_retrieved']} of {len(ret)} questions (no system message there)")

    def plan(self, found=False):
        a = self.args
        names = [p["name"] for p in self.S["phases"]]
        dels = [p for p in self.S["phases"] if p["name"] not in ("baseline", "teach")]
        if a.phases:
            want = []
            for x in a.phases.split(","):
                x = x.strip()
                if x == "deletes":
                    want += [p["name"] for p in dels]
                elif x in names:
                    want.append(x)
                else:
                    raise RunError(f"--phases: {x!r} is not a phase of {self.S['id']} ({', '.join(names)}; or 'deletes')")
        else:
            want = names
        modes = ["zero", "reteach"] if a.delete_mode == "both" else [a.delete_mode]
        probe_flavors = {"slide": [("", "slide")], "restart": [(".restart", "plain")],
                         "both": [("", "slide"), (".restart", "plain")]}[a.teach_probe]
        steps = []
        if "baseline" in want:
            s = Step("baseline", "baseline", "baseline")
            s.copy_baseline = a.baseline_from == "copy"
            s.flavors = [("baseline", "", "plain" if s.copy_baseline else "plain-src", "original")]
            steps.append(s)
        if "teach" in want:
            teach = next(p for p in self.S["phases"] if p["name"] == "teach")
            s = Step("teach", "teach", "teach", store=teach["store"])
            s.flavors = [("teach" + suf, suf, how, "slide" if how == "slide" else "restart") for suf, how in probe_flavors]
            steps.append(s)
        sel = [p for p in dels if p["name"] in want]
        if "zero" in modes:
            for p in sel:
                s = Step(p["name"] + ".zero", p["name"], "zero", store=p["store"], deleted=p["deleted"])
                s.flavors = [(s.key, "", "plain", "zero")]
                steps.append(s)
        if "reteach" in modes:
            for p in sel:
                if not p["store"]:
                    s = Step(p["name"] + ".reteach", p["name"], "reteach-empty", store=[], deleted=p["deleted"])
                    s.flavors = [(s.key, "", "plain-src", "reteach (empty store: the original model)")]
                else:
                    s = Step(p["name"] + ".reteach", p["name"], "reteach", store=p["store"], deleted=p["deleted"])
                    s.flavors = [(s.key + suf, suf, how, "reteach" + suf) for suf, how in probe_flavors]
                steps.append(s)
        return steps + self.reference_steps(found)  # after the pocket's own steps, whatever --phases picked

    def sdir(self, step):
        return os.path.join(self.work, "phases", step.key)

    def copy_path(self, step):
        return os.path.join(self.models, f"{self.S['id']}.{step.key}")

    def files(self, step, suf):
        d = self.sdir(step)
        return (os.path.join(d, f"answers{suf}.jsonl"), os.path.join(d, f"verdicts{suf}.json"),
                os.path.join(d, f"report{suf}.txt"), os.path.join(d, f"errors{suf}.jsonl"))

    # ---------------------------------------------------------------------------------- guards
    def guard_teachable(self, folder):
        """--slide rewrites the folder: only a copy this runner made, never the source or the HF cache."""
        rf = os.path.realpath(folder)
        if rf == os.path.realpath(self.src) or "/.cache/huggingface/" in rf + "/":
            raise RunError(f"refusing --slide on {folder}: it is the source or inside the Hugging Face cache")
        reg = read_json(self.copies_path, {"copies": []})
        if not any(os.path.realpath(c["path"]) == rf for c in reg["copies"]):
            raise RunError(f"refusing --slide on {folder}: not a copy this runner made (copies.json)")
        bad = []
        for root, dirs, files in os.walk(folder):
            for x in files + dirs:
                p = os.path.join(root, x)
                if os.path.islink(p):
                    bad.append(f"{p} is a symlink")
                elif os.path.isfile(p) and os.stat(p).st_nlink > 1:
                    bad.append(f"{p} has {os.stat(p).st_nlink} hard links")
        if bad:
            raise RunError("refusing --slide: the copy shares files with another folder:\n" + "\n".join(bad[:10]))

    def check_pidfile(self):
        d = read_json(self.pidfile)
        if not d or d.get("state") != "running" or not d.get("pid"):
            return
        if any(s.pid == d["pid"] for s in LIVE):
            return
        cmd = pid_command(d["pid"])
        if cmd and (d.get("bin") or "") in cmd:
            raise RunError(f"{self.pidfile} names server pid {d['pid']} ({d.get('label')}), still running: {cmd}\n"
                           f"It was started by an earlier lwrun (runner pid {d.get('runner_pid')}), not by this run, "
                           f"so this run will not stop it. Stop it, then rerun.")
        LOG(f"stale pidfile: pid {d['pid']} is gone (or is now another program); going on")

    def write_pidfile(self, srv, state):
        write_json(self.pidfile, {"state": state, "pid": srv.pid, "port": self.port, "label": srv.label,
                                  "bin": self.bin, "folder": srv.folder, "cmd": srv.cmd(), "runner_pid": os.getpid(),
                                  "at": utc()})

    # ---------------------------------------------------------------------------------- copies
    def make_copy(self, step, src, kind_note):
        dst = self.copy_path(step)
        if os.path.exists(dst):
            raise RunError(f"{dst} exists; refusing to reuse a copy without --resume")
        need = tree_bytes(src)
        free = free_bytes(self.models)
        # a first save widens every shard holding a shared expert and rewrites it whole (shard_edit.grow), so a
        # taught APFS clone ends up costing about the full checkpoint; a zeroed clone costs a few KB
        want = int(need * 1.05) + 2 * GIB if (IS_LINUX or step.kind in ("teach", "reteach")) else 2 * GIB
        if free < want:
            raise RunError(f"not enough disk for {dst}: {gb(free)} free, {gb(want)} wanted ({kind_note})")
        os.makedirs(self.models, exist_ok=True)
        cmd = ["cp", "-cRL", src, dst] if IS_MAC else ["cp", "-RL", src, dst]
        LOG(f"copy [{step.key}] ({kind_note}): {' '.join(shlex.quote(x) for x in cmd)}")
        t0 = time.time()
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            raise RunError(f"copy failed ({p.returncode}): {p.stderr.strip()[:400]}")
        reg = read_json(self.copies_path, {"copies": []})
        reg["copies"].append({"path": dst, "step": step.key, "src": src, "made": utc(), "bytes": tree_bytes(dst)})
        write_json(self.copies_path, reg)
        secs = round(time.time() - t0, 1)
        LOG(f"copied in {secs} s: {dst} ({gb(tree_bytes(dst))} logical)")
        return dst, secs

    def set_aside(self, path, why):
        """A copy or file that cannot be reused: renamed aside (never deleted on macOS); Linux --cleanup removes
        the runner's own copies."""
        if not os.path.exists(path):
            return
        if IS_LINUX and self.args.cleanup and os.path.isdir(path) and self.is_our_copy(path):
            LOG(f"remove (Linux --cleanup) {path}: {why}")
            shutil.rmtree(path)
            self.mark_removed(path)
            return
        new = f"{path}.stale-{stamp()}"
        LOG(f"set aside {path} -> {new}: {why}")
        os.rename(path, new)
        reg = read_json(self.copies_path, {"copies": []})
        for c in reg["copies"]:
            if c["path"] == path:
                c["path"], c["set_aside"] = new, utc()
        if os.path.exists(self.copies_path):
            write_json(self.copies_path, reg)

    def stash(self, step, why):
        """Answers, verdicts, graphs and SSE files of an unfinished attempt, moved into a stale-STAMP folder."""
        d = self.sdir(step)
        names = [x for x in sorted(os.listdir(d)) if x not in ("state.json", "server.log") and not x.startswith("stale-")
                 and (x.startswith(("answers", "verdicts", "report", "errors", "graph", "learn-")) or x == "zero.json")]
        if not names:
            return
        dest = os.path.join(d, f"stale-{stamp()}")
        os.makedirs(dest)
        for x in names:
            os.rename(os.path.join(d, x), os.path.join(dest, x))
        LOG(f"moved {len(names)} files of {why} into {dest}")

    def is_our_copy(self, path):
        rp = os.path.realpath(path)
        reg = read_json(self.copies_path, {"copies": []})
        return rp.startswith(os.path.realpath(self.models) + os.sep) and any(
            os.path.realpath(c["path"]) == rp for c in reg["copies"])

    def mark_removed(self, path):
        reg = read_json(self.copies_path, {"copies": []})
        for c in reg["copies"]:
            if os.path.realpath(c["path"]) == os.path.realpath(path):
                c["removed"] = utc()
        write_json(self.copies_path, reg)

    def cleanup_copy(self, path, why=None):
        if not (IS_LINUX and self.args.cleanup and path and os.path.isdir(path)):
            return
        if not self.is_our_copy(path):
            LOG(f"--cleanup: {path} is not a copy this runner registered; left alone")
            return
        LOG(f"--cleanup: remove {path} ({gb(tree_bytes(path))})" + (f": {why}" if why else ""))
        shutil.rmtree(path)
        self.mark_removed(path)

    def drop_unwritten_copy(self, path, why):
        """A copy nothing was written to and nothing will reuse: a zero copy whose zeroing was skipped (a resumed zero
        step copies the taught folder again). Linux --cleanup removes it at once; otherwise it stays, like every
        copy."""
        self.cleanup_copy(path, why)

    # ---------------------------------------------------------------------------------- state
    def load_state(self, step):
        return read_json(os.path.join(self.sdir(step), "state.json"), None)

    def saver(self, step, st):
        path = os.path.join(self.sdir(step), "state.json")
        return lambda: write_json(path, st)

    def preflight(self):
        """Before anything runs: nothing reused without --resume, the source sane, the judge reachable."""
        a = self.args
        problems = []
        if not a.judge_only:
            if not (os.path.isfile(self.bin) and os.access(self.bin, os.X_OK)):
                problems.append(f"--bin {self.bin} is not an executable file")
            if not os.path.isfile(os.path.join(self.src, "config.json")):
                problems.append(f"--model-src {self.src} has no config.json")
            else:
                cfg = read_json(os.path.join(self.src, "config.json"), {})
                if "tensorfold_learned_neurons" in cfg and not a.allow_taught_src:
                    problems.append(f"--model-src {self.src} already holds learned neurons (config.json "
                                    "tensorfold_learned_neurons): baseline would not be the original model "
                                    "(--allow-taught-src to go on anyway)")
            if a.cleanup and not IS_LINUX:
                problems.append("--cleanup is Linux only: on macOS copies are never deleted (move them to the Trash)")
            if a.port in (8091,):
                problems.append(f"--port {a.port} is reserved (another server lives there)")
            elif port_busy(a.port):
                problems.append(f"--port {a.port} is in use (or cannot be bound)")
        if not os.path.isfile(self.lwlogic):
            problems.append(f"lwlogic.py not found at {self.lwlogic} (--lwlogic)")
        elif any(s.kind in ALL_REF_KINDS for s in self.steps):
            try:  # the memory's wording and the outcome readers come from it: fail now, not after hours of teaching
                lw_module(self.lwlogic)
            except RunError as e:
                problems.append(str(e))
        planned = {s.key for s in self.steps}
        for s in self.steps:
            if s.kind == "notes_pocket" and not a.judge_only and s.pocket_from not in planned:
                sst = self.load_state(self.pocket_step(s)) or {}
                if not (sst.get("folder") and os.path.isdir(sst["folder"])):
                    problems.append(f"{s.key} serves the folder step {s.pocket_from} taught, which this run does not "
                                    "make and no earlier run left: add it to --phases (and --delete-mode)")
        if not a.judge_only:
            try:  # an earlier runner's server still up: nothing may be renamed or copied under it
                self.check_pidfile()
            except RunError as e:
                problems.append(str(e))
        old = read_json(os.path.join(self.work, "set.json"))
        if old is not None and json.dumps(old, sort_keys=True) != json.dumps(self.S, sort_keys=True):
            problems.append(f"{self.work} was used for another set ({old.get('id')}); use a new --work")
        if not a.resume and not a.judge_only:
            for s in self.steps:
                if os.path.exists(self.sdir(s)):
                    problems.append(f"{self.sdir(s)} exists (an earlier run of {s.key}); --resume to continue it")
                if s.needs_copy() and os.path.exists(self.copy_path(s)):
                    problems.append(f"{self.copy_path(s)} exists; --resume to reuse it")
        if problems:
            raise RunError("refusing to start:\n  " + "\n  ".join(problems))

    # ---------------------------------------------------------------------------------- one step
    def run_step(self, step):
        a = self.args
        sdir = self.sdir(step)
        if a.judge_only and not os.path.isdir(sdir):
            return {"status": "not run"}
        os.makedirs(sdir, exist_ok=True)
        st = self.load_state(step) or {"key": step.key, "phase": step.phase, "kind": step.kind,
                                       "store": step.store, "deleted": step.deleted, "answers": {}, "judge": {},
                                       "timings": {}, "created": utc()}
        save = self.saver(step, st)
        st["status"], st["error"] = "running", None
        save()
        want = self.pairs()
        files = {f[0]: self.files(step, f[1]) for f in step.flavors}
        if step.kind == "in_context" and not a.judge_only:
            self.check_prompt(step, st, files, save)
        if step.kind in NOTE_KINDS and not a.judge_only:
            self.check_notes_prompt(step, st, files, save)

        def complete(akey):
            return want <= answered_pairs(files[akey][0])

        def judged(akey):
            j = st["judge"].get(akey) or {}
            return complete(akey) and j.get("rc") == 0 and j.get("n") == len(want)

        if a.judge_only:
            for akey, _suf, _how, _mode in step.flavors:
                if os.path.exists(files[akey][0]):
                    self.judge_step(step, st, akey, files[akey], save)
            if step.kind == "zero":  # read again from the teach records: older records have none, or older rules
                lim = self.zero_limits_of(step)
                if lim is not None:
                    st["zero_limits"] = lim
            st["status"] = NOT_APPLICABLE if st.get("not_applicable") else "judged"
            save()
            return st
        if step.kind == "zero" and st.get("not_applicable"):
            if self.zero_na_holds():  # finished: nothing to zero, no answers to ask for
                LOG(f"step {step.key}: not applicable ({st['not_applicable']}); skipped (--resume)")
                st["status"] = NOT_APPLICABLE
                save()
                return st
            LOG(f"step {step.key}: was not applicable, but the teach records or the taught folder changed since; "
                "zeroing is tried again")
            st.pop("not_applicable", None)
        if all(judged(f[0]) for f in step.flavors):
            LOG(f"step {step.key}: every answer on file and judged; skipped (--resume)")
            st["status"] = "done"
            save()
            return st
        if all(complete(f[0]) for f in step.flavors):
            LOG(f"step {step.key}: every answer on file; judging only")
            for akey, *_ in step.flavors:
                self.judge_step(step, st, akey, files[akey], save)
            st["status"] = "done"
            save()
            return st
        LOG(f"=== step {step.key} ({step.kind}): phase {step.phase}, store "
            f"{[self.lesson_name(i) for i in step.store]}" + (f", deleted {[self.lesson_name(i) for i in step.deleted]}" if step.deleted else ""))
        t_step = time.time()
        if step.kind == "baseline":
            self.step_original(step, st, save, files)
        elif step.kind == "reteach-empty":
            self.step_empty(step, st, save, files)
        elif step.kind in ("teach", "reteach"):
            self.step_teach(step, st, save, files)
        elif step.kind == "zero":
            self.step_zero(step, st, save, files)
        elif step.kind == "in_context":
            self.step_original(step, st, save, files)
        elif step.kind == "notes":
            self.write_notes(step, st, save)
            self.step_original(step, st, save, files)
        elif step.kind == "notes_pocket":
            self.write_notes(step, st, save)
            self.step_notes_pocket(step, st, save, files)
        elif step.kind == "replacement_memory":
            self.step_memory(step, st, save, files)
        st["timings"]["step_s"] = round(time.time() - t_step, 1)
        if st.get("not_applicable"):
            st["status"] = NOT_APPLICABLE
        else:
            st["status"] = "done" if all(complete(f[0]) for f in step.flavors) else "incomplete"
        save()
        return st

    def judge_step(self, step, st, akey, f, save):
        ans, ver, rep, _err = f
        j = run_judge(self, [ans], ver, rep, akey)
        j["n"] = len(answered_pairs(ans) & self.pairs())
        st["judge"][akey] = j
        save()
        if step.kind in ("teach", "reteach") and st.get("refused_lessons"):
            self.judge_refused(step, st, akey, ans, save)

    def judge_refused(self, step, st, akey, ans, save):
        """LW-TEST-LOGIC.md section 8: a refused lesson is still in the user's store; to tell a refusal from
        learning wrongly, the same answers are also judged against the phase that never had those lessons."""
        refused = set(st["refused_lessons"])
        keep = sorted(set(step.store) - refused)
        match = next((p for p in self.S["phases"] if sorted(p["store"]) == keep), None)
        if not match:
            st.setdefault("refused_judge", {})[akey] = {
                "refused": sorted(refused), "phase": None,
                "note": "no phase of the set keeps exactly the lessons that were not refused; lwlogic.py has no "
                        "CLI to add one"}
            LOG(f"refused lessons {sorted(refused)}: no phase in the set keeps exactly {keep}; not judged that way")
            save()
            return
        d, base = os.path.split(ans)                       # answers[.restart].jsonl
        tag = base[len("answers"):-len(".jsonl")] + f".as-{match['name']}"
        rel = os.path.join(d, f"answers{tag}.jsonl")
        with open(ans, encoding="utf-8") as fi, open(rel, "w", encoding="utf-8") as fo:
            for line in fi:
                if line.strip():
                    r = json.loads(line)
                    r["phase"], r["judged_as"] = match["name"], f"{step.phase} without refused lessons"
                    fo.write(json.dumps(r, ensure_ascii=False) + "\n")
        j = run_judge(self, [rel], os.path.join(d, f"verdicts{tag}.json"), os.path.join(d, f"report{tag}.txt"),
                      f"{akey} as {match['name']}")
        j["refused"], j["phase"] = sorted(refused), match["name"]
        st.setdefault("refused_judge", {})[akey] = j
        save()

    def serve_and_probe(self, step, st, save, files, folder, flavor, slide_srv=None):
        """Probe with an already running server (slide), or a plain one started here."""
        akey, _suf, how, mode = flavor
        ans, _ver, _rep, err = files[akey]
        srv = slide_srv
        own = srv is None
        if own:
            srv = Server(self, folder, False, os.path.join(self.sdir(step), "server.log"), label=akey)
            srv.start()
        try:
            res = probe(self, srv, akey, step.phase, mode, ans, err, system=step.system,
                        extra=self.notes_extra(step) if step.kind in NOTE_KINDS else None)
        finally:
            if own:
                srv.stop("probing done")
        st["answers"][akey] = res
        st["binary_sha256"] = self.bin_sha
        st["timings"][f"{akey}.server"] = srv.timings
        save()
        LOG(f"probed {akey}: {res['n']}/{res['expected']} answers in {res['seconds']} s")
        return res

    def step_original(self, step, st, save, files):
        """The original model, served read-only from the source (or a fresh copy with --baseline-from copy): the
        baseline, an empty store served again, the in_context reference (with its system prompt) and the notes
        phase (with each question's notes)."""
        flavor = step.flavors[0]
        if step.kind == "in_context":
            st["in_context"] = {"format": IN_CONTEXT_FORMAT, "lessons": len(step.store),
                                "system_sha256": hashlib.sha256(step.system.encode()).hexdigest(),
                                "system_prompt": step.system}
            save()
        if step.copy_baseline:
            folder = st.get("folder")
            if not (folder and os.path.isdir(folder)):
                folder, secs = self.make_copy(step, self.src, f"{step.kind}: a fresh copy, served without --slide")
                st["folder"], st["timings"]["copy_s"] = folder, secs
                save()
            self.serve_and_probe(step, st, save, files, folder, flavor)
        else:
            st["folder"] = self.src
            LOG(f"serve the source read-only, without --slide: {self.src}")
            before = fingerprint(self.src)
            self.serve_and_probe(step, st, save, files, self.src, flavor)
            after = fingerprint(self.src)
            changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
            st["source_unchanged"] = not changed
            if changed:
                LOG(f"WARNING: the source folder changed while served: {changed[:10]}")
        self.judge_step(step, st, flavor[0], files[flavor[0]], save)

    def check_prompt(self, step, st, files, save):
        """An in_context step's answers on file must all have been asked with today's prompt (system_sha256 in every
        line): answers asked with another prompt (another store, or another wording) are moved aside, never mixed."""
        ans = files[step.flavors[0][0]][0]
        if not os.path.exists(ans):
            return
        want = hashlib.sha256(step.system.encode()).hexdigest()
        with open(ans, encoding="utf-8") as f:
            seen = {json.loads(x).get("system_sha256") for x in f if x.strip()}
        if seen - {want}:
            self.stash(step, "answers asked with another in_context prompt")
            st["answers"], st["judge"] = {}, {}
            st.setdefault("notes", []).append(f"{utc()}: answers asked with another in_context prompt moved aside")
            save()

    def check_notes_prompt(self, step, st, files, save):
        """A notes step's answers on file must each have been asked with today's block for its question (system_sha256,
        None where nothing was retrieved): answers asked with other notes (another store, tokenizer, cap or wording)
        are moved aside and asked again, never mixed."""
        ans = files[step.flavors[0][0]][0]
        if not os.path.exists(ans):
            return
        self.notes_plan(step)
        want = {qid: hashlib.sha256(b.encode()).hexdigest() if b else None for qid, b in step.system.items()}
        with open(ans, encoding="utf-8") as f:
            rows = [json.loads(x) for x in f if x.strip()]
        if any(r.get("system_sha256") != want.get(r["qid"]) for r in rows):
            self.stash(step, "answers asked with other notes")
            st["answers"], st["judge"] = {}, {}
            st.setdefault("notes_log", []).append(f"{utc()}: answers asked with other notes moved aside")
            save()

    def pocket_step(self, step):
        """The step whose taught folder a notes_pocket step serves."""
        key = step.pocket_from
        return Step(key, step.phase, "teach" if key == "teach" else key.rsplit(".", 1)[1])

    def step_notes_pocket(self, step, st, save, files):
        """notes_pocket: the folder the phase's own step taught (teach; a delete phase's reteach copy, or zero copy),
        served without --slide as zero copies are, with each question's notes in the system prompt. Read only: the
        folder's fingerprint before and after is compared and kept (folder_unchanged)."""
        src = self.pocket_step(step)
        sst = self.load_state(src) or {}
        folder = sst.get("folder")
        if sst.get("status") not in ("done", "judged") or not (folder and os.path.isdir(folder)):
            why = (f"its status is {sst.get('status')!r}" if sst.get("status") not in ("done", "judged") else
                   f"its folder {folder} is gone" if folder else "it has no folder")
            raise RunError(f"{step.key} serves the folder step {src.key} taught, and {why}: run {src.key} first (it "
                           "is in --phases and --delete-mode), and keep its copy")
        st["folder"], st["pocket_from"] = folder, src.key
        save()
        LOG(f"serve the pocket of {src.key} without --slide, with each question's notes: {folder}")
        before = fingerprint(folder)
        self.serve_and_probe(step, st, save, files, folder, step.flavors[0])
        after = fingerprint(folder)
        changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
        st["folder_unchanged"] = not changed
        if changed:
            LOG(f"WARNING: the pocket folder changed while served: {changed[:10]}")
        self.judge_step(step, st, step.flavors[0][0], files[step.flavors[0][0]], save)

    def step_memory(self, step, st, save, files):
        """replacement_memory: no model and no server. The slots (memory_slots) answer every question
        (memory_answer); the answers are written like a probe's, once per thinking mode so the judge's tables line
        up, with finish "computed", and judged like any other answers."""
        lw = lw_module(self.lwlogic)
        akey = step.flavors[0][0]
        ans = files[akey][0]
        if os.path.exists(ans):
            self.stash(step, "an unfinished replacement-memory step")
        t0 = time.time()
        slots = memory_slots(self.S, step.store)
        want, n = self.pairs(), 0
        tmp = ans + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fo:
            for th in self.thinking:
                for q in self.S["questions"]:
                    fo.write(json.dumps({"set": self.S["id"], "key": step.key, "phase": step.phase,
                                         "mode": "replacement_memory", "qid": q["qid"], "thinking": th,
                                         "question": q["text"], "content": memory_answer(lw, self.S, slots, q),
                                         "reasoning_content": "", "finish": "computed", "ms": 0, "usage": None,
                                         "max_tokens": None, "computed": MEMORY_FORMAT, "at": utc()},
                                        ensure_ascii=False) + "\n")
                    n += 1
        os.replace(tmp, ans)
        st["folder"] = None
        st["memory"] = {"format": MEMORY_FORMAT, "slots": [{"e": e, "a": a, "value": v}
                                                           for (e, a), v in sorted(slots.items())]}
        st["answers"][akey] = {"path": ans, "n": n, "expected": len(want), "complete": want <= answered_pairs(ans),
                               "seconds": round(time.time() - t0, 1), "computed": True}
        save()
        LOG(f"step {step.key}: computed {n} answers from {len(slots)} slots (latest value per key; no model, no "
            f"server): " + ", ".join(f"{e} {a} = {v}" for (e, a), v in sorted(slots.items())))
        self.judge_step(step, st, akey, files[akey], save)

    def zero_not_applicable(self, tst, tfolder):
        """Why the zero steps have nothing to do, or None. Zeroing removes the deleted facts' rows from the taught
        folder's pocket. When that folder holds no pocket and the teach records say nothing was stored (every lesson
        missed, every save 0 bytes), there are no rows to remove: a zero copy would be the taught folder byte for
        byte, and the step is not applicable (finished, with this reason on file, no copy made). A folder with no
        pocket while the records say something was stored is a mismatch: ZeroSkip, before any copy is made."""
        if pocket_size(tfolder) is not None:
            return None
        if not os.path.isfile(os.path.join(tfolder, "config.json")):
            raise ZeroSkip(f"the taught folder {tfolder} has no config.json")
        why = nothing_stored(tst)
        if why is None:
            held = sorted(f for f, s in (tst.get("fact_states") or {}).items() if s in ("learned", "replaced"))
            raise ZeroSkip(f"{NO_POCKET}, but the teach records say something was stored (facts {held} learned or "
                           "replaced, or a save that wrote bytes)")
        return f"{why}, and the taught folder holds no pocket ({NO_POCKET.split(':')[0]}): nothing to zero"

    def zero_na_holds(self):
        """On --resume: whether a zero step found not applicable still is. The teach records must still say nothing
        was stored, and the taught folder, when it is still there (--cleanup removes it), must still hold no
        pocket."""
        tst = self.load_state(Step("teach", "teach", "teach")) or {}
        tf = tst.get("folder")
        return nothing_stored(tst) is not None and not (tf and os.path.isdir(tf) and pocket_size(tf) is not None)

    def zero_limits_of(self, step):
        """zero_limits() of a zero step, from the teach step's records on file; None when nothing was taught yet."""
        tst = self.load_state(Step("teach", "teach", "teach")) or {}
        return zero_limits(tst, step.deleted) if tst.get("taught") else None

    def reuse_problem(self):
        """Why the baseline answers cannot stand in for an empty store, or None when they can: they must be on file
        for every question and thinking mode of this run, from the source unchanged, the same binary and the same
        max_tokens (answers from an older runner carry no max_tokens and no binary: taken as the same)."""
        bstep = Step("baseline", "baseline", "baseline")
        path = self.files(bstep, "")[0]
        if not os.path.exists(path):
            return "no baseline answers on file"
        want = self.pairs()
        have = answered_pairs(path)
        if not want <= have:
            return f"the baseline answers are incomplete ({len(want & have)} of {len(want)})"
        bst = self.load_state(bstep) or {}
        if bst.get("source_unchanged") is False:
            return "the source folder changed while the baseline was served"
        if bst.get("binary_sha256") and self.bin_sha and bst["binary_sha256"] != self.bin_sha:
            return "the baseline was asked with another binary"
        a = self.args
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    mt = r.get("max_tokens")
                    if mt is not None and mt != (a.max_tokens_on if r["thinking"] else a.max_tokens_off):
                        return f"the baseline was asked with max_tokens {mt} ({r['qid']}, thinking {r['thinking']})"
        return None

    def step_empty(self, step, st, save, files):
        """An empty store is the original model, which the baseline step already asked with the same questions. Its
        answers are reused, filed under this phase, instead of loading the model again. In our stored runs, serving
        the original again gave the baseline answers word for word (TEST-LOGIC.md section 8). Served again with
        --fresh-empty-reteach, or when the baseline answers cannot stand in (reuse_problem)."""
        flavor = step.flavors[0]
        akey = flavor[0]
        why = "--fresh-empty-reteach" if self.args.fresh_empty_reteach else self.reuse_problem()
        if why:
            LOG(f"step {step.key}: empty store; the original model is served again ({why})")
            st["reused_baseline"] = None
            st["reuse_skipped"] = why
            save()
            return self.step_original(step, st, save, files)
        ans = files[akey][0]
        if os.path.exists(ans):
            self.stash(step, "an unfinished empty-store step")
        src = self.files(Step("baseline", "baseline", "baseline"), "")[0]
        want, seen, n = self.pairs(), set(), 0
        tmp = ans + ".tmp"
        with open(src, encoding="utf-8") as fi, open(tmp, "w", encoding="utf-8") as fo:
            for line in fi:
                if not line.strip():
                    continue
                r = json.loads(line)
                k = (r["qid"], bool(r["thinking"]))
                if k not in want or k in seen:
                    continue
                seen.add(k)
                r.update({"key": step.key, "phase": step.phase, "mode": "reteach (empty store: baseline answers reused)",
                          "reused_from": "baseline", "reused_at": utc()})
                fo.write(json.dumps(r, ensure_ascii=False) + "\n")
                n += 1
        os.replace(tmp, ans)
        st["folder"] = None
        st["reused_baseline"] = {"from": os.path.relpath(src, self.work), "sha256": sha256(src), "answers": n}
        st.pop("reuse_skipped", None)
        st["answers"][akey] = {"path": ans, "n": n, "expected": len(want), "complete": n == len(want), "seconds": 0,
                               "reused": True}
        save()
        LOG(f"step {step.key}: empty store; reused the {n} baseline answers (no server started; "
            "--fresh-empty-reteach serves the original model again)")
        self.judge_step(step, st, akey, files[akey], save)

    def step_teach(self, step, st, save, files):
        folder = st.get("folder")
        reuse = bool(st.get("taught") and folder and os.path.isdir(folder))
        if not reuse:
            if folder and os.path.exists(folder):
                self.set_aside(folder, "teaching did not finish there")
            self.stash(step, "an unfinished teach")
            for k in ("taught", "lessons", "fact_states", "fact_neurons", "fact_lesson", "refused_lessons"):
                st.pop(k, None)
            st["answers"], st["judge"] = {}, {}
            if os.path.exists(self.copy_path(step)):
                self.set_aside(self.copy_path(step), "a copy from an unfinished run")
            folder, secs = self.make_copy(step, self.src, "taught with --slide")
            st["folder"], st["timings"]["copy_s"] = folder, secs
            save()
        else:
            LOG(f"resume: {folder} was taught already ({len(st.get('lessons', []))} lessons); not taught again")
        graph_file = os.path.join(self.sdir(step), "graph.json")
        want = self.pairs()
        pending = [f for f in step.flavors if not want <= answered_pairs(files[f[0]][0])]
        slide_flavors = [f for f in pending if f[2] == "slide"]
        plain_flavors = [f for f in pending if f[2] == "plain"]
        if not reuse or slide_flavors:
            srv = Server(self, folder, True, os.path.join(self.sdir(step), "server.log"), graph_file, label=step.key + " --slide")
            srv.start()
            try:
                st["timings"]["slide_server"] = srv.timings
                if not reuse:
                    if os.path.exists(graph_file):
                        raise RunError(f"{graph_file} exists before teaching")
                    teach(self, srv, step.store, self.sdir(step), st, save)
                for f in slide_flavors:
                    if reuse:
                        LOG(f"resume: probing {f[0]} in a restarted --slide server (answers before the restart came "
                            "from the server that taught)")
                        st.setdefault("notes", []).append(
                            f"{utc()}: {f[0]} resumed in a restarted --slide server after "
                            f"{len(answered_pairs(files[f[0]][0]))} answers from the server that taught")
                    self.serve_and_probe(step, st, save, files, folder, f, slide_srv=srv)
                g = srv.graph()
                sv = (g.get("model") or {}).get("save") or {}
                if sv.get("state") == "saving":
                    srv.wait_save(int(sv.get("at") or 0))
            finally:
                srv.stop("teach step done" if sys.exc_info()[0] is None else "teach step failed")
        for f in plain_flavors:
            self.serve_and_probe(step, st, save, files, folder, f)
        for f in step.flavors:
            self.judge_step(step, st, f[0], files[f[0]], save)

    def step_zero(self, step, st, save, files):
        a = self.args
        tstep = Step("teach", "teach", "teach")
        tst = self.load_state(tstep)
        if not (tst and tst.get("taught")):
            raise RunError("zero mode needs the teach step's records (fact ids and neurons per lesson): run teach first")
        tfolder = tst.get("folder")
        if not (tfolder and os.path.isdir(tfolder)):
            raise RunError(f"the taught folder {tfolder} is gone; zero mode copies it")
        lim = st["zero_limits"] = zero_limits(tst, step.deleted)
        save()
        for x in lim["cannot_restore"]:
            LOG(f"NOTE zero [{step.key}]: {self.lesson_name(x['lesson'])}'s fact {x['fact']} was replaced by fact "
                f"{x['replaced_by_fact']} of deleted {self.lesson_name(x['replaced_by_lesson'])}; 1.0.8 zeroed it then, "
                "so zeroing cannot bring it back where never teaching would (a stated limit of zero mode)")
        if lim["already_zero"]:
            LOG(f"NOTE zero [{step.key}]: deleted facts {lim['already_zero']} were replaced or missed: nothing of them "
                "is left to zero")
        folder = st.get("folder")
        if not (st.get("zero") and folder and os.path.isdir(folder)):
            if folder and os.path.exists(folder):
                self.set_aside(folder, "zeroing did not finish there")
            if os.path.exists(self.copy_path(step)):
                self.set_aside(self.copy_path(step), "a copy from an unfinished run")
            self.stash(step, "an unfinished zero step")
            st["answers"], st["judge"] = {}, {}
            for k in ("not_applicable", "zero_skipped"):  # from an earlier attempt: this one decides again
                st.pop(k, None)
            states = tst.get("fact_states", {})
            neurons = tst.get("fact_neurons", {})
            owner = tst.get("fact_lesson", {})
            deleted = set(step.deleted)
            targets, kept = [], []
            for fid, lid in owner.items():
                for n in neurons.get(fid, []):
                    item = (int(fid), states.get(fid), int(n["layer"]), int(n["neuron"]))
                    (targets if lid in deleted else kept).append(item)
            facts_del = sorted(int(f) for f, l in owner.items() if l in deleted)
            LOG(f"zero [{step.key}]: lessons {[self.lesson_name(i) for i in sorted(deleted)]} -> facts {facts_del} "
                f"(states {[states.get(str(f)) for f in facts_del]}), {len(targets)} neurons; {len(kept)} kept neurons")

            def skipped(e):
                st["zero_skipped"] = str(e)
                save()
                return RunError(f"zero mode skipped: the folder does not match zero_fact.py's layout: {e}")

            try:
                na = self.zero_not_applicable(tst, tfolder)
            except ZeroSkip as e:
                raise skipped(e)
            if na:
                st["folder"], st["zero"], st["not_applicable"] = None, None, na
                save()
                LOG(f"zero [{step.key}]: not applicable, so no copy is made and nothing is served: {na}")
                return
            folder, secs = self.make_copy(step, tfolder, "copy of the taught folder, to zero")
            st["folder"], st["timings"]["copy_s"] = folder, secs
            save()
            t0 = time.time()
            try:
                rep = zero_facts(folder, targets, kept, ref=tfolder, audit=a.zero_audit)
            except ZeroSkip as e:
                err = skipped(e)
                self.drop_unwritten_copy(folder, "zero mode was skipped, so nothing was written to this copy, and a "
                                                 "resumed zero step copies the taught folder again")
                raise err
            rep["facts"] = facts_del
            rep["seconds"] = round(time.time() - t0, 1)
            write_json(os.path.join(self.sdir(step), "zero.json"), rep)
            st["zero"] = {k: v for k, v in rep.items() if k != "rows"}
            st["timings"]["zero_s"] = rep["seconds"]
            save()
            LOG(f"zeroed {rep['rows_zeroed']} rows ({rep['bytes_zeroed']} bytes; {rep['rows_already_zero']} already "
                f"zero) in {rep['shards_touched']}; kept neurons nonzero {rep['kept_nonzero_before']} -> "
                f"{rep['kept_nonzero_after']} of {rep['kept_learned']}"
                + (f"; audit: {rep['audit']['bytes_differing']} bytes differ, {rep['audit']['outside_rows']} outside the rows"
                   if rep.get("audit") else ""))
        else:
            LOG(f"resume: {folder} was zeroed already")
        flavor = step.flavors[0]
        self.serve_and_probe(step, st, save, files, folder, flavor)
        self.judge_step(step, st, flavor[0], files[flavor[0]], save)

    # ---------------------------------------------------------------------------------- the whole run
    def run(self):
        a = self.args
        os.makedirs(self.work, exist_ok=True)
        LOG.path = os.path.join(self.work, "run.log")
        if not os.path.exists(self.set_for_judge):
            write_json(self.set_for_judge, self.S)
        self.preflight()
        self.summary = read_json(os.path.join(self.work, "summary.json")) or {}
        self.summary.update({
            "runner": os.path.abspath(__file__), "set": {"id": self.S["id"], "path": self.set_path,
                                                          "sha256": sha256(self.set_path), "level": self.S.get("level"),
                                                          "seed": self.S.get("seed")},
            "host": platform.node(), "os": platform.platform(), "python": sys.version.split()[0],
            "args": {k: v for k, v in vars(a).items()}, "updated": utc()})
        self.summary.setdefault("started", utc())
        self.summary.setdefault("steps", {})
        if not a.judge_only:
            LOG(f"lwrun {self.S['id']}: {len(self.S['lessons'])} lessons, {len(self.S['questions'])} questions x "
                f"thinking {a.thinking} = {len(self.pairs())} answers per probe; steps "
                f"{[s.key for s in self.steps]}")
            digest = self.bin_sha = sha256(self.bin)
            try:
                ver = subprocess.run([self.bin, "--version"], capture_output=True, text=True, timeout=60)
                version = (ver.stdout or ver.stderr).strip()[:200]
            except (OSError, subprocess.SubprocessError) as e:
                version = f"(--version failed: {e})"
            self.summary["binary"] = {"path": self.bin, "sha256": digest, "version": version}
            LOG(f"binary {self.bin} sha256 {digest[:16]}, --version: {self.summary['binary']['version']!r}")
            cfg = read_json(os.path.join(self.src, "config.json"), {})
            self.summary["model_src"] = {"path": self.src, "bytes": tree_bytes(self.src),
                                         "pocket": cfg.get("tensorfold_learned_neurons")}
            LOG(f"source {self.src} ({gb(self.summary['model_src']['bytes'])})")
        self.write_summary()
        failed = []
        zero_keys = [s.key for s in self.steps if s.kind == "zero"]
        holds, served = self.notes_holds(), set()  # a copy a notes_pocket step will serve is kept until it is done
        held = lambda key: [k for k in holds.get(key, []) if k not in served]

        def teach_cleanup():
            tst = self.load_state(Step("teach", "teach", "teach")) or {}
            zdone = all((self.summary["steps"].get(k) or {}).get("status") in ("done", "judged", NOT_APPLICABLE)
                        for k in zero_keys)
            if tst.get("status") == "done" and zdone and not held("teach"):
                self.cleanup_copy(tst.get("folder"))
            elif IS_LINUX and self.args.cleanup:
                LOG("--cleanup: the taught folder is kept: " + (
                    "a zero step is not done, and resuming it needs the folder" if not zdone else
                    f"{', '.join(held('teach'))} will serve it" if held("teach") else "the teach step is not done"))

        for step in self.steps:
            try:
                st = self.run_step(step)
            except RunError as e:
                LOG(f"step {step.key} FAILED: {e}")
                st = self.load_state(step) or {}
                st["status"], st["error"] = "failed", str(e)
                if os.path.isdir(self.sdir(step)):
                    write_json(os.path.join(self.sdir(step), "state.json"), st)
                failed.append(step.key)
            self.summary["steps"][step.key] = self.step_summary(step, st)
            self.write_summary()
            if st.get("status") == "done" and step.kind not in ("teach",) and step.needs_copy():
                if held(step.key):
                    if IS_LINUX and self.args.cleanup:
                        LOG(f"--cleanup: {step.key}'s copy is kept: {', '.join(held(step.key))} will serve it")
                else:
                    self.cleanup_copy(st.get("folder"))
            if (step.kind == "zero" and step.key == zero_keys[-1]) or (step.kind == "teach" and not zero_keys):
                teach_cleanup()
            if step.kind == "notes_pocket" and st.get("status") == "done":
                served.add(step.key)
                if not held(step.pocket_from):
                    if step.pocket_from == "teach":   # the zero steps came first: they are planned before it
                        teach_cleanup()
                    else:
                        sst = self.load_state(self.pocket_step(step)) or {}
                        if sst.get("status") == "done":
                            self.cleanup_copy(sst.get("folder"))
        self.finish()
        return 1 if failed else 0

    def notes_holds(self):
        """{step key: the notes_pocket steps of this run that serve its taught folder}: --cleanup keeps such a copy
        until they are done."""
        holds = {}
        for s in self.steps:
            if s.kind == "notes_pocket":
                holds.setdefault(s.pocket_from, []).append(s.key)
        return holds

    def step_summary(self, step, st):
        out = {"phase": step.phase, "kind": step.kind, "status": st.get("status"), "error": st.get("error"),
               "folder": st.get("folder"), "timings": st.get("timings"), "answers": st.get("answers"),
               "judge": {k: {x: v.get(x) for x in ("rc", "report", "verdicts", "headline", "scores", "consistency", "n")}
                         for k, v in (st.get("judge") or {}).items()}}
        if st.get("lessons") is not None:
            out["teach_seconds"] = st.get("teach_seconds")
            out["lessons"] = [{"name": r["name"], "lesson": r["lesson"], "seconds": r["seconds"],
                               "recalled": r["recalled"], "facts": r["fact_ids_sse"], "graph_ids": r["fact_ids_graph"],
                               "message": r["message"], "changed_states": r["changed_states"],
                               "save": (r.get("save") or {}).get("state"),
                               "save_line": (r.get("save_log") or {}).get("line"),
                               "held_out_back": (r.get("save_log") or {}).get("held_out_back")} for r in st["lessons"]]
            out["refused_lessons"] = st.get("refused_lessons")
            out["fact_states"] = st.get("fact_states")
            out["refused_judge"] = st.get("refused_judge")
        if st.get("zero") is not None or st.get("zero_skipped"):
            out["zero"] = st.get("zero")
            out["zero_skipped"] = st.get("zero_skipped")
        if st.get("not_applicable"):
            out["not_applicable"] = st["not_applicable"]
        if st.get("zero_limits") is not None:
            out["zero_limits"] = st["zero_limits"]
        if step.kind == "reteach-empty":
            out["reused_baseline"] = st.get("reused_baseline")
            if st.get("reuse_skipped"):
                out["reuse_skipped"] = st["reuse_skipped"]
        if "source_unchanged" in st:
            out["source_unchanged"] = st["source_unchanged"]
        for k in ("in_context", "memory", "notes"):  # the reference phases: the prompt asked with, the slots answered
            if st.get(k):                            # from, the notes given
                out[k] = st[k]
        for k in ("pocket_from", "folder_unchanged"):
            if k in st:
                out[k] = st[k]
        return out

    def write_summary(self):
        if self.summary is None:
            return
        self.summary["updated"] = utc()
        write_json(os.path.join(self.work, "summary.json"), self.summary)

    def finish(self):
        """Combined reports per delete mode, zero vs reteach, the copies left on disk."""
        comb = os.path.join(self.work, "combined")
        os.makedirs(comb, exist_ok=True)
        allsteps = {s.key: s for s in self.plan_all()}
        dels = [p["name"] for p in self.S["phases"] if p["name"] not in ("baseline", "teach")]
        families = self.families(allsteps)
        self.summary["combined"] = {}
        for fam, f in families.items():
            fl = [x for x in [f["baseline"][0], f["teach"][0]] + [p for _d, p, _how in f["deletes"]] if x]
            if len(fl) < 2:
                continue
            j = run_judge(self, fl, os.path.join(comb, f"verdicts-{fam}.json"), os.path.join(comb, f"report-{fam}.txt"),
                          f"combined {fam}")
            j["files"] = fl
            self.summary["combined"][fam] = {k: j.get(k) for k in ("rc", "files", "report", "verdicts", "headline", "scores",
                                                                     "kept_stable")}
        self.summary["kept_stable_inputs"] = self.write_kept_stable(families)
        self.summary["zero_vs_reteach"] = self.compare_modes(dels, allsteps)
        ref = self.reference_report(allsteps)
        if ref:
            self.summary["reference"] = ref
        else:
            self.summary.pop("reference", None)
        reg = read_json(self.copies_path, {"copies": []})
        left = []
        for c in reg["copies"]:
            if c.get("removed") or not os.path.exists(c["path"]):
                continue
            c["bytes_now"] = tree_bytes(c["path"])
            left.append(c)
        self.summary["copies"] = reg["copies"]
        self.write_summary()
        if left:
            LOG("model copies this runner made, left on disk (logical sizes; an APFS clone shares blocks until "
                "written, but a taught copy's first save rewrites every shard whole):")
            for c in left:
                LOG(f"  {gb(c['bytes_now']):>9}  {c['path']}  [{c['step']}]")
            if IS_MAC:
                LOG("  macOS: nothing was deleted; move unwanted copies to the Trash yourself.")
        LOG(f"summary: {os.path.join(self.work, 'summary.json')}")

    def families(self, allsteps):
        """Per delete mode (a family): the baseline answers, the teach answers the mode is compared with, and the
        mode's own answers for each delete phase, each as (path or None, how it was served: slide, plain or
        original). The teach answers are the ones served like the mode where both exist: plain (a restarted server
        without --slide) for zero, the --slide server for reteach. A family is kept only when one of its delete
        phases has answers: a reteach-only run used to write combined/report-zero.txt holding just baseline and
        teach, named after a mode that never ran."""
        def pick(key, *sufs):
            s = allsteps.get(key)
            for suf in sufs if s else ():
                p = self.files(s, suf)[0]
                if os.path.exists(p):
                    how = ("original" if s.kind in ("baseline", "reteach-empty") else
                           "plain" if s.kind == "zero" or suf == ".restart" else "slide")
                    return p, how
            return None, None

        def empty(d):
            s = allsteps.get(d + ".reteach")
            return s is not None and s.kind == "reteach-empty"

        dels = [p["name"] for p in self.S["phases"] if p["name"] not in ("baseline", "teach")]
        base, ts, tp = pick("baseline", ""), pick("teach", ""), pick("teach", ".restart")
        fams = {"reteach": {"baseline": base, "teach": ts if ts[0] else tp,
                            "deletes": [(d,) + pick(d + ".reteach", "", ".restart") for d in dels]}}
        if ts[0] and tp[0]:
            fams["reteach.restart"] = {"baseline": base, "teach": tp, "deletes": [
                (d,) + pick(d + ".reteach", "" if empty(d) else ".restart") for d in dels]}
        fams["zero"] = {"baseline": base, "teach": tp if tp[0] else ts,
                        "deletes": [(d,) + pick(d + ".zero", "") for d in dels]}
        if not any(p for f in fams.values() for _d, p, _how in f["deletes"]):
            return {"no-deletes": {"baseline": base, "teach": ts if ts[0] else tp, "deletes": []}}
        return {k: f for k, f in fams.items() if any(p for _d, p, _how in f["deletes"])}

    def write_kept_stable(self, families):
        """The kept_stable inputs: WORK/kept-stable/MODE/teach.jsonl and WORK/kept-stable/MODE/PHASE.jsonl, per delete
        mode the teach answers and each delete phase's answers, copied from phases/ with their phase names as they
        are. `lwlogic.py judge WORK/set.json WORK/kept-stable/MODE/*.jsonl` then has one model's teach and delete
        answers side by side, for a score that compares a delete phase's kept questions with the teach answers
        (verdict and text) instead of with the ideal answer. manifest.json lists every file, where it came from and
        how it was served. Rebuilt at every finish; a file no longer listed is renamed *.stale-STAMP, never
        deleted."""
        root = os.path.join(self.work, "kept-stable")
        man = {"format": "lw-kept-stable/1", "set": self.S["id"], "made": utc(), "modes": {}}
        listed = set()
        deleted = {p["name"]: p["deleted"] for p in self.S["phases"]}
        want = self.pairs()

        def put(src, dst, how):
            tmp = dst + ".tmp"
            shutil.copyfile(src, tmp)
            os.replace(tmp, dst)
            listed.add(dst)
            have = answered_pairs(dst)
            return {"file": os.path.relpath(dst, self.work), "from": os.path.relpath(src, self.work), "served": how,
                    "answers": len(have), "complete": want <= have}

        for fam, f in families.items():
            dl = [(ph, p, how) for ph, p, how in f["deletes"] if p]
            if not dl or not f["teach"][0]:
                continue
            d = os.path.join(root, fam)
            os.makedirs(d, exist_ok=True)
            m = man["modes"][fam] = {"teach": put(f["teach"][0], os.path.join(d, "teach.jsonl"), f["teach"][1]),
                                     "deletes": {}}
            for ph, p, how in dl:
                m["deletes"][ph] = put(p, os.path.join(d, ph.replace(os.sep, "_") + ".jsonl"), how)
                m["deletes"][ph]["deleted"] = deleted.get(ph)
        if not os.path.isdir(root):
            return None
        for dirpath, _dirs, names in os.walk(root):
            for x in names:
                p = os.path.join(dirpath, x)
                if x.endswith(".jsonl") and p not in listed:
                    LOG(f"kept-stable: {os.path.relpath(p, self.work)} is no longer an input; renamed *.stale-STAMP")
                    os.rename(p, f"{p}.stale-{stamp()}")
        write_json(os.path.join(root, "manifest.json"), man)
        if man["modes"]:
            LOG(f"kept-stable inputs: {os.path.relpath(root, self.work)}/"
                + "; ".join(f"{fam}: teach + {', '.join(m['deletes'])}" for fam, m in man["modes"].items()))
        return man

    def plan_all(self):
        """Every step the set allows, whatever --phases picked: finish() reads what earlier runs left too."""
        a = self.args
        saved = (a.phases, a.delete_mode)
        a.phases, a.delete_mode = None, "both"
        try:
            return self.plan(found=True)
        finally:
            a.phases, a.delete_mode = saved

    def compare_modes(self, dels, allsteps):
        out = {}
        for d in dels:
            z, r = allsteps.get(d + ".zero"), allsteps.get(d + ".reteach")
            if not z or not r:
                continue
            zf = self.files(z, "")
            rsuf = ".restart" if os.path.exists(self.files(r, ".restart")[1]) else ""
            rf = self.files(r, rsuf)
            if not (os.path.exists(zf[1]) and os.path.exists(rf[1])):
                continue

            def load(fs):
                ver = {(x["qid"], x["thinking"]): x for x in read_json(fs[1], {}).get("results", [])}
                txt = {}
                with open(fs[0], encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            x = json.loads(line)
                            txt[(x["qid"], bool(x["thinking"]))] = x["content"]
                return ver, txt

            (zv, zt), (rv, rt) = load(zf), load(rf)
            res = {"zero": zf[0], "reteach": rf[0]}
            zl = self.zero_limits_of(z) or (self.load_state(z) or {}).get("zero_limits")
            if zl and (zl.get("cannot_restore") or zl.get("already_zero")):
                res["zero_limits"] = zl  # where the two may differ by design (section 8), not by delete damage
            for th in self.thinking:
                keys = sorted(k for k in set(zv) & set(rv) if k[1] == th)
                diff = [k[0] for k in keys if zv[k]["verdict"] != rv[k]["verdict"]]
                res["on" if th else "off"] = {
                    "n": len(keys), "same_verdict": len(keys) - len(diff),
                    "same_text": sum(zt.get(k) == rt.get(k) for k in keys),
                    "both_pass": sum(zv[k]["verdict"] == "pass" == rv[k]["verdict"] for k in keys),
                    "zero_only_pass": sum(zv[k]["verdict"] == "pass" != rv[k]["verdict"] for k in keys),
                    "reteach_only_pass": sum(rv[k]["verdict"] == "pass" != zv[k]["verdict"] for k in keys),
                    "differing_verdicts": diff}
            out[d] = res
            LOG(f"zero vs reteach on {d}: " + "; ".join(
                f"thinking {t}: same verdict {res[t]['same_verdict']}/{res[t]['n']}, same text {res[t]['same_text']}"
                for t in ("off", "on") if t in res))
        return out

    def reference_report(self, allsteps):
        """The pocket beside the reference phases (TEST-LOGIC.md section 8), for every phase with a reference step on
        file. `scores`: each answer source's pass counts per group and thinking mode. `gap`: per pocket answer file of
        the phase (teach, teach.restart; a delete phase's zero, reteach and reteach.restart), per group and thinking
        mode, the pass counts of the pocket and of each reference over the answers they all have, and the pocket gap:
        in_context's pass rate minus the pocket's. `lessons` and `outcomes`: each CHANGE and RETRACT lesson of the
        phase, and per answer source and thinking mode corrected only, stale only, both or neither. Written to
        summary.json (`reference`) and WORK/reference/report.txt; None when no reference step has judged answers."""
        refs = [s for s in allsteps.values() if s.kind in ALL_REF_KINDS]
        if not refs:
            return None

        def load(step, suf):
            ans, ver = self.files(step, suf)[:2]
            v = read_json(ver)
            if not (isinstance(v, dict) and os.path.exists(ans)):
                return None
            res = {(r["qid"], bool(r["thinking"])): r for r in v.get("results", []) if r.get("phase") == step.phase}
            txt = {}
            with open(ans, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        x = json.loads(line)
                        if x.get("phase") == step.phase:
                            txt[(x["qid"], bool(x["thinking"]))] = x.get("content") or ""
            served = ("computed, no model" if step.kind == "replacement_memory" else
                      "the untaught model, lessons in the system prompt" if step.kind == "in_context" else
                      "the untaught model, each question's notes in the system prompt" if step.kind == "notes" else
                      f"the pocket of {step.pocket_from} without --slide, each question's notes in the system prompt"
                      if step.kind == "notes_pocket" else
                      "zeroed copy, without --slide" if step.kind == "zero" else
                      "restarted without --slide" if suf == ".restart" else "the --slide server that taught")
            return {"answers": os.path.relpath(ans, self.work), "verdicts": os.path.relpath(ver, self.work),
                    "served": served, "res": res, "txt": txt}

        def tally(d, keys):
            return [sum(1 for k in keys if d["res"][k].get("verdict") == "pass"), len(keys)]

        out = {"format": REF_FORMAT, "asked": {"kinds": self.ref_kinds, "of": self.ref_of},
               "steps": [s.key for s in refs],
               "gap_is": "in_context pass rate minus pocket pass rate, over the answers the pocket and every reference "
                         "of the phase have; positive: the pocket is behind the model told in context",
               "phases": {}}
        try:
            lw, lw_err = lw_module(self.lwlogic), None
        except RunError as e:
            lw, lw_err = None, str(e)
        rep = [f"Reference phases of {self.S['id']} ({REF_FORMAT}), {utc()}",
               "gap = in_context pass rate minus pocket pass rate, over the answers all sources have. "
               "in_context is a reference, not a ceiling."]
        for ph in [p["name"] for p in self.S["phases"] if any(s.phase == p["name"] for s in refs)]:
            src = {}
            for key, sufs in ([("teach", ("", ".restart"))] if ph == "teach" else
                              [(ph + ".zero", ("",)), (ph + ".reteach", ("", ".restart"))]):
                s = allsteps.get(key)
                for suf in (sufs if s and s.kind != "reteach-empty" else ()):
                    d = load(s, suf)
                    if d:
                        src[key + suf] = d
            pockets = list(src)
            for k in ALL_REF_KINDS:
                s = allsteps.get(f"{ph}.{k}")
                d = load(s, "") if s else None
                if d:
                    src[k] = d
            here = [k for k in ALL_REF_KINDS if k in src]
            cols = list(REF_KINDS) + [k for k in NOTE_KINDS if k in here]   # the notes columns only when asked
            if not here:
                continue
            groups = question_groups(self.S, ph)

            def groups_of(key):
                bk = next((src[k]["res"][key].get("buckets") or [] for k in src if key in src[k]["res"]), [])
                return answer_groups(groups, key[0], bk)

            P = out["phases"][ph] = {
                "store": next(p["store"] for p in self.S["phases"] if p["name"] == ph),
                "sources": {k: {x: d[x] for x in ("answers", "verdicts", "served")} for k, d in src.items()},
                "scores": {}, "gap": {}}
            for k, d in src.items():
                for g in REF_GROUPS:
                    for th in (False, True):
                        keys = [x for x in d["res"] if x[1] == th and g in groups_of(x)]
                        if keys:
                            P["scores"].setdefault(k, {}).setdefault(g, {})["on" if th else "off"] = tally(d, keys)
            rep.append(f"\n== {ph}: store {' '.join(self.lesson_name(i) for i in P['store'])}")
            for k, d in src.items():
                rep.append(f"   {k:24} {d['served']}: {d['answers']}")
            for k in NOTE_KINDS:  # what the notes phases gave: notes, tokens and lookup time per question
                nst = (self.load_state(allsteps[f"{ph}.{k}"]) or {}).get("notes") if f"{ph}.{k}" in allsteps else None
                if k in here and nst:
                    P.setdefault("notes", {})[k] = nst
                    rep.append(f"   {k:24} {nst['notes']} notes ({nst['current']} current) from {len(P['store'])} "
                               f"lessons; tokens given p50 {nst['tokens_p50']}, p95 {nst['tokens_p95']}, max "
                               f"{nst['tokens_max']} (cap {nst['config']['cap']}, framing included); lookup p50 "
                               f"{nst['lookup_us_p50']} us, p95 {nst['lookup_us_p95']} us; nothing retrieved for "
                               f"{nst['nothing_retrieved']} of {nst['questions']} questions; per question: {nst['log']}")
            for pk in pockets:
                both = set(src[pk]["res"]).intersection(*(src[k]["res"] for k in here))
                table = P["gap"][pk] = {}
                for g in REF_GROUPS:
                    for th in (False, True):
                        keys = sorted(x for x in both if x[1] == th and g in groups_of(x))
                        if not keys:
                            continue
                        cell = {"pocket": tally(src[pk], keys)}
                        for k in here:
                            cell[k] = tally(src[k], keys)
                        ic = rate(cell.get("in_context"))
                        cell["gap"] = None if ic is None else round(ic - rate(cell["pocket"]), 3)
                        table.setdefault(g, {})["on" if th else "off"] = cell
                rep.append(f"\n   pocket {pk} beside " + " and ".join(here))
                extra = [k for k in cols if k in NOTE_KINDS]
                rep.append(f"   {'group':16} {'thinking':8} {'pocket':>8} {'in_context':>11} {'memory':>8} {'gap':>7}"
                           + "".join(f" {k:>13}" for k in extra))
                for g, row in table.items():
                    for th, c in row.items():
                        cells = [f"{c[x][0]}/{c[x][1]}" if x in c else "-" for x in ["pocket"] + cols]
                        rep.append(f"   {g:16} {th:8} {cells[0]:>8} {cells[1]:>11} {cells[2]:>8} {fmt_gap(c['gap']):>7}"
                                   + "".join(f" {x:>13}" for x in cells[3:]))
                LOG(f"reference {ph}, pocket {pk}: gap " + "; ".join(
                    f"{g} " + " ".join(f"{th} {fmt_gap(c['gap'])}" for th, c in table[g].items())
                    for g in ("CURRENT", "WHO_HAS", "negation", "dates", "two_step", "all") if g in table))
            if not pockets:
                rep.append("   no pocket answers of this phase on file: no gap; the references' own scores are in "
                           "summary.json (reference.phases.PHASE.scores)")
            if lw is None:
                P["outcomes_error"] = lw_err
                rep.append(f"   CHANGE and RETRACT outcomes not read: {lw_err}")
                continue
            P["lessons"] = change_lessons(lw, self.S, ph)
            P["outcomes"], P["outcome_counts"] = {}, {}
            for k, d in src.items():
                oc = P["outcomes"][k] = lesson_outcomes(lw, self.S, ph, P["lessons"], d["txt"])
                cnt = P["outcome_counts"][k] = {}
                for per in oc.values():
                    for th, x in per.items():
                        cnt.setdefault(th, {o: 0 for o in OUTCOMES})[x["outcome"]] += 1
            if P["lessons"]:
                rep.append("\n   CHANGE and RETRACT lessons: corrected only, stale only, both or neither")
            for rec in P["lessons"]:
                rep.append(f"   {rec['name']} {rec['op']} \"{rec['text']}\": now {rec['new']}; holds after it: "
                           f"{', '.join(rec['now']) or 'nothing'}; stale: "
                           f"{', '.join(rec['stale']) or 'nothing'}" + ("" if rec["classified"] else f"; not classified: {rec['why']}"))
                if not rec["classified"]:
                    continue
                for k in src:
                    per = P["outcomes"][k].get(rec["name"]) or {}
                    rep.append(f"      {k:24} " + "   ".join(
                        f"{th} {x['outcome'].replace('_', ' ')}"
                        + (f" (corrected {' '.join(x['corrected'])})" if x["corrected"] else "")
                        + (f" (stale {' '.join(x['stale'])})" if x["stale"] else "") for th, x in per.items()))
        if not out["phases"]:
            return None
        d = os.path.join(self.work, "reference")
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, "report.txt")
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            f.write("\n".join(rep) + "\n")
        os.replace(path + ".tmp", path)
        out["report"] = os.path.relpath(path, self.work)
        LOG(f"reference report: {path}")
        return out


# ============================================================================================ dry run
def dry_run(run):
    a = run.args
    LOG.prefix = "[dry-run] "
    S = run.S
    LOG(f"set {run.set_path}: {S['id']} level {S.get('level')} seed {S.get('seed')}, {len(S['lessons'])} lessons, "
        f"{len(S['questions'])} questions, phases {[p['name'] for p in S['phases']]}")
    ok = True

    def check(cond, good, bad):
        nonlocal ok
        LOG(("ok    " if cond else "PROBLEM ") + (good if cond else bad))
        ok = ok and bool(cond)

    exe = os.path.isfile(run.bin) and os.access(run.bin, os.X_OK)
    check(exe, f"binary {run.bin} (sha256 {sha256(run.bin)[:16] if exe else ''})",
          f"binary {run.bin} is not an executable file (yet)")
    cfgp = os.path.join(run.src, "config.json")
    cfg = read_json(cfgp, None) if os.path.isfile(cfgp) else None
    src_bytes = tree_bytes(run.src) if os.path.isdir(run.src) else 0
    check(cfg is not None, f"source {run.src} ({gb(src_bytes)} logical)", f"source {run.src} has no config.json")
    if cfg is not None:
        check("tensorfold_learned_neurons" not in cfg or a.allow_taught_src, "source holds no learned neurons",
              "source already holds learned neurons (tensorfold_learned_neurons): not the original model")
    check(not port_busy(run.port), f"port {run.port} is free", f"port {run.port} is in use")
    try:
        h = subprocess.run([sys.executable, run.lwlogic, "judge", "-h"], capture_output=True, text=True, timeout=60)
        check(h.returncode == 0 and "--json" in h.stdout, f"judge CLI: {run.lwlogic} judge SET ANSWERS --json OUT",
              f"lwlogic.py judge -h failed or has no --json: {h.stderr.strip()[-300:]}")
    except (OSError, subprocess.SubprocessError) as e:
        check(False, "", f"lwlogic.py judge -h could not run: {e}")
    if any(s.kind in ALL_REF_KINDS for s in run.steps):
        try:
            lw_module(run.lwlogic)
            check(True, f"reference phases {run.ref_kinds} beside {run.ref_of}: lwlogic.py loads as a module "
                        "(templates and readers)", "")
        except RunError as e:
            check(False, "", str(e))
    problems = []
    if not a.resume:
        for s in run.steps:
            if os.path.exists(run.sdir(s)):
                problems.append(f"{run.sdir(s)} exists")
            if s.needs_copy() and os.path.exists(run.copy_path(s)):
                problems.append(f"{run.copy_path(s)} exists")
    check(not problems, f"work dir {run.work}: nothing to reuse", "would refuse without --resume: " + "; ".join(problems))
    try:
        run.check_pidfile()
        check(True, "no server of an earlier run is still up (server.pid)", "")
    except RunError as e:
        check(False, "", str(e))
    if any(s.kind == "zero" for s in run.steps) and not any(s.kind == "teach" for s in run.steps):
        tst = read_json(os.path.join(run.work, "phases", "teach", "state.json"), {}) or {}
        check(tst.get("taught"), "zero mode will use the teach records already in the work dir",
              "zero mode needs the teach step's records: add teach to --phases (or run it first)")
    pairs = len(run.pairs())
    need = 0
    for s in run.steps:
        if s.needs_copy() and (IS_LINUX or s.kind in ("teach", "reteach")):
            need += src_bytes
    free = free_bytes(run.models)
    LOG(f"disk: {gb(free)} free at {run.models}; copies need about {gb(need)}"
        + (" (each taught copy's first save rewrites every shard: about the full checkpoint each)" if IS_MAC else "")
        + (" (peak lower with --cleanup)" if IS_LINUX else ""))
    lessons = {l["id"]: l for l in S["lessons"]}
    srv_cmd = lambda folder, slide, g=None: " ".join(shlex.quote(x) for x in (
        [run.bin, "serve", folder, "--host", "127.0.0.1", "--port", str(run.port), "--no-update-check"]
        + (["--slide", "--slide-graph", g] if slide else []) + list(a.server_arg or [])))
    shown_questions = [False]

    def probe_lines(akey, phase, mode, ans):
        LOG(f"  probe {akey}: phase {phase!r}, mode {mode!r}: {len(S['questions'])} questions x thinking "
            f"{a.thinking} = {pairs} chats, one fresh chat each, temperature 0, max_tokens off {a.max_tokens_off} / "
            f"on {a.max_tokens_on}, chat_template_kwargs.enable_thinking -> {ans}")
        if not shown_questions[0]:
            for q in S["questions"]:
                LOG(f"      {q['qid']} [{q['type']}/{q['role']}] {q['text']}")
            shown_questions[0] = True

    def judge_line(akey, f):
        LOG(f"  judge {akey}: {sys.executable} {run.lwlogic} judge {run.set_for_judge} {f[0]} --json {f[1]} > {f[2]}")

    for s in run.steps:
        LOG(f"step {s.key} ({s.kind}): phase {s.phase!r}" + (f", deleted {[run.lesson_name(i) for i in s.deleted]}" if s.deleted else "")
            + (f", store {[run.lesson_name(i) for i in s.store]}" if s.kind != "baseline" else ""))
        sd = run.sdir(s)
        if s.kind == "replacement_memory":
            f = run.files(s, "")
            slots = memory_slots(S, s.store)
            LOG(f"  no model, no server: one slot per (entity, attribute), the latest value a lesson gives for now "
                f"({MEMORY_FORMAT}); {len(S['questions'])} questions x thinking {a.thinking} answered from the slots "
                f"-> {f[0]}")
            for (e, at), v in sorted(slots.items()):
                LOG(f"      {e} {at} = {v}")
            judge_line(s.flavors[0][0], f)
            continue
        if s.kind in NOTE_KINDS:
            f = run.files(s, "")
            plan = run.notes_plan(s)
            d, st = plan["describe"], plan["stats"]
            LOG(f"  the note store of phase {s.phase} ({d['format']}): {st['notes']} notes, {st['current']} current, "
                f"from {len(s.store)} lessons; tokenizer {d['tokenizer'].get('path') or d['tokenizer']['name']}; "
                f"settings {json.dumps(d['config'], sort_keys=True)}")
            for n in sorted(plan["notes"], key=lambda n: n["id"]):
                LOG(f"      [{n['id']}] v{n['version']} {n['state']:10} {n['entity']} {n['attribute']} \"{n['text']}\"")
            shown = False
            for q in S["questions"]:
                r = plan["ret"][q["qid"]]
                LOG(f"      {q['qid']} notes {r['ids'] or 'none (no system message)'}, {r['tokens']} tokens, "
                    f"{'past' if r['past'] else 'now'}")
                if r["block"] and not shown:
                    for line in r["block"].split("\n"):
                        LOG(f"        | {line}")
                    shown = True
            if s.kind == "notes":
                if s.copy_baseline:
                    LOG(f"  copy: {'cp -cRL' if IS_MAC else 'cp -RL'} {run.src} {run.copy_path(s)}")
                    LOG(f"  serve without --slide: {srv_cmd(run.copy_path(s), False)}")
                else:
                    LOG(f"  serve the source read-only (fingerprint before/after): {srv_cmd(run.src, False)}")
            else:
                LOG(f"  serve the folder step {s.pocket_from} taught, without --slide (fingerprint before/after): "
                    f"{srv_cmd(run.copy_path(run.pocket_step(s)), False)}")
            probe_lines(s.flavors[0][0], s.phase, f"{s.kind}, each question's notes as the system message", f[0])
            LOG(f"  per question: the note ids, tokens and lookup time -> {run.sdir(s)}/notes.jsonl")
            LOG("  stop the server")
            judge_line(s.flavors[0][0], f)
            continue
        if s.kind == "in_context":
            f = run.files(s, "")
            LOG(f"  the untaught model with the phase's lessons in the system prompt ({IN_CONTEXT_FORMAT}, sha256 "
                f"{hashlib.sha256(s.system.encode()).hexdigest()[:16]}):")
            for line in s.system.split("\n"):
                LOG(f"      | {line}")
            if s.copy_baseline:
                LOG(f"  copy: {'cp -cRL' if IS_MAC else 'cp -RL'} {run.src} {run.copy_path(s)}")
                LOG(f"  serve without --slide: {srv_cmd(run.copy_path(s), False)}")
            else:
                LOG(f"  serve the source read-only (fingerprint before/after): {srv_cmd(run.src, False)}")
            probe_lines(s.flavors[0][0], s.phase, "in_context, the prompt above as the system message", f[0])
            LOG("  stop the server")
            judge_line(s.flavors[0][0], f)
            continue
        if s.kind == "reteach-empty" and not a.fresh_empty_reteach:
            f = run.files(s, "")
            bans = run.files(Step("baseline", "baseline", "baseline"), "")[0]
            now = run.reuse_problem()
            LOG(f"  empty store = the original model: copy the baseline answers {bans} to {f[0]} as phase {s.phase!r} "
                "(no server); if they cannot stand in by then (missing, incomplete, another binary or max_tokens), "
                "serve the source instead. Today: " + (now or "they can") + "; --fresh-empty-reteach always serves")
            judge_line(s.flavors[0][0], f)
            continue
        if s.kind in ("baseline", "reteach-empty") and not s.copy_baseline:
            LOG(f"  serve the source read-only (fingerprint before/after): {srv_cmd(run.src, False)}")
            LOG(f"  wait for GET /health (startup timeout {a.startup_timeout} s, warm {a.warm_timeout} s)")
            f = run.files(s, "")
            probe_lines(s.flavors[0][0], s.phase, s.flavors[0][3], f[0])
            LOG("  stop the server (Popen.terminate, kill after "
                f"{a.stop_timeout} s)")
            judge_line(s.flavors[0][0], f)
            continue
        dst = run.copy_path(s)
        if s.kind == "zero":
            tfolder = run.copy_path(Step("teach", "teach", "teach"))
            LOG(f"  copy the taught folder: {'cp -cRL' if IS_MAC else 'cp -RL'} {tfolder} {dst}")
            dl = [lessons[i] for i in s.deleted]
            LOG(f"  zero the facts the teach step recorded for {[l['name'] for l in dl]} (SSE fact ids and graph diff), "
                "each neuron (layer L, index k): backbone.layers.L.mixer.shared_experts.up_proj.{weight,scales,biases} "
                "row rows-64+k; rows already zero allowed only for facts 1.0.8 marked replaced; kept neurons must stay "
                f"nonzero; byte audit against the taught folder ({a.zero_audit})")
            lim = run.zero_limits_of(s)
            if lim:
                for x in lim["cannot_restore"]:
                    LOG(f"  limit: {run.lesson_name(x['lesson'])}'s fact {x['fact']} was replaced by deleted "
                        f"{run.lesson_name(x['replaced_by_lesson'])}: zeroing cannot bring it back (section 8)")
                if lim["already_zero"]:
                    LOG(f"  limit: deleted facts {lim['already_zero']} were replaced or missed: nothing left to zero")
            LOG("  but when the taught folder holds no pocket and the teach records say nothing was stored (every "
                "lesson missed, every save 0 bytes): not applicable, no copy made, nothing served")
            LOG(f"  serve without --slide: {srv_cmd(dst, False)}")
            f = run.files(s, "")
            probe_lines(s.key, s.phase, "zero", f[0])
            LOG("  stop the server")
            judge_line(s.key, f)
            continue
        if s.kind == "baseline":
            LOG(f"  copy: {'cp -cRL' if IS_MAC else 'cp -RL'} {run.src} {dst}")
            LOG(f"  serve without --slide: {srv_cmd(dst, False)}")
            f = run.files(s, "")
            probe_lines("baseline", "baseline", "original", f[0])
            judge_line("baseline", f)
            continue
        g = os.path.join(sd, "graph.json")
        LOG(f"  copy: {'cp -cRL' if IS_MAC else 'cp -RL'} {run.src} {dst}  (refuse if it exists; check no links)")
        LOG(f"  serve with --slide: {srv_cmd(dst, True, g)}")
        LOG(f"  GET /v1/slide/graph -> {sd}/graph-0-before.json (must hold no facts)")
        for i in s.store:
            l = lessons[i]
            LOG(f"  POST /v1/slide/learn {json.dumps({'text': l['text']}, ensure_ascii=False)}   [{l['name']} {l['op']}]")
            LOG(f"     read SSE fact/learning/learned to [DONE] -> {sd}/learn-{l['name']}.sse; wait for the save after it "
                f"(graph model.save, {a.save_timeout} s); GET graph -> graph-{l['name']}.json; new fact ids, neurons, "
                "earlier facts whose state changed (replaced)")
        for akey, suf, how, mode in s.flavors:
            f = run.files(s, suf)
            if how == "slide":
                probe_lines(akey, s.phase, mode, f[0])
        LOG("  stop the --slide server (after any save in progress)")
        for akey, suf, how, mode in s.flavors:
            f = run.files(s, suf)
            if how == "plain":
                LOG(f"  serve the taught folder again without --slide: {srv_cmd(dst, False)}")
                probe_lines(akey, s.phase, mode, f[0])
                LOG("  stop the server")
        for akey, suf, how, mode in s.flavors:
            judge_line(akey, run.files(s, suf))
        LOG("  if lessons were refused (recalled false): also judge these answers as the set's phase that keeps only "
            "the other lessons, when there is one")
        if IS_LINUX and a.cleanup and s.kind != "teach":
            LOG(f"  --cleanup: remove {dst}")
    LOG(f"finish: combined judge per delete mode -> {run.work}/combined/; kept-stable inputs -> "
        f"{run.work}/kept-stable/MODE/{{teach,PHASE}}.jsonl; zero vs reteach agreement; "
        + (f"reference report (pocket gap per group, CHANGE and RETRACT outcomes) -> {run.work}/reference/report.txt; "
           if any(s.kind in ALL_REF_KINDS for s in run.steps) else "")
        + f"{run.work}/summary.json; print the copies left on disk")
    nl = sum(len(s.store) for s in run.steps if s.kind in ("teach", "reteach"))
    nprobe = sum(len(s.flavors) for s in run.steps
                 if (s.kind != "reteach-empty" or a.fresh_empty_reteach) and s.kind != "replacement_memory") * pairs
    LOG(f"totals: {nl} learn requests (the guide says 1.5 to 3 min per fact: roughly {nl * 1.5:.0f} to {nl * 3:.0f} "
        f"min if each lesson is one fact), {nprobe} chats, {sum(1 for s in run.steps if s.needs_copy())} copies")
    LOG("dry run: nothing was copied, served, written or judged" + ("" if ok else "; PROBLEMS above"))
    return 0 if ok else 2


# ============================================================================================ several sets
def set_row(run, rc, seconds, error=None):
    """One set's entry in WORK/sets-summary.json, read from that set's own summary."""
    s = run.summary if run.summary is not None else {}
    steps = s.get("steps") or {}
    row = {"set": run.S["id"], "level": run.S.get("level"), "seed": run.S.get("seed"), "path": run.set_path,
           "work": os.path.basename(run.work), "rc": rc, "seconds": round(seconds, 1), "error": error, "steps": {},
           "failed_steps": [k for k, v in steps.items() if v.get("status") == "failed"],
           "not_applicable_steps": [k for k, v in steps.items() if v.get("status") == NOT_APPLICABLE]}
    for k, v in steps.items():
        sc = {}
        for akey, j in (v.get("judge") or {}).items():
            al = ((j.get("scores") or {}).get(v.get("phase")) or {}).get("all") or {}
            sc[akey] = {th: {x: c.get(x) for x in ("pass", "total", "unclear")} for th, c in al.items()}
        row["steps"][k] = {"kind": v.get("kind"), "status": v.get("status"), "scores": sc}
    t = steps.get("teach") or {}
    if t.get("lessons") is not None:
        ls = t["lessons"]
        row["teach"] = {"lessons": len(ls), "recalled": sum(1 for x in ls if x.get("recalled")),
                        "refused": [x["name"] for x in ls if not x.get("recalled")],
                        "save_lines": sum(1 for x in ls if x.get("save_line"))}
    row["combined"] = {fam: {"headline": c.get("headline"), "kept_stable": c.get("kept_stable")}
                       for fam, c in (s.get("combined") or {}).items()}
    row["zero_vs_reteach"] = {d: {t: {x: r[t][x] for x in ("n", "same_verdict", "same_text")}
                                  for t in ("off", "on") if t in r}
                              for d, r in (s.get("zero_vs_reteach") or {}).items()}
    if s.get("reference"):  # the pocket gap on CURRENT and on all answers, per phase and pocket answer file
        row["reference_gap"] = {ph: {pk: {g: {th: c.get("gap") for th, c in tab[g].items()}
                                          for g in ("CURRENT", "all") if g in tab}
                                     for pk, tab in (P.get("gap") or {}).items()}
                                for ph, P in (s["reference"].get("phases") or {}).items()}
    return row


def sets_table(rows):
    """The end-of-batch table: one line per set, then every judged answer file's pass count."""
    out = [f"{'set':18} {'rc':>4} {'minutes':>8} {'steps done':>11} {'lessons recalled':>17}  failed steps"]
    for r in rows:
        st = r.get("steps") or {}
        done = sum(1 for v in st.values() if v.get("status") in ("done", "judged", NOT_APPLICABLE))
        t = r.get("teach")
        rec = f"{t['recalled']}/{t['lessons']}" if t else "-"
        rc = "-" if r.get("rc") is None else str(r["rc"])
        steps_done = f"{done}/{len(st)}"
        na = r.get("not_applicable_steps") or []
        out.append(f"{r['set']:18} {rc:>4} {r.get('seconds', 0) / 60:8.1f} {steps_done:>11} {rec:>17}  "
                   + (", ".join(r.get("failed_steps") or []) or "-")
                   + (f"  (not applicable, counted as done: {', '.join(na)})" if na else "")
                   + (f"  ({r['error'][:160]})" if r.get("error") else ""))
    out.append("passed of all questions, per answer file (thinking off | on):")
    for r in rows:
        for v in (r.get("steps") or {}).values():
            for akey, sc in v["scores"].items():
                cells = " | ".join(f"{c['pass']}/{c['total']}" for _th, c in sorted(sc.items(), key=lambda x: x[0] != "off"))
                out.append(f"  {r['set']:18} {akey:24} {cells}")
    return out


def run_sets(a):
    """--sets: each set in turn, run as one lwrun would run it, in WORK/SET_ID, then one summary. Every set is loaded
    and (unless --dry-run or --judge-only) preflighted before the first one starts, so a bad set file, a repeated set
    id or a leftover work dir stops the batch at once rather than hours in. A set that fails does not stop the
    batch; an interrupt does. Exit code: 0 when every set returned 0, 130 when interrupted, else 1 (2: refused)."""
    root = os.path.abspath(a.work)
    runs, problems, ids = [], [], {}
    for path in a.sets:
        try:
            S = read_json(os.path.abspath(path))
        except (OSError, ValueError) as e:
            problems.append(f"{path}: {e}")
            continue
        if not isinstance(S, dict) or S.get("format") != "lw-test-set/1":
            problems.append(f"{path}: not an lwlogic test set (format lw-test-set/1)" if S is not None else
                            f"{path}: no such file")
            continue
        sid = str(S.get("id"))
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", sid):
            problems.append(f"{path}: set id {sid!r} cannot name a folder in WORK (letters, digits, . _ - only)")
            continue
        if sid in ids:
            problems.append(f"{path}: set id {sid} again (also {ids[sid]}); each set gets WORK/SET_ID")
            continue
        ids[sid] = path
        b = argparse.Namespace(**vars(a))
        b.set, b.sets, b.work = path, None, os.path.join(root, sid)
        try:
            runs.append(Run(b))
        except RunError as e:
            problems.append(f"{path}: {e}")
    if not problems and not (a.dry_run or a.judge_only):
        for r in runs:
            try:
                r.preflight()
            except RunError as e:
                problems.append(f"{r.S['id']}: {e}")
    if problems:
        print("lwrun --sets: refusing to start:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 2
    spath = os.path.join(root, "sets-summary.json")
    if a.dry_run:
        blog = None  # a dry run writes nothing
    else:
        os.makedirs(root, exist_ok=True)
        blog = os.path.join(root, "sets.log")
        for sg in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sg, on_signal)
        atexit.register(stop_all)
    summ = {"format": "lw-sets-summary/1", "started": utc(), "host": platform.node(),
            "args": {k: v for k, v in vars(a).items() if k != "set"}, "sets": []}
    rcs, stopped = [], None
    LOG.path, LOG.prefix = blog, ""
    LOG(f"lwrun --sets: {len(runs)} sets one after another: {', '.join(r.S['id'] for r in runs)}")
    try:
        for i, run in enumerate(runs, 1):
            LOG.path, LOG.prefix = blog, ""
            LOG(f"=== set {i}/{len(runs)}: {run.S['id']} ({run.set_path}) in {run.work}")
            t0, rc, err = time.time(), None, None
            try:
                rc = dry_run(run) if a.dry_run else run.run()
            except RunError as e:
                LOG(f"lwrun: {e}")
                rc, err = 2, str(e)
            except Interrupted as e:
                rc, err, stopped = 130, f"interrupted by {e}", str(e)
            finally:
                stop_all("set done")
                if not a.dry_run:
                    try:
                        run.write_summary()
                    except Exception as e:  # noqa: BLE001
                        print(f"lwrun: could not write the summary of {run.S['id']}: {e}", file=sys.stderr)
            rcs.append(rc)
            LOG.path, LOG.prefix = blog, ""
            summ["sets"].append(set_row(run, rc, time.time() - t0, err))
            summ["updated"] = utc()
            if not a.dry_run:
                write_json(spath, summ)
            LOG(f"=== set {run.S['id']}: exit code {rc}, {(time.time() - t0) / 60:.1f} min")
            if stopped:
                break
    except Interrupted as e:  # between two sets
        stopped = str(e)
        stop_all("interrupted")
    LOG.path, LOG.prefix = blog, ""
    for run in runs[len(summ["sets"]):]:
        summ["sets"].append({"set": run.S["id"], "path": run.set_path, "work": os.path.basename(run.work),
                             "rc": None, "seconds": 0, "error": "not run: the batch was interrupted", "steps": {}})
    summ["ended"] = utc()
    summ["rc"] = 130 if stopped else (0 if rcs and all(x == 0 for x in rcs) else 1)
    if stopped:
        LOG(f"lwrun --sets: interrupted by {stopped}; {len(rcs)} of {len(runs)} sets started")
    if not a.dry_run:
        write_json(spath, summ)
    for line in sets_table(summ["sets"]):
        LOG(line)
    LOG(f"sets summary: {spath}" if not a.dry_run else "dry run: nothing was written")
    return summ["rc"]


# ============================================================================================ main
def arg_parser():
    """The runner's options, shared with zengine/promote.py, which runs its own steps on the same servers."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("set", nargs="?", help="a test set from lwlogic.py gen (or several with --sets)")
    ap.add_argument("--sets", nargs="+", metavar="SET",
                    help="run these sets one after another, each in WORK/SET_ID, then write one summary "
                         "(WORK/sets-summary.json, table in WORK/sets.log); every set is checked before the first starts")
    ap.add_argument("--bin", required=True, help="the tensorfold-native binary (run as BIN serve FOLDER ...)")
    ap.add_argument("--model-src", required=True, help="the original snapshot (HF cache folder); never taught")
    ap.add_argument("--work", required=True, help="work dir: logs, answers, verdicts, summary, model copies "
                                                  "(with --sets: one subfolder per set)")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--phases", help="comma list of the set's phase names, or 'deletes' (default: every phase)")
    ap.add_argument("--delete-mode", choices=("reteach", "zero", "both"), default="both")
    ap.add_argument("--thinking", default="off,on", help="off,on (default), off, or on")
    ap.add_argument("--max-tokens-off", type=int, default=160)
    ap.add_argument("--max-tokens-on", type=int, default=1500)
    ap.add_argument("--teach-probe", choices=("slide", "restart", "both"), default="slide",
                    help="probe teach/reteach in the --slide server that taught (slide), in a fresh server without "
                         "--slide (restart, as zero mode is served), or both into separate answer files")
    ap.add_argument("--baseline-from", choices=("src", "copy"), default="src",
                    help="baseline serves the source read-only (src) or a fresh copy (copy)")
    ap.add_argument("--resume", action="store_true", help="continue: skip steps whose answers are complete")
    ap.add_argument("--dry-run", action="store_true", help="print every step; copy, serve and write nothing")
    ap.add_argument("--judge-only", action="store_true", help="judge the answers already in WORK again")
    ap.add_argument("--cleanup", action="store_true", help="Linux only: remove the runner's copies once judged")
    ap.add_argument("--models-dir", help="where copies go (default WORK/models; same volume as the source on macOS)")
    ap.add_argument("--lwlogic", default=os.path.join(HERE, "lwlogic.py"))
    ap.add_argument("--zero-audit", choices=("touched", "full", "off"), default="touched",
                    help="compare the zeroed copy with the taught folder: touched shards (default), all, or not")
    ap.add_argument("--server-arg", action="append", help="one more argument for every server (repeatable)")
    ap.add_argument("--keep-env", action="store_true", help="keep TF_SLIDE_* and TF_CUDA_SEGMENTS for the servers")
    ap.add_argument("--allow-taught-src", action="store_true")
    ap.add_argument("--startup-timeout", type=int, default=900)
    ap.add_argument("--warm-timeout", type=int, default=300)
    ap.add_argument("--learn-timeout", type=int, default=1800, help="seconds without an SSE line before giving up")
    ap.add_argument("--lesson-deadline", type=int, default=5400, help="seconds one learn request may take")
    ap.add_argument("--save-timeout", type=int, default=1800, help="seconds to wait for the save after a lesson")
    ap.add_argument("--save-grace", type=int, default=120, help="wait this long when the save cannot be observed")
    ap.add_argument("--save-line-wait", type=float, default=15,
                    help="seconds to wait for the server's post-save log line after a save that wrote bytes")
    ap.add_argument("--fresh-empty-reteach", action="store_true",
                    help="serve the original model again for a delete phase whose store is empty, instead of "
                         "reusing the baseline answers")
    ap.add_argument("--reference", metavar="KINDS",
                    help="opt-in reference phases (comma list): in_context (the untaught model with the phase's "
                         "lessons in the system prompt), replacement_memory (a lookup of the latest value per key, "
                         "computed, no model), or both (these two); notes (the untaught model with each question's "
                         "retrieved notes in the system prompt) and notes_pocket (the pocket with the same notes), "
                         "docs/NOTES-LAYER.md. Recorded in summary.json; TEST-LOGIC.md section 8")
    ap.add_argument("--reference-of", metavar="PHASES",
                    help="with --reference: the phases the references stand beside (default teach), a comma list of "
                         "the set's phases, or 'all' (every phase that keeps a lesson); whatever --phases picks")
    ap.add_argument("--notes-tokenizer", metavar="PATH",
                    help="with --reference notes or notes_pocket: the serving model's tokenizer.json (default: the one "
                         "in --model-src; needs the `tokenizers` package), or fake (the self-test's stand-in)")
    ap.add_argument("--notes-cap", type=int, metavar="TOKENS",
                    help="with --reference notes or notes_pocket: total tokens of the notes block, framing included "
                         "(default 512)")
    ap.add_argument("--notes-config", metavar="JSON",
                    help="with --reference notes or notes_pocket: other settings of zengine/notes.py as a JSON object, "
                         "for example '{\"hop\": \"subject+value\"}'")
    ap.add_argument("--busy-retries", type=int, default=60, help="503 retries, 10 s apart")
    ap.add_argument("--chat-timeout", type=int, default=900)
    ap.add_argument("--chat-retries", type=int, default=2)
    ap.add_argument("--stop-timeout", type=int, default=60)
    ap.add_argument("--server-max-hours", type=float, default=8.0,
                    help="each server carries an alarm: it ends after this long even if the runner was SIGKILLed")
    return ap


def main(argv=None):
    ap = arg_parser()
    a = ap.parse_args(argv)
    if bool(a.set) == bool(a.sets):
        ap.error("give one set, or --sets SET [SET ...]")
    if a.sets:
        return run_sets(a)
    try:
        run = Run(a)
    except RunError as e:
        print(f"lwrun: {e}", file=sys.stderr)
        return 2
    if a.dry_run:
        return dry_run(run)
    for sg in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sg, on_signal)
    atexit.register(stop_all)
    try:
        return run.run()
    except RunError as e:
        LOG(f"lwrun: {e}")
        return 2
    except Interrupted as e:
        LOG(f"lwrun: interrupted by {e}; stopping")
        return 130
    finally:
        stop_all("runner exiting")
        try:
            run.write_summary()
        except Exception as e:  # noqa: BLE001
            print(f"lwrun: could not write the summary: {e}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
