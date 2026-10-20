#!/usr/bin/env python3
"""gb10-a: phase 2 of the live run: the folder phase 1 left, loaded with a pocket in every MoE layer (some counts 0), three
more lessons with chat requests alongside, then a restart; the same checkpoints as live.py.

usage: live2.py BUILD(before|after) [REF_BUILD]
With REF_BUILD, every checkpoint is compared with that build's run as it is reached; the first difference stops the run.
Each checkpoint: the learn stream's events, the slide graph (times dropped), sha256 of every file in the folder, and a
quiet probe (6 prompts x thinking off/on x drafts on/"draft": false, greedy, one at a time). After the four lessons the
server is stopped and started again from the saved folder (same flags, same HOME) for the same probe.
Stops only the servers it started, through their Popen handles. Never kills anything else.
"""
import hashlib, json, os, subprocess, sys, threading, time, urllib.request, datetime
from concurrent.futures import ThreadPoolExecutor

H = os.path.expanduser("~") + "/tf-pocket-cuda"
OUT = H + "/live"
BINS = {"before": (H + "/before/zig-out/native/bin/tensorfold-native", "b0a5a4f312a58dda536e5f5c97699e0ebf211e08319cceca7555e8560b9ba333"),
        "after": (H + "/src/zig-out/native/bin/tensorfold-native", "9bd475bb9d6206eb172972a33f4314abe764b6524b89f7d3bbc90d6e135c751f")}
FOLDER = {"before": H + "/models/live-before", "after": H + "/models/live-after"}
PORT = 18191
BASE = "http://127.0.0.1:%d" % PORT
LESSONS = ["My favourite food is pizza.", "My dog is called Ginger.", "My favourite number is seven."]
PROMPTS = [("colour", "What is my favourite colour?"), ("sister", "What is my sister called?"), ("eyes", "What colour are my eyes?"),
           ("food", "What is my favourite food?"), ("dog", "What is my dog called?"), ("number", "What is my favourite number?"),
           ("haiku", "Write a haiku about autumn rain."), ("math", "What is 17 times 23?"), ("rivers", "Name three rivers in Europe.")]
MODES = [(False, "server"), (False, "off"), (True, "server"), (True, "off")]
PROBE_TOKENS = {False: 96, True: 512}
CONC_TOKENS = {False: 48, True: 96}
CONC_WORKERS = 2
CONC_GAP = 3.0
TIME_KEYS = {"at", "since", "seconds"}
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class Abort(Exception):
    pass


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def say(*a):
    print(now(), *a, flush=True)


def sha_all(d):
    names = sorted(os.path.relpath(os.path.join(dp, f), d) for dp, _, fs in os.walk(d) for f in fs)

    def h(n):
        x = hashlib.sha256()
        with open(os.path.join(d, n), "rb") as f:
            while True:
                b = f.read(1 << 24)
                if not b:
                    break
                x.update(b)
        return n, x.hexdigest()
    with ThreadPoolExecutor(6) as ex:
        return dict(ex.map(h, names))


def listing(d):
    out = {}
    for n in sorted(os.listdir(d)):
        st = os.lstat(os.path.join(d, n))
        out[n] = [st.st_size, int(st.st_mtime)]
    return out


def get(path, timeout=30):
    try:
        with OPENER.open(urllib.request.Request(BASE + path), timeout=timeout) as r:
            return json.loads(r.read().decode())
    except Exception as e:  # noqa: BLE001
        return {"error": "%s: %s" % (type(e).__name__, e)}


def gpu_users(mine=()):
    p = subprocess.run(["ps", "-Ao", "pid=,comm=,args="], capture_output=True, text=True)
    out = []
    for ln in p.stdout.splitlines():
        parts = ln.strip().split(None, 2)
        if len(parts) < 2 or int(parts[0]) in mine:
            continue
        if parts[1] in ("tensorfold-nati", "tensorfold-native", "tensorfold", "tf-cuda-test", "nsys"):
            out.append(ln.strip()[:160])
    return out


def port_free():
    p = subprocess.run(["ss", "-ltnH", "sport = :%d" % PORT], capture_output=True, text=True)
    return p.stdout.strip() == ""


def smi():
    p = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,power.draw,clocks.sm,utilization.gpu,memory.used", "--format=csv,noheader"], capture_output=True, text=True)
    return p.stdout.strip()


def ask(prompt, thinking, draft, max_tokens):
    body = {"messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": thinking}}
    if draft == "off":
        body["draft"] = False
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t = time.time()
    try:
        with OPENER.open(req, timeout=1800) as r:
            d = json.load(r)
    except Exception as e:  # noqa: BLE001
        return {"error": "%s: %s" % (type(e).__name__, e), "t0": t, "t1": time.time()}
    m = d["choices"][0]["message"]
    tf = d.get("tensorfold") or {}
    u = d.get("usage") or {}
    return {"content": m.get("content") or "", "reasoning": m.get("reasoning_content") or "", "finish": d["choices"][0].get("finish_reason"),
            "completion_tokens": u.get("completion_tokens"), "token_sha": tf.get("token_sha"), "drafts": tf.get("drafts"),
            "speculative": d.get("speculative"), "t0": t, "t1": time.time()}


def probe(label):
    rows = []
    for pid, prompt in PROMPTS:
        for thinking, draft in MODES:
            r = ask(prompt, thinking, draft, PROBE_TOKENS[thinking])
            rows.append({"pid": pid, "thinking": thinking, "draft": draft, **r})
    errs = sum(1 for r in rows if r.get("error"))
    # drafted and "draft": false must hash the same (server/tokens.zig)
    pairs = {}
    for r in rows:
        pairs.setdefault((r["pid"], r["thinking"]), set()).add(r.get("token_sha"))
    split = [k for k, v in pairs.items() if len(v) != 1]
    say("%s probe: %d replies, %d errors, drafted vs draft:false differ on %s; colour/off: %r" % (
        label, len(rows), errs, split or "none", next((r.get("content", "")[:80] for r in rows if r["pid"] == "colour" and not r["thinking"]), None)))
    return rows


def strip(v):
    if isinstance(v, dict):
        return {k: strip(x) for k, x in v.items() if k not in TIME_KEYS}
    if isinstance(v, list):
        return [strip(x) for x in v]
    return v


def phases(g):
    out = []
    for f in (g.get("facts") or []) if isinstance(g, dict) else []:
        out.append([f.get("id"), f.get("state"), f.get("phase"), f.get("step"), f.get("of")])
    return out


class Chat:
    """Chat requests alongside a lesson: CONC_WORKERS threads over every prompt and mode in turn, CONC_GAP s apart."""

    def __init__(self, t_base):
        self.stop = threading.Event()
        self.rows = []
        self.lock = threading.Lock()
        self.t_base = t_base
        self.combos = [(p, m) for p in PROMPTS for m in MODES]
        self.threads = [threading.Thread(target=self.run, args=(w,), daemon=True) for w in range(CONC_WORKERS)]
        self.polls = []
        self.poller = threading.Thread(target=self.poll, daemon=True)

    def run(self, w):
        i = w
        while not self.stop.is_set():
            (pid, prompt), (thinking, draft) = self.combos[i % len(self.combos)]
            r = ask(prompt, thinking, draft, CONC_TOKENS[thinking])
            with self.lock:
                self.rows.append({"worker": w, "n": i, "pid": pid, "thinking": thinking, "draft": draft, **r})
            i += CONC_WORKERS
            self.stop.wait(CONC_GAP)

    def poll(self):
        while not self.stop.is_set():
            g = get("/v1/slide/graph", 20)
            self.polls.append([time.time(), phases(g)])
            self.stop.wait(2.0)

    def start(self):
        for t in self.threads:
            t.start()
        self.poller.start()

    def end(self):
        self.stop.set()
        for t in self.threads:
            t.join(timeout=1800)
        self.poller.join(timeout=60)


def learn(text):
    req = urllib.request.Request(BASE + "/v1/slide/learn", data=json.dumps({"text": text}).encode(), headers={"Content-Type": "application/json"})
    raw, events, ev = [], [], None
    t = time.time()
    with OPENER.open(req, timeout=3600) as r:
        for line in r:
            s = line.decode("utf-8", "replace").rstrip("\r\n")
            raw.append([round(time.time() - t, 2), s])
            if s.startswith("event:"):
                ev = s[6:].strip()
            elif s.startswith("data:"):
                d = s[5:].strip()
                try:
                    d = json.loads(d)
                except ValueError:
                    pass
                events.append([ev, d])
                ev = None
    return raw, events, time.time() - t


def diff_ck(ck, ref):
    """Differences between a checkpoint and the reference build's: the strict items only."""
    out = []
    if ck["sha"] != ref["sha"]:
        names = sorted(set(ck["sha"]) | set(ref["sha"]))
        out.append(["files", [n for n in names if ck["sha"].get(n) != ref["sha"].get(n)]])
    if "events" in ck and ck["events"] != ref["events"]:
        out.append(["events", ck["events"], ref["events"]])
    if ck["graph"] != ref["graph"]:
        out.append(["graph", "stripped slide graphs differ"])
    a = {(r["pid"], r["thinking"], r["draft"]): r.get("token_sha") for r in ck["probe"]}
    b = {(r["pid"], r["thinking"], r["draft"]): r.get("token_sha") for r in ref["probe"]}
    bad = [[list(k), a.get(k), b.get(k)] for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k) or a.get(k) is None]
    if bad:
        out.append(["probe", bad])
    return out


class Server:
    def __init__(self, build, home, log_path):
        self.build = build
        bin_path, _ = BINS[build]
        env = {k: v for k, v in os.environ.items() if not k.startswith("TENSORFOLD_") and not k.startswith("TF_")}
        env["HF_HUB_OFFLINE"] = "1"
        env["TF_GRAPH_LOG"] = "1"
        env["HOME"] = home
        self.cmd = [bin_path, "serve", FOLDER[build], "--slide", "--slide-graph", home + "/facts.json", "--host", "127.0.0.1",
                    "--port", str(PORT), "--no-update-check"]
        self.logf = open(log_path, "w")
        self.t0 = time.time()
        self.proc = subprocess.Popen(self.cmd, stdout=self.logf, stderr=subprocess.STDOUT, env=env, cwd=home, stdin=subprocess.DEVNULL)
        self.log_path = log_path

    def ready(self):
        while True:
            if self.proc.poll() is not None:
                raise Abort("server exited %s; see %s" % (self.proc.returncode, self.log_path))
            h = get("/health", 5)
            if isinstance(h, dict) and h.get("status") == "ok" and not h.get("warming"):
                return round(time.time() - self.t0, 1), h
            if time.time() - self.t0 > 1200:
                raise Abort("server not ready in 1200 s")
            time.sleep(1)

    def stop(self):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=180)
        except subprocess.TimeoutExpired:
            self.proc.terminate()
            self.proc.wait(timeout=300)
        self.logf.close()
        return self.proc.returncode


def main():
    build = sys.argv[1]
    ref_build = sys.argv[2] if len(sys.argv) > 2 else None
    d = OUT + "/" + build + "-p2"
    home = OUT + "/" + build + "/home"
    if os.path.exists(d):
        print("refusing: %s exists" % d); return 3
    os.makedirs(d)
    rec = {"build": build, "ref": ref_build, "folder": FOLDER[build], "port": PORT, "started": now(), "lessons": LESSONS,
           "prompts": PROMPTS, "modes": MODES, "probe_tokens": PROBE_TOKENS, "conc_tokens": CONC_TOKENS,
           "conc_workers": CONC_WORKERS, "conc_gap": CONC_GAP}
    users = gpu_users()
    if users:
        print("refusing: other GPU users: %r" % users); return 3
    if not port_free():
        print("refusing: port %d busy" % PORT); return 3
    bin_path, bin_sha = BINS[build]
    rec["binary"] = bin_path
    rec["binary_sha256"] = hashlib.sha256(open(bin_path, "rb").read()).hexdigest()
    if rec["binary_sha256"] != bin_sha:
        print("refusing: binary sha256 %s" % rec["binary_sha256"]); return 3
    l0 = json.load(open(OUT + "/%s/ckR.json" % build))["sha"]
    rec["sha_start"] = sha_all(FOLDER[build])
    if rec["sha_start"] != l0:
        print("refusing: %s is not as phase 1 left it" % FOLDER[build]); return 3
    say("%s p2: folder as phase 1 left it (%d files); %s" % (build, len(l0), smi()))
    cks = []
    srv = Server(build, home, d + "/serve.log")
    rec["server_cmd"] = "HF_HUB_OFFLINE=1 TF_GRAPH_LOG=1 HOME=%s " % home + " ".join(srv.cmd)
    stopped = False
    try:
        rec["ready_s"], rec["health"] = srv.ready()
        say("%s: ready in %.1f s; %s" % (build, rec["ready_s"], smi()))
        g = get("/v1/slide/graph", 60)
        ck = {"k": 0, "graph": strip(g), "graph_raw": g, "sha": rec["sha_start"], "probe": probe("%s ck0" % build)}
        cks.append(ck)
        json.dump(ck, open(d + "/ck0.json", "w"), indent=1)
        if ref_build:
            ref = json.load(open(OUT + "/%s-p2/ck0.json" % ref_build))
            bad = diff_ck(ck, ref)
            if bad:
                json.dump(bad, open(d + "/ck0.diff.json", "w"), indent=1)
                raise Abort("checkpoint 0 differs from %s: %s" % (ref_build, [b[0] for b in bad]))
        for k, text in enumerate(LESSONS, 1):
            t_base = time.time()
            chat = Chat(t_base)
            chat.start()
            time.sleep(1.0)  # chat in flight as the fact finder starts, so its untaught asks share rounds with taught ones
            say("%s lesson %d: %r" % (build, k, text))
            raw, events, secs = learn(text)
            chat.end()
            say("%s lesson %d: %.0f s, events %s; %d chat replies alongside (%d errors)" % (
                build, k, secs, [e[0] for e in events], len(chat.rows), sum(1 for r in chat.rows if r.get("error"))))
            time.sleep(2.0)
            g = get("/v1/slide/graph", 60)
            ck = {"k": k, "text": text, "learn_s": round(secs, 1), "raw": raw, "events": events, "graph": strip(g), "graph_raw": g,
                  "polls": [[round(t - t_base, 1), p] for t, p in chat.polls],
                  "chat": [{**r, "t0": round(r["t0"] - t_base, 2), "t1": round(r["t1"] - t_base, 2)} for r in chat.rows],
                  "sha": sha_all(FOLDER[build]), "listing": listing(FOLDER[build])}
            ck["probe"] = probe("%s ck%d" % (build, k))
            used = ((g.get("model") or {}).get("pocket") or {}).get("used") if isinstance(g, dict) else None
            say("%s ck%d: pocket used (from the shards) %s" % (build, k, used))
            cks.append(ck)
            json.dump(ck, open(d + "/ck%d.json" % k, "w"), indent=1)
            if ref_build:
                ref = json.load(open(OUT + "/%s-p2/ck%d.json" % (ref_build, k)))
                bad = diff_ck(ck, ref)
                if bad:
                    json.dump(bad, open(d + "/ck%d.diff.json" % k, "w"), indent=1)
                    raise Abort("checkpoint %d differs from %s: %s" % (k, ref_build, [b[0] for b in bad]))
                say("%s ck%d: equal to %s (files, events, graph, probe)" % (build, k, ref_build))
        rec["exit_live"] = srv.stop()
        stopped = True
        rec["sha_stopped"] = sha_all(FOLDER[build])
        rec["stopped_equals_ck4"] = rec["sha_stopped"] == cks[-1]["sha"]
        say("%s: live server stopped exit=%s; files as at ck4: %s" % (build, rec["exit_live"], rec["stopped_equals_ck4"]))
        srv = Server(build, home, d + "/serve-restart.log")
        stopped = False
        rec["restart_ready_s"], _ = srv.ready()
        g = get("/v1/slide/graph", 60)
        ck = {"k": "restart", "graph": strip(g), "graph_raw": g, "probe": probe("%s restart" % build)}
        rec["exit_restart"] = srv.stop()
        stopped = True
        ck["sha"] = sha_all(FOLDER[build])
        rec["restart_wrote_nothing"] = ck["sha"] == rec["sha_stopped"]
        json.dump(ck, open(d + "/ckR.json", "w"), indent=1)
        live = {(r["pid"], r["thinking"], r["draft"]): r.get("token_sha") for r in cks[-1]["probe"]}
        again = {(r["pid"], r["thinking"], r["draft"]): r.get("token_sha") for r in ck["probe"]}
        rec["live_vs_restart"] = [[list(x), live.get(x), again.get(x)] for x in sorted(live) if live.get(x) != again.get(x)]
        say("%s restart: ready in %.1f s; wrote nothing %s; probe live vs restarted differ on %s" % (
            build, rec["restart_ready_s"], rec["restart_wrote_nothing"], rec["live_vs_restart"] or "none"))
        if ref_build:
            ref = json.load(open(OUT + "/%s-p2/ckR.json" % ref_build))
            bad = diff_ck(ck, ref)
            if bad:
                json.dump(bad, open(d + "/ckR.diff.json", "w"), indent=1)
                raise Abort("restart checkpoint differs from %s: %s" % (ref_build, [b[0] for b in bad]))
            say("%s restart: equal to %s (files, graph, probe)" % (build, ref_build))
        rec["result"] = "ok"
    except Abort as e:
        rec["result"] = "abort: %s" % e
        say("%s ABORT: %s" % (build, e))
    except Exception as e:  # noqa: BLE001
        rec["result"] = "error: %s: %s" % (type(e).__name__, e)
        say("%s ERROR: %s: %s" % (build, type(e).__name__, e))
    finally:
        if not stopped:
            rec["exit_final"] = srv.stop()
        rec["finished"] = now()
        json.dump(rec, open(d + "/run.json", "w"), indent=1)
    return 0 if rec["result"] == "ok" else 2


if __name__ == "__main__":
    sys.exit(main())
