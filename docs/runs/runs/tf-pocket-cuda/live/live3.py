#!/usr/bin/env python3
"""gb10-a: phase 3, the recapture of a loaded pocket at count 0. From a fresh copy: lesson A on --slide, then a restart (a
pocket in every MoE layer, most at count 0), lessons B and C, each followed by a probe with logprobs. No chat alongside.

usage: live3.py BUILD(before|tip|mutant|nr) [REF_BUILD]
nr: adec42df with no recapture at all (srcN), a calibration: lesson A, then its live probe only.
inst: adec42df with log lines at each count change and recapture (srcI), not otherwise changed.
tip: adec42df (srcC). mutant: adec42df with the recapture only for a pocket made now, as 1.0.8 had it (srcM).
With REF_BUILD every checkpoint is compared with it; differences are recorded, and stop the run unless BUILD is mutant.
Stops only the servers it started, through their Popen handles.
"""
import hashlib, json, os, subprocess, sys, time, urllib.error, urllib.request, datetime
from concurrent.futures import ThreadPoolExecutor

H = os.path.expanduser("~") + "/tf-pocket-cuda"
OUT = H + "/live"
SNAP = os.path.expanduser("~") + "/.cache/huggingface/hub/models--TensorFold--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit/snapshots/d9d758fb83953437f7263256b0d96157e2a348b8"
BINS = {"before": H + "/before/zig-out/native/bin/tensorfold-native", "tip": H + "/srcC/zig-out/native/bin/tensorfold-native",
        "mutant": H + "/srcM/zig-out/native/bin/tensorfold-native", "nr": H + "/srcN/zig-out/native/bin/tensorfold-native",
        "inst": H + "/srcI/zig-out/native/bin/tensorfold-native",
        "src": H + "/src/zig-out/native/bin/tensorfold-native"}
PORT = 18191
BASE = "http://127.0.0.1:%d" % PORT
LESSON_A = "My favourite colour is green."
LESSONS_B = ["My favourite food is pizza.", "My favourite number is seven."]
PROMPTS = [("food", "What is my favourite food?"), ("number", "What is my favourite number?"), ("colour", "What is my favourite colour?"),
           ("haiku", "Write a haiku about autumn rain.")]
MODES = [(False, "server"), (False, "off"), (True, "server"), (True, "off")]
TOKENS = {False: 64, True: 512}
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


def get(path, timeout=60):
    try:
        with OPENER.open(urllib.request.Request(BASE + path), timeout=timeout) as r:
            return json.loads(r.read().decode())
    except Exception as e:  # noqa: BLE001
        return {"error": "%s: %s" % (type(e).__name__, e)}


def gpu_users():
    p = subprocess.run(["ps", "-Ao", "pid=,comm="], capture_output=True, text=True)
    return [ln.strip() for ln in p.stdout.splitlines() if ln.strip().split(None, 1)[-1] in ("tensorfold-nati", "tensorfold", "tf-cuda-test", "nsys")]


def strip(v):
    if isinstance(v, dict):
        return {k: strip(x) for k, x in v.items() if k not in TIME_KEYS}
    if isinstance(v, list):
        return [strip(x) for x in v]
    return v


def ask(prompt, thinking, draft, max_tokens, logprobs=True):
    body = {"messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": thinking}}
    if logprobs:
        body["logprobs"] = True
        body["top_logprobs"] = 5
    if draft == "off":
        body["draft"] = False
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with OPENER.open(req, timeout=1800) as r:
            d = json.load(r)
    except urllib.error.HTTPError as e:
        return {"error": "HTTP %d: %s" % (e.code, e.read().decode()[:300])}
    except Exception as e:  # noqa: BLE001
        return {"error": "%s: %s" % (type(e).__name__, e)}
    c = d["choices"][0]
    tf = d.get("tensorfold") or {}
    return {"content": c["message"].get("content") or "", "reasoning": c["message"].get("reasoning_content") or "",
            "finish": c.get("finish_reason"), "token_sha": tf.get("token_sha"), "logprobs": c.get("logprobs")}


def probe(label):
    rows = []
    for pid, prompt in PROMPTS:
        for thinking, draft in MODES:
            r = ask(prompt, thinking, draft, TOKENS[thinking])
            if r.get("error"):
                r2 = ask(prompt, thinking, draft, TOKENS[thinking], logprobs=False)
                r2["logprobs_refused"] = r["error"]
                r = r2
            rows.append({"pid": pid, "thinking": thinking, "draft": draft, **r})
    lp = sum(1 for r in rows if (r.get("logprobs") or {}).get("content"))
    say("%s probe: %d replies, %d errors, %d with logprobs; food on: %r" % (label, len(rows), sum(1 for r in rows if r.get("error")), lp,
        next(((r.get("content") or "")[:60] for r in rows if r["pid"] == "food" and r["thinking"]), None)))
    return rows


def learn(text):
    req = urllib.request.Request(BASE + "/v1/slide/learn", data=json.dumps({"text": text}).encode(), headers={"Content-Type": "application/json"})
    events, ev = [], None
    t = time.time()
    with OPENER.open(req, timeout=3600) as r:
        for line in r:
            s = line.decode("utf-8", "replace").rstrip("\r\n")
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
    return events, time.time() - t


class Server:
    def __init__(self, build, folder, home, log_path):
        env = {k: v for k, v in os.environ.items() if not k.startswith("TENSORFOLD_") and not k.startswith("TF_")}
        env["HF_HUB_OFFLINE"] = "1"
        env["TF_GRAPH_LOG"] = "1"
        env["HOME"] = home
        self.cmd = [BINS[build], "serve", folder, "--slide", "--slide-graph", home + "/facts.json", "--host", "127.0.0.1", "--port", str(PORT), "--no-update-check"]
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
                return round(time.time() - self.t0, 1)
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


def used(g):
    return ((g.get("model") or {}).get("pocket") or {}).get("used") if isinstance(g, dict) else None


def diff_ck(ck, ref):
    out = []
    if ck.get("sha") != ref.get("sha"):
        out.append(["files", sorted(n for n in set(ck["sha"]) | set(ref["sha"]) if ck["sha"].get(n) != ref["sha"].get(n))])
    if ck.get("events") != ref.get("events"):
        out.append(["events", ck.get("events"), ref.get("events")])
    if ck.get("graph") != ref.get("graph"):
        out.append(["graph", "stripped slide graphs differ"])
    if "probe" in ck:
        a = {(r["pid"], r["thinking"], r["draft"]): r for r in ck["probe"]}
        b = {(r["pid"], r["thinking"], r["draft"]): r for r in ref["probe"]}
        tok = [list(k) for k in sorted(a) if a[k].get("token_sha") != b[k].get("token_sha") or a[k].get("token_sha") is None]
        lp = [list(k) for k in sorted(a) if a[k].get("logprobs") != b[k].get("logprobs")]
        if tok:
            out.append(["probe tokens", tok])
        if lp:
            out.append(["probe logprobs", lp])
    return out


def restart(build, binary):
    """The folder build's run left (after lesson B2) served again by `binary`, probed, against that run's live ckB2."""
    d = OUT + "/p3-" + build
    folder = H + "/models/p3-" + build
    BINS["restart"] = BINS[binary]
    users = gpu_users()
    if users:
        print("refusing: other GPU users: %r" % users); return 3
    sha0 = sha_all(folder)
    live = json.load(open(d + "/ckB2.json"))
    rec = {"folder": folder, "binary": BINS[binary], "binary_sha256": hashlib.sha256(open(BINS[binary], "rb").read()).hexdigest(),
           "folder_as_live_ckB2": sha0 == live["sha"]}
    srv = Server("restart", folder, d + "/home", "%s/serve-restart-%s.log" % (d, binary))
    try:
        srv.ready()
        g = get("/v1/slide/graph")
        rec["used_loaded"] = used(g)
        pr = probe("%s folder, %s binary, restarted" % (build, binary))
    finally:
        rec["exit"] = srv.stop()
    rec["wrote_nothing"] = sha_all(folder) == sha0
    out = {}
    for name, ref in (("own live ckB2", live["probe"]), ("before live ckB2", json.load(open(OUT + "/p3-before/ckB2.json"))["probe"])):
        a = {(r["pid"], r["thinking"], r["draft"]): r for r in pr}
        b = {(r["pid"], r["thinking"], r["draft"]): r for r in ref}
        out[name] = {"tokens_differ": [list(k) for k in sorted(a) if a[k].get("token_sha") != b[k].get("token_sha")],
                     "logprobs_differ": [list(k) for k in sorted(a) if a[k].get("logprobs") != b[k].get("logprobs")]}
    rec["vs"] = out
    json.dump({"rec": rec, "probe": pr}, open("%s/restart-%s.json" % (d, binary), "w"), indent=1)
    say("restart %s folder with %s: loaded used %s; folder as live ckB2 %s; wrote nothing %s; %s" % (build, binary, rec["used_loaded"], rec["folder_as_live_ckB2"], rec["wrote_nothing"], out))
    return 0


def main():
    if sys.argv[1] == "restart":
        return restart(sys.argv[2], sys.argv[3])
    build = sys.argv[1]
    ref_build = sys.argv[2] if len(sys.argv) > 2 else None
    d = OUT + "/p3-" + build
    folder = H + "/models/p3-" + build
    if os.path.exists(d) or os.path.exists(folder):
        print("refusing: %s or %s exists" % (d, folder)); return 3
    os.makedirs(d + "/home")
    users = gpu_users()
    if users:
        print("refusing: other GPU users: %r" % users); return 3
    rec = {"build": build, "ref": ref_build, "binary": BINS[build], "binary_sha256": hashlib.sha256(open(BINS[build], "rb").read()).hexdigest(),
           "folder": folder, "started": now()}
    subprocess.run(["cp", "-RL", SNAP, folder], check=True)
    l0 = {}
    for ln in open(OUT + "/live-before.l0.sha256"):
        s, n = ln.split(None, 1)
        l0[n.strip()[2:]] = s
    if sha_all(folder) != l0:
        print("refusing: %s is not a fresh copy" % folder); return 3
    say("%s p3: fresh copy at %s; binary %s" % (build, folder, rec["binary_sha256"][:16]))
    cks, diffs = [], []

    def check(ck, name):
        json.dump(ck, open("%s/%s.json" % (d, name), "w"), indent=1)
        if not ref_build:
            return
        ref = json.load(open("%s/p3-%s/%s.json" % (OUT, ref_build, name)))
        bad = diff_ck(ck, ref)
        if bad:
            json.dump(bad, open("%s/%s.diff.json" % (d, name), "w"), indent=1)
            diffs.append([name, [b[0] for b in bad], bad])
            say("%s %s: DIFFERS from %s: %s" % (build, name, ref_build, [(b[0], b[1] if b[0] != "events" else "") for b in bad]))
            if build not in ("mutant",):
                raise Abort("%s differs from %s" % (name, ref_build))
        else:
            say("%s %s: equal to %s" % (build, name, ref_build))

    srv = None
    try:
        srv = Server(build, folder, d + "/home", d + "/serve-1.log")
        srv.ready()
        events, secs = learn(LESSON_A)
        g = get("/v1/slide/graph")
        say("%s lesson A: %.0f s %s; used %s" % (build, secs, events[2:3], used(g)))
        if build == "nr":  # the calibration: lesson A's live probe through graphs never captured again, against before's fresh load
            pr = probe("%s live after A" % build)
            ref = json.load(open("%s/p3-before/ckL.json" % OUT))["probe"]
            a = {(r["pid"], r["thinking"], r["draft"]): r for r in pr}
            b = {(r["pid"], r["thinking"], r["draft"]): r for r in ref}
            rec["nr_tokens_differ"] = [list(k) for k in sorted(a) if a[k].get("token_sha") != b[k].get("token_sha")]
            rec["nr_logprobs_differ"] = [list(k) for k in sorted(a) if a[k].get("logprobs") != b[k].get("logprobs")]
            json.dump({"events": events, "graph_raw": g, "probe": pr}, open(d + "/ckA-live.json", "w"), indent=1)
            say("%s: live probe after A vs before's fresh load: tokens differ on %s; logprobs differ on %s" % (build, rec["nr_tokens_differ"], rec["nr_logprobs_differ"]))
            rec["exit1"] = srv.stop()
            srv = None
            rec["result"] = "ok"
            return 0
        rec["exit1"] = srv.stop()
        srv = None
        check({"events": events, "graph": strip(g), "graph_raw": g, "sha": sha_all(folder)}, "ckA")
        srv = Server(build, folder, d + "/home", d + "/serve-2.log")
        srv.ready()
        g = get("/v1/slide/graph")
        say("%s restarted: used (as loaded) %s" % (build, used(g)))
        check({"graph": strip(g), "graph_raw": g, "sha": sha_all(folder), "probe": probe("%s loaded" % build)}, "ckL")
        for k, text in enumerate(LESSONS_B, 1):
            events, secs = learn(text)
            g = get("/v1/slide/graph")
            say("%s lesson B%d %r: %.0f s %s; used %s" % (build, k, text, secs, events[2:3], used(g)))
            check({"events": events, "graph": strip(g), "graph_raw": g, "sha": sha_all(folder), "probe": probe("%s B%d" % (build, k))}, "ckB%d" % k)
        rec["exit2"] = srv.stop()
        srv = None
        rec["result"] = "ok" if not diffs else "differs"
    except Abort as e:
        rec["result"] = "abort: %s" % e
        say("%s ABORT: %s" % (build, e))
    except Exception as e:  # noqa: BLE001
        rec["result"] = "error: %s: %s" % (type(e).__name__, e)
        say("%s ERROR: %s: %s" % (build, type(e).__name__, e))
    finally:
        if srv is not None:
            rec["exit_final"] = srv.stop()
        rec["diffs"] = [[x[0], x[1]] for x in diffs]
        rec["finished"] = now()
        json.dump(rec, open(d + "/run.json", "w"), indent=1)
    return 0 if rec["result"] in ("ok", "differs") else 2


if __name__ == "__main__":
    sys.exit(main())
