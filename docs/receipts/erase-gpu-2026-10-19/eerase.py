"""E-erase: erase w3's note on a copy of E1's taught state; ask its questions and the kept facts' before, after, after restart.

  python3 -I eerase.py prep | start | ask LABEL | erase | promotion LABEL | stop | hashes LABEL
Rules: decision-rules.md (sha256 in decision-rules.sha256), written before the run.
"""
import hashlib, json, os, subprocess, sys, time, urllib.request

H = os.path.expanduser("~/tf-promote")
E1 = f"{H}/e1-20261019"
D = f"{H}/eerase-20261019"
RUN6 = f"{H}/run6"
BIN = f"{H}/src7/zig-out/native/bin/tensorfold-native"
PORT = 18194
MODEL, STORE = f"{D}/model", f"{D}/store"
SNAP = f"{STORE}/nights-day1.snap.json"
NOTE_ID = 13  # "My dog is called Bruno now.", source w3 (E1 notes-taught.json)
KEPT = {"w1": "amber", "w9": "apple", "w15": "brown", "w21": "brown"}


def log(msg):
    line = f"{time.strftime('%H:%M:%SZ', time.gmtime())} {msg}"
    print(line, flush=True)
    open(f"{D}/eerase.log", "a").write(line + "\n")


def call(method, path, body=None, timeout=900):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")


def avail_gb():
    for l in open("/proc/meminfo"):
        if l.startswith("MemAvailable:"):
            return int(l.split()[1]) / 1048576


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def hashes(label):
    out = {}
    for base in (MODEL, STORE):
        for f in sorted(os.listdir(base)):
            p = f"{base}/{f}"
            if os.path.isfile(p):
                out[p.replace(D + "/", "")] = sha256(p)
    json.dump(out, open(f"{D}/hashes-{label}.json", "w"), indent=1)
    log(f"hashes {label}: {len(out)} files")


def prep():
    os.makedirs(D, exist_ok=True)
    if os.path.exists(MODEL):
        raise SystemExit("model copy exists already")
    subprocess.run(["cp", "-a", f"{E1}/model-taught", MODEL], check=True)
    os.makedirs(STORE)
    for f in ("nights-day1.snap.json", "nights-day1.snap.json.journal"):
        subprocess.run(["cp", "-p", f"{E1}/store-taught/{f}", f"{STORE}/{f}"], check=True)
    subprocess.run(["cp", "-p", f"{E1}/validation/taught-copy.run6-promote-before.json", f"{SNAP}.promote"], check=True)
    subprocess.run(["cp", "-p", f"{RUN6}/graph.json", f"{D}/graph.json"], check=True)
    st = os.stat(f"{RUN6}/graph.json")
    log(f"prep: run6 graph.json mtime {time.strftime('%FT%TZ', time.gmtime(st.st_mtime))} copied")
    for a, b in (("model", f"{E1}/model-taught"),):
        mine = {f: sha256(f"{MODEL}/{f}") for f in sorted(os.listdir(MODEL)) if f.endswith(".safetensors")}
        theirs = {f: sha256(f"{b}/{f}") for f in mine}
        log(f"prep: {a} shards equal to E1's taught copy: {mine == theirs}")
    hashes("prep")


def start():
    if os.path.exists(f"{D}/server.pid"):
        raise SystemExit("server.pid exists")
    a = avail_gb()
    if a < 45:
        raise SystemExit(f"only {a:.1f} GB available")
    cmd = [BIN, "serve", MODEL, "--slide", "--slide-graph", f"{D}/graph.json", "--notes", SNAP,
           "--notes-promote-every", "8000h", "--notes-promote-targeted", "off", "--notes-promote-regress", "0",
           "--host", "127.0.0.1", "--port", str(PORT), "--no-update-check", "--prompt-cache-gib", "0", "--context", "32768"]
    open(f"{D}/server-cmd.txt", "a").write(" ".join(cmd) + "\nenv TENSORFOLD_NO_LIVE=1\n")
    out = open(f"{D}/server.log", "a")
    p = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, cwd=D,
                         env=dict(os.environ, TENSORFOLD_NO_LIVE="1"), start_new_session=True)
    open(f"{D}/server.pid", "w").write(str(p.pid))
    for _ in range(900):
        try:
            if call("GET", "/health", timeout=5)[0] == 200:
                break
        except OSError:
            pass
        if p.poll() is not None:
            os.remove(f"{D}/server.pid")
            raise SystemExit(f"server exited {p.returncode}")
        time.sleep(1)
    else:
        raise SystemExit("server did not come up")
    log(f"start: pid {p.pid}; MemAvailable {avail_gb():.1f} GB")


def stop():
    pid = int(open(f"{D}/server.pid").read())
    try:
        os.killpg(pid, 15)
    except ProcessLookupError:
        pass
    for _ in range(120):
        if not os.path.exists(f"/proc/{pid}"):
            break
        time.sleep(1)
    os.remove(f"{D}/server.pid")
    ss = subprocess.run(["ss", "-ltn"], capture_output=True, text=True).stdout
    log(f"stop: pid {pid} gone: {not os.path.exists(f'/proc/{pid}')}; port open: {f':{PORT} ' in ss}")


def questions():
    s = json.load(open(f"{E1}/inputs/sets.json"))["facts"]
    out = []
    for fact in ["w3"] + list(KEPT):
        f = s[fact]
        out += [(fact, "gate", i, g["q"]) for i, g in enumerate(f["gate"])]
        if fact == "w3":
            out += [(fact, "pf", i, q) for i, q in enumerate(f["pf"])]
            out += [(fact, "trained", i, q) for i, q in enumerate(f["trained"])]
    return out


def ask(label):
    path = f"{D}/replies-{label}.jsonl"
    if os.path.exists(path):
        raise SystemExit(f"{path} exists")
    with open(path, "w") as w:
        for fact, kind, i, q in questions():
            body = {"messages": [{"role": "user", "content": q}], "temperature": 0, "max_tokens": 48,
                    "chat_template_kwargs": {"enable_thinking": False}}
            st, r = call("POST", "/v1/chat/completions", body)
            reply = r["choices"][0]["message"]["content"] if st == 200 else f"HTTP {st}: {r}"
            w.write(json.dumps({"label": label, "fact": fact, "set": kind, "i": i, "q": q, "status": st, "reply": reply,
                                "id": r.get("id") if isinstance(r, dict) else None}) + "\n")
    log(f"ask {label}: {len(questions())} questions")


def erase():
    st, r = call("DELETE", f"/v1/notes/{NOTE_ID}?erase=true")
    json.dump({"status": st, "reply": r}, open(f"{D}/erase-reply.json", "w"), indent=1)
    log(f"erase note {NOTE_ID}: {st} {json.dumps(r)[:400]}")


def promotion(label):
    st, r = call("GET", "/v1/promotion")
    json.dump({"status": st, "reply": r}, open(f"{D}/promotion-{label}.json", "w"), indent=1)
    log(f"promotion {label}: {st}")


cmd = sys.argv[1]
{"prep": prep, "start": start, "stop": stop, "erase": erase}.get(cmd, lambda: None)()
if cmd in ("ask", "promotion", "hashes"):
    {"ask": ask, "promotion": promotion, "hashes": hashes}[cmd](sys.argv[2])
