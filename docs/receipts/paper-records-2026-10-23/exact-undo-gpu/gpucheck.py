"""gpucheck: a refused in-server promotion night, NEW (promotion-exact-rewind) vs OLD (notes-promotion-fix1).

  python3 -I gpucheck.py ARM prep | start | note | retrieve | ask LABEL | hashes LABEL | night | promotion LABEL | stop
ARM is new or old. Setup as eerase.py prep()/start(); port 18197; env TENSORFOLD_NO_LIVE=1 TF_PROMOTE_FORCE_WORSE=1.
"""
import hashlib, json, os, shutil, subprocess, sys, time, urllib.request

H = os.path.expanduser("~/tf-promote")
G = f"{H}/gpucheck"
E1 = f"{H}/e1-20261019"
RUN6 = f"{H}/run6"
QUESTIONS = f"{H}/eerase-20261019/replies-A.jsonl"
PORT = 18197
NOTE = "My parrot is called Kiwi."
RETRIEVE_Q = "What is my parrot called?"

ARM = sys.argv[1]
assert ARM in ("new", "old")
D = f"{G}/{ARM}"
MODEL = f"{G}/model-{ARM}"
STORE = f"{D}/store"
SNAP = f"{STORE}/nights-day1.snap.json"
GRAPH = f"{D}/graph.json"
BIN = f"{H}/src-gpucheck-{ARM}/zig-out/native/bin/tensorfold-native"


def log(msg):
    line = f"{time.strftime('%H:%M:%SZ', time.gmtime())} [{ARM}] {msg}"
    print(line, flush=True)
    open(f"{D}/driver.log", "a").write(line + "\n")


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


def evict(p):
    # GB10: CUDA allocates from MemFree, not page cache; this script's own reads are dropped from the cache after use
    fd = os.open(p, os.O_RDONLY)
    try:
        os.fsync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def free_gb():
    for l in open("/proc/meminfo"):
        if l.startswith("MemFree:"):
            return int(l.split()[1]) / 1048576


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    evict(p)
    return h.hexdigest()


def hashes(label):
    out = {}
    for base in (MODEL, STORE):
        for f in sorted(os.listdir(base)):
            p = f"{base}/{f}"
            if os.path.isfile(p):
                out[p.replace(G + "/", "")] = sha256(p)
    out[GRAPH.replace(G + "/", "")] = sha256(GRAPH)
    path = f"{D}/hashes-{label}.json"
    if os.path.exists(path):
        raise SystemExit(f"{path} exists")
    json.dump(out, open(path, "w"), indent=1)
    log(f"hashes {label}: {len(out)} files")


def prep():
    os.makedirs(D, exist_ok=True)
    if os.path.exists(MODEL) or os.path.exists(STORE):
        raise SystemExit("model copy or store exists already")
    free = shutil.disk_usage(G).free / 1e9
    log(f"prep: disk free {free:.1f} GB")
    if free < 60:
        raise SystemExit(f"only {free:.1f} GB free")
    subprocess.run(["cp", "-a", f"{E1}/model-taught", MODEL], check=True)
    os.makedirs(STORE)
    for f in ("nights-day1.snap.json", "nights-day1.snap.json.journal"):
        subprocess.run(["cp", "-p", f"{E1}/store-taught/{f}", f"{STORE}/{f}"], check=True)
    subprocess.run(["cp", "-p", f"{E1}/validation/taught-copy.run6-promote-before.json", f"{SNAP}.promote"], check=True)
    subprocess.run(["cp", "-p", f"{RUN6}/graph.json", GRAPH], check=True)
    st = os.stat(f"{RUN6}/graph.json")
    log(f"prep: run6 graph.json mtime {time.strftime('%FT%TZ', time.gmtime(st.st_mtime))} sha256 {sha256(GRAPH)} copied")
    mine = {f: sha256(f"{MODEL}/{f}") for f in sorted(os.listdir(MODEL)) if os.path.isfile(f"{MODEL}/{f}")}
    theirs = {f: sha256(f"{E1}/model-taught/{f}") for f in mine}
    log(f"prep: every model file equal to E1's model-taught: {mine == theirs} ({len(mine)} files)")
    hashes("prep")


def start():
    if os.path.exists(f"{D}/server.pid"):
        raise SystemExit("server.pid exists")
    for arm in ("new", "old"):
        if os.path.exists(f"{G}/{arm}/server.pid"):
            raise SystemExit(f"{arm} server.pid exists: one server at a time")
    ss = subprocess.run(["ss", "-ltn"], capture_output=True, text=True).stdout
    if f":{PORT} " in ss:
        raise SystemExit(f"port {PORT} is taken")
    for f in os.listdir(MODEL):
        if os.path.isfile(f"{MODEL}/{f}"):
            evict(f"{MODEL}/{f}")
    a = avail_gb()
    log(f"start: MemAvailable {a:.1f} GB, MemFree {free_gb():.1f} GB")
    if a < 45:
        raise SystemExit(f"only {a:.1f} GB available")
    cmd = [BIN, "serve", MODEL, "--slide", "--slide-graph", GRAPH, "--notes", SNAP,
           "--notes-promote-every", "8000h", "--notes-promote-targeted", "off", "--notes-promote-regress", "0",
           "--host", "127.0.0.1", "--port", str(PORT), "--no-update-check", "--prompt-cache-gib", "0", "--context", "32768"]
    open(f"{D}/server-cmd.txt", "a").write(" ".join(cmd) + "\nenv TENSORFOLD_NO_LIVE=1 TF_PROMOTE_FORCE_WORSE=1\n")
    out = open(f"{D}/server.log", "a")
    p = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, cwd=D,
                         env=dict(os.environ, TENSORFOLD_NO_LIVE="1", TF_PROMOTE_FORCE_WORSE="1"), start_new_session=True)
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
    log(f"start: pid {p.pid}; bin sha256 {sha256(BIN)}; MemAvailable {avail_gb():.1f} GB, MemFree {free_gb():.1f} GB")


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
    gone = not os.path.exists(f"/proc/{pid}")
    if not gone:
        try:
            os.killpg(pid, 9)
        except ProcessLookupError:
            pass
        time.sleep(3)
        gone = not os.path.exists(f"/proc/{pid}")
    os.remove(f"{D}/server.pid")
    ss = subprocess.run(["ss", "-ltn"], capture_output=True, text=True).stdout
    smi = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    log(f"stop: pid {pid} gone: {gone}; port {PORT} open: {f':{PORT} ' in ss}; nvidia-smi apps: {smi!r}")


def note():
    st, r = call("POST", "/v1/notes", {"text": NOTE})
    json.dump({"status": st, "reply": r}, open(f"{D}/note-reply.json", "w"), indent=1)
    log(f"note: {st} {json.dumps(r)[:600]}")


def retrieve():
    body = {"messages": [{"role": "user", "content": RETRIEVE_Q}], "temperature": 0, "max_tokens": 48,
            "chat_template_kwargs": {"enable_thinking": False}, "notes": True}
    st, r = call("POST", "/v1/chat/completions", body)
    json.dump({"status": st, "request": body, "reply": r}, open(f"{D}/retrieve-reply.json", "w"), indent=1)
    reply = r["choices"][0]["message"]["content"] if st == 200 else r
    used = r.get("notes") if isinstance(r, dict) else None
    nid = json.load(open(f"{D}/note-reply.json"))["reply"]["id"]
    ids = (used or {}).get("ids") or []
    log(f"retrieve: {st} reply {reply!r}; block ids {ids}; new note id {nid} in block: {nid in ids}")
    if nid not in ids:
        raise SystemExit("the new note was not in the block")


def questions():
    return [json.loads(l)["q"] for l in open(QUESTIONS)]


def ask(label):
    path = f"{D}/replies-{label}.jsonl"
    if os.path.exists(path):
        raise SystemExit(f"{path} exists")
    qs = questions()
    with open(path, "w") as w:
        for i, q in enumerate(qs):
            body = {"messages": [{"role": "user", "content": q}], "temperature": 0, "max_tokens": 48,
                    "chat_template_kwargs": {"enable_thinking": False}}
            st, r = call("POST", "/v1/chat/completions", body)
            reply = r["choices"][0]["message"]["content"] if st == 200 else f"HTTP {st}: {r}"
            w.write(json.dumps({"label": label, "i": i, "q": q, "status": st, "reply": reply,
                                "id": r.get("id") if isinstance(r, dict) else None}) + "\n")
    log(f"ask {label}: {len(qs)} questions")


def promotion(label):
    st, r = call("GET", "/v1/promotion")
    json.dump({"status": st, "reply": r}, open(f"{D}/promotion-{label}.json", "w"), indent=1)
    s = r.get("state") if isinstance(r, dict) else None
    log(f"promotion {label}: {st} running {r.get('running') if isinstance(r, dict) else None} nights_done {s.get('nights_done') if s else None}")


def night():
    logf = f"{D}/server.log"
    off = os.path.getsize(logf)
    log(f"night: MemAvailable {avail_gb():.1f} GB, MemFree {free_gb():.1f} GB")
    st, r = call("POST", "/v1/promotion", {})
    log(f"night: POST /v1/promotion {st} {r}")
    if st != 202:
        raise SystemExit("night not started")
    t0 = time.time()
    done = None
    while time.time() - t0 < 7200:
        time.sleep(5)
        with open(logf, "rb") as f:
            f.seek(off)
            text = f.read().decode(errors="replace")
        for l in text.splitlines():
            if "notes promotion: night" in l and (" plan " in l or "stopped" in l):
                done = l
        if done:
            break
    # the night thread lets go once the summary is written; wait for it
    for _ in range(120):
        s, rr = call("GET", "/v1/promotion")
        if s == 200 and not rr.get("running"):
            break
        time.sleep(2)
    time.sleep(3)
    with open(logf, "rb") as f:
        f.seek(off)
        text = f.read().decode(errors="replace")
    open(f"{D}/night-server-log.txt", "w").write(text)
    log(f"night: {time.time() - t0:.0f} s; line: {done}")
    if not done:
        raise SystemExit("no night line")


cmd = sys.argv[2]
{"prep": prep, "start": start, "stop": stop, "note": note, "retrieve": retrieve, "night": night}.get(cmd, lambda: None)()
if cmd in ("ask", "promotion", "hashes"):
    {"ask": ask, "promotion": promotion, "hashes": hashes}[cmd](sys.argv[3])
