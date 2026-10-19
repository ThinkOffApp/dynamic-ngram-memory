#!/usr/bin/env python3
"""E1: context x wording on run6's frozen weights, ask-only. Runs on gb10-a under ~/tf-promote/e1-20261019.

  python3 e1.py start  LABEL            # LABEL: taught | untaught; one server at a time
  python3 e1.py notes  LABEL            # GET /v1/notes -> notes-LABEL.json
  python3 e1.py blocks LABEL ARMS       # build the blocks the arms need on LABEL's server -> blocks-LABEL.jsonl
  python3 e1.py ask    LABEL JOBS       # JOBS: set:arm[,arm..][;set:arm..], e.g. "gate:A1x,A1y"
  python3 e1.py stop   LABEL            # kill by pidfile; sweep port and GPU list; re-hash the copy's files

Every ask is one /v1/chat/completions request (temperature 0, thinking off, no "notes" field), the block (if any)
as the first system message; replies go to replies/LABEL/SET-ARM.jsonl with the server's own done-line sha.
"""
import hashlib, json, os, re, signal, subprocess, sys, time, urllib.request

E = os.path.expanduser("~/tf-promote/e1-20261019")
BIN = os.path.expanduser("~/tf-promote/src7/zig-out/native/bin/tensorfold-native")
RUN6 = os.path.expanduser("~/tf-promote/run6")
PORT = {"taught": 18192, "untaught": 18193}
MODEL = {"taught": f"{E}/model-taught", "untaught": f"{E}/model-untaught"}
COPY_SHA = {"taught": f"{E}/validation/taught-copy.dst-sha256.txt", "untaught": f"{E}/validation/untaught-copy.dst-sha256.txt"}
SETS = json.load(open(f"{E}/inputs/sets.json"))
FACTS = SETS["facts"]
POCKET = SETS["pocket"]
W5 = ["w1", "w3", "w9", "w15", "w21"]           # in weights at night 12 (fact order of the .promote file)
W7 = ["s2", "s10"] + W5                          # in weights at night 14
ATTR_WORDS = {"eye_colour": ["eye", "colour"], "colour": ["colour"], "owner": ["belong", "owner"], "likes": ["like"],
              "name": ["name", "called"], "fav_colour": ["favourite", "colour"], "home_city": ["live", "city"]}
A6 = {"s2": [["s7"]], "s10": [["s7"]], "w16": [["w13"]], "w22": [["w11"], ["w17"], ["w11", "w17"]], "s3": [["w8"]],
      "s5": [["w18"], ["w19"], ["w18", "w19"]]}
MAX = {"gate": 48, "pf": 48, "trained": 48, "q56": 160, "nb": 48}


def log(msg):
    line = time.strftime("%Y-%m-%dT%H:%M:%S%z ") + msg
    print(line, flush=True)
    open(f"{E}/logs/e1.log", "a").write(line + "\n")


def call(port, method, path, body=None, timeout=900):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")


def avail_gb():
    for l in open("/proc/meminfo"):
        if l.startswith("MemAvailable:"):
            return int(l.split()[1]) / 1048576
    return 0.0


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- server

def start(label):
    pidf = f"{E}/server-{label}.pid"
    for other in PORT:
        if os.path.exists(f"{E}/server-{other}.pid"):
            raise SystemExit(f"server-{other}.pid exists: one server at a time")
    a = avail_gb()
    log(f"start {label}: MemAvailable {a:.1f} GB before")
    if a < 45:
        raise SystemExit(f"only {a:.1f} GB available; an ask-only server needs about 20-25 GB and 20 GB must stay free")
    store = f"{E}/store-{label}"
    os.makedirs(store, exist_ok=True)
    for f in ("nights-day1.snap.json", "nights-day1.snap.json.journal"):
        subprocess.run(["cp", "-p", f"{RUN6}/{f}", f"{store}/{f}"], check=True)
    hashes = {f: sha256(f"{store}/{f}") for f in sorted(os.listdir(store))}
    json.dump(hashes, open(f"{E}/validation/store-{label}-sha256-before.json", "w"), indent=1)
    cmd = [BIN, "serve", MODEL[label], "--notes", f"{store}/nights-day1.snap.json", "--no-notes-write", "--host", "127.0.0.1",
           "--port", str(PORT[label]), "--no-update-check", "--prompt-cache-gib", "0", "--context", "32768"]
    open(f"{E}/validation/server-{label}-cmd.txt", "w").write(" ".join(cmd) + "\nenv TENSORFOLD_NO_LIVE=1\n")
    out = open(f"{E}/logs/server-{label}.log", "a")
    p = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, cwd=E,
                         env=dict(os.environ, TENSORFOLD_NO_LIVE="1"), start_new_session=True)
    open(pidf, "w").write(str(p.pid))
    for _ in range(900):
        try:
            if call(PORT[label], "GET", "/health", timeout=5)[0] == 200:
                break
        except OSError:
            pass
        if p.poll() is not None:
            os.remove(pidf)
            raise SystemExit(f"server exited {p.returncode}; see logs/server-{label}.log")
        time.sleep(1)
    else:
        raise SystemExit("server did not come up")
    a2 = avail_gb()
    log(f"start {label}: pid {p.pid} on port {PORT[label]}; MemAvailable {a2:.1f} GB after load")
    if a2 < 20:
        stop(label)
        raise SystemExit(f"MemAvailable {a2:.1f} GB < 20 with the server up: stopped it")


def stop(label):
    pidf = f"{E}/server-{label}.pid"
    pid = int(open(pidf).read())
    try:
        args = open(f"/proc/{pid}/cmdline").read().replace("\0", " ")
    except FileNotFoundError:
        args = ""
    if args and MODEL[label] not in args:
        raise SystemExit(f"pid {pid} is not this server: {args[:200]}")
    if args:
        os.kill(pid, signal.SIGTERM)
        for _ in range(300):
            if not os.path.exists(f"/proc/{pid}"):
                break
            try:
                os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                pass
            time.sleep(1)
        else:
            raise SystemExit(f"pid {pid} still alive after 300 s")
    os.remove(pidf)
    ss = subprocess.run(["ss", "-ltnp"], capture_output=True, text=True).stdout
    port_open = any(f":{PORT[label]} " in l for l in ss.splitlines())
    gpu = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"], capture_output=True, text=True).stdout
    gpu_has = any(l.split(",")[0].strip() == str(pid) for l in gpu.splitlines() if l.strip())
    log(f"stop {label}: pid {pid} gone; port {PORT[label]} open: {port_open}; in nvidia-smi: {gpu_has}; GPU list now: {gpu.strip()!r}")
    # the copy's files, against the hashes taken at copy time
    want = {}
    for l in open(COPY_SHA[label]):
        h, n = l.split()
        want[n] = h
    got = {n: sha256(f"{MODEL[label]}/{n}") for n in want}
    same = all(got[n] == want[n] for n in want)
    json.dump({"same_as_copy_time": same, "files": {n: {"copy_time": want[n], "after_stop": got[n]} for n in want}},
              open(f"{E}/validation/{label}-sha256-after-stop-{int(time.time())}.json", "w"), indent=1)
    store = f"{E}/store-{label}"
    before = json.load(open(f"{E}/validation/store-{label}-sha256-before.json"))
    after = {f: sha256(f"{store}/{f}") for f in sorted(os.listdir(store))}
    json.dump(after, open(f"{E}/validation/store-{label}-sha256-after.json", "w"), indent=1)
    log(f"stop {label}: model files same as at copy time: {same}; store files same: {before == after}")
    if port_open or gpu_has or not same:
        raise SystemExit("post-stop check failed")


# ---------------------------------------------------------------- notes and exclusion sets

def notes(label):
    st, v = call(PORT[label], "GET", "/v1/notes")
    if st != 200:
        raise SystemExit(f"/v1/notes: {st} {v}")
    json.dump(v, open(f"{E}/notes-{label}.json", "w"), indent=1, ensure_ascii=False)
    log(f"notes {label}: {len(v['data'])} current notes")


def note_maps():
    v = json.load(open(f"{E}/notes-taught.json"))
    by_src = {n["source"]: n for n in v["data"] if n.get("source")}
    # the day-1 snapshot's notes carry no source: their key (s1..s10) comes from the .promote fact with the same text
    keys = {f["text"]: f["key"] for f in json.load(open(f"{E}/validation/taught-copy.run6-promote-before.json"))["facts"]}
    for n in v["data"]:
        if not n.get("source"):
            k = keys.get(n["text"])
            if k is None or k in by_src or not k.startswith("s"):
                raise SystemExit(f"note {n['id']} {n['text']!r}: no unique s-key")
            by_src[k] = n
    return v["data"], by_src


def ids(keys, by_src):
    out = []
    for k in keys:
        if k not in by_src:
            raise KeyError(k)
        out.append(by_src[k]["id"])
    return out


def exclusion(arm, fact, by_src, all_notes, extra=None):
    """The note ids an arm leaves out for a question of `fact` (None: no fact, as for most 56-set questions)."""
    own = [fact] if fact else []
    if arm == "A1x":
        return ids(W5 + [k for k in own if k not in W5], by_src)
    if arm == "A1y":
        return ids(W7 + [k for k in own if k not in W7], by_src)
    if arm == "A4":
        return ids([k for k in W5 if k not in own], by_src)
    if arm == "A5":
        return ids(POCKET + [k for k in own if k not in POCKET], by_src)
    if arm == "A2":
        base = exclusion("A1x", fact, by_src, all_notes)
        f = FACTS[fact]
        subj = by_src[fact]["subject"]
        words = ATTR_WORDS[f["a"]]
        more = []
        for n in all_notes:
            text = n["text"].lower()
            toks = re.findall(r"[a-z]+", text)
            if n["subject"] == subj or any(t.startswith(w) for t in toks for w in words):
                more.append(n["id"])
        return base + [i for i in more if i not in base]
    if arm.startswith("A6"):
        base = exclusion("A1x", fact, by_src, all_notes)
        return base + [i for i in ids(extra, by_src) if i not in base]
    raise ValueError(arm)


def questions(set_name):
    """(fact, index, question) for a set."""
    out = []
    if set_name == "gate":
        for k in POCKET:
            for i, g in enumerate(FACTS[k]["gate"]):
                out.append((k, i, g["q"]))
    elif set_name == "pf":
        for k, f in FACTS.items():
            for i, q in enumerate(f["pf"]):
                out.append((k, i, q))
    elif set_name == "trained":
        for k in POCKET:
            for i, q in enumerate(FACTS[k]["trained"]):
                out.append((k, i, q))
    elif set_name == "nb":  # optional leak check: run5's neighbour probes, distinct, no fact of their own
        for i, q in enumerate(json.load(open(f"{E}/inputs/neighbours.json"))["questions"]):
            out.append((None, i, q["q"]))
    elif set_name == "q56":
        for i, q in enumerate(SETS["q56"]):
            out.append((q["fact"], i, q["text"]))
    else:
        raise ValueError(set_name)
    return out


def jobs_of(arm, set_name):
    """Every (arm label, fact, index, question, exclusion keys extra) an arm asks of a set."""
    out = []
    for fact, i, q in questions(set_name):
        if arm == "A6":
            for extra in A6.get(fact, []):
                out.append((f"A6-{'+'.join(extra)}", fact, i, q, extra))
        elif arm in ("A2", "A4", "A1y") and fact is None:
            continue
        else:
            out.append((arm, fact, i, q, None))
    return out


def block_key(q, excl):
    return q + "\x00" + ",".join(str(i) for i in excl)


def load_blocks(label):
    path = f"{E}/blocks-{label}.jsonl"
    out = {}
    if os.path.exists(path):
        for l in open(path):
            r = json.loads(l)
            out[block_key(r["q"], r["exclude"])] = r
    return out


def blocks(label, arms_sets):
    all_notes, by_src = note_maps()
    have = load_blocks(label)
    path = f"{E}/blocks-{label}.jsonl"
    n_new = 0
    with open(path, "a") as f:
        for set_name, arms in arms_sets:
            for arm in arms:
                if arm in ("A0", "A3"):
                    continue
                for alabel, fact, i, q, extra in jobs_of(arm, set_name):
                    excl = exclusion(alabel if not alabel.startswith("A6") else "A6", fact, by_src, all_notes, extra) if not (arm in ("A1x", "A5") and fact is None) \
                        else (ids(W5, by_src) if arm == "A1x" else ids(POCKET, by_src))
                    key = block_key(q, excl)
                    if key in have:
                        continue
                    body = {"messages": [{"role": "user", "content": q}], "notes": True, "notes_write": False, "notes_echo": True,
                            "notes_value_hop": False,  # the snapshot's own hop, as the alone check's store.retrieve() looks up
                            "notes_exclude": excl, "temperature": 0, "max_tokens": 1, "chat_template_kwargs": {"enable_thinking": False}}
                    st, r = call(PORT[label], "POST", "/v1/chat/completions", body)
                    if st != 200:
                        raise SystemExit(f"block {q!r} {excl}: {st} {r}")
                    nb = r.get("notes", {})
                    blk = nb.get("block", "")
                    rec = {"q": q, "exclude": excl, "block": blk, "block_sha256": hashlib.sha256(blk.encode()).hexdigest(),
                           "ids": nb.get("ids"), "tokens": nb.get("tokens"), "prompt_tokens": r["usage"]["prompt_tokens"]}
                    have[key] = rec
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_new += 1
    log(f"blocks {label}: {n_new} new, {len(have)} in all")


# ---------------------------------------------------------------- asks

def done_lines(label):
    out = {}
    for l in open(f"{E}/logs/server-{label}.log", errors="replace"):
        m = re.search(r"done (chatcmpl-[0-9a-f]+) prompt=(\d+) .*?tokens=(\d+) sha=([0-9a-f]{12}) finish=(\w+)", l)
        if m:
            out[m.group(1)] = {"prompt": int(m.group(2)), "tokens": int(m.group(3)), "sha": m.group(4), "finish": m.group(5)}
    return out


def ask(label, spec):
    all_notes, by_src = note_maps()
    bl = load_blocks("taught")  # blocks are built once, on the taught server (the untaught server's are checked against them)
    a3 = open(f"{E}/inputs/a3-block.txt").read() if os.path.exists(f"{E}/inputs/a3-block.txt") else None
    os.makedirs(f"{E}/replies/{label}", exist_ok=True)
    for part in spec.split(";"):
        set_name, arms = part.split(":")
        for arm in arms.split(","):
            path = f"{E}/replies/{label}/{set_name}-{arm}.jsonl"
            if os.path.exists(path):
                raise SystemExit(f"{path} exists; not asking again")
            recs = []
            t0 = time.time()
            for alabel, fact, i, q, extra in jobs_of(arm, set_name):
                if arm == "A0":
                    system, excl = None, None
                elif arm == "A3":
                    if a3 is None:
                        raise SystemExit("inputs/a3-block.txt missing")
                    system, excl = a3, None
                else:
                    excl = exclusion(alabel if not alabel.startswith("A6") else "A6", fact, by_src, all_notes, extra) if not (arm in ("A1x", "A5") and fact is None) \
                        else (ids(W5, by_src) if arm == "A1x" else ids(POCKET, by_src))
                    b = bl.get(block_key(q, excl))
                    if b is None:
                        raise SystemExit(f"no block for {alabel} {q!r} {excl}")
                    system = b["block"] or None
                msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": q}]
                body = {"messages": msgs, "temperature": 0, "max_tokens": MAX[set_name], "chat_template_kwargs": {"enable_thinking": False}}
                st, r = call(PORT[label], "POST", "/v1/chat/completions", body)
                if st != 200:
                    raise SystemExit(f"{alabel} {q!r}: {st} {r}")
                ch = r["choices"][0]
                recs.append({"model": label, "set": set_name, "arm": alabel, "fact": fact, "i": i, "q": q, "exclude": excl,
                             "system_sha256": hashlib.sha256(system.encode()).hexdigest() if system else None,
                             "reply": ch["message"].get("content") or "", "finish": ch.get("finish_reason"),
                             "prompt_tokens": r["usage"]["prompt_tokens"], "completion_tokens": r["usage"]["completion_tokens"], "id": r["id"]})
            time.sleep(0.5)
            dl = done_lines(label)
            for rec in recs:
                d = dl.get(rec["id"])
                rec["done"] = d
            with open(path, "w") as f:
                for rec in recs:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            log(f"ask {label} {set_name}-{arm}: {len(recs)} asks in {time.time() - t0:.0f} s; done lines found {sum(1 for r in recs if r['done'])}")


def main():
    cmd, label = sys.argv[1], sys.argv[2]
    if cmd == "start":
        start(label)
    elif cmd == "stop":
        stop(label)
    elif cmd == "notes":
        notes(label)
    elif cmd == "blocks":
        spec = [(p.split(":")[0], p.split(":")[1].split(",")) for p in sys.argv[3].split(";")]
        blocks(label, spec)
    elif cmd == "ask":
        ask(label, sys.argv[3])
    else:
        raise SystemExit(cmd)


if __name__ == "__main__":
    main()
