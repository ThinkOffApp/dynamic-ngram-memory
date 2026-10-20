#!/usr/bin/env python3
"""gb10-a: one server at a time on port 18190, before (0efbd447 build) or after (pocket-cuda build), folder served in place.

usage: srv2.py SERIES LABEL BUILD(before|after) MODEL(A|T|E) ACTIONS
ACTIONS, comma list, run in this order as given:
  openai     tools/bench_openai.py defaults (drafted)
  openai256  tools/bench_openai.py --tokens 256 --reps 3 --temperatures 0 (CONTRIBUTING.md's form)
  off        decode_bench.py, "draft": false, thinking off, 2 warm-up + 16 measured requests of 256 tokens
  conc       tools/bench_concurrent.py --alone --serial (defaults)
  conc14     tools/bench_concurrent.py --tokens 256 --reps 3 --temperatures 0 --levels 1,4 --alone --serial
  facts      the L7 subset (40 questions), greedy, thinking off (160) and on (1500), drafted and "draft": false
The server gets a fresh empty HOME of its own, so it times its own draft-window cost table at startup.
sha256 of every *.safetensors and config.json before and after. Stops only its own server (Popen.terminate).
"""
import hashlib, json, os, subprocess, sys, time, urllib.request, datetime
from concurrent.futures import ThreadPoolExecutor

HOME = os.path.expanduser("~")
H = HOME + "/tf-pocket-cuda"
BENCH = H + "/tools/decode_bench.py"
TOOLS = H + "/src108/tools"
PY = HOME + "/.local/bin/python3.12"
FACTS = H + "/tools/v/facts"
BINS = {"before": (H + "/before/zig-out/native/bin/tensorfold-native", "b0a5a4f312a58dda536e5f5c97699e0ebf211e08319cceca7555e8560b9ba333"),
        "after": (H + "/src/zig-out/native/bin/tensorfold-native", "4f512013d0181a1a68fcf0cee4dd2243f6997914429e56ac8ae2b35f616ab30e")}
PORT = 18190
MODELS = {"A": HOME + "/.cache/huggingface/hub/models--TensorFold--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit/snapshots/d9d758fb83953437f7263256b0d96157e2a348b8",
          "T": HOME + "/lw-runs/gb10-a-L7-seed1/models/L7-seed1.main",
          "E": H + "/models/A-empty64"}
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def sha_dir(d):
    names = sorted(n for n in os.listdir(d) if n.endswith(".safetensors") or n == "config.json")

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


def get(url, timeout=10):
    try:
        with OPENER.open(urllib.request.Request(url), timeout=timeout) as r:
            return json.loads(r.read().decode())
    except Exception as e:  # noqa: BLE001
        return {"error": "%s: %s" % (type(e).__name__, e)}


def port_free(port):
    p = subprocess.run(["ss", "-ltnH", "sport = :%d" % port], capture_output=True, text=True)
    return p.stdout.strip() == ""


def gpu_users():
    p = subprocess.run(["ps", "-Ao", "pid=,comm=,args="], capture_output=True, text=True)
    out = []
    for ln in p.stdout.splitlines():
        parts = ln.strip().split(None, 2)
        if len(parts) < 2:
            continue
        comm, args = parts[1], parts[2] if len(parts) > 2 else ""
        if comm in ("tensorfold-nati", "tensorfold-native", "tensorfold", "tf-cuda-test", "nsys") or \
                ("dedup_identical_copies" in args and "python" in args.split(None, 1)[0]):
            out.append(ln.strip()[:160])
    return out


def smi():
    p = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,power.draw,clocks.sm,utilization.gpu", "--format=csv,noheader"], capture_output=True, text=True)
    return p.stdout.strip()


def ask(q, thinking, draft_off):
    body = {"messages": [{"role": "user", "content": q}], "temperature": 0,
            "max_tokens": 1500 if thinking else 160, "chat_template_kwargs": {"enable_thinking": thinking}}
    if draft_off:
        body["draft"] = False
    req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % PORT, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.time()
    try:
        with OPENER.open(req, timeout=900) as r:
            d = json.load(r)
    except Exception as e:  # noqa: BLE001
        return {"error": "%s: %s" % (type(e).__name__, e), "seconds": round(time.time() - t, 2)}
    m = d["choices"][0]["message"]
    tf = d.get("tensorfold") or {}
    u = d.get("usage") or {}
    return {"content": m.get("content") or "", "reasoning": m.get("reasoning_content") or "",
            "finish": d["choices"][0].get("finish_reason"), "completion_tokens": u.get("completion_tokens"),
            "prompt_tokens": u.get("prompt_tokens"), "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "token_sha": tf.get("token_sha"), "drafts": tf.get("drafts"), "server_tps": tf.get("tokens_per_second"),
            "seconds": round(time.time() - t, 2)}


def facts(label):
    qs = json.load(open(FACTS + "/questions.json"))["subset"]
    rows = []
    for draft_off in (False, True):
        for thinking in (False, True):
            for q in qs:
                r = ask(q["text"], thinking, draft_off)
                rows.append({"qid": q["qid"], "thinking": thinking, "draft": "off" if draft_off else "server", "question": q["text"], **r})
            got = [x for x in rows if x["thinking"] == thinking and x["draft"] == ("off" if draft_off else "server")]
            print("%s facts draft=%s thinking=%s: %d replies, %d errors, %d tokens, %.0f s" % (
                label, "off" if draft_off else "server", thinking, len(got), sum(1 for x in got if x.get("error")),
                sum(x.get("completion_tokens") or 0 for x in got), sum(x["seconds"] for x in got)), flush=True)
    return rows


def main():
    series, label, build, model, actions = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5].split(",")
    out_dir = H + "/raw/v/" + series
    os.makedirs(out_dir, exist_ok=True)
    folder = MODELS[model]
    bin_path, bin_sha = BINS[build]
    rec = {"series": series, "label": label, "build": build, "model": model, "folder": folder, "actions": actions,
           "port": PORT, "started": now(), "binary": bin_path}
    if gpu_users():
        print("refusing: other GPU users: %r" % gpu_users()); return 3
    if not port_free(PORT):
        print("refusing: port %d busy" % PORT); return 3
    bsha = hashlib.sha256(open(bin_path, "rb").read()).hexdigest()
    rec["binary_sha256"] = bsha
    if bsha != bin_sha:
        print("refusing: binary sha256 %s" % bsha); return 3
    home = H + "/homes/%s-%s" % (series, label)
    if os.path.exists(home):
        print("refusing: %s exists" % home); return 3
    os.makedirs(home)
    rec["server_home"] = home
    rec["sha_before"] = sha_dir(folder)
    rec["list_before"] = listing(folder)
    rec["smi_before"] = smi()
    env = {k: v for k, v in os.environ.items() if not k.startswith("TENSORFOLD_")}
    env["TENSORFOLD_NO_LIVE"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    senv = dict(env)
    senv["HOME"] = home
    cmd = [bin_path, "serve", folder, "--host", "127.0.0.1", "--port", str(PORT), "--no-update-check"]
    rec["server_cmd"] = "TENSORFOLD_NO_LIVE=1 HOME=<fresh> " + " ".join(cmd)
    log_path = os.path.join(out_dir, label + ".serve.log")
    logf = open(log_path, "w")
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=senv, cwd=home, stdin=subprocess.DEVNULL)
    rec["server_pid"] = proc.pid
    print("%s %s: server pid %d (%s) on %s; %s" % (now(), label, proc.pid, build, folder, smi()), flush=True)
    try:
        while True:
            if proc.poll() is not None:
                raise RuntimeError("server exited %s; see %s" % (proc.returncode, log_path))
            h = get("http://127.0.0.1:%d/health" % PORT, 5)
            if isinstance(h, dict) and h.get("status") == "ok" and not h.get("warming"):
                break
            if time.time() - t0 > 900:
                raise RuntimeError("server not ready in 900 s")
            time.sleep(1)
        rec["ready_s"] = round(time.time() - t0, 2)
        rec["health"] = h
        models = get("http://127.0.0.1:%d/v1/models" % PORT, 10)
        mid = ((models.get("data") or [{}])[0].get("id")) if isinstance(models, dict) else None
        rec["model_id"] = mid
        g = get("http://127.0.0.1:%d/v1/slide/graph" % PORT, 30)
        pk = ((g.get("model") or {}).get("pocket") or {}) if isinstance(g, dict) else {}
        rec["pocket"] = {"neurons": pk.get("neurons"), "layers": pk.get("layers"), "used": pk.get("used"),
                         "error": g.get("error") if isinstance(g, dict) else None}
        print("%s %s: ready in %.1f s; model %s; pocket used %s; %s" % (now(), label, rec["ready_s"], mid, pk.get("used"), smi()), flush=True)
        rec["runs"] = {}
        base = "http://127.0.0.1:%d" % PORT
        for a in actions:
            out = os.path.join(out_dir, "%s.%s.json" % (label, a))
            t1 = time.time()
            if a == "facts":
                rows = facts(label)
                json.dump(rows, open(out, "w"), indent=1)
                rec["runs"][a] = {"rc": 0, "out": out, "seconds": round(time.time() - t1, 1), "errors": sum(1 for x in rows if x.get("error")), "smi_after": smi()}
                continue
            if a == "off":
                c = [PY, "-I", BENCH, "--base", base + "/v1", "--label", "%s-%s" % (label, a), "--thinking", "off",
                     "--warmup", "2", "--runs", "16", "--max-tokens", "256", "--draft", "off",
                     "--expect-drafts", "off", "--server-cmd", rec["server_cmd"], "--out", out]
            elif a == "openai":
                c = [PY, TOOLS + "/bench_openai.py", base, mid or "local-model", "--label", "%s-%s" % (label, a), "--output", out]
            elif a == "openai256":
                c = [PY, TOOLS + "/bench_openai.py", base, mid or "local-model", "--tokens", "256", "--reps", "3", "--temperatures", "0",
                     "--label", "%s-%s" % (label, a), "--output", out]
            elif a == "conc":
                c = [PY, TOOLS + "/bench_concurrent.py", base, mid or "local-model", "--alone", "--serial", "--label", "%s-%s" % (label, a), "--output", out]
            elif a == "conc14":
                c = [PY, TOOLS + "/bench_concurrent.py", base, mid or "local-model", "--tokens", "256", "--reps", "3", "--temperatures", "0",
                     "--levels", "1,4", "--alone", "--serial", "--label", "%s-%s" % (label, a), "--output", out]
            else:
                raise RuntimeError("unknown action " + a)
            p = subprocess.run(c, env=env, capture_output=True, text=True, cwd=TOOLS)
            tail = "\n".join((p.stdout + p.stderr).splitlines()[-12:])
            print("%s %s %s rc=%d %.0fs %s\n%s" % (now(), label, a, p.returncode, time.time() - t1, smi(), tail), flush=True)
            rec["runs"][a] = {"rc": p.returncode, "out": out, "cmd": " ".join(c), "tail": tail, "seconds": round(time.time() - t1, 1), "smi_after": smi()}
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            proc.terminate()
            proc.wait(timeout=240)
        logf.close()
        rec["server_exit"] = proc.returncode
        rec["stopped"] = now()
        rec["sha_after"] = sha_dir(folder)
        rec["list_after"] = listing(folder)
        rec["bytes_unchanged"] = rec["sha_after"] == rec["sha_before"]
        rec["listing_unchanged"] = rec["list_after"] == rec["list_before"]
        rec["home_files"] = sorted(os.path.relpath(os.path.join(dp, f), home) for dp, _, fs in os.walk(home) for f in fs)
        print("%s %s: stopped exit=%s bytes_unchanged=%s listing_unchanged=%s home=%s" % (
            now(), label, proc.returncode, rec["bytes_unchanged"], rec["listing_unchanged"], rec["home_files"]), flush=True)
        with open(os.path.join(out_dir, label + ".run.json"), "w") as f:
            json.dump(rec, f, indent=1)


if __name__ == "__main__":
    sys.exit(main() or 0)
