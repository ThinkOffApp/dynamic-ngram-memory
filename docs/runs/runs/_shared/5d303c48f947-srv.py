#!/usr/bin/env python3
"""gb10-a: one 1.0.8 server at a time on port 18190, the folder served in place, sha256 before and after.

usage: srv.py LABEL MODEL(A|T|E) ACTIONS(off,server,bench,bench256)
Stops only the server it started, through its Popen handle (terminate). Never kills anything else.
"""
import hashlib, json, os, subprocess, sys, time, urllib.request, datetime
from concurrent.futures import ThreadPoolExecutor

HOME = os.path.expanduser("~")
OUT = HOME + "/tf-pocket-cuda/raw"
BENCH = HOME + "/tf-pocket-cuda/tools/decode_bench.py"
TOOLS = HOME + "/tf-pocket-cuda/src108/tools"
PY = HOME + "/.local/bin/python3.12"
BIN = HOME + "/tf108/zig-out/native/bin/tensorfold-native"
BIN_SHA = "759114dd74b1ab080b93df181a26f12d54f70a8614ff78928a91c190e626545a"
PORT = 18190
MODELS = {"A": HOME + "/.cache/huggingface/hub/models--TensorFold--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit/snapshots/d9d758fb83953437f7263256b0d96157e2a348b8",
          "T": HOME + "/lw-runs/gb10-a-L7-seed1/models/L7-seed1.main",
          "E": HOME + "/tf-pocket-cuda/models/A-empty64"}
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
        st = os.stat(os.path.join(d, n))
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
    p = subprocess.run(["ps", "-Ao", "pid=,comm="], capture_output=True, text=True)
    return [ln.strip() for ln in p.stdout.splitlines() if ln.strip().split(None, 1)[-1] in ("tensorfold-native", "tensorfold", "tf-cuda-test")]


def smi():
    p = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,power.draw,clocks.sm,utilization.gpu", "--format=csv,noheader"], capture_output=True, text=True)
    return p.stdout.strip()


def main():
    label, model, actions = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
    folder = MODELS[model]
    os.makedirs(OUT, exist_ok=True)
    rec = {"label": label, "model": model, "folder": folder, "actions": actions, "port": PORT, "started": now(), "binary": BIN}
    if gpu_users():
        print("refusing: other GPU users: %r" % gpu_users()); return 3
    if not port_free(PORT):
        print("refusing: port %d busy" % PORT); return 3
    bsha = hashlib.sha256(open(BIN, "rb").read()).hexdigest()
    rec["binary_sha256"] = bsha
    if bsha != BIN_SHA:
        print("refusing: binary sha256 %s" % bsha); return 3
    rec["sha_before"] = sha_dir(folder)
    rec["list_before"] = listing(folder)
    rec["smi_before"] = smi()
    env = {k: v for k, v in os.environ.items() if not k.startswith("TENSORFOLD_")}
    env["TENSORFOLD_NO_LIVE"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    cmd = [BIN, "serve", folder, "--host", "127.0.0.1", "--port", str(PORT), "--no-update-check"]
    rec["server_cmd"] = "TENSORFOLD_NO_LIVE=1 " + " ".join(cmd)
    log_path = os.path.join(OUT, label + ".serve.log")
    logf = open(log_path, "w")
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env, cwd=OUT, stdin=subprocess.DEVNULL)
    rec["server_pid"] = proc.pid
    print("%s: server pid %d on %s" % (label, proc.pid, folder), flush=True)
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
        print("%s: ready in %.1f s; model %s; pocket used %s; %s" % (label, rec["ready_s"], mid, pk.get("used"), smi()), flush=True)
        rec["runs"] = {}
        base = "http://127.0.0.1:%d" % PORT
        for a in actions:
            out = os.path.join(OUT, "%s.%s.json" % (label, a))
            if a in ("off", "server"):
                c = [PY, "-I", BENCH, "--base", base + "/v1", "--label", "%s-%s" % (label, a), "--thinking", "off",
                     "--warmup", "2", "--runs", "16", "--max-tokens", "256", "--draft", a,
                     "--expect-drafts", "off" if a == "off" else "on", "--server-cmd", rec["server_cmd"], "--out", out]
            elif a == "bench":
                c = [PY, TOOLS + "/bench_openai.py", base, mid or "local-model", "--output", out]
            elif a == "bench256":
                c = [PY, TOOLS + "/bench_openai.py", base, mid or "local-model", "--tokens", "256", "--reps", "3", "--temperatures", "0", "--output", out]
            else:
                raise RuntimeError("unknown action " + a)
            t1 = time.time()
            p = subprocess.run(c, env=env, capture_output=True, text=True, cwd=TOOLS)
            tail = "\n".join((p.stdout + p.stderr).splitlines()[-12:])
            print("%s %s rc=%d %.0fs %s\n%s" % (label, a, p.returncode, time.time() - t1, smi(), tail), flush=True)
            rec["runs"][a] = {"rc": p.returncode, "out": out, "tail": tail, "seconds": round(time.time() - t1, 1), "smi_after": smi()}
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
        print("%s: stopped exit=%s bytes_unchanged=%s listing_unchanged=%s" % (label, proc.returncode, rec["bytes_unchanged"], rec["listing_unchanged"]), flush=True)
        with open(os.path.join(OUT, label + ".run.json"), "w") as f:
            json.dump(rec, f, indent=1)


if __name__ == "__main__":
    sys.exit(main() or 0)
