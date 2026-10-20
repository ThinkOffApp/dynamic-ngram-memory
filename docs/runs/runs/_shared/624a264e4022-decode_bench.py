#!/usr/bin/env python3
"""Decode-speed harness for a running TensorFold OpenAI-compatible server (Living Weights overhead).

  python3 decode_bench.py --base http://127.0.0.1:PORT/v1 --label NAME [--out FILE.json]
      [--warmup 2] [--runs 24] [--thinking off|on|both] [--draft server|off] [--max-tokens 256]
      [--expect-drafts on|off] [--model ID] [--api-key KEY] [--server-cmd TEXT] [--note TEXT]
      [--gap SECONDS] [--timeout SECONDS] [--ready-timeout SECONDS] [--allow-learning] [--dry-run]

It starts nothing and stops nothing: it talks to one server that is already running.

Per thinking mode: W warm-up requests, then R measured requests, cycling through the 8 fixed prompts below
(request i asks prompt i % 8), one at a time, streaming, temperature 0, max_tokens 256 (default).
Thinking is set per request with chat_template_kwargs {"enable_thinking": false|true}.

Numbers, per request (no request is dropped; a failed one stays in the rows with its error):
  completion_tokens  from the server's usage.completion_tokens (never counted from stream chunks)
  t_first            arrival of the first chunk whose delta carries text (content / reasoning / tool calls)
  t_end              arrival of the chunk carrying finish_reason (else [DONE], else end of stream)
  decode_tps         completion_tokens / (t_end - t_first)            <- the headline number
  decode_tps_nm1     (completion_tokens - 1) / (t_end - t_first)      <- same basis as the server's own
  e2e_tps            completion_tokens / (t_end - t_send)
  server_tps         the reply's own tensorfold.tokens_per_second (chat.zig: (n-1) / (finished - prefilled))
  agree              decode_tps_nm1 / server_tps (the two instruments; far from 1.0 = stream holdback or clock skew)
Also kept: drafts on/off as the server reports it, speculative rounds/drafted/accepted, token_sha, ttft.

Server info: GET /health, /v1/models, /metrics (tensorfold: counters, before and after, diffed) and
/v1/slide/graph reduced to counts (fact states, pocket neurons and per-layer used); fact texts are not stored.
Host info: a process snapshot (other GPU users), plus pmset (macOS) or nvidia-smi (Linux), read-only.

--dry-run: no network, no subprocess. Prints the plan and the request bodies, and runs an offline self-test of
the stream parser and the statistics on a synthetic stream. Exit 0 when the self-test passes.

Python 3.8+ standard library only.
"""
import argparse
import datetime
import json
import os
import platform
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

SCHEMA = "tf-decode-bench/1"

# Neutral on purpose: nothing about colours, eyes, names, people, pets or the user, so no taught fact is asked.
# Each asks for detail, so a 256-token reply usually ends on length and the decode lengths stay comparable.
PROMPTS = [
    ("p1", "Explain in detail how a bicycle's gears change the effort needed to pedal up a steep hill."),
    ("p2", "Write a Python function that merges two sorted lists into one sorted list, then explain its time complexity step by step."),
    ("p3", "Describe the water cycle in detail, stage by stage, from evaporation to groundwater."),
    ("p4", "List and explain the main steps for baking a simple loaf of bread at home, from mixing to cooling."),
    ("p5", "Explain the difference between TCP and UDP for a beginner, with examples of when each is used."),
    ("p6", "Explain how a household refrigerator moves heat out of its interior, covering each part of the cycle."),
    ("p7", "Show step by step how to solve a quadratic equation by completing the square, with a worked example."),
    ("p8", "Outline a detailed four-week plan for learning to touch type, with daily practice goals."),
]

METRIC_FIELDS = ["decode_tps", "decode_tps_nm1", "server_tps", "agree", "e2e_tps", "ttft_s", "decode_s",
                 "completion_tokens", "spec_tokens_per_round"]
GPU_SUSPECT = re.compile(r"tensorfold|llama|ollama|mlx|vllm|sglang|lm ?studio|lmstudio|exo|torch|comfy|whisper|"
                         r"stable|diffusion|blender|ffmpeg|handbrake|xcode|simulator", re.I)


# ---------------------------------------------------------------- statistics

def percentile(xs, q):
    """Linear interpolation between closest ranks (numpy's default), q in [0, 100]."""
    v = sorted(x for x in xs if x is not None)
    if not v:
        return None
    if len(v) == 1:
        return v[0]
    pos = (len(v) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


def describe(xs):
    v = [x for x in xs if x is not None]
    if not v:
        return {"n": 0}
    return {"n": len(v), "median": percentile(v, 50), "p10": percentile(v, 10), "p90": percentile(v, 90),
            "min": min(v), "max": max(v), "mean": sum(v) / len(v)}


def summarize(rows):
    """Per thinking mode, over the measured rows. Nothing is trimmed."""
    out = {}
    for mode in sorted({r["thinking"] for r in rows}):
        rs = [r for r in rows if r["thinking"] == mode and r["phase"] == "measured"]
        ok = [r for r in rs if r.get("ok")]
        s = {"requests": len(rs), "ok": len(ok), "failed": len(rs) - len(ok),
             "failures": [{"i": r["i"], "prompt": r["prompt_id"], "error": r.get("error")} for r in rs if not r.get("ok")]}
        for f in METRIC_FIELDS:
            s[f] = describe([r.get(f) for r in ok])
        tok = sum(r["completion_tokens"] for r in ok if r.get("decode_s"))
        sec = sum(r["decode_s"] for r in ok if r.get("decode_s"))
        s["aggregate_decode_tps"] = tok / sec if sec > 0 else None
        s["finish_reasons"] = count([r.get("finish_reason") for r in ok])
        s["drafts_reported"] = count([r.get("drafts") for r in ok])
        s["sampling_reported"] = count([r.get("sampling") for r in ok])
        dr = sum(r.get("spec_drafted") or 0 for r in ok)
        ac = sum(r.get("spec_accepted") or 0 for r in ok)
        s["spec_acceptance"] = ac / dr if dr else None
        per = {}
        for pid, _ in PROMPTS:
            pr = [r for r in ok if r["prompt_id"] == pid]
            if pr:
                per[pid] = {"n": len(pr), "decode_tps_median": percentile([r.get("decode_tps") for r in pr], 50),
                            "completion_tokens": sorted({r["completion_tokens"] for r in pr}),
                            "token_shas": sorted({r.get("token_sha") for r in pr if r.get("token_sha")})}
        s["per_prompt"] = per
        s["deterministic"] = all(len(p["token_shas"]) <= 1 for p in per.values())
        out[mode] = s
    return out


def count(xs):
    d = {}
    for x in xs:
        k = json.dumps(x) if not isinstance(x, str) else x
        d[k] = d.get(k, 0) + 1
    return d


# ---------------------------------------------------------------- stream parsing

def has_text(delta):
    if not isinstance(delta, dict):
        return False
    for k, v in delta.items():
        if k == "role":
            continue
        if isinstance(v, str) and v:
            return True
        if isinstance(v, list) and v:
            return True
    return False


def parse_stream(timed_lines):
    """timed_lines: iterable of (t_seconds, raw_line_bytes). Returns what the stream said and when."""
    st = {"t_first": None, "t_finish": None, "t_usage": None, "t_done": None, "t_last": None, "usage": None,
          "finish_reason": None, "tensorfold": None, "speculative": None, "chunks": 0, "text_chunks": 0,
          "first_kind": None, "stream_error": None}
    for t, raw in timed_lines:
        st["t_last"] = t
        line = raw.decode("utf-8", "replace").strip() if isinstance(raw, (bytes, bytearray)) else str(raw).strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            st["t_done"] = t
            break
        try:
            ev = json.loads(data)
        except ValueError:
            st["stream_error"] = "unparsed event: " + data[:200]
            continue
        if not isinstance(ev, dict):
            continue
        if "error" in ev:
            st["stream_error"] = json.dumps(ev["error"])[:500]
        st["chunks"] += 1
        for c in ev.get("choices") or []:
            d = c.get("delta") or {}
            if has_text(d):
                st["text_chunks"] += 1
                if st["t_first"] is None:
                    st["t_first"] = t
                    st["first_kind"] = sorted(k for k in d if k != "role" and d[k])
            if c.get("finish_reason") and st["t_finish"] is None:
                st["t_finish"] = t
                st["finish_reason"] = c.get("finish_reason")
        if ev.get("usage"):
            st["usage"] = ev["usage"]
            st["t_usage"] = t
        if ev.get("tensorfold") is not None:
            st["tensorfold"] = ev["tensorfold"]
        if ev.get("speculative") is not None:
            st["speculative"] = ev["speculative"]
    return st


def row_numbers(st, t_send):
    """The row's derived numbers from a parsed stream; None where a number does not exist."""
    r = {}
    u = st.get("usage") or {}
    ct = u.get("completion_tokens")
    r["completion_tokens"] = ct
    r["prompt_tokens"] = u.get("prompt_tokens")
    r["reasoning_tokens"] = (u.get("completion_tokens_details") or {}).get("reasoning_tokens")
    r["cached_tokens"] = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
    r["finish_reason"] = st.get("finish_reason")
    t_end = st.get("t_finish") or st.get("t_done") or st.get("t_last")
    tf = st.get("t_first")
    r["ttft_s"] = tf - t_send if tf is not None else None
    r["decode_s"] = t_end - tf if (tf is not None and t_end is not None) else None
    r["e2e_s"] = t_end - t_send if t_end is not None else None
    ok_dec = isinstance(ct, int) and r["decode_s"] and r["decode_s"] > 0
    r["decode_tps"] = ct / r["decode_s"] if ok_dec else None
    r["decode_tps_nm1"] = (ct - 1) / r["decode_s"] if ok_dec and ct > 1 else None
    r["e2e_tps"] = ct / r["e2e_s"] if isinstance(ct, int) and r["e2e_s"] else None
    tfx = st.get("tensorfold") or {}
    r["server_tps"] = tfx.get("tokens_per_second") or None
    r["server_decode_s"] = tfx.get("seconds")
    r["server_prefill_s"] = tfx.get("prefill_seconds")
    r["server_ttft_s"] = tfx.get("time_to_first_token")
    r["drafts"] = tfx.get("drafts")
    r["sampling"] = tfx.get("sampling")
    r["enable_thinking"] = tfx.get("enable_thinking")
    r["engine"] = tfx.get("engine")
    r["token_sha"] = tfx.get("token_sha")
    sp = st.get("speculative") or {}
    r["spec_rounds"] = sp.get("rounds")
    r["spec_drafted"] = sp.get("drafted")
    r["spec_accepted"] = sp.get("accepted")
    r["spec_tokens_per_round"] = sp.get("tokens_per_round")
    r["agree"] = r["decode_tps_nm1"] / r["server_tps"] if (r["decode_tps_nm1"] and r["server_tps"]) else None
    r["chunks"] = st.get("chunks")
    r["text_chunks"] = st.get("text_chunks")
    r["first_kind"] = st.get("first_kind")
    return r


# ---------------------------------------------------------------- HTTP

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # never route localhost through a proxy


def headers(api_key):
    h = {"Content-Type": "application/json", "Accept": "text/event-stream, application/json"}
    if api_key:
        h["Authorization"] = "Bearer " + api_key
    return h


def http_get(url, api_key, timeout=15):
    req = urllib.request.Request(url, headers=headers(api_key), method="GET")
    try:
        with OPENER.open(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", "replace")
            status = resp.status
    except urllib.error.HTTPError as e:
        return {"status": e.code, "error": e.read().decode("utf-8", "replace")[:300]}
    except Exception as e:  # noqa: BLE001 (any failure is data here)
        return {"status": None, "error": "%s: %s" % (type(e).__name__, e)}
    try:
        return {"status": status, "json": json.loads(body)}
    except ValueError:
        return {"status": status, "text": body}


def parse_metrics(text):
    """Unlabelled tensorfold: samples from Prometheus text."""
    out = {}
    for line in (text or "").splitlines():
        if line.startswith("#") or "{" in line:
            continue
        m = re.match(r"^(tensorfold:[A-Za-z0-9_:]+)\s+([-+0-9.eE]+|NaN|\+Inf|-Inf)\s*$", line)
        if m:
            try:
                out[m.group(1)] = float(m.group(2))
            except ValueError:
                pass
    return out


def graph_summary(g):
    """The slide graph reduced to counts: no fact text leaves the server."""
    if not isinstance(g, dict):
        return None
    facts = g.get("facts") or []
    s = {"facts": len(facts), "states": count([f.get("state") for f in facts if isinstance(f, dict)]),
         "fact_neurons": sum(len(f.get("neurons") or []) for f in facts if isinstance(f, dict))}
    m = g.get("model")
    if isinstance(m, dict):
        p = m.get("pocket") or {}
        used = [u for u in (p.get("used") or []) if isinstance(u, int)]
        s["pocket_neurons"] = p.get("neurons")
        s["moe_layers"] = len(p.get("layers") or [])
        s["pocket_used_per_layer"] = used
        s["pocket_layers_in_use"] = sum(1 for u in used if u > 0)
        s["pocket_used_total"] = sum(used)
        s["pocket_used_max"] = max(used) if used else 0
        sv = m.get("save")
        s["save_state"] = sv.get("state") if isinstance(sv, dict) else None
    return s


def server_info(root, base, api_key):
    info = {"at": now_local()}
    h = http_get(root + "/health", api_key)
    info["health"] = h.get("json") if "json" in h else h
    mo = http_get(base + "/models", api_key)
    info["models"] = mo.get("json") if "json" in mo else mo
    me = http_get(root + "/metrics", api_key)
    info["metrics"] = parse_metrics(me.get("text")) if me.get("status") == 200 else {"error": me.get("error"), "status": me.get("status")}
    gr = http_get(base + "/slide/graph", api_key)
    info["slide_graph"] = graph_summary(gr.get("json")) if "json" in gr else {"status": gr.get("status"), "error": gr.get("error")}
    return info


def metrics_delta(before, after):
    if not isinstance(before, dict) or not isinstance(after, dict) or "error" in before or "error" in after:
        return None
    return {k: after[k] - before.get(k, 0.0) for k in after if k.endswith("_total") and isinstance(after[k], float)}


def wait_ready(root, api_key, limit, log):
    t0 = time.time()
    while True:
        h = http_get(root + "/health", api_key, timeout=10)
        j = h.get("json")
        if isinstance(j, dict) and j.get("status") == "ok" and not j.get("warming"):
            return True, j
        if time.time() - t0 > limit:
            return False, h
        log("waiting for the server: %s" % (json.dumps(j)[:160] if j else h.get("error")))
        time.sleep(2)


def one_request(base, api_key, body, timeout):
    data = json.dumps(body).encode()
    req = urllib.request.Request(base + "/chat/completions", data=data, headers=headers(api_key), method="POST")
    clock = time.perf_counter
    t_send = clock()
    try:
        resp = OPENER.open(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        return {"ok": False, "http_status": e.code, "error": e.read().decode("utf-8", "replace")[:500]}, t_send
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "http_status": None, "error": "%s: %s" % (type(e).__name__, e)}, t_send

    def lines():
        while True:
            raw = resp.readline()
            if not raw:
                return
            yield clock(), raw

    try:
        st = parse_stream(lines())
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "http_status": resp.status, "error": "stream: %s: %s" % (type(e).__name__, e)}, t_send
    finally:
        resp.close()
    r = row_numbers(st, t_send)
    r["http_status"] = resp.status
    r["ok"] = isinstance(r.get("completion_tokens"), int) and r.get("decode_tps") is not None and not st.get("stream_error")
    if not r["ok"]:
        r["error"] = st.get("stream_error") or ("no usage.completion_tokens in the stream" if r.get("completion_tokens") is None
                                                else "no decode interval (first text chunk or end missing)")
    return r, t_send


# ---------------------------------------------------------------- host snapshot (read-only)

def run_cmd(argv):
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=15)
        return p.stdout.strip() if p.returncode == 0 else ("rc=%d %s" % (p.returncode, p.stderr.strip()[:300]))
    except FileNotFoundError:
        return None
    except Exception as e:  # noqa: BLE001
        return "%s: %s" % (type(e).__name__, e)


def host_snapshot():
    snap = {"at": now_local()}
    ps = run_cmd(["ps", "-Ao", "pid=,pcpu=,rss=,comm="]) or ""
    procs = []
    for line in ps.splitlines():
        parts = line.split(None, 3)
        if len(parts) == 4:
            try:
                procs.append({"pid": int(parts[0]), "cpu": float(parts[1]), "rss_mb": int(parts[2]) // 1024, "comm": parts[3]})
            except ValueError:
                pass
    me = os.getpid()
    snap["top_cpu"] = sorted(procs, key=lambda p: -p["cpu"])[:12]
    snap["gpu_suspects"] = [p for p in procs if GPU_SUSPECT.search(os.path.basename(p["comm"])) and p["pid"] != me]
    snap["tensorfold_processes"] = sum(1 for p in procs if "tensorfold" in os.path.basename(p["comm"]))
    if sys.platform == "darwin":
        snap["pmset_batt"] = run_cmd(["pmset", "-g", "batt"])
        snap["pmset_therm"] = run_cmd(["pmset", "-g", "therm"])
    else:
        snap["nvidia_smi_apps"] = run_cmd(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"])
        snap["nvidia_smi_gpu"] = run_cmd(["nvidia-smi", "--query-gpu=name,temperature.gpu,power.draw,clocks.sm,clocks.mem,utilization.gpu,memory.used", "--format=csv,noheader"])
    return snap


# ---------------------------------------------------------------- the run

def now_local():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def request_body(model, prompt, thinking, args):
    b = {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": True,
         "stream_options": {"include_usage": True}, "temperature": 0, "max_tokens": args.max_tokens,
         "chat_template_kwargs": {"enable_thinking": thinking == "on"}}
    if args.draft == "off":
        b["draft"] = False  # chat.zig: a request can only turn drafting off, never on
    if args.ignore_eos:
        b["ignore_eos"] = True
    return b


def modes(args):
    return ["off", "on"] if args.thinking == "both" else [args.thinking]


def schedule(args):
    out = []
    for mode in modes(args):
        for phase, n in (("warmup", args.warmup), ("measured", args.runs)):
            for i in range(n):
                pid, text = PROMPTS[i % len(PROMPTS)]
                out.append((mode, phase, i, pid, text))
    return out


def fmt(x, spec=".1f"):
    return "-" if x is None else format(x, spec)


def print_summary(res, log):
    log("")
    log("label %s | model %s | host %s (%s) | base %s | drafts %s | max_tokens %d"
        % (res["label"], res.get("model"), res["client"]["hostname"], res["client"]["platform"], res["args"]["base"],
           res["args"]["draft"], res["args"]["max_tokens"]))
    log("%-8s %4s %10s %8s %8s %10s %8s %8s %8s %6s %7s" % ("thinking", "n", "decode tps", "p10", "p90", "server tps",
                                                              "agree", "e2e tps", "ttft s", "fail", "accept"))
    for mode, s in res["summary"].items():
        d, sv, ag, e2, tt = s["decode_tps"], s["server_tps"], s["agree"], s["e2e_tps"], s["ttft_s"]
        log("%-8s %4d %10s %8s %8s %10s %8s %8s %8s %6d %7s" % (
            mode, s["ok"], fmt(d.get("median")), fmt(d.get("p10")), fmt(d.get("p90")), fmt(sv.get("median")),
            fmt(ag.get("median"), ".3f"), fmt(e2.get("median")), fmt(tt.get("median"), ".3f"), s["failed"],
            fmt(s.get("spec_acceptance"), ".2f")))
    for w in res["checks"]:
        log("CHECK: " + w)


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
        f.write("\n")
    os.replace(tmp, path)


def checks_for(res, args):
    c = []
    rows = [r for r in res["rows"] if r["phase"] == "measured"]
    if any(not r.get("ok") for r in rows):
        c.append("%d measured request(s) failed; they are in rows with their error" % sum(1 for r in rows if not r.get("ok")))
    if args.expect_drafts:
        want = args.expect_drafts == "on"
        bad = [r["i"] for r in rows if r.get("ok") and r.get("drafts") is not want]
        if bad:
            c.append("drafts reported != --expect-drafts %s on %d request(s)" % (args.expect_drafts, len(bad)))
    if any(r.get("sampling") not in (None, "greedy") for r in rows if r.get("ok")):
        c.append("a reply reports sampling other than greedy")
    for mode, s in res["summary"].items():
        if not s.get("deterministic"):
            c.append("thinking %s: one prompt gave different token_sha across runs (not greedy-deterministic)" % mode)
        ag = s["agree"].get("median")
        if ag is not None and not (0.9 <= ag <= 1.1):
            c.append("thinking %s: client/server decode rates disagree (median ratio %.3f): check stream holdback" % (mode, ag))
        fr = s.get("finish_reasons", {})
        short = sum(v for k, v in fr.items() if k != "length")
        if short:
            c.append("thinking %s: %d measured repl(ies) ended before max_tokens (finish != length): lengths differ" % (mode, short))
    hs = res.get("host_before") or {}
    if hs.get("tensorfold_processes", 0) > 1:
        c.append("%d tensorfold processes on this host: another server may be using the GPU" % hs["tensorfold_processes"])
    return c


def self_test():
    """Offline: a synthetic stream through the parser and the statistics."""
    t0 = 100.0
    ev = []
    ev.append((t0 + 0.010, b'data: {"choices":[{"index":0,"delta":{"role":"assistant"}}]}\n'))
    ev.append((t0 + 0.011, b"\n"))
    for k in range(10):
        ev.append((t0 + 0.100 + 0.005 * k, ('data: {"choices":[{"index":0,"delta":{"content":"w%d"}}]}\n' % k).encode()))
    ev.append((t0 + 0.150, b'data: {"choices":[{"index":0,"delta":{"content":""},"finish_reason":"length"}],'
                           b'"tensorfold":{"tokens_per_second":180.0,"seconds":0.05,"drafts":true,"sampling":"greedy",'
                           b'"token_sha":"abc"},"speculative":{"rounds":5,"drafted":8,"accepted":5,"tokens_per_round":2.0}}\n'))
    ev.append((t0 + 0.151, b'data: {"choices":[],"usage":{"prompt_tokens":20,"completion_tokens":10,"total_tokens":30}}\n'))
    ev.append((t0 + 0.152, b"data: [DONE]\n"))
    st = parse_stream(ev)
    r = row_numbers(st, t0)
    fails = []

    def expect(name, got, want, tol=1e-9):
        if got is None or abs(got - want) > tol:
            fails.append("%s: got %r want %r" % (name, got, want))

    expect("ttft", r["ttft_s"], 0.100)
    expect("decode_s", r["decode_s"], 0.050)
    expect("decode_tps", r["decode_tps"], 200.0, 1e-6)
    expect("decode_tps_nm1", r["decode_tps_nm1"], 180.0, 1e-6)
    expect("agree", r["agree"], 1.0, 1e-6)
    expect("e2e_tps", r["e2e_tps"], 10 / 0.150, 1e-6)
    if r["completion_tokens"] != 10:
        fails.append("completion_tokens from usage: %r" % r["completion_tokens"])
    if st["text_chunks"] != 10 or st["first_kind"] != ["content"]:
        fails.append("first text chunk detection: %r %r" % (st["text_chunks"], st["first_kind"]))
    expect("p10", percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 10), 1.9)
    expect("p50", percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 50), 5.5)
    expect("p90", percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 90), 9.1)
    # negative control: a stream with no usage must not produce a rate
    st2 = parse_stream([e for e in ev if b"usage" not in e[1]])
    if row_numbers(st2, t0)["decode_tps"] is not None:
        fails.append("a stream without usage produced a decode rate")
    # thinking deltas count as the first token
    st3 = parse_stream([(1.0, b'data: {"choices":[{"delta":{"reasoning_content":"x"}}]}\n')])
    if st3["t_first"] != 1.0:
        fails.append("reasoning_content was not taken as the first token")
    g = graph_summary({"facts": [{"state": "learned", "text": "SECRET", "neurons": [[1, 2]]}],
                       "model": {"pocket": {"neurons": 64, "layers": [1, 3], "used": [0, 2]}, "save": {"state": "saved"}}})
    if "SECRET" in json.dumps(g) or g["pocket_layers_in_use"] != 1 or g["pocket_used_max"] != 2:
        fails.append("graph summary: %r" % g)
    # the summary, the checks and the table over synthetic rows; a disagreeing row must raise a check
    rows = []
    for k in range(16):
        rr = dict(r)
        rr.update({"thinking": "off", "phase": "measured", "i": k, "prompt_id": PROMPTS[k % 8][0], "ok": True})
        rows.append(rr)
    bad = dict(rows[0])
    bad.update({"i": 16, "ok": False, "error": "synthetic"})
    rows.append(bad)
    fake = {"label": "self-test", "model": "M", "client": {"hostname": "h", "platform": "p"},
            "args": {"base": "b", "draft": "server", "max_tokens": 256}, "rows": rows}
    fake["summary"] = summarize(rows)
    ns = argparse.Namespace(expect_drafts="off")
    fake["checks"] = checks_for(fake, ns)
    sm = fake["summary"]["off"]
    if sm["ok"] != 16 or sm["failed"] != 1 or abs(sm["decode_tps"]["median"] - 200.0) > 1e-6:
        fails.append("summary: %r" % {k: sm[k] for k in ("ok", "failed", "decode_tps")})
    if not any("expect-drafts" in c for c in fake["checks"]) or not any("failed" in c for c in fake["checks"]):
        fails.append("checks did not fire: %r" % fake["checks"])
    lines = []
    print_summary(fake, lines.append)
    if not any(ln.startswith("off ") for ln in lines):
        fails.append("summary table missing its row: %r" % lines)
    return fails


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="the server's OpenAI base, e.g. http://127.0.0.1:8080/v1")
    ap.add_argument("--label", required=True, help="condition name, e.g. mac-T5-slide-drafts")
    ap.add_argument("--out", help="output JSON (default: ./decode-<label>-<time>.json); rewritten after every request")
    ap.add_argument("--warmup", type=int, default=2, help="warm-up requests per thinking mode (not summarized)")
    ap.add_argument("--runs", type=int, default=24, help="measured requests per thinking mode, cycling the 8 prompts")
    ap.add_argument("--thinking", choices=["off", "on", "both"], default="off")
    ap.add_argument("--draft", choices=["server", "off"], default="server",
                    help="server: send nothing (the server's own drafting); off: send \"draft\": false per request")
    ap.add_argument("--expect-drafts", choices=["on", "off"], help="flag rows whose reported drafts differ")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--ignore-eos", action="store_true", help="send ignore_eos (off by default: replies end as the model ends them)")
    ap.add_argument("--model", help="model id (default: the first of /v1/models)")
    ap.add_argument("--api-key", default=os.environ.get("TENSORFOLD_API_KEY"), help="default: $TENSORFOLD_API_KEY (never stored)")
    ap.add_argument("--server-cmd", default="", help="the server's command line, stored for the record")
    ap.add_argument("--note", default="")
    ap.add_argument("--gap", type=float, default=0.0, help="seconds between requests")
    ap.add_argument("--timeout", type=float, default=900.0, help="per-request socket timeout, seconds")
    ap.add_argument("--ready-timeout", type=float, default=300.0, help="wait this long for /health warming=false")
    ap.add_argument("--allow-learning", action="store_true", help="measure even while the slide graph shows learning or saving")
    ap.add_argument("--dry-run", action="store_true", help="no network: print the plan, run the offline self-test")
    args = ap.parse_args()

    base = args.base.rstrip("/")
    if not base.endswith("/v1"):
        ap.error("--base must end in /v1 (got %s)" % args.base)
    root = base[: -len("/v1")]
    args.base = base
    if args.warmup < 0 or args.runs < 1 or args.max_tokens < 2:
        ap.error("--warmup >= 0, --runs >= 1, --max-tokens >= 2")
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = args.out or os.path.join(os.getcwd(), "decode-%s-%s.json" % (re.sub(r"[^A-Za-z0-9_.-]+", "_", args.label), stamp))

    def log(msg):
        print(msg, flush=True)

    res = {"schema": SCHEMA, "label": args.label, "created": now_local(), "dry_run": args.dry_run,
           "client": {"hostname": socket.gethostname(), "platform": platform.platform(), "python": platform.python_version()},
           "args": {k: v for k, v in vars(args).items() if k != "api_key"}, "api_key_sent": bool(args.api_key),
           "prompts": [{"id": p, "text": t} for p, t in PROMPTS], "rows": [], "summary": {}, "checks": []}

    if args.dry_run:
        sched = schedule(args)
        log("DRY RUN: nothing is sent. Would talk to %s (GET %s/health, %s/models, %s/metrics, %s/slide/graph)"
            % (base, root, base, root, base))
        log("label %s -> %s" % (args.label, out if args.out else "(no --out: nothing written)"))
        for mode in modes(args):
            log("thinking %s: %d warm-up + %d measured requests; body:" % (mode, args.warmup, args.runs))
            log("  " + json.dumps(request_body(args.model or "MODEL", PROMPTS[0][1], mode, args)))
        per = {}
        for s_ in sched:
            if s_[1] == "measured":
                per[s_[3]] = per.get(s_[3], 0) + 1
        log("measured requests per prompt and mode: %s (order p1..p8, repeated)" % json.dumps({k: v // len(modes(args)) for k, v in per.items()}))
        log("would run %d requests in all" % len(sched))
        fails = self_test()
        res["self_test"] = {"passed": not fails, "failures": fails}
        log("self-test: " + ("PASS" if not fails else "FAIL\n  " + "\n  ".join(fails)))
        if args.out:
            write_json(out, res)
            log("wrote " + out)
        return 0 if not fails else 1

    ok, h = wait_ready(root, args.api_key, args.ready_timeout, log)
    if not ok:
        log("the server at %s is not ready: %s" % (root, json.dumps(h)[:300]))
        return 2
    res["host_before"] = host_snapshot()
    res["server_before"] = server_info(root, base, args.api_key)
    gs = res["server_before"].get("slide_graph") or {}
    busy = (gs.get("states") or {}).get("learning", 0) + (gs.get("states") or {}).get("queued", 0)
    if (busy or gs.get("save_state") == "saving") and not args.allow_learning:
        log("the slide graph shows learning or saving in progress (%s); refusing to measure (--allow-learning overrides)" % gs)
        return 3
    model = args.model
    if not model:
        data = (res["server_before"].get("models") or {}).get("data") or []
        model = data[0].get("id") if data and isinstance(data[0], dict) else None
    if not model:
        log("no model id from /v1/models; pass --model")
        return 2
    res["model"] = model
    log("label %s | model %s | host %s | base %s | pocket %s" % (
        args.label, model, res["client"]["hostname"], base,
        {k: gs.get(k) for k in ("pocket_neurons", "pocket_layers_in_use", "pocket_used_total", "pocket_used_max")}))

    sched = schedule(args)
    try:
        for n, (mode, phase, i, pid, text) in enumerate(sched):
            body = request_body(model, text, mode, args)
            r, _ = one_request(base, args.api_key, body, args.timeout)
            row = {"thinking": mode, "phase": phase, "i": i, "prompt_id": pid, "at": now_local()}
            row.update(r)
            res["rows"].append(row)
            log("[%s] %-8s %2d/%d %s ct=%s decode %s tok/s (server %s) ttft %s s finish=%s drafts=%s%s" % (
                mode, phase, i + 1, args.warmup if phase == "warmup" else args.runs, pid, row.get("completion_tokens"),
                fmt(row.get("decode_tps")), fmt(row.get("server_tps")), fmt(row.get("ttft_s"), ".3f"),
                row.get("finish_reason"), row.get("drafts"), "" if row.get("ok") else "  ERROR " + str(row.get("error"))[:200]))
            res["summary"] = summarize(res["rows"])
            write_json(out, res)
            if args.gap > 0 and n + 1 < len(sched):
                time.sleep(args.gap)
    except KeyboardInterrupt:
        res["interrupted"] = True
        log("interrupted: writing what was measured")
    res["server_after"] = server_info(root, base, args.api_key)
    res["host_after"] = host_snapshot()
    res["metrics_delta"] = metrics_delta(res["server_before"].get("metrics"), res["server_after"].get("metrics"))
    res["summary"] = summarize(res["rows"])
    res["checks"] = checks_for(res, args)
    res["finished"] = now_local()
    write_json(out, res)
    print_summary(res, log)
    log("wrote " + out)
    return 0 if all(r.get("ok") for r in res["rows"] if r["phase"] == "measured") else 1


if __name__ == "__main__":
    sys.exit(main())
