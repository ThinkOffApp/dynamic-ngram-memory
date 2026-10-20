"""1.0.9 receipt: summary of a server series (srv109.py outputs): integrity, every token_sha before vs after per folder,
bench_concurrent's own checks, negative control, decode speeds, cache states, the first fresh/resumed chat pair, draft costs.
usage: summ_servers109.py RAW_DIR SERIES[,SERIES...] [HOMES_DIR]"""
import glob
import json
import os
import re
import statistics
import struct
import sys
from collections import Counter, defaultdict

V = sys.argv[1]
SERIES = sys.argv[2].split(",")
HOMES = sys.argv[3] if len(sys.argv) > 3 else None


def load(p):
    with open(p) as f:
        return json.load(f)


servers = []
for s in SERIES:
    d = os.path.join(V, s)
    for n in sorted(os.listdir(d)):
        if n.endswith(".run.json"):
            r = load(os.path.join(d, n))
            r["_dir"] = d
            servers.append(r)

print("== integrity (%d servers)" % len(servers))
folder_sha = defaultdict(set)
for r in servers:
    lab = "%s-%s" % (r["series"], r["label"])
    folder_sha[r["model"]].add(json.dumps(r["sha_before"], sort_keys=True))
    rcs = {a: v.get("rc") for a, v in r["runs"].items()}
    print(lab, r["build"], r["model"], "started", r["started"], "stopped", r["stopped"], "ready_s", r["ready_s"],
          "exit", r["server_exit"], "bytes_unchanged", r["bytes_unchanged"], "listing_unchanged", r["listing_unchanged"],
          "nfiles", len(r["sha_before"]), "bin", r["binary_sha256"][:12], "rcs", rcs, "pocket_used_sum",
          sum(r["pocket"]["used"] or []), "smi_before", r.get("smi_before"))
for m, v in folder_sha.items():
    print("folder", m, "distinct sha sets across servers:", len(v))


def reqs(r):
    out, flags, states = {}, [], Counter()
    d, lab = r["_dir"], r["label"]
    for a in ("openai", "openai256"):
        for row in load(os.path.join(d, "%s.%s.json" % (lab, a))):
            for i, ev in enumerate(row["evidence"]):
                out[(a, row["prompt"], row["temperature"], 0, i)] = ev["token_sha"]
                states[(a, ev["cache_state"], ev["cached_tokens"])] += 1
                if not ev["complete"] or ev["tokens"] != row["tokens"]:
                    flags.append((a, row["prompt"], row["temperature"], i, ev))
    for a in ("conc", "conc14"):
        rep = load(os.path.join(d, "%s.%s.json" % (lab, a)))
        for c in rep["cells"]:
            if c["failed"] or c["unmeasured"] or c["alone"]["unequal"] or c["alone"]["failed"] or c["serial"]["unequal"] or c["serial"]["failed"]:
                flags.append((a, c["prompt"], c["temperature"], c["streams"], {k: c[k] for k in ("failed", "unmeasured", "alone", "serial")}))
            for ri, rp in enumerate(c["reps"]):
                for si, ev in enumerate(rp["evidence"]):
                    out[(a, c["prompt"], c["temperature"], c["streams"], ri * 100 + si)] = ev["token_sha"]
                    if not ev["complete"] or ev["tokens"] != rep["tokens"]:
                        flags.append((a, c["prompt"], c["temperature"], c["streams"], ri, si, ev))
                if not all(x is True for x in rp["equal_alone"]):
                    flags.append((a, "equal_alone", rp["equal_alone"]))
    off = load(os.path.join(d, "%s.off.json" % lab))
    for row in off["rows"]:
        if row["phase"] != "measured":
            continue
        out[("off", row["prompt_id"], 0.0, 0, row["i"])] = row["token_sha"]
        if not row.get("ok") or row["finish_reason"] != "length" or row["completion_tokens"] != 256 or row["drafts"]:
            flags.append(("off", row["prompt_id"], row["i"], row.get("ok"), row["finish_reason"], row["drafts"]))
    return out, flags, states


R = {}
for r in servers:
    lab = "%s-%s" % (r["series"], r["label"])
    q, flags, states = reqs(r)
    R[lab] = (r, q)
    print("requests", lab, len(q), "flags", len(flags), flags[:3])
    print("   bench_openai cache states (tool, state, cached tokens): count", sorted(states.items()))

print("\n== bench_concurrent internal checks (alone: concurrent reply == its solo run; serial: solo == draft false)")
for lab, (r, q) in R.items():
    for a in ("conc", "conc14"):
        rep = load(os.path.join(r["_dir"], "%s.%s.json" % (r["label"], a)))
        al = [sum(c["alone"][k] for c in rep["cells"]) for k in ("equal", "unequal", "failed")]
        se = [sum(c["serial"][k] for c in rep["cells"]) for k in ("equal", "unequal", "failed")]
        nreq = sum(len(rp["evidence"]) for c in rep["cells"] for rp in c["reps"])
        print(lab, a, "cells", len(rep["cells"]), "requests", nreq, "alone eq/uneq/fail", al, "serial eq/uneq/fail", se)


def group(model, build):
    return [lab for lab, (r, q) in R.items() if r["model"] == model and r["build"] == build]


def compare(labs_a, labs_b, tools=None):
    keys = set.intersection(*[set(R[l][1]) for l in labs_a + labs_b])
    if tools:
        keys = {k for k in keys if k[0] in tools}
    same = diff = 0
    diffs = []
    for k in sorted(keys, key=str):
        va = {R[l][1][k] for l in labs_a}
        vb = {R[l][1][k] for l in labs_b}
        if None in va or None in vb:
            diffs.append((k, "missing"))
            diff += 1
        elif len(va | vb) == 1:
            same += 1
        else:
            diff += 1
            diffs.append((k, va, vb))
    return same, diff, len(keys), diffs


print("\n== exactness: before vs after, every request (token_sha), per folder and tool")
TOOLS = ["openai", "openai256", "off", "conc", "conc14"]
for m in "ATE":
    b, a = group(m, "before"), group(m, "after")
    if not b or not a:
        continue
    for t in TOOLS:
        s, d_, n, diffs = compare(b, a, [t])
        print(m, t, "before", b, "after", a, "keys", n, "same", s, "differ", d_, diffs[:2])
    s, d_, n, _ = compare(b, a)
    print(m, "ALL", "keys", n, "same", s, "differ", d_)

print("\n== negative control: T vs A and E vs A, same build, every request")
for bb in ("before", "after"):
    for m in "TE":
        if not group(m, bb) or not group("A", bb):
            continue
        for t in TOOLS:
            s, d_, n, diffs = compare(group("A", bb), group(m, bb), [t])
            print(bb, m, "vs A", t, "keys", n, "same", s, "differ", d_)
        s, d_, n, _ = compare(group("A", bb), group(m, bb))
        print(bb, m, "vs A ALL", "keys", n, "same", s, "differ", d_)

print("\n== decode speed, bench_openai (decode_tps_median of the reps per server)")
for a in ("openai", "openai256"):
    for m in "ATE":
        cells = defaultdict(dict)
        for bb in ("before", "after"):
            for lab in group(m, bb):
                r = R[lab][0]
                for row in load(os.path.join(r["_dir"], "%s.%s.json" % (r["label"], a))):
                    cells[(row["prompt"], row["temperature"], row["tokens"])][bb] = (lab, round(row["decode_tps_median"], 1), row["decode_tps_all"])
        for c, v in cells.items():
            if "before" in v and "after" in v:
                print(a, m, c, "before %.1f after %.1f after/before %.3f" % (v["before"][1], v["after"][1], v["after"][1] / v["before"][1]),
                      "| all before", v["before"][2], "after", v["after"][2])

print("\n== bench_concurrent aggregate and per-stream tok/s per cell")
for a in ("conc", "conc14"):
    for m in "ATE":
        cells = defaultdict(dict)
        for bb in ("before", "after"):
            for lab in group(m, bb):
                r = R[lab][0]
                for c in load(os.path.join(r["_dir"], "%s.%s.json" % (r["label"], a)))["cells"]:
                    cells[(c["prompt"], c["temperature"], c["streams"])][bb] = (c["aggregate_tps"], c["per_stream_tps"])
        for c, v in sorted(cells.items(), key=str):
            if "before" in v and "after" in v:
                print(a, m, c, "aggregate %.1f -> %.1f (%.3f), per stream %.1f -> %.1f" % (
                    v["before"][0], v["after"][0], v["after"][0] / v["before"][0], v["before"][1], v["after"][1]))

print("\n== undrafted server (decode_bench, draft false): median of 16 per server")
for m in "ATE":
    v = {}
    for bb in ("before", "after"):
        v[bb] = []
        for lab in group(m, bb):
            off = load(os.path.join(R[lab][0]["_dir"], "%s.off.json" % R[lab][0]["label"]))
            sm = off["summary"]["off"]
            v[bb].append((lab, round(sm["decode_tps"]["median"], 2), round(sm["server_tps"]["median"], 2), sm["ok"], sm["failed"]))
    if v["before"] and v["after"]:
        mb = statistics.mean(x[1] for x in v["before"])
        ma = statistics.mean(x[1] for x in v["after"])
        print("off", m, v, "%.2f -> %.2f ratio %.3f" % (mb, ma, ma / mb))

print("\n== first chat request (nothing cached) and the next chat request, from each server log; and prompt-cache lines")
done = re.compile(r"\[tensorfold\] done (\S+) prompt=(\d+) cached=(\d+) .*? tokens=(\d+) sha=(\w+) finish=(\w+)")
for lab, (r, q) in R.items():
    chats, cmpl_cached = [], Counter()
    for ln in open(os.path.join(r["_dir"], "%s.serve.log" % r["label"]), errors="replace"):
        m = done.search(ln)
        if m:
            cid, pt, ca, tok, sha, fin = m.groups()
            if cid.startswith("chatcmpl-"):
                chats.append((int(pt), int(ca), sha))
            else:
                cmpl_cached[(int(pt), int(ca))] += 1
    first = next((i for i, c in enumerate(chats) if c[1] == 0), None)
    pair = chats[first:first + 2] if first is not None else []
    print(lab, "first chat pair (prompt, cached, sha):", pair, "same sha" if len(pair) == 2 and pair[0][2] == pair[1][2] else "DIFFERENT or missing",
          "| chat requests", len(chats), "resumed (cached>0)", sum(1 for c in chats if c[1] > 0), "| completions (prompt, cached): count", sorted(cmpl_cached.items()))

if HOMES:
    print("\n== draft-window cost tables each server timed in its own HOME")
    for d in sorted(glob.glob(os.path.join(HOMES, "*"))):
        files = sorted(glob.glob(os.path.join(d, ".cache/tensorfold/window-costs/*")))
        for p in files[:2]:
            b = open(p, "rb").read()
            head = b[:6]
            if head == b"TFWC2\n" and len(b) == 6 + 17 * 8 + 8 + 8:
                v = struct.unpack("<17dQd", b[6:])
                print("%-10s %s 1/2/4/8/16 rows %s ms, a draft level %.3f ms" % (os.path.basename(d), os.path.basename(p)[:12],
                      "/".join("%.2f" % v[i] for i in (1, 2, 4, 8, 16)), v[18]))
            else:
                print("%-10s %s header %r, %d bytes (format not decoded)" % (os.path.basename(d), os.path.basename(p)[:12], head, len(b)))
        if not files:
            print("%-10s no window-costs files; home files: %s" % (os.path.basename(d), [os.path.relpath(os.path.join(dp, f), d) for dp, _, fs in os.walk(d) for f in fs][:6]))

print("\n== ratios over untaught in the same build (bench_openai cells; decode_bench draft false)")
for bb in ("before", "after"):
    A = {}
    for lab in group("A", bb):
        for a in ("openai", "openai256"):
            for row in load(os.path.join(R[lab][0]["_dir"], "%s.%s.json" % (R[lab][0]["label"], a))):
                A[(a, row["prompt"], row["temperature"])] = row["decode_tps_median"]
    for m in "TE":
        for lab in group(m, bb):
            out = []
            for a in ("openai", "openai256"):
                for row in load(os.path.join(R[lab][0]["_dir"], "%s.%s.json" % (R[lab][0]["label"], a))):
                    k = (a, row["prompt"], row["temperature"])
                    out.append("%s/%s/%s %.3f" % (a, row["prompt"][:4], row["temperature"], row["decode_tps_median"] / A[k]))
            print(bb, m, "/A:", "; ".join(out))
    offA = [load(os.path.join(R[l][0]["_dir"], "%s.off.json" % R[l][0]["label"]))["summary"]["off"]["decode_tps"]["median"] for l in group("A", bb)]
    for m in "TE":
        offM = [load(os.path.join(R[l][0]["_dir"], "%s.off.json" % R[l][0]["label"]))["summary"]["off"]["decode_tps"]["median"] for l in group(m, bb)]
        if offA and offM:
            print(bb, m, "/A draft false: %.3f" % (statistics.mean(offM) / statistics.mean(offA)))
