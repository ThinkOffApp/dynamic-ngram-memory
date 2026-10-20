"""Read the s1 and s2 server series (raw/v/s1, raw/v/s2): integrity, exactness before vs after, decode speeds."""
import json
import os
import statistics
import sys
from collections import defaultdict

V = sys.argv[1]
SERIES = ["s1", "s2"]


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

print("== integrity")
folder_sha = defaultdict(set)
for r in servers:
    lab = "%s-%s" % (r["series"], r["label"])
    folder_sha[r["model"]].add(json.dumps(r["sha_before"], sort_keys=True))
    rcs = {a: v.get("rc") for a, v in r["runs"].items()}
    print(lab, r["build"], r["model"], "started", r["started"], "stopped", r["stopped"], "ready", r["ready_s"],
          "exit", r["server_exit"], "bytes_unchanged", r["bytes_unchanged"], "listing_unchanged", r["listing_unchanged"],
          "nfiles", len(r["sha_before"]), "bin", r["binary_sha256"][:12], "rcs", rcs, "pocket_used_sum",
          sum(r["pocket"]["used"] or []))
for m, v in folder_sha.items():
    print("folder", m, "distinct sha sets across servers:", len(v))


def reqs(r):
    """Every request's token_sha keyed by tool and position; plus quality flags."""
    out = {}
    flags = []
    d = r["_dir"]
    lab = r["label"]
    for a in ("openai", "openai256"):
        for row in load(os.path.join(d, "%s.%s.json" % (lab, a))):
            for i, ev in enumerate(row["evidence"]):
                out[(a, row["prompt"], row["temperature"], 0, i)] = ev["token_sha"]
                if not ev["complete"] or ev["cache_state"] != "cold" or ev["tokens"] != row["tokens"]:
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
    return out, flags


R = {}
for r in servers:
    lab = "%s-%s" % (r["series"], r["label"])
    q, flags = reqs(r)
    R[lab] = (r, q)
    print("requests", lab, len(q), "flags", len(flags), flags[:3])

# conc/conc14 internal checks summed per server
print("\n== bench_concurrent internal checks (alone: concurrent reply == its solo run; serial: solo == draft false)")
for lab, (r, q) in R.items():
    for a in ("conc", "conc14"):
        rep = load(os.path.join(r["_dir"], "%s.%s.json" % (r["label"], a)))
        al = [sum(c["alone"][k] for c in rep["cells"]) for k in ("equal", "unequal", "failed")]
        se = [sum(c["serial"][k] for c in rep["cells"]) for k in ("equal", "unequal", "failed")]
        nreq = sum(len(rp["evidence"]) for c in rep["cells"] for rp in c["reps"])
        print(lab, a, "cells", len(rep["cells"]), "concurrent requests", nreq, "alone eq/uneq/fail", al, "serial eq/uneq/fail", se)


def group(model, build):
    return [lab for lab, (r, q) in R.items() if r["model"] == model and r["build"] == build]


def compare(labs_a, labs_b, tools=None):
    """Count keys where every server in both groups gives the same sha; and keys that differ."""
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
    for t in TOOLS:
        s, d_, n, diffs = compare(b, a, [t])
        print(m, t, "before", b, "after", a, "keys", n, "same", s, "differ", d_, diffs[:2])
    s, d_, n, _ = compare(b, a)
    print(m, "ALL", "keys", n, "same", s, "differ", d_)
    # within-build consistency across the 2 servers
    for bb in ("before", "after"):
        g = group(m, bb)
        s, d_, n, diffs = compare(g[:1], g[1:])
        print("   ", m, bb, "server vs server", g, "same", s, "differ", d_, [x[0] for x in diffs[:3]])

print("\n== negative control: T vs A (and E vs A), same build, every request")
for bb in ("before", "after"):
    for m in "TE":
        for t in TOOLS:
            s, d_, n, diffs = compare(group("A", bb), group(m, bb), [t])
            print(bb, m, "vs A", t, "keys", n, "same", s, "differ", d_)
        s, d_, n, _ = compare(group("A", bb), group(m, bb))
        print(bb, m, "vs A ALL", "keys", n, "same", s, "differ", d_)

print("\n== decode speed, bench_openai (median of reps per server), before vs after")
for a in ("openai", "openai256"):
    for m in "ATE":
        cells = defaultdict(lambda: defaultdict(list))
        for bb in ("before", "after"):
            for lab in group(m, bb):
                r = R[lab][0]
                for row in load(os.path.join(r["_dir"], "%s.%s.json" % (r["label"], a))):
                    cells[(row["prompt"], row["temperature"])][bb].append((lab, round(row["decode_tps_median"], 2)))
        for c, v in cells.items():
            mb = statistics.mean(x[1] for x in v["before"])
            ma = statistics.mean(x[1] for x in v["after"])
            print(a, m, c, "before", v["before"], "after", v["after"], "means %.1f -> %.1f ratio %.3f" % (mb, ma, ma / mb))

print("\n== undrafted server (decode_bench, draft false): median of 16 per server")
for m in "ATE":
    v = {}
    for bb in ("before", "after"):
        v[bb] = []
        for lab in group(m, bb):
            r = R[lab][0]
            off = load(os.path.join(r["_dir"], "%s.off.json" % r["label"]))
            sm = off["summary"]["off"]
            v[bb].append((lab, round(sm["decode_tps"]["median"], 2), round(sm["server_tps"]["median"], 2), sm["ok"], sm["failed"]))
    mb = statistics.mean(x[1] for x in v["before"])
    ma = statistics.mean(x[1] for x in v["after"])
    print("off", m, v, "means %.2f -> %.2f ratio %.3f" % (mb, ma, ma / mb))

json.dump({lab: {"|".join(map(str, k)): s for k, s in q.items()} for lab, (r, q) in R.items()},
          open(os.path.join(os.path.dirname(V.rstrip("/")), "server_shas.json"), "w"), indent=0)
