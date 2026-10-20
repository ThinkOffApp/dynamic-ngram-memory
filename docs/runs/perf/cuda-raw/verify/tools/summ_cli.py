"""Dev CLI series in raw/v: lanes-q1 (exactness), cli-c1 (decode bursts), rounds-r1 (graph ms), prefill-p1, s1 facts."""
import json
import os
import re
import statistics
import sys
from collections import defaultdict

V = sys.argv[1]


def load(p):
    with open(p) as f:
        return json.load(f)


def med(x):
    return statistics.median(x)


def rng(x):
    return "%.3f (%.3f-%.3f)" % (med(x), min(x), max(x))


print("== lanes-q1: per-question token sha, before vs after; solo vs together; drafted vs not")
L = os.path.join(V, "lanes-q1")
runs = {}
for n in os.listdir(L):
    m = re.fullmatch(r"([ATE])-(before|after)-(off-all|off-40|on-40)-(solo-d|solo-nd|together-d)\.json", n)
    if m:
        d = load(os.path.join(L, n))
        runs[m.groups()] = {s["name"]: s["sha"] for s in d["streams"]}
for m in "ATE":
    for p in ("off-all", "off-40", "on-40"):
        for mode in ("solo-d", "solo-nd", "together-d"):
            b, a = runs.get((m, "before", p, mode)), runs.get((m, "after", p, mode))
            if b is None:
                continue
            same = sum(1 for q in b if a.get(q) == b[q])
            print(m, p, mode, "questions", len(b), "before==after", same)
    for bb in ("before", "after"):
        sd, snd = runs[(m, bb, "off-all", "solo-d")], runs[(m, bb, "off-all", "solo-nd")]
        t40 = runs[(m, bb, "off-40", "together-d")]
        o_sd, o_snd, o_t = runs[(m, bb, "on-40", "solo-d")], runs[(m, bb, "on-40", "solo-nd")], runs[(m, bb, "on-40", "together-d")]
        print("  ", m, bb, "off: drafted==undrafted %d/%d, together==solo %d/%d; on: drafted==undrafted %d/%d, together==solo %d/%d" % (
            sum(sd[q] == snd[q] for q in sd), len(sd), sum(t40[q] == sd[q] for q in t40), len(t40),
            sum(o_sd[q] == o_snd[q] for q in o_sd), len(o_sd), sum(o_t[q] == o_sd[q] for q in o_t), len(o_t)))
for bb in ("before", "after"):
    for p in ("off-all", "on-40"):
        ta = runs[("T", bb, p, "solo-d")]
        aa = runs[("A", bb, p, "solo-d")]
        ea = runs[("E", bb, p, "solo-d")]
        print("  negative control", bb, p, "T==A %d/%d, E==A %d/%d" % (sum(ta[q] == aa[q] for q in ta), len(ta), sum(ea[q] == aa[q] for q in ea), len(ea)))

print("\n== cli-c1: dev CLI decode, 256 greedy tokens from p1")
C = os.path.join(V, "cli-c1")
pat = re.compile(r"(\d\d:\d\d:\d\d) ([ATE]) (before|after) (nd|d) (\d) exit=(\d+) tokens (\d+) sha (\w+) prefill ([\d.]+)s decode ([\d.]+)s ([\d.]+) ms/token rounds (\d+) accepted (\d+)")
rows = []
for ln in open(os.path.join(V, "cli-c1.out")):
    m = pat.match(ln)
    if m:
        t, f, b, d, rep, ex, tok, sha, pre, dec, mspt, rounds, acc = m.groups()
        rows.append(dict(t=t, f=f, b=b, d=d, rep=int(rep), exit=int(ex), tokens=int(tok), sha=sha, prefill=float(pre),
                         decode=float(dec), mspt=float(mspt), rounds=int(rounds), acc=int(acc)))
print("rows", len(rows), "exits", {r["exit"] for r in rows}, "tokens", {r["tokens"] for r in rows})
g = defaultdict(list)
for r in rows:
    g[(r["f"], r["b"], r["d"])].append(r)
for k in sorted(g):
    v = g[k]
    tps = [255 / r["decode"] for r in v]
    print(k, "n", len(v), "sha", {r["sha"] for r in v}, "rounds/acc", {(r["rounds"], r["acc"]) for r in v},
          "ms/token", rng([r["mspt"] for r in v]), "tok/s(255/decode)", rng(tps), "prefill s", rng([r["prefill"] for r in v]))
# check how ms/token relates to decode
r0 = rows[0]
print("ms/token check", r0["mspt"], r0["decode"] / 255 * 1000, r0["decode"] / 256 * 1000)
# same-burst ratios after/before
for f in "ATE":
    for d in ("nd", "d"):
        rat = []
        for rep in range(1, 7):
            b = [r for r in rows if (r["f"], r["b"], r["d"], r["rep"]) == (f, "before", d, rep)]
            a = [r for r in rows if (r["f"], r["b"], r["d"], r["rep"]) == (f, "after", d, rep)]
            if a and b:
                rat.append(b[0]["mspt"] / a[0]["mspt"])
        print("speedup after/before same burst", f, d, rng(rat))
for d in ("nd", "d"):
    for f in "TE":
        rat = []
        for rep in range(1, 7):
            x = [r for r in rows if (r["f"], r["b"], r["d"], r["rep"]) == (f, "after", d, rep)]
            y = [r for r in rows if (r["f"], r["b"], r["d"], r["rep"]) == ("A", "after", d, rep)]
            rat.append(y[0]["mspt"] / x[0]["mspt"])
        print("after: %s/A tok/s ratio same burst" % f, d, rng(rat))
        ms = []
        for rep in range(1, 7):
            x = [r for r in rows if (r["f"], r["b"], r["d"], r["rep"]) == (f, "after", d, rep)]
            y = [r for r in rows if (r["f"], r["b"], r["d"], r["rep"]) == ("A", "after", d, rep)]
            ms.append(x[0]["mspt"] - y[0]["mspt"])
        print("after: %s minus A ms/token same burst" % f, d, rng(ms))
print("cli-c1 hashes unchanged:", open(os.path.join(C, "hash-before.txt")).read() == open(os.path.join(C, "hash-after.txt")).read())

print("\n== rounds-r1: GPU ms per captured graph round (two measurements per load)")
pat = re.compile(r"(\d\d:\d\d:\d\d) ([ATE]) (before|after) (\d) exit=(\d+) temp0=(\d+) (.*)")
g = defaultdict(lambda: defaultdict(list))
per = defaultdict(dict)
for ln in open(os.path.join(V, "rounds-r1.out")):
    m = pat.match(ln)
    if not m:
        continue
    t, f, b, rep, ex, temp, rest = m.groups()
    assert ex == "0"
    s = [float(x) for x in re.findall(r"serial graph: ([\d.]+) ms", rest)]
    w = [float(x) for x in re.findall(r"window graph: ([\d.]+) ms", rest)]
    g[(f, b)]["serial"] += s
    g[(f, b)]["window"] += w
    per[(f, b)][int(rep)] = (s, w)
for k in sorted(g):
    print(k, "serial", rng(g[k]["serial"]), "window", rng(g[k]["window"]), "n", len(g[k]["serial"]))
for f in "ATE":
    d = []
    for rep in range(1, 5):
        d += [x - y for x, y in zip(per[(f, "before")][rep][0], per[(f, "after")][rep][0])]
    print("serial before minus after, same rep", f, rng(d))
for f in "TE":
    d = []
    for rep in range(1, 5):
        d += [x - y for x, y in zip(per[(f, "after")][rep][0], per[("A", "after")][rep][0])]
    print("after: serial %s minus A same rep" % f, rng(d))
    d = []
    for rep in range(1, 5):
        d += [x - y for x, y in zip(per[(f, "before")][rep][0], per[("A", "before")][rep][0])]
    print("before: serial %s minus A same rep" % f, rng(d))

print("\n== prefill-p1: 4,096-token prompt, 2 out (1 token), drafts off")
pat = re.compile(r"(\d\d:\d\d:\d\d) ([ATE]) (before|after) (\d) exit=(\d+) tokens (\d+) sha (\w+) prefill ([\d.]+)s")
g = defaultdict(list)
per = {}
shas = set()
for ln in open(os.path.join(V, "prefill-p1.out")):
    m = pat.match(ln)
    if m:
        t, f, b, rep, ex, tok, sha, pre = m.groups()
        g[(f, b)].append(float(pre))
        per[(f, b, int(rep))] = float(pre)
        shas.add(sha)
for k in sorted(g):
    print(k, "prompt s", rng(g[k]), "tok/s %.0f" % (4096 / med(g[k])))
print("first-token shas", shas)
for f in "ATE":
    print("after/before same rep", f, rng([per[(f, "after", r)] / per[(f, "before", r)] for r in (1, 2, 3)]))
for f in "TE":
    for b in ("before", "after"):
        print(b, "%s/A same rep" % f, rng([per[(f, b, r)] / per[("A", b, r)] for r in (1, 2, 3)]))
for f in "ATE":
    for b in ("before", "after"):
        js = [load(os.path.join(V, "prefill-p1", "%s-%s-%d.json" % (f, b, r))) for r in (1, 2, 3)]
        print(f, b, "report keys", sorted(js[0].keys())[:12], "prompt_tokens", len(js[0].get("prompt_tokens", [])))

print("\n== s1 facts (server, 40 questions; drafted and draft false; thinking off 160 and on 1500)")
S1 = os.path.join(V, "s1")
F = {}
for n in os.listdir(S1):
    if n.endswith(".facts.json"):
        lab = n.split(".")[0]
        F[lab] = {(x["qid"], x["thinking"], x["draft"]): x for x in load(os.path.join(S1, n))}
for lab, v in sorted(F.items()):
    errs = sum(1 for x in v.values() if x.get("error"))
    dvs = sum(1 for (q, th, dr), x in v.items() if dr == "server" and x["token_sha"] == v[(q, th, "off")]["token_sha"])
    print(lab, "replies", len(v), "errors", errs, "drafted==draft-false", dvs, "/", len(v) // 2)
pairs = {"T": ("1-bT", "2-aT"), "A": ("4-bA", "3-aA"), "E": ("5-bE", "6-aE")}
for m, (b, a) in pairs.items():
    same = sum(1 for k in F[b] if F[b][k]["token_sha"] == F[a][k]["token_sha"])
    print(m, "facts before==after", same, "/", len(F[b]))
print("T vs A facts same", sum(1 for k in F["1-bT"] if F["1-bT"][k]["token_sha"] == F["4-bA"][k]["token_sha"]), "/", len(F["1-bT"]))
