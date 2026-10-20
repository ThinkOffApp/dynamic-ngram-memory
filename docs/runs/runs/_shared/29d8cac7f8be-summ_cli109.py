"""1.0.9 receipt: summary of the dev CLI sanity series (sanity109.sh lines in chain109a.out). usage: summ_cli109.py CHAIN_OUT"""
import re
import statistics
import sys
from collections import defaultdict

LOG = sys.argv[1]
pat = re.compile(r"(\d\d:\d\d:\d\d) ([ATE]) (before|after) (nd|d) (\d+) exit=(\d+) temp=(\d+) tokens (\d+) sha (\w+) prefill ([\d.]+)s "
                 r"decode ([\d.]+)s ([\d.]+) ms/token rounds (\d+) accepted (\d+)")
pp = re.compile(r"(\d\d:\d\d:\d\d) p4096 ([ATE]) (before|after) (\d+) exit=(\d+) temp=(\d+) tokens (\d+) sha (\w+) prefill ([\d.]+)s")
rows, pre = [], []
for ln in open(LOG):
    m = pat.match(ln)
    if m:
        t, f, b, d, rep, ex, temp, tok, sha, pf, dec, mspt, rounds, acc = m.groups()
        rows.append(dict(f=f, b=b, d=d, rep=int(rep), exit=int(ex), temp=int(temp), tokens=int(tok), sha=sha, prefill=float(pf),
                         decode=float(dec), mspt=float(mspt), rounds=int(rounds), acc=int(acc)))
    m = pp.match(ln)
    if m:
        t, f, b, rep, ex, temp, tok, sha, pf = m.groups()
        pre.append(dict(f=f, b=b, rep=int(rep), exit=int(ex), tokens=int(tok), sha=sha, prefill=float(pf)))


def med(x):
    return statistics.median(x)


def rng(x, f="%.3f"):
    return (f + " (" + f + "-" + f + ")") % (med(x), min(x), max(x))


print("== 256 greedy tokens from p1 (36 ids): %d runs, exits %s, tokens %s, temps %d-%d C" % (
    len(rows), sorted({r["exit"] for r in rows}), sorted({r["tokens"] for r in rows}), min(r["temp"] for r in rows), max(r["temp"] for r in rows)))
g = defaultdict(list)
for r in rows:
    g[(r["f"], r["d"], r["b"])].append(r)
reps = sorted({r["rep"] for r in rows})
print("reps", reps)
for f in "ATE":
    for d in ("nd", "d"):
        shas = {b: {r["sha"] for r in g[(f, d, b)]} for b in ("before", "after")}
        eq = len(shas["before"] | shas["after"]) == 1
        print("%s %-2s hashes before %s after %s -> %s; runs %d + %d" % (f, d, sorted(shas["before"]), sorted(shas["after"]),
              "SAME" if eq else "DIFFER", len(g[(f, d, "before")]), len(g[(f, d, "after")])))
print("negative control: T hash %s, A hash %s, E hash %s" % (sorted({r["sha"] for r in rows if r["f"] == "T"}),
      sorted({r["sha"] for r in rows if r["f"] == "A"}), sorted({r["sha"] for r in rows if r["f"] == "E"})))
print()
print("folder mode build: ms/token median (range); tok/s = 255/decode median (range); prefill 36 tokens, ms median (range); rounds/accepted")
for f in "ATE":
    for d in ("nd", "d"):
        for b in ("before", "after"):
            v = g[(f, d, b)]
            print("%s %-2s %-6s ms/token %s  tok/s %s  prefill ms %s  rounds/acc %s" % (
                f, d, b, rng([r["mspt"] for r in v]), rng([255 / r["decode"] for r in v], "%.2f"),
                rng([r["prefill"] * 1000 for r in v], "%.1f"), sorted({(r["rounds"], r["acc"]) for r in v})))


def one(f, d, b, rep):
    x = [r for r in rows if (r["f"], r["d"], r["b"], r["rep"]) == (f, d, b, rep)]
    return x[0] if x else None


print()
print("same rep: speed after/before (= before ms/token / after ms/token)")
for f in "ATE":
    for d in ("nd", "d"):
        rat = [one(f, d, "before", k)["mspt"] / one(f, d, "after", k)["mspt"] for k in reps]
        print("  %s %-2s %s  per rep %s" % (f, d, rng(rat), ["%.3f" % x for x in rat]))
print("same rep: taught or empty over untaught, tok/s ratio (= A ms/token / X ms/token)")
for b in ("before", "after"):
    for d in ("nd", "d"):
        for f in "TE":
            rat = [one("A", d, b, k)["mspt"] / one(f, d, b, k)["mspt"] for k in reps]
            ms = [one(f, d, b, k)["mspt"] - one("A", d, b, k)["mspt"] for k in reps]
            print("  %-6s %-2s %s/A %s   %s minus A ms/token %s" % (b, d, f, rng(rat), f, rng(ms)))
print()
print("== 4,096-token prompt, --max-tokens 2 --no-drafts: %d runs, exits %s, tokens %s, first-token shas %s" % (
    len(pre), sorted({r["exit"] for r in pre}), sorted({r["tokens"] for r in pre}), sorted({r["sha"] for r in pre})))
P = {(r["f"], r["b"], r["rep"]): r["prefill"] for r in pre}
preps = sorted({r["rep"] for r in pre})
for f in "ATE":
    for b in ("before", "after"):
        v = [P[(f, b, k)] for k in preps]
        print("  %s %-6s prompt s %s  prompt tok/s %.0f" % (f, b, rng(v), 4096 / med(v)))
    rat = [P[(f, "after", k)] / P[(f, "before", k)] for k in preps]
    print("  %s after/before same rep %s per rep %s" % (f, rng(rat), ["%.3f" % x for x in rat]))
for f in "TE":
    for b in ("before", "after"):
        rat = [P[(f, b, k)] / P[("A", b, k)] for k in preps]
        print("  %-6s %s over A same rep %s" % (b, f, rng(rat)))
print()
print("== 36-token prompt (p1), both modes pooled (8 runs a folder and build): prompt ms median (range)")
for f in "ATE":
    for b in ("before", "after"):
        v = [r["prefill"] * 1000 for r in rows if r["f"] == f and r["b"] == b]
        print("  %s %-6s n=%d %s" % (f, b, len(v), rng(v, "%.1f")))
