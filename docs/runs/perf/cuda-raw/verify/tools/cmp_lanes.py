#!/usr/bin/env python3
"""Compare the lanes-CLI fact runs: before vs after (full token ids), solo drafted vs --no-drafts vs together, folders.

usage: cmp_lanes.py DIR [OUT.json]
Files: DIR/{M}-{build}-{set}-{mode}.json with streams [{name, tokens, ...}].
"""
import collections, json, os, sys

D = sys.argv[1]
SETS = {"off-all": ["solo-d", "solo-nd"], "off-40": ["together-d"], "on-40": ["solo-d", "solo-nd", "together-d"]}


def load(m, b, s, mode):
    p = os.path.join(D, "%s-%s-%s-%s.json" % (m, b, s, mode))
    if not os.path.exists(p):
        return None
    r = json.load(open(p))
    return {x["name"]: x["tokens"] for x in r["streams"]}


out = {"before_vs_after": {}, "modes_equal": {}, "folders": {}, "first_diff": []}
runs = {}
for m in "TAE":
    for b in ("before", "after"):
        for s, modes in SETS.items():
            for mode in modes:
                r = load(m, b, s, mode)
                if r is not None:
                    runs[(m, b, s, mode)] = r
# before vs after, per folder, set, mode
tot = collections.Counter()
for (m, b, s, mode), r in sorted(runs.items()):
    if b != "before":
        continue
    a = runs.get((m, "after", s, mode))
    if a is None:
        continue
    eq = sum(1 for q in r if q in a and r[q] == a[q])
    n = len(r)
    toks = sum(len(v) for v in r.values())
    out["before_vs_after"]["%s %s %s" % (m, s, mode)] = {"equal": eq, "of": n, "tokens": toks}
    tot["equal"] += eq
    tot["of"] += n
    tot["tokens"] += toks
    for q in r:
        if q not in a or r[q] != a[q]:
            x, y = r[q], a.get(q, [])
            k = next((i for i in range(min(len(x), len(y))) if x[i] != y[i]), min(len(x), len(y)))
            out["first_diff"].append({"folder": m, "set": s, "mode": mode, "qid": q, "index": k, "len_before": len(x), "len_after": len(y),
                                      "before": x[max(0, k - 3):k + 3], "after": y[max(0, k - 3):k + 3]})
out["before_vs_after_total"] = dict(tot)
# drafted solo vs --no-drafts vs together, within a build; 40-question subset for together at thinking off
for m in "TAE":
    for b in ("before", "after"):
        for s, ref_mode, others in (("off-all", "solo-nd", ["solo-d"]), ("on-40", "solo-nd", ["solo-d", "together-d"])):
            ref = runs.get((m, b, s, ref_mode))
            if ref is None:
                continue
            for o in others:
                r = runs.get((m, b, s, o))
                if r is None:
                    continue
                eq = sum(1 for q in r if r[q] == ref.get(q))
                out["modes_equal"]["%s %s %s %s=%s" % (m, b, s, o, ref_mode)] = {"equal": eq, "of": len(r)}
        ref = runs.get((m, b, "off-all", "solo-nd"))
        r = runs.get((m, b, "off-40", "together-d"))
        if ref is not None and r is not None:
            eq = sum(1 for q in r if r[q] == ref.get(q))
            out["modes_equal"]["%s %s off-40 together-d=solo-nd" % (m, b)] = {"equal": eq, "of": len(r)}
# folders: T vs A (must differ somewhere), E vs A (must be equal), per build, set, mode
for b in ("before", "after"):
    for s, modes in SETS.items():
        for mode in modes:
            a = runs.get(("A", b, s, mode))
            if a is None:
                continue
            for m in "TE":
                r = runs.get((m, b, s, mode))
                if r is None:
                    continue
                eq = sum(1 for q in r if r[q] == a.get(q))
                out["folders"]["%s=A %s %s %s" % (m, b, s, mode)] = {"equal": eq, "of": len(r)}
print(json.dumps({k: v for k, v in out.items() if k != "first_diff"}, indent=1))
print("first differences before vs after:", len(out["first_diff"]))
for x in out["first_diff"][:20]:
    print(x)
if len(sys.argv) > 2:
    json.dump(out, open(sys.argv[2], "w"), indent=1)
