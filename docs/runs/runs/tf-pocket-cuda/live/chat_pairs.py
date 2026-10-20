#!/usr/bin/env python3
"""Chat replies alongside each lesson paired by (worker, request number) across builds: equal shas, and the phase and
trainer step at which each unequal pair was sent."""
import json, os, sys
H = os.path.expanduser("~") + "/tf-pocket-cuda/live"
SUF = sys.argv[1] if len(sys.argv) > 1 else ""
NL = int(sys.argv[2]) if len(sys.argv) > 2 else 4


def phase_at(polls, t):
    cur = None
    for p in polls:
        if p[0] <= t:
            cur = next(([f[2], f[3]] for f in p[1] if f[1] == "learning"), None)
    return cur


tot = {"pairs": 0, "equal": 0}
for k in range(1, NL + 1):
    a, b = (json.load(open("%s/%s%s/ck%d.json" % (H, bd, SUF, k))) for bd in ("before", "after"))
    ra = {(r["worker"], r["n"]): r for r in a["chat"]}
    rb = {(r["worker"], r["n"]): r for r in b["chat"]}
    both = sorted(set(ra) & set(rb))
    eq = [x for x in both if ra[x].get("token_sha") == rb[x].get("token_sha")]
    ne = [x for x in both if x not in eq]
    tot["pairs"] += len(both); tot["equal"] += len(eq)
    print("lesson %d: %d replies before, %d after, %d paired, %d equal sha, %d unequal; only one build: %s" % (
        k, len(ra), len(rb), len(both), len(eq), len(ne), sorted(set(ra) ^ set(rb))))
    for x in ne:
        pa, pb = phase_at(a["polls"], ra[x]["t0"]), phase_at(b["polls"], rb[x]["t0"])
        print("   w%d n%d %s think=%s draft=%s: before %s at %s (t0 %.1f), after %s at %s (t0 %.1f)" % (
            x[0], x[1], ra[x]["pid"], ra[x]["thinking"], ra[x]["draft"], ra[x]["token_sha"], pa, ra[x]["t0"], rb[x]["token_sha"], pb, rb[x]["t0"]))
    # every equal pair's phase, as a count
    ph = {}
    for x in eq:
        p = phase_at(a["polls"], ra[x]["t0"])
        ph[p[0] if p else None] = ph.get(p[0] if p else None, 0) + 1
    print("   equal pairs by phase at send (before's polls):", ph)
print(tot)
