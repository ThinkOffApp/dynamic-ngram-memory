#!/usr/bin/env python3
"""Server fact replies' token_sha against the lanes CLI's full ids for the same folder (sha256 of the comma-joined ids,
first 12 hex, as the server computes it), for every server in the series given.

usage: cmp_srv_cli.py LANES_DIR SERIES_DIR...
"""
import glob, hashlib, json, os, sys

L = sys.argv[1]


def sha(ids):
    return hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()[:12]


tot = [0, 0]
for d in sys.argv[2:]:
    for p in sorted(glob.glob(os.path.join(d, "*.facts.json"))):
        run = json.load(open(p.replace(".facts.json", ".run.json")))
        m = run["model"]
        cli = {}
        for b in ("before", "after"):
            for th, fn in ((False, "off-all-solo-d"), (True, "on-40-solo-d")):
                for s in json.load(open(os.path.join(L, "%s-%s-%s.json" % (m, b, fn))))["streams"]:
                    cli.setdefault((s["name"], th), set()).add(sha(s["tokens"]))
        rows = json.load(open(p))
        eq = sum(1 for r in rows if cli.get((r["qid"], r["thinking"])) == {r["token_sha"]})
        tot[0] += eq
        tot[1] += len(rows)
        print("%s/%s (%s on %s): %d of %d server replies have the CLI's ids (both CLI builds agree)" % (
            os.path.basename(d), run["label"], run["build"], m, eq, len(rows)))
print("total %d of %d" % tuple(tot))
