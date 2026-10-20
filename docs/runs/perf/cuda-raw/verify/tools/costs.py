#!/usr/bin/env python3
"""Decode the draft-window cost tables each server timed in its own HOME (TFWC2 + Costs: verify[17] f64, rows, level).

usage: costs.py HOMES_DIR [OUT.json]
"""
import glob, json, os, struct, sys

out = {}
for d in sorted(glob.glob(os.path.join(sys.argv[1], "*"))):
    tabs = []
    for p in sorted(glob.glob(os.path.join(d, ".cache/tensorfold/window-costs/*.bin"))):
        b = open(p, "rb").read()
        if not b.startswith(b"TFWC2\n") or len(b) != 6 + 17 * 8 + 8 + 8:
            continue
        v = struct.unpack("<17dQd", b[6:])
        tabs.append({"file": os.path.basename(p)[:12], "verify_ms": [round(x, 3) for x in v[1:17]], "rows": v[17], "level_ms": round(v[18], 4)})
    out[os.path.basename(d)] = tabs
for k, v in out.items():
    for t in v[:1]:
        print("%-10s 1/2/4/8/16 rows %s ms, a draft level %.3f ms (%d files)" % (k, "/".join("%.2f" % t["verify_ms"][i - 1] for i in (1, 2, 4, 8, 16)), t["level_ms"], len(v)))
if len(sys.argv) > 2:
    json.dump(out, open(sys.argv[2], "w"), indent=1)
