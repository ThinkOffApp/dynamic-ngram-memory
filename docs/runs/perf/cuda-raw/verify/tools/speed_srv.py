#!/usr/bin/env python3
"""Server series speed: bench_openai cells, decode_bench "draft": false, bench_concurrent aggregates, per server,
then per condition and the within-series ratios (after/before per folder; T/A and E/A per build).

usage: speed_srv.py OUT.json SERIES_DIR...
"""
import collections, glob, json, os, statistics, sys

out_path, dirs = sys.argv[1], sys.argv[2:]
cells = collections.defaultdict(dict)   # (series, build, folder) -> {cell: value}
ttft = collections.defaultdict(dict)
order = {}
for d in dirs:
    series = os.path.basename(d.rstrip("/"))
    for p in sorted(glob.glob(os.path.join(d, "*.run.json"))):
        r = json.load(open(p))
        key = (series, r["build"], r["model"])
        order[key] = r["label"]
        runs = r.get("runs") or {}
        for a in ("openai", "openai256"):
            if a not in runs:
                continue
            for row in json.load(open(os.path.join(d, os.path.basename(runs[a]["out"])))):
                c = "%s %s %d T=%g" % ("openai" if a == "openai" else "openai256", row["prompt"].split("-")[0], row["tokens"], row["temperature"])
                cells[key][c] = row["decode_tps_median"]
                ttft[key][c] = row["ttft_s_median"]
        if "off" in runs:
            j = json.load(open(os.path.join(d, os.path.basename(runs["off"]["out"]))))
            s = j["summary"]["off"]
            cells[key]["off 256 draft:false"] = s["decode_tps"]["median"]
            cells[key]["off 256 draft:false nm1"] = s["decode_tps_nm1"]["median"]
            sv = [x.get("server_tps") for x in j["rows"] if x.get("phase") == "measured" and x.get("server_tps")]
            if sv:
                cells[key]["off 256 draft:false server"] = statistics.median(sv)
        for a in ("conc", "conc14"):
            if a not in runs:
                continue
            j = json.load(open(os.path.join(d, os.path.basename(runs[a]["out"]))))
            for c in j["cells"]:
                cells[key]["%s %s T=%g x%d aggregate" % (a, c["prompt"], c["temperature"], c["streams"])] = c["aggregate_tps"]
series = sorted({k[0] for k in cells})
names = sorted({c for v in cells.values() for c in v})
res = {"per_server": {"%s %s %s" % k: {"label": order[k], **v} for k, v in cells.items()}, "ttft": {"%s %s %s" % k: v for k, v in ttft.items()}}
table = {}
for c in names:
    row = {}
    for b in ("before", "after"):
        for m in "ATE":
            v = [cells[(s, b, m)].get(c) for s in series if (s, b, m) in cells and cells[(s, b, m)].get(c) is not None]
            row["%s %s" % (b, m)] = v
    ratios = {}
    for s in series:
        for m in "ATE":
            x, y = cells.get((s, "after", m), {}).get(c), cells.get((s, "before", m), {}).get(c)
            if x and y:
                ratios.setdefault("%s after/before" % m, []).append(x / y)
        for b in ("before", "after"):
            a = cells.get((s, b, "A"), {}).get(c)
            for m in "TE":
                x = cells.get((s, b, m), {}).get(c)
                if x and a:
                    ratios.setdefault("%s %s/A" % (b, m), []).append(x / a)
                    ratios.setdefault("%s %s-A ms/token" % (b, m), []).append(1000 / x - 1000 / a)
    table[c] = {"values": row, "ratios": ratios}
res["table"] = table
json.dump(res, open(out_path, "w"), indent=1)


def f(v):
    return "/".join("%.1f" % x for x in v) if v else "-"


def g(v, p="%.3f"):
    return "/".join(p % x for x in v) if v else "-"


for c in names:
    t = table[c]
    print("%-40s " % c + "  ".join("%s %s" % (k, f(v)) for k, v in t["values"].items()))
    print("%-40s " % "" + "  ".join("%s %s" % (k, g(v, "%+.3f" if "ms" in k else "%.3f")) for k, v in t["ratios"].items()))
