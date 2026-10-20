#!/usr/bin/env python3
"""Summaries of the rounds and CLI burst logs: medians, ranges, and taught/untaught pairs within each rep.

usage: summ.py rounds RAW/rounds-TAG.out | summ.py cli RAW/cli-TAG
"""
import glob, json, os, re, statistics, sys


def med(v):
    return statistics.median(v) if v else None


def rng(v):
    return "%.3f (%.3f-%.3f)" % (med(v), min(v), max(v)) if v else "-"


def rounds(path):
    rows = {}
    for ln in open(path):
        m = re.match(r"(\S+) (\w) (\d+) exit=0 temp0=(\d+) (.*)", ln)
        if not m:
            continue
        nums = [float(x) for x in re.findall(r"([0-9.]+) ms a round", m.group(5))]
        # order: serial, window, serial, window; the second pair is the steadier one
        rows[(m.group(2), int(m.group(3)))] = {"serial": nums[0::2], "window": nums[1::2], "temp": int(m.group(4))}
    reps = sorted({r for _, r in rows})
    out = {}
    for kind in ("serial", "window"):
        for m in "ATE":
            v = [x for r in reps if (m, r) in rows for x in rows[(m, r)][kind]]
            out["%s %s" % (m, kind)] = v
            print("%-9s %s  ms a round (n=%d)" % (m, kind, len(v)), rng(v))
        for m in "TE":
            d = [statistics.mean(rows[(m, r)][kind]) - statistics.mean(rows[("A", r)][kind]) for r in reps if (m, r) in rows and ("A", r) in rows]
            out["%s-A %s" % (m, kind)] = d
            print("%s minus A, %s, same rep: +%s ms" % (m, kind, rng(d)))
    return out


def cli(folder):
    rows = []
    for p in sorted(glob.glob(os.path.join(folder, "*.json"))):
        name = os.path.basename(p)[:-5]
        m, d, rep = name.split("-")
        r = json.load(open(p))
        steps = max(1, len(r["tokens"]) - 1)
        rows.append({"m": m, "d": d, "rep": int(rep), "ms": r["ms_per_token"], "tps": 1000 / r["ms_per_token"],
                     "sha": r["token_sha256"][:12], "rounds": r["rounds"], "accepted": r["accepted_drafts"], "prefill": r["prefill_seconds"]})
    for d in ("nd", "d"):
        print("== drafts", "off" if d == "nd" else "on")
        for m in "ATE":
            rs = [x for x in rows if x["m"] == m and x["d"] == d]
            print("%s tok/s %s  ms/token %s  shas %s  rounds %s" % (m, rng([x["tps"] for x in rs]), rng([x["ms"] for x in rs]),
                  sorted({x["sha"] for x in rs}), sorted({x["rounds"] for x in rs})))
        for m in "TE":
            ratio, extra = [], []
            for rep in sorted({x["rep"] for x in rows}):
                a = [x for x in rows if x["m"] == "A" and x["d"] == d and x["rep"] == rep]
                b = [x for x in rows if x["m"] == m and x["d"] == d and x["rep"] == rep]
                if a and b:
                    ratio.append(b[0]["tps"] / a[0]["tps"])
                    extra.append(b[0]["ms"] - a[0]["ms"])
            print("%s/A same burst: %s   extra ms/token: %s" % (m, rng(ratio), rng(extra)))
    return rows


if __name__ == "__main__":
    print(json.dumps((rounds if sys.argv[1] == "rounds" else cli)(sys.argv[2]), default=str)[:0])
