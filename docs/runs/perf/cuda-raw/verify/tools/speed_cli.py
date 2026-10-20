#!/usr/bin/env python3
"""Before/after summaries of the dev CLI series: rounds (GPU ms a graph round), decode bursts, 4,096-token prompts.

usage: speed_cli.py rounds RAW/v/rounds-TAG.out | speed_cli.py cli RAW/v/cli-TAG | speed_cli.py prefill RAW/v/prefill-TAG
Medians with (min-max). Pairs (T/A, E/A, after/before) are taken within one rep or burst, then summarised.
"""
import glob, json, os, re, statistics, sys

B = ("before", "after")


def med(v):
    return statistics.median(v) if v else None


def rng(v, f="%.3f"):
    return (f + " (" + f + " to " + f + ")") % (med(v), min(v), max(v)) if v else "-"


def rounds(path):
    rows = {}
    for ln in open(path):
        m = re.match(r"(\S+) (\w) (before|after) (\d+) exit=0 temp0=(\d+) (.*)", ln)
        if not m:
            continue
        nums = [float(x) for x in re.findall(r"([0-9.]+) ms a round", m.group(6))]
        rows[(m.group(2), m.group(3), int(m.group(4)))] = {"serial": nums[0::2], "window": nums[1::2], "temp": int(m.group(5))}
    reps = sorted({r for *_, r in rows})
    out = {"reps": len(reps)}
    for kind in ("serial", "window"):
        print("== %s graph, ms a round (each rep two timings)" % kind)
        for b in B:
            for m in "ATE":
                v = [x for r in reps if (m, b, r) in rows for x in rows[(m, b, r)][kind]]
                out["%s %s %s" % (kind, b, m)] = v
                print("  %-6s %s n=%d %s" % (b, m, len(v), rng(v)))
        for b in B:
            for m in "TE":
                d = [statistics.mean(rows[(m, b, r)][kind]) - statistics.mean(rows[("A", b, r)][kind]) for r in reps if (m, b, r) in rows and ("A", b, r) in rows]
                q = [statistics.mean(rows[("A", b, r)][kind]) / statistics.mean(rows[(m, b, r)][kind]) for r in reps if (m, b, r) in rows and ("A", b, r) in rows]
                out["%s %s %s-A ms" % (kind, b, m)] = d
                out["%s %s %s/A rate" % (kind, b, m)] = q
                print("  %-6s %s minus A, same rep: %s ms; %s/A rate %s" % (b, m, rng(d, "%+.3f"), m, rng(q)))
        for m in "ATE":
            d = [statistics.mean(rows[(m, "after", r)][kind]) - statistics.mean(rows[(m, "before", r)][kind]) for r in reps if (m, "after", r) in rows and (m, "before", r) in rows]
            out["%s %s after-before ms" % (kind, m)] = d
            print("  %s after minus before, same rep: %s ms" % (m, rng(d, "%+.3f")))
    return out


def cli(folder):
    rows = []
    for p in sorted(glob.glob(os.path.join(folder, "*.json"))):
        name = os.path.basename(p)[:-5]
        m, b, d, rep = name.split("-")
        r = json.load(open(p))
        rows.append({"m": m, "b": b, "d": d, "rep": int(rep), "ms": r["ms_per_token"], "tps": 1000 / r["ms_per_token"],
                     "sha": r["token_sha256"][:12], "rounds": r["rounds"], "accepted": r["accepted_drafts"], "prefill": r["prefill_seconds"],
                     "tokens": r["tokens"]})
    idx = {(x["m"], x["b"], x["d"], x["rep"]): x for x in rows}
    reps = sorted({x["rep"] for x in rows})
    out = {"runs": len(rows), "reps": len(reps)}
    for d in ("nd", "d"):
        print("== drafts", "off" if d == "nd" else "on")
        for b in B:
            for m in "ATE":
                v = [x["tps"] for x in rows if x["m"] == m and x["b"] == b and x["d"] == d]
                ms = [x["ms"] for x in rows if x["m"] == m and x["b"] == b and x["d"] == d]
                shas = sorted({x["sha"] for x in rows if x["m"] == m and x["b"] == b and x["d"] == d})
                rr = sorted({(x["rounds"], x["accepted"]) for x in rows if x["m"] == m and x["b"] == b and x["d"] == d})
                pf = [x["prefill"] * 1000 for x in rows if x["m"] == m and x["b"] == b and x["d"] == d]
                out["%s %s %s tps" % (d, b, m)] = v
                out["%s %s %s ms" % (d, b, m)] = ms
                out["%s %s %s sha" % (d, b, m)] = shas
                out["%s %s %s rounds" % (d, b, m)] = rr
                out["%s %s %s prefill_ms" % (d, b, m)] = pf
                print("  %-6s %s tok/s %s  ms/token %s  sha %s rounds/accepted %s prompt %s ms" % (b, m, rng(v, "%.2f"), rng(ms), shas, rr, rng(pf, "%.1f")))
        for b in B:
            for m in "TE":
                q = [idx[(m, b, d, r)]["tps"] / idx[("A", b, d, r)]["tps"] for r in reps if (m, b, d, r) in idx and ("A", b, d, r) in idx]
                e = [idx[(m, b, d, r)]["ms"] - idx[("A", b, d, r)]["ms"] for r in reps if (m, b, d, r) in idx and ("A", b, d, r) in idx]
                out["%s %s %s/A" % (d, b, m)] = q
                out["%s %s %s-A ms" % (d, b, m)] = e
                print("  %-6s %s/A same burst %s; extra ms a token %s" % (b, m, rng(q), rng(e, "%+.3f")))
        for m in "ATE":
            q = [idx[(m, "after", d, r)]["tps"] / idx[(m, "before", d, r)]["tps"] for r in reps if (m, "after", d, r) in idx and (m, "before", d, r) in idx]
            out["%s %s after/before" % (d, m)] = q
            print("  %s after/before same burst %s" % (m, rng(q)))
    # identity: one token list per folder and mode over both builds
    ident = {}
    for m in "ATE":
        for d in ("nd", "d"):
            toks = {json.dumps(x["tokens"]) for x in rows if x["m"] == m and x["d"] == d}
            ident["%s %s" % (m, d)] = len(toks)
        ident["%s nd=d" % m] = len({json.dumps(x["tokens"]) for x in rows if x["m"] == m}) == 1
    out["distinct_token_lists"] = ident
    print("distinct token lists per folder and mode (1 = before, after and every rep identical):", ident)
    return out


def prefill(folder):
    rows = []
    for p in sorted(glob.glob(os.path.join(folder, "*.json"))):
        m, b, rep = os.path.basename(p)[:-5].split("-")
        r = json.load(open(p))
        rows.append({"m": m, "b": b, "rep": int(rep), "s": r["prefill_seconds"], "first": r["tokens"][:1]})
    idx = {(x["m"], x["b"], x["rep"]): x for x in rows}
    reps = sorted({x["rep"] for x in rows})
    out = {}
    for b in B:
        for m in "ATE":
            v = [x["s"] for x in rows if x["m"] == m and x["b"] == b]
            out["%s %s s" % (b, m)] = v
            print("  %-6s %s prompt s %s  tok/s %.0f first %s" % (b, m, rng(v), 4096 / med(v), sorted({str(x["first"]) for x in rows if x["m"] == m and x["b"] == b})))
    for b in B:
        for m in "TE":
            q = [idx[(m, b, r)]["s"] / idx[("A", b, r)]["s"] for r in reps if (m, b, r) in idx and ("A", b, r) in idx]
            out["%s %s/A time" % (b, m)] = q
            print("  %-6s %s/A time same rep %s" % (b, m, rng(q)))
    for m in "ATE":
        q = [idx[(m, "after", r)]["s"] / idx[(m, "before", r)]["s"] for r in reps if (m, "after", r) in idx and (m, "before", r) in idx]
        out["%s after/before time" % m] = q
        print("  %s after/before time same rep %s" % (m, rng(q)))
    return out


if __name__ == "__main__":
    kind, path = sys.argv[1], sys.argv[2]
    res = {"rounds": rounds, "cli": cli, "prefill": prefill}[kind](path)
    if len(sys.argv) > 3:
        json.dump(res, open(sys.argv[3], "w"), indent=1)
