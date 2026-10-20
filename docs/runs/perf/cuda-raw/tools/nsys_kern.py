#!/usr/bin/env python3
"""Kernel times from an nsys sqlite export: every kernel's count and time, tf_pocket_rows by grid size and stream.

usage: nsys_kern.py FILE.sqlite [OUT.json]
"""
import json, sqlite3, statistics, sys


def pct(v, q):
    v = sorted(v)
    if not v:
        return None
    k = (len(v) - 1) * q / 100
    lo = int(k)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def main():
    db = sqlite3.connect(sys.argv[1])
    names = dict(db.execute("select id, value from StringIds"))
    rows = db.execute("select start, end, streamId, gridX, gridY, gridZ, shortName from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
    total = sum(e - s for s, e, *_ in rows)
    by = {}
    for s, e, st, gx, gy, gz, sn in rows:
        by.setdefault(names.get(sn, str(sn)), []).append(e - s)
    kern = sorted(({"name": k, "count": len(v), "total_ms": sum(v) / 1e6, "share": sum(v) / total, "median_us": statistics.median(v) / 1e3}
                   for k, v in by.items()), key=lambda x: -x["total_ms"])
    pocket = [(s, e, st, gx) for s, e, st, gx, gy, gz, sn in rows if names.get(sn) == "tf_pocket_rows"]
    groups = {}
    for s, e, st, gx in pocket:
        groups.setdefault((gx, st), []).append(e - s)
    pg = [{"grid": gx, "stream": st, "count": len(v), "median_us": statistics.median(v) / 1e3, "p10_us": pct(v, 10) / 1e3,
           "p90_us": pct(v, 90) / 1e3, "min_us": min(v) / 1e3, "max_us": max(v) / 1e3, "total_ms": sum(v) / 1e6}
          for (gx, st), v in sorted(groups.items())]
    span = (rows[-1][1] - rows[0][0]) / 1e6 if rows else 0
    out = {"kernels_total_ms": total / 1e6, "first_to_last_kernel_ms": span, "launches": len(rows),
           "pocket_total_ms": sum(e - s for s, e, *_ in pocket) / 1e6, "pocket_share": (sum(e - s for s, e, *_ in pocket) / total) if total else None,
           "pocket_by_grid_stream": pg, "top": kern[:25]}
    print(json.dumps({k: v for k, v in out.items() if k != "top"}, indent=1))
    for k in kern[:15]:
        print("%-60s %7d %9.2f ms %5.1f%% med %8.2f us" % (k["name"][:60], k["count"], k["total_ms"], 100 * k["share"], k["median_us"]))
    if len(sys.argv) > 2:
        json.dump(out, open(sys.argv[2], "w"), indent=1)


if __name__ == "__main__":
    main()
