"""Split each lesson's wall time into parts and summarise (median, range, share). python3 -I lesson_split.py lessons.json"""
import json
import statistics as st
import sys

recs = json.load(open(sys.argv[1]))


def split(x):
    seq = x["seq"]
    in_round, reroute = 0.0, 0.0
    for i, e in enumerate(seq):
        if e[0] == "c":
            nxt = seq[i + 1] if i + 1 < len(seq) else None
            if nxt is not None and nxt[0] == "r":
                in_round += e[1]
            else:
                reroute += e[1]
    clock = {k: sum(c[k] for c in x["clock"]) for k in ("read", "fit", "values", "judge", "check")}
    passes = {k: sum(c[k + "_passes"] for c in x["clock"]) for k in ("read", "values", "judge", "check")}
    rows = {k: sum(c[k + "_rows"] for c in x["clock"]) for k in ("read", "values")}
    learning = sum(r["learning"] for r in x["rounds"])
    checking = sum(r["checking"] for r in x["rounds"])
    W = x["written"] or 0.0
    save = x["save_s"] or 0.0
    ready = x["ready"] or 0.0
    # learning holds the trainer's start (first lesson of a server), the round's pocket clock and the learner's own overhead
    overhead = learning - in_round - ready
    total = x["total"]
    other = total - W - learning - checking - reroute - save
    earlier = x["cand"]["earlier_s"] if x.get("cand") else 0.0
    return {
        "total": total, "write": W, "ready": ready,
        "read": clock["read"], "fit_new": clock["fit"] - earlier, "fit_earlier": earlier, "values": clock["values"], "judge": clock["judge"], "pcheck": clock["check"],
        "round_overhead": overhead, "reroute_train": reroute, "verify": checking, "save": save, "other": other,
        "train_clock": sum(clock.values()), "rounds": len(x["rounds"]), "clock_lines": len(x["clock"]),
        "values_passes": passes["values"], "values_rows": rows["values"], "read_passes": passes["read"], "read_rows": rows["read"],
        "judge_passes": passes["judge"], "check_passes": passes["check"],
        "gen_write": x["gens"]["write"], "gen_check": x["gens"]["check"], "gen_other": x["gens"]["other"],
        "save_mb": x["save_mb"], "replaces": x["replaces"], "handed": x["handed"], "refresh": x["refresh"],
    }


def q(vals):
    vals = sorted(vals)
    if not vals:
        return None
    return {"n": len(vals), "median": st.median(vals), "min": vals[0], "max": vals[-1], "mean": st.mean(vals), "sum": sum(vals),
            "p10": vals[int(0.1 * (len(vals) - 1))], "p90": vals[int(0.9 * (len(vals) - 1))]}


rows = []
for x in recs:
    s = split(x)
    s.update({k: x.get(k) for k in ("run", "phase", "label", "nth_in_server", "build", "chat_alongside", "source", "outcome", "event", "sse_learned", "drv_save_s", "drv_save_bytes", "upload_gb", "upload_s")})
    s["wrote"] = x["written"] is not None
    rows.append(s)
json.dump(rows, open(sys.argv[1].replace(".json", "-split.json"), "w"), indent=0)

parts = ["write", "ready", "read", "fit_new", "fit_earlier", "values", "judge", "pcheck", "round_overhead", "reroute_train", "verify", "save", "other"]


def report(name, sel):
    if not sel:
        return
    tot = q([r["total"] for r in sel])
    print("\n## %s: %d lessons, total median %.1f s (range %.1f to %.1f), sum %.0f s" % (name, tot["n"], tot["median"], tot["min"], tot["max"], tot["sum"]))
    print("%-15s %8s %8s %8s %8s %8s  %s" % ("part", "median", "min", "max", "mean", "p90", "share of summed time"))
    for p in parts:
        z = q([r[p] for r in sel])
        print("%-15s %8.1f %8.1f %8.1f %8.1f %8.1f  %5.1f%%" % (p, z["median"], z["min"], z["max"], z["mean"], z["p90"], 100 * z["sum"] / tot["sum"]))
    for k in ("rounds", "clock_lines", "values_passes", "values_rows", "read_passes", "read_rows", "judge_passes", "check_passes"):
        z = q([r[k] for r in sel])
        print("  %-14s median %s range %s..%s" % (k, z["median"], z["min"], z["max"]))
    for k in ("gen_write", "gen_check", "gen_other"):
        for i, lab in enumerate(("calls", "prompt_tok", "gen_tok")):
            z = q([r[k][i] for r in sel])
            print("  %-10s %-10s median %s range %s..%s sum %s" % (k, lab, z["median"], z["min"], z["max"], z["sum"]))


drv = [r for r in rows if r["source"] == "driver"]
wrote = [r for r in drv if r["wrote"]]
report("driver 1.0.8, all lessons", drv)
report("driver 1.0.8, lessons that got past writing", wrote)
report("driver 1.0.8, written, not first in server", [r for r in wrote if r["nth_in_server"] > 1])
report("driver 1.0.8, written, first in server", [r for r in wrote if r["nth_in_server"] == 1])
report("driver 1.0.8, not written (missed before training)", [r for r in drv if not r["wrote"]])
live = [r for r in rows if r["source"] == "live"]
report("live with chat alongside", [r for r in live if r["chat_alongside"]])
report("p3 no chat, 1.0.8 builds", [r for r in live if not r["chat_alongside"] and "1.0.9" not in r["build"]])
report("p3 no chat, 1.0.9 builds", [r for r in live if "1.0.9" in r["build"]])
