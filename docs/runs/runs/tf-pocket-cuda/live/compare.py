#!/usr/bin/env python3
"""gb10-a: the live --slide runs of both builds compared: files, events, graphs, replies, logs. usage: compare.py [SUFFIX N]"""
import json, os, re, sys
from collections import Counter

H = os.path.expanduser("~") + "/tf-pocket-cuda/live"
SUF = sys.argv[1] if len(sys.argv) > 1 else ""
NL = int(sys.argv[2]) if len(sys.argv) > 2 else 4
CKS = ["ck%d" % i for i in range(NL + 1)] + ["ckR"]
DONE = re.compile(r"^\[tensorfold\] done (\S*)\s+prompt=(\d+) cached=\d+ thinking=(\w+) effort=\S+ tokens=(\d+) sha=(\w+) finish=(\w+)")
TAKE = re.compile(r"slide pocket: layer (\d+), neuron (\d+), at the (\w+)")
GRAPH = re.compile(r"graphs while a request on (the own|another) sequence ran: (\d+) captures in [\d.]+ ms, (\d+) replays of graphs it captured for itself, (\d+) eager windows")


def load(build, ck):
    p = "%s/%s%s/%s.json" % (H, build, SUF, ck)
    return json.load(open(p)) if os.path.exists(p) else None


def key(r):
    return (r["pid"], r["thinking"], r["draft"])


def text(r):
    return (r.get("reasoning") or "") + "\x00" + (r.get("content") or "")


def log_parts(build):
    """The server log split at each learn request: the learner's own replies (no request id) and the takes per part."""
    parts, cur = [], {"learner": [], "chat": [], "takes": [], "graphs": [], "lines": []}
    for name in ("serve.log",):
        for ln in open("%s/%s%s/%s" % (H, build, SUF, name), errors="replace"):
            ln = ln.rstrip("\n")
            if '"POST /v1/slide/learn' in ln:
                parts.append(cur)
                cur = {"learner": [], "chat": [], "takes": [], "graphs": [], "lines": []}
            m = DONE.match(ln)
            if m:
                (cur["chat"] if m.group(1) else cur["learner"]).append([int(m.group(2)), m.group(3), int(m.group(4)), m.group(5), m.group(6)])
            m = TAKE.search(ln)
            if m:
                cur["takes"].append([int(m.group(1)), int(m.group(2)), m.group(3)])
            m = GRAPH.search(ln)
            if m:
                cur["graphs"].append([m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))])
            if "slide" in ln and ("zeroed" in ln or "put back" in ln or "lesson written" in ln or "replac" in ln or "learned" in ln or "kept" in ln):
                cur["lines"].append(ln[:220])
    parts.append(cur)
    return parts


def main():
    out = {"strict": [], "notes": []}
    ok = True
    for ck in CKS:
        a, b = load("before", ck), load("after", ck)
        if a is None or b is None:
            out["notes"].append("%s missing: before %s after %s" % (ck, a is not None, b is not None))
            ok = False
            continue
        row = {"ck": ck}
        row["files_equal"] = a["sha"] == b["sha"]
        row["files"] = len(a["sha"])
        if ck not in ("ck0", "ckR"):
            prev = load("before", "ck%d" % (int(ck[2]) - 1))
            row["changed_files"] = sorted(n for n in a["sha"] if prev["sha"].get(n) != a["sha"][n])
            row["changed_sha"] = {n: a["sha"][n][:16] for n in row["changed_files"]}
            row["events_equal"] = a["events"] == b["events"]
            row["events"] = a["events"]
        row["graph_equal"] = a["graph"] == b["graph"]
        pa = {key(r): r.get("token_sha") for r in a["probe"]}
        pb = {key(r): r.get("token_sha") for r in b["probe"]}
        row["probe_equal"] = pa == pb and None not in pa.values()
        row["probe_n"] = len(pa)
        row["probe_draft_split"] = {bd: sorted({"%s/%s" % (k[0], k[1]) for k in p if p.get((k[0], k[1], "server")) != p.get((k[0], k[1], "off"))})
                                    for bd, p in (("before", pa), ("after", pb))}
        row["answers"] = {r["pid"]: (r.get("content") or "")[:90] for r in a["probe"] if not r["thinking"] and r["draft"] == "server"}
        row["answers_thinking"] = {r["pid"]: (r.get("content") or "")[:90] for r in a["probe"] if r["thinking"] and r["draft"] == "server"}
        g = a.get("graph_raw") or {}
        row["used_shards"] = ((g.get("model") or {}).get("pocket") or {}).get("used")
        row["used_shards_after"] = (((b.get("graph_raw") or {}).get("model") or {}).get("pocket") or {}).get("used")
        row["facts"] = [[f.get("id"), f.get("text"), f.get("state"), f.get("recalled"), sorted({n["layer"] for n in f.get("neurons") or []}),
                         len(f.get("neurons") or [])] for f in g.get("facts") or []]
        ok = ok and row["files_equal"] and row["graph_equal"] and row["probe_equal"] and row.get("events_equal", True)
        out["strict"].append(row)
    # chat replies alongside each lesson: same max_tokens in both builds, so token_sha sets compare directly
    out["chat"] = []
    for k in range(1, NL + 1):
        a, b = load("before", "ck%d" % k), load("after", "ck%d" % k)
        if a is None or b is None:
            continue
        qa, qb = load("before", "ck%d" % (k - 1)), load("before", "ck%d" % k)
        quiet = {}
        for tag, q in (("pre", qa), ("post", qb)):
            for r in q["probe"]:
                quiet.setdefault(key(r), {})[tag] = text(r)
        row = {"lesson": k}
        for bd, x in (("before", a), ("after", b)):
            rows = [r for r in x["chat"] if not r.get("error")]
            cls = Counter()
            for r in rows:
                t = text(r)
                q = quiet.get(key(r), {})
                pre = _prefix(t, q.get("pre"))
                post = _prefix(t, q.get("post"))
                cls["pre and post" if pre and post else "pre" if pre else "post" if post else "neither"] += 1
            writing = [p[0] for p in x["polls"] if any(f[2] == "writing" for f in p[1])]
            span = (min(writing), max(writing)) if writing else None
            during = sum(1 for r in rows if span and r["t0"] <= span[1] and r["t1"] >= span[0])
            row[bd] = {"replies": len(rows), "errors": sum(1 for r in x["chat"] if r.get("error")), "vs_quiet": dict(cls),
                       "writing_phase_s": span, "replies_overlapping_writing": during, "learn_s": x["learn_s"]}
        sa = {}
        for bd, x in (("before", a), ("after", b)):
            for r in x["chat"]:
                sa.setdefault(key(r), {}).setdefault(bd, set()).add(r.get("token_sha"))
        row["sha_sets_differ"] = [[list(k2), sorted(v.get("before", [])), sorted(v.get("after", []))] for k2, v in sorted(sa.items())
                                  if v.get("before") and v.get("after") and v["before"] != v["after"]]
        row["sha_only_one_build"] = [[list(k2), sorted(v.get("before", set()) ^ v.get("after", set()))] for k2, v in sorted(sa.items())
                                     if (v.get("before", set()) ^ v.get("after", set()))]
        out["chat"].append(row)
    # the learner's own replies (fact finder, lesson writer, its checks) and the neurons it took, per learn request
    la, lb = log_parts("before"), log_parts("after")
    out["learner"] = []
    for i in range(1, max(len(la), len(lb))):
        x, y = la[i] if i < len(la) else None, lb[i] if i < len(lb) else None
        if not x or not y:
            continue
        row = {"part": i, "learner_replies": [len(x["learner"]), len(y["learner"])],
               "learner_sha_sequence_equal": [r[3] for r in x["learner"]] == [r[3] for r in y["learner"]],
               "learner_sha_multiset_equal": Counter(r[3] for r in x["learner"]) == Counter(r[3] for r in y["learner"]),
               "learner_untaught_or_checks_thinking": Counter(r[1] for r in x["learner"]),
               "takes_equal": x["takes"] == y["takes"], "takes": x["takes"],
               "first_neuron_layers": sorted({t[0] for t in x["takes"] if t[1] == 0}),
               "graph_lines": {bd: {"own": sum(1 for g in p["graphs"] if g[0] == "the own"), "another": sum(1 for g in p["graphs"] if g[0] == "another"),
                                    "captures": sum(g[1] for g in p["graphs"]), "eager": sum(g[3] for g in p["graphs"])} for bd, p in (("before", x), ("after", y))},
               "lines_before": x["lines"], "lines_after": y["lines"]}
        out["learner"].append(row)
    for bd in ("before", "after"):
        r = json.load(open("%s/%s%s/run.json" % (H, bd, SUF))) if os.path.exists("%s/%s%s/run.json" % (H, bd, SUF)) else {}
        out[bd + "_run"] = {k: r.get(k) for k in ("result", "ready_s", "restart_ready_s", "stopped_equals_ck4", "restart_wrote_nothing", "live_vs_restart", "exit_live", "exit_restart")}
    out["all_strict_equal"] = ok
    json.dump(out, open(H + "/compare%s.json" % SUF, "w"), indent=1, default=list)
    for row in out["strict"]:
        print(row["ck"], "files", row["files_equal"], "events", row.get("events_equal"), "graph", row["graph_equal"], "probe", row["probe_equal"],
              "changed", row.get("changed_files"), "used", row["used_shards"])
    print("all strict equal:", ok)


def _prefix(t, q):
    if q is None or not t:
        return False
    r, c = t.split("\x00", 1)
    qr, qc = q.split("\x00", 1)
    if c:
        return qr == r and qc.startswith(c)
    return qr.startswith(r)


if __name__ == "__main__":
    sys.exit(main())
