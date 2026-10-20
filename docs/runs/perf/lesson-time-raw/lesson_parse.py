"""Per-lesson time split from TensorFold 1.0.8/1.0.9 slide server logs (+ the lwrun/capacity driver's run.log).

Read-only over copies in the scratchpad. Usage: python3 -I lesson_parse.py <lwruns dir> <live dir> <r109live dir> > lessons.json
"""
import json
import os
import re
import sys

RE_CLOCK = re.compile(r"slide pocket: round (\d+) took ([\d.]+) s: read ([\d.]+) s \((\d+) passes, (\d+) rows\), fit ([\d.]+) s, values ([\d.]+) s \((\d+) passes, (\d+) rows\), judge ([\d.]+) s \((\d+) passes\), check ([\d.]+) s \((\d+) passes\)")
RE_ROUND = re.compile(r"slide: round (\d+): (\d+) steps \(([\d.]+) s learning, ([\d.]+) s checking\)")
RE_WRITTEN = re.compile(r"slide: fact (\d+)'s lesson written in ([\d.]+) s")
RE_SAVE = re.compile(r"in the model's shards \((\d+) MB written in ([\d.]+) s\)")
RE_READY = re.compile(r"the trainer is ready in ([\d.]+) s")
RE_UPLOAD = re.compile(r"raw weights \(([\d.]+) GB\) uploaded in ([\d.]+) s, its buffers \(([\d.]+) GB\) in ([\d.]+) s")
RE_CAND = re.compile(r"(\d+) candidate neurons fitted on (\d+) read rows in ([\d.]+) s; earlier neurons checked in ([\d.]+) s")
RE_GEN = re.compile(r"\[tensorfold\] done  prompt=(\d+) cached=(\d+) thinking=(\w+) effort=\w+ tokens=(\d+)")
RE_POST = re.compile(r'"POST /v1/slide/learn HTTP')
RE_REPLACES = re.compile(r'slide: ".*" replaces ".*" \(fact \d+\)')
RE_HANDED = re.compile(r"is handed back to the lesson")
RE_REFRESH = re.compile(r"slide: (\d+) keep prompts and (\d+) bank questions the fact answers are read again")


def segments(path):
    """The server log cut at each learn request."""
    segs, cur = [], None
    with open(path, errors="replace") as f:
        for line in f:
            if RE_POST.search(line):
                cur = []
                segs.append(cur)
            elif cur is not None:
                cur.append(line.rstrip("\n"))
    return segs


def parse_segment(lines):
    r = {"written": None, "ready": None, "upload_gb": None, "upload_s": None, "save_mb": None, "save_s": None,
         "clock": [], "rounds": [], "cand": None, "replaces": 0, "handed": 0, "refresh": None,
         "gens": {"write": [0, 0, 0], "check": [0, 0, 0], "other": [0, 0, 0], "post": [0, 0, 0]}, "seq": []}
    state = "write"
    for ln in lines:
        m = RE_GEN.search(ln)
        if m:
            g = r["gens"][state]
            g[0] += 1
            g[1] += int(m.group(1)) - int(m.group(2))  # prompt tokens computed
            g[2] += int(m.group(4))  # tokens generated
            continue
        m = RE_WRITTEN.search(ln)
        if m:
            r["written"] = float(m.group(2))
            state = "other"
            continue
        m = RE_CLOCK.search(ln)
        if m:
            g = [float(x) if "." in x else int(x) for x in m.groups()]
            r["seq"].append(["c", g[1]])
            r["clock"].append(dict(zip(["round", "total", "read", "read_passes", "read_rows", "fit", "values", "values_passes", "values_rows", "judge", "judge_passes", "check", "check_passes"], g)))
            state = "check"
            continue
        m = RE_ROUND.search(ln)
        if m:
            r["seq"].append(["r", float(m.group(3)), float(m.group(4))])
            r["rounds"].append({"round": int(m.group(1)), "steps": int(m.group(2)), "learning": float(m.group(3)), "checking": float(m.group(4))})
            state = "other"
            continue
        m = RE_SAVE.search(ln)
        if m:
            r["save_mb"], r["save_s"] = int(m.group(1)), float(m.group(2))
            state = "post"
            continue
        m = RE_READY.search(ln)
        if m:
            r["ready"] = float(m.group(1))
            continue
        m = RE_UPLOAD.search(ln)
        if m:
            r["upload_gb"], r["upload_s"] = float(m.group(1)), float(m.group(2))
            continue
        m = RE_CAND.search(ln)
        if m:
            prev = r["cand"]["earlier_s"] if r["cand"] else 0.0
            r["cand"] = {"n": int(m.group(1)), "rows": int(m.group(2)), "fit_s": float(m.group(3)), "earlier_s": prev + float(m.group(4))}
            continue
        if RE_REPLACES.search(ln):
            r["replaces"] += 1
        if RE_HANDED.search(ln):
            r["handed"] += 1
        m = RE_REFRESH.search(ln)
        if m:
            r["refresh"] = [int(m.group(1)), int(m.group(2))]
    return r


RE_START = re.compile(r"^(\S+) server start \[([^\]]+)\]")
RE_LEARNED = re.compile(r"^(\S+) learned (\S+) in ([\d.]+) s: (.*?)(?:; save (\w+) \(([\d.]+) s, (\d+) bytes\))?$")
RE_SSE = re.compile(r"^\S+\s+(\S+) \+(\d+)s learned: (.*)$")
RE_LEARN = re.compile(r"^(\S+) learn (\S+) \(lesson (\d+)")


def driver_runs(root):
    out = []
    for run in sorted(os.listdir(root)):
        rl = os.path.join(root, run, "run.log")
        if not os.path.isdir(os.path.join(root, run)) or not os.path.exists(rl):
            continue
        phase, lessons = None, {}
        with open(rl, errors="replace") as f:
            for line in f:
                line = line.rstrip("\n")
                m = RE_START.match(line)
                if m:
                    phase = m.group(2).split(" ")[0]
                    lessons.setdefault(phase, [])
                    continue
                m = RE_LEARN.match(line)
                if m:
                    lessons[phase].append({"label": m.group(2), "lesson": int(m.group(3))})
                    continue
                m = RE_SSE.match(line)
                if m and lessons.get(phase) and lessons[phase][-1]["label"] == m.group(1):
                    lessons[phase][-1]["sse_learned"] = int(m.group(2))
                    lessons[phase][-1]["event"] = m.group(3)
                    continue
                m = RE_LEARNED.match(line)
                if m and lessons.get(phase) and lessons[phase][-1]["label"] == m.group(2):
                    d = lessons[phase][-1]
                    d["total"] = float(m.group(3))
                    d["outcome"] = m.group(4)
                    if m.group(5):
                        d["drv_save_state"], d["drv_save_s"], d["drv_save_bytes"] = m.group(5), float(m.group(6)), int(m.group(7))
        for phase, ls in lessons.items():
            ls = [x for x in ls if "total" in x]
            if not ls:
                continue
            log = None
            for sub in ("phases", "steps"):
                p = os.path.join(root, run, sub, phase, "server.log")
                if os.path.exists(p):
                    log = p
            if log is None:
                print("no server log for", run, phase, file=sys.stderr)
                continue
            segs = segments(log)
            if len(segs) != len(ls):
                print("segment mismatch", run, phase, len(segs), len(ls), file=sys.stderr)
            for i, (d, seg) in enumerate(zip(ls, segs)):
                d.update(parse_segment(seg))
                d.update({"run": run, "phase": phase, "nth_in_server": i + 1, "build": "1.0.8 (tf108)", "chat_alongside": False, "source": "driver"})
                out.append(d)
    return out


def live_runs(live, r109):
    """The live comparisons: ck*.json give the lesson's total (and polls of the graph's phase); serve*.log the parts."""
    out = []
    specs = []
    for name, build in (("before", "1.0.8 0efbd447"), ("after", "#787 on 1.0.8 07189f62"), ("before-p2", "1.0.8 0efbd447"), ("after-p2", "#787 on 1.0.8 07189f62")):
        specs.append((os.path.join(live, name), build, True, [("serve.log", ["ck%d.json" % k for k in range(1, 5)])]))
    for name, build in (("p3-before", "1.0.8 0efbd447"), ("p3-tip", "branch tip on 1.0.8"), ("p3-mutant", "mutant on 1.0.8"), ("p3-inst", "instrumented on 1.0.8")):
        specs.append((os.path.join(live, name), build, False, [("serve-1.log", ["ckA.json"]), ("serve-2.log", ["ckB1.json", "ckB2.json"])]))
    for name, build in (("p3-before", "1.0.9 c36b536d"), ("p3-after", "#787 on 1.0.9 76fe1de4")):
        specs.append((os.path.join(r109, name), build, False, [("serve-1.log", ["ckA.json"]), ("serve-2.log", ["ckB1.json", "ckB2.json"])]))
    for d, build, chat, logs in specs:
        for logname, cks in logs:
            p = os.path.join(d, logname)
            if not os.path.exists(p):
                continue
            segs = segments(p)
            cks = [c for c in cks if os.path.exists(os.path.join(d, c))]
            if len(segs) != len(cks):
                print("live mismatch", d, logname, len(segs), len(cks), file=sys.stderr)
            for i, (seg, ck) in enumerate(zip(segs, cks)):
                j = json.load(open(os.path.join(d, ck)))
                rec = parse_segment(seg)
                rec.update({"run": "live/" + os.path.basename(d) if "r109" not in d else "r109/" + os.path.basename(d), "phase": logname, "label": ck, "nth_in_server": i + 1, "build": build, "chat_alongside": chat, "source": "live"})
                if "learn_s" in j:
                    rec["total"] = j["learn_s"]
                ev = j.get("raw")
                if ev:
                    rec["sse_learned"] = next((t for t, s in ev if "event: learned" in s), None)
                if j.get("polls"):
                    rec["polls"] = j["polls"]
                out.append(rec)
    return out


if __name__ == "__main__":
    recs = [x for x in driver_runs(sys.argv[1]) if True] + live_runs(sys.argv[2], sys.argv[3])
    p3 = {"live/p3-before": [194, 153, 145], "live/p3-tip": [193, 147, 143], "live/p3-mutant": [206, 147, 143], "live/p3-inst": [197, 147, 142],
          "r109/p3-before": [184, 146, 138], "r109/p3-after": [187, 142, 135]}
    for x in recs:
        if x["run"] in p3 and x["label"] in ("ckA.json", "ckB1.json", "ckB2.json"):
            x["total"] = float(p3[x["run"]][["ckA.json", "ckB1.json", "ckB2.json"].index(x["label"])])
            x["total_rounded"] = True
    json.dump(recs, sys.stdout)
