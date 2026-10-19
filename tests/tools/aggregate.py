#!/usr/bin/env python3
"""Aggregate repeated runs of one test set: every result as "k of n", per question, per phase and per lesson.

    python3 tools/aggregate.py LABEL=RUN_FOLDER LABEL=RUN_FOLDER ... [--json out.json] [--out report.txt]
    python3 tools/aggregate.py --selftest

RUN_FOLDER is a folder `zengine/lwrun.py` wrote (set.json, summary.json, phases/STEP/answers.jsonl). The folders are
only read. LABEL names the run in the report (use a machine type, not a host name); a bare RUN_FOLDER is labelled by
its folder name. Every run must be of the same set (the same set.json), and no run may be given twice: two folders
with the same start time and host are one run copied, and one folder given under two labels is one run; both are
refused.

Each step's answers (baseline, teach, del-A.zero, del-A.reteach, ...; teach.restart for answers.restart.jsonl) are
judged with the current judge (zengine/lwlogic.py). answers.as-PHASE.jsonl files, the same answers judged as another
phase because lessons were refused, are left out, so no answer counts twice.

The report gives:
  lessons       for each lesson, how many of the tries were recalled: every teach and reteach step of every run
  per phase     for each step and thinking mode, every score (all, the buckets, and the split by the right answer:
                a value, no, don't know) as k of n answers, and each run's own count
  per question  for each step, question and thinking mode, k of n runs passed, and each run's verdict
k counts passes only: an unclear answer is not a pass. [lo, hi] is the exact (Clopper-Pearson) 95% interval for the
rate. Per question and per lesson the n tries are separate runs. Per phase the answers of one run are not separate
tries (one model copy answers them all), so that interval is only a rough range.
"""
import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "zengine"))
import lwlogic as L  # noqa: E402  (the judge)


# ------------------------------------------------------------------------------------------------- intervals
def binom_cdf(k, n, p):
    """P(X <= k) for X ~ Binomial(n, p), summed in logs. The plain sum of comb(n, i) * p**i * (1-p)**(n-i) overflows
    a float from about n = 1030 (comb(1200, 600) has 360 digits), which a per-phase count over many runs reaches."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    lp, lq, ln = math.log(p), math.log1p(-p), math.lgamma(n + 1)
    terms = [ln - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq for i in range(k + 1)]
    top = max(terms)
    return min(1.0, math.exp(top) * math.fsum(math.exp(t - top) for t in terms))


def solve(f, target, increasing):
    """The p in [0, 1] where the monotone f(p) = target, by bisection."""
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if (f(mid) < target) == increasing:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def cp_interval(k, n, conf=0.95):
    """The exact (Clopper-Pearson) interval for a rate seen k times in n: the lowest p whose chance of k or more is
    (1 - conf) / 2, and the highest p whose chance of k or fewer is (1 - conf) / 2. None when n is 0."""
    if n == 0:
        return None
    a = (1 - conf) / 2
    lo = 0.0 if k == 0 else solve(lambda p: 1 - binom_cdf(k - 1, n, p), a, True)
    hi = 1.0 if k == n else solve(lambda p: binom_cdf(k, n, p), a, False)
    return lo, hi


def kofn(k, n, ci=True):
    if not n:
        return "-"
    iv = cp_interval(k, n) if ci else None
    return f"{k} of {n}" + (f" [{iv[0]:.2f}, {iv[1]:.2f}]" if iv else "")


def is_pass(verdict):
    return verdict == "pass"


# --------------------------------------------------------------------------------------------------- reading
class RunError(Exception):
    pass


def canonical(S):
    return hashlib.sha256(json.dumps(S, sort_keys=True).encode()).hexdigest()


def answer_files(run):
    """{step: path} for a run folder: phases/STEP/answers.jsonl as STEP, answers.X.jsonl as STEP.X; the as-PHASE files
    (the same answers judged as another phase) are left out and returned apart."""
    steps, skipped = {}, []
    ph = os.path.join(run, "phases")
    for step in sorted(os.listdir(ph)) if os.path.isdir(ph) else []:
        d = os.path.join(ph, step)
        for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            if not (f.startswith("answers") and f.endswith(".jsonl")):
                continue
            mid = f[len("answers"):-len(".jsonl")].strip(".")
            if mid.startswith("as-"):
                skipped.append(os.path.join(step, f))
                continue
            steps[step + ("." + mid if mid else "")] = os.path.join(d, f)
    return steps, skipped


def lessons_of(run):
    """[(step, lesson id, text, recalled)] from every step's state.json: each lesson the step taught."""
    out = []
    ph = os.path.join(run, "phases")
    for step in sorted(os.listdir(ph)) if os.path.isdir(ph) else []:
        p = os.path.join(ph, step, "state.json")
        if not os.path.exists(p):
            continue
        for l in json.load(open(p)).get("lessons") or []:
            if l.get("done", True) and "recalled" in l:
                out.append((step, l.get("lesson"), l.get("text"), bool(l.get("recalled"))))
    return out


def load_runs(specs):
    """[(label, folder, set, summary)]; refuses runs of different sets and a run given twice."""
    runs, seen, ref, folders = [], {}, None, {}
    for spec in specs:
        label, run = spec.split("=", 1) if "=" in spec and not os.path.isdir(spec) else (None, spec)
        label = label or os.path.basename(os.path.normpath(run))
        real = os.path.realpath(run)
        if real in folders:
            raise RunError(f"{label} and {folders[real]} are the same folder: a run given twice would count twice")
        folders[real] = label
        sp = os.path.join(run, "set.json")
        if not os.path.exists(sp):
            raise RunError(f"{run}: no set.json, not a run folder")
        S = json.load(open(sp))
        summ = json.load(open(os.path.join(run, "summary.json"))) if os.path.exists(os.path.join(run, "summary.json")) else {}
        if ref is None:
            ref = (S.get("id"), canonical(S))
        elif (S.get("id"), canonical(S)) != ref:
            raise RunError(f"{label}: a run of {S.get('id')} (or an edited set file), not of {ref[0]}: only repeats of "
                           f"one set can be aggregated")
        ident = (summ.get("started"), summ.get("host"))
        if ident[0] and ident in seen:
            raise RunError(f"{label} and {seen[ident]} are the same run (started {ident[0]} on the same host): a "
                           f"run given twice would count twice")
        seen[ident] = label
        if label in [r[0] for r in runs]:
            raise RunError(f"label {label} given twice")
        runs.append((label, run, S, summ))
    if not runs:
        raise RunError("no runs given")
    return runs


# ------------------------------------------------------------------------------------------------ aggregating
def aggregate(specs):
    runs = load_runs(specs)
    S = runs[0][2]
    phases = {p["name"] for p in S["phases"]}
    verdicts = {}      # (step, qid, thinking) -> {label: verdict}
    answers = {}       # (step, thinking) -> {label: [results]}
    skipped, lessons = {}, {}
    for label, run, _, _ in runs:
        steps, sk = answer_files(run)
        skipped[label] = sk
        for step, path in steps.items():
            rows = L.load_answers([path], phases)
            res, _, _ = L.judge(S, rows)
            for x in res:
                verdicts.setdefault((step, x["qid"], x["thinking"]), {})[label] = x
                answers.setdefault((step, x["thinking"]), {}).setdefault(label, []).append(x)
        for step, lid, text, rec in lessons_of(run):
            lessons.setdefault(lid, {"text": text, "tries": []})["tries"].append((label, step, rec))
    labels = [r[0] for r in runs]
    out = {"set": S["id"], "runs": [{"label": l, "folder": os.path.abspath(r), "started": s.get("started")}
                                    for l, r, _, s in runs],
           "skipped_files": skipped, "lessons": [], "phases": [], "questions": []}
    for lid in sorted(lessons, key=lambda x: (x is None, x)):
        t = lessons[lid]["tries"]
        k = sum(r for _, _, r in t)
        out["lessons"].append({"lesson": lid, "text": lessons[lid]["text"], "k": k, "n": len(t),
                               "interval": cp_interval(k, len(t)),
                               "tries": [{"run": a, "step": b, "recalled": c} for a, b, c in t]})
    order = {p["name"]: i for i, p in enumerate(S["phases"])}
    step_key = lambda st: (order.get(st.split(".")[0], 99), st)
    for step in sorted({s for s, _ in answers}, key=step_key):
        for th in (False, True):
            per = answers.get((step, th))
            if not per:
                continue
            scores = {}
            groups = [("all", lambda x: True)] + [(bk, lambda x, bk=bk: bk in x["buckets"]) for bk in L.BUCKETS] \
                + [(f"right: {L.CLASS_WORDS[c]}", lambda x, c=c: x["class"] == c) for c in L.CLASSES]
            for name, f in groups:
                runs_kn = {lab: (sum(is_pass(x["verdict"]) for x in xs if f(x)), sum(1 for x in xs if f(x)))
                           for lab, xs in per.items()}
                k, n = sum(v[0] for v in runs_kn.values()), sum(v[1] for v in runs_kn.values())
                if n:
                    scores[name] = {"k": k, "n": n, "interval": cp_interval(k, n), "runs": runs_kn}
            out["phases"].append({"step": step, "thinking": th, "runs": sorted(per, key=labels.index), "scores": scores})
    qorder = {q["qid"]: i for i, q in enumerate(S["questions"])}
    for (step, qid, th) in sorted(verdicts, key=lambda k: (step_key(k[0]), qorder.get(k[1], 999), k[2])):
        v = verdicts[(step, qid, th)]
        k, n = sum(is_pass(x["verdict"]) for x in v.values()), len(v)
        x0 = next(iter(v.values()))
        out["questions"].append({"step": step, "qid": qid, "thinking": th, "question": x0["question"],
                                 "expected": x0["expected"], "class": x0["class"], "k": k, "n": n,
                                 "interval": cp_interval(k, n),
                                 "runs": {lab: v[lab]["verdict"] for lab in labels if lab in v}})
    return out


def report(A):
    iv = lambda d: f" [{d['interval'][0]:.2f}, {d['interval'][1]:.2f}]" if d.get("interval") else ""
    labels = [r["label"] for r in A["runs"]]
    L_ = [f"Aggregate of {len(labels)} runs of {A['set']}: " + ", ".join(
        f"{r['label']} (started {r['started']})" if r["started"] else r["label"] for r in A["runs"]),
        "k of n: k passes in n tries; an unclear answer is not a pass. [lo, hi]: the exact (Clopper-Pearson) 95% "
        "interval. Per question and per lesson the tries are separate runs; per phase the answers of one run are not "
        "separate tries, so read that interval as a rough range.", ""]
    sk = {k: v for k, v in A["skipped_files"].items() if v}
    if sk:
        L_.append("Left out (the same answers judged as another phase): "
                  + "; ".join(f"{k}: {', '.join(v)}" for k, v in sk.items()))
        L_.append("")
    if A["lessons"]:
        L_.append("Lessons, recalled in every step that taught them (teach and reteach copies):")
        for l in A["lessons"]:
            tries = ", ".join(f"{t['run']} {t['step']} {'yes' if t['recalled'] else 'no'}" for t in l["tries"])
            L_.append(f"   L{l['lesson']} \"{l['text']}\"  recalled {l['k']} of {l['n']}{iv(l)}  ({tries})")
        L_.append("")
    for ph in A["phases"]:
        L_.append(f"== {ph['step']}, thinking {'on' if ph['thinking'] else 'off'}: {len(ph['runs'])} runs "
                  f"({', '.join(ph['runs'])})")
        for name, d in ph["scores"].items():
            per = ", ".join(f"{d['runs'][r][0]}/{d['runs'][r][1]}" for r in ph["runs"] if r in d["runs"])
            L_.append(f"   {name.replace('_', ' '):20} {d['k']} of {d['n']}{iv(d)}   per run: {per}")
    L_.append("")
    L_.append("== per question: k of n runs passed; each run's verdict (P pass, F fail, U unclear, . not asked)")
    for q in A["questions"]:
        marks = "".join({"pass": "P", "fail": "F"}.get(q["runs"].get(r), "U" if r in q["runs"] else ".") for r in labels)
        L_.append(f"   {q['step']:16} {q['qid']} {'think' if q['thinking'] else 'plain'} {q['k']} of {q['n']}{iv(q)}"
                  f"  {marks}  \"{q['question']}\" expected {q['expected']}")
    return "\n".join(L_) + "\n"


# ------------------------------------------------------------------------------------------------- self-test
def selftest():
    """Synthetic run folders with known answers: the counts, the intervals and both refusals, each with a negative
    control. Needs no model and no run records."""
    C = L.Check()
    tmp = tempfile.mkdtemp(prefix="aggregate-selftest-")
    try:
        S = L.generate(1, 1)
        S2 = L.generate(2, 1)

        def make(name, SS, started, teach, recalled, delete=True):
            d = os.path.join(tmp, name)
            os.makedirs(os.path.join(d, "phases", "teach"))
            json.dump(SS, open(os.path.join(d, "set.json"), "w"))
            json.dump({"started": started, "host": "box"} if started else {}, open(os.path.join(d, "summary.json"), "w"))
            rows = [{"qid": q["qid"], "phase": "teach", "thinking": th, "question": q["text"],
                     "content": teach.get(q["qid"], "I don't know.")} for q in SS["questions"] for th in (False, True)]
            open(os.path.join(d, "phases", "teach", "answers.jsonl"), "w").write(
                "".join(json.dumps(r) + "\n" for r in rows))
            # the same answers judged as another phase: must be left out
            open(os.path.join(d, "phases", "teach", "answers.as-baseline.jsonl"), "w").write(
                "".join(json.dumps(dict(r, phase="baseline")) + "\n" for r in rows))
            # lesson 1 is a try; a lesson not done (an interrupted step) and one with no "recalled" are not
            json.dump({"lessons": [{"lesson": 1, "text": SS["lessons"][0]["text"], "recalled": recalled, "done": True},
                                   {"lesson": 2, "text": "not done", "recalled": True, "done": False},
                                   {"lesson": 3, "text": "no recalled field", "done": True}]},
                      open(os.path.join(d, "phases", "teach", "state.json"), "w"))
            if delete:
                os.makedirs(os.path.join(d, "phases", "del-A.reteach"))
                open(os.path.join(d, "phases", "del-A.reteach", "answers.jsonl"), "w").write("".join(
                    json.dumps({"qid": q["qid"], "phase": "del-A", "thinking": False, "content": "I don't know."}) + "\n"
                    for q in SS["questions"]))
            return d

        r1 = make("r1", S, "2026-10-12T10:00:00Z", {"Q01": "Your favourite colour is blue.", "Q04": "Yes."}, True)
        r2 = make("r2", S, "2026-10-12T11:00:00Z", {"Q01": "I don't know.", "Q04": "Yes."}, False)
        r3 = make("r3", S, "2026-10-12T12:00:00Z", {"Q01": "Your favourite colour is probably blue.", "Q04": "No."},
                  True, delete=False)
        A = aggregate([f"a={r1}", f"b={r2}", f"c={r3}"])
        q = {(x["step"], x["qid"], x["thinking"]): x for x in A["questions"]}
        ph = {(x["step"], x["thinking"]): x for x in A["phases"]}
        want = {("teach", "Q01", False): (1, 3, {"a": "pass", "b": "fail", "c": "unclear"}),
                ("teach", "Q04", False): (2, 3, None), ("teach", "Q04", True): (2, 3, None),
                ("del-A.reteach", "Q01", False): (2, 2, None)}
        bad = [(k, q.get(k)) for k, (kk, nn, rv) in want.items()
               if not q.get(k) or (q[k]["k"], q[k]["n"]) != (kk, nn) or (rv and q[k]["runs"] != rv)]
        rec = A["lessons"][0] if len(A["lessons"]) == 1 else {"lessons": [x["lesson"] for x in A["lessons"]]}
        tr = ph[("teach", False)]["scores"]["recall"]
        ok = (not bad and (rec.get("k"), rec.get("n")) == (2, 3) and ("teach", True) in ph
              and ("del-A.reteach", True) not in ph and tr["k"] == 1 and tr["n"] == 9
              and tr["runs"] == {"a": (1, 3), "b": (0, 3), "c": (0, 3)} and all(A["skipped_files"][x] for x in "abc")
              and not [s for s, _ in ph if s.startswith("teach.as")])
        global is_pass
        keep = is_pass
        is_pass = lambda v: v in ("pass", "unclear")             # counts unclear as a pass
        try:
            Ab = aggregate([f"a={r1}", f"b={r2}", f"c={r3}"])
        finally:
            is_pass = keep
        qb = {(x["step"], x["qid"], x["thinking"]): x for x in Ab["questions"]}
        C.add("aggregate: every result as k of n per question, per phase and per lesson, over three synthetic runs; "
              "a step only some runs have counts only those; as-PHASE files are left out",
              ok and qb[("teach", "Q01", False)]["k"] == 2, len(want) + 3,
              (f"wrong: {bad}; lesson {rec}; recall {tr}; " if not ok else "") + "control: counting unclear as a pass "
              f"gives Q01 {qb[('teach', 'Q01', False)]['k']} of 3, not 1 of 3")

        # refusals: the same run twice (a copy, or one folder under two labels, also with no start time on file), and
        # a run of another set
        dup = make("r1-copy", S, "2026-10-12T10:00:00Z", {}, True)
        other = make("other", S2, "2026-10-12T13:00:00Z", {}, True)
        nostart = make("no-start", S, None, {}, True)
        refused = []
        for specs in ([f"a={r1}", f"b={dup}"], [f"a={r1}", f"b={other}"], [f"a={nostart}", f"b={nostart}"]):
            try:
                aggregate(specs)
            except RunError as e:
                refused.append(str(e)[:60])
        okr = len(refused) == 3
        try:
            n_ok = len(aggregate([f"a={nostart}", f"b={r1}"])["runs"])
        except RunError:
            n_ok = 0
        C.add("aggregate: refuses one run given twice (same start and host, or the same folder under two labels) and a "
              "run of another set; control: two different runs of the same set are accepted, also one with no start "
              "time on file", okr and bool(A["questions"]) and n_ok == 2, 5,
              "; ".join(refused) if not okr else "")

        # intervals: known Clopper-Pearson values; control: the normal (Wald) interval misses them
        # the two large ones were computed apart, with 50-digit decimals and exact binomial coefficients
        known = [((0, 5), (0.0, 0.5218)), ((5, 5), (0.4782, 1.0)), ((3, 10), (0.0667, 0.6525)),
                 ((1, 2), (0.0126, 0.9874)), ((7, 7), (0.5904, 1.0)), ((600, 1200), (0.471323, 0.528677)),
                 ((3, 1500), (0.000413, 0.005834))]
        badi = [(kn, cp_interval(*kn)) for kn, (lo, hi) in known
                if abs(cp_interval(*kn)[0] - lo) > 5e-4 or abs(cp_interval(*kn)[1] - hi) > 5e-4]

        def wald(k, n):
            p = k / n
            h = 1.96 * math.sqrt(p * (1 - p) / n)
            return max(0.0, p - h), min(1.0, p + h)
        wald_bad = sum(abs(wald(*kn)[0] - lo) > 5e-4 or abs(wald(*kn)[1] - hi) > 5e-4 for kn, (lo, hi) in known)

        def plain_cdf(k, n, p):                        # the first version: overflows a float from about n = 1030
            return 0.0 if k < 0 else 1.0 if k >= n else sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i)
                                                             for i in range(k + 1))
        global binom_cdf
        keep_cdf, binom_cdf = binom_cdf, plain_cdf
        try:
            cp_interval(600, 1200)
            plain_fails = "no error"
        except OverflowError as e:
            plain_fails = f"OverflowError ({e})"
        finally:
            binom_cdf = keep_cdf
        C.add("aggregate: exact (Clopper-Pearson) 95% intervals match known values, also for 600 of 1200; controls: the "
              "normal-approximation interval misses them, and the plain binomial sum cannot compute 600 of 1200",
              not badi and wald_bad > 0 and plain_fails != "no error", len(known) + 1,
              f"{badi}; " if badi else f"the normal interval misses {wald_bad} of {len(known)} (it gives [0, 0] for 0 of 5); "
              f"the plain sum: {plain_fails}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    ok = all(r[1] for r in C.rows)
    print(f"aggregate selftest {'PASSED' if ok else 'FAILED'}: {sum(r[1] for r in C.rows)}/{len(C.rows)} checks")
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="LABEL=RUN_FOLDER (or RUN_FOLDER)")
    ap.add_argument("--json", help="write the aggregate here")
    ap.add_argument("--out", help="write the report here (default: print it)")
    ap.add_argument("--selftest", action="store_true", help="run the self-test (no run records needed)")
    a = ap.parse_args(argv)
    if a.selftest:
        return 0 if selftest() else 1
    if not a.runs:
        ap.error("give run folders, or --selftest")
    try:
        A = aggregate(a.runs)
    except RunError as e:
        print(f"aggregate: {e}", file=sys.stderr)
        return 2
    txt = report(A)
    if a.out:
        open(a.out, "w", encoding="utf-8").write(txt)
        print(f"wrote {a.out}")
    else:
        print(txt, end="")
    if a.json:
        json.dump(A, open(a.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
