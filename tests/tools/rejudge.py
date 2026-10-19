#!/usr/bin/env python3
"""Re-judge stored run records with the current judge, and list every verdict that changed.

    python3 tools/rejudge.py --old OLD_lwlogic.py --out docs/rejudge LABEL=RUN_FOLDER [LABEL=RUN_FOLDER ...]
    python3 tools/rejudge.py --old OLD_lwlogic.py --split-only --out docs/rejudge/split LABEL=RUN_FOLDER ...
    python3 tools/rejudge.py --old OLD_lwlogic.py --fixes likely,truth,entity,first --out DIR LABEL=RUN_FOLDER ...

RUN_FOLDER is a folder `zengine/lwrun.py` wrote (set.json, summary.json, phases/*/answers*.jsonl). It is only read,
never written. LABEL names the run in the reports (use a machine type, not a host name). --old is an earlier copy
of lwlogic.py, for example `git show main:zengine/lwlogic.py > /tmp/old.py`.

Writes into --out:
  LABEL.FAMILY.txt      the new judge's report on the runner's combined judge input of that delete mode (baseline,
                        teach and the family's delete phases), with kept_stable
  LABEL.STEP.txt        the new judge's report on an answer file that is in no combined input (a teach-only run,
                        or answers judged as another phase because lessons were refused)
  CHANGES.md            every verdict that differs between the old and the new judge, with the fix that changed it
                        and an answer excerpt
  SPLIT.md, split.json  the split scores of every report (plan items 2 and 8): each score by what the rule book
                        expects (a value, no, don't know), should change and should stay, "don't know" as precision
                        and recall, the consistency in two parts, paraphrase agreement, the literature's tags and
                        neighbourhood failures; and the two dummy systems on the same sets, as negative controls
Every verdict comes from judging the same answer rows twice, once with each judge. A fix is named as the cause of
a change when switching that one fix off in the new judge gives back the old verdict. --fixes names the fixes to try:
only those the old judge does not have, since switching off a fix both judges share can give back an old verdict by
accident.
"""
import argparse
import glob
import importlib.util
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


NEW = load_module("lwlogic_new", os.path.join(HERE, "..", "zengine", "lwlogic.py"))

# each fix of the new judge, and how to switch it off again (for naming the cause of a change)
FIXES = [
    ("dates", "a dated claim that contradicts the lessons fails as a wrong date",
     {"dated_claims": lambda *a, **k: []}),
    ("who", "reverse questions read entity names, refusals and general answers",
     None),                                            # filled in main(): the old entity reader
    ("self", "\"No, I am a language model ...\" is a refusal, not a no",
     {"SELF_REPLY": re.compile(r"(?!)")}),
    ("guess", "the right answer said as a guess is unclear, not fail",
     {"hedged": lambda g, right: ("fail", g, "a guess, not a recall")}),
    # 13 Oct, branch judge-misreads: four misreadings
    ("likely", "\"It is very unlikely that ...\" is a guess that leans no, not a yes; \"It is very likely ...\" leans yes",
     {"LIKELY": re.compile(r"(?!)")}),
    ("truth", "\"It is false.\", \"That is false.\" and \"False.\" on a plain yes/no question are a no",
     {"TRUTH_PLAIN": False}),
    ("entity", "a value said of another entity of the set (\"Your cat's eyes are grey.\" for my dog) fails",
     {"ENTITY_CHECK": False}),
    ("first", "a first-person answer in the user's words is about the user; the model about itself is a self-reply",
     {"first_person": lambda text: None}),
    # 13 Oct, branch judge-first-person: what a first-person answer also says of the user decides
    ("you-part", "a first-person answer that also says something of the user (\"... You still do.\") is read by that part",
     {"you_part": lambda *x, **k: None}),
]


def switched_off(patch, fn):
    keep = {k: getattr(NEW, k) for k in patch}
    for k, v in patch.items():
        setattr(NEW, k, v)
    try:
        return fn()
    finally:
        for k, v in keep.items():
            setattr(NEW, k, v)


def rows_of(path, S):
    return NEW.load_answers([path], {p["name"] for p in S["phases"]})


def question_of(S, r):
    qs = {q["qid"]: q for q in S["questions"]}
    by_text = {q["text"]: q for q in S["questions"]}
    q = qs.get(r["qid"]) or by_text.get(r.get("question") or "")
    if q and r.get("question") and q["text"] != r["question"]:
        q = by_text.get(r["question"], q)
    return q


def excerpt(t, n=160):
    return " ".join(NEW.clean(t).split())[:n]


def local(run, p):
    """A path from the run's summary.json (written on the machine that ran it) as a path inside RUN_FOLDER."""
    tail = p.split("/phases/", 1)
    return os.path.join(run, "phases", tail[1]) if len(tail) == 2 else None


def md_cell(s):
    return str(s).replace("|", "/").replace("\n", " ")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", required=True, help="an earlier lwlogic.py to compare with")
    ap.add_argument("--old-name", default="the old judge", help="how the reports name the old judge")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fixes", default=",".join(k for k, _, _ in FIXES),
                    help="the fixes to name as causes, comma separated (default: all). Give only the ones the old "
                         "judge does not have: a fix both judges share can restore an old verdict by accident")
    ap.add_argument("--split-only", action="store_true",
                    help="write the reports, SPLIT.md and split.json, not CHANGES.md and changes.json (the changed "
                         "verdicts are still counted and printed)")
    ap.add_argument("runs", nargs="+", help="LABEL=RUN_FOLDER")
    a = ap.parse_args(argv)
    OLD = load_module("lwlogic_old", a.old)
    FIXES[:] = [(k, t, {"extract_entities": lambda text, ents, *x, **kw: OLD.extract_entities(text, ents)})
                if k == "who" else (k, t, patch) for k, t, patch in FIXES]
    chosen = a.fixes.split(",")
    unknown = [k for k in chosen if k not in {f[0] for f in FIXES}]
    if unknown:
        ap.error(f"unknown fixes: {unknown}")
    FIXES[:] = [f for f in FIXES if f[0] in chosen]
    os.makedirs(a.out, exist_ok=True)
    changes, index = [], []
    for spec in a.runs:
        label, run = spec.split("=", 1)
        S = json.load(open(os.path.join(run, "set.json")))
        recompute = NEW.check_expected(S)
        summ = json.load(open(os.path.join(run, "summary.json")))
        files = sorted(glob.glob(os.path.join(run, "phases", "*", "answers*.jsonl")))
        rel = {f: os.path.relpath(f, os.path.join(run, "phases")) for f in files}
        n_rows = n_same_stored = n_stored = 0
        # 1. every answer row, judged by both judges
        for f in files:
            step = rel[f].split(os.sep)[0]
            stored = {}
            vf = os.path.join(os.path.dirname(f), os.path.basename(f).replace("answers", "verdicts").replace(".jsonl", ".json"))
            if os.path.exists(vf):
                for x in json.load(open(vf)).get("results", []):
                    stored.setdefault((x["phase"], x["qid"], x["thinking"]), x["verdict"])
            for r in rows_of(f, S):
                q = question_of(S, r)
                if not q or r["phase"] not in {p["name"] for p in S["phases"]}:
                    continue
                vo, go, wo = OLD.judge_one(S, r["phase"], q, r["content"])
                vn, gn, wn = NEW.judge_one(S, r["phase"], q, r["content"])
                if vo == "skip":
                    continue
                n_rows += 1
                sv = stored.get((r["phase"], q["qid"], r["thinking"]))
                if sv is not None:
                    n_stored += 1
                    n_same_stored += sv == vo
                if vo == vn:
                    continue
                cause = [k for k, _, patch in FIXES
                         if switched_off(patch, lambda: NEW.judge_one(S, r["phase"], q, r["content"]))[0] == vo]
                changes.append({"run": label, "set": S["id"], "file": rel[f].replace(os.sep, "/"), "step": step,
                                "phase": r["phase"], "qid": q["qid"], "thinking": r["thinking"], "question": q["text"],
                                "expected": NEW.words(S["expected"][r["phase"]][q["qid"]]),
                                "old": f"{vo} ({OLD.got_words(go)}{'; ' + wo if wo else ''})",
                                "new": f"{vn} ({NEW.got_words(gn)}{'; ' + wn if wn else ''})",
                                "old_v": vo, "new_v": vn, "cause": cause or ["several"], "excerpt": excerpt(r["content"])})
        # 2. the runner's combined judge inputs, re-judged; then any answer file outside them
        used = set()
        reports = []
        for fam, c in sorted((summ.get("combined") or {}).items()):
            fl = [local(run, p) for p in c.get("files", [])]
            if not fl or not all(x and os.path.exists(x) for x in fl):
                continue
            used.update(fl)
            reports.append((f"{label}.{fam}.txt", fam, fl))
        for f in files:
            if f not in used:
                reports.append((f"{label}.{rel[f].replace(os.sep, '.').replace('.answers', '').replace('.jsonl', '')}.txt",
                                None, [f]))
        for name, fam, fl in reports:
            rows = []
            for f in fl:
                rows += rows_of(f, S)
            res, cons, skipped = NEW.judge(S, rows)
            head = [f"Re-judged with the current judge (lwlogic.py, this branch). Run {label}, set {S['id']}"
                    + (f", combined input of the '{fam}' delete mode" if fam else "") + ".",
                    "Answer files: " + ", ".join(os.path.relpath(f, os.path.join(run, "phases")).replace(os.sep, "/") for f in fl),
                    "The set's expected answers recompute with this oracle: " + ("yes" if recompute else "NO"), ""]
            body = f"judged {len(res)} answers against {S['id']}" + (f" ({skipped} lines skipped)" if skipped else "")
            with open(os.path.join(a.out, name), "w", encoding="utf-8") as fh:
                fh.write("\n".join(head) + body + "\n" + NEW.score_table(S, res, cons) + "\n")
            kept = {k[1]: v for k, v in cons.items() if isinstance(k, tuple) and k[0] == "kept"}
            index.append({"run": label, "report": name, "family": fam, "kept": kept, "results": res, "S": S,
                          "cons": cons})
        print(f"{label}: {n_rows} answers judged by both judges; stored verdicts equal to {a.old_name}: "
              f"{n_same_stored}/{n_stored}; changed: {sum(c['run'] == label for c in changes)}")
    if not a.split_only:
        write_changes(a, changes, index)
    write_split(a, index)


def write_changes(a, changes, index):
    L = ["# Re-judged run records: every changed verdict", "",
         f"Each answer was judged twice: by {a.old_name} and by the judge on this branch. "
         "This page lists every answer whose verdict changed. The run records themselves were only read.", "",
         "A fix is named as the cause when switching that one fix off in the new judge gives back the old verdict.", "",
         "| Fix | What it does | Changed verdicts |", "| --- | --- | --- |"]
    for k, title, _ in FIXES:
        L.append(f"| {k} | {title} | {sum(k in c['cause'] for c in changes)} |")
    other = sum(c["cause"] == ["several"] for c in changes)
    if other:
        L.append(f"| several | more than one fix together | {other} |")
    moves = {}
    for c in changes:
        moves[(c["old_v"], c["new_v"])] = moves.get((c["old_v"], c["new_v"]), 0) + 1
    L += ["", "Moves: " + ", ".join(f"{o} to {n}: {k}" for (o, n), k in sorted(moves.items())) + f". Total {len(changes)}.", ""]
    # kept_stable next to delete_kept, per delete phase of each combined report
    L += ["## kept_stable next to delete_kept", "",
          "delete_kept scores the questions a delete should not touch against the rule book's right answer. "
          "kept_stable compares the questions whose right answer did not change with the same model's own teach "
          "answers: the same verdict, and the very same text. Thinking off / thinking on.", "",
          "| Run | Report | Phase | delete_kept pass | kept_stable same verdict | kept_stable same text |",
          "| --- | --- | --- | --- | --- | --- |"]
    for ix in index:
        if not ix["kept"]:
            continue
        for ph, k in ix["kept"].items():
            dk = []
            for th in (False, True):
                xs = [x for x in ix["results"] if x["phase"] == ph and x["thinking"] == th and "delete_kept" in x["buckets"]]
                dk.append(f"{sum(x['verdict'] == 'pass' for x in xs)}/{len(xs)}" if xs else "-")
            cell = lambda f: " / ".join(f"{k[t][f]}/{k[t]['n']}" if t in k else "-" for t in ("off", "on"))
            L.append(f"| {ix['run']} | {ix['report']} | {ph} | {' / '.join(dk)} | {cell('same_verdict')} | {cell('same_text')} |")
    L += ["", "## Every changed verdict", ""]
    for run in dict.fromkeys(c["run"] for c in changes):
        cs = [c for c in changes if c["run"] == run]
        L += [f"### {run} ({cs[0]['set']}), {len(cs)} changed", "",
              "| File | Phase | Q | Thinking | Question | Expected | Old verdict | New verdict | Fix | Answer (start) |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for c in cs:
            L.append("| " + " | ".join(md_cell(x) for x in [
                c["file"], c["phase"], c["qid"], "on" if c["thinking"] else "off", c["question"], c["expected"],
                c["old"], c["new"], ", ".join(c["cause"]), c["excerpt"]]) + " |")
        L.append("")
    with open(os.path.join(a.out, "CHANGES.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    json.dump([{k: v for k, v in c.items()} for c in changes], open(os.path.join(a.out, "changes.json"), "w"), indent=1)


def cell2(d, f):
    """thinking off / thinking on of one figure."""
    return " / ".join(f(d[t]) if t in d else "-" for t in ("off", "on"))


def write_split(a, index):
    """SPLIT.md and split.json: the split view of every report, and the two dummy systems on the same sets."""
    def one(t):
        return f"{t['pass']}/{t['n']}" + (f" ({t['dk_miss']} dk)" if t["dk_miss"] else "")
    sc = lambda c: lambda m: one(m["split"]["all"][c]) if c in m["split"]["all"] else "-"
    same_pairs = lambda d: NEW.ratio(d["same"], d["pairs"])
    ch = lambda k: lambda m: NEW.split_cell(m["split"].get(k)) if k in m["split"] else "-"
    ab = lambda m: (f"P {NEW.ratio(m['abstention']['said_dk_right'], m['abstention']['said_dk'])} "
                    f"R {NEW.ratio(m['abstention']['said_dk_right'], m['abstention']['dk_expected'])}")
    out, seen, rows, rows2, rows3, rows4 = [], set(), [], [], [], []
    for ix in index:
        S, res, cons = ix["S"], ix["results"], ix["cons"]
        sp = NEW.split_scores(S, res)
        fam = f" ({ix['family']})" if ix["family"] else f" ({ix['report'].split('.', 1)[1].rsplit('.', 1)[0]})"
        out.append({"run": ix["run"], "report": ix["report"], "set": S["id"], "split": sp,
                    "consistency": {k[1]: v for k, v in cons.items() if k[0] == "cons"},
                    "paraphrase": {k[1]: v for k, v in cons.items() if k[0] == "para"}})
        for ph in [p["name"] for p in S["phases"]] + ["all phases"]:
            if ph not in sp:
                continue
            key = (ix["run"], ph, json.dumps(sp[ph], sort_keys=True))
            if ph in ("baseline", "teach") and key in seen:
                continue                                   # the same baseline and teach answers in another report
            seen.add(key)
            name = ph + ("" if ph in ("baseline", "teach", "all phases") else fam)
            m = sp[ph]
            if ph == "all phases":
                rows4.append(f"| {ix['run']} | {ix['report']} | {cell2(m, ab)} | {cell2(m, sc('value'))} |"
                             f" {cell2(m, sc('no'))} | {cell2(m, sc('dont_know'))} |"
                             f" {cell2(m, lambda d: str(d['closed_world']))} |")
                continue
            rows.append(f"| {ix['run']} | {name} | {cell2(m, sc('value'))} | {cell2(m, sc('no'))} | "
                        f"{cell2(m, sc('dont_know'))} | {cell2(m, ch('should_change'))} | {cell2(m, ch('should_stay'))} |"
                        f" {cell2(m, ab)} |")
            co, pa = cons.get(("cons", ph), {}), cons.get(("para", ph), {})
            cpart = lambda part: lambda d: NEW.ratio(d[part]["ok"], d[part]["n"]) if part in d else "-"
            nb = [f"{x} ({t})" for t in ("off", "on") if t in m for x in m[t]["neighbour"]]
            rows2.append(f"| {ix['run']} | {name} | {cell2(co, cpart('value'))} | {cell2(co, cpart('dont_know'))} | "
                         f"{cell2(pa, same_pairs)} | "
                         f"{cell2(pa, lambda d: NEW.ratio(d['all_right'], d['groups']))} | {'; '.join(nb) or '-'} |")
            tg = lambda t: lambda d: NEW.ratio(d["tags"][t]["pass"], d["tags"][t]["n"]) if t in d["tags"] else "-"
            rows3.append(f"| {ix['run']} | {name} | " + " | ".join(cell2(m, tg(t)) for t in NEW.TAGS) + " |")
    # the dummies, on every set the runs used
    dummies, rows5 = {}, []
    for ix in index:
        S = ix["S"]
        if S["id"] in dummies:
            continue
        dummies[S["id"]] = {}
        for says in NEW.DUMMIES:
            res, _, _ = NEW.judge(S, NEW.dummy_rows(S, says))
            sp = NEW.split_scores(S, res)
            dummies[S["id"]][says] = sp
            for ph in [p["name"] for p in S["phases"]]:
                m = sp[ph]["off"]
                xs = [x for x in res if x["phase"] == ph and not x["thinking"]]
                pooled = lambda bk: (f"{sum(x['verdict'] == 'pass' for x in xs if bk in x['buckets'])}/"
                                     f"{sum(1 for x in xs if bk in x['buckets'])}") if any(bk in x["buckets"] for x in xs) else "-"
                rows5.append(f"| {S['id']} | always {NEW.CLASS_WORDS[says] if says != 'yes' else 'yes'} | {ph} | "
                             f"{sum(x['verdict'] == 'pass' for x in xs)}/{len(xs)} | {pooled('delete_affected')} | "
                             f"{sc('value')(m)} | {sc('no')(m)} | {sc('dont_know')(m)} |")
    L = ["# Split scores of the re-judged run records", "",
         "Every answer of the stored run records, judged by the judge on this branch and split by what the rule book "
         "expects. Counted from the run records, which were only read (MEASURED). Each cell is thinking off / thinking "
         "on. `4/7` is 4 right of 7; `(3 dk)` counts the misses where the model said it does not know.", "",
         "- **a value**: the right answer is a value, a set, a yes, one of two options, someone, or the textbook answer.",
         "- **no**: the right answer is no, neither or none. **don't know**: the right answer is \"don't know\".",
         "- **should change**: questions whose right answer differs from the phase before (teach against baseline, a "
         "delete against teach); **should stay**: the others. Each split as above (v, n, ?).",
         "- **\"don't know\" P, R**: precision (of the \"don't know\" answers, how many were right to say it) and "
         "recall (of the questions whose right answer is \"don't know\", how many got it).", "",
         "## By the right answer", "",
         "| Run | Phase | a value | no | don't know | should change | should stay | \"don't know\" P, R |",
         "| --- | --- | --- | --- | --- | --- | --- | --- |"] + rows + [
        "", "## Over all phases of each report", "",
        "Closed world: the fails that would pass if a \"no\" counted where the right answer is \"don't know\" (the "
        "closed-world reading, where anything not said is false; the test suite reads it open-world).", "",
        "| Run | Report | \"don't know\" P, R | a value | no | don't know | closed world |",
        "| --- | --- | --- | --- | --- | --- | --- |"] + rows4 + [
        "", "## Consistency, paraphrases and neighbourhood failures", "",
        "Consistency: yes/no and either-or answers that follow from the model's own answer to every wording of the "
        "plain question, in two parts: the plain answer gave a value, or \"don't know\". Same answer: pairs of wordings "
        "of one question that gave the same answer, right or wrong. All ok: questions asked in several wordings that "
        "were right in every wording. Neighbourhood: failed answers that give a value taught about another entity.", "",
        "| Run | Phase | consistency, plain gave a value | consistency, plain said don't know | same answer | all ok |"
        " neighbourhood failures |", "| --- | --- | --- | --- | --- | --- | --- |"] + rows2 + [
        "", "## The literature's names", "",
        "Pass of n per tag (`zengine/lwlogic.py`, `question_tags`).", "",
        "| Run | Phase | " + " | ".join(NEW.TAGS) + " |", "| --- | --- | --- | --- | --- | --- |"] + rows3 + [
        "", "## The dummy systems on the same sets (negative controls)", "",
        "Two systems that answer every question the same way, judged on each set the runs used. Thinking off only: "
        "both modes give the same answers. Pooled is the score of the whole phase as the reports gave it before.", "",
        "| Set | System | Phase | pooled | delete_affected | a value | no | don't know |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |"] + rows5
    with open(os.path.join(a.out, "SPLIT.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    json.dump({"reports": out, "dummies": dummies}, open(os.path.join(a.out, "split.json"), "w"), separators=(",", ":"))


if __name__ == "__main__":
    sys.exit(main())
