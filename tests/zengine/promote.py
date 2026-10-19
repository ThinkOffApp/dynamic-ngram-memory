#!/usr/bin/env python3
"""Promotion (paper Section 5): notes that earn it move into the pocket's neurons, night by night, on a real
TensorFold server. The design is docs/PROMOTION.md; this is the first build of it.

  python3 zengine/promote.py make-set BASE.json --days 1-10,11-20,21-29,30-32, [--extra EXTRA.json] --out NIGHTS.json
  python3 zengine/promote.py run NIGHTS.json --bin PATH --model-src SNAPSHOT --work DIR --port N
      [--hold 2] [--alone 2] [--thinking off] [--notes-config JSON] [--tolerance 0] [--cleanup]
  python3 zengine/promote.py selftest [--dir DIR]

A nights set is a level 1 to 6 set (lwlogic.py gen) whose lessons are told over days: phase dayK holds the lessons
told up to day K (a checkpoint, not a delete), and --extra adds lessons told later (changes and take-backs of facts
that may have been promoted by then). The rule book gives every question's right answer on every day.

Each day K, with the lessons told so far as notes (zengine/notes.py, written at once):
  system    the serving folder (the source model until a night promotes something) with each question's notes,
            leaving out the notes whose facts the neurons answer alone ("in weights"); probe; judge as phase dayK.
  control   the untaught source with every note (the notes-only arm of Section 7.7); probe; judge.
  qualify   a note qualifies today when it is a current fact (a statement or a change, not a take-back or a dated
            value), at least one question rests on it (notes.needed_notes), and every such question passed in the
            system arm. A note that qualifies on --hold days in a row (default 2) is due for promotion.
Each night K (not after the last day):
  stale     a promoted fact whose note is no longer current (changed or taken back) is stale. Its neurons are not
            zeroed: zeroing can leave the fact or hit a neighbour (Section 7.4). The pocket is rebuilt instead, from
            the untaught source, with the promoted facts that still hold, then the due ones.
  teach     a copy of the serving folder (or of the source, when something is stale) is served with --slide and
            taught each due fact's own sentence (POST /v1/slide/learn), oldest first. TensorFold keeps a fact only
            when its own held-out checks pass; a fact it does not keep stays a note (missed).
  regress   the taught copy, served without --slide, answers today's questions with today's system notes. Any
            question that passed today and does not now (more than --tolerance of them) rejects the night: the copy
            is dropped and the serving folder stays as it was; the due facts are marked rejected. With stale facts,
            a rebuild without the due facts is tried next, and then the untaught source (every fact back to notes).
  alone     on the accepted folder, each promoted fact's questions are asked without its own note. Right --alone
            nights in a row (default 2): the note leaves the prompt from the next day on ("in weights"); it stays in
            the store as the record, and a change or take-back still goes to the notes at once. Wrong once: back
            to the prompt (the streak starts again).
Files: WORK/promote-state.json (state, written after every step), WORK/promote-report.txt (one row per day),
WORK/days/dayK/{system,control}/ and WORK/nights/nightK/{teach-*,regress-*,alone}/ (answers, verdicts, notes.jsonl,
server logs, learn SSE and graphs). The serving and taught folders are lwrun.py copies (copies.json); on Linux
--cleanup removes a copy once nothing serves from it.

It holds no TensorFold code: it talks to the server over HTTP through lwrun.py's Server, learn_one and ask.
"""
import argparse
import copy as _copy
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lwlogic as L  # noqa: E402
import lwrun  # noqa: E402
import notes as nm  # noqa: E402

LOG = lwrun.LOG
FORMAT = "lw-promote/1"
PROMOTABLE_OPS = ("ASSERT", "CHANGE")   # a take-back is not taught as a fact; a dated value has two values
ACTIVE = ("promoted", "in_weights")


class PromoteError(lwrun.RunError):
    pass


# ============================================================================================ the nights set
def parse_days(spec, n_lessons):
    """"1-10,11-20,21-29,30-32," -> [[1..10], [11..20], [21..29], [30..32], []]: each day's new lesson ids; an empty
    item is a day with no new lesson (a measurement after the night before it)."""
    days = []
    for part in spec.split(","):
        part = part.strip()
        ids = []
        if part:
            a, _, b = part.partition("-")
            lo, hi = int(a), int(b or a)
            if lo > hi:
                raise PromoteError(f"--days: {part!r} runs backwards")
            ids = list(range(lo, hi + 1))
        days.append(ids)
    seen = [i for d in days for i in d]
    if sorted(seen) != list(range(1, n_lessons + 1)) or len(seen) != len(set(seen)):
        raise PromoteError(f"--days must tell every lesson 1..{n_lessons} once, in order; got {spec!r}")
    if seen != sorted(seen):
        raise PromoteError("--days must tell the lessons in their order")
    if len(days) < 2:
        raise PromoteError("--days needs two days or more: a night sits between two days")
    return days


def make_set(base, extra, days_spec):
    """The nights set: the base set's lessons and questions, the extra lessons after them (told later), phases
    baseline, day1..dayN (each the lessons told so far; kind checkpoint) and teach (every lesson), right answers
    from the rule book for each."""
    S = _copy.deepcopy(base)
    if S.get("format") != "lw-test-set/1" or S.get("l7"):
        raise PromoteError("make-set takes a level 1 to 6 set from lwlogic.py gen")
    step = 100
    last = max(l["told"] for l in S["lessons"])
    for x in extra or []:
        i = len(S["lessons"]) + 1
        for k in ("op", "e", "a", "text"):
            if k not in x:
                raise PromoteError(f"extra lesson {x}: {k} missing")
        if x["e"] not in S["entities"] and x["e"] != "me":
            raise PromoteError(f"extra lesson {x}: entity {x['e']!r} is not one of the set's")
        if x["a"] not in S["schema"]:
            raise PromoteError(f"extra lesson {x}: attribute {x['a']!r} is not in the set's schema")
        last += step
        S["lessons"].append(dict(x, id=i, name=f"L{i}", told=last, para=x.get("para", 0)))
    days = parse_days(days_spec, len(S["lessons"]))
    ids = [l["id"] for l in S["lessons"]]

    def phase(name, store, kind, note):
        return {"name": name, "deleted": [i for i in ids if i not in store], "note": note, "store": list(store),
                "kind": kind}

    phases, told = [phase("baseline", [], "baseline", "before any teaching (the original model)")], []
    for k, d in enumerate(days, 1):
        told += d
        phases.append(phase(f"day{k}", told, "checkpoint", f"day {k}: lessons {d[0]} to {d[-1]} told" if d else
                            f"day {k}: nothing new told (a measurement after night {k - 1})"))
    phases.append(phase("teach", ids, "teach", "every lesson (the store of the last day)"))
    S["phases"] = phases
    S["id"] = f"{base['id']}-nights"
    S["nights"] = {"format": FORMAT, "base": base["id"], "days": days, "extra": len(extra or [])}
    S.setdefault("notes", [])
    S["notes"] = list(S["notes"]) + [f"Nights set (zengine/promote.py): lessons told over {len(days)} days, "
                                     f"{len(extra or [])} of them added to {base['id']}."]
    L.fill_expected(S)
    return S


# ============================================================================================ the run
def night_plans(stale, keep, due):
    """What a night tries, in order. Nothing stale: add the due facts to the serving folder (or nothing to do). Stale
    facts: rebuild from the untaught source with the facts that still hold and the due ones; then without the due
    ones; then the untaught source itself, always last and always taken (a review of f0cbaff: with nothing
    left to keep, the old plan list had no last resort and the stale pocket kept serving)."""
    if not stale:
        return [("add", list(due))] if due else []
    plans = []
    if keep or due:
        plans.append(("rebuild", list(keep) + list(due)))
    if keep and due:
        plans.append(("rebuild-old", list(keep)))
    return plans + [("source", [])]


class Nights:
    def __init__(self, a):
        self.a = a
        argv = [a.set, "--bin", a.bin, "--model-src", a.model_src, "--work", a.work, "--port", str(a.port),
                "--phases", "baseline", "--thinking", a.thinking, "--reference", "notes", "--reference-of", "teach",
                "--max-tokens-on", str(a.max_tokens_on)]
        if a.notes_config:
            argv += ["--notes-config", a.notes_config]
        if a.notes_tokenizer:
            argv += ["--notes-tokenizer", a.notes_tokenizer]
        if a.cleanup:
            argv += ["--cleanup"]
        for x in a.server_arg or []:
            argv += ["--server-arg", x]
        self.run = lwrun.Run(lwrun.arg_parser().parse_args(argv))
        self.S = self.run.S
        if not self.S.get("nights"):
            raise PromoteError(f"{a.set}: not a nights set (make it with `promote.py make-set`)")
        self.days = self.S["nights"]["days"]
        self.work = self.run.work
        self.nm, self.tok, self.cfg = self.run.notes_setup()
        self.by_id = {l["id"]: l for l in self.S["lessons"]}
        self.state_path = os.path.join(self.work, "promote-state.json")
        self.st = lwrun.read_json(self.state_path, None) or {
            "format": FORMAT, "set": self.S["id"], "hold": a.hold, "alone": a.alone, "tolerance": a.tolerance,
            "folder": None, "graph": None, "facts": {}, "days": {}, "nights": {}}
        for k in ("hold", "alone", "tolerance"):
            if self.st[k] != getattr(a, k):
                raise PromoteError(f"{self.state_path} was started with --{k} {self.st[k]}, not {getattr(a, k)}")
        self.thinking = self.run.thinking

    # ---------------------------------------------------------------------------------- small helpers
    def save(self):
        lwrun.write_json(self.state_path, self.st)

    def store_of(self, k):
        return [i for d in self.days[:k] for i in d]

    def fact(self, lid):
        return self.st["facts"].setdefault(str(lid), {"lesson": lid, "text": self.by_id[lid]["text"],
                                                       "status": "note", "streak": 0, "alone": 0, "log": []})

    def note(self, f, what, k):
        f["log"].append(f"{what} ({k})")

    def serving(self):
        return self.st["folder"] or self.run.src

    def notes_day(self, k):
        """The store of day k and, per lesson, its note (id, state, kind) or None (intake wrote nothing)."""
        ns, _w = self.nm.set_store(self.S, self.store_of(k), self.tok, self.cfg)
        by_src = {}
        for n in ns.notes_of("user"):
            by_src[n["source"]] = {"id": n["id"], "state": n["state"], "kind": n["kind"], "op": n["op"]}
        return ns, by_src

    def exclude_ids(self, by_src, extra=()):
        out = {by_src[int(l)]["id"] for l, f in self.st["facts"].items()
               if f["status"] == "in_weights" and by_src.get(int(l)) and by_src[int(l)]["state"] == nm.CURRENT}
        return out | set(extra)

    def blocks(self, ns, exclude=()):
        ret = {q["qid"]: ns.retrieve("user", q["text"], exclude=exclude) for q in self.S["questions"]}
        return ret, {qid: (r["block"] or None) for qid, r in ret.items()}

    def write_notes(self, d, ret):
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "notes.jsonl"), "w", encoding="utf-8") as f:
            for q in self.S["questions"]:
                r = ret[q["qid"]]
                f.write(json.dumps({"qid": q["qid"], "notes": r["ids"], "tokens": r["tokens"], "block": r["block"]},
                                   ensure_ascii=False) + "\n")

    def serve(self, folder, slide, d, label, graph=None):
        os.makedirs(d, exist_ok=True)
        return lwrun.Server(self.run, folder, slide, os.path.join(d, "server.log"), graph, label=label).start()

    def ask(self, srv, d, phase, systems, qids=None):
        """Ask the questions (all, or `qids`) not yet on file in d/answers.jsonl, then judge them as `phase`.
        Returns {(qid, thinking): verdict}."""
        os.makedirs(d, exist_ok=True)
        path, errs = os.path.join(d, "answers.jsonl"), os.path.join(d, "errors.jsonl")
        done = lwrun.answered_pairs(path)
        want = [q for q in self.S["questions"] if qids is None or q["qid"] in qids]
        a = self.run.args
        with open(path, "a", encoding="utf-8") as f:
            for th in self.thinking:
                for q in want:
                    if (q["qid"], th) in done:
                        continue
                    sysp = systems.get(q["qid"])
                    for attempt in range(a.chat_retries + 1):
                        try:
                            r = lwrun.ask(srv.base, q["text"], th, a.max_tokens_on if th else a.max_tokens_off,
                                          a.chat_timeout, sysp)
                            break
                        except (lwrun.RunError, OSError, ValueError, KeyError) as e:
                            with open(errs, "a", encoding="utf-8") as ef:
                                ef.write(json.dumps({"qid": q["qid"], "thinking": th, "error": str(e)}) + "\n")
                            time.sleep(5)
                    else:
                        raise PromoteError(f"{d}: no answer to {q['qid']} after {a.chat_retries + 1} attempts")
                    f.write(json.dumps({"set": self.S["id"], "key": os.path.basename(d), "phase": phase,
                                        "mode": "promote", "qid": q["qid"], "thinking": th, "question": q["text"],
                                        "content": r["content"], "reasoning_content": r["reasoning_content"],
                                        "finish": r["finish"], "ms": r["ms"], "usage": r["usage"], "at": lwrun.utc()},
                                       ensure_ascii=False) + "\n")
                    f.flush()
        return self.judge(d, path, phase)

    def judge(self, d, path, phase):
        vpath, rpath = os.path.join(d, "verdicts.json"), os.path.join(d, "report.txt")
        out = lwrun.run_judge(self.run, [path], vpath, rpath, os.path.relpath(d, self.work))
        if out.get("rc") != 0:
            raise PromoteError(f"the judge failed on {path}: {out}")
        v = lwrun.read_json(vpath, {})
        return {(r["qid"], bool(r["thinking"])): r["verdict"] for r in v.get("results", []) if r["phase"] == phase}

    def scores(self, verdicts, ret=None, answers=None):
        """Pass counts per thinking mode, the notes given (notes_tokens: the notes blocks as the tokenizer counts
        them, framing included) and, from the server's own usage records in `answers`, the prompt tokens it read
        (prompt_tokens: system block, chat template and question; a review: the block sum is not the
        prompt)."""
        out = {}
        for th in self.thinking:
            key = "on" if th else "off"
            vs = {qid: v for (qid, t), v in verdicts.items() if t == th}
            out[key] = {"pass": sum(v == "pass" for v in vs.values()), "total": len(vs)}
        if ret is not None:
            toks = [r["tokens"] for r in ret.values()]
            out["notes_tokens"] = sum(toks)
            out["notes_tokens_p50"] = lwrun.pctl(toks, 50)
            out["notes_given"] = sum(len(r["ids"]) for r in ret.values())
        if answers and os.path.exists(answers):
            pt = {}
            for line in open(answers, encoding="utf-8"):
                if line.strip():
                    r = json.loads(line)
                    n = (r.get("usage") or {}).get("prompt_tokens")
                    pt[(r["qid"], bool(r["thinking"]))] = n if isinstance(n, int) and not isinstance(n, bool) else None
            want = [q["qid"] for q in self.S["questions"]]
            for th in self.thinking:
                got = [pt.get((q, th)) for q in want]
                have = [n for n in got if n is not None]
                # a missing answer or usage record stays missing: no total unless every question has one
                out["on" if th else "off"].update(
                    prompt_tokens=sum(have) if len(have) == len(want) else None,
                    prompt_tokens_p50=lwrun.pctl(have, 50) if len(have) == len(want) else None,
                    prompt_tokens_coverage=f"{len(have)}/{len(want)}")
        return out

    # ---------------------------------------------------------------------------------- copies and teaching
    def copy_of(self, src, key, graph_from=None):
        step = lwrun.Step(key, "teach", "teach")
        if os.path.exists(self.run.copy_path(step)):
            self.run.set_aside(self.run.copy_path(step), "a copy from an unfinished night")
        folder, _secs = self.run.make_copy(step, src, f"promotion {key}")
        graph = folder + ".graph.json"
        if graph_from:
            shutil.copyfile(graph_from, graph)
        elif os.path.exists(graph):
            os.remove(graph)
        return folder, graph

    def drop(self, folder):
        if folder and folder != self.run.src and os.path.exists(folder):
            self.run.cleanup_copy(folder, "no longer serving")

    def teach(self, folder, graph, lessons, d, label):
        """Teach each lesson's own sentence on `folder` with --slide: {lesson id: kept (bool)}."""
        os.makedirs(d, exist_ok=True)
        srv = self.serve(folder, True, d, label, graph)
        kept = {}
        try:
            g = srv.graph()
            seen = {}
            for lid in lessons:
                rec, g = lwrun.learn_one(self.run, srv, self.by_id[lid], d, g, seen)
                kept[lid] = bool(rec["recalled"])
                lwrun.write_json(os.path.join(d, f"learn-{lid}.json"), rec)
        finally:
            srv.stop("promotion teaching done")
        return kept

    # ---------------------------------------------------------------------------------- a day
    def day(self, k):
        key = f"day{k}"
        rec = self.st["days"].get(key)
        if rec and rec.get("done"):
            LOG(f"{key}: done already")
            return
        d = os.path.join(self.work, "days", key)
        if rec is not None:
            self.set_aside(d, f"{key} did not finish")
        rec = self.st["days"][key] = {"started": lwrun.utc()}
        self.save()
        ns, by_src = self.notes_day(k)
        excl = self.exclude_ids(by_src)
        sret, sys_blocks = self.blocks(ns, excl)
        cret, ctl_blocks = self.blocks(ns)
        self.write_notes(os.path.join(d, "system"), sret)
        self.write_notes(os.path.join(d, "control"), cret)
        folder = self.serving()
        LOG(f"{key}: {len(self.store_of(k))} lessons told; serving {folder}; {len(excl)} notes in weights, left out")
        srv = self.serve(folder, False, os.path.join(d, "system"), f"{key} system")
        try:
            sv = self.ask(srv, os.path.join(d, "system"), key, sys_blocks)
        finally:
            srv.stop(f"{key} system done")
        if folder == self.run.src and not excl:
            cv = sv
            rec["control_is_system"] = True
        else:
            srv = self.serve(self.run.src, False, os.path.join(d, "control"), f"{key} control")
            try:
                cv = self.ask(srv, os.path.join(d, "control"), key, ctl_blocks)
            finally:
                srv.stop(f"{key} control done")
        rec["system"] = self.scores(sv, sret, os.path.join(d, "system", "answers.jsonl"))
        rec["control"] = self.scores(cv, cret, os.path.join(d, "system" if rec.get("control_is_system") else "control",
                                                            "answers.jsonl"))
        rec["serving"] = folder
        rec["in_weights"] = sorted(int(l) for l, f in self.st["facts"].items() if f["status"] == "in_weights")
        rec["pocket"] = sorted(int(l) for l, f in self.st["facts"].items() if f["status"] in ACTIVE)
        rec["changed"] = self.changed_questions(k, sv, cv)
        self.qualify(k, ns, by_src, sv)
        rec["notes"] = {str(l): n for l, n in by_src.items()}
        rec["done"] = True
        self.save()
        s, c = rec["system"], rec["control"]
        LOG(f"{key}: system {s['off']['pass']}/{s['off']['total']} ({s['off'].get('prompt_tokens')} prompt tokens by the "
            f"server, {s['notes_tokens']} of them notes), control {c['off']['pass']}/{c['off']['total']} "
            f"({c['off'].get('prompt_tokens')} prompt, {c['notes_tokens']} notes)")

    def changed_questions(self, k, sv, cv):
        """Questions whose right answer changed since yesterday: how many each arm got right (stale answers show
        here)."""
        if k == 1:
            return None
        e0, e1 = self.S["expected"][f"day{k - 1}"], self.S["expected"][f"day{k}"]
        qs = [q["qid"] for q in self.S["questions"] if e0[q["qid"]] != e1[q["qid"]]]
        return {"questions": qs, "system": sum(sv.get((q, False)) == "pass" for q in qs),
                "control": sum(cv.get((q, False)) == "pass" for q in qs)}

    def dependents(self, ns, k, note_id):
        lessons = [self.by_id[i] for i in self.store_of(k)]
        return [q["qid"] for q in self.S["questions"] if q["role"] == "probe"
                and note_id in nm.needed_notes(ns, "user", self.S, lessons, q)]

    def qualify(self, k, ns, by_src, sv):
        for lid in self.store_of(k):
            n = by_src.get(lid)
            f = self.fact(lid)
            f["note_state"] = n["state"] if n else None
            if f["status"] != "note":
                continue
            ok = bool(n and n["state"] == nm.CURRENT and n["kind"] == nm.FACT and n["op"] in PROMOTABLE_OPS)
            deps = self.dependents(ns, k, n["id"]) if ok else []
            f["dependents"] = deps
            good = ok and deps and all(sv.get((q, False)) == "pass" for q in deps)
            f["streak"] = f["streak"] + 1 if good else 0
            if good and f["streak"] >= self.a.hold:
                self.note(f, f"due: qualified {f['streak']} days running", f"day{k}")

    # ---------------------------------------------------------------------------------- a night
    def night(self, k):
        key = f"night{k}"
        rec = self.st["nights"].get(key)
        if rec and rec.get("done"):
            LOG(f"{key}: done already")
            return
        d = os.path.join(self.work, "nights", key)
        if rec is not None:
            if rec.get("decision"):
                # the night accepted a folder and moved facts on, then stopped: its alone checks and its state are
                # half done, and a rerun would count the night twice or certify another folder with these answers
                raise PromoteError(f"{key} accepted {rec['decision']} but did not finish: not resumable; start a new "
                                   f"work dir (a review of f5fdbe3)")
            self.set_aside(d, f"{key} did not finish")
        rec = self.st["nights"][key] = {"started": lwrun.utc()}
        self.save()                         # on file before any answer: a stop from here on is seen by the rerun
        ns, by_src = self.notes_day(k)
        facts = self.st["facts"]
        active = [int(l) for l, f in facts.items() if f["status"] in ACTIVE]
        stale = [l for l in active if not by_src.get(l) or by_src[l]["state"] != nm.CURRENT]
        due = sorted(int(l) for l, f in facts.items() if f["status"] == "note" and f["streak"] >= self.a.hold)
        keep = sorted(set(active) - set(stale))
        rec.update(active=sorted(active), stale=stale, due=due)
        LOG(f"{key}: {len(active)} facts in the pocket, {len(stale)} stale {stale}, {len(due)} due {due}")
        dv = self.read_verdicts(os.path.join(self.work, "days", f"day{k}", "system"), f"day{k}")
        # the notes each candidate folder would be served with: adding keeps today's notes in weights out of the
        # prompt (their neurons are still there); a rebuild teaches every fact again, so every note is given
        _sret, add_blocks = self.blocks(ns, self.exclude_ids(by_src))
        _fret, full_blocks = self.blocks(ns)
        plans = night_plans(stale, keep, due)
        accepted, cur = None, self.serving()
        for name, lessons in plans:
            p = {"plan": name, "lessons": lessons}
            rec.setdefault("tries", []).append(p)
            pd = os.path.join(d, name)
            if name == "source" or not lessons:
                folder, graph, kept = self.run.src, None, {}
            elif name == "add":
                folder, graph = self.copy_of(cur, f"{key}-{name}", self.st["graph"])
                kept = self.teach(folder, graph, lessons, os.path.join(pd, "teach"), f"{key} {name} --slide")
            else:
                folder, graph = self.copy_of(self.run.src, f"{key}-{name}")
                kept = self.teach(folder, graph, sorted(lessons, key=lambda i: self.by_id[i]["told"]),
                                  os.path.join(pd, "teach"), f"{key} {name} --slide")
            p["kept"] = {str(x): v for x, v in kept.items()}
            srv = self.serve(folder, False, os.path.join(pd, "regress"), f"{key} {name} regress")
            try:
                rv = self.ask(srv, os.path.join(pd, "regress"), f"day{k}", add_blocks if name == "add" else full_blocks)
                worse = [q for (q, th), v in dv.items() if v == "pass" and rv.get((q, th)) != "pass"]
                better = [q for (q, th), v in rv.items() if v == "pass" and dv.get((q, th)) != "pass"]
                p.update(worse=worse, better=better)
                LOG(f"{key} {name}: {len(worse)} questions worse {worse}, {len(better)} better")
                # the untaught source is the last resort with stale facts: it is taken whatever it scores, because a
                # pocket that still answers a changed fact must not keep serving
                if len(worse) <= self.a.tolerance or name == "source":
                    accepted = (name, folder, graph, kept)
                    self.accept(k, rec, name, folder, graph, kept, stale, keep, due)
                    self.alone(k, srv, ns, by_src, os.path.join(d, "alone"), rec)
                    break
            finally:
                srv.stop(f"{key} {name} checked")
            self.drop(folder)
        if accepted is None:
            rec["decision"] = "unchanged"
            for lid in due:
                f = self.fact(lid)
                f["status"] = "rejected"
                self.note(f, "rejected: the night made answers worse", key)
            if not plans or not stale:
                if self.st["folder"]:
                    srv = self.serve(self.serving(), False, os.path.join(d, "alone-serving"), f"{key} alone")
                    try:
                        self.alone(k, srv, ns, by_src, os.path.join(d, "alone"), rec)
                    finally:
                        srv.stop(f"{key} alone checked")
        rec["done"] = True
        self.save()

    def set_aside(self, d, why):
        """An unfinished day's or night's folder, renamed so a rerun asks every question again and reuses nothing."""
        if os.path.exists(d):
            new = f"{d}.stale-{lwrun.stamp()}"
            LOG(f"set aside {d} -> {new}: {why}")
            os.rename(d, new)

    def read_verdicts(self, d, phase):
        v = lwrun.read_json(os.path.join(d, "verdicts.json"), {})
        return {(r["qid"], bool(r["thinking"])): r["verdict"] for r in v.get("results", []) if r["phase"] == phase}

    def accept(self, k, rec, name, folder, graph, kept, stale, keep, due):
        key = f"night{k}"
        old = self.st["folder"]
        rec["decision"] = name
        self.save()                         # the marker first: a stop after this point is not resumed
        for lid in stale:
            f = self.fact(lid)
            f.update(status="freed", alone=0)
            self.note(f, "freed: its note changed; the pocket was rebuilt without it", key)
        rebuilt = name in ("rebuild", "rebuild-old", "source")
        for lid in keep:
            f = self.fact(lid)
            if not rebuilt:
                continue
            if name == "source" or not kept.get(lid):
                f.update(status="note", streak=0, alone=0)
                self.note(f, "back to notes: the rebuild left it out" if name == "source" else
                          "back to notes: not kept when the pocket was rebuilt", key)
            else:
                f.update(status="promoted", alone=0)
                self.note(f, "taught again in the rebuilt pocket", key)
        for lid in due:
            f = self.fact(lid)
            if name in ("rebuild-old", "source"):
                f["status"] = "rejected"
                self.note(f, "rejected: the night with it made answers worse", key)
            elif kept.get(lid):
                f.update(status="promoted", night=k, alone=0)
                self.note(f, "promoted", key)
            else:
                f.update(status="missed", streak=0)
                self.note(f, "missed: TensorFold did not keep it; it stays a note", key)
        self.st["folder"] = None if folder == self.run.src else folder
        self.st["graph"] = graph
        if old and old != folder:
            self.drop(old)
        self.save()

    def alone(self, k, srv, ns, by_src, d, rec):
        """Each fact in the pocket answers its questions without its own note: --alone nights running moves the note
        out of the prompt; one miss brings it back."""
        out = rec.setdefault("alone", {})
        for l, f in sorted(self.st["facts"].items(), key=lambda x: int(x[0])):
            if f["status"] not in ACTIVE:
                continue
            n = by_src.get(int(l))
            deps = self.dependents(ns, k, n["id"]) if n and n["state"] == nm.CURRENT else []
            if not deps:
                continue
            excl = self.exclude_ids(by_src, {n["id"]})
            _ret, blocks = self.blocks(ns, excl)
            v = self.ask(srv, os.path.join(d, f"L{l}"), f"day{k}", blocks, qids=set(deps))
            ok = all(v.get((q, False)) == "pass" for q in deps)
            f["alone"] = f["alone"] + 1 if ok else 0
            was = f["status"]
            f["status"] = "in_weights" if f["alone"] >= self.a.alone else "promoted"
            if f["status"] != was:
                self.note(f, "in weights: its note leaves the prompt" if f["status"] == "in_weights" else
                          "back in the prompt: answered wrong without its note", f"night{k}")
            out[l] = {"questions": deps, "right": ok, "streak": f["alone"], "status": f["status"]}
        self.save()

    # ---------------------------------------------------------------------------------- the whole run
    def go(self):
        os.makedirs(self.work, exist_ok=True)
        LOG.path = os.path.join(self.work, "promote.log")
        lwrun.write_json(self.run.set_for_judge, self.S)
        self.save()
        for k in range(1, len(self.days) + 1):
            self.day(k)
            if k < len(self.days):
                self.night(k)
        self.report()

    def report(self):
        rows = ["day   system right (off)  prompt tokens   control right (off)  prompt tokens   in weights  pocket  "
                "changed (sys/ctl)   (prompt tokens: the server's own count, summed over the day's questions)"]
        for k in range(1, len(self.days) + 1):
            r = self.st["days"].get(f"day{k}") or {}
            if not r.get("done"):
                continue
            s, c, ch = r["system"], r["control"], r.get("changed")
            pocket = len(r.get("pocket", []))       # that day's pocket, not today's (a review)
            rows.append(f"day{k}  {s['off']['pass']:>3}/{s['off']['total']:<15} {s['off'].get('prompt_tokens', '-'):>13}   "
                        f"{c['off']['pass']:>3}/{c['off']['total']:<16} {c['off'].get('prompt_tokens', '-'):>13}   "
                        f"{len(r.get('in_weights', [])):>10}  {pocket:>6}  "
                        + (f"{ch['system']}/{ch['control']} of {len(ch['questions'])}" if ch else "-"))
        nights = [f"{k}: {v.get('decision')} (due {v.get('due')}, stale {v.get('stale')})"
                  for k, v in self.st["nights"].items()]
        text = "\n".join(rows + ["", "nights:"] + nights + ["", "facts:"] +
                         [f"L{l} {f['status']}: {'; '.join(f['log'])}" for l, f in
                          sorted(self.st["facts"].items(), key=lambda x: int(x[0])) if f["log"]]) + "\n"
        with open(os.path.join(self.work, "promote-report.txt"), "w", encoding="utf-8") as f:
            f.write(text)
        LOG("\n" + text)


# ============================================================================================ self-test
def selftest(root):
    """Against lwrun_selftest.py's mock TensorFold (no GPU). The mock's answers say which facts are live, not whether
    the model is right, so the judge is replaced by a scripted one: every answer passes, except where a check makes
    a night worse. Checked: the nights set; promotion after --hold days; a note leaving the prompt after --alone
    nights, and the prompts the server actually got; a change cancelling a due fact; a stale fact rebuilding the
    pocket from the untaught source; a night that makes answers worse kept out. Each with a negative control."""
    import lwrun_selftest as T
    C = L.Check()
    root = os.path.abspath(root)
    os.makedirs(root, exist_ok=True)
    ex = lwrun.read_json(os.path.join(os.path.dirname(HERE), "examples", "L6-seed1.json"))
    extra = [{"op": "CHANGE", "e": "my dog", "a": "eye_colour", "v": "brown", "text": "My dog's eyes are brown now."}]
    S = make_set(ex, extra, "1-10,11-20,21-29,30,")
    ok_set = ([p["name"] for p in S["phases"]][:3] == ["baseline", "day1", "day2"] and S["nights"]["days"][-1] == []
              and L.check_expected(S) and S["expected"]["day1"] != S["expected"]["day4"])
    try:
        make_set(ex, extra, "1-10,12-20,21-30,")
        refused = False
    except PromoteError:
        refused = True
    C.add("promote: make-set tells every lesson once over the days and the rule book answers every day; control: a "
          "day plan that skips a lesson is refused", ok_set and refused, 2, f"phases {[p['name'] for p in S['phases']]}")
    spath = os.path.join(root, "nights.json")
    lwrun.write_json(spath, S)
    lab = T.Lab(os.path.join(root, "lab"))

    def go(name, worse_on=None):
        """One run. worse_on: (night key, plan): the scripted judge fails one question there that passed."""
        sc = {"starts_log": os.path.join(root, f"{name}.starts.jsonl"), "chat_log": os.path.join(root, f"{name}.chats.jsonl")}
        with open(os.path.join(root, f"{name}.script.json"), "w") as f:
            json.dump(sc, f)
        os.environ["MOCKTF_SCRIPT"] = os.path.join(root, f"{name}.script.json")
        a = argparse.Namespace(set=spath, bin=lab.bin, model_src=lab.src, work=os.path.join(root, name),
                               port=T.free_port(), hold=2, alone=1, tolerance=0, thinking="off", max_tokens_on=64,
                               notes_config=None, notes_tokenizer="fake", server_arg=None, cleanup=False)
        n = Nights(a)

        def judge(d, path, phase):
            v = []
            for r in (json.loads(x) for x in open(path, encoding="utf-8") if x.strip()):
                bad = (worse_on and f"/nights/{worse_on[0]}/" in d and "/regress" in d and r["qid"] == "Q01"
                       and (worse_on[1] == "*" or f"/{worse_on[1]}/regress" in d))
                v.append({"phase": phase, "qid": r["qid"], "thinking": r["thinking"], "verdict": "fail" if bad else "pass"})
            lwrun.write_json(os.path.join(d, "verdicts.json"), {"results": v})
            return {(x["qid"], bool(x["thinking"])): x["verdict"] for x in v}
        n.judge = judge
        n.go()
        chats = [json.loads(x) for x in open(sc["chat_log"], encoding="utf-8") if x.strip()]
        return n, chats

    n1, chats1 = go("w1")
    f1 = n1.st["facts"]
    st = lambda f, i: f.get(str(i), {}).get("status")
    # day 1 facts qualify on days 1 and 2 and are promoted on night 2; L1 (my dog is Luna) is changed on day 2 by L13,
    # so it never qualifies twice; L6 (my dog has grey eyes) is changed on day 4 by L30: stale on night 4
    in_w = [i for i in range(1, 11) if st(f1, i) == "in_weights"]
    ok_promo = (n1.st["nights"]["night2"].get("decision") == "add" and len(in_w) >= 5 and st(f1, 1) == "note"
                and "promoted (night2)" in "; ".join(f1["2"]["log"]))
    C.add("promote: day 1 facts qualify two days running and are promoted on night 2, then leave the prompt after an "
          "alone check; control: L1, changed on day 2, is never promoted", ok_promo, len(in_w),
          f"in weights {in_w}; night2 {n1.st['nights']['night2'].get('decision')}; L1 {st(f1, 1)}")
    # day 3: the blocks the system arm was to send (notes.jsonl) leave the in-weights notes out, every one of them
    # reached the serving folder as a chat's system message, and the control arm's blocks still hold those notes
    day3 = n1.st["days"]["day3"]
    served = os.path.realpath(day3["serving"])
    rd = lambda arm: [json.loads(x) for x in open(os.path.join(n1.work, "days", "day3", arm, "notes.jsonl"),
                                                   encoding="utf-8") if x.strip()]
    sysb, ctlb = rd("system"), rd("control")
    got = {c["system"] for c in chats1 if c["folder"] == served and not c["slide"] and c["system"]}
    texts = [S["lessons"][i - 1]["text"] for i in in_w]
    leaked = [t for t in texts if any(json.dumps(t) in (b["block"] or "") for b in sysb)]
    shown = [t for t in texts if any(json.dumps(t) in (b["block"] or "") for b in ctlb)]
    unsent = [b["qid"] for b in sysb if b["block"] and b["block"] not in got]
    C.add("promote: on day 3 the notes in weights are left out of every block the serving folder gets, and each block "
          "reached it as the chat's system message; control: the notes-only arm's blocks hold them",
          bool(sysb) and not leaked and not unsent and len(shown) == len(texts), len(sysb),
          f"leaked {leaked[:2]}; not sent {unsent[:3]}; shown to control {len(shown)} of {len(texts)}")
    n4 = n1.st["nights"].get("night4", {})
    reg = lwrun.read_json(os.path.join(n1.work, "copies.json"), {"copies": []})["copies"]
    rebuilt = [c for c in reg if c["step"] == "night4-rebuild"]
    taught4 = sorted(int(x.split("-")[1].split(".")[0]) for x in os.listdir(os.path.join(n1.work, "nights", "night4",
                     "rebuild", "teach")) if x.startswith("learn-") and x.endswith(".json")) if rebuilt else []
    ok_stale = (6 in n4.get("stale", []) and n4.get("decision") == "rebuild" and st(f1, 6) == "freed"
                and rebuilt and os.path.realpath(rebuilt[0]["src"]) == os.path.realpath(lab.src) and 6 not in taught4
                and set(in_w) - {6} <= set(taught4))
    C.add("promote: a promoted fact changed on day 4 is stale on night 4: the pocket is rebuilt from the untaught "
          "source with the facts that still hold, and the stale one is freed; control: the facts that still hold are "
          "taught again", ok_stale, len(taught4), f"night4 {n4.get('decision')} stale {n4.get('stale')} taught {taught4}")
    # the plans of a night, as a table; control: the plan list as first built (f0cbaff) leaves a stale pocket serving
    def plans_f0cbaff(stale, keep, due):
        plans = []
        if stale:
            plans.append(("rebuild", keep + due))
            if due:
                plans.append(("rebuild-old", keep))
            if keep:
                plans.append(("source", []))
        elif due:
            plans.append(("add", due))
        return plans
    cases = [(([6], [], []), ["source"]), (([6], [], [2]), ["rebuild", "source"]),
             (([6], [3], [2]), ["rebuild", "rebuild-old", "source"]), (([6], [3], []), ["rebuild", "source"]),
             (([], [3], [2]), ["add"]), (([], [3], []), [])]
    got = [[p[0] for p in night_plans(*c)] for c, _ in cases]
    old = [[p[0] for p in plans_f0cbaff(*map(list, c))] for c, _ in cases]
    last_resort = lambda ps, c: not c[0] or (ps and ps[-1] == "source")
    C.add("promote: every night with a stale fact ends with the untaught source as its last plan; control: the plan "
          "list as first built has no last resort when nothing is left to keep",
          got == [w for _, w in cases] and not all(last_resort(o, c) for o, (c, _) in zip(old, cases)), len(cases),
          f"plans {got}; first built {old}")
    n3, _chats3 = go("w3", worse_on=("night4", "*"))
    f3 = n3.st["facts"]
    n34 = n3.st["nights"].get("night4", {})
    ok_src = (n34.get("decision") == "source" and [t["plan"] for t in n34.get("tries", [])][-1] == "source"
              and st(f3, 6) == "freed" and not any(f.get("status") in ACTIVE for f in f3.values())
              and os.path.realpath(n3.st["days"]["day5"]["serving"]) == os.path.realpath(lab.src))
    C.add("promote: when every rebuild of a stale night makes answers worse, the untaught source serves anyway and "
          "every fact is back to notes; control: run w1, the same night with no worse answer, kept its rebuild",
          ok_src and ok_stale, len(n34.get("tries", [])), f"night4 {n34.get('decision')} tries "
          f"{[t['plan'] for t in n34.get('tries', [])]}; day5 serving {n3.st['days'].get('day5', {}).get('serving')}")
    # an interrupted night: stopped inside its alone checks (after it accepted a folder), a rerun refuses; stopped
    # before accepting (inside its regress answers), a rerun sets the old answers aside and asks again. Control: the
    # first run of each got that far (the marker, the answers on file).
    class Stop(Exception):
        pass

    def interrupted(name, where):
        sc = {"starts_log": os.path.join(root, f"{name}.starts.jsonl"), "chat_log": os.path.join(root, f"{name}.chats.jsonl")}
        with open(os.path.join(root, f"{name}.script.json"), "w") as f:
            json.dump(sc, f)
        os.environ["MOCKTF_SCRIPT"] = os.path.join(root, f"{name}.script.json")
        a = argparse.Namespace(set=spath, bin=lab.bin, model_src=lab.src, work=os.path.join(root, name),
                               port=T.free_port(), hold=2, alone=1, tolerance=0, thinking="off", max_tokens_on=64,
                               notes_config=None, notes_tokenizer="fake", server_arg=None, cleanup=False)
        out = []
        for attempt in (1, 2):
            n = Nights(a)

            def judge(d, path, phase, n=n, attempt=attempt):
                if attempt == 1 and f"/nights/night2/{where}" in d:
                    raise Stop()
                v = [{"phase": phase, "qid": r["qid"], "thinking": r["thinking"], "verdict": "pass"}
                     for r in (json.loads(x) for x in open(path, encoding="utf-8") if x.strip())]
                lwrun.write_json(os.path.join(d, "verdicts.json"), {"results": v})
                return {(x["qid"], bool(x["thinking"])): x["verdict"] for x in v}
            n.judge = judge
            try:
                n.go()
                out.append("done")
            except Stop:
                out.append(("stopped", n.st["nights"].get("night2", {}).get("decision")))
            except PromoteError as e:
                out.append(("refused", str(e)[:60]))
            finally:
                lwrun.stop_all("self-test")
        aside = [x for x in os.listdir(os.path.join(a.work, "nights")) if x.startswith("night2.stale-")]
        return out, aside
    after, _ = interrupted("w4", "alone")
    before, aside = interrupted("w5", "add/regress")
    ok_int = (after[0] == ("stopped", "add") and after[1][0] == "refused" and before[0] == ("stopped", None)
              and before[1] == "done" and len(aside) == 1)
    C.add("promote: a night stopped after it accepted a folder is not resumed (refused, start a new work dir); one "
          "stopped before is rerun from scratch with its old answers set aside; controls: each first run got that far",
          ok_int, 2, f"after accept {after}; before accept {before}, set aside {aside}")
    n2, chats2 = go("w2", worse_on=("night2", "add"))
    f2 = n2.st["facts"]
    rej = [i for i in range(1, 11) if st(f2, i) == "rejected"]
    day3b = n2.st["days"]["day3"]
    ok_rej = (n2.st["nights"]["night2"].get("decision") == "unchanged" and len(rej) >= 5
              and os.path.realpath(day3b["serving"]) == os.path.realpath(lab.src) and not day3b["in_weights"]
              and any(json.dumps(S["lessons"][rej[0] - 1]["text"]) in (c["system"] or "") for c in chats2))
    C.add("promote: a night that makes one question worse is kept out: the source keeps serving, its facts are "
          "rejected and their notes stay in the prompt; control: run w1, the same night without the worse answer, was "
          "kept", ok_rej and ok_promo, len(rej), f"night2 {n2.st['nights']['night2'].get('decision')} rejected {rej}")
    ok = all(r[1] for r in C.rows)
    print(f"promote selftest {'PASSED' if ok else 'FAILED'}: {sum(r[1] for r in C.rows)}/{len(C.rows)} checks; "
          f"work in {root}")
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make-set")
    m.add_argument("base")
    m.add_argument("--days", required=True)
    m.add_argument("--extra", help="a JSON list of lessons told after the base set's: {op, e, a, v, text}")
    m.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("set")
    r.add_argument("--bin", required=True)
    r.add_argument("--model-src", required=True)
    r.add_argument("--work", required=True)
    r.add_argument("--port", type=int, required=True)
    r.add_argument("--hold", type=int, default=2, help="days a note must qualify in a row before it is promoted")
    r.add_argument("--alone", type=int, default=2, help="nights a promoted fact must answer without its note")
    r.add_argument("--tolerance", type=int, default=0, help="questions a night may make worse and still be kept")
    r.add_argument("--thinking", default="off")
    r.add_argument("--max-tokens-on", type=int, default=6144)
    r.add_argument("--notes-config")
    r.add_argument("--notes-tokenizer")
    r.add_argument("--server-arg", action="append")
    r.add_argument("--cleanup", action="store_true")
    s = sub.add_parser("selftest")
    s.add_argument("--dir", default=os.path.join("/tmp", f"promote-selftest-{os.getpid()}"))
    a = ap.parse_args(argv)
    if a.cmd == "make-set":
        base = lwrun.read_json(a.base)
        extra = lwrun.read_json(a.extra) if a.extra else []
        try:
            S = make_set(base, extra, a.days)
        except PromoteError as e:
            print(f"promote: {e}", file=sys.stderr)
            return 2
        lwrun.write_json(a.out, S)
        print(f"{a.out}: {S['id']}, {len(S['lessons'])} lessons over {len(S['nights']['days'])} days, "
              f"{len(S['questions'])} questions")
        return 0
    if a.cmd == "selftest":
        return 0 if selftest(a.dir) else 1
    try:
        n = Nights(a)
    except lwrun.RunError as e:
        print(f"promote: {e}", file=sys.stderr)
        return 2
    import atexit
    import signal
    for sg in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sg, lwrun.on_signal)
    atexit.register(lwrun.stop_all)
    try:
        n.go()
        return 0
    except lwrun.RunError as e:
        LOG(f"promote: {e}")
        return 2
    except lwrun.Interrupted as e:
        LOG(f"promote: interrupted by {e}")
        return 130
    finally:
        lwrun.stop_all("promote exiting")


if __name__ == "__main__":
    sys.exit(main())
