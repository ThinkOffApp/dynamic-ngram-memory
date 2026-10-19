#!/usr/bin/env python3
"""Level 7, fill to capacity: teach a level 7 set's facts one at a time to a TensorFold server, and ask as it fills.

  python3 zengine/capacity.py run SET.json --bin PATH --model-src SNAPSHOT_DIR --work DIR --port N
      [--when-full refuse|replace|forget] [--evict-max 3] [--gate mark|skip] [--known-check on|off]
      [--steps baseline,main,pairs,deletes] [--delete-mode zero|endpoint|reteach|none] [--norms on|off]
      [--thinking off,on] [--dry-run]                                                    (more: --help)
  python3 zengine/capacity.py report WORK      judge the answers on file again and write the report again
  python3 zengine/capacity.py selftest [--dir DIR]      against the runner's mock TensorFold server, no GPU

A level 7 set (`lwlogic.py gen --level 7`, examples/L7-seed1.json) holds N plain facts, each on an entity-attribute
pair of its own, checkpoint phases ck-NNN (the first k lessons), pair phases and cumulative delete phases.

Steps, in this order:
  baseline  serve SNAPSHOT_DIR read-only, without --slide; ask every question (the original model).
  main      a fresh copy, served with --slide. For each lesson, in the set's order:
              intake   the rule book sorts it against what is taught so far (zengine/intake.py): NEW, REPLACES,
                       ALREADY_KNOWN or IMPLIED; the parser must read the sentence back to the set's own lesson.
              known    the model is asked the lesson's open CURRENT question (thinking off). "already known (model)"
                       only when the judge reads the lesson's exact value from that answer, never from a yes/no.
                       The yes/no is asked too, and only recorded: it shows how often a yes/no would have said known.
              gate     --gate mark (default) teaches every lesson and records the two checks; --gate skip does not
                       teach a lesson the intake calls ALREADY_KNOWN or IMPLIED or the model already knows.
              learn    POST /v1/slide/learn, as lwrun.py does. A lesson that comes back with 1.0.8's PocketFull
                       message (READ: pocket.zig fitAll, error.PocketFull when no layer the fact may use has a free
                       neuron) is handled by --when-full:
                         refuse   the lesson is not learned; teaching goes on with the next one (another fact may
                                  still fit: reply-keyed neurons, as a fact about you takes, may also use layers
                                  after the subject-keyed range; READ, pocket.zig candidate).
                         forget   gradual forgetting, oldest first: DELETE /v1/slide/fact/{id} of the oldest learned
                                  fact, then the lesson again, up to --evict-max times.
                         replace  the same, but the fact to give way is chosen: one the rule book says the others
                                  cover, then one the model knew before it was taught, then the largest weight-norm
                                  outlier, then the oldest.
                       1.0.8 has no per-fact delete: a 404 (405, 501) there turns forget and replace into refuse for
                       the rest of the run. A delete that does not free room (the lesson comes back PocketFull
                       again) does the same: 1.0.8's own replace zeroes a fact and keeps its places taken.
              norms    the fact's pocket weight norm, read from the saved folder (lwrun.pocket_norms): its input
                       weights (up_proj rows), its output weights (down_proj columns), both.
              ask      after the first lesson and after every 5th (the set's checkpoints): one CURRENT wording and
                       one yes/no of every fact taught so far, and the controls, in the --slide server; every
                       fact's norm is read again then, so a norm that moves later is seen.
            With --delete-mode zero, the taught folder is then served again without --slide and asked every
            question (teach, restarted): the zero copies are served that way.
  pairs     per pair (similar, far): a fresh copy taught A, asked A's and B's questions and the controls; then
            taught B and asked again. The interference matrix: A's answers after A, and after A then B.
  deletes   six facts of the set's delete order, one at a time, asked after the 1st, 3rd and 6th:
              zero      a copy of the taught folder; each fact's neurons zeroed as lwrun.py's zero mode does, one
                        lesson after another in the same copy; served without --slide.
              endpoint  a copy of the taught folder served with --slide; DELETE /v1/slide/fact/{id} (not in 1.0.8:
                        recorded as unavailable, nothing asked).
              reteach   a fresh copy taught the lessons the phase keeps (never taught): about N lessons per phase.
            kept_stable: the judge compares each delete phase with this model's own teach answers.
Report: WORK/capacity.json and WORK/capacity.txt: the retention curve at 1, 5, 10, 20, 40, 64 (and 100) facts, by
teaching order and for the first 10 facts on their own, what was learned, refused and evicted, the norms and their
outliers, the interference matrix and the deletes. Everything the run records is in WORK/capacity-state.json.

Usage ranking (which facts' neurons fire on ordinary text) needs a TensorFold activation endpoint that 1.0.8 does not
have; it is a request (docs/NIGHT-PRUNE.md), not built here.
No resume: a stopped run starts again in a new --work. Copies are never deleted on macOS. Standard library only.
"""
import argparse
import atexit
import os
import shlex
import signal
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import intake  # noqa: E402  (the rule-book check before teaching)
import lwlogic  # noqa: E402  (the rule book and the judge)
import lwrun  # noqa: E402  (servers, learning, probing, zeroing, norms)
from lwrun import LOG, RunError, ZeroSkip, read_json, utc, write_json  # noqa: E402

FORMAT = "lw-capacity/1"
NO_ENDPOINT = (404, 405, 501)
ACTIVATION_REQUEST = ("not available: TensorFold 1.0.8 has no per-request record of which facts' neurons fire, so "
                      "facts cannot be ranked by usage; requested in docs/NIGHT-PRUNE.md")


class View:
    """What lwrun.probe needs of a run, with only some of the set's questions."""

    def __init__(self, run, questions):
        self.args, self.thinking, self.run = run.args, run.thinking, run
        self.S = {"id": run.S["id"], "questions": questions}

    def pairs(self):
        return {(q["qid"], th) for q in self.S["questions"] for th in self.thinking}


class Capacity:
    def __init__(self, a):
        self.args = a
        self.set_path = os.path.abspath(a.set)
        self.S = read_json(self.set_path)
        if not isinstance(self.S, dict) or self.S.get("format") != "lw-test-set/1":
            raise RunError(f"{a.set}: not an lwlogic test set (format lw-test-set/1)")
        if not isinstance(self.S.get("l7"), dict):
            raise RunError(f"{a.set}: not a level 7 set (no l7 section): run it with lwrun.py")
        self.L7 = self.S["l7"]
        self.work = os.path.abspath(a.work)
        self.models = os.path.abspath(a.models_dir or os.path.join(self.work, "models"))
        self.pidfile = os.path.join(self.work, "server.pid")
        self.copies_path = os.path.join(self.work, "copies.json")
        self.port = a.port
        self.bin = os.path.abspath(a.bin)
        self.src = os.path.abspath(a.model_src)
        self.lwlogic = os.path.abspath(a.lwlogic)
        self.set_for_judge = os.path.join(self.work, "set.json")
        self.thinking = lwrun_thinking(a.thinking)
        self.les = {l["id"]: l for l in self.S["lessons"]}
        self.qs = {q["qid"]: q for q in self.S["questions"]}
        self.steps = [x.strip() for x in a.steps.split(",") if x.strip()]
        bad = [x for x in self.steps if x not in ("baseline", "main", "pairs", "deletes")]
        if bad:
            raise RunError(f"--steps: {bad} (baseline, main, pairs, deletes)")
        if a.evict_max < 0:
            raise RunError(f"--evict-max {a.evict_max}: give 0 or more")
        if "deletes" in self.steps and "main" not in self.steps and a.delete_mode in ("zero", "endpoint"):
            raise RunError("the zero and endpoint deletes copy the main step's taught folder: add main to --steps")
        self.state = None
        self.bin_sha = None

    # lwrun.Server, learn_one and probe use these, and args, port, bin, S, thinking, pairs()
    check_pidfile = lwrun.Run.check_pidfile
    write_pidfile = lwrun.Run.write_pidfile
    guard_teachable = lwrun.Run.guard_teachable

    def pairs(self):
        return {(q["qid"], th) for q in self.S["questions"] for th in self.thinking}

    def lesson_name(self, i):
        return self.les[i]["name"] if i in self.les else f"#{i}"

    # ---------------------------------------------------------------------------------- the set
    def fact_qids(self, lid):
        fq = self.L7["fact_questions"][str(lid)]
        return [fq["current"], fq["yes_no"]]

    def questions_for(self, lesson_ids):
        want = [x for lid in lesson_ids for x in self.fact_qids(lid)] + list(self.L7["controls"])
        return [self.qs[x] for x in want]

    def checkpoint_at(self):
        return {c["after"]: c["phase"] for c in self.L7["checkpoints"]}

    # ---------------------------------------------------------------------------------- files
    def sdir(self, name):
        d = os.path.join(self.work, "steps", name)
        os.makedirs(d, exist_ok=True)
        return d

    def save(self):
        if self.state is not None:
            self.state["updated"] = utc()
            write_json(os.path.join(self.work, "capacity-state.json"), self.state)

    def copy(self, name, src, note):
        """A copy of `src` at WORK/models/ID.name, made as lwrun.py makes its copies and registered in copies.json
        (so lwrun's guard lets --slide write it). Never reused: refused when it exists."""
        dst = os.path.join(self.models, f"{self.S['id']}.{name}")
        if os.path.exists(dst):
            raise RunError(f"{dst} exists; use a new --work")
        os.makedirs(self.models, exist_ok=True)
        cmd = ["cp", "-cRL", src, dst] if lwrun.IS_MAC else ["cp", "-RL", src, dst]
        LOG(f"copy [{name}] ({note}): {' '.join(shlex.quote(x) for x in cmd)}")
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            raise RunError(f"copy failed ({p.returncode}): {p.stderr.strip()[:400]}")
        reg = read_json(self.copies_path, {"copies": []})
        reg["copies"].append({"path": dst, "step": name, "src": src, "made": utc(), "bytes": lwrun.tree_bytes(dst)})
        write_json(self.copies_path, reg)
        return dst

    def preflight(self):
        a = self.args
        problems = []
        if not (os.path.isfile(self.bin) and os.access(self.bin, os.X_OK)):
            problems.append(f"--bin {self.bin} is not an executable file")
        cfg = read_json(os.path.join(self.src, "config.json"))
        if not isinstance(cfg, dict):
            problems.append(f"--model-src {self.src} has no config.json")
        elif "tensorfold_learned_neurons" in cfg and not a.allow_taught_src:
            problems.append(f"--model-src {self.src} already holds learned neurons: not the original model")
        if a.port in (8091,):
            problems.append(f"--port {a.port} is reserved (another server lives there)")
        elif lwrun.port_busy(a.port):
            problems.append(f"--port {a.port} is in use (or cannot be bound)")
        if not os.path.isfile(self.lwlogic):
            problems.append(f"lwlogic.py not found at {self.lwlogic} (--lwlogic)")
        if os.path.exists(os.path.join(self.work, "capacity-state.json")):
            problems.append(f"{self.work} holds an earlier capacity run; use a new --work (there is no resume)")
        try:
            self.check_pidfile()
        except RunError as e:
            problems.append(str(e))
        if problems:
            raise RunError("refusing to start:\n  " + "\n  ".join(problems))

    # ---------------------------------------------------------------------------------- asking and judging
    def ask_phase(self, srv, step, phase, questions, mode, tag=None):
        """Ask `questions` (thinking as --thinking) with server `srv`, filed as `phase`; then judge the file."""
        d = self.sdir(step)
        tag = tag or phase
        ans, err = os.path.join(d, f"answers-{tag}.jsonl"), os.path.join(d, f"errors-{tag}.jsonl")
        res = lwrun.probe(View(self, questions), srv, f"{step}:{tag}", phase, mode, ans, err)
        j = lwrun.run_judge(self, [ans], os.path.join(d, f"verdicts-{tag}.json"), os.path.join(d, f"report-{tag}.txt"),
                            f"{step} {tag}")
        rec = {"phase": phase, "tag": tag, "step": step, "mode": mode, "answers": os.path.relpath(ans, self.work),
               "verdicts": os.path.relpath(os.path.join(d, f"verdicts-{tag}.json"), self.work), "n": res["n"],
               "expected": res["expected"], "complete": res["complete"], "seconds": res["seconds"],
               "judge_rc": j.get("rc")}
        self.state["probes"].append(rec)
        self.save()
        return rec

    def ask_one(self, srv, q, thinking=False):
        a = self.args
        return lwrun.ask(srv.base, q["text"], thinking, a.max_tokens_on if thinking else a.max_tokens_off,
                         a.chat_timeout)["content"]

    def judge_against(self, q, content, store_ids):
        """(verdict, reading) of one answer against the rule book's answer for the store `store_ids`."""
        store = [self.les[i] for i in store_ids]
        exp = lwlogic.answer(self.S["schema"], store, self.S["entities"], q)
        mini = dict(self.S, phases=[{"name": "_now", "store": list(store_ids), "deleted": []}],
                    expected={"_now": {q["qid"]: exp}})
        verdict, g, _why = lwlogic.judge_one(mini, "_now", q, content)
        return verdict, g, exp

    def known_check(self, srv, lid, store_ids):
        """Ask the lesson's open CURRENT question before teaching it. Known only when the answer gives the lesson's
        exact value (the judge passes it against the store with the lesson added, and reads a value, not a guess).
        The yes/no is asked too and recorded, never used: wrong yeses about untaught facts are common (L1 purple)."""
        fq = self.L7["fact_questions"][str(lid)]
        qc, qy = self.qs[fq["current"]], self.qs[fq["yes_no"]]
        after = list(store_ids) + [lid]
        c = self.ask_one(srv, qc)
        vc, gc, _ = self.judge_against(qc, c, after)
        y = self.ask_one(srv, qy)
        vy, gy, ey = self.judge_against(qy, y, after)
        return {"current": {"question": qc["text"], "answer": c, "verdict": vc, "got": lwlogic.got_words(gc)},
                "yes_no": {"question": qy["text"], "answer": y, "verdict": vy, "got": lwlogic.got_words(gy),
                           "expected": lwlogic.words(ey)},
                "known_model": vc == "pass" and gc.get("got") in ("VALUE", "SET"),
                "yes_no_would_say_known": vy == "pass"}

    # ---------------------------------------------------------------------------------- norms
    def norms_of(self, folder, facts):
        """{fact: norms} for {fact: neurons}; None and a reason when the folder cannot be read that way."""
        if not facts:
            return {}, None
        try:
            return lwrun.pocket_norms(folder, facts), None
        except (ZeroSkip, OSError, KeyError, ValueError) as e:
            return None, f"{type(e).__name__}: {e}"

    # ---------------------------------------------------------------------------------- main step
    def victim(self, recs, now_pos):
        """The lesson to give way when the pocket is full, or None."""
        live = [r for r in recs if r.get("status") == "learned" and r["pos"] != now_pos and r.get("facts")]
        if not live:
            return None
        if self.args.when_full == "forget":
            return min(live, key=lambda r: r["pos"])
        zs = lwrun.norm_outliers({str(r["lesson"]): (r.get("norm") or {}).get("total") for r in live})["facts"]
        store = [self.les[r["lesson"]] for r in live]

        def covered(r):
            others = [x for x in store if x["id"] != r["lesson"]]
            return intake.classify(others, self.les[r["lesson"]])["verdict"] in (intake.ALREADY_KNOWN, intake.IMPLIED)

        return min(live, key=lambda r: (not covered(r), not (r.get("known") or {}).get("known_model"),
                                        not (zs.get(str(r["lesson"])) or {}).get("outlier"), r["pos"]))

    def step_main(self):
        a = self.args
        st = self.state["main"] = {"status": "running", "lessons": [], "evictions": [], "norm_checks": [],
                                   "evict": "on" if a.when_full != "refuse" else "off", "evict_note": None}
        self.save()
        folder = self.copy("main", self.src, "taught with --slide, one lesson at a time")
        st["folder"] = folder
        d = self.sdir("main")
        graph_file = os.path.join(d, "graph.json")
        srv = lwrun.Server(self, folder, True, os.path.join(d, "server.log"), graph_file, label="main --slide")
        srv.start()
        try:
            g = srv.graph()
            write_json(os.path.join(d, "graph-0-before.json"), g)
            if g.get("facts"):
                raise RunError(f"the slide graph of a fresh copy is not empty ({len(g['facts'])} facts)")
            neurons, recs = {}, st["lessons"]
            ck = self.checkpoint_at()
            for pos, lid in enumerate(self.L7["sequence"], 1):
                l = self.les[lid]
                store = [r["lesson"] for r in recs if r.get("status") == "learned"]  # what the pocket holds now
                rec = {"pos": pos, "lesson": lid, "name": l["name"], "text": l["text"], "key": [l["e"], l["a"]],
                       "value": l.get("v"), "attempts": []}
                recs.append(rec)
                parsed = intake.parse(l["text"])
                r = intake.classify([self.les[i] for i in store], l)
                rec["intake"] = {"reads_back": parsed is not None and intake.formal(parsed) == intake.formal(l),
                                 "verdict": r["verdict"], "same_as": r["same_as"], "covered_by": r["covered_by"],
                                 "supersedes": r["supersedes"]}
                if a.known_check == "on":
                    rec["known"] = self.known_check(srv, lid, store)
                known = (rec.get("known") or {}).get("known_model")
                if a.gate == "skip" and (r["verdict"] in (intake.ALREADY_KNOWN, intake.IMPLIED) or known):
                    rec["status"], rec["why"] = "skipped", ("already known (model)" if known else
                                                            f"intake: {r['verdict']}")
                    LOG(f"skip {l['name']} ({rec['why']}): {l['text']!r}")
                else:
                    g = self.teach_one(srv, d, l, rec, g, neurons, recs, pos)
                if rec["status"] == "learned" and a.norms == "on":
                    facts = {f: neurons[f] for f in map(str, rec["facts"]) if neurons.get(f)}
                    nm, why = self.norms_of(folder, facts)
                    rec["norm"] = combine(nm) if nm else None
                    if why:
                        rec["norm_skipped"] = why
                self.save()
                if pos in ck:
                    self.checkpoint(srv, folder, ck[pos], pos, recs, neurons)
            write_json(os.path.join(d, "graph-final.json"), g)
            st["fact_states"] = {str(f["id"]): f.get("state") for f in g.get("facts", [])}
            st["fact_neurons"] = neurons
            st["fact_lesson"] = {str(f): r["lesson"] for r in recs for f in r.get("all_facts", [])}
            sv = (g.get("model") or {}).get("save") or {}
            if sv.get("state") == "saving":
                srv.wait_save(int(sv.get("at") or 0))
        finally:
            srv.stop("main step done" if sys.exc_info()[0] is None else "main step failed")
        st["status"] = "done"
        self.save()
        if a.delete_mode == "zero" and "deletes" in self.steps:  # the zero copies are served without --slide
            srv = lwrun.Server(self, folder, False, os.path.join(self.sdir("main-restart"), "server.log"),
                               label="main restarted")
            srv.start()
            try:
                st["restart_probe"] = self.ask_phase(srv, "main-restart", "teach", self.S["questions"], "restart",
                                                     tag="teach")
            finally:
                srv.stop("restart probe done")
            self.save()

    def teach_one(self, srv, d, l, rec, g, neurons, recs, pos):
        """Learn one lesson; when the pocket is full, the --when-full policy. Returns the graph after it."""
        a, st = self.args, self.state["main"]
        tries = 0
        while True:
            tries += 1
            lesson = dict(l, name=l["name"] if tries == 1 else f"{l['name']}-try{tries}")
            res, g = lwrun.learn_one(self, srv, lesson, d, g, neurons)
            rec["attempts"].append({k: res.get(k) for k in ("name", "seconds", "recalled", "message", "fact_ids_sse",
                                                             "pocket_full", "changed_states")})
            rec.setdefault("all_facts", []).extend(x for x in res["fact_ids_sse"] if x is not None)
            for c in res.get("changed_states") or []:  # an earlier fact the server marked replaced (or anything else)
                for r in recs:
                    if c["id"] in (r.get("facts") or []) and r["status"] == "learned" and c.get("to") != "learned":
                        r["status"], r["why"] = f"server: {c.get('to')}", f"after {l['name']}: {c.get('message')}"
            if not res["pocket_full"]:
                rec["status"] = "learned" if res["recalled"] else "missed"
                rec["facts"] = [x for x in res["fact_ids_sse"] if x is not None]
                rec["loss"] = [f.get("loss") for f in res["facts"]]
                rec["neurons"] = sum(len(f.get("neurons") or []) for f in res["facts"])
                if rec["status"] == "missed":
                    rec["why"] = res.get("message")
                return g
            if st["evict"] != "on":
                rec["status"], rec["why"] = "refused_full", "the pocket is full (PocketFull); --when-full " + (
                    a.when_full if a.when_full == "refuse" else f"{a.when_full}, but {st['evict_note']}")
                return g
            if tries > a.evict_max:
                rec["status"], rec["why"] = "refused_full", f"still PocketFull after {a.evict_max} evictions"
                return g
            if tries >= 2 and pocket_used(g) is None:
                # the graph does not say how full each layer is, and the lesson came back PocketFull after an eviction
                return self.stop_evicting(rec, "the lesson came back PocketFull after a fact was deleted for it, and "
                                               "the graph does not show whether places were freed", g)
            v = self.victim(recs, pos)
            if v is None:
                rec["status"], rec["why"] = "refused_full", "the pocket is full and no learned fact can give way"
                return g
            used_before = pocket_used(g)
            codes, body = [], ""
            for fid in v["facts"]:
                code, body = srv.delete_fact(fid)
                codes.append(code)
                if code != 200:
                    break
            if any(c in NO_ENDPOINT for c in codes):
                self.stop_evicting(rec, f"the server has no per-fact delete (DELETE /v1/slide/fact/{v['facts'][0]} "
                                        f"answered HTTP {codes[-1]}), so nothing can give way")
                return g
            if codes[-1] != 200:
                raise RunError(f"DELETE /v1/slide/fact for {v['name']}: HTTP {codes[-1]}: {body[:200]}")
            g = srv.graph()
            used_after = pocket_used(g)
            v["status"], v["why"] = "evicted", f"gave way to {l['name']} ({a.when_full})"
            st["evictions"].append({"lesson": v["lesson"], "name": v["name"], "pos": v["pos"], "facts": v["facts"],
                                    "for_lesson": l["id"], "for_pos": pos, "policy": a.when_full,
                                    "pocket_used": [used_before, used_after]})
            LOG(f"pocket full at {l['name']}: {v['name']} (position {v['pos']}) deleted to make room ({a.when_full})"
                + (f"; neurons in use {used_before} -> {used_after}" if used_before is not None else ""))
            if used_before is not None and used_after is not None and used_after >= used_before:
                return self.stop_evicting(rec, f"deleting {v['name']} freed no place (the graph's pocket.used stayed "
                                               f"at {used_after}), so evicting more facts would only lose them", g)

    def stop_evicting(self, rec, why, g=None):
        """Forget and replace cannot make room on this server: the lesson is refused, and so is every later one."""
        st = self.state["main"]
        st["evict"], st["evict_note"] = "off", why + ": refused from here on"
        LOG(f"NOTE: {st['evict_note']}")
        rec["status"], rec["why"] = "refused_full", f"the pocket is full; {st['evict_note']}"
        return g

    def checkpoint(self, srv, folder, phase, pos, recs, neurons):
        taught = [r["lesson"] for r in recs]
        self.ask_phase(srv, "main", phase, self.questions_for(taught), "slide")
        if self.args.norms != "on":
            return
        facts = {str(f): neurons[str(f)] for r in recs if r.get("status") == "learned" for f in r.get("facts") or []
                 if neurons.get(str(f))}
        nm, why = self.norms_of(folder, facts)
        self.state["main"]["norm_checks"].append({"after": pos, "phase": phase, "skipped": why,
                                                  "facts": {f: x["total"] for f, x in (nm or {}).items()},
                                                  "up": {f: x["up"] for f, x in (nm or {}).items()}})
        self.save()

    # ---------------------------------------------------------------------------------- pairs
    def step_pairs(self):
        out = self.state["pairs"] = {}
        for kind, p in self.L7["pairs"].items():
            rec = out[kind] = {"a": p["a"], "b": p["b"], "why": p["why"], "status": "running", "learned": {}}
            self.save()
            folder = self.copy(f"ix-{kind}", self.src, f"{kind} pair: A, then B")
            d = self.sdir(f"ix-{kind}")
            srv = lwrun.Server(self, folder, True, os.path.join(d, "server.log"), os.path.join(d, "graph.json"),
                               label=f"ix-{kind} --slide")
            srv.start()
            try:
                g, neurons = srv.graph(), {}
                qs = self.questions_for([p["a"], p["b"]])
                for lid, phase in ((p["a"], p["phases"][0]), (p["b"], p["phases"][1])):
                    res, g = lwrun.learn_one(self, srv, self.les[lid], d, g, neurons)
                    rec["learned"][str(lid)] = {"recalled": res["recalled"], "message": res["message"],
                                                "pocket_full": res["pocket_full"]}
                    self.ask_phase(srv, f"ix-{kind}", phase, qs, "slide")
            finally:
                srv.stop(f"{kind} pair done")
            rec["status"] = "done"
            self.save()

    # ---------------------------------------------------------------------------------- deletes
    def step_deletes(self):
        a = self.args
        dl = self.L7["deletes"]
        out = self.state["deletes"] = {"mode": a.delete_mode, "order": dl["order"], "phases": dl["phases"],
                                       "steps": [], "status": "running"}
        self.save()
        if a.delete_mode == "none" or not dl["phases"]:
            out["status"] = "not run"
            return
        when = {c: name for name, c in dl["phases"].items()}
        if a.delete_mode == "reteach":
            for name, c in sorted(dl["phases"].items(), key=lambda x: x[1]):
                ph = next(p for p in self.S["phases"] if p["name"] == name)
                folder = self.copy(f"{name}.reteach", self.src, f"never taught: the {len(ph['store'])} lessons kept")
                d = self.sdir(f"{name}.reteach")
                srv = lwrun.Server(self, folder, True, os.path.join(d, "server.log"), os.path.join(d, "graph.json"),
                                   label=f"{name}.reteach --slide")
                srv.start()
                try:
                    g, neurons = srv.graph(), {}
                    for lid in ph["store"]:
                        res, g = lwrun.learn_one(self, srv, self.les[lid], d, g, neurons)
                        out["steps"].append({"phase": name, "lesson": lid, "recalled": res["recalled"],
                                             "pocket_full": res["pocket_full"]})
                    self.ask_phase(srv, f"{name}.reteach", name, self.S["questions"], "reteach")
                finally:
                    srv.stop("reteach done")
            out["status"] = "done"
            return
        main = self.state.get("main") or {}
        if main.get("status") != "done":
            raise RunError("the deletes need the main step's taught folder and records")
        states, neurons, owner = main["fact_states"], main["fact_neurons"], main["fact_lesson"]
        name = f"del-{a.delete_mode}"
        folder = self.copy(name, main["folder"], "a copy of the taught folder, facts deleted one at a time")
        d = self.sdir(name)
        gone = set()
        if a.delete_mode == "zero":
            zeroed_rows = []
            for i, lid in enumerate(dl["order"], 1):
                gone.add(lid)
                targets, kept = [], []
                for fid, ns in neurons.items():
                    for n in ns:
                        item = (int(fid), states.get(fid), int(n["layer"]), int(n["neuron"]))
                        if owner.get(fid) == lid:
                            targets.append(item)
                        elif owner.get(fid) not in gone and states.get(fid) == "learned":
                            kept.append(item)
                step = {"n": i, "lesson": lid, "facts": sorted({t[0] for t in targets})}
                live = [t for t in targets if t[1] == "learned"]
                if not live:
                    step["note"] = "nothing to zero: the lesson holds no learned fact (missed, refused or evicted)"
                else:
                    z = lwrun.zero_facts(folder, live, kept, ref=None, audit="off")
                    zeroed_rows += [r for r in z["rows"] if r["nonzero_before"]]
                    step["rows_zeroed"] = z["rows_zeroed"]
                    step["kept_nonzero"] = [z["kept_nonzero_before"], z["kept_nonzero_after"]]
                    if a.zero_audit != "off":
                        au = lwrun.audit_copy(folder, main["folder"], zeroed_rows, full=(a.zero_audit == "full"))
                        step["audit"] = {k: au[k] for k in ("ok", "bytes_differing", "outside_rows", "problems")}
                        if not au["ok"]:
                            raise RunError(f"byte audit failed after deleting {self.lesson_name(lid)}: {au}")
                out["steps"].append(step)
                self.save()
                LOG(f"zero delete {i}: {self.lesson_name(lid)}" + (f": {step['note']}" if step.get("note") else
                                                                   f", {step['rows_zeroed']} rows"))
                if i in when:
                    srv = lwrun.Server(self, folder, False, os.path.join(d, "server.log"), label=f"{when[i]} zero")
                    srv.start()
                    try:
                        self.ask_phase(srv, name, when[i], self.S["questions"], "zero")
                    finally:
                        srv.stop("probe done")
        else:  # endpoint: TensorFold's own per-fact delete, on a copy served with --slide
            graph = os.path.join(d, "graph.json")
            src_graph = os.path.join(self.work, "steps", "main", "graph.json")
            if os.path.exists(src_graph):
                with open(src_graph, "rb") as fi, open(graph, "wb") as fo:
                    fo.write(fi.read())
            srv = lwrun.Server(self, folder, True, os.path.join(d, "server.log"), graph, label=f"{name} --slide")
            srv.start()
            try:
                for i, lid in enumerate(dl["order"], 1):
                    facts = sorted(int(f) for f, o in owner.items() if o == lid)
                    codes = [srv.delete_fact(f)[0] for f in facts]
                    out["steps"].append({"n": i, "lesson": lid, "facts": facts, "http": codes})
                    self.save()
                    if any(c in NO_ENDPOINT for c in codes):
                        out["status"] = "unavailable"
                        out["note"] = (f"DELETE /v1/slide/fact answered HTTP {codes}: this server has no per-fact "
                                       "delete (1.0.8 has none); nothing was asked")
                        LOG(f"NOTE: {out['note']}")
                        return
                    if any(c != 200 for c in codes):
                        raise RunError(f"DELETE of {self.lesson_name(lid)}'s facts {facts}: HTTP {codes}")
                    if i in when:
                        self.ask_phase(srv, name, when[i], self.S["questions"], "endpoint")
            finally:
                srv.stop("endpoint deletes done")
        out["status"] = "done"
        self.save()

    # ---------------------------------------------------------------------------------- the whole run
    def run(self):
        a = self.args
        os.makedirs(self.work, exist_ok=True)
        LOG.path = os.path.join(self.work, "run.log")
        self.preflight()
        write_json(self.set_for_judge, self.S)
        self.bin_sha = lwrun.sha256(self.bin)
        self.state = {"format": FORMAT, "set": {"id": self.S["id"], "path": self.set_path,
                                               "sha256": lwrun.sha256(self.set_path), "facts": self.L7["facts"]},
                      "started": utc(), "args": {k: v for k, v in vars(a).items()},
                      "binary": {"path": self.bin, "sha256": self.bin_sha}, "probes": [], "status": "running"}
        self.save()
        LOG(f"capacity {self.S['id']}: {self.L7['facts']} facts, --when-full {a.when_full}, --gate {a.gate}, "
            f"steps {self.steps}, deletes {a.delete_mode}, thinking {a.thinking}")
        try:
            if "baseline" in self.steps:
                srv = lwrun.Server(self, self.src, False, os.path.join(self.sdir("baseline"), "server.log"),
                                   label="baseline")
                srv.start()
                try:
                    self.ask_phase(srv, "baseline", "baseline", self.S["questions"], "original")
                finally:
                    srv.stop("baseline done")
            if "main" in self.steps:
                self.step_main()
            if "pairs" in self.steps:
                self.step_pairs()
            if "deletes" in self.steps:
                self.step_deletes()
            self.state["status"] = "done"
        except RunError as e:
            LOG(f"capacity FAILED: {e}")
            self.state["status"], self.state["error"] = "failed", str(e)
        except lwrun.Interrupted as e:
            LOG(f"capacity interrupted by {e}")
            self.state["status"], self.state["error"] = "interrupted", f"interrupted by {e}"
        finally:
            lwrun.stop_all("capacity exiting")
            self.state["finished"] = utc()
            self.save()
            try:
                rep = build_report(self.work, judge=self)
                LOG(f"report: {os.path.join(self.work, 'capacity.txt')}")
                for line in rep["headline"]:
                    LOG("  " + line)
            except Exception as e:  # noqa: BLE001  (the records stay; `capacity.py report WORK` can try again)
                LOG(f"report could not be written: {type(e).__name__}: {e}")
            LOG.path = None
        return {"done": 0, "interrupted": 130}.get(self.state["status"], 1)


def lwrun_thinking(s):
    out = []
    for x in s.split(","):
        x = x.strip().lower()
        if x not in ("off", "on"):
            raise RunError(f"--thinking takes off and/or on, not {x!r}")
        out.append(x == "on")
    return out


def pocket_used(g):
    """Neurons in use over every pocket layer, from the graph's model.pocket.used (1.0.8 lists one count per pocket
    layer; READ from our stored graphs), or None when the graph does not say."""
    used = ((g or {}).get("model") or {}).get("pocket", {}).get("used")
    return sum(used) if isinstance(used, list) and all(isinstance(x, int) for x in used) else None


def combine(norms):
    """One lesson's norms from its facts' (a lesson is normally one fact)."""
    xs = list(norms.values())
    if not xs:
        return None
    sq = lambda k: sum((x.get(k) or 0.0) ** 2 for x in xs)
    has_down = all(x.get("down") is not None for x in xs)
    return {"up": round(sq("up") ** 0.5, 6), "down": round(sq("down") ** 0.5, 6) if has_down else None,
            "total": round(sq("total") ** 0.5, 6), "neurons": sum(x["neurons"] for x in xs), "facts": sorted(norms)}


# ============================================================================================ the report
def frac(xs):
    return [sum(xs), len(xs)]


def pct(c):
    return "-" if not c or not c[1] else f"{c[0]}/{c[1]}"


def load_verdicts(work, probes):
    """{(phase, tag): {(qid, thinking): result}} from every probe's verdicts file."""
    out = {}
    for p in probes:
        v = read_json(os.path.join(work, p["verdicts"]), {}) or {}
        out[(p["step"], p["tag"])] = {(r["qid"], bool(r["thinking"])): r for r in v.get("results", [])}
    return out


def build_report(work, judge=None):
    """capacity.json and capacity.txt from WORK/capacity-state.json and the verdicts on file. With `judge` (a
    Capacity), the kept_stable inputs are judged in one call first (teach and delete answers side by side)."""
    st = read_json(os.path.join(work, "capacity-state.json"))
    S = read_json(os.path.join(work, "set.json"))
    if not st or not S:
        raise RunError(f"{work}: no capacity-state.json or set.json")
    L7 = S["l7"]
    les = {l["id"]: l for l in S["lessons"]}
    V = load_verdicts(work, st.get("probes", []))
    modes = sorted({th for vs in V.values() for (_q, th) in vs}) or [False]
    mname = lambda th: "on" if th else "off"
    main = st.get("main") or {}
    recs = {r["lesson"]: r for r in main.get("lessons", [])}
    pos = {lid: i for i, lid in enumerate(L7["sequence"], 1)}
    status_at = {}  # the status a lesson had at each checkpoint: evicted only from the lesson that evicted it on
    for ev in main.get("evictions", []):
        status_at.setdefault(ev["lesson"], []).append(ev["for_pos"])
    rep = {"format": FORMAT, "set": S["id"], "facts": L7["facts"], "made": utc(), "status": st.get("status"),
           "args": {k: st["args"].get(k) for k in ("when_full", "evict_max", "gate", "known_check", "delete_mode",
                                                   "norms", "thinking", "steps")},
           "error": st.get("error")}
    # --- per lesson
    lessons = []
    for lid in L7["sequence"]:
        r = recs.get(lid) or {}
        k = r.get("known") or {}
        lessons.append({"pos": pos[lid], "lesson": lid, "text": les[lid]["text"], "status": r.get("status", "not run"),
                        "why": r.get("why"), "intake": (r.get("intake") or {}).get("verdict"),
                        "intake_reads_back": (r.get("intake") or {}).get("reads_back"),
                        "known_model": k.get("known_model"), "yes_no_would_say_known": k.get("yes_no_would_say_known"),
                        "facts": r.get("facts"), "neurons": r.get("neurons"), "loss": r.get("loss"),
                        "norm": r.get("norm"), "tries": len(r.get("attempts") or [])})
    rep["lessons"] = lessons
    by_status = {}
    for x in lessons:
        by_status.setdefault(x["status"], []).append(x["lesson"])
    full_at = next((x["pos"] for x in lessons if x["status"] == "refused_full" or
                    any(e["for_lesson"] == x["lesson"] for e in main.get("evictions", []))), None)
    rep["capacity"] = {"by_status": by_status, "first_full_at": full_at,
                       "learned_before_full": None if full_at is None else
                       sum(1 for x in lessons if x["pos"] < full_at and x["status"] in ("learned", "evicted")),
                       "evictions": main.get("evictions", []), "evict_note": main.get("evict_note")}
    rep["intake"] = {}
    for x in lessons:
        if x["intake"]:
            rep["intake"][x["intake"]] = rep["intake"].get(x["intake"], 0) + 1
    rep["intake_reads_back"] = frac([bool(x["intake_reads_back"]) for x in lessons if x["intake"]])
    rep["known"] = {"known_model": [x["lesson"] for x in lessons if x["known_model"]],
                    "yes_no_would_say_known": [x["lesson"] for x in lessons if x["yes_no_would_say_known"]],
                    "rule": "known only when the CURRENT answer gives the exact value; the yes/no is recorded, not used"}
    # --- the curve, at every checkpoint
    curve, matrix = [], {mname(th): {} for th in modes}
    first = L7["sequence"][:L7["first"]]
    for c in L7["checkpoints"]:
        vs = V.get(("main", c["phase"]))
        if not vs:
            continue
        upto = [lid for lid in L7["sequence"] if pos[lid] <= c["after"]]
        held = [lid for lid in upto if (recs.get(lid) or {}).get("status") in ("learned",) or
                ((recs.get(lid) or {}).get("status") == "evicted" and min(status_at.get(lid, [10 ** 9])) > c["after"])]
        point = {"after": c["after"], "phase": c["phase"], "taught_so_far": len(upto), "held_so_far": len(held)}
        for th in modes:
            def ok(lid, kind):
                x = vs.get((L7["fact_questions"][str(lid)][kind], th))
                return None if x is None else x["verdict"] == "pass"
            cur = {lid: ok(lid, "current") for lid in upto}
            yn = {lid: ok(lid, "yes_no") for lid in upto}
            ctl = [vs[(q, th)]["verdict"] == "pass" for q in L7["controls"] if (q, th) in vs]
            point[mname(th)] = {
                "current": frac([v for v in cur.values() if v is not None]),
                "yes_no": frac([v for v in yn.values() if v is not None]),
                "current_held": frac([cur[l] for l in held if cur.get(l) is not None]),
                "first_current": frac([cur[l] for l in first if l in cur and cur[l] is not None]),
                "first_yes_no": frac([yn[l] for l in first if l in yn and yn[l] is not None]),
                "first_current_held": frac([cur[l] for l in first if l in held and cur.get(l) is not None]),
                "controls": frac(ctl),
                "by_order": [[f"{i + 1}-{min(i + 10, c['after'])}",
                              frac([cur[l] for l in upto[i:i + 10] if cur.get(l) is not None])]
                             for i in range(0, len(upto), 10)]}
            for lid in upto:
                matrix[mname(th)].setdefault(str(lid), {})[c["phase"]] = (
                    "-" if cur[lid] is None else "P" if cur[lid] else
                    "U" if (vs.get((L7["fact_questions"][str(lid)]["current"], th)) or {}).get("verdict") == "unclear"
                    else "F")
        curve.append(point)
    rep["curve"] = curve
    rep["report_at"] = {str(k): next((p for p in curve if p["after"] == k), None) for k in L7["report_at"]}
    rep["matrix_current"] = matrix
    first_held = {mname(th): [p[mname(th)]["first_current_held"] for p in curve if mname(th) in p] for th in modes}
    rep["first"] = {"facts": first, "current_held": first_held,
                    "dropped": {m: any(s[i][0] * s[j][1] < s[j][0] * s[i][1] for i in range(len(s))
                                       for j in range(i) if s[i][1] and s[j][1]) for m, s in first_held.items()}}
    # --- norms
    norms = {str(x["lesson"]): (x["norm"] or {}).get("total") for x in lessons if x["norm"]}
    out = lwrun.norm_outliers(norms)
    drift = []
    checks = main.get("norm_checks", [])
    seen = {}
    for chk in checks:
        for f, v in (chk.get("facts") or {}).items():
            if f in seen and abs(v - seen[f][1]) > 1e-6 * max(1.0, abs(seen[f][1])):
                drift.append({"fact": int(f), "from": seen[f][1], "to": v, "first_after": seen[f][0],
                              "after": chk["after"]})
            seen.setdefault(f, (chk["after"], v))
    rep["norms"] = {"per_lesson": {k: (recs[int(k)].get("norm")) for k in norms}, "outliers": out,
                    "drift": drift, "checks": len(checks),
                    "skipped": sorted({r.get("norm_skipped") for r in recs.values() if r.get("norm_skipped")}),
                    "flagged": [int(x) for x in out["outliers"]]}
    # --- pairs
    pairs = {}
    for kind, p in L7["pairs"].items():
        cell = {}
        for ph_i, ph in enumerate(p["phases"]):
            vs = V.get((f"ix-{kind}", ph)) or {}
            for th in modes:
                for who, lid in (("A", p["a"]), ("B", p["b"])):
                    qids = L7["fact_questions"][str(lid)]
                    xs = [vs[(q, th)]["verdict"] == "pass" for q in (qids["current"], qids["yes_no"]) if (q, th) in vs]
                    cell.setdefault(mname(th), {})[f"{who} after {'A' if ph_i == 0 else 'A then B'}"] = frac(xs)
        drop = {m: (c.get("A after A", [0, 0])[0] - c.get("A after A then B", [0, 0])[0]) for m, c in cell.items()}
        pairs[kind] = {"a": p["a"], "b": p["b"], "why": p["why"], "matrix": cell, "a_lost_after_b": drop,
                       "learned": ((st.get("pairs") or {}).get(kind) or {}).get("learned")}
    rep["pairs"] = pairs
    # --- deletes
    dl = st.get("deletes") or {}
    rep["deletes"] = {"mode": dl.get("mode"), "status": dl.get("status"), "note": dl.get("note"), "phases": {}}
    step = {"zero": "del-zero", "endpoint": "del-endpoint"}.get(dl.get("mode"))
    kept_inputs = []
    for name, c in (L7["deletes"]["phases"] or {}).items():
        sname = step or f"{name}.reteach"
        vs = V.get((sname, name))
        if not vs:
            continue
        ph = next(x for x in S["phases"] if x["name"] == name)
        gone = ph["deleted"]
        rec = {"deleted": gone}
        for th in modes:
            g_ok = [vs[(L7["fact_questions"][str(l)]["current"], th)]["verdict"] == "pass" for l in gone
                    if (L7["fact_questions"][str(l)]["current"], th) in vs]
            kept = [vs[(q["qid"], th)]["verdict"] == "pass" for q in S["questions"]
                    if q.get("fact") not in gone and (q["qid"], th) in vs]
            rec[mname(th)] = {"deleted_now_dont_know": frac(g_ok), "rest_right": frac(kept)}
        rep["deletes"]["phases"][name] = rec
        p = next((x for x in st["probes"] if x["step"] == sname and x["tag"] == name), None)
        if p:
            kept_inputs.append(os.path.join(work, p["answers"]))
    teach = next((x for x in st.get("probes", []) if x["step"] == ("main-restart" if dl.get("mode") == "zero" else "main")
                  and x["tag"] == "teach"), None)
    if judge is not None and teach and kept_inputs:
        comb = os.path.join(work, "combined")
        os.makedirs(comb, exist_ok=True)
        j = lwrun.run_judge(judge, [os.path.join(work, teach["answers"])] + kept_inputs,
                            os.path.join(comb, "verdicts-kept.json"), os.path.join(comb, "report-kept.txt"),
                            "kept_stable: teach and deletes")
        st["kept_stable"] = j.get("kept_stable") or {}
        write_json(os.path.join(work, "capacity-state.json"), st)
    rep["deletes"]["kept_stable"] = st.get("kept_stable")
    rep["activation"] = ACTIVATION_REQUEST
    rep["headline"] = headline(rep)
    write_json(os.path.join(work, "capacity.json"), rep)
    with open(os.path.join(work, "capacity.txt"), "w", encoding="utf-8") as f:
        f.write(report_text(rep) + "\n")
    return rep


def headline(rep):
    out = []
    cap = rep["capacity"]
    bs = cap["by_status"]
    out.append(f"{rep['set']}: {rep['facts']} facts; " + ", ".join(f"{k} {len(v)}" for k, v in sorted(bs.items())))
    if cap["first_full_at"]:
        out.append(f"pocket full at position {cap['first_full_at']} ({cap['learned_before_full']} learned before it); "
                   f"--when-full {rep['args']['when_full']}: {len(cap['evictions'])} evicted"
                   + (f" ({cap['evict_note']})" if cap["evict_note"] else ""))
    for k, p in rep["report_at"].items():
        if p:
            out.append(f"after {k:>3}: " + "; ".join(
                f"thinking {m} CURRENT {pct(p[m]['current'])}, yes/no {pct(p[m]['yes_no'])}, first 10 "
                f"{pct(p[m]['first_current'])}" for m in ("off", "on") if m in p))
    if rep["norms"]["flagged"]:
        out.append(f"weight-norm outliers: lessons {rep['norms']['flagged']}")
    return out


def report_text(rep):
    L = [f"Level 7 capacity report: {rep['set']} ({rep['facts']} facts), status {rep['status']}"
         + (f", error: {rep['error']}" if rep.get("error") else ""),
         "settings: " + ", ".join(f"{k} {v}" for k, v in rep["args"].items()), ""]
    L += rep["headline"] + [""]
    L.append("Retention curve (CURRENT right / facts taught so far; held = learned and not evicted at that point):")
    L.append(f"  {'after':>5} {'phase':8} {'mode':4} {'CURRENT':>9} {'held':>9} {'yes/no':>9} {'first10':>9} "
             f"{'controls':>9}  by order (CURRENT)")
    for p in rep["curve"]:
        for m in ("off", "on"):
            if m in p:
                x = p[m]
                L.append(f"  {p['after']:>5} {p['phase']:8} {m:4} {pct(x['current']):>9} {pct(x['current_held']):>9} "
                         f"{pct(x['yes_no']):>9} {pct(x['first_current']):>9} {pct(x['controls']):>9}  "
                         + " ".join(f"{b}:{pct(c)}" for b, c in x["by_order"]))
    L.append("")
    fr = rep["first"]
    L.append(f"First {len(fr['facts'])} facts, held ones only (CURRENT): " + "; ".join(
        f"thinking {m}: {' '.join(pct(c) for c in s)}" + (" (drops)" if fr["dropped"].get(m) else "")
        for m, s in fr["current_held"].items()))
    cap = rep["capacity"]
    L.append(f"Capacity: first PocketFull at position {cap['first_full_at']}; evictions "
             + (", ".join(f"L{e['lesson']} (pos {e['pos']}) for L{e['for_lesson']}" for e in cap["evictions"]) or "none")
             + (f"; {cap['evict_note']}" if cap["evict_note"] else ""))
    L.append(f"Intake verdicts: {rep['intake']}; sentences read back {pct(rep['intake_reads_back'])}")
    k = rep["known"]
    L.append(f"Already known (model): lessons {k['known_model'] or 'none'}; a yes/no would have said known for "
             f"{k['yes_no_would_say_known'] or 'none'} (recorded, not used)")
    n = rep["norms"]
    o = n["outliers"]
    L.append(f"Weight norms: {len(n['per_lesson'])} lessons, median {o['median']}, outliers (modified z > "
             f"{o['z_cut']} or > {o['ratio_cut']}x the median): {n['flagged'] or 'none'}; drift at later checkpoints: "
             f"{len(n['drift'])}" + (f"; not read: {n['skipped']}" if n["skipped"] else ""))
    for kind, p in rep["pairs"].items():
        L.append(f"Pair {kind} (A L{p['a']}, B L{p['b']}: {p['why']}): " + "; ".join(
            f"thinking {m}: " + ", ".join(f"{c} {pct(v)}" for c, v in cells.items()) + f", A lost {p['a_lost_after_b'][m]}"
            for m, cells in p["matrix"].items()))
    d = rep["deletes"]
    L.append(f"Deletes ({d['mode']}, {d['status']})" + (f": {d['note']}" if d.get("note") else ""))
    for name, x in d["phases"].items():
        L.append(f"  {name}: deleted {x['deleted']}; " + "; ".join(
            f"thinking {m}: deleted facts now don't know {pct(x[m]['deleted_now_dont_know'])}, the rest right "
            f"{pct(x[m]['rest_right'])}" for m in ("off", "on") if m in x))
    for name, ks in (d.get("kept_stable") or {}).items():
        L.append(f"  kept_stable {name}: " + "; ".join(
            f"thinking {m}: same verdict {c['same_verdict']}/{c['n']}, same text {c['same_text']}/{c['n']}"
            + (f", moved {c['moved']}" if c["moved"] else "") for m, c in ks.items()))
    L.append(f"Usage ranking: {rep['activation']}")
    return "\n".join(L)


# ============================================================================================ dry run
def dry_run(c):
    a = c.args
    LOG.prefix = "[dry-run] "
    ok = True

    def check(cond, good, bad):
        nonlocal ok
        LOG(("ok    " if cond else "PROBLEM ") + (good if cond else bad))
        ok = ok and bool(cond)

    try:
        S, L7 = c.S, c.L7
        LOG(f"set {c.set_path}: {S['id']}, {L7['facts']} facts, {len(S['questions'])} questions, checkpoints "
            f"{[x['after'] for x in L7['checkpoints']]}, pairs {list(L7['pairs'])}, delete order {L7['deletes']['order']}")
        check(os.path.isfile(c.bin) and os.access(c.bin, os.X_OK), f"binary {c.bin}", f"binary {c.bin} is not executable")
        check(os.path.isfile(os.path.join(c.src, "config.json")), f"source {c.src}", f"source {c.src} has no config.json")
        check(not lwrun.port_busy(c.port), f"port {c.port} is free", f"port {c.port} is in use")
        check(not os.path.exists(os.path.join(c.work, "capacity-state.json")), f"work dir {c.work}: no earlier run",
              f"{c.work} holds an earlier capacity run")
        bad = [l["name"] for l in S["lessons"] if not (intake.parse(l["text"]) and
                                                       intake.formal(intake.parse(l["text"])) == intake.formal(l))]
        check(not bad, "the intake parser reads every lesson back to the set's own lesson", f"not read back: {bad}")
        nth = len(c.thinking)
        chats = learns = copies = 0
        if "baseline" in c.steps:
            chats += len(S["questions"]) * nth
            LOG(f"baseline: serve the source read-only; {len(S['questions'])} questions x {nth}")
        if "main" in c.steps:
            copies += 1
            learns += L7["facts"]
            pre = 2 * L7["facts"] if a.known_check == "on" else 0
            ck = sum(len(c.questions_for(L7["sequence"][:x["after"]])) for x in L7["checkpoints"]) * nth
            chats += pre + ck
            LOG(f"main: copy, serve --slide, {L7['facts']} lessons in order; before each: intake, "
                + ("the CURRENT and yes/no question (thinking off), " if pre else "")
                + f"gate {a.gate}; when full: {a.when_full}"
                + (f" (up to {a.evict_max} evictions a lesson)" if a.when_full != "refuse" else "") + "; "
                f"norms {a.norms}; {len(L7['checkpoints'])} checkpoints, {ck} chats")
            if a.delete_mode == "zero" and "deletes" in c.steps:
                chats += len(S["questions"]) * nth
                LOG("main: then serve the taught folder without --slide and ask every question (teach, restarted)")
        if "pairs" in c.steps:
            for kind, p in L7["pairs"].items():
                copies += 1
                learns += 2
                chats += 2 * len(c.questions_for([p["a"], p["b"]])) * nth
                LOG(f"pair {kind}: a fresh copy taught L{p['a']} then L{p['b']} ({p['why']}), asked after each")
        if "deletes" in c.steps and a.delete_mode != "none":
            np_ = len(L7["deletes"]["phases"])
            chats += np_ * len(S["questions"]) * nth
            copies += 1 if a.delete_mode != "reteach" else np_
            if a.delete_mode == "reteach":
                learns += sum(len(p["store"]) for p in S["phases"] if p["name"] in L7["deletes"]["phases"])
            LOG(f"deletes ({a.delete_mode}): {L7['deletes']['order']} one at a time, asked at "
                f"{sorted(L7['deletes']['phases'].values())}")
        LOG(f"totals: {learns} learn requests (2 to 4 min each in our 1.0.8 runs: about {learns * 2 / 60:.1f} to "
            f"{learns * 4 / 60:.1f} h), {chats} chats, {copies} copies of the model")
        LOG("dry run: nothing was copied, served, written or judged" + ("" if ok else "; PROBLEMS above"))
    finally:
        LOG.prefix = ""
    return 0 if ok else 2


# ============================================================================================ self-test
def say_answer(S, q, x):
    """A plain answer the judge reads as the expected answer `x` (the mock's answer when the fact is live)."""
    k, you = x["kind"], (lambda v: "you" if v == "me" else v)
    if k == "value":
        return f"It is {you(x['value'])}."
    if k == "set":
        return f"{lwlogic.render('{nom}', q['e'])} like{'' if q['e'] == 'me' else 's'} {' and '.join(x['yes'])}."
    if k == "yn":
        return {"yes": "Yes.", "no": "No.", "unk": "I don't know."}[x["value"]]
    if k == "gk":
        return " and ".join(x["any_of"][0]).capitalize() + "."
    return "I don't know."


def mock_answers(S, base_knows=(), yes_to_all=False, base_hedge=(), base_wrong=()):
    """The mock's answer table for a level 7 set: each fact question answered right while its lesson is live, else
    "I don't know." (base_knows: lessons whose CURRENT the untaught model already answers right; yes_to_all: the
    untaught model says yes to every yes/no, the wrong-yes habit of our 1.0.8 runs; base_hedge and base_wrong:
    lessons whose CURRENT the untaught model answers with the right value as a guess, or with another value)."""
    les = {l["id"]: l for l in S["lessons"]}
    out = {}
    for q in S["questions"]:
        x = S["expected"]["teach"][q["qid"]]
        if q.get("fact"):
            t = les[q["fact"]]["text"]
            right = say_answer(S, q, x)
            rows = [[t, right]]
            if q["fact"] in base_knows and q["type"] == "CURRENT":
                rows.append([None, right])
            elif q["fact"] in base_hedge and q["type"] == "CURRENT":
                rows.append([None, f"It is probably {x['value']}."])
            elif q["fact"] in base_wrong and q["type"] == "CURRENT":
                rows.append([None, f"It is {next(v for v in S['schema'][q['a']]['domain'] if v != x['value'])}."])
            elif yes_to_all and q["type"] == "YES_NO":
                rows.append([None, "Yes, you told me that."])
            out[q["text"]] = rows
        else:
            out[q["text"]] = [[None, say_answer(S, q, x)]]
    return out


def selftest(root=None):
    import lwrun_selftest as T  # the runner's mock TensorFold server and its test lab
    t0 = time.time()
    root = root or tempfile.mkdtemp(prefix="capacity-selftest-")
    print(f"capacity selftest in {root} (kept: nothing is deleted)")
    C = lwlogic.Check()
    lab = T.Lab(root, down=True)
    S = lwlogic.generate(7, 1, n_facts=12)
    setp = os.path.join(root, "L7-12.json")
    write_json(setp, S)
    L7 = S["l7"]
    les = {l["id"]: l for l in S["lessons"]}
    seq = L7["sequence"]
    text = lambda pos: les[seq[pos - 1]]["text"]
    big, known_pos, miss_pos = 3, 2, 4  # positions: a planted large norm, a fact the untaught model knows, a miss
    prog = [sys.executable, os.path.join(HERE, "capacity.py"), "run"]
    base = ["--thinking", "off", "--stop-timeout", "10", "--startup-timeout", "60", "--save-line-wait", "0"]

    def scen(**kw):
        sc = {"answers": mock_answers(S, base_knows=kw.pop("base_knows", [seq[known_pos - 1]]),
                                      yes_to_all=kw.pop("yes_to_all", True), base_hedge=kw.pop("base_hedge", ()),
                                      base_wrong=kw.pop("base_wrong", ())),
              "capacity": 8, "big_norm": [text(big)]}
        sc.update(kw)
        return {"l7": sc, "miss": [text(miss_pos)] if kw.get("with_miss", True) else []}

    def go(name, extra, sc):
        r = lab.run(name, setp, base + list(extra), sc, prog=prog)
        rep = read_json(os.path.join(r.work, "capacity.json"), {}) or {}
        state = read_json(os.path.join(r.work, "capacity-state.json"), {}) or {}
        return r, rep, state

    def statuses(rep):
        return [x["status"] for x in rep.get("lessons", [])]

    # 1. refuse: capacity 8, one lesson missed for another reason, one with a planted large norm, one the untaught
    # model already knows, and an untaught model that says yes to every yes/no
    rA, A, sA = go("refuse", ["--when-full", "refuse", "--delete-mode", "zero"], scen())
    want = ["missed" if i == miss_pos else "learned" if i <= 9 else "refused_full" for i in range(1, 13)]
    C.add("refuse: the run finishes; 8 facts fit (a missed lesson takes no place), the 9th to 12th come back "
          "PocketFull and are refused; teaching went on after the first", rA.rc == 0 and statuses(A) == want
          and A["capacity"]["first_full_at"] == 10 and not A["capacity"]["evictions"], len(want),
          f"rc {rA.rc}, statuses {statuses(A)}, first full at {A.get('capacity', {}).get('first_full_at')}"
          + (f"; {rA.err.strip()[-300:]}" if rA.rc else ""))
    pts = {p["after"]: p["off"] for p in A.get("curve", [])}
    got = {k: (v["current"], v["current_held"], v["yes_no"], v["controls"]) for k, v in pts.items()}
    held = [i for i in range(1, 13) if i != miss_pos and i <= 9]       # what the mock holds: 8 places, one missed
    yes = lambda i: S["expected"]["teach"][L7["fact_questions"][str(seq[i - 1])]["yes_no"]]["value"] == "yes"
    nc = len(L7["controls"])
    want_c = {}
    for k in (1, 5, 10, 12):  # a fact it does not hold: "I don't know.", and "Yes, you told me that." to the yes/no
        h = [i for i in held if i <= k]
        want_c[k] = ([len(h), k], [len(h), len(h)], [len(h) + sum(yes(i) for i in range(1, k + 1) if i not in h), k],
                     [nc, nc])
    C.add("retention curve at 1, 5, 10 and 12 facts: CURRENT and yes/no right exactly for the facts the mock holds, "
          "the controls right; 'held' leaves out the missed and refused ones", got == want_c, len(got), f"{got}")
    kn = A.get("known") or {}
    C.add("already known (model): only the lesson whose CURRENT the untaught model answers with the exact value; "
          "control: reading the yes/no instead would have marked the lessons whose yes/no asks the taught value",
          kn.get("known_model") == [seq[known_pos - 1]] and len(kn.get("yes_no_would_say_known") or []) >= 2
          and set(kn["known_model"]) != set(kn["yes_no_would_say_known"]), 1,
          f"known {kn.get('known_model')}, a yes/no would say {kn.get('yes_no_would_say_known')}")
    C.add("intake: every lesson read back by the parser and sorted NEW before it was taught",
          A.get("intake") == {"NEW": 12} and A.get("intake_reads_back") == [12, 12], 12, f"{A.get('intake')}")
    # norms: the reader's numbers against the ones the mock wrote (worked out apart from lwrun's reader)
    nm = (A.get("norms") or {}).get("per_lesson") or {}
    fid = {x["lesson"]: (x["facts"] or [None])[0] for x in A.get("lessons", [])}
    bad_n = []
    for lid, x in nm.items():
        up, down = T.l7_norms(fid[int(lid)], big=les[int(lid)]["text"] == text(big))
        if abs(x["up"] - up) > 1e-3 * up or abs(x["down"] - down) > 1e-3 * down:
            bad_n.append((lid, x["up"], up, x["down"], down))
    C.add("pocket weight norms: read from the saved folder (up_proj rows, down_proj columns, 4-bit codes with their "
          "scales) equal the values the mock wrote, for every learned fact; the planted large one is the only "
          "outlier", len(nm) == 8 and not bad_n and A["norms"]["flagged"] == [seq[big - 1]] and not A["norms"]["drift"],
          len(nm), f"mismatches {bad_n[:3]}, flagged {A.get('norms', {}).get('flagged')}")
    main_folder = (sA.get("main") or {}).get("folder")
    neur = (sA.get("main") or {}).get("fact_neurons") or {}
    shifted = {f: [dict(n, neuron=n["neuron"] + 1) for n in ns] for f, ns in neur.items()}
    try:
        sh = lwrun.pocket_norms(main_folder, shifted) if main_folder else {}
    except ZeroSkip:
        sh = {}
    sh_bad = [f for f, x in sh.items() if abs(x["up"] - T.l7_norms(int(f), big=False)[0]) > 1e-3]
    C.add("negative control: a reader one neuron off reads other rows, and the norms no longer match the mock's",
          len(sh_bad) >= len(sh) - 1 and len(sh) > 0, len(sh), f"{len(sh_bad)} of {len(sh)} differ")
    fr = A.get("first") or {}
    pr = A.get("pairs") or {}
    C.add("pairs: A learned on a fresh copy, then B; with no interference A keeps its answers after B (similar and "
          "far)", all((p.get("a_lost_after_b") or {}).get("off") == 0 and p["matrix"]["off"]["A after A"] == [2, 2]
                      for p in pr.values()) and len(pr) == 2, 2, f"{ {k: p.get('matrix') for k, p in pr.items()} }")
    dl = A.get("deletes") or {}
    ks = dl.get("kept_stable") or {}
    # a deleted fact says don't know again, except the one the untaught model knew anyway: it still answers
    gone_ok = all(x["off"]["deleted_now_dont_know"] == [len([g for g in x["deleted"] if g != seq[known_pos - 1]]),
                                                        len(x["deleted"])] for x in (dl.get("phases") or {}).values())
    stable = all(c["same_verdict"] == c["n"] and c["same_text"] == c["n"] for k in ks.values() for c in k.values())
    C.add("deletes by zeroing, one at a time in one copy: after 1, 3 and 6 deletes the deleted facts say don't know "
          "(but the one the untaught model knew) and kept_stable finds every other answer as in the teach phase", sorted(dl.get("phases") or {}) ==
          ["del-1", "del-3", "del-6"] and gone_ok and stable and sorted(ks) == ["del-1", "del-3", "del-6"], 3,
          f"phases {dl.get('phases')}, kept_stable {ks}")

    # 2. forget and replace with a delete endpoint that frees places
    rF, F, _ = go("forget", ["--when-full", "forget", "--steps", "main", "--delete-mode", "none"],
                  scen(delete_endpoint=True))
    ev = [e["pos"] for e in (F.get("capacity") or {}).get("evictions", [])]
    # held at each checkpoint: a fact evicted later still counts where it was held (lessons 1 to 3 at 1 and 5)
    heldF = [(p_["after"], p_["held_so_far"], p_["off"]["current_held"]) for p_ in F.get("curve", [])]
    C.add("forget: each lesson that comes back PocketFull deletes the oldest learned fact and is learned again; the "
          "facts held at the end are right; the curve counts an evicted fact as held at the checkpoints before its "
          "eviction (1 of 1, 4 of 5, 8 of 10, 8 of 12)", rF.rc == 0 and ev == [1, 2, 3]
          and statuses(F)[:3] == ["evicted"] * 3 and statuses(F)[9:] == ["learned"] * 3
          and (F["report_at"]["12"] or {}).get("off", {}).get("current_held") == [8, 8]
          and heldF == [(1, 1, [1, 1]), (5, 4, [4, 4]), (10, 8, [8, 8]), (12, 8, [8, 8])], len(ev),
          f"evicted positions {ev}, statuses {statuses(F)}, held (after, held, CURRENT right of held) {heldF}")
    rR, R, _ = go("replace", ["--when-full", "replace", "--steps", "main", "--delete-mode", "none"],
                  scen(delete_endpoint=True))
    evr = [e["pos"] for e in (R.get("capacity") or {}).get("evictions", [])]
    C.add("replace: the facts that give way are chosen: first the one the model already knew, then the weight-norm "
          "outlier, then the oldest; control: forget gave way oldest first", rR.rc == 0
          and evr == [known_pos, big, 1] and evr != ev, len(evr), f"replace {evr}, forget {ev}")
    # 3. forget on a server with no per-fact delete (1.0.8), and on one whose delete keeps the places taken
    r4, F4, _ = go("forget-no-endpoint", ["--when-full", "forget", "--steps", "main", "--delete-mode", "none"],
                   scen(delete_endpoint=False))
    note4 = (F4.get("capacity") or {}).get("evict_note") or ""
    r5, F5, _ = go("forget-keeps-places", ["--when-full", "forget", "--steps", "main", "--delete-mode", "none"],
                   scen(delete_endpoint=True, delete_frees=False))
    ev5 = [e["pos"] for e in (F5.get("capacity") or {}).get("evictions", [])]
    C.add("forget without a per-fact delete (HTTP 404, as 1.0.8) refuses instead and says why; with a delete that "
          "keeps the places taken, one fact is lost, the run notices and evicts no more", r4.rc == 0
          and statuses(F4) == want and "no per-fact delete" in note4 and r5.rc == 0 and ev5 == [1]
          and statuses(F5)[9:] == ["refused_full"] * 3 and "freed no place" in ((F5.get("capacity") or {}).get(
              "evict_note") or ""), 2,
          f"404: {statuses(F4)}; places kept: evicted {ev5}, {statuses(F5)}")

    # 4. gradual forgetting inside the model, and interference in the similar pair
    sim = L7["pairs"]["similar"]
    rG, G, _ = go("fade", ["--steps", "main,pairs", "--delete-mode", "none"],
                  scen(capacity=64, fade_after=5, interfere={les[sim["b"]]["text"]: les[sim["a"]]["text"]},
                       with_miss=False, base_knows=[], yes_to_all=False))
    s = (G.get("first") or {}).get("current_held", {}).get("off")
    gp = G.get("pairs") or {}
    C.add("first 10 facts tracked on their own: when early facts fade (the mock forgets a fact once 5 more are "
          "learned), the series drops and is flagged; control: the refuse run's series does not drop",
          s is not None and (G["first"]["dropped"] or {}).get("off") is True and (fr.get("dropped") or {}).get("off")
          is False, len(s or []), f"fade: {s}; refuse run: {fr.get('current_held', {}).get('off')}")
    C.add("interference matrix: teaching B of the similar pair takes A's answers away and the matrix shows it; the far "
          "pair keeps A; control: the refuse run (no interference) lost nothing", (gp.get("similar") or {}).get(
              "a_lost_after_b", {}).get("off") == 2 and (gp.get("far") or {}).get("a_lost_after_b", {}).get("off") == 0,
          2, f"{ {k: (p.get('matrix', {}).get('off'), p.get('a_lost_after_b')) for k, p in gp.items()} }")

    # 5. deletes through a per-fact delete endpoint, where one kept fact leans on a deleted one
    order = L7["deletes"]["order"]
    lean_on = next(l for l in seq if l not in order)
    rH, H, _ = go("endpoint-deletes", ["--steps", "main,deletes", "--delete-mode", "endpoint"],
                  scen(capacity=64, delete_endpoint=True, with_miss=False, base_knows=[], yes_to_all=False,
                       depends={les[lean_on]["text"]: les[order[0]]["text"]}))
    kh = (H.get("deletes") or {}).get("kept_stable") or {}
    moved = [m for c in (kh.get("del-1") or {}).values() for m in c["moved"]]
    want_moved = {L7["fact_questions"][str(lean_on)]["current"], L7["fact_questions"][str(lean_on)]["yes_no"]}
    C.add("deletes through DELETE /v1/slide/fact: kept_stable flags the kept fact that leaned on the deleted one, and "
          "only it; control: the zero deletes of the refuse run kept every answer", rH.rc == 0 and
          {m.split()[0] for m in moved} == want_moved and stable, len(moved), f"moved at del-1: {moved}")

    # 6. the gate: lessons the model already knows are not taught, and still count. Two more lessons the untaught
    # model answers, one with the right value said as a guess and one with another value: neither is known.
    vpos = [i for i in range(3, 13) if S["expected"]["teach"][L7["fact_questions"][str(seq[i - 1])]["current"]]["kind"]
            == "value"]
    hedge_pos, wrong_pos = vpos[0], vpos[1]
    rK, K, sK = go("gate-skip", ["--gate", "skip", "--steps", "main", "--delete-mode", "none", "--thinking", "off,on"],
                   scen(capacity=64, with_miss=False, base_hedge=[seq[hedge_pos - 1]], base_wrong=[seq[wrong_pos - 1]]))
    taught_texts = [x["text"] for x in (sK.get("main") or {}).get("lessons", []) if x.get("attempts")]
    at12 = K.get("report_at", {}).get("12") or {}
    C.add("--gate skip: the lesson the model already knows is not sent to the server, the others are; it is still "
          "answered right, with thinking off and on", rK.rc == 0 and statuses(K).count("skipped") == 1
          and text(known_pos) not in taught_texts and len(taught_texts) == 11
          and [at12.get(m, {}).get("current") for m in ("off", "on")] == [[12, 12], [12, 12]], 1,
          f"statuses {statuses(K)}, at 12: {[at12.get(m, {}).get('current') for m in ('off', 'on')]}")
    kk = {x["pos"]: (x.get("known") or {}) for x in (sK.get("main") or {}).get("lessons", [])}
    hk, wk = kk.get(hedge_pos) or {}, kk.get(wrong_pos) or {}
    lax = sorted(p_ for p_, x in kk.items() if (x.get("current") or {}).get("verdict") in ("pass", "unclear"))
    C.add("already known (model) only on the exact value: an untaught model that names the right value as a guess "
          "('It is probably ...') or names another value is not marked known, and --gate skip teaches both; control: "
          "a rule that also took a guess would have marked the guessed lesson", rK.rc == 0
          and hk.get("known_model") is False and (hk.get("current") or {}).get("verdict") == "unclear"
          and wk.get("known_model") is False and (wk.get("current") or {}).get("verdict") == "fail"
          and statuses(K)[hedge_pos - 1] == "learned" and statuses(K)[wrong_pos - 1] == "learned"
          and hedge_pos in lax and wrong_pos not in lax, 2,
          f"guess at {hedge_pos}: {(hk.get('current') or {}).get('verdict')}, known {hk.get('known_model')}; other value "
          f"at {wrong_pos}: {(wk.get('current') or {}).get('verdict')}, known {wk.get('known_model')}; a lax rule marks {lax}")
    before = read_json(os.path.join(rA.work, "capacity.json"), {}) or {}
    pr = subprocess.run([sys.executable, os.path.join(HERE, "capacity.py"), "report", rA.work], capture_output=True,
                        text=True, timeout=300)
    after = read_json(os.path.join(rA.work, "capacity.json"), {}) or {}
    C.add("`capacity.py report WORK` judges the kept_stable inputs again and writes the same report from the records",
          pr.returncode == 0 and "Retention curve" in pr.stdout and after.get("curve") == before.get("curve")
          and after.get("deletes") == before.get("deletes") and after.get("pairs") == before.get("pairs"), 1,
          f"rc {pr.returncode}")

    # 7. lwrun refuses a level 7 set; the dry run writes nothing
    r8 = lab.run("lwrun-L7", setp, ["--thinking", "off"])
    r9 = lab.run("dry", setp, base + ["--dry-run"], prog=prog)
    C.add("lwrun.py refuses a level 7 set (its checkpoints would be read as deletes); capacity.py --dry-run plans "
          "the run and writes nothing", r8.rc == 2 and "capacity.py" in r8.err and r9.rc == 0
          and not os.path.exists(r9.work) and "totals:" in r9.out, 2, f"lwrun rc {r8.rc}, dry run rc {r9.rc}")
    ok = all(r[1] for r in C.rows)
    print(f"capacity selftest {'PASSED' if ok else 'FAILED'}: {sum(r[1] for r in C.rows)}/{len(C.rows)} checks, "
          f"{time.time() - t0:.0f} s; work folders kept in {root}")
    return ok


# ============================================================================================ main
def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run a level 7 set (with --dry-run: only plan it)")
    r.add_argument("set", help="a level 7 set (lwlogic.py gen --level 7)")
    r.add_argument("--bin", required=True, help="the tensorfold-native binary (run as BIN serve FOLDER ...)")
    r.add_argument("--model-src", required=True, help="the original snapshot; never taught")
    r.add_argument("--work", required=True, help="work dir: state, answers, verdicts, report, copies (WORK/models)")
    r.add_argument("--port", type=int, required=True)
    r.add_argument("--when-full", choices=("refuse", "replace", "forget"), default="refuse",
                   help="when a lesson comes back PocketFull: refuse it (default), or delete a fact to make room: the "
                        "oldest (forget) or a chosen one (replace)")
    r.add_argument("--evict-max", type=int, default=3, help="facts deleted for one lesson at most (forget, replace)")
    r.add_argument("--gate", choices=("mark", "skip"), default="mark",
                   help="mark: teach every lesson and record the checks; skip: do not teach a lesson the intake calls "
                        "ALREADY_KNOWN or IMPLIED or the model already knows")
    r.add_argument("--known-check", choices=("on", "off"), default="on",
                   help="ask each lesson's CURRENT (and yes/no) question before teaching it")
    r.add_argument("--steps", default="baseline,main,pairs,deletes", help="any of baseline,main,pairs,deletes")
    r.add_argument("--delete-mode", choices=("zero", "endpoint", "reteach", "none"), default="zero")
    r.add_argument("--norms", choices=("on", "off"), default="on", help="read each fact's pocket weight norm")
    r.add_argument("--thinking", default="off,on", help="off,on (default), off, or on")
    r.add_argument("--max-tokens-off", type=int, default=160)
    r.add_argument("--max-tokens-on", type=int, default=1500)
    r.add_argument("--zero-audit", choices=("touched", "full", "off"), default="touched")
    r.add_argument("--models-dir", help="where copies go (default WORK/models; same volume as the source on macOS)")
    r.add_argument("--lwlogic", default=os.path.join(HERE, "lwlogic.py"))
    r.add_argument("--allow-taught-src", action="store_true")
    r.add_argument("--server-arg", action="append", help="one more argument for every server (repeatable)")
    r.add_argument("--keep-env", action="store_true", help="keep TF_SLIDE_* and TF_CUDA_SEGMENTS for the servers")
    r.add_argument("--startup-timeout", type=int, default=900)
    r.add_argument("--warm-timeout", type=int, default=300)
    r.add_argument("--learn-timeout", type=int, default=1800, help="seconds without an SSE line before giving up")
    r.add_argument("--lesson-deadline", type=int, default=5400, help="seconds one learn request may take")
    r.add_argument("--save-timeout", type=int, default=1800, help="seconds to wait for the save after a lesson")
    r.add_argument("--save-grace", type=int, default=120)
    r.add_argument("--save-line-wait", type=float, default=15)
    r.add_argument("--busy-retries", type=int, default=60)
    r.add_argument("--chat-timeout", type=int, default=900)
    r.add_argument("--chat-retries", type=int, default=2)
    r.add_argument("--stop-timeout", type=int, default=60)
    r.add_argument("--server-max-hours", type=float, default=12.0,
                   help="each server carries an alarm: it ends after this long even if the runner was SIGKILLed")
    r.add_argument("--dry-run", action="store_true", help="check and print every step; copy, serve and write nothing")
    p = sub.add_parser("report", help="judge the kept_stable inputs again and rewrite capacity.json and capacity.txt")
    p.add_argument("work")
    p.add_argument("--lwlogic", default=os.path.join(HERE, "lwlogic.py"))
    t = sub.add_parser("selftest", help="against the runner's mock TensorFold server, with negative controls")
    t.add_argument("--dir", help="a new folder for the test (default: a new temp folder); it is never deleted")
    return ap


def main(argv=None):
    a = parser().parse_args(argv)
    if a.cmd == "selftest":
        root = os.path.abspath(a.dir) if a.dir else None
        if root and os.path.exists(root) and os.listdir(root):
            print(f"{root} is not empty: give a new folder", file=sys.stderr)
            return 2
        return 0 if selftest(root) else 1
    if a.cmd == "report":
        work = os.path.abspath(a.work)
        judge = argparse.Namespace(lwlogic=os.path.abspath(a.lwlogic), set_for_judge=os.path.join(work, "set.json"))
        try:
            rep = build_report(work, judge=judge)
        except RunError as e:
            print(f"capacity: {e}", file=sys.stderr)
            return 2
        print(report_text(rep))
        return 0
    try:
        c = Capacity(a)
    except RunError as e:
        print(f"capacity: {e}", file=sys.stderr)
        return 2
    if a.dry_run:
        return dry_run(c)
    for sg in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sg, lwrun.on_signal)
    atexit.register(lwrun.stop_all)
    try:
        return c.run()
    except RunError as e:
        LOG(f"capacity: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
