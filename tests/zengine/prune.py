#!/usr/bin/env python3
"""Night clean-up prototype: free the pocket neurons of taught facts that the stored questions do not need.

  python3 zengine/prune.py run --model TAUGHT_DIR --graph GRAPH.json --set SET.json --bin PATH --work DIR --port N
      [--phase teach] [--questions set|graph|both] [--order oldest|newest] [--facts 3,1,2] [--same text|verdict]
      [--no-noise-check] [--no-final-probe] [--thinking off,on] [--zero-audit touched|full|off] [--dry-run]
  python3 zengine/prune.py selftest [--tmp DIR]

Steps, in this order:
  copy        WORK/models/NAME.pruned, a copy of the taught folder (macOS cp -cRL, an APFS clone; Linux cp -RL), as
              lwrun.py copies. It must share no file with the original: no links, another inode for every file.
  reference   serve the copy without --slide; ask every stored question, thinking off then on, one fresh chat each,
              temperature 0 (lwrun.py's probe); judge the answers with lwlogic.py against --phase (default teach).
  repeat      serve the unchanged copy again and ask again. If the same weights give different answers, a change after
              zeroing cannot be told from noise: the run stops (--no-noise-check skips this).
  per fact    one fact at a time, cumulatively, in the order chosen (oldest first by default): zero its neurons in the
              copy as lwrun.py's zero mode does (zero_facts: up_proj rows own+neuron, weight, scales and biases),
              check that the copy differs from the original only inside zeroed rows, serve, ask every question,
              judge, and compare each answer with the copy's answer before this step.
                every answer the same   the fact stays zeroed: FREED
                any answer differs      its rows are written back and checked against the original: KEPT
              A fact whose rows are already zero (replaced or missed) changes no byte: FREED without a probe.
              A fact with no neurons in the graph, or whose neurons another fact also uses: SKIPPED.
  finish      every freed row zero, every other row as in the original (byte audit), and one more probe of the
              finished copy, which must give the reference answers (--no-final-probe skips that probe).
Report: WORK/prune.json, written as each step starts and ends. Answers, verdicts and server logs: WORK/steps/NN-*/.

What FREED shows: the stored questions got the same answers with that fact's neurons zeroed. It does not show that the
model is unchanged: a question nobody stored may now get another answer. The result also depends on the order: facts
are tried cumulatively, so of two facts that say the same thing only the first one tried is freed.

It never serves with --slide, never writes the original folder or its graph, and holds no TensorFold code.
Standard library only. macOS and Linux.
"""
import argparse
import atexit
import contextlib
import hashlib
import importlib.machinery
import importlib.util
import io
import json
import os
import random
import shlex
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lwlogic  # noqa: E402  (the judge)
import lwrun  # noqa: E402  (the server, copy, zero and probe helpers)
from lwrun import LOG, RunError, ZeroSkip, gb, read_json, utc, write_json  # noqa: E402

IS_MAC, IS_LINUX = lwrun.IS_MAC, lwrun.IS_LINUX
FORMAT = "lw-prune/1"
PARTS = ("weight", "scales", "biases")
SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
CAVEAT = ("FREED means the stored questions got the same answers with that fact's neurons zeroed, after the facts "
          "freed before it. It does not mean the model is unchanged: a question nobody stored may now get another "
          "answer. The result also depends on the order the facts were tried in.")


# ============================================================================================ inputs
def neuron_list(ns):
    out = []
    for n in ns or []:
        key = (int(n["layer"]), int(n["neuron"]))
        if key not in out:
            out.append(key)
    return out


def load_graph(path):
    """Facts and their pocket neurons, from a slide graph (GET /v1/slide/graph, lwrun's graph-final.json) or from
    lwrun's teach state.json. The state also remembers the neurons of facts replaced later; a final graph lists none
    for them (1.0.8 zeroes a replaced fact's rows)."""
    g = read_json(path)
    if not isinstance(g, dict):
        raise RunError(f"{path}: not a JSON object")
    facts = []
    if isinstance(g.get("facts"), list):
        kind = "slide graph"
        for f in g["facts"]:
            facts.append({"id": int(f["id"]), "text": f.get("text") or "", "state": f.get("state"),
                          "recalled": f.get("recalled"), "neurons": neuron_list(f.get("neurons")),
                          "questions": list(f.get("questions") or [])})
    elif isinstance(g.get("fact_neurons"), dict):
        kind = "lwrun teach state"
        info = {str(f.get("id")): f for rec in g.get("lessons") or [] for f in rec.get("facts") or []}
        states = g.get("fact_states") or {}
        for i in sorted({int(x) for x in states} | {int(x) for x in g["fact_neurons"]}):
            f = info.get(str(i), {})
            facts.append({"id": i, "text": f.get("text") or "", "state": states.get(str(i)),
                          "recalled": f.get("recalled"), "neurons": neuron_list(g["fact_neurons"].get(str(i))),
                          "questions": list(f.get("questions") or []),
                          "lesson": (g.get("fact_lesson") or {}).get(str(i))})
    else:
        raise RunError(f"{path}: neither a slide graph (facts) nor an lwrun teach state.json (fact_neurons)")
    return kind, facts


USAGE_REQUEST = ("--order usage needs to know how strongly each fact's neurons fire on the stored questions and on "
                 "ordinary text. TensorFold 1.0.8 has no endpoint or log that says which facts' neurons fired for a "
                 "request, so this is a request to TensorFold (docs/NIGHT-PRUNE.md), not something this tool fakes. "
                 "Use --order norm (weight-norm outliers first), oldest or newest.")
RELEARN_NOTE = ("1.0.8 zeroes a replaced fact's neurons and keeps their places taken while the server runs (READ: "
                "pocket.zig retire). On load it counts a layer's neurons in use up to the last one whose input row is "
                "nonzero (READ: weights.zig pocketOf), so in the relearn copy, which is loaded fresh, a freed neuron above "
                "every live one in its layer is free again and a freed neuron below a live one stays taken. TensorFold "
                "picks the related fact's places itself; on_freed records which of them a freed fact held.")


def landed_on(places, freed):
    """The related fact's (layer, neuron) places that a freed fact held, sorted: [[layer, neuron], ...]."""
    held = {(int(l), int(n)) for l, n in freed}
    return sorted([int(l), int(n)] for l, n in places or [] if (int(l), int(n)) in held)


def relearn_verdict(p, r=None):
    """(ok, why) of one relearning probe, from the freed copy's counts `p` and the never-taught reference's `r` (or
    None). Stays "don't know": where the rule book says don't know, the answers say it. Without a reference every
    one must. With one, no more of them may miss than in the reference, which never learned the fact: an untaught
    model guesses too (our 1.0.8 runs had wrong yeses about facts nobody taught), and only what the freed copy adds
    is relearning. And no thinking mode may have more wrong answers than the reference. None: inconclusive."""
    if not p["related_learned"] or (r and not r["related_learned"]):
        return None, "the related fact was not learned: nothing was relearned"
    miss = lambda x, m: (x["unknown_kept"].get(m) or [0, 0])[1] - (x["unknown_kept"].get(m) or [0, 0])[0]
    modes = sorted(set(p["unknown_kept"]) | set(p["wrong"]))
    lost = [m for m in modes if miss(p, m) > (miss(r, m) if r else 0)]
    worse = [m for m in modes if r and p["wrong"].get(m, 0) > r["wrong"].get(m, 0)]
    if not lost and not worse:
        return True, "stays don't know" + (" as often as the reference, no more often wrong" if r else "")
    return False, "; ".join(([f"don't know not kept {p['unknown_kept']}"
                              + (f", the reference {r['unknown_kept']}" if r else "") + f" (thinking {lost})"]
                             if lost else [])
                            + ([f"more often wrong than the reference with thinking {worse}"] if worse else []))


def rank_by_norm(folder, facts):
    """Candidates for freeing by their pocket weight norm, read from the taught folder (read only; lwrun.pocket_norms):
    the large outliers first (modified z score, lwrun.norm_outliers), then the rest from the largest norm down, then
    facts already switched off (their input rows zero: replaced or zeroed), then facts with no neurons. Gupta et al.
    (2024) report single edits with outsized weight changes that disable a model; those are worth trying first."""
    fn = {str(f["id"]): [{"layer": l, "neuron": n} for l, n in f["neurons"]] for f in facts if f["neurons"]}
    norms = lwrun.pocket_norms(folder, fn)
    on = {k: v["total"] for k, v in norms.items() if v["up"] > 0}
    out = lwrun.norm_outliers(on)
    zs = out["facts"]
    order = sorted(on, key=lambda f: (not zs[f]["outlier"], -zs[f]["z"], -on[f], int(f)))
    order += sorted((f for f in norms if f not in on), key=int)
    order = [int(f) for f in order] + sorted(f["id"] for f in facts if str(f["id"]) not in norms)
    return {"order": order, "median": out["median"], "outliers": [int(x) for x in out["outliers"]],
            "facts": {f: dict(up=v["up"], down=v["down"], total=v["total"], neurons=v["neurons"], off=f not in on,
                              z=(zs.get(f) or {}).get("z"), outlier=(zs.get(f) or {}).get("outlier", False))
                      for f, v in norms.items()}}


def related_lesson(S, l):
    """The related fact the relearning probe teaches after freeing lesson `l`: the same attribute and value, said of
    the entity the rule book calls similar (my sister for me, my dog for my cat), or of another entity of the same
    kind; in the attribute's first wording. Its id and tick come after every lesson of the set."""
    e, a = l["e"], l["a"]
    kinds = lwlogic.SCHEMA[a]["applies"]
    cands = [lwlogic.SIMILAR.get(e)] + [x for x, ent in lwlogic.ENTITIES.items()
                                       if ent["kind"] == lwlogic.ENTITIES[e]["kind"]]
    e2 = next(x for x in cands if x and x != e and lwlogic.ENTITIES[x]["kind"] in kinds)
    v = l.get("v") if l.get("v") is not None else l.get("v1")
    nid = max(x["id"] for x in S["lessons"]) + 1
    return {"id": nid, "name": f"R{nid}", "op": "ASSERT", "e": e2, "a": a, "v": v,
            "told": max(x["told"] for x in S["lessons"]) + 100,
            "text": lwlogic.render(lwlogic.T[a]["ASSERT"][0], e2, v=v), "para": 0}


def relearn_questions(S, l, rel):
    """The freed fact asked in several wordings: every CURRENT wording of its pair, yes/no about its value in every
    wording, an either-or with another value (one value at a time only); and the related fact's CURRENT, to see it
    was learned. qids R01, R02, ..."""
    e, a = l["e"], l["a"]
    v = l.get("v") if l.get("v") is not None else l.get("v1")
    dom = lwlogic.SCHEMA[a]["domain"]
    T = lwlogic.T[a]
    qs = [("CURRENT", e, t, {}) for t in T["Q_CURRENT"]]
    qs += [("YES_NO", e, t, {"v": v}) for t in T["Q_YES_NO"]]
    if not lwlogic.SCHEMA[a]["multi"] and T["Q_EITHER_OR"]:
        w = next(x for x in dom if x != v)
        qs.append(("EITHER_OR", e, T["Q_EITHER_OR"][0], {"x": v, "y": w}))
    qs.append(("CURRENT", rel["e"], T["Q_CURRENT"][0], {}))
    out = []
    for i, (t, ent, tmpl, kw) in enumerate(qs, 1):
        q = {"qid": f"R{i:02d}", "type": t, "e": ent, "a": a, "text": lwlogic.render(tmpl, ent, **kw), "para": None,
             "role": "probe" if ent == e else "related"}
        q.update(kw)
        out.append(q)
    return out


def parse_thinking(s):
    out = []
    for x in s.split(","):
        x = x.strip().lower()
        if x not in ("off", "on"):
            raise RunError(f"--thinking takes off and/or on, not {x!r}")
        out.append(x == "on")
    return out


# ============================================================================================ pocket rows
class Pocket:
    """Where zero_facts (lwrun.py) writes for one (layer, neuron): the same indexing, read only, so a dry run can plan
    without writing. Every real step checks that zero_facts wrote exactly these rows."""

    def __init__(self, folder):
        self.folder = folder
        cfg = read_json(os.path.join(folder, "config.json"), {}) or {}
        self.P = cfg.get("tensorfold_learned_neurons")
        self.wide = cfg.get("moe_shared_expert_intermediate_size")
        if not isinstance(self.P, int) or self.P <= 0:
            raise ZeroSkip("config.json has no tensorfold_learned_neurons: the folder holds no pocket")
        if self.P != 64:
            raise ZeroSkip(f"tensorfold_learned_neurons is {self.P}, not the 64 of 1.0.8's pocket.zig")
        self.wm = (read_json(os.path.join(folder, "model.safetensors.index.json"), {}) or {}).get("weight_map")
        if not self.wm:
            raise ZeroSkip("no model.safetensors.index.json weight_map")
        self.headers = {}

    def rows(self, layer, neuron):
        """[(part, shard file, offset, width)] of one neuron, for weight, scales and biases."""
        out = []
        for part in PARTS:
            name = f"backbone.layers.{layer}.mixer.shared_experts.up_proj.{part}"
            if name not in self.wm:
                raise ZeroSkip(f"{name} is not in the index")
            shard = self.wm[name]
            if shard not in self.headers:
                self.headers[shard] = lwrun.st_header(os.path.join(self.folder, shard))
            h, base = self.headers[shard]
            m = h[name]
            dtype, shape, (s, e) = m["dtype"], m["shape"], m["data_offsets"]
            if dtype != lwrun.PART_DTYPE[part] or len(shape) != 2 or shape[0] != self.wide:
                raise ZeroSkip(f"{name} is {dtype} {shape}, expected {lwrun.PART_DTYPE[part]} [{self.wide}, ...]")
            if not 0 <= neuron < self.P:
                raise ZeroSkip(f"neuron {neuron} outside the pocket of {self.P}")
            width = shape[1] * lwrun.DTYPE_SIZE[dtype]
            if e - s != shape[0] * width:
                raise ZeroSkip(f"{name}: data_offsets do not match its shape")
            out.append((part, shard, base + s + (shape[0] - self.P + neuron) * width, width))
        return out


def read_at(folder, shard, off, width):
    with open(os.path.join(folder, shard), "rb") as f:
        f.seek(off)
        return f.read(width)


@contextlib.contextmanager
def signals_held():
    """Signals that arrive inside this block wait until it ends, then go to the usual handler (lwrun.on_signal stops
    the run). A write-back is never left half done, except by SIGKILL."""
    held, old = [], {}
    try:
        for sg in SIGNALS:
            old[sg] = signal.signal(sg, lambda n, _f: held.append(n))
    except ValueError:  # not the main thread: no handler can run here anyway
        pass
    done = False
    try:
        yield
        done = True
    finally:
        for sg, h in old.items():
            signal.signal(sg, signal.SIG_DFL if h is None else h)
    if held and done:
        signal.raise_signal(held[0])


# ============================================================================================ answers
def scores(answers):
    out = {}
    for (_qid, th), x in answers.items():
        c = out.setdefault("on" if th else "off", {"pass": 0, "fail": 0, "unclear": 0, "unjudged": 0})
        v = x.get("verdict")
        c["unjudged" if v is None else v if v in ("pass", "fail") else "unclear"] += 1
    return out


def same_stats(before, after):
    out = {}
    for k, b in before.items():
        x = after.get(k) or {}
        c = out.setdefault("on" if k[1] else "off", {"n": 0, "same_text": 0, "judged": 0, "same_verdict": 0})
        c["n"] += 1
        c["same_text"] += b["content"] == x.get("content")
        if b.get("verdict") is not None:
            c["judged"] += 1
            c["same_verdict"] += (b["verdict"], b["got"]) == (x.get("verdict"), x.get("got"))
    return out


def excerpt(t, n=240):
    return None if t is None else " ".join(t.split())[:n]


# ============================================================================================ the run
class Prune:
    def __init__(self, a):
        self.args = a
        self.model = os.path.realpath(a.model)
        self.work = os.path.abspath(a.work)
        self.models = os.path.abspath(a.models_dir or os.path.join(self.work, "models"))
        self.copy = os.path.join(self.models, os.path.basename(self.model) + ".pruned")
        self.pidfile = os.path.join(self.work, "server.pid")
        self.port = a.port
        self.bin = os.path.abspath(a.bin)
        self.phase = a.phase
        self.thinking = parse_thinking(a.thinking)
        self.graph_kind, self.facts = load_graph(a.graph)
        self.by_id = {f["id"]: f for f in self.facts}
        self.judge_set = None
        qs = []
        if a.questions in ("set", "both"):
            if not a.set:
                raise RunError("--set is needed for --questions set or both (or use --questions graph)")
            S = read_json(a.set)
            if not isinstance(S, dict) or S.get("format") != "lw-test-set/1":
                raise RunError(f"{a.set}: not an lwlogic test set (format lw-test-set/1)")
            if a.phase not in {p["name"] for p in S["phases"]}:
                raise RunError(f"--phase {a.phase!r} is not a phase of {S['id']} ({[p['name'] for p in S['phases']]})")
            self.judge_set = S
            qs += [dict(q) for q in S["questions"]]
        if a.questions in ("graph", "both"):  # TensorFold's own checks per fact: no expected answers, compared by text
            seen, ids, n = {q["text"] for q in qs}, {q["qid"] for q in qs}, 0
            for f in self.facts:
                for t in f["questions"]:
                    if t not in seen:
                        seen.add(t)
                        n += 1
                        while f"G{n:02d}" in ids:  # G01, G02, ...: never a qid the set already uses
                            n += 1
                        qs.append({"qid": f"G{n:02d}", "text": t, "type": "GRAPH", "role": "graph", "fact": f["id"]})
        if not qs:
            raise RunError("no questions to ask")
        self.S = {"id": (self.judge_set or {}).get("id", "graph"), "questions": qs}  # what lwrun.probe asks
        if a.order == "usage":
            raise RunError(USAGE_REQUEST)
        self.src = self.model  # lwrun's guard: --slide never serves the original folder
        self.copies_path = os.path.join(self.work, "copies.json")
        try:  # read only; also for the report when another order is used
            self.ranking = rank_by_norm(self.model, self.facts)
        except (ZeroSkip, OSError, KeyError, ValueError) as e:
            self.ranking = {"skipped": f"{type(e).__name__}: {e}"}
        if a.relearn and not self.judge_set:
            raise RunError("--relearn needs --set: its questions and right answers come from the set's lessons")
        if a.relearn and not (a.slide_graph or self.graph_kind == "slide graph"):
            raise RunError("--relearn serves the freed copy with --slide, which needs the slide graph: give "
                           "--slide-graph (the runner's phases/teach/graph.json) when --graph is a state.json")
        self.order = self.pick_order()
        self.pocket = None
        self.rep = None
        self.fp_before = None
        self.freed = {}       # fact id -> its rows, zero in the copy now
        self.dead = set()     # facts whose rows were all zero already: no kept neuron of theirs to protect
        self.reference = None  # the unchanged copy's answers: {(qid, thinking): {...}}
        self.current = None    # the copy's answers as it is now
        self.n = 0
        self.dry_plan = None

    # lwrun.Server and lwrun.probe use these, and run.args, run.port, run.bin, run.S, run.thinking, run.pairs()
    check_pidfile = lwrun.Run.check_pidfile
    write_pidfile = lwrun.Run.write_pidfile
    guard_teachable = lwrun.Run.guard_teachable

    def pairs(self):
        return {(q["qid"], th) for q in self.S["questions"] for th in self.thinking}

    def pick_order(self):
        a = self.args
        if a.facts:
            ids = [int(x) for x in a.facts.split(",") if x.strip()]
            missing = [i for i in ids if i not in self.by_id]
            if missing:
                raise RunError(f"--facts {missing}: not in the graph (facts {sorted(self.by_id)})")
            if len(set(ids)) != len(ids):
                raise RunError("--facts names a fact twice")
            return ids
        if a.order == "norm":
            if "order" not in self.ranking:
                raise RunError(f"--order norm: the norms cannot be read from {self.model}: {self.ranking['skipped']}")
            return list(self.ranking["order"])
        ids = sorted(self.by_id)  # fact ids grow in teaching order
        return ids if a.order == "oldest" else ids[::-1]

    def rows_of(self, f, pocket=None):
        """[(part, shard, offset, width, layer, neuron)] of every neuron of fact f."""
        pocket = pocket or self.pocket
        return [(part, shard, off, width, layer, neuron) for layer, neuron in f["neurons"]
                for part, shard, off, width in pocket.rows(layer, neuron)]

    def owners(self):
        own = {}
        for f in self.facts:
            for key in f["neurons"]:
                own.setdefault(key, []).append(f["id"])
        return own

    def save(self):
        if self.rep is not None:
            write_json(os.path.join(self.work, "prune.json"), self.rep)

    # ---------------------------------------------------------------------------------- guards
    def nesting_problems(self):
        out = []
        for name, path in (("--work", self.work), ("--models-dir", self.models), ("the copy", self.copy)):
            rp = os.path.realpath(path)
            if rp == self.model or (rp + os.sep).startswith(self.model + os.sep):
                out.append(f"{name} {path} is the original folder or lies inside it")
        return out

    def preflight(self):
        """Before anything is copied: the inputs sane, nothing reused, no earlier server still up."""
        a = self.args
        problems = self.nesting_problems()
        if not (os.path.isfile(self.bin) and os.access(self.bin, os.X_OK)):
            problems.append(f"--bin {self.bin} is not an executable file")
        cfg = read_json(os.path.join(self.model, "config.json"))
        if not isinstance(cfg, dict):
            problems.append(f"--model {self.model} has no config.json")
        elif "tensorfold_learned_neurons" not in cfg:
            problems.append(f"--model {self.model} holds no learned neurons (config.json tensorfold_learned_neurons): "
                            "not a taught folder")
        if not any(f["neurons"] for f in self.facts):
            problems.append(f"no fact in {a.graph} lists neurons: nothing to try")
        if a.port in (8091,):
            problems.append(f"--port {a.port} is reserved (another server lives there)")
        elif lwrun.port_busy(a.port):
            problems.append(f"--port {a.port} is in use (or cannot be bound)")
        if os.path.exists(os.path.join(self.work, "prune.json")):
            problems.append(f"{self.work} holds an earlier prune run (prune.json); use a new --work")
        if os.path.exists(self.copy):
            problems.append(f"{self.copy} exists; use a new --work or --models-dir")
        try:
            self.check_pidfile()
        except RunError as e:
            problems.append(str(e))
        if problems:
            raise RunError("refusing to start:\n  " + "\n  ".join(problems))

    def copy_model(self):
        """A fresh copy of the taught folder, made the way lwrun.py makes its copies."""
        need = lwrun.tree_bytes(self.model)
        free = lwrun.free_bytes(self.models)
        want = int(need * 1.05) + 2 * lwrun.GIB if IS_LINUX else 2 * lwrun.GIB  # zeroing a clone rewrites a few blocks
        if free < want:
            raise RunError(f"not enough disk for {self.copy}: {gb(free)} free, {gb(want)} wanted")
        os.makedirs(self.models, exist_ok=True)
        cmd = ["cp", "-cRL", self.model, self.copy] if IS_MAC else ["cp", "-RL", self.model, self.copy]
        LOG(f"copy: {' '.join(shlex.quote(x) for x in cmd)}")
        t0 = time.time()
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            raise RunError(f"copy failed ({p.returncode}): {p.stderr.strip()[:400]}")
        LOG(f"copied in {time.time() - t0:.1f} s ({gb(lwrun.tree_bytes(self.copy))} logical)")

    def guard_copy(self):
        """Zeroing writes the copy in place, so the copy must share no file with the original: no symlink, no hard
        link, another inode for every file; and it must hold the same files at the same sizes."""
        def files(top):
            out = {}
            for root, _dirs, names in os.walk(top):
                for x in names:
                    out[os.path.relpath(os.path.join(root, x), top)] = os.path.join(root, x)
            return out
        orig, mine = files(self.model), files(self.copy)
        bad = []
        if set(orig) != set(mine):
            bad.append(f"file lists differ: {sorted(set(orig) ^ set(mine))[:10]}")
        for rel, p in sorted(mine.items()):
            if os.path.islink(p):
                bad.append(f"{rel} is a symlink")
                continue
            st = os.stat(p)
            if st.st_nlink > 1:
                bad.append(f"{rel} has {st.st_nlink} hard links")
            if rel in orig:
                so = os.stat(orig[rel])
                if (so.st_dev, so.st_ino) == (st.st_dev, st.st_ino):
                    bad.append(f"{rel} is the original's own file")
                elif so.st_size != st.st_size:
                    bad.append(f"{rel}: size differs from the original")
        if bad:
            raise RunError("refusing to zero: the copy shares files with the original or does not match it:\n  "
                           + "\n  ".join(bad[:10]))

    def check_original(self):
        if self.fp_before is not None and lwrun.fingerprint(self.model) != self.fp_before:
            raise RunError(f"the original folder {self.model} changed during the run (another program?): stopping")

    # ---------------------------------------------------------------------------------- probing
    def probe_state(self, tag, label):
        """Serve the copy without --slide, ask every question (lwrun.probe), judge: (answers, probe record)."""
        d = os.path.join(self.work, "steps", tag)
        os.makedirs(d, exist_ok=True)
        ans, err = os.path.join(d, "answers.jsonl"), os.path.join(d, "errors.jsonl")
        srv = lwrun.Server(self, self.copy, False, os.path.join(d, "server.log"), label=label)
        srv.start()
        try:
            res = lwrun.probe(self, srv, label, self.phase, "prune", ans, err)
        finally:
            srv.stop("probing done")
        res["server"] = srv.timings
        return self.judged(ans, os.path.join(d, "verdicts.json")), res

    def judged(self, ans, out):
        """The answers on file, with the judge's verdict and reading for the set's questions (lwlogic.judge)."""
        rows = []
        with open(ans, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        answers = {(r["qid"], bool(r["thinking"])): {"qid": r["qid"], "thinking": bool(r["thinking"]),
                                                     "question": r["question"], "content": r["content"] or "",
                                                     "verdict": None, "got": None, "expected": None} for r in rows}
        if self.judge_set:
            jrows = [{"phase": self.phase, "qid": r["qid"], "thinking": bool(r["thinking"]),
                      "content": r["content"] or "", "question": r["question"]} for r in rows]
            results, _cons, _skipped = lwlogic.judge(self.judge_set, jrows)
            for x in results:
                k = (x["qid"], x["thinking"])
                if k in answers:
                    answers[k].update(verdict=x["verdict"], got=x["got"], expected=x["expected"])
            write_json(out, {"set": self.judge_set["id"], "phase": self.phase, "results": results})
        return answers

    def compare(self, before, after):
        """The answers that changed. --same text: word for word. --same verdict: the judge's verdict and reading of
        the answer; answers the judge does not score (the graph's own questions) are compared by text."""
        out = []
        for k in sorted(before):
            b, x = before[k], after.get(k)
            if x is None:
                out.append({"qid": k[0], "thinking": k[1], "question": b["question"], "how": "no answer",
                            "before": excerpt(b["content"]), "after": None})
                continue
            same_text = b["content"] == x["content"]
            same_verdict = (b["verdict"], b["got"]) == (x["verdict"], x["got"])
            if self.args.same == "verdict" and b["verdict"] is not None and x["verdict"] is not None:
                changed = not same_verdict
            else:
                changed = not same_text
            if changed:
                out.append({"qid": k[0], "thinking": k[1], "question": b["question"],
                            "how": "text" if same_verdict else "verdict",
                            "before": excerpt(b["content"]), "after": excerpt(x["content"]),
                            "verdict_before": b["verdict"], "verdict_after": x["verdict"],
                            "got_before": b["got"], "got_after": x["got"]})
        return out

    # ---------------------------------------------------------------------------------- rows
    def audit(self, extra=()):
        """The copy against the original: every freed row zero, and (lwrun.audit_copy) every differing byte inside
        a zeroed row."""
        rows = [r for rs in self.freed.values() for r in rs] + list(extra)
        nonzero = [(r[1], r[2]) for r in rows if any(read_at(self.copy, r[1], r[2], r[3]))]
        out = {"rows": len(rows), "rows_not_zero": len(nonzero), "ok": not nonzero}
        if self.args.zero_audit != "off":
            plan = [{"shard": r[1], "offset": r[2], "bytes": r[3], "nonzero_before": True} for r in rows]
            au = lwrun.audit_copy(self.copy, self.model, plan, full=(self.args.zero_audit == "full"))
            out.update({k: au[k] for k in ("files", "bytes_differing", "outside_rows", "problems")},
                       compared=len(au["compared"]), size_only=len(au["size_only"]))
            out["ok"] = out["ok"] and au["ok"]
        if not out["ok"]:
            raise RunError(f"byte audit failed: {out}" + (f"; freed rows not zero: {nonzero[:5]}" if nonzero else ""))
        return out

    def put_back(self, rows, backup):
        """Write a fact's rows back as they were before zeroing, then check them against the original. A signal
        (a second Ctrl-C) waits until this is done."""
        with self.hold_signals():
            self.restore_rows(rows, backup)
            self.verify_restored(rows)

    def hold_signals(self):
        return signals_held()

    def restore_rows(self, rows, backup):
        todo = [(shard, off) for _part, shard, off, width, *_ in rows
                if read_at(self.copy, shard, off, width) != backup[(shard, off)]]
        lwrun.check_inplace_writable(os.path.join(self.copy, s) for s, _o in todo)  # no symlink or hard link, first
        for shard, off in todo:
            with lwrun.open_for_inplace_write(os.path.join(self.copy, shard)) as f:
                f.seek(off)
                f.write(backup[(shard, off)])
                f.flush()
                os.fsync(f.fileno())

    def verify_restored(self, rows):
        bad = [(shard, off) for _part, shard, off, width, *_ in rows
               if read_at(self.copy, shard, off, width) != read_at(self.model, shard, off, width)]
        if bad:
            raise RunError(f"restore failed: {len(bad)} rows differ from the original after writing them back "
                           f"{bad[:5]}; the copy is not usable")

    def accept(self, f, rows, backup, answers):
        """FREED: the rows stay zero, and the next fact is compared with these answers."""
        self.freed[f["id"]] = rows
        self.current = answers

    # ---------------------------------------------------------------------------------- one fact
    def try_fact(self, fid):
        f = self.by_id[fid]
        self.n += 1
        rec = {"step": self.n, "fact": fid, "text": f["text"], "state": f["state"],
               "neurons": [list(x) for x in f["neurons"]], "started": utc(), "result": "running"}
        self.rep["steps"].append(rec)
        self.save()  # on disk before anything is zeroed: after a SIGKILL the report names the step under way
        LOG(f"=== step {self.n}: fact {fid} ({f['state']}) {f['text']!r}, {len(f['neurons'])} neurons")
        t0 = time.time()
        try:
            self.decide(f, rec, f"{self.n:02d}-fact-{fid}")
        finally:
            rec["seconds"] = round(time.time() - t0, 1)
            if rec["result"] == "running":
                rec["result"], rec["why"] = "stopped", "the run stopped during this step (see the report's error)"
            self.save()
        LOG(f"fact {fid}: {rec.get('result')}" + (f": {rec['why']}" if rec.get("why") else ""))

    def decide(self, f, rec, tag):
        fid = f["id"]

        def mark(result, why):
            rec["result"], rec["why"] = result, why

        if not f["neurons"]:
            return mark("SKIPPED", "no neurons in the graph (a missed fact, or one the graph no longer lists)")
        shared = sorted({o for key in f["neurons"] for o in self.owners()[key] if o != fid and o in self.freed})
        if shared:
            return mark("SKIPPED", f"shares neurons with freed facts {shared}")
        self.check_original()
        rows = self.rows_of(f)
        backup = {(r[1], r[2]): read_at(self.copy, r[1], r[2], r[3]) for r in rows}
        if any(read_at(self.model, r[1], r[2], r[3]) != backup[(r[1], r[2])] for r in rows):
            raise RunError(f"fact {fid}: the copy's rows differ from the original's before zeroing")
        rec["rows"], rec["rows_nonzero"] = len(rows), sum(1 for v in backup.values() if any(v))
        if not rec["rows_nonzero"]:
            self.dead.add(fid)
            if f["state"] == "learned":
                return mark("SKIPPED", "learned, but every row is already zero: wrong indexing, or a stale graph")
            self.freed[fid] = rows
            rec["probed"] = False
            return mark("FREED", f"every row already zero (state {f['state']}): no byte changed, so no probe")
        targets = [(fid, f["state"], layer, neuron) for layer, neuron in f["neurons"]]
        kept = [(g["id"], g["state"], layer, neuron) for g in self.facts
                if g["id"] != fid and g["id"] not in self.freed and g["id"] not in self.dead
                for layer, neuron in g["neurons"]]
        try:
            z = lwrun.zero_facts(self.copy, targets, kept, ref=None, audit="off")
        except RunError as e:  # neurons shared with a kept fact, or rows that look wrongly indexed: nothing to free
            self.put_back(rows, backup)
            if isinstance(e, lwrun.SharedFileError):  # a shard that is a symlink or a hard link: stop the run
                raise
            return mark("SKIPPED", f"zero_facts refused: {e}")
        except BaseException:  # ZeroSkip (a layout zero_facts does not know: stop), or a signal
            self.put_back(rows, backup)
            raise
        settled = False
        try:
            wrote = sorted((r["shard"], r["offset"], r["bytes"]) for r in z["rows"])
            if wrote != sorted((r[1], r[2], r[3]) for r in rows):
                raise RunError(f"fact {fid}: zero_facts planned other rows than this tool's index; stopping")
            rec["zero"] = {k: z[k] for k in ("rows_zeroed", "rows_already_zero", "bytes_zeroed", "kept_learned",
                                             "kept_nonzero_before", "kept_nonzero_after", "shards_touched")}
            rec["zero_rows"] = [[r["shard"], r["offset"], r["bytes"]] for r in z["rows"]]
            rec["audit"] = self.audit(extra=rows)
            answers, res = self.probe_state(tag, f"fact {fid}")
            changes = self.compare(self.current, answers)
            rec.update(probed=True, answers=res["path"], complete=res["complete"], probe_seconds=res["seconds"],
                       same=same_stats(self.current, answers), scores=scores(answers), n_changed=len(changes),
                       changed=changes[:40])
            with self.hold_signals():  # FREED, or written back and KEPT: a signal now waits until the fact is settled
                if changes:
                    self.put_back(rows, backup)
                    how = sorted({c["how"] for c in changes})
                    mark("KEPT", f"{len(changes)} of {len(self.current)} answers changed ({', '.join(how)}); rows "
                                 "written back and checked against the original")
                else:
                    self.accept(f, rows, backup, answers)
                    mark("FREED", f"all {len(answers)} answers the same ({self.args.same})")
                settled = True
        except BaseException:  # a failed audit or probe, or a signal: the copy goes back to its last accepted state
            if not settled:
                self.put_back(rows, backup)
            raise

    # ---------------------------------------------------------------------------------- whole run
    def run(self):
        a = self.args
        os.makedirs(self.work, exist_ok=True)
        LOG.path = os.path.join(self.work, "prune.log")
        try:
            self.preflight()
            S = self.judge_set
            self.rep = {
                "format": FORMAT, "caveat": CAVEAT, "status": "running", "started": utc(),
                "model": self.model, "copy": self.copy,
                "graph": {"path": os.path.abspath(a.graph), "kind": self.graph_kind, "facts": len(self.facts)},
                "set": {"id": S["id"], "path": os.path.abspath(a.set), "sha256": lwrun.sha256(a.set)} if S else None,
                "phase": self.phase if S else None,
                "questions": {"source": a.questions, "n": len(self.S["questions"])}, "thinking": a.thinking,
                "answers_per_probe": len(self.pairs()), "same": a.same, "order_rule": "given" if a.facts else a.order,
                "order": self.order, "binary": {"path": self.bin, "sha256": lwrun.sha256(self.bin)},
                "zero_audit": a.zero_audit, "ranking": self.ranking, "steps": []}
            self.save()
            return self.go()
        finally:
            LOG.path = None

    def go(self):
        status = "failed"
        try:
            self.body()
            status = self.rep.get("status") if self.rep.get("status") != "running" else "done"
        except (RunError, ZeroSkip, OSError) as e:  # ZeroSkip: a layout zero_facts does not know; OSError: the disk
            msg = str(e) if isinstance(e, RunError) else f"{type(e).__name__}: {e}"
            LOG(f"prune FAILED: {msg}")
            self.rep["error"] = msg
        except lwrun.Interrupted as e:
            LOG(f"prune interrupted by {e}; stopping")
            status = "interrupted"
            self.rep["error"] = f"interrupted by {e}"
        finally:
            lwrun.stop_all("prune exiting")
            self.rep["status"] = status
            self.wrap_up()
        return 0 if status == "done" else 130 if status == "interrupted" else 1  # relearn-failed: 1

    def body(self):
        a = self.args
        self.fp_before = lwrun.fingerprint(self.model)
        self.copy_model()
        self.guard_copy()
        self.pocket = Pocket(self.copy)
        self.rep["audit_start"] = self.audit()
        LOG(f"prune {len(self.order)} facts in the order {self.order} (cumulative), {len(self.pairs())} answers per "
            f"probe, same = {a.same}")
        self.reference, res = self.probe_state("00-reference", "reference")
        self.current = self.reference
        self.rep["reference"] = {"answers": res["path"], "n": res["n"], "expected": res["expected"],
                                 "complete": res["complete"], "seconds": res["seconds"],
                                 "scores": scores(self.reference)}
        self.save()
        if not res["complete"]:
            raise RunError(f"the reference probe is incomplete ({res['n']}/{res['expected']} answers): stopping")
        if not a.no_noise_check:
            again, res2 = self.probe_state("00-repeat", "reference again")
            diff = self.compare(self.reference, again)
            self.rep["noise_check"] = {"answers": res2["path"], "same": same_stats(self.reference, again),
                                       "n_changed": len(diff), "changed": diff[:40], "ok": not diff}
            self.save()
            if diff:
                LOG(f"noise check: {len(diff)} answers differ between two probes of the same weights; a change after "
                    "zeroing could not be told from noise. Nothing was zeroed.")
                self.rep["status"] = "noisy"
                return
            LOG(f"noise check: the unchanged copy gave the same {len(again)} answers twice")
        for fid in self.order:
            self.try_fact(fid)
        self.finish()
        if a.relearn and self.rep.get("status") == "running":
            self.relearn()

    def finish(self):
        """The finished copy: freed rows zero, every other fact's rows as in the original, and one more probe."""
        a = self.args
        self.check_original()
        self.rep["final_audit"] = self.audit()
        freed_keys = {key for fid in self.freed for key in self.by_id[fid]["neurons"]}
        bad = [(g["id"], r[1], r[2]) for g in self.facts
               if g["neurons"] and g["id"] not in self.freed and not freed_keys & set(g["neurons"])
               for r in self.rows_of(g) if read_at(self.copy, r[1], r[2], r[3]) != read_at(self.model, r[1], r[2], r[3])]
        self.rep["final_audit"]["other_rows_as_original"] = not bad
        if bad:
            raise RunError(f"rows of facts that were not freed differ from the original: {bad[:5]}")
        if a.no_final_probe:
            return
        answers, res = self.probe_state(f"{self.n + 1:02d}-final", "final")
        diff = self.compare(self.reference, answers)
        self.rep["final_probe"] = {"answers": res["path"], "same": same_stats(self.reference, answers),
                                   "n_changed": len(diff), "changed": diff[:40], "ok": not diff}
        if diff:
            LOG(f"final probe: {len(diff)} answers differ from the reference: do not use this copy")
            self.rep["status"] = "final-differs"
        else:
            LOG(f"final probe: the finished copy gave the reference answers ({len(answers)})")

    # ---------------------------------------------------------------------------------- relearning probe
    def lesson_of(self, f):
        """The set lesson a graph fact holds: by lwrun's teach state (lesson id), else by its text."""
        S = self.judge_set
        if f.get("lesson") is not None:
            return next((l for l in S["lessons"] if l["id"] == f["lesson"]), None)
        hits = [l for l in S["lessons"] if l["text"] == f["text"]]
        return hits[0] if len(hits) == 1 else None

    def relearn(self):
        """After freeing: teach a related fact to a copy of the freed copy, then ask the freed fact in several
        wordings. Where the rule book (the phase's lessons without the freed ones, plus the related fact) says don't
        know, the answers must say don't know, no less often than in --reference-model, a taught folder that never
        learned the freed lessons (lwrun's reteach copy of a phase without them), and no more of them may be wrong
        than there (relearn_verdict)."""
        a, S = self.args, self.judge_set
        out = self.rep["relearn"] = {"note": RELEARN_NOTE, "reference": a.reference_model, "facts": [], "ok": None}
        cands = [(fid, self.lesson_of(self.by_id[fid])) for fid in self.order if fid in self.freed]
        out["no_lesson"] = [fid for fid, l in cands if l is None]
        gone = {l["id"] for _f, l in cands if l}
        store = [l for l in S["lessons"] if l["id"] in lwlogic.phase_store(S, self.phase) and l["id"] not in gone]
        graph = a.slide_graph or a.graph
        for fid, l in [(f, l) for f, l in cands if l][:a.relearn_max]:
            rel = related_lesson(S, l)
            qs = relearn_questions(S, l, rel)
            rec = {"fact": fid, "lesson": l["id"], "text": l["text"], "related": rel["text"], "questions": len(qs)}
            out["facts"].append(rec)
            self.save()
            LOG(f"relearn fact {fid} ({l['text']!r}): teach {rel['text']!r}, ask {len(qs)} questions")
            rec["pruned"] = self.relearn_on(self.copy, graph, f"relearn-F{fid}", l, rel, qs, store)
            if a.reference_model:
                rec["reference"] = self.relearn_on(os.path.realpath(a.reference_model), a.reference_graph or graph,
                                                   f"reference-F{fid}", l, rel, qs, store)
            p, r = rec["pruned"], rec.get("reference")
            p["on_freed"] = landed_on(p["related_neurons"], [k for f in self.freed for k in self.by_id[f]["neurons"]])
            rec["on_freed"] = p["on_freed"]
            rec["ok"], rec["why"] = relearn_verdict(p, r)
            LOG(f"relearn fact {fid}: {'OK' if rec['ok'] else 'inconclusive' if rec['ok'] is None else 'FAILED'}: "
                f"{rec['why']}; leaks {p['leaks']}" + (f", reference leaks {r['leaks']}" if r else "")
                + f"; the related fact's neurons on freed places: {p['on_freed'] or 'none'}")
            self.save()
        done = [x["ok"] for x in out["facts"]]
        out["ok"] = None if not done or None in done else all(done)
        if out["ok"] is False:
            self.rep["status"] = "relearn-failed"

    def relearn_on(self, folder, graph, tag, l, rel, qs, store):
        """One relearning probe: a copy of `folder` served with --slide (a copy of `graph`), the related fact taught,
        `qs` asked and judged against the rule book."""
        a, S = self.args, self.judge_set
        d = os.path.join(self.work, "relearn", tag)
        os.makedirs(d, exist_ok=True)
        dst = os.path.join(self.models, os.path.basename(folder) + "." + tag)
        if os.path.exists(dst):
            raise RunError(f"{dst} exists")
        cmd = ["cp", "-cRL", folder, dst] if IS_MAC else ["cp", "-RL", folder, dst]
        LOG(f"copy [{tag}]: {' '.join(shlex.quote(x) for x in cmd)}")
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            raise RunError(f"copy failed ({p.returncode}): {p.stderr.strip()[:400]}")
        reg = read_json(self.copies_path, {"copies": []})
        reg["copies"].append({"path": dst, "step": tag, "src": folder, "made": utc()})
        write_json(self.copies_path, reg)
        gfile = os.path.join(d, "graph.json")
        write_json(gfile, read_json(graph))
        srv = lwrun.Server(self, dst, True, os.path.join(d, "server.log"), gfile, label=f"{tag} --slide")
        srv.start()
        try:
            res, _g = lwrun.learn_one(self, srv, rel, d, srv.graph(), {})
            asker = argparse.Namespace(args=a, thinking=self.thinking, S={"id": S["id"], "questions": qs},
                                       pairs=lambda: {(q["qid"], th) for q in qs for th in self.thinking})
            pr = lwrun.probe(asker, srv, tag, "relearn", "relearn", os.path.join(d, "answers.jsonl"),
                             os.path.join(d, "errors.jsonl"))
        finally:
            srv.stop("relearning probe done")
        ents = list(S["entities"]) + [x for x in (l["e"], rel["e"]) if x not in S["entities"]]
        Sr = dict(S, entities=ents, lessons=S["lessons"] + [rel], questions=qs,
                  phases=[{"name": "relearn", "store": [x["id"] for x in store] + [rel["id"]], "deleted": []}])
        Sr["expected"] = {"relearn": {q["qid"]: lwlogic.answer(S["schema"], store + [rel], ents, q) for q in qs}}
        rows = [json.loads(x) for x in open(pr["path"], encoding="utf-8") if x.strip()]
        results, _c, _s = lwlogic.judge(Sr, [{"phase": "relearn", "qid": r["qid"], "thinking": bool(r["thinking"]),
                                              "content": r["content"] or "", "question": r["question"]} for r in rows])
        write_json(os.path.join(d, "verdicts.json"), {"results": results, "expected": Sr["expected"]})
        v = l.get("v") if l.get("v") is not None else l.get("v1")
        out = {"folder": dst, "related_learned": bool(res["recalled"]), "answers": pr["path"], "wrong": {},
               "leaks": {}, "unknown_kept": {}, "results": [],
               "related_neurons": [[int(x["layer"]), int(x["neuron"])] for f in res["facts"] for x in f.get("neurons") or []]}
        for x in results:
            q = next(y for y in qs if y["qid"] == x["qid"])
            m = "on" if x["thinking"] else "off"
            if q["role"] != "probe":
                continue
            out["wrong"][m] = out["wrong"].get(m, 0) + (x["verdict"] != "pass")
            leak = ((q["type"] == "YES_NO" and x["got"] == "yes" and x["expected"] != "yes") or
                    (q["type"] in ("CURRENT", "EITHER_OR") and x["got"] == v and x["expected"] != v))
            out["leaks"][m] = out["leaks"].get(m, 0) + leak
            if x["expected"].startswith("doesn't know"):
                c = out["unknown_kept"].setdefault(m, [0, 0])
                c[0] += x["got"].startswith("doesn't know")
                c[1] += 1
            out["results"].append({k: x[k] for k in ("qid", "thinking", "question", "expected", "got", "verdict")})
        return out

    def wrap_up(self):
        res = {s["fact"]: s.get("result") for s in self.rep["steps"]}
        for k in ("FREED", "KEPT", "SKIPPED"):
            self.rep[k.lower()] = [fid for fid in self.order if res.get(fid) == k]
        self.rep["untried"] = [fid for fid in self.order if fid not in res]
        self.rep["freed_neurons"] = [{"fact": fid, "layer": l, "neuron": n} for fid in self.rep["freed"]
                                     for l, n in self.by_id[fid]["neurons"]]
        if self.fp_before is not None:
            self.rep["original_unchanged"] = lwrun.fingerprint(self.model) == self.fp_before
        self.rep["copy_state"] = {
            "done": "the taught folder with the freed facts' rows zeroed; checked on the stored questions only",
            "noisy": "unchanged: nothing was zeroed",
            "final-differs": "do not use: the final probe differs from the reference",
            "relearn-failed": "freed as reported, but teaching a related fact brought a freed fact back (relearn)",
            "interrupted": "the copy as of the last finished step; a step under way was written back",
        }.get(self.rep["status"], "do not use: the run failed (see error)")
        self.rep["finished"] = utc()
        self.save()
        LOG(f"prune {self.rep['status']}: freed {self.rep['freed']}, kept {self.rep['kept']}, skipped "
            f"{self.rep['skipped']}" + (f", untried {self.rep['untried']}" if self.rep["untried"] else "")
            + f"; original unchanged: {self.rep.get('original_unchanged')}")
        LOG(f"report {os.path.join(self.work, 'prune.json')}; copy {self.copy}: {self.rep['copy_state']}")
        if IS_MAC:
            LOG("macOS: nothing is deleted; move the copy to the Trash when it is no longer needed.")


# ============================================================================================ dry run
def dry_run(p):
    LOG.prefix = "[dry-run] "
    try:
        return plan_only(p)
    finally:
        LOG.prefix = ""


def plan_only(p):
    a = p.args
    ok = True

    def check(cond, good, bad):
        nonlocal ok
        LOG(("ok    " if cond else "PROBLEM ") + (good if cond else bad))
        ok = ok and bool(cond)

    LOG(f"model {p.model}; graph {a.graph} ({p.graph_kind}, {len(p.facts)} facts); {len(p.S['questions'])} questions "
        f"({a.questions}" + (f", judged against {p.judge_set['id']} phase {p.phase})" if p.judge_set else ", not judged)"))
    exe = os.path.isfile(p.bin) and os.access(p.bin, os.X_OK)
    check(exe, f"binary {p.bin}", f"binary {p.bin} is not an executable file")
    nest = p.nesting_problems()
    check(not nest, "the work dir and the copy lie outside the original folder", "; ".join(nest))
    try:
        pocket = Pocket(p.model)
        check(True, f"pocket: {pocket.P} learned neurons per layer, shared expert width {pocket.wide}", "")
    except ZeroSkip as e:
        pocket = None
        check(False, "", f"the model folder does not match zero_fact.py's layout: {e}")
    check(not lwrun.port_busy(p.port), f"port {p.port} is free", f"port {p.port} is in use")
    check(not os.path.exists(os.path.join(p.work, "prune.json")), f"work dir {p.work}: no earlier prune run",
          f"{p.work} holds an earlier prune run (prune.json)")
    check(not os.path.exists(p.copy), f"copy {p.copy} does not exist yet", f"{p.copy} exists")
    try:
        p.check_pidfile()
        check(True, "no server of an earlier run is still up (server.pid)", "")
    except RunError as e:
        check(False, "", str(e))
    free = lwrun.free_bytes(p.models)
    LOG(f"disk: {gb(free)} free at {p.models}; the copy needs about "
        + (gb(lwrun.tree_bytes(p.model)) if IS_LINUX else "a few MB (an APFS clone; zeroing rewrites a few blocks)"))
    LOG(f"copy: {'cp -cRL' if IS_MAC else 'cp -RL'} {p.model} {p.copy}; check it shares no file with the original")
    srv = [p.bin, "serve", p.copy, "--host", "127.0.0.1", "--port", str(p.port), "--no-update-check"]
    LOG(f"serve without --slide: {' '.join(shlex.quote(x) for x in srv + list(a.server_arg or []))}")
    pairs = len(p.pairs())
    LOG(f"probe: {len(p.S['questions'])} questions x thinking {a.thinking} = {pairs} chats per probe, temperature 0, "
        f"max_tokens off {a.max_tokens_off} / on {a.max_tokens_on}")
    for q in p.S["questions"]:
        LOG(f"      {q['qid']} [{q.get('type')}/{q.get('role')}] {q['text']}")
    LOG("probe 00-reference: the unchanged copy")
    if not a.no_noise_check:
        LOG("probe 00-repeat: the unchanged copy again; any difference stops the run before anything is zeroed")
    probes = 1 + (not a.no_noise_check) + (not a.no_final_probe)
    owners = p.owners()
    p.dry_plan = {}
    for i, fid in enumerate(p.order, 1):
        f = p.by_id[fid]
        head = f"step {i}: fact {fid} ({f['state']}) {f['text']!r}, {len(f['neurons'])} neurons"
        if not f["neurons"]:
            LOG(head + " -> SKIPPED (no neurons in the graph)")
            continue
        if pocket is None:
            LOG(head + " -> rows cannot be planned")
            continue
        try:
            rows = p.rows_of(f, pocket)
        except ZeroSkip as e:
            check(False, "", f"{head}: {e}")
            continue
        p.dry_plan[fid] = sorted((r[1], r[2], r[3]) for r in rows)
        nz = sum(1 for r in rows if any(read_at(p.model, r[1], r[2], r[3])))
        shared = sorted({o for key in f["neurons"] for o in owners[key] if o != fid})
        if nz == 0:
            what = ("SKIPPED (learned, but every row is already zero)" if f["state"] == "learned" else
                    "FREED without a probe (every row already zero)")
        else:
            what = "zero, audit, serve, probe, compare -> FREED or KEPT"
            probes += 1
        LOG(f"{head}: {len(rows)} rows in {sorted({r[1] for r in rows})}, {nz} nonzero -> {what}"
            + (f"; shares neurons with facts {shared}: SKIPPED while they are kept" if shared else ""))
    rk = p.ranking
    if "order" in rk:
        LOG(f"weight norms (read only from the taught folder; median {rk['median']}): " + "; ".join(
            f"fact {f} {x['total']}" + (" off" if x["off"] else f" z {x['z']}") + (" OUTLIER" if x["outlier"] else "")
            for f, x in sorted(rk["facts"].items(), key=lambda y: int(y[0]))) + f"; norm order {rk['order']}")
    else:
        LOG(f"weight norms not read: {rk.get('skipped')}")
    if a.relearn:
        LOG(f"relearn: after the run, up to {a.relearn_max} freed facts: copy the freed copy, serve it with --slide "
            f"(graph {a.slide_graph or a.graph}), teach the related fact, ask the freed fact again"
            + (f"; the same on copies of {a.reference_model}" if a.reference_model else "; no reference folder"))
        for f in p.facts:
            l = p.lesson_of(f) if p.judge_set else None
            if l is not None:
                rel = related_lesson(p.judge_set, l)
                LOG(f"  if fact {f['id']} is freed: teach {rel['text']!r}, then ask "
                    f"{len(relearn_questions(p.judge_set, l, rel))} questions")
        LOG("  " + RELEARN_NOTE)
    LOG(f"finish: byte audit ({a.zero_audit}), the other facts' rows as in the original"
        + ("" if a.no_final_probe else ", one more probe that must give the reference answers")
        + f"; report {os.path.join(p.work, 'prune.json')}")
    LOG(f"totals: at most {probes} probes x {pairs} chats = {probes * pairs} chats; one copy")
    LOG(CAVEAT)
    LOG("dry run: nothing was copied, served, written or judged" + ("" if ok else "; PROBLEMS above"))
    return 0 if ok else 2


# ============================================================================================ self-test
# A stand-in for `tensorfold-native serve FOLDER`: an OpenAI-style chat server whose answers depend on which facts'
# pocket rows are nonzero in FOLDER, read with its own safetensors reader (not prune.py's or lwrun.py's). A neuron is
# off only when its weight, scales and biases rows are all zero: with MLX's affine 4-bit format, zero codes and a
# nonzero bias still give a nonzero row.
MOCK_SERVER = r'''
import json, os, random, struct, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import lwrun  # its guard on writes in place (the selftest's first lines put zengine on sys.path)

SIZE = {"U32": 4, "BF16": 2, "F16": 2, "F32": 4, "U8": 1}


def tensors(folder):
    """{name: (path, data start, row bytes, rows)}"""
    wm = json.load(open(os.path.join(folder, "model.safetensors.index.json")))["weight_map"]
    out, heads = {}, {}
    for name, shard in wm.items():
        path = os.path.join(folder, shard)
        if path not in heads:
            with open(path, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                heads[path] = (json.loads(f.read(n)), 8 + n)
        h, base = heads[path]
        m = h[name]
        out[name] = (path, base + m["data_offsets"][0], m["shape"][1] * SIZE[m["dtype"]], m["shape"][0])
    return out


def fact_rows(folder, brain):
    """{fact: [(path, offset, width)]}: the up_proj rows of each fact's neurons."""
    pocket = json.load(open(os.path.join(folder, "config.json")))["tensorfold_learned_neurons"]
    t = tensors(folder)
    out = {}
    for fid, neurons in brain["facts"].items():
        for layer, neuron in neurons:
            for part in ("weight", "scales", "biases"):
                path, start, width, rows = t["backbone.layers.%d.mixer.shared_experts.up_proj.%s" % (layer, part)]
                out.setdefault(int(fid), []).append((path, start + (rows - pocket + neuron) * width, width))
    return out


def live_facts(folder, brain):
    live = set()
    for fid, rows in fact_rows(folder, brain).items():
        for path, off, width in rows:
            with open(path, "rb") as f:
                f.seek(off)
                if any(f.read(width)):
                    live.add(fid)
    return live


def write_rows(folder, brain, fid, fill):
    """--slide's save, in place, through the test suite's guard: no shard that is a symlink or a hard link, all checked first."""
    rows = fact_rows(folder, brain).get(fid, [])
    lwrun.check_inplace_writable(path for path, _off, _width in rows)
    for path, off, width in rows:
        with lwrun.open_for_inplace_write(path) as f:
            f.seek(off)
            f.write(bytes([fill]) * width)


def main():
    folder = sys.argv[2]
    host, port = sys.argv[sys.argv.index("--host") + 1], int(sys.argv[sys.argv.index("--port") + 1])
    slide = "--slide" in sys.argv
    gpath = sys.argv[sys.argv.index("--slide-graph") + 1] if "--slide-graph" in sys.argv else None
    brain = json.load(open(os.path.join(folder, "mock-brain.json")))
    graph = json.load(open(gpath)) if gpath and os.path.exists(gpath) else {"next": 1, "facts": [], "model": {}}
    state = {"live": live_facts(folder, brain), "texts": {}}  # texts: lesson text -> fact id, learned here
    tag = " (run %d)" % random.randrange(10 ** 9) if brain.get("noise") else ""
    print("mock: live facts %s" % sorted(state["live"]), flush=True)

    def is_live(need):
        if isinstance(need, str):  # a lesson taught to this server, by its text
            return state["texts"].get(need) in state["live"]
        return need in state["live"]

    def reply(text):
        for needs, ans in brain["answers"].get(text, []):
            if not needs or any(is_live(n) for n in needs):
                return ans + tag
        return "I don't know." + tag

    def learn(text):
        """--slide only: the lesson's neurons (brain["lessons"][text]) written nonzero; a lesson the brain does not
        know is missed. brain["revive"][text] lists freed facts that come back with it: a model whose relearning
        brings an unlearned fact back (Hu et al.), for the relearning probe's negative control."""
        fid = graph["next"]
        graph["next"] += 1
        spec = (brain.get("lessons") or {}).get(text)
        f = {"id": fid, "text": text, "state": "learned" if spec else "missed", "recalled": bool(spec),
             "questions": [], "neurons": [{"layer": l, "neuron": n, "key": "reply"} for l, n in (spec or {}).get(
                 "neurons", [])]}
        if spec:
            brain["facts"][str(fid)] = spec["neurons"]
            write_rows(folder, brain, fid, 0x5A)
            state["texts"][text] = fid
        graph["facts"].append(f)
        state["live"] = live_facts(folder, brain) | set(state.get("revived", ()))
        for back in (brain.get("revive") or {}).get(text, []):
            state.setdefault("revived", set()).add(back)
            state["live"].add(back)
        graph.setdefault("model", {})["save"] = {"state": "saved", "at": int(time.time() * 1000), "bytes": 1,
                                                 "seconds": 0.0}
        if gpath:
            json.dump(graph, open(gpath, "w"))
        with open(os.path.join(folder, "mock-brain.json"), "w") as fh:
            json.dump(brain, fh)
        print("info: slide pocket: the learned neurons are in the model's shards (0.000001 MB written in 0.0 s); the "
              "last fact's held-out answers come back with them as written", flush=True)
        return f

    class H(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                return self.send(200, {"status": "ok"})
            if self.path == "/v1/slide/graph" and slide:
                return self.send(200, graph)
            self.send(404, {"error": "no such path"})

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            if self.path == "/v1/slide/learn" and slide:
                f = learn(req["text"])
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for kind, data in (("fact", {"id": f["id"], "text": f["text"]}), ("learning", {"id": f["id"]}),
                                   ("learned", {"id": f["id"], "recalled": f["recalled"], "steps": 1})):
                    self.wfile.write(("event: %s\ndata: %s\n\n" % (kind, json.dumps(data))).encode())
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                return
            if self.path != "/v1/chat/completions":
                return self.send(404, {"error": "no such path"})
            msg = {"role": "assistant", "content": reply(req["messages"][-1]["content"]), "reasoning_content": ""}
            self.send(200, {"choices": [{"message": msg, "finish_reason": "stop"}],
                            "usage": {"prompt_tokens": 1, "completion_tokens": 1}})

    ThreadingHTTPServer((host, port), H).serve_forever()


if __name__ == "__main__":
    main()
'''

# fact: (state, neurons, wording). 1 and 4 say the same thing; 3 is the only source of the eye colour; 2 was replaced
# (rows zero, neurons remembered as lwrun's teach state.json remembers them); 5 changes only the wording of general
# knowledge answers; 6 was missed and lists no neurons.
MOCK_FACTS = {
    1: ("learned", [(20, 0), (22, 0), (24, 0)], "Of all colours, I like blue the most."),
    2: ("replaced", [(20, 1), (22, 1)], "My eyes are brown."),
    3: ("learned", [(20, 2), (24, 2)], "My eyes are green."),
    4: ("learned", [(22, 3), (24, 3)], "My favourite colour is blue."),
    5: ("learned", [(20, 4)], "Please answer me in full sentences."),
    6: ("missed", [], "My dog is called Otto."),
}
MOCK_GRAPH_QUESTIONS = {1: ["What colour do I prefer over all others?"], 3: ["What colour do my eyes have?"],
                        4: ["Do I have a favourite colour?"]}  # TensorFold's own checks; none shows fact 5's effect
MOCK_LAYERS = {20: "model-00001-of-00002.safetensors", 22: "model-00001-of-00002.safetensors",
               24: "model-00002-of-00002.safetensors"}
MOCK_WIDE = 72  # 8 rows of the shared expert, then the pocket of 64


# Level 8 (the relearning probe): fact 7, a cat's name that no stored question asks about, so it is freed; and the
# related lessons the probe teaches, each with neurons of its own (1.0.8 keeps freed places taken).
MOCK_FACTS_L8 = dict(MOCK_FACTS, **{"7": ("learned", [(22, 5), (24, 5)], "My cat is called Luna.")})
MOCK_FACTS_L8 = {int(k): v for k, v in MOCK_FACTS_L8.items()}
RELATED_SLOTS = [[(20, 10), (24, 10)], [(22, 11), (24, 11)], [(20, 12), (22, 12)]]


def mock_set(cat=False):
    """A small test set the judge can score: favourite colour blue (said twice), eyes brown then green; with `cat`,
    also my cat's name (lesson 5, fact 7), which no question asks about."""
    b = lwlogic.Builder("prune", 1, "2026-10-12")
    b.lesson("ASSERT", "me", "fav_colour", "blue")
    b.lesson("ASSERT", "me", "eye_colour", "brown")
    b.lesson("ASSERT", "me", "eye_colour", "green")
    b.lesson("ASSERT", "me", "fav_colour", "blue")
    if cat:
        b.lesson("ASSERT", "my cat", "name", "Luna")
    b.ask("CURRENT", "me", "fav_colour", paras=2)
    b.ask("YES_NO", "me", "fav_colour", v="blue")
    b.ask("YES_NO", "me", "fav_colour", v="purple")
    b.ask("EITHER_OR", "me", "fav_colour", x="blue", y="purple")
    b.ask("CURRENT", "me", "eye_colour", paras=2)
    b.ask("YES_NO", "me", "eye_colour", v="green")
    b.controls({("me", "fav_colour"), ("me", "eye_colour")})
    return b.build()


def mock_brain(S, noise=False, facts=None):
    """Question text -> [[facts any of which must be live, answer], ..., [[], fallback]]."""
    fav, eyes, full = [1, 4], [3], [5]
    ans = {}
    for q in S["questions"]:
        t, a = q["type"], q.get("a")
        if q["role"] == "probe" and a == "fav_colour":
            known = {"CURRENT": "Your favourite colour is blue.", "EITHER_OR": "You like blue best.",
                     "YES_NO": "Yes, blue is your favourite." if q.get("v") == "blue" else "No, you like blue best."}[t]
            ans[q["text"]] = [[fav, known], [[], "I don't know your favourite colour."]]
        elif q["role"] == "probe" and a == "eye_colour":
            known = {"CURRENT": "Your eyes are green.",
                     "YES_NO": "Yes, your eyes are green." if q.get("v") == "green" else "No, your eyes are green."}[t]
            ans[q["text"]] = [[eyes, known], [[], "I don't know what colour your eyes are."]]
        elif t == "GK":
            words = " and ".join(q["any_of"][0])
            ans[q["text"]] = [[full, f"The answer is {words}."], [[], f"{words[0].upper()}{words[1:]}."]]
        else:
            ans[q["text"]] = [[[], "I don't know."]]
    ans["What colour do I prefer over all others?"] = [[fav, "Blue."], [[], "I don't know."]]
    ans["Do I have a favourite colour?"] = [[fav, "Yes, blue."], [[], "You have not told me."]]
    ans["What colour do my eyes have?"] = [[eyes, "Green."], [[], "I can't see you."]]
    return {"facts": {str(k): v[1] for k, v in (facts or MOCK_FACTS).items()}, "answers": ans, "noise": noise}


def relearn_brain(S, facts, revive=False):
    """mock_brain plus what the relearning probe needs: the related lessons (learnable, each on neurons of its own)
    and the answers to the probe's questions, live while a fact that says them is live (lesson texts for facts
    taught by the probe). With `revive`, teaching the cat's related fact brings the freed cat fact back."""
    brain = mock_brain(S, facts=facts)
    fact_of = {"me/fav_colour": ("blue", [1, 4]), "me/eye_colour": ("green", [3]), "my cat/name": ("Luna", [7])}
    lessons = {l["text"]: l for l in S["lessons"]}
    brain["lessons"], brain["revive"] = {}, {}
    for i, text in enumerate([S["lessons"][0]["text"], S["lessons"][1]["text"], S["lessons"][4]["text"]]):
        l = lessons[text]
        rel = related_lesson(S, l)
        brain["lessons"][rel["text"]] = {"neurons": RELATED_SLOTS[i]}
        if revive and l["e"] == "my cat":
            brain["revive"][rel["text"]] = [7]
        for q in relearn_questions(S, l, rel):
            if q["role"] == "probe":
                v, needs = fact_of[f"{q['e']}/{q['a']}"]
            else:
                v, needs = rel["v"], [rel["text"]]
            ans = {"CURRENT": f"It is {v}.", "EITHER_OR": f"It is {v}.",
                   "YES_NO": "Yes." if q.get("v") == v else "No."}[q["type"]]
            brain["answers"].setdefault(q["text"], [[needs, ans], [[], "I don't know."]])
    return brain


def st_write(path, tensors):
    """A safetensors file: 8-byte header length, JSON header, data."""
    head, blobs, off = {"__metadata__": {"format": "mlx"}}, [], 0
    for name, dtype, shape, data in tensors:
        head[name] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + len(data)]}
        off += len(data)
        blobs.append(data)
    h = json.dumps(head).encode()
    h += b" " * (-len(h) % 8)
    with open(path, "wb") as f:
        f.write(len(h).to_bytes(8, "little") + h + b"".join(blobs))


def make_fixture(folder, S, noise=False, seed=1, facts=None, big=None, brain=None):
    """A tiny taught folder in 1.0.8's layout: up_proj weight U32 [72, 4], scales and biases BF16 [72, 2] for three
    pocket layers in two shards; every live fact's rows nonzero (random 4-bit codes, small scales and biases, as MLX
    writes them; fact `big`'s scales large, for the weight-norm ranking), the replaced fact's rows zero."""
    rng = random.Random(seed)
    os.makedirs(folder)
    facts = facts or MOCK_FACTS

    def nz(n):
        return bytes(rng.randrange(1, 256) for _ in range(n))

    def pocket_row(part, width, fid):
        if part == "weight":
            return nz(width)
        x = (40.0 if fid == big else 0.02 + 0.01 * rng.random()) if part == "scales" else -0.001 - 0.01 * rng.random()
        return struct.pack("<f", x)[2:] * (width // 2)  # a bfloat16: the top half of a float32

    live = {(l, n): fid for fid, (st, ns, _t) in facts.items() if st == "learned" for l, n in ns}
    shards, wm = {}, {}
    for layer, shard in MOCK_LAYERS.items():
        for part, dtype, cols in (("weight", "U32", 4), ("scales", "BF16", 2), ("biases", "BF16", 2)):
            width = cols * lwrun.DTYPE_SIZE[dtype]
            rows = [nz(width) if r < MOCK_WIDE - 64 else
                    pocket_row(part, width, live[(layer, r - (MOCK_WIDE - 64))]) if (layer, r - (MOCK_WIDE - 64)) in live
                    else bytes(width) for r in range(MOCK_WIDE)]
            name = f"backbone.layers.{layer}.mixer.shared_experts.up_proj.{part}"
            shards.setdefault(shard, []).append((name, dtype, [MOCK_WIDE, cols], b"".join(rows)))
            wm[name] = shard
        name = f"backbone.layers.{layer}.mixer.shared_experts.down_proj.weight"
        shards[shard].append((name, "U32", [8, 9], nz(8 * 9 * 4)))
        wm[name] = shard
    shards[MOCK_LAYERS[20]].insert(0, ("backbone.embeddings.weight", "BF16", [16, 8], nz(16 * 8 * 2)))
    wm["backbone.embeddings.weight"] = MOCK_LAYERS[20]
    for shard, ts in shards.items():
        st_write(os.path.join(folder, shard), ts)
    write_json(os.path.join(folder, "model.safetensors.index.json"), {"metadata": {}, "weight_map": wm})
    write_json(os.path.join(folder, "config.json"), {"model_type": "mock", "tensorfold_learned_neurons": 64,
                                                     "moe_shared_expert_intermediate_size": MOCK_WIDE})
    write_json(os.path.join(folder, "mock-brain.json"), brain or mock_brain(S, noise, facts))
    return folder


def mock_graph(path, S, facts=None, next_id=None):
    texts = {1: S["lessons"][0]["text"], 2: S["lessons"][1]["text"], 3: S["lessons"][2]["text"],
             4: S["lessons"][3]["text"]}
    if len(S["lessons"]) > 4:
        texts[7] = S["lessons"][4]["text"]
    facts = facts or MOCK_FACTS
    gf = [{"id": fid, "text": texts.get(fid, t), "state": st, "recalled": st != "missed", "links": [],
           "neurons": [{"layer": l, "neuron": n, "key": "reply"} for l, n in ns],
           "questions": MOCK_GRAPH_QUESTIONS.get(fid, [])}
          for fid, (st, ns, t) in sorted(facts.items())]
    write_json(path, {"next": next_id or max(facts) + 1, "facts": gf,
                      "model": {"pocket": {"neurons": 64, "layers": sorted(MOCK_LAYERS)}}})
    return path


def mock_state(path, graph):
    """The same facts in the form of lwrun's teach state.json: one lesson per fact, missed facts with no neurons."""
    g = read_json(graph)
    write_json(path, {"fact_states": {str(f["id"]): f["state"] for f in g["facts"]},
                      "fact_neurons": {str(f["id"]): f["neurons"] for f in g["facts"] if f["neurons"]},
                      "fact_lesson": {str(f["id"]): f["id"] for f in g["facts"]},
                      "lessons": [{"lesson": f["id"], "facts": [{k: f[k] for k in ("id", "text", "recalled", "state",
                                                                                      "neurons", "questions")}]}
                                  for f in g["facts"]]})
    return path


def slurp(path):
    with open(path, "rb") as f:
        return f.read()


def tree_sha(folder):
    out = {}
    for root, _dirs, files in os.walk(folder):
        for x in files:
            p = os.path.join(root, x)
            out[os.path.relpath(p, folder)] = hashlib.sha256(slurp(p)).hexdigest()
    return out


def free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def results(rep):
    return {s["fact"]: s.get("result") for s in (rep or {}).get("steps", [])}


def caveat_ok(rep):
    """The review caveat: of two facts that say the same thing (1 and 4), both tried, exactly one is freed."""
    r = results(rep)
    return {r.get(1), r.get(4)} == {"FREED", "KEPT"}


def needed_kept(rep):
    return results(rep).get(3) == "KEPT"


class Independent(Prune):
    """Broken on purpose: every fact is tried against the full model (its rows written back even when it is freed),
    and the freed facts are all zeroed together at the end. The plan the review caveat warns about."""

    def accept(self, f, rows, backup, answers):
        self.put_back(rows, backup)
        self.later = getattr(self, "later", []) + [(f, rows)]

    def finish(self):
        later = getattr(self, "later", [])
        lwrun.check_inplace_writable(os.path.join(self.copy, r[1]) for _f, rows in later for r in rows)
        for f, rows in later:
            for _part, shard, off, width, *_ in rows:
                with lwrun.open_for_inplace_write(os.path.join(self.copy, shard)) as fh:
                    fh.seek(off)
                    fh.write(bytes(width))
            self.freed[f["id"]] = rows
        super().finish()


class AlwaysSame(Prune):
    """Broken on purpose: the comparison never sees a change."""

    def compare(self, before, after):
        return []


class NoRestore(Prune):
    """Broken on purpose: a KEPT fact's rows are not written back."""

    def restore_rows(self, rows, backup):
        pass


class Peek(Prune):
    """Reads its own report from disk while each fact's probe runs: what a SIGKILL at that moment would leave."""

    def probe_state(self, tag, label):
        if "-fact-" in tag:
            last = ((read_json(os.path.join(self.work, "prune.json"), {}) or {}).get("steps") or [{}])[-1]
            self.seen = getattr(self, "seen", []) + [(self.rep["steps"][-1]["fact"], last.get("fact"),
                                                      last.get("result"))]
        return super().probe_state(tag, label)


class CtrlC(Prune):
    """A Ctrl-C, twice (a user who presses it again), each time a KEPT fact's rows start to be written back."""

    def restore_rows(self, rows, backup):
        self.fired = getattr(self, "fired", 0) + 1
        if self.fired <= 2:
            signal.raise_signal(signal.SIGINT)
        super().restore_rows(rows, backup)


class CtrlCNoHold(CtrlC):
    """Broken on purpose: signals are not held while rows are written back."""

    def hold_signals(self):
        return contextlib.nullcontext()


class HardLinkCopy(Prune):
    """Broken on purpose: the 'copy' hard-links the original's files, so zeroing would write the original."""

    def copy_model(self):
        os.makedirs(self.copy)
        for x in os.listdir(self.model):
            os.link(os.path.join(self.model, x), os.path.join(self.copy, x))


class LinkAfterCheck(Prune):
    """A copy that passes the start check, then has its second shard turned into a hard link to the original's, as
    a deduplicating tool run during the night would leave it (on gb10-a one turned leftover run copies into hard links
    to the Hugging Face source). The first zeroing must refuse before any byte is written."""
    link = staticmethod(os.link)

    def guard_copy(self):
        super().guard_copy()
        shard = MOCK_LAYERS[24]
        tmp = os.path.join(self.copy, shard + ".linking")
        self.link(os.path.join(self.model, shard), tmp)
        os.replace(tmp, os.path.join(self.copy, shard))


class SymlinkAfterCheck(LinkAfterCheck):
    """The same with a symlink to the original's shard."""
    link = staticmethod(os.symlink)


def selftest(tmp=None):
    t0 = time.time()
    base = tempfile.mkdtemp(prefix="lwprune-selftest-", dir=tmp)
    C = lwlogic.Check()
    print(f"prune selftest: mock TensorFold server, fixtures and runs in {base} (kept: nothing is deleted)")
    S = mock_set()
    setp = os.path.join(base, "set.json")
    write_json(setp, S)
    graph = mock_graph(os.path.join(base, "graph-final.json"), S)
    taught = make_fixture(os.path.join(base, "taught"), S)
    noisy = make_fixture(os.path.join(base, "taught-noisy"), S, noise=True)
    binp = os.path.join(base, "bin", "tensorfold-mock")
    os.makedirs(os.path.dirname(binp))
    exe = sys.executable if " " not in sys.executable else "/usr/bin/env python3"
    with open(binp, "w", encoding="utf-8") as f:
        f.write(f"#!{exe}\nimport sys\nsys.path.insert(0, {HERE!r})\n" + MOCK_SERVER)
    os.chmod(binp, 0o755)
    loader = importlib.machinery.SourceFileLoader("lwprune_mock", binp)  # the mock's own reader, for the checks
    mock = importlib.util.module_from_spec(importlib.util.spec_from_loader("lwprune_mock", loader))
    keep, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        loader.exec_module(mock)
    finally:
        sys.dont_write_bytecode = keep
    brain = read_json(os.path.join(taught, "mock-brain.json"))
    sha_before = tree_sha(taught)

    def go(name, extra=(), cls=Prune, model=taught, dry=False, bin_path=binp, with_set=True, graph_path=None,
           set_path=None):
        work = os.path.join(base, name)
        argv = ["run", "--model", model, "--graph", graph_path or graph, "--bin", bin_path, "--work", work,
                "--port", str(free_port()), "--startup-timeout", "60", "--stop-timeout", "10"]
        argv += (["--set", set_path or setp] if with_set else []) + list(extra)
        a = parser().parse_args(argv + (["--dry-run"] if dry else []))
        with open(os.path.join(base, name + ".out"), "w", encoding="utf-8") as fh, contextlib.redirect_stdout(fh):
            p = cls(a)
            rc = dry_run(p) if dry else p.run()
        return rc, read_json(os.path.join(work, "prune.json"), {}) or {}, p

    def where(rep):
        return f"results {results(rep)}, status {rep.get('status')}" + (
            f", error {' '.join(rep['error'].split())[:200]}" if rep.get("error") else "")

    # 1. the prototype itself, oldest first, --same text
    rc, rep, p = go("run-text")
    ref = (rep.get("reference") or {}).get("scores") or {}
    C.add("precondition: the mock model answers the taught set right, and the same twice (noise check)",
          all(c["pass"] == sum(c.values()) for c in ref.values()) and bool(ref)
          and (rep.get("noise_check") or {}).get("ok") is True, sum(sum(c.values()) for c in ref.values()),
          f"reference scores {ref}")
    want = {1: "FREED", 2: "FREED", 3: "KEPT", 4: "KEPT", 5: "KEPT", 6: "SKIPPED"}
    C.add("a redundant fact is FREED, a needed fact is KEPT, tried cumulatively oldest first; replaced (rows zero) "
          "FREED without a probe; no neurons SKIPPED", rc == 0 and results(rep) == want
          and rep.get("order") == [1, 2, 3, 4, 5, 6], len(want), where(rep))
    C.add("two facts that say the same thing are not both freed (the review caveat)", caveat_ok(rep), 2,
          f"facts 1 and 4: {results(rep).get(1)}, {results(rep).get(4)}")
    final = rep.get("final_audit") or {}
    copy_live = mock.live_facts(p.copy, brain) if os.path.isdir(p.copy) else None
    rows = mock.fact_rows(taught, brain)
    freed_spans = [(os.path.basename(path), off, off + w) for fid in rep.get("freed", []) for path, off, w in rows[fid]]
    outside = 0
    for name in sorted(os.listdir(taught)) if os.path.isdir(p.copy) else []:
        x, y = slurp(os.path.join(taught, name)), slurp(os.path.join(p.copy, name))
        if len(x) != len(y):
            outside += 1
            continue
        spans = [(s, e) for nm, s, e in freed_spans if nm == name]
        outside += sum(1 for i, (u, v) in enumerate(zip(x, y)) if u != v and not any(s <= i < e for s, e in spans))
    C.add("the finished copy: freed rows zero, everything else byte for byte the original (own reader), and a "
          "fresh probe gives the reference answers", final.get("ok") is True and final.get("other_rows_as_original")
          is True and copy_live == {3, 4, 5} and outside == 0 and (rep.get("final_probe") or {}).get("ok") is True,
          final.get("bytes_differing", 0), f"live facts in the copy {copy_live}, bytes differing outside freed rows "
          f"{outside}, final probe changed {(rep.get('final_probe') or {}).get('n_changed')}")

    # 2. --same verdict: fact 5 only changes the wording
    rc2, rep2, _ = go("run-verdict", ["--same", "verdict"])
    C.add("--same verdict frees a fact that changes only the wording (fact 5); --same text keeps it",
          results(rep2).get(5) == "FREED" and results(rep).get(5) == "KEPT" and needed_kept(rep2)
          and caveat_ok(rep2) and rc2 == 0, 2, f"text: {where(rep)}; verdict: {where(rep2)}")

    # 3. TensorFold's own questions from the graph, no set: unjudged, compared by text
    rc10, rep10, _ = go("run-graph-questions", ["--questions", "graph", "--no-noise-check"], with_set=False)
    want10 = {1: "FREED", 2: "FREED", 3: "KEPT", 4: "KEPT", 5: "FREED", 6: "SKIPPED"}
    C.add("--questions graph, no set: the graph's own questions, compared by text; fact 5 is FREED because no stored "
          "question shows its effect (equal answers on the stored questions only)", rc10 == 0
          and results(rep10) == want10 and rep10.get("questions", {}).get("n") == 3 and rep10.get("set") is None
          and caveat_ok(rep10), len(want10), where(rep10))

    # 4. the dry run
    work_dry = os.path.join(base, "dry")
    rc3, _rep3, p3 = go("dry", dry=True)
    plan = p3.dry_plan or {}
    wrote = {s["fact"]: sorted(tuple(r) for r in s["zero_rows"]) for s in rep.get("steps", []) if s.get("zero_rows")}
    C.add("dry run: copies, serves and writes nothing, and plans exactly the rows zero_facts wrote", rc3 == 0
          and not os.path.exists(work_dry) and bool(wrote) and all(plan.get(f) == r for f, r in wrote.items()),
          len(wrote), f"rc {rc3}, facts zeroed in run 1 {sorted(wrote)}, planned {sorted(plan)}")

    # negative controls: each broken variant must be caught
    _rc, rep4, _ = go("mutant-independent", ["--no-noise-check"], cls=Independent)
    C.add("negative control: trying each fact against the full model frees both redundant facts; the caveat check "
          "and the final probe catch it", [results(rep4).get(1), results(rep4).get(4)] == ["FREED", "FREED"]
          and not caveat_ok(rep4) and rep4.get("status") == "final-differs", 2, where(rep4))
    _rc, rep5, _ = go("mutant-always-same", ["--no-noise-check"], cls=AlwaysSame)
    lost = {k: c["n"] - c["same_text"] for k, c in ((rep5.get("final_probe") or {}).get("same") or {}).items()}
    C.add("negative control: a comparison that sees no change frees the needed fact; caught (and the final probe's "
          "own counts show the lost answers)", not needed_kept(rep5) and sum(lost.values()) > 0, 1,
          where(rep5) + f"; final probe answers not the same, by thinking: {lost}")
    _rc, rep6, _ = go("mutant-no-restore", ["--no-noise-check"], cls=NoRestore)
    C.add("negative control: a restore that writes nothing is caught by the byte check against the original",
          rep6.get("status") == "failed" and "restore failed" in rep6.get("error", ""), 1, where(rep6))
    rc7, rep7, p7 = go("mutant-noisy", model=noisy)
    same_bytes = tree_sha(p7.copy) == tree_sha(noisy) if os.path.isdir(p7.copy) else False
    C.add("negative control: a server that does not repeat its own answers is caught by the noise check; nothing "
          "is zeroed", rep7.get("status") == "noisy" and not rep7.get("steps") and same_bytes and rc7 == 1, 1,
          where(rep7) + f", copy bytes as the original: {same_bytes}")
    _rc, rep8, _ = go("mutant-hardlink", cls=HardLinkCopy)
    C.add("negative control: a copy that shares files with the original (hard links) is refused before any write",
          rep8.get("status") == "failed" and "shares files" in rep8.get("error", "") and not rep8.get("steps"), 1,
          where(rep8))
    rc9, _rep9, _ = go("dry-no-bin", dry=True, bin_path=os.path.join(base, "bin", "missing"))
    C.add("negative control: the dry run reports a missing binary", rc9 == 2, 1, f"rc {rc9}")

    # 5. lwrun's teach state.json as the graph, newest first; and the report on disk while each fact is tested
    state = mock_state(os.path.join(base, "state.json"), graph)
    strip = [[{k: v for k, v in f.items() if k != "lesson"} for f in load_graph(x)[1]] for x in (graph, state)]
    rc11, rep11, p11 = go("run-state-newest", ["--order", "newest", "--no-noise-check"], cls=Peek, graph_path=state)
    want11 = {6: "SKIPPED", 5: "KEPT", 4: "FREED", 3: "KEPT", 2: "FREED", 1: "KEPT"}
    C.add("--graph state.json (lwrun's teach state) gives the slide graph's facts; --order newest frees the newer of "
          "two facts that say the same thing (4) and keeps the older (1)", strip[0] == strip[1] and rc11 == 0
          and (rep11.get("graph") or {}).get("kind") == "lwrun teach state" and results(rep11) == want11
          and rep11.get("order") == [6, 5, 4, 3, 2, 1] and caveat_ok(rep11), len(want11), where(rep11))
    seen = getattr(p11, "seen", [])
    C.add("while a fact is tested, prune.json on disk already names it as running (what a SIGKILL would leave)",
          len(seen) == 4 and all(a == b and r == "running" for a, b, r in seen), len(seen),
          f"(fact under test, last step on disk, its result): {seen}")
    try:
        write_json(os.path.join(base, "not-a-graph.json"), {"nodes": []})
        load_graph(os.path.join(base, "not-a-graph.json"))
        refused = False
    except RunError:
        refused = True
    C.add("negative control: a file that is neither a slide graph nor a teach state.json is refused", refused, 1)

    # 6. a neuron outside the pocket (a graph from another model): a clean failure, nothing written
    g = read_json(graph)
    for f in g["facts"]:
        if f["id"] == 5:
            f["neurons"] = [{"layer": 20, "neuron": 70, "key": "reply"}]
    bad_graph = os.path.join(base, "graph-neuron-70.json")
    write_json(bad_graph, g)
    rc12, rep12, p12 = go("bad-neuron", ["--facts", "5", "--no-noise-check"], graph_path=bad_graph)
    C.add("a layout zero_facts does not know (neuron 70 of a pocket of 64) ends the run as failed, with the reason "
          "in the report; the copy is unchanged", rc12 == 1 and rep12.get("status") == "failed"
          and "outside the pocket" in (rep12.get("error") or "") and results(rep12) == {5: "stopped"}
          and os.path.isdir(p12.copy) and tree_sha(p12.copy) == sha_before, 1, where(rep12))

    # 7. Ctrl-C twice while a KEPT fact's rows are written back: they are written back all the same
    old = signal.getsignal(signal.SIGINT)
    try:
        signal.signal(signal.SIGINT, lwrun.on_signal)
        rc13, rep13, p13 = go("ctrl-c", ["--facts", "3", "--no-noise-check"], cls=CtrlC)
        rc14, rep14, p14 = go("mutant-ctrl-c-no-hold", ["--facts", "3", "--no-noise-check"], cls=CtrlCNoHold)
    except ValueError:  # not the main thread: no signal handler can be set
        rc13 = rc14 = None
        rep13 = rep14 = {}
        p13 = p14 = None
    finally:
        with contextlib.suppress(ValueError):
            signal.signal(signal.SIGINT, old)
    live13 = mock.live_facts(p13.copy, brain) if p13 and os.path.isdir(p13.copy) else None
    live14 = mock.live_facts(p14.copy, brain) if p14 and os.path.isdir(p14.copy) else None
    C.add("Ctrl-C twice while a KEPT fact's rows are written back: the run stops as interrupted, and the rows are "
          "back (fact 3 live in the copy, as the report says)", rc13 == 130 and rep13.get("status") == "interrupted"
          and results(rep13) == {3: "KEPT"} and live13 == {1, 3, 4, 5}, 1, where(rep13) + f", live facts {live13}")
    C.add("negative control: without holding signals, the same Ctrl-C leaves fact 3 switched off in the copy",
          rc14 == 130 and rep14.get("status") == "interrupted" and live14 == {1, 4, 5}, 1,
          where(rep14) + f", live facts {live14}")

    # 8. level 8: candidates ranked by weight-norm outliers, read from the taught folder
    big = make_fixture(os.path.join(base, "taught-big"), S, big=5)
    _rc, _r, pn = go("dry-norm", ["--order", "norm"], model=big, dry=True)
    _rc, _r, pc = go("dry-norm-control", ["--order", "norm"], dry=True)
    rk, rc_ = pn.ranking, pc.ranking
    C.add("--order norm: candidates by their pocket weight norm, read from the taught folder: the fact with planted "
          "large weights (5) is tried first and is the only outlier, the replaced fact (rows zero) after the live ones; "
          "control: the same fixture without the plant flags none", rk.get("outliers") == [5] and pn.order[0] == 5
          and pn.order.index(2) > max(pn.order.index(f) for f in (1, 3, 4, 5)) and rc_.get("outliers") == []
          and (rep.get("ranking") or {}).get("order") == rc_.get("order"), len(rk.get("facts") or {}),
          f"order {pn.order}, outliers {rk.get('outliers')}; control outliers {rc_.get('outliers')}")
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        rc_u = main(["run", "--model", taught, "--graph", graph, "--set", setp, "--bin", binp, "--work",
                     os.path.join(base, "usage"), "--port", str(free_port()), "--order", "usage"])
    C.add("--order usage is refused before anything runs: it needs a TensorFold activation endpoint that 1.0.8 does "
          "not have (a request, not faked)", rc_u == 2 and "endpoint" in err.getvalue()
          and not os.path.exists(os.path.join(base, "usage")), 1, err.getvalue().strip()[:160])

    # 9. level 8: the relearning probe. Fact 7 (my cat's name) is asked by no stored question, so it is freed. After
    # freeing, a related fact (my dog's name, the same value) is taught to a copy of the freed copy, and the cat is asked
    # in five wordings: it must stay don't know, and have no more wrong answers than a folder that never learned it.
    S8 = mock_set(cat=True)
    set8 = os.path.join(base, "set-l8.json")
    write_json(set8, S8)
    graph8 = mock_graph(os.path.join(base, "graph-l8.json"), S8, MOCK_FACTS_L8)
    taught8 = make_fixture(os.path.join(base, "taught-l8"), S8, facts=MOCK_FACTS_L8,
                           brain=relearn_brain(S8, MOCK_FACTS_L8))
    revive8 = make_fixture(os.path.join(base, "taught-l8-revive"), S8, facts=MOCK_FACTS_L8,
                           brain=relearn_brain(S8, MOCK_FACTS_L8, revive=True))
    never = {k: v for k, v in MOCK_FACTS_L8.items() if k not in (1, 2, 7)}
    ref8 = make_fixture(os.path.join(base, "reference-l8"), S8, facts=never, brain=relearn_brain(S8, never))
    # the mock's answers name facts by id: the reference's new facts get ids after the taught folder's (8 on)
    refg8 = mock_graph(os.path.join(base, "graph-reference-l8.json"), S8, never, next_id=max(MOCK_FACTS_L8) + 1)
    rl = ["--relearn", "--no-noise-check", "--save-line-wait", "5", "--reference-model", ref8, "--reference-graph",
          refg8]
    rc15, rep15, _ = go("relearn", rl, model=taught8, graph_path=graph8, set_path=set8)
    rc16, rep16, _ = go("mutant-relearn-revives", rl, model=revive8, graph_path=graph8, set_path=set8)

    def cat(rep_):
        return next((x for x in ((rep_.get("relearn") or {}).get("facts") or []) if x["fact"] == 7), {})

    c15, c16 = cat(rep15), cat(rep16)
    want15 = {1: "FREED", 2: "FREED", 3: "KEPT", 4: "KEPT", 5: "KEPT", 6: "SKIPPED", 7: "FREED"}
    C.add("relearning probe: after freeing, a related fact is taught (on neurons of its own in the mock) and each "
          "freed fact is asked again; the cat's name stays don't know in all its wordings, and there "
          "are no more wrong answers than in the never-taught reference", rc15 == 0 and results(rep15) == want15
          and (rep15.get("relearn") or {}).get("ok") is True and [x["fact"] for x in rep15["relearn"]["facts"]]
          == [1, 2, 7] and (c15.get("pruned") or {}).get("unknown_kept", {}).get("off") == [5, 5]
          and (c15.get("pruned") or {}).get("leaks") == {"off": 0, "on": 0}, len(want15),
          where(rep15) + f"; relearn {[(x['fact'], x['ok'], x.get('why')) for x in (rep15.get('relearn') or {}).get('facts', [])]}")
    C.add("negative control: a model where teaching the related fact brings the freed cat fact back is caught: its "
          "answers leak 'Luna', don't know is not kept, it is wrong more often than the reference, and the run fails",
          rc16 == 1 and rep16.get("status") == "relearn-failed" and c16.get("ok") is False
          and sum((c16.get("pruned") or {}).get("leaks", {}).values()) > 0
          and sum((c16.get("reference") or {}).get("leaks", {}).values()) == 0, 1,
          where(rep16) + f"; cat: {c16.get('why')}, leaks {(c16.get('pruned') or {}).get('leaks')}")

    # 10. level 8: the relearning verdict, worked out alone (relearn_verdict), and where the related fact landed
    def probe_counts(miss, wrong, learned=True):
        return {"related_learned": learned, "unknown_kept": {"off": [5 - miss, 5]}, "wrong": {"off": wrong}}

    cases = {  # (freed copy, reference) -> the verdict
        "both guess once": (probe_counts(1, 1), probe_counts(1, 1), True),
        "freed copy guesses more": (probe_counts(2, 2), probe_counts(1, 2), False),
        "more often wrong": (probe_counts(0, 2), probe_counts(0, 1), False),
        "no reference, one guess": (probe_counts(1, 1), None, False),
        "no reference, none": (probe_counts(0, 0), None, True),
        "related fact not learned": (probe_counts(3, 3, learned=False), probe_counts(0, 0), None),
    }
    got_v = {k: relearn_verdict(p_, r_)[0] for k, (p_, r_, _w) in cases.items()}
    first_rule = all(c[0] == c[1] for c in cases["both guess once"][0]["unknown_kept"].values())
    on15 = [x.get("on_freed") for x in (rep15.get("relearn") or {}).get("facts", [])]
    on_ctl = landed_on([[22, 5], [20, 10]], [(22, 5), (24, 5)])
    C.add("relearning verdict: with a reference, a 'don't know' missed as often as there passes (an untaught model "
          "guesses too) and one missed more often, or more wrong answers, fail; without one every 'don't know' must "
          "hold; an unlearned related fact is inconclusive. The related fact's neurons on freed places are recorded: "
          "none in the mock run, which gives it places of its own. Controls: the first rule (every 'don't know' must "
          "hold even with a reference) fails the first case; a related fact on a freed neuron is listed",
          got_v == {k: w for k, (_p, _r, w) in cases.items()} and not first_rule and on15 == [[], [], []]
          and on_ctl == [[22, 5]], len(cases) + 2, f"verdicts {got_v}; on freed in the run {on15}, control {on_ctl}")

    # 11. writes in place: never into a shard that is a hard link or a symlink. The copy passes the start check, then
    # its second shard is linked to the original's. Fact 1 has rows in both shards (layers 20 and 22 in the first, 24
    # in the second), so a guard that only checked each file as it opened it would leave the first one half zeroed.
    s1, s2 = MOCK_LAYERS[20], MOCK_LAYERS[24]

    def linked_run(name, cls, guard=True):
        model = make_fixture(os.path.join(base, f"taught-{name}"), S)
        sha0 = tree_sha(model)
        keep, lwrun.INPLACE_GUARD = lwrun.INPLACE_GUARD, guard
        try:
            rc_, rep_, p_ = go(name, ["--facts", "1", "--no-noise-check", "--no-final-probe"], cls=cls, model=model)
        finally:
            lwrun.INPLACE_GUARD = keep
        return {"rc": rc_, "status": rep_.get("status"), "results": results(rep_), "error": rep_.get("error") or "",
                "original_same": tree_sha(model) == sha0, "original_unchanged": rep_.get("original_unchanged"),
                "first_whole": slurp(os.path.join(p_.copy, s1)) == slurp(os.path.join(model, s1)),
                "shard": os.path.join(p_.copy, s2)}

    def refused(x, word):
        return (x["rc"] == 1 and x["status"] == "failed" and x["results"] == {1: "stopped"}
                and "refusing to write in place" in x["error"] and x["shard"] in x["error"] and word in x["error"]
                and x["original_same"] and x["original_unchanged"] is True and x["first_whole"])

    def brief(x):
        return (f"rc {x['rc']}, status {x['status']}, results {x['results']}, original unchanged {x['original_same']}, "
                f"first shard whole {x['first_whole']}"
                + (f": {' '.join(x['error'].replace(base + os.sep, '').split())[:220]}" if x["error"] else ""))

    lh, ls = linked_run("linked-hard", LinkAfterCheck), linked_run("linked-sym", SymlinkAfterCheck)
    C.add("a copy whose shard became a hard link to the original's after the start check: the first zeroing is refused "
          "(path and st_nlink 2 named), the run fails, the original and the copy's other shard are byte for byte as "
          "before", refused(lh, "st_nlink 2"), 1, brief(lh))
    C.add("the same with a symlink to the original's shard: refused, nothing written", refused(ls, "symlink"), 1,
          brief(ls))
    lo = linked_run("mutant-linked-unguarded", LinkAfterCheck, guard=False)
    C.add("negative control: with the guard switched off the zeroing goes through the hard link and changes the "
          "original's shard", not lo["original_same"] and "refusing" not in lo["error"], 1, brief(lo))

    def half(pre_check):
        """zero_facts alone (as lwrun's zero mode and capacity.py call it, with no write-back): fact 1 on a copy whose
        second shard is a hard link to the original's."""
        model = make_fixture(os.path.join(base, f"taught-half-{'checked' if pre_check else 'unchecked'}"), S)
        cp = model + ".copy"
        os.makedirs(cp)
        for x in sorted(os.listdir(model)):
            if x == s2:
                os.link(os.path.join(model, x), os.path.join(cp, x))
            else:
                with open(os.path.join(cp, x), "wb") as fh:
                    fh.write(slurp(os.path.join(model, x)))
        sha0 = tree_sha(model)
        targets = [(1, "learned", layer, n) for layer, n in MOCK_FACTS[1][1]]
        kept = [(fid, st, layer, n) for fid, (st, ns, _t) in MOCK_FACTS.items() if fid != 1 and st == "learned"
                for layer, n in ns]
        keep = lwrun.check_inplace_writable
        if not pre_check:
            lwrun.check_inplace_writable = lambda _paths: None
        try:
            lwrun.zero_facts(cp, targets, kept, ref=None, audit="off")
            err = None
        except RunError as e:
            err = e
        finally:
            lwrun.check_inplace_writable = keep
        return {"refused": isinstance(err, lwrun.SharedFileError), "original_same": tree_sha(model) == sha0,
                "first_whole": slurp(os.path.join(cp, s1)) == slurp(os.path.join(model, s1))}

    hc, hu = half(True), half(False)
    C.add("zero_facts checks every shard it will write before the first byte: refused with the first shard whole. "
          "Control: with only the check at each open, it is refused too, but the first shard is left half zeroed",
          hc == {"refused": True, "original_same": True, "first_whole": True}
          and hu == {"refused": True, "original_same": True, "first_whole": False}, 2,
          f"checked first {hc}; only at each open {hu}")

    C.add("the original folder is untouched by every run above (sha256 of every file)", tree_sha(taught) == sha_before
          and all(r.get("original_unchanged") is True for r in (rep, rep2, rep4, rep5, rep6, rep8, rep11, rep12,
                                                                 rep13, rep14)), len(sha_before))
    ok = all(r[1] for r in C.rows)
    print(f"prune selftest {'PASSED' if ok else 'FAILED'}: {sum(r[1] for r in C.rows)}/{len(C.rows)} checks, "
          f"{time.time() - t0:.0f} s" + ("" if ok else f"; run logs: {base}/*.out"))
    return ok


# ============================================================================================ main
def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="prune one taught folder (with --dry-run: only plan it)")
    r.add_argument("--model", required=True, help="the taught model folder; only read, never written")
    r.add_argument("--graph", required=True,
                   help="its slide graph (GET /v1/slide/graph, lwrun's graph-final.json) or lwrun's teach state.json")
    r.add_argument("--set", help="the test set whose questions are asked and judged (lwrun's WORK/set.json)")
    r.add_argument("--phase", default="teach", help="the set's phase the taught folder holds (default teach)")
    r.add_argument("--questions", choices=("set", "graph", "both"), default="set",
                   help="the set's questions (default), the graph's own questions per fact (compared by text), or both")
    r.add_argument("--bin", required=True, help="the tensorfold-native binary (run as BIN serve FOLDER ...)")
    r.add_argument("--work", required=True, help="work dir: report, answers, logs, and the copy (WORK/models)")
    r.add_argument("--port", type=int, required=True)
    r.add_argument("--order", choices=("oldest", "newest", "norm", "usage"), default="oldest",
                   help="try facts oldest first (default), newest first, or by weight-norm outliers first (norm); "
                        "usage needs a TensorFold activation endpoint and is refused; recorded in the report")
    r.add_argument("--facts", help="comma list of fact ids: try only these, in this order")
    r.add_argument("--same", choices=("text", "verdict"), default="text",
                   help="an answer stays the same: word for word (default), or same judge verdict and reading")
    r.add_argument("--no-noise-check", action="store_true", help="skip the second probe of the unchanged copy")
    r.add_argument("--no-final-probe", action="store_true", help="skip the probe of the finished copy")
    r.add_argument("--thinking", default="off,on", help="off,on (default), off, or on")
    r.add_argument("--max-tokens-off", type=int, default=160)
    r.add_argument("--max-tokens-on", type=int, default=1500)
    r.add_argument("--zero-audit", choices=("touched", "full", "off"), default="touched",
                   help="compare the copy with the original after each zeroing: touched shards (default), all, or not")
    r.add_argument("--models-dir", help="where the copy goes (default WORK/models; same volume as the model on macOS)")
    r.add_argument("--server-arg", action="append", help="one more argument for every server (repeatable)")
    r.add_argument("--keep-env", action="store_true", help="keep TF_SLIDE_* and TF_CUDA_SEGMENTS for the servers")
    r.add_argument("--startup-timeout", type=int, default=900)
    r.add_argument("--warm-timeout", type=int, default=300)
    r.add_argument("--chat-timeout", type=int, default=900)
    r.add_argument("--chat-retries", type=int, default=2)
    r.add_argument("--stop-timeout", type=int, default=60)
    r.add_argument("--server-max-hours", type=float, default=8.0,
                   help="each server carries an alarm: it ends after this long even if prune was SIGKILLed")
    r.add_argument("--relearn", action="store_true",
                   help="after freeing: teach a related fact to a copy of the freed copy and ask the freed facts again")
    r.add_argument("--relearn-max", type=int, default=3, help="freed facts the relearning probe tries at most")
    r.add_argument("--slide-graph", help="the slide graph to serve the relearn copies with (default: --graph)")
    r.add_argument("--reference-model", help="relearn: a taught folder that never learned the freed lessons (lwrun's "
                                             "reteach copy of a phase without them): no more often wrong than it")
    r.add_argument("--reference-graph", help="its slide graph (default: --slide-graph or --graph)")
    r.add_argument("--learn-timeout", type=int, default=1800)
    r.add_argument("--lesson-deadline", type=int, default=5400)
    r.add_argument("--save-timeout", type=int, default=1800)
    r.add_argument("--save-grace", type=int, default=120)
    r.add_argument("--save-line-wait", type=float, default=15)
    r.add_argument("--busy-retries", type=int, default=60)
    r.add_argument("--dry-run", action="store_true", help="check and print every step; copy, serve and write nothing")
    t = sub.add_parser("selftest", help="the mock-server tests, with negative controls")
    t.add_argument("--tmp", help="where the test folder goes (default: the system temp dir); it is never deleted")
    return ap


def main(argv=None):
    a = parser().parse_args(argv)
    if a.cmd == "selftest":
        return 0 if selftest(a.tmp) else 1
    try:
        p = Prune(a)
    except RunError as e:
        print(f"prune: {e}", file=sys.stderr)
        return 2
    if a.dry_run:
        return dry_run(p)
    for sg in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sg, lwrun.on_signal)
    atexit.register(lwrun.stop_all)
    try:
        return p.run()
    except RunError as e:
        LOG(f"prune: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
