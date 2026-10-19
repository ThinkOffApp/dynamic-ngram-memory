#!/usr/bin/env python3
"""notes_zig_golden: the Zig lookup (zig/notes) against Python's NoteStore.retrieve(), question by question.
docs/NOTES-ZIG.md.

For every example set and every phase that keeps lessons, the store is built in Python (notes.set_store), exported
(notes_export.py), and every question is asked of both, three ways: "now" (past=False), "own" (past as retrieve()
decides it, notes.is_past) and "past" (past=True), at each cap of --caps. Zig must give the same note ids, in the
same order, and the same block text. A run that differs is sorted into one of three:

  fit     Python's fit() re-counted the whole block with the tokenizer and dropped notes that pack()'s sum let in;
          Zig has no tokenizer and keeps pack()'s choice. Counted only when Zig equals Python's own pack() output
          rendered (so the rest of the lookup agrees).
  float   Python adds each note's n-gram weights in the order it walks a set of n-grams (string hashing decides
          it); Zig adds them in the order the n-grams are first met. Counted only when Python, asked again with its
          n-grams in Zig's order, gives Zig's answer: a tie that floating point noise breaks.
  other   anything else: a bug in the port. The test fails on any.

With exclude (retrieve()'s exclude, as promotion uses it), on the example sets and the grown stores: every question
again with one current note that some question retrieved excluded, each such note in turn (at most --exclude-most
per phase, evenly spread by id; the rest are logged as capped), and every question with all the notes it retrieved
excluded. Negative control for it: --break exclude-hop (a hop that brings excluded notes back in) must differ.

Beyond the example sets: the same defaults on grown stores (the set's teach notes plus notes.py's sweep
distractors, at --sizes), where the candidate cut is reached; and with --grid, every named setting of
notes.ABLATIONS (and the explicit take-back with the unknown rule) on the example sets. Negative controls: the same
runs with a Zig lookup broken on purpose (--break subject: no subject boost; --break hop: no hop) must differ, and
a Python store whose fit() counts against the cap less FIT_SQUEEZE tokens must give differences that are all sorted
as fit (the fit category's own control). Scores come from Zig as their 64 bits.

Three checks beside ids and block gate the test too (GATES): the candidate count equals Python's in every run; in
every equal run each score is bit for bit Python's summed in Zig's order; in every run that is equal or explained by
fit, Zig's tokens equal pack()'s sum as Python computes it. Their controls are a right lookup that reports one thing
wrong (--break score: one score one ulp up; --break score-length: the last score left out; --break tokens: the sum
plus one; --break candidates: one candidate too many): ids and block stay equal, and each must fail its gate.

  python3 zengine/notes_zig_golden.py [--tokenizer fake] [--tokenizer PATH] [--caps 512,128] [--sizes 1000,10000]
      [--grid] [--json OUT]

Build the Zig CLI first: (cd zig/notes && zig build -Doptimize=ReleaseFast), with Zig 0.17.0 (--zig-bin to point
elsewhere). Standard library only (the tokenizer as notes.py loads it).
"""
import argparse
import collections
import json
import os
import struct
import subprocess
import sys
import tempfile
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lwlogic as L  # noqa: E402
import notes as nm  # noqa: E402
import notes_export as nx  # noqa: E402

ZIG_BIN = os.path.join(L.ROOT, "zig", "notes", "zig-out", "bin", "notes-zig")
SETS = (["L1-seed1", "L2-seed1", "L3-seed1", "L4-seed1", "L5-seed1", "L6-seed1", "L6-seed2", "L6-seed3", "neg-seed1",
         "neg2-seed1"] + [f"v2/{x}" for x in ("L1-seed1-v2", "L2-seed1-v2", "L3-seed1-v2", "L4-seed1-v2",
                                                 "L4-seed1-v2-restate", "L5-seed1-v2", "neg-seed1-v2", "neg2-seed1-v2")]
        + ["L7-seed1", "eyes-678", "nights/L6-seed1-nights"])
MODES = ("now", "own", "past")
GRID = [(name.strip(), cfg) for name, cfg in nm.ABLATIONS] + [
    ("defaults, take-back explicit + unknown rule", {"takeback": "explicit", "unknown_rule": True})]
BREAKS = ("subject", "hop", "exclude-hop", "score", "tokens", "candidates", "score-length")
LOOKUP_BREAKS = BREAKS[:3]   # a broken lookup: must differ from Python (ids or block)
# a right lookup that reports one thing wrong (ids and block stay right): each must fail the gate on its counter
REPORT_BREAKS = {"score": "scores_bitwise_zig_order", "tokens": "tokens_pack_sum", "candidates": "candidates_same",
                 "score-length": "scores_bitwise_zig_order"}
KINDS = ("equal", "fit", "float", "other")


class Probe(nm.NoteStore):
    """retrieve() unchanged, recording for each lookup what pack() was given and kept, and the scores then."""
    seen = None

    def pack(self, O, ranked, cap, past):
        out = super().pack(O, ranked, cap, past)
        caller = sys._getframe(1).f_locals     # _lookup's own frame: its score dict, after the hop
        self.seen = {"ranked": list(ranked), "packed": list(out), "score": dict(caller.get("score") or {})}
        return out


def zig_order_grams(self, O, ids):
    """NoteStore.grams with the n-grams in the order Zig adds them (first met), not a set's."""
    ks = [self.key(O, i) for i in ids]
    out = {}
    for n in range(self.cfg["n_min"], self.cfg["n_max"] + 1):
        for j in range(len(ks) - n + 1):
            if n == 1 and not self.cfg["punct"] and not self.wordy(O, ids[j]):
                continue
            out.setdefault(tuple(ks[j:j + n]), None)
    return list(out)


def ask(ns, owner, text, cap, past, zig_order=False, exclude=()):
    """(retrieve()'s result, what pack saw). With zig_order, the question's n-grams are summed in Zig's order."""
    ns.seen = None
    if zig_order:
        ns.grams = types.MethodType(zig_order_grams, ns)
    try:
        r = ns.retrieve(owner, text, cap=cap, past=past, exclude=exclude)
    finally:
        ns.__dict__.pop("grams", None)
    return r, ns.seen


def view(ns, O, chosen):
    """(ids, block) of a chosen list, as _lookup orders the ids and render() writes the block."""
    if not chosen:
        return [], ""
    cur = sorted((i for i in chosen if O.notes[i]["state"] == nm.CURRENT), key=lambda i: O.notes[i]["told"])
    hist = sorted((i for i in chosen if O.notes[i]["state"] != nm.CURRENT), key=lambda i: O.notes[i]["told"])
    return cur + hist, ns.render(O, chosen)


def pack_sum(ns, O, chosen):
    """pack()'s own estimate of a chosen list's tokens: what Zig reports."""
    if not chosen:
        return 0
    fx, used, titles = ns.fixed(), ns.fixed()["head"], set()
    for i in chosen:
        sect = nm.CURRENT if O.notes[i]["state"] == nm.CURRENT else "history"
        used += ns.cost(O, ns.line(O, i)) + fx["nl"] + (0 if sect in titles else fx[sect] + fx["nl"])
        titles.add(sect)
    return used


def zig_bin(path=None):
    p = path or ZIG_BIN
    if not os.path.isfile(p):
        raise nm.NotesError(f"no Zig CLI at {p}: build it with (cd zig/notes && zig build -Doptimize=ReleaseFast)")
    return p


def run_zig(binp, snap_path, q_path, brk=None):
    """The Zig CLI's results for a query file, one dict per line."""
    cmd = [binp, "lookup", snap_path, q_path] + (["--break", brk] if brk else [])
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode:
        raise nm.NotesError(f"{' '.join(cmd)} failed ({p.returncode}): {p.stderr.decode(errors='replace')[:500]}")
    out = [json.loads(x) for x in p.stdout.decode().splitlines()]
    for r in out:   # the scores come as their 64 bits in hex, so no decimal round trip can blur them
        r["scores"] = [struct.unpack("<d", int(h, 16).to_bytes(8, "little"))[0] for h in r["scores"]]
    return out


def grow(S, tok, sizes, store=None, owner="u0", seed=1, config=None):
    """notes.sweep's stores: the set's teach notes for `owner`, then its distractors (notes.distractors, the same
    pool and order) until the store holds each size in turn. Yields (size, store, seconds to grow it)."""
    teach = next(p for p in S["phases"] if p["name"] == "teach")
    ns = store or nm.NoteStore(tok, schema=S["schema"], config=config)
    nm.set_store(S, teach["store"], store=ns, owner=owner)
    base = ns.stats(owner)["notes"]
    pool = nm.distractors(int((max(sizes) - base) * 1.6) + 100, seed=seed)
    k = 0
    for size in sorted(sizes):
        t0 = time.perf_counter()
        while ns.stats(owner)["notes"] < size and k < len(pool):
            ns.write(owner, pool[k])
            k += 1
        yield size, ns, time.perf_counter() - t0


class Tally:
    """Counts per row (tokenizer, kind, section, set), the first few runs of each kind that differ, and per phase
    how many retrieved notes were tried one at a time as exclude."""

    def __init__(self):
        self.rows = collections.OrderedDict()
        self.examples = collections.defaultdict(list)
        self.capped = []

    def row(self, key):
        return self.rows.setdefault(key, collections.Counter())


def spread(xs, k):
    """At most k of the sorted xs, evenly spaced, first and last included (deterministic)."""
    xs = sorted(xs)
    if len(xs) <= k:
        return xs
    return [xs[round(j * (len(xs) - 1) / (k - 1))] for j in range(k)] if k > 1 else xs[:1]


def run_jobs(ns, owner, v, sp, jobs, binp, tmp, tally, meta):
    """Each job (tally key, text, cap, mode, past flag, exclude) asked of Python's retrieve() and of the Zig CLI over
    the snapshot at sp, and of the Zig CLI with each break. Adds to the tally; returns Python's results, by job."""
    O = ns.owner(owner)
    qp = os.path.join(tmp, "q.txt")
    with open(qp, "w") as f:
        for _key, t, cap, _mode, flag, ex in jobs:
            f.write(nx.query_line(v, t, flag, cap, ex) + "\n")
    zs = {None: run_zig(binp, sp, qp)}
    for b in BREAKS:
        zs[b] = run_zig(binp, sp, qp, b)
    for b, z in zs.items():     # one result per query, or the comparisons below would pair the wrong ones
        if len(z) != len(jobs):
            raise nm.NotesError(f"notes-zig{' --break ' + b if b else ''} gave {len(z)} results for {len(jobs)} queries")
    out = []
    for key in dict.fromkeys(j[0] for j in jobs):
        tally.row(key)["snapshots"] += 1
    for k, (key, t, cap, mode, flag, ex) in enumerate(jobs):
        c = tally.row(key)
        r, seen = ask(ns, owner, t, cap, None if mode == "own" else flag, exclude=ex)
        if r["past"] != flag:
            raise nm.NotesError(f"retrieve() read {t!r} as past={r['past']}, the query said {flag}")
        out.append(r)
        r2, seen2 = ask(ns, owner, t, cap, flag, zig_order=True, exclude=ex)
        py = (r["ids"], r["block"])
        packed = seen["packed"] if seen else []
        pview = view(ns, O, packed)
        fit_moved = py != pview
        c["runs"] += 1
        c["fit_moved_python"] += fit_moved
        c[f"mode_{mode}"] += 1
        if mode == "own" and flag:
            c["own_past"] += 1
        c["float_moved_python"] += py != (r2["ids"], r2["block"])
        c["excluded_ids"] += len(ex)
        c["excluded_given_by_zig"] += bool(set(ex) & set(zs[None][k]["ids"]))
        for b, z in zs.items():
            zr = z[k]
            zv = (zr["ids"], zr["block"])
            if zv == py:
                kind = "equal"
            elif fit_moved and zv == pview:
                kind = "fit"
            elif zv in ((r2["ids"], r2["block"]), view(ns, O, seen2["packed"] if seen2 else [])):
                kind = "float"
            else:
                kind = "other"
            pre = "" if b is None else f"break_{b}_"
            c[pre + kind] += 1
            if kind != "equal" and b is None and len(tally.examples[(key, kind)]) < 4:   # the broken runs: counts only
                tally.examples[(key, pre + kind)].append({
                    "meta": meta, "question": t, "mode": mode, "past": flag, "cap": cap, "exclude": list(ex),
                    "python": r["ids"], "python_packed": view(ns, O, packed)[0], "zig": zr["ids"],
                    "python_tokens": r["tokens"], "zig_tokens": zr["tokens"], "python_candidates": r["candidates"],
                    "zig_candidates": zr["candidates"]})
            # the gated checks beside ids and block (see GATES), for every variant, so the controls can show they fail
            c[pre + "candidates_same"] += zr["candidates"] == r["candidates"]
            if kind in ("equal", "fit"):
                c[pre + "tokens_pack_sum"] += zr["tokens"] == pack_sum(ns, O, packed)
            if kind == "equal":
                d = zr["tokens"] - r["tokens"]
                c[pre + "tokens_exact"] += d == 0
                c[pre + "tokens_over"] += d > 0
                c[pre + "tokens_under"] += d < 0
                c[pre + "tokens_max_over"] = max(c[pre + "tokens_max_over"], d)
                c[pre + "tokens_max_under"] = max(c[pre + "tokens_max_under"], -d)
                # scores: bit for bit against Python summing in Zig's order, and against its own order
                s2 = seen2["score"] if seen2 else {}
                s1 = seen["score"] if seen else {}
                # a score for every note, then each bit for bit (zip alone would pass a short list)
                same_len = len(zr["scores"]) == len(zr["ids"])
                c[pre + "scores_length_wrong"] += not same_len
                c[pre + "scores_bitwise_zig_order"] += same_len and all(
                    s2.get(i) == s for i, s in zip(zr["ids"], zr["scores"]))
                c[pre + "scores_bitwise_set_order"] += same_len and all(
                    s1.get(i) == s for i, s in zip(zr["ids"], zr["scores"]))
    return out


def compare(ns, owner, texts, caps, binp, tmp, tally, key, meta, exclude=None):
    """Every text asked of the store and of its snapshot, at every cap and mode. With `exclude` (the most single
    ids per phase, the caps for them), then again with exclude = one current note that some question retrieved (each
    in turn, at most that many per phase, evenly spread: the rest are logged as capped), and with exclude = all the
    notes the question itself retrieved. Rows of the exclude runs: the key's kind + "-x1" and + "-xall"."""
    snap, v = nx.snapshot(ns, owner, asks=texts, meta=meta)
    sp = os.path.join(tmp, "snap.json")
    nx.write(snap, sp)
    jobs = [(key, t, cap, mode, {"now": False, "past": True, "own": nm.is_past(t)}[mode], ())
            for t in texts for cap in caps for mode in MODES]
    got = run_jobs(ns, owner, v, sp, jobs, binp, tmp, tally, meta)
    if len(got) != len(jobs):
        raise nm.NotesError(f"{len(got)} Python results for {len(jobs)} queries")
    if exclude is None:
        return
    most, one_caps = exclude
    k1, kall = (key[0], key[1] + "-x1") + key[2:], (key[0], key[1] + "-xall") + key[2:]
    retrieved = sorted({i for r in got for i in r["current"]})
    tried = spread(retrieved, most)
    tally.capped.append({"key": list(key), "meta": meta, "retrieved": len(retrieved), "tried": len(tried)})
    xjobs = [(k1, t, cap, mode, {"now": False, "past": True, "own": nm.is_past(t)}[mode], (e,))
             for e in tried for t in texts for cap in one_caps for mode in MODES]
    xjobs += [(kall, t, cap, mode, flag, tuple(r["ids"])) for (_k, t, cap, mode, flag, _x), r in zip(jobs, got)
              if r["ids"]]
    tally.row(kall)["nothing_to_exclude"] += sum(1 for r in got if not r["ids"])
    run_jobs(ns, owner, v, sp, xjobs, binp, tmp, tally, meta)


def table(tally, title, kinds, per_set):
    """Rows of one table: per set (the example sets, defaults), or per section (summed over its sets)."""
    rows = collections.OrderedDict()
    for (tk, kind, label, name), c in tally.rows.items():
        if kind in kinds:
            lab = name if per_set else label + {"-x1": ", exclude one", "-xall": ", exclude all retrieved"}.get(
                kind[kind.find("-"):] if "-" in kind else "", "")
            rows.setdefault((tk, lab), collections.Counter()).update(c)
    w = 24 if per_set else 60
    out = [title, f"{'tokenizer':9} {'set' if per_set else 'section':{w}} {'snapshots':>9} {'compared':>8} {'equal':>7} "
                  f"{'fit':>4} {'float':>5} {'other':>5} | {'no subject':>10} {'no hop':>7} {'excl. hop':>9}"]
    for (tk, lab), c in rows.items():
        br = [c[f"break_{b}_other"] for b in LOOKUP_BREAKS]
        out.append(f"{tk:9} {lab[:w]:{w}} {c['snapshots']:>9} {c['runs']:>8} {c['equal']:>7} {c['fit']:>4} "
                   f"{c['float']:>5} {c['other']:>5} | {br[0]:>10} {br[1]:>7} {br[2]:>9}")
    return "\n".join(out)


def xtable(tally, title):
    """The exclude runs on the example sets, per set: one retrieved note excluded at a time, and every note the
    question retrieved excluded."""
    rows = collections.OrderedDict()
    for (tk, kind, _label, name), c in tally.rows.items():
        if kind in ("examples-x1", "examples-xall"):
            rows.setdefault((tk, name), {}).setdefault(kind, collections.Counter()).update(c)
    tried = collections.Counter()
    for x in tally.capped:
        if x["key"][1] == "examples":
            tried[(x["key"][0], x["key"][3])] += x["tried"]
    out = [title, f"{'tokenizer':9} {'set':24} {'ids':>5} {'one: runs':>9} {'equal':>7} {'float':>5} {'other':>5} "
                  f"{'all: runs':>9} {'equal':>7} {'float':>5} {'other':>5} | {'excl. hop: one':>14} {'all':>6}"]
    for (tk, name), d in rows.items():
        a, b = d.get("examples-x1", collections.Counter()), d.get("examples-xall", collections.Counter())
        out.append(f"{tk:9} {name[:24]:24} {tried[(tk, name)]:>5} {a['runs']:>9} {a['equal']:>7} {a['float']:>5} "
                   f"{a['other']:>5} {b['runs']:>9} {b['equal']:>7} {b['float']:>5} {b['other']:>5} | "
                   f"{a['break_exclude-hop_other']:>14} {b['break_exclude-hop_other']:>6}")
    return "\n".join(out)


GATES = (("candidates_same", "runs", "candidates equal to Python's in every run"),
         ("scores_bitwise_zig_order", "equal",
          "a score for every note, each bit for bit Python's (summed in Zig's order), in every equal run"),
         ("tokens_pack_sum", "equal+fit", "tokens equal to pack()'s sum in every run that is equal or explained by fit"))


def gate_failures(t, pre=""):
    """The gated counters that do not hold in totals t, for the variant with counter prefix pre (unbroken: "")."""
    want = {"runs": t["runs"], "equal": t[pre + "equal"], "equal+fit": t[pre + "equal"] + t[pre + "fit"]}
    return [name for name, of, _what in GATES if t[pre + name] != want[of]]


def totals(tally, pred=lambda k: True):
    t = collections.Counter()
    for k, c in tally.rows.items():
        if pred(k):
            t.update(c)
    return t


class TightFit(Probe):
    """Control for the "fit" category: fit() counts the whole block against the cap less FIT_SQUEEZE tokens, as a
    tokenizer that counts a block above the sum of its lines would, so Python drops notes that Zig keeps."""

    def fit(self, O, chosen, cap):
        return super().fit(O, chosen, cap - FIT_SQUEEZE)


FIT_SQUEEZE = 16


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", action="append", help="fake (default) and/or a tokenizer.json; repeatable")
    ap.add_argument("--caps", default="512,128")
    ap.add_argument("--sizes", default="1000,10000", help="grown stores for L6-seed1 and L7-seed1 ('' for none)")
    ap.add_argument("--grid", action="store_true", help="also every setting of notes.ABLATIONS on the example sets")
    ap.add_argument("--sets", help="comma-separated subset of the example sets")
    ap.add_argument("--exclude-most", type=int, default=24,
                    help="most retrieved notes tried one at a time as exclude, per phase (0: no exclude runs)")
    ap.add_argument("--exclude-caps", default="512", help="caps for the one-note exclude runs")
    ap.add_argument("--zig-bin")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    try:
        binp = zig_bin(a.zig_bin)
        caps = [int(x) for x in a.caps.split(",") if x]
        sizes = [int(x) for x in a.sizes.split(",") if x]
        names = a.sets.split(",") if a.sets else SETS
        xplan = (a.exclude_most, [int(x) for x in a.exclude_caps.split(",") if x]) if a.exclude_most > 0 else None
        tally = Tally()
        t_all = time.time()
        base = os.path.join(L.ROOT, "zig-out")
        with tempfile.TemporaryDirectory(dir=base if os.path.isdir(base) else None) as tmp:
            for spec in a.tokenizer or ["fake"]:
                tok = nm.load_tokenizer(spec)
                tk = "fake" if spec == "fake" else "nemotron"
                grid = [("defaults", {})] + (GRID if a.grid else [])
                for gname, cfg in grid:
                    kind = "examples" if gname == "defaults" else "grid"
                    for name in names:
                        S = nm.load_set(os.path.join(L.ROOT, "examples", f"{name}.json"))
                        texts = [q["text"] for q in S["questions"]]
                        for ph in nm.recall_phases(S, "all"):
                            pstore = next(p["store"] for p in S["phases"] if p["name"] == ph)
                            ns = Probe(tok, schema=S["schema"], config=cfg)
                            nm.set_store(S, pstore, store=ns)
                            compare(ns, "user", texts, caps if kind == "examples" else caps[:1], binp, tmp, tally,
                                    (tk, kind, f"setting: {gname}", name), {"set": name, "phase": ph, "config": cfg},
                                    exclude=xplan if kind == "examples" else None)
                    print(f"  {tk} {gname}: {len(names)} sets done ({time.time() - t_all:.0f} s)", flush=True)
                for name in ("L6-seed1", "L7-seed1") if sizes else ():
                    S = nm.load_set(os.path.join(L.ROOT, "examples", f"{name}.json"))
                    texts = [q["text"] for q in S["questions"]]
                    for size, ns, _s in grow(S, tok, sizes, store=Probe(tok, schema=S["schema"])):
                        compare(ns, "u0", texts, caps, binp, tmp, tally,
                                (tk, "grown", f"grown store: {name}, {size:,} notes", name), {"set": name, "size": size},
                                exclude=xplan)
                        print(f"  {tk} grown {name} {size}: done ({time.time() - t_all:.0f} s)", flush=True)
                for name in ("L6-seed1", "L7-seed1"):     # the fit category's own control
                    S = nm.load_set(os.path.join(L.ROOT, "examples", f"{name}.json"))
                    ns = TightFit(tok, schema=S["schema"])
                    nm.set_store(S, next(p["store"] for p in S["phases"] if p["name"] == "teach"), store=ns)
                    compare(ns, "user", [q["text"] for q in S["questions"]], caps, binp, tmp, tally,
                            (tk, "fitctl", f"control: fit() against cap - {FIT_SQUEEZE}", name),
                            {"set": name, "phase": "teach", "fit_squeeze": FIT_SQUEEZE})
        print()
        print(table(tally, f"The example sets, default settings: every phase that keeps lessons, every question at "
                           f"caps {a.caps}, each now / own / past", ("examples",), True))
        if xplan:
            print()
            print(xtable(tally, f"The same, with exclude: 'one' asks every question with one retrieved current note "
                                f"excluded (at most {xplan[0]} notes per phase, evenly spread; caps {a.exclude_caps}; "
                                f"now / own / past); 'all' asks each question with every note it retrieved excluded "
                                f"(caps {a.caps}, now / own / past). ids: notes tried one at a time, all phases"))
        print()
        print(table(tally, "Other settings, grown stores (with their exclude runs), and the fit control (summed over "
                           "their sets)", ("grid", "grown", "grown-x1", "grown-xall", "fitctl"), False))
        print("\nLast columns (negative controls): runs where a Zig lookup broken on purpose differs from Python other "
              "than by fit or float order: no subject boost, no hop, a hop that brings in excluded notes.")
        capped = [x for x in tally.capped if x["retrieved"] > x["tried"]]
        if xplan:
            print(f"\nexclude one: {sum(x['tried'] for x in tally.capped)} notes tried of "
                  f"{sum(x['retrieved'] for x in tally.capped)} retrieved, over {len(tally.capped)} phases and stores; "
                  f"capped at {xplan[0]} in {len(capped)} of them: "
                  + ", ".join(f"{'/'.join(str(v) for v in x['meta'].values() if not isinstance(v, dict))} "
                              f"({x['tried']} of {x['retrieved']})" for x in capped[:40])
                  + (" ..." if len(capped) > 40 else ""))
        ok = True
        for tk in dict.fromkeys(k[0] for k in tally.rows):
            for part, pred, need in (
                    ("without exclude", lambda k, tk=tk: k[0] == tk and k[1] in ("examples", "grid", "grown"),
                     ("subject", "hop")),
                    ("with exclude", lambda k, tk=tk: k[0] == tk and k[1].endswith(("-x1", "-xall")),
                     ("exclude-hop",))):
                t = totals(tally, pred)
                if not t["runs"]:
                    continue
                eq = t["equal"]
                rows_k = [c for k, c in tally.rows.items() if pred(k)]
                print(f"\n{tk}, {part}: {t['runs']} runs: {eq} equal, {t['fit']} differ because of fit, {t['float']} "
                      f"because of float order, {t['other']} otherwise; own flag past in {t['own_past']} of "
                      f"{t['mode_own']} own runs")
                print(f"  Python's fit() changed its own result in {t['fit_moved_python']} runs (Zig equals Python's "
                      f"pack() in {t['fit']} of them); Python's result changes with the order it sums in, in "
                      f"{t['float_moved_python']} runs")
                print(f"  candidates equal in {t['candidates_same']} of {t['runs']}; of the {eq} equal runs, Zig's "
                      f"scores are bit for bit Python's summing in Zig's order in {t['scores_bitwise_zig_order']} (in "
                      f"Python's own order in {t['scores_bitwise_set_order']}); of the {eq + t['fit']} equal or fit "
                      f"runs, Zig's tokens equal pack()'s sum in {t['tokens_pack_sum']}")
                print(f"  Zig's tokens (pack()'s sum) against the tokenizer's count of the whole block: equal in "
                      f"{t['tokens_exact']}, above it in {t['tokens_over']} (by up to "
                      f"{max(c['tokens_max_over'] for c in rows_k)}), below it in {t['tokens_under']} (by up to "
                      f"{max(c['tokens_max_under'] for c in rows_k)})")
                if part == "with exclude":
                    print(f"  {t['excluded_ids']} note ids excluded in all; runs where Zig gave an excluded note: "
                          f"{t['excluded_given_by_zig']}")
                    ok &= t["excluded_given_by_zig"] == 0
                for b in LOOKUP_BREAKS:
                    d = t["runs"] - t[f"break_{b}_equal"]
                    print(f"  negative control, --break {b}: {d} of {t['runs']} runs differ "
                          f"({t[f'break_{b}_other']} not explained by fit or float order)")
                    if b in need:
                        ok &= t[f"break_{b}_other"] > 0
                ok &= t["other"] == 0
            # the gates, over every run of this tokenizer (fit control included), and their controls
            t = totals(tally, lambda k, tk=tk: k[0] == tk)
            fails = gate_failures(t)
            want = {"runs": t["runs"], "equal": t["equal"], "equal+fit": t["equal"] + t["fit"]}
            print(f"\n{tk}, gates over all {t['runs']} runs (the test fails unless each holds):")
            for name, of, what in GATES:
                print(f"  {name}: {t[name]} of {want[of]} ({of}): {what}: {'holds' if name not in fails else 'FAILS'}"
                      + (f" (score lists of the wrong length: {t['scores_length_wrong']})"
                         if name == "scores_bitwise_zig_order" else ""))
            ok &= not fails
            for b, target in REPORT_BREAKS.items():
                pre = f"break_{b}_"
                f = gate_failures(t, pre)
                wb = {"runs": t["runs"], "equal": t[pre + "equal"], "equal+fit": t[pre + "equal"] + t[pre + "fit"]}
                of = next(o for n, o, _w in GATES if n == target)
                print(f"  control --break {b}: ids and block as Python's in {t[pre + 'equal']} runs (unbroken: "
                      f"{t['equal']}); {target} {t[pre + target]} of {wb[of]}: the gate "
                      f"{'fails, as it must' if target in f else 'HOLDS: the control is not caught'}; gates failing: "
                      f"{', '.join(f) or 'none'}; score lists of the wrong length: {t[pre + 'scores_length_wrong']}")
                ok &= target in f
            fc = totals(tally, lambda k, tk=tk: k[0] == tk and k[1] == "fitctl")
            print(f"  fit control (Python's fit() against cap - {FIT_SQUEEZE}): Python's fit() dropped notes in "
                  f"{fc['fit_moved_python']} of {fc['runs']} runs; {fc['fit']} of them sorted as fit, {fc['other']} as "
                  f"other")
            ok &= fc["fit"] > 0 and fc["other"] == 0
        bad = [(k, x) for k, xs in tally.examples.items() if k[0][1] != "fitctl" for x in xs]
        if bad:
            print("\nfirst differing runs (unbroken Zig):")
            for (key, kind), x in bad[:12]:
                print(f"  {kind}: {key}: {json.dumps(x)}")
        print(f"\nnotes_zig_golden {'PASSED' if ok else 'FAILED'} in {time.time() - t_all:.0f} s (PYTHONHASHSEED "
              f"{os.environ.get('PYTHONHASHSEED', 'random')}: it sets the order Python walks a set of n-grams in)")
        if a.json:
            with open(a.json, "w") as f:
                json.dump({"format": "lw-notes-zig-golden/1", "caps": caps, "sizes": sizes, "grid": a.grid,
                           "pythonhashseed": os.environ.get("PYTHONHASHSEED"), "exclude": xplan,
                           "exclude_one_tried": tally.capped,
                           "rows": [{"tokenizer": k[0], "kind": k[1], "section": k[2], "set": k[3], **c}
                                    for k, c in tally.rows.items()],
                           "examples": [{"key": list(k[0]), "kind": k[1], "runs": v}
                                        for k, v in tally.examples.items()]}, f, indent=1)
        return 0 if ok else 1
    except nm.NotesError as e:
        print(f"notes_zig_golden: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
