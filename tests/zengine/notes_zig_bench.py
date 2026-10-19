#!/usr/bin/env python3
"""notes_zig_bench: lookup times of the Zig port (zig/notes) and of Python's NoteStore, on the same stores and the
same questions. docs/NOTES-ZIG.md.

The stores are notes.py's sweep stores (notes_zig_golden.grow: a set's teach notes, then notes.distractors worded
from the same templates, for owner u0) at each size of --sizes, for L6-seed1 and L7-seed1. At each size:

  zig      the store exported (notes_export.py), then `notes-zig bench`: the snapshot's load time and the index's
           memory apart, then every question (past as retrieve() decides it, the default cap) asked --reps times by
           1 thread and by 8 threads at once over the one read-only index, each thread in its own shuffled order;
           p50 and p95 per lookup and lookups per second. Every timed lookup is checked against a first pass.
  python   notes.timed_lookups, as notes.py's sweep times it: retrieve() from the question's text, by 1 and by 8
           threads (one interpreter lock). And "same work": what Zig does, from the token ids with fit() left out
           (the tokenizer is not called), 1 thread.
  check    Zig's ids and blocks against Python's retrieve() for every question (as notes_zig_golden does).

  python3 zengine/notes_zig_bench.py [--tokenizer T] [--sizes 100,1000,10000,100000] [--threads 1,8] [--reps 50]
      [--py-reps 3] [--json OUT]
  python3 zengine/notes_zig_bench.py --failure-controls     the thread start under injected faults: an error, no hang

Build the Zig CLI with -Doptimize=ReleaseFast first. Standard library only (the tokenizer as notes.py loads it).
"""
import argparse
import gc
import json
import os
import platform
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lwlogic as L  # noqa: E402
import notes as nm  # noqa: E402
import notes_export as nx  # noqa: E402
import notes_zig_golden as G  # noqa: E402

SETS = ("L6-seed1", "L7-seed1")


def sysctl(name):
    try:
        return subprocess.run(["sysctl", "-n", name], capture_output=True, text=True).stdout.strip()
    except OSError:
        return None


def machine(binp):
    """The machine and the toolchains, as stated above every table."""
    z = subprocess.run([os.environ.get("ZIG", "zig"), "version"], capture_output=True, text=True) if os.environ.get(
        "ZIG") else None
    return {"cpu": sysctl("machdep.cpu.brand_string"), "ncpu": sysctl("hw.ncpu"), "memsize": sysctl("hw.memsize"),
            "perflevels": [(sysctl(f"hw.perflevel{k}.name"), sysctl(f"hw.perflevel{k}.physicalcpu")) for k in (0, 1)],
            "os": platform.platform(), "python": sys.version.split()[0], "zig": z.stdout.strip() if z else None,
            "zig_bin": binp}


def same_work(ns, owner, qids, cap):
    """Python's lookup from token ids, as Zig does it: keys, n-grams, _lookup, pack and render, with fit() left out
    (no tokenizer call). Returns microseconds per question, in the given order."""
    O = ns.owner(owner)
    ns.fit = lambda O_, chosen, cap_: (list(chosen), ns.render(O_, chosen) if chosen else "", 0)
    out = []
    try:
        for ids, past in qids:
            t0 = time.perf_counter_ns()
            res = {}
            with O.lock.read():
                ns._lookup(O, ns.grams(O, ids), cap, past, res, [ns.key(O, i) for i in ids], frozenset())
            out.append((time.perf_counter_ns() - t0) / 1000)
    finally:
        del ns.fit
    return out


def tables(rows, info, tok, a):
    """The Zig table and the Python table, each under the machine and toolchain it was measured with."""
    if not rows:
        return ""
    z0 = rows[0]["zig"]
    mach = (f"{info['cpu']}, {info['ncpu']} cores ({', '.join(f'{n} {k}' for k, n in info['perflevels'])}), "
            f"{int(info['memsize'] or 0) / 2**30:.0f} GB; tokenizer {tok.describe()['name']}")
    th = [str(t) for t in a.threads.split(",")]
    out = [f"Zig {z0['zig']} ({z0['mode']}) on {mach}; each thread asks every question {a.reps} times in its own order",
           f"{'set':9} {'notes':>8} {'current':>8} {'snapshot':>9} {'load':>8} {'index':>8} {'peak RSS':>9} "
           + " ".join(f"{t + ' thr p50/p95 us':>22} {'lookups/s':>10}" for t in th) + f" {'same as Python':>15}"]
    for r in rows:
        zr = {str(x["threads"]): x for x in r["zig"]["runs"]}
        out.append(f"{r['set']:9} {r['notes']:>8,} {r['current']:>8,} {r['snapshot_mb']:>7} MB "
                   f"{r['zig']['load_ms']:>5.0f} ms {r['zig']['index_used_bytes'] / 1e6:>5.1f} MB "
                   f"{r['zig']['rss_peak'] / 1e6:>6.0f} MB "
                   + " ".join(f"{str(zr[t]['p50_us']) + ' / ' + str(zr[t]['p95_us']):>22} {zr[t]['per_s']:>10,.0f}"
                              for t in th)
                   + f" {str(r['zig_same_as_python'][0]) + '/' + str(r['zig_same_as_python'][1]):>15}"
                   + ("" if r.get("timing_claimed", True) else "  (float-order tie: timing not claimed)"))
    out += ["", f"Python {info['python']} on the same machine and stores; retrieve() as notes.py's sweep times it "
                f"({a.py_reps} times per thread), and the same work as Zig (from token ids, no fit), 1 thread",
            f"{'set':9} {'notes':>8} " + " ".join(f"{t + ' thr p50/p95 us':>22} {'lookups/s':>10}" for t in th)
            + f" {'same work p50/p95 us':>22} {'index (objects)':>16}"]
    for r in rows:
        p = r["python"]
        out.append(f"{r['set']:9} {r['notes']:>8,} "
                   + " ".join(f"{str(p[t]['p50_us']) + ' / ' + str(p[t]['p95_us']):>22} {p[t]['per_s']:>10,}"
                              for t in th)
                   + f" {str(p['same_work_1']['p50_us']) + ' / ' + str(p['same_work_1']['p95_us']):>22} "
                     f"{r['python_index_mb']:>13} MB")
    return "\n".join(out)


def check(ns, owner, texts, sp, qp, binp, brk=None):
    """Zig's ids and block against retrieve() on every question (its own past flag, the default cap). A question
    that differs is a float-order tie only by the golden test's rule: Python, summing its n-gram weights in Zig's
    order, gives Zig's answer. brk: a broken Zig lookup (notes-zig --break), for the control. Returns (questions
    that agree, the ties, the others)."""
    zs = G.run_zig(binp, sp, qp, brk)
    if len(zs) != len(texts):
        raise nm.NotesError(f"notes-zig gave {len(zs)} results for {len(texts)} questions")
    same, ties, wrong = 0, [], []
    for t, z in zip(texts, zs):
        r = ns.retrieve(owner, t)
        if (z["ids"], z["block"]) == (r["ids"], r["block"]):
            same += 1
            continue
        r2, _seen = G.ask(ns, owner, t, ns.cfg["cap"], nm.is_past(t), zig_order=True)
        (ties if (z["ids"], z["block"]) == (r2["ids"], r2["block"]) else wrong).append(t)
    return same, ties, wrong


def failure_controls(binp, out_dir, threads=8, limit_s=30.0, hang_s=5.0):
    """The bench's thread start under injected faults (notes-zig bench --fail-alloc K, --fail-spawn N): each must stop
    with an error within limit_s, not hang. Control: the old start (--legacy-barrier) with a failed setup must hang,
    caught by the hang_s limit, so the check can fail. On L6-seed1's teach notes, stand-in tokenizer. Returns
    (rows, all as expected)."""
    S = nm.load_set(os.path.join(L.ROOT, "examples", "L6-seed1.json"))
    ns, _w = nm.set_store(S, next(p["store"] for p in S["phases"] if p["name"] == "teach"), nm.FakeTokenizer())
    texts = [q["text"] for q in S["questions"]]
    snap, v = nx.snapshot(ns, "user", asks=texts)
    sp, qp = os.path.join(out_dir, "faults.json"), os.path.join(out_dir, "faults.q")
    nx.write(snap, sp)
    with open(qp, "w") as f:
        for t in texts:
            f.write(nx.query_line(v, t, nm.is_past(t), nm.DEFAULTS["cap"]) + "\n")
    cases = [("no fault", [], "ok")]
    cases += [(f"worker {k}'s allocations fail", ["--fail-alloc", str(k)], "error") for k in (0, threads // 2, threads - 1)]
    cases += [(f"starting worker {k} fails", ["--fail-spawn", str(k)], "error") for k in (0, threads // 2, threads - 1)]
    cases += [("one timed result taken as wrong (forced mismatch)", ["--corrupt-one"], "error")]
    cases += [("control: the old start, worker 3's allocations fail", ["--fail-alloc", "3", "--legacy-barrier"], "hang")]
    rows, ok = [], True
    for name, extra, want in cases:
        cmd = [binp, "bench", sp, qp, "--threads", str(threads), "--reps", "5"] + extra
        t0 = time.perf_counter()
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=hang_s if want == "hang" else limit_s)
            got = "ok" if p.returncode == 0 else "error"
            said = (p.stderr.strip().splitlines() or [""])[-1] if p.returncode else "bench JSON"
        except subprocess.TimeoutExpired:
            got, said = "hang", f"still running after {hang_s if want == 'hang' else limit_s:.0f} s, killed"
        rows.append({"case": name, "flags": extra, "want": want, "got": got, "seconds": round(time.perf_counter() - t0, 2),
                     "said": said})
        ok &= got == want
    # the check against Python stops the bench on a result that is not a float-order tie: with a broken lookup
    same, ties, wrong = check(ns, "user", texts, sp, qp, binp, "hop")
    got = "error" if wrong else "ok"
    rows.append({"case": "check against Python, lookup broken (--break hop)", "flags": ["--break", "hop"],
                 "want": "error", "got": got, "seconds": 0,
                 "said": f"{len(wrong)} of {len(texts)} questions differ, not by float order ({len(ties)} ties)"})
    ok &= got == "error"
    return rows, ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", default="fake")
    ap.add_argument("--sizes", default="100,1000,10000,100000")
    ap.add_argument("--threads", default="1,8")
    ap.add_argument("--reps", type=int, default=50, help="Zig: times each thread asks every question")
    ap.add_argument("--py-reps", type=int, default=3, help="Python: times each thread asks every question")
    ap.add_argument("--sets", default=",".join(SETS))
    ap.add_argument("--zig-bin")
    ap.add_argument("--keep", action="store_true", help="keep the snapshots in zig-out/bench")
    ap.add_argument("--corrupt-one", action="store_true",
                    help="control: notes-zig bench takes one timed result as wrong; the run must stop with an error")
    ap.add_argument("--check-break", help="control: check against Python with a broken Zig lookup (notes-zig --break)")
    ap.add_argument("--failure-controls", action="store_true",
                    help="only check that the bench stops with an error under injected thread-start faults")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    try:
        binp = G.zig_bin(a.zig_bin)
        if a.failure_controls:
            out_dir = os.path.join(L.ROOT, "zig-out", "bench")
            os.makedirs(out_dir, exist_ok=True)
            rows, ok = failure_controls(binp, out_dir)
            for r in rows:
                print(f"  {r['case']:52} {' '.join(r['flags']):32} want {r['want']:5} got {r['got']:5} "
                      f"{r['seconds']:>6} s  {r['said']}")
            print(f"failure controls {'PASSED' if ok else 'FAILED'}")
            if a.json:
                with open(a.json, "w") as f:
                    json.dump({"format": "lw-notes-zig-bench/failures", "rows": rows, "ok": ok}, f, indent=1)
            return 0 if ok else 1
        tok = nm.load_tokenizer(a.tokenizer)
        sizes = [int(x) for x in a.sizes.split(",")]
        threads = [int(x) for x in a.threads.split(",")]
        out_dir = os.path.join(L.ROOT, "zig-out", "bench")
        os.makedirs(out_dir, exist_ok=True)
        info = machine(binp)
        print(f"{info['cpu']}, {info['ncpu']} cores {info['perflevels']}, {int(info['memsize'] or 0) / 2**30:.0f} GB; "
              f"Python {info['python']}; tokenizer {tok.describe()['name']}", flush=True)
        rows = []
        for name in a.sets.split(","):
            S = nm.load_set(os.path.join(L.ROOT, "examples", f"{name}.json"))
            texts = [q["text"] for q in S["questions"]]
            cap = nm.DEFAULTS["cap"]
            for size, ns, grow_s in G.grow(S, tok, sizes, store=nm.NoteStore(tok, schema=S["schema"])):
                st = ns.stats("u0")
                t0 = time.perf_counter()
                snap, v = nx.snapshot(ns, "u0", asks=texts, meta={"set": name, "size": size})
                t1 = time.perf_counter()
                sp = os.path.join(out_dir, f"{name}-{size}.json")
                qp = os.path.join(out_dir, f"{name}-{size}.q")
                nx.write(snap, sp)
                t2 = time.perf_counter()
                with open(qp, "w") as f:
                    for t in texts:
                        f.write(nx.query_line(v, t, nm.is_past(t), cap) + "\n")
                # check: Zig against Python's retrieve(), every question; anything but a float-order tie stops the run
                same, ties, wrong = check(ns, "u0", texts, sp, qp, binp, a.check_break)
                if wrong:
                    raise nm.NotesError(f"{name} at {size} notes: Zig differs from retrieve() on {len(wrong)} of "
                                        f"{len(texts)} questions, not by float order: {wrong[:3]}")
                zb = subprocess.run([binp, "bench", sp, qp, "--threads", a.threads, "--reps", str(a.reps)]
                                    + (["--corrupt-one"] if a.corrupt_one else []), capture_output=True, text=True)
                if zb.returncode:
                    raise nm.NotesError(f"notes-zig bench failed ({zb.returncode}): {zb.stderr.strip()[:400]}")
                zig = json.loads(zb.stdout)
                if any(r["differ_from_reference"] for r in zig["runs"]):
                    raise nm.NotesError(f"{name} at {size} notes: timed lookups differ from the first pass")
                py = {}
                for th in threads:
                    gc.collect()
                    us, wall, got = nm.timed_lookups(ns, [("u0", t) for t in texts], th, a.py_reps)
                    py[str(th)] = {"n": len(us), "p50_us": nm.pct(us, 50), "p95_us": nm.pct(us, 95),
                                   "per_s": round(len(us) / wall)}
                qids = [(tok.encode(t), nm.is_past(t)) for t in texts]     # the tokenizer's own ids, encoded once
                sw = sorted(x for _ in range(a.py_reps) for x in same_work(ns, "u0", qids, cap))
                py["same_work_1"] = {"n": len(sw), "p50_us": round(nm.pct(sw, 50), 1),
                                     "p95_us": round(nm.pct(sw, 95), 1)}
                O = ns.owner("u0")
                row = {"set": name, "size": size, "notes": st["notes"], "current": st[nm.CURRENT],
                       "history": st["history"], "grow_s": round(grow_s, 1), "export_s": round(t1 - t0, 2),
                       "write_s": round(t2 - t1, 2), "snapshot_mb": round(os.path.getsize(sp) / 1e6, 1),
                       "questions": len(texts), "zig_same_as_python": [same, len(texts)], "float_ties": ties,
                       "timing_claimed": not ties, "zig": zig, "python": py,
                       "python_index_mb": round(nm.deep_size(O.idx) / 1e6, 1)}
                rows.append(row)
                zr = {str(r["threads"]): r for r in zig["runs"]}
                if ties:
                    print(f"  {name} {size}: Zig and Python differ by a float-order tie on {len(ties)} questions "
                          f"({ties[:2]}); this row's timings are not claimed", flush=True)
                print(f"  {name} {size:>7} notes ({st[nm.CURRENT]} current): same {same}/{len(texts)}; zig load "
                      f"{zig['load_ms']:.0f} ms, index {zig['index_used_bytes'] / 1e6:.1f} MB; zig " + ", ".join(
                          f"{t}t p50 {x['p50_us']} us p95 {x['p95_us']} us {x['per_s']:.0f}/s" for t, x in zr.items())
                      + "; python " + ", ".join(f"{t}: p50 {x['p50_us']} us p95 {x['p95_us']} us"
                                                for t, x in py.items()), flush=True)
                if not a.keep:
                    os.replace(sp, os.path.join(out_dir, "last.json"))
        print()
        print(tables(rows, info, tok, a))
        if a.json:
            with open(a.json, "w") as f:
                json.dump({"format": "lw-notes-zig-bench/1", "machine": info, "tokenizer": tok.describe(),
                           "sizes": sizes, "threads": threads, "reps": a.reps, "py_reps": a.py_reps, "rows": rows},
                          f, indent=1)
        return 0
    except nm.NotesError as e:
        print(f"notes_zig_bench: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
