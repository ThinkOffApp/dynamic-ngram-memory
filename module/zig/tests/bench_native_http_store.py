"""Capped NativeNotes (the HTTP-selected store) benchmark, never a live journal.

Example: NOTES_PROXY_TEST_TOKENIZER=... python3 module/zig/tests/bench_native_http_store.py
Each isolated child has a 45s wall limit. Timeouts are not completed scale results.
Measures the HTTP adapter methods, not socket/JSON overhead or model inference.
"""
import argparse
import contextlib
import json
import os
import pathlib
import resource
import subprocess
import sys
import tempfile
import time

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def emit(**values):
    print(json.dumps(values), flush=True)


def capped_worker(command, timeout, rss_mib):
    """Parent samples this worker's RSS every 200ms; fail closed if ps fails.

    Not an OS hard memory limit: a worker may transiently exceed the threshold
    between samples. This is explicitly reported rather than called MemoryMax.
    """
    started = time.monotonic()
    reason = None
    sampled_peak = 0
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as worker:
        while True:
            try:
                out, err = worker.communicate(timeout=0.2)
                break
            except subprocess.TimeoutExpired:
                if time.monotonic() - started >= timeout:
                    reason = "wall_limit"
                elif worker.poll() is None:
                    sample = subprocess.run(["ps", "-o", "rss=", "-p", str(worker.pid)],
                                            capture_output=True, text=True, timeout=2)
                    if sample.returncode == 0 and sample.stdout.strip().isdigit():
                        rss = int(sample.stdout.strip()) * 1024
                        sampled_peak = max(sampled_peak, rss)
                        if rss > rss_mib * 1024 * 1024:
                            reason = "sampled_rss_limit"
                    elif worker.poll() is None:
                        reason = "rss_monitor_failed"
                if reason:
                    worker.kill()
                    out, err = worker.communicate()
                    break
        print(out, end="", flush=True)
        emit(result=reason or ("completed" if worker.returncode == 0 else "failed"),
             returncode=worker.returncode, error=err[-2000:], sampled_peak_bytes=sampled_peak,
             wall_cap_seconds=timeout, rss_threshold_mib=rss_mib,
             memory_control="200ms sampled watchdog, NOT a hard MemoryMax")


def run(size, journal_root=None, engine="zig", bulk=False):
    from notes_proxy.native_notes import NativeNotes
    from notes_proxy.store import nm, Notes, Journal, JOURNAL_FORMAT, FREE, StoreError
    context = contextlib.nullcontext(journal_root) if journal_root else tempfile.TemporaryDirectory(prefix="native-notes-bench-")
    with context as root:
        tok = nm.load_tokenizer(os.environ["NOTES_PROXY_TEST_TOKENIZER"])
        path = pathlib.Path(root) / "bench.journal"
        factory = NativeNotes if engine == "zig" else Notes
        if bulk:
            started = time.monotonic()
            if engine == "zig":
                sample = factory(path, "bench", tok)
                sample.write_text("Inventory item 0 is stored on shelf 0.", subject="inventory")
                sample.close()
                header = json.loads(path.read_text().splitlines()[0])
                header.update(next_source=size + 1, next_id=size + 1)
                # Synthetic fixture uses the native writer's actual header and documented committed records.
                with path.open("w") as output:
                    output.write(json.dumps(header) + "\n")
                    for i in range(size):
                        source = f"w{i + 1}"
                        record = dict(op="add", source=source, id=i + 1, told=739916000000 + i * 100,
                                      subject="inventory", attribute="note:" + source, target=None,
                                      text=f"Inventory item {i} is stored on shelf {i % 97}.")
                        output.write(json.dumps(record) + "\n" + json.dumps({"commit": source}) + "\n")
            else:
                journal = Journal(path)
                try:
                    journal.rewrite(dict(format=JOURNAL_FORMAT, owner="bench", schema=None, next_source=size + 1),
                        [dict(op="ASSERT", e="inventory", a=f"note:w{i + 1}", id=f"w{i + 1}",
                              text=f"Inventory item {i} is stored on shelf {i % 97}.",
                              told=739916000000 + i * 100, **{FREE: "add"}) for i in range(size)])
                finally:
                    journal.close()
            emit(engine=engine, size=size, phase="bulk_fixture", seconds=time.monotonic() - started)
        started = time.monotonic()
        notes = factory(path, "bench", tok)
        emit(engine=engine, size=size, phase="open", seconds=time.monotonic() - started)
        try:
            if bulk:
                assert notes.count() == size, "bulk fixture did not replay the requested note count"
                assert notes.get("w1") is not None and notes.get(f"w{size}") is not None
            started = time.monotonic()
            for i in range(0 if bulk else size):
                notes.write_text(f"Inventory item {i} is stored on shelf {i % 97}.", subject="inventory")
                if (i + 1) % 100 == 0:
                    emit(size=size, phase="fill_progress", written=i + 1, elapsed=time.monotonic() - started)
            emit(size=size, phase="fill", seconds=time.monotonic() - started)
            started = time.monotonic()
            try:
                notes.write_text("An additional synthetic inventory item is blue.", subject="inventory")
                emit(engine=engine, size=size, phase="single_write", seconds=time.monotonic() - started)
            except StoreError as error:
                if error.status != 507:
                    raise
                emit(engine=engine, size=size, phase="single_write", result="capacity_refused", status=507,
                     seconds=time.monotonic() - started)
            started = time.monotonic()
            notes.lookup("Where is inventory item 42 stored?", cap=512)
            emit(size=size, phase="lookup", seconds=time.monotonic() - started)
            started = time.monotonic()
            notes.erase("w1")
            emit(size=size, phase="erase", seconds=time.monotonic() - started,
                 maxrss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                 maxrss_unit="bytes" if sys.platform == "darwin" else "KiB")
            assert notes.get("w1") is None and notes.get(f"w{size}") is not None
        finally:
            notes.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, choices=(1000, 10000, 100000))
    parser.add_argument("--timeout", type=float, default=45)
    parser.add_argument("--journal-root", help=argparse.SUPPRESS)
    parser.add_argument("--engine", choices=("python", "zig"), default="zig")
    parser.add_argument("--bulk", action="store_true", help="synthetic batched journal fixture, then real reopen/write/erase")
    parser.add_argument("--tiers", type=int, nargs="+", choices=(1000, 10000, 100000), default=[1000, 10000, 100000])
    parser.add_argument("--rss-mib", type=int, default=1024)
    args = parser.parse_args()
    if args.size:
        run(args.size, args.journal_root, args.engine, args.bulk)
    else:
        if args.rss_mib <= 0 or args.timeout <= 0:
            parser.error("positive memory and wall caps are required")
        for size in args.tiers:
            emit(engine=args.engine, size=size, phase="worker_start")
            # The parent owns cleanup, including when the worker is killed at either cap.
            with tempfile.TemporaryDirectory(prefix="native-notes-bench-parent-") as root:
                capped_worker([sys.executable, __file__, "--size", str(size), "--journal-root", root,
                               "--engine", args.engine] + (["--bulk"] if args.bulk else []),
                              args.timeout, args.rss_mib)
