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


def run(size, journal_root=None, engine="zig", bulk=False):
    from notes_proxy.native_notes import NativeNotes
    from notes_proxy.store import nm, Notes, Journal, JOURNAL_FORMAT, FREE
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
            notes.write_text("An additional synthetic inventory item is blue.", subject="inventory")
            emit(engine=engine, size=size, phase="single_write", seconds=time.monotonic() - started)
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
    args = parser.parse_args()
    if args.size:
        run(args.size, args.journal_root, args.engine, args.bulk)
    else:
        for size in (1000, 10000, 100000):
            try:
                # The parent owns cleanup, including when the worker is killed at the wall cap.
                with tempfile.TemporaryDirectory(prefix="native-notes-bench-parent-") as root:
                    result = subprocess.run([sys.executable, __file__, "--size", str(size), "--journal-root", root,
                                             "--engine", args.engine] + (["--bulk"] if args.bulk else []),
                                            capture_output=True, text=True, timeout=args.timeout)
                print(result.stdout, end="", flush=True)
                emit(size=size, result="completed" if result.returncode == 0 else "failed",
                     returncode=result.returncode, error=result.stderr[-2000:])
            except subprocess.TimeoutExpired as error:
                output = error.stdout or b""
                print(output.decode() if isinstance(output, bytes) else output, end="", flush=True)
                emit(size=size, result="wall_limit", limit_seconds=args.timeout,
                     qualification="NOT a completed scale/erase measurement")
