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


def run(size, journal_root=None):
    from notes_proxy.native_notes import NativeNotes
    from notes_proxy.store import nm
    context = contextlib.nullcontext(journal_root) if journal_root else tempfile.TemporaryDirectory(prefix="native-notes-bench-")
    with context as root:
        tok = nm.load_tokenizer(os.environ["NOTES_PROXY_TEST_TOKENIZER"])
        notes = NativeNotes(pathlib.Path(root) / "bench.journal", "bench", tok)
        try:
            started = time.monotonic()
            for i in range(size):
                notes.write_text(f"Inventory item {i} is stored on shelf {i % 97}.", subject="inventory")
                if (i + 1) % 100 == 0:
                    emit(size=size, phase="fill_progress", written=i + 1, elapsed=time.monotonic() - started)
            emit(size=size, phase="fill", seconds=time.monotonic() - started)
            started = time.monotonic()
            notes.lookup("Where is inventory item 42 stored?", cap=512)
            emit(size=size, phase="lookup", seconds=time.monotonic() - started)
            started = time.monotonic()
            notes.erase("w1")
            emit(size=size, phase="erase", seconds=time.monotonic() - started,
                 maxrss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                 maxrss_unit="bytes" if sys.platform == "darwin" else "KiB")
        finally:
            notes.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, choices=(1000, 10000, 100000))
    parser.add_argument("--timeout", type=float, default=45)
    parser.add_argument("--journal-root", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.size:
        run(args.size, args.journal_root)
    else:
        for size in (1000, 10000, 100000):
            try:
                # The parent owns cleanup, including when the worker is killed at the wall cap.
                with tempfile.TemporaryDirectory(prefix="native-notes-bench-parent-") as root:
                    result = subprocess.run([sys.executable, __file__, "--size", str(size), "--journal-root", root],
                                            capture_output=True, text=True, timeout=args.timeout)
                print(result.stdout, end="", flush=True)
                emit(size=size, result="completed" if result.returncode == 0 else "failed",
                     returncode=result.returncode, error=result.stderr[-2000:])
            except subprocess.TimeoutExpired as error:
                output = error.stdout or b""
                print(output.decode() if isinstance(output, bytes) else output, end="", flush=True)
                emit(size=size, result="wall_limit", limit_seconds=args.timeout,
                     qualification="NOT a completed scale/erase measurement")
