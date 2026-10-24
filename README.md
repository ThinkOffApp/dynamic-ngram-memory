# Dynamic n-gram memory

![TensorFold Memory: three notes about my sister Anna answer "What does my sister drive, and where?", with the Memory Benchmark results](docs/images/tensorfold-memory.jpg)

Explicit, editable memory for language models: retrieve relevant notes using token n-grams, place a bounded block in the prompt, and keep a journal of changes and take-backs.

The paper *Fast to Write, Slow to Learn* ([doi.org/10.5281/zenodo.23604600](https://doi.org/10.5281/zenodo.23604600)) calls the method dynamic n-gram memory, and calls its integration in the TensorFold engine TensorFold Notes.

This is the Notes module created by Petrus Pennanen and the ThinkOff team for [TensorFold](https://github.com/ashhart/TensorFold). This repository is its separate home for source, integration instructions and examples. TensorFold supplies the model engine, tokenizer, HTTP server and prompt handling. The original lookup comes from ThinkOff's test suite, which is now in [tests/](tests/).

**Pre-release integration snapshot.** This package contains an exact module export at the commit recorded in [SOURCE.json](SOURCE.json). It runs inside the pinned TensorFold host; it is not yet an independently buildable library. The test suite the paper uses is in [tests/](tests/); its working history is not included.

## What it does

- Ranks notes using token n-grams, IDF weighting, subject matches and selected value links.
- Fits the retrieved block to a token cap using the served model's tokenizer.
- Supports adding, changing, taking back and erasing notes over HTTP.
- Uses a durable journal and publishes new index versions for concurrent readers.
- Requires explicit server and request switches to use Notes.

The pinned version still rebuilds indexes on writes. Bounded write latency at one million notes is unfinished. An erase removes note-store material and tokenizer-kept encodes, but comprehensive removal from saved conversations, prompt caches and other derived artifacts is an open production gate. Do not interpret erase as a complete privacy guarantee.

## Reproduce the integration

Install Git, Python 3.10 or later, and Zig 0.17.0. From this repository:

```sh
python3 tools/stage_tensorfold.py work/tensorfold --check
```

The command creates a fresh checkout of the exact public host commit, verifies every exported module file, and runs the host's CPU server tests. It refuses to replace an existing destination. It downloads source only, with no model download. For an existing local source repository, use `--source-repository /path/to/TensorFold` to stage without network access. Use `--zig /path/to/zig` to choose the compiler.

Build the native engine in the staged host using TensorFold's platform instructions. Serve your model with a compatible Notes snapshot:

```sh
tensorfold-native serve MODEL --notes SNAPSHOT
```

A snapshot must contain note token IDs for the model's tokenizer. Prefer a snapshot with its `tokenizer.json` SHA-256, which the server verifies. A vocabulary-size-only snapshot provides weaker compatibility checking. The existing exporter is part of the test suite ([tests/zengine/notes_export.py](tests/zengine/notes_export.py)); a self-contained public snapshot creation tool is still needed before this becomes a fresh-install quick start.

See [examples/http.md](examples/http.md) for the actual HTTP contract. The module starts with one owner per server. Use the host's authentication and deployment documentation when exposing a server.

See [TensorFold release integration](docs/tensorfold-releases.md) for the upstream plan and the distinction between this pinned fork setup and an official TensorFold release.

## Production gates

1. Review and integrate the experimental uncertainty annotations and improved index; remove whole-store copying from the write path and measure p99/max at the configured hard limit.
2. Prove erase across prompt caches, saved conversations and restart, including derived text.
3. Run real Metal/CUDA streaming, cancellation, resumed-generation and prefix-cache comparisons, plus one-million-note load with eight lookup threads.
4. Exercise explicit opt-in switches and rollback. Keep nightly weight promotion experimental while longer learning/regression runs remain pending.

These gates are a work plan, not completed qualifications. The baseline source and its previously recorded experiments are linked through [docs/provenance.md](docs/provenance.md); this repository does not claim a new benchmark run.

## Paper material

- [docs/PAPER-RESULTS-DETAIL.md](docs/PAPER-RESULTS-DETAIL.md): every measured number of the paper, with the file it comes from, and the numbers no file here backs yet.
- [docs/NOTES-LAYER.md](docs/NOTES-LAYER.md) and [docs/NOTES-ZIG.md](docs/NOTES-ZIG.md), with raw data in [docs/notes-raw/](docs/notes-raw/): the note store measured offline.
- [docs/PROMOTION.md](docs/PROMOTION.md): how promotion from notes into neurons decides.
- [docs/RESULTS.md](docs/RESULTS.md) and [docs/rejudge/](docs/rejudge/): the first fact tests and the corrections to the grading rules.
- [docs/receipts/](docs/receipts/): two GPU runs with every answer, [E1](docs/receipts/e1-no-notes-control/) (do the neurons answer without the notes?) and [erasing a promoted fact](docs/receipts/erase-gpu-2026-10-19/).
- [tests/](tests/): the test suite: rule book in Python and Lean, generator, judge and runner.

## Attribution and license

The TensorFold export retains its Apache-2.0 license and notices. The lookup derived from the test suite, the test suite in [tests/](tests/), and the documents copied from it into [docs/](docs/) keep their MIT license; see [LICENSES/TEST-SUITE-MIT.txt](LICENSES/TEST-SUITE-MIT.txt). The included TensorFold third-party notice inventory is preserved for provenance and describes components of the full host as well as this subset. See [NOTICE](NOTICE), [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
