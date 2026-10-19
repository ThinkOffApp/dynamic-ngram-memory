# Raw outputs of the offline note measurements

These are the raw outputs behind [../NOTES-LAYER.md](../NOTES-LAYER.md) (the Python note store) and [../NOTES-ZIG.md](../NOTES-ZIG.md) (its lookup ported to Zig). They were measured on an Apple M5 Max MacBook Pro with Nemotron 3.5 Lightning's own tokenizer and no model. The code that wrote them is in [../../tests/](../../tests/) (`zengine/notes.py`, `zengine/notes_zig_golden.py`, `zengine/notes_zig_bench.py`).

Each measurement has two files: the JSON the command wrote, and its console output as text. [runs.txt](runs.txt) lists each command with its exit code and how long it took. The commands themselves are in the "How to run" sections of the two pages above.

| Files | What they hold |
| --- | --- |
| `recall-L1-L6.*`, `recall-L7.*` | Whether the lookup finds every note a question needs, on the example test sets |
| `ablate-tuning.*`, `ablate-examples.*` | The same with each lookup setting changed, on other generated sets and on the example sets |
| `sweep.*`, `sweep-subject0.*`, `sweep-value-hop.*`, `sweep-misses.*` | Stores grown to 100,000 notes with distractor notes: recall, lookup time, write time and memory |
| `zig-golden.*` | The Zig lookup against the Python lookup, note for note |
| `zig-bench.*`, `zig-bench-spot.*`, `zig-bench-failures.*` | Speed of the Zig lookup with 1 and 8 threads, and checks that a failure stops the benchmark |

These files come from the test suite's working repository (commit `2083ee9`, MIT license: [../../LICENSES/LLWZ-MIT.txt](../../LICENSES/LLWZ-MIT.txt)). For publication, home-directory paths inside the JSON files are shown as `~`. Nothing else was changed.
