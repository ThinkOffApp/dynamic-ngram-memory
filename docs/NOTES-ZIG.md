# The note lookup in Zig

> **In this repository.** This page comes from the test suite's working repository (commit `2083ee9`, MIT license: [LICENSES/LLWZ-MIT.txt](../LICENSES/LLWZ-MIT.txt)). The code it names (`zengine/`, `tools/`, `examples/`, `lean/`, `zig/notes/`, `docs/TEST-LOGIC.md`) is in [tests/](../tests/) here. For publication, home-directory paths are shown as `~`, a few names were removed, and links were pointed to their new places; nothing else was changed.

The read path of the note memory (`zengine/notes.py`, [NOTES-LAYER.md](NOTES-LAYER.md)) ported to Zig, and checked against the Python question by question. Written 15 October 2026, branch `notes-zig`.

Updated the same day: `exclude` ported and tested; the branch merged with `promotion` at `ec874b3`.

Grades, as in NOTES-LAYER.md: **MEASURED** means we ran it, on the machine stated above each table. **READ** means we read it in code or a document. **INFERRED** means our own reasoning, not checked by a run. Every number below was measured by us on this Mac (an Apple M5 Max MacBook Pro), on 15 October 2026.

## In short

- **The port gives Python's answers.** Over every example set, every phase that keeps lessons, every question asked three ways (about now, as `retrieve()` reads it, about the past) at caps 512 and 128, over two grown stores of 1,000 and 10,000 notes, and over all 13 named settings of `notes.ABLATIONS` plus the explicit take-back with the unknown rule: 279,096 runs per tokenizer, with the stand-in tokenizer and with Nemotron 3.5 Lightning's own. Zig returned the same note ids, in the same order, and the same block text, byte for byte, in 279,004 of them (MEASURED). Every one of the other 92 is a tie between two notes that floating point noise breaks, and Python itself breaks it either way depending on the order it walks a set (below). No run differed for any other reason.
- **`exclude` gives Python's answers too.** Asking every question again with one retrieved note excluded (each in turn, up to 24 per phase) and with every note it retrieved excluded, on the example sets and the grown stores: 299,538 runs per tokenizer, 299,424 equal (stand-in) and 299,422 (Nemotron), the rest float-order ties (114 and 116), 0 other. Zig never gave an excluded note. A Zig lookup whose hop brings excluded notes back differs in 94,376 and 94,257 of them (MEASURED).
- **Scores are the same doubles.** In every run where the answers agree, Zig's score for each chosen note is bit for bit the score Python computes when it adds the n-gram weights in Zig's order (279,004 of 279,004, each tokenizer). That took one fix: Zig's own `log` and the C library's differ in the last bit for some inputs, so the port calls the C library's, as CPython does.
- **The test can fail.** A Zig lookup with the subject boost switched off differs from Python in 13,583 runs (stand-in) and 13,364 (Nemotron); with the hop switched off, in 141,849 and 142,646. A Python `fit()` made deliberately stricter makes 956 and 554 runs differ, and the test sorts every one of them as a `fit` difference, with Zig equal to Python's own `pack()`. Three more checks gate the test, beside ids and block: candidate counts equal to Python's in every run, every score bit for bit in every equal run (one score per note), and the token sum in every equal or fit run. Each has a control that leaves ids and block right and reports one thing wrong, and each fails its gate (below).
- **`fit()` never changed a result in the real runs.** Python's `fit()` re-counts the whole block with the tokenizer and drops notes while it is over the cap. With Nemotron's tokenizer, `pack()`'s sum of the parts was never below the count of the whole block (above it in 265,475 runs, by up to 38 tokens; equal in 13,529), so `fit()` dropped nothing. Zig reports `pack()`'s sum as the block's tokens.
- **Speed.** On the sweep stores with Nemotron's tokenizer, a Zig lookup takes 2 to 2.5 µs at 100 notes and 146 to 167 µs at 100,000 (p50, one thread), against 0.44 to 0.50 ms and 12.5 to 13.7 ms for Python's `retrieve()`, and 51 to 89 times less than Python doing the same work from token ids. Eight threads over one shared index give 7.0 to 8.0 times the lookups per second of one thread; Python's eight give about what one gives. At 100,000 notes the index takes 34 MB and loads from its 37 MB snapshot in 147 ms (MEASURED).

## What is ported, and what is not

Ported (the lookup, `NoteStore._lookup` and what it calls):

| Python | Zig |
| --- | --- |
| `key`, `wordy` (fold, punctuation) | a table of every token id's key and wordy flag, from the snapshot |
| `grams` (n from `n_min` to `n_max`, punctuation-only unigrams out unless `punct`) | the same, for the question and, at load, for every note (the postings of both views are rebuilt from the notes' token ids and checked against the store's own gram counts) |
| the IDF weight `ln(1 + (N - df + 0.5)/(df + 0.5))`, df and N over the views searched | the same, with the C library's `log` |
| `subjects` (entity phrases, 2 to 8 keys, longest first, none inside a found one; a phrase found twice counts twice) and the subject boost (`ent_cur`, and `ent_keys`/`by_key` for the past) | the same |
| the candidate cut (`room`, the 4·room best of more than 8·room), `min_rel`, the hop (`subject`, `subject+value`, `none`, `hop_history`, `hop_from`, `hop_weight`, read from the live scores in the same order) | the same |
| the order `(-score, -told, id)`, `pack`, `render` | the same; the ranking key is one 128-bit number per note |
| `exclude` (note ids never given: promotion keeps a note whose fact the neurons answer out of the prompt): their scores go after the subject boost, before the candidates are counted; they still count in the document frequencies; a hop skips them | the same; `lookup(..., exclude, ...)` takes the ids, and a CLI query line takes them after a bar |

Not ported:

- **Writes.** The intake check, supersede, take-back, delete and the write lock stay in Python. A snapshot is a copy taken under the owner's read lock: a write after it reaches Zig only when the store is exported again.
- **Erasure reaches the snapshot only by a new export.** Python's `delete` leaves the lesson in no structure of the store (NOTES-LAYER.md). A snapshot file written before the delete still holds the note's text and token ids until that file is removed or overwritten (READ, from the code: the file is plain JSON). Whatever runs the Zig lookup must export again and remove the old file after a delete.
- **`fit()`'s exact count.** There is no tokenizer in Zig. The block's tokens are `pack()`'s sum; see the results for how it compares.
- **`is_past`.** The caller passes the past flag (Python's regular expression is not ported). The golden test passes `notes.is_past(question)`.
- **More than one owner.** A snapshot holds one owner; several owners are several snapshots.
- **`n_max` over 4.** N-grams are packed into one 128-bit key, four 32-bit keys; a snapshot with `n_max` over 4 is refused at load.

## The snapshot

`zengine/notes_export.py` writes one owner's store as compact JSON (`lw-notes-snapshot/1`):

| Key | What it holds |
| --- | --- |
| `settings` | every key of `notes.DEFAULTS` as the store has it |
| `head`, `cur_title`, `hist_title`, `fixed` | the block's head (with the unknown rule if set), the two titles, and the tokens of each and of one line break (`NoteStore.fixed`) |
| `vocab` | `key`: every token id's index key as one 32-bit number (a fold string's place in `folds`, or `0x80000000` with the id itself when the key is the id); `wordy`: one `0`/`1` per id; `folds`: the interned fold strings (Zig does not read them) |
| `notes` | per note: `id`, `state` (0 current, 1 superseded, 2 withdrawn), `kind`, `entity`, `attribute`, `pair` (into `pairs`), `value_ent` (the entity its value names, for the `subject+value` hop, or -1), `told`, `line` (exactly `NoteStore.line()`), `cost` (`NoteStore.cost`), `tokens` |
| `entities` | per entity: `cur` (`ent_cur`), `keys` (`ent_keys`, as places in `pairs`) |
| `pairs` | `by_key`: per (entity, attribute), its note ids, oldest first |
| `phrases` | `O.phrases`: each entity's phrase as keys |
| `n`, `grams` | notes and distinct n-grams per view, which the loader checks |

With the model's tokenizer the vocabulary is every id it has (131,072 for Nemotron), so Zig folds any question. The stand-in tokenizer has no fixed vocabulary (an id is the piece's own bytes), so its snapshot lists the pieces of the notes and of the texts passed as `asks` (the questions), numbered from 0, and `Vocab.ids()` refuses a question with a piece outside them. The JSON loads in 6 ms (100 notes, a 2.3 MB file, most of it the vocabulary) to 147 ms (100,000 notes, 37 MB), once per snapshot and apart from the lookups (MEASURED, below); a binary format was not needed for this.

## Running it

Zig 0.17.0, the version it was written and measured with:

```sh
cd zig/notes && ~/zig/zig-aarch64-macos-0.17.0/zig build -Doptimize=ReleaseFast && ~/zig/zig-aarch64-macos-0.17.0/zig build test
```

The CLI (`zig/notes/zig-out/bin/notes-zig`):

```sh
python3 zengine/notes_export.py examples/L6-seed1.json --phase teach --out zig-out/l6.json --queries zig-out/l6.q
zig/notes/zig-out/bin/notes-zig lookup zig-out/l6.json zig-out/l6.q          # one JSON line per question
zig/notes/zig-out/bin/notes-zig lookup zig-out/l6.json zig-out/l6.q --break exclude-hop   # a negative control
zig/notes/zig-out/bin/notes-zig bench zig-out/l6.json zig-out/l6.q --threads 1,8 --reps 50
```

A query line is `past cap id id ...`, and `past cap id id ... | note note ...` to exclude notes (`retrieve()`'s `exclude`). `lookup` prints the note ids (current ones first, each section by when it was told, as `retrieve()` gives them), the scores (as their 64 bits, so nothing is lost in decimal), `pack()`'s tokens, the candidates and the block. The library is `zig/notes/src/notes.zig`: `Index.load` (or `loadFile`), then per thread a `Scratch` and `lookup(index, scratch, ids, past, cap, exclude, .{})`. An index is read-only once loaded.

The golden test and the bench (the commands that produced the results below; `TOK` is Nemotron's `tokenizer.json`, and `~/tmp/tensorfold/.venv/bin/python` has the `tokenizers` package):

```sh
PYTHONHASHSEED=0 ~/tmp/tensorfold/.venv/bin/python zengine/notes_zig_golden.py --tokenizer fake --tokenizer $TOK --grid --json docs/notes-raw/zig-golden.json
```

```sh
~/tmp/tensorfold/.venv/bin/python zengine/notes_zig_bench.py --tokenizer $TOK --json docs/notes-raw/zig-bench.json
```

Their console output is beside the JSON in `docs/notes-raw/` (`zig-golden.txt`, `zig-bench.txt`; `runs.txt` has exit codes and seconds). `python3 zengine/notes_zig_golden.py` alone runs the stand-in tokenizer, the defaults, the exclude runs and the grown stores (the example sets with their exclude runs took 117 seconds in the run below); `--exclude-most 0` leaves the exclude runs out (about 30 seconds in all). `python3 zengine/notes_zig_bench.py --failure-controls` checks that the bench stops with an error under injected faults (a few seconds).

## Results: the golden test (MEASURED)

**Machine:** Apple M5 Max (`machdep.cpu.brand_string`), 18 cores (`hw.ncpu`; 6 "Super" and 12 "Performance" by `hw.perflevel0/1.name`), 137,438,953,472 bytes (128 GiB, `hw.memsize`), macOS 27.0.1. **Zig 0.17.0**, ReleaseFast. **CPython 3.12.8**, `tokenizers` 0.23.2, `PYTHONHASHSEED=0`. Code commit `dd357a1` (after the merge with `promotion` at `ec874b3`; the lookup is as in `3f0855a`, the test now gates three more checks and runs seven Zig variants per query); 2,384 seconds for everything in this section, both tokenizers, with another agent's test running on the same Mac. The counts are the same as in the run from `3f0855a` (978 seconds) in all 742 rows of `zig-golden.json`, and those without `exclude` the same as in the first full run (code commit `101b1de`, 362 seconds) in all 642 of its rows.

What is compared: for each set and each phase with lessons (`notes.recall_phases(S, "all")`, level 7's checkpoints and pairs included), the store is built by `notes.set_store`, exported, and every question is asked of Python's `retrieve()` and of Zig, about now (`past=False`), as `retrieve()` reads it (`past=None`, so `notes.is_past`) and about the past (`past=True`), at caps 512 and 128. A run is equal when the note ids, their order and the block text are all the same.

The example sets, default settings, both tokenizers (each row: every phase, every question, 2 caps, 3 ways):

| Set | Phases | Compared (per tokenizer) | Equal, stand-in | Float order, stand-in | Equal, Nemotron | Float order, Nemotron | Other, either |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| L1-seed1 | 1 | 72 | 72 | 0 | 72 | 0 | 0 |
| L2-seed1 | 3 | 288 | 288 | 0 | 288 | 0 | 0 |
| L3-seed1 | 3 | 396 | 396 | 0 | 396 | 0 | 0 |
| L4-seed1 | 4 | 528 | 528 | 0 | 528 | 0 | 0 |
| L5-seed1 | 3 | 648 | 648 | 0 | 648 | 0 | 0 |
| L6-seed1 | 4 | 1,344 | 1,344 | 0 | 1,344 | 0 | 0 |
| L6-seed2 | 4 | 1,344 | 1,344 | 0 | 1,344 | 0 | 0 |
| L6-seed3 | 4 | 1,344 | 1,344 | 0 | 1,344 | 0 | 0 |
| neg-seed1 | 1 | 72 | 72 | 0 | 72 | 0 | 0 |
| neg2-seed1 | 3 | 216 | 216 | 0 | 216 | 0 | 0 |
| v2/L1-seed1-v2 | 1 | 138 | 138 | 0 | 138 | 0 | 0 |
| v2/L2-seed1-v2 | 3 | 504 | 504 | 0 | 504 | 0 | 0 |
| v2/L3-seed1-v2 | 3 | 1,044 | 1,044 | 0 | 1,044 | 0 | 0 |
| v2/L4-seed1-v2 | 6 | 2,700 | 2,700 | 0 | 2,700 | 0 | 0 |
| v2/L4-seed1-v2-restate | 6 | 2,700 | 2,698 | 2 | 2,698 | 2 | 0 |
| v2/L5-seed1-v2 | 3 | 1,386 | 1,386 | 0 | 1,386 | 0 | 0 |
| v2/neg-seed1-v2 | 1 | 114 | 114 | 0 | 114 | 0 | 0 |
| v2/neg2-seed1-v2 | 3 | 414 | 414 | 0 | 414 | 0 | 0 |
| L7-seed1 | 21 | 16,884 | 16,863 | 21 | 16,863 | 21 | 0 |
| eyes-678 | 5 | 450 | 450 | 0 | 450 | 0 | 0 |
| nights/L6-seed1-nights | 6 | 2,016 | 2,016 | 0 | 2,016 | 0 | 0 |

Other settings and grown stores (each row summed over its sets; a grid row is every example set and phase at cap 512, three ways):

| Section | Compared (per tokenizer) | Equal, stand-in | Float order, stand-in | Equal, Nemotron | Float order, Nemotron | Other, either |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 as written: exact ids, n 1-3, subject hop, no version hop | 17,301 | 17,301 | 0 | 17,301 | 0 | 0 |
| 2 + folded ids | 17,301 | 17,292 | 9 | 17,291 | 10 | 0 |
| 3 + version hop for the past | 17,301 | 17,292 | 9 | 17,291 | 10 | 0 |
| 4 + no punctuation-only keys | 17,301 | 17,295 | 6 | 17,294 | 7 | 0 |
| 5 + subject match 1.0 (the defaults) | 17,301 | 17,295 | 6 | 17,295 | 6 | 0 |
| defaults, subject match 2.0 | 17,301 | 17,295 | 6 | 17,295 | 6 | 0 |
| defaults, hop none | 17,301 | 17,301 | 0 | 17,301 | 0 | 0 |
| defaults, hop subject+value | 17,301 | 17,294 | 7 | 17,295 | 6 | 0 |
| defaults, n-grams 2 to 4 (paper 4.1) | 17,301 | 17,301 | 0 | 17,301 | 0 | 0 |
| defaults, floor 0.1 | 17,301 | 17,291 | 10 | 17,292 | 9 | 0 |
| defaults, floor 0.2 | 17,301 | 17,294 | 7 | 17,294 | 7 | 0 |
| defaults, floor 0.3 | 17,301 | 17,299 | 2 | 17,299 | 2 | 0 |
| defaults, floor 0.5 | 17,301 | 17,301 | 0 | 17,301 | 0 | 0 |
| defaults, take-back explicit + unknown rule | 17,301 | 17,294 | 7 | 17,295 | 6 | 0 |
| grown store: L6-seed1, 1,000 notes | 336 | 336 | 0 | 336 | 0 | 0 |
| grown store: L6-seed1, 10,000 notes | 336 | 336 | 0 | 336 | 0 | 0 |
| grown store: L7-seed1, 1,000 notes | 804 | 804 | 0 | 804 | 0 | 0 |
| grown store: L7-seed1, 10,000 notes | 804 | 804 | 0 | 804 | 0 | 0 |
| control: fit() against cap - 16 (fit: 956 and 554) | 1,140 | 184 | 0 | 586 | 0 | 0 |

Over everything, per tokenizer: 279,096 runs; 279,004 equal; 0 differ because of `fit()`; 92 because of float order; **0 otherwise**. The candidate counts agree in all 279,096, and in all 279,004 equal runs Zig's tokens are `pack()`'s sum as Python computes it. The `own` flag read a question as about the past in 3,184 of 93,032 `own` runs.

Negative controls, the same runs with a Zig lookup broken on purpose, and a Python `fit()` made stricter on purpose:

| Control | Stand-in: runs that differ | Nemotron: runs that differ |
| --- | ---: | ---: |
| `--break subject` (no subject boost) | 13,583 of 279,096 | 13,364 of 279,096 |
| `--break hop` (no hop) | 141,849 of 279,096 | 142,646 of 279,096 |
| Python's `fit()` against cap - 16 (L6-seed1 and L7-seed1, teach) | 956 of 1,140, all sorted `fit`, Zig equal to Python's `pack()` | 554 of 1,140, the same |
| `--break exclude-hop` (a hop brings excluded notes back), over the exclude runs below | 94,469 of 299,538 (94,376 not fit or float) | 94,350 of 299,538 (94,257 not fit or float) |

`--break exclude-hop` changes nothing without `exclude` (the 92 runs it "differs" in are the float-order ties above). The breaks show nothing on the stores of 1 to 3 notes (L1, neg, neg2 and their v2 forms): every note fits there whatever the scores. The per-set counts are in `docs/notes-raw/zig-golden.txt`.

### Gates

Beside ids and block, three checks fail the test unless they hold over every run of a tokenizer (the fit control's included). Each has a control: a right lookup that reports one thing wrong (`notes-zig lookup --break score|score-length|tokens|candidates`), so ids and block stay equal and only that check can see it.

| Gate | Stand-in | Nemotron |
| --- | ---: | ---: |
| candidate count equal to Python's, in every run | 579,774 of 579,774 | 579,774 of 579,774 |
| one score per note, each bit for bit Python's summed in Zig's order, in every equal run | 578,612 of 578,612 | 579,012 of 579,012 |
| tokens equal to `pack()`'s sum, in every equal or fit run | 579,568 of 579,568 | 579,566 of 579,566 |

| Control (ids and block stay as Python's: 578,612 and 579,012 runs, as unbroken) | Gate it must fail | Stand-in | Nemotron |
| --- | --- | ---: | ---: |
| `--break score`: the first note's score one ulp up | scores | fails: 541,826 equal runs off (the 36,786 that pass give no notes) | fails: 542,473 off |
| `--break score-length`: the last score left out | scores | fails: 541,826 lists of the wrong length | fails: 542,473 |
| `--break tokens`: `pack()`'s sum plus one | tokens | fails: 0 of 579,568 hold | fails: 0 of 579,566 |
| `--break candidates`: one candidate too many | candidates | fails: 0 of 579,774 hold | fails: 0 of 579,774 |

The score check first requires one score per note; it used to zip the two lists, which would have let a short list pass. Every Zig variant must also give exactly one result per query, and Python one per job, before results are paired.

### With exclude

`retrieve(..., exclude=ids)` (promotion keeps a note whose fact the neurons answer out of the prompt). For each set and phase above, with the default settings, every question is asked twice more of both: **one**, with one current note that some question of the phase retrieved excluded, for each such note in turn (at most 24 per phase, evenly spread by id; cap 512; now, own and past); and **all**, with every note the question itself retrieved excluded (caps 512 and 128; now, own and past; a question that retrieved nothing is skipped). The same on the four grown stores. A run counts as equal on the same terms as above, and the test also checks that Zig never gives an excluded note.

| Set | Notes tried one at a time (all phases) | One: runs | Equal, stand-in / Nemotron | Float order | All: runs | Equal | Float order | Other | `--break exclude-hop` caught, stand-in / Nemotron |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| L1-seed1 | 1 | 36 | 36 / 36 | 0 / 0 | 54 | 54 / 54 | 0 / 0 | 0 | 0 / 0 |
| L2-seed1 | 3 | 144 | 144 / 144 | 0 / 0 | 224 | 224 / 224 | 0 / 0 | 0 | 34 / 34 |
| L3-seed1 | 7 | 462 | 462 / 462 | 0 / 0 | 378 | 378 / 378 | 0 / 0 | 0 | 378 / 378 |
| L4-seed1 | 34 | 2,244 | 2,244 / 2,244 | 0 / 0 | 504 | 504 / 504 | 0 / 0 | 0 | 1,632 / 1,628 |
| L5-seed1 | 12 | 1,296 | 1,296 / 1,296 | 0 / 0 | 634 | 634 / 634 | 0 / 0 | 0 | 904 / 941 |
| L6-seed1 | 92 | 15,456 | 15,456 / 15,456 | 0 / 0 | 1,326 | 1,326 / 1,326 | 0 / 0 | 0 | 8,534 / 8,525 |
| L6-seed2 | 92 | 15,456 | 15,456 / 15,454 | 0 / 2 | 1,338 | 1,338 / 1,338 | 0 / 0 | 0 | 8,639 / 8,648 |
| L6-seed3 | 92 | 15,456 | 15,456 / 15,456 | 0 / 0 | 1,344 | 1,344 / 1,344 | 0 / 0 | 0 | 9,228 / 9,215 |
| neg-seed1 | 1 | 36 | 36 / 36 | 0 / 0 | 60 | 60 / 60 | 0 / 0 | 0 | 0 / 0 |
| neg2-seed1 | 3 | 108 | 108 / 108 | 0 / 0 | 162 | 162 / 162 | 0 / 0 | 0 | 9 / 9 |
| v2/L1-seed1-v2 | 1 | 69 | 69 / 69 | 0 / 0 | 90 | 90 / 90 | 0 / 0 | 0 | 0 / 0 |
| v2/L2-seed1-v2 | 3 | 252 | 252 / 252 | 0 / 0 | 404 | 404 / 404 | 0 / 0 | 0 | 48 / 48 |
| v2/L3-seed1-v2 | 7 | 1,218 | 1,218 / 1,218 | 0 / 0 | 1,014 | 1,014 / 1,014 | 0 / 0 | 0 | 1,017 / 1,017 |
| v2/L4-seed1-v2 | 50 | 11,250 | 11,250 / 11,250 | 0 / 0 | 2,664 | 2,664 / 2,664 | 0 / 0 | 0 | 9,979 / 9,948 |
| v2/L4-seed1-v2-restate | 44 | 9,900 | 9,900 / 9,900 | 0 / 0 | 2,664 | 2,664 / 2,664 | 0 / 0 | 0 | 8,689 / 8,535 |
| v2/L5-seed1-v2 | 12 | 2,772 | 2,772 / 2,772 | 0 / 0 | 1,372 | 1,372 / 1,372 | 0 / 0 | 0 | 1,874 / 1,946 |
| v2/neg-seed1-v2 | 1 | 57 | 57 / 57 | 0 / 0 | 102 | 102 / 102 | 0 / 0 | 0 | 0 / 0 |
| v2/neg2-seed1-v2 | 3 | 207 | 207 / 207 | 0 / 0 | 360 | 360 / 360 | 0 / 0 | 0 | 20 / 20 |
| L7-seed1 | 345 | 138,690 | 138,579 / 138,579 | 111 / 111 | 16,554 | 16,551 / 16,551 | 3 / 3 | 0 | 31,110 / 31,106 |
| eyes-678 | 9 | 405 | 405 / 405 | 0 / 0 | 306 | 306 / 306 | 0 / 0 | 0 | 214 / 215 |
| nights/L6-seed1-nights | 124 | 20,832 | 20,832 / 20,832 | 0 / 0 | 1,998 | 1,998 / 1,998 | 0 / 0 | 0 | 11,765 / 11,755 |

| Grown store | Runs | Equal, stand-in / Nemotron | Float order | Other | `--break exclude-hop` caught |
| --- | ---: | ---: | ---: | ---: | ---: |
| L6-seed1, 1,000 notes, every retrieved note excluded | 336 | 336 / 336 | 0 / 0 | 0 | 18 / 16 |
| L6-seed1, 1,000 notes, one note excluded at a time (24 tried) | 4,032 | 4,032 / 4,032 | 0 / 0 | 0 | 65 / 49 |
| L6-seed1, 10,000 notes, every retrieved note excluded | 336 | 336 / 336 | 0 / 0 | 0 | 13 / 12 |
| L6-seed1, 10,000 notes, one note excluded at a time (24 tried) | 4,032 | 4,032 / 4,032 | 0 / 0 | 0 | 9 / 13 |
| L7-seed1, 1,000 notes, every retrieved note excluded | 804 | 804 / 804 | 0 / 0 | 0 | 51 / 59 |
| L7-seed1, 1,000 notes, one note excluded at a time (24 tried) | 9,648 | 9,648 / 9,648 | 0 / 0 | 0 | 103 / 100 |
| L7-seed1, 10,000 notes, every retrieved note excluded | 804 | 804 / 804 | 0 / 0 | 0 | 16 / 13 |
| L7-seed1, 10,000 notes, one note excluded at a time (24 tried) | 9,648 | 9,648 / 9,648 | 0 / 0 | 0 | 27 / 27 |

Over all exclude runs, per tokenizer: 299,538 runs; 299,424 equal (stand-in), 299,422 (Nemotron); 0 differ because of `fit()`; 114 and 116 because of float order; **0 otherwise**. About 636,000 (stand-in) and 632,000 (Nemotron) note ids were excluded across them, and Zig gave an excluded note in none. Candidate counts agree in every run, and in every equal run the scores are bit for bit Python's summed in Zig's order.

**What was capped** (logged in `exclude_one_tried` in `docs/notes-raw/zig-golden.json`): on the example sets, 936 of the 1,234 retrieved current notes were tried one at a time; the cap of 24 bit in 12 of 88 phases, all of them level 7 (ck-025 to ck-060, teach, del-1, del-3 and del-6, which retrieve 25 to 64 notes each). On the grown stores, 24 of 324 to 1,270 retrieved notes each (all four capped). The "all" runs are not capped.

### Why 92 runs differ: float order

Python adds each note's n-gram weights in the order it walks the question's *set* of n-grams, and that order follows string hashing, so it changes with `PYTHONHASHSEED`. Zig adds them in the order the n-grams are first met. Two notes that match the same n-grams get the same sum either way. Two notes that match *different* n-grams whose weights add up to the same real number can come out one bit apart, and the tie-break `(-told, id)` then never runs. Example (stand-in, L7-seed1, phase ck-040, "In which city is my brother's home?", cap 128): notes 6 ("My parrot's name is Otto.") and 30 ("My cousin is living in Bergen.") have scores that are equal in exact arithmetic. Summed in Python's order both are `0x1.39b42416bb1ecp+1`, the tie goes to the later note (30); summed in Zig's order note 30 gets `...1ebp+1`, one bit less, and note 6 wins the last place in the block.

The test sorts a run as `float` only when Python, asked again with its n-grams summed in Zig's order, gives exactly Zig's answer; so in all 92 Python's own answer changes with nothing but the order of its additions. Which runs these are depends on the hash seed: two runs of the stand-in defaults (examples and grown stores, 36,882 runs) without a fixed seed had 9 and 6 such runs; with `PYTHONHASHSEED=0` the same runs have 23. Zig's order is fixed, so Zig gives the same answer every time; Python does not. **Both are equally right**: the scores are equal, and the order between them is noise. Making Python deterministic (adding the weights in sorted order) would remove the category; that is a change to `notes.py`, not made here.

### Tokens

Zig reports `pack()`'s sum: the head, each title, each line and each line break counted apart. With the stand-in tokenizer that equals the count of the whole block in every run. With Nemotron's tokenizer it was above the whole block's count in 265,475 of 279,004 equal runs (by up to 38 tokens) and never below it: pieces merge across the line breaks. So `fit()` never had to drop a note, and a Zig block is within the cap as the model's tokenizer counts it in every run measured. Nothing guarantees "never below" for other text or another tokenizer (INFERRED); if it ever happens, the test sorts the run as `fit` and checks Zig against Python's `pack()`.

### The log function

An early run had answers that agreed while Zig's scores were one bit off Python's (377 note scores over three sets, stand-in tokenizer). The cases we looked at were the weight `ln(12/11)` (a unigram in every note of a 5-note store). Python's `math.log` calls the C library's `log`; a Zig `extern fn log` was bound at link time to Zig's own compiler_rt copy, which is one bit lower for that input (`0x3fb64660aa8ce620` against `...621`). A `zig run` of the same call bound the C library's instead, so it depends on how the binary is linked. The port now looks `log` up by name in the C library (`dlsym`) when an index loads. After the fix, scores agree bit for bit in every equal run.

How often the two differ: over every input the weight formula forms for N up to 3,000 and every df (and the subject boost's), 4,504,500 inputs, Zig's own `log` and the C library's give different bits for 6,310 (0.14%), always by one bit, in either direction (MEASURED: a Debug build, where `@log` is Zig's own, against CPython's `math.log`). The unit test is bit for bit: for six of those inputs (N, df = 5, 5, which is `log(12/11)`; 32, 12; 57, 48; 62, 30; 62, 54; 73, 29) the looked-up `log` must give exactly the C library's bits as CPython's `math.log` gives them on macOS 27.0.1, and, as its negative control, Zig's own `log` of each input (evaluated by the compiler, the same code as compiler_rt's) must give other bits. With Zig's own `log` put in place of the looked-up one, the test fails (expected `...621`, found `...620`); it passes in Debug and ReleaseFast builds. It is skipped off macOS, where the C library's bits were not measured.

## Results: speed (MEASURED)

**Machine:** Apple M5 Max, 18 cores (6 "Super" and 12 "Performance"), 128 GiB, macOS 27.0.1. **Zig 0.17.0**, ReleaseFast (the binary reports both). Python side: **CPython 3.12.8**, `tokenizers` 0.23.2 with Nemotron 3.5 Lightning's `tokenizer.json`. Code commit `101b1de`; 1,154 seconds, of which 508 and 515 grew the two 100,000-note stores in Python.

The stores are `notes.py`'s sweep stores, built the same way (`notes_zig_golden.grow`: the set's teach notes for owner `u0`, then `notes.distractors`, seed 1, until the size is reached). The questions are the set's own (56 for L6-seed1, 134 for L7-seed1), each with the past flag `retrieve()` gives it and the default cap of 512. Zig: each thread asks every question 50 times in its own shuffled order, over one shared read-only index; every timed lookup is checked against a first single-thread pass (0 differed, in every row). Each lookup is timed alone with the monotonic clock, the two clock reads included. Python: `notes.timed_lookups`, as the sweep times it, 3 times per thread. "Same as Python": Zig's ids and block against `retrieve()` on every question at that size.

Zig:

| Set | Notes | Current | Snapshot | Load | Index | Peak RSS | 1 thread p50 / p95 (µs) | Lookups/s, 1 thread | 8 threads p50 / p95 (µs) | Lookups/s, 8 threads | Same as Python |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| L6-seed1 | 100 | 60 | 2.3 MB | 6 ms | 0.8 MB | 7 MB | 2.0 / 2.83 | 497,258 | 2.08 / 3.04 | 3,502,120 | 56/56 |
| L6-seed1 | 1,000 | 472 | 2.6 MB | 8 ms | 1.2 MB | 9 MB | 6.42 / 11.17 | 159,510 | 6.71 / 12.13 | 1,145,787 | 56/56 |
| L6-seed1 | 10,000 | 4,113 | 5.7 MB | 21 ms | 4.4 MB | 26 MB | 31.92 / 39.25 | 36,323 | 32.96 / 42.67 | 274,112 | 56/56 |
| L6-seed1 | 100,000 | 24,262 | 36.9 MB | 146 ms | 34.0 MB | 151 MB | 145.54 / 241.33 | 7,321 | 161.92 / 236.25 | 56,187 | 56/56 |
| L7-seed1 | 100 | 86 | 2.3 MB | 6 ms | 0.7 MB | 7 MB | 2.46 / 3.46 | 384,672 | 2.58 / 3.83 | 2,676,437 | 134/134 |
| L7-seed1 | 1,000 | 499 | 2.6 MB | 8 ms | 1.2 MB | 9 MB | 7.04 / 11.42 | 137,026 | 7.42 / 12.21 | 1,013,250 | 134/134 |
| L7-seed1 | 10,000 | 4,144 | 5.7 MB | 21 ms | 4.4 MB | 26 MB | 32.92 / 40.08 | 31,286 | 33.96 / 42.54 | 240,578 | 134/134 |
| L7-seed1 | 100,000 | 24,299 | 36.9 MB | 147 ms | 34.0 MB | 151 MB | 166.75 / 259.88 | 5,989 | 169.67 / 223.63 | 48,139 | 134/134 |

Python, the same stores and questions:

| Set | Notes | retrieve(), 1 thread p50 / p95 (µs) | Lookups/s, 1 thread | retrieve(), 8 threads p50 / p95 (µs) | Lookups/s, 8 threads | Same work as Zig, 1 thread p50 / p95 (µs) | Index (objects) | Zig 1 thread against Python's same work, p50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| L6-seed1 | 100 | 436 / 580 | 2,305 | 443 / 14,925 | 2,259 | 101.2 / 126.1 | 0.4 MB | 51x |
| L6-seed1 | 1,000 | 925 / 1,281 | 1,159 | 940 / 41,355 | 1,167 | 572.7 / 713.8 | 3.3 MB | 89x |
| L6-seed1 | 10,000 | 2,162 / 2,720 | 495 | 2,329 / 68,061 | 491 | 1,677.4 / 2,279.5 | 26.1 MB | 53x |
| L6-seed1 | 100,000 | 13,731 / 19,566 | 79 | 83,344 / 204,682 | 86 | 12,140.2 / 16,509.2 | 223.4 MB | 83x |
| L7-seed1 | 100 | 501 / 829 | 1,891 | 501 / 25,955 | 1,937 | 135.3 / 150.7 | 0.3 MB | 55x |
| L7-seed1 | 1,000 | 960 / 1,168 | 1,051 | 996 / 49,573 | 1,001 | 604.4 / 771.4 | 3.2 MB | 86x |
| L7-seed1 | 10,000 | 2,310 / 3,012 | 433 | 18,739 / 22,008 | 422 | 1,874.3 / 2,615.0 | 26.3 MB | 57x |
| L7-seed1 | 100,000 | 12,503 / 16,243 | 82 | 92,811 / 205,261 | 78 | 12,591.1 / 17,154.8 | 223.5 MB | 76x |

- **Zig looks up in 2 to 2.5 µs at 100 notes and 146 to 167 µs at 100,000** (p50, one thread; p95 at most 260 µs). The time follows the candidates: at 100,000 notes the commonest n-grams ("my", "is") are in most of the 24,000 current notes, and each of them gets a score.
- **8 threads give 7.0 to 8.0 times the lookups per second of one** (6,000 to 7,300 a second with one thread at 100,000 notes, 48,000 to 56,000 with eight), with p50 within 1.12 times of one thread's. The Mac has 18 cores; 8 threads were measured, as asked. Python's 8 threads give about as many lookups per second as one (within 10% in every row: the interpreter lock), and their p95 is mostly waiting for it.
- **Against Python doing the same work** (keys, n-grams, `_lookup`, `pack` and `render` from the token ids, without the tokenizer), Zig is 51 to 89 times faster at p50. Against `retrieve()` as the runner calls it (which also encodes the question and counts the whole block with the tokenizer), 68 to 218 times.
- **Load and memory, apart from the lookups.** Loading parses the JSON and rebuilds the postings: 6 ms at 100 notes, 146 to 147 ms at 100,000 (reading the file is 0.2 to 3.5 ms of it). The index's own structures take 0.7 to 0.8 MB at 100 notes (the vocabulary alone is 131,072 ids at 5 bytes, 0.66 MB) and 34.0 MB at 100,000; the arena that holds them has 80.6 MB reserved at 100,000, and the process's peak resident memory, which also held the JSON text and its parse, was 151 MB. Python's index at 100,000 notes, counted object by object (as the sweep counts it): 223.4 MB.
- Exporting the 100,000-note store from Python took 3.8 s (each line's cost needs the tokenizer).
- **Failures stop the bench** (code commit `f54344f`, after these timings; the successful runs did not change). A timed lookup that differs from its first pass makes `notes-zig bench` exit 2, and `notes_zig_bench.py` stops when that happens or when Zig differs from `retrieve()` on a question other than by a float-order tie (the golden test's rule; a row with such a tie would be marked and its timings not claimed). In the recorded rows every timed lookup matched and every question agreed. A failed worker setup or thread start now stops the bench with an error instead of leaving it waiting: every started worker counts itself, set up or not, and on any error the others are told to stop and are joined. `notes_zig_bench.py --failure-controls` (MEASURED, `docs/notes-raw/zig-bench-failures.txt`): worker 0, 4 or 7's allocations failing and starting worker 0, 4 or 7 failing each end in an error (exit 2) in 0.01 s; one timed result taken as wrong (`--corrupt-one`) exits 2; a check against Python with a broken lookup stops the run (47 of 56 questions differ, not by float order); and the old start with a failed setup (`--legacy-barrier`, the control) still hangs, caught by the 5-second limit.
- These numbers are from code commit `101b1de`, before `exclude` was ported. A spot check with `3f0855a` (exclude empty, as here) at 100 to 10,000 notes gave the same within the run-to-run spread: at 10,000 notes p50 31.67 and 32.42 µs with one thread against 31.92 and 32.92 above (`docs/notes-raw/zig-bench-spot.txt`, MEASURED). The 100,000-note rows were not rerun.

## What would come next (not done)

- Make Python's sums deterministic (sorted n-gram order), so the float category is empty by construction.
- A serving engine would call `lookup` in-process with the ids it already has; this port reads a JSON snapshot that Python writes. Keeping the index in step with writes (or porting writes) is open.
