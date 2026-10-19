# Results behind the paper

This page collects the measured numbers of the paper *Fast to Write, Slow to Learn* (Hart and Pennanen, draft of 19 October 2026) and names the file in this repository that each one comes from. The tables are copied from those files. This page adds no new measurements.

Some numbers in the paper are not backed by a file in this repository yet. They are listed at the end, under [Not yet backed by a file here](#not-yet-backed-by-a-file-here), with where they may come from.

The source files grade their own numbers: **MEASURED** means we ran it, **READ** means we read it in code or a document, and **INFERRED** means our own reading, not checked by a run.

## The test suite (paper Section 6 and Appendix A)

The test suite is in [../tests/](../tests/): the rule book in Python and in Lean, the test generator, the judge, the runner and the intake check.

| Paper | Source |
| --- | --- |
| The Python rule book passes 102 self-checks | [../tests/README.md](../tests/README.md), Status table, row "Rule book, generator, judge, intake": "102 of 102 checks" |
| The Lean copy states 93 theorems, all checked by Lean 4.35.0 | [../tests/README.md](../tests/README.md), Status table, row "Lean rules"; the proofs are in [../tests/lean/LwLogic.lean](../tests/lean/LwLogic.lean) |
| The two copies agree on all 5,910 expected answers of the example sets | [../tests/README.md](../tests/README.md), Status table, row "Lean checks of the example sets": "5,910 of 5,910 pass" |

We ran the moved self-checks again in their new place on 19 October 2026 (MEASURED; the full list is at the end of the Status section of [../tests/README.md](../tests/README.md)). The Python rule book now passes **104 of 104** checks, two more than the paper's 102. Lean 4.35.0 proves all 93 theorems and passes all 5,910 checks of the 18 example sets.

## The note store, measured offline (paper Sections 4.6 and 4.7, and Table 1)

These are measurements of the Python prototype of the note store, with Nemotron 3.5 Lightning's own tokenizer and no model, on an Apple M5 Max MacBook Pro.

**Finding the needed notes on the example test sets.** Source: [NOTES-LAYER.md](NOTES-LAYER.md), section "Recall on the example sets". Raw data: [notes-raw/recall-L1-L6.json](notes-raw/recall-L1-L6.json) and [notes-raw/recall-L7.json](notes-raw/recall-L7.json).

| Question type | Asked | Recall at cap | Rule book on retrieved | Tokens p50 / p95 | False per question | With any false |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **all** (levels 1 to 6) | **994** | **714/726 (98%)** | **868/880 (99%)** | **295 / 431** | **13.8** | **932/994** |

On level 7 the same source gives recall at the cap of 492 of 492, with 471 tokens at the median. Of the 12 misses on levels 1 to 6, 7 share no token with their note and 5 are chains of ownership (same section).

**Growing the store with distractor notes.** Source: [NOTES-LAYER.md](NOTES-LAYER.md), section "The store-size sweep". Raw data: [notes-raw/sweep.json](notes-raw/sweep.json), [notes-raw/sweep-subject0.json](notes-raw/sweep-subject0.json), [notes-raw/sweep-value-hop.json](notes-raw/sweep-value-hop.json) and [notes-raw/sweep-misses.json](notes-raw/sweep-misses.json).

| Set | Notes in the store | Current / earlier | Recall at cap (defaults) | Without the subject match | With the subject+value hop |
| --- | ---: | ---: | ---: | ---: | ---: |
| L6-seed1 | 100 | 60 / 40 | 42/45 (93%) | 42/45 (93%) | 43/45 (96%) |
| L6-seed1 | 1,000 | 472 / 528 | 30/45 (67%) | 28/45 (62%) | 34/45 (76%) |
| L6-seed1 | 10,000 | 4,113 / 5,887 | 27/45 (60%) | 24/45 (53%) | 28/45 (62%) |
| L6-seed1 | 100,000 | 24,262 / 75,738 | 27/45 (60%) | 21/45 (47%) | 28/45 (62%) |
| L7-seed1 | 100 | 86 / 14 | 128/128 (100%) | 128/128 (100%) | 128/128 (100%) |
| L7-seed1 | 1,000 | 499 / 501 | 123/128 (96%) | 108/128 (84%) | 123/128 (96%) |
| L7-seed1 | 10,000 | 4,144 / 5,856 | 115/128 (90%) | 80/128 (62%) | 115/128 (90%) |
| L7-seed1 | 100,000 | 24,299 / 75,701 | 111/128 (87%) | 70/128 (55%) | 111/128 (87%) |

**What the Python prototype costs.** Source: [NOTES-LAYER.md](NOTES-LAYER.md), section "The store-size sweep", the cost table. Raw data: [notes-raw/sweep.json](notes-raw/sweep.json).

| Set | Notes | Lookup p50 / p95, 1 thread (µs) | 8 threads, one store (µs) | 8 threads, 8 owners (µs) | Lookups per second, 8 threads one store (1 thread) | Index (MB) | Notes (MB) | Write p50 / p95 (µs) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| L6-seed1 | 100 | 420 / 480 | 430 / 18,032 | 444 / 23,432 | 2,397 (2,428) | 0.4 | 0.2 | 212 / 499 |
| L6-seed1 | 1,000 | 964 / 1,344 | 940 / 43,792 | 960 / 44,326 | 1,164 (1,074) | 3.3 | 1.6 | 260 / 702 |
| L6-seed1 | 10,000 | 1,956 / 2,639 | 2,261 / 70,838 | 2,880 / 83,843 | 500 (528) | 26.1 | 16.3 | 297 / 964 |
| L6-seed1 | 100,000 | 7,834 / 11,837 | 63,457 / 86,770 | - | 124 (135) | 223.4 | 163.1 | 775 / 12,117 |
| L7-seed1 | 100 | 452 / 542 | 471 / 15,383 | 477 / 28,268 | 2,118 (2,181) | 0.3 | 0.2 | 151 / 360 |
| L7-seed1 | 1,000 | 943 / 1,079 | 1,026 / 36,142 | 997 / 55,562 | 1,061 (1,079) | 3.2 | 1.6 | 240 / 699 |
| L7-seed1 | 10,000 | 2,118 / 2,813 | 2,308 / 79,681 | 3,194 / 128,186 | 459 (475) | 26.3 | 16.3 | 290 / 933 |
| L7-seed1 | 100,000 | 8,641 / 11,252 | 67,684 / 111,417 | - | 116 (118) | 223.5 | 163.1 | 834 / 12,040 |

The same section gives the memory as "about 3.9 KB per note in all" at 100,000 notes (Table 1 of the paper). Table 1 also gives the prototype's write time as "0.2 to 0.8 ms". The write medians above run from 151 to 834 µs, so the paper's lower end rounds up the smallest values.

**The Zig lookup, one thread against eight.** Source: [NOTES-ZIG.md](NOTES-ZIG.md), section "Results: speed". Raw data: [notes-raw/zig-bench.json](notes-raw/zig-bench.json). The paper's Section 4.7 says the Zig version does "7 to 8 times as many lookups per second on eight threads as on one". The source says "8 threads give 7.0 to 8.0 times the lookups per second of one".

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

**A cache that kept an erased note (paper Section 4.6).** The paper says the erase self-test fails when run against the old cache. Source: [NOTES-LAYER.md](NOTES-LAYER.md), "How erasure is checked": on commit `d831eb7` the same walk found the erased notes "in 10 places after the delete".

## First fact tests (paper Section 7.2)

Source: [RESULTS.md](RESULTS.md), "Headline findings". The run records themselves are not in this repository (RESULTS.md says why).

| Paper | Source in RESULTS.md |
| --- | --- |
| On the Mac a single fact was learned in 218 s | Finding 1: "learned in 218.4 s, including an 18.4 s save" |
| and recalled 3 of 3 times | Finding 1: "blue, 3 of 3, thinking off and on" |
| Zeroing its neurons gave back the original model word for word in 24 of 24 answers | Finding 2: "All 24 answers (12 questions, thinking off and on) were identical to the original model's" |
| On the GB10, teaching the same fact failed 4 times in 6 tries | Finding 3: "4 misses in 6 tries" |
| A confident yes to values it was never taught | Finding 1: "Is my favourite colour purple?" got a yes |
| Take-backs were not retired | Finding 4 |
| Facts that look alike were confused | Split scores, point 7, "Neighbourhood failures, all in L4" |
| The control questions held at 5 of 5 in every phase | Finding 7: "5 of 5 in every phase", and 6 of 6 in the level 4 set |

## Corrections to the grading rules (paper Section 9)

Source: [rejudge/README.md](rejudge/README.md). The paper says the fixes "changed the grade of 51 of 1,470 answers". The source counts 1,470 answers and says "51 verdicts changed", each listed with its answer in [rejudge/CHANGES.md](rejudge/CHANGES.md). A later round of fixes, over 2,190 answers, is in [rejudge/misreads/](rejudge/misreads/).

## Promotion (paper Sections 5 and 7.8)

**How promotion decides.** [PROMOTION.md](PROMOTION.md) describes the first build of promotion in the test suite's Python code (`tests/zengine/promote.py`): a fact is due after `--hold` days right (default 2), and its note leaves the prompt after `--alone` nights answered without it (default 2). The longest run in the paper used the promotion built into our TensorFold build. Its saved state records `hold` 1 and `alone` 2 ([receipts/e1-no-notes-control/validation/taught-copy.run6-promote-before.json](receipts/e1-no-notes-control/validation/taught-copy.run6-promote-before.json)).

**The longest run (called `run6`).** The paper says the server "taught 15 facts to the neurons in its first six nights, and the notes of 5 of them left the prompt and stayed out". Source: the run's saved promotion state, copied before the E1 test, [receipts/e1-no-notes-control/validation/taught-copy.run6-promote-before.json](receipts/e1-no-notes-control/validation/taught-copy.run6-promote-before.json), field `nights`. The rows below are copied from it; "pocket" is the number of facts in the neurons after the night, and "in_weights" the number of facts whose note had left the prompt.

| night | due | taught | kept | missed | worse | rejected | left_prompt | in_weights | pocket |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 10 | 10 | 7 | 3 | 0 | 0 | 0 | 0 | 7 |
| 2 | 11 | 11 | 8 | 3 | 2 | 1 | 0 | 0 | 0 |
| 3 | 22 | 22 | 17 | 5 | 0 | 0 | 0 | 0 | 17 |
| 4 | 6 | 34 | 24 | 10 | 3 | 2 | 0 | 0 | 0 |
| 5 | 16 | 16 | 14 | 2 | 0 | 0 | 0 | 0 | 14 |
| 6 | 1 | 1 | 1 | 0 | 0 | 0 | 3 | 3 | 15 |
| 7 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 4 | 15 |
| 8 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 5 | 15 |
| 95 (the last) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 15 |

**Do the neurons answer without the notes?** Source: [receipts/e1-no-notes-control/tables.md](receipts/e1-no-notes-control/tables.md); the receipt's [README](receipts/e1-no-notes-control/README.md) explains the arms. "No notes" is arm A0; "with its notes" is arm A1x, the server's block with the fact's own note and the 5 promoted notes left out.

| Paper (Section 7.8) | Source in tables.md |
| --- | --- |
| The taught model answered all 14 questions about the 5 promoted facts right, with no notes | Q_gate table, "taught A0" column, the five rows marked "(in weights)": 2/2, 3/3, 4/4, 2/2 and 3/3 |
| The untaught copy answered none | the same rows, "untaught A0" column: 0/2, 0/3, 0/4, 0/2 and 0/3 |
| Of all 15 taught facts, the neurons answer 12 with no notes | Totals, row "taught\|gate\|A0", facts passing: 12 |
| On the 56 questions, 36 with its notes and 22 without them | Totals, rows "taught\|q56\|A1x": 36 and "taught\|q56\|A0": 22 (both judge versions agree) |

## Erasing a promoted fact on the GPU (paper Sections 4.5 and 5.4)

Source: [receipts/erase-gpu-2026-10-19/README.md](receipts/erase-gpu-2026-10-19/README.md). This run is newer than the paper's text. Section 5.4 says that so far only the server's own tests, with a stand-in model and no GPU, check that erasing a note also takes its fact out of the neurons and keeps it out after a restart. This receipt checks it once on a GB10:

| | A, before | B, after erase | C, after restart |
| --- | ---: | ---: | ---: |
| The 13 questions about the erased fact that name "Bruno" | 13 | 0 | 0 |
| The 11 questions about the four other facts the neurons answer, right | 11 | 10 | 10 |

## Not yet backed by a file here

The paper states the numbers below, but no file in this repository holds them yet. "Candidate" names where the number appears elsewhere, found by searching our files on 19 October 2026. A candidate is where to look, not a check that it backs the number.

**Table 1, our rows.**

- TensorFold Notes in the engine: "about 434 bytes per note at 1,000,000 notes" and "8.2 times the note text". Candidate for 434 bytes: the cost report `NOTES-COST.md` on a branch of the test suite's working repository (not public), which measured 433.7 bytes per note on the build this repository pins. We found no file with "8.2 times".
- "3.9 to 4.8 ms median per write in the engine's write benchmark, plus an index merge of about 3.5 s every ~1,000 writes at 1,000,000 notes". The same `NOTES-COST.md` says these come from a benchmark run on 17 October whose raw files are only on a local drive, made with a binary that merges on a separate thread. It reports that on the build this repository pins, a write over HTTP takes 8.9 ms at 1,000 notes and 71.1 ms at 1,000,000 (median), and the merging write 1.9 to 7.6 s at 1,000,000.
- "a fact is a median 7 tokens": no source found.
- "lookup 0.03 ms at 1,000 notes to 5 ms at 1,000,000 notes in the engine" (also Sections 4.2 and 7.5: 0.032 ms, 0.72 ms at 100,000, 4.96 ms): `NOTES-COST.md` names the same local benchmark as their source. On the pinned build it measured 0.025 ms and 2.96 ms in process, one thread. We found no source for 0.72 ms.
- Living Weights: "180 s median on a GB10", "decode 102 tok/s", "35 of 40 still right after 64 lessons". Candidates: `perf/LESSON-TIME.md` and `perf/CUDA-POCKET.md` on the `run-files` branch of the working repository (not public).
- Everything in the prompt: "207 tokens for 29 facts, 436 for 64" and "L6: 43 of 56 right": no source file here; the L6 reference runs are on the `run-files` branch.

**Section 3.** Box 1's values for the neuron of "I like blue." (z = +1.54, a = 2.36; z = -1.22 for "Tell me a joke."), and Section 3.4's "13.8% of decode speed": no source found. The 13.8% is the drop from 102.3 to 88.2 tok/s listed under Section 7.5 below.

**Section 4.2.** On 1,568 test questions, the server built the same notes blocks as the Python version, byte for byte: candidate `NOTES-IN-ENGINE.md` in our public TensorFold fork at the pinned commit, linked from [provenance.md](provenance.md).

**Section 4.3 and Figure 6.** "468 tokens median ... against 418 without the hop", "19 of 56 blocks ... differ" and "8 of 56" replies paused: candidate `NOTES-DECODE.md` in the same public fork. The Figure 6 counts (23, 28 and 32 of 45; 29, 29 and 28 of 56; 26, 29 and 29 at 1,000 notes; 3, 5 and 5 of 16), the 28 hand-built two-step questions (1 to 27, 0 to 28), "about 0.2 to 0.3 s per added note" and the twenty unrelated questions: no source found. `NOTES-COST.md` says the Figure 6 runs used stores on a local drive.

**Section 7.3** (test set L7: 40 learned, 20 missed, 4 replaced; 35 and 29 of 64), **Section 7.4** (the L6 table; "only 1 of the 16 two-step questions"), **Section 7.6** (8 of 26; 3 of 17 against 17 of 17) and **Section 7.7** (the per-set table; 148 of 182 and 77 of 81 against 138 and 79; 39 against 43 of 56 with notes plus neurons): candidates are the run folders on the `run-files` branch of the working repository (not public), which hold machine names and home paths.

**Section 7.5.** 179.5 s over 306 lessons; 102.3 to 88.2 tok/s; 102.1 tok/s with the fix; 2.0% against 12.6% on a 4,096-token prompt; the RTX PRO 6000 check within 0.3% (philip-pentatonic, TensorFold issue #678): candidates `perf/LESSON-TIME.md` and `perf/CUDA-POCKET.md` on the `run-files` branch.

**Section 7.7, after the measurements.** Names and "I", "my", "me" raising right answers from 28 to 34 of 56; the new block's 62 of 86 against 43; German 11 to 24 of 45 and Finnish 16 to 20: no source file found.

**Section 7.8.** The brother's amber eyes spreading to other answers, the earlier shorter runs, and the learner bug and its first fix: no source file here.

**Appendix B.** "Raw run files (every answer, its grade, and the server and run logs)": only the two receipts in [receipts/](receipts/) are here. The other run files are on the `run-files` branch of the working repository (not public).

**Sections 2 and 9.** "32 of 32 unrelated prompts on Metal and 28 of 32 on CUDA gave exactly the same answers" after promotion, and "when we repeated three lessons on 1.0.9, each part of a lesson took the same time as on 1.0.8 to within 1 s": no source here. Candidate for the second: `perf/LESSON-TIME.md` on the `run-files` branch.
