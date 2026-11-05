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

## The note store, measured offline (paper Sections 4.2 and 4.7, and Table 1)

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

Source: [RESULTS.md](RESULTS.md), "Headline findings". The GB10 run records behind these findings are in [runs/runs/](runs/runs/), folders `gb10-a-L1-seed1` (with `gb10-a-L1-seed1-teach2` and the later repeats `-r2` to `-r5`), `gb10-a-L2-seed1`, `gb10-a-L3-seed1` and `gb10-a-L4-seed1` ([runs/README.md](runs/README.md) explains the layout). The Mac's run records are not in this repository.

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

## The raw run files (paper Sections 7.3 to 7.7)

[runs/](runs/) holds the raw files of the GPU runs and cost measurements: every answer, its grade, the judge reports, the run summaries, the server and run logs, and the scripts. They were cleaned for privacy only; [runs/README.md](runs/README.md) lists the folders, which paper section each one backs, and what was replaced. The GB10 is called `gb10-a` there. File names in backticks below are relative to [runs/runs/](runs/runs/); many of those files are stored once under `_shared/`, and [runs/runs/MANIFEST.tsv](runs/runs/MANIFEST.tsv) gives each one's place.

**Filling the pocket (paper Section 7.3, test set L7).** Source: [runs/runs/gb10-a-L7-seed1/capacity.txt](runs/runs/gb10-a-L7-seed1/capacity.txt), which says "L7-seed1: 64 facts; learned 40, missed 20, server: replaced 4". Its retention table, at the end (64 lessons):

| Paper | Source row "64 teach" |
| --- | --- |
| 35 of 64 right with thinking off, 29 with it on | CURRENT 35/64 (off), 29/64 (on) |
| 35 of the 40 held facts still answer | held 35/40 (off); 29/40 (on) |
| Yes/no right in 30 of 64 and 19 of 64 | yes/no 30/64 (off), 19/64 (on) |
| Controls 5 of 6 and 4 of 6 | controls 5/6 (off), 4/6 (on) |
| The held facts of the first ten lessons went from 5 of 5 to 4 of 5 | "First 10 facts, held ones only (CURRENT): thinking off: ... 5/5 4/5 (drops)" |

The precision of "don't know", 2 of 36 with thinking off, is in [runs/runs/gb10-a-L7-seed1/combined/report-kept.txt](runs/runs/gb10-a-L7-seed1/combined/report-kept.txt), teach phase: "abstention P 2/36 R 2/3".

**Two versions of the judge.** Next to each phase's answers there are the verdicts of the judge at run time (`verdicts.json`, summed in `report.txt`, the `.out` files and `sets.log.gz`) and of a later judge (`verdicts-judge2.json`, and `verdicts-judgefix.json`, which passes the same answers). No report or log sums the later verdicts, so the "re-judged" counts below are our own count of those files. They differ on a few answers, mostly where an arm read the boat's new owner in test set L6: the first judge failed "My brother." to "Whose is the boat?" (Q03) and called a "Yes." to "Does the boat belong to my brother?" unclear (Q40); the later judge passes both. A few more flip in some arms (L6 Q32, Q33 and Q36; L5 Q17 with every lesson in the prompt, thinking on). Several of the paper's L6 numbers are the re-judged ones, as marked.

**Two-step questions and the references (paper Section 7.4, test set L6).** Sources: [runs/runs/gb10-a-L6-seed1b/reference/report.txt](runs/runs/gb10-a-L6-seed1b/reference/report.txt) (pocket and exact lookup), [runs/runs/gb10-a-notes/L6-seed1/reference/report.txt](runs/runs/gb10-a-notes/L6-seed1/reference/report.txt) (notes) and [runs/runs/gb10-a-cap/L6-seed1/reference/report.txt](runs/runs/gb10-a-cap/L6-seed1/reference/report.txt) (every lesson in the prompt, thinking on rerun at 6,144 tokens).

| Paper column | Source |
| --- | --- |
| Living Weights: 8/7, 3/5, 0/2, 3/1, 0/0, 8/9; all 22/22 of 56 | the pocket column, row for row; "all off 22/56", "all on 22/56" |
| Exact lookup: 14, 4, 3, 8, 1, 3; all 29 of 56 | the memory column, row for row; also `gb10-a-L6-seed1b-ref.out`: "teach: thinking off 29/56 pass (0 unclear)" |
| TensorFold Notes only: all 43 / 48 of 56 | re-judged (`gb10-a-notes/L6-seed1/phases/teach.notes/verdicts-judge2.json`): every cell of the paper's row, all 43/48. The report gives 42/56 and 47/56 (Q03 and Q40) |
| All lessons in prompt: all 43 / 49 of 56 | re-judged (`gb10-a-cap/L6-seed1/phases/teach.in_context/verdicts-judge2.json`): current values 12/13, all 43/49; two-step 16/19, and/or/not 5/5, yes/no 7/9, negation 1/1 and controls 8/8 as in the report. The report gives all 42/56 and 48/56 |
| The untaught model passed 46 of 56 | `gb10-a-L6-seed1b.out`: "baseline: thinking off 46/56 pass (1 unclear)" |
| Thinking on at 1,500 tokens: 13 of 56 answers ran out with no reply; 1 of 56 at 6,144 | `gb10-a-L6-seed1b/summary.json` (1,500) and `gb10-a-cap/L6-seed1/summary.json` (`max_tokens_on` 6144); counted from the answer files, answers with `finish` "length" and an empty reply: 13 (Q13, Q20, Q21, Q23, Q24, Q28, Q29, Q30, Q32, Q33, Q34, Q44, Q48), then 1 (Q24) |

The group rows of 24 two-step questions are the report's 16 `two_step` questions plus its 8 `compound` ones (and, or, not).

**Cost (paper Section 7.5 and Table 1's Living Weights row).** Sources: [runs/perf/LESSON-TIME.md](runs/perf/LESSON-TIME.md) and [runs/perf/CUDA-POCKET.md](runs/perf/CUDA-POCKET.md), with their raw files in [runs/perf/lesson-time-raw/](runs/perf/lesson-time-raw/), [runs/perf/cuda-raw/](runs/perf/cuda-raw/) and [runs/runs/tf-pocket-cuda/](runs/runs/tf-pocket-cuda/).

| Paper | Source |
| --- | --- |
| Learning one fact: median 179.5 s over 306 lessons, 117 s at the 10th percentile and 319 s at the 90th ("180 s median" in Table 1) | LESSON-TIME.md, row "Whole lesson": "179.5 (117.0 to 318.5; 93.8 to 571.6)", median (p10 to p90; min to max), over "all 306 lessons"; its summary says "median 180 s" |
| The shares 44.2%, 26.1%, 14.3%, 10.1% and 6.5% | LESSON-TIME.md, the same table, column "Share" |
| Saving rewrites 263 MB; the first save 18.5 GB | LESSON-TIME.md ("263 MB", "18.5 GB") |
| Undrafted decode 102.3 tok/s untaught, 88.2 taught, 88.1 empty; 102.1, 102.1 and 102.4 with the fix ("decode 102 tok/s" in Table 1) | CUDA-POCKET.md: "Drafts off, 64 facts: 88.2 to 102.1 tok/s in the dev CLI (1.158x) ... Empty pocket: 88.1 to 102.4 (1.163x) ... Untaught: unchanged (0.998x and 1.000x)", and "102.3 in section 3.3" for untaught |
| The pocket cost about 1.6 ms a token, about 0.045 ms after the fix | CUDA-POCKET.md: "of 1.0.8's +1.6 ms a token, about 0.045 ms is left with 64 facts and nothing with the empty pocket" |
| On 1.0.9 the fix gave 1.17x in the dev CLI and 1.16x on the server | CUDA-POCKET.md: "Undrafted, the fix gives 1.17x in the dev CLI (1.0.8: 1.16x) and 1.16x on the server" |
| A 4,096-token prompt went from 12.6% to 2.0% slower than untaught (1.0.9) | CUDA-POCKET.md: "4,096-token prompt with 64 facts: 2.0% over untaught after, against 2.8% on 1.0.8; before the fix 12.6% (1.0.8: 10.5%)" |

Section 3.4's "13.8% of decode speed" is (102.3 - 88.2) / 102.3 from these numbers. CUDA-POCKET.md's own figure for the undrafted cost is "13.6% (+1.56 to +1.59 ms a token)", from another instrument.

**Question forms v2 (paper Section 7.6, thinking off).** Each phase's `report.txt` splits the answers by the right answer (v = a value, n = "no", ? = "don't know"):

| Paper | Source |
| --- | --- |
| The pocket: 8 of 26 "no" in test set L4 | `gb10-a-v2/L4-seed1-v2/phases/teach/report.txt`: "n 8/26 (1 dk)" |
| Every lesson in the prompt: 3 of 17 "don't know" in test set L2 | `gb10-a-v2/L2-seed1-v2/phases/teach.in_context/report.txt`: "v 6/6  n 4/5  ? 3/17" |
| The exact lookup: 17 of 17 | `gb10-a-v2/L2-seed1-v2/phases/teach.replacement_memory/report.txt`: "? 17/17" |

**TensorFold Notes only, on the GPU (paper Section 7.7).** The per-set table's cells, thinking off, are in each phase's `report.txt`: the notes column in `gb10-a-notes/<set>/phases/teach.notes/`, the pocket in `gb10-a-v2/<set>/phases/teach/`, every lesson in the prompt in `gb10-a-v2/<set>/phases/teach.in_context/` (for L6, the pocket in `gb10-a-L6-seed1b/phases/teach/` and the prompt in `gb10-a-cap/L6-seed1/phases/teach.in_context/`). For example L5, notes: "v 37/50 (3 dk)  n 23/23  ? 3/4". Every cell matches, except the L6 values: the reports give 34 of 41 for notes and for the prompt, the re-judged verdicts 35 (Q03). No file states these:

- The prompt-token medians. We counted them as the median of `usage.prompt_tokens` in each phase's `answers.jsonl`: notes 91, 99, 127, 194, 130 and 351.5 for L1 to L6 (the paper says 351), every lesson in the prompt 74, 94, 106, 169, 324 and 375.
- The totals, which are sums over L1 to L6: "no" 77 of 81 (notes) and 79 (prompt) from the reports; values 148 and 138 of 182 only with the re-judged L6 (the reports give 147 and 137).

L6 as a whole, notes only: 43 / 48 of 56 re-judged; the report and `gb10-a-notes/sets.log.gz` give 42/56 and 47/56. The neg set's "don't know": `gb10-a-notes/neg-seed1-v2/phases/teach.notes/report.txt`, "? 3/13" with thinking off and "? 9/13" with it on.

**Notes plus Living Weights (paper Section 7.7).** Source: [runs/runs/gb10-a-both/](runs/runs/gb10-a-both/), run from 14 October 2026 21:07 UTC to 15 October 00:25 UTC. From `sets.log.gz` (thinking off / on): Living Weights only "teach 22/56 | 22/56" (L6) and "38/77 | 30/77" (L5); notes plus Living Weights "teach.notes_pocket 59/77 | 62/77" (L5) and "39/56 | 46/56" (L6; 39 / 47 re-judged, as in the paper); notes only on L5 63 / 62 (from `gb10-a-notes`). The two-step questions: `gb10-a-both/L6-seed1/reference/report.txt`, "two_step off ... 16/24  13/24" (notes, then notes plus neurons). The "no" questions of L5: "n 21/23" in `gb10-a-both/L5-seed1-v2/phases/teach.notes_pocket/report.txt` against "n 23/23" with notes only.

**Stating a take-back more strongly (paper Section 7.7).** Sources: [runs/runs/gb10-a-tb-explicit/](runs/runs/gb10-a-tb-explicit/) (the clearer label) and [runs/runs/gb10-a-tb-rule/](runs/runs/gb10-a-tb-rule/) (label plus a line), against `gb10-a-notes`. The 175 questions are the sets neg (19), neg2 (23), L5 (77) and L6 (56). Summed from the `teach.notes` lines of each `sets.log.gz`, thinking off / on:

| Arm | Logs | Re-judged | Paper |
| --- | ---: | ---: | ---: |
| As first measured (`gb10-a-notes`) | 130 / 141 | 131 / 142 | 131 / 142 |
| Clearer label (`gb10-a-tb-explicit`) | 136 / 134 | 137 / 137 | 137 / 137 |
| Label plus a line (`gb10-a-tb-rule`) | 138 / 148 | 139 / 150 | 139 / 150 |

The "don't know" answers after a take-back, thinking off, from each `teach.notes/report.txt`: neg 3/13 and neg2 9/16 (notes), 9/13 and 12/16 (clearer label), 11/13 and 14/16 (label plus a line), as in the paper. The L6 values lost with the added line, the paper's 35 to 29 of 41, are 34 to 28 in the reports ("v 28/41 (6 dk)" in `gb10-a-tb-rule/L6-seed1/phases/teach.notes/report.txt`) and 35 to 29 re-judged.

**Repeating three lessons on 1.0.9 (paper Sections 2 and 9).** LESSON-TIME.md: "On the same three facts the GPU trainer parts match within 1 s (output fitting 47.2 against 46.4 s, 29.2 against 29.2, 37.0 against 36.7; MEASURED); the CPU fit differs by up to 1.6 s." So the paper's "each part of a lesson took the same time ... to within 1 s" holds for the GPU parts; the CPU fit differed by up to 1.6 s.

## Records added on 23 October 2026

These records back numbers that this list named as missing on 19 October. They are in [receipts/paper-records-2026-10-23/](receipts/paper-records-2026-10-23/), or in our public TensorFold fork at a fixed commit.

- **Note store cost (Table 1, Sections 4.2 and 7.5).** [NOTES-COST.md](receipts/paper-records-2026-10-23/NOTES-COST.md), measured on the M5 Max on 18 October 2026. It has 434 bytes per note at 1,000,000 notes. A write takes 8.9 ms at 1,000 notes and 71.1 ms at 1,000,000 (median). The index merges every 1,025 writes, and the merging write takes 2.5 s (median) at 1,000,000. One lookup in process, on one thread, takes 0.025, 0.094, 0.50 and 2.96 ms (median) at 1,000, 10,000, 100,000 and 1,000,000 notes.
- **German and Finnish lookup (Section 7.7).** [STEM-LOOKUP.md](receipts/paper-records-2026-10-23/STEM-LOOKUP.md).
- **The exact undo on a GPU (Section 5.5).** [exact-undo-gpu/](receipts/paper-records-2026-10-23/exact-undo-gpu/), run on the GB10 on 19 October 2026. With the exact undo, all four weight files have the same hash after a refused night and 24 of 24 replies are the same. With zeroing, three of the four weight files change and 0 of 24 replies stay the same.
- **The same blocks as the Python version, byte for byte, on 1,568 questions (Section 4.2).** [NOTES-IN-ENGINE.md](https://github.com/ThinkOffApp/TensorFold/blob/2c9e759964fd2af1b47d79fde04b77ee564f2e54/NOTES-IN-ENGINE.md), line 12.
- **The value hop (Section 4.3): 468 against 418 tokens, 19 of 56 blocks differ, 8 of 56 replies pause.** [NOTES-DECODE.md](https://github.com/ThinkOffApp/TensorFold/blob/2c9e759964fd2af1b47d79fde04b77ee564f2e54/NOTES-DECODE.md), lines 28, 31 and 170.
- **Unrelated prompts after promotion (Sections 2 and 9): 32 of 32 on Metal, 28 of 32 on CUDA.** [PROMOTION-IN-SERVER.md](https://github.com/ThinkOffApp/TensorFold/blob/7824f05dc3a2f5266b029637801d3c0f61466999/PROMOTION-IN-SERVER.md) on branch `promotion-alone-check`, lines 276 and 283. On Metal the 32 replies are the same before and after promotion. On CUDA, 28 of 32 replies equal the untaught model's, and the other 4 part after 76 to 166 characters with the same answer.

In NOTES-COST.md and STEM-LOOKUP.md we replaced one mention each of the test suite's old working name. Nothing else in them is changed.

## Not yet backed by a file here

The paper states the numbers below, but no file here holds them yet. Checked against the draft of 23 October 2026.

- **Table 1:** "8.2 times the note text" and "a fact is a median 7 tokens": no source found.
- **Table 1:** "207 tokens for 29 facts, 436 for 64": no source found. The run files give the L6 prompt with all 29 lessons as 372 to 384 tokens a question (median 375, `gb10-a-cap/L6-seed1/phases/teach.in_context/answers.jsonl`). There is no L7 run with every lesson in the prompt.
- **Section 3, Box 1:** the values for the neuron of "I like blue." (z = +1.54, a = 2.36; z = -1.22 for "Tell me a joke."): no source found.
- **Figure 6:** the counts (23, 28 and 32 of 45; 29, 29 and 28 of 56) come from runs whose files are on a local drive. The figure's plot script holds the same numbers. The 28 hand-built two-step questions and "3, 5 and 5 of 16": no source found.
- **Section 7.4:** "Of the 16 two-step questions, only 1 had both of its facts answered right when asked directly". The file says something else. The two-step part of the pocket's teach report (`gb10-a-L6-seed1b/phases/teach/report.txt`, thinking off) counts 10 chains of two steps. 2 of them had both steps right, and the model answered 1 of those 2 right as a chain.
- **Section 7.5:** "With 40 facts taught, a 4,096-token prompt is 2.0% slower". [runs/perf/CUDA-POCKET.md](runs/perf/CUDA-POCKET.md) gives the 2.0% for the pocket of test set L7, which has 64 lessons. The RTX PRO 6000 check within 0.3% is philip-pentatonic's own measurement, in [TensorFold issue #678](https://github.com/ashhart/TensorFold/issues/678), not ours.
- **Section 7.7:** names and "I", "my", "me" raising right answers from 28 to 34 of 56, and the new block's 62 of 86 against 43. These were measured on the M5 Max with a 1,000,000-note store. Their files are on the same local drive as Figure 6.
- **Section 7.8:** the brother's amber eyes spreading to other answers, and the learner bug and its first fix: no file here or on a public branch.
