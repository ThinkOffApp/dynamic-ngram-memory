# What the notes cost: the paper's Section 6.5, run as one set

- **Machine:** Apple M5 Max MacBook Pro, 18 cores, 128 GiB (`hw.memsize` 137,438,953,472), macOS 27.0.1 (26A434), Metal.
- **Model:** TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit, snapshot `d9d758fb83953437f7263256b0d96157e2a348b8` (17.27 GiB of files), from the Hugging Face cache.
- **Engine:** our TensorFold fork, branch `notes-review` at `2c9e759964fd2af1b47d79fde04b77ee564f2e54`, checked out clean at `/Volumes/t705/tf-work/notes-review` and not edited.
- **Build:** Zig 0.17.0, `zig build native notes-live-bench notes-sa-bench notes-size-bench -Dcpu=apple_m1 -Dversion=1.0.11`, output and cache in a separate folder (`-p ~/tmp/notes-cost/out --cache-dir ~/tmp/notes-cost/cache`). The build defaults apply: the engine is ReleaseFast, the HTTP side and the notes code are ReleaseSafe, and so are the three bench programs. `tensorfold-native --version` printed 1.0.11, and the binary's SHA-256 starts `2ae939e9c7b6c135`.
- **Date:** 18 October 2026, 12:49 to 16:57 EEST (Helsinki, UTC+3).
- **Grades:** MEASURED means we ran it here, on this date. READ means we read it in code or a document. INFERRED means our own reasoning, not checked by a run. Every number in the tables below is MEASURED unless its row or note says otherwise.

The paper's Section 6.5 lists these costs: write time; prompt tokens per answer (p50 and p95); lookup time; time to first token; decode speed; and memory, all with 8 lookup threads on a native index. They had never been measured as one set on one build. This page measures all of them on one build, one model and one machine, with the same 56 questions in every arm.

## In short

| Cost | 1,000 notes | 10,000 | 100,000 | 1,000,000 | Grade |
| --- | ---: | ---: | ---: | ---: | --- |
| Write through `/v1/notes`, median (ms) | 8.9 | 10.0 | 24.2 | 71.1 | MEASURED |
| Write, p95 (ms) | 10.0 | 12.0 | 45.8 | 237 | MEASURED |
| Index merge: how often | every 1,025 written notes | same | same | same | MEASURED (and READ) |
| Index merge: the merging write (ms, median) | 11 | 21 | 245 | 2,541 | MEASURED |
| Prompt tokens per answer, notes on, p50 / p95 | 509 / 517 | 512 / 520 | 511 / 521 | 512 / 519 | MEASURED |
| Prompt tokens per answer, notes off, p50 / p95 | 24 / 32 | 24 / 32 | 24 / 32 | 24 / 32 | MEASURED |
| Lookup in the server, one request at a time, p50 / p95 (ms) | 0.15 / 0.46 | 0.37 / 0.70 | 2.0 / 3.9 | 12.1 / 39.7 | MEASURED |
| Lookup in process, 1 thread, p50 / p95 (ms) | 0.025 / 0.054 | 0.094 / 0.146 | 0.50 / 0.85 | 2.96 / 5.16 | MEASURED |
| Lookup in process, 8 threads, p50 / p95 (ms) | 0.030 / 0.058 | 0.074 / 0.116 | 0.32 / 0.52 | 1.53 / 5.91 | MEASURED |
| Time to first token, notes on / off, p50 (ms) | 494 / 85 | 394 / 78 | 346 / 71 | 331 / 69 | MEASURED, on a loaded machine |
| Decode, 256-token answer, notes on (no live fetch) / off (tok/s) | 227 / 233 | 270 / 282 | 286 / 306 | 339 / 266 | MEASURED, drafts on |
| Decode, same answer with drafts off, on / off (tok/s) | 103 / 104 | 97 / 101 | 88 / 87 | 95 / 97 | MEASURED |
| Server RSS after the questions (GiB) | 18.60 | 18.61 | 18.69 | 19.48 | MEASURED |
| Store bytes per note | 1,248 | 526 | 451 | 434 | MEASURED |

Each figure is the median over 3 runs. The spread over the runs is in the tables below.

What stands out:

- **The notes add about 485 prompt tokens to every answer, at every store size.** The block fills its 512-token cap whether the store holds 1,000 notes or 1,000,000. Most of the time to first token with notes is the model reading these tokens, not the lookup.
- **On this build, a write gets slower as the store grows.** A write takes 9 ms at 1,000 notes and 71 ms at 1,000,000 (median). The write that merges the index takes 2.5 s (median) at 1,000,000 notes, and it holds up that request.
- **At 1,000,000 notes the live checks during decode cost real time.** With the server's defaults, a 256-token answer made 88 checks, each one a lookup. Together they took 1.1 to 3.8 s per answer (median 3.4 s), and decode fell from 339 tok/s without them to 70 tok/s. At 100,000 notes the same checks took about 0.2 s and decode did not slow down measurably.
- **The 8-thread figure is not a server option.** The server has no setting for lookup threads (READ, `serve --help` and the code). A lookup runs on the thread of the request that asks for it. The 8-thread figures come from the branch's own bench (`notes-live-bench`), which runs 8 lookup threads on the store with no model loaded.
- **The paper's write numbers do not come from this build.** The 3.9 to 4.8 ms median write and the roughly 3.5 s merge at 1,000,000 notes come from `bench/final1/after.jsonl`. That bench used a different binary, one that runs merges on a separate thread (`"merges_on": "a thread"`). On this build a merge runs inside the write. The "before" binary of the same folder does reproduce here once the files sit on the same drive (section 1).

## How it was run

**Stores and questions.** The stores are the sweep stores of 1,000, 10,000, 100,000 and 1,000,000 notes in `/Volumes/t705/tf-work/stores/sweep-N.snap.json`. Each one holds test set L6-seed1's teach notes for owner `u0`, plus distractor notes grown by the test suite's generator (`zengine/notes_sweep_fast.py`, seed 1, `PYTHONHASHSEED=0`; for the 1,000,000-note store see `/Volumes/t705/tf-work/store1m`). These files are byte-identical to the copies in `stores/recall/` that fig6-live used. The questions are L6-seed1's 56, taken from each store's `sweep-N.ref.json`. None of them is about the past, as `notes.is_past` reads it.

**Server.** `tensorfold-native serve MODEL --host 127.0.0.1 --port 18731 --prompt-cache-gib 0 --context 32768 --keep-warm 0 --no-update-check --notes COPY/sweep-N.snap.json`, with drafts on (the default) and the `sa` index (the default). The prompt cache is off, so every reply reported 0 cached tokens and no answer reuses another's prefill. Each run started its own server on a fresh copy of the snapshot, with no journal. Only that server was stopped (`tools/run.sh`).

**Requests.** Every request used temperature 0, thinking off (`chat_template_kwargs: {"enable_thinking": false}`), `max_tokens` 160 and no streaming. There were three arms:

- **off** sends no notes field. This is the control.
- **on** sends `"notes": true` with the server's defaults. These include the value hop and the live checks during decode (`notes_decode`), which can fetch more notes while the answer is written.
- **on_nofetch** sends `"notes": true, "notes_decode": false`. The lookup block goes in, and no checks run during decode.

Every notes request also sent `"notes_write": false`, so no question was ever written as a note. Per question the order was off, X, off, Y, where (X, Y) is (on, on_nofetch) for even questions and (on_nofetch, on) for odd ones. The second off request only separates the two long prefills: an earlier run found that a long prefill straight after another one runs slower (NOTES-IN-ENGINE.md, READ). Six questions were asked first in all three arms as a warm-up and not kept.

**Runs.** Three runs per store size, each in its own server process. A figure is computed per run (a p50 or p95 over the 56 questions), and we report the median of the three runs with the lowest and highest in brackets. All three runs gave the same answers token for token (the server's token SHA), the same prompt token counts and the same note ids, in every arm and at every size.

**The control.** The off arm ran on the same binary, the same model, the same machine, the same server process and the same questions as the notes arms, interleaved with them. It differs only in the notes fields. The notes arms also answer differently: the off answers have a median of 69 tokens, the notes answers 8 to 10. So decode speed on the answers themselves does not compare like with like. A fixed 256-token probe answer is the comparison that does (section 5).

**Load on the machine.** This Mac was shared, and we could not keep it quiet. Google Drive and Finder each used about one core the whole time. Other agents ran Zig builds and test suites, and the TensorFold.app engine and a Python TensorFold server sat idle. The one-minute load average at each run's start was 3.9 to 8.4 for 1,000 and 10,000 notes, 25 to 87 for 100,000, and 23 to 75 for 1,000,000 (`notes-cost-raw/run-log.txt`). The arms are interleaved within each run, so comparisons between arms of the same run hold. Absolute times, and comparisons across store sizes, carry this load.

**Memory check before every model load.** Free plus inactive plus speculative memory was 76 to 94 GiB before each server start, and the script refuses to start under 43 GiB. The server holds about 19 to 20 GiB resident (below), well under the 110 GB limit.

**Where the files were.** Snapshot copies and journals lived on the internal SSD (`~/tmp`, APFS). `/Volumes/t705` had only 18 GiB free. This matters for write times, because every write syncs its journal to the drive (section 1).

## 1. Write time, and the index merge

**What was done.** After the questions, each run sent 2,100 new notes through `POST /v1/notes`, one at a time on one kept-alive connection: `{"text": "My item K is red."}`, with K running from 0 to 2,099. The server called every one of them `NEW`, so each write added one note. The time is the client's wall time per request, from sending to the full reply. It includes the intake check, the build of the new index version, the journal append with its `F_FULLFSYNC` (which asks the drive to flush its cache), and the HTTP round trip on localhost.

**How the merge shows.** The `sa` index keeps its suffix-array base while at most 1,024 written notes follow it (READ, `notes_store.zig`, `side_max` 1024). The write that adds the 1,025th note builds a new base over all the notes, inside that write and under the writer lock. With one note per write, the merges fall on writes 1,025 and 2,050. Those two writes are each run's slowest at 100,000 and 1,000,000 notes (MEASURED). At 1,000 and 10,000 notes a merge is too cheap to stand out from the drive sync.

| Notes at start | Write p50 (ms) | p95 (ms) | Slowest write that is not a merge (ms) | Merges per 2,100 writes | Merging write (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 8.92 [8.04 to 9.00] | 10.03 [9.99 to 10.18] | 67.6 [60.8 to 108.9] | 2 | 11 [9 to 11] |
| 10,000 | 9.96 [9.88 to 11.98] | 11.99 [11.95 to 18.41] | 71.3 [67.4 to 72.8] | 2 | 21 [19 to 32] |
| 100,000 | 24.16 [19.01 to 27.41] | 45.75 [27.56 to 57.09] | 127.4 [79.3 to 173.7] | 2 | 245 [140 to 426] |
| 1,000,000 | 71.05 [54.09 to 127.27] | 237.13 [65.70 to 288.26] | 484.5 [174.4 to 504.8] | 2 | 2,541 [1,946 to 7,570] |

In the merge column the brackets span all 6 merging writes of the 3 runs.

- **Every write between merges still grows with the store.** Writes take 9 ms at 1,000 notes and 71 ms at 1,000,000 (median). The side buffer keeps the n-gram work small. But a new index version still copies per-note arrays that grow with the store (NOTES-WRITE-ERASE-DESIGN.md, READ; the copying is INFERRED from that design note and was not profiled here).
- **At 1,000,000 notes the merge took 1.9 to 7.6 s, with a median of 2.5 s.** The 7.6 s merge came in the run that started at load average 40.
- **The drive sets the floor.** On the same build, `notes-sa-bench` (3,000 writes, no HTTP) at 1,000 notes took 7.91 ms per write (median of 3 runs) with its files on the internal SSD, and 4.46 ms with them on `/Volumes/t705`. Below about 10,000 notes, the journal sync is most of a write.

**The paper's write numbers, checked against their raw results.** The paper's "3.9 to 4.8 ms" median write and its "index merge of about 3.5 s every ~1,000 writes at 1,000,000 notes" have raw results only in `/Volumes/t705/tf-work/bench/final1` (17 October). That folder holds two runs of `notes-sa-bench` with 3,000 writes and side_max 1,024, both with their files on t705:

- `before.jsonl` comes from a binary that merges inside the write, as this build does. It covers 1,000 to 100,000 notes.
- `after.jsonl` comes from a binary that merges on a separate thread (`"merges_on": "a thread"`). It covers 1,000 to 1,000,000 notes.

The paper's numbers are the `after` rows. Its sa writes take 3.93, 4.15, 4.03 and 4.82 ms at the four sizes, so between merges no write waits. Its 3,527 ms at 1,000,000 notes is `merge_ms`, the time of a merge called on its own, not of a write. We ran the same bench on this build (MEASURED; 3 runs, except 1 at 100,000):

| Notes | Index | Source | Write p50 (ms) | p95 (ms) | Merging write (ms) | Merge on its own (ms) |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 1,000 | sa | this build, files on t705 | 4.46 [4.43 to 4.51] | 5.40 [5.37 to 5.95] | 6.3 [6.2 to 6.4] | 3.9 [3.9 to 4.1] |
| 1,000 | sa | this build, files on the internal SSD | 7.91 [7.38 to 7.98] | 11.92 [9.21 to 11.97] | 9.9 [9.1 to 10.1] | 5.1 [4.5 to 7.0] |
| 1,000 | sa | final1/before (READ, 1 run) | 4.49 | 6.66 | 9.1 | 4.0 |
| 1,000 | sa | final1/after, merges on a thread (READ) | 3.93 | 5.35 | 3.8 | 6.6 |
| 10,000 | sa | this build, internal SSD | 9.03 [9.02 to 9.23] | 10.99 [10.26 to 12.48] | 25.0 [24.0 to 26.1] | 19.9 [19.8 to 29.1] |
| 10,000 | sa | final1/before (READ) | 5.55 | 7.82 | 22.0 | 26.5 |
| 100,000 | sa | this build, internal SSD (1 run) | 17.98 | 20.10 | 180.2 | 185.7 |
| 100,000 | sa | final1/before (READ) | 14.04 | 15.91 | 255.8 | 279.7 |
| 1,000,000 | sa | final1/after, merges on a thread (READ) | 4.82 | 7.42 | 13.4 | 3,527 |

- **They reproduce where the code and the drive match.** At 1,000 notes with the files on t705, this build writes in 4.46 ms against `before`'s 4.49 ms. At 10,000 and 100,000 notes this build is slower than `before` by 3.5 to 4 ms per write, which is about the internal SSD's extra sync time measured at 1,000 notes (INFERRED). Its merges are within the same range as `before`'s.
- **They do not hold for this build at 1,000,000 notes.** The 4.82 ms write and the 3.5 s off-write merge need merges on a separate thread, which `2c9e7599` does not have (READ: `Store.build` and `merge` run in the write, under the writer lock). On this build, at 1,000,000 notes, a write over HTTP takes 54 to 127 ms (median per run), and the merging write takes 1.9 to 7.6 s.
- The 1,000,000-note row of `before` was never run. The postings bench at 1,000,000 notes rebuilds the whole index on every write (about 1.6 s each, `final1/after.jsonl`), so we did not run 3,000 of them either.

**Writes during lookups.** `notes-live-bench` writes its own mix (adds, changes, same-as adds and take-backs, with an erase every 100th write) while 8 threads look up without pause. Files were on the internal SSD.

| Notes at start | Runs | Write p50 (ms) | p95 (ms) | Erase p50 (ms) | 8-thread lookup during the writes, p50 / p95 (us) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 3 | 8.86 [8.85 to 8.87] | 10.08 [9.98 to 10.22] | 62 [60 to 63] | 32.6 / 70.1 |
| 10,000 | 3 | 21.51 [21.19 to 21.57] | 23.71 [22.90 to 23.86] | 124 [118 to 124] | 77.6 / 125.2 |
| 100,000 | 3 | 157 [150 to 166] | 191 [165 to 232] | 806 [756 to 920] | 335 / 562 |
| 1,000,000 | 1 | 2,325 | 3,476 | 10,883 | 1,627 / 3,951 |

At 1,000,000 notes, every write in this mix took seconds. The single run took 38 minutes for 990 writes and 10 erases, so we stopped after one run. The difference from the 71 ms adds above is INFERRED, not measured apart: the erases and take-backs in this mix seem to make later writes build a full new base instead of using the side buffer.

## 2. Prompt tokens per answer

**What was done.** For each answer we took the server's own `usage.prompt_tokens`, not a count of streamed chunks. That count is the tokens the model read before its first answer token, with the chat template included. With the live checks on (arm on), a check that finds a missing note re-reads the prompt with the new block, and the server reports that prompt's size in `notes.decode.injected[].prompt_tokens`. The last two columns add those re-read tokens.

| Notes | Arm | p50 | p95 | Live-fetch re-reads per run (of 56 answers) | Prompt plus re-read tokens, p50 | p95 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1,000 | off | 24 | 32 | | 24 | 32 |
| 1,000 | on | 509 | 517 | 5 | 509 | 1,068 |
| 1,000 | on_nofetch | 509 | 517 | | 509 | 517 |
| 10,000 | off | 24 | 32 | | 24 | 32 |
| 10,000 | on | 512 | 520 | 14 | 512 | 1,073 |
| 10,000 | on_nofetch | 512 | 520 | | 512 | 520 |
| 100,000 | off | 24 | 32 | | 24 | 32 |
| 100,000 | on | 511 | 521 | 11 | 512 | 1,072 |
| 100,000 | on_nofetch | 511 | 521 | | 511 | 521 |
| 1,000,000 | off | 24 | 32 | | 24 | 32 |
| 1,000,000 | on | 512 | 519 | 13 | 512 | 1,074 |
| 1,000,000 | on_nofetch | 512 | 519 | | 512 | 519 |

The counts were the same in all three runs. With notes, every answer reads about 485 more prompt tokens: the block, which `fit()` keeps under its 512-token cap. When a live check re-reads the prompt (5 to 14 answers in 56), that answer reads about 1,070 tokens in all. The re-reads all reported 0 cached tokens, because the prompt cache was off. With a prompt cache, much of each re-read could come from the cache (INFERRED; not measured here).

## 3. Lookup time

**What was done, in the server.** Each notes reply carries the server's own timings for its notes step (`notes`): `lookup_us` covers the lookup and the rendering of the block, and `total_us` the whole step (encoding the question, the lookup, `fit()` counting the block, and putting it into the messages). One request ran at a time, so the server made one lookup at a time. The main run's figures come from the 112 notes replies per run (arms on and on_nofetch, both with the value hop on, the default). After the 2,100 writes, the 56 questions were asked once more (on_nofetch, `max_tokens` 1), on a store with those notes in its side buffer.

| Notes | Candidates (p50) | Lookup and render, p50 (us) | p95 | Whole notes step, p50 (us) | p95 | Whole step after the 2,100 writes, p50 | p95 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 466 | 48 [40 to 68] | 116 [76 to 143] | 145 [95 to 185] | 460 [297 to 590] | 247 [242 to 275] | 449 [375 to 491] |
| 10,000 | 4,107 | 201 [172 to 241] | 380 [273 to 533] | 368 [270 to 454] | 696 [520 to 1,374] | 232 [209 to 349] | 432 [352 to 710] |
| 100,000 | 24,256 | 1,761 [1,555 to 2,169] | 3,432 [2,806 to 9,727] | 2,002 [1,851 to 2,615] | 3,909 [3,281 to 10,240] | 1,294 [1,203 to 1,354] | 2,700 [2,677 to 4,061] |
| 1,000,000 | 85,782 | 11,715 [10,294 to 13,061] | 39,506 [19,676 to 40,529] | 12,057 [10,540 to 13,178] | 39,731 [19,890 to 41,360] | 8,352 [7,024 to 10,317] | 16,404 [13,465 to 17,946] |

A separate pass (three fresh servers per size, the 56 questions asked three times each with `max_tokens` 1 and the live checks off, `extra.jsonl`) asked every question with the value hop on and with it off. The lookup and render took the same time either way: at 1,000,000 notes, 12.0 ms with the hop on and 12.3 ms off (p50). At 100,000 notes, 1.62 and 1.64 ms. The hop is not what makes the server's lookup slow.

**What was done, in process, with 1 and 8 threads.** The server cannot be told to use 8 lookup threads, so the thread counts come from the branch's own benches. They load the model's tokenizer and the store, but no weights, and they call the same `Store.retrieve` the server calls (encode, lookup and render, `fit()`). **1 thread** is `notes-sa-bench`: the 56 questions asked 50 times in turn (20 times from 100,000 notes up), with the first round not counted. **8 threads** is `notes-live-bench` with no writes: 8 threads ask the 56 questions without pause for 2 s.

| Notes | 1 thread, p50 (us) | p95 | 8 threads, p50 (us) | p95 | Lookups per second, 8 threads together |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 25.2 [23.8 to 27.5] | 54.4 [49.8 to 63.0] | 29.5 [29.4 to 30.0] | 58.4 [58.3 to 59.8] | 238,172 [236,093 to 240,754] |
| 10,000 | 94.0 [93.1 to 104.2] | 145.8 [143.7 to 166.3] | 73.8 [73.8 to 74.8] | 115.7 [114.6 to 117.8] | 108,740 [107,770 to 110,008] |
| 100,000 | 497 [495 to 534] | 855 [835 to 861] | 319 [309 to 325] | 517 [504 to 556] | 26,386 [23,136 to 27,434] |
| 1,000,000 | 2,959 [2,679 to 3,574] | 5,165 [4,468 to 5,963] | 1,528 [1,477 to 1,557] | 5,912 [2,540 to 6,982] | 3,916 [3,896 to 5,688] |

- **Eight threads do not slow each lookup.** Per lookup they are no slower than one thread, so the lookups served per second grow about 8 times. At 1,000,000 notes the 8-thread p50 is even below the 1-thread one. These are two programs with different loops (8 threads asking without pause against one round-robin pass), and we did not look into why the p50s differ (INFERRED: different cache warmth).
- **The server's lookup is 4 to 6 times slower than the bench's** at 100,000 and 1,000,000 notes: 12 ms against 3.0 ms at 1,000,000 (p50). The lookup code and store are the same. The cause was not measured. Two things point to the caches. Back-to-back requests with `max_tokens` 1 after the writes were faster (8.4 ms) than lookups between full answers (12 ms). And NOTES-IN-ENGINE.md found the same gap on 15 October at 100,000 notes (395 us in the server against 146 to 167 us standalone). INFERRED.

## 4. Time to first token

**What was done.** For each answer we took the server's own `time_to_first_token`. It starts after the notes step, so we also report it with the step's `total_us` added. The prefill column is the engine's `prefill_seconds`.

| Notes | Arm | p50 (ms) | p95 (ms) | p50 with the notes step | Prefill p50 (ms) |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1,000 | off | 85.0 [77.7 to 93.4] | 103.7 [92.2 to 130.1] | | 84.8 |
| 1,000 | on | 493.6 [422.8 to 516.4] | 562.8 [492.1 to 642.1] | 493.8 | 493.2 |
| 1,000 | on_nofetch | 478.8 [421.5 to 529.1] | 585.5 [512.4 to 617.6] | 478.9 | 478.6 |
| 10,000 | off | 78.3 [74.2 to 79.8] | 103.2 [94.3 to 127.6] | | 75.5 |
| 10,000 | on | 394.4 [378.5 to 447.9] | 491.8 [478.8 to 529.0] | 394.8 | 394.0 |
| 10,000 | on_nofetch | 398.3 [378.6 to 429.9] | 500.6 [482.7 to 574.3] | 399.0 | 397.9 |
| 100,000 | off | 70.5 [64.4 to 73.7] | 79.8 [79.6 to 105.2] | | 67.2 |
| 100,000 | on | 346.3 [335.7 to 348.4] | 396.4 [370.3 to 411.8] | 348.0 | 336.2 |
| 100,000 | on_nofetch | 340.1 [339.3 to 347.7] | 378.5 [359.3 to 418.1] | 342.4 | 339.4 |
| 1,000,000 | off | 68.9 [64.5 to 74.4] | 91.9 [76.0 to 100.1] | | 64.0 |
| 1,000,000 | on | 330.8 [318.5 to 334.3] | 356.7 [354.2 to 368.6] | 337.1 | 322.4 |
| 1,000,000 | on_nofetch | 328.3 [323.6 to 331.2] | 365.4 [356.9 to 384.3] | 339.2 | 320.5 |

- **With notes, the first token comes 260 to 410 ms later** (p50, within a size). Nearly all of that is prefill: about 485 more tokens to read. The notes step adds 0.2 ms at 1,000 notes and 12 ms at 1,000,000 (about 4% of time to first token).
- **Compare arms within a size, not across sizes.** The notes cost the same 485 tokens at every size, yet time to first token falls from 494 to 331 ms as the store grows. The sizes ran one after another under a changing load, so the drift comes from the machine, not the store (INFERRED).
- **These times are slower than the 15 October numbers.** NOTES-IN-ENGINE.md measured 41 to 53 ms without notes and 205 to 279 ms with them, on a different commit (`c11e8d26`) under a lighter load. Ours are 69 to 85 ms and 331 to 494 ms. Not the same build, so not compared further (INFERRED: the load is the likely cause).

## 5. Decode speed

**What was done.** For each answer we took the server's own `tokens_per_second`: completion tokens over decode seconds, after the first token. The answers to the 56 questions do not compare like with like. With notes, the median answer is 8 to 10 tokens (a short fact), and a short answer's rate is dominated by its first steps. Without notes it is 69 tokens. So each run also asked one fixed probe 3 times in each arm, an off request before each: "Count from 1 to 120, one number per line, digits only." with `max_tokens` 256. Every probe ran to its 256 tokens. The prompt was 34 tokens without notes and 521 to 528 with them. That difference is the cost being measured, so it was not padded away. The extra pass asked the same probe with drafts off (`"draft": false`), 3 times per arm and server.

| Notes | Probe, drafts on: off (tok/s) | on_nofetch | on (with live checks) | Probe, drafts off: off | on_nofetch |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 233 [222 to 251] | 227 [216 to 254] | 220 [201 to 253] | 104.1 [103.7 to 108.0] | 103.0 [102.2 to 106.8] |
| 10,000 | 282 [271 to 288] | 270 [269 to 279] | 269 [266 to 275] | 100.6 [98.1 to 102.0] | 97.2 [96.4 to 100.5] |
| 100,000 | 306 [300 to 308] | 286 [281 to 288] | 285 [284 to 287] | 86.8 [82.0 to 90.9] | 88.2 [83.9 to 90.2] |
| 1,000,000 | 266 [265 to 307] | 339 [286 to 341] | 70 [63 to 227] | 96.6 [88.4 to 99.7] | 94.6 [92.3 to 97.4] |

On the answers to the 56 questions themselves, decode ran at 120 to 162 tok/s (p50) without notes and 79 to 119 tok/s with them. Answers of 32 or more tokens decoded at 122 to 163 tok/s without notes and 122 to 158 with on_nofetch, though only 4 to 10 notes answers reach that length (full table in `notes-cost-raw/tables.md`).

- **The block alone barely changes decode.** With drafts off, a 256-token answer decodes within 1 to 4% of the same answer without notes, in both directions, at every size. With drafts on, the probe is a best case for drafting (acceptance 0.98 to 1.00). There the notes probe ran 2 to 7% slower at 1,000 to 100,000 notes, and 28% faster at 1,000,000. That run-to-run noise is larger than the effect.
- **The live checks are what cost decode at 1,000,000 notes.** In arm on, the probe made 88 checks, each one a lookup. They took 12 ms in all per answer at 1,000 notes, 32 ms at 10,000, 198 ms at 100,000, and 1,087 to 3,747 ms (median 3,435 ms) at 1,000,000. None of them fetched anything. At 1,000,000 notes the probe's decode took 3.6 s (median) with the checks, against 0.75 s without them. On the short answers, the checks cost 16 ms per answer at 1,000,000 notes (median), and up to 1.9 s for the longest. At 100,000 notes, the probe's decode time with and without checks is the same (0.90 and 0.89 s), though the checks summed to 198 ms, so up to that size they seem to run alongside decode (INFERRED from the timings; not read in code).
- A live fetch that did fetch (5 to 14 of 56 answers) paused its answer for a re-read of about 554 tokens. These pauses are inside that answer's decode time. Their size is in `notes.decode.injected[].pause_ms` in the raw rows (for example 475 ms at 1,000 notes).

## 6. Memory

**What was done.** We read the server's resident set (`ps -o rss`) and its physical footprint (the server's `/memory`, from the kernel's `rusage_info_v4`) three times per run: once the store had loaded, after the questions, and after the 2,100 writes. Separately, `notes-size-bench` loads each store with counting allocators and reports the bytes the store asked for and the bytes its malloc blocks hold. That excludes the lookup scratch, the journal, the tokenizer, the model and the KV cache.

| Notes | Store loaded at start (ms) | RSS loaded (GiB) | after the questions | after the writes | Footprint after the questions (GiB) | Store bytes per note | held by malloc, per note | Store total (MB) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 20 to 27 | 18.58 | 18.60 | 18.67 | 24.74 | 1,248 | 1,301 | 1.2 |
| 10,000 | 50 to 68 | 18.59 | 18.61 | 18.68 | 24.75 | 526 | 567 | 5.3 |
| 100,000 | 558 to 906 | 18.66 | 18.69 | 18.77 | 24.82 | 451 | 478 | 45.1 |
| 1,000,000 | 4,234 to 19,110 | 19.40 | 19.48 | 19.57 | 25.62 | 434 | 479 | 433.7 |

The RSS and the bytes per note were the same in all three runs, to two decimals and to the byte. The physical footprint is larger than the RSS because it includes the GPU buffers the engine has written, about 6 GiB here. The store's bytes per note fall toward 434 as fixed tables (the vocabulary maps, about 0.65 MB) spread over more notes. From 1,000 to 1,000,000 notes the server's RSS grew by 0.82 GiB against the store's 0.43 GB. The rest is INFERRED to be the load's parse peak (`notes-size-bench` reports 2.6 GB asked during loading at 1,000,000 notes) that malloc keeps. Loading the 1,000,000-note store took 4.2, 6.3 and 19.1 s in the three runs (the server's log line).

## Against the earlier numbers

| Earlier number | Its source | Here | Agrees? |
| --- | --- | --- | --- |
| About 434 bytes per note at 1,000,000 notes | `notes-integrated-size1m-f419b916.md` (432.9 B) | 433.7 B (434), `notes-size-bench` on this build | Yes |
| Median lookup 0.032 ms at 1,000 notes, one thread | `bench/final1/after.jsonl`, sa, 32.3 us | 25.2 us in process (1 thread); 48 us lookup in the server (145 us for the whole step) | In process, close (25 against 32 us). The server is slower. |
| Median lookup 4.96 ms at 1,000,000 notes, one thread | `bench/final1/after.jsonl`, sa, 4,963 us | 2.96 ms in process; 11.7 ms in the server (12.1 ms for the whole step) | No: in process it is 1.7 times faster than before, and the server is 2.4 times slower. Different binary and load. Neither is the server's figure. |
| Write median 3.9 to 4.8 ms | `bench/final1/after.jsonl` (merges on a thread) | 8.9 to 71 ms over HTTP; 4.46 ms in process at 1,000 notes with files on t705 | Only at small stores and on the same drive. Not on this build at 100,000 or 1,000,000 notes (section 1). |
| Index merge of about 3.5 s every ~1,000 writes at 1,000,000 notes | `bench/final1/after.jsonl`, a merge called on its own (3,527 ms) | every 1,025 written notes; the merging write took 1.9 to 7.6 s (median 2.5 s) | The interval agrees. On this build the merge is part of a write and blocks that request. Its length was 1.9 to 7.6 s here. |

## What did not run, or ran differently

- **8 lookup threads in the server**: the server has no such option (READ: `tensorfold-native serve --help` lists `--notes`, `--no-notes-write` and `--notes-index` only, and a lookup runs on its request's thread). As instructed, no code was added. The 8-thread numbers are `notes-live-bench`'s, with no model loaded. Eight concurrent requests to the server were not measured.
- **`notes-live-bench` with writes at 1,000,000 notes**: 1 run, not 3 (one run took 38 minutes).
- **`notes-sa-bench` with 3,000 writes**: 3 runs at 1,000 and 10,000 notes, 1 at 100,000, none at 1,000,000 (its postings half rebuilds the whole index on each write). Only 1,000 notes was also run with the files on t705.
- **A first try of the extra pass and the 1-thread bench was thrown away.** Two copies of the script ran at once by our mistake, shared a port and folders, and broke each other (`notes-cost-raw/discarded-collided-run/`). Both were run again alone, and only those reruns are reported. No main-sweep run overlapped them. The main sweep had finished at 13:43, the reruns ran from 16:03.
- **Not measured**: TTFT and decode with a prompt cache (it was off), streamed timings, and the size of each live fetch's pause beyond the raw rows.

## Files

`docs/notes-cost-raw/`:

- `sweep-N-runR.chat.jsonl`: every request of the main run. Each row has its arm, its order, the server's usage, timings and notes fields, and the first 200 characters of the answer.
- `.writes.jsonl` (one row per write), `.after.jsonl`, `.mem.jsonl` and `.server.log`.
- `sweep-N-runR-extra.extra.jsonl`: the value-hop pass and the probe with drafts off.
- `live-bench-N-runR.jsonl`, `live-bench-writes-N-runR.jsonl`, `sa-bench-N-runR.jsonl`, `sa-bench-3000w-*.jsonl` and `size-bench-N-runR.jsonl`: the in-process benches.
- `run-log.txt`, `extra-log.txt`, `bench-log.txt`, `sa-log.txt` and `final1x-log.txt`, with the free memory and load average at each start; `build.log`.
- `tables.md` holds every table, computed by `tools/report.py` from these files. `summary.json` holds the per-run values behind them.

`tools/`: `cost.py` (the requests), `run.sh` (the main sweep), `extra.sh`, `bench.sh`, `sa.sh`, `final1x.sh` and `report.py`. The scripts hold this Mac's paths.

## Reproducing

```sh
zig build native notes-live-bench notes-sa-bench notes-size-bench -Dcpu=apple_m1 -Dversion=1.0.11 -p OUT --cache-dir CACHE
```

```sh
RUNS=3 WRITES=2100 tools/run.sh 1000 10000 100000 1000000
```

```sh
tools/extra.sh 1000 10000 100000 1000000
```

```sh
tools/sa.sh
```

```sh
tools/bench.sh
```

```sh
python3 -I tools/report.py docs/notes-cost-raw > tables.md
```
