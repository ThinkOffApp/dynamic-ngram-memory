# Where one Living Weights lesson spends its time

Model: Nemotron 3.5 Lightning 30B-A3B, MLX 4-bit (`TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit`, snapshot `d9d758fb`).
Machine: gb10-a, NVIDIA GB10 (CUDA, memory shared with the host), ext4 on NVMe.
Software: TensorFold 1.0.8 (`tensorfold-native serve --slide --slide-graph`, build `~/tf108`) for the main set; 1.0.9 and our #787 branch for small check sets.

Written 14 Oct 2026 by a reviewer agent (agent-authored). Nothing was run on a GPU for this page: every number comes from logs that already existed on gb10-a (12 to 14 Oct 2026) or from reading the source (`~/tmp/tf-pocket-cuda`, tag `v1.0.8` and `origin/main` = 1.0.9).

Grades: **MEASURED** = taken from a log, path given. **READ** = the code says it, file:line given (paths under `zig/src/` unless stated). **INFERRED** = our arithmetic or reasoning, not checked by a run.

## Summary for the project owner

- Teaching one fact takes about 3 minutes (median 180 s; most lessons take 2 to 5 minutes).
- Almost half of that (44%) is one step: teaching the new neurons what to say. The trainer runs the model forward and backward about 170 times, over about 54,000 tokens of example answers.
- A quarter (26%) is the model writing the lesson: about 280 short generations (questions, answers, yes or no checks). The first lesson after a server start takes about 40 s more, because the server first writes its reference answers one at a time.
- 14% is checking after each round (about 44 more generations a round, usually 2 to 5 rounds). 10% is starting over when a check finds the lesson answering a question about something else.
- Saving takes half a second, but it rewrites 263 MB for a fact that adds about 30 KB. The very first save into a model rewrites all 18.5 GB once.
- "Heavy memory" is true for memory size (the trainer keeps its own 18.3 GB copy of the weights) and for disk writes. The time itself goes into work per training token, mostly the backward pass, which costs about 5 times a forward pass per token.
- Biggest likely savings: a faster trainer backward, not starting over after a hand-back, cheaper checks, and writing the reference answers in parallel. Saving only the pocket saves disk writes but little time. The trainer's pocket kernel showed no measurable cost.

## 1. The split (MEASURED)

**Model and machine:** Nemotron 3.5 Lightning 30B-A3B MLX 4-bit on the GB10 (gb10-a), TensorFold 1.0.8, one fact per `/v1/slide/learn` request, no other traffic on the server during a lesson.

**Lessons:** 306 lessons that reached training, from 26 runs of the lwrun/capacity driver (51 server processes), 12 to 14 Oct 2026: L1 (11 runs, 1 lesson each), L2 (3 runs, 15 lessons), L3 (7), L4 (40), L5 (47), L6-seed1b (103), L7-seed1 (60), eyes-678 (14), neg and neg2 (9). Another 19 lessons stopped before training ("too few clean answers to learn from") after 5 to 24 s; they are left out.

**Logs:** `gb10-a:~/lw-runs/<run>/run.log` (the driver: each lesson's wall time, and the save's seconds and bytes) and `gb10-a:~/lw-runs/<run>/phases/<phase>/server.log` or `.../steps/main/server.log` (the server's own timers). Per-lesson numbers and the two scripts that produced them: `~/tmp/lw-logic/perf/lesson-time-raw/` (`lesson_parse.py`, `lesson_split.py`, `lessons-split.json`, `split-report.txt`).

**Share** is a part's seconds summed over all 306 lessons, divided by their summed wall time (61,474 s). The rows add up to each lesson's wall time (checked lesson by lesson; "rest" is the remainder).

| # | Phase | What it does | Seconds per lesson: median (p10 to p90; min to max) | Share | Kind of work | Work per lesson (median), bytes |
|---|---|---|---|---|---|---|
| 1 | Writing the lesson | The shipped model writes about 12 questions, its own answers to them, near misses, extra wordings, yes/no judgements of each; and asks whether the fact replaces an earlier one | 45.0 (35.9 to 80.4; 31.7 to 95.5) | 26.1% | GPU text generation (the serving engine, 8 at once) | 278 model calls; 6,774 tokens written; 9,773 new prompt tokens read |
| 2 | Trainer start | The trainer loads its own copy of the weights; once per server process | 4.0 in each server's first lesson (51 lessons), 0 otherwise | 0.3% | GPU upload | 18.3 GB of weights + 4.1 GB of buffers |
| 3 | Reading rows | Forward passes over the fact's questions and all the prompts it must stay silent on, keeping each MoE layer's input | 7.7 (6.1 to 15.4; 4.2 to 18.8) | 4.5% | GPU, forward only | 40 passes, 39,256 token rows; 0.19 s a pass of about 970 rows (0.197 ms a row) |
| 4 | Fitting where the neurons fire | Adam, 500 steps for each of 10 to 16 candidate layers (one CPU thread each) over the read rows; then refitting earlier facts' neurons that fire on the new rows | 8.5 (4.6 to 26.2; 3.1 to 74.4) | 6.5% (1.9% of it the earlier facts' refit) | CPU | about 16,000 to 39,000 rows of 2,688 values per layer |
| 5 | Fitting what the neurons add | Adam steps through the real model: each step is a forward pass through all 52 layers, the LM head, and a backward pass down to the fact's lowest layer, over 8 of the fact's answers plus up to 8 answers that must not change | **60.6** (32.7 to 185.7; 22.3 to 346.3) | **44.2%** | GPU, forward + backward | 174 passes, 54,294 token rows; 1.13 ms a row |
| 6 | Judging | The held-out answers' loss, read every 4 steps, for the early stop | 2.5 (1.2 to 5.1; 0.8 to 9.4) | 1.4% | GPU, forward only | 33 passes, 0.075 s each |
| 7 | Trainer recall check | Do the held-out answers come back token for token | 0.8 (0.2 to 1.4; 0.2 to 3.0) | 0.4% | GPU, forward only | 12 passes |
| 8 | Checking each round | The live model, new neurons in, answers the held-out and never-trained wordings and the questions it must leave alone; a judge compares before and after | 23.3 (11.7 to 45.0; 6.1 to 124.4) | 14.3% | GPU text generation | 135 model calls (44 a round); 4,503 tokens written; 7.9 s a round; 3 rounds |
| 9 | Saving | The pocket written into the model's shards at the end of the request, then the last fact's held-out answers read again | 0.5 in place (0.3 to 1.4; 265 saves). 14.8 the first time a model gets a pocket (11.6 to 29.6; 37 saves) | 1.2% | disk + CPU | 262,628,352 bytes in place. 18.53 GB on the first save (section 3) |
| 10 | Rest | Learner bookkeeping, HTTP, setting up a hand-back, re-asking reference answers the fact changed | 1.7 (1.4 to 2.1) | 1.1% | mixed | |
| | **Whole lesson** | | **179.5** (117.0 to 318.5; 93.8 to 571.6) | 100% | | |

**Hand-backs (inside rows 3 to 7):** in 63 of the 306 lessons a check found the lesson answering a question about something else. The question is handed back and the trainer starts the fact over from reading (`whole.zig:125-139`). These reruns are **10.1%** of all lesson time (median 0, p90 69.7 s, max 216.6 s a lesson).

**Grouped:** model text generation (rows 1 and 8) 40.4%; trainer GPU passes (rows 3, 5, 6, 7) 50.4%; CPU fitting 6.5%; saving 1.2%; trainer start and rest 1.4%.

**Rounds a lesson took:** 1 round in 37 lessons, 2 in 88, 3 in 42, 4 in 37, 5 in 74, 6 to 14 in 28. A round after the first takes a median 13.1 s of training (rows 5 to 7; p10 6.0, p90 32.6 s, 784 rounds) plus 7.9 s of checking.

Where each number comes from (log line, and the code that prints it):

- Row 1: `slide: fact N's lesson written in X s` (`server/slide.zig:122`). Its timer starts before the replace check (`slide.zig:109-113`), so the replace decision is inside row 1.
- Rows 3 to 7: `slide pocket: round N took X s: read ..., fit ..., values ... (P passes, R rows), judge ..., check ...` (`families/nemotron/pocket_clock.zig:28`), one line for every trainer run, hand-back reruns included.
- Row 4's split: `N candidate neurons fitted on R read rows in X s; earlier neurons checked in Y s` (`families/nemotron/pocket.zig:399`).
- Row 8: `slide: round N: k steps (X s learning, Y s checking)` (`slide.zig:173`).
- Row 2: `slide: the trainer is ready in X s` and `the trainer's raw weights (18.3 GB) uploaded in 3.4 s, its buffers (4.1 GB) in 0.1 s` (`families/nemotron/learner.zig:207`, `cuda_train.zig:235`).
- Row 9: `the learned neurons are in the model's shards (N MB written in X s)` (`families/nemotron/whole.zig:170`) and the driver's `save saved (X s, B bytes)`.
- Whole lesson: the driver's `learned Ln in X s`, from opening the request to the stream closing. In 1.0.8 the save runs after `[DONE]` but before the stream closes (`slide.zig:101`, `:135`).
- Model calls and tokens: the server's own generations, the `[tensorfold] done  prompt=P cached=C ... tokens=T` lines with no request id, counted in the phase they appear in (new prompt tokens = P minus C).

### The same split in other sets (MEASURED)

| Set | Lessons | Whole lesson | Writing | Reading | Fitting (CPU) | Output fitting | Judge + recall | Checking | Saving | Rest | Logs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.0.8, not the first lesson in its server | 255 | 174.5 | 42.0 | 8.2 | 8.5 | 71.1 | 3.3 | 21.4 | 0.5 | 1.7 | `~/lw-runs/*` as above |
| 1.0.8, first lesson in its server | 51 | 195.7 | 80.7 | 7.4 | 8.8 | 47.6 | 3.6 | 32.7 | 11.9 | 1.5 | same |
| 1.0.8 builds, live test p3, no other traffic | 12 | 147.0 | 78.8 | 7.2 | 8.5 | 37.0 | 3.2 | 32.5 | 0.5 | 0.3 | `~/tf-pocket-cuda/live/p3-{before,tip,mutant,inst}/serve-{1,2}.log`; totals from `live/run_live3.log`, `live/run_inst.log` |
| **1.0.9** builds, live test p3 | 6 | 144.0 | 74.5 | 7.2 | 8.8 | 36.8 | 3.1 | 28.8 | 0.6 | 0.3 | `~/tf-pocket-cuda/r109/live/p3-{before,after}/serve-{1,2}.log`; totals from `r109/raw/live.out` |
| 1.0.8, with chat requests alongside | 14 | 172.9 | 49.3 | 7.2 | 9.2 | 46.8 | 2.5 | 21.3 | 0.6 | **21.1** | `~/tf-pocket-cuda/live/{before,after,before-p2,after-p2}/serve.log`, `ck*.json` |

All medians in seconds. The p3 sets are the same three facts on each build (lesson A on a fresh model, B1 and B2 after a restart), each a first or second lesson in its server, so their writing is high.

- 1.0.9 learns exactly like 1.0.8: no file of the learner, the lesson writer, the trainer, the save or `kernels/cuda/train.cu` changed between `v1.0.8` and `origin/main` (READ: `git diff v1.0.8 origin/main`). On the same three facts the GPU trainer parts match within 1 s (output fitting 47.2 against 46.4 s, 29.2 against 29.2, 37.0 against 36.7; MEASURED); the CPU fit differs by up to 1.6 s.
- With chat requests running alongside, each lesson's training rounds took 20.8 s longer than the trainer's own clock (median; 10.7 to 26.5 s), against 0.2 s without (MEASURED). INFERRED: the trainer waits for the GPU while chat requests run.
- The graph polls in the chat-alongside `ck*.json` (`polls`, every 2 s) show the same phase order: writing, reading, fitting, values, then checking and values in turn (MEASURED).

## 2. What dominates, and why it is "heavy for few memories"

**Output fitting dominates (44%), then the model's own writing (26%) and checking (14%).** Training work on the GPU is half of the lesson; text generation is 40%.

**What a fact keeps** (MEASURED from the 59 learned facts in the runs' `graph-final.json`; sizes READ from `families/nemotron/pocket_shards.zig:109-162`):

- 10 neurons a fact, in every one of the 59. Each neuron has an input row and an output column of 2,688 values: 53,760 numbers a fact.
- While the server runs they are float32: 215 KB a fact. On disk at 4-bit: about 3,024 bytes a neuron (1,344 + 168 for the input row, 1,344 + its share of 168 for the output column), so about **30 KB a fact**.
- The whole pocket for comparison: 64 neurons x 2,688 x 2 matrices x 4 bytes = 1.38 MB a layer as float32, 31.7 MB over the 23 MoE layers; 4.45 MB at 4-bit.

**What it takes to get them, per lesson (medians, MEASURED):**

- 263 trainer passes. The reading and output-fitting passes alone carry 95,600 token rows forward through all 52 layers; 54,300 of those rows also go backward from the top down to the fact's lowest neuron. That lowest neuron is at layer 10 in 48 of 59 facts (the other 11: layers 17 to 22), so the backward crosses about 42 of the 52 layers.
- 430 model generations writing 12,600 tokens (rows 1 and 8).
- A second 18.3 GB copy of the model's 4-bit weights held only for the trainer's backward (`cuda_train.zig:226`, `loadRaw`): the server reserves "25.54 GiB kept for --slide's trainer" next to "model 17.99 GiB" (server start line in every `--slide` log).
- 262.6 MB written to the shards. That is about **8,700 times** the fact's 30 KB, and 59 times the whole 4-bit pocket.

**Why the time is per token, not per pass (MEASURED, reading INFERRED):**

- An output-fitting pass costs 1.13 ms for every token row in it. Over 1,177 trainer rounds with 150 to 730 rows a pass, the time a pass is `-0.010 s + 1.13 ms x rows` (near zero fixed cost). Passes under 150 rows cost a floor of about 0.165 s each.
- A forward-only reading pass costs 0.197 ms a row.
- So the LM head plus the backward cost about 0.93 ms a row, about 5 times the whole forward, over about 42 layers: up to about 6 times a forward layer, layer for layer (INFERRED; the head's share is not logged).
- The work grows with how many example tokens go through and how many steps, never with how many neurons are learned (READ: `pocket_values.zig:18-27` sets up to 60 steps a round, at least 12, 8 + 8 + 8 answers a step).
- Memory traffic (INFERRED): with about 420 rows a pass and 6 of 128 experts a row in each MoE layer, every expert is used in every pass (the chance a given expert is missed is about (122/128)^420, about 2 in a billion). So each forward pass reads close to all 17.8 GB of 4-bit weights, about 4.7 TB a lesson in forward passes alone. The 0.165 s floor for small passes fits that read. But at the usual 420 rows the per-row work (about 0.47 s a pass) is about three times that floor, so the weights' reading is not what sets the lesson's time.

So the project owner's "heavy memory op for very few memories" is right three ways: memory held (a second 18.3 GB weight copy), memory read (several TB of weight reads a lesson), and disk written (263 MB a lesson for 30 KB). The **time** is set by compute per training token, mostly the backward.

## 3. The save path, checked (READ + MEASURED)

A reviewer agent's reading of `core/shard_edit.zig` holds, with one correction: later saves do not write "just the pocket".

READ:

- `pocket_shards.zig:18-20`: the model is widened only when `config.json` has no pocket size yet. A pocket of another size is refused (`PocketOfAnotherSize`).
- **First save** (`pocket_shards.zig:38-50`, `shard_edit.zig:87-114`): `se.grow` cannot write in place (each shared expert grows by 64 rows), so every shard holding an MoE layer is rewritten whole through `.tmp`, fsync and rename (`shard_edit.zig:219-271`, `:326-339`). The MTP file is rewritten too, then the index's `total_size` and `config.json`.
- **Later saves** (`shard_edit.zig:139-150`): `se.overwrite` pwrites each tensor where it lies and fsyncs, one open and one fsync per tensor: 138 a save (6 tensors x 23 layers).
- **The correction:** what gets written is the whole shared expert of **all 23 MoE layers**, its own 3,712 rows plus the 64 pocket rows, as 4-bit codes, scales and biases (`pocket_shards.zig:28-37`, `:109-161`). That is 11,418,624 bytes a layer, 262,628,352 bytes a save. Layers the fact did not touch are written too, with zeros where no neuron is. The pocket itself is 1.7% of those bytes (64 of 3,776 rows).
- A request whose lesson failed still saves, once the model has a pocket (`whole.zig:166` skips only an empty pocket).

MEASURED:

| Save | Count | Server's time | Bytes the server reports | Bytes actually written | Evidence |
|---|---|---|---|---|---|
| First save into a model with no pocket | 37 | median 14.8 s (11.6 to 29.6) | 274,046,976 ("261 MB") | **18.53 GB**: 4 shards 17.78 GB + MTP 0.75 GB + index + config | `live/before/ck1.json` listing: the shards' mtimes 14:37:11, :17, :23, :25 UTC, index :26, MTP and config :27, all at their new sizes. The request's stream stayed open 20.9 s after `learned` (`ck1.json` `raw`), the server says 20.5 s (chat alongside). Same in `live/after/ck1.json` (shards 14:53:08 to :20, 13.6 s). |
| Every later save | 265 | median 0.5 s (0.3 to 1.4) | 262,628,352 ("250 MB") | 262,628,352 bytes in place, in all 4 shards | `live/{before,after}/ck2-4.json`, `live/{before,after}-p2/ck2-3.json`: all 4 shards' mtimes change within one second, sizes unchanged; MTP, index and config untouched. Also L7 L10, a failed lesson: "save saved (0.4 s, 262628352 bytes)". |

- The reported bytes undercount the first save: the server counts only the replaced tensors. The disk takes all 18.53 GB (file sizes; READ `rewrite` writes the whole file).
- The widened shards grew by exactly the pocket: shard 1 by 1,161,222 bytes (6 layers x 193,536 + header), shards 2 and 3 by 1,354,757 and 1,354,753 bytes (7 layers each), shard 4 by 580,609 (3 layers), MTP by 193,536 (MEASURED: `stat` of the HF snapshot against the taught copies' listing).
- Earlier hash checks saw 2 or 3 shards change on later lessons. INFERRED: all 4 are rewritten in place, but a shard whose pocket bytes came out the same keeps its hash.

## 4. What would cut each part, ranked by likely saving (all IDEA; savings INFERRED)

1. **IDEA: a faster trainer backward (row 5, 44%).** The backward plus head costs about 0.93 ms a token row, about 6 times a forward layer, layer for layer. The trainer only needs gradients for the activations (READ: no weight gradients; `cuda_train_back.zig:37-63`), which should cost about one to two forward passes. If it came down to about 0.4 ms a row, output fitting would fall from about 61 to about 32 s: about 28 s a lesson (16%). First step: an nsys profile of one lesson's values passes (not run here, GPU rule). Where: `families/nemotron/cuda_train_back.zig` (`moe`, `product`, `adapterBack`, `mamba`, `attention`), `cuda_train_ops.zig`, the pass in `cuda_train.zig:448-560`.
2. **IDEA: no full restart after a hand-back (10% of all time; in 1 lesson of 5).** Today a handed-back question undoes the pocket and starts again from reading (`whole.zig:125-139` → `startPocket`). Keep the read rows and the fitted inputs, add the question to the stay-silent rows, refit the inputs on the CPU, and continue the outputs from where they were. Saving: most of the 10%, about 20 s a lesson on average (mean of the reruns 20.4 s), up to 3.6 minutes on a bad lesson.
3. **IDEA: cheaper checks each round (row 8, 14%).** About 44 generations a round, about 7.9 s. Score the check questions with the trainer's teacher-forced loss (as the judge does, 0.075 s a pass) and generate only where the loss moved; or check only every second round. The questions about the user that nothing told it, and the bank questions, do not change between rounds unless the pocket fires on them. Saving: about 10 to 15 s a lesson. Where: `server/slide.zig:169-171`, `server/slide_verify.zig:27-103`.
4. **IDEA: fewer or batched generations while writing (row 1, 26%).** Three parts:
   - The reference answers (40 everyday prompts with and without thinking, 89 personal questions) are written one at a time on a server's first lesson and again after every replace (READ: serial `for` loops at `server/slide_teacher.zig:130` and `:146`; `forgetBank` at `slide.zig:256`, `:272`). That costs about 39 s (first lessons median 80.7 s against 42.0 s; 13,300 tokens written against 6,300). `side.askAll` (`server/slide_side.zig:45`) already runs 8 at once; caching them on disk next to the model would remove them after restarts too. Saving: about 30 s on each server's first lesson.
   - The replace decision costs about 0.5 to 0.8 s and about 5.5 short calls for every earlier learned chat fact, up to 32 facts, so up to about 16 to 25 s (section 5). Pre-filter the candidates by shared words or subject before asking the model, or ask once over all of them. Where: `server/slide_replace.zig:17-52`.
   - The lesson itself: about 280 calls, including a thinking-on answer for every question, 12 thinking near misses, 12 extra wordings each judged, up to 96 questions about others (`server/slide_lesson.zig:64-75`, `server/slide_take.zig:132-136`). Fewer of each would save in proportion, at some cost to locality.
5. **IDEA: train only toward the layers that take neurons (rows 3 and 5).** The backward already stops at the fact's lowest neuron (READ `cuda_train_back.zig:60`); the forward always runs all 52 layers from layer 0 (`cuda_train.zig:502`).
   - Starting each pass at the lowest neuron's layer from a kept residual would skip layers 0 to 9 for most facts: about 19% of the forward, about 3% of output fitting, about 2 s a lesson.
   - Letting facts take only deeper layers would shorten the backward much more. Subject facts may start at a fifth of the depth (`pocket.zig:75-79`), and 48 of 59 facts did start at layer 10. Starting at layer 24 instead would cut the backward from about 42 to about 28 layers: a third of the backward, about a quarter of row 5, about 15 s a lesson. But it changes where facts can live, so recall and locality need re-testing.
6. **IDEA: input fitting on the GPU (row 4, 6.5%).** 16 layers x 500 Adam steps, each a 2,688-wide product over 16,000 to 39,000 rows, on CPU threads (`pocket.zig:358-400`, `pocket_fit.zig:46-147`). The refit of earlier facts' neurons grows with the number of facts (p90 15.9 s, max 58 s; `pocket.zig:471-513`). On the GPU each step is one small matrix-vector product. Saving: about 7 s a lesson, more as facts pile up.
7. **IDEA: several facts in one training run.** Today each fact is its own lesson (`slide.zig:102-103`). Output fitting costs the same per token row however rows are grouped (MEASURED, section 2), so the fact's own answer rows do not get cheaper. What batching shares is the per-lesson overhead: reading the stay-silent prompts (most of the 39,000 read rows), the 8 answers-to-keep in every step, the checks that are the same for every fact, and the reference answers. Saving: perhaps 15 to 30% a fact when 2 to 4 facts share a run. Where: `slide.zig:102-134`, the pocket learner's `begin` (`pocket.zig`) and `valueOnce`'s draws (`pocket_values.zig:132-232`), with an undo for each fact.
8. **IDEA: save only the pocket (row 9, 1.2%).** This saves bytes, not much time: at most 0.5 s a lesson, plus the 14.8 s first save once a model.
   - The pocket rows of `up_proj` are contiguous (the last 64 rows). The pocket columns of `down_proj` are interleaved, one run of 64 per output row, so 2,688 short writes a layer, unless the pocket moves to tensors of its own.
   - A pocket file of its own, instead of widened shards, would also remove the 18.53 GB first rewrite. Per lesson the writes drop from 262.6 MB to 4.45 MB, or to about 30 KB plus the rescaled down groups if only changed neurons are written. Note the down groups are shared by all 64 neurons of a row (TF108-SOURCE-NOTES (c)), so one neuron's change rescales its whole group.
   - Skip the save when no neuron changed (failed lessons write 262.6 MB today).
   - Use one open and one fsync per shard, not per tensor.
   - Where: `families/nemotron/pocket_shards.zig:14-52`, `core/shard_edit.zig:139-150`, the loaders that read the pocket back (`cuda_slide.zig:92-96`, `weights.zig:259-264`).
9. **IDEA: the trainer's pocket kernel `tf_train_pocket_rows` (`zig/kernels/cuda/train.cu:562`).** It runs one block a row and always loops over all 64 neurons, used or not (`cuda_train_fwd.zig:99` passes the width, not the count). It reads 1.38 MB of float32 pocket a row in each MoE layer. #787 fixed this pattern for decoding, not here. But reading passes cost the same per row with a pocket present as without: 0.1966 ms against 0.1975 ms a row (median, 274 against 32 lessons, MEASURED). INFERRED: the kernel is under about 1% of a forward pass, so fixing it would save under about 1 s a lesson. Lowest priority for time.

Not a time saving, but memory: the backward reads its own 18.3 GB copy of the weights (`cuda_train.zig:226`). If it read the engine's 4-bit buffers, about 18 GB would be freed, and the 3.4 s trainer start would go.

## 5. What the logs cannot time (estimates, kept out of the table)

- **The replace decision** is inside row 1 and has no timer of its own. INFERRED from regressions of writing time on the number of earlier learned facts (capped at 32):
  - L7-seed1 (56 lessons): 0.51 s and 5.5 model calls an earlier fact, from 35.4 s with none.
  - L6-seed1b teach (26 lessons): 0.79 s an earlier fact.
  - At the cap of 32: about 16 to 25 s of the writing.
- **The reference answers** on a server's first lesson: MEASURED as a difference of groups (+38.7 s writing, +7,000 tokens written). INFERRED that this is `keepExamples` and `bankAnswers` (READ `server/slide_teacher.zig:123-155`).
- **The rest of the writing** (questions, answers, near misses, wordings, judgements) is not split further by any log line. MEASURED only as model calls by reply length. In L7, the 1 to 4 token yes/no calls grow from 65 to 214 a lesson as facts pile up, while longer generations stay about the same.
- **CUDA graph captures** (the "reload/recapture" step): MEASURED sums of the `TF_GRAPH_LOG` capture times inside each lesson were 2.3 to 3.9 s (1.0.9, 3 lessons) and 4.8 to 8.5 s (1.0.8, 2 lessons) (`r109/live/p3-before/serve-*.log`, `live/p3-before/serve-2.log`). They happen during the generations, so they are already inside rows 1 and 8, not added on top. The recapture after a pocket upload (a layer going from 0 to 1 neuron, READ `cuda_slide.zig:85-88`) is among them.
- **Weight bytes read per pass** (section 2): INFERRED; no bandwidth counter was logged.
- **The backward's own kernels**: their split is unknown without a profile; ranked first in section 4 on the per-row arithmetic only.
