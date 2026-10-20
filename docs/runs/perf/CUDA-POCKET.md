# The Living Weights pocket on CUDA: how 1.0.8 runs it, and what it costs on the GB10

**Source:** TensorFold v1.0.8, `0efbd447` (upstream main), read in `~/tmp/tf108`. On gb10-a the source in `~/tf108` matches all 1,704 tracked files of `0efbd447` byte for byte (MEASURED, sha256, 13 Oct 2026).
**Model:** Nemotron 3.5 Lightning 30B-A3B, MLX 4-bit (`d9d758fb`): hidden size d = 2,688, 52 layers, of which 23 are MoE layers (layers 1, 3, 6, ..., 49, 51), plus one MoE layer in the MTP draft head (READ, `config.json`).
**Written by** a reviewer agent, 13 Oct 2026. Grades: MEASURED (we ran it), READ (in the source or files), INFERRED (our reasoning).

The problem this file is about is philip-pentatonic's finding in ashhart/TensorFold#678 (comment 6125486909): on an RTX PRO 6000 with 1.0.7, a Nemotron model with a pocket decodes about 31% slower (290 to 199 tok/s), even when the pocket is empty, and his nsys profile puts it on one kernel, `tf_pocket_rows`. Both fixes discussed below are his proposals; the `used`-count detail is ours.

## 1. How 1.0.8 runs the pocket on CUDA

### 1.1 Where the pocket comes from, and what `n` is

- **The pocket's width is one number for the whole model,** `tensorfold_learned_neurons` in `config.json` (READ, `zig/src/families/nemotron/config.zig:29`). The config reader stores it as `c.pocket` and takes it off the shared expert's own width (`config.zig:250-253`). It must be a multiple of 64.
- **The learner always writes 64.** `pocket.zig:23` sets `neurons = 64`. The first save widens the shared expert of *every* MoE layer by 64 neurons, writing zeros for the layers that hold no fact, and also widens the MTP head's shared expert with zeros; then it sets `tensorfold_learned_neurons = 64` (READ, `pocket_shards.zig:28-49`). So a taught folder has 64 pocket neurons in all 23 MoE layers and in the MTP head, most of them zero.
  - The learner never places a neuron in the last MoE layer (layer 51) (READ, `pocket.zig:195-196`), and later saves never touch the MTP head (READ, `pocket_shards.zig:39-50`, only when widening). Those two pockets are always all zero, even in a model with many facts.
- **Loading on CUDA:** for every MoE layer, when `c.pocket > 0` (`cuda_weights.zig:394`), the loader reads the extra up rows and down columns, dequantizes them to fp32 on the host and uploads two device buffers, `up` and `down`, each `[n, d]` fp32 (READ, `cuda_weights.zig:267-314`). The same `moe()` loader runs for the MTP head (`cuda_weights.zig:504`), so the head gets a pocket too.
  - `n` is `c.pocket`, the config's width, in every layer (`cuda_weights.zig:270, 277`). A pocket wider than 64 is refused, because the kernel keeps a row's activations in 64 floats (`cuda_weights.zig:271`).
  - The CUDA pocket struct is `{ up, down, n }` and nothing else (READ, `cuda_weights.zig:28-29`). **There is no per-layer count of neurons in use on the CUDA forward path.** With 64 neurons per layer, each layer's pocket is 2 x 64 x 2,688 x 4 bytes = 1.38 MB of fp32 (READ + arithmetic).
- **Where per-layer "used" counts do exist (READ):**
  - **Metal forward:** `weights.zig:41-47` keeps `used` beside `up` and `down`; the loader sets it to the last neuron whose up row is not all zero (`weights.zig:259-264`); the learner sets it on every change (`slide.zig:86`). `layers.zig:205` runs the pocket only when `used > 0`, and `layers.zig:268-290` launches over `used` neurons only. So Metal already does both halves of "CUDA fix 1".
  - **The learner (both backends):** `pocket.zig:111` keeps `used` per MoE layer, filled at load from `host.pocketRead` (`pocket.zig:201`) and passed back with every upload (`pocket.zig:328-331`).
  - **CUDA's `pocketRead`** recomputes it by scanning the up rows for the last nonzero one (`cuda_slide.zig:91-96`), and returns 0 when the buffer is not exactly `n x d` (`cuda_slide.zig:87`). This is why shrinking `n` itself would make the learner read a loaded pocket as empty: keep `n = 64` as the buffer size and add a separate count (READ + INFERRED; our addition in llwz `docs/DECODE-COST.md`).
  - **CUDA's `pocket()` setter ignores the count it is given** (`_ = used;`, `cuda_slide.zig:60`), allocates the buffers on first use and uploads (`cuda_slide.zig:59-79`).
  - **The server's Sliding Weights view** (`/v1/slide/graph`) computes its own per-layer `used` from the shards' bf16 scales and biases (`zig/src/server/slide_model.zig:79-145`); it does not feed the forward.

### 1.2 The kernel

`tf_pocket_rows`, `zig/kernels/cuda/nemotron_ops.cu:126-158` (READ):

- One block of 256 threads (8 warps) per row. `__shared__ float act[64]` holds that row's activations.
- **Phase 1 (lines 133-145):** warp w takes neurons w, w+8, w+16, ... (8 neurons a warp when n = 64). For each, its 32 lanes stride over d = 2,688 (84 products a lane, one running sum), reduce by `__shfl_xor_sync`, and lane 0 stores relu(s)^2 into `act[j]`.
- `__syncthreads()` (line 146).
- **Phase 2 (lines 147-157):** thread k takes output elements k, k+256, ... (10 or 11 of them); for each it sums `act[j] * down[j*d + k]` for j = 0..n-1 in order, then adds the sum into the output row (fp32 in decode, bf16 in prompt chunks).
- The loops run over all n neurons whatever their values: an all-zero pocket costs the same as a full one (READ). Each block reads the whole 1.38 MB; with several rows, each row's block reads it again (INFERRED: from L2 after the first).
- The output row is `blockIdx.x * row_mul + row_add` (line 147). The caller passes `row_mul = top_k + 2 = 8` and `row_add = top_k = 6`, the slot of the shared expert's first half in the pair rows (`cuda_forward.zig:259`, `config.zig:80-81`), so the pocket adds into the shared expert's output before the combine.

### 1.3 The launch

- `Ops.pocketRows`, `zig/src/families/nemotron/cuda_kernels.zig:234-248` (READ): refuses n > 64, packs `y, d, up, down, n, out, out_f32, row_mul, row_add`, and launches grid `(rows, 1, 1)`, block 256, no dynamic shared memory (`o.go`, line 247). The function handle is loaded from the `nemotron_ops` module at `cuda_kernels.zig:84`.
- Called from `Forward.pocketRun` (`cuda_forward.zig:255-260`), through `Forward.pocket` (`cuda_forward.zig:247-253`), which runs all rows unless some sequences in the pass are marked untaught (a lesson writer's reads); then it runs the rows between them (`lesson_route.Runs`).
- `Forward.sharedRest` (`cuda_forward.zig:262-273`) calls it whenever the layer has a pocket: `if (m.pocket) |p| try f.pocket(...)` (line 268). There is no check of whether any neuron is in use.

### 1.4 The side stream

- The engine makes a second stream and two events at load (READ, `cuda_engine.zig:136`). A decode forward uses it when the model has at most 128 experts and 8 slots and no dump is attached (`cuda_engine.zig:339`); Nemotron 3.5 has 128 and 8.
- For decode (not prompt chunks), `Forward.experts` hands each MoE layer to `forked` (`cuda_forward.zig:225`). `forked` (`cuda_forward.zig:277-303`):
  - **side stream:** records `fork` on the main stream and waits on it, then the shared expert's up and down projections (two fixed items), then `sharedRest`, which is where the pocket runs (line 290), then records `join`;
  - **main stream:** the router, the routed plan, the routed experts' up and down projections, then waits on `join` (line 301) before the combine.
- So in decode the pocket sits at the end of the side stream. It adds to decode time when the side stream (shared expert + pocket) takes longer than the routed path, and through whatever it takes from the other work while both run (INFERRED). The engine's own window profile measures that wait as the class `shared_wait` (`cuda_forward.zig:30, 302`).
- Prompt chunks take the one-stream path: routed experts, then `sharedRest` on the main stream (`cuda_forward.zig:244`). There the pocket is fully on the critical path, with one block per prompt row (up to the chunk size) (READ + INFERRED).
- **The MTP draft head** calls `f.experts(m.moe, 1, false)` (`cuda_mtp.zig:154`), the same forked path with one row, so every draft level runs one more `tf_pocket_rows` over the head's all-zero pocket (READ).

### 1.5 CUDA graphs

- At load the engine captures the serial round (one row plus the feed) and, with drafting, one verify window per row count 1..16 as CUDA graphs (READ, `cuda_engine.zig:172, 182-193`), each recorded by stream capture (`cuda_engine.zig:202-220`). The MTP head's levels are captured too (`cuda_drafts.zig:123`). Other sequences capture their own copies on first use (`cuda_engine.zig:235-253`).
- The kernel arguments, including `n` and the buffer addresses, are fixed in the graph when it is captured (READ, by how stream capture works).
- When the learner makes a pocket for the first time, `cuda_slide.pocket` captures the graphs again (`cuda_slide.zig:78`, `recapture` at `cuda_slide.zig:123-140`); later uploads only overwrite the buffers' contents, with no recapture (READ).
- **Consequence for a fix (INFERRED):** a per-layer `used` count passed as a launch argument, or a launch skipped for a layer with none in use, is frozen into the captured graphs. When the learner changes a layer's count, the graphs must be captured again (as on first use), or the count must live in device memory and be read by the kernel. With no pocket at all (an untaught folder), no launch is captured in the first place.

### 1.6 Launches per token (READ + arithmetic)

| Mode | `tf_pocket_rows` launches | Each reads |
|---|---|---|
| Undrafted decode, one sequence | 23 a token (one per MoE layer), grid 1 | 1.38 MB |
| Drafted round of r rows | 23 in the verify window, grid r; plus 1 per MTP draft level, grid 1 | 1.38 MB per block |
| Prompt chunk of R rows | 23, grid R, on the main stream | 1.38 MB per block |

Per undrafted token that is 23 x 1.38 MB = 31.7 MB, read by 23 single blocks, against roughly 1.9 GB of weights the rest of the token reads (INFERRED, the 1.9 GB figure from llwz `docs/DECODE-COST.md`). The bytes are small; the problem is that each launch is one block on one SM, so it is limited by how fast one SM can pull 1.38 MB, not by the GPU's bandwidth (INFERRED; philip-pentatonic's reading in #678 too).

## 2. Cost on the GB10 (1.0.8)

**Machine:** gb10-a, an ASUS Ascent GX10 (NVIDIA GB10, sm_121, CUDA at `/usr/local/cuda`, aarch64 Ubuntu). Measured by a reviewer agent on 13 Oct 2026, 08:16-09:13 UTC, after the LLWZ L7 run had finished (`DONE L7-seed1` at 07:58:56 UTC). No other engine process ran on the GPU during these numbers (MEASURED, `ps`; each script holds a lock and refuses to start beside another engine process). GPU temperature 43-58 C at the start of each run.

**Model:** Nemotron 3.5 Lightning 30B-A3B, MLX 4-bit, three folders:

| Label | Folder | Pocket |
|---|---|---|
| **A** (untaught) | the Hugging Face snapshot `d9d758fb`, served in place, read-only | none (no `tensorfold_learned_neurons`) |
| **T** (taught) | `~/lw-runs/gb10-a-L7-seed1/models/L7-seed1.main`, the L7 run's main folder (64 facts taught; LLWZ capacity run), served in place, read-only | 64 neurons a layer; neurons in use (last nonzero up row) 41-44 in layers 10-31, 1-3 in layers 34-45, 0 in layers 1, 3, 6, 8, 47, 49, 51 and in the MTP head: 440 in use in 16 of 23 layers (MEASURED from the shards) |
| **E** (empty) | `~/tf-pocket-cuda/models/A-empty64`: A widened to 64 pocket neurons a layer, all zero (philip-pentatonic's "pocket widened, no facts" case) | 64 neurons a layer, all zero |

**How E was made without copying the model (MEASURED):** the engine opens the index's shard files in name order and takes each tensor from the first file that holds it (READ, `zig/src/core/checkpoint.zig:11-50, 93-100`). E holds symlinks to A's four shards, one new 251 MB shard named to sort first with the 23 widened shared experts (A's own bytes, then 64 all-zero neurons), its own 718 MB MTP file with the head's shared expert widened the same way, and a `config.json` with the two keys the engine's writer sets. Checks:
- E's MTP file is **byte-identical** to the one the engine itself wrote into T's folder (`cmp`), and E's widened shared experts are byte-identical to T's in the 7 layers where T holds no fact (42 of 42 tensors). So E is what the engine writes for "widened, no facts".
- Every tensor's own bytes in T and in E equal A's (763 of 763 tensors), the only config keys that differ from A are `moe_shared_expert_intermediate_size` and `tensorfold_learned_neurons`, and E's config equals T's.
- E decodes the same tokens as A in every run below (sha `2e5c4aac8a41`, 24 dev CLI runs, drafted and not), as it must: an all-zero neuron adds exactly zero. T decodes a different reply to the same neutral prompt (sha `716feda01d52`), so these runs can see the pocket.
- Disk: no model copy was made; E is 0.98 GB of new data.

**Binaries:**
- Server: the existing 1.0.8 build `~/tf108/zig-out/native/bin/tensorfold-native`, sha256 `759114dd…545a`, built from `0efbd447` with `zig build native -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 -Dversion=1.0.8` and Zig 0.17.0 (READ, `~/tf108/_notes/build.sh`).
- Dev CLI `tensorfold` and `tf-cuda-test`: built by me from a clean `git archive 0efbd447` in `~/tf-pocket-cuda/src108` with the same Zig, nvcc and flags (`zig build -Dnvcc=... -Dsm=121 -Dversion=1.0.8`). All 32 kernel fatbins are byte-identical to the server build's (MEASURED, sha256); the host executables differ only as builds in different directories do (my rebuild of `tensorfold-native` is not byte-identical either, so I serve the existing one).

### 2.1 GPU time per decode step (the cleanest number)

`tensorfold rounds MODEL --tokens <p1, 36 ids> --max-tokens 400`: after the prompt, the captured serial graph (one token plus the feed) launched 400 times back to back, timed with CUDA events, then the 1-row verify window graph the same way; each measured twice per load. 4 reps, order rotated (A T E, T E A, E A T, A T E), 60 s idle before each rep. Files: `~/tf-pocket-cuda/raw/rounds-r2*`.

| Graph | A | T | E | T minus A, same rep | E minus A, same rep |
|---|---|---|---|---|---|
| Serial round, ms | 9.944 (9.800-9.968) | 11.507 (11.467-11.570) | 11.498 (11.449-11.518) | **+1.591** (+1.507 to +1.706) | **+1.559** (+1.539 to +1.659) |
| 1-row window, ms | 9.409 (9.267-9.447) | 10.987 (10.947-11.080) | 11.034 (10.909-11.076) | +1.616 (+1.525 to +1.714) | +1.632 (+1.582 to +1.681) |

Medians of 8, with the range. In tok/s, the serial round is 100.6 untaught against 86.9 taught and 87.0 empty: **taught/untaught 0.864**, and the empty pocket costs the same as the 64-fact one (MEASURED). That is about 69 µs a MoE layer (1.59 ms / 23, arithmetic).

### 2.2 Undrafted and drafted decode, dev CLI

`tensorfold run MODEL --tokens <p1> --max-tokens 256 [--no-drafts]`, greedy. 6 bursts, each one run per folder and mode, order rotated, 90 s idle before each burst. Files: `~/tf-pocket-cuda/raw/cli-c1/`.

| Mode | A tok/s | T tok/s | E tok/s | T/A, same burst | E/A, same burst | Extra ms a token, E |
|---|---|---|---|---|---|---|
| Drafts off | 100.72 (100.56-101.18) | 87.19 (86.70-87.76) | 87.13 (86.79-87.33) | **0.866** (0.862-0.868) | **0.864** (0.861-0.868) | +1.563 (+1.507 to +1.600) |
| Drafts on (the CLI's MTP rounds) | 150.83 (149.95-151.14) | 127.59 (127.38-127.89) | 138.67 (138.14-139.57) | 0.848 (0.843-0.850), different text | **0.922** (0.916-0.924) | +0.560 (+0.545 to +0.603) |

- Drafts off, the extra time per token is the same as in 2.1: +1.54 ms (T) and +1.56 ms (E).
- Drafts on, A and E write the same 256 tokens in the same 100 rounds with 155 accepted drafts, because all three used the same cached draft-cost table (MEASURED: identical "measured: 9.76/12.32/16.19/22.35/31.39 ms" line in every drafted run; READ: the cache key has no pocket width, `cuda_costs.zig:35`). So E/A = 0.922 is the pocket's drafted cost on identical work: about 1.4 ms a round of 2.56 tokens.
- T's drafted reply is a different text (107 rounds, 148 accepted), so its 0.848 mixes the pocket's cost with a different draft acceptance.
- Prompt (36 tokens, one chunk), drafts off: 52.8 ms A against 54.7 ms T and 54.8 ms E (medians of 6, MEASURED, `prefill` in the same runs); nsys puts 2.0 ms of that on the pocket (2.4).

### 2.3 Where the time goes inside a window (engine's own profile)

`tf-cuda-test window-profile MODEL 1,4,16 p1-off.ids`: a verify window at position 20, timed as its graph, then eager with timing events between kernel classes, medians of 5 (READ, `zig/tests/cuda/window_profile.zig`). `shared_wait` is the main stream's wait for the side stream (the shared expert plus the pocket) after the routed experts. Files: `~/tf-pocket-cuda/raw/wp-w1/`.

| Rows | Graph ms A / T / E | `shared_wait`, 23 calls, ms A / T / E | Routed down projection (`experts_down`), ms A / T / E |
|---|---|---|---|
| 1 | 9.957 / 11.346 / 11.357 | **0.020 / 1.574 / 1.534** | 1.861 / 1.804 / 1.816 |
| 4 | 16.016 / 16.755 / 16.793 | 0.026 / 0.871 / 0.825 | 4.801 / 5.008 / 4.953 |
| 16 | 32.403 / 33.479 / 33.253 | 0.040 / 0.125 / 0.210 | 12.607 / 13.558 / 13.408 |

- At one row, nearly all of the cost is the main stream waiting for the side stream: 1.53-1.57 ms over 23 layers, about 67 µs a layer, against 0.02 ms untaught (the negative control). This is the check our llwz plan asked for: the wait grows on a pocket and does not move without one (MEASURED).
- At 16 rows the routed experts take longer, so the wait mostly disappears, but the routed down projection gets 0.8-0.95 ms slower: the pocket's 16 blocks now compete with it (INFERRED from the class times). The 16-row window still costs 0.85-1.08 ms more.
- With the shared expert forced onto one stream, the pocket's time lands in the class after it (`norm`): +1.97 ms (T) and +1.87 ms (E) over 23 calls at one row, about 81-86 µs a launch with nothing to hide behind.

### 2.4 nsys

`nsys profile -t cuda --cuda-graph-trace=node` (nsys 2025.3.2) around one dev CLI run of 256 tokens; kernel records read from the sqlite export (`tools/nsys_kern.py`). Files: `~/tf-pocket-cuda/raw/nsys/n1-*`.

| Run | `tf_pocket_rows` launches in decode | Per launch, µs: median (p10-p90) | Pocket share of decode kernel time | Decode ms/token under nsys |
|---|---|---|---|---|
| T, drafts off | 5,865 = 23 x 255 tokens, grid 1 | **132.0** (122.2-142.6) | **18.9%** (791 of 4,184 ms) | 11.506 |
| E, drafts off | 5,865, grid 1 | 131.7 (122.0-142.4) | 18.8% | 11.491 |
| A, drafts off | 0 | - | 0% | 9.884 |
| T, drafts on | 2,776, grids 1-16 | 128 (grid 2, 1,127 launches) to 495 (grid 16) | 17.6% | 7.935 |
| E, drafts on | 2,582, grids 1-16 | 136 (grid 2) to 491 (grid 16) | 17.1% | 7.279 |

- The other kernels change little: `expert_kernel` 78.7 µs (T) against 78.1 µs (A), `gemv_kernel` 35.7 against 34.0 µs (MEASURED, median per launch). The 5% on `gemv_kernel` may be the pocket competing for memory while both streams run (INFERRED).
- In the 36-row prompt chunk each of the 23 launches takes 86.5 µs (2.0 ms a prompt chunk).
- One launch takes 132 µs, but the decode step grows by only about 68 µs a layer (2.1-2.3), so about half of each launch is hidden behind the routed experts on the other stream (INFERRED, arithmetic).
- Kernel time adds to more than the decode span (4,184 ms of kernels in a 2,934 ms span for T) because the two streams overlap.

### 2.5 The 1.0.8 server with TensorFold's client and ours

Six servers, one at a time on port 18190, in the order A T E E T A, 120 s idle before each, each folder served in place with `TENSORFOLD_NO_LIVE=1 tensorfold-native serve FOLDER --host 127.0.0.1 --port 18190 --no-update-check` (no `--slide`). Each server ran, in order:
- **undrafted:** our `decode_bench.py`, `"draft": false`, thinking off, 8 neutral prompts, 2 warm-up and 16 measured requests of 256 tokens, greedy;
- **drafted, the brief's form:** `tools/bench_openai.py BASE MODEL` with its defaults (64 tokens, 5 reps, temperatures 1.0 and 0, the two built-in prompts);
- **drafted, 256 tokens:** `tools/bench_openai.py BASE MODEL --tokens 256 --reps 3 --temperatures 0`.

Files: `~/tf-pocket-cuda/raw/s1-*`. Every server's `*.safetensors` and `config.json` hashed the same before and after, and the folders' file lists, sizes and times did not change (MEASURED, 6 of 6). No server wrote into a folder.

**Undrafted** (decode tok/s, median of 16 requests per server, two servers per folder):

| Folder | Server 1 | Server 2 | Over A |
|---|---|---|---|
| A | 100.10 | 99.77 | |
| T | 86.42 | 86.30 | **0.864** |
| E | 86.36 | 86.23 | **0.863** |

Every request ended on length, the server's own rate agreed with the stream's (1.000), and each folder gave the same `token_sha` per prompt on both of its servers. E's 8 replies equal A's; T's differ from A's on 7 of 8 prompts.

**Drafted, `bench_openai.py`** (decode tok/s, mean of the two servers' medians):

| Cell | A | T | E | T/A | E/A |
|---|---|---|---|---|---|
| code, 64, greedy | 141.6 | 128.0 | 127.0 | 0.904 (same text) | **0.897** |
| chat, 64, greedy | 156.4 | 146.1 | 142.5 | 0.934 (same text) | **0.911** |
| code, 64, temperature 1 | 159.8 | 146.0 | 146.4 | 0.913 | 0.916 |
| chat, 64, temperature 1 | 133.1 | 125.8 | 124.5 | 0.946 | 0.936 |
| code, 256, greedy | 201.5 | 181.1 | 187.4 | 0.898 | **0.930** |
| chat, 256, greedy | 145.5 | 143.6 | 133.1 | 0.987 | **0.915** |

- E's tokens equal A's in every cell (`token_sha` of every request), so E/A is the pocket's cost on identical work: **6.4% to 10.3% slower drafted**.
- T writes A's text in the two greedy 64-token cells (0.904 and 0.934) and its own text elsewhere, where drafting acceptance differs, so its other ratios are not like for like.
- The draft-cost table is cached per executable, GPU and model shape, and the shape key has no pocket width (READ, `zig/src/families/nemotron/cuda_costs.zig:35-62`). So all six servers read the same table and planned drafts with the same prices, as did all dev CLI runs in 2.2 (the same "measured" line in each).
- Note on E and the server's Sliding Weights view: `/v1/slide/graph` reported 64 neurons in use in every layer for E. The view reads every shard's header, and E's symlinked published shards hold the unwidened shared experts, whose last 64 own rows the view takes for a pocket (INFERRED from `slide_model.zig:79-145`). The forward reads the overlay, as the identical tokens show. For T the view's counts equal the ones I read from the shards: 0,0,0,0,41,41,41,43,43,42,43,44,43,44,3,3,3,3,2,1,0,0,0.

### 2.6 Prompt cost

`tensorfold run MODEL --tokens-file <4,096 ids> --max-tokens 2 --no-drafts` (p1's ids repeated, so two full 2,048-row chunks), 3 reps, order rotated, 60 s idle before each. Files: `~/tf-pocket-cuda/raw/prefill-p1/`.

| Folder | Prompt seconds, median (range) | Prompt tok/s |
|---|---|---|
| A | 0.536 (0.525-0.537) | 7,640 |
| T | 0.595 (0.586-0.597) | 6,880 |
| E | 0.595 (0.591-0.598) | 6,880 |

The pocket makes a 4,096-token prompt **about 11% slower** (+59 ms), empty or full. With 2,048 blocks per launch, each re-reading the pocket's 1.38 MB, that is about 1.3 ms per launch (46 launches; INFERRED, arithmetic). All three gave the same first token.

### 2.7 Summary, and what it changes

| Instrument | Untaught (A) | Taught, 64 facts (T) | Empty pocket (E) | Pocket cost |
|---|---|---|---|---|
| Serial decode graph, GPU ms a token (2.1) | 9.944 | 11.507 | 11.498 | +1.56 to +1.59 ms, 0.864 |
| Dev CLI, drafts off, tok/s (2.2) | 100.7 | 87.2 | 87.1 | 0.864-0.866 |
| Server, `"draft": false`, tok/s (2.5) | 99.9 | 86.4 | 86.3 | 0.863-0.864 |
| Server, `bench_openai.py` defaults, drafted (2.5) | | | | E/A 0.897-0.936 |
| Prompt, 4,096 tokens (2.6) | 0.536 s | 0.595 s | 0.595 s | +11% |
| `tf_pocket_rows`, one decode launch (2.4) | none | 132 µs | 132 µs | 18.9% of decode kernel time |

What this changes (INFERRED unless marked):
- **The cost on the GB10 is not small.** Undrafted it is 13.6% (+1.56 to +1.59 ms a token, MEASURED). That is almost the same absolute cost per token as philip-pentatonic's +1.58 ms on the RTX PRO 6000 (290 to 199 tok/s on 1.0.7); it is a smaller share only because the GB10 decodes slower. A fixed cost per token that is the same on a much larger GPU fits a kernel limited by one SM, as he read it.
- **Empty costs the same as full on the GB10 too** (MEASURED: E against T in every instrument). philip-pentatonic's fix 1 (skip a layer whose pocket is all zero, decided at load) removes the whole cost for E. For a taught model it removes the 7 empty layers here plus the MTP head's launch in every drafted round, because that head's pocket is always zero (READ, 1.1).
- **For a well-filled pocket, fix 1 alone is not enough to be sure.** In T, 16 of 23 layers hold neurons and 10 of them hold 41-44, so most launches stay. Each launch takes 132 µs, of which about 64 µs is already hidden behind the routed experts. Launching over only the neurons in use would shorten the 10 heavy layers' launches by about a third if the time scales with the neuron count, which might bring T's cost down to a few tenths of a ms. That is a rough model, not a measurement. philip-pentatonic's fix 2 (many blocks, then a second kernel over the output dimension) is the one that should remove the cost whatever the fill.
- **Prompt chunks need it as well.** The pocket makes prompts 11% slower (2.6), and the PR template requires prompt speed no slower than the last release. In a prompt chunk the launch already has one block per row; the cost there is every block reading the whole pocket again. A fix that only handles decode leaves that 11%.
- **CUDA graphs:** a per-layer count or a skipped launch is captured into the graphs. The fix must capture them again when the learner changes a count (it already does so when it first makes a pocket), or keep the count in device memory (READ + INFERRED, 1.5). Shrinking `n` itself would make `pocketRead` return 0 for a loaded pocket (1.1).
- **Who writes it:** in #678 philip-pentatonic offered to send the PR for fixes 1 and 2 himself, with exactness receipts. That should be settled with him before we write a CUDA PR (READ, his comment of 12 Oct 2026).
- **Test folders for the fix:** A, T and E are in place on gb10-a and need no copies. E (empty) must give A's tokens and, after fix 1, A's speed; T must give its own tokens before and after.

### 2.8 Not measured

- More than one stream at a time, `bench_concurrent.py`, thinking on, sampled decoding beyond `bench_openai.py`'s defaults.
- A folder with a full pocket (64 neurons in use in every layer).
- Prompt lengths other than 36 and 4,096 tokens.
- Any GPU other than this GB10.

### 2.9 Raw data, scripts and incidents

- Everything is on gb10-a under `~/tf-pocket-cuda/` (`raw/`, `tools/`, `models/A-empty64*`, `src108/`). A copy without the nsys traces (1.5 MB) is in `~/tmp/lw-logic/perf/cuda-raw/`. The nsys traces (`raw/nsys/*.nsys-rep` and `.sqlite`, 52 MB) stay on gb10-a.
- Scripts: `rounds.sh` (2.1), `cli.sh` (2.2), `wp.sh` (2.3), `nsys.sh` and `nsys_kern.py` (2.4), `servers.sh` and `srv.py` (2.5), `prefill.sh` (2.6), `mk_empty.py` (folder E and its checks), `summ.py` (the medians). Every GPU script takes `~/tf-pocket-cuda/GPU.lock` and waits while another engine process runs.
- **Discarded runs (incident, 08:01-08:10 UTC):** a second copy of this agent, resumed by the coordinator while the first was still running, started the same `rounds.sh` at the same time as mine. Both ran on the GPU together, and the serial round read 21-25 ms instead of about 10 ms. Those rounds (`raw/rounds-r1*`) and the three sanity runs just before them (`raw/sanity/`) are not used for any number here. Their token hashes agree with the clean runs. Every number above comes from 08:16 UTC onwards, after a check that no other engine process was running.
- Disk: gb10-a had 24 GB free throughout (98% used after the L7 run). I made no model copy; my files take about 1.4 GB (E's two files 0.98 GB, the source tree and build cache 0.36 GB, the traces 52 MB).

## 3. Result: the fix in ashhart/TensorFold#787 (`07189f62`) against 1.0.8 on the GB10

**In short (MEASURED):** with `07189f62` the pocket no longer costs anything measurable on the GB10, empty or with 64 facts, drafted or not, and no token changes.
- Drafts off, 64 facts: 88.2 to 102.1 tok/s in the dev CLI (1.158x) and 86.9 to 100.0 tok/s on the server (1.150x). Empty pocket: 88.1 to 102.4 (1.163x) and 86.7 to 101.2 (1.167x). Untaught: unchanged (0.998x and 1.000x).
- Drafted, `tools/bench_openai.py`: 1.04x to 1.09x with 64 facts, 1.04x to 1.12x empty, 0.98x to 1.01x untaught. The empty pocket now decodes at the untaught speed in every cell (E/A 0.994 to 1.019).
- A 4,096-token prompt: 8% faster with 64 facts, 10% faster empty, 1% faster untaught. With 64 facts it is still 2.8% slower than untaught (10.5% in 1.0.8).
- Every one of the 120 requests a server sends, on 12 servers, gives the same `token_sha` before and after on its folder, and so do all 428 dev CLI fact replies per folder.

**Who and when:** the runs below were made on gb10-a on 13 Oct 2026, 10:21-13:46 UTC, by the scripts in `~/tf-pocket-cuda/tools/v`, started by an earlier measuring session of a reviewer agent. That session lost its network at about 13:15 UTC; its stage chain kept running on gb10-a and ended every stage with exit 0 (`raw/v/chain.out`, "chain done" at 13:45:49). This section was written by a second reviewer agent session at 14:28-14:50 UTC, which read every file, checked it as described below and redid nothing: no stage was missing and none failed a check. This session ran no GPU job. Grades as in sections 1-2; "the measuring session" marks what that session ran and I only read.

### 3.1 Builds and conditions

- **Before:** `0efbd447` (v1.0.8) in `~/tf-pocket-cuda/before`. **After:** `07189f62` (branch `pocket-cuda`) in `~/tf-pocket-cuda/src`. Every file in each tree has the sha256 of `git archive` of its commit, 1,704 and 1,705 files (MEASURED by me at 14:35 UTC). Both were built on gb10-a by `tools/build_tree.sh` with Zig 0.17.0, nvcc 13.0.88, `-Dsm=121 -Dversion=1.0.8` (install and native), 09:31-10:00 UTC, before any run below.
  - This "before" is a fresh build of `0efbd447`, not the `~/tf108` server binary of section 2, so both builds were made the same way.
  - sha256: `tensorfold-native` before `b0a5a4f3…a333`, after `4f512013…630e`; dev CLI `tensorfold` before `0f0fe4b0…80d7`, after `37efd269…535f`. Every script refused to run on another hash, and the files still have these hashes at 14:36 UTC (MEASURED).
- **Machine:** gb10-a as in section 2; Ubuntu 24.04.5, kernel 7.0.0-1019-nvidia, driver 580.178.04, CUDA 13.0. Clients: Python 3.12.14 with `0efbd447`'s `tools/` (`bench_openai.py` and `bench_concurrent.py` are the same in both commits, READ: the PR touches no tool).
- **Folders:** A, T and E of section 2, served in place, read-only.
- **Servers:** `TENSORFOLD_NO_LIVE=1 tensorfold-native serve FOLDER --host 127.0.0.1 --port 18190 --no-update-check`, no `--slide`, one at a time, each with a fresh empty HOME so it timed its own draft-window costs at startup (`tools/v/srv2.py`). sha256 of every `*.safetensors` and `config.json`, and the folder's file list with sizes and times, before and after every server: unchanged on 12 of 12 (MEASURED by the measuring session, `run.json`; the three folders also hashed the same across all 12 servers). The dev CLI bursts and the fact runs hashed the folders before and after too: unchanged.
- **Order and idle:** two series of six servers, `s1` = bT aT aA bA bE aE and `s2` = aE bE bA aA aT bT (b = before, a = after), so each folder ran ABBA or BAAB across the series, 120 s idle before each server. Dev CLI: builds alternated in each burst and their order flipped every burst, folders rotated, 60-90 s idle before each burst.
- **The other agent on gb10-a:** its live-learning test copied two model folders at 12:24:30-12:24:45 UTC, hashed them until 12:27:12 and ran a CPU-only build at nice 19 from 12:28:56 to 12:30:01 (its `~/tf-pocket-cuda/live/*.log`). That falls in the `s1` server 3-aA's fact questions (exactness only, no timed number from them) and the idle gap after it. Its first GPU job I saw was a server on port 18191 at 14:41 UTC, after all of this. Every script held `GPU.lock` for its whole stage and checked for other engine, test and nsys processes before each run (READ, `tools/v/common.sh`). The same build and folder agree between `s1` and `s2` within 1.3% undrafted and within 2.3% drafted, except one cell explained in 3.4 (MEASURED). I take the series as clean (INFERRED).

### 3.2 Exactness

**Server.** Nemotron 3.5 Lightning 30B-A3B MLX 4-bit, gb10-a (GB10), 12 servers. Each server sends 120 requests that carry a `token_sha`: `bench_openai.py` defaults (20), its 256-token form (6), our `decode_bench.py` with `"draft": false` (16), `bench_concurrent.py --alone --serial` with its defaults (48: code and chat, temperatures 1 and 0, 4 streams, 3 reps) and with CONTRIBUTING.md's form `--tokens 256 --reps 3 --temperatures 0 --levels 1,4 --alone --serial` (30). Files `raw/v/s1`, `raw/v/s2`; summary `perf/cuda-raw/verify/raw/summ_servers.txt`.

| Check (MEASURED) | A | T | E |
|---|---|---|---|
| Same `token_sha` before and after, every request, all 4 servers of the folder | 120 of 120 | 120 of 120 | 120 of 120 |
| The two servers of one build agree | 120 of 120, both builds | 120 of 120, both builds | 120 of 120, both builds |
| `bench_concurrent`: concurrent reply equals its solo run, per server | 78 of 78 | 78 of 78 | 78 of 78 |
| `bench_concurrent --serial`: solo run equals `"draft": false`, per server | 12 of 12 | 12 of 12 | 12 of 12 |
| Failed or unmeasured requests | 0 | 0 | 0 |

- **Negative control:** in the same build, T's `token_sha` differs from A's on 98 of the 120 requests: all 30 of the CONTRIBUTING.md form, 45 of 48 of the `bench_concurrent` defaults, 14 of 16 `"draft": false`, all 6 of the 256-token `bench_openai` form and 3 of its 20 defaults. T gives A's tokens only in the two greedy 64-token `bench_openai` cells (10 requests), on 10 sampled requests and on 2 of the 8 `"draft": false` prompts. So the hashes can see the pocket. E equals A on 120 of 120, as it must: an all-zero neuron adds exactly zero.
- **Resumed prompt:** on all 12 servers the first chat request (cached 0) and the next one, the same prompt and seed resumed with 29 of its 35 tokens from the prompt cache, gave the same `token_sha` (`657018d897ad`; MEASURED from the server logs). bench_openai's other chat requests are resumed the same way, before and after alike.
- Every request was complete with the requested token count; every `"draft": false` request ended on length.
- `--serial` per server is 10 checks from the defaults and 2 from the CONTRIBUTING.md form (the tool reports those 2 once per level, so its own cells add to 14).

**Dev CLI.** Same model and machine. `tensorfold lanes FOLDER QUESTIONS --solo [--no-drafts]` and all at once (the lane core), full token ids: the 134 L7 fact questions with thinking off (160 tokens) and 40 of them with thinking on (1,500 tokens) (measuring session, 10:21-11:22 UTC, `raw/v/lanes-q1`).

| Check (MEASURED) | A | T | E |
|---|---|---|---|
| Same tokens before and after (6 sets: 134 + 134 + 40 + 40 + 40 + 40) | 428 of 428 | 428 of 428 | 428 of 428 |
| Drafted equals `--no-drafts`, each build | 174 of 174 | 174 of 174 | 174 of 174 |
| 40 at once equal their solo run, each build | 80 of 80 | 80 of 80 | 80 of 80 |

- **Negative control:** in each build T equals A on only 19 of 134 replies with thinking off and 0 of 40 with it on; E equals A on 174 of 174.
- T's replies before and after equal the L7 run's own last probe of that folder (1.0.8 server, `~/lw-runs/gb10-a-L7-seed1/steps/main-restart/answers-teach.jsonl`) in text and token count: 134 of 134 with thinking off and 40 of 40 with it on (MEASURED: the measuring session's `cmp_l7.py`, which I ran again on the same files with the same result). The fix keeps every taught fact's answer.
- Server, 40 fact questions, thinking off and on, drafted and `"draft": false` (`s1` only): 160 of 160 replies equal before and after on each folder, and drafted equals `"draft": false` on 80 of 80 per server (MEASURED).
- 256 greedy tokens from p1 (3.3): every run has its folder's hash before and after, drafts on and off (`2e5c4aac8a41` on A and E, `716feda01d52` on T, 24 runs per folder). The 4,096-token prompt gives the same first token in all 18 runs.

**Kernel test** (measuring session, 09:34-10:00 UTC, `raw/fix/`; I read the logs). `tf-cuda-test pocket` on the after build: 194 cases byte-equal to `tf_pocket_rows` (widths 2,688 and 2,728, 1 to 300 rows plus two 2,048-row prompt chunks, 0 to 64 neurons in use, fp32 and bf16 out; READ, `pocket_tests.zig`). The controls (neurons summed in reverse, inputs rotated by one) change 4,140 to 323,207 values, so the test can see an order change. Four deliberately broken kernels (`tools/mutants.sh`: reversed batch order, no relu, no fma, last output dropped) each FAIL it. On 1.0.8 the command does not exist (`UnknownCommand`), so the test cannot run before the change.
- One case is not bit-identical by construction (READ, the PR text and `cuda_forward.zig`): a skipped layer leaves an output of exactly -0.0 as -0.0, where 1.0.8 added +0.0 and made it +0.0. No reply above changed.

### 3.3 Decode speed, drafts off

Nemotron 3.5 Lightning 30B-A3B MLX 4-bit, gb10-a (GB10), p1 (36 tokens) for the dev CLI. Medians with the range; "same burst" pairs runs of one rep or burst.

| Instrument (MEASURED) | A before | A after | T before | T after | E before | E after |
|---|---|---|---|---|---|---|
| Serial graph round, GPU ms (`tensorfold rounds`, 4 reps x 2) | 9.864 | 9.857 | 11.487 | **9.893** | 11.474 | **9.812** |
| 1-row window graph, GPU ms | 9.330 | 9.343 | 11.010 | 9.376 | 11.025 | 9.325 |
| Dev CLI `run --no-drafts`, 256 tokens, tok/s (6 bursts) | 102.26 | 102.13 | 88.19 | **102.06** | 88.08 | **102.41** |
| Server `"draft": false`, 16 x 256 tokens, tok/s (mean of 2 servers) | 100.25 | 100.28 | 86.94 | **99.96** | 86.73 | **101.19** |

- Serial round, before minus after, same rep: A +0.002 ms (-0.018 to +0.044), T **+1.596** (+1.563 to +1.653), E **+1.646** (+1.606 to +1.686). After, T minus A is +0.045 ms (+0.008 to +0.060) and E minus A -0.040 (-0.070 to +0.040): of 1.0.8's +1.6 ms a token, about 0.045 ms is left with 64 facts and nothing with the empty pocket.
- Dev CLI, after over before in the same burst: A 0.998 (0.993-1.002), T **1.158** (1.144-1.174), E **1.163** (1.152-1.169). After, T/A 1.003 (0.989-1.013) and E/A 1.007 (0.996-1.009). In ms a token: T 11.338 to 9.798, E 11.354 to 9.764, A 9.779 to 9.791.
- Server: after over before A 1.000, T 1.150, E 1.167; after, T/A 0.997 and E/A 1.009. Per server: A 100.13, 100.37 to 100.07, 100.48; T 87.28, 86.60 to 100.60, 99.32; E 86.60, 86.85 to 101.79, 100.59. Every request ended on length after 255 rounds and the server's own rate agreed with the stream's (0.993 to 1.000).
- This session's A runs about 1.5% faster than section 2's (9.78 against 9.93 ms a token in the dev CLI, the same commit rebuilt); only the before and after pairs within this section are compared.

### 3.4 Decode speed, drafted

Nemotron 3.5 Lightning 30B-A3B MLX 4-bit, gb10-a (GB10). `tools/bench_openai.py BASE MODEL` with its defaults (64 tokens, 5 reps, seeds 1234 + i, temperatures 1 and 0) and with `--tokens 256 --reps 3 --temperatures 0`. Decode tok/s, the median of the reps on each server, then the mean of the folder's two servers per build (MEASURED).

| Cell | A before | A after | after/before | T before | T after | after/before | E before | E after | after/before | E/A after |
|---|---|---|---|---|---|---|---|---|---|---|
| code, 64, temperature 1 | 161.9 | 159.4 | 0.984 | 147.1 | 160.1 | **1.089** | 148.5 | 162.4 | **1.093** | 1.019 |
| chat, 64, temperature 1 | 134.9 | 136.4 | 1.011 | 127.1 | 137.3 | **1.080** | 121.8 | 136.7 | **1.123** | 1.002 |
| code, 64, greedy | 141.8 | 140.5 | 0.991 | 130.5 | 142.7 | **1.093** | 130.7 | 139.7 | **1.069** | 0.994 |
| chat, 64, greedy | 157.0 | 158.4 | 1.009 | 154.4 | 160.9 | **1.043** | 151.6 | 158.0 | **1.043** | 0.997 |
| code, 256, greedy | 203.4 | 203.2 | 0.999 | 184.5 | 195.9 | **1.062** | 191.7 | 202.4 | **1.056** | 0.996 |
| chat, 256, greedy | 148.1 | 147.7 | 0.997 | 147.3 | 155.7 | **1.057** | 137.6 | 147.9 | **1.075** | 1.001 |

- E writes A's tokens in every cell, so E/A is the pocket's remaining drafted cost on identical work: 0.994 to 1.019 after, against 0.902 to 0.965 before (MEASURED).
- T writes A's text in the two greedy 64-token cells: there T/A is 0.920 and 0.983 before and 1.016 and 1.016 after. In the other cells T's reply is its own text and its ratios to A are not like for like.
- **Dev CLI, drafts on** (`run`, 256 greedy tokens, 6 bursts): A 153.07 to 152.54 tok/s (0.996, 0.991-1.005), T 128.13 to 138.73 (**1.083**, 1.078-1.087), E 140.81 to 152.51 (**1.082**, 1.079-1.088); after, E/A 0.999 (0.998-1.002). A and E take 100 rounds with 155 accepted drafts in both builds; T 108 rounds and 147 accepted before, 107 and 148 after.
- **The draft-window costs each server timed** (MEASURED, decoded from each server's HOME by `costs.py`): a 1-row verify window 11.15-11.18 ms before on T and E and 9.46-9.72 ms after, against 9.63-9.66 ms on A in both builds; one draft level 0.764-0.788 ms before and 0.712-0.718 ms after (A 0.702-0.716). So the planner now prices T and E as A, which is part of the drafted gain (INFERRED).
- **The one cell that moves between series:** A's greedy 64-token code cell reads 136.6 and 139.5 tok/s in `s1` against 144.4 and 144.2 in `s2`. The `s1` servers drafted it in 30 rounds and the `s2` servers in 29 (MEASURED from the server logs), from their own timed costs. It moves the same way in both builds, so the ABBA mean cancels it (INFERRED).

### 3.5 Prompt speed

`tools/prefill_cold.py` is not in 1.0.8. As in 2.6: `tensorfold run MODEL --tokens-file <4,096 ids> --max-tokens 2 --no-drafts` (p1's ids repeated, two full 2,048-row chunks), 3 reps, folders rotated, builds alternated, 60 s idle before each rep (measuring session, 11:53-11:57 UTC, `raw/v/prefill-p1`). Nemotron 3.5 Lightning 30B-A3B MLX 4-bit, gb10-a (GB10).

| Folder (MEASURED) | Before, s | After, s | After/before, same rep | Prompt tok/s, before to after | Over A, before / after |
|---|---|---|---|---|---|
| A | 0.533 (0.529-0.536) | 0.524 (0.523-0.534) | 0.991 (0.980-0.995) | 7,678 to 7,818 | |
| T | 0.592 (0.586-0.593) | 0.543 (0.539-0.550) | **0.917** (0.911-0.940) | 6,924 to 7,536 | 1.105 / **1.028** |
| E | 0.592 (0.581-0.596) | 0.533 (0.525-0.534) | **0.901** (0.882-0.917) | 6,915 to 7,679 | 1.117 / **1.005** |

- With 64 facts a 4,096-token prompt still costs about 2.8% over untaught (1.018-1.053), down from 10.5%; the empty pocket's cost is gone. Prompt processing is no slower than 1.0.8 on any folder.
- The 36-token prompt of 3.3 (one chunk, drafts off, medians of 6): A 52.5 to 51.7 ms, T 54.1 to 52.3 ms, E 54.4 to 51.9 ms.

### 3.6 Where the time went

`nsys profile -t cuda --cuda-graph-trace=node` around one dev CLI run of 256 tokens each (measuring session, 13:38-13:42 UTC, `raw/v/nsys-n1`), and `tf-cuda-test window-profile MODEL 1,4,16` once per folder and build (13:42-13:46 UTC, `raw/v/wp-w1`). Nemotron 3.5 Lightning 30B-A3B MLX 4-bit, gb10-a (GB10). MEASURED unless marked.

| Run, drafts off | Pocket launches in decode | Per launch, µs (median) | Pocket share of kernel time | Decode ms a token under nsys |
|---|---|---|---|---|
| T before | 5,865 `tf_pocket_rows` (23 x 255), grid 1 | 134.5 | 18.2% | 11.433 |
| T after | 4,080 `tf_pocket_in` + 4,080 `tf_pocket_out` (16 x 255) | 31.3 + 13.7 | 4.7% | 9.883 |
| E after | none | - | 0% | 9.781 |
| A after | none | - | 0% | 9.839 |

- After the fix T launches the pocket only in its 16 layers with neurons in use, and E launches none: E's run has the same 93,150 kernel launches as A's. Drafted, T's pocket share is 2.5%.
- Window profile, the main stream's wait for the side stream (`shared_wait`, 23 calls) at 1 row: T 1.529 to 0.020 ms, E 1.546 to 0.024 ms, A 0.020 and 0.020 ms; at 4 rows T 0.720 to 0.022 ms and E 0.805 to 0.036 ms. With the shared expert forced onto one stream, the class after it (`norm`) at 1 row: T 2.728 to 0.967 ms, E 2.731 to 0.835 ms, against A 0.805 and 0.830 ms, so about 0.14 ms over 16 layers is what the pocket still takes when nothing hides it (INFERRED, arithmetic). Each window profile is a single run; their graph times move by up to 1.3 ms between runs, so only the class times are used here.
- So the decode cost is gone for two reasons together: empty layers launch nothing, and a used layer's launch drops from about 134 µs on one SM to about 45 µs spread over many blocks, short enough to hide behind the routed experts on the main stream (INFERRED from the two instruments).

### 3.7 Tests and checks

- `zig build test`, both trees (measuring session, 10:07-10:08 UTC, `raw/fix/test-*.log`): 616 of 621 passed, 5 skipped, 0 failed, 44 of 44 steps, the same counts before and after (MEASURED; I summed the logs).
- `python3 tools/zig/lean_check.py` on clean `git archive` trees (MEASURED by me): 539 problems before and 539 after, the same list; the new test file adds none (1,265 and 1,266 files).
- `zig fmt --check zig build.zig` with Zig 0.17.0 on both trees: clean (MEASURED by me, 14:38 UTC).
- `cuobjdump -res-usage` of the after build's `nemotron_ops.fatbin` for sm_86, sm_120 and sm_121 (`~/tf-pocket-cuda/fatbins-3sm-src`): all five pocket kernels use no stack and no local memory, so nothing spills; registers 38-128 (MEASURED by me). The 3-SM build itself ended rc 0 (`raw/fix/fatbins-3sm.log`).

### 3.8 Not measured

- Learning with `--slide` on the after build: the count upload and the recapture when a skipped layer gets its first neuron are read, not exercised (READ, `cuda_slide.zig`).
- Any GPU other than this GB10, including philip-pentatonic's RTX PRO 6000; sm_86 and sm_120 were compiled, not run.
- Cold prefill at 2k, 8k, 32k and 64k tokens (no `prefill_cold.py` in 1.0.8); prompt lengths other than 36 and 4,096.
- More than 4 concurrent server streams (the dev CLI lane core ran 40 at once); thinking on only in the fact questions.
- A folder with a full pocket (64 neurons in use in every layer).

### 3.9 Files

- On gb10-a: `~/tf-pocket-cuda/raw/v/` (all series), `raw/fix/` (kernel test, mutants, host tests), `tools/v/` (the scripts), `homes/` (each server's timed draft costs). A copy without the nsys traces (8.7 MB) is in `~/tmp/lw-logic/perf/cuda-raw/verify/raw/`, with the summaries `summ_servers.txt`, `summ_cli.txt`, `costs.txt`, `cmp-l7.txt`, `lean-*.txt` and `test-*.log`; the summary scripts are `verify/tools/summ_servers.py` and `summ_cli.py`.
- The measuring session's own partial summaries in `verify/raw/` (`*.summary.json`, `lanes-q1.cmp.json`, `srv-cmp-partial.json`) agree with these numbers where they overlap.

## 4. Against 1.0.9

**Why:** Ash released TensorFold 1.0.9 on 13 Oct 2026, and #787 was rebased onto it as branch `pocket-cuda-109`: `c36b536d` (tag v1.0.9: the release commit `72b45368` plus a CHANGELOG commit) and five commits: `e7b65ff4`, `9803c8fc`, `0ffaefc2`, `a3df10ea` and `76fe1de4` (the last one: the pocket test adds the 4,096-row cases, since 1.0.9's serial prompt pass is `pass_rows` = 4,096). This section is the receipt against 1.0.9: before = `c36b536d`, after = `76fe1de4`.

**Who and when:** a reviewer agent, 13 Oct 2026, 16:48-18:05 UTC (18:48-20:05 CEST), on gb10-a. Every GPU job ran under `~/tf-pocket-cuda/GPU.lock`, one at a time: the kernel test and the dev CLI 16:56-17:10 UTC (`tools/chain109a.sh`), then the servers and live learning 17:10-18:02 UTC (`tools/chain109b.sh`). Another reviewer agent job (`review/tools/chainS.sh`, a 1.0.8-based build) had released the lock at 16:52 UTC; nothing else took it, and every script checked for other engine, test and nsys processes before each run. The builds and host tests ran on the CPU before the GPU jobs; the 3-SM fatbins build (16:59:10-16:59:56 UTC, nice 19, `raw/fatbins-3sm.log`) overlapped rep 1's four E runs. E's same-rep ratios in rep 1 (1.171 drafts off, 1.090 on) are inside the other reps' (1.168-1.173 and 1.089-1.103), so I kept them. Grades as in sections 1-3. Raw logs and scripts: `~/tmp/lw-logic/perf/cuda-raw/r109/` (`raw/` from the box, `local/` from the Mac, `tools/`); on gb10-a under `~/tf-pocket-cuda/r109/`.

**In short (MEASURED):** every check passed. On 1.0.9 the pocket costs what it cost on 1.0.8 (+1.59 ms a token undrafted with 64 facts, +1.58 ms empty, dev CLI). With the fix it costs nothing measurable undrafted: taught and empty decode at the untaught speed, 1.17x in the dev CLI and 1.16x on the server, and a 4,096-token prompt is 9-10% faster on them. The tokens are 1.0.9's everywhere checked: 120 of 120 server requests per folder, every dev CLI run, the kernel test's 196 cases, and every live-learning checkpoint.

### 4.1 Builds and conditions

- **Trees (MEASURED):** `git archive` of each commit, hashed file by file on the Mac, against the trees extracted on gb10-a: `c36b536d` = `r109/before`, 1,744 of 1,744 files; `76fe1de4` = `r109/after`, 1,745 of 1,745 files; same names and sha256 (`local/tree-verify.txt`).
- **Builds (MEASURED, `raw/build109.out`):** `tools/build109.sh`, the way of `review/tools/build.sh`: `zig build install` then `zig build native`, `-Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 -Dversion=1.0.9 -j4`, nice 19, Zig 0.17.0 (`build.zig.zon` asks for at least 0.17.0, `.zig-version` says 0.17.0), nvcc 13.0.88, 16:50-16:55 UTC, all rc 0. sha256 (`raw/bin-*.sha256`):
  - before: `tensorfold` `2566d2cb…5b77`, `tensorfold-native` `c00c78fa…348a`, `tf-cuda-test` `bf981cc9…0ca8`;
  - after: `tensorfold` `bf71ff30…01c2`, `tensorfold-native` `c8197b09…8058`, `tf-cuda-test` `756d85c2…60ed`.
  - Of the 34 fatbins in each build cache, 33 are byte-identical and only `nemotron_ops.fatbin` differs (`raw/fatbin-*.sha256`), as on 1.0.8.
- **Machine:** gb10-a as in section 2: NVIDIA GB10 (sm_121), Ubuntu 24.04.5 LTS, kernel 7.0.0-1019-nvidia, driver 580.178.04, CUDA 13.0 (`raw/machine.txt`). Clients: Python 3.12.14.
- **Clients:** `tools/bench_openai.py`, `bench_concurrent.py` and `bench_evidence.py` are the same in v1.0.8 and v1.0.9 (READ: `git diff 0efbd447 c36b536d -- tools/` touches only `tools/zig/gen_glm_kernels.py`), so they gained no option for 1.0.9's resumed prompts or concurrent rounds. The servers ran them from the `c36b536d` tree. There is no `tools/prefill_cold.py` in 1.0.9 either (READ, `git ls-tree`).
- **Folders:** A, T and E of section 2, served in place, read-only. `config.json` and every `*.safetensors` hashed before and after the dev CLI series (19 files, unchanged) and before and after every server (unchanged, 6 of 6).

### 4.2 Host checks and the kernel test

- **`zig build test`** (MEASURED, `raw/test-before.log`, `raw/test-after.log`): before and after both 46 of 46 steps, 700 of 705 tests passed, 5 skipped, 0 failed. 1.0.8 had 44 steps and 616 of 621 (section 3.7); the counts grew with 1.0.9, and they are the same before and after the PR.
- **`zig fmt --check`** with Zig 0.17.0 (MEASURED, `raw/fmt.log`): the seven changed `.zig` files and `zig build.zig` on after, and `zig build.zig` on before: clean.
- **`tools/zig/lean_check.py`** on the two clean `git archive` trees (MEASURED, `local/lean-*.txt`): 538 problems before and 538 after, the same list (1,304 and 1,305 files). The one listed file the PR touches, `zig/tests/cuda/main.zig`, has its line-1 problem on 1.0.9 already. 1.0.8 had 539.
- **Spills** (MEASURED, `raw/resusage-pocket.txt` from `cuobjdump -res-usage` of a `zig build fatbins -Dsm=86,120,121` of after, `raw/fatbins-3sm.log` rc 0): all five pocket kernels (`tf_pocket_rows`, `tf_pocket_in`, `tf_pocket_in_rows`, `tf_pocket_out`, `tf_pocket_out_rows`) use STACK 0 and LOCAL 0 on sm_86, sm_120 and sm_121, so nothing spills; registers 38 to 128, the same as section 3.7.
- **`tf-cuda-test pocket 30`** on after (MEASURED, `raw/pocket-after.log`, 16:56 UTC):
  - `PASS pocket: 196 cases byte-equal to tf_pocket_rows (rows 1 to 4096, 0 to 64 neurons in use, fp32 and bf16 out)`. 194 on 1.0.8; the two new cases are a whole 4,096-row serial pass with 44 and 64 neurons in use.
  - Controls: neurons summed in reverse change 17,633 and 324,472 values, inputs rotated by one 4,028 and 82,080. These differ from 1.0.8's 18,391 / 323,207 and 4,140 / 84,382 because the new cases draw from the same seeded generator first, so the controls get other random inputs (INFERRED from `pocket_tests.zig`, READ).
  - `PASS pocket -0.0: tf_pocket_rows turned 568064 exact -0.0 to +0.0 and kept the rest; a skipped launch and a launch reading a count of 0 keep them all`, as on 1.0.8.
  - Timings, one run, L2 flushed, 44 neurons unless marked, `tf_pocket_rows` against `tf_pocket_in` + `tf_pocket_out`, µs: 1 row 93.9 / 18.2 (64 neurons 93.9 / 22.2), 4 rows 91.9 / 22.2, 16 rows 93.9 / 26.4, 36 rows 98.0 / 35.6, 128 rows 108.3 / 67.3, 512 rows 249.6 / 179.0, 2,048 rows 1,238.8 / 483.0 (64 neurons 1,265.3 / 532.2). Close to the 1.0.8 PR table's four runs on `adec42df` (1 row 94.0-96.0 / 18.1-20.3, 2,048 rows 1,200-1,279 / 487-491); one run here, so no range.
  - On before: `FAIL pocket: UnknownCommand` (`raw/pocket-before.log`), as on 1.0.8: the test cannot run before the change.

### 4.3 Dev CLI: same tokens and speed

`tools/sanity109.sh c1 4 45` (the way of `review/tools/sanity.sh`): A, T, E x before and after x drafts off and on, `tensorfold run FOLDER --tokens <p1, 36 ids> --max-tokens 256 [--no-drafts]`, 4 reps, folders rotated each rep, build order flipped each rep (before first in reps 1 and 3), 45 s idle before each rep; then the 4,096-token prompt `--tokens-file p4096.ids --max-tokens 2 --no-drafts`, 4 reps alike. 16:57-17:10 UTC, GPU 48-57 C after each run. The binaries' sha256 were checked against `raw/bin-*.sha256` before the first run. Folders hashed before and after: unchanged, 19 files. Summary `raw/summ_cli109.txt` (`tools/summ_cli109.py`), runs in `raw/chain109a.out`, reports in `raw/sanity-c1/`. Nemotron 3.5 Lightning 30B-A3B MLX 4-bit, gb10-a (GB10), MEASURED:

| Check | A | T | E |
|---|---|---|---|
| 256 greedy tokens: one hash before and after, drafts off (8 runs) | `2e5c4aac8a41` | `716feda01d52` | `2e5c4aac8a41` |
| the same, drafts on (8 runs) | `2e5c4aac8a41` | `716feda01d52` | `2e5c4aac8a41` |
| 4,096-token prompt: the first token (8 runs) | `95bce61a71a7` | `95bce61a71a7` | `95bce61a71a7` |

The hashes are 1.0.8's too (section 3.2). T differs from A, E equals A, so the runs see the pocket.

| Drafts off, medians (range) | A before | A after | T before | T after | E before | E after |
|---|---|---|---|---|---|---|
| ms a token | 9.328 (9.099-9.382) | 9.287 (9.182-9.348) | 10.880 (10.854-10.935) | **9.283** (9.258-9.315) | 10.867 (10.838-10.913) | **9.285** (9.241-9.325) |
| tok/s | 107.21 | 107.69 | 91.92 | **107.72** | 92.02 | **107.71** |

- After over before, same rep: A 0.999 (0.991-1.015), T **1.174** (1.166-1.177), E **1.171** (1.168-1.173).
- Over A, same rep: before T 0.855 (0.838-0.864), E 0.855 (0.840-0.865), that is +1.588 and +1.576 ms a token; after T **0.998** (0.992-1.007), E **1.001** (0.992-1.002), +0.014 and -0.008 ms.
- 1.0.9's untaught decode is about 5% faster than 1.0.8's was in section 3.3 (107.2 against 102.3 tok/s, MEASURED in each section; different sessions), and the pocket's cost before the fix is the same in ms (T +1.59 and E +1.58 against +1.56 and +1.58 in section 3.3). So the fix is worth a little more on 1.0.9 as a ratio (1.17x against 1.16x) (INFERRED, arithmetic).

| Drafts on, medians | A before | A after | T before | T after | E before | E after |
|---|---|---|---|---|---|---|
| tok/s | 160.05 | 160.75 | 135.34 | 146.13 | 146.55 | 160.68 |
| rounds / accepted drafts | 100 / 155 | 100 / 155 | 107 / 148 | 108 / 147 | 100 / 155 | 100 / 155 |

- After over before, same rep: A 1.005 (0.998-1.010), T **1.077** (1.073-1.081), E **1.094** (1.089-1.103). After, E/A 0.998 (0.982-1.012); before 0.917.
- T drafts its own text in a different number of rounds, so T/A drafted (0.845 before, 0.906 after) is not like for like, as on 1.0.8 (0.909 after there).

| 4,096-token prompt (one 4,096-row pass in 1.0.9) | Before, s | After, s | After/before, same rep | Prompt tok/s | Over A, before / after |
|---|---|---|---|---|---|
| A | 0.516 (0.507-0.521) | 0.513 (0.506-0.528) | 0.997 (0.987-1.017) | 7,938 to 7,986 | |
| T | 0.577 (0.574-0.588) | 0.527 (0.522-0.534) | **0.907** (0.903-0.925) | 7,098 to 7,779 | 1.126 / **1.020** |
| E | 0.576 (0.574-0.580) | 0.517 (0.504-0.533) | **0.897** (0.878-0.920) | 7,107 to 7,922 | 1.116 / **1.002** |

- 1.0.9 runs this prompt as one 4,096-row pass where 1.0.8 ran two 2,048-row chunks (READ, CHANGELOG 1.0.9 and `cuda_state.zig`), so each pocket launch has 4,096 rows; the new kernel test cases cover that size. With 64 facts the prompt still costs 2.0% over untaught (1.005-1.052), down from 12.6%; the empty pocket's cost is gone. Prompt processing is no slower than 1.0.9 on any folder.
- The 36-token prompt (one pass), both modes pooled, 8 runs: A 51.3 to 51.2 ms, T 52.9 to 51.1 ms, E 53.0 to 50.8 ms (medians). A's after range reaches 56.4 ms in one run (rep 3, drafts off); its median moves 0.1 ms.

### 4.4 Server, the maintainer's tools

`tools/servers109.sh s1 120 "bT aT aA bA bE aE"` with `tools/srv109.py` (`tools/v/srv2.py` with the 1.0.9 binaries, clients and output folders): one series of six servers, one at a time on port 18190, each folder's two builds next to each other in mirrored order (T before first, A after first, E before first), 120 s idle before each server, a fresh empty HOME for each, `TENSORFOLD_NO_LIVE=1 tensorfold-native serve FOLDER --host 127.0.0.1 --port 18190 --no-update-check`, no `--slide`. 17:12-17:43 UTC, GPU 46-50 C at each start. Each server ran, in order: `bench_openai.py` defaults; `bench_openai.py --tokens 256 --reps 3 --temperatures 0`; our `decode_bench.py` with `"draft": false` (8 prompts, 2 warm-up and 16 measured requests of 256 greedy tokens, thinking off); `bench_concurrent.py --alone --serial` defaults; `bench_concurrent.py --tokens 256 --reps 3 --temperatures 0 --levels 1,4 --alone --serial`. No fact questions this time. Summary `raw/summ_servers109.txt` (`tools/summ_servers109.py`), per-server files in `raw/s1/`, the series log `raw/servers-s1.out`. Nemotron 3.5 Lightning 30B-A3B MLX 4-bit, gb10-a (GB10), MEASURED unless marked.

- **Integrity:** every server started and became ready in 9.2 s, every client exited 0, every server stopped with exit 0, and each folder's `*.safetensors` and `config.json` hashes, file list, sizes and times were the same before and after its server (6 of 6), and the same across its two servers. Every server ran the binary with the recorded sha256. `/v1/slide/graph` gave T's 440 neurons in use, as in section 2.5.

| Check | A | T | E |
|---|---|---|---|
| Same `token_sha` before and after, every request, both servers of the folder | 120 of 120 | 120 of 120 | 120 of 120 |
| `bench_concurrent`: concurrent reply equals its solo run, per server | 78 of 78 | 78 of 78 | 78 of 78 |
| `bench_concurrent --serial`: solo run equals `"draft": false`, per server | 12 of 12 | 12 of 12 | 12 of 12 |
| Failed, incomplete or unmeasured requests | 0 | 0 | 0 |

- `--serial` per server is 10 checks from the defaults and 2 from the CONTRIBUTING.md form, which the tool reports once per level (its cells add to 14), as in section 3.2.
- **Negative control:** in each build T's `token_sha` differs from A's on 98 of 120 requests, the same split as on 1.0.8 (3 of 20 `bench_openai` defaults, 6 of 6 of its 256-token form, 14 of 16 `"draft": false`, 45 of 48 `bench_concurrent` defaults, 30 of 30 of its CONTRIBUTING.md form). E equals A on 120 of 120.
- **Resumed against fresh:** on all 6 servers the first chat request (prompt 35 tokens, nothing cached) and the next one (the same prompt, 29 of 35 tokens from the prompt cache) gave the same `token_sha`, `657018d897ad`, as on 1.0.8 (server logs). On each server 87 of the 88 chat requests resumed from the cache; the 70 raw completions (15-token prompts) never did (cached 0), before and after alike. 1.0.9's "resumable short prompts" are the 27B and Flash Next on a Mac (READ, CHANGELOG), so this is expected.
- **The clients:** the same files as in 1.0.8, so no new option for 1.0.9's resumed prompts or concurrent rounds; `bench_concurrent.py`'s 4-stream cells carry a `steady_tps`, as on 1.0.8 (MEASURED, both runs' JSON).

**Undrafted** (`"draft": false`, decode tok/s, median of 16 requests, one server per build; every request ended on length after 256 tokens, and the server's own median rate is within 0.5% of the stream's):

| Folder | Before | After | After/before | Over A, before / after |
|---|---|---|---|---|
| A | 106.09 | 105.67 | 0.996 | |
| T | 91.46 | **106.39** | **1.163** | 0.862 / **1.007** |
| E | 90.93 | **105.77** | **1.163** | 0.857 / **1.001** |

**Drafted, `bench_openai.py`** (decode tok/s, the median of the reps on the one server of each build):

| Cell | A before | A after | after/before | T before | T after | after/before | E before | E after | after/before | E/A before / after |
|---|---|---|---|---|---|---|---|---|---|---|
| code, 64, temperature 1 | 165.9 | 166.1 | 1.001 | 153.1 | 167.5 | **1.094** | 155.7 | 169.9 | **1.091** | 0.938 / 1.023 |
| chat, 64, temperature 1 | 145.0 | 141.5 | 0.976 | 133.8 | 145.2 | **1.085** | 128.3 | 142.5 | **1.111** | 0.885 / 1.007 |
| code, 64, greedy | 146.3 | 149.5 | 1.022 | 145.8 | 150.3 | 1.031 | 139.9 | 144.2 | 1.031 | 0.956 / 0.965 |
| chat, 64, greedy | 166.7 | 166.1 | 0.996 | 164.9 | 169.9 | 1.030 | 164.1 | 165.5 | 1.009 | 0.984 / 0.996 |
| code, 256, greedy | 212.4 | 214.4 | 1.009 | 196.7 | 207.1 | **1.053** | 200.9 | 212.8 | **1.059** | 0.946 / 0.993 |
| chat, 256, greedy | 155.9 | 155.1 | 0.995 | 155.2 | 167.1 | **1.077** | 145.6 | 155.6 | **1.069** | 0.934 / 1.003 |

- E writes A's tokens in every cell, so E/A is the pocket's drafted cost on identical work: 0.885 to 0.984 before, 0.965 to 1.023 after.
- The one after cell under 0.99, E's greedy 64-token code: the six servers drafted those identical 64 tokens in 27 to 30 rounds, A after in 29 and E after in 30 (`raw/code-greedy-rounds.txt`, from the server logs), each from its own timed costs. So that cell's E/A holds a different draft plan, not only the pocket (INFERRED). This is one server a build, where 1.0.8's tables were the mean of two.
- T writes A's text in the two greedy 64-token cells: there T/A is 0.997 and 0.989 before and 1.006 and 1.022 after. Elsewhere T's reply is its own text.
- **`bench_concurrent.py` speeds:** the 1-stream cells move like the single-request ones (CONTRIBUTING.md form, greedy: A 0.991 and 1.009, T 1.075 and 1.055, E 1.068 and 1.058 after/before). The 4-stream greedy cells jump between two levels from rep to rep on every folder and build: about 347-385 or 420-452 tok/s for chat (one rep at 409), about 359-402 or 477-530 for code (`raw/conc4-greedy-reps.txt`). So their after/before reads 0.73 to 1.40, untaught included (0.727 and 0.765 on A, which has no pocket). They are not used for any speed claim here. The 4-stream sampled cells (one mode each): A 0.998 and 1.011, T 1.039 and 1.041, E 1.037 and 1.011.
- **Draft-window costs each server timed** (decoded from each HOME, the same TFWC2 format as 1.0.8): a 1-row verify window 10.42 ms (T) and 10.39 ms (E) before, 8.98 and 8.88 ms after, against 9.06 and 8.92 ms on A; one draft level 0.741 ms before on T and E, 0.667 and 0.682 after, A 0.680 and 0.678. As on 1.0.8, the planner now prices T and E as A (INFERRED).

### 4.5 Live learning

`tools/live109.sh` with `tools/live109.py` (`live/live3.py`, phase 3, with the 1.0.9 binaries; a difference is now recorded and the run goes on instead of stopping, and events are compared without time keys like the graph). For each build: a fresh `cp -RL` copy of A under `~/tf-pocket-cuda/r109/models/`, checked against the 14 sha256 of the earlier fresh copy (`live/l0-reference.sha256`); `serve --slide --slide-graph` on port 18191; lesson A ("My favourite colour is green."); stop and restart; a quiet probe; lessons B1 ("My favourite food is pizza.") and B2 ("My favourite number is seven."), each followed by a quiet probe. A probe is 4 prompts (food, number, colour, a haiku) in all four modes (thinking off and on, drafted and `"draft": false`), greedy, 64 or 512 tokens, with logprobs (top 5). At each checkpoint: sha256 of every file in the folder, the slide graph without its time keys, the lesson events, and the probe. Then each folder after B2 served again by its own build, probed and compared with its own live B2 probe and with before's. No chat alongside. 17:42-18:02 UTC. Nemotron 3.5 Lightning 30B-A3B MLX 4-bit, gb10-a (GB10), MEASURED (`raw/live.out`, `live/p3-*/`):

| Checkpoint | Files (sha256) | Slide graph | Lesson events | Probe tokens | Probe logprobs |
|---|---|---|---|---|---|
| A: after lesson A, server stopped | equal | equal | equal | | |
| L: restarted, loaded | equal | equal | | 16 of 16 equal | equal |
| B1: after lesson B1, live | equal | equal | equal | 16 of 16 equal | equal |
| B2: after lesson B2, live | equal | equal | equal | 16 of 16 equal | equal |
| B2 folder restarted, against its own live B2 | unchanged by the restart | | | 16 of 16 equal, both builds | equal |
| after's restart against before's live B2 | | | | 16 of 16 equal | equal |

- Every cell compares after with before (1.0.9) at the same checkpoint. `run.json` of after: result ok, no differences recorded. The two folders' final files are identical (14 of 14, `live/final-p3-*.sha256`), and each restart wrote nothing.
- The lessons: A learned (recalled, 5 steps), B1 learned (recalled, 2 steps), B2 learned but not recalled (5 steps), in both builds. Neurons in use per MoE layer: after A, 1 in 10 layers; after B1, 2 in 9 layers and 1 in 2. One layer that had none after the restart got its first neuron in B1 while the server ran, which is the case where this branch captures the graphs again (INFERRED from the counts; as section 3's live runs showed, the replies alone cannot prove a recapture).
- Logprobs are compared where the server returned them: 14 of 16 replies in the loaded probe and 12 of 16 later. The others are thinking-on replies whose answer stayed in the reasoning (empty content, so no content logprobs); their `token_sha` is compared.
- **1.0.9 against 1.0.8 (MEASURED, `raw/live-vs-108.txt`):** 1.0.9's own run (before) wrote the same files, slide graph and events, and gave the same probe tokens and logprobs, as the 1.0.8 run of section 3 at every checkpoint. So 1.0.9 learns these lessons bit for bit as 1.0.8 did.
- **The copies stay in place:** `~/tf-pocket-cuda/r109/models/p3-before` and `~/tf-pocket-cuda/r109/models/p3-after`, 18,548,876,048 bytes each (18 GB), 14 files, the same bytes.
- A, T and E (the read-only folders) hashed after the whole series at 18:03 UTC: the same 19 files as before the dev CLI series (`raw/hash-final.check`).

### 4.6 What differs from the 1.0.8 receipt

- **Counts that moved with 1.0.9, not with the PR (MEASURED):** `zig build test` 46 steps and 700 of 705 (1.0.8: 44 and 616 of 621); `lean_check.py` 538 (539); the kernel test 196 cases (194), with other control counts because the generator feeds the new cases first.
- **Speed baseline:** 1.0.9 decodes untaught about 5% faster in the dev CLI (107.2 tok/s, against 102.3 in section 3.3) and on the server (106.1 against 100.3), and its 4,096-token prompt is one pass (0.516 s against 0.533 s). The pocket's cost before the fix is unchanged in ms. Undrafted, the fix gives 1.17x in the dev CLI (1.0.8: 1.16x) and 1.16x on the server for taught and empty alike (1.0.8: 1.15x and 1.17x) (MEASURED).
- **Drafted, `bench_openai.py`:** taught 1.030 to 1.094 and empty 1.009 to 1.111 after/before, against 1.043 to 1.093 and 1.043 to 1.123 on 1.0.8. E/A after 0.965 to 1.023 (1.0.8: 0.994 to 1.019); the low cell is explained in 4.4. One server a build here against two on 1.0.8, so these cells are noisier.
- **4,096-token prompt with 64 facts:** 2.0% over untaught after, against 2.8% on 1.0.8; before the fix 12.6% (1.0.8: 10.5%).
- **Draft-window costs:** 1.0.9 times them on a different window (READ, CHANGELOG 1.0.9), so its tables are lower than 1.0.8's (1-row window 8.9-9.1 ms untaught against 9.6-9.7 on 1.0.8), and close between the two builds of each version on untaught.
- Of the checks re-run, nothing that was equal on 1.0.8 differs on 1.0.9.

### 4.7 Not run on 1.0.9

- `tools/prefill_cold.py` (check 6): not in 1.0.9 (READ), so no cold prefill at 2k to 8k. The only prompt lengths are 36 and 4,096 tokens.
- A second server series (1.0.8 had two), the 40 fact questions on the server, `tensorfold lanes` (the 428 fact replies and 40 at once), `tensorfold rounds` (the captured serial graph), `tf-cuda-test window-profile`, nsys, and the four broken kernels. Their 1.0.8 results stand as 1.0.8 results.
- Any GPU other than this GB10; sm_86 and sm_120 compiled only. A full pocket.

### 4.8 Files

- On gb10-a, `~/tf-pocket-cuda/r109/`: `before/`, `after/` (trees and builds), `fatbins-3sm-after/`, `raw/` (every log and summary), `live/` (checkpoints, probes, server logs), `homes/` (each server's timed costs), `models/p3-before`, `models/p3-after` (the live copies), `tools/` (the scripts).
- On the Mac, `~/tmp/lw-logic/perf/cuda-raw/r109/`: copies of `raw/`, `live/`, `homes/` and `tools/` (6 MB), and `local/` (the `git archive` hash lists, the tree check and the `lean_check.py` outputs, run on the Mac). Scripts: `build109.sh`, `chain109a.sh`, `sanity109.sh`, `chain109b.sh`, `servers109.sh`, `srv109.py`, `live109.sh`, `live109.py`, and the summaries `summ_cli109.py` and `summ_servers109.py`; `patch_srv.py` and `patch_live.py` made `srv109.py` and `live109.py` from the 1.0.8 scripts.
- The interim status after checks 1-3: `~/tmp/lw-logic/perf/r109-interim.md`. The PR body against 1.0.9: `~/tmp/lw-logic/perf/PR-CUDA-BODY-109.md`.
