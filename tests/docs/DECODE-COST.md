# Why taught facts slow down decoding

A short summary of our review of the decode overhead in TensorFold 1.0.8's Living Weights. Written 12 October 2026.

**Status: mostly analysis.** We read the source and did the arithmetic. The section "First Mac measurements (draft)" adds a first, unfinished set of Mac measurements of the proposed Mac fix: a profile and a same-tokens check. The speed comparison was still being written up and is not here yet.

Grades: **MEASURED** = a number someone measured (we name who). **READ** = we read it in TensorFold's source (tag `v1.0.8`, commit `0efbd447`). **INFERRED** = our own reasoning or arithmetic, not checked by a run.

Model: Nemotron 3.5 Lightning 30B-A3B, MLX 4-bit.

## The problem

Once a model has learned facts, it writes text more slowly.

| Who | Machine | Before | After | Extra time per token | Grade |
| --- | --- | --- | --- | --- | --- |
| Ash Hart | Apple M5 Ultra (Metal), 1.0.8 | 248 tok/s | 213 tok/s with 10 facts | +0.66 ms | MEASURED by Ash Hart; ms by us |
| Ash Hart | Apple M5 Ultra (Metal), 1.0.8 | 248 tok/s | 205 tok/s with 100 facts | +0.85 ms | MEASURED by Ash Hart; ms by us |
| philip-pentatonic | NVIDIA RTX PRO 6000 Blackwell (CUDA), 1.0.7, `--no-drafts` | 290 tok/s | 199 tok/s | +1.58 ms | MEASURED by philip-pentatonic; ms by us |

philip-pentatonic also saw about the same 199 tok/s with an empty pocket as with 2 facts. (MEASURED by him, [TensorFold issue #678](https://github.com/ashhart/TensorFold/issues/678#issuecomment-6125486909))

Ash Hart's numbers are also in TensorFold's 1.0.8 release notes, which report a much smaller cost on a GB10: thirty facts cost 1 to 6%. (MEASURED by TensorFold)

The extra time per token is `1000/after - 1000/before`, in ms. For example `1000/199 - 1000/290` = 5.025 - 3.448 = 1.58 ms. (INFERRED, arithmetic)

## Where the time goes

Learned facts live in a small "pocket" of extra neurons beside the shared expert in some of the model's MoE layers.

### Mac (Metal)

- Each layer that holds learned neurons adds 3 small GPU dispatches per token. Each one waits behind a full barrier. (READ)
- One more kernel loses its overlap with another and moves onto the critical path. (READ)
- Only 16 of the 23 MoE layers can ever hold learned neurons. So the most it can add is 48 dispatches per token. One fact uses about 10 layers, so about 30. (READ; our earlier estimate of 69 was wrong)
- 0.66 ms spread over 48 dispatches is about 14 µs each. (INFERRED, arithmetic on Ash Hart's numbers)
- The pocket's size cannot explain it. 10 facts are about 2.2 MB of weights, against about 1.9 GB read per token. That is about 0.1%. (INFERRED)
- **What the source suggested: waiting at barriers, not reading bytes.** (INFERRED) The first Mac profile only partly confirms this. Most of the cost is one slow kernel on the critical path. See "First Mac measurements (draft)" below.

### CUDA

- The pocket kernel runs in every MoE layer of a model that has a pocket. It always runs over all 64 pocket neurons, even when they are all zero. (READ)
- While writing text, it runs as one block of 256 threads on one of the GPU's many processors, reading 1.38 MB per layer. That is 23 launches per token, plus one more for each draft level of the speculative head. (READ; philip-pentatonic reached the same reading in #678)
- philip-pentatonic measured the kernel itself with nsys: 85.3 µs per launch, 28.8% of all GPU kernel time ([TensorFold issue #678](https://github.com/ashhart/TensorFold/issues/678#issuecomment-6125486909); MEASURED by him). Of that, about 69 µs per launch (1.58 ms over 23 launches) is not hidden by other work. (INFERRED, arithmetic)
- **Most likely cause: too little work in flight per launch.** The cost depends on whether a pocket exists, not on how many facts it holds. That fits "empty pocket about the same as 2 facts". (INFERRED; philip-pentatonic's reading in #678 too)

## Proposed fixes

Both first fixes are designed to give **exactly the same tokens** as today. Any token difference would be a bug. (INFERRED)

**Mac: let the pocket run alongside the rest instead of behind barriers.**
- Fold the activation step into the pocket's input kernel, so there is one kernel fewer.
- Start the two remaining pocket kernels next to work that already runs at that point, with no added barriers.
- Every floating point operation keeps its order, so the tokens should be identical. (INFERRED)
- Expected: about 242 to 248 tok/s with 10 facts (213 today) and about 239 to 245 with 100 facts (205 today). (INFERRED) If 10 facts still decode below about 235 tok/s, the barrier explanation is wrong. (Written before the first profile. That profile, below, still finds 0.169 ms per token after the fix, more than this estimate allows, so treat these rates as superseded until the speed write-up is in.)

**CUDA, fix 1: launch only over the neurons in use, and skip layers with none** (philip-pentatonic's first proposed fix, extended to skip unused neurons too).
- This is what the Mac already does.
- Keep the buffer size at 64 and add a separate "used" count. Changing the size itself would make the learner read a loaded pocket as empty and could overwrite earlier facts. (READ + INFERRED)
- The dropped neurons are all zero, so they only ever add zero. The tokens cannot change. (INFERRED, IEEE 754)
- Expected: about 290 tok/s with an empty pocket, about 280 to 290 with 2 facts, but still about 220 with a full pocket of about 100 facts (on philip-pentatonic's RTX PRO 6000). (INFERRED)

**CUDA, fix 2 (proposed by philip-pentatonic in #678): spread the pocket kernel over many blocks**, in two launches, keeping each sum in the same order. Expected: 287 tok/s or more, probably fully hidden. (INFERRED)

## How to prove a fix

1. **Confirm the cause first.** Profile one taught and one untaught model per machine. On the Mac, the three pocket kernels alone should add up to far less than the 0.66 ms gap. On CUDA, the wait for the side stream should grow by about 23 times the pocket's time, and not move on a model without a pocket (the negative control).
2. **Same tokens, old build against new.** Greedy, thinking off and on, drafting on and off:
   - kernel outputs byte for byte;
   - 8 neutral prompts at 256 tokens, same token hash per prompt;
   - every test wording of every taught fact, same full token ids and same recall;
   - a taught and an untaught request in one batch;
   - drafted equal to undrafted, as today;
   - **a negative control that must fail**: a taught model against the untaught one on the fact questions must give different tokens. If it does not, the check cannot see the pocket;
   - the model files' sha256 unchanged after every server run.
3. **Speed**, with `tools/decode_bench.py`: one server per machine, a fresh copy of the model for each, old build against new. Pass on the Mac: 10 facts within 3% of the untaught model.

## The harness

`tools/decode_bench.py` talks to one running TensorFold server. It sends 8 neutral prompts that touch no taught fact, streams the replies at temperature 0, and records two speed measurements side by side: its own (tokens from the server's usage count, over the stream's time) and the server's own `tokens_per_second`. It also records whether drafting was on and a hash of the tokens. Standard library only.

```sh
python3 tools/decode_bench.py --base http://127.0.0.1:8090/v1 --label untaught --thinking both --out untaught.json
```

```sh
python3 tools/decode_bench.py --base http://127.0.0.1:8090/v1 --label dry --dry-run
```

The second command needs no server. It prints the plan and runs an offline self-test of the stream parser and the statistics.

## First Mac measurements (draft)

Measured by the authors on an Apple M5 Max MacBook Pro on 12 October 2026, with a first build of the Mac fix above, based on 1.0.8. The fix is on our own branch and is not part of any TensorFold release. This write-up was unfinished when this page was made: the speed comparison is missing, and the numbers below may still be revised. All numbers are MEASURED, one run each unless stated. The readings are INFERRED.

Two model copies: the untaught model, and a taught copy with 40 learned neurons in 16 of the 23 MoE layers. The rates come from an undrafted run of TensorFold's command-line tool on an M5 Max, so they are lower than Ash Hart's M5 Ultra numbers in the first table; the per-kernel times are taken with each dispatch alone. Compare them only with each other.

| Build | Model | Decode tok/s | ms per token | Pocket kernels alone, ms per token |
| --- | --- | --- | --- | --- |
| 1.0.8 | untaught | 183.5 | 5.450 | 0 |
| 1.0.8 | taught | 166.1 | 6.019 | 0.526 |
| fix | untaught | 183.6 | 5.446 | 0 |
| fix | taught | 178.1 | 5.615 | 0.492 |

**Is it the barriers? Only partly.**
- On 1.0.8 the taught model is 0.569 ms per token slower. The three pocket kernels on their own take 0.526 ms of that, 92%.
- Most of it (0.439 of 0.526 ms) is one kernel, the pocket's input step: about 27 µs per call. A dispatch that does almost nothing costs 2 to 3 µs here.
- So the cost is mostly that kernel's own run time, sitting on the critical path. The barriers themselves are a small part. (INFERRED)
- The fix still helps, because the slow kernel now runs beside other work. In this profile the gap falls from 0.569 to 0.169 ms per token.

**Same tokens? Yes, on every path tested.**
- Fact questions on the server: 72 of 72 replies identical between 1.0.8 and the fix, on the taught model and on the untaught one. Identical means the same token hash, text, reasoning, token count and finish reason.
- Full token ids (taught model): 60 of 60 identical, 9,440 tokens.
- The 8 neutral prompts of `tools/decode_bench.py`: identical in every case.
- Drafted equal to undrafted, in every check.
- Negative control: the taught model against the untaught one differs in 56 of 72 server replies. So the check can see the pocket.
- Not tested yet: a taught and an untaught request in one batch. The model-file hash check is not written up yet.

**A measuring trap.** This MacBook's GPU slowed by about a third after a few minutes of continuous decoding. It recovered after about 5 minutes idle, and the system reported no thermal warning. So the speed runs rest 4 minutes before each server. Runs made back to back moved by up to 10% between servers of the same condition, which would hide a 5 to 10% effect. Other apps were also busy on the machine during these runs (up to about two CPU cores, and a backup near the end). (MEASURED; the cause is INFERRED)

## Thanks

The measurements in the first table are Ash Hart's (Mac) and philip-pentatonic's (CUDA). philip-pentatonic also profiled the CUDA cost to one kernel and proposed both CUDA fixes in [TensorFold issue #678](https://github.com/ashhart/TensorFold/issues/678#issuecomment-6125486909); our additions there are the source reading and the `used`-count detail of CUDA fix 1. The Mac analysis and the first Mac measurements of the Mac fix are ours.
