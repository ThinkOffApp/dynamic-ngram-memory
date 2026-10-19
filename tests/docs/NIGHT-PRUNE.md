# Freeing neurons at night (prototype)

A first try at the night clean-up: find taught facts the model does not need, and switch their neurons off for good.

**Status: a prototype.** It has run against a stand-in server in its own self-test, and a dry run has planned one real taught folder. It has never run against a TensorFold server.

## In short

Each fact TensorFold learns gets its own small set of neurons in the model's "pocket". In our runs every fact that listed neurons used 10 of them, and the pocket has 64 neurons in each of 23 layers. (MEASURED, from the slide graphs of our 1.0.8 runs)

Some facts may not be needed. A fact can be said twice in different words. A newer fact can make an older one useless. Their neurons could be freed for new facts.

`zengine/prune.py` tests that, one fact at a time:

1. Make a copy of the taught model. The original is only read.
2. Ask every stored question and keep the answers.
3. Take one fact. Set its neurons to zero in the copy.
4. Ask every stored question again.
5. If every answer is the same as before, the fact stays off. It is **FREED**.
6. If any answer changed, the fact's neurons are written back. It is **KEPT**.
7. Go on with the next fact, keeping the freed ones off.

At the end, the copy is the taught model with the freed facts switched off.

## What passing shows, and what it does not

**FREED shows that the stored questions got the same answers with that fact's neurons off.** That is all.

It does not show that the model is unchanged. A question nobody stored may now get another answer. The self-test shows this on purpose: when only TensorFold's own questions per fact are asked, a fact that changes only how general-knowledge answers are worded is freed, because no stored question shows its effect.

So the result is only as good as the stored questions. More questions, and more kinds of questions, make it safer.

## Why one at a time, and in order

Facts are tried **cumulatively**: each one is tested with the facts freed before it already off.

This matters when two facts say the same thing. Say fact 1 is "Of all colours, I like blue the most." and fact 4 is "My favourite colour is blue.".

- Fact 1 is tried first. Fact 4 still answers "blue", so every answer stays the same. Fact 1 is freed.
- Fact 4 is tried next, with fact 1 already off. Now nothing says "blue", the answers change, and fact 4 is kept.

If each fact were tested alone against the full model, both would look unneeded, and freeing both would lose the favourite colour. A teammate raised this case in review. The self-test has a broken version that tests facts alone, and it frees both; the check catches it.

Because of this, **the result depends on the order**. It is not the smallest set of facts the model could keep. The default order is oldest first, so of two facts that say the same thing, the newer one is kept. `--order newest`, `--order norm` (below) and `--facts 4,1,3` change the order, and the report records it.

## Which facts to try first (level 8)

**By weight norm.** `--order norm` reads each fact's pocket weight norm from the taught folder, read only (`lwrun.pocket_norms`; the layout is in section 15 of `docs/TEST-LOGIC.md`). It tries the large outliers first (a modified z score above 3.5, or more than 10 times the median), then the others from the largest norm down, then facts already switched off (their input rows zero: replaced or zeroed), then facts with no neurons. Gupta et al. (2024) report single edits whose weight change is far larger than the rest, and that such edits can disable a model, so they are the first worth a look. The ranking is in every report (`prune.json`, `ranking`) and in the dry run, whatever the order.

**By usage: a request to TensorFold.** The plan also asks to rank facts by how strongly their neurons fire on the stored questions, compared with ordinary background text (as Sparse Memory Finetuning ranks memory slots). That needs, for each request, a record of which facts' pocket neurons fired and how strongly. TensorFold 1.0.8 has no endpoint or log line for it (READ, tag v1.0.8: its slide routes are the `/slide` viewer, the graph, to read or clear, and learn; the viewer colours a neuron by the fact that owns it, not by what fired). So `--order usage` is refused, before anything runs, with that reason. Nothing stands in for it.

What we would ask for: a debug option on `POST /v1/chat/completions` (or a separate endpoint) that returns, per request, every (fact, layer, neuron) whose activation went above a threshold, with the largest activation and the token position. With that, a night run could ask the stored questions and some ordinary text, and rank facts by how much they are used. The same record would answer plan item 22: whether a failed answer is a fact that did not fire or one that fired with the wrong value.

## After freeing: does a related fact bring it back? (level 8)

Hu et al. report "unlearned" knowledge coming back after a little related fine-tuning. `--relearn` checks that for the freed facts, after the run:

1. For up to `--relearn-max` (3) freed facts that a lesson of the set says, a **related fact**: the same attribute and value, said of the entity the rule book calls similar ("My dog's name is Luna." after freeing "My cat is called Luna."; "My sister's favourite colour is blue." for me).
2. A copy of the freed copy is served with `--slide` (with a copy of `--slide-graph`, by default `--graph`), and the related fact is taught.
3. The freed fact is asked in every wording: each CURRENT wording, the yes/no about its value in each wording, an either-or with another value; and the related fact's own question, to see it was learned.
4. The answers are judged against the rule book: the phase's lessons without the freed ones, plus the related fact. **Where it says "don't know", the answers must say "don't know".** The report also counts leaks: answers that give the freed value where the rule book does not.
5. With `--reference-model` (and `--reference-graph`), the same is done on a copy of a folder that never learned the freed lessons, for example `lwrun.py`'s reteach copy of a phase that deletes them. Then the reference is the yardstick: **in each thinking mode, the freed copy may miss "don't know" no more often than the reference, and have no more wrong answers.** An untaught model guesses too (our 1.0.8 runs had wrong yeses about facts nobody taught), so without a reference a single guess fails the probe.

A failed probe ends the run as `relearn-failed`, exit code 1: the copy is freed as reported, but freeing it was not safe.

**Where the related fact lands on 1.0.8.** TensorFold picks the places itself; this tool cannot. While a server runs, a zeroed fact's places stay taken (READ, `pocket.zig` `retire`). But the relearn copy is loaded fresh, and on load TensorFold counts a layer's neurons in use up to the last one whose input row is not zero (READ, `weights.zig` `pocketOf`). So a freed neuron above every live one in its layer is free again, and the related fact may be taught into it; a freed neuron below a live one stays taken. The report lists, for each probe, the related fact's neurons that a freed fact held (`on_freed`). The graph still lists the freed facts as learned, but 1.0.8 keeps its replay of earlier answers in the server's memory, so a fresh server replays none of them into the new fact (READ, `learner.zig`).

Not built: comparing the lesson sentence's loss in the zero and reteach copies. 1.0.8's API guide says its native CUDA engine returns prompt log-probabilities on `/v1/completions` (`prompt_logprobs`), and other engines refuse them (READ), so this could be built for the GB10. It is left for a later change.

## The rules for each fact

| Case | Result |
| --- | --- |
| Every answer the same after zeroing | FREED: the neurons stay off |
| Any answer changed | KEPT: the neurons are written back and checked against the original, byte for byte |
| Every neuron already zero, and the fact was replaced or missed | FREED without asking: no byte changed |
| Every neuron already zero, but the fact is marked learned | SKIPPED: the index may be wrong, or the graph is stale |
| No neurons listed in the graph | SKIPPED |
| Its neurons are also used by another fact | SKIPPED |

"The same" means word for word by default (`--same text`). With `--same verdict`, an answer counts as the same when the judge gives it the same verdict and reads the same value from it, even if the wording moved. TensorFold's own questions have no right answers to judge, so they are always compared word for word.

## Safety

- **The original folder is only read.** The copy is made the same way the runner makes its copies (an APFS clone on macOS). Before anything is written, the copy is checked to share no file with the original: no links, and another file on disk for every file. The original's file sizes and times are checked again before every zeroing and at the end.
- **No write into a shared file.** Zeroing and write-backs go through `open_for_inplace_write` (`zengine/lwrun.py`). A shard that is a symlink or has more than one hard link (`st_nlink` above 1) is refused, with its path and link count, before any byte of that step is written, and the run stops. The check above catches such a copy at the start; this one also catches a link made during the run, such as a deduplicating tool turning copies into hard links to their source, as one did on gb10-a.
- **Zeroing is the runner's own.** It uses `zero_facts` from `zengine/lwrun.py`: philip-pentatonic's removal method, as in the runner's "zero" delete mode. The prototype also works out the same rows itself, read-only, for the dry run. Every real step checks that both agree.
- **Byte audit.** After each zeroing, the copy is compared with the original. Every byte that differs must lie inside the rows of a fact that is off. By default this reads every shard that holds such rows, and every small file, byte for byte; the other shards are compared by size (`--zero-audit full` reads them all).
- **Noise check.** Before anything is zeroed, the unchanged copy is asked every question twice, with a fresh server each time. If the same weights give different answers, a change after zeroing could not be told from noise, so the run stops.
- **Final check.** At the end, every freed row must be zero, every other fact's rows must match the original, and one more round of questions must give the first answers again.
- **Signals.** On Ctrl-C or a stop signal while a fact is being tested, its rows are written back before the run ends. A signal that lands while rows are being written back, such as a second Ctrl-C, waits until they are back. A hard kill (SIGKILL) during a step leaves that fact switched off in the copy. The report is saved as each step starts, so it shows that step as `running`.
- It never serves with `--slide`, never writes the slide graph, and contains no TensorFold code.

## Commands

Plan a run first. The dry run checks everything, reads the fact's rows in the taught folder and prints every step. It copies, serves and writes nothing:

```sh
python3 zengine/prune.py run --model PATH/TO/TAUGHT --graph PATH/TO/graph-final.json --set set.json --bin PATH/TO/tensorfold-native --work runs/prune --port 18190 --dry-run
```

Then the same command without `--dry-run`.

- `--model` is a taught folder, for example the runner's `models/SET.teach` copy.
- `--graph` is its slide graph (`GET /v1/slide/graph`, or the runner's `graph-final.json`), or the runner's teach `state.json`. The state also remembers the neurons of facts that were replaced later; the final graph no longer lists them.
- `--set` gives the questions and the right answers for the judge. `--phase` names the phase the taught folder holds (default `teach`).
- `--questions graph` asks TensorFold's own questions per fact instead; `both` asks both.

The self-test needs no model and no GPU. It writes a tiny fake model, starts a stand-in server that answers from which facts' neurons are switched on, and runs the prototype against it:

```sh
python3 zengine/prune.py selftest
```

The report is `WORK/prune.json`. It lists each step in order: the fact, its neurons, the result and why, the rows zeroed, how many answers stayed the same, and the answers that changed. It also lists the freed facts and their neurons. Answers, verdicts and server logs are kept under `WORK/steps/`.

## Status

Grades: **MEASURED** means we ran it. **INFERRED** means our own reading, not checked by a run. **UNCHECKED** means written but never run.

| Part | Status | Grade |
| --- | --- | --- |
| Self-test | 29 of 29 checks pass against the stand-in server, with Python 3.12 and 3.14 (13 Oct 2026), in about 160 seconds. The four new ones are the link guard of Safety: a shard turned into a hard link, or a symlink, to the original's after the start check is refused at the first zeroing, with the original and the copy's other shard unchanged (control: with the guard switched off the zeroing changes the original's shard); and `zero_facts` checks every shard before its first byte (control: with only the check at each open, the first shard is left half zeroed; with that pre-check removed from a copy of `lwrun.py`, this check fails). Before them, 25 of 25 with Python 3.9, 3.13 and 3.14 (13 Oct 2026), in about 150 seconds; before the level 8 checks, 20 of 20 in about 100 seconds, with Python 3.12 and 3.14. The five level 8 ones: `--order norm` tries the planted large-norm fact first and flags only it (control: the same fixture without the plant flags none); `--order usage` is refused and says it needs an activation endpoint; the relearning probe keeps the freed cat's name "don't know" in all 5 wordings, with no more wrong answers than a folder that never learned it; its control, a model whose relearning brings the freed fact back, leaks the name and fails the run; and the verdict rule worked out alone on six cases, with the related fact's neurons on freed places listed (controls: the first rule, every "don't know" kept even with a reference, fails the case where the reference guesses as often; a related fact on a freed neuron is listed). It covers: a fact said twice frees one copy and keeps the other; a needed fact is kept; a replaced fact with zero rows is freed without asking; `--same verdict` against `--same text`; TensorFold's own questions; the runner's teach `state.json` as the graph; `--order newest`; the report naming the step under way; a neuron outside the pocket (the run fails cleanly); Ctrl-C twice during a write-back | MEASURED |
| Negative controls | Each must be caught, and is: testing each fact alone (both copies freed), a comparison that sees no change (the needed fact freed), a write-back that writes nothing, a server that does not repeat itself, a copy that shares files with the original, a missing binary, a file in neither graph format, signals not held during a write-back (the fact is left switched off) | MEASURED |
| Planted bug | Kept facts not written back, put into a scratch copy by hand: the byte audit stops the run and 9 of 20 checks fail | MEASURED |
| Dry run on a real taught folder | Our Mac L1 taught folder (TensorFold 1.0.8): the one fact's 10 neurons give 30 rows in 2 shards, all 30 nonzero. On the runner's zeroed copy of it, all 30 are zero, so the dry run predicts SKIPPED. Nothing was written | MEASURED |
| A real run against TensorFold | Not run yet: the GPUs were busy | UNCHECKED |

## Limits and open questions

- **Cost.** One round of questions per fact that still has nonzero rows, plus up to three more (first answers, the repeat, the final check). A round is every question with thinking off and on. For the level 1 set that is 24 chats per round and at most 4 rounds, as the dry run printed. No real timing yet.
- **Disk reads.** After each zeroing the byte audit reads every touched shard in both folders: about 10 GB for each shard it touches on the Nemotron model, whose shards are about 5 GB. `--zero-audit off` skips it; the freed rows are still checked. (INFERRED)
- **No resume.** A stopped run starts again in a new `--work`.
- **TensorFold does not know.** The freed neurons are zero in the copy, but TensorFold's own bookkeeping still counts them as used. Reusing them for new facts would need support in TensorFold. The report lists them for that. (INFERRED)
- **Facts that share neurons** are skipped, not tested. None of the facts in our 1.0.8 runs shared a neuron. (MEASURED, from the run records)
