# E1: do the taught neurons answer without the notes? (19 October 2026)

**Question.** In our longest promotion run (paper Section 7.8, called `run6` in these files), the server taught 15 facts into the model's neurons, and the notes of 5 of them left the prompt. Do the neurons really carry those facts, or were the answers coming from other notes still in the prompt?

**Setup.** One run on one NVIDIA GB10 machine (called `gb10-a` here), with our TensorFold build `src7` (commit `3bf4e54f`). Two copies of Nemotron 3.5 Lightning were served one after the other: the **taught** copy, with run6's neurons, and an **untaught** copy of the same model as the control. Every question was asked in a fresh chat at temperature 0 with thinking off. The rules for reading the result were written before the first question, in [inputs/decision-rules.md](inputs/decision-rules.md), with one later note on how the notes were looked up ([inputs/addendum-1.md](inputs/addendum-1.md)).

Each question was asked in several **arms**, which differ only in the notes block put in front of it. The two that matter most:

- **A0**: no notes at all. Only the neurons can answer.
- **A1x**: the block the server would build, with the fact's own note and the 5 promoted notes left out.

The other arms (A1y to A6) test why some facts fail; [inputs/decision-rules.md](inputs/decision-rules.md) defines each one.

**What it shows** (all from [tables.md](tables.md)):

| | Taught model | Untaught model |
| --- | ---: | ---: |
| The 5 facts whose notes left the prompt: their 14 check questions, no notes (A0) | 14 of 14 | 0 of 14 |
| All 15 taught facts: facts whose every check question was right, no notes (A0) | 12 of 15 | 0 of 15 |
| The 56 questions of test set L6, no notes (A0) | 22 of 56 | 17 of 56 |
| The 56 questions of test set L6, with the A1x block | 36 of 56 | 28 of 56 |

- The neurons carry the 5 promoted facts. With no notes at all, the taught model answered all 14 of their questions, and the untaught model none.
- Of all 15 taught facts, the neurons alone answer 12.
- The notes still do most of the work on the full question set: 36 right with the block against 22 without it.

**What it does not show.** This is one run on one machine with one seed, and thinking was off. The 56-question counts come from the test suite's judge, read with two versions of it (`judge-main` and `llwz-promo2`), which agree on every count here.

## Files

| Folder or file | What it is |
| --- | --- |
| [tables.md](tables.md) | Every count, per fact and per arm, and the decision rules applied one by one |
| [inputs/](inputs/) | The rules, the question sets (`sets.json`, `pf.json`), the fixed block of arm A3, and the neighbour questions |
| [replies/](replies/) | Every question and answer, per model and per arm (`gate` = the server's own check questions, `pf` = six hand-written rewordings per fact, `trained` = questions the learner trained on, `q56` = test set L6, `nb` = questions about neighbouring facts) |
| [judge/](judge/) | The judge's inputs and verdicts, the 56-question verdicts under both judge versions (`q56/`), and the judge's source code from our build (`judge-src/`) |
| [validation/](validation/) | Checks that the copies were exact and unchanged, how the servers were started, the promotion state copied from run6, and a first attempt that was set aside (`attempt1/`, see the addendum) |
| [scripts/](scripts/) | The scripts that built the sets, asked the questions, judged them and wrote `tables.md` |
| [SHA256SUMS](SHA256SUMS) | SHA-256 of every file in this folder as published |

## What was changed for publication

Home-directory paths are shown as `~`, and the machine's name is replaced by `gb10-a`. A writer's handle was removed from `inputs/decision-rules.md` and `inputs/pf.json`, and the test suite's former name was replaced in prose (the judge version label `llwz-promo2` is an identifier and stays). The two hash files that froze `decision-rules.md` and `pf.json` before the run are not published, because they hash the files before these changes. `SHA256SUMS` was computed fresh over the files as published. Left out: the model copies, the run logs, the raw notes blocks and the note stores, and compiled Python files.
