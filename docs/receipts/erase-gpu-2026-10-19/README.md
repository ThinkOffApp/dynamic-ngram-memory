# Erasing a promoted fact on the GPU (19 October 2026)

**Question.** A fact can move from the notes into the model's neurons (paper Section 5). If the user then erases that fact's note, do the neurons stop answering it, do the other promoted facts keep answering, and does this survive a server restart?

**Setup.** One run on one NVIDIA GB10 machine (called `gb10-a` here), with our TensorFold build `src7` (commit `3bf4e54f`). The model is the taught copy from the E1 receipt next to this folder ([../e1-no-notes-control/](../e1-no-notes-control/)): Nemotron 3.5 Lightning with 15 facts in its neurons, 5 of them answered with no note in the prompt. Every question was asked with temperature 0, thinking off, and no notes in the prompt, so only the neurons can answer. The erased fact is "My dog is called Bruno now." (fact `w3`, note 13). The rules for reading the result were written before the run, in [decision-rules.md](decision-rules.md).

The run has three phases: **A** before the erase, **B** right after it, and **C** after stopping and restarting the server on the erased copy.

| | A, before | B, after erase | C, after restart |
| --- | ---: | ---: | ---: |
| The 13 questions about the erased fact that name "Bruno" | 13 | 0 | 0 |
| The 11 questions about the four other facts the neurons answer, right | 11 | 10 | 10 |

**What it shows.**

- The erase reached the neurons. The server zeroed the fact's 10 neurons and saved the model in 4.25 s ([erase-reply.json](erase-reply.json)). Afterwards, no answer named Bruno, and the promotion state lists the fact as freed ([promotion-B.json](promotion-B.json), [promotion-C.json](promotion-C.json)).
- It stayed erased after a restart. Every answer in C equals the answer in B, and no file changed between B and C ([hashes-after-erase.json](hashes-after-erase.json), [hashes-C.json](hashes-C.json)).
- The control holds. Every answer in A equals the E1 taught model's answer to the same question, so this copy is the E1 state.
- It cost one other answer. "How are my dog's eyes right now?" went from "Your dog's eyes are brown." to a reply that it has no access to the dog's status. Four more answers about the dog changed wording and stayed right. The answers about the brother and the sister did not change at all.

**What it does not show.** This is one run, one fact and one machine. It does not tell whether the lost answer comes from zeroing the erased fact's neurons, which fire on "my dog", or from saving the pocket again at 4 bits. It also does not cover the server's prompt cache or saved conversations.

## Files

| File | What it is |
| --- | --- |
| [decision-rules.md](decision-rules.md), [decision-rules.time](decision-rules.time) | The rules, written before the run, and when they were written |
| [run-summary.md](run-summary.md) | The run's own short summary, as written right after it |
| [run.sh](run.sh), [eerase.py](eerase.py) | The driver script and the program it runs |
| [server-cmd.txt](server-cmd.txt), [server.log](server.log), [eerase.log](eerase.log) | How the server was started, its log, and the run's log |
| [replies-A.jsonl](replies-A.jsonl), [replies-B.jsonl](replies-B.jsonl), [replies-C.jsonl](replies-C.jsonl) | Every question and answer in each phase |
| [promotion-A.json](promotion-A.json), [promotion-B.json](promotion-B.json), [promotion-C.json](promotion-C.json) | The server's promotion state in each phase |
| [erase-reply.json](erase-reply.json) | The server's reply to the erase request |
| [hashes-prep.json](hashes-prep.json), [hashes-after-erase.json](hashes-after-erase.json), [hashes-C.json](hashes-C.json) | SHA-256 of the model and note-store files at each step |
| [SHA256SUMS](SHA256SUMS) | SHA-256 of every file in this folder as published |

## What was changed for publication

Home-directory paths are shown as `~`, and the machine's name is replaced by `gb10-a`. A writer's handle was removed from the first line of `decision-rules.md`, and the folder's README was renamed to `run-summary.md`. The original hash files are not published, because they hash the files before these changes. `SHA256SUMS` was computed fresh over the files as published. The model, the note store and the source build are not included.
