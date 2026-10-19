# Re-judged run records (13 October 2026)

> **In this repository.** This page comes from the test suite's working repository (commit `2083ee9`, MIT license: [LICENSES/LLWZ-MIT.txt](../../LICENSES/LLWZ-MIT.txt)). The code it names (`zengine/`, `tools/`, `examples/`, `lean/`, `zig/notes/`, `docs/TEST-LOGIC.md`) is in [tests/](../../tests/) here. For publication, home-directory paths are shown as `~`, a few names were removed, and links were pointed to their new places; nothing else was changed.

Every stored run record of our first TensorFold 1.0.8 runs, judged again with the fixed judge. The model's answers are the same. Only the judge changed. The run records were only read.

Grades, as in [RESULTS.md](../RESULTS.md): **MEASURED** is counted from the run records, **INFERRED** is our reading.

## What was re-judged

| Label | Machine | Set | Answers judged |
| --- | --- | --- | --- |
| mac-L1-seed1 | Mac (M5 Max) | L1-seed1 | 120 |
| gb10-L1-seed1 | GB10 | L1-seed1 | 96 |
| gb10-L1-seed1-teach2 | GB10 | L1-seed1, teach phase again | 48 |
| gb10-L2-seed1 | GB10 | L2-seed1 | 192 |
| gb10-L3-seed1 | GB10 | L3-seed1 | 308 |
| gb10-eyes-678 | GB10 | eyes-678 | 354 |
| gb10-L4-seed1 | GB10 | L4-seed1 | 352 |

That is 1,470 answers. An answer file that the runner also judged as another phase (`answers.as-*.jsonl`, written when lessons were refused) counts once per file. The two GB10 L1 runs are kept twice in our records with identical answers; each was judged once. There were no newer run records. (MEASURED)

Each set file in the records is identical to the one in `examples/`, and its expected answers recompute with the current rule book. (MEASURED)

## What changed

51 verdicts changed. Every one is listed in [CHANGES.md](CHANGES.md), with the answer. (MEASURED)

| Fix | Changed | What happened |
| --- | --- | --- |
| Reverse questions | 32 | All in L4. The model answered "Who has grey eyes?" and "Who likes olives?" with famous people or people in general, in every phase. The old judge called these unclear. Now they name nobody you know: a pass when nothing was taught (4), a fail when the dog was taught (28). |
| Self-replies | 11 | "No, I am a language model developed by researchers from NVIDIA." (10 times) and "No, I am Nemotron, ..." (once) were read as "no". Now they mean "doesn't know": 4 fail to pass where the right answer was "don't know", 7 pass to fail where it was "no". |
| Dates | 8 | Answers with a wrong date now fail: "Until October 11, 2023, your eyes were brown." (3 times), "... brown until 16 October 2026." (once), "blue until 9 May 2024" (3 times, one answer judged as two phases) and a made-up timeline "Until May 14, 2025 ... From May 15, 2025" (once). |
| Guesses | 0 | No stored answer gave the right value as a guess, so this rule changed nothing here. |

Moves: unclear to fail 28, pass to fail 15, unclear to pass 4, fail to pass 4. (MEASURED)

A review of the fixed judge (13 Oct) changed how it reads dates and reverse questions in answers our runs never gave: two readings of a date that opens a later clause, dates about another attribute or entity, colours outside the test's list, and people the test never taught about. Run again with the reviewed judge, every report here and CHANGES.md came out byte for byte the same. (MEASURED)

Three later fixes (13 Oct) read a clause with no subject of its own as going on with the subject before it ("Your dog does not like apples but likes olives." names the dog), subjects joined by "and" as sharing one predicate ("Your dog and your cat do not like olives." names nobody) a value in a list as the verb's ("Your dog likes apples and olives." names the dog), and a later "no longer does" as taking an earlier credit back ("Your dog and your cat like olives, but your dog no longer does." names only the cat). They changed no stored verdict: judged against main's judge at f274a03, 0 of the 1,470 answers changed, and every report here, CHANGES.md and changes.json came out byte for byte the same. None of the 32 stored answers to a reverse question has a clause these rules read differently: no subject is carried, no joined subject or list item is found, and nothing is taken back. (MEASURED)

The verdicts stored in the L1 run folders differ from main's judge in 9 places, before this branch. All 9 are the same either-or answer ("Both are beautiful colours. Purple is often associated with royalty ..."). The judge learned to read it as "doesn't know" after those runs were judged. (MEASURED)

## What this changes in RESULTS.md

- **#678 case, deleting "My eyes are green." by zeroing:** 4 of 5 affected eye answers right with thinking off, not 5 of 5. "Until October 11, 2023, your eyes were brown." is now a wrong date. The reteach stays at 5 of 5. (MEASURED)
- **#678 case, deleting the dated lesson by never teaching it (reteach):** 2 of 5 eye answers right with thinking off, not 3 of 5. "No, I am a language model ..." no longer passes as a "no". (MEASURED)
- **L3, teach phase, thinking off:** past-time questions 3 of 6 (was 4 of 6), yes/no 2 of 6 (was 3 of 6). Both answers are named above. (MEASURED)
- **L4 reverse lookups:** 2 of 2 in the baseline, both modes. In every taught phase 0 of 2, now as fails instead of unclear: the model never named the dog. (MEASURED)

## Four misreadings, judged again (13 Oct, later)

A later branch (`judge-misreads`) fixed four misreadings of the judge. Every stored answer was judged again, with main's judge at 5f444ce as the old judge, and so were three newer GB10 runs. The model's answers are the same; only the judge changed. The run records were only read. The new reports are in [misreads/](misreads/). (MEASURED)

| Fix | What it does |
| --- | --- |
| likely | "It is very unlikely that your favourite colour has always been the same." is a guess that leans no, as "Probably not." is; "It is very likely that it has!" leans yes. Both opened with "It is", which was read as a plain yes |
| truth | on a plain yes/no question, "It is false.", "That is false." and "False." are a no, and "That's true." and "True." a yes |
| entity | an answer that gives the value to another of the set's entities fails: "Your cat's eyes are grey." for "What is my dog's eye colour?" |
| first | a first-person answer in the words of a lesson ("I do not like olives anymore.") is read as about the user, as "You do not like olives anymore." is; the model talking about itself ("Personally, ...", "I love olives! Do you?") is a self-reply, "doesn't know" |

The three newer runs:

| Label | Machine | Set | Answers judged |
| --- | --- | --- | --- |
| gb10-L5-seed1 | GB10 | L5-seed1 | 432 |
| gb10-neg-seed1 | GB10 | neg-seed1 | 96 |
| gb10-neg2-seed1 | GB10 | neg2-seed1 | 192 |

With the 1,470 above that is 2,190 answers. Each of the three set files is identical to the one in `examples/`. (MEASURED)

**15 verdicts changed, all by the first fix.** Every one is listed in [misreads/CHANGES.md](misreads/CHANGES.md), with the answer. (MEASURED)

| Run | Changed | What happened |
| --- | --- | --- |
| gb10-L2-seed1 | 10 | "It is very unlikely that your favourite colour has always been the same." (5 times, the right answer is no): fail to unclear, the right answer said as a guess. "It is very likely that it has!" and "It is very likely that your favourite colour has changed ..." (5 times, the right answer is yes): pass to unclear, for the same reason |
| gb10-L5-seed1 | 5 | "It is very unlikely ..." on "Has my favourite colour always been the same?" (4 times) and "It is very unlikely!" (once), the right answer is no: fail to unclear |

Moves: fail to unclear 10, pass to unclear 5. (MEASURED)

6 more answers keep their verdict, fail, but are now read as a guess that leans no instead of a yes. In all 6 the right answer is "don't know": "Has my favourite colour always been the same?" in the L2 and L5 baselines, "Have my dog's eyes always been the same colour?" in the L5 baseline, and "Have my eyes ever changed colour?" in the L3 baseline and in both copies of L3's first delete. (MEASURED)

The other three fixes change no stored verdict and no reading: no stored answer to a plain yes/no question opens with "true" or "false", none gives a value to another of the set's entities, the one first-person statement of a value ("My eyes are green.", #678, the reteach of "delete fact 5", thinking on) is read as green, as before, and the 3 first-person answers where the model talks about itself ("I don't have the ability to see you ... I'm an AI text model ...") were already read as "doesn't know". The reports differ from main's judge's only in L2, L3 and L5. (MEASURED)

The verdicts the runner stored in the three new run folders differ from main's judge in 17 of 432 (L5), 0 of 96 (neg) and 4 of 192 (neg2) places, before this branch; they were not looked into here. (MEASURED)

A cross-model audit then found a false pass in the whose-value fix: the asked entity named with no value of its own ("Your cat has grey eyes. Your dog is asleep.") passed for the dog. Fixed on the same branch and judged again: against the judge of 35fc940, 0 of the 2,190 verdicts or readings changed, and every file in `misreads/` came out byte for byte the same. No stored answer has such a clause. (MEASURED)

A second audit (of 0c782f4) found that a first-person answer that also says something of the user ("I do not like olives anymore. You still do.") was read as the user's no. Fixed on the branch judge-first-person: the "I" is then the model itself, and what is said of the user decides. Judged again: against the judge of 0c782f4, 0 of the 2,190 verdicts or readings changed, and every file in `misreads/` came out byte for byte the same. No stored answer opens in the first person and also says something of the user. (MEASURED)

[misreads/](misreads/) holds every report with the split lines, CHANGES.md, changes.json, SPLIT.md and split.json, for the 2,190 answers. The reports at the top of this folder and in `split/` carry the verdicts of the judges before this change; for L2 and L3 their verdicts and readings differ from the ones in `misreads/` in the answers above. L5, neg and neg2 are only in `misreads/`.

## kept_stable

A new score. In a delete phase, it takes the questions whose right answer did not change. It compares each answer with the same model's answer in the teach phase: the same verdict, and the very same text. It measures stability, not correctness. A wrong answer that stays wrong the same way is stable. The table per run is in [CHANGES.md](CHANGES.md); the per-phase lines are in each report.

- **Where the zero step changed no bytes, no answer moved.** In the #678 case, the zero copies of "delete fact 3" and "delete fact 4" gave the teach phase's text in 15 of 15 answers, both modes: those facts had been replaced, and their 30 rows were already zero. So did L2's second delete (16 of 16, rows already zero) and L3's second delete (19 of 19, a lesson that was never learned, so no rows). (MEASURED)
- **Reteach keeps verdicts better than words.** For example, L4 with the dog's name never taught: same verdict 19 of 20 and same text 10 of 20 with thinking off; 17 of 20 and 2 of 20 with thinking on. (MEASURED) A freshly taught copy gives other words for mostly the same verdicts, as RESULTS.md finding 5 found for zero against reteach. (INFERRED)
- **Zeroing the dated #678 lesson moved answers it should not have.** "What colour are my eyes?" and "What is my eye colour?" passed in the teach phase and failed after zeroing, in both modes, and so did "Are my eyes brown or green?" with thinking off, although "My eyes are green." still holds. This is RESULTS.md finding 9 seen from the new score. (MEASURED)
- **New: in L4, zeroing the cat's eye colour also lost the cat's name.** Delete phase C removed "My cat's eyes are hazel." and "My favourite colour is no longer blue." The second was never learned, so only the first had neurons to zero: 30 rows (10 neurons). The other learned facts' 70 neurons stayed nonzero. "What is my cat called?" was "Your cat is called Nora." in the teach phase. After zeroing it was "My name is Nemotron, and I'm a language model developed by NVIDIA." (thinking off) and "I don't know your cat's name!" (thinking on). Never teaching those two lessons (reteach) kept "Your cat is called Nora." in both modes. (MEASURED, one run, one question in two modes) Our reading: the name fact leaned on neurons of the other fact about the same cat, as the dated fact did on 1.0.7 in #678. (INFERRED, not checked)

## Files

| File | What it is |
| --- | --- |
| [CHANGES.md](CHANGES.md) | every changed verdict with the fix that changed it, and kept_stable next to delete_kept |
| `changes.json` | the same list, for scripts |
| `LABEL.zero.txt`, `LABEL.reteach.txt` | the new judge's report on the runner's combined judge input of that delete mode |
| `LABEL.reteach.restart.txt` | the same, with the Mac run's teach answers probed after a restart without `--slide` (`answers.restart.jsonl`) |
| `LABEL.teach.txt`, `LABEL.teach.as-*.txt` | answer files outside every combined input: a teach-only run, and teach answers judged as another phase because lessons were refused |
| [misreads/CHANGES.md](misreads/CHANGES.md) | added 13 Oct: the four misreadings fixed later; every changed verdict against main's judge at 5f444ce, for the 1,470 answers above and three newer GB10 runs (2,190 answers). `misreads/` also holds every report with the split lines, `changes.json`, `SPLIT.md` and `split.json` |
| [split/SPLIT.md](split/SPLIT.md) | added 13 Oct: every report split by the right answer (a value, no, "don't know"), the fixed consistency score, paraphrase agreement, the literature's tags, neighbourhood failures, and the two dummy systems on the same sets. `split/` also holds the reports with these lines and `split.json`. No verdict changed: 0 of 1,470 against main's judge (MEASURED) |

## How to run it again

`tools/rejudge.py` judges every answer row twice, with an older judge and with the current one. It names a fix as the cause of a change when switching that one fix off gives back the old verdict. It only reads the run folders.

```sh
git show main:zengine/lwlogic.py > /tmp/lwlogic-main.py
```

```sh
python3 tools/rejudge.py --old /tmp/lwlogic-main.py --out docs/rejudge mac-L1-seed1=PATH/TO/RUN gb10-L4-seed1=PATH/TO/RUN
```

The split scores were written with `--split-only`, which leaves `CHANGES.md` alone:

```sh
python3 tools/rejudge.py --old /tmp/lwlogic-main.py --split-only --out docs/rejudge/split mac-L1-seed1=PATH/TO/RUN gb10-L4-seed1=PATH/TO/RUN
```

The four misreadings were judged with `--fixes`, which names only the fixes the old judge does not have (a fix both judges share can give back an old verdict by accident), and with the three newer GB10 runs added:

```sh
python3 tools/rejudge.py --old /tmp/lwlogic-main.py --fixes likely,truth,entity,first --out docs/rejudge/misreads mac-L1-seed1=PATH/TO/RUN gb10-L5-seed1=PATH/TO/RUN gb10-neg-seed1=PATH/TO/RUN gb10-neg2-seed1=PATH/TO/RUN
```

Since the split view, the judge's reports also carry the split lines. So the first command above, run with this judge, writes reports that differ from the ones in this folder by those lines; the verdicts and CHANGES.md are the same. The reports with the split lines are in `split/`.
