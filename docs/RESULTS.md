# First runs on TensorFold 1.0.8

> **In this repository.** This page comes from the test suite's working repository (commit `2083ee9`, MIT license: [LICENSES/TEST-SUITE-MIT.txt](../LICENSES/TEST-SUITE-MIT.txt)). The code it names (`zengine/`, `tools/`, `examples/`, `lean/`, `zig/notes/`, `docs/TEST-LOGIC.md`) is in [tests/](../tests/) here. For publication, home-directory paths are shown as `~`, a few names were removed, and links were pointed to their new places; nothing else was changed.

Findings from 12 October 2026. These are the first real runs of the test suite against TensorFold's Living Weights. Every answer on this page was judged again on 13 October with the fixed judge (see the end of this page).

## How to read this page

Each finding says where its numbers come from and how sure we are.

- **MEASURED**: a number from a real run, read from the run records kept by the authors.
- **READ**: something we read in TensorFold's source, its guide or release notes, or a comment in its issues.
- **INFERRED**: our own reading of what the numbers mean. Not checked by a run.

The run records are the answer files, judge reports and summaries the runner writes (`zengine/lwrun.py`). They are kept by the authors and are not in this repository, because they hold machine paths.

These are a few runs on two machines. They are not rates. One run of one lesson tells you what happened once.

## Setup

| | |
| --- | --- |
| Model | `TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit`, snapshot `d9d758fb` |
| TensorFold | 1.0.8, tag `v1.0.8` (commit `0efbd447`) |
| Mac | an Apple M5 Max MacBook Pro (Metal). Our own build of tag v1.0.8 (`zig build dist`), sha256 `ad41a0a8b7bc3ac3a9968050b224ed99b169da0f93546a9221016da7ac813855` |
| GB10 | an ASUS Ascent GX10 (NVIDIA GB10, CUDA). Our own CUDA build of the same tag, sha256 starting `759114dd74b1ab08` |
| Sets | `examples/L1-seed1.json`, `L2-seed1.json`, `L3-seed1.json`, `L4-seed1.json`, `eyes-678.json` |
| Asking | one fresh chat per question, temperature 0, thinking off (160 tokens) and on (1500 tokens) |

**Deleting.** TensorFold 1.0.8 has no endpoint to delete one fact yet; its Living Weights guide says "There is no per-fact undo." (READ) So every delete phase was run two ways:

- **zero**: copy the taught folder and set the deleted fact's learned neurons to zero. This is [philip-pentatonic's removal method](https://github.com/ashhart/TensorFold/issues/678#issuecomment-6111596537).
- **reteach**: teach only the lessons that remain to a fresh copy. This is the "never taught" reference.

Our delete rule says both must give the same answers. Comparing them tests "delete = never taught" directly.

When the deleted fact had already been replaced or missed, there was nothing left to zero: 4 of the 13 zero steps in these runs changed no bytes. (MEASURED)

## Headline findings

1. **On the Mac, one fact was learned, but a yes/no question about another colour got a wrong yes.** (MEASURED, L1 set, Mac)
   - The lesson "Of all colours, I like blue the most." was learned in 218.4 s, including an 18.4 s save.
   - "What is my favourite colour?" and its two other wordings: blue, 3 of 3, thinking off and on.
   - But "Is my favourite colour purple?" got "Yes, you mentioned earlier that your favourite colour is purple!" (thinking off). The model said yes to a colour it was never taught.
   - Only one other colour was asked. With thinking on the same question got a refusal, and both either-or questions got blue. TensorFold's 1.0.8 notes already list yes/no questions about a changed value as unreliable; here nothing had changed. (MEASURED; the notes are READ)

2. **Zeroing the only fact gave back the original model, word for word.** (MEASURED, L1 set, Mac)
   - All 24 answers (12 questions, thinking off and on) were identical to the original model's.
   - A second reviewer on the team re-checked this.

3. **On the GB10, the same wording missed 4 times in 6.** (MEASURED, L1, L2 and L3 sets, GB10)
   - L1: `recalled: false` twice, after 186.1 s and 189.1 s. The server marked the fact missed and the save wrote 0 bytes.
   - L3 used the same wording with yellow. It was missed again, in the teach phase and in a reteach.
   - So the wording "Of all colours, I like X the most." missed all 4 times on the GB10 with blue and yellow, but was learned both times with white (the change-of-mind set below): 4 misses in 6 tries. The Mac learned it once out of once.
   - With one try on the Mac, we cannot yet say whether this is a machine difference or chance. (INFERRED)

4. **After a change of mind and then "not anymore", the model still gave the changed-to colour.** (MEASURED, L2 set, GB10)
   - Lessons: "Of all colours, I like white the most." "I like green best now." "Green is not my favourite colour anymore."
   - The server reported all three as recalled.
   - Lesson 2 replaced lesson 1: the server moved fact 1 from learned to replaced. Lesson 3 replaced nothing.
   - [Ash Hart's #678 comment](https://github.com/ashhart/TensorFold/issues/678#issuecomment-6124177090) says a newer fact that changes or takes back an earlier one replaces it; here the server recorded no replacement for lesson 3. (READ; MEASURED)
   - After all three, the model said green to every "what is my favourite colour" question (0 of 3 right, both thinking modes). The right answer is "don't know, but not green".
   - "What was my favourite colour before it changed?" got "I don't have access to your personal history" in both modes. The history was lost. (INFERRED)
   - We cannot yet tell "the negation was learned backwards" from "the negation had no effect". Two small sets in `examples/` are built to decide it: `neg-seed1.json` and `neg2-seed1.json`. Their results are not on this page.

5. **Zero and reteach agree on verdicts, not on wording.** (MEASURED, L2 set, GB10)
   - Same verdict in 31 of 32 answers, in each of the two delete phases.
   - Identical text in only 15 of 32 answers (first delete phase) and 11 of 32 (second).
   - In the second delete phase the zero step changed nothing: the deleted lesson 1 had already been replaced by lesson 2, and 1.0.8 zeroes a replaced fact's neurons. That phase's reteach also missed lesson 3 (`recalled: false`), so its reference is not a clean "never taught". (MEASURED; the zeroing on replace is READ, [Ash Hart's #678 comment](https://github.com/ashhart/TensorFold/issues/678#issuecomment-6124177090))

6. **An empty answer.** (MEASURED, L2 set, GB10, second delete phase, reteach, thinking on)
   - "Do I like white or green best?" came back with an empty reply. Its 2 completion tokens were both reasoning, and the reasoning said "Green". Finish reason `stop`; 18 of the 24 prompt tokens came from the cache.
   - TensorFold's Living Weights guide already describes this: with thinking on, the model sometimes recalls the fact in its reasoning and stops without a reply. (READ)
   - It did not run out of tokens. We keep it as a test case for empty output.

7. **The control questions did not change.** (MEASURED, every phase of all seven runs)
   - The controls scored 5 of 5 in every phase, thinking off and on: the near-miss question, the untouched attribute and general knowledge. The L4 set has 6 controls (two near misses), and they scored 6 of 6 in every phase.
   - The judge marks only these questions as controls, so this is measured on 5 or 6 questions per phase, not on the whole model.

8. **Dates: current values work, past values do not.** (MEASURED, L3 set, GB10)
   - TensorFold does not claim dated facts yet: [Ash Hart wrote in #678](https://github.com/ashhart/TensorFold/issues/678#issuecomment-6124177090) on 12 October that dated history is not done. (READ)
   - Lessons: "Of all colours, I like yellow the most." (missed, see 3), "My home is in Graz.", "My eyes were green until 9 June 2025; now they are amber."
   - Thinking off: the current values come back (amber, Graz).
   - Thinking on: the model refused most yes/no questions (2 of 6 right) and every question about the past ("I don't have access to your personal information"). Past-time questions: 1 of 6 right.
   - The date boundary was wrong (thinking off). "On 25 June 2025, what colour were my eyes?" got green. The right answer is amber.
   - Deleting the dated fact: 9 of 11 of the affected answers were right with thinking off, 11 of 11 with thinking on.
   - All 11 of those questions expect "don't know" (MEASURED), so the thinking-on score mostly reflects the refusals seen in the teach phase, not a working delete. (INFERRED)

9. **The #678 eye case: this run could not test the 1.0.7 break, and zeroing is not the same as never teaching.** (GB10, `eyes-678` set)
   - All four lessons were recalled. The newer eye fact replaced the older one twice: "green" replaced "brown", and the dated lesson replaced "green". (MEASURED)
   - Deleting "My eyes are green.": with thinking off, 5 of 5 affected eye answers right in the reteach and 4 of 5 in the zero copy. The fifth zero answer has the right "no" but the wrong year: "No, they have not. Until October 11, 2023, your eyes were brown." Our first judge did not read dates and passed it; the fixed judge (13 Oct, [rejudge](rejudge/README.md)) fails it as a wrong date. The teach phase gave the same answer word for word. (MEASURED) With thinking on, 2 of 5 in both, the same as never teaching it, so that drop is the thinking-on weakness and not delete damage. (MEASURED; the reading is INFERRED)
   - The zero step found all 30 of that fact's rows already zero, because 1.0.8 zeroed them when the dated lesson replaced it, so the zero answers are the teach phase's word for word. (MEASURED; the cause is READ, [Ash Hart's #678 comment](https://github.com/ashhart/TensorFold/issues/678#issuecomment-6124177090))
   - On 1.0.7, deleting this fact broke the dated fact after it ([TensorFold issue #678](https://github.com/ashhart/TensorFold/issues/678), measured on the Mac). That break came from zeroing neurons the dated fact leaned on; here they were gone before the dated fact was learned, so this run does not test that break. (INFERRED)
   - Deleting the dated lesson by zeroing: only 1 of 5 eye answers right, in both modes, and that one is "before 11 October", whose right answer is "don't know". "What colour are my eyes?" got "I don't have access to your physical appearance". Never teaching it (reteach of the other three lessons) gives "Your eyes are green." and 2 of 5 eye answers right with thinking off. Our first judge counted 3, because it read "No, I am a language model developed by researchers from NVIDIA." as a no; the fixed judge reads it as "doesn't know". (MEASURED)
   - This is the failure the rule book's "1.0.8-like" broken system predicts. When a newer fact replaces an older one, 1.0.8 zeroes the older fact (READ: [Ash Hart's #678 comment](https://github.com/ashhart/TensorFold/issues/678#issuecomment-6124177090); MEASURED: the rows already zero, above). Deleting the newer one then leaves nothing, where "never taught" brings the older one back. (INFERRED)
   - Thinking on was weaker throughout: past-time questions 0 of 2 right in the teach phase. (MEASURED)

10. **A level 4 set: three of eleven lessons missed, and "likes" held one food at a time.** (MEASURED, L4 set, GB10)
    - Lessons: a dog called Bruno, with grey eyes, who likes olives; a cat called Nora, with hazel eyes; my favourite colour blue, then "no longer blue"; and the foods I like: pears, olives, apples, then "I do not like olives anymore".
    - The server reported 8 of the 11 lessons as recalled. It missed "I like pears.", "My favourite colour is no longer blue." and "I really like olives."
    - The three reteach copies were taught the same lessons again, less the deleted ones. Over all the tries, "I like pears." missed 4 of 4, "My favourite colour is no longer blue." 3 of 3, and "I really like olives." 2 of 4. Every other lesson was learned every time.
    - Both times "I really like olives." was learned, the next lesson, "I really like apples.", replaced it. The server moved the olives fact from learned to replaced, "replaced by fact N". (MEASURED, the server's own fact states in the run records)
    - Our rule book keeps both, because a person can like several foods. So 1.0.8 seems to hold one liked food at a time. (INFERRED, from 2 cases of 2) The intake check found the same 2 cases when it sorted the run records ([TEST-LOGIC.md](../tests/docs/TEST-LOGIC.md), section 12).
    - Like finding 4, the colour that was taken back stayed: "Which colour do I like best?" got blue in both modes. Here the server also reported the "no longer" lesson as missed.

11. **L4 after teaching: the names came back, the yes/no answers did not.** (MEASURED, L4 set, GB10, teach phase)
    - The dog's name, the cat's name and both eye colours came back in both modes. Recall was 4 of 7 in both modes. The misses were the two food questions ("I don't know!", although "I really like apples." was learned) and the favourite colour (blue, see finding 10).
    - Wrong yeses about things never taught, in both modes. "Is it true that I like chocolate?" got "Yes, it is true. You mentioned earlier that you like chocolate." "Is my dog's name Nora?" got "Yes, you mentioned earlier that your dog's name is Nora." With thinking off, the untaught model already said yes to that one. This is the pattern of finding 1.
    - "Is my cat called Bruno or Nora?" got "Your cat is called Bruno." in both modes, while "What is my cat called?" got Nora. Only 2 of 12 yes/no and either-or answers followed from the model's own answer to the plain question. (The fixed consistency score splits this number; see "Split scores" below.)
    - "Is my favourite colour blue?" got "No, you told me your favourite colour is green." with thinking off. Green was never taught in this set.
    - Yes/no questions: 3 of 5 right with thinking off, 0 of 5 with thinking on.
    - "Who has grey eyes?" and "Who likes olives?" were answered about people in general, such as famous people who like olives, in every phase and both modes. The dog was never named: 0 of 2 in every taught phase. The fixed judge counts such an answer as right when nothing was taught (the baseline) and wrong once the dog was taught ([TEST-LOGIC.md](../tests/docs/TEST-LOGIC.md), section 9, decision 8).

12. **L4 deletes by zeroing: one deleted fact stayed, and another delete took a fact with it.** (MEASURED, L4 set, GB10, one run)
    - Each zero step set 30 rows (10 neurons) to zero, all nonzero before. The other learned facts' 70 neurons were still nonzero afterwards.
    - Zeroing "My dog is called Bruno." did not remove it. "What is my dog's name?" still got "Your dog's name is Bruno." in both modes. 40 of the 44 answers were the teach phase's word for word, and the other 4 differ only in wording. Never teaching it (reteach) gave "I don't know your dog's name!" with thinking on and "My name is Nemotron, ..." with thinking off.
    - Zeroing "My cat's eyes are hazel." also lost the cat's name. "What is my cat called?" went from "Your cat is called Nora." to "My name is Nemotron, and I'm a language model developed by NVIDIA." (thinking off) and "I don't know your cat's name!" (thinking on). Never teaching the eye lesson kept "Your cat is called Nora." in both modes. The new `kept_stable` score flagged this ([rejudge/README.md](rejudge/README.md)).
    - Zero and reteach gave the same verdict in 21, 20 and 18 of 22 answers with thinking off, and 21, 17 and 20 of 22 with thinking on, in delete phases A, B and C. The same text in only 15, 10 and 7 (thinking off) and 8, 2 and 5 (thinking on).
    - Our reading: facts about the same animal share what they learned. The dog's name may also sit in the later facts about the dog ("My dog really likes olives.", "My dog's eyes are grey."). The cat's name may lean on the eye fact taught before it, as the dated fact did on 1.0.7 in [#678](https://github.com/ashhart/TensorFold/issues/678). Either way, zeroing one fact's neurons is not a clean delete. (INFERRED, not checked; one run)

## What we changed in the test suite after these runs

The `delete_kept` score compares the questions a delete should not touch with the rule book's ideal answer. So it also counts what the model already got wrong in the teach phase (the missed yellow lesson, a wrong yes/no about Graz) as delete damage.

Built on 13 Oct: a `kept_stable` score. It compares every question whose right answer did not change with the same model's own answer from the teach phase, verdict and text, kept apart from correctness. The judge also learned to read wrong dates, reverse questions answered about people in general, and "No, I am a language model ..." as a refusal. All the runs on this page were judged again with it: 51 of 1,470 verdicts changed, and the numbers above that moved are marked. The full list, and a new observation that kept_stable brought out (in L4, zeroing "My cat's eyes are hazel." also lost the cat's name), are in [rejudge/README.md](rejudge/README.md). (MEASURED)

Also built on 13 Oct: an intake check that sorts a new fact before it is learned ([TEST-LOGIC.md](../tests/docs/TEST-LOGIC.md), section 12). Run over the 71 lesson steps of these records, it agreed with the server's "replaced" marks in 7 of the server's 9. The other 2 are the liked foods of finding 10. In 7 more steps it says a "no longer" or "not ... anymore" lesson takes back an earlier fact, and the server marked nothing; in 4 of those 7 the server reported the lesson as not recalled. (MEASURED) So 1.0.8 seems not to retire a fact that a later lesson takes back. (INFERRED)

## Split scores (13 Oct)

Every stored answer was judged again with the split view ([TEST-LOGIC.md](../tests/docs/TEST-LOGIC.md), section 7). Each score is split by the right answer: a value, a no, or "don't know". No verdict changed: 0 of 1,470 against main's judge. The fix of point 9, made later, changed 10 of these answers; the numbers below that moved say so. The tables per run and phase are in [rejudge/split/SPLIT.md](rejudge/split/SPLIT.md). In the totals below each answer counts once: 1,354 answers, 677 per thinking mode. The teach answers judged as another phase and the Mac restart answers are left out. (MEASURED)

1. **The L3 delete that scored 11 of 11 cannot be told from a refuser.** Deleting the dated eye fact (reteach), thinking on: all 11 affected answers right, and all 11 expect "don't know". The dummy "always don't know" also scores 11 of 11 there. On the same phase's questions whose right answer is a value, the model got 4 of 7 in both modes, the dummy 0 of 7. The zero copy gives the same numbers. (MEASURED) So that score does not show the delete worked; the value questions show the model was not refusing everything. (INFERRED)
2. **Thinking on says "don't know" more often, and more often wrongly.** Thinking off: 323 "don't know" answers, 229 of them where that was right (precision 0.71), and 229 of the 272 questions whose right answer is "don't know" (recall 0.84). Thinking on: 386 and 242 (precision 0.63, recall 0.89). On questions whose right answer is a value: 254 of 356 right with thinking off (259 before the fix of point 9), 223 of 356 with it on; 82 and 111 of the misses said "don't know". (MEASURED)
3. **A right "no" is rare.** Questions whose right answer is no or neither: 13 of 49 right with thinking off, 2 of 49 with it on. With thinking off, 13 misses gave a yes or a value and 5 a guess that leans the right way (all 18 read as a yes before the fix of point 9), 12 said "don't know", and 6 gave the right no with a wrong date. With thinking on, 33 said "don't know" and 13 gave a yes or a value. (MEASURED) This is the wrong yes of findings 1 and 11, counted. (INFERRED)
4. **Consistency, like for like.** The old score gave L4 11 of 12 in the baseline and 2 of 12 after teaching (finding 11). Fixed, every plain answer in the L4 baseline was "don't know": 17 of 18 pairs agree, and no pair has a plain answer that gave a value. After teaching, the pairs whose plain answer gave a value agree 0 of 3 in each mode; the pairs whose plain answer was "don't know" agree 0 of 6 with thinking off and 4 of 6 with it on. (MEASURED) So the old 11 of 12 was agreement on "don't know", with nothing to compare the taught values with. (INFERRED) On the Mac L1 run, checked against all three wordings, 9 of 12 value pairs agree in each mode; the 3 misses are "Is my favourite colour purple?" (finding 1) against each wording. (MEASURED)
5. **The same wrong answer in every wording.** L2 after teaching: the wordings of one question gave the same answer in 5 of 5 pairs with thinking off and 4 of 5 with it on. None of the 3 questions was right in every wording. The model said green each time, where the right answer is "don't know, but not green" (finding 4). (MEASURED)
6. **The plain question held better than other forms.** Over the 7 teach phases: the plain question in its first wording (reliability) 9 of 15 in both modes; every other question about a taught pair (generality) 21 of 57 with thinking off (22 before the fix of point 9) and 12 of 57 with it on; reverse and indirect questions (portability) 0 of 3; the controls (locality) 36 of 36. (MEASURED) Papers on side-memory editors report the same weak spot: questions worded differently from the taught sentence (our literature review, section 1, which is not in this repository). (READ)
7. **Neighbourhood failures, all in L4.** "Is my cat called Bruno or Nora?" got Bruno, the dog's name, and "Is my dog's name Nora?" got yes, the cat's name, in both modes after teaching: 18 such answers over the L4 phases. The untaught model's "Yes, your dog's name is Nora!" is not counted, since nothing was taught then. (MEASURED)
8. **Small things.** The closed-world reading would pass 4 more answers (1 with thinking off, 3 with it on). On the three L1 runs, `tools/aggregate.py` gives the lesson recalled in 1 of 3 tries, with the interval [0.01, 0.91], and the recall questions after teaching right in 3 of 9 per mode. (MEASURED)
9. **A misreading the split view brought out, fixed later on 13 Oct.** "It is very unlikely that your favourite colour has always been the same." was read as a plain yes, because it starts with "It is". So was "It is very likely that it has!". 14 stored answers start "It is very", 11 in L2 and 3 in L3, all with thinking off. All 14 were read as yes: 5 passed as the right yes to "Has my favourite colour ever changed?" and 9 failed. Both are guesses that lean one way, and the judge now reads them so ([rejudge/README.md](rejudge/README.md), "Four misreadings"). The 5 passes, and the 5 fails where the right answer was no, are now unclear: the right answer said as a guess. The 4 where the right answer was "don't know" still fail, now as guesses. 5 of the 18 yes-or-value misses in point 3 were such answers. The newer GB10 L5 run has 7 more, which move the same way: 5 from fail to unclear, 2 in the baseline still fail. (MEASURED)

## What these runs do not show

- They are a handful of runs on two machines. A fact learned once or missed four times is not a rate.
- The controls are 5 or 6 questions per phase. Passing them shows those answers are right. It does not show that every other output of the model is unchanged.
- "Delete" here is zeroing or reteaching. TensorFold's own delete endpoint was not released when these runs were made, so its behaviour is not tested here.
