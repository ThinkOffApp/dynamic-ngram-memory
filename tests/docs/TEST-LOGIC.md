# Living Weights test logic

The design of the test suite: what the rule book knows, how test sets are made, how answers are judged, and how a set is run against TensorFold.

## In short

We teach the model short facts, one at a time. Then we ask it questions.

A small rule book gives the one right answer to each question. It uses only the facts we taught.

The rule book understands change ("my eyes are green now"). It understands taking back ("not blue anymore"). It understands dates ("until 11 October").

It also reads what a sentence implies. "Not blue anymore" means it was blue before. "Green now" means it was not green before.

Anything never said is "don't know". It is never "no".

A test set is a list of facts and a list of questions. It also holds the right answer at each stage: before teaching, after teaching, and after deleting some facts.

A generator makes test sets at six levels. Level 1 is one fact. Level 5 is up to twenty facts about several people, pets and things. Level 6 asks two-step questions, which need two facts put together: "Which country do I live in?" after "I live in Oslo." (section 14). The same set number always gives the same set.

Level 7 fills the pocket. It teaches 64 facts one at a time, each about a different person, pet or thing, and asks about every earlier fact after every fifth lesson. That gives a retention curve. Its own runner, `zengine/capacity.py`, also handles a full pocket, reads each fact's weight norm and deletes facts one at a time (section 15). It has run only against a mock server so far.

A judge reads the model's answers. It marks each one pass, fail or unclear. A wrong guess ("maybe brown") fails. The right answer said as a guess ("probably green") is unclear, for a person to look at. An answer about the model itself ("My name is Nemotron") never counts as an answer about you. A dated answer with a wrong date ("Until October 11, 2023, your eyes were brown") fails.

Every score is also split by what the right answer is: a value, a "no", or "don't know". A model that says "I don't know" to everything passes every "don't know" question, so a pooled score can look good. Split, it scores zero on the value questions (section 7).

An intake check sorts a new fact before it is learned: new, replaces an old fact, already known, or implied by what is stored. Only the first two need learning (section 12).

A night clean-up prototype frees the neurons of facts that the stored questions do not need, one fact at a time, in a copy of the model. It can rank the facts by weight norm first, and after freeing it can teach a related fact and check that the freed fact stays gone (level 8, section 16). It has run only against a stand-in server so far (`docs/NIGHT-PRUNE.md`).

Question forms v2 (`gen --forms v2`) add three kinds of probes to a set, and keep everything the set had (section 13). They ask yes/no about values never taught, to measure wrong "yes" answers. They ask "who has it?" three ways, to tell the reversal curse from a question read as a general one. And they ask about likes as a set, after every new like, to see whether earlier likes are kept.

Two optional reference phases put the taught model beside two others. One is the untaught model with the lessons written into its system prompt. The other is a lookup that keeps only the latest value per fact, with no model at all. The runner reports the gap between the taught model and the first one, per kind of question, and for every change or take-back lesson whose effect today's answers can show, whether the old value, what holds now, both or neither came out (section 8). They have run only against the mock server so far.

Two rules matter most to us:

- Deleting a fact must give the same answers as never teaching it. This is our choice, not a TensorFold promise yet (see section 9). One case of it is the eye-colour break we found in [TensorFold issue #678](https://github.com/ashhart/TensorFold/issues/678): a dated fact must keep its answers when an older fact is deleted.
- Teaching a fact must not change the answers about anything else. This is TensorFold's own aim.

The rule book is written twice. The Python copy runs, and all 102 of its self-checks pass.

The Lean copy is for a proof checker, a program that checks proofs line by line. Lean checks all 93 of its theorems. It also recomputes every expected answer of the example sets, and agrees with the Python copy on all 5,910.

We also built eight systems that are broken on purpose. Two break like 1.0.7 did, and one copies 1.0.8's replace step, which breaks our delete rule. The self-checks catch every one. Level 6 adds eight more, two broken scorers and three broken test sets (section 14).

Files, relative to the repository root:

| File | What it is |
| --- | --- |
| `docs/TEST-LOGIC.md` | this document |
| `zengine/lwlogic.py` | the rule book, the generator, the judge and the self-test. Python 3, standard library only |
| `zengine/lwrun.py` | the runner: teaches, deletes and asks against a real TensorFold server, then judges. Optional reference phases: the lessons in the system prompt, and a latest-value lookup (section 8). Standard library only |
| `zengine/lwrun_selftest.py` | the runner's self-test, against a mock TensorFold server that also honours a system prompt (section 8) |
| `zengine/intake.py` | the intake check: sorts a new fact before it is learned (section 12). Standard library only |
| `zengine/prune.py` | the night clean-up prototype, with its own self-test against a stand-in server (`docs/NIGHT-PRUNE.md`). Standard library only |
| `zengine/levels.py` | level 6, two-step questions: the generator, the "echo" verdict, the stated-against-implied scores and their self-test checks (section 14). The rules themselves are in `lwlogic.py`. Standard library only |
| `zengine/capacity.py` | the level 7 runner: fill the pocket one fact at a time and ask as it fills, with its own self-test against the runner's mock server (section 15). Standard library only |
| `zengine/notes.py` | the note memory, notes only: notes written through the intake check, found by token n-grams under a token cap, given to the model as quoted data; its self-test and the offline measurements (section 17, `docs/NOTES-LAYER.md`). Standard library only, except a real tokenizer, which needs the `tokenizers` package |
| `lean/LwLogic.lean` | the same rules in Lean 4, with 93 theorems, all proved. Checked by Lean 4.35.0 |
| `lean-toolchain` | the Lean version the proofs are checked with |
| `examples/L1..L5-seed1.json` | one generated set per level, set number 1 |
| `examples/L6-seed1..3.json` | three level 6 sets (two-step questions), set numbers 1 to 3 |
| `examples/L7-seed1.json` | the level 7 set: 64 facts, 14 checkpoints, two pairs, three delete phases |
| `examples/eyes-678.json` | our #678 eye-colour trial written as a test set |
| `examples/neg-seed1.json`, `examples/neg2-seed1.json` | two small hand-made sets: is "not my favourite anymore" stored as not-X, ignored, or as X? |
| `examples/v2/` | the same sets with the question forms v2 added (section 13): L1 to L5 set 1, the two negation sets, and L4 with every like restating the whole set |
| `examples/*.check.lean`, `examples/v2/*.check.lean` | Lean checks that recompute every expected answer of a set (5,910 checks in 18 sets, all pass; 2,882 of them are level 7's) |
| `examples/eyes-678.judged-1.0.7.txt` | the judge run on our existing 1.0.7 removal answers |
| `docs/RESULTS.md` | findings from the first runs on TensorFold 1.0.8 |
| `docs/NIGHT-PRUNE.md` | how the night clean-up prototype works, and what passing does and does not show |
| `docs/rejudge/` | every stored 1.0.8 run record judged again with the fixed judge, and every verdict that changed |
| `tools/rejudge.py` | judges stored run records with two versions of the judge and lists what changed; also writes the split scores |
| `docs/rejudge/split/` | the split scores of every stored 1.0.8 run record, and the two dummy systems on the same sets (section 7) |
| `docs/rejudge/misreads/` | every stored run record and three newer GB10 runs judged again after the four misreadings of section 7 were fixed, with every verdict that changed |
| `tools/aggregate.py` | reports repeated runs of one set as "k of n", per question, per phase and per lesson (section 8) |

---

## 1. What is modelled

**Entities** are the things facts are about: the user ("me"), people ("my sister", "my brother"), animals ("my cat", "my dog"), vehicles ("my car", "my bike") and objects ("the boat", "the tent"). Some are deliberately similar, so a fact about one must not leak to the other.

Level 7 needs a pair of its own for each of 64 facts, so 16 more entities were added on 13 Oct: eight people (my aunt, uncle, friend, colleague, grandmother, grandfather, niece, nephew), three animals (my horse, parrot, hamster), two vehicles (my van, scooter) and three objects (the kite, lamp, clock). Each has a similar entity of its own (aunt and uncle, lamp and clock). That makes 100 entity-attribute pairs. Levels 1 to 5 name their own entities, so their sets did not change: the self-test regenerates the committed examples and compares them byte for byte (MEASURED).

**Attributes** belong to entities. Each has a finite list of values. The model may answer with a value outside the list ("hazel" when no fact says hazel), and the judge notices that.

| Attribute | Applies to | Kind | Values (generator) |
| --- | --- | --- | --- |
| eye_colour | me, people, animals | single | brown, green, blue, grey, hazel, amber |
| fav_colour | me, people | single | red, blue, green, yellow, purple, orange, black, white |
| name | people, animals, vehicles, objects | single | Ana, Milo, Luna, Otto, Bruno, Nora, Pablo, Elsa |
| colour | vehicles, objects | single | red, blue, green, yellow, black, white, silver, orange |
| home_city | me, people | single | Lisbon, Oslo, Tampere, Porto, Ghent, Graz, Bergen, Turku |
| likes | me, people, animals | **multi** | apples, cheese, olives, chocolate, mushrooms, pears |
| owner | objects | single | me, my sister, my brother, my neighbour |

**Single-valued** means one value at a time: a new eye colour replaces the old one. **Multi-valued** means a set: "I like olives" adds olives and keeps cheese.

"I like blue." is ambiguous. It could be a favourite (single) or one of several likes (multi). Our #678 trials read it as a favourite, as TensorFold 1.0.8 does. The generator avoids the ambiguity: colours go in `fav_colour` ("My favourite colour is blue.", "I like blue best."), and `likes` holds foods.

**Truth has three values**: yes, no and unknown. Anything never said is unknown, not no. The state of an attribute at a moment is the truth of each of its values: "eyes are green" makes green yes and every other colour no. "Eyes are not green anymore" makes green no and the others unknown.

**Time.** Every fact holds over an interval of time.

- A present-tense fact ("My eyes are green.") holds from the moment it is told. It says nothing about earlier dates, because the model is never told when a lesson was given.
- "were brown until 11 October 2026" holds up to the start of 11 October.
- "have been red since 3 March 2025" holds from the start of 3 March.
- "have always been blue" holds at every time.
- "now" is after everything.
- No generated question asks about the boundary day itself ("on 11 October"), because people read "until 11 October" both ways. The rule book says the new value holds on that day.

**What a wording implies (hints).** Some wordings say something about the other side of the change they describe. A person reading them would use that.

| Wording | Also means |
| --- | --- |
| "My eyes are green now." / "has changed; it is green now" | they were not green just before it was said |
| "My eyes are not green anymore." | they were green just before it was said |
| "My eyes have been green since D." | they were not green just before D |
| "My eyes were green until D." | they are not green from D on |

A hint only fills a gap. It acts only where the value is still unknown at that moment, so it never overrides anything a lesson says outright. "I only like cheese now" carries no hint, since it says nothing about cheese before.

Inside the code, time is a whole number ("tick"): a date D is D × 1,000,000, and a lesson told today gets a tick inside today, after every date its text mentions. A hint about "just before" sits on the tick just before.

## 2. Lessons: the operations and how they are said

A **lesson** is one learn request, one sentence or two. Each lesson becomes one or more **writes**: (entity, attribute, interval, effect).

| Lesson | Meaning | Writes |
| --- | --- | --- |
| `ASSERT(e, a, v)` | e's a is v (present tense) | from now on: v (single: v and nothing else; multi: add v) |
| `ASSERT(e, a, v, since D / until D / between D1 D2 / always)` | e's a was or is v during that interval | that interval: v. Hint: not v just before the start, and not v from the end on |
| `CHANGE(e, a, v)` | it is v now; it was something else before | from now on: v and nothing else (on likes: "now I only like v"). Hint (single-valued): not v just before |
| `RETRACT(e, a, v)` | v no longer holds | from now on: v is no. Hint: v just before |
| `RETRACT(e, a)` | no value holds any more ("I have no favourite colour anymore") | from now on: every value is no |
| `DATED(e, a, v0, D, v1)` | it was v0 until D, and v1 since D | before D: v0; from D on: v1 |
| `DELETE(lesson)` | remove a lesson from the store | the lesson's writes are gone, as if never taught |

`CHANGE` keeps history. The old value still holds up to the moment of the change, so "What was my favourite colour before it changed?" has an answer.

A newer lesson does not need to say "now" to replace an older one. On a single-valued attribute, a later "My eyes are green." replaces "My eyes are brown." from the moment it is told. `CHANGE` adds the wording that marks a change of mind, and with it the hint.

Each lesson has one to three English wordings, picked by the set number. For eye colour:

| Lesson | Wordings |
| --- | --- |
| ASSERT | "My eyes are green." / "I have green eyes." / "The colour of my eyes is green." |
| ASSERT since | "My eyes have been green since 3 March 2025." |
| ASSERT until | "My eyes were green until 3 March 2025." |
| ASSERT always | "My eyes have always been green." |
| CHANGE | "My eyes are green now." / "My eyes have changed colour; they are green now." / "These days my eyes are green." |
| RETRACT | "My eyes are not green anymore." / "My eyes are no longer green." |
| DATED | "My eyes were brown until 11 October 2026; now they are green." / "Until 11 October 2026, my eyes were brown. Since then they have been green." / "My eyes changed from brown to green on 11 October 2026." |

Every attribute has its own table in `lwlogic.py` (the dictionary `T`). The same entity wording works for anyone: "My sister has green eyes.", "My cat's eyes are green." Sets 1 to 30 of the five levels use every question wording. Five rare lesson forms (for example "My favourite colour has always been blue." and "I have no favourite colour anymore.") appear only in later set numbers of level 5.

Questions also have one to three wordings each:

| Question | Asks | Eye-colour wordings |
| --- | --- | --- |
| `CURRENT(e, a)` | the value now | "What colour are my eyes?" / "What is my eye colour?" / "Tell me the colour of my eyes." |
| `AT(e, a, D)` | the value on a date | "What colour were my eyes on 1 March 2025?" |
| `BEFORE(e, a, D)` | the value just before a date | "What colour were my eyes before 11 October 2026?" |
| `PREVIOUS(e, a)` | the last known value before the current one | "What colour did my eyes use to be?" |
| `ALWAYS(e, a)` / `CHANGED(e, a)` | the same yes/no question, opposite wording | "Have my eyes always been the same colour?" / "Have my eyes ever changed colour?" |
| `EITHER_OR(e, a, x, y)` | which of two | "Are my eyes brown or green?" (asked in both orders) |
| `YES_NO(e, a, v)` | is it v | "Are my eyes green?" |
| `WHO_HAS(a, v)` | reverse lookup | "Who has green eyes?" |
| `GK` | general knowledge, must not change | "What colour is fresh grass?" |

## 3. The rules, as pseudocode

```
store   = list of lessons still in force, in the order they were taught
log     = list of TEACH(lesson) and DELETE(lesson id) entries

effective(log):                                   # deleting = never taught
    deleted = { id | DELETE(id) in log }
    return [ l | TEACH(l) in log, l.id not in deleted ]

writes(lesson):                                   # t = the lesson's tick; "just before t" = [t-1, t)
    ASSERT(e,a,v)            -> [(e, a, [t, ∞),  multi(a) ? ADD v : SET v)]
    ASSERT(e,a,v, [lo, hi))  -> [(e, a, [lo, hi), multi(a) ? ADD v : SET v),
                                 (e, a, [lo-1, lo), PRESUME_NOT v) if lo is a date,     # "since D"
                                 (e, a, [hi, ∞),   PRESUME_NOT v) if hi is a date]      # "until D"
    CHANGE(e,a,v)            -> [(e, a, [t-1, t), PRESUME_NOT v) unless multi(a),
                                 (e, a, [t, ∞),  SET v)]
    RETRACT(e,a,v)           -> [(e, a, [t-1, t), PRESUME v), (e, a, [t, ∞), RETRACT v)]
    RETRACT(e,a)             -> [(e, a, [t, ∞),  CLEAR)]
    DATED(e,a,v0,D,v1)       -> [(e, a, (-∞, D), SET v0), (e, a, [D, ∞), SET v1)]

step(effect, state):                              # state: value -> yes | no | unknown
    SET v          : v -> yes, every other value -> no
    ADD v          : v -> yes, others unchanged       (single-valued: same as SET v)
    RETRACT v      : single-valued and state(v) = yes : v -> no, every other value -> unknown
                     otherwise                         : v -> no, others unchanged
    CLEAR          : every value -> no
    PRESUME v      : if state(v) = unknown: ADD v      else unchanged    # a hint fills a gap only
    PRESUME_NOT v  : if state(v) = unknown: v -> no    else unchanged

state_at(store, e, a, t):                         # t is a tick, or NOW
    s = every value unknown
    for lesson in store:                          # in teaching order: later lessons win
        for (e', a', interval, effect) in writes(lesson):
            if (e', a') = (e, a) and t in interval:
                s = step(effect, s)
    return s

summary(s):                                       # the answer to "what is it?"
    single-valued:  the value with yes            -> VALUE v
                    every value no                -> NONE
                    otherwise                     -> UNKNOWN (but not the values with no)
    multi-valued:   some value yes                -> SET (yes values, no values)
                    every value no                -> NONE
                    otherwise                     -> UNKNOWN (but not the values with no)

answer(store, question):
    CURRENT(e,a)          = summary(state_at(store, e, a, NOW))
    AT(e,a,D)             = summary(state_at(store, e, a, start of D))
    BEFORE(e,a,D)         = summary(state_at(store, e, a, start of D minus one tick))
    PREVIOUS(e,a)         = cur = CURRENT; B = all interval ends of writes on (e,a)
                            C = [ b in B | summary(state_at(b - 1)) ≠ cur ]       # pieces below, unlike now
                            K = [ b in C | summary(state_at(b - 1)) is not UNKNOWN ]
                            K nonempty -> summary(state_at(max K - 1))            # last known value
                            else C nonempty -> summary(state_at(max C - 1))       # e.g. "not green"
                            else UNKNOWN
    ALWAYS(e,a)           = pieces = CURRENT and summary(state_at(b - 1)) for every b in B
                            yes      if every piece is the same VALUE
                            no       if two pieces conflict (two values, or a value and its exclusion)
                            unknown  otherwise
    CHANGED(e,a)          = not ALWAYS(e,a)       (yes <-> no, unknown stays unknown)
    YES_NO(e,a,v)         = state_at(store, e, a, NOW)[v]
    EITHER_OR(e,a,x,y)    = x if x is yes; y if y is yes; NEITHER if both are no; else UNKNOWN
    WHO_HAS(a,v)          = [ e in the set's entities | state_at(store, e, a, NOW)[v] = yes ]
    GK                    = the fixed general-knowledge answer
```

**Conflicts.** Where two lessons cover the same time, the later one wins, because it is applied later in the fold. A timeless "have always been blue" taught after a dated history overrides the history. A dated history taught after "My eyes are green." overrides it too. Hints are the exception: they never override, whichever lesson came first.

**Retractions.** "Green is not my favourite colour anymore" leaves the favourite unknown, not empty: we only learned that it is not green. "Is white my favourite colour?" is then "don't know", not "no".

**PREVIOUS skips gaps.** "I like blue best." "Blue is not my favourite colour anymore." "I like red best now." Then "What was my favourite colour before it changed?" is blue. The unknown stretch between the second and third lesson is skipped, as a person would skip it. "Do I like blue best?" is no.

**Deletes.** A delete removes the lesson from the store and the answers are recomputed. Deleting a retraction brings the taken-back value back. Deleting a change brings the old value back. Deleting the fact before a change leaves the change, and its hint: before it, only "not that value" is known. There is no "un-retract" lesson: saying the fact again is a new `ASSERT`.

**Multi-valued attributes.** "I like apples. I like cheese. I do not like apples anymore." gives likes = {cheese}, not apples, everything else unknown. "Now I only like cheese." makes cheese the only yes and every other food no. The generator asks either-or, previous and always questions only about single-valued attributes.

## 4. The properties that must hold

Each property is checked by `python3 zengine/lwlogic.py selftest` on random stores (1,500 per property, seed 1). Each is also stated as a Lean theorem, and Lean checks every proof. In the rule book most of them hold by construction. That is the point: they become checks we run **on the model**, where nothing makes them automatic.

| Property | Plain meaning | Lean (checked) | Python (ran) |
| --- | --- | --- | --- |
| Delete = never taught | After `DELETE(i)` every answer equals the answer from the store without lesson i. Delete order does not matter. Deleting everything gives the original model's answers. | `delete_is_never_taught`, `delete_answers`, `delete_comm` (proved) | PASS, 1,373 logs with interleaved deletes |
| Locality | Lessons about other (entity, attribute) pairs never change an answer | `locality` (proved) | PASS, 1,500 stores, extra lessons inserted anywhere |
| Self-contained dated fact | A `DATED` lesson answers its own questions whatever was taught or deleted before it. This is what 1.0.7 broke: deleting fact 4 broke fact 5. | `dated_screens_off`; for PREVIOUS and ALWAYS `previous_screened` (both proved) | PASS, 1,500 stores, random earlier lessons deleted. Control: without the dated lesson, the same deletions moved answers in 985 of 1,500 |
| Idempotence | The same lesson taught twice in a row = once | `step_idem`, `dup_lesson`, `dup_change`, `dup_retract`, `dup_dated` (proved) | PASS, 1,500. Control: the same fact again after a conflicting change did move answers (399 cases), as it should |
| Order of independent lessons | Lessons on different keys can be taught in any order; so can two "I like X" lessons | `swap_independent`, `add_comm`; store level for likes `adds_commute_store` (all proved) | PASS, 1,344. Control: swapping two lessons on the same key moved answers in 99 cases |
| Derivable from CURRENT | YES_NO, EITHER_OR and WHO_HAS follow from the answer to CURRENT; a single-valued attribute never says yes to two values | `functional_stateAt`, `yesNo_from_summary`, `eitherOr_from_summary`, `whoHas_from_current` (proved) | PASS, 1,500, read from the answers alone |

**Negative controls: a check that cannot fail is not a check.** Every property above is a function that takes a whole system as input. The self-test runs the same functions on eight systems that are broken on purpose. Each must be caught by the property it breaks, and each is (300 random cases per check):

| Broken system | Caught by | Caught in |
| --- | --- | --- |
| 1.0.8-like: a newer fact zeroes the older one for good (teach blue, teach green, delete green: "don't know" where the rules say blue) | delete | 26 of 272 |
| 1.0.7-like: deleting a lesson also loses the later lessons on its key | delete; dated | 42 of 260; 242 of 300 |
| the #678 eye case: a dated fact is lost when the fact before it is deleted | delete; dated | 7 of 277; 186 of 300 |
| entity mix-up: what is taught about my sister is also believed about me | locality | 51 of 300 |
| overflow: only the last 4 lessons are kept | locality; order | 67 of 300; 32 of 265 |
| repetition hardens: a fact said twice is read as "always" | idempotence | 156 of 300 |
| either-or picks the first option when it does not know | derive | 300 of 300 |
| single-valued attributes kept as sets ("red best", then "blue best" keeps both) | derive | 240 of 300 |

The self-test also runs:

- **An independent second implementation.** A timeline that splits time into pieces lesson by lesson. Its answers must equal the fold's on every question of 1,500 random stores. PASS. Control: a rule book where the earliest lesson wins fails it in 607 stores.
- **Golden cases.** 75 hand-written answers that pin the meaning of change, retract, hints, none, sets, dates, "always", deletes, and "nothing taught means don't know". The second implementation cannot catch a wrong rule that both share; these can.
- **The #678 case.** Facts 1, 3, 4 and 5 of the #678 trial (built into the generator). All four give green now, brown before 11 October and "not always the same". Deleting fact 4 leaves all five of fact 5's eye answers exactly as they were. Deleting 3 changes nothing either. Deleting 5 leaves green now and makes "before 11 October" unknown. Deleting all four gives the original model's answers, and "I like blue" survives every partial deletion. All PASS.
- **Planted bugs (mutation test, run by hand on 12 Oct).** Ten deliberate bugs put into a copy of the rule book: each hint missing or misplaced, a hint that overrides, PREVIOUS not skipping gaps, an off-by-one at "until", a wrong retract rule, either-or reading unknown as yes, and deletes ignored. The self-test fails on all ten. The "until" off-by-one got through at first; the golden answer for the boundary day was added because of it. (The mutated copies are not in this repository.)
- **The judge** against 20 hand-labelled answers from the 1.0.7 runs, 5 mixed-tense or negated sentences, and 32 hand-written refusals, guesses, off-topic answers and honest "don't know"s on every kind of question.
- **The judge fixes of 13 Oct**, each with real 1.0.8 answers where we have them: 36 dated answers (wrong dates fail; right, undated or unsupported ones are judged as before), 28 answers to reverse questions ("Who has grey eyes?"), and 6 yes/no answers that talk about the model itself. Each of these checks, and the guess check above (35 answers), runs a second time with its fix switched off, and must break there. It does: with dates not read 13 labels break; with taught names, general answers and the careful reading of "you" switched off 11; with self-replies read as yes or no 4; and with guesses failing again 8. A check that cannot fail is not a check.
- **A clause with no subject of its own** (an audit of the judge, 13 Oct): 15 answers to "Who likes olives?". "Your dog does not like apples but likes olives." named nobody; now it names the dog, and so do the same answer with "and", "although" or "while", with "it" in the second clause, and with the taught name "Bruno". "... but your cat likes olives." names the cat. "... but does not like olives.", an aside ("does not, however, like olives") and a question back name nobody. With the subject not carried on, 9 labels break. Changing one part of the new rule at a time (8 single edits to a copy, run by hand on 13 Oct) breaks at least one label each time.
- **Joined subjects and lists** (13 Oct): 20 answers to "Who likes olives?". "Your dog and your cat do not like olives." named the dog; now it names nobody, and neither do "Neither your dog nor your cat likes olives.", "You and your dog do not like olives." and "Bruno and Nora don't like olives.". "Your dog and your cat like olives." names both, "Your dog, unlike your cat, likes olives." only the dog. "Your dog likes apples and olives." named nobody; now it names the dog, and so does "apples, pears and olives". "... but not olives.", "..., not olives." and "Your dog does not like apples and olives." name nobody. With joined subjects and lists not read, 9 labels break. Changing one part of the new rules at a time (10 single edits to a copy, run by hand on 13 Oct) breaks at least one label of this check or of the reverse-question check each time.
- **Taking a value back** (a cross-model audit, 13 Oct): 19 answers to "Who likes olives?". "Your dog and your cat like olives, but your dog no longer does." named both; now it names only the cat. "..., but it no longer does.", "..., but not any more.", "... but stopped liking them.", "Your dog likes olives. Your dog no longer does." and "You like olives, but you no longer do." name nobody. "... but your dog does too." still names both, "... but your cat no longer does." still names the dog, and so do clauses about another value ("does not like apples"), the past ("did not as a puppy") or more than the value ("does not like them in salads", "not all of them"). With nothing taken back, 11 labels break. Changing one part of the new rule at a time (11 single edits to a copy, run by hand on 13 Oct) breaks at least one label each time.
- **Four misreadings** (later on 13 Oct, branch judge-misreads): 105 answers in four checks, 7 of them real 1.0.8 answers cut to their first sentences. 20 likelihood openings: "It is very unlikely that your favourite colour has always been the same." is a guess that leans no, "It is very likely that it has!" one that leans yes, "Likely false." and "Most likely, no." lean no, "It is likely that your eyes are brown." leans no on green, and tied to what the user said it is a plain no. 14 true/false openings on plain yes/no questions ("It is false.", "That is false.", "False.", "Incorrect.", "That's true.", "Probably false.", "False, I am a language model ..."). 38 answers about whose the value is: said of the cat, of the cat's taught name, of the user, of the dog in a list, after a subject that waits for its predicate, bare and of the cat, after a yes, and with the asked entity only named ("Your cat has grey eyes. Your dog is asleep.", "Your sister has green eyes. You are tired."); and, read as before, a value given to the asked entity (also as "yours", "so do you", "and so does your dog", "Your dog's are grey too.", "Your dog, like your cat, has grey eyes."), no entity, "you have a dog with grey eyes", "unlike your cat", and a no that says whose it is. 33 first-person answers: the lesson said back is about the user ("I do not like olives anymore." on "Is it true that I like chocolate?" is a no), the model talking about itself is a self-reply ("Personally, ...", "I love olives! Do you like them too?", "I don't have eyes, ..."), what the answer also says of the user decides ("I do not like olives anymore. You still do." and "I don't like olives, but you love them." are a yes, "..., and neither do you." a no, "I like green. You like blue." a yes to "Do I still like blue?", "I don't. You do." a yes), "You are right to ask.", "You told me so." and "Do you?" say nothing of the user, and the model's own act ("I think you like olives. Do you?"), a refusal, an offer of help and an answer tied to memory are read as before. Main's judge gets 14, 9, 16 and 20 of them wrong. Each check runs again with its fix switched off and must break: 13, 9, 16 and 20 labels do. The whose-value check has a second control: with the asked entity counted wherever it is named, as first built, 3 labels break. That first version passed "Your cat has grey eyes. Your dog is asleep." for the dog's eyes (a cross-model audit of 35fc940). The first-person check has one too: with what the answer says of the user not read apart, 8 labels break. The judge of 0c782f4 read "I do not like olives anymore. You still do." as the user's no (an audit of 0c782f4). One canned right answer of the round trips had to change: it answered "Which foods does my sister like?" with "You like cheese.", which the new check reads as said of the user; it is now "She likes cheese.".
- **Every part of the new readers has a case that needs it.** A review on 13 Oct took out one part at a time, 33 single edits to a copy of the judge: the date-of-telling rule, the question's own date, each kind of dated claim, both readings, the other-attribute rule, the "you" rules, the general answer, kept_stable's thinking modes, and so on. The self-test fails on all 33. On the first version of these fixes, 18 of 30 such edits went unnoticed, because their cases came out the same with or without that part; the review added a case for each. (The edited copies are not in this repository.)
- **kept_stable** on a planted case: identical teach and delete answers give 7 of 7; one changed text and one changed verdict are both reported, a question whose right answer moved is left out, and each thinking mode is compared with its own teach answers.
- **The split view** (section 7) on the 12 example sets in `examples/` and 50 generated sets (levels 1 to 5, set numbers 1 to 10), 273 phases (after the merges of 13 Oct; 8 example sets and 236 phases before levels 6 and 7). In a delete phase the "should change" questions must be among its affected ones; a level 7 checkpoint or pair is not a delete and has no such list. Right answers pass every class, and "don't know" scores precision 1 and recall 1. The dummy "always don't know" passes no value question in any phase, and every miss there is a "don't know". The dummy "always yes" passes no "no" and no "don't know" question. Three controls. Pooled, "always don't know" passes half or more of the affected answers in 48 delete phases (L3 set 1, first delete: 11 of 11). A split that leaves the "don't know" answers out of the value count lets it through in all 62 sets. Comparing every phase with the baseline moves "should change" in all 62. One more case keeps the thinking modes apart: right answers with thinking off and "I don't know." with it on, in one judge call.
- **Consistency against every wording** (section 7). L1 set 1, teach phase: the second of three wordings of "What is my favourite colour?" says purple, the others blue. 8 of 12 pairs agree, and all 4 misses are against the purple wording. The baseline gives 12 "don't know" pairs and no value pair. Control: the first wording alone, as before 13 Oct, finds 4 of 4 and misses all four.
- **Paraphrase agreement** on the same answers: 2 of 4 pairs agree in the teach phase, 1 of 2 questions right in every wording. Control: grouping questions by their own id finds nothing to compare.
- **Tags and neighbourhood failures**: 37 hand-set tags on three sets and 9 hand-labelled answers. "Is my cat called Bruno or Nora?" answered Bruno is a neighbourhood failure in the teach phase. It is not one in the baseline, or in the phase where the dog's name is deleted. "Do I like olives?" answered yes, after "I do not like olives anymore.", is a stale value of its own, not the dog's. Controls: a tagger that calls every wording reliability breaks 5 tags; a leak finder that ignores the phase's lessons breaks 2 answers.
- **The split view, part by part** (review, 13 Oct). A review made 36 single edits to a copy of the first version of the split view, and 14 of them went unnoticed by the four checks above. Three of those cannot change any output: a right value is always one of the entity's own values, and no control asks about a taught pair. A new check has a case for each of the other 11. It labels the right answer of every kind (a value, a set, yes, no, neither, none, "don't know", nobody known) and 20 real readings: a value, a set, a refusal, "You do not have one anymore", yes, no, "No, I am a language model ...", an option, neither, a person, a general answer about famous people, a textbook answer, a guess, off topic and two values at once. It pins the closed-world count, the misses that said "don't know", and abstention on six planted answers. It checks that a lesson that takes a value back ("My dog does not like olives anymore.") teaches no value to its neighbours, that a probe about a pair no lesson is about is locality, that either-or questions with other options and questions on other dates are not paraphrases, that a guess is left out of the paraphrase pairs, and that the indirect question ("If you were buying me a shirt ...") is not a wording of the plain one. Controls: 13 parts switched off one at a time, each caught. The same review found two faults, now fixed. A refusal in the fourth sentence ("... I do not know yours.") was counted as a "no", because the first version looked for it in three sentences only; it now reads the answer the way the judge does. And a hand-made set without a baseline phase crashed the judge, which compared the teach phase with a baseline it did not have; such a teach phase now has no "should change" (a case and a control).
- **A judge round trip** on 50 generated sets (levels 1 to 5, set numbers 1 to 10). Answers written from the expected answers must all pass (4,607 of 4,607 do). Answers with the value swapped must all fail (4,607 of 4,607 do).
- **The generator.** Levels 1 to 5 × set numbers 1 to 30: same set every time, no duplicate questions, every set changes at least one answer, and the expected answers recompute exactly.

- **What feeds Lean.** Three checks that need no Lean. The exported schema keeps both fields in one column, as Lean 4.35 requires (controls: the first exports' layout is rejected, and so is a second field indented less than the first). The committed `examples/*.check.lean` and `examples/v2/*.check.lean` files are exactly what the generator makes from the sets (control: a copy with one wrong answer is told apart, in exactly one check). The reader of Lean's output puts each error on its own check, and counts a `sorry`, an extra axiom or a missing theorem as not proved. It also finds a `sorry` outside a theorem, and an axiom the file declares itself.

- **The question forms v2** (section 13): 13 checks. v1 sets stay as they were, the v2 generator is complete, restated likes, the judge on each new form, the answers a review found misread at first, the new scores on planted answers, a judge round trip on 50 v2 sets, and the committed `examples/v2` sets. Each has a control that is broken on purpose.
- **Level 7 sets.** 32 sets of 4, 12, 64 and 100 facts: the same set every time, one plain fact per entity-attribute pair, each fact's CURRENT answer "don't know" before its lesson and its value from then on, the two pairs and the cumulative deletes as described in section 15, and every expected answer recomputes. Control: one fact more than there are pairs is refused.
- **Level 7 yes/no questions tell held from not held.** In 16 sets (540 facts), each fact's yes/no is "don't know" before its lesson and yes or no from then on, at every checkpoint. Control: with the first rule, which also asked likes about another value, 5 yes/no questions of `L7-seed1` stay "don't know" in every phase, so they could not show a fact lost.
- **Older sets unchanged.** The committed examples of levels 1 to 5 and #678 are regenerated byte for byte, after the 16 entities were added. Control: another near-miss entity for "me" changes L1-seed1.
- **Level 7 through the judge.** In every phase of three level 7 sets, all 3,668 right answers pass and all 3,668 wrong ones fail. Only delete phases get the delete scores and kept_stable. Control: read the way levels 1 to 5 are read, the checkpoints get them too.

Result on an Apple M5 Max MacBook Pro (13 Oct 2026): **102 of 102 checks pass** with Python 3.12 and 3.14 on the branch judge-misreads, in 53 to 79 seconds (again after each of the two audit fixes), with other tests running beside them; the runner (32 of 32; 35 of 35 with the cleanup checks of section 8), the night clean-up (25 of 25), the level 7 runner (18 of 18) and the aggregate (3 of 3) pass again with both (MEASURED). Before it, **98 of 98 checks passed** with Python 3.9, 3.12, 3.13 and 3.14 after merging the split view, the question forms v2, level 6, levels 7 and 8 and the reference phases, in 36 to 64 seconds, with other tests running beside them (MEASURED). Before these merges, main's 53 checks passed with Python 3.9, 3.12, 3.13 and 3.14 in 23 to 72 seconds. On their own branches, the split view's 55 checks passed with Python 3.9, 3.13 and 3.14 in 26 to 35 seconds, the forms v2 branch's 63 with Python 3.9, 3.12, 3.13 and 3.14 in 29 to 37 seconds, level 6's 73 with Python 3.9, 3.13 and 3.14 in 33 to 45 seconds, and levels 7 and 8's 54 with Python 3.9, 3.13 and 3.14 in 25 to 41 seconds, with the machine busy with other work (MEASURED). On 12 Oct the 28 checks of then took 22 to 33 seconds. The 15 intake checks of section 12 add about 3 seconds, and the 23 level 6 checks (section 14) about 2 seconds.

**Lean status: checked.** `LwLogic.lean` holds the same definitions and 93 theorems (63 before level 6; section 14 lists the 30 new ones). Lean 4.35.0 checks all of them: no errors, no warnings, no `sorry`, and every theorem rests only on Lean's three standard axioms (MEASURED, 13 Oct 2026). It uses the core library only, no Mathlib. The version is pinned in `lean-toolchain`.

What happened at the first compile:

- One proof needed a fix. In `dup_lesson`, `dup_write _ e a p _ _ rfl rfl` let Lean read the two writes as one, so `cover_mono` no longer fit. Naming the two writes fixed it. The statement did not change.
- Every other proof compiled as written. Lean 4.35 warned about 17 unused simp lemmas and 6 uses of the renamed `if_pos` and `if_neg`. These were cleaned up, so any new warning stands out.
- The generated check files did not parse: Lean 4.35 rejects a structure field indented less than the first one. The generator now puts both schema fields in one column. Only that line of each check file changed.
- The two theorems left as `sorry` are now proved. `previous_screened`: PREVIOUS and ALWAYS read the boundary list, which differs between the two stores. But every piece they look at is one of the dated lesson's two pieces, and its date is a boundary in both stores, so both see the same pieces. `adds_commute_store`: proved through a stronger theorem, `adds_commute_stateAt`, which holds at every point for the whole state. The tick condition in the first statement turned out not to be needed.
- Three small cases, evaluated by Lean's kernel, show that the hypotheses matter. Without a dated lesson last, PREVIOUS does move. Two plain lessons on a single-valued attribute do not commute. On a multi-valued one they do.

The theorems are about the Lean copy of the rules. That the Python copy gives the same answers is checked on test sets, not proved:

- **The example sets.** Lean's kernel recomputes every expected answer of the ten example sets in `examples/` (4,098 checks: 466 in the six sets of levels 1 to 5 and #678, 750 in the three level 6 sets, 2,882 in the level 7 set) and of the eight forms v2 sets in `examples/v2` (1,812 checks). All 5,910 agree (MEASURED, 13 Oct 2026). The level 7 set alone takes Lean 20 to 30 seconds.
- **A wider run.** Levels 1 to 5 × set numbers 1 to 30, plus the two negation probes: 152 sets, and all 11,578 checks agree, in 3 to 4 minutes (MEASURED twice, 13 Oct 2026; these sets are generated on the fly, not stored here).
- **A control.** A copy of a check file with one expected answer replaced by the answer the same question has in another phase. Lean rejects exactly that check (MEASURED). A check that cannot fail is not a check. A second control does the same with a two-step question of a level 6 set ("Which country do I live in?"); Lean rejects exactly that one too (MEASURED).

One command runs all of it from the repository root, with [elan](https://github.com/leanprover/elan) installed:

```sh
python3 zengine/lwlogic.py leancheck
```

It compiles `LwLogic.lean` with `#print axioms` for every theorem. It fails on any `sorry`, also one in an `example` or a `def`, which `#print axioms` does not look at, and on any axiom the file declares. It runs the checks of every example set, maps each Lean error to its check, compares each committed check file with the generator's output, and runs the control, and the two-step control when a level 6 set is among the sets. With a set of restated likes among them (section 13), it also runs a second control on it. Give it other set files to check those too. The same steps by hand:

```sh
lean lean/LwLogic.lean
```

```sh
cat lean/LwLogic.lean examples/L3-seed1.check.lean > /tmp/c.lean && lean /tmp/c.lean
```

CI runs `leancheck` in a job of its own. It installs elan with the official installer, and elan installs the pinned Lean version. The installer script is pinned; the elan it fetches is the latest release.

## 5. Making test sets

```sh
python3 zengine/lwlogic.py gen --level 3 --seed 7 --out set.json
```

```sh
python3 zengine/lwlogic.py show set.json
```

A set holds the lessons (text, formal form, wording choice), the questions (all types, several wordings), the phases, and the expected answer to every question in every phase. The same level and set number always give the same set.

| Level | Content |
| --- | --- |
| L1 | One entity (me), one attribute, one ASSERT. Current in three wordings, yes/no for the true and a false value, either-or in both orders. Delete: the only lesson, which must give back the original model. |
| L2 | Change of mind: ASSERT, then CHANGE, and by set number one of: nothing more, change back, change again, or take it back. Adds PREVIOUS and the ALWAYS/CHANGED pair. Deletes: the last lesson (the earlier value must return); the first lesson (the change stays; before it, only "not that value" is known). |
| L3 | Dated history plus two more attributes, one of them sometimes "since D". Patterns by set number: the #678 shape (brown, green, dated), a dated fact alone, or a dated fact then a change. Adds AT before and after D and BEFORE D, never AT on D itself. Deletes for the #678 shape: the fact just before the dated one (#678's fact 4), the dated one, both plain facts; the other patterns delete the dated fact and a change or an unrelated fact. |
| L4 | Several entities with similar names (my cat and my dog, or my sister and my brother), names and eye colours for both, likes with three adds and one retraction, a retracted favourite colour, reverse lookups ("Who has grey eyes?", "Who likes olives?"). Deletes: the retraction (the food comes back), one of the two names (the other must stay), a retraction and an unrelated fact together. |
| L5 | 8 to 20 random lessons on 4 to 6 (entity, attribute) pairs over 2 to 5 entities, every lesson type (the "no favourite anymore" form is rare, about 1 set in 15). Every pair gets at least one lesson. "Now v" is never said while v already holds, and "not v anymore" never names a value already ruled out. Questions on every pair, including PREVIOUS and the ALWAYS/CHANGED pair where a pair has more than one lesson (13 to 45 questions, capped at 45, plus controls). Two delete variants of 1 to 3 random lessons. |
| L6 | Two-step questions (section 14). 29 lessons on 8 people, pets and things, all 7 attributes and two linking facts. Chains through a city's country, the boat's owner, a name and a link, and compound yes/no questions. Some chains change after a change of owner, home or name. Every single step is also asked alone. Deletes: a fact the kept lessons still imply (it must stay known), one link of each chain (they must become "don't know"), and the changes. |
| L7 | Fill to capacity: 64 plain facts by default (`--n-facts`, 1 to 100), each on an (entity, attribute) pair of its own, so no fact replaces another by the rules. One CURRENT and one yes/no question per fact (the yes/no asks the taught value or another one, half and half; for likes always the taught one). Checkpoint phases, a similar and a far pair, and deletes one at a time (section 15). Run with `zengine/capacity.py`, not `lwrun.py`. |

`--forms v2` adds the question forms of section 13 to any level from 1 to 5; `gen` refuses it for levels 6 and 7, which it is not built for yet. `--likes restate` (with `--forms v2`) makes every "I like X" after the first restate the whole set. `--from SET.json --forms v2` adds the forms to an existing set, such as the hand-made negation sets. Without these options the generator writes the same bytes as before.

**Phases.** Every set has `baseline` (nothing taught: the original model), `teach` (every lesson, in order), and one or more delete variants. Each delete variant starts from the fully taught folder. Every phase is probed with every question, thinking off and on.

Level 7 phases also carry a `kind`: `checkpoint` (the first k lessons, taught so far), `pair` (two lessons on a fresh copy) or `delete`. Only delete phases get the judge's delete scores and kept_stable. Sets without kinds (levels 1 to 5, #678) are read as before: every phase after `teach` is a delete.

**Controls in every set:**

- a **near miss**: the same attribute of a similar entity nobody taught ("What is my sister's favourite colour?"). Expected: don't know;
- an **untouched attribute** of a taught entity ("What colour are my eyes?" when only the favourite colour was taught). Expected: don't know;
- **general knowledge**, two questions near the taught values ("What colour is fresh grass?" when colours are taught; "What is the capital of Norway?" when cities are) and one far away. Expected: the textbook answer, in every phase.

Controls test Living Weights' own promise. Its guide says "Nothing the model already had is changed." (READ)

**Hand check (12 Oct).** Twenty random questions from the level 1 to 5 example sets, ten from any phase and ten from taught or deleted phases only, checked by reading the lessons. All twenty expected answers are right. Example: level 5, lessons "Until 23 December 2025, my dog was called Ana. Since then it has been called Otto.", "My dog was called Elsa until 20 February 2026; now it is called Milo.", "My dog's name is Otto." Then "What was my dog called before the name changed?" is Milo.

## 6. An example, in plain English (L2, set 1)

Three lessons, taught one at a time on 12 October 2026:

1. "Of all colours, I like white the most."
2. "I like green best now."
3. "Green is not my favourite colour anymore."

So white was the favourite, then green, and now we only know it is not green. We do not know whether white came back.

Four phases: nothing taught; all three taught; all taught then lesson 3 deleted (A); all taught then lesson 1 deleted (B).

| Question | Nothing taught | All taught | Lesson 3 deleted | Lesson 1 deleted |
| --- | --- | --- | --- | --- |
| What is my favourite colour? (and two other wordings) | don't know | don't know, but not green | green | don't know, but not green |
| What was my favourite colour before it changed? | don't know | green | white | green |
| Is white my favourite colour? | don't know | don't know | no | don't know |
| Is my favourite colour green? | don't know | no | yes | no |
| Do I like white or green best? (both orders) | don't know | don't know | green | don't know |
| Has my favourite colour always been the same? | don't know | no | no | no |
| Has my favourite colour ever changed? | don't know | yes | yes | yes |
| What is my sister's favourite colour? (near miss) | don't know | don't know | don't know | don't know |
| What colour are my eyes? (untouched) | don't know | don't know | don't know | don't know |
| Name three primary colours / sky / capital of France | textbook | textbook | textbook | textbook |

Deleting lesson 3 must bring green back. Deleting lesson 1 must change nothing about now: the change and the retraction still hold. Only "before it changed" moves: without lesson 1, the last known value before now is green.

The full set is `examples/L2-seed1.json`; `python3 zengine/lwlogic.py show examples/L2-seed1.json` prints it.

## 7. Judging the model's answers

```sh
python3 zengine/lwlogic.py judge set.json answers.jsonl
```

The answers file has one JSON line per answer: `qid` (or `id`), `content`, `thinking` (true or false) and `phase`. You can also give the phase on the command line: `answers.jsonl:del-A`.

**Reading an answer.** The judge strips markdown and looks at the first sentence first, then widens to three sentences, then the whole answer.

- **Values**: it finds every value word of the attribute, with synonyms (grey and gray, apple and apples, "you" for me). It also knows values outside the test's list, so "hazel" or "navy" are noticed.
- **Negation**: a value after "not", "n't", "no longer" or "never", or followed by "anymore", does not count.
- **Tense**: when an answer names two values ("were brown until 11 October and are now green"), it takes the one after "now", "are" or "is" for CURRENT, and the one after "were", "was" or "used to" for past questions.
- **Refusals**: "I don't know", "I can't see", "I'm an AI", "you haven't told me", "only you can answer" and similar mean "doesn't know". On an either-or question, an answer that describes both options in general terms ("Purple is often associated with royalty...") and refuses later still means "doesn't know", as long as neither option is tied to the user, picked or guessed. A refusal in the first sentence wins over a value named later as general information ("Common eye colours include brown..."). The exception is a value tied to memory ("you said", "earlier", "based on").
- **Guesses**: a value inside "maybe", "perhaps", "probably", "likely", "might be", "could be", "I guess" in the first three sentences is a guess, not a recall, unless it is tied to memory. The right answer said as a guess ("I'm not sure, maybe green" when the answer is green) is unclear, for a person to look at. A wrong guess fails, and so does any guess where the right answer is "don't know": "I don't know. It might be brown." A list of three or more values counts as general information, not a guess. "I think" is not a guess: models often put it in front of a correct recall.
- **Off topic**: an answer that opens by talking about the model ("My name is Nemotron.") and then neither names a value nor says it does not know fails. So does an answer that never touches the topic.
- **Yes/no**: read from the start of the answer ("Yes, you do", "No, ..."). A refusal or "it depends" in the first sentence means "doesn't know", so "No, I am an AI and cannot know" is not a no. Neither is a yes or no followed by the model talking about itself: "No, I am a language model developed by researchers from NVIDIA." and "No, I don't have eyes." mean "doesn't know". "Probably not" and "Maybe, yes" are guesses that lean one way, and so are answers that open with how likely it is: "It is very unlikely that your favourite colour has always been the same.", "Highly unlikely.", "Not likely." lean no, "It is very likely that it has!" leans yes (until 13 Oct the opening "It is" read them as a plain yes). "It is likely that your eyes are brown." on "Are my eyes green?" leans no; tied to what the user said ("It is very unlikely, since you told me it changed.") it is a plain answer. An opening "true" or "false" is read as in the true/false form: "It is false.", "That is false.", "False." and "Incorrect." are a no, "That's true." and "True." a yes, "Probably false." a guess (until 13 Oct "It is false." read as a yes and "False." as no answer). With no yes or no word, the judge reads the value: "Are my eyes green?" answered "Your eyes are brown." is a no.
- **Whose value** (13 Oct): on a question about one entity, an answer that gives the value to another of the set's entities does not answer it. "Your cat's eyes are grey." for "What is my dog's eye colour?" fails, although grey is right, and so do "Nora has grey eyes." (Nora is the cat), "Your eyes are grey." and "Your cat is called Bruno." for "Is my dog's name Nora?". The judge reads whose each value is from the clause that states it: the entity before its predicate or its "'s", a subject that waits for its predicate ("Your cat, Nora, has grey eyes."), the clause before when this one has no subject, the verb of a list ("Your dog likes apples and pears."). The user holds a value as "your eyes", "your favourite colour", "you have", "you like", not as "you have a dog with grey eyes". The value said also with no holder ("Grey. Your cat's eyes are grey too.") or after a yes ("Yes, your dog likes olives." for "Do I like olives?") is unclear. An answer that gives a value to the asked entity ("Your dog's eyes are grey, like your cat's.", "Your dog's are grey too."), also by repeating the predicate for it ("and so does your dog", "..., and so are yours.", "You do too."), or that names no entity of the set ("They are grey."), is judged as before, and so is a no that says whose the value is ("No, your cat is called Nora."). Naming the asked entity is not enough: "Your cat has grey eyes. Your dog is asleep." fails, and so does "... but your dog does not.". It never makes a verdict better. Reverse questions are read as before.
- **First person** (13 Oct): the lessons are in the user's words, in the first person, and a model sometimes says one back. "I do not like olives anymore." is read as about the user, as "You do not like olives anymore." is: on "Is it true that I like chocolate?" it is a no, not a "doesn't know". So are "I don't." and "I do.". "My eyes are green." is green. The model talking about itself is a self-reply, as "No, I am a language model ..." is, and means "doesn't know": when it says so ("Personally, ...", "as an AI", "I don't have eyes") or asks the user back ("I love olives! Do you like them too?", "My favourite colour is blue! What's yours?"). First-person openings that are the model's own act ("I think you ...", "I remember that you ...", "I'd say ...") are read as before, and so are refusals and answers tied to what the user said. When the answer also says something of the user for the same attribute, the "I" is the model itself: the first person is a self-reply, and what is said of the user decides. "I do not like olives anymore. You still do." is a yes, and so are "I don't like olives. You do." and "I don't like olives, but you love them."; "I do not like olives anymore, and neither do you." is a no; "My favourite colour is green. Yours is blue." is blue. A clause that only repeats the predicate ("You still do.", "and neither do you") stands for the first sentence's statement said of the user, or, with no value there ("I don't. You do."), for the yes/no question's own. "Do you?", "You told me so." and "You are right to ask." say nothing of the user, so the first person is read as above. Questions about the user only. Plain questions only: a fill-in (section 13) is finished in the user's own voice.
- **Who has**: the judge reads every statement, clause by clause, for the set's entities ("your sister", "your cat", "you") and for names taught to them ("Bruno" is my dog). A clause that refuses ("I don't know whether your dog..."), negates ("not your cat"), gives another value ("your cat's are hazel"), sets a condition ("if you have grey eyes") or asks back ("Do you like olives?") names nobody. A name followed by a comma is read on to its value: in "Your cat, Nora, has hazel eyes, and your dog, Bruno, has grey eyes." only the dog has grey eyes. A negation after the value does not count ("Your dog has grey eyes that never change."). "You" counts only as an answer: "You do.", "You and your dog.", "you like olives", not "you told me", "you did tell me", "You have a dog named Bruno ..." or "Thank you!". An answer that names none of the set's entities and talks about people in general ("Here are some famous people with grey eyes: ...") is a general answer: it names nobody you know. One that names someone of yours the test never taught about ("Your brother has grey eyes.") is unclear, not general. A clause with no subject of its own goes on with the subject of the clause before it, after "but", "and", "while", "whereas" or "although": "Your dog does not like apples but likes olives." and "..., although it likes olives." name the dog. It does not when that clause negates the value ("but does not like olives"), asks ("but does it like olives?") or comes after an aside ("does not, however, like olives"), or when a clause between them has someone else as its subject ("and your brother does not like pears but likes olives"). Subjects joined by "and" or a comma share the predicate that follows them: "Your dog and your cat do not like olives." names nobody, "Your dog and your cat like olives." names both. "Neither", "unlike", "except", "other than" and similar words leave out the entity after them, as "not" does. A value in a list after "and" or a comma belongs to the subject of the verb before it, when that verb is not negated: "Your dog likes apples and olives." names the dog; "Your dog does not like apples and olives.", "... but not olives." and "..., not olives." do not. A later clause that takes the value back removes an earlier credit of its subject, in the same sentence or a later one: "Your dog and your cat like olives, but your dog no longer does." names only the cat. Only short forms take a value back ("no longer does", "doesn't anymore", "does not", "but not any more", "stopped liking them", "does not like olives anymore"); a clause that names another value, says more, talks about the past or asks takes nothing back. "It" is carried only within a sentence, so "Your dog likes olives. It no longer does." still names the dog.
- **Dates**: a date in an answer ("Until October 11, 2023, your eyes were brown.") is tied to the value named before its "until", "since" or "on" in the same clause, or after it when the date opens the sentence. When the date opens a later clause and values stand on both sides ("Your eyes are green now; until 11 October 2026 they were brown."), both readings are tried, and the claim is wrong only if both are. The judge checks that claim against the lessons. If no moment of the named day, month or year can make it true, the answer fails as a wrong date, whatever else it got right. "Until D" also says it stopped at D, and "since D" that it started there, as the lessons' own wordings do. A claim about a time the lessons say nothing about is not wrong, only unsupported. Not checked: the question's own date, a date of telling ("you told me on 11 October") or of the model ("as of my knowledge cutoff in 2023"), since the model is never told when a lesson was given, a negated value ("not brown since 2020"), and a clause about another attribute or another of the set's entities ("you have liked blue since 2020" on an eye question, "your sister's eyes were blue until 2025" on a question about you). A colour the test never uses ("red") is "no" wherever another colour is known. Answers without a date are judged as before.

**Verdicts:**

| Expected | Pass | Fail | Unclear |
| --- | --- | --- | --- |
| a value | names that value | another value, "don't know", a wrong guess, off topic, or the value said of another of the set's entities | several values with no tense cue; the right value as a guess; the value said both bare and of another entity |
| don't know / none | refuses, or says it is no longer so | names any value (an invented value), a guess, or off topic | no value, no refusal, but on topic |
| a set (likes) | names exactly the liked ones | misses one, or names a retracted or never-taught one, a wrong guess | the right set as a guess |
| yes / no / don't know | the same | the other, a wrong guess, or off topic | no yes, no and no value found; the right yes or no as a guess |
| who has | names exactly those entities; for nobody, also a refusal or a general answer | another entity, a missing one, a refusal or general answer when one was taught, a wrong guess | no entity, no refusal, nothing about the value; someone the test never taught about; the right entities as a guess |
| general knowledge | contains the textbook words | does not | |

Any of these also fails when the answer states a wrong date about the asked pair (see Dates above).

An empty answer is unclear. Unclear never counts as a pass. The report lists every fail and unclear with an excerpt, so a person can look.

Two-step questions (level 6) have one more verdict, **echo**: the answer names only a step on the way ("You live in Oslo." when the country was asked). See section 14.

**Scores**, per phase, thinking off and on separately:

- **recall**: CURRENT questions on taught pairs;
- **temporal**: AT, BEFORE, PREVIOUS, ALWAYS, CHANGED;
- **either_or**, **yes_no**, **reverse** (WHO_HAS);
- **controls**: near misses, untouched attributes, general knowledge;
- **delete_affected**: in a delete phase, questions whose right answer changed, or whose pair lost a lesson;
- **delete_kept**: in a delete phase, all other probe questions, which must not move;
- **consistency**: does the model's yes/no or either-or answer follow from its own answer to the plain question? This checks the model against itself, whatever the right answer is. 1.0.8's "Do I like blue? Yes, you said 'Now I like red.'" is exactly this failure. Since 13 Oct each answer is checked against every wording of the plain question, not only the first. The pairs are counted in two parts: the plain answer gave a value, or it said "don't know". A plain "don't know" makes "don't know" the consistent yes/no answer, so a model that refuses everything agrees with itself. Only the first part compares like for like between the baseline and the teach phase;
- **same answer** (paraphrase agreement, as in ParaRel): the wordings of one question, such as the three wordings of "What is my favourite colour?" or an either-or question in both orders, should give the same answer, right or wrong. Counted in pairs, split into pairs that agree on a value and pairs that agree on "don't know". **all wordings ok** counts the questions that are right in every wording. The baseline phase is the untaught model's reference;
- **yes/no groups**: every yes/no question on a pair, in both wordings, must be right together, so a model that says "no" to everything cannot score;
- **kept_stable**: in a delete phase, every question whose right answer did not change, compared with the same model's own answer in the teach phase: the same verdict, and the very same text. It measures stability, not correctness, so it is reported apart from the scores above. delete_kept counts a teach-phase mistake again as delete damage; kept_stable does not, since a wrong answer that stays wrong the same way is stable. It needs the teach answers in the same judge call, which the runner's combined judge has. The report lists every question that moved, and the `--json` file has a `kept_stable` section.

### The split view (13 Oct)

A pooled score mixes questions whose right answer is a value with questions whose right answer is "don't know". A model that refuses everything then passes whole delete phases. In our L3 run, all 11 affected answers of the first delete expected "don't know" (`docs/RESULTS.md`, finding 8). So every score is also split by what the rule book expects.

| Class | The right answer is |
| --- | --- |
| a value (v) | a value, a set of likes, yes, one of the two options, someone ("my dog"), or the textbook answer |
| no (n) | no, neither, or none ("taken back") |
| don't know (?) | "don't know", also "nobody known" on a reverse question |

The judge also reads what the model gave: a value, a no, "don't know", a guess, off topic or unclear. A refusal ("I don't know") is "don't know". An answer that says the value is gone ("You don't have one any more") is a no. As the judge does, it reads the first sentence, then three, then the whole answer, and the first part that refuses or says the value is gone decides. A reverse answer about people in general names nobody, so it counts as "don't know".

Each report shows, per phase and thinking mode:

- every score split by class, as `v 4/7 (3 dk)  n 1/1  ? 2/2`. (3 dk) counts the misses where the model said "don't know". On a value question that is always a miss;
- **should change** and **should stay**: the questions whose right answer differs from the phase before, and the others. The phase before is the baseline for the teach phase, and the teach phase for a delete phase. should change is part of delete_affected, which also holds the questions whose pair lost a lesson;
- **abstention**: "don't know" as precision and recall, as AbstentionBench scores it. Precision: of the "don't know" answers, how many were right to say it. Recall: of the questions whose right answer is "don't know", how many got it. The report ends with both over all phases, per thinking mode;
- **closed world**: the fails that would pass if a "no" counted where the right answer is "don't know". The test suite reads the world as open (section 1, "unknown is not no"). The count shows how much that choice matters;
- the literature's names for each question (below), and the **neighbourhood failures**.

No verdict changes. A score that penalizes wrong answers (Kalai et al. 2025) is not built. If one is wanted, it must be computed inside each class, never pooled, or it rewards refusing.

**The dummy systems.** Two systems answer every question the same way: "always don't know" ("I don't know.") and "always yes" ("Yes."). They are the negative controls of the split view. `python3 zengine/lwlogic.py dummy SET --says dont_know --out FILE` writes one as an answer file, and `judge` scores it like any other. On our example sets (MEASURED, `docs/rejudge/split/SPLIT.md`):

- "always don't know" passes all 11 affected answers of L3's first delete, all 7 of L1's, and all 9 of the #678 set's "delete everything". Split, it passes no value question in any phase: 0 of 7 in L3's first delete, and every miss is a "don't know".
- "always yes" passes only questions whose right answer is yes. It passes no "no" and no "don't know" question.

One gap stays. On a "none" question ("I have no favourite colour anymore."), the judge passes "I don't know." as it passes "You have none." It cannot tell them apart. The split view counts such an answer as "don't know" given on a "no" question, so it lowers the abstention precision. The example sets have no such question; later level 5 sets can.

**The literature's names.** Editing papers (ROME's CounterFact, KnowEdit, GRACE, WISE, MEMOIR) score an edit by four names. The test suite gives each question one of them, from the set alone (`question_tags` in `lwlogic.py`):

| Name | Test-suite questions |
| --- | --- |
| reliability | the plain question (CURRENT) about a taught entity and attribute, in its first wording. We never ask the lesson sentence back, so this is the nearest to "the taught wording" |
| generality | every other question about a taught pair: other wordings, yes/no, either-or, past and "always" |
| portability | reverse questions ("Who has grey eyes?") and indirect ones ("If you were buying me a shirt ..."). Two-step and alias questions come with level 6 |
| locality (neighbourhood) | the controls (near misses, untouched attributes, general knowledge) and pairs no lesson is about |

**Neighbourhood failures.** A failed answer that gives a value taught about another of the set's entities in that phase is tagged as a neighbourhood failure, as CounterFact names it. "Is my cat called Bruno or Nora?" answered Bruno, the dog's name, is one. So is "Yes." to "Is my dog's name Nora?" when Nora is the cat. It is not one when nothing is taught yet, when the value is right for both, or when the entity's own lessons also name the value ("Do I like olives?" answered yes after "I do not like olives anymore." is its own stale value). The report lists them per phase.

**Tried on real answers** (no new model run: the judge read our existing 1.0.7 removal answers, kept by the authors; full report in `examples/eyes-678.judged-1.0.7.txt`). It agrees with what we wrote in #678 by hand:

- **Fact 4 deleted:** the five eye questions keep the same right answers (fact 5 still holds). The model got 0 of 5 right, thinking off and on. The judge flags "Your eyes are hazel." and "My name is Nemotron." (off topic). "No, your eye color has likely changed over time" is the right no, but said as a general guess about babies' eyes, so it is unclear. kept_stable shows the break directly: 7 answers that passed in the teach phase moved, 3 with thinking off and 4 with thinking on.
- **Fact 5 deleted:** it flags "Your eyes were green." for "before 11 October" as an invented value. With only facts 3 and 4, nothing is known before 11 October.
- **All four deleted:** every expected answer right, both modes, the same as the original model.
- **Controls:** all general-knowledge and near-miss answers right in every phase.

It also finds two things we had not called out:

- **Fact 3 deleted is not clean.** "What is my eye colour?" goes to "I don't know" in both modes, but facts 4 and 5 still say green. We called this removal fine in #678; by the rules it is a fail.
- **The taught folder contradicts itself.** "Are my eyes brown or green?" says brown while "What colour are my eyes?" says green (thinking off). "Do I like blue or green?" with thinking on says "I don't know" while "What colour do I like?" says blue. 3 of 6 such pairs disagree.

**Limits of the judge.** It reads words, not meaning. Indirect questions ("If you were buying me a shirt...") are scored only where a value is known. Generic answers without a guess word can still slip through, which is why generated sets ask both wordings (ALWAYS and CHANGED) and score them together. Dates are read only in the common forms ("11 October 2026", "October 11, 2026", "2026-10-11", "October 2026", and a year after "in", "since" or "until"). A dated clause that uses another attribute's word is skipped, even when it means the asked one: "You have liked blue since 2020." is not checked on a favourite-colour question, because "like" belongs to the likes. A reverse answer that mentions "your friends" in passing is unclear rather than general. A clause after "likely that" that turns the question around is not read: "It is likely that it has not changed." on "Has my favourite colour ever changed?" leans yes. Whose a value is comes from the clause's own words: "The eyes of your cat are grey." names no holder, so it is judged on the value alone, as before. A first-person answer with no sign of the model talking about itself is taken as the user's lesson said back; one that is the model's own view without saying so ("I like blue.") is read as about the user too. Unclear answers need a person.

A gap closed on 13 Oct: "No, I am a language model developed by researchers from NVIDIA." used to be read as a "no", and in one 1.0.8 run it passed as the right answer to "Have my eyes always been the same colour?" It now means "doesn't know". Every verdict the fixes changed in our stored runs is listed in `docs/rejudge/CHANGES.md`. Four more gaps closed later on 13 Oct (branch judge-misreads): the likelihood and true/false openings of yes/no answers, whose value an answer names, and the first person (the rules above). Judged again, 15 of 2,190 stored answers changed verdict, all by the first: "It is very unlikely ..." and "It is very likely ..." are now guesses (`docs/rejudge/misreads/CHANGES.md`).

## 8. Running it against TensorFold

`zengine/lwrun.py` runs one set end to end against a real TensorFold server. Standard library only, macOS and Linux.

**Teach.** Start a fresh copy with `--slide`. Send each lesson as its own `POST /v1/slide/learn` request with `{"text": lesson}`, in order. Wait for `learned`, and for the save that follows, before the next one. Record `recalled` and the fact ids the slide graph gives it. A lesson can become one fact or two; a dated lesson may split.

The runner also keeps the server's own log line after each save, in the lesson's `save_log`. In 1.0.8 it reads like this:

> info: slide pocket: the learned neurons are in the model's shards (261 MB written in 18.4 s); the last fact's held-out answers do not all come back with them as written

`save_log` holds the line, the MB and seconds, and `held_out_back`: whether the fact's held-out answers came back with the neurons as saved. That is the server's own check of the saved fact, and it is not the same as `recalled`. Any other ending leaves `held_out_back` out, so a new wording is not read as a pass. A save that writes 0 bytes logs no such line, so its `save_log` is empty. In our runs that happened only when the first lesson was missed. A missed lesson after earlier facts still wrote the shards and logged the line: 9 of the 14 missed lessons in our GB10 runs, and the other 5 were first lessons (MEASURED). The runner waits up to `--save-line-wait` seconds (15) for the line after a save that wrote bytes, and warns when none came.

**Refused lessons.** A lesson with `"recalled": false` is still in the user's store, so it still counts against the model. To see whether a miss came from a refusal or from learning wrongly, add a phase whose `deleted` lists the refused lessons and judge against that too.

**Delete.** For each delete variant, start from a copy of the fully taught folder and call `DELETE /v1/slide/fact/{id}` for every fact that came from the variant's lessons. That endpoint is **not released yet**: [Ash Hart says](https://github.com/ashhart/TensorFold/issues/678#issuecomment-6124177090) it comes in the next release. Until then the runner has two stand-ins (`--delete-mode zero|reteach|both`):

- **zero**: [philip-pentatonic's zeroing method](https://github.com/ashhart/TensorFold/issues/678#issuecomment-6111596537). The deleted facts' learned neurons are set to zero in a copy of the taught folder.
- **reteach**: teach only the remaining lessons to a fresh copy. This is the "never taught" reference.

Running both and comparing them tests "delete = never taught" directly.

**An empty store** (every lesson deleted) is the original model, which the baseline step has already asked. So the runner reuses the baseline answers for it, filed under the delete phase, and loads no model. In our stored runs, the 3 empty-store steps that served the original model again gave the baseline answers word for word: 78 of 78 answers, on the Mac and the GB10 (MEASURED). `--fresh-empty-reteach` serves the original model again. The runner also does that, and says why, when the baseline answers are missing or incomplete, or were asked with another binary or other `max_tokens`.

**Limits of the zero stand-in.** These are stated design limits, not delete damage. They follow from how TensorFold 1.0.8 stores facts, and they belong to the stand-in, not to TensorFold's coming delete endpoint (section 9, decision 1).

- **Zeroing cannot bring back a fact that the deleted fact had replaced.** When a newer fact replaces an older one, 1.0.8 sets the older fact's neurons to zero (READ: [Ash Hart's #678 comment](https://github.com/ashhart/TensorFold/issues/678#issuecomment-6124177090); MEASURED: those rows were already zero in our runs). Zeroing the newer fact then leaves neither. Never teaching the newer fact keeps the older one, and that is what the rule book expects. So in such a phase zero and reteach may differ, and only reteach can match the rule book. Example: in the #678 set, deleting the dated eye lesson by zeroing cannot bring back "My eyes are green.", which it had replaced (`docs/RESULTS.md`, finding 9). This is not a limit when a kept fact taught later replaced the deleted one in turn. Never teaching the deleted fact, that kept fact replaces the older one itself, so the older one is gone both ways. In the #678 set, "My eyes are green." replaced "My eyes are brown.", and the dated lesson replaced it. Deleting "green" therefore does not bring "brown" back: the reteach without "green" had the dated lesson replace "brown" (MEASURED, one run on the GB10).
- **Nothing is left to zero when the deleted fact was itself replaced or missed.** Its rows are already zero, or it never had any, so the zero copy answers like the taught model. In our runs of 12 October, 4 of the 13 zero steps changed no bytes (MEASURED, `docs/RESULTS.md`).
- **Later facts were learned with the deleted fact in place.** Zeroing removes the deleted fact's own neurons. It does not learn the later facts again without it. On 1.0.7, deleting a fact this way broke the dated fact taught after it ([#678](https://github.com/ashhart/TensorFold/issues/678), MEASURED on the Mac). Our reading is that the later fact leaned on the zeroed neurons (INFERRED). Reteach has no such history.

The runner marks the first two in each zero step's `zero_limits`: `cannot_restore` lists the kept facts a deleted fact had replaced, unless a kept fact replaced it in turn (read from the server's own "replaced by fact N" notes in the teach records). `already_zero` lists the deleted facts with nothing left to zero. The zero-vs-reteach comparison in `summary.json` repeats them for the phases they touch, so a difference there is not read as delete damage. `--judge-only` works them out again from the teach records, so older run records get them too.

**Nothing stored at all: not applicable.** When every lesson of the teach step was missed and no earlier fact was there, every save wrote 0 bytes and the taught folder holds no pocket: its `config.json` has no `tensorfold_learned_neurons`. Zeroing then has nothing to remove, and a zero copy would be the taught folder byte for byte. Since 13 Oct the runner marks such a zero step `not applicable`. The reason is in the step's `not_applicable` field in `summary.json`. No copy is made and nothing is served. The step counts as finished, so the run's exit code is 0, and `--cleanup` removes the taught copy as it does after a finished run. Before, the step failed as a layout mismatch ("the folder holds no pocket"). `--cleanup` then kept the taught copy, because a zero step was not done, and the zero copy too. That left two copies of 18.5 GB on disk at the end of each of 9 runs on the GB10 with `--cleanup` (MEASURED, their `run.log` and `copies.json`, 13 Oct). Other cases are unchanged:

- A folder with no pocket while the teach records say a fact was stored is still a mismatch. The step fails before any copy is made.
- A zero step skipped for its layout (a pocket that is not 64 neurons, say) still fails. It no longer leaves its copy behind: nothing was written to it, and a resumed step copies the taught folder again, so `--cleanup` removes it at once. The taught copy stays for the resume.
- A zero step that failed after zeroing (its server died, say) keeps both copies. `--resume` reuses the zeroed copy and finishes, and `--cleanup` then removes both.

On `--resume`, a not-applicable step stays so while the teach records still say nothing was stored. `--resume --cleanup` on a run folder the older runner left removes both of its copies. All of this is checked on the mock server only (the self-test below), never on a GB10 run folder.

**Writes in place.** Zero mode writes the copy's shards in place, so it must never write a file that shares its bytes with another one. On gb10-a, leftover run copies had been deduplicated into hard links to the Hugging Face source, so zeroing one of them would have silently changed the source model. Since 13 Oct every write in place (zero mode, the night clean-up's write-backs, and the mock servers) goes through `open_for_inplace_write` in `zengine/lwrun.py`. It refuses a symlink or a file with `st_nlink` above 1, and names the path, the link count and the reason. `zero_facts` checks every shard it will write before the first byte, so a refused step fails with nothing written and no copy half zeroed. The guard looks at the file itself, not at the folders above it. The runner self-test checks a hard-linked and a symlinked shard: both are refused, with both files unchanged. Controls: with the guard switched off the write goes through and changes the source, and a real copy is zeroed. With these 5 checks the runner self-test passes 40 of 40 with Python 3.12 and 3.14, 35 of 35 before; with the guard made to do nothing, the 3 refusal checks fail (MEASURED, 13 Oct 2026, on the mock only).

**Probe.** Use `POST /v1/chat/completions`, one fresh chat per question, `temperature: 0`, thinking off (`max_tokens` 160) and on (1500). The runner writes one JSON line per answer with `qid`, `phase`, `thinking`, `max_tokens` and `content`, then runs `judge`.

**Kept-stable inputs.** The `delete_kept` score compares the questions a delete should not touch with the rule book's ideal answer, so it also counts what the model already got wrong after teaching. The judge's kept_stable score (section 7) compares them with the same model's own teach answers instead, verdict and text. For it, the runner puts each delete mode's teach answers and delete answers side by side, with names that never change:

```
WORK/kept-stable/MODE/teach.jsonl      the teach answers this mode is compared with
WORK/kept-stable/MODE/PHASE.jsonl      one file per delete phase, for example del-A.jsonl
WORK/kept-stable/manifest.json         where each file came from, how it was served, whether it is complete
```

`MODE` is `zero`, `reteach`, or `reteach.restart` (with `--teach-probe both`). The files are the runner's answer files, copied line for line, so every line keeps its own `phase`. One judge call then sees one model's teach and delete answers together:

```sh
python3 zengine/lwlogic.py judge WORK/set.json WORK/kept-stable/zero/*.jsonl
```

For zero, the teach answers come from a restarted server without `--slide` when there are some (`--teach-probe restart` or `both`), because zero copies are served that way. Otherwise they come from the `--slide` server that taught, and the manifest says so. When the judge's JSON holds a `kept_stable` score (a top-level key, filled when teach and delete answers are judged in one call), the runner copies it into `summary.json`, under `combined`.

**Several sets.** `--sets A.json B.json ...` runs the sets one after another with the same options, each in its own `WORK/SET_ID`. This is what our shell scripts around the runner did. Every set is loaded and checked before the first one starts, so a bad file, a set given twice, a set id that cannot name a folder or a leftover work folder stops the batch at once, not hours in. A set that fails does not stop the next one; an interrupt does. At the end, `WORK/sets-summary.json` holds each set's exit code, steps and scores, and `WORK/sets.log` ends with a table.

### Reference phases (opt-in)

A low score alone does not say why. The pocket may not have stored the fact, or the model may not answer that kind of question even when it is told. Two reference phases separate the two (plan item 5 in `docs/LITERATURE.md`):

- **in_context**: the untaught model, with the phase's lessons written into the system prompt, asked the same questions.
- **replacement_memory**: no model at all. A lookup that keeps the latest value of each fact and answers from it.

They are opt-in. Without `--reference` the runner does exactly what it did before.

```sh
python3 zengine/lwrun.py set.json --bin PATH/TO/tensorfold-native --model-src PATH/TO/MODEL --work runs/set --port 18180 --reference in_context,replacement_memory
```

`--reference` takes `in_context`, `replacement_memory`, or `both`. `--reference-of` names the phases they stand beside: `teach` (the default), a comma list of the set's phases, or `all` (every phase that keeps at least one lesson). `--reference-of` needs `--reference`: given alone, or naming no phase, it is refused before anything runs, as are an unknown kind, an unknown phase and a phase with no lessons. The reference steps run after the taught model's own steps, whatever `--phases` picks. Their answers carry the phase's own name, so the judge scores them against that phase's right answers. They go to `WORK/phases/PHASE.in_context/` and `WORK/phases/PHASE.replacement_memory/`. The flags are kept in `summary.json`, under `args` and under `reference.asked`. `--dry-run` prints the prompt and the lookup's contents. `--judge-only` also judges reference steps it finds on file, with or without the flag.

**Neither is a ceiling.** Both are references. PersonaMem (2025) reports frontier models at about 50% overall at tracking a user's traits and preferences as they change, with the facts in the conversation (READ: its abstract, as summarised in `docs/LITERATURE.md`). So the model told in context can get questions wrong that the pocket gets right. The gap shows where the pocket is behind the same model told the same lessons. It does not show how far the pocket is from perfect.

**The in_context prompt** (format `lw-in-context/1`). One fixed line, then the phase's lessons in teaching order, numbered, each in double quotes, word for word as taught. Then one fixed line that says whose "I" they are. For the teach phase of `examples/L2-seed1.json` the system message is exactly:

```
Earlier, the user told you the following, one message at a time, oldest first:
1. "Of all colours, I like white the most."
2. "I like green best now."
3. "Green is not my favourite colour anymore."
In these messages, "I", "me" and "my" mean the user.
```

Nothing else goes in:

- no date of telling, since the taught model is never told one either (section 9, decision 3);
- no rule of the rule book, such as "a later fact replaces an earlier one";
- no request to say "I don't know".

A lesson the server reported as not recalled stays in the prompt. It is still in the user's store (see Refused lessons above). The model is served like the baseline: the source folder read-only, or a fresh copy with `--baseline-from copy`. Each question is a fresh chat with the system message first and the question second, at temperature 0, thinking off and on. Each answer line keeps the sha256 of the prompt. On `--resume`, answers asked with another prompt are moved aside and asked again, never mixed. Changing the wording needs a new format name.

Does TensorFold give the system message to the model? READ in the TensorFold source at tag v1.0.8 (`zig/src/server/messages.zig`, `normalize`): the chat endpoint merges leading system messages into one system turn at the start, and keeps a later one as a system turn where it stands. The runner sends the system message first. How Nemotron's chat template renders that turn, and whether the model uses it, is UNCHECKED. The first real run should look at a few in_context answers by hand before trusting the gap.

**The replacement memory** (format `lw-replacement-memory/1`). One slot per (entity, attribute). The store's lessons are replayed in teaching order:

| Lesson | The slot |
| --- | --- |
| ASSERT v (present tense, "since D", "always"), CHANGE v | v |
| DATED v0 until D, v1 since D | v1 |
| ASSERT v "until D" (or between two dates) | stays as it is: the lesson says nothing about now |
| RETRACT v | emptied if it holds v, else stays as it is |
| RETRACT with no value ("I have no favourite colour anymore.") | none |

A slot holds one value, also for likes. So after "I like pears." and "I really like apples." it holds apples only, as 1.0.8 did with olives and apples (`docs/RESULTS.md`, finding 10). It keeps no history, no "not v" and no general knowledge.

It answers only about now:

- CURRENT gets the value in the attribute's first wording, said to the user: "Your favourite colour is green.", "Your dog's name is Bruno.", "You have no favourite colour anymore."
- YES_NO gets "Yes." or "No.".
- EITHER_OR gets the value, or "Neither.".
- WHO_HAS gets the entities whose slot holds the value ("You and your dog."), or "No one you told me about.".
- Every other question, and any key without a slot, gets "I don't know.".

The answers are computed, with no server. They are written like a probe's answers, once per thinking mode so the tables line up, and judged as usual.

What it scores on the eight example sets, teach phase (computed and judged, MEASURED 13 Oct 2026):

| Group | Passed |
| --- | --- |
| CURRENT | 30 of 32. The two misses are "Which foods do I like?" in two wordings: likes holds two foods there |
| WHO_HAS | 3 of 3 |
| negation | 24 of 42. It has no "not v": "Is my favourite colour green?" after "not green anymore" gets "I don't know." |
| dates | 0 of 10 |
| yes_no | 18 of 24. On likes it says "No." where the right answer is "don't know" or "yes" |
| either_or | 16 of 16 |
| temporal | 0 of 31. It keeps no history |
| controls | 17 of 41. It says "I don't know." to general-knowledge questions |
| all | 84 of 147 |

The runner's self-test checks this table, through the report's own grouping and the real judge. So it is a sanity reference. It shows what keeping only the latest value is worth, question by question.

**The report.** `WORK/reference/report.txt`, and `reference` in `summary.json`:

- `scores`: for each answer source of the phase, the passed and asked counts per group and thinking mode.
- `gap`: for each of the taught model's answer files of the phase, per group and thinking mode. For teach these are `teach` and, with `--teach-probe restart` or `both`, `teach.restart`. For a delete phase they are its `zero`, `reteach` and `reteach.restart` answers. Each cell holds the counts of that file, of in_context and of replacement_memory, over the answers they all have. It also holds the **pocket gap**: the in_context pass rate minus the pocket's. A positive gap means the pocket is behind the model told in context.
- `lessons` and `outcomes`: each CHANGE and RETRACT lesson, as below.

The groups:

| Group | Questions |
| --- | --- |
| CURRENT | the value now (the judge's recall group) |
| WHO_HAS | reverse questions (the judge's reverse group) |
| negation | every question on a key that a RETRACT lesson in the phase's store touches. For WHO_HAS: a retraction of its value, or of every value, on its attribute |
| dates | questions that name a date (AT, BEFORE) |
| two_step | level 6's two-step questions: type CHAIN (from a start through `hops` to a final attribute) or COMPOUND (two yes/no parts joined by and, or, not). the same two types as `lwlogic.TWO_STEP`; the three `examples/L6-seed*.json` sets have 24 each (MEASURED after the level 6 merge, 13 Oct 2026) |
| compound | the judge's own group for COMPOUND questions (level 6), when present |
| temporal, either_or, yes_no, controls, delete_affected, delete_kept | the judge's own groups (section 7) |
| all | every answer |

**Corrected, stale, both or neither.** The split is plan item 5's, after Beyond Perplexity (2026), which asks for evidence from behaviour against matched in-context baselines. For each CHANGE and RETRACT lesson of the phase, and for each answer source and thinking mode, the report says what the answers on the lesson's key showed. It reads the CURRENT, YES_NO and EITHER_OR answers on that key, with the judge's own readers.

- **Stale values.** For CHANGE v: the values that held just before it, other than v. For RETRACT v: v, also when v did not hold, so an answer that names it is "learned backwards". For a RETRACT with no value: every value that held just before.
- **What holds just after it.** The rule book's state of the key right after the lesson (`now` in `lessons`). For CHANGE v it is v. For a RETRACT it is what is left: often nothing, the other foods on likes, or the value that still holds when a RETRACT takes back a value that did not hold.
- **Classified only when today's answers can show it.** A lesson is left out, with the reason, when a later lesson in the store writes the same key again. On likes, only a later CHANGE, or a lesson about v or a stale value, counts. It is also left out when no other value held just before it (a CHANGE, or a RETRACT with no value), so nothing can be stale. In L2 set 1, "I like green best now." is left out in teach, because "Green is not my favourite colour anymore." comes after it. It is classified in del-A, where that lesson is deleted.

| Answer | Shows |
| --- | --- |
| CURRENT names a stale value | stale |
| CURRENT names a value that holds just after the lesson | corrected |
| CURRENT says "don't know", none, or "not anymore", after a RETRACT that leaves nothing | corrected |
| YES_NO: yes to a stale value | stale |
| YES_NO: yes to a value that holds just after the lesson | corrected |
| YES_NO: no to the stale value (RETRACT) | corrected |
| EITHER_OR picks a stale value | stale |
| EITHER_OR picks a value that holds just after the lesson | corrected |

After a CHANGE, "no" to the old value does not show the new one, so it counts for nothing. A guess counts for nothing either. After a RETRACT that leaves nothing, "I don't know" is the right answer, so it counts as corrected. After "Of all colours, I like white the most." and "Green is not my favourite colour anymore.", white still holds: "Your favourite colour is white." is corrected, and "I don't know." is neither. On likes, after "I do not like olives anymore." with pears and apples left, "You like pears and apples." is corrected and "You like cheese." is neither. On a single CURRENT answer, corrected only falls exactly where the judge passes it, in the self-test's cases. A model that refuses everything therefore looks corrected on retractions; compare with its baseline answers before reading a corrected only as learned. Corrected and stale together are **both**: for example "I don't know." to "What is my favourite colour?" with "Yes." to "Is my favourite colour green?". Neither side is **neither**. `outcome_counts` adds the outcomes up per source and thinking mode.

For the two negation sets this is the classification that plan item 1 asks for. "Learned backwards" or "no effect" is stale only, "learned" is corrected only.

**Runner self-test.** `python3 zengine/lwrun_selftest.py` runs the runner against a mock TensorFold server, in about two minutes, with no GPU. The mock keeps its facts in a tiny model folder laid out like 1.0.8's, so zero mode zeroes real bytes, and a newer fact that replaces an older one zeroes the older one's rows. It also honours a system prompt in the in_context wording. It then answers from the last lesson the prompt quotes, as it answers from its last live fact otherwise. Each feature above is checked on real runner runs. Each also has a negative control: a run where the feature is missing, switched off or given bad input, where the check must fail or the runner must refuse.

The 14 checks of the reference phases, each with its control:

- the prompt is the documented wording above, word for word. Controls: a store without L3, and the lessons in reverse order, give other prompts; an empty store is refused;
- the memory's slots on L2, L3, L4 and a hand-made store, as in the table above. Control: a memory where the earliest lesson wins gets 4 of the 5 stores wrong;
- on every example set and phase, the memory passes the judge on all 142 CURRENT, yes/no and either-or questions whose key's last lesson gives a value now. Control: the earliest-wins memory fails 43 of them;
- a full run with `--reference`: both steps, every answer under the phase's name, the flags in `summary.json`, the exact prompt as the first message of all 24 in_context chats and on no other chat, and no server for the memory. Control: the run without `--reference` has none of it;
- the gap: with the lesson missed by the pocket, CURRENT is 0 of 3 in the pocket and 3 of 3 in context and in the memory, a gap of +1.00 in both thinking modes. Control: a mock that drops the system message gives in_context 0 of 3, the memory still 3 of 3, and a gap of 0.00;
- outcomes on a run: the pocket that missed "Green is not my favourite colour anymore." is stale only, and in_context and the memory are corrected only. Control: with the lesson learned the pocket is corrected only;
- outcomes of 9 hand-written answer sets: one per outcome for a RETRACT and for a CHANGE, and none for a CHANGE that a later lesson overrides. Control: a reader deaf to negation gets 2 of them wrong;
- `--judge-only` without the flag judges the reference steps on file again (their verdict files, moved aside first, are written again) and rebuilds the report. Control: with those folders moved aside there is no report;
- on `--resume`, answers asked with another prompt are moved aside and asked again. Control: unchanged answers are kept, and nothing is asked;
- `--dry-run` shows both steps, the prompt and the slots, and writes nothing. Control: a dry run without the flag shows none of it;
- the memory's table above, on the eight example sets, through the report's grouping and the real judge. Controls: negation read from every taught key, and a memory that names nobody for "who has", give other tables;
- outcomes follow what holds just after the lesson, on 7 hand-written answer sets: on likes a retraction that leaves nothing, one that leaves two foods, and a retraction of a value that did not hold. On every single CURRENT answer, corrected only falls exactly where the judge passes. Control: the rule as first built gets 5 of the 7 wrong;
- bad flags are refused before anything runs: an unknown kind, an unknown phase, a phase with no lessons, `--reference-of` without `--reference`, and an empty `--reference-of`. Control: `--reference in_context --reference-of del-A` is accepted and plans `del-A.in_context` only;
- the two_step group holds level 6's CHAIN and COMPOUND questions, made by hand in level 6's shape, and no question of the eight example sets. Controls: the test as first built misses COMPOUND, and one that takes every question flags all 147 example questions.

**Planted bugs (run by hand on 13 Oct).** Ten single edits to a copy of the runner: no system prompt sent, a retraction that does not empty the slot, an "until" lesson that overwrites it, the prompt in reverse order, "no" to the old value counted as corrected after a CHANGE, the gap's sign flipped, the prompt check on `--resume` switched off, `--judge-only` not looking for reference steps on file, "in force" ignoring later lessons, and a memory that says yes to every value. At first the self-test failed on 8 of them. The `--judge-only` edit and the "in force" edit got through: the report still came from the old verdict files, and no check looked at a lesson that should not be classified. Those two checks were tightened, and now the self-test fails on all 10 (MEASURED). The edited copies are not in this repository.

**Review round (13 Oct, a second agent).** Fifteen more single edits, made without looking at the first ten. On the 28 checks then in place, 8 got through (MEASURED): the system message sent after the question, a memory that names nobody for "who has", no negation group, no dates group, no WHO_HAS group, likes read as one value, the gap read from the memory's column instead of in_context's, and a `--reference-of` list ignored. Two checks were tightened (the prompt must be the first message; the control of the gap must move with in_context, not the memory), four were added (the last four above), and the outcome rule was fixed to follow what holds just after the lesson. On the 32 checks, the self-test fails on all 20 edits of this round (MEASURED). The likes-as-one-value edit was left out of the 20: under the fixed rule it gives the same outcomes as the real reader on every case tried. The edited copies are not in this repository.

The 3 checks of `--cleanup` and the zero step (13 Oct), each with its control. `--cleanup` deletes only on Linux, so these runs import the runner with its Linux paths switched on; the mock servers still start as they do on the Mac:

- a teach step that stored nothing (the mock's missed first lesson saves 0 bytes and writes no pocket): the zero step is not applicable, the reason is in `summary.json`, no zero copy is made, the taught copy is removed, and the exit code is 0. Control: the zero step as first built fails this check, keeps both copies, and logs the same two lines as the GB10 runs;
- `--resume`: the step stays not applicable, with no server and no copy; this runner resumed on the first build's run folder removes both copies left there. Control: with the resume rule switched off, the resumed step fails, because the taught folder is gone;
- a zero step skipped for its layout (the mock writes a pocket of 32) removes its own copy and keeps the taught one; a zero step whose server dies at start keeps both, and `--resume` reuses the zeroed copy, finishes and removes both. Control: without the removal of the skipped step's copy, that copy stays.

Three single edits to a copy of the runner, made by hand on 13 Oct: the taught copy removed after a failed zero step too, a not-applicable step left as `incomplete`, and a copy made for a not-applicable step. The cleanup checks fail on all three, with Python 3.12 (MEASURED). The edited copies are not in this repository. With `--records`, one more check reads our stored runs: every stored zero step that failed with "the folder holds no pocket" has teach records that say nothing was stored. Control: no stored teach record with a learned fact says so.

Result (13 Oct 2026, MEASURED): with the cleanup checks, 35 of 35 checks pass (40 of 40 after the link guard was added the same day) with Python 3.12 and 3.14, in 117 to 119 seconds, and 39 of 39 with `--records` and our stored runs (Python 3.12; the new records check found 10 such zero steps, 9 of them from the GB10, and 17 teach records with a learned fact). Before them, 32 of 32 checks passed with Python 3.12 and 3.14, in 75 to 77 seconds, and again after the merges of 13 Oct with Python 3.12 and 3.14, in 78 to 79 seconds. Before the reference phases there were 18, in 46 seconds; the first build of the reference phases had 28.

```sh
python3 zengine/lwrun.py examples/L1-seed1.json --bin PATH/TO/tensorfold-native --model-src PATH/TO/SNAPSHOT --work runs/L1 --port 18180 --dry-run
```

`--dry-run` prints every step and checks the binary, the source folder, the port and the judge, and copies, serves and writes nothing. Drop it to run. Each model copy is about 18.5 GB. On macOS the runner never deletes its copies; on Linux `--cleanup` removes them once judged. It removes the taught copy once every zero step is done or not applicable, and a zero copy at once when its zeroing was skipped.

**Also recorded since 13 Oct** (for level 7, used by `capacity.py`): each lesson's `pocket_full`, true when its message is 1.0.8's `PocketFull` (below); each fact's teach `loss` from the graph; and `Server.delete_fact`, which calls `DELETE /v1/slide/fact/{id}` and reads a 404 as "no such endpoint". `lwrun.py` refuses a level 7 set: it would read the checkpoints as deletes.

```sh
python3 zengine/lwrun.py --sets examples/L1-seed1.json examples/L2-seed1.json --bin PATH/TO/tensorfold-native --model-src PATH/TO/SNAPSHOT --work runs/night --port 18180 --dry-run
```

**Repeated runs.** One run of one lesson tells what happened once. `tools/aggregate.py` reads several run folders of the same set and reports every result as "k of n":

- each lesson: recalled in k of n tries, over every teach and reteach step of every run;
- each step and thinking mode: every score (all, the buckets, and the split by the right answer) as k of n answers, and each run's own count;
- each question: passed in k of n runs, and each run's verdict.

An unclear answer is not a pass. Each figure has the exact (Clopper-Pearson) 95% interval for the rate. Per question and per lesson the tries are separate runs. Per phase they are not: one model copy answers all of a run's questions. So the per-phase interval is only a rough range. The tool reads the folders only, and judges every answer with the current judge. It refuses a run given twice (the same start time and host, or the same folder) and a run of another set. It leaves out the `answers.as-PHASE.jsonl` files, so no answer counts twice. Runs from different machines can be given together; the report names each run, so keep them apart when the machine matters.

```sh
python3 tools/aggregate.py mac-1=runs/L1-a gb10-1=runs/L1-b gb10-2=runs/L1-c --out L1-repeats.txt --json L1-repeats.json
```

On our three stored L1 runs, "Of all colours, I like blue the most." was recalled in 1 of 3 tries, with the interval [0.01, 0.91]: once on the Mac, missed twice on the GB10 (MEASURED). Its self-test, `python3 tools/aggregate.py --selftest`, runs on synthetic run folders: 3 checks, all PASS (MEASURED, 13 Oct 2026). Controls: counting an unclear answer as a pass, a run given twice, the same folder under two labels, a run of another set, the normal-approximation interval and the plain binomial sum are each caught. The interval is summed in logs: the plain sum stops with an overflow error from about 1,030 answers (a review on 13 Oct found it at 600 of 1,200), which a per-phase count over many runs can reach. Lessons are counted only from steps that finished, and only when the record says whether they were recalled.

The first runs on TensorFold 1.0.8 are in `docs/RESULTS.md`.

## 9. Decisions to confirm

1. **What delete means.** We chose "never taught": deleting "I like green" after "I like blue" brings blue back. TensorFold 1.0.8 zeroes the older fact when a newer one replaces it (READ: [Ash Hart's #678 comment](https://github.com/ashhart/TensorFold/issues/678#issuecomment-6124177090); MEASURED: in our 1.0.8 runs the replaced facts' rows were already zero, see `docs/RESULTS.md`), so deleting the newer one would leave nothing. Our broken 1.0.8-like system shows that design fails this rule. Worth asking Ash Hart which meaning `DELETE /v1/slide/fact/{id}` will have before writing tests against it.
2. **A plain contradiction is a change, not a correction.** "My eyes are brown." then "My eyes are green." means green from the moment it was said, with brown before it. A user who means "I was wrong all along" would need a separate `CORRECT` lesson that rewrites the past. We have not added it.
3. **Present-tense facts say nothing about past dates.** The model is never told when a lesson was given. So "What colour were my eyes before 11 October?" after only "My eyes are brown." is "don't know".
4. **Wordings carry hints.** "Now", "anymore", "since" and "until" also say what held on the other side, and a hint never overrides what was said outright. Without hints, "My favourite colour has changed; it is green now." alone would answer "Has it ever changed?" with "don't know", where any person says yes.
5. **Taking back a single value leaves "don't know, but not that"**, not "none".
6. **"I like blue." is a favourite colour** in our #678 case and in 1.0.8. The generator keeps colours and likes apart so new sets are not ambiguous.
7. **The right answer said as a guess is unclear, not fail** (decided 13 Oct 2026). "Your eyes are probably green." when they are green leaves the call to a person. A wrong guess still fails, and so does any guess where nothing is known.
8. **A reverse question answered about the world names nobody you know.** "Who has grey eyes?" answered with famous people passes when nothing was taught and fails when one of your people or pets was taught. The generated wording does not say "of the people I told you about", so the model may fairly read it as a general question; the verdict still asks whether it used what it was taught. An answer about someone of yours the test never taught about ("Your brother has grey eyes.") is not about the world, so it is unclear. One change on main (12 Oct) failed a general answer in the baseline too. The merge on 13 Oct kept the rule above. **Decided by petrus on 13 Oct 2026 (01:39 CEST):** the general answer is right when nothing was taught ("a whole essay better than saying do not know anyone"). After teaching, the answer should be "a whole with some prioritisation for learned": it passes when it names the taught one, with or without general information around it, and fails when it gives only general information. A separate score for whether the taught one is named first is planned; it will not change pass or fail.
9. **Level 6 readings** (section 14). A linking fact ("My cat's eyes are the same colour as mine.") is read one way, one link deep, as a standing rule, and only fills gaps. A name means the current name. "Or" is inclusive. Each is a choice a person could make differently; the generator avoids the questions where the choices would differ.
10. **The in_context reference gets the lessons and nothing else** (section 8). No date of telling, no rule of the rule book, no request to say "I don't know". A prompt with rules would score higher, but it would no longer be the same model told the same lessons. The replacement memory is a computed lookup, as built on 13 Oct. Plan item 5 first had the latest values written into the model's prompt instead; that variant is not built.

## 10. Review notes addressed

From our teammates' reviews of the first draft:

1. **Order vs valid time:** separate. Present tense holds from when it is told; "always" is its own lesson form (A5 in the Lean header, section 1 here).
2. **Delete replays the remaining lesson ids** (`effective`), so independent history survives.
3. **Deleting a retraction restores the earlier fact.** It is a golden case. There is no retract-a-retraction; saying it again is a new ASSERT.
4. **Unknown is not no.** Truth has three values; YES_NO and EITHER_OR answer "don't know" when nothing decides them.
5. **Lean's assumptions are listed** in the file header (A1 to A8).
6. **A reviewer's outline is the data format:** facts in teaching order, queries, removal sets, and a checker that never needs the model.

## 11. Commands

```sh
python3 zengine/lwlogic.py selftest
```

```sh
python3 zengine/lwlogic.py gen --level 4 --seed 3 --out /tmp/L4s3.json
```

```sh
python3 zengine/lwlogic.py show /tmp/L4s3.json
```

```sh
python3 zengine/lwlogic.py gen --level 4 --seed 3 --forms v2 --out /tmp/L4s3-v2.json
```

```sh
python3 zengine/lwlogic.py gen --from examples/neg-seed1.json --forms v2 --out /tmp/neg-v2.json
```

```sh
python3 zengine/lwlogic.py judge /tmp/L4s3.json answers.jsonl --json verdicts.json
```

```sh
python3 zengine/lwlogic.py dummy /tmp/L4s3.json --says dont_know --out /tmp/L4s3.always-dont-know.jsonl
```

```sh
python3 tools/aggregate.py runs/L1-a runs/L1-b runs/L1-c
```

```sh
python3 tools/aggregate.py --selftest
```

```sh
python3 zengine/lwlogic.py lean /tmp/L4s3.json --out /tmp/L4s3.check.lean
```

```sh
python3 zengine/lwlogic.py leancheck /tmp/L4s3.json
```

```sh
python3 zengine/lwlogic.py intake --store /tmp/L4s3.json --text "My dog's eyes are grey."
```

```sh
python3 zengine/lwrun_selftest.py
```

```sh
python3 zengine/prune.py selftest
```

`gen --level 6` writes a level 6 set; `python3 zengine/levels.py selftest` runs the level 6 checks alone. `gen --level eyes` writes the #678 case. `selftest --probes DIR` also grades the real 1.0.7 removal answers of the #678 trial, when you have that folder of answer files (they are not in this repository).

## 12. The intake check

Before a fact is learned, the rule book can sort it. `zengine/intake.py` does this. Each new lesson gets one of four verdicts. Only the first two need learning.

| Verdict | Meaning | Learn it? |
| --- | --- | --- |
| NEW | Some answer changes, and it is not REPLACES | yes |
| REPLACES | The attribute holds one value at a time, its current value was known (a value, or "none"), and the lesson changes it | yes |
| ALREADY_KNOWN | No answer changes, and an earlier lesson says the same thing | no |
| IMPLIED | No answer changes, but no earlier lesson says the same thing: it follows from them | no |

**How it decides.** The rule book answers every question about the lesson's entity and attribute twice: from the store as it is, and from the store with the lesson added. If any answer differs, the lesson is news. The questions are:

- the current value, the value before it changed, "always the same?" and "ever changed?";
- yes/no for every value, and either-or for every pair of values;
- the value on, just before, the day before and the day after every date that the lessons on this pair mention, or were told on. Answers can only change on those days, so this covers every date;
- "who has" for the lesson's values.

For my eye colour that is 45 questions or more.

**"Says the same".** Two lessons say the same when they write the same thing, leaving out the hints (section 1), and counting "from the moment it is told" as one moment. Hints are left out because two wordings of one fact can differ there. "My eyes are green now." hints that they were not green just before. After "My eyes are green." that hint loses, since a hint never overrides. So "green now" after "green" is ALREADY_KNOWN.

**Examples.** All of these are golden cases in the self-test.

| Store, in order | New lesson | Verdict |
| --- | --- | --- |
| "My eyes are green." | "My eyes are green." | ALREADY_KNOWN |
| "My eyes are green." | "My eyes are green now." | ALREADY_KNOWN |
| "I like blue best." | "My favourite colour is blue." | ALREADY_KNOWN |
| "My eyes were brown until 11 October 2026; now they are green." | "My eyes are green." | IMPLIED: the dated fact covers it and says more |
| the same dated fact | "My eyes have been green since 11 October 2026." | IMPLIED |
| "My eyes are brown." | "My eyes are not green anymore." | IMPLIED |
| "My eyes are brown." | "My eyes are green." | REPLACES |
| "Of all colours, I like white the most." "I like green best now." | "Green is not my favourite colour anymore." | REPLACES: green becomes "don't know, but not green" |
| "My eyes are green." | "My eyes were brown until 11 October 2026." | NEW: only the past changes |
| "I like apples." | "I do not like apples anymore." | NEW: likes holds several values, so it is never REPLACES |
| "Red is not my favourite colour anymore." | "My favourite colour is blue now." | NEW: the value was not known |

**Through the gate.** Several sentences can go through in order, and only NEW and REPLACES join the store. "My eyes are green." twice, then "My eyes are green now.", then the dated fact, then "My eyes are green." again gives NEW, ALREADY_KNOWN, ALREADY_KNOWN, NEW, ALREADY_KNOWN. The last one is ALREADY_KNOWN, not IMPLIED as in the table above, because the first "green" is still in the store.

**What else it reports.**

- `same_as`: the earlier lessons that say the same (ALREADY_KNOWN).
- `covered_by`: the earlier lessons whose deletion alone would make the lesson news again (ALREADY_KNOWN, IMPLIED).
- `supersedes`: the earlier lessons that decided the current answer before and no longer do (NEW, REPLACES). This is our counterpart of TensorFold's "replaced" state.

**Reading the sentence.** There are two ways in.

- **The parser** reads the generator's own wordings (the tables `T` in `lwlogic.py`), for every entity they apply to. It is exact and never guesses. A sentence that ends in a question mark is a question, not a fact, and is not read. It also reads "I like blue." as a favourite colour (decision 6 in section 9). Names and cities are open: "My dog is called Rex." works, though Rex is in no list. Any other sentence is refused, not misread.
- **The extractor** asks any OpenAI-compatible chat endpoint to return the lesson as JSON in our schema. The reply is checked before use: a known entity, an attribute that applies to it, a known value, dates written as YYYY-MM-DD and no later than today, and the right fields for the operation. A reply that fails a check is refused. The extractor is only asked when the parser cannot read the sentence. The self-test never calls it; a mock endpoint stands in.

**Commands.**

```sh
python3 zengine/lwlogic.py intake --store examples/L2-seed1.json --phase del-A --text "Green is not my favourite colour anymore."
```

```sh
python3 zengine/lwlogic.py intake --store me.json --save --text "My eyes are green." --text "My eyes are green now."
```

```sh
python3 zengine/lwlogic.py intake --store me.json --text "I have had green eyes since spring." --endpoint http://localhost:8080/v1 --model MODEL --api-key-env KEY_VARIABLE
```

`--store` takes a test set (all its lessons, or one `--phase`) or a store file. Lessons are told on the test set's own day, so its examples always give the same answers. With a store file they are told on the real day, and never before the store's own day: a store saved last week still takes a fact from today. `--today` sets the day by hand. `--save` writes the learned lessons to a store file (`lw-store/1`); it never changes a test set. `--json FILE` writes the results. The key is read from the named environment variable and only sent in the request header. The command exits with 1 when a sentence cannot be read. `python3 zengine/intake.py --help` lists every option.

**Self-test.** 15 checks, all PASS with Python 3.12 and 3.14 (13 Oct 2026, MEASURED):

- **Parser round trip.** Every lesson wording in `T`, for every entity it applies to and every list value: 8,788 sentences since the level 7 entities came in (3,312 before), each read back to exactly its lesson and wording, with one reading only. Control: a parser that ignores the entity fails 8,426 of them (MEASURED, 13 Oct 2026).
- **Generated sets.** Every lesson of levels 1 to 5 × set numbers 1 to 30, and of the #678 set, reads back to its formal lesson: 1,012 lessons.
- **Refusals.** 14 sentences outside the tables, with an unknown entity, value or date, or ending in a question mark, are refused.
- **Golden cases.** 31 hand-sorted lessons and the gate sequence above.
- **Agreement.** On 300 random stores, the intake says "learn" exactly when the lesson moves an answer to the self-test's own question set. That set does not look at the lesson: it asks about four fixed days (the days the random lessons use), the day after each, and today. 131 lessons were kept out and 169 learned. A longer run by hand (16,000 cases over 8 seeds, 13 Oct) found no disagreement.
- **Broken classifiers, each caught:** comparing only the current value (caught by the agreement check and by 4 golden cases), comparing sentences, counting hints as part of what a lesson says, calling the same value the same lesson, and calling every change of the current answer REPLACES.
- **The gate loses nothing.** 300 random sequences on one or two keys, half of them with a lesson said twice, are taught two ways: through the gate, and in full. Every answer of the self-test's question set agrees at the end; 375 lessons were kept out. Control: a gate that compares only the current value loses answers in 71 sequences. A longer run by hand (9,500 sequences over 5 seeds, about 15,000 lessons kept out, 13 Oct) found no difference either.
- **The command line,** run in-process with its output captured: no store, a test set's phase, and an unreadable sentence, which must exit with 1. `--phase` without a test set and a `--today` that is not a date must be refused with exit 2. It caught two bugs put back in by hand: a crash when no store is given, and the report mixed into the JSON. The same check pins the day a store file tells new lessons on.
- **Extractor, against a mock endpoint.** 5 good replies are accepted and normalised ("Gray" becomes grey; a fenced or chatty reply is fine). 12 bad ones are refused: an unknown entity, an unknown value, a dated change of likes, a future date, a name for "me", no JSON, "NONE", a `when` that is not an object, an entity that is a list, an attribute that is an object, a date written as 20261011, and a reply whose content is not plain text. A bad reply is refused, never a crash. The request has the right URL, model, temperature 0 and key header.

**Against the 1.0.8 run records.** `python3 zengine/intake.py --records RUN_DIR ...` sorts every lesson of a run folder against the lessons taught before it in the same phase, and sets our `supersedes` beside the facts TensorFold marked "replaced". It only reads. On our seven run folders from 12 October (71 lesson steps in 16 phases; the folders are kept by the authors, see `docs/RESULTS.md`):

- The parser read all 71 lesson texts back to the set's own formal lesson. (MEASURED)
- No lesson was ALREADY_KNOWN or IMPLIED: 61 NEW and 10 REPLACES. The generator never repeats a fact. (MEASURED)
- In 9 steps TensorFold marked an older fact replaced. Our `supersedes` named the same lesson in 7. The other 2 are "I really like apples." replacing "I really like olives.": likes holds several values, so by our rules both stay. (MEASURED; TensorFold's marks are read from the records)
- In 7 steps we name a superseded lesson and TensorFold marked none. All 7 are "no longer" or "not ... anymore" lessons, and in 4 of them the server reported the lesson as not recalled. (MEASURED) So 1.0.8 seems not to retire a fact that a later lesson takes back. (INFERRED)
- In the other 55 steps neither marked anything.

**Open questions.**

1. **The gate and delete.** A lesson kept out by the gate is not in the store. If the lessons that cover it are deleted later, it is lost, although the user said it. "Delete = never taught" would bring it back. `covered_by` names those lessons, so a delete could run the intake again for the lessons they covered. Not built.
2. **REPLACES means the current value changes.** A dated fact that keeps "green" now but adds a past is NEW, though 1.0.8 marks the older "green" replaced. `supersedes` reports that lesson.
3. **One fact per sentence.** "My eyes are green and I live in Oslo." is two lessons. The parser refuses it, and the extractor is asked for one lesson only.
4. **The extractor has not met a real model yet.** Only the mock endpoint has been used. (UNCHECKED)
5. **`supersedes` needs one lesson that decides.** A lesson decides the current answer when deleting it alone changes that answer. After "My eyes are green." and the dated fact that ends in green, neither decides alone, since the other still gives green. So a later "My eyes are blue now." is REPLACES and names no superseded lesson. (MEASURED)
6. **The gate assumes the value lists stay the same.** "I have no favourite colour anymore." after each listed colour was taken back one by one is IMPLIED. A later fact about a colour outside the lists, such as pink, makes the list longer. Then the kept-out sentence would have mattered: with it, my favourite colour is "none"; without it, "don't know, but none of those eight". The gate check above uses fixed lists. (MEASURED, one constructed case)

## 13. Question forms v2: wrong yeses, who has it, and likes as a set

Plan items 7, 9 and 10 of `docs/LITERATURE.md`, built on 13 Oct 2026. They are new question forms, and a few new phases, added to a finished test set. Nothing the set had changes.

```sh
python3 zengine/lwlogic.py gen --level 4 --seed 1 --forms v2 --out set.json
```

**What stays the same.** Without `--forms v2` the generator writes the same bytes as before. A v2 set holds its v1 set unchanged: the same lessons, the v1 questions first, the v1 phases first, and the same expected answers. The self-test checks both for levels 1 to 5 × set numbers 1 to 30, and the committed v1 examples byte for byte (MEASURED). The v2 additions come from a random stream of their own, so they cannot move a v1 choice. A v2 set's id ends in `-v2` and it carries `"forms": "v2"`. Its format is still `lw-test-set/1`, so the runner takes it as it is.

**How the rule book answers them.** Every new question has one of the rule book's own types (CURRENT, YES_NO or WHO_HAS). So the rule book, the second implementation and the Lean export answer it unchanged, with three-valued truth: a value that was never taught is "don't know", or "no" where another value is known. A field `form` tells the judge how the question is asked.

| Form | Example (L4 set 1) | Type | Item |
| --- | --- | --- | --- |
| yes/no on an untaught value | "Is my dog's name Ana?" | YES_NO | 7 |
| true or false | "True or false: my dog is called Nora." | YES_NO | 7 |
| fill-in, no value named | 'Finish this sentence: "My dog's name is ..."' | CURRENT | 7 |
| recall, then yes or no | "First say what my dog is called, then answer yes or no: is it called Nora?" | YES_NO | 7 |
| reverse, bare | "Who has grey eyes?" | WHO_HAS | 9 |
| reverse, listed | "Of me, my dog and my cat, who has grey eyes?" | WHO_HAS | 9 |
| reverse yes/no | "Is it my cat who has grey eyes?" | YES_NO | 9 |
| the full list of likes | "What food does my dog like?" | CURRENT | 10 |
| a yes/no per like | "Do I like apples?" | YES_NO | 10 |

### Wrong yeses (item 7)

For every single-valued (entity, attribute) that a lesson is about, the set asks yes/no about at least 3 values no lesson on that pair ever named (fewer only when the list runs out). When another entity of the set was taught a value of the same attribute, that value comes first: the cat's name asked about the dog. A v1 yes/no about an untaught value counts toward the 3. These questions carry `"probe": "wrong_yes"`, and the neighbour's ones `"neighbour"`.

Each such pair also gets the other forms: true or false twice (an untaught value and the value of the last lesson on the pair), one fill-in, and recall-then-answer twice (the same two values). Where that lesson holds, its value is the right "yes" that a model saying "no" to everything gets wrong. Where a later lesson took it back ("My favourite colour is no longer blue."), both are "no" or "don't know" in the teach phase, and only the delete phases that bring it back ask for a "yes".

The new scores, per phase and thinking mode:

- **wrong yes**: the share of yes/no questions (every form) whose right answer is not yes, where the model said a firm yes. The baseline is scored on exactly the same questions, so the change from the baseline is like for like. It is split by form, and by the neighbour's values.
- **coherence breaks**: pairs of a single-valued attribute where the model said yes to two values, in any yes/no form.
- **recall agrees**: recall-then-answer answers whose yes or no follows from the value they recalled first.

### Who has it (item 9)

Each reverse value is asked three ways: bare, listed and as yes/no. The values are the set's own reverse questions first, then one held value per taught attribute, at most 3 per set. The list names everyone with a lesson on that attribute, you (where the attribute can be yours), and similar entities up to three. So the listed form names everyone who can hold the value, and its right answer is the bare question's in every phase; the generator checks this. The reverse yes/no asks about one holder and one who does not hold the value. Every entity a reverse question is about gets its forward question ("What colour are my dog's eyes?") if the set had none. When no lesson is about that entity's attribute, the forward question is a near miss, a control, so the recall score stays on taught pairs; a near miss the set already has serves as the forward question.

The judge reads the new forms like this:

- **True or false**: a leading "True" is yes and "False" is no, also after a lead-in: "Definitely false.", "I'd say false.", "This statement is false.", "That's not true.", "Incorrect.", "That's correct.". "Probably false." is a guess. The rest is read as a yes/no answer, so "False, I am a language model ..." is about the model and means "doesn't know". A first sentence that echoes the question is skipped: "True or false? I can't know that." is a refusal, and "True or false: my favourite colour is purple. False." is a no. An answer that only echoes the question is unclear, not a yes.
- **Fill-in**: read as a recall. One the model finished about itself ("My favourite colour is blue! What's yours?") is unclear: it opens in the first person and asks you back. A refusal that asks back ("I don't know. What's your favourite colour?") is a refusal.
- **Recall, then yes or no**: the answer is the last yes or no given: after "so", "therefore" or "the answer is", or a later sentence that opens with yes or no. "Your favourite colour is blue. Yes." to "... is it purple?" is a yes, so it fails. The yes or no must stand on its own: "so I can't answer yes or no" (the question echoed), "so no answer is possible" and "No other colour was mentioned." give none. The recall is kept beside it.
- **Reverse yes/no**: "Yes" and "No" as usual. With no yes or no word, it is read for who it names: "Your dog has grey eyes." is yes to "Is it my dog ...?" and no to "Is it my cat ...?". An answer that names no holder but gives the asked one a value is read for that value: "Your cat's eyes are hazel." is no to "Is it my cat ...?". An answer that names nobody ("Grey eyes are quite rare.") stays unclear.
- **Reverse, bare and listed**: as before (section 7), and on these forms "Yours.", "It's yours.", "I think it's yours." and "Your favourite colour is blue." name you; "yours are unknown to me" does not. v1 sets keep the old reading, so their verdicts do not move.

The new scores:

- **forward**: every reverse result is marked with whether this model passed the forward question in the same phase and mode (any plain wording of CURRENT; on likes also the plain yes/no, or a list that names the value). Results where it did also count in a `reverse_fwd` score line.
- **reversal curse**: counted only where the forward question passed. The listed form failing is the reversal curse. The listed form right and the bare one failing is counted as "bare only": our reading is that the model took the bare question as a general one (INFERRED).
- **reverse as a set**: of the bare and listed answers, how many of the holders were named, and how many extra entities.

### Likes as a set (item 10)

Every entity with a like gets the full list question and a yes/no for every food a lesson added. After every "I like X" from the second on, a phase holds the lessons up to that one (`after-L9`), at most 4 per set. Where a v1 phase already holds exactly those lessons, it is used instead: in L4 set 1 that is del-A, the lessons up to the third like. One more delete phase removes every like lesson but the last.

The runner runs these phases as delete variants, so reteach teaches exactly the lessons up to that like. Their kind is `prefix`, and the judge counts them as delete phases (`is_delete_phase`), so they keep their delete scores and kept_stable after the level 7 merge, which reads any other named kind as "not taught yet". The self-test check of the committed `examples/v2` sets catches it if they do not: without it, the regenerated L4 set loses the `affected` list of its after-L9 phase (MEASURED during the merge, 13 Oct 2026). That is "after each add". The zero stand-in zeroes the later lessons instead. That is not "after each add". In our 1.0.8 runs "I really like apples." replaced "I really like olives." (README, first runs), and 1.0.8 zeroes a replaced fact (section 8), so zeroing the later like cannot bring the earlier one back (INFERRED). Read Preservation in these phases from reteach only. The runner does not ask questions in the middle of teaching.

`--likes restate` makes each "I like X" after the first restate the whole set: "I like pears, olives and apples." It writes one "add" per value, with no hint. Lean gets one `assert` per value, with the lesson's own id and tick, which are the same writes. Ids, phases and questions stay those of the one-at-a-time set (`examples/v2/L4-seed1-v2-restate.json`). The two sets give the same answers in the teach phase and after each like. They differ only where earlier like lessons are deleted: in the restated set the last lesson still holds the earlier likes (MEASURED: 39 of 45 restated sets of levels 4 and 5 differ there, 13 Oct 2026).

The new score, **earlier likes kept** (Preservation): in a phase that adds at least two likes on one entity, the likes from before the newest one that must still hold. A firm yes to each one's yes/no, and each one named in the full list, counts as kept. Every other answer counts as lost, an off-topic or unclear one too, so a model cannot look better by not answering. What is lost is the **Ignore Rate**. The newest like is counted on its own line.

### The report

The judge prints a "forms v2" block for each phase, and the `--json` file has a `forms` section. Planted answers in the L4 set, thinking off, teach phase:

```
   forms v2                               thinking off                  thinking on
   wrong yes                  2/33 (baseline 0/33, +2)                            -
     yes/no                                       1/18                            -
     true/false                                    1/6                            -
   coherence breaks                        2 of 5 keys                            -
   reverse, forward ok    bare 2/3, listed 2/3, yes/no 6/6                            -
   reversal curse                 1 of 3 (bare only 1)                            -
   earlier likes kept              3/3, Ignore Rate 0%                            -
```

### What was checked

- **Self-test** (MEASURED, 13 Oct 2026): 13 new checks. On the forms v2 branch, 63 of 63 passed with Python 3.9, 3.12, 3.13 and 3.14; after the merges see section 4. Each new check has a control that is broken on purpose:
  - v1 unchanged. Controls: an edited v1 question and an edited v1 expected answer are caught.
  - The v2 generator is complete (levels 1 to 5 × set numbers 1 to 30). Controls: five sets broken on purpose are caught: no neighbour's value, a holder left off a list, no phase after a like, no true/false form, and a forward question about an untaught pair scored as a probe.
  - Restated likes. Controls: a restated lesson read as its newest like only, and as "only these", are caught.
  - The judge on each new form, with hand-written answers: true or false (12 answers), fill-in (12), recall then answer (11), and the reverse forms (20). With each reader switched off, 2 to 4 labels break.
  - The review of 13 Oct: 26 more answers, 18 that the first readers misread and 8 guards. "Definitely false." and "I can't say what it is, so I can't answer yes or no." were read as yes, which made wrong yeses that were not; "I don't know. What's your favourite colour?" was unclear instead of a refusal. Each of the 5 fixes is taken back in turn, and every one of the 18 answers that needs it breaks; the 8 guards hold both ways.
  - The new scores on planted answers: wrong yes and coherence, the reversal curse, and earlier likes kept, also with off-topic answers. Controls: a rate over all yes/no questions, coherence read from the plain form only, counting without the forward question, counting the newest like as an earlier one, and leaving out the answers that are not a yes, no or list are each caught.
  - A judge round trip on 50 v2 sets: 13,819 right answers pass and 13,819 wrong ones fail.
  - The committed `examples/v2` sets are what the generator makes.
- **Lean** (MEASURED, 13 Oct 2026): the 8 sets in `examples/v2` add 1,812 checks. Lean agrees with Python on all of them, 2,278 of 2,278 across the 14 example sets on the forms v2 branch (section 4 has the count after the merges). `leancheck` has a second control for the restated set: exported with each restated lesson read as its newest like only, Lean rejects exactly the 3 checks whose answer that changes, all in the phase that deletes the earlier like lessons.
- **v1 verdicts did not move** (MEASURED, 13 Oct 2026): `tools/rejudge.py` judged all 1,470 stored 1.0.8 answers with main's judge and with this one. 0 verdicts changed, and all 17 reports in `docs/rejudge` came out byte for byte the same. The run records were only read.
- **The runner takes v2 sets** (MEASURED, 13 Oct 2026, mock server only): `lwrun.py` ran `examples/v2/L1-seed1-v2.json` (4 steps), `L4-seed1-v2.json` (12 steps, the new del-D and after-L9 among them) and `L4-seed1-v2-restate.json` against the runner self-test's mock server, with both delete stand-ins. All three finished, and the judge's JSON had its `forms` section.

### Limits and what is not built

- **Never run on TensorFold.** Every number above is from the self-test, stored answers or a mock server. (UNCHECKED on a model)
- **Longer runs.** L4 set 1 goes from 22 to 75 questions and from 5 to 7 phases; levels 1 to 5 go from 12 to 50 questions to 23 to 109. Each new phase is one more reteach copy in the runner.
- **Fill-in.** A model may finish the sentence as itself without asking back. The judge then reads it as a recall. (INFERRED risk)
- **"Yours"** is read only on the v2 reverse forms. v1 sets keep the old reading on purpose.
- **Forward** means any plain wording of CURRENT passed, not all of them.
- **A firm yes only.** A guess that leans yes ("Maybe, yes") is not counted as a wrong yes. It already fails or is unclear in the verdicts.
- **Restated sentences** are not read by the intake parser: "I like pears and olives." is two facts, and the parser takes one fact per sentence (section 12).
- **No new theorem** is about restated lessons. Lean checks them through the expansion, on the example sets.
- **neg-seed1** gets no reverse questions: no value is ever held in it.
- **"Is it my cat who has grey eyes?"** is scored as a plain yes/no about the cat. Where nothing is known about the cat's eyes, the right answer is "don't know", so "No, it's your dog." fails there, although the wording invites that reading. Whether such a question should be read as "Is the cat the one?" is still to confirm.
- **v1 questions inside a v2 set** can be judged a little differently from the same question in its v1 set. The listed forms add entities to the set, so "Your sister likes olives." names someone of the set and fails, where in the v1 set it names someone the test never taught about and is unclear. The v1 bare reverse question also gets the "yours" reading. Compare v1 and v2 runs by the v1 set's own judge where it matters.
- **Mixed lists.** A list can mix people and things ("Of my brother, the boat and the tent, who is called Milo?"), and a reverse yes/no can say "who" of a thing ("Is it the tent who is called Milo?"). The answers are right; the wording is clumsy.
- **One sentence the v1 reader misreads** shows on these forms too: in "Your favourite colour is blue, your sister's is unknown.", the sister counts as named. Fixing it would move v1 verdicts, so it is left for a judge change of its own.
- **Not done:** telling Ash Hart that replacement on a list attribute should key on (entity, attribute, value) (item 10's feedback part), teaching a reversed sentence as its own lesson (item 9's later step), and asking questions in the middle of a teach step instead of reteaching.

## 14. Level 6: two-step questions

Plan item 19 of `docs/LITERATURE.md`. A two-step question can only be answered by putting two facts together. Published work reports that facts learned separately are often not combined: MQuAKE, the two-hop paper and User as Engram, among others (READ: the numbers are in `docs/LITERATURE.md`, item 19). Level 6 measures this for Living Weights.

`zengine/levels.py` makes the sets, judges the answers and scores them. The rules are in the rule book (`lwlogic.py`, "two-step questions") and in `lean/LwLogic.lean`, so every expected answer is computed, not written by hand.

**Five families.**

| Family | Example | Steps |
| --- | --- | --- |
| natural | "Which country do I live in?" after "I live in Oslo." | my city, then its country (general knowledge) |
| owner | "What colour are the eyes of the owner of the boat?" | the boat's owner, then that person's eyes |
| alias | "What colour are Ana's eyes?" after "My sister's name is Ana." | who is called Ana, then her eyes |
| link | "What colour are my cat's eyes?" after "My cat's eyes are the same colour as mine." | the link, then my eyes |
| compound | "Is it true that my eyes are green and my car is red?" | each part |

Families combine. "In which country does the owner of the boat live?" can take four steps: the owner (my brother), his link to my sister's city, her city, and its country.

**What is new in the rule book.**

- A linking fact is a lesson on an attribute of its own. "My cat's eyes are the same colour as mine." is ASSERT(my cat, eyes_like, me). "My brother lives in the same city as my sister." is ASSERT(my brother, city_like, my sister). Change, retract and delete work on them as on any lesson.
- The owner and the two link attributes have values that name people of the set ("my sister").
- The country of each city is fixed general knowledge, kept in the set's own schema: Lisbon and Porto in Portugal, Oslo and Bergen in Norway, Tampere and Turku in Finland, Ghent in Belgium, Graz in Austria. So a move can keep the country (Oslo to Bergen) or change it (Oslo to Porto).

**The rules.**

- A chain asks about now.
- A name points at the one entity called that now. Nobody with that name, or two: "don't know".
- A hop (the owner) must name someone of the set. "The tent belongs to my neighbour." names nobody the set knows, so "the eyes of the owner of the tent" are "don't know".
- A link only fills gaps, like a hint (section 1). What is said outright about my cat wins over the link, whichever was taught first.
- A link is read one way and one link deep. "My cat's eyes are the same colour as mine." does not tell the rule book my eyes from my cat's. A person might conclude it; the generator never asks it.
- A link is a standing rule: if my eye colour changes, my cat's linked colour follows. A person could also read the link as a snapshot of that moment. The generator never changes a linked value after the link, so its sets do not depend on this choice.
- My cat "not blue anymore", then linked to my blue eyes: no colour fits, so "none". The generator never makes such a contradiction.
- The natural step maps a known city to its country. "I do not live in Oslo anymore." says nothing about the country: "don't know".
- Compound questions use Kleene's strong three-valued logic. "And" is no as soon as one part is no, even when the other is unknown. "Or" is yes as soon as one part is yes. "Not" turns yes and no around and leaves "don't know" alone. "Or" is inclusive.

```
linked(e, a)           = e's own state of a now; where a value is unknown, the value of t, if e's link for a
                         names an entity t ("same as mine")
named(v)               = the one entity whose name is v now, else nothing
chain(start, hops, a, natural):
    e = start, or named(v)
    for each hop h:  e = the entity the value of (e, h) now names; none: stop, "don't know"
    s = summary(linked(e, a))
    natural:  a known city gives its country; none stays none; anything else is "don't know"
chain yes/no (... , v) = what that summary says about v
compound               = Kleene's and / or / not of the parts' yes/no answers
```

**A level 6 set.** `python3 zengine/lwlogic.py gen --level 6 --seed N`. Sets 1 to 3 are in `examples/`. Each one has (MEASURED for sets 1 to 3):

- 29 lessons on 8 people, pets and things, all 7 attributes and the two links; 56 questions; 5 phases.
- My home moves (within the country or to another one), the boat's owner changes between my sister and my brother, and my dog gets a new name. So some chains must change after a CHANGE lesson.
- My cat's eye colour is said outright and also implied by the link to mine. My brother's city is only implied, by his link to my sister's.
- 16 chain questions and 8 compound questions, one for each of eight Kleene cases in the teach phase ("yes and yes", "unknown and no", "unknown or yes", "no or no", "not unknown", ...). Not every row of the truth tables is asked.
- Every single step of every chain, in every phase, is also asked alone. The country questions ("Which country is Turku in?") are asked in every phase, the baseline included.
- A pair that has a link is only asked through chains. A plain question there would ignore the link.
- The usual controls: near misses ("What colour are my cousin's eyes?") and general knowledge.
- Phases: baseline, teach and three deletes:
  - del-A, a deep delete: the cat's own eye colour. The link to my eyes still implies it, so it must stay known.
  - del-B, one link of each chain: the boat's owner, my sister's name and my brother's city link. The chains through them must become "don't know".
  - del-C, the changes: the chains through them go back to the value before the change.

**The judge.** A chain is judged like a plain question about its answer (the country, the owner's eye colour). A compound question is judged like a yes/no question, with three more readings, because it is asked as "Is it true that ...?" and its answers often reason through the parts:

- An opening "true" or "false" counts: "That's true.", "False.", "It is false." (the plain yes/no reading takes "It is ..." as a yes, and the other three as no answer at all).
- A conclusion after the parts counts when the opening gives no yes or no: "I don't know your brother's favourite colour, but you no longer like cheese, so no." is no. The plain reading takes the opening "I don't know" as the answer and fails this right answer.
- For "Is it true that my car is not red?", "Your car is red." is read as not true, the way a plain yes/no question reads a named value.
- An opening and a conclusion that disagree are unclear. Guesses, refusals without a conclusion, and answers about the model itself keep the plain reading.

There is one new verdict:

- **echo**: the answer names only a step on the way and no value of the answer. "You live in Oslo." for the country, "The boat belongs to your sister." for her eye colour, "Ana is your sister.", "They are the same colour as yours.", or "The tent belongs to your neighbour." where the answer is "don't know". It also counts when the answer adds "but I don't know her eye colour". Echo is never a pass.
- In a delete phase, an answer that names only a step the chain took before the delete ("The boat belongs to your brother." after the owner lessons were deleted) keeps its verdict, and its reason says so instead of "off topic".
- In the standard score table and in the runner's scores an echo counts with "unclear"; the two-step table counts it apart.
- The date check of plain questions is not run on level 6 questions. None of them has a date.

**The scores.** One unit is one two-step question in one phase and thinking mode.

- **raw accuracy**: the `two_step` and `compound` rows of the standard table, per phase.
- **stated**: accuracy on the taught single steps a chain rests on, asked alone in the same phase and thinking mode. Each chain counts its own steps.
- **implied**: accuracy on the chains themselves.
- **gap** (the headline): stated minus implied, in percentage points.
- **given steps right**: accuracy on chains whose every step, the country included, was answered right in the same phase and thinking mode.
- Each is given by number of steps (1 to 4) and by family. The number of steps is the number of facts the answer rests on in that phase: my cat's eyes take one step in the teach phase (said outright) and two after the deep delete (the link, then my eyes). The headline pools 2 steps or more.
- A unit counts only in phases with lessons taught and when the rule book knows its answer. Chains whose answer is "don't know" are scored apart.
- **Baseline drop.** A chain whose country question the base model gets wrong in the baseline, in the same thinking mode, is dropped from these scores. Without a baseline answer nothing is dropped, and the table says so.
- Compound questions: accuracy, accuracy given both parts right, and accuracy per Kleene case.

`lwlogic.py judge` prints a two-step table under the standard one, and writes the same numbers under `two_step` in its `--json` file. The runner calls `lwlogic.py judge`, so its reports carry the table with no change to the runner.

**Lean.** 30 new theorems, 93 in all, all proved (MEASURED, Lean 4.35.0, 13 Oct 2026). The definitions mirror the Python names (`chainSum` and `chain_sum`, `linkedState` and `linked_state`, ...). The main theorems:

| Theorem | Plain meaning |
| --- | --- |
| `chain_known_iff` | a two-step chain is known exactly when both links are known |
| `chain_value_iff` | its value is the second link's value, read on the entity the first link names |
| `delete_link_unknown` | deleting the only lesson on a chain's first link makes the chain "don't know" |
| `named_chain_unknown`, `namedEnt_untouched` | a name nobody holds, for example when every name lesson is deleted, points at nobody |
| `deep_delete_implied` | deleting what was said outright keeps the answer the kept lessons imply through a link |
| `linked_fills_gap`, `own_wins` | a link fills gaps only; what is said outright wins |
| `functional_linked` | a linked answer on a single-valued attribute never holds two values, contradictions included |
| `kAnd_no_iff`, `kOr_yes_iff`, `kAnd_yes_iff`, `kOr_no_iff` | Kleene's truth tables |
| `deMorgan_and`, `deMorgan_or`, `kleene_classical`, `kleene_mono` | De Morgan's laws; ordinary logic on known parts; when an unknown part of a compound becomes known, a yes or a no of the whole stays (this is about the connectives, not about lessons: a CHANGE lesson can still turn an answer around) |

The schema got four fields for level 6 (which values name entities, the link of each attribute, the country map, the name attribute). Their defaults switch them off, so the check files of levels 1 to 5 did not change. `leancheck` recomputes all 750 expected answers of the three level 6 example sets, and they all agree with Python (MEASURED). Its second negative control changes one two-step answer, and Lean rejects exactly that one (MEASURED).

**Self-test.** 23 checks, run by `lwlogic.py selftest`, or alone by `python3 zengine/levels.py selftest` (about 2 seconds). Results of 13 Oct 2026, after the review (MEASURED):

- **Kleene's logic.** The 21 rows of the truth tables and De Morgan's laws. Controls: a closed-world logic (unknown read as no) gets 10 rows wrong, and a weak one (any unknown part makes the answer unknown) gets 4 wrong.
- **Seven properties of the chain rules**, each on 300 random stores (2,400 questions for the first): equal to a second implementation that asks the single-step rule book one plain question at a time, as a person would; known exactly when both links are known; "don't know" after its link is deleted; the deep delete; a chain follows a CHANGE of its link; what is said outright wins; never two values.
- **Six broken systems, each caught** by the property it breaks: an oracle that answers chains from the first link only (by the second implementation in 198 of 1,200 questions, and by "both links known" in 33 of 150); hops read from the first lesson taught on them, so a change of owner is not seen (by the change property in 128 of 150, and by the second implementation in 5 of 1,200); links ignored (by the deep delete in 95 of 149, and by the second implementation in 9 of 1,200); chains that do not see deletes (89 of 150); a link that overrides what is said outright (150 of 150); a link that adds its value to what is said (55 of 150). The override is rare in random stores, so only its own property is asked to catch it.
- **38 golden answers**: owners, a change of owner, an owner the set knows nothing about, names (two holders, a new name), countries (a move within and out of a country, a taken-back city), links (filling a gap, said outright wins, one way, one link deep, a standing rule, a contradiction), Kleene, deep deletes and deleted links. Controls: four broken systems each get some of them wrong (chains from the first link only 18, a link that overrides what is said 2, links ignored 6, a closed-world and/or 2, of 38).
- **The generator**, sets 1 to 30: the same set every time; 25 to 35 lessons on 8 entities with every attribute taught; the expected answers recompute; no plain question on a linked pair; every family and the eight Kleene cases; every step of the 1,499 chain answers the rule book knows has its own question; and in every set a change moves a chain, the deep delete keeps the implied fact known, and a deleted link makes a chain "don't know". Controls: three broken copies of set 1 are caught, each by its own rule (a step question removed, a plain question on a linked pair added, a deep delete that also deletes the link).
- **Levels 1 to 5 are untouched.** The committed L1 to L5 and #678 sets, and the three level 6 examples, are byte for byte what the generator makes. A one-off comparison also hashed all 150 sets of levels 1 to 5 (sets 1 to 30), the #678 set and the Lean exports of the 150, before and after level 6: the same (MEASURED). Control: one space added to one lesson of L1-seed1 is seen.
- **The intake check** (section 12) refuses a level 6 set with a reason, instead of failing on the link attributes it does not model. Control: a level 5 set is read.
- **The judge.** 31 hand-written answers on the families, on set 1 (the tent is mine) and set 2 (the tent is the neighbour's): echoes (also "You own the tent.", "It's yours."), right answers, refusals, wrong values, and a step named after its delete. Control: with echo switched off, 16 of them get other verdicts.
- **Compound answers.** 28 hand-written answers, 14 on each of sets 1 and 2: "true" and "false", a conclusion after the parts, a named value under "not", and answers that must keep their verdict (a refusal without a conclusion, "so no one can say", a guess, an answer about the model). Control: the plain yes/no reading alone gets the 14 marked ones wrong.
- **A judge round trip** on sets 1 to 10: 2,800 right answers pass, 2,800 wrong answers fail, and 430 answers that name only a step on the way are echo.
- **The scores**, on four planted answer files: all right (49 chain answers with thinking off on set 1: stated and implied 100%, gap 0); steps right but chains echoed (implied 30.6%, gap 69.4 points, 34 echoes); one country missed in the baseline (6 chain answers dropped, with thinking off only); one step answered wrong (given steps right counts 36 of 49 chains, gap -13.7 points). Controls: a scorer without the condition on the steps, and one without the drop, are caught.
- **The Lean export**: the six schema fields start in one column in sets 1 to 30. Control: a field one column off is told apart.

**The runner.** Unchanged. One run end to end against the runner's own stand-in server (no model): L6-seed1, thinking off, both delete modes. Exit code 0, all 8 steps taught or deleted, asked and judged, and both combined reports carry the two-step table (MEASURED, 13 Oct 2026). The same again after the review's judge changes: exit code 0, all 8 step reports and both combined reports carry the table (MEASURED, 13 Oct 2026). The stand-in answers with the text of its last live fact, so its scores mean nothing. Level 6 has never run on TensorFold (UNCHECKED). A real run teaches 29 lessons per model copy. At 2 to 4 minutes per lesson (`docs/RESULTS.md`) that is 1 to 2 hours per copy, and the reteach mode needs three more copies (INFERRED).

**Limits.**

- The judge reads words. An echo that also names a value of the answer is judged as a plain answer, right or wrong.
- "You" is in almost every answer, so it does not count as an echo of "me" alone. For me the judge looks for "belongs to you", "is yours", "you own", "you are the owner" or "the same ... as yours".
- The judge reads the asked attribute and the steps of this phase. A step the model invents is not checked: "You live in Oslo." in the baseline, for "Which country do I live in?", is unclear, not fail.
- The compound readings are for compound questions only. Levels 1 to 5 also ask "Is it true that my eyes are amber?" (L3-seed1 Q11, L4-seed1 Q10). There the plain reading passed "It is false." as a yes, and failed "That's true." and "False." as off topic, when the answer is yes (MEASURED on those two questions, 13 Oct 2026). Since the branch judge-misreads (later on 13 Oct) the plain reading reads an opening "true" or "false" as the true/false form does (section 7), on chain questions too; compound questions keep the readings above. No stored answer changed (`docs/rejudge/misreads/`).
- Links are read one way and one link deep; see the rules above.
- A name means the current name only. The generator never asks for an old name ("What colour are Luna's eyes?" after Luna was renamed), because people read that both ways.
- The intake check does not model linking facts. The night clean-up prototype has not been tried on a level 6 set.
- Plan item 19 expects chains of two taught facts to fail most of the time with thinking off. That is the literature's finding, not a measurement of ours (INFERRED).

## 15. Level 7: filling the pocket

Plan item 20 of `docs/LITERATURE.md`, the runner side. Papers on editing models one fact at a time report slow forgetting, then a sudden collapse, and the earliest edits lost first (Gupta et al. 2024; MEMoE). Level 7 measures that on TensorFold's pocket.

**What TensorFold's code says about room** (READ, tag v1.0.8: `pocket.zig`, `learner.zig`, `slide.zig`):

- Every MoE layer's pocket has 64 neurons, one 4-bit group of the shared expert. A fact takes up to 10 of them, at most one per layer for each kind of key.
- A fact's neurons are keyed either at the start of the reply or at the subject its questions name. Reply-keyed neurons may sit from a fifth to seven eighths of the way in, subject-keyed ones from a fifth to five eighths. How a fact's 10 are split between the two depends on how many of its questions name a subject. A layer whose pocket is full is passed over.
- When fewer layers have room than a fact asks for, it takes fewer neurons. Only when no layer it may use has room does learning fail with `PocketFull`. The server sends it as the lesson's message, with `"recalled": false`.
- A replaced fact's neurons are zeroed, and their places stay taken while the server runs. On load, a layer's neurons in use are counted up to the last one whose input row is not zero (`weights.zig`), so zeroed neurons above every live one come free after a restart.
- Each new lesson keeps earlier lessons' answers as they are, from a replay of at most 256 of them, 8 per lesson, the oldest giving way (`learner.zig`). The replay is kept in memory: a restarted server starts with none.

Our reading (INFERRED, not run). Of the 52 layers, the 23 MoE layers hold pockets, and all but the last can take neurons. Subject-keyed neurons may use the 10 of them from layer 10 to layer 31, 640 neurons in all; reply-keyed ones may use 16 layers. A fact about you is reply-keyed: the stored Mac L1 fact sits at layers 20 to 40. A fact whose questions name someone, as level 7's do, is mostly subject-keyed, so about 64 such facts may fill those 10 layers. Facts taught near the edge may get fewer than 10 neurons before any lesson comes back `PocketFull`; the runner records each lesson's neuron count. Level 7's 64 facts sit near that edge, and its 100-fact variant should cross it. And at 8 answers a lesson, the replay is full after about 32 lessons: from then on the oldest facts are no longer kept steady, so the first facts may be the first to fade. Level 7 is built to see both.

### The set

`python3 zengine/lwlogic.py gen --level 7 --seed 1` makes `examples/L7-seed1.json`:

- **64 facts**, each a plain fact on an entity-attribute pair of its own: "My nephew lives in Tampere.", "The clock is black." No fact replaces another by the rules. `--n-facts` takes 1 to 100 (there are 100 pairs).
- **Two questions per fact**: one CURRENT wording, and one yes/no that asks the taught value or another value, half and half. For likes (several values at once) the yes/no always asks the taught value: by the rules another value stays "don't know" whatever is taught, so it could not show a fact lost.
- **Controls**: two near misses (the same attribute of the similar entity, never taught), one untouched attribute, and three general-knowledge questions. Near the full 100 pairs fewer of them are left untaught: from about 98 facts the near misses and the untouched pair drop out one by one, and with 100 only the general-knowledge questions remain.
- **Checkpoint phases** `ck-001`, `ck-005`, ..., `ck-060`, and `teach` after the 64th: the first k lessons.
- **Pair phases** for the interference matrix: a similar pair (one attribute, two similar entities, two values: in seed 1, my colleague's and my friend's names) and a far pair (other attributes and kinds of entity). `ix-KIND-A` holds A alone, `ix-KIND-AB` holds A then B.
- **Delete phases** `del-1`, `del-3` and `del-6`: six facts, one from each sixth of the teaching order, deleted one at a time in a random order.

The rule book gives every expected answer, as for the other levels. Lean recomputes all 2,882 of them (MEASURED).

### The runner

`zengine/capacity.py run SET.json --bin ... --model-src ... --work ... --port ...` works through the set:

1. **Baseline.** The original model is asked every question.
2. **Main.** A fresh copy is served with `--slide`, and the lessons are taught in order. Before each lesson:
   - **Intake.** The rule book sorts it against what the pocket holds (section 12). The parser must read the sentence back to the set's own lesson.
   - **Already known (model).** The model is asked the lesson's open CURRENT question, thinking off. The lesson is marked "already known (model)" only when the judge reads its exact value from the answer, said as a fact: "It is probably Tampere." is a guess and does not count. A yes/no is never used for this: our runs had wrong yeses about facts nobody taught ("Is my favourite colour purple?" in L1; "Yes, your dog's name is Nora!" in the L4 baseline). The yes/no is asked too and recorded, to show how often it would have said "known".
   - **Gate.** `--gate mark` (the default) teaches every lesson and only records the two checks. `--gate skip` does not teach a lesson the intake calls ALREADY_KNOWN or IMPLIED, or one the model already knows. Such a lesson still counts: the user said it.
3. **When the pocket is full** (`--when-full`):
   - `refuse` (the default): the lesson is not learned, and teaching goes on, since a later fact may still find room: reply-keyed neurons (a fact about you) may also use the six layers after layer 31 that subject-keyed ones cannot.
   - `forget`: gradual forgetting, oldest first. The oldest learned fact is deleted with `DELETE /v1/slide/fact/{id}`, and the lesson is taught again, up to `--evict-max` times (3).
   - `replace`: the same, but the fact to give way is chosen. First one the rule book says other facts cover, then one the model knew before it was taught, then a weight-norm outlier, then the oldest.
   - 1.0.8 has no per-fact delete. A 404 there turns `forget` and `replace` into `refuse` for the rest of the run, and the report says why. So does a delete after which the graph's `pocket.used` does not go down: on 1.0.8 a zeroed fact keeps its places.
4. **Norms.** After each save, the fact's pocket weight norm is read from the saved folder (below). At every checkpoint all facts' norms are read again, so a norm that moves later is seen.
5. **Checkpoints.** After the first lesson and after every fifth, the model is asked one CURRENT and one yes/no question of every fact taught so far, and the controls, in the `--slide` server.
6. **Pairs.** For each pair, a fresh copy is taught A and asked A's and B's questions, then taught B and asked again.
7. **Deletes** (`--delete-mode`), one at a time, asked after the 1st, 3rd and 6th:
   - `zero` (the default): one copy of the taught folder, each lesson's neurons zeroed as `lwrun.py`'s zero mode does, served without `--slide`. The taught folder is also asked once without `--slide`, so the delete answers are compared with answers served the same way.
   - `endpoint`: TensorFold's own `DELETE /v1/slide/fact/{id}` on a copy served with `--slide`. On 1.0.8 this records "unavailable" and asks nothing.
   - `reteach`: a fresh copy taught the lessons each phase keeps (never taught). About 60 lessons per phase.
   - kept_stable: the judge compares each delete phase with the same model's teach answers (section 7).

`--dry-run` checks everything and prints the plan, and writes nothing. For `examples/L7-seed1.json` with the defaults it plans 68 learn requests (about 2.3 to 4.5 hours at 2 to 4 minutes each, as in our 1.0.8 runs), 3,536 chats and 4 copies of the model (MEASURED, the dry run's own count). There is no resume: a stopped run starts again in a new `--work`.

### Pocket weight norms

`lwrun.pocket_norms` reads a fact's neurons from a saved folder, read only. The layout is READ from our stored 1.0.8 run records:

- `config.json` has `tensorfold_learned_neurons: 64`.
- Each MoE layer's `shared_experts.up_proj` and `down_proj` are in MLX's 4-bit format: 8 codes per 32-bit word, a scale and a bias per group of 64. A value is scale × code + bias.
- Learned neuron k is row 3712 + k of `up_proj` (its input weights) and column 3712 + k of `down_proj` (its output weights). The 64 pocket columns of `down_proj` form one group, so they share its scale and bias.

The report gives each fact's input norm, output norm and both together. It flags outliers among the facts' total norms: a modified z score above 3.5, or more than 10 times the median. Only large norms are flagged.

On our stored Mac L1 folder (TensorFold 1.0.8), the one fact's 10 neurons sit at layers 20 to 40. Its input norm is 0.94 and its output norm 52.0. A neuron no fact uses has an input row of exactly zero. In the zeroed copy the input norm is 0 and the output norm is still 52.0: zeroing touches only `up_proj` (MEASURED, read only).

Because the output columns share their group's scale, saving a new fact may move the output weights of the facts already in that layer (INFERRED). The checkpoint norm reads would show that as drift.

### The report

`WORK/capacity.json` and `WORK/capacity.txt`; `capacity.py report WORK` writes them again from the records.

- **Retention curve**, at every checkpoint and named at 1, 5, 10, 20, 40 and 64 facts (and 100), per thinking mode: CURRENT right of the facts taught so far, the same over the facts the pocket held then, yes/no right, the controls, and accuracy by teaching order in groups of ten.
- **The first 10 facts on their own**, at every checkpoint. The report says when that series drops.
- A per-fact matrix of CURRENT verdicts by checkpoint.
- What was learned, missed, refused because the pocket was full, skipped by the gate and evicted, and where the pocket first filled.
- The intake verdicts, "already known (model)", and the lessons a yes/no would have called known.
- The norms, their outliers, and any drift between checkpoints.
- The interference matrix: A's and B's answers after A, and after A then B, for both pairs, and how many of A's answers B took away.
- The deletes: the deleted facts' answers (they should say "don't know"), the rest, and kept_stable.

### Self-test

`python3 zengine/capacity.py selftest` runs the runner against the runner's own mock server (section 8), on a 12-fact level 7 set, in about 45 seconds. 18 of 18 checks pass with Python 3.9, 3.13 and 3.14 (MEASURED, 13 Oct 2026; the 18th came with the review), and again with 3.12 and 3.14 after the merges. For level 7 the mock has a pocket of 8 places, writes real 4-bit rows with known norms, and can forget, interfere, delete or keep places taken, as a scenario says. Each feature has a negative control:

| Feature | The check | Its control |
| --- | --- | --- |
| Refuse | 8 facts fit (a lesson missed for another reason takes no place); lessons 10 to 12 come back PocketFull and are refused; teaching goes on | a missed lesson is not counted as full |
| Retention curve | right exactly for the facts the mock holds, at 1, 5, 10 and 12 facts | the counts are worked out from the set, apart from the report |
| Already known (model) | only the lesson the untaught mock answers with the exact value | a yes/no reading would have marked seven lessons, six of them unknown to the mock |
| Not known from a guess | an untaught answer with the right value said as a guess ("It is probably ..."), or with another value, is not known, and `--gate skip` teaches both | a rule that also took a guess would have marked the guessed lesson |
| Norms | equal to the values the mock wrote, for all 8 facts; the planted large one is the only outlier | a reader one neuron off reads other rows: 8 of 8 differ |
| Forget | lessons 10 to 12 each evict the oldest fact and are learned; the curve counts an evicted fact as held at the checkpoints before its eviction | a curve that left evicted facts out of every checkpoint fails it (tried in review) |
| Replace | evicts the fact the model knew, then the outlier, then the oldest | forget evicted 1, 2, 3 instead |
| No delete endpoint | a 404 turns forget into refuse, with the reason | |
| A delete that keeps places | one fact is lost, the run notices and evicts no more | |
| First 10 facts | when early facts fade, the series drops and is flagged | the refuse run's series does not drop |
| Interference | B takes A's answers in the similar pair; the far pair keeps them | the refuse run lost nothing |
| Deletes, zero | deleted facts say don't know (except one the untaught model knew anyway); kept_stable keeps every other answer | |
| Deletes, endpoint | kept_stable flags the one kept fact that leaned on a deleted one, and only it | the zero run kept everything |
| Gate skip | the known lesson is not sent and is still answered right, thinking off and on | |
| Report again | `capacity.py report` rebuilds the same report | |
| Refusals | `lwrun.py` refuses a level 7 set; the dry run writes nothing | |

### Not built, and why

- **Similarity measured inside the model.** The plan picks the pairs by how alike the base model's activations are at the pocket layers (an MLX script). That needs a model run, so the pairs come from the rule book instead: the same attribute of two entities it calls similar, against other attributes and kinds.
- **An in-context baseline** beside the curve (plan item 5) is not part of this runner.
- **Usage ranking** needs a record of which facts' neurons fire, which 1.0.8 does not give (section 16).
- **Never run on TensorFold.** Everything above is checked against the mock server only (UNCHECKED on a real server).
- **The judge does not check which entity an answer names.** "Your nephew's eyes are blue." passes for "What is my niece's eye colour?" when the niece's are blue, and "Your nephew is called Otto." passes as a yes to "Is my parrot called Otto?" (MEASURED in review, with this branch's judge and with main's at 07db179). Level 7 gives many entities the same values, and the similar pair invites mix-ups, so a mix-up that keeps the value counts as right. A fix belongs in the judge, with every stored run judged again.

## 16. Level 8: freeing at night, with relearning and ranking

Plan item 21 of `docs/LITERATURE.md`. It extends the night clean-up prototype, `zengine/prune.py` (`docs/NIGHT-PRUNE.md`). Freeing one fact at a time already compares every stored answer before and after, by text or verdict, which is kept_stable on the stored questions.

- **Ranking by weight norm.** `--order norm` tries the facts by their pocket weight norm, read from the taught folder: large outliers first, then from the largest norm down, then facts already switched off (input rows zero). Gupta et al. (2024) report single edits whose weight change is far larger than the rest, and that such edits can disable a model. The ranking is in every report, whatever the order.
- **Ranking by usage** needs to know how strongly each fact's neurons fire on the stored questions and on ordinary text. TensorFold 1.0.8 has no endpoint or log for that, so `--order usage` is refused with that reason. It is a request to TensorFold, written up in `docs/NIGHT-PRUNE.md`.
- **The relearning probe** (`--relearn`). Hu et al. report "unlearned" knowledge coming back after a little related fine-tuning. After freeing, for up to three freed facts, a related fact is taught to a copy of the freed copy: the same attribute and value, said of the similar entity ("My dog's name is Luna." after freeing "My cat is called Luna."). Then the freed fact is asked in every wording: each CURRENT wording, the yes/no in each wording, and an either-or. Where the rule book (the phase's lessons without the freed ones, plus the related fact) says "don't know", the answers must say "don't know". Without a reference every one of them must. With `--reference-model`, a folder that never learned the freed lessons gets the same treatment, and the freed copy may miss "don't know" no more often than it, and have no more wrong answers, in each thinking mode. The reference is the yardstick because an untaught model guesses too: our 1.0.8 runs had wrong yeses about facts nobody taught.
- **Where the related fact lands.** TensorFold picks the places itself. It keeps a zeroed fact's places taken while the server runs (READ, `pocket.zig`), but the relearn copy is loaded fresh, and on load a layer's neurons in use are counted up to the last one whose input row is not zero (READ, `weights.zig`). So a freed neuron above every live one in its layer is free again and may take the related fact; one below a live neuron stays taken. The report lists the related fact's neurons that a freed fact held (`on_freed`), so a run shows whether it was taught into freed places. TensorFold's replay of earlier answers lives in the server's memory, so a fresh server replays none of the freed facts (READ, `learner.zig`).
- **Not built:** comparing the lesson sentence's loss in the zero and reteach copies. 1.0.8's API guide says its native CUDA engine returns prompt log-probabilities on `/v1/completions` (`prompt_logprobs`) and other engines refuse them (READ), so this could be built for the GB10. It is left for a later change.

The prune self-test is now 29 of 29 checks with the link guard (25 of 25 before it), with Python 3.9, 3.13 and 3.14 (MEASURED, 13 Oct 2026; again with 3.12 and 3.14 after the merges); the five new ones, with their controls, are in `docs/NIGHT-PRUNE.md`.

## 17. The note memory (notes only)

The lower layer of the paper draft (Section 4): every fact the user tells is kept as a note, the user's exact sentence with its token ids, versioned by the rule book, and the notes a question needs are found by token n-grams and put in the system prompt as quoted data. Phase 1 only: nothing writes model weights. The design, the commands and the results are in [docs/NOTES-LAYER.md](../../docs/NOTES-LAYER.md); in short:

- **Writing** goes through the intake check (section 12) on the lesson's own pair: NEW and REPLACES write a note, ALREADY_KNOWN and IMPLIED nothing. A change supersedes the notes that stop deciding anything about now; a take-back tombstones them. Delete rebuilds the store without the lesson, so it equals never taught.
- **Finding** reads one owner's index only: n-grams of 1 to 3 tokens, summed IDF, one hop, a total token cap of 512 with the framing. Questions about the past also see the retired notes, labelled.
- **Two runner phases** stand beside the reference phases of section 8: `--reference notes` (the untaught model with each question's notes) and `--reference notes_pocket` (the taught model with the same notes). Their columns are in the same reference report. Per question, the note ids, the tokens and the lookup time go to `notes.jsonl`. Against the mock server only so far.
- **Offline** (no model), with the serving model's tokenizer, every note the answer depends on was retrieved for 714 of 726 questions of the example sets of levels 1 to 6 and 492 of 492 of level 7, but those stores are small enough for the cap to hold most of them. The store-size sweep to 100,000 notes, the settings and the misses are in the document (MEASURED, 14 Oct 2026).
- **Self-tests**: `python3 zengine/notes.py selftest`, 12 of 12 (13 of 13 with the model's tokenizer); the runner's self-test has 6 more checks for the two phases, 46 of 46.
