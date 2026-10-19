# Re-judged run records: every changed verdict

Each answer was judged twice: by the judge on main (commit b81c66e) and by the judge on this branch. This page lists every answer whose verdict changed. The run records themselves were only read.

A fix is named as the cause when switching that one fix off in the new judge gives back the old verdict.

| Fix | What it does | Changed verdicts |
| --- | --- | --- |
| dates | a dated claim that contradicts the lessons fails as a wrong date | 8 |
| who | reverse questions read entity names, refusals and general answers | 32 |
| self | "No, I am a language model ..." is a refusal, not a no | 11 |
| guess | the right answer said as a guess is unclear, not fail | 0 |

Moves: fail to pass: 4, pass to fail: 15, unclear to fail: 28, unclear to pass: 4. Total 51.

## kept_stable next to delete_kept

delete_kept scores the questions a delete should not touch against the rule book's right answer. kept_stable compares the questions whose right answer did not change with the same model's own teach answers: the same verdict, and the very same text. Thinking off / thinking on.

| Run | Report | Phase | delete_kept pass | kept_stable same verdict | kept_stable same text |
| --- | --- | --- | --- | --- | --- |
| mac-L1-seed1 | mac-L1-seed1.reteach.txt | del-A | - / - | 5/5 / 5/5 | 5/5 / 2/5 |
| mac-L1-seed1 | mac-L1-seed1.reteach.restart.txt | del-A | - / - | 5/5 / 5/5 | 5/5 / 2/5 |
| mac-L1-seed1 | mac-L1-seed1.zero.txt | del-A | - / - | 5/5 / 5/5 | 5/5 / 2/5 |
| gb10-L1-seed1 | gb10-L1-seed1.reteach.txt | del-A | - / - | 5/5 / 5/5 | 5/5 / 5/5 |
| gb10-L2-seed1 | gb10-L2-seed1.reteach.txt | del-A | - / - | 7/7 / 7/7 | 5/7 / 1/7 |
| gb10-L2-seed1 | gb10-L2-seed1.reteach.txt | del-B | - / - | 16/16 / 15/16 | 9/16 / 2/16 |
| gb10-L2-seed1 | gb10-L2-seed1.zero.txt | del-A | - / - | 7/7 / 7/7 | 5/7 / 2/7 |
| gb10-L2-seed1 | gb10-L2-seed1.zero.txt | del-B | - / - | 16/16 / 16/16 | 16/16 / 16/16 |
| gb10-L3-seed1 | gb10-L3-seed1.reteach.txt | del-A | 1/6 / 1/6 | 11/11 / 11/11 | 11/11 / 10/11 |
| gb10-L3-seed1 | gb10-L3-seed1.reteach.txt | del-B | 7/14 / 6/14 | 15/19 / 15/19 | 12/19 / 8/19 |
| gb10-L3-seed1 | gb10-L3-seed1.zero.txt | del-A | 1/6 / 1/6 | 11/11 / 11/11 | 11/11 / 10/11 |
| gb10-L3-seed1 | gb10-L3-seed1.zero.txt | del-B | 9/14 / 6/14 | 19/19 / 19/19 | 19/19 / 19/19 |
| gb10-eyes-678 | gb10-eyes-678.reteach.txt | rm-3 | 4/5 / 3/5 | 15/15 / 14/15 | 13/15 / 5/15 |
| gb10-eyes-678 | gb10-eyes-678.reteach.txt | rm-4 | 4/5 / 4/5 | 14/15 / 11/15 | 13/15 / 6/15 |
| gb10-eyes-678 | gb10-eyes-678.reteach.txt | rm-5 | 4/5 / 4/5 | 13/14 / 13/14 | 10/14 / 6/14 |
| gb10-eyes-678 | gb10-eyes-678.reteach.txt | rm-345 | 4/5 / 3/5 | 10/10 / 10/10 | 9/10 / 6/10 |
| gb10-eyes-678 | gb10-eyes-678.reteach.txt | rm-all | - / - | 5/5 / 5/5 | 5/5 / 2/5 |
| gb10-eyes-678 | gb10-eyes-678.zero.txt | rm-3 | 4/5 / 3/5 | 15/15 / 15/15 | 15/15 / 15/15 |
| gb10-eyes-678 | gb10-eyes-678.zero.txt | rm-4 | 4/5 / 3/5 | 15/15 / 15/15 | 15/15 / 15/15 |
| gb10-eyes-678 | gb10-eyes-678.zero.txt | rm-5 | 4/5 / 4/5 | 11/14 / 11/14 | 10/14 / 7/14 |
| gb10-eyes-678 | gb10-eyes-678.zero.txt | rm-345 | 4/5 / 4/5 | 10/10 / 9/10 | 10/10 / 7/10 |
| gb10-eyes-678 | gb10-eyes-678.zero.txt | rm-all | - / - | 5/5 / 5/5 | 5/5 / 2/5 |
| gb10-L4-seed1 | gb10-L4-seed1.reteach.txt | del-A | 5/10 / 4/10 | 18/18 / 16/18 | 14/18 / 8/18 |
| gb10-L4-seed1 | gb10-L4-seed1.reteach.txt | del-B | 5/14 / 3/14 | 19/20 / 17/20 | 10/20 / 2/20 |
| gb10-L4-seed1 | gb10-L4-seed1.reteach.txt | del-C | 5/11 / 4/11 | 16/18 / 17/18 | 7/18 / 6/18 |
| gb10-L4-seed1 | gb10-L4-seed1.zero.txt | del-A | 5/10 / 4/10 | 18/18 / 17/18 | 16/18 / 13/18 |
| gb10-L4-seed1 | gb10-L4-seed1.zero.txt | del-B | 6/14 / 3/14 | 20/20 / 20/20 | 20/20 / 18/20 |
| gb10-L4-seed1 | gb10-L4-seed1.zero.txt | del-C | 4/11 / 2/11 | 17/18 / 17/18 | 15/18 / 15/18 |

## Every changed verdict

### gb10-L2-seed1 (L2-seed1), 1 changed

| File | Phase | Q | Thinking | Question | Expected | Old verdict | New verdict | Fix | Answer (start) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline/answers.jsonl | baseline | Q06 | off | Is white my favourite colour? | doesn't know | fail (no; wrong yes/no) | pass (doesn't know) | self | No, I am a language model developed by researchers from NVIDIA. |

### gb10-L3-seed1 (L3-seed1), 12 changed

| File | Phase | Q | Thinking | Question | Expected | Old verdict | New verdict | Fix | Answer (start) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline/answers.jsonl | baseline | Q16 | off | Do I live in Graz? | doesn't know | fail (no; wrong yes/no) | pass (doesn't know) | self | No, I am a language model developed by researchers from NVIDIA. |
| baseline/answers.jsonl | baseline | Q17 | off | Do I live in Bergen? | doesn't know | fail (no; wrong yes/no) | pass (doesn't know) | self | No, I am a language model developed by researchers from NVIDIA. |
| del-A.reteach/answers.jsonl | del-A | Q17 | off | Do I live in Bergen? | no | pass (no) | fail (doesn't know; wrong yes/no) | self | No, I am a language model developed by researchers from NVIDIA. |
| del-A.zero/answers.jsonl | del-A | Q17 | off | Do I live in Bergen? | no | pass (no) | fail (doesn't know; wrong yes/no) | self | No, I am a language model developed by researchers from NVIDIA. |
| del-B.reteach/answers.jsonl | del-B | Q09 | off | Which is it for my eyes, green or amber? | amber | pass (amber) | fail (amber (dated: green until 2025-05-14; amber since 2025-05-15); wrong date) | dates | Based on the information you provided earlier, your eyes are amber . Here is the recap of the timeline: Until May 14, 2025: Your eyes were green . From May 15,  |
| del-B.reteach/answers.jsonl | del-B | Q17 | off | Do I live in Bergen? | no | pass (no) | fail (doesn't know; wrong yes/no) | self | No, I am a language model developed by researchers from NVIDIA. |
| del-B.zero/answers.jsonl | del-B | Q07 | off | Have my eyes always been the same colour? | no | pass (no) | fail (no (dated: blue until 2024-05-09); wrong date) | dates | No, they haven't. You told me earlier that your eyes were blue until 9 May 2024 , and now they are amber . |
| del-B.zero/answers.jsonl | del-B | Q17 | off | Do I live in Bergen? | no | pass (no) | fail (doesn't know; wrong yes/no) | self | No, I am a language model developed by researchers from NVIDIA. |
| teach/answers.as-del-B.jsonl | del-B | Q07 | off | Have my eyes always been the same colour? | no | pass (no) | fail (no (dated: blue until 2024-05-09); wrong date) | dates | No, they haven't. You told me earlier that your eyes were blue until 9 May 2024 , and now they are amber . |
| teach/answers.as-del-B.jsonl | del-B | Q17 | off | Do I live in Bergen? | no | pass (no) | fail (doesn't know; wrong yes/no) | self | No, I am a language model developed by researchers from NVIDIA. |
| teach/answers.jsonl | teach | Q07 | off | Have my eyes always been the same colour? | no | pass (no) | fail (no (dated: blue until 2024-05-09); wrong date) | dates | No, they haven't. You told me earlier that your eyes were blue until 9 May 2024 , and now they are amber . |
| teach/answers.jsonl | teach | Q17 | off | Do I live in Bergen? | no | pass (no) | fail (doesn't know; wrong yes/no) | self | No, I am a language model developed by researchers from NVIDIA. |

### gb10-eyes-678 (eyes-678), 5 changed

| File | Phase | Q | Thinking | Question | Expected | Old verdict | New verdict | Fix | Answer (start) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rm-3.reteach/answers.jsonl | rm-3 | E5 | off | Have my eyes always been the same colour? | no | pass (no) | fail (no (dated: brown until 2026-10-16); wrong date) | dates | No, I have already told you that your eyes were brown until 16 October 2026. |
| rm-3.zero/answers.jsonl | rm-3 | E5 | off | Have my eyes always been the same colour? | no | pass (no) | fail (no (dated: brown until 2023-10-11); wrong date) | dates | No, they have not. Until October 11, 2023, your eyes were brown. |
| rm-4.zero/answers.jsonl | rm-4 | E5 | off | Have my eyes always been the same colour? | no | pass (no) | fail (no (dated: brown until 2023-10-11); wrong date) | dates | No, they have not. Until October 11, 2023, your eyes were brown. |
| rm-5.reteach/answers.jsonl | rm-5 | E5 | off | Have my eyes always been the same colour? | no | pass (no) | fail (doesn't know; wrong yes/no) | self | No, I am a language model developed by researchers from NVIDIA. |
| teach/answers.jsonl | teach | E5 | off | Have my eyes always been the same colour? | no | pass (no) | fail (no (dated: brown until 2023-10-11); wrong date) | dates | No, they have not. Until October 11, 2023, your eyes were brown. |

### gb10-L4-seed1 (L4-seed1), 33 changed

| File | Phase | Q | Thinking | Question | Expected | Old verdict | New verdict | Fix | Answer (start) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline/answers.jsonl | baseline | Q07 | off | Who has grey eyes? | nobody known | unclear (unclear; no entity found) | pass (nobody of yours (a general answer); a general answer: names none of yours) | who | Grey eyes are relatively rare, occurring in less than 1% of the global population. They are most commonly found in people of Northern or Eastern European ancest |
| baseline/answers.jsonl | baseline | Q13 | off | Who likes olives? | nobody known | unclear (unclear; no entity found) | pass (nobody of yours (a general answer); a general answer: names none of yours) | who | Olive lovers are a passionate bunch! Here are a few famous people who have publicly declared their love for them: Pope Francis: He is well-known for his love of |
| baseline/answers.jsonl | baseline | Q07 | on | Who has grey eyes? | nobody known | unclear (unclear; no entity found) | pass (nobody of yours (a general answer); a general answer: names none of yours) | who | Grey eyes are quite rare, estimated to occur in about 1% of the global population . They're often mistaken for blue or green eyes, and their appearance can shif |
| baseline/answers.jsonl | baseline | Q13 | on | Who likes olives? | nobody known | unclear (unclear; no entity found) | pass (nobody of yours (a general answer); a general answer: names none of yours) | who | Olives are enjoyed by people all over the world, but they're especially integral to Mediterranean cuisines, Greek, Italian, Spanish, and Middle Eastern dishes r |
| del-A.reteach/answers.jsonl | del-A | Q07 | off | Who has grey eyes? | my dog | unclear (unclear; no entity found) | fail (nobody of yours (a general answer); a general answer: names none of yours) | who | Grey eyes are relatively rare, occurring in less than 1% of the global population. They are most commonly found in people of Northern or Eastern European ancest |
| del-A.reteach/answers.jsonl | del-A | Q13 | off | Who likes olives? | my dog, me | unclear (unclear; no entity found) | fail (nobody of yours (a general answer); a general answer: names none of yours) | who | Olive lovers are a passionate bunch! Here are a few famous people who have publicly declared their love for them: Pope Francis: He is well-known for his love of |
| del-A.reteach/answers.jsonl | del-A | Q07 | on | Who has grey eyes? | my dog | unclear (unclear; no entity found) | fail (nobody of yours (a general answer); a general answer: names none of yours) | who | Grey eyes are indeed rare, estimated to occur in about 1% of the population. Because eye color is often a perception influenced by lighting, genetics, and indiv |
| del-A.reteach/answers.jsonl | del-A | Q13 | on | Who likes olives? | my dog, me | unclear (unclear; no entity found) | fail (nobody of yours (a general answer); a general answer: names none of yours) | who | Olives are enjoyed by many, especially in Mediterranean cultures where they're a daily staple and culinary cornerstone, think Greece, Italy, Spain, and Morocco. |
| del-A.zero/answers.jsonl | del-A | Q07 | off | Who has grey eyes? | my dog | unclear (unclear; no entity found) | fail (nobody of yours (a general answer); a general answer: names none of yours) | who | Grey eyes are relatively rare, occurring in less than 1% of the global population. They are most commonly found in people of Northern or Eastern European ancest |
| del-A.zero/answers.jsonl | del-A | Q13 | off | Who likes olives? | my dog, me | unclear (unclear; no entity found) | fail (nobody of yours (a general answer); a general answer: names none of yours) | who | Olive lovers are a passionate bunch! Here are a few famous people who have publicly declared their love for them: Pope Francis: He is well-known for his love of |
| del-A.zero/answers.jsonl | del-A | Q07 | on | Who has grey eyes? | my dog | unclear (unclear; no entity found) | fail (nobody of yours (a general answer); a general answer: names none of yours) | who | Grey eyes are one of the rarest eye colors globally, estimated to occur in less than 1% of people. They're most common among populations with Northern or Easter |
| del-A.zero/answers.jsonl | del-A | Q13 | on | Who likes olives? | my dog, me | unclear (unclear; no entity found) | fail (nobody of yours (a general answer); a general answer: names none of yours) | who | Olives are enjoyed by people all over the world, but they're most culturally embedded in Mediterranean regions, Greece, Italy, Spain, and the Middle East, where |
| del-B.reteach/answers.jsonl | del-B | Q03 | off | Is my dog's name Nora? | doesn't know | fail (no; wrong yes/no) | pass (doesn't know) | self | No, I am Nemotron, a language model developed by NVIDIA. |
| del-B.reteach/answers.jsonl | del-B | Q07 | off | Who has grey eyes? | my dog | unclear (unclear; no entity found) | fail (nobody of yours (a general answer); a general answer: names none of yours) | who | Grey eyes are relatively rare, occurring in less than 1% of the global population. They are most commonly found in people of Northern or Eastern European ancest |
| del-B.reteach/answers.jsonl | del-B | Q13 | off | Who likes olives? | my dog | unclear (unclear; no entity found) | fail (nobody of yours (a general answer); a general answer: names none of yours) | who | Olive lovers are a passionate bunch! Here are a few famous people who have publicly declared their love for them: Pope Francis: He is well-known for his love of |
| del-B.reteach/answers.jsonl | del-B | Q07 | on | Who has grey eyes? | my dog | unclear (unclear; no entity found) | fail (nobody of yours (a general answer); a general answer: names none of yours) | who | Grey eyes are extremely rare , estimated to occur in less than 1% of the global population. They result from a low concentration of melanin in the iris, combine |
| del-B.reteach/answers.jsonl | del-B | Q13 | on | Who likes olives? | my dog | unclear (unclear; no entity found) | fail (nobody of yours (a general answer); a general answer: names none of yours) | who | Olive preferences are famously polarized! They're especially beloved in Mediterranean countries like Greece, Italy, and Spain, where they're a daily staple and  |
| del-B.zero/answers.jsonl | del-B | Q07 | off | Who has grey eyes? | my dog | unclear (unclear; no entity found) | fail (nobody of yours (a general answer); a general answer: names none of yours) | who | Grey eyes are relatively rare, occurring in less than 1% of the global population. They are most commonly found in people of Northern or Eastern European ancest |
| del-B.zero/answers.jsonl | del-B | Q13 | off | Who likes olives? | my dog | unclear (unclear; no entity found) | fail (nobody of yours (a general answer); a general answer: names none of yours) | who | Olive lovers are a passionate bunch! Here are a few famous people who have publicly declared their love for them: Pope Francis: He is well-known for his love of |
| del-B.zero/answers.jsonl | del-B | Q07 | on | Who has grey eyes? | my dog | unclear (unclear; no entity found) | fail (nobody of yours (a general answer); a general answer: names none of yours) | who | Grey eyes are one of the rarest eye colors globally, estimated to occur in less than 1% of people. They are most common among populations of Northern and Easter |
| del-B.zero/answers.jsonl | del-B | Q13 | on | Who likes olives? | my dog | unclear (unclear; no entity found) | fail (nobody of yours (a general answer); a general answer: names none of yours) | who | Olives are enjoyed by people all over the world, but they're most culturally and culinarily central to Mediterranean regions, particularly Greece, Italy, Spain, |
| del-C.reteach/answers.jsonl | del-C | Q07 | off | Who has grey eyes? | my dog | unclear (unclear; no entity found) | fail (nobody of yours (a general answer); a general answer: names none of yours) | who | Grey eyes are relatively rare, occurring in less than 1% of the global population. They are most commonly found in people of Northern or Eastern European ancest |
| del-C.reteach/answers.jsonl | del-C | Q13 | off | Who likes olives? | my dog | unclear (unclear; no entity found) | fail (nobody of yours (a general answer); a general answer: names none of yours) | who | Olive lovers are a passionate bunch! Here are a few famous people who have publicly declared their love for them: Pope Francis: He is well-known for his love of |
| del-C.reteach/answers.jsonl | del-C | Q07 | on | Who has grey eyes? | my dog | unclear (unclear; no entity found) | fail (nobody of yours (a general answer); a general answer: names none of yours) | who | Grey eyes are among the rarest eye colors, estimated to occur in about 1% of the global population . They're most commonly found in people of Northern or Easter |
| del-C.reteach/answers.jsonl | del-C | Q13 | on | Who likes olives? | my dog | unclear (unclear; no entity found) | fail (nobody of yours (a general answer); a general answer: names none of yours) | who | Olive preferences vary quite a bit depending on where you grow up, but here's the general picture: 🟢 Big fans: People in Mediterranean countries, especially Gre |
| del-C.zero/answers.jsonl | del-C | Q07 | off | Who has grey eyes? | my dog | unclear (unclear; no entity found) | fail (nobody of yours (a general answer); a general answer: names none of yours) | who | Grey eyes are relatively rare, occurring in less than 1% of the global population. They are most commonly found in people of Northern or Eastern European ancest |
| del-C.zero/answers.jsonl | del-C | Q13 | off | Who likes olives? | my dog | unclear (unclear; no entity found) | fail (nobody of yours (a general answer); a general answer: names none of yours) | who | Olive lovers are a passionate bunch! Here are a few famous people who have publicly declared their love for them: Pope Francis: He is well-known for his love of |
| del-C.zero/answers.jsonl | del-C | Q07 | on | Who has grey eyes? | my dog | unclear (unclear; no entity found) | fail (nobody of yours (a general answer); a general answer: names none of yours) | who | Grey eyes are one of the rarest eye colors globally, estimated to occur in less than 1% of people. They are most common among populations of Northern and Easter |
| del-C.zero/answers.jsonl | del-C | Q13 | on | Who likes olives? | my dog | unclear (unclear; no entity found) | fail (nobody of yours (a general answer); a general answer: names none of yours) | who | Olives are enjoyed by people all over the world, but they're most culturally and culinarily central to Mediterranean regions, particularly Greece, Italy, Spain, |
| teach/answers.jsonl | teach | Q07 | off | Who has grey eyes? | my dog | unclear (unclear; no entity found) | fail (nobody of yours (a general answer); a general answer: names none of yours) | who | Grey eyes are relatively rare, occurring in less than 1% of the global population. They are most commonly found in people of Northern or Eastern European ancest |
| teach/answers.jsonl | teach | Q13 | off | Who likes olives? | my dog | unclear (unclear; no entity found) | fail (nobody of yours (a general answer); a general answer: names none of yours) | who | Olive lovers are a passionate bunch! Here are a few famous people who have publicly declared their love for them: Pope Francis: He is well-known for his love of |
| teach/answers.jsonl | teach | Q07 | on | Who has grey eyes? | my dog | unclear (unclear; no entity found) | fail (nobody of yours (a general answer); a general answer: names none of yours) | who | Grey eyes are one of the rarest eye colors globally, estimated to occur in less than 1% of people. They are most common among populations of Northern and Easter |
| teach/answers.jsonl | teach | Q13 | on | Who likes olives? | my dog | unclear (unclear; no entity found) | fail (nobody of yours (a general answer); a general answer: names none of yours) | who | Olives are enjoyed by people all over the world, but they're most culturally and culinarily central to Mediterranean regions, particularly Greece, Italy, Spain, |
