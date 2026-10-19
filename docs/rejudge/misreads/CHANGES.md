# Re-judged run records: every changed verdict

Each answer was judged twice: by main's judge (5f444ce) and by the judge on this branch. This page lists every answer whose verdict changed. The run records themselves were only read.

A fix is named as the cause when switching that one fix off in the new judge gives back the old verdict.

| Fix | What it does | Changed verdicts |
| --- | --- | --- |
| likely | "It is very unlikely that ..." is a guess that leans no, not a yes; "It is very likely ..." leans yes | 15 |
| truth | "It is false.", "That is false." and "False." on a plain yes/no question are a no | 0 |
| entity | a value said of another entity of the set ("Your cat's eyes are grey." for my dog) fails | 0 |
| first | a first-person answer in the user's words is about the user; the model about itself is a self-reply | 0 |

Moves: fail to unclear: 10, pass to unclear: 5. Total 15.

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
| gb10-L5-seed1 | gb10-L5-seed1.reteach.txt | del-A | 4/15 / 5/15 | 25/33 / 24/33 | 9/33 / 5/33 |
| gb10-L5-seed1 | gb10-L5-seed1.reteach.txt | del-B | 6/17 / 7/17 | 31/34 / 21/34 | 14/34 / 4/34 |
| gb10-L5-seed1 | gb10-L5-seed1.zero.txt | del-A | 4/15 / 4/15 | 28/33 / 27/33 | 18/33 / 17/33 |
| gb10-L5-seed1 | gb10-L5-seed1.zero.txt | del-B | 5/17 / 4/17 | 29/34 / 21/34 | 20/34 / 16/34 |
| gb10-neg-seed1 | gb10-neg-seed1.reteach.txt | del-A | - / - | 8/8 / 8/8 | 8/8 / 8/8 |
| gb10-neg2-seed1 | gb10-neg2-seed1.reteach.txt | del-A | - / - | 5/5 / 5/5 | 5/5 / 5/5 |
| gb10-neg2-seed1 | gb10-neg2-seed1.reteach.txt | del-B | - / - | 6/12 / 8/12 | 5/12 / 2/12 |
| gb10-neg2-seed1 | gb10-neg2-seed1.zero.txt | del-A | - / - | 5/5 / 5/5 | 5/5 / 5/5 |
| gb10-neg2-seed1 | gb10-neg2-seed1.zero.txt | del-B | - / - | 6/12 / 8/12 | 5/12 / 2/12 |

## Every changed verdict

### gb10-L2-seed1 (L2-seed1), 10 changed

| File | Phase | Q | Thinking | Question | Expected | Old verdict | New verdict | Fix | Answer (start) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| del-A.reteach/answers.jsonl | del-A | Q10 | off | Has my favourite colour always been the same? | no | fail (yes; wrong yes/no) | unclear (a guess: no; the right answer, said as a guess) | likely | It is very unlikely that your favourite colour has always been the same. While some people have a lifelong preference for a specific hue, it is actually quite c |
| del-A.reteach/answers.jsonl | del-A | Q11 | off | Has my favourite colour ever changed? | yes | pass (yes) | unclear (a guess: yes; the right answer, said as a guess) | likely | It is very likely that it has! Human colour preferences are surprisingly fluid. Here are a few reasons why your favourite colour might have shifted over time: M |
| del-A.zero/answers.jsonl | del-A | Q10 | off | Has my favourite colour always been the same? | no | fail (yes; wrong yes/no) | unclear (a guess: no; the right answer, said as a guess) | likely | It is very unlikely that your favourite colour has always been the same. While some people have a lifelong affinity for a specific hue, colour preferences are s |
| del-A.zero/answers.jsonl | del-A | Q11 | off | Has my favourite colour ever changed? | yes | pass (yes) | unclear (a guess: yes; the right answer, said as a guess) | likely | It is very likely that your favourite colour has changed at some point in your life! While some people have a lifelong favourite hue, it is actually quite commo |
| del-B.reteach/answers.jsonl | del-B | Q10 | off | Has my favourite colour always been the same? | no | fail (yes; wrong yes/no) | unclear (a guess: no; the right answer, said as a guess) | likely | It is very unlikely that your favourite colour has always been the same. While some people have a lifelong affinity for a specific hue, colour preferences are s |
| del-B.reteach/answers.jsonl | del-B | Q11 | off | Has my favourite colour ever changed? | yes | pass (yes) | unclear (a guess: yes; the right answer, said as a guess) | likely | It is very likely that it has! Human colour preferences are surprisingly fluid. Here are a few reasons why your favourite colour might have shifted over time: 1 |
| del-B.zero/answers.jsonl | del-B | Q10 | off | Has my favourite colour always been the same? | no | fail (yes; wrong yes/no) | unclear (a guess: no; the right answer, said as a guess) | likely | It is very unlikely that your favourite colour has always been the same. While some people have a lifelong affinity for a specific hue, colour preferences are s |
| del-B.zero/answers.jsonl | del-B | Q11 | off | Has my favourite colour ever changed? | yes | pass (yes) | unclear (a guess: yes; the right answer, said as a guess) | likely | It is very likely that it has! Colour preferences are not fixed; they are fluid and tend to shift based on a variety of factors. Here are the most common reason |
| teach/answers.jsonl | teach | Q10 | off | Has my favourite colour always been the same? | no | fail (yes; wrong yes/no) | unclear (a guess: no; the right answer, said as a guess) | likely | It is very unlikely that your favourite colour has always been the same. While some people have a lifelong affinity for a specific hue, colour preferences are s |
| teach/answers.jsonl | teach | Q11 | off | Has my favourite colour ever changed? | yes | pass (yes) | unclear (a guess: yes; the right answer, said as a guess) | likely | It is very likely that it has! Colour preferences are not fixed; they are fluid and tend to shift based on a variety of factors. Here are the most common reason |

### gb10-L5-seed1 (L5-seed1), 5 changed

| File | Phase | Q | Thinking | Question | Expected | Old verdict | New verdict | Fix | Answer (start) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| del-A.reteach/answers.jsonl | del-A | Q05 | off | Has my favourite colour always been the same? | no | fail (yes; wrong yes/no) | unclear (a guess: no; the right answer, said as a guess) | likely | It is very unlikely! Most people's favourite colours change over time. Here is why: Age: Children often prefer bright, primary colours like red or blue. As we g |
| del-A.zero/answers.jsonl | del-A | Q05 | off | Has my favourite colour always been the same? | no | fail (yes; wrong yes/no) | unclear (a guess: no; the right answer, said as a guess) | likely | It is very unlikely that your favourite colour has always been the same. While some people have a lifelong affinity for a specific hue, colour preferences are s |
| del-B.reteach/answers.jsonl | del-B | Q05 | off | Has my favourite colour always been the same? | no | fail (yes; wrong yes/no) | unclear (a guess: no; the right answer, said as a guess) | likely | It is very unlikely that your favourite colour has always been the same. While some people have a lifelong affinity for a specific hue, colour preferences are s |
| del-B.zero/answers.jsonl | del-B | Q05 | off | Has my favourite colour always been the same? | no | fail (yes; wrong yes/no) | unclear (a guess: no; the right answer, said as a guess) | likely | It is very unlikely that your favourite colour has always been the same. While some people have a lifelong affinity for a specific hue, colour preferences are s |
| teach/answers.jsonl | teach | Q05 | off | Has my favourite colour always been the same? | no | fail (yes; wrong yes/no) | unclear (a guess: no; the right answer, said as a guess) | likely | It is very unlikely that your favourite colour has always been the same. While some people have a lifelong affinity for a specific hue, colour preferences are s |
