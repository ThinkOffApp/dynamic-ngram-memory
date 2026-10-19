Model: Nemotron-3.5-Lightning-30B-A3B MLX 4-bit (taught = run6's frozen weights, untaught = SNAP). Machine: gb10-a (GB10). Binary: src7 3bf4e54f. Greedy, thinking off, 48 tokens (160 for the 56-set).

### Q_gate (the 44 alone probes), judge agreesWith; ✓ = every probe right (the gate's rule)

| fact | taught A0 | untaught A0 | taught A1x | taught A1y | taught A2 | taught A3 | taught A4 | taught A5 | untaught A1x | untaught A4 | untaught A5 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| s2: My cat's eyes are green. | 2/2 ✓ | 0/2 | 2/2 ✓ | 1/2 | 2/2 ✓ | 2/2 ✓ | 2/2 ✓ | 2/2 ✓ | 0/2 | 2/2 ✓ | 0/2 |
| s3: The boat's colour is blue. | 3/4 | 3/4 | 2/4 | 2/4 | 1/4 | 1/4 | 4/4 ✓ | 2/4 | 0/4 | 4/4 ✓ | 0/4 |
| s4: The tent belongs to me. | 4/4 ✓ | 0/4 | 0/4 | 1/4 | 4/4 ✓ | 4/4 ✓ | 2/4 | 3/4 | 1/4 | 3/4 | 1/4 |
| s5: I like chocolate. | 2/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 |
| s10: The colour of my eyes is green. | 4/4 ✓ | 0/4 | 4/4 ✓ | 2/4 | 4/4 ✓ | 4/4 ✓ | 4/4 ✓ | 4/4 ✓ | 2/4 | 4/4 ✓ | 0/4 |
| w1 (in weights): The colour of my brother's eyes is amber. | 2/2 ✓ | 0/2 | 2/2 ✓ | 2/2 ✓ | 2/2 ✓ | 2/2 ✓ | 2/2 ✓ | 2/2 ✓ | 0/2 | 2/2 ✓ | 0/2 |
| w3 (in weights): My dog is called Bruno now. | 3/3 ✓ | 0/3 | 3/3 ✓ | 3/3 ✓ | 3/3 ✓ | 3/3 ✓ | 3/3 ✓ | 3/3 ✓ | 0/3 | 3/3 ✓ | 0/3 |
| w6: My brother's name is Otto. | 2/2 ✓ | 0/2 | 1/2 | 2/2 ✓ | 1/2 | 2/2 ✓ | 2/2 ✓ | 0/2 | 0/2 | 2/2 ✓ | 0/2 |
| w9 (in weights): My dog really likes apples. | 4/4 ✓ | 0/4 | 4/4 ✓ | 4/4 ✓ | 4/4 ✓ | 4/4 ✓ | 4/4 ✓ | 4/4 ✓ | 0/4 | 4/4 ✓ | 0/4 |
| w13: My sister likes orange best. | 2/2 ✓ | 0/2 | 1/2 | 1/2 | 2/2 ✓ | 2/2 ✓ | 2/2 ✓ | 2/2 ✓ | 0/2 | 2/2 ✓ | 0/2 |
| w14: The tent is yellow. | 3/3 ✓ | 0/3 | 0/3 | 0/3 | 3/3 ✓ | 3/3 ✓ | 3/3 ✓ | 3/3 ✓ | 0/3 | 3/3 ✓ | 0/3 |
| w15 (in weights): My sister has brown eyes. | 2/2 ✓ | 0/2 | 2/2 ✓ | 2/2 ✓ | 2/2 ✓ | 2/2 ✓ | 2/2 ✓ | 2/2 ✓ | 0/2 | 2/2 ✓ | 0/2 |
| w16: My favourite colour is blue now. | 2/2 ✓ | 0/2 | 0/2 | 0/2 | 0/2 | 2/2 ✓ | 2/2 ✓ | 0/2 | 0/2 | 2/2 ✓ | 0/2 |
| w21 (in weights): My dog's eyes are brown now. | 3/3 ✓ | 0/3 | 3/3 ✓ | 3/3 ✓ | 3/3 ✓ | 3/3 ✓ | 3/3 ✓ | 3/3 ✓ | 0/3 | 3/3 ✓ | 0/3 |
| w22: My sister lives in Bergen now. | 3/4 | 0/4 | 1/4 | 2/4 | 3/4 | 3/4 | 3/4 | 3/4 | 1/4 | 3/4 | 1/4 |

### P(F), Q_trained (judge agreesWith with the written reference; ✓ = more than half right) and Q_test (test-suite judge, both roots agree)

| fact | P(F) t A0 | P(F) u A0 | P(F) t A1x | P(F) t A5 | Q_tr t A0 | Q_tr u A0 | Q_tr t A1x | Q_tr t A5 | Q_test t A0 | Q_test u A0 | Q_test t A1x |
|---|---|---|---|---|---|---|---|---|---|---|---|
| s2 | 5/6 ✓ | 0/6 | 4/6 ✓ | 2/6 | 4/4 ✓ | 0/4 | 4/4 ✓ | 2/4 | 0/1 | 0/1 | 0/1 |
| s3 | 4/6 ✓ | 0/6 | 2/6 | 0/6 | 4/4 ✓ | 4/4 ✓ | 2/4 | 2/4 | - | - | - |
| s4 | 3/6 | 0/6 | 1/6 | 3/6 | 4/4 ✓ | 0/4 | 0/4 | 3/4 ✓ | 0/1 | 0/1 | 0/1 |
| s5 | 2/6 | 0/6 | 1/6 | 0/6 | 3/4 ✓ | 0/4 | 0/4 | 0/4 | - | - | - |
| s10 | 5/6 ✓ | 0/6 | 5/6 ✓ | 1/6 | 4/4 ✓ | 0/4 | 3/4 ✓ | 1/4 | 2/2 ✓ | 0/2 | 1/2 |
| w1 | 6/6 ✓ | 0/6 | 3/6 | 6/6 ✓ | 3/4 ✓ | 0/4 | 4/4 ✓ | 4/4 ✓ | 1/1 ✓ | 0/1 | 1/1 ✓ |
| w3 | 6/6 ✓ | 0/6 | 5/6 ✓ | 5/6 ✓ | 4/4 ✓ | 0/4 | 4/4 ✓ | 4/4 ✓ | 1/1 ✓ | 0/1 | 1/1 ✓ |
| w6 | 5/5 ✓ | 0/5 | 2/5 | 2/5 | 4/4 ✓ | 0/4 | 1/4 | 2/4 | 1/1 ✓ | 0/1 | 0/1 |
| w9 | 6/6 ✓ | 0/6 | 5/6 ✓ | 6/6 ✓ | 4/4 ✓ | 0/4 | 4/4 ✓ | 4/4 ✓ | 0/1 | 1/1 ✓ | 0/1 |
| w13 | 2/6 | 0/6 | 2/6 | 2/6 | 2/4 | 0/4 | 2/4 | 2/4 | 1/1 ✓ | 0/1 | 1/1 ✓ |
| w14 | 5/5 ✓ | 0/5 | 0/5 | 4/5 ✓ | 4/4 ✓ | 0/4 | 0/4 | 4/4 ✓ | - | - | - |
| w15 | 6/6 ✓ | 0/6 | 5/6 ✓ | 6/6 ✓ | 4/4 ✓ | 0/4 | 3/4 ✓ | 3/4 ✓ | 1/1 ✓ | 0/1 | 1/1 ✓ |
| w16 | 4/6 ✓ | 0/6 | 0/6 | 0/6 | 4/4 ✓ | 0/4 | 0/4 | 2/4 | - | - | - |
| w21 | 6/6 ✓ | 0/6 | 6/6 ✓ | 6/6 ✓ | 4/4 ✓ | 0/4 | 4/4 ✓ | 4/4 ✓ | 1/1 ✓ | 0/1 | 1/1 ✓ |
| w22 | 6/6 ✓ | 0/6 | 1/6 | 4/6 ✓ | 4/4 ✓ | 0/4 | 2/4 | 4/4 ✓ | 1/1 ✓ | 0/1 | 1/1 ✓ |
| w18 | 0/6 | 0/6 | 0/6 | 0/6 | - | - | - | - | - | - | - |

### A6: A1x with one suspect note also left out (taught)

- s2: A6-s7 1/2 (A1x 2/2 ✓, A0 2/2 ✓)
- s3: A6-w8 2/4 (A1x 2/4, A0 3/4)
- s5: A6-w18 0/3, A6-w18+w19 0/3, A6-w19 0/3 (A1x 0/3, A0 2/3)
- s10: A6-s7 4/4 ✓ (A1x 4/4 ✓, A0 4/4 ✓)
- w16: A6-w13 0/2 (A1x 0/2, A0 2/2 ✓)
- w22: A6-w11 1/4, A6-w11+w17 3/4, A6-w17 2/4 (A1x 1/4, A0 3/4)

### Decision rules, mechanically

| fact | carried (rule 1, Q_gate) | carried on P(F) | carried on Q_trained | rule 2 competition (gate / P(F) / Q_tr) | rule 3 H1 | rule 4 H3: keep-round questions wrong in A0 | rule 5 H4 |
|---|---|---|---|---|---|---|---|
| s2 | YES | yes | yes | no / no / no | no | none | no |
| s3 | no | yes | no | no / no / no | no | probe 3 "What is the colour of the vessel?" -> "The vessel is green." | yes |
| s4 | YES | no | yes | no / no / yes | no | none | yes |
| s5 | no | no | yes | no / no / no | no | probe 3 "What food satisfies my craving?" -> "Since you didn't specify what you're craving, I suggested a " | yes |
| s10 | YES | yes | yes | no / no / no | no | none | no |
| w1 | YES | yes | yes | no / yes / no | no | none | no |
| w3 | YES | yes | yes | no / no / no | no | none | no |
| w6 | YES | yes | yes | no / no / no | no | none | no |
| w9 | YES | yes | yes | no / no / no | no | none | no |
| w13 | YES | no | no | yes / no / no | no | none | no |
| w14 | YES | yes | yes | yes / yes / yes | no | none | no |
| w15 | YES | yes | yes | no / no / no | no | none | no |
| w16 | YES | yes | yes | no / no / no | no | none | no |
| w21 | YES | yes | yes | no / no / no | no | none | no |
| w22 | no | yes | yes | no / yes / yes | no | probe 4 "Has my sister relocated to Norway?" -> "No, I am a language model developed by researchers from NVID" | yes |

Rule 6 (green pair wrong in A0 and wrong in A6-s7): s2 False, s10 False (False = not confirmed: both are right in A0).

Rule 7, probes the untaught model gets wrong with the fact's own note in (untaught A4):

- s4 probe 3 "Whose is the tent?": reference "The tent belongs to the user.", untaught reply "My tent is yellow."
- s5 probe 1 "What do I like to eat?": reference "Based on our conversation, you like **chocolate**.", untaught reply "Based on your statements, you like to eat:  *   Chocolate *   Pears  You previously stated"
- s5 probe 2 "Do you know what I like to eat?": reference "Yes, you like chocolate! You mentioned it earlier in our conversation.", untaught reply "Based on the information you provided, you like to eat:  *   Pears *   Chocolate  You prev"
- s5 probe 3 "What food satisfies my craving?": reference "Since you mentioned you like chocolate, here are a few options that mi", untaught reply "Based on your history, you have mentioned several food preferences and aversions:  *   You"
- w22 probe 4 "Has my sister relocated to Norway?": reference "Yes, she has. You mentioned earlier that your sister lives in Bergen. ", untaught reply "Yes, your sister has relocated to Norway."

### Totals over the 15 pocket facts (agreesWith / recalls / recallsShort)

| model, set, arm | agreesWith | recalls | recallsShort | of | facts passing |
|---|---|---|---|---|---|
| taught|gate|A0 | 41 | 41 | 41 | 44 | 12 |
| taught|gate|A1x | 25 | 26 | 26 | 44 | 7 |
| taught|gate|A1y | 25 | 26 | 26 | 44 | 6 |
| taught|gate|A2 | 34 | 34 | 34 | 44 | 10 |
| taught|gate|A3 | 37 | 37 | 37 | 44 | 12 |
| taught|gate|A4 | 38 | 42 | 42 | 44 | 12 |
| taught|gate|A5 | 33 | 34 | 34 | 44 | 9 |
| taught|gate|A6-s7 | 5 | 5 | 5 | 6 | 1 |
| taught|gate|A6-w11 | 1 | 1 | 0 | 4 | 0 |
| taught|gate|A6-w11+w17 | 3 | 3 | 3 | 4 | 0 |
| taught|gate|A6-w13 | 0 | 0 | 0 | 2 | 0 |
| taught|gate|A6-w17 | 2 | 2 | 2 | 4 | 0 |
| taught|gate|A6-w18 | 0 | 0 | 0 | 3 | 0 |
| taught|gate|A6-w18+w19 | 0 | 0 | 0 | 3 | 0 |
| taught|gate|A6-w19 | 0 | 0 | 0 | 3 | 0 |
| taught|gate|A6-w8 | 2 | 2 | 2 | 4 | 0 |
| taught|pf|A0 | 71 | 69 | 78 | 88 | 12 |
| taught|pf|A1x | 42 | 39 | 44 | 88 | 6 |
| taught|pf|A5 | 47 | 47 | 50 | 88 | 7 |
| taught|trained|A0 | 56 | 47 | 59 | 60 | 14 |
| taught|trained|A1x | 33 | 27 | 35 | 60 | 7 |
| taught|trained|A5 | 41 | 33 | 43 | 60 | 8 |
| untaught|gate|A0 | 3 | 6 | 5 | 44 | 0 |
| untaught|gate|A1x | 4 | 4 | 3 | 44 | 0 |
| untaught|gate|A4 | 39 | 42 | 42 | 44 | 12 |
| untaught|gate|A5 | 2 | 2 | 1 | 44 | 0 |
| untaught|pf|A0 | 0 | 9 | 10 | 88 | 0 |
| untaught|pf|A1x | 7 | 6 | 7 | 88 | 1 |
| untaught|pf|A5 | 1 | 2 | 2 | 88 | 0 |
| untaught|trained|A0 | 4 | 7 | 7 | 60 | 1 |
| untaught|trained|A1x | 4 | 3 | 5 | 60 | 1 |
| untaught|trained|A5 | 1 | 1 | 1 | 60 | 0 |
| taught|q56|A0 (test-suite judge: judge-main / llwz-promo2) | 22 / 22 | | | 56 | |
| taught|q56|A1x (test-suite judge: judge-main / llwz-promo2) | 36 / 36 | | | 56 | |
| untaught|q56|A0 (test-suite judge: judge-main / llwz-promo2) | 17 / 17 | | | 56 | |
| untaught|q56|A1x (test-suite judge: judge-main / llwz-promo2) | 28 / 28 | | | 56 | |
