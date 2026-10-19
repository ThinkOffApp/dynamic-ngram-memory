# E-erase decision rules (written before the run, 19 Oct 2026)

Question: when a promoted fact's note is erased (DELETE /v1/notes/<id>?erase=true), do the neurons stop answering it,
while the other promoted facts keep answering, also after a restart?

Copy: E1's taught copy of run6 (gb10-a:~/tf-promote/e1-20261019/model-taught) and its notes store, with run6's
promotion state as copied at E1 time (validation/taught-copy.run6-promote-before.json). Binary: src7 (3bf4e54f),
the same as run6 and E1. Promotion on with --notes-promote-every 8000h, so no night runs during the test.
Every ask: /v1/chat/completions, temperature 0, thinking off, no "notes" field (no notes in the prompt), 48 tokens.

Erased fact: w3 "My dog is called Bruno now." (in weights). Its 13 questions: 3 gate, 6 P(F), 4 trained (E1 sets.json).
Kept facts: w1 amber, w9 apples, w15 brown, w21 brown; their 11 gate questions (E1: 11 of 11 right with no notes).

Phases: A before the erase, B after the erase, C after a server restart on the erased copy.
Rules:
1. Reproduction (control): in A, every reply equals E1's taught A0 reply for the same question where E1 asked it
   (gate questions), and w3 names Bruno on at least 12 of 13. If not, the copy is not E1's state: stop, no result.
2. Erased: in B and in C, no w3 reply names "Bruno" (case-insensitive). E1's untaught copy named it on 0 of 13.
3. Kept: in B and in C, count kept-fact replies that name the right value (amber / apples / brown / brown).
   Report the count; any drop from A is damage done by the erase.
4. Files: report which model files changed between A and C (sha256), and what the erase reply and
   GET /v1/promotion say about w3.
One run, one machine (GB10), greedy decoding.
