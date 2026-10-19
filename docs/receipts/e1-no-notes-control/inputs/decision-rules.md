# E1 decision rules (written before any E1 ask; frozen by sha256 in decision-rules.sha256)

Written 19 Oct 2026, about 12:50 CEST, from synth-promotion.md section 5 "E1".
Nothing below may change after the first ask. Anything added later goes in a separate, dated file.

## Models
- taught: a fresh copy of ~/tf-promote/run6/model (shard hashes must equal run6's).
- untaught: a fresh copy of SNAP (models--TensorFold--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit, d9d758fb...), blobs checked against their names.

## Serving (both models, one server at a time)
src7 binary (3bf4e54f, sha256 5b04e79d...), `serve MODEL --notes STORECOPY --no-notes-write --host 127.0.0.1 --port 1819x --no-update-check --prompt-cache-gib 0 --context 32768`, TENSORFOLD_NO_LIVE=1, no --slide, no --notes-promote-*.
Every ask: /v1/chat/completions, temperature 0, chat_template_kwargs.enable_thinking false, no "notes" field (so no lookup), max_tokens 48 (160 for the 56-set), the block (if any) as the first system message.
Blocks: built once with notes:true, notes_write:false, notes_echo:true, notes_exclude:[ids], max_tokens 1, on a store copy of run6's nights-day1.snap.json + .journal (no .promote file).

## Arms (exclusion sets are note ids; "own" = the fact's current note; "W5" = the notes of w1 w3 w9 w15 w21; "W7" = W5 + s2 + s10)
- A0: no system message.
- A1x: production block, exclude own + W5 (night 12, reply set X).
- A1y: production block, exclude own + W7 (night 14, reply set Y).
- A2: A1x's set, plus every current note whose subject equals the fact's subject, plus every current note whose text has one of the fact's attribute words (prefix match, case folded). Attribute words by the fact's L6 attribute: eye_colour -> eye, colour; colour -> colour; owner -> belong, owner; likes -> like; name -> name, called; fav_colour -> favourite, colour; home_city -> live, city.
- A3: one fixed block for every question: the production head and "Current:" lines with 8 made-up notes on subjects and attributes the store lacks (no colour, eye, name, city, like, favourite, owner word), padded to about the mean token length of the A1x Q_gate blocks.
- A4: production block, exclude W5 minus the fact's own note (the own note is always in; for an in-weights fact it is taken out of W5).
- A5: production block, exclude own + the notes of all 15 pocket facts.
- A6: A1x minus one suspect note (that note put back is impossible, so: A1x's set plus the suspect note excluded). Sub-arms: s2 and s10 with s7 excluded; w16 with w13 excluded; w22 with w11, with w17, and with both; s3 with w8; s5 with w18, with w19 (the cheese take-back), and with both. If a suspect has no current note id, that sub-arm is reported as not runnable.

## Questions per arm
- Q_gate: the 44 probes in run6's .promote file (2 to 4 per pocket fact, with the stored reference).
- P(F): pf.json (frozen), minus collisions.
- Q_trained: per pocket fact, the first 4 questions of the last "slide: N questions kept for: F (...)" line in run6's server.log before the night F entered the pocket (the fact's "night" in .promote), after removing that fact's Q_gate questions.
- Q_test: the 56-set of L6-seed1-nights.json, 160 tokens, judged by the test suite's judge (judge-main and llwz-promo2) for phase day5; per fact, the questions whose (e, a) is the fact's.
- Taught: Q_gate in A0, A1x, A1y, A2, A3, A4, A5, A6; P(F) and Q_trained in A0, A1x, A5; Q_test in A0, A1x.
- Untaught: Q_gate in A0, A1x, A4, A5; P(F) and Q_trained in A0, A1x, A5; Q_test in A0, A1x.
- In A1x and A5 the exclusion for a P(F) question of "I like pears." (not a pocket fact) is its own note + the arm's set; for a 56-set question its own note is the current note on its (e, a) if that is a pocket fact, else none.

## Judges
- Primary for Q_gate: the branch's agreesWith(fact, question, stored ref, reply, others) with others = every current note's text but the fact's (GET /v1/notes, as othersOf lists them).
- Primary for P(F) and Q_trained: agreesWith with the fact's written reference in pf.json.
- Secondary, reported beside: recalls(fact, q, ref, reply) and recallsShort(fact, q, short, reply) (the learner's judges, no hedge filter).
- Q_test: the test suite's judge; a fact's Q_test is "right" when every one of its questions passes under both judge roots.
- Manual reads of replies are reported as INFERRED and never change a primary verdict.

## Pass rules
- Q_gate, one arm: the fact passes when every one of its probes is judged right (the gate's rule on one night).
- P(F) and Q_trained, one arm: the fact passes when strictly more than half of its (non-colliding) questions are judged right.
- "right" below means passes; "wrong" means does not pass.

## Validation (stop if it fails)
- Taught A1x on Q_gate must reproduce run6 night 12's 44 alone asks, and A1y night 14's: the multiset of (prompt tokens, reply tokens, 12-hex sha) of my 44 asks must equal the night's 44 "done" lines (run6_alone_lines.json). Also 25/44 by the primary judge in both.
- Shard hashes of both copies unchanged after each ask-only server stops.
- If prompt counts match but shas do not: compare serving flags with serve.sh before trusting any number; report, do not proceed silently.

## Decision rules
1. Carried by neurons (headline): the taught model passes A0 on Q_gate AND the untaught model fails A0 on Q_gate. Reported for all 15 pocket facts; the 5 in-weights facts are the question.
   Secondary views (reported, not headline): the same rule on P(F) and on Q_trained, each with its own pass rule.
2. Right in A0 and A5 but wrong in A1x: competition from other notes; the fix is the gate's context (E3/E5 cannot help).
3. Right in A0 but wrong in A3: H1 (a notes block of any kind shifts the decision row); a training problem.
4. Wrong in A0 on questions the keep round passed (held questions = Q_gate probes 1-2, plus probes 3-4 that the keep round's log does not name as failing): H3; E2 matters most.
5. Q_trained right but P(F) or Q_gate wrong in A0: H4; run E3 (E5 only on its own condition).
6. Green pair (s2, s10) wrong in A0 and wrong in A6 with s7 excluded: they were passing through the other notes.
7. Untaught A4: a probe the untaught model gets wrong with the fact's own note in is a broken probe or reference (H5); listed per probe.
8. Untaught A0: any pocket fact it passes is a judge false pass or a guessable answer; such a fact can never count as carried.
