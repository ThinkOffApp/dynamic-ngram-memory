# Split scores of the re-judged run records

Every answer of the stored run records, judged by the judge on this branch and split by what the rule book expects. Counted from the run records, which were only read (MEASURED). Each cell is thinking off / thinking on. `4/7` is 4 right of 7; `(3 dk)` counts the misses where the model said it does not know.

- **a value**: the right answer is a value, a set, a yes, one of two options, someone, or the textbook answer.
- **no**: the right answer is no, neither or none. **don't know**: the right answer is "don't know".
- **should change**: questions whose right answer differs from the phase before (teach against baseline, a delete against teach); **should stay**: the others. Each split as above (v, n, ?).
- **"don't know" P, R**: precision (of the "don't know" answers, how many were right to say it) and recall (of the questions whose right answer is "don't know", how many got it).

## By the right answer

| Run | Phase | a value | no | don't know | should change | should stay | "don't know" P, R |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mac-L1-seed1 | baseline | 3/3 / 3/3 | - / - | 9/9 / 9/9 | - / - | - / - | P 9/9 R 9/9 / P 9/9 R 9/9 |
| mac-L1-seed1 | teach | 9/9 / 9/9 | 0/1 / 0/1 (1 dk) | 2/2 / 2/2 | v 6/6  n 0/1 / v 6/6  n 0/1 (1 dk) | v 3/3  ? 2/2 / v 3/3  ? 2/2 | P 2/2 R 2/2 / P 2/3 R 2/2 |
| mac-L1-seed1 | del-A (reteach) | 3/3 / 3/3 | - / - | 9/9 / 9/9 | ? 7/7 / ? 7/7 | v 3/3  ? 2/2 / v 3/3  ? 2/2 | P 9/9 R 9/9 / P 9/9 R 9/9 |
| mac-L1-seed1 | del-A (reteach.restart) | 3/3 / 3/3 | - / - | 9/9 / 9/9 | ? 7/7 / ? 7/7 | v 3/3  ? 2/2 / v 3/3  ? 2/2 | P 9/9 R 9/9 / P 9/9 R 9/9 |
| mac-L1-seed1 | del-A (zero) | 3/3 / 3/3 | - / - | 9/9 / 9/9 | ? 7/7 / ? 7/7 | v 3/3  ? 2/2 / v 3/3  ? 2/2 | P 9/9 R 9/9 / P 9/9 R 9/9 |
| gb10-L1-seed1 | baseline | 3/3 / 3/3 | - / - | 9/9 / 9/9 | - / - | - / - | P 9/9 R 9/9 / P 9/9 R 9/9 |
| gb10-L1-seed1 | teach | 3/9 (6 dk) / 3/9 (6 dk) | 0/1 (1 dk) / 0/1 (1 dk) | 2/2 / 2/2 | v 0/6 (6 dk)  n 0/1 (1 dk) / v 0/6 (6 dk)  n 0/1 (1 dk) | v 3/3  ? 2/2 / v 3/3  ? 2/2 | P 2/9 R 2/2 / P 2/9 R 2/2 |
| gb10-L1-seed1 | del-A (reteach) | 3/3 / 3/3 | - / - | 9/9 / 9/9 | ? 7/7 / ? 7/7 | v 3/3  ? 2/2 / v 3/3  ? 2/2 | P 9/9 R 9/9 / P 9/9 R 9/9 |
| gb10-L1-seed1-teach2 | baseline | 3/3 / 3/3 | - / - | 9/9 / 9/9 | - / - | - / - | P 9/9 R 9/9 / P 9/9 R 9/9 |
| gb10-L1-seed1-teach2 | teach | 3/9 (6 dk) / 3/9 (6 dk) | 0/1 (1 dk) / 0/1 (1 dk) | 2/2 / 2/2 | v 0/6 (6 dk)  n 0/1 (1 dk) / v 0/6 (6 dk)  n 0/1 (1 dk) | v 3/3  ? 2/2 / v 3/3  ? 2/2 | P 2/9 R 2/2 / P 2/9 R 2/2 |
| gb10-L2-seed1 | baseline | 3/3 / 3/3 | - / - | 11/13 / 13/13 | - / - | - / - | P 11/11 R 11/13 / P 13/13 R 13/13 |
| gb10-L2-seed1 | teach | 4/6 (2 dk) / 3/6 (3 dk) | 0/2 / 0/2 (1 dk) | 2/8 / 2/8 | v 1/3 (2 dk)  n 0/2  ? 0/3 / v 0/3 (3 dk)  n 0/2 (1 dk)  ? 0/3 | v 3/3  ? 2/5 / v 3/3  ? 2/5 | P 2/4 R 2/8 / P 2/6 R 2/8 |
| gb10-L2-seed1 | del-A (reteach) | 10/12 (2 dk) / 7/12 (4 dk) | 0/2 / 0/2 (1 dk) | 2/2 / 2/2 | v 6/8 (2 dk)  n 0/1 / v 4/8 (3 dk)  n 0/1 | v 4/4  n 0/1  ? 2/2 / v 3/4 (1 dk)  n 0/1 (1 dk)  ? 2/2 | P 2/4 R 2/2 / P 2/7 R 2/2 |
| gb10-L2-seed1 | del-B (reteach) | 4/6 (2 dk) / 3/6 (3 dk) | 0/2 / 0/2 (1 dk) | 2/8 / 2/8 | - / - | v 4/6 (2 dk)  n 0/2  ? 2/8 / v 3/6 (3 dk)  n 0/2 (1 dk)  ? 2/8 | P 2/4 R 2/8 / P 2/6 R 2/8 |
| gb10-L2-seed1 | del-A (zero) | 10/12 (2 dk) / 8/12 (3 dk) | 0/2 / 0/2 (1 dk) | 2/2 / 2/2 | v 6/8 (2 dk)  n 0/1 / v 5/8 (2 dk)  n 0/1 | v 4/4  n 0/1  ? 2/2 / v 3/4 (1 dk)  n 0/1 (1 dk)  ? 2/2 | P 2/4 R 2/2 / P 2/6 R 2/2 |
| gb10-L2-seed1 | del-B (zero) | 4/6 (2 dk) / 3/6 (3 dk) | 0/2 / 0/2 (1 dk) | 2/8 / 2/8 | - / - | v 4/6 (2 dk)  n 0/2  ? 2/8 / v 3/6 (3 dk)  n 0/2 (1 dk)  ? 2/8 | P 2/4 R 2/8 / P 2/6 R 2/8 |
| gb10-L3-seed1 | baseline | 3/3 / 3/3 | - / - | 17/19 / 19/19 | - / - | - / - | P 17/17 R 17/19 / P 19/19 R 19/19 |
| gb10-L3-seed1 | teach | 11/16 (4 dk) / 8/16 (8 dk) | 1/4 (2 dk) / 1/4 (3 dk) | 2/2 / 2/2 | v 8/13 (4 dk)  n 1/4 (2 dk) / v 5/13 (8 dk)  n 1/4 (3 dk) | v 3/3  ? 2/2 / v 3/3  ? 2/2 | P 2/8 R 2/2 / P 2/13 R 2/2 |
| gb10-L3-seed1 | del-A (reteach) | 4/7 (3 dk) / 4/7 (3 dk) | 0/2 (2 dk) / 0/2 (2 dk) | 11/13 / 13/13 | ? 9/11 / ? 11/11 | v 4/7 (3 dk)  n 0/2 (2 dk)  ? 2/2 / v 4/7 (3 dk)  n 0/2 (2 dk)  ? 2/2 | P 11/16 R 11/13 / P 13/18 R 13/13 |
| gb10-L3-seed1 | del-B (reteach) | 8/14 (2 dk) / 9/14 (4 dk) | 2/3 (1 dk) / 0/3 (2 dk) | 5/5 / 5/5 | ? 3/3 / ? 3/3 | v 8/14 (2 dk)  n 2/3 (1 dk)  ? 2/2 / v 9/14 (4 dk)  n 0/3 (2 dk)  ? 2/2 | P 5/8 R 5/5 / P 5/11 R 5/5 |
| gb10-L3-seed1 | del-A (zero) | 4/7 (3 dk) / 4/7 (3 dk) | 0/2 (2 dk) / 0/2 (2 dk) | 11/13 / 13/13 | ? 9/11 / ? 11/11 | v 4/7 (3 dk)  n 0/2 (2 dk)  ? 2/2 / v 4/7 (3 dk)  n 0/2 (2 dk)  ? 2/2 | P 11/16 R 11/13 / P 13/18 R 13/13 |
| gb10-L3-seed1 | del-B (zero) | 11/14 (2 dk) / 8/14 (6 dk) | 1/3 (1 dk) / 1/3 (2 dk) | 5/5 / 5/5 | ? 3/3 / ? 3/3 | v 11/14 (2 dk)  n 1/3 (1 dk)  ? 2/2 / v 8/14 (6 dk)  n 1/3 (2 dk)  ? 2/2 | P 5/8 R 5/5 / P 5/13 R 5/5 |
| gb10-L3-seed1 | del-B (teach.as-del-B) | 11/14 (2 dk) / 8/14 (6 dk) | 1/3 (1 dk) / 1/3 (2 dk) | 5/5 / 5/5 | ? 3/3 / ? 3/3 | v 11/14 (2 dk)  n 1/3 (1 dk)  ? 2/2 / v 8/14 (6 dk)  n 1/3 (2 dk)  ? 2/2 | P 5/8 R 5/5 / P 5/13 R 5/5 |
| gb10-eyes-678 | baseline | 3/3 / 3/3 | - / - | 11/11 / 11/11 | - / - | - / - | P 11/11 R 11/11 / P 11/11 R 11/11 |
| gb10-eyes-678 | teach | 11/12 (1 dk) / 8/12 (2 dk) | 0/1 / 0/1 (1 dk) | 2/2 / 2/2 | v 8/9 (1 dk)  n 0/1 / v 5/9 (2 dk)  n 0/1 (1 dk) | v 3/3  ? 2/2 / v 3/3  ? 2/2 | P 2/3 R 2/2 / P 2/5 R 2/2 |
| gb10-eyes-678 | rm-3 (reteach) | 11/12 (1 dk) / 9/12 (2 dk) | 0/1 / 0/1 (1 dk) | 2/2 / 2/2 | - / - | v 11/12 (1 dk)  n 0/1  ? 2/2 / v 9/12 (2 dk)  n 0/1 (1 dk)  ? 2/2 | P 2/3 R 2/2 / P 2/5 R 2/2 |
| gb10-eyes-678 | rm-4 (reteach) | 11/12 (1 dk) / 9/12 (1 dk) | 1/1 / 0/1 | 2/2 / 2/2 | - / - | v 11/12 (1 dk)  n 1/1  ? 2/2 / v 9/12 (1 dk)  n 0/1  ? 2/2 | P 2/3 R 2/2 / P 2/3 R 2/2 |
| gb10-eyes-678 | rm-5 (reteach) | 9/11 (1 dk) / 9/11 (1 dk) | 0/1 (1 dk) / 0/1 (1 dk) | 2/3 / 2/3 | ? 0/1 / ? 0/1 | v 9/11 (1 dk)  n 0/1 (1 dk)  ? 2/2 / v 9/11 (1 dk)  n 0/1 (1 dk)  ? 2/2 | P 2/4 R 2/3 / P 2/4 R 2/3 |
| gb10-eyes-678 | rm-345 (reteach) | 7/8 (1 dk) / 6/8 (1 dk) | - / - | 7/7 / 7/7 | ? 5/5 / ? 5/5 | v 7/8 (1 dk)  ? 2/2 / v 6/8 (1 dk)  ? 2/2 | P 7/8 R 7/7 / P 7/8 R 7/7 |
| gb10-eyes-678 | rm-all (reteach) | 3/3 / 3/3 | - / - | 11/11 / 11/11 | ? 9/9 / ? 9/9 | v 3/3  ? 2/2 / v 3/3  ? 2/2 | P 11/11 R 11/11 / P 11/11 R 11/11 |
| gb10-eyes-678 | rm-3 (zero) | 11/12 (1 dk) / 8/12 (2 dk) | 0/1 / 0/1 (1 dk) | 2/2 / 2/2 | - / - | v 11/12 (1 dk)  n 0/1  ? 2/2 / v 8/12 (2 dk)  n 0/1 (1 dk)  ? 2/2 | P 2/3 R 2/2 / P 2/5 R 2/2 |
| gb10-eyes-678 | rm-4 (zero) | 11/12 (1 dk) / 8/12 (2 dk) | 0/1 / 0/1 (1 dk) | 2/2 / 2/2 | - / - | v 11/12 (1 dk)  n 0/1  ? 2/2 / v 8/12 (2 dk)  n 0/1 (1 dk)  ? 2/2 | P 2/3 R 2/2 / P 2/5 R 2/2 |
| gb10-eyes-678 | rm-5 (zero) | 7/11 (4 dk) / 7/11 (3 dk) | 0/1 (1 dk) / 0/1 (1 dk) | 3/3 / 3/3 | ? 1/1 / ? 1/1 | v 7/11 (4 dk)  n 0/1 (1 dk)  ? 2/2 / v 7/11 (3 dk)  n 0/1 (1 dk)  ? 2/2 | P 3/8 R 3/3 / P 3/7 R 3/3 |
| gb10-eyes-678 | rm-345 (zero) | 7/8 (1 dk) / 7/8 | - / - | 7/7 / 7/7 | ? 5/5 / ? 5/5 | v 7/8 (1 dk)  ? 2/2 / v 7/8  ? 2/2 | P 7/8 R 7/7 / P 7/7 R 7/7 |
| gb10-eyes-678 | rm-all (zero) | 3/3 / 3/3 | - / - | 11/11 / 11/11 | ? 9/9 / ? 9/9 | v 3/3  ? 2/2 / v 3/3  ? 2/2 | P 11/11 R 11/11 / P 11/11 R 11/11 |
| gb10-L4-seed1 | baseline | 3/3 / 3/3 | - / - | 18/19 / 19/19 | - / - | - / - | P 18/18 R 18/19 / P 19/19 R 19/19 |
| gb10-L4-seed1 | teach | 8/14 (5 dk) / 7/14 (6 dk) | 2/3 / 0/3 (2 dk) | 3/5 / 3/5 | v 5/11 (5 dk)  n 2/3  ? 0/1 / v 4/11 (6 dk)  n 0/3 (2 dk)  ? 0/1 | v 3/3  ? 3/4 / v 3/3  ? 3/4 | P 3/8 R 3/5 / P 3/11 R 3/5 |
| gb10-L4-seed1 | del-A (reteach) | 9/15 (5 dk) / 8/15 (6 dk) | 1/2 / 0/2 (1 dk) | 3/5 / 4/5 | v 1/4 (3 dk) / v 0/4 (4 dk) | v 8/11 (2 dk)  n 1/2  ? 3/5 / v 8/11 (2 dk)  n 0/2 (1 dk)  ? 4/5 | P 3/8 R 3/5 / P 4/11 R 4/5 |
| gb10-L4-seed1 | del-B (reteach) | 7/13 (5 dk) / 6/13 (5 dk) | 1/2 / 0/2 | 4/7 / 5/7 | ? 1/2 / ? 2/2 | v 7/13 (5 dk)  n 1/2  ? 3/5 / v 6/13 (5 dk)  n 0/2  ? 3/5 | P 4/9 R 4/7 / P 5/10 R 5/7 |
| gb10-L4-seed1 | del-C (reteach) | 10/14 (3 dk) / 7/14 (7 dk) | 0/2 / 0/2 (1 dk) | 5/6 / 6/6 | v 2/2  ? 2/2 / v 1/2 (1 dk)  ? 2/2 | v 8/12 (3 dk)  n 0/2  ? 3/4 / v 6/12 (6 dk)  n 0/2 (1 dk)  ? 4/4 | P 5/8 R 5/6 / P 6/14 R 6/6 |
| gb10-L4-seed1 | del-A (zero) | 8/15 (6 dk) / 7/15 (7 dk) | 1/2 / 0/2 (1 dk) | 3/5 / 4/5 | v 0/4 (4 dk) / v 0/4 (4 dk) | v 8/11 (2 dk)  n 1/2  ? 3/5 / v 7/11 (3 dk)  n 0/2 (1 dk)  ? 4/5 | P 3/9 R 3/5 / P 4/12 R 4/5 |
| gb10-L4-seed1 | del-B (zero) | 7/13 (5 dk) / 6/13 (6 dk) | 2/2 / 0/2 (2 dk) | 3/7 / 3/7 | ? 0/2 / ? 0/2 | v 7/13 (5 dk)  n 2/2  ? 3/5 / v 6/13 (6 dk)  n 0/2 (2 dk)  ? 3/5 | P 3/8 R 3/7 / P 3/11 R 3/7 |
| gb10-L4-seed1 | del-C (zero) | 7/14 (5 dk) / 6/14 (8 dk) | 1/2 / 0/2 (1 dk) | 5/6 / 5/6 | v 1/2  ? 2/2 / v 1/2 (1 dk)  ? 2/2 | v 6/12 (5 dk)  n 1/2  ? 3/4 / v 5/12 (7 dk)  n 0/2 (1 dk)  ? 3/4 | P 5/10 R 5/6 / P 5/14 R 5/6 |

## Over all phases of each report

Closed world: the fails that would pass if a "no" counted where the right answer is "don't know" (the closed-world reading, where anything not said is false; the test suite reads it open-world).

| Run | Report | "don't know" P, R | a value | no | don't know | closed world |
| --- | --- | --- | --- | --- | --- | --- |
| mac-L1-seed1 | mac-L1-seed1.reteach.txt | P 20/20 R 20/20 / P 20/21 R 20/20 | 15/15 / 15/15 | 0/1 / 0/1 (1 dk) | 20/20 / 20/20 | 0 / 0 |
| mac-L1-seed1 | mac-L1-seed1.reteach.restart.txt | P 20/20 R 20/20 / P 20/21 R 20/20 | 15/15 / 15/15 | 0/1 / 0/1 (1 dk) | 20/20 / 20/20 | 0 / 0 |
| mac-L1-seed1 | mac-L1-seed1.zero.txt | P 20/20 R 20/20 / P 20/21 R 20/20 | 15/15 / 15/15 | 0/1 / 0/1 (1 dk) | 20/20 / 20/20 | 0 / 0 |
| gb10-L1-seed1 | gb10-L1-seed1.reteach.txt | P 20/27 R 20/20 / P 20/27 R 20/20 | 9/15 (6 dk) / 9/15 (6 dk) | 0/1 (1 dk) / 0/1 (1 dk) | 20/20 / 20/20 | 0 / 0 |
| gb10-L1-seed1 | gb10-L1-seed1.zero.txt | P 11/18 R 11/11 / P 11/18 R 11/11 | 6/12 (6 dk) / 6/12 (6 dk) | 0/1 (1 dk) / 0/1 (1 dk) | 11/11 / 11/11 | 0 / 0 |
| gb10-L1-seed1 | gb10-L1-seed1.teach.as-baseline.txt | P 9/9 R 9/9 / P 9/9 R 9/9 | 3/3 / 3/3 | - / - | 9/9 / 9/9 | 0 / 0 |
| gb10-L1-seed1-teach2 | gb10-L1-seed1-teach2.teach.as-baseline.txt | P 9/9 R 9/9 / P 9/9 R 9/9 | 3/3 / 3/3 | - / - | 9/9 / 9/9 | 0 / 0 |
| gb10-L1-seed1-teach2 | gb10-L1-seed1-teach2.teach.txt | P 2/9 R 2/2 / P 2/9 R 2/2 | 3/9 (6 dk) / 3/9 (6 dk) | 0/1 (1 dk) / 0/1 (1 dk) | 2/2 / 2/2 | 0 / 0 |
| gb10-L2-seed1 | gb10-L2-seed1.reteach.txt | P 17/23 R 17/31 / P 19/32 R 19/31 | 21/27 (6 dk) / 16/27 (10 dk) | 0/6 / 0/6 (3 dk) | 17/31 / 19/31 | 0 / 1 |
| gb10-L2-seed1 | gb10-L2-seed1.zero.txt | P 17/23 R 17/31 / P 19/31 R 19/31 | 21/27 (6 dk) / 17/27 (9 dk) | 0/6 / 0/6 (3 dk) | 17/31 / 19/31 | 0 / 2 |
| gb10-L3-seed1 | gb10-L3-seed1.reteach.txt | P 35/49 R 35/39 / P 39/61 R 39/39 | 26/40 (9 dk) / 24/40 (15 dk) | 3/9 (5 dk) / 1/9 (7 dk) | 35/39 / 39/39 | 0 / 0 |
| gb10-L3-seed1 | gb10-L3-seed1.zero.txt | P 35/49 R 35/39 / P 39/63 R 39/39 | 29/40 (9 dk) / 23/40 (17 dk) | 2/9 (5 dk) / 2/9 (7 dk) | 35/39 / 39/39 | 0 / 0 |
| gb10-L3-seed1 | gb10-L3-seed1.teach.as-del-B.txt | P 5/8 R 5/5 / P 5/13 R 5/5 | 11/14 (2 dk) / 8/14 (6 dk) | 1/3 (1 dk) / 1/3 (2 dk) | 5/5 / 5/5 | 0 / 0 |
| gb10-eyes-678 | gb10-eyes-678.reteach.txt | P 37/43 R 37/38 / P 37/47 R 37/38 | 55/61 (5 dk) / 47/61 (7 dk) | 1/4 (1 dk) / 0/4 (3 dk) | 37/38 / 37/38 | 0 / 0 |
| gb10-eyes-678 | gb10-eyes-678.zero.txt | P 38/47 R 38/38 / P 38/51 R 38/38 | 53/61 (8 dk) / 44/61 (9 dk) | 0/4 (1 dk) / 0/4 (4 dk) | 38/38 / 38/38 | 0 / 0 |
| gb10-L4-seed1 | gb10-L4-seed1.reteach.txt | P 33/51 R 33/42 / P 37/65 R 37/42 | 37/59 (18 dk) / 31/59 (24 dk) | 4/9 / 0/9 (4 dk) | 33/42 / 37/42 | 1 / 1 |
| gb10-L4-seed1 | gb10-L4-seed1.zero.txt | P 32/53 R 32/42 / P 34/67 R 34/42 | 33/59 (21 dk) / 29/59 (27 dk) | 6/9 / 0/9 (6 dk) | 32/42 / 34/42 | 0 / 0 |

## Consistency, paraphrases and neighbourhood failures

Consistency: yes/no and either-or answers that follow from the model's own answer to every wording of the plain question, in two parts: the plain answer gave a value, or "don't know". Same answer: pairs of wordings of one question that gave the same answer, right or wrong. All ok: questions asked in several wordings that were right in every wording. Neighbourhood: failed answers that give a value taught about another entity.

| Run | Phase | consistency, plain gave a value | consistency, plain said don't know | same answer | all ok | neighbourhood failures |
| --- | --- | --- | --- | --- | --- | --- |
| mac-L1-seed1 | baseline | - / - | 12/12 / 12/12 | 4/4 / 4/4 | 2/2 / 2/2 | - |
| mac-L1-seed1 | teach | 9/12 / 9/12 | - / - | 4/4 / 4/4 | 2/2 / 2/2 | - |
| mac-L1-seed1 | del-A (reteach) | - / - | 12/12 / 12/12 | 4/4 / 4/4 | 2/2 / 2/2 | - |
| mac-L1-seed1 | del-A (reteach.restart) | - / - | 12/12 / 12/12 | 4/4 / 4/4 | 2/2 / 2/2 | - |
| mac-L1-seed1 | del-A (zero) | - / - | 12/12 / 12/12 | 4/4 / 4/4 | 2/2 / 2/2 | - |
| gb10-L1-seed1 | baseline | - / - | 12/12 / 12/12 | 4/4 / 4/4 | 2/2 / 2/2 | - |
| gb10-L1-seed1 | teach | - / - | 12/12 / 12/12 | 4/4 / 4/4 | 0/2 / 0/2 | - |
| gb10-L1-seed1 | del-A (reteach) | - / - | 12/12 / 12/12 | 4/4 / 4/4 | 2/2 / 2/2 | - |
| gb10-L1-seed1-teach2 | baseline | - / - | 12/12 / 12/12 | 4/4 / 4/4 | 2/2 / 2/2 | - |
| gb10-L1-seed1-teach2 | teach | - / - | 12/12 / 12/12 | 4/4 / 4/4 | 0/2 / 0/2 | - |
| gb10-L2-seed1 | baseline | - / - | 9/9 / 12/12 | 4/4 / 5/5 | 2/3 / 3/3 | - |
| gb10-L2-seed1 | teach | 9/12 / 9/12 | - / - | 5/5 / 4/5 | 0/3 / 0/3 | - |
| gb10-L2-seed1 | del-A (reteach) | 9/12 / 3/12 | - / - | 5/5 / 4/5 | 2/3 / 1/3 | - |
| gb10-L2-seed1 | del-B (reteach) | 9/12 / 6/9 | - / - | 5/5 / 4/4 | 0/3 / 0/3 | - |
| gb10-L2-seed1 | del-A (zero) | 9/12 / 6/12 | - / - | 5/5 / 4/5 | 2/3 / 1/3 | - |
| gb10-L2-seed1 | del-B (zero) | 9/12 / 9/12 | - / - | 5/5 / 4/5 | 0/3 / 0/3 | - |
| gb10-L3-seed1 | baseline | - / - | 8/8 / 10/10 | 1/1 / 1/1 | 1/1 / 1/1 | - |
| gb10-L3-seed1 | teach | 6/8 / 4/8 | 2/2 / 2/2 | 1/1 / 1/1 | 1/1 / 1/1 | - |
| gb10-L3-seed1 | del-A (reteach) | 0/2 / 0/2 | 6/6 / 8/8 | 1/1 / 1/1 | 1/1 / 1/1 | - |
| gb10-L3-seed1 | del-B (reteach) | 4/8 / 0/8 | 2/2 / 2/2 | 1/1 / 1/1 | 1/1 / 1/1 | - |
| gb10-L3-seed1 | del-A (zero) | 0/2 / 0/2 | 6/6 / 8/8 | 1/1 / 1/1 | 1/1 / 1/1 | - |
| gb10-L3-seed1 | del-B (zero) | 6/8 / 4/8 | 2/2 / 2/2 | 1/1 / 1/1 | 1/1 / 1/1 | - |
| gb10-L3-seed1 | del-B (teach.as-del-B) | 6/8 / 4/8 | 2/2 / 2/2 | 1/1 / 1/1 | 1/1 / 1/1 | - |
| gb10-eyes-678 | baseline | - / - | 6/6 / 6/6 | 2/2 / 2/2 | 2/2 / 2/2 | - |
| gb10-eyes-678 | teach | 6/6 / 2/6 | - / - | 2/2 / 2/2 | 2/2 / 2/2 | - |
| gb10-eyes-678 | rm-3 (reteach) | 6/6 / 2/6 | - / - | 2/2 / 2/2 | 2/2 / 2/2 | - |
| gb10-eyes-678 | rm-4 (reteach) | 6/6 / 4/5 | - / - | 2/2 / 1/1 | 2/2 / 1/2 | - |
| gb10-eyes-678 | rm-5 (reteach) | 4/6 / 4/6 | - / - | 2/2 / 2/2 | 2/2 / 2/2 | - |
| gb10-eyes-678 | rm-345 (reteach) | 4/4 / 2/4 | 2/2 / 2/2 | 2/2 / 2/2 | 2/2 / 2/2 | - |
| gb10-eyes-678 | rm-all (reteach) | - / - | 6/6 / 6/6 | 2/2 / 2/2 | 2/2 / 2/2 | - |
| gb10-eyes-678 | rm-3 (zero) | 6/6 / 2/6 | - / - | 2/2 / 2/2 | 2/2 / 2/2 | - |
| gb10-eyes-678 | rm-4 (zero) | 6/6 / 2/6 | - / - | 2/2 / 2/2 | 2/2 / 2/2 | - |
| gb10-eyes-678 | rm-5 (zero) | 4/4 / 4/4 | 2/2 / 2/2 | 2/2 / 2/2 | 1/2 / 1/2 | - |
| gb10-eyes-678 | rm-345 (zero) | 4/4 / 4/4 | 2/2 / 2/2 | 2/2 / 2/2 | 2/2 / 2/2 | - |
| gb10-eyes-678 | rm-all (zero) | - / - | 6/6 / 6/6 | 2/2 / 2/2 | 2/2 / 2/2 | - |
| gb10-L4-seed1 | baseline | - / - | 8/9 / 9/9 | 1/1 / 1/1 | 1/1 / 1/1 | - |
| gb10-L4-seed1 | teach | 0/3 / 0/3 | 0/6 / 4/6 | 1/1 / 1/1 | 0/1 / 0/1 | Q03 Nora (my cat) (off); Q04 Bruno (my dog) (off); Q03 Nora (my cat) (on); Q04 Bruno (my dog) (on) |
| gb10-L4-seed1 | del-A (reteach) | 0/3 / 0/3 | 0/6 / 4/6 | 1/1 / 1/1 | 0/1 / 0/1 | Q03 Nora (my cat) (off); Q04 Bruno (my dog) (off); Q03 Nora (my cat) (on); Q04 Bruno (my dog) (on) |
| gb10-L4-seed1 | del-B (reteach) | 1/2 / 1/1 | 0/6 / 1/7 | 1/1 / 1/1 | 0/1 / 0/1 | - |
| gb10-L4-seed1 | del-C (reteach) | 2/3 / 0/3 | 0/3 / 6/6 | - / 1/1 | 0/1 / 0/1 | Q03 Nora (my cat) (off); Q03 Nora (my cat) (on) |
| gb10-L4-seed1 | del-A (zero) | 0/3 / 0/3 | 2/6 / 6/6 | 1/1 / 1/1 | 0/1 / 0/1 | Q03 Nora (my cat) (off); Q04 Bruno (my dog) (off); Q03 Nora (my cat) (on); Q04 Bruno (my dog) (on) |
| gb10-L4-seed1 | del-B (zero) | 0/3 / 0/3 | 0/6 / 4/6 | 1/1 / 1/1 | 0/1 / 0/1 | Q03 Nora (my cat) (off); Q03 Nora (my cat) (on) |
| gb10-L4-seed1 | del-C (zero) | 0/2 / 0/2 | 0/6 / 5/7 | 1/1 / 1/1 | 0/1 / 0/1 | Q03 Nora (my cat) (off); Q03 Nora (my cat) (on) |

## The literature's names

Pass of n per tag (`zengine/lwlogic.py`, `question_tags`).

| Run | Phase | reliability | generality | locality | portability |
| --- | --- | --- | --- | --- | --- |
| mac-L1-seed1 | baseline | 1/1 / 1/1 | 6/6 / 6/6 | 5/5 / 5/5 | - / - |
| mac-L1-seed1 | teach | 1/1 / 1/1 | 5/6 / 5/6 | 5/5 / 5/5 | - / - |
| mac-L1-seed1 | del-A (reteach) | 1/1 / 1/1 | 6/6 / 6/6 | 5/5 / 5/5 | - / - |
| mac-L1-seed1 | del-A (reteach.restart) | 1/1 / 1/1 | 6/6 / 6/6 | 5/5 / 5/5 | - / - |
| mac-L1-seed1 | del-A (zero) | 1/1 / 1/1 | 6/6 / 6/6 | 5/5 / 5/5 | - / - |
| gb10-L1-seed1 | baseline | 1/1 / 1/1 | 6/6 / 6/6 | 5/5 / 5/5 | - / - |
| gb10-L1-seed1 | teach | 0/1 / 0/1 | 0/6 / 0/6 | 5/5 / 5/5 | - / - |
| gb10-L1-seed1 | del-A (reteach) | 1/1 / 1/1 | 6/6 / 6/6 | 5/5 / 5/5 | - / - |
| gb10-L1-seed1-teach2 | baseline | 1/1 / 1/1 | 6/6 / 6/6 | 5/5 / 5/5 | - / - |
| gb10-L1-seed1-teach2 | teach | 0/1 / 0/1 | 0/6 / 0/6 | 5/5 / 5/5 | - / - |
| gb10-L2-seed1 | baseline | 1/1 / 1/1 | 8/10 / 10/10 | 5/5 / 5/5 | - / - |
| gb10-L2-seed1 | teach | 0/1 / 0/1 | 1/10 / 0/10 | 5/5 / 5/5 | - / - |
| gb10-L2-seed1 | del-A (reteach) | 1/1 / 1/1 | 6/10 / 3/10 | 5/5 / 5/5 | - / - |
| gb10-L2-seed1 | del-B (reteach) | 0/1 / 0/1 | 1/10 / 0/10 | 5/5 / 5/5 | - / - |
| gb10-L2-seed1 | del-A (zero) | 1/1 / 1/1 | 6/10 / 4/10 | 5/5 / 5/5 | - / - |
| gb10-L2-seed1 | del-B (zero) | 0/1 / 0/1 | 1/10 / 0/10 | 5/5 / 5/5 | - / - |
| gb10-L3-seed1 | baseline | 3/3 / 3/3 | 12/14 / 14/14 | 5/5 / 5/5 | - / - |
| gb10-L3-seed1 | teach | 2/3 / 2/3 | 7/14 / 4/14 | 5/5 / 5/5 | - / - |
| gb10-L3-seed1 | del-A (reteach) | 2/3 / 2/3 | 8/14 / 10/14 | 5/5 / 5/5 | - / - |
| gb10-L3-seed1 | del-B (reteach) | 3/3 / 3/3 | 7/14 / 6/14 | 5/5 / 5/5 | - / - |
| gb10-L3-seed1 | del-A (zero) | 2/3 / 2/3 | 8/14 / 10/14 | 5/5 / 5/5 | - / - |
| gb10-L3-seed1 | del-B (zero) | 3/3 / 3/3 | 9/14 / 6/14 | 5/5 / 5/5 | - / - |
| gb10-L3-seed1 | del-B (teach.as-del-B) | 3/3 / 3/3 | 9/14 / 6/14 | 5/5 / 5/5 | - / - |
| gb10-eyes-678 | baseline | 2/2 / 2/2 | 7/7 / 7/7 | 5/5 / 5/5 | - / - |
| gb10-eyes-678 | teach | 2/2 / 2/2 | 6/7 / 3/7 | 5/5 / 5/5 | 0/1 / 0/1 |
| gb10-eyes-678 | rm-3 (reteach) | 2/2 / 2/2 | 6/7 / 4/7 | 5/5 / 5/5 | 0/1 / 0/1 |
| gb10-eyes-678 | rm-4 (reteach) | 2/2 / 2/2 | 7/7 / 4/7 | 5/5 / 5/5 | 0/1 / 0/1 |
| gb10-eyes-678 | rm-5 (reteach) | 2/2 / 2/2 | 4/7 / 4/7 | 5/5 / 5/5 | 0/1 / 0/1 |
| gb10-eyes-678 | rm-345 (reteach) | 2/2 / 2/2 | 7/7 / 6/7 | 5/5 / 5/5 | 0/1 / 0/1 |
| gb10-eyes-678 | rm-all (reteach) | 2/2 / 2/2 | 7/7 / 7/7 | 5/5 / 5/5 | - / - |
| gb10-eyes-678 | rm-3 (zero) | 2/2 / 2/2 | 6/7 / 3/7 | 5/5 / 5/5 | 0/1 / 0/1 |
| gb10-eyes-678 | rm-4 (zero) | 2/2 / 2/2 | 6/7 / 3/7 | 5/5 / 5/5 | 0/1 / 0/1 |
| gb10-eyes-678 | rm-5 (zero) | 1/2 / 1/2 | 4/7 / 4/7 | 5/5 / 5/5 | 0/1 / 0/1 |
| gb10-eyes-678 | rm-345 (zero) | 2/2 / 2/2 | 7/7 / 7/7 | 5/5 / 5/5 | 0/1 / 0/1 |
| gb10-eyes-678 | rm-all (zero) | 2/2 / 2/2 | 7/7 / 7/7 | 5/5 / 5/5 | - / - |
| gb10-L4-seed1 | baseline | 6/6 / 6/6 | 7/8 / 8/8 | 6/6 / 6/6 | 2/2 / 2/2 |
| gb10-L4-seed1 | teach | 4/6 / 4/6 | 3/8 / 0/8 | 6/6 / 6/6 | 0/2 / 0/2 |
| gb10-L4-seed1 | del-A (reteach) | 4/6 / 4/6 | 3/8 / 2/8 | 6/6 / 6/6 | 0/2 / 0/2 |
| gb10-L4-seed1 | del-B (reteach) | 3/6 / 3/6 | 3/8 / 2/8 | 6/6 / 6/6 | 0/2 / 0/2 |
| gb10-L4-seed1 | del-C (reteach) | 5/6 / 5/6 | 4/8 / 2/8 | 6/6 / 6/6 | 0/2 / 0/2 |
| gb10-L4-seed1 | del-A (zero) | 4/6 / 4/6 | 2/8 / 1/8 | 6/6 / 6/6 | 0/2 / 0/2 |
| gb10-L4-seed1 | del-B (zero) | 3/6 / 3/6 | 3/8 / 0/8 | 6/6 / 6/6 | 0/2 / 0/2 |
| gb10-L4-seed1 | del-C (zero) | 4/6 / 4/6 | 3/8 / 1/8 | 6/6 / 6/6 | 0/2 / 0/2 |

## The dummy systems on the same sets (negative controls)

Two systems that answer every question the same way, judged on each set the runs used. Thinking off only: both modes give the same answers. Pooled is the score of the whole phase as the reports gave it before.

| Set | System | Phase | pooled | delete_affected | a value | no | don't know |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L1-seed1 | always don't know | baseline | 9/12 | - | 0/3 (3 dk) | - | 9/9 |
| L1-seed1 | always don't know | teach | 2/12 | - | 0/9 (9 dk) | 0/1 (1 dk) | 2/2 |
| L1-seed1 | always don't know | del-A | 9/12 | 7/7 | 0/3 (3 dk) | - | 9/9 |
| L1-seed1 | always yes | baseline | 0/12 | - | 0/3 | - | 0/9 |
| L1-seed1 | always yes | teach | 1/12 | - | 1/9 | 0/1 | 0/2 |
| L1-seed1 | always yes | del-A | 0/12 | 0/7 | 0/3 | - | 0/9 |
| L2-seed1 | always don't know | baseline | 13/16 | - | 0/3 (3 dk) | - | 13/13 |
| L2-seed1 | always don't know | teach | 8/16 | - | 0/6 (6 dk) | 0/2 (2 dk) | 8/8 |
| L2-seed1 | always don't know | del-A | 2/16 | 0/11 | 0/12 (12 dk) | 0/2 (2 dk) | 2/2 |
| L2-seed1 | always don't know | del-B | 8/16 | 6/11 | 0/6 (6 dk) | 0/2 (2 dk) | 8/8 |
| L2-seed1 | always yes | baseline | 0/16 | - | 0/3 | - | 0/13 |
| L2-seed1 | always yes | teach | 1/16 | - | 1/6 | 0/2 | 0/8 |
| L2-seed1 | always yes | del-A | 2/16 | 2/11 | 2/12 | 0/2 | 0/2 |
| L2-seed1 | always yes | del-B | 1/16 | 1/11 | 1/6 | 0/2 | 0/8 |
| L3-seed1 | always don't know | baseline | 19/22 | - | 0/3 (3 dk) | - | 19/19 |
| L3-seed1 | always don't know | teach | 2/22 | - | 0/16 (16 dk) | 0/4 (4 dk) | 2/2 |
| L3-seed1 | always don't know | del-A | 13/22 | 11/11 | 0/7 (7 dk) | 0/2 (2 dk) | 13/13 |
| L3-seed1 | always don't know | del-B | 5/22 | 3/3 | 0/14 (14 dk) | 0/3 (3 dk) | 5/5 |
| L3-seed1 | always yes | baseline | 0/22 | - | 0/3 | - | 0/19 |
| L3-seed1 | always yes | teach | 4/22 | - | 4/16 | 0/4 | 0/2 |
| L3-seed1 | always yes | del-A | 2/22 | 0/11 | 2/7 | 0/2 | 0/13 |
| L3-seed1 | always yes | del-B | 3/22 | 0/3 | 3/14 | 0/3 | 0/5 |
| eyes-678 | always don't know | baseline | 11/14 | - | 0/3 (3 dk) | - | 11/11 |
| eyes-678 | always don't know | teach | 2/15 | - | 0/12 (12 dk) | 0/1 (1 dk) | 2/2 |
| eyes-678 | always don't know | rm-3 | 2/15 | 0/5 | 0/12 (12 dk) | 0/1 (1 dk) | 2/2 |
| eyes-678 | always don't know | rm-4 | 2/15 | 0/5 | 0/12 (12 dk) | 0/1 (1 dk) | 2/2 |
| eyes-678 | always don't know | rm-5 | 3/15 | 1/5 | 0/11 (11 dk) | 0/1 (1 dk) | 3/3 |
| eyes-678 | always don't know | rm-345 | 7/15 | 5/5 | 0/8 (8 dk) | - | 7/7 |
| eyes-678 | always don't know | rm-all | 11/14 | 9/9 | 0/3 (3 dk) | - | 11/11 |
| eyes-678 | always yes | baseline | 0/14 | - | 0/3 | - | 0/11 |
| eyes-678 | always yes | teach | 1/15 | - | 1/12 | 0/1 | 0/2 |
| eyes-678 | always yes | rm-3 | 1/15 | 0/5 | 1/12 | 0/1 | 0/2 |
| eyes-678 | always yes | rm-4 | 1/15 | 0/5 | 1/12 | 0/1 | 0/2 |
| eyes-678 | always yes | rm-5 | 1/15 | 0/5 | 1/11 | 0/1 | 0/3 |
| eyes-678 | always yes | rm-345 | 1/15 | 0/5 | 1/8 | - | 0/7 |
| eyes-678 | always yes | rm-all | 0/14 | 0/9 | 0/3 | - | 0/11 |
| L4-seed1 | always don't know | baseline | 19/22 | - | 0/3 (3 dk) | - | 19/19 |
| L4-seed1 | always don't know | teach | 5/22 | - | 0/14 (14 dk) | 0/3 (3 dk) | 5/5 |
| L4-seed1 | always don't know | del-A | 5/22 | 1/6 | 0/15 (15 dk) | 0/2 (2 dk) | 5/5 |
| L4-seed1 | always don't know | del-B | 7/22 | 2/2 | 0/13 (13 dk) | 0/2 (2 dk) | 7/7 |
| L4-seed1 | always don't know | del-C | 6/22 | 2/5 | 0/14 (14 dk) | 0/2 (2 dk) | 6/6 |
| L4-seed1 | always yes | baseline | 0/22 | - | 0/3 | - | 0/19 |
| L4-seed1 | always yes | teach | 1/22 | - | 1/14 | 0/3 | 0/5 |
| L4-seed1 | always yes | del-A | 2/22 | 2/6 | 2/15 | 0/2 | 0/5 |
| L4-seed1 | always yes | del-B | 1/22 | 0/2 | 1/13 | 0/2 | 0/7 |
| L4-seed1 | always yes | del-C | 2/22 | 1/5 | 2/14 | 0/2 | 0/6 |
