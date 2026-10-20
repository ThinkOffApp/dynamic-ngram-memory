# Raw run files behind the paper

Text results only (no weights, checkpoints, tarballs, sources). Copied from the GB10 machine `gb10-a` (`~/lw-runs`, `~/tf-pocket-cuda`) and the Mac (`perf/`).

Rules used to keep the repo small:

- Every `.log` is gzipped individually (`name.log.gz`, `zcat` to read). Answers, verdicts, reports (`.json`, `.jsonl`, `.txt`, `.md`, `.out`) are plain text.
- Byte-identical files of 4 KB or more are stored once under `runs/_shared/<sha256-12>-<name>`. `runs/MANIFEST.tsv` maps every original path to its sha256, original size and where the bytes are stored. The same L6 teach/reteach server logs recur in L6-seed1b, cap, notes, both, tb-explicit and tb-rule. tb-rule and tb-explicit are NOT fully identical (about 50 of 488 files differ or are missing on one side), so both are kept; their identical files are shared.
- Skipped: `.sse`, `.orig`, `.pid`, anything under `models/`, `checkpoints/`, `adapters/`, files over 20 MB, directories not listed here (for example L1 seeds 2 to 6, neg*, eyes-678, nights, reviewer-*), PR-body drafts, `perf/mac-raw`.

| Paper section | Directories | Original files | Original text size (MB) |
|---|---|---|---|
| 7.2 First fact tests L1-L4 | `gb10-a-L1-seed1`, `gb10-a-L1-seed1-r2`, `gb10-a-L1-seed1-r2.out`, `gb10-a-L1-seed1-r3`, `gb10-a-L1-seed1-r3.out`, `gb10-a-L1-seed1-r4`, `gb10-a-L1-seed1-r4.out`, `gb10-a-L1-seed1-r5`, `gb10-a-L1-seed1-r5.out`, `gb10-a-L1-seed1-teach2`, `gb10-a-L1-seed1-teach2.console.log`, `gb10-a-L1-seed1-teach2.rc`, `gb10-a-L1-seed1.console.log`, `gb10-a-L1-seed1.rc`, `gb10-a-L2-seed1`, `gb10-a-L2-seed1.out`, `gb10-a-L3-seed1`, `gb10-a-L3-seed1.out`, `gb10-a-L4-seed1`, `gb10-a-L4-seed1.out` | 528 | 16.5 |
| 7.3 L7 filling | `gb10-a-L7-seed1`, `gb10-a-L7-seed1.out` | 157 | 19.7 |
| 7.4 L6 two-step and references (also L5) | `gb10-a-L5-seed1`, `gb10-a-L5-seed1.out`, `gb10-a-L6-seed1.out`, `gb10-a-L6-seed1b`, `gb10-a-L6-seed1b-ref.out`, `gb10-a-L6-seed1b.out` | 338 | 33.5 |
| 7.5 Cost (CUDA logs; perf/) | `perf`, `tf-pocket-cuda` | 2249 | 40.6 |
| 7.6 v2 question forms (v2, cap) | `gb10-a-cap`, `gb10-a-cap.out`, `gb10-a-v2`, `gb10-a-v2.out` | 1138 | 95.7 |
| 7.7 Notes only / notes plus neurons / take-back wordings (notes, both, tb-explicit, tb-rule) | `gb10-a-both`, `gb10-a-both.out`, `gb10-a-notes`, `gb10-a-notes.out`, `gb10-a-tb-explicit`, `gb10-a-tb-explicit.out`, `gb10-a-tb-rule`, `gb10-a-tb-rule.out` | 2177 | 180.8 |
| 7.8 Promotion (nights2) | `gb10-a-nights2`, `gb10-a-nights2.out` | 106 | 4.0 |

`perf/` (repo root, not under runs/) holds `LESSON-TIME.md`, `CUDA-POCKET.md`, `lesson-time-raw/`, `cuda-raw/` (the `.tar` source snapshots were dropped). Its manifest rows start with `perf/`. Sizes above are before gzip and dedupe. This branch also un-ignores `runs/` in `.gitignore`.
