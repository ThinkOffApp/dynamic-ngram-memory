# Raw run files behind the paper

These are the raw files of the GPU runs and cost measurements behind Section 7 of the paper *Fast to Write, Slow to Learn*: every answer, its grade, the judge reports, the run summaries, the server and run logs, and the scripts that ran them. They come from the test suite's working repository (branch `run-files`, commit `0aaba45`, folders `runs/` and `perf/`), and were cleaned for privacy only. [../PAPER-RESULTS-DETAIL.md](../PAPER-RESULTS-DETAIL.md) names the file behind each number.

The test suite was called LLWZ while these runs were made, so its old names (`llwz`, `lw-logic`, `zengine`, `lw-runs`) appear in paths and labels. They are left as they are: they are part of the data.

## How the files are stored

`runs/` and `perf/` keep the layout of the source branch. Two rules from the source kept it small, and are explained in [runs/README.md](runs/README.md), the source's own note:

- Every `.log` is gzipped on its own (`name.log.gz`; read it with `gzcat` or `zcat`).
- Files of 4 KB or more that occur more than once are stored once, under `runs/_shared/<first 12 hex digits of the sha256>-<name>`. [runs/MANIFEST.tsv](runs/MANIFEST.tsv) maps each of the 6,693 original paths to its sha256, its size and the file that holds its bytes. Paths in its `stored_at` column are relative to this folder.

## The folders

| Folder (under `runs/` unless marked) | What it holds | Paper section |
| --- | --- | --- |
| `gb10-a-L1-seed1`, `gb10-a-L1-seed1-teach2`, `gb10-a-L1-seed1-r2` to `-r5`, with their `.rc` files | Test set L1 (one fact) on the GB10: the first run, the teach phase run again, and four later repeats | 7.2 |
| `gb10-a-L2-seed1`, `gb10-a-L3-seed1`, `gb10-a-L4-seed1` | Test sets L2 to L4 on the GB10: teach, delete by zeroing and by reteaching, judged answers | 7.2, and the deletion paragraph of 7.4 |
| `gb10-a-L7-seed1` | Test set L7: 64 facts taught in a row, with a check after every fifth lesson (`capacity.txt` is the summary) | 7.3 |
| `gb10-a-L5-seed1`, `gb10-a-L6-seed1b`, `gb10-a-L6-seed1.out`, `gb10-a-L6-seed1b.out`, `gb10-a-L6-seed1b-ref.out` | Test sets L5 and L6: the pocket, the untaught model, every lesson in the prompt and the exact lookup (`gb10-a-L6-seed1b/reference/report.txt`) | 7.4, and the L6 pocket column of 7.7 |
| `tf-pocket-cuda`; `perf/` (outside `runs/`) | What learning and serving cost on the GB10: per-lesson timings, decode and prompt benchmarks before and after the CUDA fix, build logs and scripts. `perf/LESSON-TIME.md` and `perf/CUDA-POCKET.md` are the write-ups; `perf/lesson-time-raw/` and `perf/cuda-raw/` their raw files | 7.5 (and Table 1's Living Weights row) |
| `gb10-a-v2`, `gb10-a-cap` | Question forms v2 on L1 to L5 and the two take-back sets (`neg`, `neg2`); `cap` reruns L5 and L6 with every lesson in the prompt at a 6,144-token answer cap for thinking on | 7.6; the pocket and prompt columns of 7.7; `cap` also the prompt column of 7.4 |
| `gb10-a-notes`, `gb10-a-both`, `gb10-a-tb-explicit`, `gb10-a-tb-rule` | TensorFold Notes only; notes plus Living Weights; the two stronger take-back wordings (clearer label; label plus a line) | 7.7 |
| `gb10-a-nights2` | Promotion, first run: five days and their nights | 7.8 |
| `_shared` | The stored bytes of files that occur more than once (see above) | all |
| `MANIFEST.tsv`, `README.md` | The map of every original path, and the source's own note | all |

A `.out` file next to a folder is the console output of that run. Some of them, and some files inside the folders, are stored under `_shared`; `MANIFEST.tsv` says where. A phase folder holds the answers (`answers.jsonl`) and the verdicts of the judge at run time (`verdicts.json`, summed in `report.txt`); most also hold the verdicts of a later judge (`verdicts-judge2.json`, and `verdicts-judgefix.json`, which passes the same answers); [../PAPER-RESULTS-DETAIL.md](../PAPER-RESULTS-DETAIL.md) says which numbers come from which.

Three later results of Section 7.7 (names and the first person as subjects, the new notes block, and German and Finnish lookup) were measured on the Mac with a 1,000,000-note store, not in these runs, and their files are not here. [../PAPER-RESULTS-DETAIL.md](../PAPER-RESULTS-DETAIL.md) says where they are.

## What was changed for publication

Only names were replaced. The table counts every replacement, inside the gzipped logs as well.

| What | Replaced by | Occurrences | Files |
| --- | --- | ---: | ---: |
| The GB10 machine's network name | `gb10-a` | 15,711 | 482 |
| The same machine's system host name (for example the `host` field of `summary.json`) | `gb10-a` | 116 | 96 |
| Linux home directory, `/home/<user>` | `~` | 13,347 | 671 |
| macOS home directory, `/Users/<user>` | `~` | 9 | 2 |
| The user name on its own in prose (`perf/LESSON-TIME.md`) | "the project owner" | 2 | 1 |
| Handles of the AI agents that ran or wrote things | "a reviewer agent" in prose (8), "reviewer" in identifiers (7: a process name in host snapshots, and a folder pattern in `runs/README.md`) | 15 | 6 |
| File and folder names with the machine's network name | `gb10-a` | 19 folders and 23 files; 1,352 file paths in all | |

Searched for and not found: other machine names, IP addresses other than the loopback address `127.0.0.1`, email addresses, and API keys, tokens or passwords. The words "api_key", "password" and "secret" occur only as words: in a benchmark script (an `--api-key` option whose value is never stored) and its results (`"api_key_sent": false`), in file names of the engine's source tree, and in test questions and model answers.

`home/facts.json` (26 times) is not a home directory: it is the file `facts.json` in a folder called `home/` inside a run folder, the server's own HOME for that run. It names no person and was left as it is.

### Machine labels

| Label | Hardware |
| --- | --- |
| `gb10-a` | ASUS Ascent GX10 with an NVIDIA GB10 (sm_121, memory shared with the host), aarch64, Ubuntu 24.04.5 LTS, kernel 7.0.0-1019-nvidia, driver 580.178.04, CUDA 13.0 (as recorded in `perf/CUDA-POCKET.md` and `perf/cuda-raw/r109/raw/machine.txt`) |

Every run on a GPU here ran on this one machine. It appears in the source under two names, its network name and its system host name, and both became `gb10-a`. The Mac that `perf/` mentions, where the files were hashed and read, is called only "the Mac" in them, so there was nothing to replace.

### Everything else is as it was

Apart from the replacements above and the manifest's hashes (last point but one), every file is byte-identical to the source:

- 3,664 of the 4,348 files are unchanged byte for byte (comparing the logs after decompression). The other 684 differ only by the replacements in the table, and `runs/MANIFEST.tsv` also by its hashes. This was checked by aligning every file with its source, token by token.
- A `.log.gz` whose log needed a replacement was recompressed with the settings of the originals (gzip level 9, modification time 0, no stored name); recompressing an unchanged log that way gives back the original file byte for byte. 219 of the 1,063 `.log.gz` files were recompressed; the other 844 are the original files.
- `runs/MANIFEST.tsv` describes the published bytes. For the 965 of its 6,693 rows whose file changed, the `sha256_of_original_bytes` and `size_bytes` columns give the redacted file, and the 185 files under `runs/_shared/` whose bytes changed are named after their new hash. The hashes of the files before redaction are not published, because they would let anyone confirm a guess of a replaced name.
- Every `.json` file (2,489) and every line of every `.jsonl` file (258 files) parses.

[SHA256SUMS](SHA256SUMS) lists the sha256 of every file in this folder except itself (`shasum -a 256 -c SHA256SUMS` from this folder checks them).
