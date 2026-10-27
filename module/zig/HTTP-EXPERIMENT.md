# Experimental HTTP opt-in

Build the shared library, then pass `--store-engine zig` with a real local HF
tokenizer and a new single-owner journal path. Python is still the default.
The native journal is not the Python JSONL format. Existing Python journals
are refused, not converted. Keep the original journals and do not point a
production service at this experiment yet.

The first HTTP gate covers write, stable `w1` GET, lookup, DELETE and reopen
without source reuse. Native handle/journal ownership tests also pass.
This is not full HTTP parity or production qualification: multi-user mode,
seed migration, test vocabularies, wording/schema options, atomic multi-target
retirement and semantic no-op intake remain unfinished. Numeric positions and
renumbered reporting still need compatibility work. No scale result is claimed.

The adapter retains numeric-ID/source pairs and the last committed timestamp,
not a Python note/index mirror. Listing and erase still enumerate native rows.

## First MacBook measurement (27 October 2026)

`bench_native_http_store.py --size 1000`, real local Nemotron tokenizer, synthetic
single-owner free-text writes, no model, 45-second external subprocess cap:

| path | fill 1k | one lookup | erase w1 | process max RSS |
| --- | --- | --- | --- | --- |
| cc86591 adapter | 14.909 s | 5.084 ms | 145.975 ms | 293306368 bytes |
| source-ID/timestamp metadata optimization | 12.710 s | 1.639 ms | 153.705 ms | 295878656 bytes |

One sample each, same machine; not a cross-machine comparison or scaling proof.
The initial 15-second-capped 10k/100k attempts timed out during fill at around
900 writes: neither produced an erase measurement. No completed 10k/100k result
yet. This measures the HTTP-selected adapter methods, not HTTP socket overhead.

Same-machine bulk-fixture control, 1k notes, one sample each (synthetic committed
journal, real reopen, then one real extra write/lookup/erase):

| engine | reopen | single write | lookup | erase w1 | max RSS bytes |
| --- | --- | --- | --- | --- | --- |
| Python | 1.111 s | 8.201 ms | 16.554 ms | 1.126 s | 260243456 |
| Zig adapter | 0.197 s | 19.796 ms | 1.747 ms | 0.163 s | 296222720 |

The native bulk fixture copies the actual writer's snapshot-bound header and
uses normal committed add records; Python uses Journal.rewrite. Fixture creation
is not a native bulk-ingestion API or a measurement of normal HTTP teaching.

Larger native tiers (same synthetic workload, single samples):

| notes | reopen | lookup | erase | max RSS bytes |
| --- | --- | --- | --- | --- |
| 10k, 5250934 | 1.061 s | 9.796 ms | 1.102 s | 363282432 |
| 100k, lightweight metadata boundary | 10.550 s | 65.629 ms | 10.670 s | 1020411904 |

The first 100k attempt exceeded the 1024 MiB sampled-RSS threshold during erase.
After eliminating full note/record JSON copies for get/erase/startup metadata,
100k completed under the same 180-second/1024-MiB watchdog. The watchdog samples
every 200 ms; it is not an OS hard MemoryMax. The additional write at 100k is
correctly refused with 507 at the reviewed 100000-note limit, not timed as a
successful write. These points are consistent with roughly linear erase cost
for this workload, not a proof for structured/dependent records or all inputs.

Do not attribute the difference from old raw-core timings solely to the Python
adapter: this C ABI invokes the actual reviewed Storage, which builds an index
on every write, and erase replays kept records with real tokenizer callbacks.
Workload/tokenizer/build equivalence needs checking before that comparison.

Completed same-Mac Python 10k bulk control: reopen 109.908 s, single write
26.830 ms, lookup 166.806 ms, erase 108.793 s, max RSS 398163968 bytes;
240-second wall cap and the same 1024-MiB sampled watchdog. No Python 100k
measurement was attempted. The current native library is a **Debug build**, not
an optimized release, another reason not to equate it with old raw-core timings.
Current-native repeat at 1k/10k: erase 0.142/1.005 s, max RSS
295534592/353271808 bytes. These are single runs, not statistical confidence
intervals. All three current-native tiers used the same workload and Debug
implementation; 100k was measured earlier in a separate capped worker.

## ReleaseFast rerun

Built from `3dd723298560859333e77770d59889cc57952b07` using Zig 0.17.0,
`zig build --build-file module/zig/build.zig -Doptimize=ReleaseFast`.
Library SHA-256: `f6cec12e3def0c6d5ee9f4dcc2b8641fd4324560bc00351ee80f1cc387d306cf`.
Same real tokenizer, synthetic bulk fixtures and 180-second/1024-MiB sampled
watchdog; single samples:

| notes | reopen | single write | lookup | erase | max RSS bytes |
| --- | --- | --- | --- | --- | --- |
| 10k | 0.612 s | 17.330 ms | 1.830 ms | 0.690 s | 349011968 |
| 100k | 6.199 s | capacity 507 | 9.471 ms | 6.865 s | 1049378816 |

Both workers completed, but 100k has little headroom below the memory threshold;
this is not a hard-memory-limit qualification. Python remains the default until
the unfinished behavioral cases and full native-path suite are green.
