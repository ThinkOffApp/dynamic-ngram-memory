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
