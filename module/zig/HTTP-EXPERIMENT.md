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

The adapter currently lists native rows for mapping responses and timestamps;
the capped benchmark must measure this path, including that overhead, before
any speed claim. Native lookup does not use a Python note/index mirror.
