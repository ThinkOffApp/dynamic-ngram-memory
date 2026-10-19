# Promotion: notes that earn it move into neurons

> **In this repository.** This page comes from the test suite's working repository (commit `2083ee9`, MIT license: [LICENSES/LLWZ-MIT.txt](../LICENSES/LLWZ-MIT.txt)). The code it names (`zengine/`, `tools/`, `examples/`, `lean/`, `zig/notes/`, `docs/TEST-LOGIC.md`) is in [tests/](../tests/) here. For publication, home-directory paths are shown as `~`, a few names were removed, and links were pointed to their new places; nothing else was changed.

Paper Section 5, first build: `zengine/promote.py`. Status: built and self-tested against the mock server
(`python3 zengine/promote.py selftest`); the first GPU run is `examples/nights/L6-seed1-nights.json` on a GB10.

## What it does

Facts are told over days. Every fact is a note at once (`zengine/notes.py`). Each night, notes that earned it are
taught into the pocket on the running server, and a note whose fact the neurons answer alone leaves the prompt.

| Step | Rule |
|---|---|
| Qualify (each day) | A current fact note (a statement or a change; not a take-back, not a dated value) with at least one question resting on it (`notes.needed_notes`), all of them right in the system arm that day. |
| Due | Qualified `--hold` days running (default 2). A change or take-back of the fact ends the streak at once. |
| Teach (night) | A copy of the serving folder, `--slide`, each due fact's own sentence, oldest first. TensorFold keeps a fact only when its own held-out checks pass; one it does not keep stays a note (missed). |
| Regress (night) | The taught copy answers today's questions with today's notes. More than `--tolerance` questions (default 0) that passed today and fail now: the night is dropped and the due facts are rejected. |
| Alone (night) | Each fact in the pocket answers its questions without its own note. Right `--alone` nights running (default 2): the note leaves the prompt ("in weights"). One wrong answer brings it back. |
| Stale (night) | A fact in the pocket whose note changed or was taken back. The neurons are not zeroed (zeroing can leave the fact or hit a neighbour: paper Section 7.4) and TensorFold's per-fact delete is not used. The pocket is rebuilt from the untaught source with the facts that still hold, then the due ones; if that is worse, without the due ones; if that is worse too, the untaught source (every fact back to notes) is served anyway. |

The note stays the record of truth throughout: a fact in weights keeps its note in the store, and a change or
take-back goes to the notes the moment it is said.

## Arms

Every day has two arms on the same questions, judged by the rule book as phase `dayK`:

- **system**: the serving folder (the source until a night promotes something) with each question's notes, minus the
  notes in weights;
- **control**: the untaught source with every note (the notes-only arm of paper Section 7.7).

The report (`WORK/promote-report.txt`) gives per day: questions right in each arm, the prompt tokens each arm sent,
the facts in weights and in the pocket, and the questions whose right answer changed since the day before (where a
stale pocket would show).

## The nights set

`promote.py make-set BASE --days SPEC [--extra EXTRA.json]`: the base set's lessons told over days, plus extra lessons
told later. `examples/nights/L6-seed1-nights.json` is L6 seed 1 (29 lessons) over days 1-10, 11-20, 21-29, then day 4 adds
three lessons from `examples/nights/L6-seed1-nights.extra.json` that change or take back day 1 and day 2 facts (my car is no
longer red; my dog's eyes are brown now; my sister lives in Bergen now), and day 5 tells nothing new. On day 4, 11
of the 56 questions have a new right answer, several of them two-step (where the brother lives follows the sister).

## Not yet

- One seed, one set, thinking off for the nightly decisions.
- A night interrupted after it accepted a folder but before it finished its alone checks is rerun from its start.
- Rejected facts are not retried on a later night.
