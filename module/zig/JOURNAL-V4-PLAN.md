# Native adapter journal v4 — implementation contract (not implemented)

Keep SOURCE.json and its exported files pristine. The adapter extension must
have a reproducible derivation/patch manifest recording original and derived
hashes; do not silently change what the upstream receipts qualified.

The current default stays Python. v4 is not a sidecar and must not be emitted by
the opt-in until its reader, replay, rollback and erase closure are tested.
An old v3 reader must reject the v4 header before truncating or rewriting it.
The v4 reader must retain support for valid v1/v2/v3 inputs and upgrade only via
an atomic, checked rewrite. A failed upgrade leaves the original bytes intact.

## Record and API contract

Expose `ZStore.commit_intake(text, told, subject, attribute, verdict, retire=(),
depends=())`. Resolve caller numeric IDs to durable source identities before
preparing one record. No loop of independently committed native operations.

- Learned record: raw text plus an explicit retirement target list. Validate
  every target and its owner/current state before mutation. Allocate the reply,
  replacement index and undo state before durable commit. All retirements and
  the one learned note become visible together, or none do.
- No-op record: raw text, verdict ALREADY_KNOWN/IMPLIED, explicit dependencies,
  no learned note and no state/index change. It still consumes a write source
  and updates the monotonic told floor. No fabricated note or changed raw text.
- Targets/dependencies are stored as source identities, not display numbers.
  Replay validates lists, duplicates, bounds and references; invalid records
  fail closed, never partially retire targets.
- Preserve the record/commit-marker protocol: a torn/uncommitted record cannot
  become half an intake. Erase traverses dependency edges (including no-ops)
  and rewrites the remaining committed history with source floors preserved.
- The old parser's change-with-null-target constraint must be addressed with
  an explicit v4 shape and round-trip test, not a nonexistent/self target hack.

## Required fail-first gates

1. Two current facts -> one atomic set change; reopen has exactly the new
   current fact and both old facts in history; erase the change restores both.
2. Semantic no-op retains differing raw text without adding a note; reopen is
   unchanged; erase its dependency removes the no-op's raw text as well.
3. Kill the actual writer after partial record bytes and before its commit
   marker; reopen has all old facts and no partial retirement/no-op record.
   Repeat after the marker and verify the whole committed operation replays.
4. Malformed/truncated lists, missing sources, wrong owner, cap refusal and
   allocation/I/O failure leave both disk and live state unchanged or explicitly
   poisoned, following the existing uncertain-disk contract.
5. Legacy formats reopen/upgrade, old-reader v4 refusal, manifest integrity and
   existing core/binding/HTTP regression tests. Only then wire structured intake
   and implement legacy Python journal migration with stable wN/sN identities.

The first API contract tests intentionally fail until this extension exists;
they are not passing-feature evidence or a production qualification.
