# Addendum 1 (19 Oct 2026, 12:57 CEST): how the blocks are looked up

Written after the first validation attempt and before any other arm was asked. decision-rules.md is unchanged.

- First attempt: blocks built over HTTP with the route's default lookup. The HTTP route's default is the subject+value
  hop on (notes_decode_fields.zig: default_value_hop = true), while the alone check calls store.retrieve(), which is
  retrieveWith(..., hop = null), the snapshot's own hop (notes_store.zig:270-273). Result: A1x matched night 12 on 38 of
  44 (prompt, tokens, sha) and A1y matched night 14 on 31 of 44; every mismatch had a longer prompt (more notes).
  Those replies are kept in validation/attempt1/ and are not used.
- Fix: every block request also sends "notes_value_hop": false ("the snapshot's own hop with false"), which makes the
  HTTP lookup the same call the alone check makes. Nothing else changes.
