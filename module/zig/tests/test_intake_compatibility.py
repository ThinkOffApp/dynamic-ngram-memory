"""Integration gates: do not replace HTTP Notes until native intake preserves all supported behaviour.

Run with NOTES_PROXY_TEST_TOKENIZER pointing at a real local tokenizer.json. The first replacement mapping is
covered here; other structured cases explicitly refuse and the adapter is not yet qualified for HTTP use.
"""
import os
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from notes_proxy.store import Notes, nm
from notes_proxy.zstore import IntakeStore, NativeStoreError
import notes_export


class IntakeCompatibility(unittest.TestCase):
    def test_dated_change_keeps_current_and_past_block_parity(self):
        spec = os.environ.get("NOTES_PROXY_TEST_TOKENIZER")
        self.assertTrue(spec and spec != "fake")
        tok = nm.load_tokenizer(spec)
        with tempfile.TemporaryDirectory(prefix="notes-dated-parity-") as root:
            ref = Notes(pathlib.Path(root) / "reference.jsonl", "alice", tok)
            self.addCleanup(ref.close)
            ref.store.owner("alice", create=True)
            snapshot, vocab = notes_export.snapshot(ref.store, "alice")
            with IntakeStore(snapshot, pathlib.Path(root) / "native.journal", vocab.ids) as native:
                for text in ("My eyes are brown.", "My eyes were brown until 11 October 2026; now they are green."):
                    expected = ref.write_text(text)
                    result = native.write_text(text, expected["told"])
                    self.assertEqual(result["verdict"], expected["verdict"])
                for past in (False, True):
                    question = "What colour were my eyes?" if past else "What colour are my eyes?"
                    self.assertEqual(native.lookup(question, 512, past=past)["block"],
                                     ref.lookup(question, cap=512, past=past)["block"])

    def test_retraction_uses_native_tombstone_and_erases_dependent_record(self):
        spec = os.environ.get("NOTES_PROXY_TEST_TOKENIZER")
        self.assertTrue(spec and spec != "fake")
        tok = nm.load_tokenizer(spec)
        with tempfile.TemporaryDirectory(prefix="notes-retract-parity-") as root:
            ref = Notes(pathlib.Path(root) / "reference.jsonl", "alice", tok)
            self.addCleanup(ref.close)
            ref.store.owner("alice", create=True)
            snapshot, vocab = notes_export.snapshot(ref.store, "alice")
            with IntakeStore(snapshot, pathlib.Path(root) / "native.journal", vocab.ids) as native:
                first = None
                for text in ("I live in Helsinki.", "I no longer live in Helsinki."):
                    expected = ref.write_text(text)
                    result = native.write_text(text, expected["told"])
                    first = first or result["note"]["id"]
                actual_current = [r["text"] for r in native.listing() if r["state"] == "current"]
                self.assertEqual(actual_current, [r["text"] for r in ref.listing()])
                self.assertEqual(native.lookup("Where do I live?", 512)["block"],
                                 ref.lookup("Where do I live?", cap=512)["block"])
                native.erase(first)
                self.assertEqual(native.listing(), [])
                self.assertEqual(native.records(), [])

    def test_parsed_home_city_replacement_keeps_only_new_fact_current(self):
        spec = os.environ.get("NOTES_PROXY_TEST_TOKENIZER")
        self.assertTrue(spec and spec != "fake", "a real local tokenizer is required")
        tok = nm.load_tokenizer(spec)
        with tempfile.TemporaryDirectory(prefix="notes-intake-parity-") as root:
            ref = Notes(pathlib.Path(root) / "reference.jsonl", "alice", tok)
            self.addCleanup(ref.close)
            ref.store.owner("alice", create=True)
            snapshot, vocab = notes_export.snapshot(ref.store, "alice")
            with IntakeStore(snapshot, pathlib.Path(root) / "native.journal", vocab.ids) as native:
                for text in ("I live in Helsinki.", "I live in Berlin."):
                    expected = ref.write_text(text)
                    native.write_text(text, expected["told"])
                expected_current = [row["text"] for row in ref.listing()]
                actual_current = [row["text"] for row in native.listing() if row["state"] == "current"]
                self.assertEqual(actual_current, expected_current)
                # The existing native change path, not a second retirement implementation, keeps history.
                self.assertEqual(len(native.listing()), 2)
                before = native.listing()
                duplicate = native.write_text("I live in Berlin.", expected["told"] + 100)
                self.assertEqual(duplicate["verdict"], "ALREADY_KNOWN")
                self.assertEqual(native.listing(), before)
            with IntakeStore(snapshot, pathlib.Path(root) / "native.journal", vocab.ids) as reopened:
                changed = reopened.write_text("I live in Stockholm.", expected["told"] + 200)
                self.assertEqual(changed["verdict"], "REPLACES")
                self.assertEqual([row["text"] for row in reopened.listing() if row["state"] == "current"],
                                 ["I live in Stockholm."])
                reopened.erase(changed["note"]["id"])
                self.assertEqual([row["text"] for row in reopened.listing() if row["state"] == "current"],
                                 ["I live in Berlin."])


if __name__ == "__main__":
    unittest.main()
