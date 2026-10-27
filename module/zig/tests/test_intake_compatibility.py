"""RED integration gate: do not replace HTTP Notes until native intake preserves this behaviour.

Run with NOTES_PROXY_TEST_TOKENIZER pointing at a real local tokenizer.json. This intentionally exposes the
remaining schema-intake gap between the low-level native API and the existing proxy, not a qualified adapter.
"""
import os
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from notes_proxy.store import Notes, nm
from notes_proxy.zstore import ZStore
import notes_export


class IntakeCompatibility(unittest.TestCase):
    def test_parsed_home_city_replacement_keeps_only_new_fact_current(self):
        spec = os.environ.get("NOTES_PROXY_TEST_TOKENIZER")
        self.assertTrue(spec and spec != "fake", "a real local tokenizer is required")
        tok = nm.load_tokenizer(spec)
        with tempfile.TemporaryDirectory(prefix="notes-intake-parity-") as root:
            ref = Notes(pathlib.Path(root) / "reference.jsonl", "alice", tok)
            self.addCleanup(ref.close)
            ref.store.owner("alice", create=True)
            snapshot, vocab = notes_export.snapshot(ref.store, "alice")
            with ZStore(snapshot, pathlib.Path(root) / "native.journal", vocab.ids) as native:
                for text in ("I live in Helsinki.", "I live in Berlin."):
                    expected = ref.write_text(text)
                    native.write(text, expected["told"])
                expected_current = [row["text"] for row in ref.listing()]
                actual_current = [row["text"] for row in native.listing() if row["state"] == "current"]
                self.assertEqual(actual_current, expected_current)


if __name__ == "__main__":
    unittest.main()
