"""Real-tokenizer parity check, no download. Set NOTES_PROXY_TEST_TOKENIZER to tokenizer.json."""
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


class HFBinding(unittest.TestCase):
    def test_free_text_write_lookup_and_erase_with_real_vocabulary(self):
        tokenizer = os.environ.get("NOTES_PROXY_TEST_TOKENIZER")
        if not tokenizer or tokenizer == "fake":
            self.fail("set NOTES_PROXY_TEST_TOKENIZER to a local real tokenizer.json; no fake fallback")
        tok = nm.load_tokenizer(tokenizer)
        with tempfile.TemporaryDirectory(prefix="notes-hf-parity-") as root:
            ref = Notes(pathlib.Path(root) / "reference.jsonl", "alice", tok)
            self.addCleanup(ref.close)
            ref.store.owner("alice", create=True)
            snapshot, vocab = notes_export.snapshot(ref.store, "alice")
            self.assertIsNone(vocab.dense)
            with ZStore(snapshot, pathlib.Path(root) / "native.journal", vocab.ids) as native:
                text = "My neighbour Oskarina keeps seven tortoises."
                expected = ref.write_text(text, subject="me")
                written = native.write(text, expected["told"])
                self.assertEqual(written["source"], expected["source"])
                question = "What does my neighbour Oskarina keep?"
                for cap in (512, 160):
                    actual = native.lookup(question, cap)
                    wanted = ref.lookup(question, cap=cap)
                    self.assertEqual(actual["block"], wanted["block"])
                    self.assertEqual(actual["tokens"], wanted["tokens"])
                native.erase(written["note"]["id"])
                self.assertEqual(native.lookup(question, 512)["block"], "")


if __name__ == "__main__":
    unittest.main()
