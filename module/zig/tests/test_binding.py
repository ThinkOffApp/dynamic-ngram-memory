"""Native binding lifecycle and serialization; build module/zig before running."""
import concurrent.futures
import json
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from notes_proxy.zstore import NativeStoreError, ZStore


class Binding(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="notes-binding-")
        self.addCleanup(self.tmp.cleanup)
        self.snapshot = json.loads((ROOT / "module/zig/src/server/notes/testdata/tb_empty_bytes_snapshot.json").read_bytes())

    def open(self, owner="alice", encode=None):
        snapshot = dict(self.snapshot, owner=owner)
        store = ZStore(snapshot, pathlib.Path(self.tmp.name) / owner,
                       encode or (lambda text: list(text.encode())))
        self.addCleanup(store.close)
        return store

    def test_concurrent_calls_are_serialized_and_return_owned_results(self):
        store = self.open()
        def write(k):
            return store.write(f"My shelf contains item alpha{k}.", 739900000000 + k)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as workers:
            results = list(workers.map(write, range(12)))
        ids = [r["note"]["id"] for r in results]
        self.assertEqual(len(set(ids)), 12)
        self.assertEqual(len(store.listing()), 12)
        store.close()
        store.close()  # idempotent
        with self.assertRaises(NativeStoreError):
            store.listing()

    def test_separate_owner_handles_do_not_share_notes(self):
        alice, bob = self.open(), self.open("bob")
        alice.write("My aunt Brigitta grows tomatoes.", 739900000000)
        question = "What does my aunt Brigitta grow?"
        self.assertIn("Brigitta", alice.lookup(question, 1000)["block"])
        self.assertEqual(bob.lookup(question, 1000)["block"], "")

    def test_tokenizer_exception_is_closed_without_exposing_its_text(self):
        def failing(_):
            raise ValueError("private-note-sentinel")
        store = self.open(encode=failing)
        with self.assertRaises(NativeStoreError) as error:
            store.write("A secret fact.", 739900000000)
        self.assertNotIn("private-note-sentinel", str(error.exception))
        self.assertEqual(store.listing(), [])


if __name__ == "__main__":
    unittest.main()
