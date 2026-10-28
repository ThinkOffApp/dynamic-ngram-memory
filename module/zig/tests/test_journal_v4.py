"""Fail-first v4 API gates. Deliberately red until the native extension exists."""
import json
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from notes_proxy.zstore import ZStore


class JournalV4(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="notes-v4-contract-")
        self.addCleanup(self.tmp.cleanup)
        self.path = pathlib.Path(self.tmp.name) / "native.journal"
        self.snapshot = json.loads((ROOT / "module/zig/src/server/notes/testdata/tb_empty_bytes_snapshot.json").read_bytes())

    def open(self):
        return ZStore(self.snapshot, self.path, lambda text: list(text.encode()))

    def test_atomic_retirement_replays_and_erasing_change_restores_both(self):
        with self.open() as store:
            first = store.write("I like apples.", 739900000000)["note"]["id"]
            second = store.write("I like cheese.", 739900000100)["note"]["id"]
            result = store.commit_intake("I only like bananas now.", 739900000200,
                subject="me", attribute="schema:likes", verdict="REPLACES", retire=(first, second))
            changed = result["note"]["id"]
        with self.open() as store:
            self.assertEqual([r["text"] for r in store.listing() if r["state"] == "current"],
                             ["I only like bananas now."])
            store.erase(changed)
            self.assertEqual([r["text"] for r in store.listing() if r["state"] == "current"],
                             ["I like apples.", "I like cheese."])

    def test_noop_keeps_raw_text_without_state_change_and_follows_erase_dependency(self):
        raw = "My residence is Helsinki, as already recorded."
        with self.open() as store:
            first = store.write("I live in Helsinki.", 739900000000)["note"]["id"]
            before = store.listing()
            result = store.commit_intake(raw, 739900000100, subject="me", attribute="schema:home_city",
                                         verdict="IMPLIED", depends=(first,))
            self.assertIsNone(result["note"])
            self.assertEqual(store.listing(), before)
        self.assertIn(raw.encode(), self.path.read_bytes())
        with self.open() as store:
            self.assertEqual(store.listing(), before)
            store.erase(first)
        self.assertNotIn(raw.encode(), self.path.read_bytes())


if __name__ == "__main__":
    unittest.main()
