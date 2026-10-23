"""The proxy's block is retrieve()'s, byte for byte: the L6-seed1 teach store and its 56 questions.

The reference is notes.set_store + NoteStore.retrieve, built here independently of the proxy. The proxy's side is a
store seeded into a journal on disk and read back (so persistence is in the path). Controls: another phase's store
and another set's store must give different blocks, or the comparison could not tell stores apart.
"""
import unittest

from notes_proxy.store import Notes, nm
from notes_proxy.tests import helpers as H


def compare(notes, ref, questions, **kw):
    """The questions whose block, ids, tokens or past flag differ between the proxy's lookup and retrieve()."""
    out = []
    for q in questions:
        a = notes.lookup(q["text"], **kw)
        b = ref.retrieve("user", q["text"], **kw)
        if (a["block"], a["ids"], a["tokens"], a["past"]) != (b["block"], b["ids"], b["tokens"], b["past"]):
            out.append(q["qid"])
    return out


class BlockEqualsRetrieve(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = H.TempDir()
        cls.S, cls.ref = H.reference()
        cls.notes = Notes(H.seeded(cls.tmp, "teach.jsonl"), "user", H.tokenizer())
        print(f"\n[test_block] tokenizer: {H.tokenizer_spec()}")

    @classmethod
    def tearDownClass(cls):
        cls.notes.close()
        cls.tmp.cleanup()

    def test_56_questions_as_asked(self):
        qs = self.S["questions"]
        self.assertEqual(len(qs), 56)
        self.assertEqual(compare(self.notes, self.ref, qs), [])
        found = sum(1 for q in qs if self.ref.retrieve("user", q["text"])["block"])
        print(f"[test_block] 56 of 56 blocks equal; {found} of them hold notes")
        self.assertGreater(found, 40)            # the comparison is mostly of blocks that hold notes

    def test_past_forced_and_exclusions(self):
        qs = self.S["questions"]
        ids = sorted(n["id"] for n in self.ref.notes_of("user"))[:2]
        self.assertEqual(compare(self.notes, self.ref, qs, past=True), [])
        self.assertEqual(compare(self.notes, self.ref, qs, exclude=ids), [])
        self.assertEqual(compare(self.notes, self.ref, qs, past=True, exclude=ids), [])
        self.assertEqual(compare(self.notes, self.ref, qs, cap=128), [])

    def test_control_another_phase(self):
        other = Notes(H.seeded(self.tmp, "del-b.jsonl", phase="del-B"), "user", H.tokenizer())
        try:
            diff = compare(other, self.ref, self.S["questions"])
        finally:
            other.close()
        print(f"[test_block] control, phase del-B's store: {len(diff)} of 56 blocks differ")
        self.assertGreater(len(diff), 0)

    def test_control_another_set(self):
        other = Notes(H.seeded(self.tmp, "seed2.jsonl", set_path=H.L6_OTHER), "user", H.tokenizer())
        try:
            diff = compare(other, self.ref, self.S["questions"])
        finally:
            other.close()
        print(f"[test_block] control, L6-seed2's teach store: {len(diff)} of 56 blocks differ")
        self.assertGreater(len(diff), 40)

    def test_restart_gives_the_same_blocks(self):
        path = H.seeded(self.tmp, "again.jsonl")
        first = Notes(path, "user", H.tokenizer())
        a = [first.lookup(q["text"])["block"] for q in self.S["questions"]]
        first.close()
        second = Notes(path, "user", H.tokenizer())
        b = [second.lookup(q["text"])["block"] for q in self.S["questions"]]
        second.close()
        self.assertEqual(a, b)

    def test_abstain_line(self):
        line = "If the notes do not say, answer that you do not know."
        path = H.seeded(self.tmp, "abstain.jsonl")
        n = Notes(path, "user", H.tokenizer(), abstain_line=line)
        try:
            for cap in (512, 160):
                for q in self.S["questions"]:
                    r = n.lookup(q["text"], cap=cap)
                    if not r["block"]:
                        continue
                    self.assertEqual(r["block"].split("\n")[0], nm.HEAD + " " + line)
                    self.assertLessEqual(len(H.tokenizer().encode(r["block"])), cap)   # the cap counts the line
                    self.assertEqual(r["tokens"], len(H.tokenizer().encode(r["block"])))
            # without the line, the first line is the reference's own
            q = self.S["questions"][0]["text"]
            self.assertEqual(self.notes.lookup(q)["block"].split("\n")[0], nm.HEAD)
        finally:
            n.close()
        with self.assertRaises(nm.NotesError):
            Notes(H.seeded(self.tmp, "bad.jsonl"), "user", H.tokenizer(), abstain_line="two\nlines")

    def test_one_owner_per_store(self):
        path = H.seeded(self.tmp, "owner.jsonl")
        with self.assertRaises(Exception) as cm:
            Notes(path, "someone-else", H.tokenizer())
        self.assertIn("another owner", str(cm.exception))

    def test_one_proxy_per_store(self):
        path = H.seeded(self.tmp, "locked.jsonl")
        a = Notes(path, "user", H.tokenizer())
        try:
            with self.assertRaises(Exception) as cm:
                Notes(path, "user", H.tokenizer())
            self.assertIn("another notes proxy", str(cm.exception))
        finally:
            a.close()


if __name__ == "__main__":
    unittest.main()
