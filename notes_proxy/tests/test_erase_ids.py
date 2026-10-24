"""A note id names one note forever.

The case that lost data: three notes, erase the first, then send the same erase again (a client retrying after a
timeout). When ids were the notes' places in the store, the erase moved every later note down by one, so the retry
erased a different note. An id is now the note's own write id (wN for the proxy's writes, sN for a seeded lesson), and
no erase, restart or later write gives it to another note. The retry is a 404 and touches nothing.

Each case runs on the store itself, through HTTP with one owner, and through HTTP with many users.
"""
import os
import unittest

from notes_proxy.store import Notes, StoreError
from notes_proxy.tests import helpers as H
from notes_proxy.tests import test_users as TU

TEXTS = ["My neighbour Oskarina keeps seven tortoises.", "My aunt Brigitta grows tomatoes.",
         "My cousin Aurelio repairs clocks."]


def by_id(listing):
    """id -> (text, source) of every note listed: what must not change for the notes an erase leaves."""
    return {n["id"]: (n["text"], n["source"]) for n in listing}


def disk(path):
    with open(path, "rb") as f:
        return f.read()


# ============================================================================================ the store itself
class StoreIds(unittest.TestCase):
    def setUp(self):
        self.tmp = H.TempDir()
        self.addCleanup(self.tmp.cleanup)
        self.path = self.tmp.file("me.jsonl")

    def open(self):
        n = Notes(self.path, "me", H.tokenizer())
        self.addCleanup(n.close)
        return n

    def three(self, n):
        ids = [n.write_text(t)["id"] for t in TEXTS]
        self.assertEqual(len(set(ids)), 3)
        return ids

    def test_a_repeated_erase_is_a_404_and_touches_no_other_note(self):
        n = self.open()
        ids = self.three(n)
        before = by_id(n.listing())
        self.assertEqual(n.erase(ids[0])["erased"], ids[0])
        on_disk = disk(self.path)
        with self.assertRaises(StoreError) as e:
            n.erase(ids[0])                                                 # the retry
        self.assertEqual(e.exception.status, 404)
        self.assertEqual([x["text"] for x in n.listing()], TEXTS[1:])      # the other two are still there
        self.assertEqual(by_id(n.listing()), {i: before[i] for i in ids[1:]})   # under the ids they had
        self.assertEqual(disk(self.path), on_disk)
        self.assertIsNone(n.get(ids[0]))
        self.assertEqual([n.get(i)["text"] for i in ids[1:]], TEXTS[1:])

    def test_a_repeated_erase_after_a_restart(self):
        n = self.open()
        ids = self.three(n)
        before = by_id(n.listing())
        n.erase(ids[0])
        n.close()
        n = self.open()
        with self.assertRaises(StoreError) as e:
            n.erase(ids[0])
        self.assertEqual(e.exception.status, 404)
        self.assertEqual([x["text"] for x in n.listing()], TEXTS[1:])
        self.assertEqual(by_id(n.listing()), {i: before[i] for i in ids[1:]})

    def test_an_erased_id_is_never_given_to_a_later_note(self):
        """Erase the newest note, then write: the new note gets a new id, before and after a restart. So a late
        retry of the erase cannot reach it."""
        n = self.open()
        ids = self.three(n)
        n.erase(ids[2])
        w = n.write_text("My uncle Bertil sails a ketch.")["id"]
        self.assertNotIn(w, ids)
        n.erase(w)
        n.close()
        n = self.open()
        w2 = n.write_text("My piano teacher Ludmila lives in Tampere.")["id"]
        self.assertNotIn(w2, ids + [w])
        for gone in (ids[2], w):
            with self.assertRaises(StoreError):
                n.erase(gone)
        self.assertEqual([x["text"] for x in n.listing()], TEXTS[:2] + ["My piano teacher Ludmila lives in Tampere."])


# ============================================================================================ HTTP, one owner
class HttpOneOwner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = H.TempDir()
        cls.stub = H.Stub()
        cls.stub.release.set()

    @classmethod
    def tearDownClass(cls):
        cls.stub.stop()
        cls.tmp.cleanup()

    def proxy(self, store):
        p = H.Running(self.stub.url, store)
        self.addCleanup(p.stop)
        return p

    def test_a_repeated_erase_is_a_404_and_touches_no_other_note(self):
        store = self.tmp.file("one-retry.jsonl")
        p = self.proxy(store)
        ids = [p.json("POST", "/v1/notes", {"text": t})[1]["id"] for t in TEXTS]
        before = by_id(p.json("GET", "/v1/notes?history=1")[1]["data"])
        st, e = p.json("DELETE", f"/v1/notes/{ids[0]}")
        self.assertEqual((st, e["erased"]), (200, ids[0]))
        on_disk = disk(store)
        st, e = p.json("DELETE", f"/v1/notes/{ids[0]}")                  # the retry
        self.assertEqual(st, 404)
        listed = p.json("GET", "/v1/notes?history=1")[1]["data"]
        self.assertEqual([x["text"] for x in listed], TEXTS[1:])
        after = {i: before[i] for i in ids[1:]}
        self.assertEqual(by_id(listed), after)
        self.assertEqual(disk(store), on_disk)
        self.assertEqual(p.json("GET", f"/v1/notes/{ids[0]}")[0], 404)
        self.assertEqual([p.json("GET", f"/v1/notes/{i}")[1]["text"] for i in ids[1:]], TEXTS[1:])
        p.stop()
        p = self.proxy(store)                                               # and after a restart
        self.assertEqual(p.json("DELETE", f"/v1/notes/{ids[0]}")[0], 404)
        self.assertEqual(by_id(p.json("GET", "/v1/notes?history=1")[1]["data"]), after)
        self.assertEqual(disk(store), on_disk)

    def test_a_number_is_not_an_id(self):
        """An old client's DELETE /v1/notes/1 is refused and erases nothing: 1 could be any note by now."""
        store = self.tmp.file("one-number.jsonl")
        p = self.proxy(store)
        for t in TEXTS:
            p.json("POST", "/v1/notes", {"text": t})
        before, on_disk = p.json("GET", "/v1/notes")[1]["data"], disk(store)
        for m in ("DELETE", "GET"):
            st, e = p.json(m, "/v1/notes/1")
            self.assertEqual(st, 400, m)
            self.assertIn("GET /v1/notes", e["error"]["message"])
        self.assertEqual(p.json("GET", "/v1/notes")[1]["data"], before)
        self.assertEqual(disk(store), on_disk)

    def test_a_seeded_note_is_erased_once(self):
        store = H.seeded(self.tmp, "one-seeded.jsonl")
        p = self.proxy(store)
        before = by_id(p.json("GET", "/v1/notes?history=1")[1]["data"])
        first = sorted(before, key=lambda i: int(i[1:]))[0]
        self.assertEqual(first, "s1")
        self.assertEqual(p.json("DELETE", f"/v1/notes/{first}")[0], 200)
        after = by_id(p.json("GET", "/v1/notes?history=1")[1]["data"])
        self.assertTrue(set(after) < set(before) and first not in after)
        self.assertEqual(after, {i: before[i] for i in after})
        self.assertEqual(p.json("DELETE", f"/v1/notes/{first}")[0], 404)
        self.assertEqual(by_id(p.json("GET", "/v1/notes?history=1")[1]["data"]), after)

    def test_the_chat_reply_names_notes_by_their_ids_and_exclude_takes_them(self):
        p = self.proxy(self.tmp.file("one-chat.jsonl"))
        ids = [p.json("POST", "/v1/notes", {"text": t})[1]["id"] for t in TEXTS]
        p.json("DELETE", f"/v1/notes/{ids[0]}")
        q = "What does Aurelio repair?"
        st, h, _b = p.call("POST", "/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": q}]})
        r = p.json("POST", "/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": q}],
                                                    "notes_echo": True})[1]["notes"]
        self.assertIn(ids[2], r["ids"])
        self.assertEqual(h["X-Notes-Ids"], ",".join(r["ids"]))
        self.assertEqual(len(r["numbers"]), len(r["ids"]))
        for i, k in zip(r["ids"], r["numbers"]):                            # the number is the note's [k] in the block
            self.assertEqual(p.json("GET", f"/v1/notes/{i}")[1]["number"], k)
            self.assertIn(f"[{k}] ", r["block"])
        r = p.json("POST", "/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": q}],
                                                    "notes_exclude": [ids[2]], "notes_echo": True})[1]["notes"]
        self.assertNotIn(ids[2], r["ids"])
        self.assertNotIn("Aurelio", r["block"])
        self.assertEqual(p.json("POST", "/v1/chat/completions",
                                {"model": "m", "messages": [{"role": "user", "content": q}],
                                 "notes_exclude": [3]})[0], 400)


# ============================================================================================ HTTP, many users
class HttpManyUsers(unittest.TestCase):
    def setUp(self):
        self.tmp = H.TempDir()
        self.addCleanup(self.tmp.cleanup)
        self.team = TU.Team(self.tmp, "team")
        self.A, self.B = TU.auth(self.team.keys["alice"]), TU.auth(self.team.keys["bob"])
        self.stub = H.Stub()
        self.stub.release.set()
        self.addCleanup(self.stub.stop)

    def proxy(self):
        p = H.Running(self.stub.url, self.team.stores, users=self.team.users)
        self.addCleanup(p.stop)
        return p

    def test_a_repeated_erase_is_a_404_for_its_user_and_touches_no_other_note(self):
        p = self.proxy()
        mine = [p.json("POST", "/v1/notes", {"text": t}, headers=self.A)[1]["id"] for t in TEXTS]
        his = [p.json("POST", "/v1/notes", {"text": "Bob says " + t}, headers=self.B)[1]["id"] for t in TEXTS]
        hers_before = by_id(p.json("GET", "/v1/notes?history=1", headers=self.A)[1]["data"])
        bobs = p.json("GET", "/v1/notes?history=1", headers=self.B)[1]
        bob_disk = self.team.bytes_of("bob")
        self.assertEqual(p.json("DELETE", f"/v1/notes/{mine[0]}", headers=self.A)[0], 200)
        alice_disk = self.team.bytes_of("alice")
        self.assertEqual(p.json("DELETE", f"/v1/notes/{mine[0]}", headers=self.A)[0], 404)   # the retry
        listed = p.json("GET", "/v1/notes?history=1", headers=self.A)[1]["data"]
        self.assertEqual([x["text"] for x in listed], TEXTS[1:])
        hers = {i: hers_before[i] for i in mine[1:]}
        self.assertEqual(by_id(listed), hers)
        self.assertEqual(self.team.bytes_of("alice"), alice_disk)
        self.assertEqual(mine, his)                     # the same ids in two stores: each names that user's note
        self.assertEqual(p.json("GET", "/v1/notes?history=1", headers=self.B)[1], bobs)        # bob keeps his w1
        self.assertEqual(p.json("GET", f"/v1/notes/{his[0]}", headers=self.B)[1]["text"], "Bob says " + TEXTS[0])
        self.assertEqual(self.team.bytes_of("bob"), bob_disk)
        p.stop()
        p = self.proxy()                                                    # and after a restart
        self.assertEqual(p.json("DELETE", f"/v1/notes/{mine[0]}", headers=self.A)[0], 404)
        self.assertEqual(by_id(p.json("GET", "/v1/notes?history=1", headers=self.A)[1]["data"]), hers)
        self.assertEqual(self.team.bytes_of("alice"), alice_disk)
        self.assertEqual(p.json("GET", "/v1/notes?history=1", headers=self.B)[1], bobs)
        self.assertEqual(self.team.bytes_of("bob"), bob_disk)


if __name__ == "__main__":
    unittest.main()
