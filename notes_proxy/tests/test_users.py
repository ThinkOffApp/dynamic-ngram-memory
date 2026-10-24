"""Many users on one proxy (--users, --store-dir): each key is one owner, and no owner's notes reach another's block,
journal or scores.

bob's store is L6-seed1's teach store, so his blocks are also checked against retrieve()'s own. alice's store holds
L6-seed2's teach store and, for every L6-seed1 question, a note in that question's own words with a word no other note
has (MARK). If her notes could reach bob's lookup, these are the ones it would find.
"""
import contextlib
import http.client
import io
import json
import os
import random
import secrets
import shutil
import stat
import threading
import unittest
import unittest.mock

from notes_proxy import proxy, users
from notes_proxy.store import Notes, ProxyStore, nm, seed
from notes_proxy.tests import helpers as H

MARK = "Zyxquor"
ENV = "NOTES_PROXY_TEST_UPSTREAM_KEY"
UPSTREAM_KEY = "sk-upstream-test-0123456789abcdef"
ROUTES = [("POST", "/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": "Where do I live?"}]}),
          ("GET", "/v1/notes", None), ("GET", "/v1/notes/1", None), ("POST", "/v1/notes", {"text": "I am a cat."}),
          ("DELETE", "/v1/notes/1", None), ("GET", "/v1/models", None), ("POST", "/v1/completions", {"prompt": "hi"})]


def auth(key):
    return {"Authorization": f"Bearer {key}"}


def ask(q, **kw):
    return dict({"model": "stub-model", "messages": [{"role": "user", "content": q}]}, **kw)


def lessons(path):
    """(header, lessons) of a journal file."""
    with open(path, "rb") as f:
        lines = f.read().splitlines()
    return json.loads(lines[0]), [json.loads(x)["lesson"] for x in lines[1:]]


def cli(*argv):
    """(stdout, stderr) of `python -m notes_proxy ARGV`, run in this process."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        proxy.main(list(argv))
    return out.getvalue(), err.getvalue()


class Team:
    """A users file with alice and bob, and their store folder. bob's store is L6-seed1's teach store."""

    def __init__(self, tmp, name):
        self.dir = tmp.file(name)
        self.stores = os.path.join(self.dir, "stores")
        os.makedirs(self.stores)
        self.users = os.path.join(self.dir, "users.json")
        self.keys = {n: users.add_user(self.users, n) for n in ("alice", "bob")}
        seed(self.journal("bob"), "bob", H.L6, "teach")

    def journal(self, owner):
        return users.journal_path(self.stores, owner)

    def bytes_of(self, owner):
        with open(self.journal(owner), "rb") as f:
            return f.read()


# ============================================================================================ owner names
class Names(unittest.TestCase):
    BAD = ["", ".", "..", "../alice", "../../etc/passwd", "alice/..", "a/b", "/abs", "a\\b", "alice\x00", "a\x00b",
           "a" * 33, "a" * 4096, "ALICE", "Alice", " alice", "alice ", "alice\n", "al ice", "-alice", "_alice",
           "al.ice", "alice.jsonl", "аlice", "alicе", "ａlice", "alice​", "Kate", "ſam",
           "café", "١٢", None, 7, ["alice"]]
    GOOD = ["alice", "bob", "a", "0", "x_y-z", "user1", "a" * 32]

    def setUp(self):
        self.tmp = H.TempDir()
        self.addCleanup(self.tmp.cleanup)

    def test_path_tricks_are_refused_everywhere(self):
        os.mkdir(self.tmp.file("stores"))
        path = self.tmp.file("users.json")
        users.add_user(path, "alice")
        with open(path, "rb") as f:
            before = f.read()
        listing = sorted(os.listdir(self.tmp.path))
        for bad in self.BAD:
            with self.subTest(name=bad):
                with self.assertRaises(users.UsersError):
                    users.check_name(bad)
                with self.assertRaises(users.UsersError):               # at user creation
                    users.add_user(path, bad)
                with self.assertRaises(users.UsersError):               # and where a store's file name is made
                    users.journal_path(self.tmp.file("stores"), bad)
                if isinstance(bad, str):                                # and at load, from a hand-edited file
                    hand = self.tmp.file("hand.json")
                    with open(hand, "w") as f:
                        json.dump({"format": users.USERS_FORMAT, "users": {bad: {"sha256": "0" * 64}}}, f)
                    with self.assertRaises(users.UsersError):
                        users.Users(hand)
                    os.unlink(hand)
        with open(path, "rb") as f:
            self.assertEqual(f.read(), before)
        self.assertEqual(sorted(os.listdir(self.tmp.path)), listing)
        self.assertEqual(os.listdir(self.tmp.file("stores")), [])
        self.assertFalse(os.path.exists(os.path.join(os.path.dirname(self.tmp.path), "alice.jsonl")))

    def test_good_names(self):
        path = self.tmp.file("users.json")
        for good in self.GOOD:
            self.assertEqual(users.check_name(good), good)
            users.add_user(path, good)
            self.assertEqual(os.path.dirname(users.journal_path(self.tmp.path, good)), self.tmp.path)
        self.assertEqual(sorted(users.Users(path).names()), sorted(self.GOOD))


# ============================================================================================ the users file
class UsersFile(unittest.TestCase):
    def setUp(self):
        self.tmp = H.TempDir()
        self.addCleanup(self.tmp.cleanup)

    def test_the_file_never_holds_a_key(self):
        path = self.tmp.file("users.json")
        printed = {}
        for name in ("alice", "bob", "carol", "dave", "erin"):
            out, err = cli("add-user", "--users", path, "--owner", name)
            printed[name] = out.strip()
            self.assertEqual(out, printed[name] + "\n")                 # the key alone, on stdout, once
            self.assertGreaterEqual(len(printed[name]), 43)              # token_urlsafe(32): 32 random bytes
            self.assertNotIn(printed[name], err)
        with open(path, "rb") as f:
            data = f.read()
        for name, key in printed.items():
            self.assertNotIn(key.encode(), data, name)
            for j in range(len(key) - 12):                               # not even a part of it
                self.assertNotIn(key[j:j + 12].encode(), data, name)
            self.assertIn(users.digest(key).encode(), data)              # control: the file holds this user's hash
            self.assertEqual(users.Users(path).who(key), name)
        self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o600)
        self.assertEqual(len(set(printed.values())), 5)

    def test_add_user_twice_is_refused(self):
        path = self.tmp.file("users.json")
        users.add_user(path, "alice")
        with open(path, "rb") as f:
            before = f.read()
        with self.assertRaises(users.UsersError):
            users.add_user(path, "alice")
        with open(path, "rb") as f:
            self.assertEqual(f.read(), before)

    def test_remove_user_must_say_what_happens_to_the_notes(self):
        path = self.tmp.file("users.json")
        users.add_user(path, "alice")
        for argv in (["remove-user", "--users", path, "--owner", "alice"],
                     ["remove-user", "--users", path, "--owner", "alice", "--keep-notes", "--erase-notes"]):
            with self.assertRaises(SystemExit):
                with contextlib.redirect_stderr(io.StringIO()):
                    proxy.main(argv)
        with self.assertRaises(users.UsersError):
            users.remove_user(path, "alice", notes="maybe")
        self.assertEqual(users.Users(path).names(), ["alice"])

    def test_who_is_none_for_anything_else(self):
        path = self.tmp.file("users.json")
        key = users.add_user(path, "alice")
        u = users.Users(path)
        self.assertEqual(u.who(key), "alice")
        for other in (None, "", key[:-1], key + "x", key.upper(), " " + key, secrets.token_urlsafe(32),
                      users.digest(key)):
            self.assertIsNone(u.who(other), other)


# ============================================================================================ through HTTP
class ManyUsers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = H.TempDir()
        cls.S, cls.ref = H.reference()
        cls.Q = [q["text"] for q in cls.S["questions"]]
        cls.stub = H.Stub()
        cls.stub.release.set()
        cls.alice_full = cls.tmp.file("alice-full.jsonl")
        seed(cls.alice_full, "alice", H.L6_OTHER, "teach")
        n = Notes(cls.alice_full, "alice", H.tokenizer())
        try:
            for q in cls.Q:
                n.write_text(f"{MARK} once asked me: {q}")
        finally:
            n.close()

    @classmethod
    def tearDownClass(cls):
        cls.stub.stop()
        cls.tmp.cleanup()

    def setUp(self):
        self.team = Team(self.tmp, self._testMethodName)
        self.keys = self.team.keys

    def proxy(self, extra=(), team=None):
        t = team or self.team
        p = H.Running(self.stub.url, t.stores, users=t.users, extra=extra)
        self.addCleanup(p.stop)
        return p

    def fill_alice(self, team=None):
        shutil.copyfile(self.alice_full, (team or self.team).journal("alice"))

    def blocks(self, p, who, questions):
        """The block the upstream received for each question asked with this user's key ("" when none)."""
        out = []
        for q in questions:
            st, r = p.json("POST", "/v1/chat/completions", ask(q), headers=auth(self.keys[who]))
            self.assertEqual(st, 200)
            m = r["received_messages"]
            out.append(m[0]["content"] if len(m) == 2 else "")
        return out

    def test_the_proxy_refuses_a_bad_name_at_start(self):
        with open(self.team.users) as f:
            data = json.load(f)
        data["users"]["../bob"] = data["users"].pop("bob")
        with open(self.team.users, "w") as f:
            json.dump(data, f)
        with self.assertRaises(users.UsersError):
            self.proxy()
        self.assertFalse(os.path.exists(os.path.join(self.team.dir, "bob.jsonl")))

    # ------------------------------------------------------------------------------------------ 1. isolation
    def test_alices_notes_never_reach_bobs_block(self):
        self.fill_alice()
        p = self.proxy()
        mine = self.blocks(p, "alice", self.Q)
        found = sum(MARK in b for b in mine)
        self.assertGreater(found, 50)                 # control: these questions do find alice's notes, in her store
        bobs = self.blocks(p, "bob", self.Q)
        self.assertEqual([q for q, b in zip(self.Q, bobs) if MARK in b], [])
        self.assertEqual(bobs, [self.ref.retrieve("user", q)["block"] for q in self.Q])   # nobody else's note at all
        listed = p.json("GET", "/v1/notes?history=1", headers=auth(self.keys["bob"]))[1]
        self.assertEqual(listed["owner"], "bob")
        self.assertNotIn(MARK, json.dumps(listed))
        for n in p.json("GET", "/v1/notes", headers=auth(self.keys["alice"]))[1]["data"][-3:]:
            st, one = p.json("GET", f"/v1/notes/{n['id']}", headers=auth(self.keys["bob"]))
            self.assertNotIn(MARK, json.dumps(one))   # an id is the asking owner's own
        print(f"\n[test_users] 56 of 56 of bob's blocks are retrieve()'s and hold none of alice's notes; alice's "
              f"own notes are in {found} of her 56")

    # ------------------------------------------------------------------------------------------ 2. scores
    def lookups(self, p):
        """bob's blocks through HTTP, and his lookup's scores, for every question."""
        blocks = self.blocks(p, "bob", self.Q)
        bob = p.notes.get("bob")
        return blocks, [bob.lookup(q).get("scores") for q in self.Q]

    def test_alices_store_does_not_move_bobs_scores(self):
        p = self.proxy()
        empty = self.lookups(p)
        self.assertFalse(os.path.exists(self.team.journal("alice")))        # alice's store never opened: empty
        p.stop()
        self.fill_alice()
        p = self.proxy()
        st, _ = p.json("POST", "/v1/notes", {"text": f"{MARK} says my home is in Atlantis."},
                       headers=auth(self.keys["alice"]))
        self.assertEqual(st, 201)                                           # her store open, full, and written
        full = self.lookups(p)
        p.stop()
        p = self.proxy()
        self.blocks(p, "alice", self.Q[:1])                                 # after a restart, hers opens first
        again = self.lookups(p)
        self.assertEqual(full[0], empty[0])
        self.assertEqual(again[0], empty[0])
        self.assertEqual(full[1], empty[1])
        self.assertEqual(again[1], empty[1])
        # control: alice's notes in bob's own index would move his blocks (both the matches and the weights)
        hb, bl = lessons(self.team.journal("bob"))
        _ha, al = lessons(self.alice_full)
        merged = ProxyStore(H.tokenizer(), schema=hb.get("schema"))
        for l in bl + [dict(l, id="a" + l["id"]) for l in al if MARK in l["text"]]:
            merged.write("bob", l)
        moved = sum(merged.retrieve("bob", q)["block"] != b for q, b in zip(self.Q, empty[0]))
        print(f"[test_users] bob's 56 blocks and scores are the same with alice's store empty, full, and full after "
              f"a restart; control, her notes in his index: {moved} of 56 blocks move")
        self.assertGreater(moved, 40)

    # ------------------------------------------------------------------------------------------ 3. erase
    def test_erase_for_alice_leaves_bob_alone(self):
        self.fill_alice()
        p = self.proxy()
        A, B = auth(self.keys["alice"]), auth(self.keys["bob"])
        text = "My neighbour Oskarina keeps seven tortoises."
        st, w = p.json("POST", "/v1/notes", {"text": text}, headers=A)
        self.assertEqual(st, 201)
        q = "How many tortoises does Oskarina keep?"
        self.assertIn("Oskarina", self.blocks(p, "alice", [q])[0])          # control: the note is found before
        bob_disk, bob_blocks = self.team.bytes_of("bob"), self.blocks(p, "bob", self.Q)
        bob_list = p.json("GET", "/v1/notes?history=1", headers=B)[1]
        alice_disk = self.team.bytes_of("alice")
        st, e = p.json("DELETE", f"/v1/notes/{w['id']}", headers=A)
        self.assertEqual((st, e["erased"], e["removed"]), (200, w["id"], [w["source"]]))
        self.assertEqual(nm.walk_for(p.notes, texts=["Oskarina", "tortoises"]), [])    # before a question brings them
        self.assertNotEqual(self.team.bytes_of("alice"), alice_disk)         # control: the erase did rewrite hers
        self.assertNotIn(b"Oskarina", self.team.bytes_of("alice"))
        self.assertEqual(self.team.bytes_of("bob"), bob_disk)
        self.assertEqual(self.blocks(p, "bob", self.Q), bob_blocks)
        self.assertEqual(p.json("GET", "/v1/notes?history=1", headers=B)[1], bob_list)
        self.assertNotIn("Oskarina", self.blocks(p, "alice", [q])[0])
        p.stop()
        p = self.proxy()                                                     # and after a restart
        texts = [n["text"] for n in p.json("GET", "/v1/notes?history=1", headers=A)[1]["data"]]
        self.assertNotIn(text, texts)
        self.assertGreater(sum(MARK in t for t in texts), 50)                # her other notes are kept
        self.assertNotIn("Oskarina", self.blocks(p, "alice", [q])[0])
        self.assertEqual(self.blocks(p, "bob", self.Q), bob_blocks)
        self.assertEqual(p.json("GET", "/v1/notes?history=1", headers=B)[1], bob_list)
        self.assertEqual(self.team.bytes_of("bob"), bob_disk)

    def test_write_from_chat_goes_to_the_speaker_only(self):
        p = self.proxy(extra=["--write"])
        bob_disk = self.team.bytes_of("bob")
        st, r = p.json("POST", "/v1/chat/completions", ask("My neighbour Oskarina keeps seven tortoises."),
                       headers=auth(self.keys["alice"]))
        self.assertEqual(r["notes"]["write"]["verdict"], "NEW")
        self.assertEqual([n["text"] for n in p.json("GET", "/v1/notes", headers=auth(self.keys["alice"]))[1]["data"]],
                         ["My neighbour Oskarina keeps seven tortoises."])
        self.assertIn(b"Oskarina", self.team.bytes_of("alice"))
        self.assertEqual(self.team.bytes_of("bob"), bob_disk)
        self.assertNotIn("Oskarina", self.blocks(p, "bob", ["How many tortoises does Oskarina keep?"])[0])
        bob = p.json("GET", "/v1/notes?history=1", headers=auth(self.keys["bob"]))[1]["data"]
        self.assertNotIn("Oskarina", json.dumps(bob))

    # ------------------------------------------------------------------------------------------ 4. keys
    def test_no_key_a_wrong_key_or_a_removed_users_key_is_a_401(self):
        carol = users.add_user(self.team.users, "carol")
        alice = self.keys["alice"]
        wrong = [None, "", "Bearer", "Bearer ", f"Basic {alice}", f"Token {alice}", f"Bearer {alice[:-1]}",
                 f"Bearer {alice}x", f"Bearer {secrets.token_urlsafe(32)}", f"Bearer {users.digest(alice)}"]
        p = self.proxy()
        n = len(self.stub.requests)
        for value in wrong:
            for method, path, body in ROUTES:
                h = {"X-Api-Key": alice} if value is None else {"Authorization": value}
                st, hdr, raw = p.call(method, path, body, headers=h)
                self.assertEqual(st, 401, (value, method, path))
                self.assertEqual(hdr.get("WWW-Authenticate"), "Bearer")
                self.assertNotIn(alice.encode(), raw)
        self.assertEqual(len(self.stub.requests), n)                        # the upstream received nothing
        self.assertEqual(p.notes.opened, {})                                # and no store was opened, nor looked up
        self.assertFalse(os.path.exists(self.team.journal("alice")))
        # a kept-alive connection asks for the key again at every request
        c = http.client.HTTPConnection("127.0.0.1", p.port, timeout=30)
        c.request("GET", "/v1/notes", headers=auth(alice))
        r = c.getresponse()
        self.assertEqual((r.status, json.loads(r.read())["owner"]), (200, "alice"))
        c.request("GET", "/v1/notes")
        self.assertEqual(c.getresponse().status, 401)
        c.close()
        # control: carol's key works, so the 401 after her removal is the removal
        st, _ = p.json("POST", "/v1/notes", {"text": "My aunt Brigitta grows tomatoes."}, headers=auth(carol))
        self.assertEqual(st, 201)
        n = len(self.stub.requests)
        self.assertEqual(p.call("GET", "/v1/models", headers=auth(carol))[0], 200)
        self.assertEqual(len(self.stub.requests), n + 1)                    # control: the stub counts what reaches it
        p.stop()
        out, err = cli("remove-user", "--users", self.team.users, "--owner", "carol", "--keep-notes")
        self.assertIn("kept", err)
        self.assertTrue(os.path.exists(self.team.journal("carol")))
        p = self.proxy()
        n = len(self.stub.requests)
        for method, path, body in ROUTES:
            self.assertEqual(p.call(method, path, body, headers=auth(carol))[0], 401, (method, path))
        self.assertEqual(len(self.stub.requests), n)
        self.assertEqual(p.notes.opened, {})
        self.assertEqual(p.call("GET", "/v1/notes", headers=auth(alice))[0], 200)

    def test_remove_user_with_erase(self):
        p = self.proxy()
        self.assertEqual(p.json("POST", "/v1/notes", {"text": "My aunt Brigitta grows tomatoes."},
                                headers=auth(self.keys["alice"]))[0], 201)
        bob_disk = self.team.bytes_of("bob")
        argv = ["remove-user", "--users", self.team.users, "--owner", "alice", "--erase-notes", "--store-dir",
                self.team.stores]
        with self.assertRaises(SystemExit) as cm:                           # not while a proxy serves the folder
            cli(*argv)
        self.assertIn("stop it first", str(cm.exception))
        self.assertTrue(os.path.exists(self.team.journal("alice")))
        self.assertEqual(sorted(users.Users(self.team.users).names()), ["alice", "bob"])
        p.stop()
        out, err = cli(*argv)
        self.assertIn("erased", err)
        self.assertEqual([f for f in os.listdir(self.team.stores) if f.startswith("alice")], [])
        self.assertEqual(users.Users(self.team.users).names(), ["bob"])
        self.assertEqual(self.team.bytes_of("bob"), bob_disk)
        with self.assertRaises(SystemExit):                                 # erase needs to know where the notes are
            with contextlib.redirect_stderr(io.StringIO()):
                proxy.main(["remove-user", "--users", self.team.users, "--owner", "bob", "--erase-notes"])

    # ------------------------------------------------------------------------------------------ 5. upstream key
    def upstream_requests(self, p, key):
        """Every kind of request alice can send, and what the stub received from them."""
        n = len(self.stub.requests)
        h = dict(auth(key), **{"X-Api-Key": key})
        self.assertEqual(p.call("POST", "/v1/chat/completions", ask("Where do I live?"), headers=h)[0], 200)
        self.assertEqual(p.call("POST", "/v1/chat/completions", ask("Where do I live?", stream=True), headers=h)[0],
                         200)
        self.assertEqual(p.call("POST", "/v1/chat/completions", ask("hi", notes=False), headers=h)[0], 200)
        self.assertEqual(p.call("GET", "/v1/models", headers=h)[0], 200)
        p.call("POST", "/v1/completions", {"prompt": "hi"}, headers=h)
        got = self.stub.requests[n:]
        self.assertEqual(len(got), 5)
        return got

    def test_the_users_key_never_reaches_the_upstream(self):
        key = self.keys["alice"]
        with unittest.mock.patch.dict(os.environ, {ENV: UPSTREAM_KEY}):
            p = self.proxy(extra=["--upstream-key-env", ENV])
        for r in self.upstream_requests(p, key):
            self.assertNotIn(key, json.dumps(r["headers"]))
            self.assertNotIn(key.encode(), r["body"])
            self.assertEqual({k.lower(): v for k, v in r["headers"].items()}.get("authorization"),
                             f"Bearer {UPSTREAM_KEY}")
        self.assertFalse(any(key in line for line in p.logs))
        self.assertTrue(any("[alice]" in line for line in p.logs))
        self.assertEqual(nm.walk_for([p.srv.cfg, p.srv.users, p.notes], texts=[key]), [])
        p.stop()
        p = self.proxy()                                                     # without --upstream-key-env: no key at all
        for r in self.upstream_requests(p, key):
            self.assertNotIn(key, json.dumps(r["headers"]))
            self.assertNotIn("authorization", {k.lower() for k in r["headers"]})

    def test_an_upstream_key_variable_that_is_not_set_stops_the_start(self):
        env = dict(os.environ)
        env.pop(ENV, None)
        for value in ({ENV: ""}, {ENV: "two words"}, {ENV: "line\nbreak"}, None):
            with unittest.mock.patch.dict(os.environ, value or env, clear=value is None):
                with self.assertRaises(SystemExit) as cm:
                    self.proxy(extra=["--upstream-key-env", ENV])
                self.assertIn(ENV, str(cm.exception))

    def test_one_owner_mode_with_an_upstream_key(self):
        with unittest.mock.patch.dict(os.environ, {ENV: UPSTREAM_KEY}):
            p = H.Running(self.stub.url, self.team.journal("bob"), owner="bob", extra=["--upstream-key-env", ENV])
        self.addCleanup(p.stop)
        p.call("GET", "/v1/models", headers={"Authorization": "Bearer sk-client"})
        self.assertEqual(self.stub.requests[-1]["headers"].get("Authorization"), f"Bearer {UPSTREAM_KEY}")

    def test_the_two_modes_do_not_mix(self):
        for argv in (["--owner", "bob", "--store", self.team.journal("bob"), "--users", self.team.users],
                     ["--users", self.team.users], ["--store-dir", self.team.stores], ["--owner", "bob"], []):
            args = proxy.parser().parse_args(["--upstream", self.stub.url, "--tokenizer", "fake"] + argv)
            with self.assertRaises(SystemExit) as cm:
                proxy.build(args)
            self.assertIn("--store-dir", str(cm.exception))

    # ------------------------------------------------------------------------------------------ 7. concurrency
    def test_concurrent_users_get_the_blocks_they_get_one_at_a_time(self):
        self.fill_alice()
        self.keys["carol"] = users.add_user(self.team.users, "carol")
        self.keys["dave"] = users.add_user(self.team.users, "dave")
        seed(self.team.journal("carol"), "carol", H.L6, "del-B")
        seed(self.team.journal("dave"), "dave", H.L6_OTHER, "teach")
        who = ["alice", "bob", "carol", "dave"]
        p = self.proxy()
        alone = {(u, q): b for u in who for q, b in zip(self.Q, self.blocks(p, u, self.Q))}
        p.stop()
        differ = sum(len({alone[(u, q)] for u in who}) == 4 for q in self.Q)
        self.assertGreater(differ, 40)              # control: a request answered from another user's store would show
        jobs = [(u, q) for u in who for q in self.Q] * 2
        random.Random(7).shuffle(jobs)
        for run in range(2):
            p = self.proxy()                         # fresh, so the stores also open while the requests race
            got, errors, start = {}, [], threading.Barrier(16)

            def work(part):
                start.wait()
                for u, q in part:
                    try:
                        b = self.blocks(p, u, [q])[0]
                    except Exception as e:  # noqa: BLE001 (a failure in a thread is reported below)
                        errors.append(repr(e))
                        continue
                    if got.setdefault((u, q), b) != b:
                        errors.append(f"{u}: two blocks for {q}")

            threads = [threading.Thread(target=work, args=(jobs[k::16],)) for k in range(16)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            p.stop()
            self.assertEqual(errors, [])
            self.assertEqual(got, alone)
        print(f"[test_users] {len(jobs)} requests of 4 users on 16 threads, twice: every block as one at a time "
              f"({differ} of 56 questions get 4 different blocks)")


if __name__ == "__main__":
    unittest.main()
