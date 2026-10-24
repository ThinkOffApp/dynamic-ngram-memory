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


if __name__ == "__main__":
    unittest.main()
