"""Many users: what one user can learn about another through the upstream, the operator's commands when they hold a
mistake, and stores that fail. test_users.py has the isolation of the notes themselves."""
import contextlib
import errno
import http.client
import io
import json
import os
import re
import resource
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import unittest.mock
from http.server import BaseHTTPRequestHandler

from notes_proxy import proxy, store, users
from notes_proxy.store import seed
from notes_proxy.tests import helpers as H
from notes_proxy.tests import test_users as TU

CHAT = "/v1/chat/completions"
SECRET = "My secret island is Zanzibar."
GUESSES = [SECRET, "My secret island is Madeira.", "My secret lake is Zanzibar."]
README = os.path.join(H.REPO, "notes_proxy", "README.md")


def auth(key):
    return {"Authorization": f"Bearer {key}"}


def run_cli(*argv):
    """(exit code, stdout, stderr) of `python -m notes_proxy ARGV`, run in this process."""
    out, err = io.StringIO(), io.StringIO()
    code = 0
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            proxy.main(list(argv))
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else 1
            if not isinstance(e.code, int) and e.code is not None:
                err.write(str(e.code))
    return code, out.getvalue(), err.getvalue()


# ============================================================================================ a stand-in prompt cache
class LlamaStub:
    """llama.cpp's prompt cache, as its server keeps one: a slot holds the last prompt. A request with cache_prompt on
    (llama.cpp's default) reuses the prefix it shares with that prompt, and the reply counts it, in timings.cache_n and
    usage.prompt_tokens_details.cached_tokens (characters here, tokens there). vLLM's cache_salt is modelled too: a
    request reuses nothing of a prompt sent with another salt."""

    def __init__(self):
        self.slot = (None, "")
        self.log = []                    # (path, body, characters reused)
        stub = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path == CHAT:
                    prompt = "".join(f"<{m['role']}>{m['content']}" for m in body["messages"])
                else:
                    prompt = body["prompt"]
                salt = body.get("cache_salt")
                old_salt, old = stub.slot
                n = 0
                if body.get("cache_prompt", True) is not False and salt == old_salt:
                    while n < min(len(old), len(prompt)) and old[n] == prompt[n]:
                        n += 1
                stub.slot = (salt, prompt)
                stub.log.append((self.path, body, n))
                counts = {"usage": {"prompt_tokens": len(prompt), "completion_tokens": 1,
                                    "total_tokens": len(prompt) + 1, "prompt_tokens_details": {"cached_tokens": n}},
                          "timings": {"cache_n": n, "prompt_n": len(prompt) - n}}
                if body.get("stream"):
                    chunks = [{"choices": [{"index": 0, "delta": {"content": "ok"}}]},
                              dict({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}, **counts)]
                    data = b"".join(b"data: " + json.dumps(c).encode() + b"\n\n" for c in chunks) + b"data: [DONE]\n\n"
                    ctype = "text/event-stream"
                else:
                    data = json.dumps(dict({"object": "chat.completion", "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": "ok"}}]}, **counts)).encode()
                    ctype = "application/json"
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.srv = H.Server(("127.0.0.1", 0), Handler)
        self.srv.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def stop(self):
        self.srv.shutdown()
        self.srv.server_close()


# ============================================================================================ through the upstream
class Upstream(unittest.TestCase):
    """What reaches the upstream with many users, and what comes back from it."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = H.TempDir()
        cls.stub = H.Stub()
        cls.stub.release.set()

    @classmethod
    def tearDownClass(cls):
        cls.stub.stop()
        cls.tmp.cleanup()

    def setUp(self):
        self.team = TU.Team(self.tmp, self._testMethodName)
        self.A, self.B = auth(self.team.keys["alice"]), auth(self.team.keys["bob"])

    def proxy(self, url=None, extra=()):
        p = H.Running(url or self.stub.url, self.team.stores, users=self.team.users, extra=extra)
        self.addCleanup(p.stop)
        return p

    # ------------------------------------------------------------------------------------------ the prompt cache
    def probe(self, extra):
        """alice keeps a secret note and asks about it. Then, for each guess, bob sends notes.py's framing plus the
        guess with notes off, right after alice's question. Returns (what the stub reused for each of bob's 9 probes,
        every reply bob got, (who, the body the stub got) for every request)."""
        up = LlamaStub()
        self.addCleanup(up.stop)
        p = self.proxy(up.url, extra)
        sent = []

        def call(who, method, path, body):
            n = len(up.log)
            out = p.call(method, path, body, headers=self.A if who == "alice" else self.B)
            sent.extend((who, b) for _p, b, _n in up.log[n:])
            return out

        st, _ = p.json("POST", "/v1/notes", {"text": SECRET}, headers=self.A)
        self.assertEqual(st, 201)
        st, _h, raw = call("alice", "POST", CHAT, TU.ask("Where is my island?", notes_echo=True))
        block = json.loads(raw)["notes"]["block"]
        frame = block[:block.index(SECRET)]                   # notes.py's framing: the same for every user
        reused, replies = [], []
        for guess in GUESSES:
            msgs = [{"role": "system", "content": frame + guess}, {"role": "user", "content": "hi"}]
            for path, body in ((CHAT, {"model": "m", "notes": False, "messages": msgs}),
                               (CHAT, {"model": "m", "notes": False, "stream": True, "messages": msgs}),
                               ("/v1/completions", {"model": "m", "prompt": "<system>" + frame + guess,
                                                    "cache_prompt": True})):
                call("alice", "POST", CHAT, TU.ask("Where is my island?"))       # her block is in the slot
                st, _h, raw = call("bob", "POST", path, body)
                self.assertEqual(st, 200)
                reused.append(up.log[-1][2])
                replies.append(raw)
        st, _h, raw = call("bob", "POST", CHAT, TU.ask("Where is my island?", stream=True))
        replies.append(raw)                                   # bob's own chat with notes, streamed
        return reused, replies, sent

    def test_one_users_prompt_cache_tells_another_nothing(self):
        reused, replies, sent = self.probe(())
        self.assertEqual(reused, [0] * 9)                     # bob's requests reuse nothing of alice's prompt
        for raw in replies:                                   # and no reply counts reused tokens
            for word in (b"timings", b"cache_n", b"cached_tokens", b"prompt_n"):
                self.assertNotIn(word, raw)
        self.assertIn(b": notes {", replies[-1])               # the scrubbed stream still carries bob's notes line
        salts = {"alice": set(), "bob": set()}
        for who, body in sent:                                # every request goes with the cache off and a salt
            self.assertIs(body.get("cache_prompt"), False)
            self.assertRegex(body.get("cache_salt", ""), r"^[0-9a-f]{64}$")
            salts[who].add(body["cache_salt"])
        self.assertEqual(len(sent), 20)
        self.assertEqual([len(salts["alice"]), len(salts["bob"])], [1, 1])   # one salt per user
        self.assertNotEqual(salts["alice"], salts["bob"])
        # control: with the cache shared, the same stub and the same probe tell the right guess from the wrong ones
        self.team = TU.Team(self.tmp, self._testMethodName + "-shared")
        self.A, self.B = auth(self.team.keys["alice"]), auth(self.team.keys["bob"])
        shared, replies, _sent = self.probe(["--share-upstream-cache"])
        for k in range(3):
            self.assertGreater(shared[k], shared[3 + k])      # the right guess reuses more than "Madeira"
            self.assertGreater(shared[3 + k], shared[6 + k])  # which reuses more than "lake"
        for raw in replies:                                   # the counts are still taken out of the replies
            self.assertNotIn(b"cache_n", raw)
            self.assertNotIn(b"cached_tokens", raw)
        print(f"\n[test_many_safety] bob's 9 probes of alice's cached block reuse {reused}. Control, with "
              f"--share-upstream-cache: {shared}")

    def test_a_client_cannot_turn_the_cache_back_on(self):
        up = LlamaStub()
        self.addCleanup(up.stop)
        p = self.proxy(up.url)
        p.call("POST", CHAT, TU.ask("Where do I live?"), headers=self.A)
        alices = up.log[-1][1]["cache_salt"]
        p.call("POST", CHAT, TU.ask("Where do I live?", cache_prompt=True, cache_salt=alices), headers=self.B)
        body = up.log[-1][1]
        self.assertIs(body["cache_prompt"], False)
        self.assertNotEqual(body["cache_salt"], alices)
        self.assertEqual(up.log[-1][2], 0)

    # ------------------------------------------------------------------------------------------ routes passed on
    def test_only_the_model_routes_are_passed_on(self):
        p = self.proxy()
        n = len(self.stub.requests)
        for method, path in (("GET", "/slots"), ("GET", "/metrics"), ("GET", "/props"), ("POST", "/props"),
                             ("POST", "/slots/0?action=save"), ("POST", "/slots/0?action=erase"), ("GET", "/health"),
                             ("GET", CHAT), ("POST", "/completion"), ("POST", "/v1/responses"),
                             ("POST", "/v1/messages"), ("GET", "/v1/models/../../slots"), ("GET", "//slots"),
                             ("GET", "/v1/models/stub-model"), ("POST", "/v1/models"), ("DELETE", "/v1/completions")):
            st, _h, raw = p.call(method, path, {} if method == "POST" else None, headers=self.B)
            self.assertEqual(st, 404, (method, path))
            self.assertIn(b"/v1/models", raw)                  # the reply names the routes that do pass
        self.assertEqual(len(self.stub.requests), n)          # none of them reached the upstream
        # control: the routes on the list do reach it
        self.assertEqual(p.call("GET", "/v1/models", headers=self.B)[0], 200)
        p.call("POST", "/v1/completions", {"model": "m", "prompt": "hi"}, headers=self.B)
        p.call("POST", "/v1/embeddings", {"model": "m", "input": "hi"}, headers=self.B)
        self.assertEqual([r["path"] for r in self.stub.requests[n:]], ["/v1/models", "/v1/completions",
                                                                       "/v1/embeddings"])

    # ------------------------------------------------------------------------------------------ the key elsewhere
    def test_a_request_that_carries_the_key_outside_its_header_is_refused(self):
        key = self.team.keys["alice"]
        p = self.proxy()
        p.json("POST", "/v1/notes", {"text": "My aunt Brigitta grows tomatoes."}, headers=self.A)
        disk = self.team.bytes_of("alice")
        n = len(self.stub.requests)
        coded = "".join(f"%{ord(c):02X}" for c in key)
        for method, path, body in (("GET", f"/v1/models?api_key={key}", None), ("GET", f"/v1/models?key={coded}", None),
                                   ("GET", f"/v1/notes/{key}", None), ("GET", f"/v1/{coded}", None),
                                   ("POST", CHAT, TU.ask("Where do I live?", user=key)),
                                   ("POST", CHAT, TU.ask(f"My key is {key}.", notes=False)),
                                   ("POST", "/v1/completions", {"model": "m", "prompt": key}),
                                   ("POST", "/v1/notes", {"text": f"My key is {key}."}),
                                   ("GET", f"/v1/notes?{key}=1", None)):
            st, _h, raw = p.call(method, path, body, headers=self.A)
            self.assertEqual(st, 400, (method, path))
            self.assertNotIn(key.encode(), raw)
        self.assertEqual(len(self.stub.requests), n)          # the upstream got none of them
        self.assertEqual(self.team.bytes_of("alice"), disk)   # and no note holds the key
        self.assertFalse(any(key in line or coded in line for line in p.logs), p.logs)
        # control: the same requests without the key go through
        p.call("GET", "/v1/models?api_key=none", headers=self.A)
        self.assertEqual(self.stub.requests[-1]["path"], "/v1/models?api_key=none")
        self.assertEqual(p.call("POST", CHAT, TU.ask("Where do I live?", user="alice"), headers=self.A)[0], 200)
        self.assertEqual(len(self.stub.requests), n + 2)

    def test_a_key_hidden_by_json_escapes_is_refused(self):
        key = self.team.keys["alice"]
        p = self.proxy()
        p.json("POST", "/v1/notes", {"text": "My aunt Brigitta grows tomatoes."}, headers=self.A)
        disk = self.team.bytes_of("alice")
        n = len(self.stub.requests)
        every = "".join(f"\\u{ord(c):04x}" for c in key)       # each character as \uXXXX
        i = next(j for j, c in enumerate(key) if not c.isdigit())   # an escaped digit ends in that digit again
        first = key[:i] + f"\\u{ord(key[i]):04x}" + key[i + 1:]       # one character is enough to hide it from the bytes
        for coded in (every, first):
            self.assertNotIn(key.encode(), coded.encode())
            for path, raw in ((CHAT, '{"model": "m", "messages": [{"role": "user", "content": "My key is %s."}]}'),
                              (CHAT, '{"model": "m", "messages": [{"role": "user", "content": "hi"}], "x": ["%s"]}'),
                              ("/v1/completions", '{"model": "m", "prompt": "%s"}'),
                              ("/v1/embeddings", '{"model": "m", "input": ["%s"]}'),
                              ("/v1/notes", '{"text": "My key is %s."}')):
                st, _h, out = p.call("POST", path, raw=(raw % coded).encode(), headers=self.A)
                self.assertEqual(st, 400, (path, raw))
                self.assertNotIn(key.encode(), out)
        self.assertEqual(len(self.stub.requests), n)          # the upstream got none of them
        self.assertEqual(self.team.bytes_of("alice"), disk)   # and no note holds the key
        # control: the same escaped text without the key goes through
        plain = "".join(f"\\u{ord(c):04x}" for c in "tomatoes")
        raw = '{"model": "m", "messages": [{"role": "user", "content": "I like %s."}]}' % plain
        self.assertEqual(p.call("POST", CHAT, raw=raw.encode(), headers=self.A)[0], 200)
        self.assertEqual(len(self.stub.requests), n + 1)

    # ------------------------------------------------------------------------------------------ the times
    def test_replies_carry_no_lookup_times(self):
        p = self.proxy(extra=["--write"])
        st, r = p.json("POST", CHAT, TU.ask("My neighbour Oskarina keeps seven tortoises."), headers=self.A)
        self.assertEqual(r["notes"]["write"]["verdict"], "NEW")   # a write ran, so write_us would have a value
        st, _h, raw = p.call("POST", CHAT, TU.ask("How many tortoises does Oskarina keep?", stream=True),
                             headers=self.A)
        line = next(x for x in raw.split(b"\n") if x.startswith(b": notes "))
        for used in (r["notes"], json.loads(line[len(b": notes "):])):
            for k in ("lookup_us", "total_us", "write_us"):
                self.assertNotIn(k, used)
            self.assertIn("ids", used)
        self.assertTrue(any(" us; write " in line for line in p.logs))   # the operator's log keeps the time
        # control: one owner's replies keep them
        p.stop()
        one = H.Running(self.stub.url, self.team.journal("bob"), owner="bob")
        self.addCleanup(one.stop)
        st, r = one.json("POST", CHAT, TU.ask("Where do I live?"))
        self.assertIn("lookup_us", r["notes"])


# ============================================================================================ the operator's commands
class Commands(unittest.TestCase):
    def setUp(self):
        self.tmp = H.TempDir()
        self.addCleanup(self.tmp.cleanup)
        self.team = TU.Team(self.tmp, "team")

    def users_bytes(self):
        with open(self.team.users, "rb") as f:
            return f.read()

    def give_alice_notes(self):
        n = store.Notes(self.team.journal("alice"), "alice", H.tokenizer())
        try:
            n.write_text("I live in Oulu.")
        finally:
            n.close()

    def test_erase_with_the_wrong_folder_erases_nothing_and_keeps_the_user(self):
        self.give_alice_notes()
        other = self.tmp.file("other")
        os.mkdir(other)
        before = self.users_bytes()
        code, out, err = run_cli("remove-user", "--users", self.team.users, "--owner", "alice", "--erase-notes",
                                 "--store-dir", other)
        self.assertEqual(code, 1, err)
        self.assertIn("nothing was erased", err)
        self.assertIn(other, err)
        self.assertEqual(self.users_bytes(), before)          # alice is still a user
        self.assertEqual(os.listdir(other), [])               # and the wrong folder got no lock file
        self.assertTrue(os.path.exists(self.team.journal("alice")))
        # with the right folder the same command erases them
        code, out, err = run_cli("remove-user", "--users", self.team.users, "--owner", "alice", "--erase-notes",
                                 "--store-dir", self.team.stores)
        self.assertEqual(code, 0, err)
        self.assertFalse(os.path.exists(self.team.journal("alice")))
        self.assertEqual(users.Users(self.team.users).names(), ["bob"])
        self.assertNotIn("Restart", err)                      # no proxy serves the folder during an erase

    def test_a_new_user_never_gets_an_old_users_notes_by_accident(self):
        self.give_alice_notes()
        code, _out, err = run_cli("remove-user", "--users", self.team.users, "--owner", "alice", "--keep-notes",
                                  "--store-dir", self.team.stores)
        self.assertEqual(code, 0, err)
        self.assertIn(self.team.journal("alice"), err)        # the message says where the kept notes are
        before = self.users_bytes()
        code, out, err = run_cli("add-user", "--users", self.team.users, "--owner", "alice", "--store-dir",
                                 self.team.stores)
        self.assertEqual((code, out), (1, ""))
        self.assertIn("--reuse-notes", err)
        self.assertEqual(self.users_bytes(), before)
        code, out, err = run_cli("add-user", "--users", self.team.users, "--owner", "alice")
        self.assertEqual((code, out), (2, ""))                # add-user must be told where the notes are
        code, out, err = run_cli("add-user", "--users", self.team.users, "--owner", "alice", "--store-dir",
                                 self.tmp.file("nowhere"))
        self.assertEqual((code, out), (1, ""))                # and the folder must be there
        self.assertEqual(self.users_bytes(), before)
        # the same person, on purpose: the new key reads the kept notes
        code, out, err = run_cli("add-user", "--users", self.team.users, "--owner", "alice", "--store-dir",
                                 self.team.stores, "--reuse-notes")
        self.assertEqual(code, 0, err)
        stub = H.Stub()
        self.addCleanup(stub.stop)
        p = H.Running(stub.url, self.team.stores, users=self.team.users)
        self.addCleanup(p.stop)
        st, v = p.json("GET", "/v1/notes", headers=auth(out.strip()))
        self.assertEqual((st, [n["text"] for n in v["data"]]), (200, ["I live in Oulu."]))
        # control: a name with no notes left needs no flag
        code, out, err = run_cli("add-user", "--users", self.team.users, "--owner", "carol", "--store-dir",
                                 self.team.stores)
        self.assertEqual(code, 0, err)

    def test_remove_user_says_only_what_it_knows(self):
        code, _out, err = run_cli("remove-user", "--users", self.team.users, "--owner", "alice", "--keep-notes")
        self.assertEqual(code, 0, err)
        self.assertIn("if any", err)                          # alice never had a journal, and no folder was named
        code, _out, err = run_cli("remove-user", "--users", self.team.users, "--owner", "bob", "--keep-notes",
                                  "--store-dir", self.team.stores)
        self.assertEqual(code, 0, err)
        self.assertIn(self.team.journal("bob"), err)

    def test_a_users_file_others_can_change_stops_the_start(self):
        stub = H.Stub()
        self.addCleanup(stub.stop)
        for mode in (0o666, 0o620, 0o602):
            os.chmod(self.team.users, mode)
            with self.assertRaises(users.UsersError) as cm:
                H.Running(stub.url, self.team.stores, users=self.team.users)
            self.assertIn("chmod", str(cm.exception))
        os.chmod(self.team.users, 0o600)
        with unittest.mock.patch("os.getuid", return_value=os.getuid() + 1):
            with self.assertRaises(users.UsersError) as cm:
                H.Running(stub.url, self.team.stores, users=self.team.users)
            self.assertIn("another account", str(cm.exception))
        for mode in (0o600, 0o640, 0o400):                    # control: private files start
            os.chmod(self.team.users, mode)
            H.Running(stub.url, self.team.stores, users=self.team.users).stop()

    def test_a_store_folder_that_is_a_file_is_a_clear_refusal(self):
        f = self.tmp.file("afile")
        open(f, "w").close()
        for d in (f, os.path.join(f, "sub")):
            args = proxy.parser().parse_args(["--upstream", "http://127.0.0.1:9", "--users", self.team.users,
                                              "--store-dir", d, "--tokenizer", "fake", "--port", "0"])
            with self.assertRaises(users.UsersError) as cm:
                proxy.build(args)
            self.assertIn("store folder", str(cm.exception))
        code, _out, err = run_cli("--upstream", "http://127.0.0.1:9", "--users", self.team.users, "--store-dir", f,
                                  "--tokenizer", "fake", "--port", "0")
        self.assertEqual(code, 1)
        self.assertTrue(err.startswith("notes-proxy: "), err)
        self.assertNotIn("Traceback", err)

    def test_mode_mistakes_are_usage_errors_as_on_main(self):
        base = ["--upstream", "http://127.0.0.1:9", "--tokenizer", "fake", "--port", "0"]
        for argv, want in ((["--store", "a.jsonl"], "the following arguments are required: --owner"),
                           (["--owner", "me"], "the following arguments are required: --store"),
                           ([], "the following arguments are required: --owner, --store"),
                           (["--users", self.team.users], "--store-dir"),
                           (["--owner", "me", "--store", "a.jsonl", "--share-upstream-cache"], "--users")):
            code, out, err = run_cli(*(base + argv))
            self.assertEqual(code, 2, (argv, err))
            self.assertTrue(err.startswith("usage: python -m notes_proxy"), err)
            self.assertIn("python -m notes_proxy: error: ", err)
            self.assertIn(want, err)

    def test_the_remote_warning_names_the_upstream_key(self):
        args = proxy.parser().parse_args(["--upstream", "http://127.0.0.1:9", "--owner", "bob", "--store",
                                          self.team.journal("bob"), "--tokenizer", "fake", "--host", "0.0.0.0"])
        with self.assertRaises(SystemExit) as cm:
            proxy.build(args)
        self.assertEqual(str(cm.exception), "--host 0.0.0.0 is not a loopback address: anyone who can reach it could "
                                            "read and erase this owner's notes. Pass --allow-remote to do it anyway.")
        args.upstream_key_env = TU.ENV
        with unittest.mock.patch.dict(os.environ, {TU.ENV: TU.UPSTREAM_KEY}):
            with self.assertRaises(SystemExit) as cm:
                proxy.build(args)
        self.assertIn(f"the upstream key in ${TU.ENV}", str(cm.exception))


class Readme(unittest.TestCase):
    def test_the_many_users_commands_work_in_order_on_a_fresh_machine(self):
        with open(README) as f:
            section = f.read().split("\n## Many users\n", 1)[1].split("\n## ", 1)[0]
        blocks = re.findall(r"```sh\n(.*?)\n```", section, re.S)
        home = tempfile.mkdtemp(prefix="notes-proxy-home-")
        self.addCleanup(shutil.rmtree, home, True)
        env = dict(os.environ, HOME=home, PYTHONPATH=H.REPO)

        def sh(block):
            r = subprocess.run(["/bin/sh", "-c", block.replace("python3 -m ", f"{sys.executable} -m ")], cwd=H.REPO,
                               env=env, capture_output=True, text=True, timeout=60)
            self.assertEqual(r.returncode, 0, (block, r.stderr))
            return r.stdout

        first = [b for b in blocks if "--tokenizer" not in b and not b.startswith("curl")]
        adds = [b for b in first if " add-user " in b]
        setup = first[:first.index(adds[0]) + 1]               # everything up to the first add-user, in order
        key = sh("set -e\n" + "\n".join(setup)).strip().splitlines()[-1]
        self.assertGreaterEqual(len(key), 43)
        keep = next(b for b in first if "--keep-notes" in b)
        sh(keep)
        erase = next(b for b in first if "--erase-notes" in b)
        notes = os.path.join(home, "notes")
        users.add_user(os.path.join(notes, "users.json"), "alice")
        seed(os.path.join(notes, "alice.jsonl"), "alice", H.L6, "teach")
        sh(erase)
        self.assertFalse(os.path.exists(os.path.join(notes, "alice.jsonl")))


# ============================================================================================ stores that fail
class Stores(unittest.TestCase):
    def setUp(self):
        self.tmp = H.TempDir()
        self.addCleanup(self.tmp.cleanup)
        self.team = TU.Team(self.tmp, "team")
        self.A, self.B = auth(self.team.keys["alice"]), auth(self.team.keys["bob"])
        self.stub = H.Stub()
        self.stub.release.set()
        self.addCleanup(self.stub.stop)

    def proxy(self, extra=()):
        p = H.Running(self.stub.url, self.team.stores, users=self.team.users, extra=extra)
        self.addCleanup(p.stop)
        return p

    def test_routes_without_notes_open_no_store(self):
        p = self.proxy()
        self.assertEqual(p.call("GET", "/v1/models", headers=self.A)[0], 200)
        self.assertEqual(p.call("POST", CHAT, TU.ask("hi", notes=False), headers=self.A)[0], 200)
        self.assertEqual(p.notes.opened, {})
        self.assertFalse(os.path.exists(self.team.journal("alice")))
        self.assertEqual(p.call("GET", "/v1/notes", headers=self.A)[0], 200)   # control: the notes route opens it
        self.assertEqual(sorted(p.notes.opened), ["alice"])

    @unittest.skipIf(os.geteuid() == 0, "root reads a file of mode 000")
    def test_a_store_that_cannot_be_opened_is_a_clear_reply_for_its_owner_only(self):
        p = self.proxy()
        self.assertEqual(p.json("POST", "/v1/notes", {"text": "My dog is Rex."}, headers=self.A)[0], 201)
        p.stop()
        os.chmod(self.team.journal("alice"), 0)
        self.addCleanup(os.chmod, self.team.journal("alice"), 0o600)
        p = self.proxy()
        for method, path, body in (("GET", "/v1/notes", None), ("POST", CHAT, TU.ask("What is my dog called?")),
                                   ("POST", "/v1/notes", {"text": "My cat is Mau."})):
            st, h, raw = p.call(method, path, body, headers=self.A)
            self.assertEqual(st, 503, (path, raw))
            err = json.loads(raw)["error"]
            self.assertIn("operator", err["message"])
            self.assertNotIn(self.tmp.path, err["message"])   # the server's folders are the operator's business
        self.assertTrue(any(line.startswith("notes-proxy: [alice] GET /v1/notes 503") and "Permission denied" in line
                            for line in p.logs), p.logs)
        self.assertEqual(p.call("GET", "/v1/models", headers=self.A)[0], 200)
        self.assertEqual(p.json("GET", "/v1/notes", headers=self.B)[0], 200)

    def test_a_store_of_another_owner_is_reported_to_the_operator(self):
        os.rename(self.team.journal("bob"), self.tmp.file("bob-away.jsonl"))
        seed(self.team.journal("bob"), "someone-else", H.L6, "teach")
        p = self.proxy()
        st, h, raw = p.call("GET", "/v1/notes", headers=self.B)
        self.assertEqual(st, 500)
        msg = json.loads(raw)["error"]["message"]
        self.assertNotIn(self.tmp.path, msg)
        self.assertNotIn("one owner per proxy", msg)
        line = [x for x in p.logs if " /v1/notes " in x][-1]
        self.assertTrue(line.startswith("notes-proxy: [bob] GET /v1/notes 500"), line)
        self.assertIn("another owner", line)
        self.assertEqual(p.json("GET", "/v1/notes", headers=self.A)[0], 200)

    def test_a_journal_that_failed_mid_erase_is_opened_anew_for_its_owner_alone(self):
        p = self.proxy()
        st, w = p.json("POST", "/v1/notes", {"text": "My neighbour Oskarina keeps seven tortoises."}, headers=self.A)
        self.assertEqual(st, 201)
        self.assertEqual(p.json("POST", "/v1/notes", {"text": "My aunt Brigitta grows tomatoes."},
                                headers=self.A)[0], 201)
        mine = self.team.journal("alice")
        real = store._sync_dir

        def flaky(path):
            if path == mine:
                raise OSError(errno.EIO, "Input/output error")
            return real(path)

        with unittest.mock.patch.object(store, "_sync_dir", flaky):
            st, e = p.json("DELETE", f"/v1/notes/{w['id']}", headers=self.A)
        self.assertEqual(st, 500)
        self.assertIn("disk state is unknown", e["error"]["message"])
        self.assertTrue(p.notes.opened["alice"].journal.poisoned)          # control: her journal took no writes now
        bob_disk = self.team.bytes_of("bob")
        st, w2 = p.json("POST", "/v1/notes", {"text": "My cousin Ilmari plays the tuba."}, headers=self.A)
        self.assertEqual(st, 201, w2)
        texts = [n["text"] for n in p.json("GET", "/v1/notes", headers=self.A)[1]["data"]]
        self.assertEqual(texts, ["My aunt Brigitta grows tomatoes.", "My cousin Ilmari plays the tuba."])
        self.assertNotIn(b"Oskarina", self.team.bytes_of("alice"))
        self.assertIn(b"Ilmari", self.team.bytes_of("alice"))
        self.assertEqual(self.team.bytes_of("bob"), bob_disk)
        self.assertEqual(p.json("POST", "/v1/notes", {"text": "My uncle Pentti fishes."}, headers=self.B)[0], 201)
        p.stop()
        p = self.proxy()                                                   # a restart reads the same notes
        self.assertEqual([n["text"] for n in p.json("GET", "/v1/notes", headers=self.A)[1]["data"]], texts)


class OpenFiles(unittest.TestCase):
    """The real proxy process, with many users and a low limit on open files."""
    N = 60

    def setUp(self):
        self.tmp = H.TempDir()
        self.addCleanup(self.tmp.cleanup)
        self.stores = self.tmp.file("stores")
        os.mkdir(self.stores)
        self.users = self.tmp.file("users.json")
        self.keys = {f"u{i:03d}": secrets.token_urlsafe(32) for i in range(self.N)}
        data = {"format": users.USERS_FORMAT,
                "users": {n: {"sha256": users.digest(k), "added": "2026-10-24"} for n, k in self.keys.items()}}
        fd = os.open(self.users, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump(data, f)

    def start(self, soft, hard):
        """(the process, its port or None, its stderr lines) for a proxy started under these limits."""
        def limits():
            resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))

        proc = subprocess.Popen([sys.executable, "-m", "notes_proxy", "--upstream", "http://127.0.0.1:9", "--users",
                                 self.users, "--store-dir", self.stores, "--tokenizer", "fake", "--port", "0"],
                                cwd=H.REPO, env=dict(os.environ, PYTHONPATH=H.REPO), stderr=subprocess.PIPE,
                                stdout=subprocess.DEVNULL, text=True, preexec_fn=limits)

        def end():
            if proc.poll() is None:
                proc.kill()
            proc.wait()
            reader.join(5)
            proc.stderr.close()

        self.addCleanup(end)
        lines = []
        reader = threading.Thread(target=lambda: lines.extend(iter(proc.stderr.readline, "")), daemon=True)
        reader.start()
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            m = next((re.search(r"listening on http://127\.0\.0\.1:(\d+)", x) for x in lines if "listening" in x), None)
            if m:
                return proc, int(m.group(1)), lines
            if proc.poll() is not None:
                reader.join(5)
                return proc, None, lines
            time.sleep(0.05)
        self.fail(f"the proxy neither started nor stopped: {lines}")

    def test_every_user_is_served_under_a_low_soft_limit(self):
        hard = resource.getrlimit(resource.RLIMIT_NOFILE)[1]
        proc, port, lines = self.start(64, hard)
        self.assertIsNotNone(port, lines)
        got = {}
        for name, key in self.keys.items():
            c = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
            try:
                c.request("GET", "/v1/notes", headers=auth(key))
                got[name] = c.getresponse().status
            except (OSError, http.client.HTTPException) as e:
                got[name] = type(e).__name__
            finally:
                c.close()
        self.assertEqual(got, {n: 200 for n in self.keys}, "".join(lines[-5:]))
        self.assertEqual(len([f for f in os.listdir(self.stores) if f.endswith(".jsonl")]), self.N)

    def test_a_limit_too_low_for_every_user_stops_the_start(self):
        proc, port, lines = self.start(128, 128)
        self.assertIsNone(port, "the proxy started with room for only some of its users")
        self.assertEqual(proc.wait(10), 1)
        self.assertIn("open files", "".join(lines))
        self.assertTrue("".join(lines).startswith("notes-proxy: "), lines)


if __name__ == "__main__":
    unittest.main()
