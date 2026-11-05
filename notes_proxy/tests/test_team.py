"""A team's shared store: one store behind many keys, a lookup that needs no chat, read-only keys, an agent's proxy that
adds the team's notes to its own, and a block that leaves the prompt cache alone."""
import json
import os
import stat
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from notes_proxy import proxy, team, users
from notes_proxy.tests import helpers as H

FACT = "The Bosgame serves Flash Next to hermes on port 8080."
OTHER = "The router is at 192.168.50.1 on the home network."
QUESTION = "Which port does the Bosgame serve Flash Next on?"


def auth(key):
    return {"Authorization": f"Bearer {key}"}


def ask(q, **kw):
    return dict({"model": "stub-model", "messages": [{"role": "user", "content": q}]}, **kw)


def sent(stub):
    """The messages the upstream got in its last chat request."""
    return stub.last_body()["messages"]


class TeamCase(unittest.TestCase):
    def setUp(self):
        self.tmp = H.TempDir()
        self.addCleanup(self.tmp.cleanup)
        self.stub = H.Stub()
        self.addCleanup(self.stub.stop)
        self.stores = self.tmp.file("stores")
        os.mkdir(self.stores, 0o700)
        self.users = self.tmp.file("users.json")
        self.keys = {
            "hermes": users.add_user(self.users, "hermes", store_dir=self.stores, shares="team"),
            "deep": users.add_user(self.users, "deep", store_dir=self.stores, shares="team"),
            "cloud": users.add_user(self.users, "cloud", store_dir=self.stores, shares="team", role="read"),
        }
        self.p = None

    def serve(self):
        self.p = H.Running(self.stub.url, self.stores, users=self.users)
        self.addCleanup(self.p.stop)
        return self.p

    def write(self, who, text, **kw):
        return self.p.json("POST", "/v1/notes", dict({"text": text, "subject": "fleet"}, **kw), auth(self.keys[who]))

    def journal(self):
        with open(os.path.join(self.stores, "team.jsonl"), "rb") as f:
            return [json.loads(x) for x in f.read().splitlines()]


class SharedStore(TeamCase):
    def test_two_keys_one_store_and_who_wrote_each_note(self):
        self.serve()
        self.assertEqual(self.write("hermes", FACT)[0], 201)
        self.assertEqual(self.write("deep", OTHER, cite="room:thinkoff-development/abc")[0], 201)
        for who in ("hermes", "deep", "cloud"):
            st, v = self.p.json("GET", "/v1/notes", headers=auth(self.keys[who]))
            self.assertEqual(st, 200)
            self.assertEqual(v["owner"], "team")
            self.assertEqual([(n["text"], n["by"]) for n in v["data"]], [(FACT, "hermes"), (OTHER, "deep")])
        self.assertEqual(v["data"][1]["cite"], "room:thinkoff-development/abc")
        self.assertIsNone(v["data"][0]["cite"])
        self.assertFalse(os.path.exists(os.path.join(self.stores, "hermes.jsonl")))
        print(f"[test_team] 3 keys, 1 store: 2 notes, each naming the key that wrote it")

    def test_credits_survive_a_restart_and_leave_with_an_erase(self):
        self.serve()
        _, w = self.write("hermes", FACT, cite="room:x/1")
        self.write("deep", OTHER)
        self.p.stop()
        self.serve()
        _, v = self.p.json("GET", "/v1/notes", headers=auth(self.keys["deep"]))
        self.assertEqual([n["by"] for n in v["data"]], ["hermes", "deep"])
        self.assertEqual(self.p.json("DELETE", f"/v1/notes/{w['id']}", headers=auth(self.keys["deep"]))[0], 200)
        raw = open(os.path.join(self.stores, "team.jsonl"), "rb").read()
        self.assertNotIn(b"room:x/1", raw)
        self.assertNotIn(b"hermes", raw)
        self.assertIn(b'"by":"deep"', raw.replace(b": ", b":"))

    def test_a_bad_cite_is_refused_and_writes_nothing(self):
        self.serve()
        for bad in ("", "x" * 201, "a\nb", 7):
            st, _ = self.p.json("POST", "/v1/notes", {"text": FACT, "subject": "fleet", "cite": bad},
                                auth(self.keys["hermes"]))
            self.assertEqual(st, 400, bad)
        self.assertEqual(self.p.json("GET", "/v1/notes", headers=auth(self.keys["hermes"]))[1]["data"], [])

    def test_sharing_notes_shares_no_identity_or_cache_upstream(self):
        self.serve()
        self.write("hermes", FACT)
        seen = {}
        for who in ("hermes", "deep"):
            st, _ = self.p.json("POST", "/v1/chat/completions", ask(QUESTION, user="x", safety_identifier="y"),
                                auth(self.keys[who]))
            self.assertEqual(st, 200)
            seen[who] = self.stub.last_body()
        for field in ("user", "safety_identifier", "cache_salt"):
            self.assertTrue(seen["hermes"].get(field), field)
            self.assertNotEqual(seen["hermes"][field], seen["deep"][field], field)
        self.assertIs(seen["hermes"]["cache_prompt"], False)
        self.assertIn(FACT, seen["deep"]["messages"][0]["content"])

    def test_a_private_store_keeps_its_old_shape(self):
        key = users.add_user(self.users, "solo", store_dir=self.stores)
        data = json.load(open(self.users))["users"]["solo"]
        self.assertEqual(set(data), {"sha256", "added"})
        self.serve()
        st, w = self.p.json("POST", "/v1/notes", {"text": FACT, "subject": "fleet"}, auth(key))
        self.assertEqual(st, 201)
        self.assertEqual(self.p.json("GET", "/v1/notes", headers=auth(key))[1]["owner"], "solo")
        self.assertEqual(self.p.json("GET", "/v1/notes", headers=auth(self.keys["deep"]))[1]["data"], [])

    def test_a_member_cannot_erase_the_team_by_leaving(self):
        self.serve()
        self.write("hermes", FACT)
        self.p.stop()
        with self.assertRaises(users.UsersError):
            users.remove_user(self.users, "hermes", "erase", store_dir=self.stores)
        self.assertTrue(os.path.getsize(os.path.join(self.stores, "team.jsonl")))
        users.remove_user(self.users, "hermes", "keep")

    def test_a_users_file_with_a_bad_role_or_store_is_refused(self):
        for field, bad in (("role", "admin"), ("store", "../etc")):
            data = json.load(open(self.users))
            keep = dict(data["users"]["cloud"])
            data["users"]["cloud"][field] = bad
            with open(self.users, "w") as f:
                json.dump(data, f)
            with self.assertRaises(users.UsersError):
                users.Users(self.users)
            data["users"]["cloud"] = keep
            with open(self.users, "w") as f:
                json.dump(data, f)


class Lookup(TeamCase):
    def test_it_is_the_chat_routes_own_lookup_and_never_reaches_the_upstream(self):
        self.serve()
        self.write("hermes", FACT)
        self.write("deep", OTHER)
        before = len(self.stub.requests)
        st, v = self.p.json("POST", "/v1/notes/lookup", {"question": QUESTION}, auth(self.keys["cloud"]))
        self.assertEqual(st, 200)
        self.assertEqual(len(self.stub.requests), before)
        self.assertIn(FACT, v["block"])
        self.assertEqual(v["notes_in_store"], 2)
        self.assertEqual(v["notes"][0]["by"], "hermes")
        st, c = self.p.json("POST", "/v1/chat/completions", ask(QUESTION, notes_echo=True), auth(self.keys["deep"]))
        self.assertEqual(st, 200)
        self.assertEqual(c["notes"]["block"], v["block"])
        self.assertEqual(c["notes"]["ids"], v["ids"])

    def test_a_small_cap_chooses(self):
        self.serve()
        self.write("hermes", FACT)
        self.write("deep", OTHER)
        _, full = self.p.json("POST", "/v1/notes/lookup", {"question": QUESTION}, auth(self.keys["cloud"]))
        _, cut = self.p.json("POST", "/v1/notes/lookup", {"question": QUESTION, "cap": full["tokens"] - 1},
                             auth(self.keys["cloud"]))
        self.assertLess(len(cut["ids"]), len(full["ids"]))
        self.assertEqual(cut["cap"], full["tokens"] - 1)

    def test_an_empty_store_answers_with_nothing(self):
        self.serve()
        st, v = self.p.json("POST", "/v1/notes/lookup", {"question": QUESTION}, auth(self.keys["cloud"]))
        self.assertEqual(st, 200)
        self.assertEqual((v["block"], v["ids"], v["notes"]), ("", [], []))

    def test_bad_requests(self):
        self.serve()
        k = auth(self.keys["cloud"])
        for body in ({}, {"question": ""}, {"question": 3}, {"question": "x" * 5000}, {"question": "q", "cap": 0},
                     {"question": "q", "cap": True}, {"question": "q", "past": "no"}, {"question": "q", "exclude": [1]},
                     {"question": "q", "other": 1}, []):
            self.assertEqual(self.p.call("POST", "/v1/notes/lookup", body, k)[0], 400, body)
        self.assertEqual(self.p.call("GET", "/v1/notes/lookup", headers=k)[0], 405)
        self.assertEqual(self.p.call("POST", "/v1/notes/lookup", {"question": "q"})[0], 401)

    def test_it_works_for_one_owner_too(self):
        store = self.tmp.file("one.jsonl")
        p = H.Running(self.stub.url, store)
        self.addCleanup(p.stop)
        p.json("POST", "/v1/notes", {"text": FACT, "subject": "fleet"})
        st, v = p.json("POST", "/v1/notes/lookup", {"question": QUESTION})
        self.assertEqual(st, 200)
        self.assertIn(FACT, v["block"])
        self.assertIsNone(v["notes"][0]["by"])


class ReadOnly(TeamCase):
    def test_it_reads_and_does_nothing_else(self):
        self.serve()
        _, w = self.write("hermes", FACT)
        k = auth(self.keys["cloud"])
        size = os.path.getsize(os.path.join(self.stores, "team.jsonl"))
        before = len(self.stub.requests)
        self.assertEqual(self.p.call("GET", "/v1/notes", headers=k)[0], 200)
        self.assertEqual(self.p.call("GET", f"/v1/notes/{w['id']}", headers=k)[0], 200)
        refused = [("POST", "/v1/notes", {"text": OTHER}), ("DELETE", f"/v1/notes/{w['id']}", None),
                   ("POST", f"/v1/notes/{w['id']}/weights", {"in_weights": False}),
                   ("POST", "/v1/chat/completions", ask(QUESTION)), ("GET", "/v1/models", None),
                   ("POST", "/v1/completions", {"prompt": "x"}), ("POST", "/v1/embeddings", {"input": "x"}),
                   ("GET", "/v1/slide/graph", None), ("GET", "/anything", None)]
        for method, path, body in refused:
            self.assertEqual(self.p.call(method, path, body, k)[0], 403, path)
        self.assertEqual(len(self.stub.requests), before)
        self.assertEqual(os.path.getsize(os.path.join(self.stores, "team.jsonl")), size)
        print(f"[test_team] a read-only key: 2 reads answered, {len(refused)} other routes refused, 0 upstream requests")


class Slow(BaseHTTPRequestHandler):
    """A team proxy that takes too long, or answers with something else."""
    protocol_version = "HTTP/1.1"
    delay, status, reply = 0.0, 200, b"{}"

    def log_message(self, *a):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        time.sleep(type(self).delay)
        self.send_response(type(self).status)
        self.send_header("Content-Length", str(len(type(self).reply)))
        self.end_headers()
        self.wfile.write(type(self).reply)


class AgentProxy(TeamCase):
    """deep's own proxy: its private notes, and the team's, in front of its own model."""

    def setUp(self):
        super().setUp()
        self.serve()
        self.write("hermes", FACT)
        self.keyfile = self.tmp.file("team.key")
        fd = os.open(self.keyfile, os.O_WRONLY | os.O_CREAT, 0o600)
        os.write(fd, (self.keys["deep"] + "\n").encode())
        os.close(fd)
        self.model = H.Stub()
        self.addCleanup(self.model.stop)

    def agent(self, url=None, *extra):
        url = url or f"http://127.0.0.1:{self.p.port}"
        a = H.Running(self.model.url, self.tmp.file("deep.jsonl"), owner="deep",
                      extra=["--team-url", url, "--team-key-file", self.keyfile] + list(extra))
        self.addCleanup(a.stop)
        return a

    def test_both_stores_reach_the_prompt(self):
        a = self.agent()
        a.json("POST", "/v1/notes", {"text": "Builds never run on asus2.", "subject": "asus2"})
        st, v = a.json("POST", "/v1/chat/completions", ask(QUESTION + " And where do builds not run?"))
        self.assertEqual(st, 200)
        block = sent(self.model)[0]["content"]
        self.assertLess(block.index("Builds never run"), block.index(FACT))
        self.assertTrue(v["notes"]["team"]["ids"])
        self.assertGreater(v["notes"]["team"]["tokens"], 0)
        self.assertTrue(any("; team [" in x for x in a.logs))
        self.assertEqual(len(self.stub.requests), 0)       # the team proxy's own upstream played no part

    def test_only_team_notes(self):
        a = self.agent()
        a.json("POST", "/v1/chat/completions", ask(QUESTION))
        self.assertIn(FACT, sent(self.model)[0]["content"])

    def test_notes_off_asks_nobody(self):
        a = self.agent()
        a.json("POST", "/v1/chat/completions", ask(QUESTION, notes=False))
        self.assertEqual(len(sent(self.model)), 1)
        self.assertFalse(any("lookup" in x for x in self.p.logs))

    def test_notes_query_is_what_the_team_is_asked(self):
        a = self.agent()
        wrapper = "A long wrapper\nwith many lines\nabout nothing here"
        a.json("POST", "/v1/chat/completions", ask(wrapper, notes_query=QUESTION))
        self.assertIn(FACT, sent(self.model)[0]["content"])

    def failed(self, a):
        st, v = a.json("POST", "/v1/chat/completions", ask(QUESTION))
        self.assertEqual(st, 200)
        self.assertEqual(sent(self.model)[-1]["content"], QUESTION)
        self.assertEqual(len(sent(self.model)), 1)
        return v["notes"]["team"]["failed"]

    def test_a_team_proxy_that_is_down_costs_only_its_notes(self):
        a = self.agent("http://127.0.0.1:9")
        self.assertEqual(self.failed(a), "ConnectionRefusedError")

    def bad_server(self, delay=0.0, status=200, reply=b"{}"):
        handler = type("Bad", (Slow,), {"delay": delay, "status": status, "reply": reply})
        srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.server_close)
        self.addCleanup(srv.shutdown)
        return f"http://127.0.0.1:{srv.server_address[1]}"

    def test_a_slow_team_proxy_is_given_up_on(self):
        a = self.agent(self.bad_server(delay=3.0), "--team-timeout", "0.2")
        t0 = time.monotonic()
        self.assertEqual(self.failed(a), "timed out")
        self.assertLess(time.monotonic() - t0, 2.0)

    def test_a_team_proxy_that_drips_is_given_up_on(self):
        class Drip(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
                self.send_response(200)
                self.send_header("Content-Length", "100000")
                self.end_headers()
                try:
                    for _ in range(100):      # each byte inside the socket timeout, the whole reply far past it
                        self.wfile.write(b" ")
                        self.wfile.flush()
                        time.sleep(0.1)
                except OSError:
                    pass

        srv = ThreadingHTTPServer(("127.0.0.1", 0), Drip)
        srv.daemon_threads = True
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.server_close)
        self.addCleanup(srv.shutdown)
        a = self.agent(f"http://127.0.0.1:{srv.server_address[1]}", "--team-timeout", "0.5")
        t0 = time.monotonic()
        self.assertEqual(self.failed(a), "timed out")
        self.assertLess(time.monotonic() - t0, 3.0)
        for _ in range(50):
            if not any(t.name == "team-lookup" for t in threading.enumerate()):
                break
            time.sleep(0.05)
        self.assertFalse([t for t in threading.enumerate() if t.name == "team-lookup"])

    def test_a_reply_past_the_limit_is_refused(self):
        a = self.agent(self.bad_server(reply=b" " * (team.REPLY_LIMIT + 2)))
        self.assertEqual(self.failed(a), "the reply is too long")

    def test_a_refusal_or_a_strange_reply_is_a_failure_with_no_words_of_theirs(self):
        for server, why in ((self.bad_server(status=401, reply=b"secret words"), "HTTP 401"),
                            (self.bad_server(reply=b"not json"), "the reply is not JSON"),
                            (self.bad_server(reply=b"{}"), "the reply is not a lookup")):
            a = self.agent(server)
            self.assertEqual(self.failed(a), why)
            a.stop()

    def test_the_key_stays_out_of_replies_and_logs(self):
        a = self.agent("http://127.0.0.1:9")
        st, _h, raw = a.call("POST", "/v1/chat/completions", ask(QUESTION))
        self.assertEqual(st, 200)
        self.assertNotIn(self.keys["deep"].encode(), raw)
        self.assertFalse(any(self.keys["deep"] in x for x in a.logs))
        self.assertNotIn(self.keys["deep"], json.dumps(self.model.last_body()))

    def test_the_key_file_must_be_private_and_hold_a_key(self):
        os.chmod(self.keyfile, 0o644)
        with self.assertRaises(team.TeamError):
            team.read_key(self.keyfile)
        with self.assertRaises(SystemExit):
            self.agent()
        os.chmod(self.keyfile, 0o600)
        with open(self.keyfile, "w") as f:
            f.write("\n")
        with self.assertRaises(team.TeamError):
            team.read_key(self.keyfile)
        with self.assertRaises(team.TeamError):
            team.read_key(self.tmp.file("nothing"))

    def test_the_two_team_flags_go_together(self):
        args = proxy.parser().parse_args(["--upstream", self.model.url, "--owner", "o", "--store", "s", "--tokenizer",
                                          H.tokenizer_spec(), "--team-url", "http://x:1"])
        self.assertIn("--team-key-file", proxy.usage_error(args))


class BlockPlace(unittest.TestCase):
    MSGS = [{"role": "system", "content": "You are an agent."}, {"role": "user", "content": "first"},
            {"role": "assistant", "content": "ok"}, {"role": "user", "content": "second"}]

    def test_the_default_is_unchanged(self):
        self.assertEqual(proxy.with_block(self.MSGS, "B"), [{"role": "system", "content": "B"}] + self.MSGS)

    def test_last_user_leaves_everything_before_it_alone(self):
        out = proxy.with_block(self.MSGS, "B", at="last-user")
        self.assertEqual(out[:3], self.MSGS[:3])
        self.assertEqual(out[3], {"role": "user", "content": "B\n\nsecond"})
        self.assertEqual(self.MSGS[3]["content"], "second")

    def test_a_tool_loop_keeps_the_block_where_it_was(self):
        msgs = self.MSGS + [{"role": "assistant", "content": "calling"}, {"role": "tool", "content": "done"}]
        out = proxy.with_block(msgs, "B", at="last-user")
        self.assertEqual(out[3]["content"], "B\n\nsecond")
        self.assertEqual(out[4:], msgs[4:])

    def test_parts_and_no_user_message(self):
        parts = [{"role": "user", "content": [{"type": "text", "text": "look"}, {"type": "image_url", "image_url": {}}]}]
        out = proxy.with_block(parts, "B", at="last-user")
        self.assertEqual(out[0]["content"][0], {"type": "text", "text": "B\n\n"})
        self.assertEqual(out[0]["content"][1:], parts[0]["content"])
        sysonly = [{"role": "system", "content": "s"}]
        self.assertEqual(proxy.with_block(sysonly, "B", at="last-user")[0], {"role": "system", "content": "B"})

    def test_through_the_proxy(self):
        tmp, stub = H.TempDir(), H.Stub()
        self.addCleanup(tmp.cleanup)
        self.addCleanup(stub.stop)
        p = H.Running(stub.url, tmp.file("n.jsonl"), extra=["--block-at", "last-user"])
        self.addCleanup(p.stop)
        p.json("POST", "/v1/notes", {"text": FACT, "subject": "fleet"})
        st, _ = p.json("POST", "/v1/chat/completions", {"model": "m", "messages": [
            {"role": "system", "content": "You are an agent."}, {"role": "user", "content": QUESTION}]})
        self.assertEqual(st, 200)
        got = sent(stub)
        self.assertEqual(got[0], {"role": "system", "content": "You are an agent."})
        self.assertIn(FACT, got[1]["content"])
        self.assertTrue(got[1]["content"].endswith(QUESTION))


if __name__ == "__main__":
    unittest.main()
