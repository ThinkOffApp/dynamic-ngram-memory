"""Filling a store from a transcript: what the model sees, what reaches the candidates file, and that only what a person
approved reaches the store."""
import contextlib
import io
import json
import os
import secrets
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from notes_proxy import prefill, proxy
from notes_proxy.tests import helpers as H

KEY = "xfb_" + "a1" * 32
FACT = "The Bosgame serves Flash Next to hermes on port 8080."


class Model:
    """A chat server that answers each request with `reply`, and keeps what it was sent."""

    def __init__(self, reply):
        self.seen = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                outer.seen.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                r = reply(outer.seen[-1]) if callable(reply) else reply
                data = json.dumps({"choices": [{"message": {"content": r}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"

    def stop(self):
        self.srv.shutdown()
        self.srv.server_close()


def message(i, body, who="petrus", human=True, at=None):
    return {"id": f"id-{i}", "created_at": at or f"2026-11-0{i % 5 + 1}T10:00:00+00:00", "from": who, "body": body,
            "isHuman": human}


class Case(unittest.TestCase):
    def setUp(self):
        self.tmp = H.TempDir()
        self.addCleanup(self.tmp.cleanup)
        self.path = self.tmp.file("room.jsonl")
        self.out = self.tmp.file("candidates.jsonl")

    def transcript(self, rows):
        with open(self.path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    def extract(self, reply, *extra):
        m = Model(reply)
        self.addCleanup(m.stop)
        with contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()):
            code = prefill.main(["extract", "--transcript", self.path, "--model-url", m.url, "--model", "x",
                                 "--out", self.out] + list(extra))
        self.said = out.getvalue()
        self.code = code
        return m, [json.loads(x) for x in open(self.out)]


class Extract(Case):
    def test_a_good_fact_keeps_its_message(self):
        self.transcript([message(1, "the box is on 8080")])
        _, rows = self.extract(json.dumps([{"fact": FACT, "subject": "Bosgame", "id": "m0"}]))
        self.assertEqual(rows[0]["fact"], FACT)
        self.assertEqual((rows[0]["id"], rows[0]["from"]), ("id-1", "petrus"))
        self.assertIs(rows[0]["approved"], False)
        self.assertIn("Nothing was written", self.said)

    def test_keys_never_reach_the_model(self):
        for secret in (KEY, "sk-" + "b" * 30, "Bearer " + "c" * 40, "password: hunter2hunter2", "d" * 64,
                       secrets.token_urlsafe(32), "-" + secrets.token_urlsafe(32), "ykn_" + "e" * 32,
                       "Authorization: Basic " + "f" * 24, "authorization: Token abc123", "Basic " + "g" * 24):
            self.transcript([message(1, f"here it is {secret} ok")])
            m, _ = self.extract("[]")
            sent = m.seen[0]["messages"][1]["content"]
            self.assertNotIn(secret, sent, secret)
            self.assertIn("[removed]", sent)

    def test_a_fact_holding_a_key_is_dropped(self):
        self.transcript([message(1, "x")])
        _, rows = self.extract(json.dumps([{"fact": f"The room key for the MacBook is {KEY} today.", "id": "m0"}]))
        self.assertEqual(rows, [])

    def test_a_subject_holding_a_key_is_dropped(self):
        self.transcript([message(1, "x")])
        _, rows = self.extract(json.dumps([{"fact": FACT, "subject": KEY, "id": "m0"}]))
        self.assertEqual(rows, [])

    def test_a_made_up_tag_a_repeat_and_odd_sizes_are_dropped(self):
        self.transcript([message(1, "x"), message(2, "y")])
        reply = json.dumps([{"fact": FACT, "id": "m0"}, {"fact": FACT.upper(), "id": "m1"},
                            {"fact": "Another fact about the router here.", "id": "m9"}, {"fact": "short", "id": "m0"},
                            {"fact": "x" * 400, "id": "m0"}, "not an object", {"id": "m0"}])
        _, rows = self.extract(reply)
        self.assertEqual([r["fact"] for r in rows], [FACT])

    def test_a_report_of_what_someone_said_is_dropped(self):
        self.transcript([message(1, "x")])
        said = ["Petrus asked why the software is not used by the team.", "hermes confirmed the restart went well today.",
                "Petrus wants every local install to run the Notes Proxy.", "Deep reports the store has 14 notes now."]
        reply = json.dumps([{"fact": f, "id": "m0"} for f in said] + [{"fact": FACT, "id": "m0"}])
        self.assertEqual([r["fact"] for r in self.extract(reply)[1]], [FACT])

    def test_an_id_of_the_wrong_type_is_dropped_and_the_run_goes_on(self):
        self.transcript([message(1, "x")])
        reply = json.dumps([{"fact": FACT + " One.", "id": []}, {"fact": FACT + " Two.", "id": {"a": 1}},
                            {"fact": FACT + " Three.", "id": None}, {"fact": FACT, "id": "m0"}])
        self.assertEqual([r["fact"] for r in self.extract(reply)[1]], [FACT])
        self.assertEqual(self.code, 0)

    def test_a_reply_that_is_not_a_list_gives_nothing(self):
        self.transcript([message(1, "x")])
        for reply in ("nothing here", "{\"a\": 1}", "[broken", ""):
            self.assertEqual(self.extract(reply)[1], [])

    def test_it_reads_oldest_first_within_the_dates_and_skips_approvals(self):
        self.transcript([message(3, "third"), message(1, "first"), message(2, "[Confirmation needed] approve"),
                         message(4, "too new", at="2026-12-01T00:00:00+00:00")])
        m, _ = self.extract("[]", "--since", "2026-11-01", "--until", "2026-11-30")
        sent = m.seen[0]["messages"][1]["content"]
        self.assertLess(sent.index("first"), sent.index("third"))
        self.assertNotIn("approve", sent)
        self.assertNotIn("too new", sent)

    def test_the_model_keeps_the_transcript_out_of_its_own_memory(self):
        self.transcript([message(1, "x")])
        m, _ = self.extract("[]")
        self.assertIs(m.seen[0]["store"], False)

    def test_an_older_fact_on_the_same_subject_is_marked(self):
        self.transcript([message(1, "a", at="2026-08-01T00:00:00+00:00"), message(2, "b", at="2026-11-05T00:00:00+00:00")])
        reply = json.dumps([{"fact": "The Bosgame's LAN address is 192.168.50.127.", "subject": "Bosgame", "id": "m0"},
                            {"fact": "The Bosgame's LAN address is 192.168.50.205.", "subject": "bosgame", "id": "m1"}])
        _, rows = self.extract(reply)
        self.assertEqual([r["newer_same_subject"] for r in rows], [1, 0])

    def test_a_failed_chunk_is_reported_and_the_rest_goes_on(self):
        self.transcript([message(1, "x"), message(2, "y")])
        m = Model("[]")
        self.addCleanup(m.stop)
        m.stop()
        with contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()):
            code = prefill.main(["extract", "--transcript", self.path, "--model-url", m.url, "--model", "x",
                                 "--out", self.out, "--chunk", "1"])
        self.assertEqual(code, 1)
        self.assertIn("2 chunks failed", out.getvalue())


class Apply(Case):
    def setUp(self):
        super().setUp()
        self.stub = H.Stub()
        self.addCleanup(self.stub.stop)
        self.p = H.Running(self.stub.url, self.tmp.file("store.jsonl"))
        self.addCleanup(self.p.stop)
        self.key = self.tmp.file("key")
        with open(self.key, "w") as f:
            f.write("any\n")
        os.chmod(self.key, 0o600)

    def candidates(self, rows):
        with open(self.out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    def apply(self, *extra):
        args = ["apply", "--candidates", self.out] + list(extra)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.code = prefill.main(args)
        return out.getvalue()

    def to_store(self):
        return self.apply("--url", f"http://127.0.0.1:{self.p.port}", "--key-file", self.key,
                          "--cite-prefix", "room:team/")

    def stored(self):
        return self.p.json("GET", "/v1/notes")[1]["data"]

    def test_only_what_was_approved_goes_in_with_its_message(self):
        self.candidates([{"fact": FACT, "subject": "Bosgame", "id": "a", "at": "2", "approved": True},
                         {"fact": "The router is at 192.168.50.1 at home.", "subject": "router", "id": "b", "at": "1",
                          "approved": False},
                         {"fact": "Builds never run on the asus2 machine.", "subject": "asus2", "id": "c", "at": "3"}])
        self.assertIn("1 notes written", self.to_store())
        self.assertEqual([(n["text"], n["cite"]) for n in self.stored()], [(FACT, "room:team/a")])

    def test_a_second_run_writes_nothing_new(self):
        self.candidates([{"fact": FACT, "subject": "Bosgame", "id": "a", "at": "1", "approved": True}])
        self.to_store()
        self.assertIn("0 notes written, 1 already known", self.to_store())
        self.assertEqual(len(self.stored()), 1)

    def test_a_dry_run_writes_nothing(self):
        self.candidates([{"fact": FACT, "subject": "Bosgame", "id": "a", "at": "1", "approved": True}])
        self.assertIn("1 of 1 approved", self.apply("--dry-run"))
        self.assertEqual(self.stored(), [])

    def test_a_key_in_an_approved_fact_is_still_refused(self):
        self.candidates([{"fact": f"The key is {KEY} for now.", "subject": "x", "id": "a", "at": "1", "approved": True}])
        self.assertIn("1 failed", self.to_store())
        self.assertEqual(self.code, 1)
        self.assertEqual(self.stored(), [])


    def test_a_key_in_an_approved_subject_is_refused(self):
        self.candidates([{"fact": FACT, "subject": KEY, "id": "a", "at": "1", "approved": True}])
        self.assertIn("1 failed", self.to_store())
        self.assertEqual(self.stored(), [])

    def test_a_key_file_others_can_read_is_refused(self):
        self.candidates([{"fact": FACT, "subject": "Bosgame", "id": "a", "at": "1", "approved": True}])
        os.chmod(self.key, 0o644)
        with self.assertRaises(SystemExit):
            self.to_store()
        self.assertEqual(self.stored(), [])


class Cli(unittest.TestCase):
    def test_the_proxy_command_reaches_it(self):
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            proxy.main(["prefill", "--help"])

    def test_apply_needs_a_store_or_a_dry_run(self):
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            prefill.main(["apply", "--candidates", "c"])


if __name__ == "__main__":
    unittest.main()
