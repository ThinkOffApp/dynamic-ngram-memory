"""End to end: client -> proxy -> a stub OpenAI-shaped upstream that echoes the messages it received."""
import http.client
import json
import os
import time
import unittest
import unittest.mock

from notes_proxy import proxy
from notes_proxy.store import nm
from notes_proxy.tests import helpers as H

CLIENT = [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Hello."},
          {"role": "assistant", "content": "Hi. What can I do?"}]


def chat(q, **kw):
    return dict({"model": "stub-model", "temperature": 0, "messages": CLIENT + [{"role": "user", "content": q}]}, **kw)


def sse_lines(resp):
    """The SSE lines of a streamed reply as they arrive."""
    while True:
        line = resp.readline()
        if not line:
            return
        yield line


class E2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = H.TempDir()
        cls.S, cls.ref = H.reference()
        cls.stub = H.Stub()

    @classmethod
    def tearDownClass(cls):
        cls.stub.stop()
        cls.tmp.cleanup()

    def proxy(self, store, extra=(), handler=None):
        p = H.Running(self.stub.url, store, extra=extra, handler=handler)
        self.addCleanup(p.stop)
        return p

    # ------------------------------------------------------------------------------------------ the block
    def test_block_first_unchanged_and_user_messages_untouched(self):
        p = self.proxy(H.seeded(self.tmp, "e2e-block.jsonl"))
        with_block = 0
        for q in self.S["questions"]:
            body = chat(q["text"], max_tokens=7)
            ref = self.ref.retrieve("user", q["text"])
            st, reply = p.json("POST", "/v1/chat/completions", body)
            self.assertEqual(st, 200)
            got = self.stub.last_body()
            if ref["block"]:
                with_block += 1
                self.assertEqual(got["messages"][0], {"role": "system", "content": ref["block"]})
                self.assertEqual(got["messages"][1:], body["messages"])
            else:
                self.assertEqual(got["messages"], body["messages"])
            self.assertEqual({k: v for k, v in got.items() if k != "messages"},
                             {k: v for k, v in body.items() if k != "messages"})
            n = reply["notes"]
            self.assertEqual((n["ids"], n["tokens"], n["past"]), (ref["ids"], ref["tokens"], ref["past"]))
            self.assertEqual(reply["received_messages"], got["messages"])          # the upstream's reply, kept
        print(f"\n[test_e2e] 56 of 56 requests: the block first and byte for byte retrieve()'s ({with_block} with "
              f"a block), every client message after it unchanged")

    def test_notes_echo_and_sha(self):
        p = self.proxy(H.seeded(self.tmp, "e2e-echo.jsonl"))
        q = self.S["questions"][3]["text"]
        st, reply = p.json("POST", "/v1/chat/completions", chat(q, notes_echo=True))
        ref = self.ref.retrieve("user", q)
        self.assertEqual(reply["notes"]["block"], ref["block"])
        import hashlib
        self.assertEqual(reply["notes"]["block_sha256"], hashlib.sha256(ref["block"].encode()).hexdigest())
        self.assertNotIn("notes_echo", self.stub.last_body())

    def test_notes_false_sends_the_request_unchanged(self):
        p = self.proxy(H.seeded(self.tmp, "e2e-off.jsonl"))
        body = chat(self.S["questions"][0]["text"], notes=False)
        before = len(self.stub.requests)
        st, h, raw = p.call("POST", "/v1/chat/completions", body)
        self.assertEqual(st, 200)
        self.assertEqual(len(self.stub.requests), before + 1)
        self.assertEqual(self.stub.last_body(), {k: v for k, v in body.items() if k != "notes"})
        self.assertNotIn(b'"notes"', raw)
        self.assertNotIn("X-Notes-Ids", h)
        # control: the same question with notes on does change the request
        p.call("POST", "/v1/chat/completions", dict(body, notes=True))
        self.assertNotEqual(self.stub.last_body()["messages"], body["messages"])

    def test_nothing_found_puts_nothing_in(self):
        p = self.proxy(self.tmp.file("e2e-empty.jsonl"))
        body = chat("What is the capital of Peru?")
        st, reply = p.json("POST", "/v1/chat/completions", body)
        self.assertEqual(self.stub.last_body(), body)
        self.assertEqual((reply["notes"]["ids"], reply["notes"]["tokens"], reply["notes"]["block_sha256"]),
                         ([], 0, None))

    def test_streaming_with_notes_off_passes_through(self):
        p = self.proxy(H.seeded(self.tmp, "e2e-stream-off.jsonl"))
        self.stub.release.set()
        st, h, raw = p.call("POST", "/v1/chat/completions", chat("hi", stream=True, notes=False))
        self.assertEqual(st, 200)
        self.assertNotIn(b": notes", raw)
        self.assertTrue(raw.rstrip().endswith(b"data: [DONE]"))
        self.assertEqual(json.loads(raw.split(b"\n")[0][len(b"data: "):])["received_messages"], chat("hi")["messages"])

    def test_join_system(self):
        p = self.proxy(H.seeded(self.tmp, "e2e-join.jsonl"), extra=["--join-system"])
        q = self.S["questions"][0]["text"]
        p.json("POST", "/v1/chat/completions", chat(q))
        got = self.stub.last_body()["messages"]
        ref = self.ref.retrieve("user", q)["block"]
        self.assertEqual(got[0], {"role": "system", "content": ref + "\n\nYou are terse."})
        self.assertEqual(got[1:], CLIENT[1:] + [{"role": "user", "content": q}])

    def test_refusals(self):
        p = self.proxy(H.seeded(self.tmp, "e2e-bad.jsonl"))
        n = len(self.stub.requests)
        for bad in [{"notes": "yes"}, {"notes": False, "notes_exclude": [1]}, {"notes_cap": 0},
                    {"notes_exclude": ["a"]}, {"notes": False, "notes_echo": True}, {"notes_language": "fi"}]:
            st, reply = p.json("POST", "/v1/chat/completions", chat("hi", **bad))
            self.assertEqual(st, 400, bad)
            self.assertIn("message", reply["error"])
        self.assertEqual(len(self.stub.requests), n)            # nothing refused reached the upstream

    # ------------------------------------------------------------------------------------------ streaming
    def stream_first_chunk_early(self, p):
        """Whether the first SSE chunk reached the client while the stub was still holding the rest back."""
        self.stub.release.clear()
        self.stub.released_by = None
        q = self.S["questions"][0]["text"]
        c = http.client.HTTPConnection("127.0.0.1", p.port, timeout=30)
        c.request("POST", "/v1/chat/completions", body=json.dumps(chat(q, stream=True)),
                  headers={"Content-Type": "application/json"})
        r = c.getresponse()
        lines = sse_lines(r)
        first = next(l for l in lines if l.startswith(b"data: "))
        early = self.stub.finished_at is None
        self.stub.release.set()
        rest = list(lines)
        c.close()
        return first, rest, early and self.stub.released_by in (None, "event")

    def test_streaming_is_incremental(self):
        p = self.proxy(H.seeded(self.tmp, "e2e-stream.jsonl"))
        self.stub.finished_at = None
        first, rest, early = self.stream_first_chunk_early(p)
        self.assertTrue(early)
        self.assertEqual(self.stub.released_by, "event")      # the stub went on only because the client had chunk 1
        msgs = json.loads(first[len(b"data: "):])["received_messages"]
        q = self.S["questions"][0]["text"]
        self.assertEqual(msgs[0], {"role": "system", "content": self.ref.retrieve("user", q)["block"]})
        datas = [l for l in rest if l.startswith(b"data: ")]
        self.assertEqual(datas[-1].strip(), b"data: [DONE]")
        notes = [l for l in rest if l.startswith(b": notes ")]
        self.assertEqual(len(notes), 1)
        self.assertLess(rest.index(notes[0]), rest.index(datas[-1]))          # before [DONE]
        self.assertEqual(json.loads(notes[0][len(b": notes "):])["ids"], self.ref.retrieve("user", q)["ids"])

    def test_control_a_buffering_proxy_fails_the_streaming_check(self):
        class Buffering(proxy.Handler):
            def relay_sse(self, r, headers, notes_hdr, used, t0, extra):
                data = r.read()                           # the whole reply first, then on: what must not happen
                self.send_response(r.status)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                self.wfile.flush()

        p = self.proxy(H.seeded(self.tmp, "e2e-buffer.jsonl"), handler=Buffering)
        self.stub.finished_at = None
        t0 = time.monotonic()
        first, rest, early = self.stream_first_chunk_early(p)
        self.assertFalse(early)
        self.assertEqual(self.stub.released_by, "timeout")
        self.assertGreater(time.monotonic() - t0, 4.5)

    # ------------------------------------------------------------------------------------------ erase and restart
    def test_erase_removes_the_note_and_the_next_request_lacks_it(self):
        store = self.tmp.file("e2e-erase.jsonl")
        p = self.proxy(store)
        text = "My neighbour Oskarina keeps seven tortoises."
        st, w = p.json("POST", "/v1/notes", {"text": text})
        self.assertEqual((st, w["verdict"], w["written"]), (201, "NEW", True))
        keep = p.json("POST", "/v1/notes", {"text": "My aunt Brigitta grows tomatoes."})[1]
        q = "How many tortoises does Oskarina keep?"
        p.json("POST", "/v1/chat/completions", chat(q))
        self.assertIn(json.dumps(text), self.stub.last_body()["messages"][0]["content"])
        st, e = p.json("DELETE", f"/v1/notes/{w['id']}")
        self.assertEqual(st, 200)
        self.assertEqual(e["removed"], [w["source"]])
        self.assertEqual(e["renumbered"], [{"from": keep["id"], "to": w["id"]}])
        # right after the erase (a later question brings its own words into the lookup's memos again): gone from
        # every structure the proxy holds (notes.py's own walker), and from the journal on disk
        self.assertEqual(nm.walk_for(p.notes, texts=["Oskarina", "tortoises"]), [])
        self.assertEqual(nm.walk_for(p.srv.cfg, texts=["Oskarina", "tortoises"]), [])
        with open(store, "rb") as f:
            self.assertNotIn(b"Oskarina", f.read())
        self.assertEqual([f for f in os.listdir(self.tmp.path) if ".tmp." in f], [])
        st, reply = p.json("POST", "/v1/chat/completions", chat(q, notes_echo=True))
        self.assertNotIn("Oskarina", json.dumps(self.stub.last_body()["messages"][:-1]))   # the last is the question
        self.assertNotIn("Oskarina", reply["notes"]["block"])
        self.assertEqual([n["text"] for n in p.json("GET", "/v1/notes?history=1")[1]["data"]],
                         ["My aunt Brigitta grows tomatoes."])
        self.assertEqual(p.json("GET", f"/v1/notes/{w['id']}")[1]["text"], "My aunt Brigitta grows tomatoes.")
        p.stop()
        p2 = self.proxy(store)                                      # and after a restart
        self.assertEqual([n["text"] for n in p2.json("GET", "/v1/notes?history=1")[1]["data"]],
                         ["My aunt Brigitta grows tomatoes."])
        p2.json("POST", "/v1/chat/completions", chat(q))
        self.assertNotIn("Oskarina", json.dumps(self.stub.last_body()["messages"][:-1]))

    def test_control_the_walker_finds_a_kept_note(self):
        p = self.proxy(self.tmp.file("e2e-walk.jsonl"))
        p.json("POST", "/v1/notes", {"text": "My neighbour Oskarina keeps seven tortoises."})
        self.assertNotEqual(nm.walk_for(p.notes, texts=["Oskarina"]), [])

    def test_erase_also_takes_out_the_same_as_writes(self):
        p = self.proxy(self.tmp.file("e2e-same.jsonl"))
        a = p.json("POST", "/v1/notes", {"text": "My boss Ingrid rides a unicycle."})[1]
        st, b = p.json("POST", "/v1/notes", {"text": "my boss  Ingrid rides a UNICYCLE"})
        self.assertEqual((st, b["verdict"], b["written"], b["same_as"]), (200, "ALREADY_KNOWN", False, [a["id"]]))
        e = p.json("DELETE", f"/v1/notes/{a['id']}")[1]
        self.assertEqual(sorted(e["removed"]), sorted([a["source"], b["source"]]))
        self.assertEqual(nm.walk_for(p.notes, texts=["unicycle"]), [])

    def test_erase_with_a_failure_after_the_rename(self):
        """Every step after the new journal is renamed into place can fail. The erase must then hold in memory, say
        the disk state is unknown, refuse later writes until a restart, and stay erased after the restart."""
        from notes_proxy import store as S

        def opening_fails(j):
            raise OSError(5, "Input/output error")

        def sync_fails(path):
            raise OSError(5, "Input/output error")

        for name, patch in (("open", lambda p: unittest.mock.patch.object(S.Journal, "open_append", opening_fails)),
                            ("sync", lambda p: unittest.mock.patch.object(S, "_sync_dir", sync_fails))):
            with self.subTest(step=name):
                store = self.tmp.file(f"e2e-erase-fault-{name}.jsonl")
                p = self.proxy(store)
                text = "My neighbour Oskarina keeps seven tortoises."
                w = p.json("POST", "/v1/notes", {"text": text})[1]
                p.json("POST", "/v1/notes", {"text": "My aunt Brigitta grows tomatoes."})
                with patch(p):
                    st, e = p.json("DELETE", f"/v1/notes/{w['id']}")
                self.assertEqual(st, 500)
                self.assertIn("disk state is unknown", e["error"]["message"])
                self.assertNotIn(text, [n["text"] for n in p.json("GET", "/v1/notes")[1]["data"]])
                self.assertEqual(nm.walk_for(p.notes, texts=["tortoises"]), [])
                self.assertTrue(p.notes.journal.poisoned)
                st, r = p.json("POST", "/v1/notes", {"text": "My cousin Henrik sails."})
                self.assertEqual(st, 503)
                with open(store, encoding="utf-8") as f:
                    self.assertNotIn("tortoises", f.read())
                p.stop()
                q = self.proxy(store)
                self.assertNotIn(text, [n["text"] for n in q.json("GET", "/v1/notes")[1]["data"]])
                self.assertIn("My aunt Brigitta grows tomatoes.", [n["text"] for n in q.json("GET", "/v1/notes")[1]["data"]])
                self.assertEqual(q.json("POST", "/v1/notes", {"text": "My cousin Henrik sails."})[0], 201)

    def test_restart_keeps_notes(self):
        store = H.seeded(self.tmp, "e2e-restart.jsonl")
        p = self.proxy(store)
        w = p.json("POST", "/v1/notes", {"text": "My piano teacher Ludmila lives in Tampere."})[1]
        before = p.json("GET", "/v1/notes?history=1")[1]["data"]
        q = "Where does Ludmila live?"
        p.json("POST", "/v1/chat/completions", chat(q))
        block = self.stub.last_body()["messages"][0]["content"]
        self.assertIn("Ludmila", block)
        p.stop()
        p2 = self.proxy(store)
        self.assertEqual(p2.json("GET", "/v1/notes?history=1")[1]["data"], before)
        p2.json("POST", "/v1/chat/completions", chat(q))
        self.assertEqual(self.stub.last_body()["messages"][0]["content"], block)
        w2 = p2.json("POST", "/v1/notes", {"text": "My cousin Aurelio repairs clocks."})[1]
        self.assertNotEqual(w2["source"], w["source"])                       # sources are never used twice

    def test_a_torn_last_line_is_cut(self):
        store = self.tmp.file("e2e-torn.jsonl")
        p = self.proxy(store)
        p.json("POST", "/v1/notes", {"text": "My uncle Bertil sails a ketch."})
        p.stop()
        with open(store, "ab") as f:
            f.write(b'{"lesson":{"op":"ASSERT","text":"half')
        p2 = self.proxy(store)
        self.assertEqual([n["text"] for n in p2.json("GET", "/v1/notes")[1]["data"]],
                         ["My uncle Bertil sails a ketch."])

    # ------------------------------------------------------------------------------------------ writes from chat
    def test_write_mode(self):
        p = self.proxy(self.tmp.file("e2e-write.jsonl"), extra=["--write"])
        say = lambda msgs, **kw: p.json("POST", "/v1/chat/completions",
                                        dict({"model": "m", "messages": msgs}, **kw))[1]["notes"]["write"]
        u = lambda t: {"role": "user", "content": t}
        a = lambda t: {"role": "assistant", "content": t}
        # a statement the rule book reads, and one it does not (the server's free-text intake)
        w1 = say([u("My favourite colour is teal.")])
        self.assertEqual(w1["verdict"], "NEW")
        w2 = say([u("My sister Valpuri drives a red Volvo.")])
        self.assertEqual(w2["verdict"], "NEW")
        self.assertEqual(say([u("my sister Valpuri drives a red Volvo")])["verdict"], "ALREADY_KNOWN")
        # the next request finds what was written
        n = p.json("POST", "/v1/chat/completions", {"model": "m", "messages": [u("What car does Valpuri drive?")]})[1]
        self.assertEqual(n["notes"]["write"], {"skipped": "a question: it ends in a question mark"})
        self.assertIn(w2["id"], n["notes"]["ids"])
        self.assertIn("Valpuri", self.stub.last_body()["messages"][0]["content"])
        # questions, requests, the model's answers and earlier messages are never written
        self.assertEqual(say([u("what colour do I like")])["skipped"], "a question: it opens as one")
        self.assertEqual(say([u("Tell me about my sister.")])["skipped"], "a request: it asks to be told something")
        self.assertEqual(say([u("My dog Rufus eats carrots."), a("Your cat Zorro hates rain.")])["skipped"],
                         "the last message is not the user's: only the user's own words are written")
        self.assertEqual(say([u("I collect brass compasses."), a("Your cat Zorro hates rain."),
                              u("Remind me what I collect.")])["skipped"], "a request: it asks to be told something")
        self.assertEqual(say([u("I keep bees.")], notes_write=False)["skipped"],
                         "off: the request sends \"notes_write\": false")
        self.assertEqual(say([{"role": "system", "content": "The user owns a yacht."}, u("Hmm.")])["verdict"], "NEW")
        texts = [x["text"] for x in p.json("GET", "/v1/notes?history=1")[1]["data"]]
        self.assertEqual(texts, ["My favourite colour is teal.", "My sister Valpuri drives a red Volvo.", "Hmm."])
        O = p.notes.store.owner("user")
        with open(p.notes.journal.path, "rb") as f:
            journal = f.read().decode()
        for never in ("Rufus", "Zorro", "compasses", "bees", "yacht"):   # in no note, no write and not on disk
            self.assertEqual(nm.walk_for([O.notes, O.log], texts=[never]), [], never)
            self.assertNotIn(never, journal)
        self.assertNotEqual(nm.walk_for([O.notes, O.log], texts=["Valpuri"]), [])          # the check can fail
        # a change the rule book reads retires the old note (the intake's REPLACES)
        w3 = say([u("My favourite colour is red now.")])
        self.assertEqual(w3["verdict"], "REPLACES")
        cur = [x["text"] for x in p.json("GET", "/v1/notes")[1]["data"]]
        self.assertNotIn("My favourite colour is teal.", cur)

    def test_write_mode_off_by_default(self):
        p = self.proxy(self.tmp.file("e2e-nowrite.jsonl"))
        r = p.json("POST", "/v1/chat/completions", {"model": "m", "messages": [
            {"role": "user", "content": "My favourite colour is teal."}]})[1]
        self.assertEqual(r["notes"]["write"], {"skipped": "off: the proxy runs without --write"})
        self.assertEqual(p.json("GET", "/v1/notes")[1]["data"], [])

    # ------------------------------------------------------------------------------------------ the rest
    def test_other_routes_pass_through_with_the_key(self):
        p = self.proxy(H.seeded(self.tmp, "e2e-pass.jsonl"))
        key = "sk-test-not-a-real-key-123"
        st, h, raw = p.call("GET", "/v1/models", headers={"Authorization": f"Bearer {key}"})
        self.assertEqual(st, 200)
        direct = http.client.HTTPConnection("127.0.0.1", self.stub.srv.server_address[1])
        direct.request("GET", "/v1/models")
        self.assertEqual(raw, direct.getresponse().read())
        direct.close()
        self.assertEqual(self.stub.requests[-2]["headers"].get("Authorization"), f"Bearer {key}")
        p.json("POST", "/v1/chat/completions", chat("hi"), headers={"Authorization": f"Bearer {key}"})
        self.assertEqual(self.stub.requests[-1]["headers"].get("Authorization"), f"Bearer {key}")
        self.assertTrue(p.logs)
        self.assertFalse(any(key in l for l in p.logs))
        self.assertEqual(nm.walk_for(p.srv.cfg, texts=[key]), [])
        self.assertEqual(p.call("GET", "/v1/nowhere")[0], 404)

    def test_web_pages_are_refused(self):
        p = self.proxy(self.tmp.file("e2e-origin.jsonl"))
        n = len(self.stub.requests)
        st, h, raw = p.call("POST", "/v1/notes", {"text": "I am a page."}, headers={"Origin": "https://evil.example"})
        self.assertEqual(st, 403)
        st, h, raw = p.call("POST", "/v1/chat/completions", chat("hi"), headers={"Origin": "https://evil.example"})
        self.assertEqual(st, 403)
        self.assertEqual(len(self.stub.requests), n)
        self.assertEqual(p.json("GET", "/v1/notes")[1]["data"], [])

    def test_other_host_names_are_refused(self):
        """DNS rebinding: a page's own name pointed at 127.0.0.1 is same-origin, so it sends no Origin header."""
        p = self.proxy(self.tmp.file("e2e-host.jsonl"))
        self.assertEqual(p.call("POST", "/v1/notes", {"text": "My sister Anna drives a red Volvo."})[0], 201)
        n = len(self.stub.requests)
        for host in ("evil.example", "evil.example:8090", "127.0.0.1.evil.example"):
            self.assertEqual(p.call("GET", "/v1/notes", headers={"Host": host})[0], 403, host)
            self.assertEqual(p.call("POST", "/v1/chat/completions", chat("hi"), headers={"Host": host})[0], 403)
        self.assertEqual(len(self.stub.requests), n)
        for host in ("localhost", "127.0.0.1:%d" % p.port, "[::1]:%d" % p.port, "LOCALHOST"):
            self.assertEqual(p.call("GET", "/v1/notes", headers={"Host": host})[0], 200, host)
        # control: --allow-remote serves any name, so the refusal above is the Host check and nothing else
        q = self.proxy(self.tmp.file("e2e-host-remote.jsonl"), extra=["--allow-remote"])
        self.assertEqual(q.call("GET", "/v1/notes", headers={"Host": "evil.example"})[0], 200)

    def test_upstream_down_is_a_502(self):
        p = H.Running("http://127.0.0.1:9", self.tmp.file("e2e-down.jsonl"))
        self.addCleanup(p.stop)
        st, reply = p.json("POST", "/v1/chat/completions", chat("hi"))
        self.assertEqual(st, 502)


if __name__ == "__main__":
    unittest.main()
