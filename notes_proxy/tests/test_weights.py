"""Notes × Living Weights (A+B): weights markers in the journal, replay, auto-exclude, demote-on-change,
annotated GET /v1/slide/graph, POST /v1/notes/ID/weights, and no client reaching /v1/slide/learn through the proxy.

Written against the fake child (its folder may hold slide_graph.json: what the pocket says it knows)."""
import http.client
import json
import os
import threading
import unittest

from notes_proxy import proxy, users
from notes_proxy.tests import helpers as H

FAKE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fake_tensorfold.py")


def _call(port, method, path, body=None, headers=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    data = json.dumps(body).encode() if body is not None else None
    h = {"Content-Type": "application/json"} if data is not None else {}
    h.update(headers or {})
    c.request(method, path, body=data, headers=h)
    r = c.getresponse()
    out = (r.status, dict(r.getheaders()), r.read())
    c.close()
    return out


def _json(port, method, path, body=None):
    st, _h, raw = _call(port, method, path, body)
    return st, json.loads(raw) if raw else None


class ChildBase(unittest.TestCase):
    def setUp(self):
        self.tmp = H.TempDir()
        self.addCleanup(self.tmp.cleanup)

    def running(self, extra, store=None, path=None):
        model = self.tmp.file("model")
        if not os.path.isdir(model):
            os.mkdir(model)
        if "--users" in extra:
            store = None
        else:
            store = store or self.tmp.file("s.jsonl")
        args = proxy.parser().parse_args([
            "--tensorfold", model, "--tensorfold-bin", FAKE, "--port", "0", "--host", "127.0.0.1",
            "--tensorfold-quiet", "0.15", "--tensorfold-restart-max", "2",
        ] + extra + (store and ["--store", store] or []))
        srv, notes = proxy.build(args, log=lambda *_: None)
        self.addCleanup(notes.close)
        self.addCleanup(srv.cfg.engine.close)
        self.addCleanup(srv.server_close)
        self.addCleanup(srv.shutdown)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        return srv, notes, srv.server_address[1]


def write_graph(model_dir, facts):
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "slide_graph.json"), "w") as f:
        json.dump({"next": len(facts) + 1, "facts": facts, "model": {}}, f)


class Markers(unittest.TestCase):
    def setUp(self):
        self.tmp = H.TempDir()
        self.addCleanup(self.tmp.cleanup)
        self.model = self.tmp.file("model")
        os.makedirs(self.model, exist_ok=True)
        self.store = self.tmp.file("s.jsonl")

    def build(self, extra=(), learned=()):
        """learned: the sentences the pocket's graph says it holds (state learned) — the mark route only accepts
        what the graph confirms."""
        write_graph(self.model, [{"id": i + 1, "text": t, "state": "learned", "neurons": [], "questions": []}
                                 for i, t in enumerate(learned)])
        args = proxy.parser().parse_args(["--tensorfold", self.model, "--tensorfold-bin", FAKE,
                                          "--port", "0", "--host", "127.0.0.1",
                                          "--tensorfold-quiet", "0.15", "--tensorfold-restart-max", "2",
                                          "--owner", "user", "--store", self.store] + list(extra))
        srv, notes = proxy.build(args, log=lambda *_: None)
        self.addCleanup(notes.close)
        self.addCleanup(srv.cfg.engine.close)
        self.addCleanup(srv.server_close)
        self.addCleanup(srv.shutdown)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        return srv, notes

    def test_mark_then_auto_exclude_and_restart(self):
        srv, _notes = self.build(learned=["My uncle Bertil sails a ketch."])
        port = srv.server_address[1]
        w = _json(port, "POST", "/v1/notes", {"text": "My uncle Bertil sails a ketch."})[1]
        nid = w["id"]
        q = {"model": "m", "messages": [{"role": "user", "content": "What does my uncle Bertil do?"}]}
        used = _json(port, "POST", "/v1/chat/completions", q)[1]["notes"]
        self.assertIn(nid, used["ids"])                       # the note rides the block before teaching
        st, b = _json(port, "POST", f"/v1/notes/{nid}/weights", {"in_weights": True})
        self.assertEqual((st, b["changed"]), (200, True))
        used = _json(port, "POST", "/v1/chat/completions", q)[1]["notes"]
        self.assertNotIn(nid, used["ids"])                    # taught: the note leaves the prompt
        listing = _json(port, "GET", "/v1/notes")[1]["data"]
        self.assertTrue([n for n in listing if n["id"] == nid][0]["in_weights"])
        # the control arm: the request asks for the taught note back
        back = _json(port, "POST", "/v1/chat/completions", dict(q, notes_in_weights=True))[1]["notes"]
        self.assertIn(nid, back["ids"])
        # restart: the journal replays the marker, so the note stays out
        srv.shutdown()
        srv.server_close()
        srv.cfg.engine.close()
        _notes.close()                                       # free the journal's flock for the reopen
        srv2, _ = self.build()                               # same store file, new process state
        port2 = srv2.server_address[1]
        used = _json(port2, "POST", "/v1/chat/completions", q)[1]["notes"]
        self.assertNotIn(nid, used["ids"])

    def test_mark_unknown_note_404_and_shape(self):
        srv, _notes = self.build()
        port = srv.server_address[1]
        self.assertEqual(_json(port, "POST", "/v1/notes/w999/weights", {"in_weights": True})[0], 404)
        w = _json(port, "POST", "/v1/notes", {"text": "My uncle Bertil sails a ketch."})[1]["id"]
        self.assertEqual(_json(port, "POST", f"/v1/notes/{w}/weights", {"in_weights": "yes"})[0], 400)
        self.assertEqual(_json(port, "POST", f"/v1/notes/{w}/weights", {"in_weight": True})[0], 400)
        self.assertEqual(_json(port, "GET", f"/v1/notes/{w}/weights")[0], 405)

    def test_demote_on_change(self):
        srv, _notes = self.build(["--write"], learned=["My favourite colour is teal."])
        port = srv.server_address[1]
        a = _json(port, "POST", "/v1/notes", {"text": "My favourite colour is teal."})[1]
        self.assertEqual(_json(port, "POST", f"/v1/notes/{a['id']}/weights", {"in_weights": True})[0], 200)
        r = _json(port, "POST", "/v1/chat/completions",
                  {"model": "m", "messages": [{"role": "user", "content": "My favourite colour is red now."}]})[1]
        self.assertEqual(r["notes"]["write"]["verdict"], "REPLACES")
        self.assertEqual(r["notes"]["write"]["demoted"], [a["id"]])   # the taught note the write retired
        cur = _json(port, "GET", "/v1/notes?history=1")[1]["data"]
        self.assertFalse([n for n in cur if n["id"] == a["id"]][0]["in_weights"])   # the retired note: open again
        self.assertEqual([n["id"] for n in _json(port, "GET", "/v1/notes")[1]["data"]
                         if n["in_weights"]], [])                                    # nothing taught rides as earned

    def test_erase_drops_marker(self):
        srv, _notes = self.build(learned=["My cousin Aurelio repairs clocks."])
        port = srv.server_address[1]
        w = _json(port, "POST", "/v1/notes", {"text": "My cousin Aurelio repairs clocks."})[1]["id"]
        self.assertEqual(_json(port, "POST", f"/v1/notes/{w}/weights", {"in_weights": True})[0], 200)
        self.assertEqual(_json(port, "DELETE", f"/v1/notes/{w}")[0], 200)
        with open(self.store) as f:
            lines = f.read().splitlines()
        self.assertEqual([l for l in lines[1:] if '"weights"' in l], [])   # the marker left with the note


    def test_mark_needs_a_learned_fact(self):
        # a mark true retires the note from every block, so the pocket's own graph must confirm it: nothing taught,
        # no mark, and the note keeps riding the prompt.
        srv, _notes = self.build()                                  # the graph is empty: nothing taught
        port = srv.server_address[1]
        nid = _json(port, "POST", "/v1/notes", {"text": "My uncle Bertil sails a ketch."})[1]["id"]
        q = {"model": "m", "messages": [{"role": "user", "content": "What does my uncle Bertil do?"}]}
        self.assertIn(nid, _json(port, "POST", "/v1/chat/completions", q)[1]["notes"]["ids"])
        st, b = _json(port, "POST", f"/v1/notes/{nid}/weights", {"in_weights": True})
        self.assertEqual(st, 409)                                   # a claim the pocket never earned
        self.assertIn("learned fact", b["error"]["message"])
        self.assertIn(nid, _json(port, "POST", "/v1/chat/completions", q)[1]["notes"]["ids"])   # still in the prompt
        self.assertEqual(_json(port, "POST", f"/v1/notes/{nid}/weights", {"in_weights": False})[0], 200)   # open: needs no proof
        # a fact the pocket holds but has not learned is no proof either
        write_graph(self.model, [{"id": 1, "text": "My uncle Bertil sails a ketch.", "state": "missed",
                                  "neurons": [], "questions": []}])
        self.assertEqual(_json(port, "POST", f"/v1/notes/{nid}/weights", {"in_weights": True})[0], 409)
        # now teach it: the mark rides on the learned fact, and the note leaves the block
        write_graph(self.model, [{"id": 1, "text": "My uncle Bertil sails a ketch.", "state": "learned",
                                  "neurons": [{"i": 2, "w": 0.9}], "questions": []}])
        self.assertEqual(_json(port, "POST", f"/v1/notes/{nid}/weights", {"in_weights": True})[0], 200)
        self.assertNotIn(nid, _json(port, "POST", "/v1/chat/completions", q)[1]["notes"]["ids"])
        # a near miss is not the sentence: the learned text must equal the note's text exactly
        nid2 = _json(port, "POST", "/v1/notes", {"text": "My uncle Bertil sails a ketch too."})[1]["id"]
        self.assertEqual(_json(port, "POST", f"/v1/notes/{nid2}/weights", {"in_weights": True})[0], 409)


class SharedPocket(unittest.TestCase):
    def setUp(self):
        self.tmp = H.TempDir()
        self.addCleanup(self.tmp.cleanup)

    def test_slide_with_many_users_stops_at_start(self):
        # one pocket for all users: what Alice teaches lands in Bob's answers, and no flag allows it
        stores = self.tmp.file("stores")
        os.makedirs(stores)
        users_path = self.tmp.file("users.json")
        users.add_user(users_path, "alice", store_dir=stores)
        model = self.tmp.file("model")
        os.makedirs(model, exist_ok=True)
        base = ["--tensorfold", model, "--tensorfold-bin", FAKE, "--users", users_path, "--store-dir", stores,
                "--share-upstream-cache", "--port", "0"]
        for slide in ("--tensorfold-arg=--slide", "--tensorfold-arg=--slide-graph=/tmp/g.json"):
            args = proxy.parser().parse_args(base + [slide])
            err = proxy.usage_error(args)
            self.assertIn("pocket", err)
            with self.assertRaises(SystemExit) as cm:
                proxy.build(args, log=lambda *_: None)
            self.assertIn("one proxy per owner", str(cm.exception))
        # control: the same proxy without the pocket starts (the shared prompt cache is allowed here)
        args = proxy.parser().parse_args(base + ["--tensorfold-arg=--no-drafts"])
        self.assertIsNone(proxy.usage_error(args))


class Graph(ChildBase):
    def test_graph_annotated_and_slide_routes_refused(self):
        model = self.tmp.file("model")
        write_graph(model, [{"id": 1, "text": "My uncle Bertil sails a ketch.", "state": "learned",
                             "neurons": [], "questions": []},
                            {"id": 2, "text": "A sentence no note holds.", "state": "missed",
                             "neurons": [], "questions": []}])
        srv, _notes, port = self.running(["--owner", "user"])
        w = _json(port, "POST", "/v1/notes", {"text": "My uncle Bertil sails a ketch."})[1]["id"]
        self.assertEqual(_json(port, "POST", f"/v1/notes/{w}/weights", {"in_weights": True})[0], 200)
        st, g = _json(port, "GET", "/v1/slide/graph")
        self.assertEqual(st, 200)
        f1, f2 = g["facts"]
        self.assertEqual((f1["note"], f1["in_weights"]), (w, True))     # mapped by the taught sentence
        self.assertIsNone(f2["note"])                                   # a fact no note holds names no note
        # no client teaches or deletes in the weights through the proxy, at any age
        self.assertEqual(_json(port, "POST", "/v1/slide/learn", {"text": "I like blue."})[0], 404)
        self.assertEqual(_json(port, "DELETE", "/v1/slide/fact/1")[0], 404)
        self.assertEqual(_json(port, "DELETE", "/v1/slide/graph")[0], 404)

    def test_graph_refused_with_many_users(self):
        # one pocket, many users: the graph is the whole pocket — every owner's taught sentence and question —
        # so with many users the route is refused outright, and no user reads another's taught sentences.
        stores = self.tmp.file("stores")
        os.makedirs(stores)
        model = self.tmp.file("model")
        write_graph(model, [{"id": 7, "text": "My sister Anna drives a red Volvo.", "state": "learned",
                             "neurons": [{"i": 3, "w": 0.9}], "questions": ["Who drives a red Volvo?"]},
                            {"id": 8, "text": "Alice keeps a boat in Turku.", "state": "learned",
                             "neurons": [{"i": 9, "w": 0.8}], "questions": ["Where does Alice keep her boat?"]}])
        users_path = self.tmp.file("users.json")
        ka = users.add_user(users_path, "alice", store_dir=stores)
        kb = users.add_user(users_path, "bob", store_dir=stores)
        srv, _notes, port = self.running(["--users", users_path, "--store-dir", stores, "--share-upstream-cache"])
        ha = {"Authorization": f"Bearer {ka}"}
        hb = {"Authorization": f"Bearer {kb}"}
        text = "My sister Anna drives a red Volvo."
        qa = json.loads(_call(port, "POST", "/v1/notes", {"text": text}, ha)[2])["id"]
        _call(port, "POST", "/v1/notes", {"text": "Alice keeps a boat in Turku."}, ha)
        st, _h, raw = _call(port, "GET", "/v1/slide/graph", None, hb)
        self.assertEqual(st, 404)                                     # bob: the route does not exist
        self.assertNotIn(text.encode(), raw)                          # control: his reply names none of it
        self.assertNotIn(b"Turku", raw)
        self.assertNotIn(b"Anna", raw)
        st, _h = _call(port, "GET", "/v1/slide/graph", None, ha)[:2]
        self.assertEqual(st, 404)                                     # not even alice, her own teacher
        self.assertEqual(_call(port, "HEAD", "/v1/slide/graph", None, hb)[0], 404)
        # the notes routes stay per owner: bob's own block never holds alice's note
        bob = json.loads(_call(port, "GET", "/v1/notes", None, hb)[2])["data"]
        self.assertNotIn(qa, [n["id"] for n in bob])

    def test_graph_without_engine_404(self):
        store = self.tmp.file("plain.jsonl")
        p = H.Running("http://127.0.0.1:1", store, owner="user")   # no engine: the route does not exist
        self.addCleanup(p.stop)
        self.assertEqual(p.json("GET", "/v1/slide/graph")[0], 404)
        self.assertEqual(p.json("POST", "/v1/notes/w1/weights", {"in_weights": True})[0], 404)


class Allowlist(unittest.TestCase):
    def test_slide_flags_allowed(self):
        from notes_proxy import tensorfold as tfchild
        self.assertEqual(tfchild.check_extra(["--slide", "--slide-graph=/tmp/g.json"]),
                         ["--slide", "--slide-graph", "/tmp/g.json"])
        self.assertEqual(tfchild.check_extra(["--slide", "--slide-graph", "/tmp/g.json"]),
                         ["--slide", "--slide-graph", "/tmp/g.json"])


if __name__ == "__main__":
    unittest.main()
