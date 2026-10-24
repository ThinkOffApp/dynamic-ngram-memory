"""Many users: no client can name or share another user's state on the upstream, nor pick whose notes it gets.

The fields and headers come from the table under "What reaches the upstream" in the README, so the test checks the
code against what the README promises."""
import json
import os
import re
import threading
import unittest
import unittest.mock
from http.server import BaseHTTPRequestHandler

from notes_proxy import proxy
from notes_proxy.tests import helpers as H
from notes_proxy.tests import test_many_safety as TS
from notes_proxy.tests import test_users as TU

CHAT = "/v1/chat/completions"
SECRET = TS.SECRET
QUESTION = "Where is my island?"
# every field and header the review asked about: the README's table must name each one
ASKED = {("id_slot", "body"), ("slot_id", "body"), ("cache_prompt", "body"), ("cache_salt", "body"),
         ("request_id", "body"), ("user", "body"), ("prompt_cache_key", "body"), ("session_id", "body"),
         ("conversation", "body"), ("previous_response_id", "body"), ("owner", "body"), ("notes_owner", "body"),
         ("X-Session-Id", "header"), ("X-Request-Id", "header"), ("OpenAI-Organization", "header"),
         ("OpenAI-Project", "header"), ("Cookie", "header"), ("api-key", "header"), ("X-Owner", "header"),
         ("X-Notes-Owner", "header"), ("X-User", "header")}
VALUES = {"cache_prompt": True, "id_slot": 0, "slot_id": 0, "session_params": {"id": "same-session"},
          "kv_transfer_params": {"remote_request_id": "same-request", "remote_block_ids": [0, 1]}}
# with many users, the only headers the upstream gets: the three client headers and the proxy's own
ALLOWED = {"content-type", "accept", "user-agent", "host", "content-length", "accept-encoding", "authorization"}


def table():
    """{(name, "body" or "header"): the README's first word for what happens to it}, from its table in "What reaches
    the upstream"."""
    with open(TS.README) as f:
        text = f.read()
    rows = text.split("| Field | In | With many users |\n| --- | --- | --- |\n", 1)[1].split("\n\n", 1)[0]
    out = {}
    for row in rows.splitlines():
        names, where, what = [c.strip() for c in row.strip().strip("|").split("|")]
        for name in re.findall(r"`([^`]+)`", names):
            out[(name, where)] = re.match(r"[a-z]+", what).group()
    return out


def value(name):
    return VALUES.get(name, f"same-{name.lower()}")


# ============================================================================================ an upstream with state
class StateStub:
    """An upstream that keeps state by every name a client can send. Each field and header of the README's table that
    a request carries is a name. So is the prompt cache when a request asks for it (cache_prompt true, or a cache_salt
    without cache_prompt false), named by its cache_salt. Each request's prompt is kept under all its names, and the
    reply lists every prompt kept before under any of them ("seen"). So a request that names what another user's
    request named gets that user's prompt back."""

    def __init__(self, names):
        self.kept = {}
        self.got = []                    # (method, path, headers, body) of every request
        self.lock = threading.Lock()
        body_names = [n for n, where in names if where == "body" and n not in ("cache_prompt", "cache_salt")]
        header_names = [n.lower() for n, where in names if where == "header"]
        stub = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                headers = {k.lower(): v for k, v in self.headers.items()}
                prompt = json.dumps(body.get("messages") or body.get("prompt") or body.get("input"))
                keys = [("body", n, json.dumps(body[n], sort_keys=True)) for n in body_names if n in body]
                keys += [("header", n, headers[n]) for n in header_names if n in headers]
                if body.get("cache_prompt") is True or "cache_salt" in body and body.get("cache_prompt") is not False:
                    keys.append(("cache", body.get("cache_salt")))
                with stub.lock:
                    stub.got.append((self.command, self.path, headers, body))
                    seen = sorted({p for k in keys for p in stub.kept.get(k, ())})
                    for k in keys:
                        stub.kept.setdefault(k, []).append(prompt)
                data = json.dumps({"object": "chat.completion", "model": body.get("model"), "seen": seen,
                                   "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}]})
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data.encode())

        self.srv = H.Server(("127.0.0.1", 0), Handler)
        self.srv.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def stop(self):
        self.srv.shutdown()
        self.srv.server_close()


class SharedState(unittest.TestCase):
    """alice and bob send, at the same moment, requests that carry the same value for one field or header."""
    ROUNDS = 3

    def setUp(self):
        self.tmp = H.TempDir()
        self.addCleanup(self.tmp.cleanup)
        self.team = TU.Team(self.tmp, "team")
        self.A, self.B = TS.auth(self.team.keys["alice"]), TS.auth(self.team.keys["bob"])
        self.table = table()

    def stub(self):
        s = StateStub(self.table)
        self.addCleanup(s.stop)
        return s

    def exchange(self, alices, bobs, a_auth, b_auth):
        """For every field of the table and every round, alice (her chat, notes on) and bob (a chat, a completion or an
        embedding, by round) send at once, each carrying that field with the same value. The model field names the
        sender, the field and the round, for the stub's records. Returns {(who, name, where, round): (status,
        reply)}."""
        out = {}
        gate = threading.Barrier(2)

        def send(who, p, auth, name, where, rnd):
            body = {"model": f"m|{who}|{name}|{where}|{rnd}"}
            route = CHAT if who == "alice" else (CHAT, "/v1/completions", "/v1/embeddings")[rnd % 3]
            if route == CHAT:
                body["messages"] = [{"role": "user", "content": QUESTION}]
            else:
                body["prompt" if route == "/v1/completions" else "input"] = "hello"
            headers = dict(auth)
            if where == "body":
                body[name] = value(name)
            else:
                headers[name] = value(name)
            gate.wait()
            st, _h, raw = p.call("POST", route, body, headers=headers)
            out[(who, name, where, rnd)] = (st, raw)

        for rnd in range(self.ROUNDS):
            for name, where in sorted(self.table):
                ts = [threading.Thread(target=send, args=("alice", alices, a_auth, name, where, rnd)),
                      threading.Thread(target=send, args=("bob", bobs, b_auth, name, where, rnd))]
                for t in ts:
                    t.start()
                for t in ts:
                    t.join()
        return out

    def give_alice_her_secret(self, p):
        self.assertEqual(p.json("POST", "/v1/notes", {"text": SECRET}, headers=self.A)[0], 201)

    def test_the_table_names_every_field_asked_about(self):
        self.assertEqual(ASKED - set(self.table), set())
        self.assertEqual({w for w in self.table.values()} - {"set", "replaced", "dropped", "refused"}, set())

    def test_identical_ids_from_two_users_share_nothing_upstream(self):
        up = self.stub()
        p = H.Running(up.url, self.team.stores, users=self.team.users)
        self.addCleanup(p.stop)
        self.give_alice_her_secret(p)
        out = self.exchange(p, p, self.A, self.B)
        for (who, name, where, rnd), (st, raw) in out.items():
            if who == "bob":
                self.assertNotIn(SECRET.encode(), raw, (name, where, rnd))     # bob never gets alice's prompt
        self.assertTrue(any(SECRET in json.dumps(b) for _m, _p, _h, b in up.got))   # control: hers held it
        got = {}
        for _m, _p, headers, body in up.got:
            _m, who, name, where, rnd = body["model"].split("|")
            got[(who, name, where, int(rnd))] = (headers, body)
        for (name, where), what in sorted(self.table.items()):
            with self.subTest(field=name, where=where, what=what):
                sent = [k for k in out if k[1:3] == (name, where)]
                self.assertEqual(len(sent), 2 * self.ROUNDS)
                if what == "refused":
                    for k in sent:
                        st, raw = out[k]
                        self.assertEqual(st, 400, k)
                        self.assertIn(name, json.loads(raw)["error"]["message"])
                        self.assertNotIn(k, got)                                # the upstream got nothing of it
                    continue
                for k in sent:
                    self.assertEqual(out[k][0], 200, (k, out[k][1][:300]))
                    self.assertIn(k, got)
                if where == "header":
                    self.assertEqual(what, "dropped")
                    for k in sent:
                        self.assertNotIn(name.lower(), got[k][0])
                    continue
                seen = {(k[0], json.dumps(got[k][1].get(name, "(none)"))) for k in sent}
                if what == "dropped":
                    self.assertEqual({v for _w, v in seen}, {'"(none)"'})
                elif what == "set":
                    self.assertEqual({v for _w, v in seen}, {"false"})
                else:                                                           # replaced: one value per user
                    mine = {w: {v for who, v in seen if who == w} for w in ("alice", "bob")}
                    self.assertEqual([len(mine["alice"]), len(mine["bob"])], [1, 1], mine)
                    self.assertNotEqual(mine["alice"], mine["bob"])
                    self.assertNotIn(json.dumps(value(name)), mine["alice"] | mine["bob"])
        print(f"\n[test_many_state] alice and bob sent the same value for {len(self.table)} fields and headers, at "
              f"once, {self.ROUNDS} rounds: bob got nothing of alice's from the upstream")

    def test_control_two_one_owner_proxies_share_the_upstreams_state(self):
        up = self.stub()
        p = H.Running(up.url, self.team.stores, users=self.team.users)
        self.addCleanup(p.stop)
        self.give_alice_her_secret(p)
        p.stop()
        alice = H.Running(up.url, self.team.journal("alice"), owner="alice")
        bob = H.Running(up.url, self.team.journal("bob"), owner="bob")
        self.addCleanup(alice.stop)
        self.addCleanup(bob.stop)
        out = self.exchange(alice, bob, {}, {})
        leaked = {(name, where) for (who, name, where, rnd), (st, raw) in out.items()
                  if who == "bob" and SECRET.encode() in raw}
        # every field and header is a way through this stub, so the check above could fail for each one
        self.assertEqual(set(self.table) - leaked, set())
        print(f"[test_many_state] control, two one-owner proxies in front of the same stub: bob got alice's secret "
              f"through {len(leaked)} of {len(self.table)} fields and headers")


# ============================================================================================ the owner
class HonoursXOwner(proxy.Handler):
    """A broken proxy, for the control: it takes the owner from X-Owner."""

    @property
    def notes(self):
        if self._notes is None:
            self._notes = self.server.notes_of(self.headers.get("X-Owner") or self.owner)
        return self._notes


class Owner(unittest.TestCase):
    """bob names alice as the owner in every way but her key."""
    FIELDS = {"owner": "alice", "notes_owner": "alice", "user": "alice"}
    HEADERS = {"X-Owner": "alice", "X-Notes-Owner": "alice", "X-User": "alice"}

    def setUp(self):
        self.tmp = H.TempDir()
        self.addCleanup(self.tmp.cleanup)
        self.team = TU.Team(self.tmp, "team")
        self.A, self.B = TS.auth(self.team.keys["alice"]), TS.auth(self.team.keys["bob"])
        self.stub = H.Stub()
        self.stub.release.set()
        self.addCleanup(self.stub.stop)

    def proxy(self, handler=None):
        p = H.Running(self.stub.url, self.team.stores, users=self.team.users, extra=["--write"], handler=handler)
        self.addCleanup(p.stop)
        st, w = p.json("POST", "/v1/notes", {"text": SECRET}, headers=self.A)
        self.assertEqual(st, 201)
        return p, w["id"]

    def test_only_the_key_names_the_owner(self):
        p, sid = self.proxy()
        hers = self.team.bytes_of("alice")
        n = len(self.stub.requests)
        replies = []
        for name, v in self.FIELDS.items():                     # in the body
            for path, body in ((CHAT, TU.ask(QUESTION, **{name: v})),
                               (CHAT, TU.ask(QUESTION, stream=True, **{name: v})),
                               ("/v1/completions", {"model": "m", "prompt": "hi", name: v})):
                st, _h, raw = p.call("POST", path, body, headers=self.B)
                replies.append(raw)
                if name == "user":                              # the stub's own status: it has no /v1/completions
                    self.assertEqual(st, 200 if path == CHAT else 404, raw)
                else:
                    self.assertEqual(st, 400, (name, path))
                    self.assertIn(name, json.loads(raw)["error"]["message"])
        for name, v in self.HEADERS.items():                    # in a header
            h = dict(self.B, **{name: v})
            st, _h, raw = p.call("POST", CHAT, TU.ask(QUESTION), headers=h)
            self.assertEqual(st, 200)
            replies.append(raw)
            st, v2 = p.json("GET", "/v1/notes?owner=alice", headers=h)
            self.assertEqual((st, v2["owner"]), (200, "bob"))
            replies.append(json.dumps(v2).encode())
        every = dict(self.B, **self.HEADERS)
        st, r = p.json("POST", CHAT, TU.ask("My secret island is Madeira.", user="alice"), headers=every)
        self.assertEqual(r["notes"]["write"]["verdict"], "NEW")  # bob's own statement, written to bob
        st, _h, raw = p.call("POST", "/v1/notes", {"text": "My cat is Mau.", "owner": "alice"}, headers=every)
        self.assertEqual(st, 400)
        self.assertEqual(p.json("POST", "/v1/notes", {"text": "My cat is Mau."}, headers=every)[0], 201)
        st, _h, raw = p.call("GET", f"/v1/notes/{sid}", headers=every)
        replies.append(raw)
        for raw in replies:
            self.assertNotIn(SECRET.encode(), raw)            # bob never gets alice's note
        self.assertEqual(self.team.bytes_of("alice"), hers)   # and nothing of his reaches her store
        self.assertIn(b"Madeira", self.team.bytes_of("bob"))
        self.assertIn(b"Mau", self.team.bytes_of("bob"))
        for r in self.stub.requests[n:]:                      # the upstream never got a name for the owner
            body = json.loads(r["body"]) if r["body"] else {}
            self.assertFalse({"owner", "notes_owner"} & set(body), body)
            self.assertNotEqual(body.get("user"), "alice")
            self.assertFalse({k.lower() for k in self.HEADERS} & {k.lower() for k in r["headers"]}, r["headers"])
        self.assertEqual(p.json("GET", "/v1/notes", headers=self.A)[1]["data"][0]["text"], SECRET)

    def test_control_a_proxy_that_read_x_owner_would_give_bob_her_note(self):
        p, _sid = self.proxy(handler=HonoursXOwner)
        st, _h, raw = p.call("POST", CHAT, TU.ask(QUESTION), headers=dict(self.B, **{"X-Owner": "alice"}))
        self.assertEqual(st, 200)
        self.assertIn(SECRET.encode(), raw)


# ============================================================================================ headers
class Headers(unittest.TestCase):
    TENANT = {"Cookie": "session=bob", "Proxy-Authorization": "Basic Ym9iOnB3", "api-key": "sk-bob",
              "X-Api-Key": "sk-bob", "Ocp-Apim-Subscription-Key": "sk-bob", "X-Goog-Api-Key": "sk-bob",
              "OpenAI-Organization": "org-bob", "OpenAI-Project": "proj-bob", "X-Session-Id": "s-1",
              "X-Request-Id": "r-1", "X-Owner": "alice", "X-Notes-Owner": "alice", "X-User": "alice",
              "X-Forwarded-For": "10.0.0.1", "Forwarded": "for=10.0.0.1", "X-Anything": "1"}
    KEPT = {"User-Agent": "test-client/1", "Accept": "application/json"}

    def setUp(self):
        self.tmp = H.TempDir()
        self.addCleanup(self.tmp.cleanup)
        self.team = TU.Team(self.tmp, "team")
        self.stub = H.Stub()
        self.stub.release.set()
        self.addCleanup(self.stub.stop)

    def requests(self, p, auth):
        """Every route bob can send, with every header; what the stub got from them."""
        n = len(self.stub.requests)
        h = dict(auth, **self.TENANT, **self.KEPT)
        for method, path, body in (("POST", CHAT, TU.ask(QUESTION)), ("POST", CHAT, TU.ask(QUESTION, stream=True)),
                                   ("POST", CHAT, TU.ask("hi", notes=False)), ("GET", "/v1/models", None),
                                   ("POST", "/v1/completions", {"prompt": "hi"}),
                                   ("POST", "/v1/embeddings", {"input": "hi"})):
            p.call(method, path, body, headers=h)
        got = self.stub.requests[n:]
        self.assertEqual(len(got), 6)
        return got

    def test_only_three_client_headers_reach_the_upstream(self):
        with unittest.mock.patch.dict(os.environ, {TU.ENV: TU.UPSTREAM_KEY}):
            p = H.Running(self.stub.url, self.team.stores, users=self.team.users, extra=["--upstream-key-env", TU.ENV])
        self.addCleanup(p.stop)
        for r in self.requests(p, TS.auth(self.team.keys["bob"])):
            self.assertEqual({k.lower() for k in r["headers"]} - ALLOWED, set(), (r["method"], r["path"]))
            for k, v in self.KEPT.items():                    # control: the stub does record the headers that pass
                self.assertEqual(r["headers"].get(k), v)
            self.assertEqual(r["headers"].get("Authorization"), f"Bearer {TU.UPSTREAM_KEY}")

    def test_control_one_owner_passes_the_headers_on_as_before(self):
        p = H.Running(self.stub.url, self.team.journal("bob"), owner="bob")
        self.addCleanup(p.stop)
        for r in self.requests(p, {"Authorization": "Bearer sk-client"}):
            names = {k.lower() for k in r["headers"]}
            want = {k.lower() for k in self.TENANT} - {"proxy-authorization"}   # a hop-by-hop header, in both modes
            self.assertEqual(want - names, set(), (r["method"], r["path"]))
            self.assertNotIn("proxy-authorization", names)
            self.assertEqual(r["headers"].get("Authorization"), "Bearer sk-client")


if __name__ == "__main__":
    unittest.main()
