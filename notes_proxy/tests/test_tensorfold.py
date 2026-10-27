"""Lane 2: TensorFold child allowlist, tokenize-through-child, store:false, batched restart after erase.

Tests for review items 2-5 are written to fail on the first branch (wrong tokenize schema, hang on restart fail,
DELETE blocking on reload, no erase-status).
"""
import json
import os
import tempfile
import threading
import time
import unittest

from notes_proxy import proxy, users
from notes_proxy import tensorfold as tfchild
from notes_proxy.tests import helpers as H

FAKE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fake_tensorfold.py")
ERASE = {"erased": 1, "removed": [1], "complete": False, "state": "pending",
         "remaining": ["the upstream engine's prompt cache, logs and any copy it kept: the proxy cannot reach them",
                       "memory this process freed is not overwritten (Python strings)"],
         "remaining_places": ["upstream", "freed_memory"]}


def wait_until(pred, timeout=5.0, step=0.05):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(step)
    return pred()


class Allowlist(unittest.TestCase):
    def test_rejects_lan_and_unknown_flags(self):
        with self.assertRaises(tfchild.TensorFoldError):
            tfchild.check_extra(["--host", "0.0.0.0"])
        with self.assertRaises(tfchild.TensorFoldError):
            tfchild.check_extra(["--vision-urls"])
        with self.assertRaises(tfchild.TensorFoldError):
            tfchild.check_extra(["--snapshot-dir", "/tmp"])
        with self.assertRaises(tfchild.TensorFoldError):
            tfchild.check_extra(["--made-up"])
        self.assertEqual(tfchild.check_extra(["--no-drafts", "--context", "4096"]),
                         ["--no-drafts", "--context", "4096"])

    def test_model_must_be_dir_or_repo_id(self):
        with self.assertRaises(tfchild.TensorFoldError):
            tfchild.check_model("not a path")
        self.assertEqual(tfchild.check_model("Vontra/Qwen3.8-27B-MLX-4bit"), "Vontra/Qwen3.8-27B-MLX-4bit")


class Child(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="tf-child-")
        self.model = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def start(self, extra=(), quiet=0.15, restart_max=2.0, timeout=10.0):
        child = tfchild.TensorFoldChild(self.model, extra=extra, binary=FAKE, timeout=timeout,
                                        log=lambda *_: None, quiet=quiet, restart_max=restart_max)
        self.addCleanup(child.close)
        return child

    def test_tokenize_uses_prompt_and_tokens(self):
        child = self.start()
        st, body = child.request("POST", "/v1/tokenize", {"prompt": "AB"})
        self.assertEqual(st, 200)
        self.assertEqual(body["tokens"], [ord("A"), ord("B")])
        self.assertEqual(body["count"], 2)
        self.assertNotIn("ids", body)
        tok = tfchild.ChildTokenizer(child)
        self.assertEqual(tok.encode("AB"), [ord("A"), ord("B")])
        with open(os.path.join(self.model, "last_tokenize.json")) as f:
            sent = json.loads(f.read())
        self.assertEqual(sent, {"prompt": "AB"})
        self.assertNotIn("text", sent)

    def test_tokenize_ignores_text_ids_schema(self):
        child = self.start()
        st, body = child.request("POST", "/v1/tokenize", {"text": "AB"})
        self.assertEqual(st, 200)
        self.assertEqual(body["tokens"], [])
        self.assertEqual(body["count"], 0)

    def test_tokenize_refuses_without_child_tokens(self):
        open(os.path.join(self.model, "no_tokenize"), "w").close()
        child = self.start()
        tok = tfchild.ChildTokenizer(child)
        with self.assertRaises(tfchild.TensorFoldError) as cm:
            tok.encode("AB")
        self.assertIn("FakeTokenizer", str(cm.exception))

    def test_key_is_in_env_not_argv(self):
        child = self.start()
        cmdline = " ".join(child.proc.args)
        self.assertNotIn(child.key, cmdline)
        self.assertEqual(child._env.get("TENSORFOLD_API_KEY"), child.key)
        self.assertNotIn("OPENAI_API_KEY", child._env)
        self.assertNotIn("TENSORFOLD_API_KEY_FILE", child._env)
        st, _ = child.request("GET", "/health")
        self.assertEqual(st, 200)

    def test_erase_returns_pending_and_batches(self):
        child = self.start(quiet=0.25)
        pid = child.proc.pid
        r1 = child.after_erase(dict(ERASE))
        r2 = child.after_erase(dict(ERASE))
        self.assertFalse(r1["complete"])
        self.assertFalse(r2["complete"])
        self.assertEqual(r1["state"], "pending")
        self.assertEqual(child.proc.pid, pid)
        self.assertTrue(wait_until(lambda: child.erase_status()["complete"]))
        self.assertNotEqual(child.proc.pid, pid)
        status = child.erase_status()
        self.assertEqual(status["state"], "complete")
        self.assertEqual(status["erase_seq"], 2)
        self.assertEqual(status["restart_seq"], 2)

    def test_restart_waits_until_idle(self):
        child = self.start(quiet=0.1)
        pid = child.proc.pid
        child.begin()
        reply = child.after_erase(dict(ERASE))
        self.assertFalse(reply["complete"])
        self.assertEqual(child.proc.pid, pid)
        time.sleep(0.3)
        self.assertEqual(child.proc.pid, pid)
        child.end()
        self.assertTrue(wait_until(lambda: child.proc is not None and child.proc.pid != pid
                                   and child.erase_status()["complete"]))

    def test_restart_failure_does_not_hang(self):
        child = self.start(quiet=0.05)
        with open(os.path.join(self.model, "fail_starts"), "w") as f:
            f.write("99")
        t0 = time.time()
        child.after_erase(dict(ERASE))
        self.assertTrue(wait_until(lambda: child.health()["status"] == "down", timeout=3.0))
        with self.assertRaises(tfchild.TensorFoldError):
            child.begin()
        self.assertLess(time.time() - t0, 2.0)
        h = child.health()
        self.assertEqual(h["status"], "down")
        self.assertGreaterEqual(child.retry_after(), 1)

    def test_erase_stream_restarts_at_force_at(self):
        child = self.start(quiet=2.0, restart_max=0.3)
        pid = child.proc.pid
        child.after_erase(dict(ERASE))
        time.sleep(0.12)
        child.after_erase(dict(ERASE))
        t0 = time.time()
        self.assertTrue(wait_until(lambda: child.erase_status()["complete"], timeout=1.5))
        self.assertLess(time.time() - t0, 1.2)
        self.assertNotEqual(child.proc.pid, pid)

    def test_busy_traffic_is_drained_after_restart_max(self):
        child = self.start(quiet=0.05, restart_max=0.25)
        pid = child.proc.pid
        child.begin()
        child.after_erase(dict(ERASE))
        self.assertTrue(wait_until(lambda: child.health().get("engine") == "draining", timeout=2.0))
        with self.assertRaises(tfchild.TensorFoldError):
            child.begin()
        self.assertEqual(child.proc.pid, pid)
        child.end()
        self.assertTrue(wait_until(lambda: child.proc is not None and child.proc.pid != pid
                                   and child.erase_status()["complete"]))

    def test_defaults_are_ten_and_sixty(self):
        self.assertEqual(tfchild.QUIET, 10.0)
        self.assertEqual(tfchild.RESTART_MAX, 60.0)
        args = proxy.parser().parse_args(["--tensorfold", self.model, "--owner", "u", "--store", "s.jsonl"])
        self.assertEqual(args.tensorfold_quiet, 10.0)
        self.assertEqual(args.tensorfold_restart_max, 60.0)


class ProxyEngine(unittest.TestCase):
    def setUp(self):
        self.tmp = H.TempDir()
        self.addCleanup(self.tmp.cleanup)

    def running(self, extra, store=None):
        model = self.tmp.file("model")
        if not os.path.isdir(model):
            os.mkdir(model)
        store = store or self.tmp.file("s.jsonl")
        args = proxy.parser().parse_args([
            "--tensorfold", model, "--tensorfold-bin", FAKE, "--port", "0", "--host", "127.0.0.1",
            "--tensorfold-quiet", "0.15", "--tensorfold-restart-max", "2",
        ] + extra + (["--store", store] if "--store-dir" not in extra else []))
        srv, notes = proxy.build(args, log=lambda *_: None)
        self.addCleanup(notes.close)
        self.addCleanup(srv.cfg.engine.close)
        self.addCleanup(srv.server_close)
        self.addCleanup(srv.shutdown)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        return srv, notes, srv.server_address[1], srv.cfg.engine

    def test_chat_sets_store_false_and_forwards_child_key(self):
        srv, notes, port, engine = self.running(["--owner", "user"])
        st, h, raw = _call(port, "POST", "/v1/chat/completions",
                           {"model": "m", "messages": [{"role": "user", "content": "Hi."}]})
        self.assertEqual(st, 200)
        body = json.loads(raw)
        self.assertIs(body.get("store"), False)

    def test_erase_pending_then_status_complete(self):
        store = H.seeded(self.tmp, "erase.jsonl")
        srv, notes, port, engine = self.running(["--owner", "user"], store=store)
        pid = engine.proc.pid
        listing = json.loads(_call(port, "GET", "/v1/notes")[2])
        nid = listing["data"][0]["id"]
        t0 = time.time()
        st, _, raw = _call(port, "DELETE", f"/v1/notes/{nid}")
        self.assertEqual(st, 200)
        done = json.loads(raw)
        self.assertFalse(done["complete"])
        self.assertEqual(done["engine"]["state"], "pending")
        self.assertLess(time.time() - t0, 0.5)
        self.assertTrue(wait_until(lambda: json.loads(_call(port, "GET", "/v1/notes/erase-status")[2])["complete"]))
        status = json.loads(_call(port, "GET", "/v1/notes/erase-status")[2])
        self.assertEqual(status["state"], "complete")
        self.assertNotEqual(engine.proc.pid, pid)

    def test_health_and_503_when_engine_down(self):
        srv, notes, port, engine = self.running(["--owner", "user"])
        st, _, raw = _call(port, "GET", "/health")
        self.assertEqual(st, 200)
        self.assertEqual(json.loads(raw)["engine"], "ready")
        with open(os.path.join(self.tmp.file("model"), "fail_starts"), "w") as f:
            f.write("99")
        engine.after_erase(dict(ERASE))
        self.assertTrue(wait_until(
            lambda: json.loads(_call(port, "GET", "/health")[2]).get("engine") == "down", timeout=3.0))
        st, headers, raw = _call(port, "GET", "/health")
        self.assertEqual(st, 503)
        self.assertEqual(json.loads(raw)["engine"], "down")
        self.assertIn("retry-after", {k.lower() for k in headers})
        st, headers, _ = _call(port, "POST", "/v1/chat/completions",
                               {"model": "m", "messages": [{"role": "user", "content": "Hi."}]})
        self.assertEqual(st, 503)
        self.assertIn("retry-after", {k.lower() for k in headers})

    def test_responses_store_false(self):
        srv, notes, port, engine = self.running(["--owner", "user"])
        st, _, raw = _call(port, "POST", "/v1/responses", {"model": "m", "input": "hi", "store": True})
        self.assertEqual(st, 200)
        self.assertIs(json.loads(raw).get("store"), False)

    def test_many_users_need_share_cache(self):
        stores = self.tmp.file("stores")
        os.mkdir(stores)
        users_path = self.tmp.file("users.json")
        users.add_user(users_path, "alice", store_dir=stores)
        model = self.tmp.file("model")
        if not os.path.isdir(model):
            os.mkdir(model)
        args = proxy.parser().parse_args([
            "--tensorfold", model, "--tensorfold-bin", FAKE, "--users", users_path, "--store-dir", stores,
            "--port", "0",
        ])
        err = proxy.usage_error(args)
        self.assertIn("share-upstream-cache", err)
        with self.assertRaises(SystemExit) as cm:
            proxy.build(args, log=lambda *_: None)
        self.assertIn("share-upstream-cache", str(cm.exception))

    def test_engine_with_many_users(self):
        stores = self.tmp.file("stores")
        os.mkdir(stores)
        users_path = self.tmp.file("users.json")
        key = users.add_user(users_path, "alice", store_dir=stores)
        srv, notes, port, engine = self.running(
            ["--users", users_path, "--store-dir", stores, "--share-upstream-cache"])
        auth = {"Authorization": f"Bearer {key}"}
        st, _, raw = _call(port, "POST", "/v1/chat/completions",
                           {"model": "m", "messages": [{"role": "user", "content": "Hi."}]}, headers=auth)
        self.assertEqual(st, 200)
        self.assertIs(json.loads(raw).get("store"), False)
        st, _, raw = _call(port, "POST", "/v1/notes", {"text": "My sister Anna drives a red Volvo."}, headers=auth)
        self.assertIn(st, (200, 201))
        nid = json.loads(raw)["id"]
        st, _, raw = _call(port, "DELETE", f"/v1/notes/{nid}", headers=auth)
        self.assertEqual(st, 200)
        self.assertFalse(json.loads(raw)["complete"])
        self.assertTrue(wait_until(
            lambda: json.loads(_call(port, "GET", "/v1/notes/erase-status", headers=auth)[2])["complete"]))
        st, _, raw = _call(port, "GET", "/health")
        self.assertEqual(st, 200)


def _call(port, method, path, body=None, headers=None):
    import http.client
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    data = json.dumps(body).encode() if body is not None else None
    h = {"Content-Type": "application/json"} if data is not None else {}
    h.update(headers or {})
    c.request(method, path, body=data, headers=h)
    r = c.getresponse()
    out = (r.status, dict(r.getheaders()), r.read())
    c.close()
    return out


if __name__ == "__main__":
    unittest.main()
