"""Experimental single-owner native HTTP smoke gate; needs the built library and real tokenizer."""
import json
import copy
import os
import pathlib
import sys
import tempfile
import threading
import unittest
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from notes_proxy.proxy import build, parser
from notes_proxy.native_notes import NativeNotes
from notes_proxy.store import Notes, nm, StoreError


class NativeHTTP(unittest.TestCase):
    def test_custom_schema_is_reused_on_reopen_and_bound_to_journal_identity(self):
        tok = nm.load_tokenizer(os.environ["NOTES_PROXY_TEST_TOKENIZER"])
        schema = copy.deepcopy(nm.L.SCHEMA)
        schema["likes"]["multi"] = False
        with tempfile.TemporaryDirectory(prefix="native-schema-") as root:
            path = pathlib.Path(root) / "zig"
            ref = Notes(pathlib.Path(root) / "python", "alice", tok, schema=schema)
            self.addCleanup(ref.close)
            native = NativeNotes(path, "alice", tok, schema=schema)
            try:
                for text in ("I like apples.", "I like cheese."):
                    self.assertEqual(native.write_text(text, today="2026-10-27"),
                                     ref.write_text(text, today="2026-10-27"))
            finally:
                native.close()
            native = NativeNotes(path, "alice", tok, schema=schema)
            try:
                self.assertEqual(native.listing(), ref.listing())
            finally:
                native.close()
            before = path.read_bytes()
            changed = copy.deepcopy(schema)
            changed["likes"]["multi"] = True
            with self.assertRaises(StoreError):
                NativeNotes(path, "alice", tok, schema=changed)
            self.assertEqual(path.read_bytes(), before)

    def test_framing_and_abstain_are_fitted_before_the_token_cap(self):
        tok = nm.load_tokenizer(os.environ["NOTES_PROXY_TEST_TOKENIZER"])
        with tempfile.TemporaryDirectory(prefix="native-framing-") as root:
            for framing, abstain in (("quoted", None), ("firm", None), ("firm", "Say unknown when absent.")):
                with self.subTest(framing=framing, abstain=abstain):
                    stem = framing + str(abstain is not None)
                    opts = dict(framing=framing, abstain_line=abstain)
                    ref = Notes(pathlib.Path(root) / (stem + "-python"), "alice", tok, **opts)
                    native = NativeNotes(pathlib.Path(root) / (stem + "-zig"), "alice", tok, **opts)
                    try:
                        for text in ("My aunt Brigitta grows tomatoes.", "Brigitta owns a blue bicycle."):
                            ref.write_text(text, subject="me")
                            native.write_text(text, subject="me")
                        for cap in (512, 160, 60):
                            expected = ref.lookup("What does Brigitta grow?", cap=cap)
                            actual = native.lookup("What does Brigitta grow?", cap=cap)
                            self.assertEqual(actual["block"], expected["block"])
                            self.assertEqual(actual["tokens"], expected["tokens"])
                    finally:
                        native.close()
                        ref.close()

    def test_invalid_writes_match_python_and_do_not_consume_sources(self):
        tok = nm.load_tokenizer(os.environ["NOTES_PROXY_TEST_TOKENIZER"])
        with tempfile.TemporaryDirectory(prefix="native-validation-") as root:
            reference = Notes(pathlib.Path(root) / "python", "alice", tok)
            native = NativeNotes(pathlib.Path(root) / "zig", "alice", tok)
            try:
                for text, subject in (("   ", None), (None, None), ("x" * 16385, None),
                                      ("bad\ud800", None), ("valid", " spaced"), ("valid", 42),
                                      ("valid", "bad\nsubject")):
                    with self.subTest(text_type=type(text).__name__, subject_type=type(subject).__name__):
                        with self.assertRaises(StoreError) as expected:
                            reference.write_text(text, subject=subject)
                        with self.assertRaises(StoreError) as actual:
                            native.write_text(text, subject=subject)
                        self.assertEqual((str(actual.exception), actual.exception.status),
                                         (str(expected.exception), expected.exception.status))
                self.assertEqual(native.listing(), [])
                self.assertEqual(native.write_text("A valid note.", subject="me")["id"], "w1")
            finally:
                native.close()
                reference.close()

    def test_exact_duplicate_reply_matches_python_without_claiming_another_write(self):
        tok = nm.load_tokenizer(os.environ["NOTES_PROXY_TEST_TOKENIZER"])
        with tempfile.TemporaryDirectory(prefix="native-reply-") as root:
            reference = Notes(pathlib.Path(root) / "python", "alice", tok)
            native = NativeNotes(pathlib.Path(root) / "zig", "alice", tok)
            try:
                for text in ("I live in Helsinki.", "I live in Helsinki."):
                    self.assertEqual(native.write_text(text, today="2026-10-27"),
                                     reference.write_text(text, today="2026-10-27"))
                native.erase("w1")
                self.assertEqual(native.listing(history=True), [])
                self.assertEqual(native.native.records(), [])
            finally:
                native.close()
                reference.close()

    def test_opt_in_write_list_get_erase_and_reopen(self):
        tok = os.environ["NOTES_PROXY_TEST_TOKENIZER"]
        with tempfile.TemporaryDirectory(prefix="native-http-") as root:
            argv = ["--upstream", "http://127.0.0.1:1", "--owner", "alice", "--store",
                    str(pathlib.Path(root) / "native.journal"), "--tokenizer", tok, "--port", "0"]
            self.assertEqual(parser().parse_args(argv).store_engine, "python")
            args = parser().parse_args(argv + ["--store-engine", "zig"])
            server, notes = build(args, log=lambda _: None)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            def request(path="", data=None, method=None):
                req = urllib.request.Request(f"http://127.0.0.1:{server.server_port}/v1/notes" + path,
                    data=json.dumps(data).encode() if data is not None else None, method=method,
                    headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=5) as reply:
                    return json.load(reply)
            try:
                written = request(data={"text": "My aunt Brigitta grows tomatoes.", "subject": "me"})
                self.assertEqual(written["id"], "w1")
                self.assertEqual(request("/w1")["id"], "w1")
                self.assertIn("Brigitta", notes.lookup("What does Brigitta grow?")["block"])
                self.assertEqual(request("/w1", method="DELETE")["erased"], "w1")
                self.assertEqual(notes.listing(), [])
            finally:
                server.shutdown()
                server.server_close()
                thread.join()
                notes.close()
            server, reopened = build(args, log=lambda _: None)
            try:
                self.assertEqual(reopened.listing(), [])
                self.assertEqual(reopened.write_text("My aunt owns a bicycle.", subject="me")["id"], "w2")
            finally:
                server.server_close()
                reopened.close()


if __name__ == "__main__":
    unittest.main()
