"""Experimental single-owner native HTTP smoke gate; needs the built library and real tokenizer."""
import json
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
