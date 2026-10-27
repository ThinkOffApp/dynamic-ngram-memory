"""Explicit C ABI smoke tests: build module/zig first. Byte tokenizer is fixture-only."""
import ctypes as C
import json
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
ENCODE = C.CFUNCTYPE(C.c_ssize_t, C.c_void_p, C.c_void_p, C.c_size_t,
                    C.POINTER(C.c_uint32), C.c_size_t)


class CApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        suffix = "dylib" if sys.platform == "darwin" else "so"
        cls.lib = C.CDLL(str(ROOT / "zig-out/lib" / ("libnotes_store." + suffix)))
        cls.lib.np_open.argtypes = [C.c_char_p, C.c_size_t, C.c_char_p, C.c_size_t,
                                    C.c_uint32, ENCODE, C.c_void_p]
        cls.lib.np_open.restype = C.c_void_p
        cls.lib.np_close.argtypes = [C.c_void_p]
        cls.lib.np_last_error.restype = C.c_char_p
        cls.lib.np_write.argtypes = [C.c_void_p, C.c_char_p, C.c_size_t, C.c_int64]
        cls.lib.np_write.restype = C.c_char_p
        cls.lib.np_list.argtypes = [C.c_void_p]
        cls.lib.np_list.restype = C.c_char_p
        cls.lib.np_erase.argtypes = [C.c_void_p, C.c_uint32]
        cls.lib.np_erase.restype = C.c_char_p
        cls.lib.np_lookup.argtypes = [C.c_void_p, C.c_char_p, C.c_size_t, C.c_uint32, C.c_bool,
                                      C.POINTER(C.c_uint32), C.c_size_t]
        cls.lib.np_lookup.restype = C.c_char_p
        cls.snapshot = (ROOT / "src/server/notes/testdata/tb_empty_bytes_snapshot.json").read_bytes()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="notes-cabi-")
        self.addCleanup(self.tmp.cleanup)
        self.path = str(pathlib.Path(self.tmp.name) / "journal").encode()
        self.handles = []
        self.addCleanup(self.close_all)

        def byte_encode(_, text, length, out, capacity):
            data = C.string_at(text, length)
            if out:
                if capacity < len(data):
                    return -1
                for k, byte in enumerate(data):
                    out[k] = byte
            return len(data)
        self.callback = ENCODE(byte_encode)

    def close_all(self):
        for handle in self.handles:
            self.lib.np_close(handle)
        self.handles.clear()

    def open(self, callback=None):
        handle = self.lib.np_open(self.snapshot, len(self.snapshot), self.path,
                                  len(self.path), 256, callback or self.callback, None)
        self.assertTrue(handle, self.lib.np_last_error())
        self.handles.append(handle)
        return handle

    def write(self, handle, text):
        data = text.encode()
        result = self.lib.np_write(handle, data, len(data), 739900000000)
        self.assertIsNotNone(result, self.lib.np_last_error())
        return json.loads(result)

    def test_durable_write_erase_reopen_preserves_other_note(self):
        handle = self.open()
        first = self.write(handle, "My aunt Brigitta grows tomatoes.")["note"]["id"]
        other = self.write(handle, "My cousin Aurelio repairs clocks.")["note"]["id"]
        question = b"What does my aunt Brigitta grow?"
        found = json.loads(self.lib.np_lookup(handle, question, len(question), 1000, False, None, 0))
        self.assertIn(first, found["ids"])
        self.assertIn("Brigitta", found["block"])
        result = json.loads(self.lib.np_erase(handle, first))
        self.assertFalse(result["complete"])
        self.assertEqual(result["remaining"], "upstream")
        self.close_all()
        handle = self.open()
        rows = json.loads(self.lib.np_list(handle))
        self.assertEqual([(r["id"], r["text"]) for r in rows],
                         [(other, "My cousin Aurelio repairs clocks.")])
        self.assertNotIn(b"Brigitta", pathlib.Path(self.path.decode()).read_bytes())
        found = json.loads(self.lib.np_lookup(handle, question, len(question), 1000, False, None, 0))
        self.assertNotIn(first, found["ids"])
        self.assertNotIn("Brigitta", found["block"])

    def test_tokenizer_failure_does_not_commit_a_write(self):
        failing = ENCODE(lambda *_: -1)
        handle = self.open(failing)
        self.assertIsNone(self.lib.np_write(handle, b"A fact.", 7, 739900000000))
        self.assertTrue(self.lib.np_last_error())
        self.assertEqual(json.loads(self.lib.np_list(handle)), [])

    def test_unsafe_journal_refused_after_close(self):
        self.write(self.open(), "My aunt Brigitta grows tomatoes.")
        self.close_all()
        pathlib.Path(self.path.decode()).chmod(0o644)
        handle = self.lib.np_open(self.snapshot, len(self.snapshot), self.path,
                                  len(self.path), 256, self.callback, None)
        self.assertFalse(handle)
        self.assertEqual(self.lib.np_last_error(), b"UnsafeJournal")


if __name__ == "__main__":
    unittest.main()
