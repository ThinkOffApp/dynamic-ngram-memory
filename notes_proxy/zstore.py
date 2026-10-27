"""Serialized ctypes ownership of the standalone Zig store (not yet the HTTP store adapter).

Build module/zig with the pinned Zig first, or set NOTES_PROXY_STORE_LIBRARY to that build's library.
The caller supplies a snapshot and its exact tokenizer's encode function. No fallback tokenizer is used.
"""
import ctypes as C
import json
import os
import pathlib
import sys
import threading

Encode = C.CFUNCTYPE(C.c_ssize_t, C.c_void_p, C.c_void_p, C.c_size_t,
                    C.POINTER(C.c_uint32), C.c_size_t)


class NativeStoreError(RuntimeError):
    pass


class ZStore:
    """One handle per owner. Explicitly close it (or use a context manager)."""

    def __init__(self, snapshot, journal, encode, library=None):
        self._lock = threading.RLock()
        self._handle = None
        suffix = "dylib" if sys.platform == "darwin" else "so"
        default = pathlib.Path(__file__).resolve().parents[1] / "module/zig/zig-out/lib" / (
            "libnotes_store." + suffix)
        self._lib = C.CDLL(str(library or os.environ.get("NOTES_PROXY_STORE_LIBRARY") or default))
        lib = self._lib
        lib.np_open.argtypes = [C.c_char_p, C.c_size_t, C.c_char_p, C.c_size_t,
                               C.c_uint32, Encode, C.c_void_p]
        lib.np_open.restype = C.c_void_p
        lib.np_close.argtypes = [C.c_void_p]
        lib.np_close.restype = None
        lib.np_last_error.argtypes = []
        lib.np_last_error.restype = C.c_char_p
        lib.np_write.argtypes = [C.c_void_p, C.c_char_p, C.c_size_t, C.c_int64]
        lib.np_write.restype = C.c_char_p
        lib.np_list.argtypes = [C.c_void_p]
        lib.np_list.restype = C.c_char_p
        lib.np_erase.argtypes = [C.c_void_p, C.c_uint32]
        lib.np_erase.restype = C.c_char_p
        lib.np_lookup.argtypes = [C.c_void_p, C.c_char_p, C.c_size_t, C.c_uint32, C.c_bool,
                                 C.POINTER(C.c_uint32), C.c_size_t]
        lib.np_lookup.restype = C.c_char_p

        def tokenize(_, text, size, out, capacity):
            try:
                ids = encode(C.string_at(text, size).decode("utf-8"))
                if out:
                    if len(ids) > capacity:
                        return -1
                    for k, token in enumerate(ids):
                        if isinstance(token, bool) or not isinstance(token, int) or not 0 <= token < vocabulary:
                            return -1
                        out[k] = token
                return len(ids)
            except Exception:
                return -1  # fail closed; do not log tokenizer exceptions containing note text

        if isinstance(snapshot, bytes):
            raw = snapshot
            vocabulary = json.loads(raw)["vocab"]["size"]
        else:
            vocabulary = snapshot["vocab"]["size"]
            raw = json.dumps(snapshot, ensure_ascii=False, separators=(",", ":")).encode()
        if isinstance(vocabulary, bool) or not isinstance(vocabulary, int) or not 0 < vocabulary < (1 << 31):
            raise ValueError("invalid snapshot vocabulary size")
        path = os.fsencode(journal)
        if b"\0" in path:
            raise ValueError("journal path contains NUL")
        self._callback = Encode(tokenize)  # its lifetime must exceed the native handle's
        self._handle = lib.np_open(raw, len(raw), path, len(path), vocabulary, self._callback, None)
        if not self._handle:
            self._raise()

    def _raise(self):
        raise NativeStoreError((self._lib.np_last_error() or b"native store failure").decode("ascii", "replace"))

    def _call(self, function, *args):
        with self._lock:
            if self._handle is None:
                raise NativeStoreError("store is closed")
            result = function(self._handle, *args)
            if result is None:
                self._raise()
            return json.loads(result)  # decode under the lock, before the next native arena reset

    def write(self, text, told):
        raw = text.encode("utf-8")
        return self._call(self._lib.np_write, raw, len(raw), told)

    def listing(self):
        return self._call(self._lib.np_list)

    def erase(self, note_id):
        if isinstance(note_id, bool) or not isinstance(note_id, int) or not 0 < note_id < (1 << 32):
            raise ValueError("invalid native note id")
        return self._call(self._lib.np_erase, note_id)

    def lookup(self, question, cap=512, past=False, exclude=()):
        if isinstance(cap, bool) or not isinstance(cap, int) or not 0 <= cap < (1 << 32):
            raise ValueError("invalid token cap")
        if any(isinstance(i, bool) or not isinstance(i, int) or not 0 < i < (1 << 32) for i in exclude):
            raise ValueError("invalid excluded note id")
        raw = question.encode("utf-8")
        ids = (C.c_uint32 * len(exclude))(*exclude)
        return self._call(self._lib.np_lookup, raw, len(raw), cap, bool(past), ids, len(ids))

    def close(self):
        with self._lock:
            if self._handle is not None:
                self._lib.np_close(self._handle)
                self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
