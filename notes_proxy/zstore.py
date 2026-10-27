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
    def __init__(self, message, status=500):
        super().__init__(message)
        self.status = status


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
        lib.np_last_status.argtypes = []
        lib.np_last_status.restype = C.c_uint16
        lib.np_write.argtypes = [C.c_void_p, C.c_char_p, C.c_size_t, C.c_int64]
        lib.np_write.restype = C.c_char_p
        lib.np_operate.argtypes = [C.c_void_p, C.c_uint8, C.c_uint32, C.c_char_p, C.c_size_t,
                                   C.c_char_p, C.c_size_t, C.c_int64]
        lib.np_operate.restype = C.c_char_p
        lib.np_add_attribute.argtypes = [C.c_void_p, C.c_char_p, C.c_size_t, C.c_char_p, C.c_size_t,
                                         C.c_char_p, C.c_size_t, C.c_int64]
        lib.np_add_attribute.restype = C.c_char_p
        lib.np_records.argtypes = [C.c_void_p]
        lib.np_records.restype = C.c_char_p
        lib.np_structured_records.argtypes = [C.c_void_p]
        lib.np_structured_records.restype = C.c_char_p
        lib.np_last_told.argtypes = [C.c_void_p]
        lib.np_last_told.restype = C.c_int64
        lib.np_list.argtypes = [C.c_void_p]
        lib.np_list.restype = C.c_char_p
        lib.np_get.argtypes = [C.c_void_p, C.c_uint32]
        lib.np_get.restype = C.c_char_p
        lib.np_metadata.argtypes = [C.c_void_p]
        lib.np_metadata.restype = C.c_char_p
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
        raise NativeStoreError((self._lib.np_last_error() or b"native store failure").decode("ascii", "replace"),
                               self._lib.np_last_status())

    def _call(self, function, *args):
        with self._lock:
            if self._handle is None:
                raise NativeStoreError("store is closed")
            result = function(self._handle, *args)
            if result is None:
                self._raise()
            return json.loads(result)  # decode under the lock, before the next native arena reset

    def write(self, text, told, subject=None):
        return self._operate(0, 0, text, told, subject)

    def change(self, target, text, told, subject=None):
        return self._operate(1, target, text, told, subject)

    def take_back(self, target, told, text="I take that back."):
        return self._operate(2, target, text, told, None)

    def _operate(self, operation, target, text, told, subject):
        if isinstance(target, bool) or not isinstance(target, int) or not 0 <= target < (1 << 32):
            raise ValueError("invalid target id")
        if isinstance(told, bool) or not isinstance(told, int) or not -(1 << 63) <= told < (1 << 63):
            raise ValueError("invalid told timestamp")
        raw = text.encode("utf-8")
        sub = None if subject is None else subject.encode("utf-8")
        return self._call(self._lib.np_operate, operation, target, sub, len(sub) if sub is not None else 0,
                          raw, len(raw), told)

    def listing(self):
        return self._call(self._lib.np_list)

    def records(self):
        return self._call(self._lib.np_records)

    def metadata(self):
        return self._call(self._lib.np_metadata)

    def last_told(self):
        with self._lock:
            if self._handle is None:
                raise NativeStoreError("store is closed")
            return self._lib.np_last_told(self._handle)

    def get(self, note_id):
        return self._call(self._lib.np_get, note_id)

    def _add_attribute(self, text, told, subject, attribute):
        raw, sub, attr = text.encode(), subject.encode(), attribute.encode()
        return self._call(self._lib.np_add_attribute, attr, len(attr), sub, len(sub), raw, len(raw), told)

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


class IntakeStore(ZStore):
    """WIP rule-book adapter. Not the default HTTP Notes implementation.

    Reuse the Python rule book's classification, then the reviewed native change/retire operation.
    Structured slot identity is stored in native journal attributes; no non-atomic metadata sidecar.
    Unsupported cases still refuse before mutation; legacy journal migration is not implemented.
    """
    def __init__(self, *args, schema=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._schema = schema
        self._lessons = {}
        try:
            from .store import intake
            notes = {r["source"]: r for r in self.metadata()}
            for record in self._call(self._lib.np_structured_records):
                if not record["attribute"].startswith("schema:") or record["source"] not in notes:
                    continue
                parsed = intake.parse(record["text"])
                if parsed is None or "schema:" + parsed["a"] != record["attribute"] or parsed["e"] != record["subject"]:
                    raise NativeStoreError("inconsistent structured record metadata", 500)
                nid = notes[record["source"]]["id"]
                self._lessons[nid] = dict(parsed, told=record["told"], id=nid)
        except BaseException:
            self.close()
            raise

    def write_text(self, text, told, subject=None):
        from .store import intake, nm, lnorm
        with self._lock:
            parsed = None if subject is not None else intake.parse(text)
            if parsed is None:
                return self.write(text, told, subject)
            if parsed["op"] not in ("ASSERT", "RETRACT", "DATED"):
                raise NativeStoreError("this structured operation needs the remaining compatibility adapter", 501)
            lesson = dict(parsed, told=told, id=-1)
            previous = [l for l in self._lessons.values() if (l["e"], l["a"]) == (lesson["e"], lesson["a"])]
            schema = nm.widen(self._schema or nm.L.SCHEMA, previous + [lesson])
            verdict = intake.classify(previous, lesson, ents=[lesson["e"]], sch=schema)
            rows = {r["id"]: r for r in self.listing()}
            if not verdict["learn"]:
                # Exact normalized duplicates use the reviewed journal's same-as closure, including erase.
                if verdict["verdict"] == intake.ALREADY_KNOWN and any(
                        rows[l["id"]]["state"] == "current" and lnorm(l["text"]) == lnorm(text) for l in previous):
                    return self._add_attribute(text, told, lesson["e"], "schema:" + lesson["a"])
                raise NativeStoreError("this structured no-op needs the remaining compatibility adapter", 501)
            current = [l for l in previous if rows[l["id"]]["state"] == "current"]
            retired = nm.NoteStore.no_longer_current(schema, current, lesson)
            if len(retired) > 1:
                raise NativeStoreError("atomic multi-target retirement is not implemented", 501)
            if retired:
                if parsed["op"] == "RETRACT":
                    result = self.take_back(current[retired[0]]["id"], told, text=text)
                else:
                    result = self.change(current[retired[0]]["id"], text, told, subject=lesson["e"])
            else:
                if parsed["op"] == "RETRACT":
                    raise NativeStoreError("this retraction needs the remaining compatibility adapter", 501)
                result = self._add_attribute(text, told, lesson["e"], "schema:" + lesson["a"])
            self._lessons[result["note"]["id"]] = dict(lesson, id=result["note"]["id"])
            result["verdict"] = verdict["verdict"]  # the rule book distinguishes added history from replacement
            return result

    def erase(self, note_id):
        with self._lock:
            result = super().erase(note_id)
            retained = {r["id"] for r in self.metadata()}
            self._lessons = {i: l for i, l in self._lessons.items() if i in retained}
            return result
