"""Experimental HTTP adapter for new native journals. Python remains the default.

Legacy Python journals are refused by the native journal's snapshot/header check;
this adapter never migrates or overwrites them. Remaining intake parity gaps raise
explicit errors rather than silently falling back to a second store.
"""
import datetime
import functools
import threading
from types import SimpleNamespace

from .store import ProxyStore, StoreError, L, nm, validate_write
from .zstore import IntakeStore, NativeStoreError


def mapped(fn):
    @functools.wraps(fn)
    def call(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except NativeStoreError as error:
            raise StoreError(str(error), error.status) from error
    return call


class NativeNotes:
    @mapped
    def __init__(self, path, owner, tokenizer, config=None, abstain_line=None, schema=None, framing="quoted"):
        if not isinstance(owner, str) or not owner:
            raise StoreError("the owner is a name", 500)
        if not isinstance(tokenizer, nm.HFTokenizer):
            raise StoreError("experimental Zig store requires a local HF tokenizer", 501)
        if abstain_line is not None or framing != "quoted" or schema is not None:
            raise StoreError("experimental Zig store does not yet support these wording/schema settings", 501)
        import notes_export
        scratch = ProxyStore(tokenizer, config=config)
        scratch.owner(owner, create=True)
        snapshot, vocab = notes_export.snapshot(scratch, owner)
        self.owner, self.lock = owner, threading.RLock()
        self.store = SimpleNamespace(tok=tokenizer)  # startup description, not a Python note/index mirror
        self.cap = (config or {}).get("cap", 512)
        self.native = IntakeStore(snapshot, path, vocab.ids)
        self._sources = {r["id"]: r["source"] for r in self.native.metadata()}
        self._last_told = self.native.last_told()

    def close(self):
        self.native.close()

    def _views(self):
        rows = self.native.listing()
        sources = {r["id"]: r["source"] for r in rows}
        return [self._view(r, sources) for r in rows]

    @staticmethod
    def _view(r, sources):
        if not r["source"]:
            raise StoreError("native snapshot notes need legacy ID migration", 501)
        by = sources.get(r["by"])
        return {"id": r["source"], "number": r["id"], "version": r["version"],
                           "state": r["state"], "kind": r["kind"], "subject": r["subject"],
                           "text": r["text"], "line": r["line"], "source": r["source"], "told": r["told"],
                           "superseded_by": by if r["state"] == "superseded" else None,
                           "tombstoned_by": by if r["state"] == "tombstoned" else None}

    @mapped
    def listing(self, history=False):
        with self.lock:
            return [r for r in self._views() if history or r["state"] == "current"]

    def count(self):
        with self.lock:
            return len(self._sources)

    def get(self, nid):
        with self.lock:
            number = next((i for i, source in self._sources.items() if source == nid), None)
            if number is None:
                return None
            try:
                return self._view(self.native.get(number), self._sources)
            except NativeStoreError as error:
                raise StoreError(str(error), error.status) from error

    @mapped
    def lookup(self, question, cap=None, past=None, exclude=()):
        with self.lock:
            ids = {source: number for number, source in self._sources.items()}
            result = self.native.lookup(question, self.cap if cap is None else cap,
                                        nm.is_past(question) if past is None else past,
                                        [ids[i] for i in exclude if i in ids])
            result["note_ids"] = [self._sources[i] for i in result["ids"]]
            return result

    @mapped
    def write_text(self, text, subject=None, today=None):
        validate_write(text, subject)
        with self.lock:
            told = max(L.tick(today or datetime.date.today().isoformat()), self._last_told + 100)
            result = self.native.write_text(text, told, subject)
            self._last_told = told
            if result["note"]:
                self._sources[result["note"]["id"]] = result["note"]["source"]
            # Native Written.note may point at an existing same-as note. The HTTP contract
            # reserves note/id/number/written for a newly learned note, not its duplicate.
            learned = result["verdict"] in ("NEW", "REPLACES")
            note = self._view(result["note"], self._sources) if result["note"] and learned else None
            retired = [self._sources[i] for i in result["retired"]]
            retract = note and note["kind"] == "take_back"
            return {"id": note["id"] if note else None, "number": note["number"] if note else None,
                    "version": note["version"] if note else None, "state": note["state"] if note else None,
                    "verdict": result["verdict"], "written": note is not None, "source": result["source"],
                    "told": told, "superseded": [] if retract else retired,
                    "tombstoned": retired if retract else [],
                    "same_as": [self._sources[i] for i in result["same_as"]], "note": note}

    @mapped
    def erase(self, nid):
        with self.lock:
            row = self.get(nid)
            if row is None:
                raise StoreError("no note has this id: it was never written, or it was erased", 404)
            result = self.native.erase(row["number"])
            self._sources = {r["id"]: r["source"] for r in self.native.metadata()}
            return dict(result, erased=nid, renumbered=[],
                        complete=False, state="pending",
                        remaining=["the upstream engine's cache and logs are outside this store",
                                   "memory this process freed is not overwritten"],
                        remaining_places=["upstream", "freed_memory"])
