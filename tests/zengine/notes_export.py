#!/usr/bin/env python3
"""notes_export: one owner's note store as a snapshot file for the Zig lookup (zig/notes). docs/NOTES-ZIG.md.

The Zig code ports the read path only: given a snapshot and a question's token ids, it returns the notes and the
block that NoteStore.retrieve() returns. Writes (intake, supersede, take-back, delete) stay in Python, so a snapshot
is everything _lookup, pack and render read, taken under the owner's read lock:

  settings    cap, n_min, n_max, fold, hop, hop_weight, hop_from, hop_history, min_rel, subject, punct (and the two
              wording settings, already applied to the head and the lines)
  vocab       every token id's index key (NoteStore.key, fold strings interned to integers) and whether it is wordy
              (NoteStore.wordy), so Zig folds a question and builds its n-grams itself
  notes       id, state, kind, entity, attribute, told, the line exactly as NoteStore.line() gives it, its cost
              (NoteStore.cost) and its token ids; Zig rebuilds the two views' postings from the ids and checks the
              gram counts against the store's own
  the rest    entity phrases (O.phrases), ent_cur, ent_keys, by_key, the fixed costs and the head and titles

A key is one 32-bit number: a fold string's place in the interned list, or KEY_ID | the token's id when the key is
the id itself (fold off, an empty piece, part of a multi-byte character). For the serving model's tokenizer the
vocab is every id it has. The stand-in tokenizer has no fixed vocabulary (an id is the piece's own bytes), so its
snapshot lists the pieces of the notes and of the texts passed as `asks`, numbered from 0; Vocab.ids() maps a
question to those numbers and refuses a piece the snapshot lacks.

  python3 zengine/notes_export.py SET.json [--phase teach] [--tokenizer T] --out SNAP.json [--queries Q.txt]

--queries also writes the set's questions as the Zig CLI reads them: one line each, "past cap id id ...", past as
retrieve() decides it (notes.is_past); a line may end in "| note note ..." for retrieve()'s exclude. Standard library only (the tokenizer as notes.py loads it).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import notes as nm  # noqa: E402

SNAP_FORMAT = "lw-notes-snapshot/1"
KEY_ID = 0x8000_0000            # a key that is the token id itself
STATES = {nm.CURRENT: 0, nm.SUPERSEDED: 1, nm.TOMBSTONED: 2}
KINDS = {nm.FACT: 0, nm.TOMBSTONE: 1}
_FULL = {}                      # (tokenizer sha256, fold) -> the full vocab table, made once per process


class Vocab:
    """Token ids as the snapshot numbers them, and their keys. For the model's tokenizer the numbers are its ids;
    for the stand-in, a dense numbering of the pieces seen (dense: stand-in id -> number)."""

    def __init__(self, ns, dense=None):
        self.ns, self.dense = ns, dense
        self.folds, self.fold_ix = [], {}
        self.keys, self.wordy = [], []

    def enc(self, k):
        """A key as one 32-bit number."""
        if isinstance(k, str):
            j = self.fold_ix.get(k)
            if j is None:
                j = self.fold_ix[k] = len(self.folds)
                self.folds.append(k)
            return j
        return KEY_ID | (k if self.dense is None else self.dense[k])

    def ids(self, text):
        """A text's token ids as the snapshot numbers them."""
        raw = self.ns.tok.encode(text)
        if self.dense is None:
            return raw
        miss = [self.ns.tok.piece(i) for i in raw if i not in self.dense]
        if miss:
            raise nm.NotesError(f"the snapshot's stand-in vocabulary lacks {miss[:5]}; pass the text in `asks`")
        return [self.dense[i] for i in raw]


def vocab_for(ns, token_ids):
    """The vocab table: every id of the model's tokenizer (cached per tokenizer and fold), or for the stand-in the
    given ids in first-seen order. Keys come from NoteStore.key and NoteStore.wordy themselves, on a scratch owner,
    so the store's own memos are not touched."""
    scratch = nm.Owner("export")
    hf = isinstance(ns.tok, nm.HFTokenizer)
    if hf:
        memo = (ns.tok.sha256, ns.cfg["fold"])
        v = _FULL.get(memo)
        if v is None:
            v = Vocab(ns)
            n = ns.tok.tok.get_vocab_size(with_added_tokens=True)
            for i in range(n):
                v.keys.append(v.enc(ns.key(scratch, i)))
                v.wordy.append(ns.wordy(scratch, i))
            _FULL[memo] = v
        v.ns = ns
        return v
    dense = {}
    for i in token_ids:
        if i not in dense:
            dense[i] = len(dense)
    v = Vocab(ns, dense)
    for i in dense:
        v.keys.append(v.enc(ns.key(scratch, i)))
        v.wordy.append(ns.wordy(scratch, i))
    return v


def snapshot(ns, owner="user", asks=(), meta=None):
    """(the snapshot of one owner's store as plain values, its Vocab). `asks`: texts whose pieces the stand-in
    vocabulary must hold (the questions to be asked); the model's tokenizer needs none."""
    O = ns.owner(owner)
    if O is None:
        raise nm.NotesError(f"no notes for owner {owner!r}")
    with O.lock.read():
        order = sorted(O.notes)
        ids_seen = [t for i in order for t in O.notes[i]["tokens"]]
        ids_seen += [k for ph in O.phrases for k in ph if not isinstance(k, str)]
        for t in asks:
            ids_seen += ns.tok.encode(t)
        v = vocab_for(ns, ids_seen)
        ents = list(O.ent_cur)
        ent_ix = {e: k for k, e in enumerate(ents)}
        attrs = sorted({O.notes[i]["attribute"] for i in order})
        attr_ix = {a: k for k, a in enumerate(attrs)}
        pairs = list(O.by_key)
        pair_ix = {p: k for k, p in enumerate(pairs)}
        notes = []
        for i in order:
            n = O.notes[i]
            line = ns.line(O, i)
            val = n["value"]
            hop_to = ent_ix[val] if isinstance(val, str) and val in O.ent_cur and val != n["entity"] else -1
            notes.append({"id": i, "state": STATES[n["state"]], "kind": KINDS[n["kind"]],
                          "entity": ent_ix[n["entity"]], "attribute": attr_ix[n["attribute"]],
                          "pair": pair_ix[(n["entity"], n["attribute"])], "value_ent": hop_to, "told": n["told"],
                          "line": line, "cost": ns.cost(O, line), "tokens": _note_ids(v, n)})
        snap = {
            "format": SNAP_FORMAT, "owner": owner, "tokenizer": ns.tok.describe(), "meta": meta or {},
            "settings": {k: ns.cfg[k] for k in nm.DEFAULTS},
            "head": ns.head(), "cur_title": nm.CUR_TITLE, "hist_title": nm.HIST_TITLE,
            "fixed": {"head": ns.fixed()["head"], "current": ns.fixed()[nm.CURRENT],
                      "history": ns.fixed()["history"], "nl": ns.fixed()["nl"]},
            "entities": [{"name": e, "cur": sorted(O.ent_cur[e]),
                          "keys": sorted(pair_ix[k] for k in O.ent_keys.get(e, ()))} for e in ents],
            "attributes": attrs,
            "pairs": [{"entity": ent_ix[e], "attribute": attr_ix[a], "ids": list(O.by_key[(e, a)])}
                      for e, a in pairs],
            "phrases": [{"keys": [v.enc(k) for k in ph], "entity": ent_ix[e]} for ph, e in O.phrases.items()],
            "notes": notes,
            "n": {"current": O.n[nm.CURRENT], "history": O.n["history"]},
            "grams": {"current": len(O.idx[nm.CURRENT]), "history": len(O.idx["history"])},
        }
    snap["vocab"] = {"size": len(v.keys), "key": v.keys, "wordy": "".join("1" if w else "0" for w in v.wordy),
                     "folds": v.folds, "dense": v.dense is not None}
    return snap, v


def _note_ids(v, n):
    """A note's token ids as the snapshot numbers them."""
    return n["tokens"] if v.dense is None else [v.dense[t] for t in n["tokens"]]


def write(snap, path):
    """The snapshot as compact JSON (ASCII, so any text survives the trip)."""
    with open(path, "w") as f:
        json.dump(snap, f, separators=(",", ":"))


def query_line(v, text, past, cap, exclude=()):
    """One question as the Zig CLI reads it: "past cap id id ...", and "| note note ..." for retrieve()'s exclude."""
    line = " ".join([str(int(bool(past))), str(cap)] + [str(i) for i in v.ids(text)])
    return line + (" | " + " ".join(str(i) for i in exclude) if exclude else "")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("set")
    ap.add_argument("--phase", default="teach")
    ap.add_argument("--tokenizer", default="fake")
    ap.add_argument("--out", required=True)
    ap.add_argument("--queries", help="also write the set's questions here, one Zig query line each")
    a = ap.parse_args(argv)
    try:
        tok = nm.load_tokenizer(a.tokenizer)
        S = nm.load_set(a.set)
        ph = next((p for p in S["phases"] if p["name"] == a.phase), None)
        if ph is None:
            raise nm.NotesError(f"{a.set} has no phase {a.phase}")
        ns, _w = nm.set_store(S, ph["store"], tok)
        qs = [q["text"] for q in S["questions"]]
        snap, v = snapshot(ns, "user", asks=qs, meta={"set": S["id"], "phase": a.phase})
        write(snap, a.out)
        if a.queries:
            with open(a.queries, "w") as f:
                for t in qs:
                    f.write(query_line(v, t, nm.is_past(t), ns.cfg["cap"]) + "\n")
        print(f"{a.out}: {len(snap['notes'])} notes, vocab {snap['vocab']['size']}, "
              f"{os.path.getsize(a.out)} bytes")
    except nm.NotesError as e:
        print(f"notes_export: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
