#!/usr/bin/env python3
"""Self-test of the runner (lwrun.py) against a mock TensorFold server. No GPU, no model, about two minutes.

  python3 zengine/lwrun_selftest.py [--dir DIR] [--records DIR]

The mock is this file, run as `lwrun_selftest.py mock serve FOLDER --host H --port N [--slide --slide-graph G]`. It
speaks the parts of TensorFold 1.0.8's HTTP API the runner uses: GET /health, POST /v1/slide/learn (server-sent
events), GET /v1/slide/graph with model.save, and POST /v1/chat/completions. The model folder is a tiny safetensors
shard laid out like 1.0.8's pocket, so a taught fact's neurons are real rows, zero mode zeroes real bytes, and a newer
fact that replaces an older one zeroes the older one's rows, as 1.0.8 does. After a save that wrote bytes the mock
logs 1.0.8's post-save line; as in our 1.0.8 runs, a missed lesson saves 0 bytes when it is the first fact, and writes
the shards again (and logs the line) when earlier facts are there. It answers every question with the last fact whose rows are still nonzero, or "I don't
know" when there is none: the answers say which facts are live, not whether the model is right. A system prompt in the
runner's in_context wording is honoured: the mock then answers from the last lesson the prompt quotes. A scenario file
(env MOCKTF_SCRIPT, JSON) says which lessons are missed, which replace which, which fail, whether the post-save line is
left out or late, whether the system message is dropped (ignore_system), the pocket size a save writes to config.json
(pocket_neurons), which folders the server dies on at start (fail_serve), and where to note each server start and
chat.

Each feature is checked on a real lwrun.py run against the mock, and each has a negative control: a run or input
where the feature is missing, switched off or given bad input, where the check must fail or the runner must refuse
(a check that cannot fail is not a check):

  save line      every lesson that saved bytes has the server's post-save line in its record and in the summary; a
                 missed first lesson (0 bytes) has none and costs no wait. Control: a mock that logs no such line.
  empty store    a delete phase with an empty store reuses the baseline answers and starts no server.
                 Control: --fresh-empty-reteach serves the original model again.
  kept-stable    WORK/kept-stable/MODE/teach.jsonl and PHASE.jsonl per delete mode, complete, with the right phase
                 names, and the judge reads them; a kept_stable score reaches summary.json. Controls: one file moved
                 aside; an empty score.
  zero limits    zero mode records which kept facts it cannot bring back (a deleted fact had replaced them, and no
                 kept fact replaced it in turn). Controls: the teach records without the server's replace notes, and
                 without only the last one.
  --cleanup      run as on Linux (AS_LINUX). A teach step that stored nothing (a missed first lesson, 0 bytes, no
                 pocket): its zero step is not applicable, with the reason in summary.json, no zero copy is made and
                 no copy is left; on --resume it stays so, and the fixed runner resumed on a folder the old logic
                 left removes both copies. A zero step skipped for its layout drops its own copy; one whose server
                 died keeps both, and its --resume finishes and removes them. Controls: the zero step as first built
                 (both copies kept, the GB10 runs' log lines), the resume rule off, the drop off.
  --sets         several sets in one call, one summary, a failing set does not stop the next. Control: a repeated
                 set, a leftover work dir and a set id that is not a folder name are refused before anything runs.
  reference      --reference in_context,replacement_memory (TEST-LOGIC.md section 8): the prompt word for word, the
                 memory's slots and its answers through the real judge, the steps and flags of a run, the pocket gap,
                 CHANGE and RETRACT outcomes on a run and on hand-written answers, --judge-only and --resume with
                 reference steps on file, the dry run, the memory's table per report group, outcomes that follow what
                 holds after a lesson, bad flags refused, and level 6's two-step questions grouped. Controls: other
                 stores and orders, an earliest-wins memory, a run without the flag, a mock that drops the system
                 message, the lesson learned, a reader deaf to negation, the folders moved aside, answers unchanged, a
                 dry run without the flag, wrong groupings, the outcome rule as first built, an explicit phase list,
                 and the two-step test as first built.
  in place       zero mode refuses a copy whose shard is a hard link to the source's (as on gb10-a, where leftover run
                 copies were deduplicated into hard links to the Hugging Face source) or a symlink to it, before any
                 byte is written, and both files stay as they were; the mock's own writes go through the same guard.
                 Controls: the guard switched off (the write goes through and changes the source), a real copy
                 (zeroed, the source untouched), and the path check patched out (the open itself still refuses).

--records DIR also reads stored run records under DIR (read only): the zero limits of a real #678 run, whether the
zero steps skipped for "no pocket" had teach records that stored nothing, and whether an empty-store step served
again gave the baseline answers word for word.
Nothing is deleted but the mock model copies that the --cleanup checks' runs remove, as --cleanup does: the work
folders stay under --dir (default: a new temp folder), about a megabyte in all.
"""
import argparse
import glob
import http.server
import json
import os
import random
import re
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
NOTES_HEAD_START = "Notes from memory: "   # how a notes block (zengine/notes.py HEAD) starts
LWRUN = os.path.join(HERE, "lwrun.py")
LWLOGIC = os.path.join(HERE, "lwlogic.py")

# ============================================================================================ the mock model
P, WIDE, BASE = 64, 72, 8                 # pocket of 64 learned neurons at the end of 72 shared-expert rows
LAYERS, POCKET_LAYERS = (0, 1, 2, 3), (1, 3)
PARTS = {"weight": ("U32", 4, 4), "scales": ("BF16", 2, 2), "biases": ("BF16", 2, 2)}  # dtype, columns, bytes each
SHARD = "model-00001-of-00001.safetensors"


HIDDEN = 8                                # down_proj rows of the level 7 mock model
DOWN = {"weight": ("U32", WIDE // 8, 4), "scales": ("BF16", WIDE // 8, 2), "biases": ("BF16", WIDE // 8, 2)}  # groups of 8


def tname(layer, part):
    return f"backbone.layers.{layer}.mixer.shared_experts.up_proj.{part}"


def dname(layer, part):
    return f"backbone.layers.{layer}.mixer.shared_experts.down_proj.{part}"


def part_of(name):
    return name.rsplit(".", 1)[1]


def bf16_bytes(x):
    return struct.pack("<f", x)[2:]  # the top half of a float32 (exact for the values the mock writes)


def l7_code(fid):
    """The 4-bit code a level 7 mock fact writes in every one of its up_proj and down_proj places."""
    return 1 + fid % 7


def l7_norms(fid, big=False):
    """The norms the level 7 mock writes for fact `fid` (up, down), worked out here, apart from lwrun's reader: up_proj
    rows of 32 codes (2 groups of 16) at scale 0.5 (64 when `big`), bias 0, in the 2 pocket layers; down_proj columns
    of HIDDEN rows at scale 1, bias 0."""
    c, sc = l7_code(fid), (64.0 if big else 0.5)
    return (2 * 32 * (sc * c) ** 2) ** 0.5, (2 * HIDDEN * c ** 2) ** 0.5


def make_model(folder, down=False):
    """A tiny source snapshot: config.json, an index and one shard whose up_proj tensors have WIDE rows, the last P
    of them the empty pocket (rows of zeros), the first BASE the model's own (nonzero). With `down` (the level 7
    tests) each layer also has a down_proj of HIDDEN rows and WIDE columns in groups of 8, its pocket columns zero."""
    os.makedirs(folder)
    rng = random.Random(7)
    header, blobs, off = {}, [], 0
    names = [(tname(layer, part), dt, cols, size) for layer in LAYERS for part, (dt, cols, size) in PARTS.items()]
    names.append(("backbone.embeddings.weight", "BF16", 8, 2))
    if down:
        names += [(dname(layer, part), dt, cols, size) for layer in LAYERS for part, (dt, cols, size) in DOWN.items()]
    for name, dt, cols, size in names:
        rows = HIDDEN if "down_proj" in name else WIDE
        n = rows * cols * size
        data = bytearray(n)
        if "down_proj" in name:  # the model's own columns (the first BASE inputs: word 0, group 0) nonzero
            for r in range(rows):
                for i in range(size if part_of(name) != "weight" else 4):
                    data[r * cols * size + i] = rng.randrange(1, 256)
        else:
            own = n if "embeddings" in name else BASE * cols * size
            for i in range(own):
                data[i] = rng.randrange(1, 256)
        header[name] = {"dtype": dt, "shape": [rows, cols], "data_offsets": [off, off + n]}
        blobs.append(bytes(data))
        off += n
    h = json.dumps(header).encode()
    h += b" " * (-len(h) % 8)
    with open(os.path.join(folder, SHARD), "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h + b"".join(blobs))
    with open(os.path.join(folder, "config.json"), "w") as f:
        json.dump({"model_type": "mock", "num_hidden_layers": len(LAYERS), "moe_shared_expert_intermediate_size": WIDE}, f)
    with open(os.path.join(folder, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": {name: SHARD for name in header}}, f)


def open_rw(path):
    """The mock's writes in place go through the runner's guard (lwrun.open_for_inplace_write), as the test suite's own do:
    never into a shard that is a symlink or a hard link."""
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    import lwrun  # noqa: E402
    return lwrun.open_for_inplace_write(path)


class Pocket:
    """The mock's view of a model folder: the pocket rows in the shard, and mock-pocket.json (which fact owns which
    neuron, and its state)."""

    def __init__(self, folder):
        self.folder = folder
        self.shard = os.path.join(folder, SHARD)
        self.meta = os.path.join(folder, "mock-pocket.json")
        with open(self.shard, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.h, self.base = json.loads(f.read(n)), 8 + n

    def span(self, layer, k, part):
        m = self.h[tname(layer, part)]
        width = m["shape"][1] * PARTS[part][2]
        return self.base + m["data_offsets"][0] + (WIDE - P + k) * width, width

    def write(self, k, fill):
        """Neuron k's rows in every pocket layer set to `fill` (0 zeroes them); the number of bytes written."""
        n = 0
        with open_rw(self.shard) as f:
            for layer in POCKET_LAYERS:
                for part in PARTS:
                    off, w = self.span(layer, k, part)
                    f.seek(off)
                    f.write(bytes([fill]) * w)
                    n += w
        return n

    def write_l7(self, k, fid, big=False):
        """Neuron k as a level 7 mock fact: every up_proj code l7_code(fid), scale 0.5 (64 when big), bias 0; and, when
        the model has down_proj, its column's codes l7_code(fid) with every pocket group's scale 1 and bias 0."""
        c, n = l7_code(fid), 0
        sc = bf16_bytes(64.0 if big else 0.5)
        word = struct.pack("<I", sum(c << (4 * i) for i in range(8)))
        with open_rw(self.shard) as f:
            for layer in POCKET_LAYERS:
                for part in PARTS:
                    off, w = self.span(layer, k, part)
                    f.seek(off)
                    f.write(word * (w // 4) if part == "weight" else (sc if part == "scales" else b"\0\0") * (w // 2))
                    n += w
                if dname(layer, "weight") not in self.h:
                    continue
                mw, ms, mb = (self.h[dname(layer, x)] for x in ("weight", "scales", "biases"))
                cols, groups = mw["shape"][1], ms["shape"][1]
                col = WIDE - P + k
                for r in range(HIDDEN):
                    at = self.base + mw["data_offsets"][0] + (r * cols + col // 8) * 4
                    f.seek(at)
                    x = struct.unpack("<I", f.read(4))[0]
                    x = (x & ~(15 << (4 * (col % 8)))) | (c << (4 * (col % 8)))
                    f.seek(at)
                    f.write(struct.pack("<I", x))
                    for g in range(1, groups):  # the pocket's groups: scale 1, bias 0 (shared by its neurons)
                        f.seek(self.base + ms["data_offsets"][0] + (r * groups + g) * 2)
                        f.write(bf16_bytes(1.0))
                        f.seek(self.base + mb["data_offsets"][0] + (r * groups + g) * 2)
                        f.write(b"\0\0")
                    n += 4
        return n

    def live(self, k):
        off, w = self.span(POCKET_LAYERS[0], k, "weight")
        with open(self.shard, "rb") as f:
            f.seek(off)
            return any(f.read(w))

    def facts(self):
        try:
            with open(self.meta) as f:
                return json.load(f)["facts"]
        except FileNotFoundError:
            return []


def mock_main(argv):
    """`mock --version`, or `mock serve FOLDER ...`: the mock TensorFold server, until SIGTERM."""
    if argv[:1] == ["--version"]:
        print("tensorfold-native mock (lwrun_selftest.py)")
        return 0
    ap = argparse.ArgumentParser(prog="mock")
    ap.add_argument("cmd")
    ap.add_argument("folder")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--slide", action="store_true")
    ap.add_argument("--slide-graph")
    a, _rest = ap.parse_known_args(argv)
    sc = {}
    if os.environ.get("MOCKTF_SCRIPT"):
        with open(os.environ["MOCKTF_SCRIPT"]) as f:
            sc = json.load(f)
    if sc.get("starts_log"):
        with open(sc["starts_log"], "a") as f:
            f.write(json.dumps({"folder": os.path.realpath(a.folder), "slide": a.slide, "port": a.port}) + "\n")
    if any(x in os.path.basename(os.path.realpath(a.folder)) for x in sc.get("fail_serve", [])):
        print(f"[tensorfold] mock: fail_serve: will not serve {a.folder}", flush=True)  # a server that dies at start
        return 3
    pocket = Pocket(a.folder)
    lock = threading.Lock()
    graph = {"next": 1, "facts": [], "model": {"layers": len(LAYERS)}}
    if a.slide and a.slide_graph and os.path.exists(a.slide_graph):
        with open(a.slide_graph) as f:
            graph = json.load(f)

    def say(line):
        print(line, flush=True)

    L7 = sc.get("l7")  # the level 7 scenario (capacity.py's self-test); without it the mock is as it always was

    def write_fact(g):
        if L7:
            return pocket.write_l7(g["k"], g["id"], big=g["text"] in L7.get("big_norm", []))
        return pocket.write(g["k"], (g["id"] * 37) % 255 + 1)

    def held_slots():
        """Level 7: the pocket slots taken. A missed fact takes none; a deleted one keeps its slot unless the scenario's
        delete frees it (1.0.8's replace zeroes a fact's neurons and keeps their places taken: READ, pocket.zig)."""
        return {g["k"] for g in graph["facts"] if g["state"] in ("learned", "replaced")
                or (g["state"] == "deleted" and not L7.get("delete_frees", True))}

    def l7_live(facts):
        """The texts of the facts that answer now: learned, rows nonzero, not faded (more than fade_after learned facts
        after it), not overwritten by a live fact it interferes with, and the fact it depends on live by its rows."""
        rows = {g["text"] for g in facts if g["state"] == "learned" and pocket.live(g["k"])}
        order = [g["text"] for g in facts if g["state"] == "learned"]
        live = set(rows)
        fade = L7.get("fade_after")
        if fade is not None:
            live = {t for t in live if sum(1 for u in order[order.index(t) + 1:] if u in rows) <= fade}
        for b, a_ in (L7.get("interfere") or {}).items():
            if b in live:
                live.discard(a_)
        for y, x in (L7.get("depends") or {}).items():
            if x not in rows:
                live.discard(y)
        return live

    def save(new, replaced):
        """1.0.8 saves after the stream's [DONE]; the mock saves before it, so the runner's first poll sees it."""
        t0 = time.time()
        graph["model"]["save"] = {"state": "saving", "at": int(t0 * 1000)}
        nbytes = 0
        if new["state"] == "learned":
            nbytes += write_fact(new)
        else:  # a missed lesson: 1.0.8 still writes the shards when earlier facts are there, and 0 bytes when none are
            for g in [g for g in graph["facts"] if g["state"] == "learned"][-1:]:
                nbytes += write_fact(g)
        for g in replaced:  # pocket.zig `retire`: a replaced fact's rows are zeroed
            nbytes += pocket.write(g["k"], 0)
        if nbytes:
            cfgp = os.path.join(a.folder, "config.json")
            with open(cfgp) as f:
                cfg = json.load(f)
            cfg["tensorfold_learned_neurons"] = sc.get("pocket_neurons", P)  # another value: a layout zero mode skips
            with open(cfgp, "w") as f:
                json.dump(cfg, f)
        with open(pocket.meta, "w") as f:
            json.dump({"facts": [{k: g[k] for k in ("id", "text", "state", "k")} for g in graph["facts"]]}, f)
        if L7:  # as 1.0.8's graph: model.pocket.used, the neurons in use at each pocket layer
            graph["model"]["pocket"] = {"neurons": P, "layers": list(POCKET_LAYERS),
                                        "used": [len(held_slots())] * len(POCKET_LAYERS)}
        if a.slide_graph:
            with open(a.slide_graph, "w") as f:
                json.dump(graph, f)
        dt = time.time() - t0
        graph["model"]["save"] = {"state": "saved", "at": int(time.time() * 1000), "seconds": round(dt, 1),
                                  "bytes": nbytes, "files": [SHARD] if nbytes else []}
        if nbytes and not sc.get("no_save_line"):
            back = "come back" if new["id"] % 2 else "do not all come back"
            line = (f"info: slide pocket: the learned neurons are in the model's shards ({nbytes / 1e6:.6f} MB "
                    f"written in {dt:.1f} s); the last fact's held-out answers {back} with them as written")
            if sc.get("save_line_delay"):
                threading.Timer(float(sc["save_line_delay"]), say, (line,)).start()
            else:
                say(line)

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            say(f"[tensorfold] 127.0.0.1 \"{self.command} {self.path} HTTP/1.1\" {args[1] if len(args) > 1 else '-'} -")

        def reply(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def body(self):
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}")

        def do_GET(self):
            if self.path == "/health":
                return self.reply(200, {"status": "ok", "warming": False})
            if self.path == "/v1/slide/graph" and a.slide:
                with lock:
                    return self.reply(200, graph)
            return self.reply(404, {"error": "not found"})

        def do_DELETE(self):
            if not (L7 and L7.get("delete_endpoint") and a.slide and self.path.startswith("/v1/slide/fact/")):
                return self.reply(404, {"error": "not found"})  # 1.0.8 has no per-fact delete
            fid = int(self.path.rsplit("/", 1)[1])
            with lock:
                g = next((x for x in graph["facts"] if x["id"] == fid), None)
                if g is None:
                    return self.reply(404, {"error": f"no fact {fid}"})
                if g["state"] in ("learned", "replaced"):
                    pocket.write(g["k"], 0)
                g["state"] = "deleted"
                g.pop("neurons", None)
                graph["model"]["pocket"] = {"neurons": P, "layers": list(POCKET_LAYERS),
                                            "used": [len(held_slots())] * len(POCKET_LAYERS)}
                with open(pocket.meta, "w") as f:
                    json.dump({"facts": [{k: x[k] for k in ("id", "text", "state", "k")} for x in graph["facts"]]}, f)
                if a.slide_graph:
                    with open(a.slide_graph, "w") as f:
                        json.dump(graph, f)
            say(f"[tensorfold] slide: fact {fid} deleted")
            return self.reply(200, {"id": fid, "state": "deleted"})

        def do_POST(self):
            if self.path == "/v1/chat/completions":
                return self.chat(self.body())
            if self.path == "/v1/slide/learn" and a.slide:
                return self.learn(self.body()["text"])
            return self.reply(404, {"error": "not found"})

        def chat(self, b):
            msgs = b.get("messages") or []
            system = next((m.get("content") for m in msgs if m.get("role") == "system"), None)
            if sc.get("chat_log"):
                with open(sc["chat_log"], "a") as f:
                    f.write(json.dumps({"folder": os.path.realpath(a.folder), "slide": a.slide, "system": system,
                                        "roles": [m.get("role") for m in msgs],
                                        "question": next((m.get("content") for m in msgs if m.get("role") == "user"),
                                                         None)}) + "\n")
            # a system prompt in the runner's in_context wording is honoured as a model told in context would: the
            # last lesson it quotes is answered from, as the pocket answers from its last live fact. ignore_system
            # (a control) drops it, as a server that never passes the system message to the model would.
            told = [] if sc.get("ignore_system") or not system else re.findall(r'^\d+\. "(.*)"$', system, re.M)
            # a notes block (zengine/notes.py) is honoured the same way: the last current note it quotes is answered from
            if not sc.get("ignore_system") and system and system.startswith(NOTES_HEAD_START):
                cur = system.split("\nCurrent:\n", 1)[1].split("\nEarlier, ", 1)[0] if "\nCurrent:\n" in system else ""
                told = [json.loads(m) for m in re.findall(r'^\[\d+\] ("(?:[^"\\]|\\.)*")', cur, re.M)]
            live = [g for g in pocket.facts() if g["state"] == "learned" and pocket.live(g["k"])]
            if told:    # the in_context system prompt first, then the level 7 answers, then the last live fact
                text = f"You told me: {told[-1]}"
            elif L7:  # the first entry whose lesson is live (or that needs none) answers; else "I don't know."
                on = l7_live(pocket.facts())
                q = b["messages"][-1]["content"]
                text = next((ans for need, ans in L7["answers"].get(q, []) if need is None or need in on),
                            "I don't know.")
            else:
                text = f"You told me: {live[-1]['text']}" if live else "I don't know that about you."
            th = bool((b.get("chat_template_kwargs") or {}).get("enable_thinking"))
            return self.reply(200, {"choices": [{"message": {"content": text,
                                                             "reasoning_content": "mock reasoning" if th else ""},
                                                 "finish_reason": "stop"}],
                                    "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18}})

        def learn(self, text):
            if text in sc.get("fail_learn", []):
                return self.reply(500, {"error": "mock: this lesson fails"})
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()

            def event(kind, data):
                self.wfile.write(f"event: {kind}\ndata: {json.dumps(data)}\n\n".encode())
                self.wfile.flush()

            with lock:
                fid = graph["next"]
                graph["next"] += 1
                missed = text in sc.get("miss", [])
                k = fid - 1
                if L7:  # the lowest free slot; none left within the scenario's capacity: 1.0.8's PocketFull
                    held = held_slots()
                    free = [i for i in range(min(P, L7.get("capacity", P))) if i not in held]
                    k = free[0] if free else None
                new = {"id": fid, "text": text, "source": "chat", "at": int(time.time() * 1000), "k": k,
                       "questions": [f"mock question {fid}?"], "recalled": not missed}
                if L7:
                    new["loss"] = round(0.05 + 0.001 * fid, 4)
                event("fact", {"id": fid, "text": text})
                event("learning", {"id": fid})
                replaced = []
                if L7 and k is None:  # as 1.0.8 reports it: the learner's error name as the lesson's message
                    new.update({"state": "missed", "message": "PocketFull", "recalled": False})
                    event("learned", {"id": fid, "recalled": False, "steps": 0, "message": "PocketFull"})
                elif missed:
                    new.update({"state": "missed", "message": "mock: the fact does not come back"})
                    event("learned", {"id": fid, "recalled": False, "steps": 5, "message": new["message"]})
                else:
                    new.update({"state": "learned", "neurons": [{"layer": layer, "neuron": new["k"], "key": "reply"}
                                                                for layer in POCKET_LAYERS]})
                    event("learned", {"id": fid, "recalled": True, "steps": 3})
                    old = (sc.get("replaces") or {}).get(text)
                    for g in graph["facts"]:
                        if old and g["text"] == old and g["state"] == "learned":
                            g.update({"state": "replaced", "message": f"replaced by fact {fid}"})
                            g.pop("neurons", None)  # as 1.0.8's graph shows a replaced fact
                            replaced.append(g)
                graph["facts"].append(new)
                save(new, replaced)
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

    srv = http.server.ThreadingHTTPServer((a.host, a.port), H)
    say(f"[tensorfold] mock serving {a.folder} on {a.host}:{a.port}" + (" with --slide" if a.slide else ""))
    srv.serve_forever()
    return 0


# ============================================================================================ test helpers
class Check:
    def __init__(self):
        self.rows = []

    def add(self, name, ok, n, detail=""):
        self.rows.append((name, bool(ok), n, detail))
        print(f"  {'PASS' if ok else 'FAIL'}  {name}  [{n}]" + (f"  {detail}" if detail else ""), flush=True)


def read_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default


def lines(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class Lab:
    """One test folder: the fake binary, the source model and one work folder per runner call."""

    def __init__(self, root, down=False):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.bin = os.path.join(root, "bin", "tensorfold-native")
        os.makedirs(os.path.dirname(self.bin), exist_ok=True)
        with open(self.bin, "w") as f:
            f.write(f'#!/bin/sh\nexec "{sys.executable}" "{os.path.abspath(__file__)}" mock "$@"\n')
        os.chmod(self.bin, 0o755)
        self.src = os.path.join(root, "src-model")
        make_model(self.src, down=down)

    def run(self, name, sets, extra=(), script=None, timeout=900, prog=None, work=None):
        """lwrun.py on `sets` (one path, or a list for --sets) in work folder `name`, against the mock. `prog` runs
        another tool the same way (capacity.py's self-test: [python, capacity.py, "run"]). `work` names the work
        folder when it is not `name`: a later call on an earlier call's folder, with logs of its own."""
        work = os.path.join(self.root, work or name)
        sc = dict(script or {})
        sc.setdefault("starts_log", os.path.join(self.root, f"{name}.starts.jsonl"))
        sc.setdefault("chat_log", os.path.join(self.root, f"{name}.chats.jsonl"))
        spath = os.path.join(self.root, f"{name}.script.json")
        with open(spath, "w") as f:
            json.dump(sc, f)
        env = dict(os.environ, MOCKTF_SCRIPT=spath)
        what = ["--sets"] + list(sets) if isinstance(sets, list) else [sets]
        cmd = list(prog or [sys.executable, LWRUN]) + what + ["--bin", self.bin, "--model-src", self.src, "--work", work,
                                                "--port", str(free_port())] + list(extra)
        t0 = time.time()
        p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
        with open(os.path.join(self.root, f"{name}.console.log"), "w") as f:
            f.write(p.stdout + "\n--- stderr ---\n" + p.stderr)
        return types.SimpleNamespace(rc=p.returncode, out=p.stdout, err=p.stderr, work=work,
                                     starts=lines(sc["starts_log"]), chats=lines(sc["chat_log"]),
                                     seconds=round(time.time() - t0, 1))


def ex(name):
    return os.path.join(REPO, "examples", name)


# ============================================================================================ the checks
def save_lines_ok(work):
    """Every taught lesson whose save wrote bytes carries the server's post-save line, parsed, in its record and in
    the summary; a lesson whose save wrote nothing carries none."""
    st = read_json(os.path.join(work, "phases", "teach", "state.json"), {}) or {}
    summ = {x["name"]: x for x in (((read_json(os.path.join(work, "summary.json"), {}) or {}).get("steps") or {})
                                   .get("teach") or {}).get("lessons") or []}
    bad, n = [], 0
    for r in st.get("lessons") or []:
        n += 1
        wrote = int((r.get("save") or {}).get("bytes") or 0) > 0
        sl = r.get("save_log")
        if not wrote:
            if sl:
                bad.append(f"{r['name']}: a line for a save of 0 bytes")
            continue
        if not (sl and sl.get("line", "").startswith("info: slide pocket: the learned neurons are in the model's shards")
                and sl.get("mb", 0) > 0 and sl.get("held_out_back") == bool(r["fact_ids_sse"][0] % 2)):
            bad.append(f"{r['name']}: save_log {sl}")
        elif (summ.get(r["name"]) or {}).get("save_line") != sl["line"]:
            bad.append(f"{r['name']}: the summary does not carry the line")
    return bool(n) and not bad, n, "; ".join(bad)


def reuse_ok(lab, res, phase="del-A"):
    """The empty-store step reused the baseline answers: same answers word for word, filed under its phase, marked
    as reused, and no server was started for it (the source served once, for the baseline)."""
    sd = os.path.join(res.work, "phases", f"{phase}.reteach")
    st = read_json(os.path.join(sd, "state.json"), {}) or {}
    base = {(x["qid"], x["thinking"]): x["content"] for x in lines(os.path.join(res.work, "phases", "baseline", "answers.jsonl"))}
    got = lines(os.path.join(sd, "answers.jsonl"))
    src_starts = sum(1 for s in res.starts if s["folder"] == os.path.realpath(lab.src))
    same = sum(base.get((x["qid"], x["thinking"])) == x["content"] for x in got)
    ok = (bool(st.get("reused_baseline")) and bool(got) and same == len(got) == len(base)
          and all(x.get("reused_from") == "baseline" and x["phase"] == phase for x in got) and src_starts == 1
          and os.path.exists(os.path.join(sd, "verdicts.json")))
    return ok, len(got), (f"{same}/{len(got)} answers equal the baseline's, source served {src_starts} time(s), "
                          f"reused_baseline {'set' if st.get('reused_baseline') else 'not set'}")


def kept_stable_ok(work, modes):
    """WORK/kept-stable/MODE/teach.jsonl and MODE/PHASE.jsonl for every delete phase the mode has answers for: every
    line under the file's own phase name, every question and thinking mode there, as manifest.json lists them."""
    S = read_json(os.path.join(work, "set.json"))
    man = read_json(os.path.join(work, "kept-stable", "manifest.json"))
    if not man:
        return False, 0, "no manifest.json"
    th = {x["thinking"] for x in lines(os.path.join(work, "phases", "teach", "answers.jsonl"))}
    want = {(q["qid"], t) for q in S["questions"] for t in th}
    bad, n = [], 0
    for mode in modes:
        m = man["modes"].get(mode)
        if not m:
            bad.append(f"mode {mode} missing")
            continue
        files = {"teach": "teach.jsonl"}
        for p in S["phases"][2:]:
            if os.path.isdir(os.path.join(work, "phases", f"{p['name']}.{mode.split('.')[0]}")):
                files[p["name"]] = f"{p['name']}.jsonl"
        for ph, fn in files.items():
            rows = lines(os.path.join(work, "kept-stable", mode, fn))
            n += len(rows)
            listed = m["teach"] if ph == "teach" else m["deletes"].get(ph)
            if not rows or {(x["qid"], x["thinking"]) for x in rows} != want:
                bad.append(f"{mode}/{fn} missing or incomplete ({len(rows)} lines)")
            elif any(x["phase"] != ph for x in rows):
                bad.append(f"{mode}/{fn} holds lines of another phase")
            elif not listed or listed["file"] != os.path.join("kept-stable", mode, fn) or not listed["complete"]:
                bad.append(f"{mode}/{fn} not listed in manifest.json")
    return not bad, n, "; ".join(bad)


def judge_reads(work, mode):
    """lwlogic.py judge takes the folder as it is: teach and delete answers of one mode, side by side."""
    files = sorted(glob.glob(os.path.join(work, "kept-stable", mode, "*.jsonl")))
    out = os.path.join(work, f"kept-stable-{mode}.verdicts.json")
    p = subprocess.run([sys.executable, LWLOGIC, "judge", os.path.join(work, "set.json")] + files + ["--json", out],
                       capture_output=True, text=True, timeout=300)
    v = read_json(out, {}) or {}
    phases = sorted({r["phase"] for r in v.get("results", [])})
    total = sum(len(lines(f)) for f in files)
    return p.returncode == 0 and len(v.get("results", [])) == total and "teach" in phases, total, \
        f"judged {len(v.get('results', []))} of {total} lines, phases {phases}"


# lwrun.py as on Linux, the only place its --cleanup deletes: imported with IS_LINUX set, so cleanup, copies (cp -RL)
# and the disk estimate take their Linux paths. The server child's setup still runs as on this machine (no prctl
# outside Linux). OFF switches parts of the zero-step cleanup back to how they were first built, for the controls:
# "na" no not-applicable rule (a pocketless taught folder is a layout mismatch), "drop" a skipped zero step's copy
# stays on disk, "resume" a not-applicable step is tried again on --resume.
AS_LINUX = '''import sys
sys.path.insert(0, {here!r})
import lwrun  # noqa: E402

OFF = {off!r}
lwrun.IS_LINUX, lwrun.IS_MAC = True, False
_child_setup = lwrun.child_setup


def child_setup(max_seconds):
    lwrun.IS_LINUX = sys.platform.startswith("linux")
    try:
        return _child_setup(max_seconds)
    finally:
        lwrun.IS_LINUX = True


lwrun.child_setup = child_setup
if "na" in OFF:
    lwrun.Run.zero_not_applicable = lambda self, tst, tfolder: None
if "drop" in OFF:
    lwrun.Run.drop_unwritten_copy = lambda self, path, why: None
if "resume" in OFF:
    lwrun.Run.zero_na_holds = lambda self: False
if "hold" in OFF:
    lwrun.Run.notes_holds = lambda self: dict()
sys.exit(lwrun.main(sys.argv[1:]))
'''


def linux_runner(root, off=()):
    """The command (Lab.run's `prog`) that runs lwrun.py as on Linux, with the parts named in `off` as first built."""
    path = os.path.join(root, "lwrun-as-linux" + "".join(f"-{x}-off" for x in off) + ".py")
    with open(path, "w") as f:
        f.write(AS_LINUX.format(here=HERE, off=tuple(off)))
    return [sys.executable, path]


def model_dirs(work):
    """The folders in WORK/models: the model copies still on disk, whatever copies.json says."""
    d = os.path.join(work, "models")
    return sorted(x for x in os.listdir(d) if os.path.isdir(os.path.join(d, x))) if os.path.isdir(d) else []


def copies_of(work):
    """copies.json by step: {step: (on disk, marked removed)}."""
    reg = read_json(os.path.join(work, "copies.json"), {}) or {}
    return {c["step"]: (os.path.isdir(c["path"]), bool(c.get("removed"))) for c in reg.get("copies") or []}


def zero_na_ok(res, phase="del-A", made_before=False):
    """A run whose teach step stored nothing, with --cleanup: exit code 0; the zero step not applicable, with the reason
    in summary.json and no error; no zero copy made (with `made_before`, the one an earlier run left is removed) and
    no server started on one; the taught copy removed (marked so in copies.json); no model folder left on disk; the
    reteach step done."""
    steps = summary_of(res.work).get("steps") or {}
    z, r = steps.get(f"{phase}.zero") or {}, steps.get(f"{phase}.reteach") or {}
    cp = copies_of(res.work)
    why = z.get("not_applicable") or ""
    bad = []
    if res.rc != 0:
        bad.append(f"rc {res.rc}")
    if z.get("status") != "not applicable" or z.get("error") or "stored nothing" not in why or "no pocket" not in why:
        bad.append(f"{phase}.zero status {z.get('status')!r}, error {z.get('error')!r}, not_applicable {why!r}")
    want_zero = (False, True) if made_before else None  # (on disk, marked removed), or never registered
    if cp.get(f"{phase}.zero") != want_zero or any(s["folder"].endswith(f".{phase}.zero") for s in res.starts):
        bad.append(f"a zero copy was made, kept or served: copies {cp}")
    if cp.get("teach") != (False, True):
        bad.append(f"taught copy (on disk, marked removed) {cp.get('teach')}")
    if model_dirs(res.work):
        bad.append(f"left in models/: {model_dirs(res.work)}")
    if r.get("status") != "done":
        bad.append(f"{phase}.reteach {r.get('status')}")
    return not bad, len(cp), "; ".join(bad) or f"not_applicable: {why}"


# the GB10 run logs' two lines (gb10-a-neg-seed1 and nine more, 13 Oct 2026), as the first built zero step wrote them
OLD_ZERO_LINES = ("step del-A.zero FAILED: zero mode skipped: the folder does not match zero_fact.py's layout: config.json "
                  "has no tensorfold_learned_neurons: the folder holds no pocket",
                  "--cleanup: the taught folder is kept: a zero step is not done, and resuming it needs the folder")


def cleanup_checks(C, lab, root):
    """Section 4 of the self-test: --cleanup and zero steps, on Linux paths (AS_LINUX), each check with its control."""
    S1 = read_json(ex("L1-seed1.json"))
    miss = {"miss": [S1["lessons"][0]["text"]]}  # the mock's missed first lesson saves 0 bytes and leaves no pocket
    flags = ["--delete-mode", "both", "--thinking", "off", "--cleanup"]
    linux = linux_runner(root)

    # a teach step that stored nothing: the zero step is not applicable and the copies go. Control: the old logic
    ra = lab.run("zna-fixed", ex("L1-seed1.json"), flags, miss, prog=linux)
    ok, n, d = zero_na_ok(ra)
    rc = lab.run("zna-old", ex("L1-seed1.json"), flags, miss, prog=linux_runner(root, ("na", "drop")))
    okc, _nc, dc = zero_na_ok(rc)
    old_sig = (rc.rc == 1 and all(x in rc.out for x in OLD_ZERO_LINES) and model_dirs(rc.work) == [
        "L1-seed1.del-A.zero", "L1-seed1.teach"] and copies_of(rc.work) == {"teach": (True, False),
                                                                         "del-A.zero": (True, False)})
    C.add("--cleanup, a teach step that stored nothing (missed first lesson, 0 bytes, no pocket): the zero step is "
          "not applicable with the reason in summary.json, no zero copy is made, the taught copy is removed, rc 0; "
          "control: the old logic fails the check, keeps both copies and logs the GB10 runs' two lines",
          ok and not okc and old_sig, n, d + ("" if old_sig else f"; control rc {rc.rc}, models/ {model_dirs(rc.work)}, "
                                              f"copies {copies_of(rc.work)}") + ("" if not okc else "; control passed"))

    # --resume: the step stays not applicable; the fixed runner resumed on the old logic's folder removes both copies.
    # Control: with the resume rule off, the resumed step is tried again and fails, as the taught folder is gone
    shutil.copytree(ra.work, os.path.join(root, "zna-fixed-copy"), symlinks=True)
    rd = lab.run("zna-fixed-resume", ex("L1-seed1.json"), flags + ["--resume"], miss, prog=linux, work="zna-fixed")
    okd, _nd, dd = zero_na_ok(rd)
    rf = lab.run("zna-old-resume", ex("L1-seed1.json"), flags + ["--resume"], miss, prog=linux, work="zna-old")
    okf, _nf, df = zero_na_ok(rf, made_before=True)
    rx = lab.run("zna-resume-off", ex("L1-seed1.json"), flags + ["--resume"], miss,
                 prog=linux_runner(root, ("resume",)), work="zna-fixed-copy")
    zx = (summary_of(rx.work).get("steps") or {}).get("del-A.zero") or {}
    ctl = rx.rc == 1 and zx.get("status") == "failed" and "is gone" in (zx.get("error") or "")
    C.add("--resume, --cleanup: a not-applicable zero step stays so (no server started, no copy, rc 0); the fixed "
          "runner resumed on the old logic's folder removes both copies it left; control: with the resume rule off the "
          "resumed step fails (the taught folder is gone)",
          okd and not rd.starts and "step del-A.zero: not applicable (" in rd.out and okf and not rf.starts and ctl, 3,
          "; ".join(x for x in (dd if not okd else "", f"{len(rd.starts)} server starts on resume" if rd.starts else "",
                                df if not okf else "", "" if ctl else f"control rc {rx.rc}, zero {zx.get('status')} "
                                f"{zx.get('error')!r}") if x))

    # a zero step skipped for its layout leaves no copy of its own, and keeps the taught folder (a resume needs it); a
    # genuinely unfinished zero step keeps both, and its --resume finishes and removes them. Control: with the drop
    # switched off, the skipped step's copy stays
    zflags = ["--delete-mode", "zero", "--thinking", "off", "--cleanup"]
    rb = lab.run("zskip-layout", ex("L1-seed1.json"), zflags, {"pocket_neurons": 32}, prog=linux)
    zb = read_json(os.path.join(rb.work, "phases", "del-A.zero", "state.json"), {}) or {}
    okb = (rb.rc == 1 and zb.get("status") == "failed" and "is 32, not the 64" in (zb.get("zero_skipped") or "")
           and copies_of(rb.work) == {"teach": (True, False), "del-A.zero": (False, True)}
           and model_dirs(rb.work) == ["L1-seed1.teach"] and OLD_ZERO_LINES[1] in rb.out)
    rbx = lab.run("zskip-layout-drop-off", ex("L1-seed1.json"), zflags, {"pocket_neurons": 32},
                  prog=linux_runner(root, ("drop",)))
    ctlb = rbx.rc == 1 and copies_of(rbx.work) == {"teach": (True, False), "del-A.zero": (True, False)}
    re_ = lab.run("zunfinished", ex("L1-seed1.json"), zflags, {"fail_serve": [".del-A.zero"]}, prog=linux)
    ze = read_json(os.path.join(re_.work, "phases", "del-A.zero", "state.json"), {}) or {}
    oke = (re_.rc == 1 and ze.get("status") == "failed" and (ze.get("zero") or {}).get("rows_zeroed", 0) > 0
           and copies_of(re_.work) == {"teach": (True, False), "del-A.zero": (True, False)} and OLD_ZERO_LINES[1] in re_.out)
    re2 = lab.run("zunfinished-resume", ex("L1-seed1.json"), zflags + ["--resume"], {}, prog=linux, work="zunfinished")
    ze2 = (summary_of(re2.work).get("steps") or {}).get("del-A.zero") or {}
    oke2 = (re2.rc == 0 and ze2.get("status") == "done" and "was zeroed already" in re2.out and "copy [" not in re2.out
            and copies_of(re2.work) == {"teach": (False, True), "del-A.zero": (False, True)} and not model_dirs(re2.work))
    C.add("--cleanup: a zero step skipped for its layout (pocket of 32) removes its own unwritten copy and keeps the "
          "taught one; a zero step whose server died keeps both, and --resume reuses the zeroed copy, finishes and "
          "removes both; control: with the drop off the skipped step's copy stays",
          okb and ctlb and oke and oke2, 4,
          f"skipped: copies {copies_of(rb.work)}, rc {rb.rc}; control copies {copies_of(rbx.work)}; unfinished: rc "
          f"{re_.rc}, status {ze.get('status')}; resumed: rc {re2.rc}, status {ze2.get('status')}, copies "
          f"{copies_of(re2.work)}")


REF_PROMPT_L2 = ('Earlier, the user told you the following, one message at a time, oldest first:\n'
                 '1. "Of all colours, I like white the most."\n'
                 '2. "I like green best now."\n'
                 '3. "Green is not my favourite colour anymore."\n'
                 'In these messages, "I", "me" and "my" mean the user.')  # the wording in TEST-LOGIC.md section 8
EXAMPLES = ("L1-seed1", "L2-seed1", "L3-seed1", "L4-seed1", "L5-seed1", "eyes-678", "neg-seed1", "neg2-seed1")


def summary_of(work):
    return read_json(os.path.join(work, "summary.json"), {}) or {}


def reference_ok(lab, lwrun, res, S, phase="teach"):
    """A run with --reference in_context,replacement_memory beside `phase`: both steps done, every question in both
    thinking modes under the phase's own name, the flags in summary.json, the mock asked with exactly the documented
    prompt on every in_context chat (and with none anywhere else), served from the source without --slide, and no
    server started for the memory."""
    summ = summary_of(res.work)
    steps = summ.get("steps") or {}
    store = next(p["store"] for p in S["phases"] if p["name"] == phase)
    prompt = lwrun.in_context_prompt(S, store)
    want = {(q["qid"], th) for q in S["questions"] for th in (False, True)}
    rows = {k: lines(os.path.join(res.work, "phases", f"{phase}.{k}", "answers.jsonl")) for k in lwrun.REF_KINDS}
    sys_chats = [c for c in res.chats if c.get("system") is not None]
    bad = []
    if res.rc != 0:
        bad.append(f"rc {res.rc}")
    for k, xs in rows.items():
        if (steps.get(f"{phase}.{k}") or {}).get("status") != "done":
            bad.append(f"step {phase}.{k} not done")
        if {(x["qid"], x["thinking"]) for x in xs} != want or any(x["phase"] != phase or x["mode"] != k for x in xs):
            bad.append(f"{k}: {len(xs)} answers, not every question under phase {phase}")
    asked = (summ.get("reference") or {}).get("asked")
    if asked != {"kinds": list(lwrun.REF_KINDS), "of": [phase]} or (summ.get("args") or {}).get("reference") is None:
        bad.append(f"summary.json reference.asked {asked}, args.reference {(summ.get('args') or {}).get('reference')}")
    if len(sys_chats) != len(want) or any(c["system"] != prompt or c["slide"] or c["folder"] != os.path.realpath(lab.src)
                                          or c.get("roles") != ["system", "user"] for c in sys_chats):
        bad.append(f"{len(sys_chats)} chats carried a system prompt, {sum(c['system'] == prompt for c in sys_chats)} "
                   f"of them the documented one, {sum(c.get('roles') == ['system', 'user'] for c in sys_chats)} as "
                   f"[system, user], of {len(want)} in_context questions")
    if len(res.starts) != 2:
        bad.append(f"{len(res.starts)} server starts (want 2: teach, in_context)")
    return not bad, len(sys_chats), "; ".join(bad)


def gap_cell(res, group="CURRENT", phase="teach", pocket="teach"):
    P = ((summary_of(res.work).get("reference") or {}).get("phases") or {}).get(phase) or {}
    return ((P.get("gap") or {}).get(pocket) or {}).get(group) or {}


def outcomes_golden(lwrun, lw, S, reader):
    """Hand-written answers on L2-seed1, one per outcome, for its RETRACT (teach: L3, stale green) and its CHANGE (del-A,
    where L3 is deleted: L2, stale white), read by `reader`; and the same CHANGE in teach, where the later L3 writes its
    key again, so it gets no outcome at all. (outcomes got, outcomes wanted)"""
    cases = [("teach", "L3", {"Q01": "I don't know your favourite colour; it is no longer green.", "Q07": "No."}, "corrected_only"),
             ("teach", "L3", {"Q01": "Your favourite colour is green.", "Q07": "Yes, it is green."}, "stale_only"),
             ("teach", "L3", {"Q01": "I don't know your favourite colour.", "Q07": "Yes."}, "both"),
             ("teach", "L3", {"Q01": "My name is Nemotron.", "Q07": "Maybe."}, "neither"),
             ("del-A", "L2", {"Q01": "Your favourite colour is green.", "Q08": "Green."}, "corrected_only"),
             ("del-A", "L2", {"Q01": "Your favourite colour is white.", "Q06": "Yes."}, "stale_only"),
             ("del-A", "L2", {"Q01": "Green.", "Q09": "White."}, "both"),
             ("del-A", "L2", {"Q01": "I don't know.", "Q06": "No.", "Q07": "I don't know."}, "neither"),
             ("teach", "L2", {"Q01": "Your favourite colour is green.", "Q06": "Yes."}, None)]  # L3 rewrites the key
    got = []
    for ph, name, ans, _want in cases:
        L = lwrun.change_lessons(lw, S, ph)
        oc = lwrun.lesson_outcomes(lw, S, ph, L, {(q, False): c for q, c in ans.items()}, reader=reader)
        got.append(((oc.get(name) or {}).get("off") or {}).get("outcome"))
    return got, [c[3] for c in cases]


def memory_sane(lwrun, lw, slots_of):
    """On every example set and phase: each CURRENT, yes/no and either-or question on a single-valued key whose last
    lesson in the store gives a value now (ASSERT present, since or always; CHANGE; DATED) is one a latest-value
    lookup must get right; the memory's answer is judged by the real judge. (checked, failures)"""
    n, bad = 0, []
    for name in EXAMPLES:
        S = read_json(ex(f"{name}.json"))
        for p in S["phases"]:
            keep = set(p["store"])
            store = [l for l in S["lessons"] if l["id"] in keep]
            if not store:
                continue
            slots = slots_of(S, p["store"])
            for q in S["questions"]:
                if q.get("role") != "probe" or q["type"] not in ("CURRENT", "YES_NO", "EITHER_OR") \
                        or S["schema"][q["a"]]["multi"]:
                    continue
                last = [l for l in store if (l["e"], l["a"]) == (q["e"], q["a"])][-1:]
                if not last or not (last[0]["op"] in ("CHANGE", "DATED") or (
                        last[0]["op"] == "ASSERT" and not {"until", "between"} & set(last[0].get("when") or {}))):
                    continue
                n += 1
                if lw.judge_one(S, p["name"], q, lwrun.memory_answer(lw, S, slots, q))[0] != "pass":
                    bad.append(f"{name} {p['name']} {q['qid']}")
    return n, bad


# The replacement memory on the eight example sets, teach phase, per report group: the table in TEST-LOGIC.md
# section 8 (MEASURED 13 Oct 2026). [passed, asked]
MEMORY_TABLE = {"CURRENT": [30, 32], "WHO_HAS": [3, 3], "negation": [24, 42], "dates": [0, 10], "two_step": None,
                "yes_no": [18, 24], "either_or": [16, 16], "temporal": [0, 31], "controls": [17, 41], "all": [84, 147]}


def memory_table(lwrun, lw, groups_fn, answer_fn):
    """The memory's teach-phase answers on the eight example sets, judged by the real judge and grouped as the report
    groups them (answer_groups over groups_fn's groups and the judge's buckets): {group: [passed, asked] or None}."""
    tot = {}
    for name in EXAMPLES:
        S = read_json(ex(f"{name}.json"))
        slots = lwrun.memory_slots(S, next(p["store"] for p in S["phases"] if p["name"] == "teach"))
        rows = [{"phase": "teach", "qid": q["qid"], "thinking": False, "question": q["text"],
                 "content": answer_fn(lw, S, slots, q)} for q in S["questions"]]
        groups = groups_fn(S, "teach")
        for r in lw.judge(S, rows)[0]:
            for g in lwrun.answer_groups(groups, r["qid"], r["buckets"]):
                c = tot.setdefault(g, [0, 0])
                c[0] += r["verdict"] == "pass"
                c[1] += 1
    return {g: tot.get(g) for g in MEMORY_TABLE}


def with_phase(lw, S, name, store):
    """A copy of set S with one more phase that keeps `store`, its right answers worked out by the rule book."""
    T = json.loads(json.dumps(S))
    T["phases"].append({"name": name, "store": list(store),
                        "deleted": [l["id"] for l in S["lessons"] if l["id"] not in store]})
    lw.fill_expected(T)
    return T


def marks_before_now(lw, S, rec, q, r):
    """The outcome rule as first built (a control): corrected was the CHANGE's new value, or for a RETRACT any "don't
    know" on a single-valued key and any foods without the stale one on likes, whatever still held after it."""
    marks, stale, v = set(), set(rec["stale"]), (rec["new"] if rec["op"] == "CHANGE" else None)
    if q["type"] == "CURRENT":
        named = set(r["named"])
        if named & stale:
            marks.add("stale")
        if v is not None and v in named:
            marks.add("corrected")
        if rec["op"] == "RETRACT":
            if S["schema"][q["a"]]["multi"]:
                if r["got"] == "SET" and named and not named & stale:
                    marks.add("corrected")
            elif r["absent"]:
                marks.add("corrected")
    elif q["type"] == "YES_NO":
        if r["yn"] == lw.YES and q["v"] in stale:
            marks.add("stale")
        if r["yn"] == lw.YES and v is not None and q["v"] == v:
            marks.add("corrected")
        if r["yn"] == lw.NO and rec["op"] == "RETRACT" and q["v"] in stale:
            marks.add("corrected")
    elif q["type"] == "EITHER_OR":
        if r["choice"] in stale:
            marks.add("stale")
        if v is not None and r["choice"] == v:
            marks.add("corrected")
    return marks


def outcomes_after(lwrun, lw, reader, marks):
    """Hand-written answers where what holds after the lesson matters: a retraction on likes that leaves nothing (L4
    with a phase that keeps only "I really like olives." and "I do not like olives anymore."), one that leaves two
    foods (L4 teach), and a retraction of a value that did not hold while another did (L2 with a phase that keeps only
    "Of all colours, I like white the most." and "Green is not my favourite colour anymore."). Returns (outcomes got,
    outcomes wanted, the judge's verdicts on the single CURRENT answers, those cases' outcomes)."""
    S2, S4 = read_json(ex("L2-seed1.json")), read_json(ex("L4-seed1.json"))
    S2x, S4x = with_phase(lw, S2, "x-white-not-green", [1, 3]), with_phase(lw, S4, "x-olives-gone", [9, 11])
    cases = [(S4x, "x-olives-gone", "L11", {"Q08": "I don't know which foods you like."}, "corrected_only"),
             (S4x, "x-olives-gone", "L11", {"Q08": "You like olives."}, "stale_only"),
             (S4, "teach", "L11", {"Q08": "You like pears and apples."}, "corrected_only"),
             (S4, "teach", "L11", {"Q08": "You like cheese."}, "neither"),
             (S2x, "x-white-not-green", "L3", {"Q01": "Your favourite colour is white."}, "corrected_only"),
             (S2x, "x-white-not-green", "L3", {"Q01": "I don't know your favourite colour."}, "neither"),
             (S2x, "x-white-not-green", "L3", {"Q01": "Your favourite colour is green.", "Q08": "White."}, "both")]
    got, verdicts, single = [], [], []
    for S, ph, name, ans, _want in cases:
        L = lwrun.change_lessons(lw, S, ph)
        oc = lwrun.lesson_outcomes(lw, S, ph, L, {(q, False): c for q, c in ans.items()}, reader=reader, marks=marks)
        got.append(((oc.get(name) or {}).get("off") or {}).get("outcome"))
        if len(ans) == 1:
            (qid, c), = ans.items()
            verdicts.append(lw.judge_one(S, ph, next(q for q in S["questions"] if q["qid"] == qid), c)[0])
            single.append(got[-1])
    return got, [c[4] for c in cases], verdicts, single


def two_step_before(q):
    """The two-step test as first built (a control): hops or steps of 2 or more, or a type named for chains."""
    n = q.get("hops") or q.get("steps") or 1
    return (isinstance(n, int) and n >= 2) or str(q.get("type") or "").upper().startswith(
        ("TWO_STEP", "CHAIN", "MULTI_HOP"))


def reference_checks(C, lab, lwrun, root, r1, r9):
    """Section 6 of the self-test: the reference phases (TEST-LOGIC.md section 8), each check with its control."""
    lw = lwrun.lw_module(LWLOGIC)
    S1, S2 = read_json(ex("L1-seed1.json")), read_json(ex("L2-seed1.json"))
    S3, S4 = read_json(ex("L3-seed1.json")), read_json(ex("L4-seed1.json"))

    # the prompt, word for word
    got = lwrun.in_context_prompt(S2, [1, 2, 3])
    rev = json.loads(json.dumps(S2))
    rev["lessons"].reverse()
    others = [lwrun.in_context_prompt(S2, [1, 2]), lwrun.in_context_prompt(rev, [1, 2, 3])]
    try:
        lwrun.in_context_prompt(S2, [])
        refused = False
    except lwrun.RunError:
        refused = True
    C.add("reference: the in_context prompt is the documented wording, the store's lessons in teaching order, word for "
          "word; controls: the store without L3 and the lessons in reverse order give other prompts, an empty store is "
          "refused", got == REF_PROMPT_L2 and all(x != REF_PROMPT_L2 for x in others) and refused, 4,
          "" if got == REF_PROMPT_L2 else f"got {got!r}")

    # the replacement memory: slots as documented
    me = "me"
    hand = {"entities": [me], "lessons": [
        {"id": 1, "op": "ASSERT", "e": me, "a": "eye_colour", "v": "green"},
        {"id": 2, "op": "ASSERT", "e": me, "a": "eye_colour", "v": "brown", "when": {"until": "2025-03-03"}},
        {"id": 3, "op": "ASSERT", "e": me, "a": "fav_colour", "v": "red"},
        {"id": 4, "op": "RETRACT", "e": me, "a": "fav_colour", "v": None}]}
    want = [((S2, [1, 2, 3]), {}),
            ((S2, [1, 2]), {(me, "fav_colour"): "green"}),
            ((S3, [1, 2, 3]), {(me, "fav_colour"): "yellow", (me, "home_city"): "Graz", (me, "eye_colour"): "amber"}),
            ((S4, list(range(1, 12))), {("my dog", "name"): "Bruno", (me, "likes"): "apples", ("my cat", "eye_colour"): "hazel",
                                        ("my cat", "name"): "Nora", ("my dog", "likes"): "olives",
                                        ("my dog", "eye_colour"): "grey"}),
            ((hand, [1, 2, 3, 4]), {(me, "eye_colour"): "green", (me, "fav_colour"): lwrun.MEM_NONE})]
    clear = lwrun.memory_answer(lw, hand, lwrun.memory_slots(hand, [1, 2, 3, 4]),
                                {"qid": "X", "type": "CURRENT", "e": me, "a": "fav_colour"})
    ok = all(lwrun.memory_slots(S, st) == w for (S, st), w in want) and clear == "You have no favourite colour anymore." \
        and lw.extract_value(clear, lw.SCHEMA, "fav_colour", "present")["got"] == "ABSENT"
    early = [lwrun.memory_slots(dict(S, lessons=S["lessons"][::-1]), st) == w for (S, st), w in want]
    C.add("reference: the replacement memory keeps the latest value per key (L2, L3, L4 and a hand-made store: a "
          "retraction empties only its own value, likes keep one food, 'until' leaves now alone, 'no favourite "
          "anymore' is none and is read as such); control: a memory where the earliest lesson wins gets "
          f"{sum(not x for x in early)} of {len(want)} stores wrong", ok and not all(early), len(want),
          "" if ok else f"slots {[lwrun.memory_slots(S, st) for (S, st), _w in want]}, none answer {clear!r}")
    n, bad = memory_sane(lwrun, lw, lwrun.memory_slots)
    nb, badb = memory_sane(lwrun, lw, lambda S, st: lwrun.memory_slots(dict(S, lessons=S["lessons"][::-1]), st))
    C.add("reference: on every example set and phase, the memory's answer passes the judge on each CURRENT, yes/no and "
          "either-or question whose key's last lesson gives a value now; control: the earliest-wins memory fails "
          f"{len(badb)} of them", n > 0 and not bad and nb == n and bool(badb), n,
          f"{len(bad)} fail" + (f": {', '.join(bad[:6])}" if bad else ""))

    # a whole run: L1, the pocket misses the lesson; both references beside teach. Controls: r1 (no --reference) and
    # a server that drops the system message
    miss1 = {"miss": [S1["lessons"][0]["text"]]}
    flags = ["--phases", "teach", "--delete-mode", "reteach", "--reference", "in_context,replacement_memory"]
    rr = lab.run("L1-ref", ex("L1-seed1.json"), flags, miss1)
    ok, n, d = reference_ok(lab, lwrun, rr, S1)
    okc, _nc, dc = reference_ok(lab, lwrun, r1, S1)
    C.add("reference: --reference adds teach.in_context and teach.replacement_memory, every answer under phase teach, the "
          "flags in summary.json, the documented prompt as the first message of every in_context chat and nowhere "
          "else, no server for the memory; control: the run without --reference has none of it", ok and not okc, n,
          d or f"control: {dc[:160]}")
    ri = lab.run("L1-ref-ignore", ex("L1-seed1.json"), flags, dict(miss1, ignore_system=True))
    want_cell = {"pocket": [0, 3], "in_context": [3, 3], "replacement_memory": [3, 3], "gap": 1.0}
    want_ignored = {"pocket": [0, 3], "in_context": [0, 3], "replacement_memory": [3, 3], "gap": 0.0}
    g, gi = gap_cell(rr), gap_cell(ri)
    rep = os.path.join(rr.work, "reference", "report.txt")
    ok = all(g.get(th) == want_cell for th in ("off", "on")) and os.path.exists(rep) and "CURRENT" in open(rep).read()
    # the control must move the gap with in_context, not with the memory: a gap read from the wrong column stays +1.00
    oki = all(gi.get(th) == want_ignored for th in ("off", "on"))
    C.add("reference: pocket gap on CURRENT, pocket 0/3, in_context 3/3, memory 3/3: +1.00 in both thinking modes, also "
          "in reference/report.txt; control: a server that drops the system message gives in_context "
          f"{(gi.get('off') or {}).get('in_context')}, memory {(gi.get('off') or {}).get('replacement_memory')} and "
          f"gap {(gi.get('off') or {}).get('gap')}", ok and oki and g != gi, 2,
          "" if ok and oki else f"got {g}; control {gi}")

    # CHANGE and RETRACT outcomes on a run: L2, the pocket misses L3 ("not green anymore"), so it still says green.
    # Control: the same run with L3 learned
    flags2 = ["--phases", "teach", "--delete-mode", "reteach", "--thinking", "off", "--reference", "both",
              "--reference-of", "all"]
    ro = lab.run("L2-ref", ex("L2-seed1.json"), flags2, {"miss": [S2["lessons"][2]["text"]]})
    rl = lab.run("L2-ref-learned", ex("L2-seed1.json"), flags2)

    def outcomes(res):
        ph = ((summary_of(res.work).get("reference") or {}).get("phases")) or {}
        oc = lambda p, src, l: ((((ph.get(p) or {}).get("outcomes") or {}).get(src) or {}).get(l) or {}).get("off", {}).get("outcome")
        why = {x["name"]: x.get("why", "") for x in (ph.get("teach") or {}).get("lessons") or []
               if not x.get("classified")}
        return ({src: oc("teach", src, "L3") for src in ("teach", "in_context", "replacement_memory")},
                {src: oc("del-A", src, "L2") for src in ("in_context", "replacement_memory")},
                why.get("L2", "") if not any(oc("teach", src, "L2") for src in ("teach", "in_context")) else "")

    t3, a2, why = outcomes(ro)
    t3l, _a2l, _whyl = outcomes(rl)
    want3 = {"teach": "stale_only", "in_context": "corrected_only", "replacement_memory": "corrected_only"}
    C.add("reference: CHANGE and RETRACT outcomes on a run; the pocket that missed 'not green anymore' is stale only, "
          "in_context and the memory corrected only; in del-A the CHANGE to green is classified (stale: white); in teach "
          "it is not, a later lesson writes the key; control: with the lesson learned the pocket is "
          f"{t3l.get('teach')}", ro.rc == 0 and t3 == want3 and a2 == {"in_context": "corrected_only",
                                                                         "replacement_memory": "corrected_only"}
          and "later lesson (L3)" in why and rl.rc == 0 and t3l.get("teach") != "stale_only", 3,
          f"teach L3 {t3}, del-A L2 {a2}")
    gotg, wantg = outcomes_golden(lwrun, lw, S2, lwrun.read_answer)

    def deaf(lw_, S, ph, q, c):  # a reader that does not hear negation ("no longer green" read as "green")
        return lwrun.read_answer(lw_, S, ph, q, c.replace("no longer ", "").replace("don't ", "do ").replace(" not ", " "))

    gotb, _w = outcomes_golden(lwrun, lw, S2, deaf)
    C.add("reference: outcomes of 9 hand-written answer sets: one per outcome for a RETRACT and for a CHANGE, and none for a "
          "CHANGE a later lesson overrides; control: a "
          f"reader deaf to negation gets {sum(x != y for x, y in zip(gotb, wantg))} of them wrong",
          gotg == wantg and gotb != wantg, len(wantg), "" if gotg == wantg else f"got {gotg}, want {wantg}")

    # --judge-only finds reference steps on file without --reference; control: with them moved aside it reports none
    jo = os.path.join(root, "L2-ref-judge")
    shutil.copytree(ro.work, jo, symlinks=True)
    ref_dirs = [os.path.join(jo, "phases", x) for x in sorted(os.listdir(os.path.join(jo, "phases")))
                if x.endswith(("in_context", "replacement_memory"))]
    for x in ref_dirs:  # the old verdicts go aside: only a judge run of the steps found on file brings them back
        os.rename(os.path.join(x, "verdicts.json"), os.path.join(x, "verdicts.json.aside"))
    rj = lab.run("L2-ref-judge", ex("L2-seed1.json"), ["--judge-only", "--thinking", "off"])
    rejudged = sum(os.path.exists(os.path.join(x, "verdicts.json")) for x in ref_dirs)
    jc = os.path.join(root, "L2-ref-judge-none")
    shutil.copytree(ro.work, jc, symlinks=True)
    for x in os.listdir(os.path.join(jc, "phases")):
        if x.endswith(("in_context", "replacement_memory")):
            os.rename(os.path.join(jc, "phases", x), os.path.join(jc, f"aside-{x}"))
    rjc = lab.run("L2-ref-judge-none", ex("L2-seed1.json"), ["--judge-only", "--thinking", "off"])
    ref_j = summary_of(rj.work).get("reference") or {}
    C.add("reference: --judge-only without --reference judges the reference steps on file again and rebuilds the report "
          "(same outcomes); control: with those folders moved aside summary.json has no reference",
          rj.rc == 0 and len(ref_j.get("steps") or []) == 6 and rejudged == 6 and outcomes(rj)[0] == want3
          and rjc.rc == 0 and "reference" not in summary_of(rjc.work), len(ref_j.get("steps") or []),
          f"{rejudged} of {len(ref_dirs)} verdict files written again; steps {ref_j.get('steps')}")

    # resume: in_context answers asked with another prompt are moved aside and asked again; control: an untouched
    # copy resumes with no chat at all
    rs, rsc = os.path.join(root, "L2-ref-resume"), os.path.join(root, "L2-ref-resume-same")
    for d in (rs, rsc):
        shutil.copytree(ro.work, d, symlinks=True)
    ans = os.path.join(rs, "phases", "teach.in_context", "answers.jsonl")
    xs = lines(ans)
    xs[0]["system_sha256"] = "0" * 64
    with open(ans, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(x) + "\n" for x in xs)
    res_flags = ["--phases", "teach", "--delete-mode", "reteach", "--thinking", "off", "--reference", "in_context",
                 "--resume"]
    ra = lab.run("L2-ref-resume", ex("L2-seed1.json"), res_flags)
    rb = lab.run("L2-ref-resume-same", ex("L2-seed1.json"), res_flags)
    stale = [x for x in os.listdir(os.path.dirname(ans)) if x.startswith("stale-")]
    C.add("reference: on --resume, in_context answers asked with another prompt are moved aside and asked again with "
          "today's; control: unchanged answers are kept and nothing is asked",
          ra.rc == 0 and len(stale) == 1 and len([c for c in ra.chats if c["system"]]) == len(xs)
          and all(x.get("system_sha256") != "0" * 64 for x in lines(ans)) and rb.rc == 0 and not rb.chats,
          len(xs), f"{len(stale)} stale folder(s), {len(ra.chats)} chats; control {len(rb.chats)} chats")

    # the dry run shows the reference steps and the prompt, and writes nothing; control: a dry run without --reference
    rd = lab.run("L1-ref-dry", ex("L1-seed1.json"), ["--dry-run", "--reference", "both"])
    shown = ("step teach.in_context" in rd.out and "| Earlier, the user told you the following" in rd.out
             and "step teach.replacement_memory" in rd.out and "me fav_colour = blue" in rd.out)
    C.add("reference: --dry-run shows both reference steps, the prompt and the memory's slots, and writes nothing; "
          "control: the dry run without --reference shows none", rd.rc == 0 and shown and not os.path.exists(rd.work)
          and "in_context" not in r9.out, 2, f"rc {rd.rc}")

    # the report's groups and the memory's answers on the eight example sets: the table in TEST-LOGIC.md section 8.
    # Controls: negation read from every key the store touches (not only retracted ones), and a memory that names
    # nobody for "who has ...?"
    tab = memory_table(lwrun, lw, lwrun.question_groups, lwrun.memory_answer)

    def wide_negation(S, ph):
        out = lwrun.question_groups(S, ph)
        keep = set(next(p["store"] for p in S["phases"] if p["name"] == ph))
        keys = {(l["e"], l["a"]) for l in S["lessons"] if l["id"] in keep}
        for q in S["questions"]:
            if q.get("role") == "probe" and (q.get("e"), q.get("a")) in keys and "negation" not in out[q["qid"]]:
                out[q["qid"]].append("negation")
        return out

    def nobody(lw_, S, slots, q):
        return "No one you told me about." if q["type"] == "WHO_HAS" else lwrun.memory_answer(lw_, S, slots, q)

    tw = memory_table(lwrun, lw, wide_negation, lwrun.memory_answer)
    tn = memory_table(lwrun, lw, lwrun.question_groups, nobody)
    C.add("reference: the report's groups and the memory's answers on the eight example sets, teach phase, judged: "
          "CURRENT 30/32, WHO_HAS 3/3, negation 24/42, dates 0/10, no two-step question, and the judge's groups yes/no "
          "18/24, either-or 16/16, temporal 0/31, controls 17/41, all 84/147, as documented; "
          f"controls: negation from every taught key gives {tw['negation']}, a memory that names nobody gives WHO_HAS "
          f"{tn['WHO_HAS']}", tab == MEMORY_TABLE and tw != MEMORY_TABLE and tn != MEMORY_TABLE,
          (tab.get("all") or [0, 0])[1], "" if tab == MEMORY_TABLE else f"got {tab}")

    # corrected follows what holds just after the lesson. Control: the rule as first built
    gota, wanta, verd, single = outcomes_after(lwrun, lw, lwrun.read_answer, lwrun.outcome_marks)
    gotb, _w, _v, _s = outcomes_after(lwrun, lw, lwrun.read_answer, marks_before_now)
    agree = all((o == "corrected_only") == (v == "pass") for o, v in zip(single, verd))
    C.add("reference: outcomes follow what holds just after the lesson: on likes a retraction that leaves nothing makes "
          "\"don't know\" corrected, one that leaves two foods needs one of them; a retraction of a value that did "
          "not hold leaves the other value corrected and \"don't know\" not; on every single answer corrected only "
          "exactly where the judge passes it; control: the rule as first built gets "
          f"{sum(x != y for x, y in zip(gotb, wanta))} of {len(wanta)} wrong", gota == wanta and agree and gotb != wanta,
          len(wanta), "" if gota == wanta and agree else f"got {gota}, want {wanta}; verdicts {verd}")

    # bad --reference flags are refused before anything runs; control: an explicit --reference-of list is honoured
    bad = [(["--reference", "nope"], "--reference takes"),
           (["--reference", "both", "--reference-of", "nope"], "is not a phase"),
           (["--reference", "both", "--reference-of", "baseline"], "keeps no lesson"),
           (["--reference-of", "del-A"], "needs --reference"),
           (["--reference", "both", "--reference-of", ","], "names no phase")]
    refused = []
    for i, (fl, why) in enumerate(bad):
        rb = lab.run(f"L2-ref-bad{i}", ex("L2-seed1.json"), ["--dry-run"] + fl)
        refused.append(rb.rc == 2 and why in rb.err and not os.path.exists(rb.work) and not rb.starts)
    rg = lab.run("L2-ref-of", ex("L2-seed1.json"), ["--dry-run", "--reference", "in_context", "--reference-of", "del-A"])
    honoured = (rg.rc == 0 and "step del-A.in_context" in rg.out and "step teach.in_context" not in rg.out
                and "replacement_memory" not in rg.out)
    C.add("reference: bad flags are refused before anything runs: an unknown kind, an unknown phase, a phase with no "
          "lesson, --reference-of without --reference, an empty --reference-of; control: --reference in_context "
          "--reference-of del-A is accepted and plans del-A.in_context only", all(refused) and honoured, len(bad),
          f"refused {sum(refused)} of {len(bad)}; control rc {rg.rc}")

    # two-step questions as level 6 makes them (CHAIN with hops, COMPOUND with parts) are grouped; no example question
    # is. Controls: the test as first built misses COMPOUND, and one that takes every question flags the example sets
    chain = {"qid": "X1", "role": "probe", "type": "CHAIN", "start": "the boat", "hops": ["owner"],
             "final": "eye_colour", "text": "What colour are the eyes of the owner of the boat?"}
    compound = {"qid": "X2", "role": "probe", "type": "COMPOUND", "op": "and", "text": "Is my favourite colour blue and "
                "is my dog called Bruno?", "parts": [{"type": "YES_NO", "e": "me", "a": "fav_colour", "v": "blue"},
                                                     {"type": "YES_NO", "e": "my dog", "a": "name", "v": "Bruno"}]}
    exq = [q for name in EXAMPLES for q in read_json(ex(f"{name}.json"))["questions"]]
    S6 = json.loads(json.dumps(S1))
    S6["questions"] += [chain, compound]
    grouped = lwrun.question_groups(S6, "teach")

    def finds(fn):
        return [fn(q) for q in (chain, compound)] == [True, True] and not any(fn(q) for q in exq)

    C.add("reference: the two_step group holds level 6's CHAIN and COMPOUND questions and no question of the eight "
          "example sets; controls: the test as first built misses COMPOUND, one that takes every question flags "
          f"{sum(1 for q in exq)} example questions", finds(lwrun.two_step) and "two_step" in grouped["X1"]
          and "two_step" in grouped["X2"] and not finds(two_step_before) and not finds(lambda q: True), 2 + len(exq),
          f"X1 {grouped['X1']}, X2 {grouped['X2']}")


def notes_blocks(S, phase, cap=None):
    """{qid: block or None} of one phase, as zengine/notes.py gives them with the fake tokenizer and the defaults."""
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    import notes  # noqa: E402  (the note memory, next to this file)
    b, _ns = notes.phase_blocks(S, phase, notes.FakeTokenizer(), {"cap": cap} if cap else None)
    return {q: (r["block"] or None) for q, r in b.items()}


def notes_run_ok(res, S, phases, pocket_of, thinking=(False, True)):
    """A run with --reference notes,notes_pocket beside `phases`: every notes step done; every question in each thinking
    mode under its phase's name; every chat served without --slide is one of these steps', with exactly the block
    notes.py gives that question (none where nothing is retrieved) as [system, user]; notes served from the source,
    notes_pocket from the folder its phase's own step taught (`pocket_of`: phase -> copy name), read only; notes.jsonl
    one line per question within the cap; the answer lines carry the note ids; the flags and the notes in
    summary.json; the notes columns in the report. (ok, chats checked, what is wrong)"""
    summ = summary_of(res.work)
    steps = summ.get("steps") or {}
    want = {(q["qid"], th) for q in S["questions"] for th in thinking}
    src = os.path.realpath(os.path.join(os.path.dirname(res.work), "src-model"))
    models = os.path.join(res.work, "models")
    exp, bad = [], []
    if res.rc != 0:
        bad.append(f"rc {res.rc}")
    for ph in phases:
        blocks = notes_blocks(S, ph)
        for k in ("notes", "notes_pocket"):
            key = f"{ph}.{k}"
            st = steps.get(key) or {}
            if st.get("status") != "done":
                bad.append(f"{key} {st.get('status')}")
            xs = lines(os.path.join(res.work, "phases", key, "answers.jsonl"))
            if {(x["qid"], x["thinking"]) for x in xs} != want or any(x["phase"] != ph or x["mode"] != k for x in xs):
                bad.append(f"{key}: {len(xs)} answers, not every question under {ph}")
            if any(x.get("notes") is None or "notes_tokens" not in x for x in xs):
                bad.append(f"{key}: answer lines without their note ids")
            log = lines(os.path.join(res.work, "phases", key, "notes.jsonl"))
            if [x["qid"] for x in log] != [q["qid"] for q in S["questions"]] or any(
                    x["tokens"] > x["cap"] or x["lookup_us"] < 0 or (x["block"] or None) != blocks[x["qid"]] for x in log):
                bad.append(f"{key}: notes.jsonl does not hold one line per question with its block")
            folder = src if k == "notes" else os.path.realpath(os.path.join(models, f"{S['id']}.{pocket_of[ph]}"))
            if k == "notes_pocket" and (st.get("folder_unchanged") is not True or st.get("pocket_from") != pocket_of[ph]):
                bad.append(f"{key}: pocket_from {st.get('pocket_from')}, folder_unchanged {st.get('folder_unchanged')}")
            for q in S["questions"]:
                for _th in thinking:
                    exp.append((folder, q["text"], blocks[q["qid"]]))
    got = [(c["folder"], c["question"], c["system"]) for c in res.chats if not c["slide"]]
    roles_ok = all(c.get("roles") == (["system", "user"] if c["system"] else ["user"]) for c in res.chats)
    if sorted(got, key=repr) != sorted(exp, key=repr) or not roles_ok:
        bad.append(f"{len(got)} chats without --slide, {len(exp)} wanted; same multiset "
                   f"{sorted(got, key=repr) == sorted(exp, key=repr)}; roles as sent {roles_ok}")
    if not any(b for _f, _q, b in exp) or not any(b is None for _f, _q, b in exp):
        bad.append("the case should have both questions with notes and questions without")
    ref = summ.get("reference") or {}
    if (summ.get("args") or {}).get("reference") is None or set((ref.get("phases") or {}).get(phases[0], {}).get(
            "notes", {})) != {"notes", "notes_pocket"}:
        bad.append(f"summary.json: args.reference {(summ.get('args') or {}).get('reference')}, notes "
                   f"{list(((ref.get('phases') or {}).get(phases[0]) or {}).get('notes', {}))}")
    rep = os.path.join(res.work, "reference", "report.txt")
    if not (os.path.exists(rep) and "notes_pocket" in open(rep).read() and "lookup p50" in open(rep).read()):
        bad.append("reference/report.txt has no notes columns or notes line")
    return not bad, len(got), "; ".join(bad)


def notes_checks(C, lab, lwrun, root, r1, r9):
    """Section 6b of the self-test: the notes phases (docs/NOTES-LAYER.md), each check with its control."""
    S1, S2 = read_json(ex("L1-seed1.json")), read_json(ex("L2-seed1.json"))
    nflags = ["--notes-tokenizer", "fake"]

    # a whole run: notes and notes_pocket beside teach and del-A of L2; control: r1 has none of it
    flags = ["--phases", "teach,del-A", "--delete-mode", "reteach", "--reference", "notes,notes_pocket",
             "--reference-of", "teach,del-A"] + nflags
    rn = lab.run("L2-notes", ex("L2-seed1.json"), flags)
    ok, n, d = notes_run_ok(rn, S2, ["teach", "del-A"], {"teach": "teach", "del-A": "del-A.reteach"})
    okc, _nc, _dc = notes_run_ok(r1, S1, ["teach"], {"teach": "teach"})
    C.add("notes: --reference notes,notes_pocket adds PHASE.notes and PHASE.notes_pocket beside teach and del-A, every "
          "answer under its phase, each chat with exactly the block zengine/notes.py gives its question (no system "
          "message where nothing is retrieved), notes from the source and notes_pocket from the folder the phase's own "
          "step taught (teach; del-A's reteach copy), read only and without --slide; notes.jsonl per question within "
          "the cap; the note ids on every answer line; the notes in summary.json and in the report; control: the run "
          "without the flags has none of it", ok and not okc, n, d)

    # what the notes add, through the judge: L1, the lesson learned. The question that shares no word with the lesson
    # ("What is my favourite colour?" against "Of all colours, I like blue the most.") gets no notes, so the untaught
    # model with notes is 2 of 3 on CURRENT, the pocket 3 of 3 and the pocket with notes 3 of 3. Control: a server that
    # drops the system message gives notes 0 of 3 (the pocket with notes stays 3 of 3: its own fact answers)
    lflags = ["--phases", "teach", "--delete-mode", "reteach", "--reference", "notes,notes_pocket"] + nflags
    rg = lab.run("L1-notes", ex("L1-seed1.json"), lflags)
    ri = lab.run("L1-notes-ignore", ex("L1-seed1.json"), lflags, {"ignore_system": True})
    cur = lambda res: {k: (gap_cell(res).get("off") or {}).get(k) for k in ("pocket", "notes", "notes_pocket")}
    b1 = notes_blocks(S1, "teach")
    shares = sum(1 for q in S1["questions"] if q["type"] == "CURRENT" and q["role"] == "probe" and b1[q["qid"]])
    want = {"pocket": [3, 3], "notes": [shares, 3], "notes_pocket": [3, 3]}
    wanti = {"pocket": [3, 3], "notes": [0, 3], "notes_pocket": [3, 3]}
    C.add(f"notes: through the judge on L1 (the lesson learned), CURRENT: the pocket 3/3, the untaught model with notes "
          f"{shares}/3 (the wording that shares no word with the lesson gets no notes), the pocket with notes 3/3, in "
          "the same table as the gap; control: a server that drops the system message gives notes 0/3",
          rg.rc == 0 and shares == 2 and cur(rg) == want and ri.rc == 0 and cur(ri) == wanti, 2,
          f"got {cur(rg)}; control {cur(ri)}")

    # --resume: answers asked with other notes are moved aside and asked again; control: the same flags ask nothing
    rs, rsc = os.path.join(root, "L2-notes-resume"), os.path.join(root, "L2-notes-resume-same")
    for x in (rs, rsc):
        shutil.copytree(rn.work, x, symlinks=True)
    ra = lab.run("L2-notes-resume", ex("L2-seed1.json"), flags + ["--resume", "--notes-cap", "60"])
    rb = lab.run("L2-notes-resume-same", ex("L2-seed1.json"), flags + ["--resume"])
    stale = [x for x in os.listdir(os.path.join(rs, "phases", "teach.notes")) if x.startswith("stale-")]
    log60 = lines(os.path.join(rs, "phases", "teach.notes", "notes.jsonl"))
    C.add("notes: on --resume, answers asked with other notes (here a cap of 60 tokens, which no note fits under the "
          "framing) are moved aside and asked again with today's, now with no system message; control: unchanged "
          "flags keep the answers and ask nothing",
          ra.rc == 0 and len(stale) == 1 and log60 and all(x["cap"] == 60 and not x["notes"] for x in log60)
          and len([c for c in ra.chats if not c["slide"]]) == len(S2["questions"]) * 2 * 4
          and all(c["system"] is None for c in ra.chats) and rb.rc == 0 and not rb.chats, len(ra.chats),
          f"rc {ra.rc}, {len(stale)} stale, {len(ra.chats)} chats; control {len(rb.chats)} chats")

    # the dry run shows the note store, each question's notes and a block, and writes nothing; control: r9
    rd = lab.run("L2-notes-dry", ex("L2-seed1.json"), ["--dry-run"] + flags)
    shown = ("step teach.notes (notes)" in rd.out and "step del-A.notes_pocket (notes_pocket)" in rd.out
             and "the note store of phase teach" in rd.out and "| Notes from memory:" in rd.out
             and "notes none (no system message)" in rd.out and "serve the folder step del-A.reteach taught" in rd.out)
    C.add("notes: --dry-run shows both notes steps, the note store, each question's notes and a block, and the folder "
          "notes_pocket would serve, and writes nothing; control: the dry run without the flags shows none",
          rd.rc == 0 and shown and not os.path.exists(rd.work) and "the note store" not in r9.out, 2, f"rc {rd.rc}")

    # bad notes flags are refused before anything runs; control: a good set of flags is accepted
    bad = [(["--reference", "notes"], "tokenizer.json"),              # the mock model has no tokenizer.json
           (["--reference", "notes", "--notes-tokenizer", os.path.join(root, "none", "tokenizer.json")], "no tokenizer"),
           (["--reference", "notes"] + nflags + ["--notes-cap", "0"], "positive number"),
           (["--reference", "notes"] + nflags + ["--notes-config", '{"nope": 1}'], "--notes-config takes"),
           (["--reference", "notes"] + nflags + ["--notes-config", '{"hop": "sideways"}'], "hop is"),
           (["--notes-cap", "100"], "need --reference notes"),
           # a known setting of the wrong type or out of range (PR #1 review): refused, not a crash in a lookup
           (["--reference", "notes"] + nflags + ["--notes-config", '{"hop_from": 1.5}'], "hop_from is"),
           (["--reference", "notes"] + nflags + ["--notes-config", '{"subject": true}'], "subject is"),
           (["--reference", "notes"] + nflags + ["--notes-config", '{"min_rel": 2}'], "min_rel is"),
           (["--reference", "notes"] + nflags + ["--notes-config", '{"cap": true}'], "cap is")]
    real = [(["--reference", "notes_pocket", "--reference-of", "del-A", "--phases", "teach"] + nflags,
             "which this run does not make")]                        # refused by the run itself, not the dry run
    refused = []
    for i, (fl, why) in enumerate(bad + real):
        rb_ = lab.run(f"L2-notes-bad{i}", ex("L2-seed1.json"), ["--dry-run"] + fl if i < len(bad) else fl)
        refused.append(rb_.rc == 2 and why in (rb_.err + rb_.out) and not rb_.starts and not rb_.chats)
    rk = lab.run("L2-notes-good", ex("L2-seed1.json"), ["--dry-run", "--reference", "notes", "--reference-of", "del-A",
                                                       "--notes-cap", "300"] + nflags)
    C.add("notes: bad flags are refused before anything runs: no tokenizer.json in the source, a missing tokenizer file, "
          "a cap of 0, an unknown setting, a bad hop, a known setting of the wrong type or range (hop_from 1.5, "
          "subject true, min_rel 2, cap true), --notes-cap without a notes phase, and notes_pocket of a phase whose "
          "pocket this run does not make and no earlier run left; control: --reference notes --reference-of del-A "
          "--notes-cap 300 is accepted and plans del-A.notes only", all(refused) and rk.rc == 0
          and "step del-A.notes (notes)" in rk.out and "step teach.notes" not in rk.out, len(bad + real),
          f"refused {refused}")

    # --cleanup (as on Linux): the copies notes_pocket serves are kept until it is done, then removed. Control: with
    # the hold switched off, del-A's reteach copy is removed first and its notes_pocket step fails
    linux = linux_runner(root)
    cflags = flags + ["--thinking", "off", "--cleanup"]
    rc_ = lab.run("L2-notes-cleanup", ex("L2-seed1.json"), cflags, prog=linux)
    st_ = summary_of(rc_.work).get("steps") or {}
    pocket_starts = [x["folder"] for x in rc_.starts if not x["slide"] and "models" in x["folder"]]
    okc = (rc_.rc == 0 and all((st_.get(k) or {}).get("status") == "done" for k in st_)
           and copies_of(rc_.work) == {"teach": (False, True), "del-A.reteach": (False, True)}
           and not model_dirs(rc_.work) and sorted(os.path.basename(x) for x in pocket_starts)
           == ["L2-seed1.del-A.reteach", "L2-seed1.teach"] and "will serve it" in rc_.out)
    rh = lab.run("L2-notes-cleanup-hold-off", ex("L2-seed1.json"), cflags, prog=linux_runner(root, ("hold",)))
    sh = summary_of(rh.work).get("steps") or {}
    ctl = rh.rc == 1 and (sh.get("del-A.notes_pocket") or {}).get("status") == "failed" and "is gone" in (
        (sh.get("del-A.notes_pocket") or {}).get("error") or "")
    C.add("notes: --cleanup (as on Linux): the taught copy and del-A's reteach copy are kept until the notes_pocket "
          "steps that serve them are done, then removed; every step done, no model folder left; control: with the "
          "hold off, del-A's reteach copy goes first and its notes_pocket step fails",
          okc and ctl, 2, f"rc {rc_.rc}, copies {copies_of(rc_.work)}, served {pocket_starts}; control rc {rh.rc} "
          f"{(sh.get('del-A.notes_pocket') or {}).get('error')!r}")


def inplace_checks(C, lwrun, root):
    """Zero mode writes the copy in place (zero_facts). A copy whose shard is a hard link to the source's (as on gb10-a,
    where leftover run copies were deduplicated into hard links to the Hugging Face source) or a symlink to it must
    be refused before any byte is written, both files left byte for byte as they were."""
    targets = [(1, "learned", layer, 0) for layer in POCKET_LAYERS]
    kept = [(2, "learned", layer, 1) for layer in POCKET_LAYERS]

    def attempt(how, guard=True):
        src = os.path.join(root, f"inplace-{how}-{'on' if guard else 'off'}-src")
        make_model(src)
        for k, fill in ((0, 0x11), (1, 0x22)):  # fact 1 on pocket neuron 0, fact 2 (kept) on neuron 1
            Pocket(src).write(k, fill)
        cfg = read_json(os.path.join(src, "config.json"))
        cfg["tensorfold_learned_neurons"] = P
        with open(os.path.join(src, "config.json"), "w") as f:
            json.dump(cfg, f)
        dst = src[:-len("src")] + "copy"
        os.makedirs(dst)
        for x in sorted(os.listdir(src)):
            make = {"hardlink": os.link, "symlink": os.symlink}.get(how) if x == SHARD else None
            (make or shutil.copyfile)(os.path.join(src, x), os.path.join(dst, x))
        sa, sb = os.path.join(src, SHARD), os.path.join(dst, SHARD)
        before = (lwrun.sha256(sa), lwrun.sha256(sb))
        keep, lwrun.INPLACE_GUARD = lwrun.INPLACE_GUARD, guard
        try:
            rep, err = lwrun.zero_facts(dst, targets, kept, ref=src), None
        except lwrun.RunError as e:
            rep, err = None, e
        finally:
            lwrun.INPLACE_GUARD = keep
        return {"refused": isinstance(err, lwrun.SharedFileError), "error": " ".join(str(err).split()) if err else "",
                "rows_zeroed": (rep or {}).get("rows_zeroed"), "source_same": lwrun.sha256(sa) == before[0],
                "copy_same": lwrun.sha256(sb) == before[1], "source_live": Pocket(src).live(0), "shard": sb}

    def brief(x):
        return (f"refused {x['refused']}, rows zeroed {x['rows_zeroed']}, source unchanged {x['source_same']}, copy "
                f"unchanged {x['copy_same']}" + (f": {x['error'].replace(root + os.sep, '')[:200]}" if x["error"] else ""))

    hard, sym = attempt("hardlink"), attempt("symlink")
    C.add("in place: zero mode refuses a copy whose shard is a hard link to the source's (st_nlink 2), before any byte "
          "is written; source and copy byte for byte as before", hard["refused"] and hard["shard"] in hard["error"]
          and "st_nlink 2" in hard["error"] and hard["source_same"] and hard["copy_same"], 1, brief(hard))
    C.add("in place: zero mode refuses a copy whose shard is a symlink to the source's, before any byte is written; "
          "both files byte for byte as before", sym["refused"] and sym["shard"] in sym["error"] and "symlink" in
          sym["error"] and sym["source_same"] and sym["copy_same"], 1, brief(sym))
    offs = {how: attempt(how, guard=False) for how in ("hardlink", "symlink")}
    C.add("negative control: with the guard switched off the same zeroing goes through and changes the linked source "
          "(hard link and symlink)", all(not o["refused"] and o["rows_zeroed"] and not o["source_same"]
                                         and not o["source_live"] for o in offs.values()), 2,
          "; ".join(f"{k}: {brief(o)}" for k, o in offs.items()))
    real = attempt("copy")
    C.add("control: a real copy (its own file) is zeroed with the guard on, and the source is untouched",
          not real["refused"] and real["rows_zeroed"] == 2 * len(PARTS) and real["source_same"]
          and not real["copy_same"] and real["source_live"], 1, brief(real))
    # the open re-checks the file it opened: a link made after the path was checked is still refused
    paths = {"hardlink": hard["shard"], "symlink": sym["shard"]}
    keep, lwrun.shared_file_problem = lwrun.shared_file_problem, lambda _p: None
    got = {}
    try:
        for how, p in paths.items():
            try:
                lwrun.open_for_inplace_write(p).close()
                got[how] = "opened"
            except lwrun.SharedFileError:
                got[how] = "refused"
    finally:
        lwrun.shared_file_problem = keep
    C.add("in place: with the path check patched out, the open itself still refuses (the opened file's st_nlink, and "
          "O_NOFOLLOW for a symlink)", got == {"hardlink": "refused", "symlink": "refused"}, 2, f"{got}")


def selftest(root, records=None):
    sys.path.insert(0, HERE)
    import lwrun  # noqa: E402  (the module under test)
    t0 = time.time()
    C = Check()
    print(f"lwrun selftest in {root}")
    lab = Lab(root)

    # 1. the post-save line, unit level: 1.0.8's two shapes parse; a log line quoting the model never matches
    log = os.path.join(root, "unit-server.log")
    real = [
        "info: slide pocket: the learned neurons are in the model's shards (261 MB written in 18.4 s); the last "
        "fact's held-out answers do not all come back with them as written",
        "info: slide pocket: the learned neurons are in the model's shards (250 MB written in 0.4 s); the last "
        "fact's held-out answers come back with them as written"]
    quoted = ('[tensorfold] slide: "Where are the shards?" is learned as the model answers it knowing the fact: '
              "info: slide pocket shards")
    other = "[tensorfold] slide: fact 2's lesson written in 31.7 s"
    half = real[0][:80]  # a matching line the server is still writing
    with open(log, "w") as f:
        f.write(real[0] + "\n")
        mark = f.tell()
        f.write(other + "\n" + real[1] + "\n" + quoted + "\n")
        mark2 = f.tell()
        f.write(half)
    a, b, c = lwrun.find_save_line(log, mark), lwrun.find_save_line(log, 0), lwrun.find_save_line(log, mark2)
    with open(log, "a") as f:
        f.write("\n")
    d = lwrun.find_save_line(log, mark2)
    # another wording of the ending (a later version, say) gives no verdict rather than a guessed one
    odd = [lwrun.parse_save_line(real[0].replace("do not all come back", x)) for x in ("don't come back", "none come back")]
    ok = (a and a["line"] == real[1] and a["mb"] == 250 and a["seconds"] == 0.4 and a["held_out_back"] is True
          and b["line"] == real[1] and lwrun.parse_save_line(real[0])["held_out_back"] is False and c is None
          and d and d["line"] == half and all("held_out_back" not in x and x["mb"] == 261 for x in odd))
    C.add("save line: 1.0.8's post-save line parsed (MB, seconds, held-out answers back or not), only after the "
          "lesson's log offset; quoted model text never matches, a half-written line only once complete; an unknown "
          "ending gives no held-out verdict", bool(ok), 7, "" if ok else f"got {a}, {b}, {c}, {d}, {odd}")

    # 2. one full run, both delete modes, L1: save line, empty-store reuse, kept-stable folders
    r1 = lab.run("L1-both", ex("L1-seed1.json"))
    C.add("L1 run against the mock, both delete modes, finishes with exit code 0", r1.rc == 0, 1,
          f"{r1.seconds} s" + ("" if r1.rc == 0 else f"; rc {r1.rc}: {r1.err.strip()[-300:]}"))
    ok, n, d = save_lines_ok(r1.work)
    C.add("save line: every lesson that saved bytes has the server's post-save line in its record and the summary",
          ok, n, d)
    r2 = lab.run("L1-no-save-line", ex("L1-seed1.json"),
                 ["--phases", "teach", "--delete-mode", "reteach", "--thinking", "off", "--save-line-wait", "1"],
                 {"no_save_line": True})
    ok2, n2, d2 = save_lines_ok(r2.work)
    C.add("negative control: a server that logs no post-save line fails the save-line check (and the runner warns)",
          r2.rc == 0 and not ok2 and "shows no post-save line" in r2.out, n2, d2)
    # missed lessons, as on 1.0.8: the first one saves 0 bytes (no line, no wait); one after a learned fact still
    # writes the shards and logs the line. A wait of --save-line-wait on the 0-byte save would show in the time.
    S2 = read_json(ex("L2-seed1.json"))
    rm = lab.run("L2-missed", ex("L2-seed1.json"),
                 ["--phases", "teach", "--delete-mode", "reteach", "--thinking", "off", "--save-line-wait", "30"],
                 {"miss": [S2["lessons"][0]["text"], S2["lessons"][2]["text"]]})
    okm, nm, dm = save_lines_ok(rm.work)
    ls = (read_json(os.path.join(rm.work, "phases", "teach", "state.json"), {}) or {}).get("lessons") or []
    shape = [(x["recalled"], int((x.get("save") or {}).get("bytes") or 0) > 0, bool(x.get("save_log"))) for x in ls]
    C.add("save line: a missed first lesson saves 0 bytes and gets no line, without a wait or a warning; a missed "
          "lesson after a learned one still saves and gets its line",
          rm.rc == 0 and okm and shape == [(False, False, False), (True, True, True), (False, True, True)]
          and "shows no post-save line" not in rm.out and rm.seconds < 25, nm,
          f"(recalled, wrote bytes, line) per lesson {shape}, {rm.seconds} s" + (f"; {dm}" if dm else ""))

    ok, n, d = reuse_ok(lab, r1)
    C.add("empty store: the baseline answers are reused under the delete phase, judged, and no server is started", ok,
          n, d)
    r3 = lab.run("L1-fresh-empty", ex("L1-seed1.json"),
                 ["--phases", "baseline,del-A", "--delete-mode", "reteach", "--thinking", "off", "--fresh-empty-reteach"])
    ok3, n3, d3 = reuse_ok(lab, r3)
    C.add("negative control: --fresh-empty-reteach serves the original model again and fails the reuse check",
          r3.rc == 0 and not ok3 and len(r3.starts) == 2, n3, d3)
    r4 = lab.run("L1-no-baseline", ex("L1-seed1.json"), ["--phases", "del-A", "--delete-mode", "reteach", "--thinking", "off"])
    st4 = read_json(os.path.join(r4.work, "phases", "del-A.reteach", "state.json"), {}) or {}
    C.add("empty store with no baseline on file: the original model is served instead, and the reason is recorded",
          r4.rc == 0 and st4.get("reuse_skipped") == "no baseline answers on file" and len(r4.starts) == 1
          and len(lines(os.path.join(r4.work, "phases", "del-A.reteach", "answers.jsonl"))) == 12, 1,
          f"reuse_skipped {st4.get('reuse_skipped')!r}, {len(r4.starts)} server start(s)")

    ok, n, d = kept_stable_ok(r1.work, ["zero", "reteach"])
    C.add("kept-stable: WORK/kept-stable/MODE/{teach,del-A}.jsonl per delete mode, complete, phase names kept, listed "
          "in manifest.json", ok, n, d)
    ok, n, d = judge_reads(r1.work, "zero")
    C.add("kept-stable: `lwlogic.py judge SET kept-stable/zero/*.jsonl` reads teach and delete answers in one call", ok,
          n, d)
    moved = os.path.join(r1.work, "kept-stable", "zero", "del-A.jsonl")
    os.rename(moved, moved + ".moved-by-selftest")
    try:
        okn, nn, dn = kept_stable_ok(r1.work, ["zero", "reteach"])
    finally:
        os.rename(moved + ".moved-by-selftest", moved)
    C.add("negative control: with zero/del-A.jsonl moved aside the kept-stable check fails", not okn, nn, dn)
    # a judge that writes a kept_stable score (filled, or {} as the judge does for a call without teach answers),
    # run over a copy of the L1 records with --judge-only: the score reaches summary.json only when it holds something
    got = {}
    for tag, score in (("filled", {"del-A": {"off": [3, 4]}}), ("empty", {})):
        fake = os.path.join(root, f"fake-judge-{tag}.py")
        with open(fake, "w") as f:
            f.write("import json, sys\nprint('judged 0 answers')\n"
                    f"json.dump({{'results': [], 'kept_stable': {score!r}}}, open(sys.argv[sys.argv.index('--json') + 1], 'w'))\n")
        shutil.copytree(r1.work, os.path.join(root, f"L1-judge-{tag}"), symlinks=True)
        rj = lab.run(f"L1-judge-{tag}", ex("L1-seed1.json"), ["--judge-only", "--lwlogic", fake])
        comb = (read_json(os.path.join(rj.work, "summary.json"), {}) or {}).get("combined") or {}
        got[tag] = (rj.rc, sorted(comb), [c.get("kept_stable") for c in comb.values()])
    C.add("kept-stable: a kept_stable score in the judge's JSON is carried into summary.json (combined, per mode); "
          "control: an empty one ({}) is not", got["filled"] == (0, ["reteach", "zero"], [{"del-A": {"off": [3, 4]}}] * 2)
          and got["empty"] == (0, ["reteach", "zero"], [None, None]), 2, f"got {got}")

    # 3. zero limits on the #678 shape: F4 replaced F3, F5 replaced F4 (as 1.0.8 did on the GB10)
    S6 = read_json(ex("eyes-678.json"))
    text = {l["id"]: l["text"] for l in S6["lessons"]}
    r5 = lab.run("eyes-zero", ex("eyes-678.json"), ["--phases", "teach,rm-4,rm-5,rm-345", "--delete-mode", "zero",
                                                    "--thinking", "off"],
                 {"replaces": {text[4]: text[3], text[5]: text[4]}, "save_line_delay": 1.0})
    lim = {k: (read_json(os.path.join(r5.work, "phases", f"{k}.zero", "state.json"), {}) or {}).get("zero_limits")
           for k in ("rm-4", "rm-5", "rm-345")}
    zr = {k: (read_json(os.path.join(r5.work, "phases", f"{k}.zero", "zero.json"), {}) or {}).get("rows_zeroed")
          for k in lim}

    def flagged(limits):
        return {k: sorted(x["lesson"] for x in (v or {}).get("cannot_restore", [])) for k, v in limits.items()}

    # F5 replaced F4, which had replaced F3. Deleting F5 loses F4 (kept): flagged. Deleting F4 does not lose F3: kept
    # F5 replaced F4 in turn, and never taught F4, F5 replaces F3 itself (the stored #678 reteach of rm-4 did).
    want = {"rm-4": [], "rm-5": [4], "rm-345": []}
    ok = (r5.rc == 0 and flagged(lim) == want and lim["rm-4"]["already_zero"] and not lim["rm-5"]["already_zero"]
          and zr["rm-4"] == 0 and zr["rm-5"] > 0)
    C.add("zero limits: deleting F5 flags the kept F4 it had replaced; deleting F4 flags nothing (kept F5 replaced it "
          "in turn); deleting all three flags none; an already-replaced fact zeroes 0 rows", ok, 3,
          f"flagged {flagged(lim)}, rows zeroed {zr}")
    ok, n, d = save_lines_ok(r5.work)
    C.add("save line: a line logged 1 s after the save is still found (the runner waits for it)", r5.rc == 0 and ok, n, d)
    tst = read_json(os.path.join(r5.work, "phases", "teach", "state.json"), {}) or {}
    blind = json.loads(json.dumps(tst))
    for rec in blind.get("lessons") or []:
        rec["changed_states"] = []
    got = flagged({k: lwrun.zero_limits(blind, p["deleted"]) for p in S6["phases"] for k in [p["name"]] if k in want})
    no_f5 = json.loads(json.dumps(tst))  # only F5's note gone: nothing kept replaced F4, so deleting F4 loses F3 again
    for rec in no_f5.get("lessons") or []:
        if rec["lesson"] == 5:
            rec["changed_states"] = []
    got5 = flagged({k: lwrun.zero_limits(no_f5, p["deleted"]) for p in S6["phases"] for k in [p["name"]] if k in want})
    C.add("negative control: without the server's replace notes the zero-limits check finds nothing to flag; without "
          "only F5's note, deleting F4 flags F3", got != want and got5 == {"rm-4": [3], "rm-5": [], "rm-345": []}, 6,
          f"flagged {got}; without F5's note {got5}")

    # 4. --cleanup and zero steps (as on Linux): a teach step that stored nothing, --resume, a skipped and an
    # unfinished zero step
    cleanup_checks(C, lab, root)

    # 5. --sets: two sets in one call; the first fails in teach, the second still runs; one summary
    sets = [ex("L1-seed1.json"), ex("L2-seed1.json")]
    r6 = lab.run("sets", sets, ["--thinking", "off", "--delete-mode", "reteach"],
                 {"fail_learn": [read_json(sets[0])["lessons"][0]["text"]]})
    ss = read_json(os.path.join(r6.work, "sets-summary.json"), {}) or {}
    rows = {x["set"]: x for x in ss.get("sets", [])}
    l1, l2 = rows.get("L1-seed1") or {}, rows.get("L2-seed1") or {}
    tbl = open(os.path.join(r6.work, "sets.log")).read() if os.path.exists(os.path.join(r6.work, "sets.log")) else ""
    ok = (r6.rc == 1 and ss.get("rc") == 1 and l1.get("rc") == 1 and l1.get("failed_steps") == ["teach"]
          and l2.get("rc") == 0 and sorted(l2.get("steps", {})) == ["baseline", "del-A.reteach", "del-B.reteach", "teach"]
          and all(v["status"] == "done" for v in l2["steps"].values())
          and all(os.path.exists(os.path.join(r6.work, s, "summary.json")) for s in rows)
          and "L1-seed1" in tbl and "L2-seed1" in tbl and "lessons recalled" in tbl)
    C.add("--sets: two sets one after another, each in WORK/SET_ID; a failed teach in the first does not stop the "
          "second; one sets-summary.json and table", ok, len(rows),
          f"rc {r6.rc}; per set {[(x['set'], x['rc'], x.get('failed_steps')) for x in ss.get('sets', [])]}")
    r7 = lab.run("sets-dup", [ex("L1-seed1.json"), ex("L1-seed1.json")], ["--thinking", "off"])
    os.makedirs(os.path.join(root, "sets-left", "L2-seed1", "phases", "baseline"))
    r8 = lab.run("sets-left", sets, ["--thinking", "off"])
    esc = read_json(ex("L2-seed1.json"))
    esc["id"] = "../escaped"  # the set id names the folder WORK/SET_ID: it must not lead out of WORK
    bad_id = os.path.join(root, "bad-id.json")
    with open(bad_id, "w") as f:
        json.dump(esc, f)
    r10 = lab.run("sets-bad-id", [ex("L1-seed1.json"), bad_id], ["--thinking", "off"])
    C.add("negative control: --sets refuses a repeated set id, a leftover work dir of a later set, and a set id that "
          "is not a plain folder name, before any set starts",
          r7.rc == 2 and not os.path.exists(r7.work) and "again" in r7.err and r8.rc == 2
          and not os.path.exists(os.path.join(r8.work, "L1-seed1")) and not r7.starts and not r8.starts
          and r10.rc == 2 and "cannot name a folder" in r10.err and not os.path.exists(r10.work)
          and not os.path.exists(os.path.join(root, "escaped")) and not r10.starts, 3, f"rc {r7.rc}, {r8.rc} and {r10.rc}")
    r9 = lab.run("sets-dry", sets, ["--dry-run", "--thinking", "off"])
    C.add("--sets --dry-run checks both sets and writes nothing", r9.rc == 0 and not os.path.exists(r9.work)
          and r9.out.count("dry run: nothing was copied") == 2, 2, f"rc {r9.rc}")

    # 6. reference phases (TEST-LOGIC.md section 8): in_context and replacement_memory beside the pocket
    reference_checks(C, lab, lwrun, root, r1, r9)

    # 6b. the notes phases (docs/NOTES-LAYER.md): the retrieved notes beside the pocket, and with it
    notes_checks(C, lab, lwrun, root, r1, r9)

    # 7. writes in place: never into a shard that is a hard link or a symlink
    inplace_checks(C, lwrun, root)

    # 8. optional: stored run records (read only)
    if records:
        for sp in sorted(glob.glob(os.path.join(records, "**", "phases", "teach", "state.json"), recursive=True)):
            work = os.path.dirname(os.path.dirname(os.path.dirname(sp)))
            S = read_json(os.path.join(work, "set.json")) or {}
            tst = read_json(sp, {}) or {}
            if S.get("id") == "eyes-678" and tst.get("taught"):
                got = {p["name"]: sorted(x["lesson"] for x in lwrun.zero_limits(tst, p["deleted"])["cannot_restore"])
                       for p in S["phases"][2:]}
                C.add(f"records: zero limits of a stored #678 run ({os.path.basename(work)}): rm-5 flagged, rm-3, "
                      "rm-4, rm-345 and rm-all not", got == {"rm-3": [], "rm-4": [], "rm-5": [4], "rm-345": [],
                                                             "rm-all": []}, len(got), f"{got}")
        # the stored zero steps the first built runner failed with "holds no pocket": the teach records of each say
        # nothing was stored, so the runner now calls them not applicable. Control: no teach record with a learned
        # fact says so
        na, held, bad, seen = 0, 0, [], set()
        for sp in sorted(glob.glob(os.path.join(records, "**", "phases", "teach", "state.json"), recursive=True)):
            work = os.path.dirname(os.path.dirname(os.path.dirname(sp)))
            tst = read_json(sp, {}) or {}
            ident = ((read_json(os.path.join(work, "set.json")) or {}).get("id"), tst.get("created"))
            if not tst.get("taught") or ident in seen:
                continue
            seen.add(ident)
            said = lwrun.nothing_stored(tst)
            for zp in sorted(glob.glob(os.path.join(work, "phases", "*.zero", "state.json"))):
                if lwrun.NO_POCKET in ((read_json(zp, {}) or {}).get("zero_skipped") or ""):
                    na += 1
                    if said is None:
                        bad.append(f"{os.path.relpath(zp, records)}: the teach records say something was stored")
            if any(s == "learned" for s in (tst.get("fact_states") or {}).values()):
                held += 1
                if said is not None:
                    bad.append(f"{os.path.relpath(sp, records)}: a learned fact, yet {said!r}")
        if na:
            C.add(f"records: the {na} stored zero steps skipped as \"the folder holds no pocket\" all have teach records "
                  f"that say nothing was stored (not applicable now); control: none of the {held} teach records with a "
                  "learned fact says so", not bad, na + held, "; ".join(bad[:3]))
        same = total = 0
        seen = set()  # a run folder copied elsewhere counts once: its step state is the same file
        for sp in sorted(glob.glob(os.path.join(records, "**", "phases", "*", "state.json"), recursive=True)):
            st = read_json(sp, {}) or {}
            if st.get("kind") != "reteach-empty" or st.get("reused_baseline"):
                continue
            work = os.path.dirname(os.path.dirname(os.path.dirname(sp)))
            ident = ((read_json(os.path.join(work, "set.json")) or {}).get("id"), st.get("key"), st.get("created"))
            if ident in seen:
                continue
            seen.add(ident)
            base = {(x["qid"], x["thinking"]): x["content"] for x in lines(os.path.join(work, "phases", "baseline", "answers.jsonl"))}
            got = lines(os.path.join(os.path.dirname(sp), "answers.jsonl"))
            total += len(got)
            same += sum(base.get((x["qid"], x["thinking"])) == x["content"] for x in got)
        if seen:
            print(f"  info  records: {len(seen)} empty-store steps that served the original model again gave the "
                  f"baseline answers word for word in {same} of {total} answers")
        # pocket weight norms read from the stored taught folders and their zeroed copies, read only: every learned
        # fact's input rows are nonzero in the taught folder, every zeroed fact's are zero in its zero copy, and a
        # neuron no fact uses has an input row of zeros
        rows, bad = 0, []
        for sp in sorted(glob.glob(os.path.join(records, "**", "phases", "teach", "state.json"), recursive=True)):
            work = os.path.dirname(os.path.dirname(os.path.dirname(sp)))
            tst = read_json(sp, {}) or {}
            sid = (read_json(os.path.join(work, "set.json")) or {}).get("id")
            taught = os.path.join(work, "models", f"{sid}.teach")
            neur = tst.get("fact_neurons") or {}
            if not (sid and os.path.isdir(taught) and neur):
                continue
            states = tst.get("fact_states") or {}
            try:
                nm = lwrun.pocket_norms(taught, neur)
                rd = lwrun.PocketReader(taught)
                used = {(int(n["layer"]), int(n["neuron"])) for ns in neur.values() for n in ns}
                free = next((layer, k) for layer, _ in sorted(used) for k in range(64) if (layer, k) not in used)
                if rd.up_norm2(*free) != 0.0:
                    bad.append(f"{sid}: unused neuron {free} has a nonzero input row")
            except lwrun.ZeroSkip as e:
                bad.append(f"{sid}: {e}")
                continue
            for f, x in nm.items():
                rows += 1
                if (states.get(f) == "learned") != (x["up"] > 0):
                    bad.append(f"{sid} fact {f} ({states.get(f)}): input norm {x['up']}")
            for zp in sorted(glob.glob(os.path.join(work, "phases", "*.zero", "zero.json"))):
                z = read_json(zp, {}) or {}
                zf = os.path.join(work, "models", f"{sid}.{os.path.basename(os.path.dirname(zp))}")
                if os.path.isdir(zf):
                    for f in z.get("facts") or []:
                        if str(f) in neur:
                            rows += 1
                            if lwrun.pocket_norms(zf, {str(f): neur[str(f)]})[str(f)]["up"] != 0.0:
                                bad.append(f"{sid} zero copy: fact {f} still has input weights")
        if rows:
            C.add("records: pocket weight norms read from the stored taught folders, read only: learned facts' input "
                  "rows nonzero, zeroed facts' rows zero in their zero copies, an unused neuron's row zero", not bad,
                  rows, "; ".join(bad[:3]))

    ok = all(r[1] for r in C.rows)
    print(f"lwrun selftest {'PASSED' if ok else 'FAILED'}: {sum(r[1] for r in C.rows)}/{len(C.rows)} checks, "
          f"{time.time() - t0:.0f} s; work folders kept in {root}")
    return ok


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["mock"]:
        return mock_main(argv[1:])
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", help="a new folder for the test's work folders (default: a new temp folder)")
    ap.add_argument("--records", help="optional: a folder of stored lwrun work folders to read (never written)")
    a = ap.parse_args(argv)
    root = os.path.abspath(a.dir) if a.dir else tempfile.mkdtemp(prefix="lwrun-selftest-")
    if a.dir and os.path.exists(root) and os.listdir(root):
        print(f"{root} is not empty: give a new folder", file=sys.stderr)
        return 2
    return 0 if selftest(root, a.records) else 1


if __name__ == "__main__":
    sys.exit(main())
