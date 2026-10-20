#!/usr/bin/env python3
"""An empty-pocket folder from the published one, without copying it: the published shards by symlink, plus one
small shard that holds every widened shared expert (own bytes as published, 64 pocket neurons all zero).

usage: mk_empty.py SOURCE OUT [--neurons 64]
       mk_empty.py --check SOURCE TAUGHT OUT

Why this loads as an ordinary widened folder (READ, zig/src/core/checkpoint.zig): the engine opens the index's shard
files in name order and takes each tensor from the first file that holds it, and counts a name as used once taken.
The new shard is named to sort first, so its widened shared experts win; the published unwidened ones are never read.
An all-zero neuron is the bytes the engine itself writes for a layer with no fact (affine4_host.quantize of zeros:
words, scales and biases 0; pocket_shards.zig widens every MoE layer and the MTP head that way).
The MTP head is its own file (mtp-4bit.safetensors), loaded alone, so it is written whole with its expert widened.
"""
import json, os, shutil, struct, sys

GROUP = 64
OVERLAY = "model-00000-pocket.safetensors"
EL = {"U32": 4, "BF16": 2, "F32": 4, "F16": 2, "U8": 1, "I32": 4}


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    meta = h.pop("__metadata__", None)
    return 8 + n, h, meta


def read_at(path, off, ln):
    with open(path, "rb") as f:
        f.seek(off)
        return f.read(ln)


def is_shared(name):
    return ".shared_experts.up_proj." in name or ".shared_experts.down_proj." in name


def widen(name, dtype, shape, data, n):
    rows, cols = shape
    el = EL[dtype]
    if ".up_proj." in name:
        return [rows + n, cols], data + b"\0" * (n * cols * el)
    unit = 8 if dtype == "U32" else GROUP  # inputs one stored column holds
    add = n // unit
    per = cols * el
    out = bytearray()
    for r in range(rows):
        out += data[r * per:(r + 1) * per]
        out += b"\0" * (add * el)
    return [rows, cols + add], bytes(out)


def write_st(path, items, meta):
    """items: [(name, dtype, shape, bytes)] in order."""
    hdr, off = {}, 0
    if meta is not None:
        hdr["__metadata__"] = meta
    for name, dtype, shape, data in items:
        hdr[name] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + len(data)]}
        off += len(data)
    hb = json.dumps(hdr, separators=(",", ":")).encode()
    hb += b" " * ((8 - len(hb) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for _, _, _, data in items:
            f.write(data)
        f.flush()
        os.fsync(f.fileno())


def build(src, out, n):
    if os.path.exists(out):
        sys.exit("refusing: %s exists" % out)
    os.makedirs(out)
    idx = json.load(open(os.path.join(src, "model.safetensors.index.json")))
    files = sorted(set(idx["weight_map"].values()))
    items = []
    for fn in files:
        p = os.path.realpath(os.path.join(src, fn))
        base, h, _ = header(p)
        for name in sorted(h):
            if is_shared(name):
                v = h[name]
                a, b = v["data_offsets"]
                shape, data = widen(name, v["dtype"], v["shape"], read_at(p, base + a, b - a), n)
                items.append((name, v["dtype"], shape, data))
        os.symlink(p, os.path.join(out, fn))
    write_st(os.path.join(out, OVERLAY), items, {"format": "mlx"})
    for name, *_ in items:
        idx["weight_map"][name] = OVERLAY
    json.dump(idx, open(os.path.join(out, "model.safetensors.index.json"), "w"), indent=2)
    # the MTP head, whole, its shared expert widened
    mp = os.path.realpath(os.path.join(src, "mtp-4bit.safetensors"))
    base, h, meta = header(mp)
    mitems = []
    for name in sorted(h, key=lambda k: h[k]["data_offsets"][0]):
        v = h[name]
        a, b = v["data_offsets"]
        data = read_at(mp, base + a, b - a)
        shape = v["shape"]
        if is_shared(name):
            shape, data = widen(name, v["dtype"], shape, data, n)
        mitems.append((name, v["dtype"], shape, data))
    write_st(os.path.join(out, "mtp-4bit.safetensors"), mitems, meta)
    cfg = json.load(open(os.path.join(src, "config.json")))
    cfg["moe_shared_expert_intermediate_size"] += n
    cfg["tensorfold_learned_neurons"] = n
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=2)
    for fn in os.listdir(src):
        if fn.endswith(".safetensors") or fn in ("config.json", "model.safetensors.index.json", ".gitattributes"):
            continue
        shutil.copyfile(os.path.join(src, fn), os.path.join(out, fn))
    print("built %s: %d widened tensors in %s, MTP head %d tensors" % (out, len(items), OVERLAY, len(mitems)))


def resolved(folder):
    """name -> (path, offset, nbytes, dtype, shape) as the engine resolves it: first file in name order."""
    idx = json.load(open(os.path.join(folder, "model.safetensors.index.json")))
    out = {}
    for fn in sorted(set(idx["weight_map"].values())):
        p = os.path.join(folder, fn)
        base, h, _ = header(p)
        for k, v in h.items():
            if k not in out:
                a, b = v["data_offsets"]
                out[k] = (p, base + a, b - a, v["dtype"], v["shape"])
    p = os.path.join(folder, "mtp-4bit.safetensors")
    base, h, _ = header(p)
    for k, v in h.items():
        a, b = v["data_offsets"]
        out["mtp:" + k] = (p, base + a, b - a, v["dtype"], v["shape"])
    return out


def split(name, t, own, n, data):
    """(own bytes, pocket bytes) of a widened shared expert tensor; (data, b'') otherwise."""
    if not is_shared(name) or t[4][0 if ".up_proj." in name else 1] in (own, own // 8, own // GROUP):
        return data, b""
    rows, cols = t[4]
    el = EL[t[3]]
    if ".up_proj." in name:
        return data[:own * cols * el], data[own * cols * el:]
    unit = 8 if t[3] == "U32" else GROUP
    keep, per = own // unit * el, cols * el
    return (b"".join(data[r * per:r * per + keep] for r in range(rows)),
            b"".join(data[r * per + keep:(r + 1) * per] for r in range(rows)))


def check(src, taught, out):
    cfg = json.load(open(os.path.join(taught, "config.json")))
    n = cfg["tensorfold_learned_neurons"]
    own = cfg["moe_shared_expert_intermediate_size"] - n
    rs, rt, re_ = resolved(src), resolved(taught), resolved(out)
    rep = {"neurons": n, "own": own, "names_equal": set(rs) == set(rt) == set(re_), "taught_own_equals_source": True,
           "empty_own_equals_source": True, "empty_pocket_all_zero": True, "taught_pocket_used": {}, "mismatch": []}
    cs, ce = json.load(open(os.path.join(src, "config.json"))), json.load(open(os.path.join(out, "config.json")))
    rep["config_keys_changed_taught"] = sorted(k for k in set(cs) | set(cfg) if cs.get(k) != cfg.get(k))
    rep["config_keys_changed_empty"] = sorted(k for k in set(cs) | set(ce) if cs.get(k) != ce.get(k))
    rep["config_empty_equals_taught"] = ce == cfg
    for name in sorted(rs):
        s = read_at(*rs[name][:3])
        for label, tab, key in (("taught", rt, "taught_own_equals_source"), ("empty", re_, "empty_own_equals_source")):
            t = tab[name]
            own_b, pocket_b = split(name, t, own, n, read_at(*t[:3]))
            if own_b != s:
                rep[key] = False
                rep["mismatch"].append([label, name])
            if label == "empty" and pocket_b.count(0) != len(pocket_b):
                rep["empty_pocket_all_zero"] = False
            if label == "taught" and name.endswith("shared_experts.up_proj.scales") and pocket_b:
                cols = t[4][1]
                used = 0
                for k in range(n):
                    if pocket_b[k * cols * 2:(k + 1) * cols * 2].count(0) != cols * 2:
                        used = k + 1
                rep["taught_pocket_used"][name.split(".mixer")[0]] = used
    print(json.dumps(rep, indent=1))
    json.dump(rep, open(out.rstrip("/") + ".check.json", "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "--check":
        check(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        build(sys.argv[1], sys.argv[2], int(sys.argv[sys.argv.index("--neurons") + 1]) if "--neurons" in sys.argv else 64)
