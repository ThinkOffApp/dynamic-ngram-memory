import sys
src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()
rep = [
 ('''"""gb10-a: phase 3, the recapture of a loaded pocket at count 0.''',
  '''"""1.0.9 receipt, adapted from live/live3.py: phase 3, the recapture of a loaded pocket at count 0.'''),
 ('''usage: live3.py BUILD(before|tip|mutant|nr) [REF_BUILD]
nr: adec42df with no recapture at all (srcN), a calibration: lesson A, then its live probe only.
inst: adec42df with log lines at each count change and recapture (srcI), not otherwise changed.
tip: adec42df (srcC). mutant: adec42df with the recapture only for a pocket made now, as 1.0.8 had it (srcM).
With REF_BUILD every checkpoint is compared with it; differences are recorded, and stop the run unless BUILD is mutant.''',
  '''usage: live109.py BUILD(before|after) [REF_BUILD]   or   live109.py restart FOLDER_BUILD BINARY_BUILD
before: c36b536d (r109/before). after: 76fe1de4 (r109/after).
With REF_BUILD every checkpoint is compared with it; differences are recorded (the run does not stop on them).
Events and slide graphs are compared without time keys; graph_diff lists the paths that still differ.'''),
 ('OUT = H + "/live"', 'R = H + "/r109"\nOUT = R + "/live"'),
 ('''BINS = {"before": H + "/before/zig-out/native/bin/tensorfold-native", "tip": H + "/srcC/zig-out/native/bin/tensorfold-native",
        "mutant": H + "/srcM/zig-out/native/bin/tensorfold-native", "nr": H + "/srcN/zig-out/native/bin/tensorfold-native",
        "inst": H + "/srcI/zig-out/native/bin/tensorfold-native",
        "src": H + "/src/zig-out/native/bin/tensorfold-native"}''',
  '''BINS = {"before": R + "/before/zig-out/native/bin/tensorfold-native", "after": R + "/after/zig-out/native/bin/tensorfold-native"}'''),
 ('''def diff_ck(ck, ref):
    out = []
    if ck.get("sha") != ref.get("sha"):
        out.append(["files", sorted(n for n in set(ck["sha"]) | set(ref["sha"]) if ck["sha"].get(n) != ref["sha"].get(n))])
    if ck.get("events") != ref.get("events"):
        out.append(["events", ck.get("events"), ref.get("events")])
    if ck.get("graph") != ref.get("graph"):
        out.append(["graph", "stripped slide graphs differ"])''',
  '''def paths(a, b, p=""):
    if type(a) is not type(b):
        return [p or "/"]
    if isinstance(a, dict):
        return [x for k in sorted(set(a) | set(b)) for x in (paths(a.get(k), b.get(k), p + "/" + str(k)) if k in a and k in b else [p + "/" + str(k)])]
    if isinstance(a, list):
        if len(a) != len(b):
            return [p + "[len]"]
        return [x for i, (u, v) in enumerate(zip(a, b)) for x in paths(u, v, "%s[%d]" % (p, i))]
    return [] if a == b else [p or "/"]


def diff_ck(ck, ref):
    out = []
    if ck.get("sha") != ref.get("sha"):
        out.append(["files", sorted(n for n in set(ck["sha"]) | set(ref["sha"]) if ck["sha"].get(n) != ref["sha"].get(n))])
    if strip(ck.get("events")) != strip(ref.get("events")):
        out.append(["events", ck.get("events"), ref.get("events")])
    if ck.get("graph") != ref.get("graph"):
        out.append(["graph", "stripped slide graphs differ", paths(ck.get("graph"), ref.get("graph"))[:40]])'''),
 ('''    sha0 = sha_all(folder)
    live = json.load(open(d + "/ckB2.json"))''',
  '''    sha0 = sha_all(folder)
    live = json.load(open(d + "/ckB2.json"))
    rec_b = {"folder_build": build, "binary_build": binary}'''),
 ('''            if build not in ("mutant",):
                raise Abort("%s differs from %s" % (name, ref_build))''',
  '''            # recorded, not fatal: the receipt wants every checkpoint'''),
 ('''    for ln in open(OUT + "/live-before.l0.sha256"):''', '''    for ln in open(H + "/live/live-before.l0.sha256"):'''),
]
for a, b in rep:
    assert s.count(a) == 1, a[:70]
    s = s.replace(a, b)
# drop the nr branch (unused) is harmless; keep it.
open(dst, "w").write(s)
