import sys
src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()
rep = [
 ('"""gb10-a: one server at a time on port 18190, before (0efbd447 build) or after (pocket-cuda build), folder served in place.',
  '"""1.0.9 receipt: one server at a time on port 18190, before (c36b536d build) or after (76fe1de4 build), folder served in place.'),
 ('The server gets a fresh empty HOME of its own', 'Clients: tools/ of the c36b536d tree (1.0.9). The server gets a fresh empty HOME of its own'),
 ('TOOLS = H + "/src108/tools"', 'R = H + "/r109"\nTOOLS = R + "/before/tools"'),
 ('''BINS = {"before": (H + "/before/zig-out/native/bin/tensorfold-native", "b0a5a4f312a58dda536e5f5c97699e0ebf211e08319cceca7555e8560b9ba333"),
        "after": (H + "/src/zig-out/native/bin/tensorfold-native", "4f512013d0181a1a68fcf0cee4dd2243f6997914429e56ac8ae2b35f616ab30e")}''',
  '''def _rec_sha(b):
    for ln in open(R + "/raw/bin-%s.sha256" % b):
        s, n = ln.split(None, 1)
        if n.strip() == "zig-out/native/bin/tensorfold-native":
            return s
    raise RuntimeError("no recorded sha256 for " + b)


BINS = {b: (R + "/" + b + "/zig-out/native/bin/tensorfold-native", _rec_sha(b)) for b in ("before", "after")}'''),
 ('out_dir = H + "/raw/v/" + series', 'out_dir = R + "/raw/" + series'),
 ('home = H + "/homes/%s-%s" % (series, label)', 'home = R + "/homes/%s-%s" % (series, label)'),
]
for a, b in rep:
    assert s.count(a) == 1, a[:60]
    s = s.replace(a, b)
open(dst, "w").write(s)
