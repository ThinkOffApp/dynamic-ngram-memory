#!/usr/bin/env python3
"""The chat replies alongside a lesson for some prompts and modes, in both builds: time, phase, sha, quiet class, text."""
import json, os, sys
H = os.path.expanduser("~") + "/tf-pocket-cuda/live"
k = int(sys.argv[1]); pids = sys.argv[2].split(",")


def text(r):
    return (r.get("reasoning") or "") + "\x00" + (r.get("content") or "")


def prefix(t, q):
    if q is None or not t:
        return False
    r, c = t.split("\x00", 1)
    qr, qc = q.split("\x00", 1)
    return (qr == r and qc.startswith(c)) if c else qr.startswith(r)


for bd in ("before", "after"):
    x = json.load(open("%s/%s/ck%d.json" % (H, bd, k)))
    pre = {(r["pid"], r["thinking"], r["draft"]): text(r) for r in json.load(open("%s/before/ck%d.json" % (H, k - 1)))["probe"]}
    post = {(r["pid"], r["thinking"], r["draft"]): text(r) for r in json.load(open("%s/before/ck%d.json" % (H, k)))["probe"]}
    print("==", bd, "lesson", k, "learn", x["learn_s"], "s")
    polls = x["polls"]
    def phase(t):
        cur = None
        for p in polls:
            if p[0] <= t:
                cur = [f for f in p[1] if f[1] == "learning"]
        return cur
    for r in x["chat"]:
        if r["pid"] not in pids:
            continue
        kk = (r["pid"], r["thinking"], r["draft"])
        t = text(r)
        c = ("pre+post" if prefix(t, pre.get(kk)) and prefix(t, post.get(kk)) else "pre" if prefix(t, pre.get(kk)) else "post" if prefix(t, post.get(kk)) else "NEITHER")
        print("w%d n%-3d %6.1f-%6.1f %-6s think=%-5s draft=%-6s %s %-8s phase=%s | %r" % (r["worker"], r["n"], r["t0"], r["t1"], r["pid"], r["thinking"], r["draft"], r.get("token_sha"), c, phase(r["t0"]), (r.get("content") or r.get("reasoning") or "")[:70]))
