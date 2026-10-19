#!/usr/bin/env python3
"""Build E1's question sets from run6's files (read-only copies).

  python3 build_sets.py PROMOTE SERVER_LOG L6_SET PF_JSON OUT_DIR

Writes OUT_DIR/sets.json (facts with Q_gate, Q_trained, P(F) after collisions; the 56-set with each question's fact)
and OUT_DIR/pf-collisions.json.
"""
import json, re, sys

POCKET = ["s2", "s3", "s4", "s5", "s10", "w1", "w3", "w6", "w9", "w13", "w14", "w15", "w16", "w21", "w22"]
IN_WEIGHTS = ["w1", "w3", "w9", "w15", "w21"]


def norm(q):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s']", " ", q.lower().replace("‑", "-"))).strip()


def lesson_of(key):
    n = int(key[1:])
    return n if key[0] == "s" else n + 10


def main():
    promote, logp, setp, pfp, out = sys.argv[1:6]
    P = json.load(open(promote))
    S = json.load(open(setp))
    PF = json.load(open(pfp))
    by_key = {f["key"]: f for f in P["facts"]}
    lessons = {l["id"]: l for l in S["lessons"]}
    lines = open(logp, errors="replace").read().split("\n")
    # night summary line index per night
    summary = {}
    for i, l in enumerate(lines):
        m = re.search(r"notes promotion: night (\d+): \d+ due", l)
        if m:
            summary[int(m.group(1))] = i
    kept_re = re.compile(r"slide: (\d+) questions kept for: (.*?) \((.*)\)$")
    kept = []  # (line, fact text, [questions])
    for i, l in enumerate(lines):
        m = kept_re.search(l)
        if m:
            qs = [q.strip() for q in m.group(3).split(" | ")] if m.group(3).strip() else []
            if len(qs) != int(m.group(1)):
                raise SystemExit(f"line {i+1}: {m.group(1)} questions announced, {len(qs)} parsed")
            kept.append((i, m.group(2), qs))
    # every question any lesson kept, near misses, gate probes: the collision set
    lesson_qs = set()
    for _, _, qs in kept:
        lesson_qs.update(norm(q) for q in qs)
    for l in lines:
        m = re.search(r"slide: near misses kept steady: (.*)$", l)
        if m:
            lesson_qs.update(norm(q) for q in m.group(1).split(" | "))
    for f in P["facts"]:
        for p in f["probes"]:
            lesson_qs.add(norm(p["q"]))
    for q in S["questions"]:
        lesson_qs.add(norm(q["text"]))
    facts = {}
    for key in POCKET + ["w18"]:
        f = by_key[key]
        L = lessons[lesson_of(key)]
        assert L["text"] == f["text"], (key, L["text"], f["text"])
        gate = [{"q": p["q"], "ref": p["ref"]} for p in f["probes"]]
        trained, trained_line = [], None
        if key in POCKET:
            end = summary[f["night"]]
            cands = [(i, qs) for i, t, qs in kept if t == f["text"] and i < end]
            if not cands:
                raise SystemExit(f"{key}: no kept line before night {f['night']}")
            trained_line, qs = cands[-1]
            gq = {norm(g["q"]) for g in gate}
            trained = [q for q in qs if norm(q) not in gq][:4]
        pf = PF["facts"][key]
        facts[key] = {"key": key, "text": f["text"], "status": f["status"], "night": f["night"], "e": L["e"], "a": L["a"], "v": L["v"],
                      "in_weights": key in IN_WEIGHTS, "pocket": key in POCKET, "gate": gate, "trained": trained,
                      "trained_log_line": None if trained_line is None else trained_line + 1,
                      "ref": pf["ref"], "short": pf["short"], "pf_all": pf["q"]}
    collisions = []
    for key, f in facts.items():
        keep = []
        for q in f["pf_all"]:
            if norm(q) in lesson_qs:
                collisions.append({"fact": key, "q": q})
            else:
                keep.append(q)
        f["pf"] = keep
    # the 56-set: each question's fact is the pocket fact (or pears) on its (e, a)
    pairs = {}
    for k, f in facts.items():
        pairs.setdefault((f["e"], f["a"]), []).append(k)
    pair = {p: ks[0] for p, ks in pairs.items() if len(ks) == 1}  # a pair two facts share (me, likes) maps to neither
    q56 = []
    for q in S["questions"]:
        e = q.get("e") or (q.get("start") if q.get("family") == "link" else None)
        q56.append({"qid": q["qid"], "text": q["text"], "e": e, "a": q.get("a"), "type": q["type"], "fact": pair.get((e, q.get("a")))})
    json.dump({"facts": facts, "q56": q56, "pocket": POCKET, "in_weights": IN_WEIGHTS}, open(f"{out}/sets.json", "w"), indent=1, ensure_ascii=False)
    json.dump(collisions, open(f"{out}/pf-collisions.json", "w"), indent=1, ensure_ascii=False)
    print("gate", sum(len(facts[k]["gate"]) for k in POCKET), "trained", sum(len(facts[k]["trained"]) for k in POCKET),
          "pf", sum(len(f["pf"]) for f in facts.values()), "collisions", len(collisions), "q56", len(q56),
          "q56 with a pocket fact", sum(1 for q in q56 if q["fact"] in POCKET))


if __name__ == "__main__":
    main()
