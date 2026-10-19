#!/usr/bin/env python3
"""Judge E1's replies with the branch's own judge (3bf4e54f) and apply decision-rules.md.

  python3 analyze.py PULL_DIR JUDGE_BIN OUT_DIR
"""
import json, os, subprocess, sys
from collections import defaultdict

D, JUDGE, OUT = sys.argv[1:4]
os.makedirs(OUT, exist_ok=True)
SETS = json.load(open(f"{D}/inputs/sets.json"))
FACTS, POCKET, INW = SETS["facts"], SETS["pocket"], SETS["in_weights"]
NOTES = [n["text"] for n in json.load(open(f"{D}/notes-taught.json"))["data"]]
# keep-round failures: probe wordings the last keep round named as not bringing the fact back (filled from the log below)
KEEP_FAIL = json.load(open(f"{OUT}/keep-round-fails.json")) if os.path.exists(f"{OUT}/keep-round-fails.json") else {}

# ---- judge every reply of gate / pf / trained
recs = []
for model in ("taught", "untaught"):
    for fn in sorted(os.listdir(f"{D}/replies/{model}")):
        st = fn.split("-")[0]
        if st not in ("gate", "pf", "trained"):
            continue
        for l in open(f"{D}/replies/{model}/{fn}"):
            r = json.loads(l)
            f = FACTS[r["fact"]]
            ref = f["gate"][r["i"]]["ref"] if st == "gate" else f["ref"]
            r["judge_in"] = {"fact": f["text"], "q": r["q"], "ref": ref, "short": f["short"], "reply": r["reply"],
                             "others": [t for t in NOTES if t != f["text"]]}
            recs.append(r)
inp = f"{OUT}/judge-input.jsonl"
with open(inp, "w") as fh:
    for r in recs:
        fh.write(json.dumps(r["judge_in"], ensure_ascii=False) + "\n")
subprocess.run([JUDGE, inp, f"{OUT}/judge-output.txt"], check=True)
for r, line in zip(recs, open(f"{OUT}/judge-output.txt")):
    a, b, c = line.split()
    r["agrees"], r["recalls"], r["recalls_short"] = a == "1", b == "1", c == "1"
with open(f"{OUT}/verdicts.jsonl", "w") as fh:
    for r in recs:
        x = {k: r[k] for k in ("model", "set", "arm", "fact", "i", "q", "reply", "agrees", "recalls", "recalls_short", "exclude", "system_sha256", "prompt_tokens", "completion_tokens", "finish")}
        x["sha"] = r["done"]["sha"]
        x["ref"] = r["judge_in"]["ref"]
        fh.write(json.dumps(x, ensure_ascii=False) + "\n")

# ---- 56-set verdicts (test-suite judge, both roots)
q56 = {}
for model in ("taught", "untaught"):
    for arm in ("A0", "A1x"):
        R = [json.loads(l) for l in open(f"{D}/replies/{model}/q56-{arm}.jsonl")]
        v = {}
        for name in ("judge-main", "llwz-promo2"):
            res = json.load(open(f"{D}/judge/q56/{model}-{arm}.verdicts-{name}.json"))["results"]
            v[name] = {x["qid"]: x["verdict"] for x in res}
        rows = []
        for r in R:
            qid = SETS["q56"][r["i"]]["qid"]
            rows.append({"qid": qid, "fact": r["fact"], "q": r["q"], "reply": r["reply"], "main": v["judge-main"][qid], "promo2": v["llwz-promo2"][qid]})
        q56[(model, arm)] = rows

# ---- tables
V = defaultdict(list)  # (model, set, arm, fact) -> [verdict rows]
for r in recs:
    V[(r["model"], r["set"], r["arm"], r["fact"])].append(r)


def score(model, st, arm, fact, key="agrees"):
    rows = V.get((model, st, arm, fact))
    if not rows:
        return None
    n = len(rows)
    k = sum(1 for r in rows if r[key])
    ok = (k == n) if st == "gate" else (2 * k > n)
    return {"right": k, "of": n, "pass": ok}


arms_gate_t = ["A0", "A1x", "A1y", "A2", "A3", "A4", "A5"]
table = {}
for fact in POCKET + ["w18"]:
    row = {"text": FACTS[fact]["text"], "in_weights": fact in INW}
    for model in ("taught", "untaught"):
        for st in ("gate", "pf", "trained"):
            for arm in arms_gate_t + sorted({a for (m, s, a, f) in V if a.startswith("A6") and f == fact and m == model and s == st}):
                for key in ("agrees", "recalls", "recalls_short"):
                    s = score(model, st, arm, fact, key)
                    if s is not None:
                        row[f"{model}|{st}|{arm}|{key}"] = s
    for model in ("taught", "untaught"):
        for arm in ("A0", "A1x"):
            rows = [x for x in q56[(model, arm)] if x["fact"] == fact]
            if rows:
                row[f"{model}|q56|{arm}"] = {"right": sum(1 for x in rows if x["main"] == "pass" and x["promo2"] == "pass"), "of": len(rows),
                                            "pass": all(x["main"] == "pass" and x["promo2"] == "pass" for x in rows)}
    table[fact] = row


def P(fact, model, st, arm, key="agrees"):
    s = table[fact].get(f"{model}|{st}|{arm}|{key}")
    return None if s is None else s["pass"]


# ---- decision rules
dec = {}
for fact in POCKET:
    d = {}
    d["carried_headline"] = bool(P(fact, "taught", "gate", "A0")) and P(fact, "untaught", "gate", "A0") is False
    d["carried_pf"] = bool(P(fact, "taught", "pf", "A0")) and P(fact, "untaught", "pf", "A0") is False
    d["carried_trained"] = bool(P(fact, "taught", "trained", "A0")) and P(fact, "untaught", "trained", "A0") is False
    d["untaught_passes_A0_gate"] = bool(P(fact, "untaught", "gate", "A0"))
    comp = {}
    for st in ("gate", "pf", "trained"):
        comp[st] = bool(P(fact, "taught", st, "A0")) and bool(P(fact, "taught", st, "A5")) and P(fact, "taught", st, "A1x") is False
    d["rule2_competition"] = comp
    d["rule3_H1"] = bool(P(fact, "taught", "gate", "A0")) and P(fact, "taught", "gate", "A3") is False
    # rule 4: held questions (probes 0-1) and probes 2-3 the keep round did not name, wrong in A0 (gate judge, and recalls)
    passed_keep = [i for i in range(len(FACTS[fact]["gate"])) if i < 2 or FACTS[fact]["gate"][i]["q"] not in KEEP_FAIL.get(fact, [])]
    a0 = {r["i"]: r for r in V[("taught", "gate", "A0", fact)]}
    d["rule4_H3_wrong_in_A0"] = [{"i": i, "q": a0[i]["q"], "reply": a0[i]["reply"], "agrees": a0[i]["agrees"], "recalls": a0[i]["recalls"]}
                                 for i in passed_keep if not a0[i]["agrees"]]
    d["rule5_H4"] = bool(P(fact, "taught", "trained", "A0")) and (P(fact, "taught", "pf", "A0") is False or P(fact, "taught", "gate", "A0") is False)
    dec[fact] = d
for fact in ("s2", "s10"):
    dec[fact]["rule6_green_pair_confirmed"] = P(fact, "taught", "gate", "A0") is False and P(fact, "taught", "gate", "A6-s7") is False
broken = []
for r in recs:
    if r["model"] == "untaught" and r["set"] == "gate" and r["arm"] == "A4" and not r["agrees"]:
        broken.append({"fact": r["fact"], "i": r["i"], "q": r["q"], "ref": r["judge_in"]["ref"], "reply": r["reply"]})
totals = {}
for model in ("taught", "untaught"):
    for st in ("gate", "pf", "trained"):
        for arm in sorted({a for (m, s, a, f) in V if m == model and s == st}):
            rows = [r for (m, s, a, f), rr in V.items() if m == model and s == st and a == arm for r in rr]
            pocket_rows = [r for r in rows if r["fact"] in POCKET]
            totals[f"{model}|{st}|{arm}"] = {"agrees": sum(r["agrees"] for r in pocket_rows), "recalls": sum(r["recalls"] for r in pocket_rows),
                                             "recalls_short": sum(r["recalls_short"] for r in pocket_rows), "of": len(pocket_rows),
                                             "facts_passing": sum(1 for f in POCKET if P(f, model, st, arm)) if st != "gate" or arm != "A6" else None}
for (model, arm), rows in q56.items():
    totals[f"{model}|q56|{arm}"] = {"main": sum(x["main"] == "pass" for x in rows), "promo2": sum(x["promo2"] == "pass" for x in rows), "of": len(rows)}
json.dump({"table": table, "decisions": dec, "broken_probes_untaught_A4": broken, "totals": totals,
           "q56": {f"{m}|{a}": rows for (m, a), rows in q56.items()}}, open(f"{OUT}/results.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(totals, indent=0)[:4000])
