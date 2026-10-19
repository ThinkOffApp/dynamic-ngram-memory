#!/usr/bin/env python3
"""Per-fact tables (markdown) from results.json.  python3 tables.py RESULTS_DIR"""
import json, sys

D = sys.argv[1]
R = json.load(open(f"{D}/results.json"))
T, DEC = R["table"], R["decisions"]
INW = ["w1", "w3", "w9", "w15", "w21"]


def c(f, k):
    v = T[f].get(k)
    return "-" if v is None else f"{v['right']}/{v['of']}" + (" ✓" if v["pass"] else "")


out = []
out.append("Model: Nemotron-3.5-Lightning-30B-A3B MLX 4-bit (taught = run6's frozen weights, untaught = SNAP). Machine: gb10-a (GB10). Binary: src7 3bf4e54f. Greedy, thinking off, 48 tokens (160 for the 56-set).\n")
out.append("### Q_gate (the 44 alone probes), judge agreesWith; ✓ = every probe right (the gate's rule)\n")
cols = [("taught A0", "taught|gate|A0|agrees"), ("untaught A0", "untaught|gate|A0|agrees"), ("taught A1x", "taught|gate|A1x|agrees"),
        ("taught A1y", "taught|gate|A1y|agrees"), ("taught A2", "taught|gate|A2|agrees"), ("taught A3", "taught|gate|A3|agrees"),
        ("taught A4", "taught|gate|A4|agrees"), ("taught A5", "taught|gate|A5|agrees"), ("untaught A1x", "untaught|gate|A1x|agrees"),
        ("untaught A4", "untaught|gate|A4|agrees"), ("untaught A5", "untaught|gate|A5|agrees")]
out.append("| fact | " + " | ".join(h for h, _ in cols) + " |")
out.append("|---" * (len(cols) + 1) + "|")
for f in T:
    if f == "w18":
        continue
    name = f"{f}{' (in weights)' if f in INW else ''}: {T[f]['text']}"
    out.append(f"| {name} | " + " | ".join(c(f, k) for _, k in cols) + " |")
out.append("\n### P(F), Q_trained (judge agreesWith with the written reference; ✓ = more than half right) and Q_test (test-suite judge, both roots agree)\n")
cols2 = [("P(F) t A0", "taught|pf|A0|agrees"), ("P(F) u A0", "untaught|pf|A0|agrees"), ("P(F) t A1x", "taught|pf|A1x|agrees"), ("P(F) t A5", "taught|pf|A5|agrees"),
         ("Q_tr t A0", "taught|trained|A0|agrees"), ("Q_tr u A0", "untaught|trained|A0|agrees"), ("Q_tr t A1x", "taught|trained|A1x|agrees"), ("Q_tr t A5", "taught|trained|A5|agrees"),
         ("Q_test t A0", "taught|q56|A0"), ("Q_test u A0", "untaught|q56|A0"), ("Q_test t A1x", "taught|q56|A1x")]
out.append("| fact | " + " | ".join(h for h, _ in cols2) + " |")
out.append("|---" * (len(cols2) + 1) + "|")
for f in T:
    out.append(f"| {f} | " + " | ".join(c(f, k) for _, k in cols2) + " |")
out.append("\n### A6: A1x with one suspect note also left out (taught)\n")
for f in T:
    ks = [k for k in T[f] if k.startswith("taught|gate|A6") and k.endswith("|agrees")]
    if ks:
        out.append(f"- {f}: " + ", ".join(f"{k.split('|')[2]} {c(f, k)}" for k in ks) + f" (A1x {c(f, 'taught|gate|A1x|agrees')}, A0 {c(f, 'taught|gate|A0|agrees')})")
out.append("\n### Decision rules, mechanically\n")
out.append("| fact | carried (rule 1, Q_gate) | carried on P(F) | carried on Q_trained | rule 2 competition (gate / P(F) / Q_tr) | rule 3 H1 | rule 4 H3: keep-round questions wrong in A0 | rule 5 H4 |")
out.append("|---|---|---|---|---|---|---|---|")
for f, d in DEC.items():
    comp = " / ".join("yes" if d["rule2_competition"][s] else "no" for s in ("gate", "pf", "trained"))
    h3 = "; ".join(f"probe {x['i'] + 1} \"{x['q']}\" -> \"{x['reply'][:60]}\"" for x in d["rule4_H3_wrong_in_A0"]) or "none"
    out.append(f"| {f} | {'YES' if d['carried_headline'] else 'no'} | {'yes' if d['carried_pf'] else 'no'} | {'yes' if d['carried_trained'] else 'no'} | {comp} | {'yes' if d['rule3_H1'] else 'no'} | {h3} | {'yes' if d['rule5_H4'] else 'no'} |")
out.append("\nRule 6 (green pair wrong in A0 and wrong in A6-s7): s2 " + str(DEC["s2"].get("rule6_green_pair_confirmed")) + ", s10 " + str(DEC["s10"].get("rule6_green_pair_confirmed")) + " (False = not confirmed: both are right in A0).")
out.append("\nRule 7, probes the untaught model gets wrong with the fact's own note in (untaught A4):\n")
for b in R["broken_probes_untaught_A4"]:
    out.append(f"- {b['fact']} probe {b['i'] + 1} \"{b['q']}\": reference \"{b['ref'][:70]}\", untaught reply \"{b['reply'][:90].replace(chr(10), ' ')}\"")
out.append("\n### Totals over the 15 pocket facts (agreesWith / recalls / recallsShort)\n")
out.append("| model, set, arm | agreesWith | recalls | recallsShort | of | facts passing |")
out.append("|---|---|---|---|---|---|")
for k, v in R["totals"].items():
    if "|q56|" in k:
        continue
    out.append(f"| {k} | {v['agrees']} | {v['recalls']} | {v['recalls_short']} | {v['of']} | {v['facts_passing'] if v['facts_passing'] is not None else '-'} |")
for k, v in R["totals"].items():
    if "|q56|" in k:
        out.append(f"| {k} (test-suite judge: judge-main / llwz-promo2) | {v['main']} / {v['promo2']} | | | {v['of']} | |")
open(f"{D}/tables.md", "w").write("\n".join(out) + "\n")
print("\n".join(out))
