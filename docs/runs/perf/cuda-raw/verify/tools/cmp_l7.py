#!/usr/bin/env python3
"""The lanes CLI's replies on T, decoded, against the L7 run's last probe of the same folder (1.0.8 server, drafted).

usage: cmp_l7.py SNAPSHOT LANES_DIR ANSWERS.jsonl
Thinking off: the decoded text is the reply's content. Thinking on: text before </think> is reasoning, the rest content;
without </think> all of it is reasoning (as the server splits it). Token counts compared with usage.completion_tokens.
"""
import json, sys
from tokenizers import Tokenizer

tok = Tokenizer.from_file(sys.argv[1] + "/tokenizer.json")
ans = [json.loads(x) for x in open(sys.argv[3]) if x.strip()]
for b in ("before", "after"):
    for th, f in ((False, "off-all-solo-d"), (True, "on-40-solo-d")):
        srv = {a["qid"]: a for a in ans if a["thinking"] is th}
        r = json.load(open("%s/T-%s-%s.json" % (sys.argv[2], b, f)))
        eq, n, bad = 0, 0, []
        for s in r["streams"]:
            n += 1
            t = s["tokens"]
            a = srv[s["name"]]
            txt = tok.decode([x for x in t if x not in (11, 2)], skip_special_tokens=False)
            if th:
                rs, ct = (txt.split("</think>", 1) if "</think>" in txt else (txt, ""))
            else:
                rs, ct = "", txt
            ok = ct.strip() == a["content"].strip() and rs.replace("<think>", "").strip() == (a.get("reasoning_content") or "").strip() \
                and len(t) == a["usage"]["completion_tokens"]
            eq += ok
            if not ok:
                bad.append(s["name"])
        print("T %s thinking %s: %d of %d replies equal the L7 server's (text and token count); unequal %s" % (b, th, eq, n, bad))
