#!/usr/bin/env python3
"""Validation: my 44 A1x / A1y asks against run6's night-12 / night-14 alone "done" lines.

  python3 validate.py ALONE_LINES.json REPLIES_DIR OUT.json
"""
import json, sys
from collections import Counter
lines = json.load(open(sys.argv[1]))
rd = sys.argv[2]
out = {}
for arm, night in (("A1x", "12"), ("A1y", "14")):
    mine = [json.loads(l) for l in open(f"{rd}/gate-{arm}.jsonl")]
    want = Counter((d[0], d[1], d[2]) for d in lines[night]["done"])
    got = Counter((r["done"]["prompt"], r["done"]["tokens"], r["done"]["sha"]) for r in mine)
    want_p = Counter(d[0] for d in lines[night]["done"])
    got_p = Counter(r["done"]["prompt"] for r in mine)
    usage_ok = all(r["prompt_tokens"] == r["done"]["prompt"] and r["completion_tokens"] == r["done"]["tokens"] for r in mine)
    out[arm] = {"night": int(night), "asks": len(mine), "night_lines": sum(want.values()),
                "prompt_token_multiset_equal": want_p == got_p, "full_multiset_equal": want == got,
                "missing_from_mine": [list(k) + [c] for k, c in (want - got).items()],
                "extra_in_mine": [list(k) + [c] for k, c in (got - want).items()],
                "usage_matches_done_lines": usage_ok,
                "per_question": [{"fact": r["fact"], "i": r["i"], "q": r["q"], "prompt": r["done"]["prompt"], "tokens": r["done"]["tokens"],
                                  "sha": r["done"]["sha"], "in_night": (r["done"]["prompt"], r["done"]["tokens"], r["done"]["sha"]) in want,
                                  "reply": r["reply"]} for r in mine]}
    print(arm, "night", night, "prompt multiset equal:", want_p == got_p, "| full (prompt, tokens, sha) multiset equal:", want == got,
          "| missing", sum((want - got).values()), "extra", sum((got - want).values()))
json.dump(out, open(sys.argv[3], "w"), indent=1, ensure_ascii=False)
