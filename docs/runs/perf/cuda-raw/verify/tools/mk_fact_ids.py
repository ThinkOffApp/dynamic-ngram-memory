#!/usr/bin/env python3
"""L7-seed1 fact questions as chat prompt token ids (the snapshot's own chat template and tokenizer).

usage: mk_fact_ids.py SNAPSHOT L7-seed1.json OUTDIR
Writes questions.json (the 40-question subset and all 134, with qids) and lanes prompts files
facts-off-all.json (134, thinking off), facts-off-40.json and facts-on-40.json: {qid: [ids]} in question order.
"""
import json, os, sys
import jinja2
from tokenizers import Tokenizer

snap, setp, out = sys.argv[1], sys.argv[2], sys.argv[3]
S = json.load(open(setp))
qs = S["questions"]
keep_facts = set(range(1, 65, 4)) | {64}
sub = [q for q in qs if (q.get("fact") in keep_facts) or q.get("fact") is None]
tpl = open(os.path.join(snap, "chat_template.jinja")).read()
env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)


def raise_exception(m):
    raise Exception(m)


env.globals["raise_exception"] = raise_exception
tok = Tokenizer.from_file(os.path.join(snap, "tokenizer.json"))


def ids(text, thinking):
    s = env.from_string(tpl).render(messages=[{"role": "user", "content": text}], add_generation_prompt=True,
                                    enable_thinking=thinking, bos_token="", eos_token="")
    return tok.encode(s, add_special_tokens=False).ids


os.makedirs(out, exist_ok=True)
json.dump({"set": S["id"], "subset_rule": "facts 1,5,...,61 and 64 (CURRENT and YES_NO), plus every question without a fact",
           "subset": [{"qid": q["qid"], "type": q["type"], "role": q["role"], "fact": q.get("fact"), "text": q["text"]} for q in sub],
           "all": [{"qid": q["qid"], "type": q["type"], "role": q["role"], "fact": q.get("fact"), "text": q["text"]} for q in qs]},
          open(os.path.join(out, "questions.json"), "w"), indent=1)
for name, group, th in (("facts-off-all", qs, False), ("facts-off-40", sub, False), ("facts-on-40", sub, True)):
    json.dump({q["qid"]: ids(q["text"], th) for q in group}, open(os.path.join(out, name + ".json"), "w"))
    print(name, len(group))
print("subset", len(sub), [q["qid"] for q in sub])
