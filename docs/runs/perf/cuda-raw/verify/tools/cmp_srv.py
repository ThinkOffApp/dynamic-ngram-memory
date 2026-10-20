#!/usr/bin/env python3
"""Server series: exactness (token_sha per request, before vs after; alone/serial counts; facts) and bytes.

usage: cmp_srv.py OUT.json SERIES_DIR...
Each SERIES_DIR holds LABEL.run.json and LABEL.{openai,openai256,off,conc,conc14,facts}.json from srv2.py.
"""
import collections, glob, json, os, sys

out_path, dirs = sys.argv[1], sys.argv[2:]
servers = []
for d in dirs:
    for p in sorted(glob.glob(os.path.join(d, "*.run.json"))):
        r = json.load(open(p))
        r["dir"] = d
        servers.append(r)

# token_sha per request key, per folder: {folder: {key: {sha: [server labels]}}}
seen = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
alone = []
facts_rows = collections.defaultdict(dict)  # (folder, build) -> {(qid, thinking, draft): row}
problems = []


def add(folder, key, sha, who):
    seen[folder][key][sha].append(who)


for s in servers:
    who = "%s/%s" % (os.path.basename(s["dir"]), s["label"])
    m, b = s["model"], s["build"]
    if not s.get("bytes_unchanged") or not s.get("listing_unchanged"):
        problems.append("%s: folder changed" % who)
    for a, run in (s.get("runs") or {}).items():
        if run.get("rc") != 0:
            problems.append("%s %s rc=%s" % (who, a, run.get("rc")))
        p = run.get("out")
        p = os.path.join(s["dir"], os.path.basename(p)) if p else None
        if not p or not os.path.exists(p):
            problems.append("%s %s: no output" % (who, a))
            continue
        data = json.load(open(p))
        if a in ("openai", "openai256"):
            for row in data:
                for i, ev in enumerate(row["evidence"]):
                    if not ev.get("complete") or not ev.get("token_sha"):
                        problems.append("%s %s %s incomplete" % (who, a, row["prompt"]))
                    add(m, ("bench_openai", row["prompt"], row["temperature"], row["tokens"], 1234 + i), ev.get("token_sha"), who + ":" + a)
        elif a in ("conc", "conc14"):
            for cell in data["cells"]:
                n = cell["streams"]
                t = cell["temperature"]
                al, se = cell.get("alone") or {}, cell.get("serial") or {}
                alone.append({"server": who, "build": b, "folder": m, "client": a, "prompt": cell["prompt"], "temperature": t, "streams": n,
                              "alone": al, "serial": se, "failed": cell.get("failed"), "unmeasured": cell.get("unmeasured")})
                for rep in cell["reps"]:
                    for i, ev in enumerate(rep["evidence"]):
                        seed = 1234 + i if t > 0 else 1234
                        add(m, ("bench_concurrent", cell["prompt"], t, data["tokens"], seed), ev.get("token_sha"), who + ":" + a)
        elif a == "off":
            for row in data["rows"]:
                add(m, ("decode_bench draft off", row["prompt_id"], 0.0, 256, None), row.get("token_sha"), who + ":" + a)
        elif a == "facts":
            for row in data:
                if row.get("error"):
                    problems.append("%s facts %s error %s" % (who, row["qid"], row["error"]))
                facts_rows[(m, b)][(row["qid"], row["thinking"], row["draft"])] = row
                add(m, ("facts", row["qid"], row["thinking"], row["draft"]), row.get("token_sha"), who + ":" + a)

res = {"servers": len(servers), "problems": problems}
# one token_sha per (folder, request) over every server of both builds
single = {}
multi = []
for m, keys in seen.items():
    single[m] = {"distinct_requests": len(keys), "single_valued": sum(1 for v in keys.values() if len(v) == 1),
                 "replies": sum(len(w) for v in keys.values() for w in v.values())}
    for k, v in keys.items():
        if len(v) > 1:
            multi.append({"folder": m, "key": list(k), "shas": {sha: ws for sha, ws in v.items()}})
res["token_sha_single_valued"] = single
res["token_sha_multi"] = multi
# per build: same counts separately, so before=after is visible
for m, keys in seen.items():
    for k, v in keys.items():
        pass
# alone / serial
agg = collections.defaultdict(lambda: collections.Counter())
for x in alone:
    c = agg[x["build"]]
    for f in ("equal", "unequal", "failed"):
        c["alone_" + f] += x["alone"].get(f, 0)
    c["failed_requests"] += x.get("failed") or 0
    c["unmeasured"] += x.get("unmeasured") or 0
# serial counts are repeated in each level's cell of the same (prompt, temperature): count them once per client run
ser = collections.defaultdict(lambda: collections.Counter())
done = set()
for x in alone:
    k = (x["server"], x["client"], x["prompt"], x["temperature"])
    if k in done:
        continue
    done.add(k)
    for f in ("equal", "unequal", "failed"):
        ser[x["build"]]["serial_" + f] += x["serial"].get(f, 0)
res["alone"] = {b: dict(c) for b, c in agg.items()}
res["serial"] = {b: dict(c) for b, c in ser.items()}
res["alone_by_server"] = alone
# folders: T and E against A, on keys every folder has, per request (single-valued shas only)
fold = {}
for m in ("T", "E"):
    if m not in seen or "A" not in seen:
        continue
    keys = [k for k in seen[m] if k in seen["A"] and len(seen[m][k]) == 1 and len(seen["A"][k]) == 1]
    diff = [k for k in keys if list(seen[m][k])[0] != list(seen["A"][k])[0]]
    byclient = collections.Counter(k[0] for k in diff)
    fold[m + " vs A"] = {"requests": len(keys), "differ": len(diff), "differ_by_client": dict(byclient)}
res["folders"] = fold
# facts: before vs after on every field; drafted vs "draft": false within a build
fx = {}
for m in "TAE":
    bb, aa = facts_rows.get((m, "before")), facts_rows.get((m, "after"))
    if bb and aa:
        keys = sorted(set(bb) & set(aa))
        eq = [k for k in keys if all(bb[k].get(f) == aa[k].get(f) for f in ("token_sha", "content", "reasoning", "completion_tokens", "finish"))]
        fx[m + " before=after"] = {"equal": len(eq), "of": len(keys)}
    for b in ("before", "after"):
        r = facts_rows.get((m, b))
        if not r:
            continue
        pairs = [(k, (k[0], k[1], "off")) for k in r if k[2] == "server" and (k[0], k[1], "off") in r]
        eq = sum(1 for k, k2 in pairs if r[k]["token_sha"] == r[k2]["token_sha"] and r[k]["content"] == r[k2]["content"] and r[k]["reasoning"] == r[k2]["reasoning"])
        fx["%s %s drafted=draft false" % (m, b)] = {"equal": eq, "of": len(pairs)}
        pt = [(row["qid"], row["thinking"], row.get("prompt_tokens")) for row in r.values()]
        fx["%s %s replies" % (m, b)] = {"n": len(r), "tokens": sum(row.get("completion_tokens") or 0 for row in r.values()),
                                       "finish_length": sum(1 for row in r.values() if row.get("finish") == "length")}
for b in ("before", "after"):
    a = facts_rows.get(("A", b))
    for m in "TE":
        r = facts_rows.get((m, b))
        if a and r:
            keys = sorted(set(a) & set(r))
            fx["%s vs A %s differ" % (m, b)] = {"differ": sum(1 for k in keys if a[k]["token_sha"] != r[k]["token_sha"]), "of": len(keys)}
res["facts"] = fx
res["prompt_tokens"] = {}
for (m, b), r in facts_rows.items():
    res["prompt_tokens"]["%s %s" % (m, b)] = sorted({(row["qid"], row["thinking"], row.get("prompt_tokens")) for row in r.values()})
json.dump(res, open(out_path, "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k not in ("alone_by_server", "prompt_tokens")}, indent=1)[:6000])
