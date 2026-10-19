#!/usr/bin/env python3
"""intake: sort a new fact before it is learned, by the rule book (lwlogic.py).

Every new lesson gets one of four verdicts. Only the first two are learned.

  NEW            some answer changes, and it is not REPLACES
  REPLACES       the current value of a single-valued attribute was known (a value, or "none"), and changes
  ALREADY_KNOWN  no answer changes, and an earlier lesson says the same thing (hints aside, see `claim`)
  IMPLIED        no answer changes, but no earlier lesson says the same thing (it follows from them)

"No answer changes" is decided by the oracle: every question the rule book can be asked about the lesson's
(entity, attribute), plus WHO_HAS for its values, is answered with and without the lesson (`key_questions`).

Two ways in. The parser (`parse`) reads sentences in the generator's own wording tables (lwlogic.T) back into
lessons; it is exact and never guesses. The extractor (`extract`) asks any OpenAI-compatible chat endpoint for
the lesson as JSON and checks it (`check_lesson`) before use. The self-test never calls a real endpoint.

  python3 zengine/intake.py --store STORE.json --text "My eyes are green now."      (or: lwlogic.py intake ...)
  python3 zengine/intake.py --store set.json --phase teach --text "..." --text "..."  several, through the gate
  python3 zengine/intake.py --store me.json --text "..." --save                      learn NEW and REPLACES into me.json
  python3 zengine/intake.py --text "..." --endpoint URL --model NAME --api-key-env VAR   free English via a model
  python3 zengine/intake.py --records RUN_DIR [RUN_DIR ...]                          against TensorFold's run records

STORE.json is a test set (lw-test-set/1, all its lessons or one --phase) or a store file (lw-store/1) that
--save writes. Standard library only.
"""
import argparse
import datetime as dt
import itertools
import json
import os
import random
import re
import sys
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lwlogic as L  # noqa: E402  (the rule book, next to this file)

NEW, REPLACES, ALREADY_KNOWN, IMPLIED = "NEW", "REPLACES", "ALREADY_KNOWN", "IMPLIED"
VERDICTS = [NEW, REPLACES, ALREADY_KNOWN, IMPLIED]
LEARN = {NEW: True, REPLACES: True, ALREADY_KNOWN: False, IMPLIED: False}
STORE_FORMAT = "lw-store/1"
FIELDS = ("op", "e", "a", "v", "when", "v0", "d", "v1")


# ======================================================================================== reading English
# The lesson slots of the wording tables and the operation each one is.
SLOT_OP = {"ASSERT": "ASSERT", "SINCE": "ASSERT", "UNTIL": "ASSERT", "ALWAYS_A": "ASSERT", "CHANGE": "CHANGE",
           "RETRACT": "RETRACT", "CLEAR": "RETRACT", "DATED": "DATED"}
# Wordings the parser reads beyond T. "I like blue." is a favourite colour, as in our #678 case and in 1.0.8
# (TEST-LOGIC.md, decision 6); the values keep it apart from "I like apples." (likes holds foods).
EXTRA = {"fav_colour": {"ASSERT": ["{nom} like{s} {v}."]}}
# Names and cities are open: a capitalised word the lists lack ("My dog is called Rex.") is still a value.
OPEN = {"name", "home_city"}
# a marker for each slot that no template text contains (private-use characters)
HOLE = {k: "\ue000" + k + "\ue001" for k in ("v", "v0", "v1", "d")}
GROUP = {"v": r"(?P<v>[^.;,]+?)", "v0": r"(?P<v0>[^.;,]+?)", "v1": r"(?P<v1>[^.;,]+?)",
         "d": r"(?P<d>\d{1,2} [a-z]+ \d{4})"}
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
          "november", "december"]
_PATTERNS = []


def norm(text):
    """One line, straight quotes, single spaces, no closing full stop."""
    return " ".join((text or "").translate(L.APOS).split()).rstrip(".!? ")


def vocab_map(a):
    """lower-case word -> the rule book's value, for attribute a (list values, extras and their synonyms)."""
    out = {}
    for v in list(L.SCHEMA[a]["domain"]) + list(L.SCHEMA[a].get("extra", [])):
        for w in ([v] if a == "owner" else L.SYN.get(v, [v])):  # owner values are people; their SYN is "you"
            out.setdefault(w.lower(), v)
    return out


VOCAB = {a: vocab_map(a) for a in L.SCHEMA}


def value_of(a, raw):
    """The rule book's value for a word as the user wrote it, or None."""
    w = " ".join((raw or "").split())
    if w.lower() in VOCAB[a]:
        return VOCAB[a][w.lower()]
    parts = w.split(" ")
    if a in OPEN and 1 <= len(parts) <= 3 and all(p[:1].isupper() and re.fullmatch(r"[^\W\d_][\w'-]*", p) for p in parts):
        return w
    return None


def date_of(raw):
    """'11 October 2026' -> '2026-10-11', or None."""
    m = re.fullmatch(r"(\d{1,2}) ([A-Za-z]+) (\d{4})", (raw or "").strip())
    if not m:
        return None
    mon = [i for i, x in enumerate(MONTHS) if x == m.group(2).lower() or x[:3] == m.group(2).lower()]
    try:
        return dt.date(int(m.group(3)), mon[0] + 1, int(m.group(1))).isoformat() if mon else None
    except ValueError:
        return None


def patterns():
    """One anchored regular expression per (attribute, slot, wording, entity), built from the wording tables."""
    if _PATTERNS:
        return _PATTERNS
    for a in L.ATTR_ORDER:
        tables = [("T", L.T[a]), ("extra", EXTRA.get(a, {}))]
        for src, table in tables:
            for slot in SLOT_OP:
                for k, tmpl in enumerate(table.get(slot, [])):
                    for e, ent in L.ENTITIES.items():
                        if ent["kind"] not in L.SCHEMA[a]["applies"]:
                            continue
                        rx = re.escape(norm(L.render(tmpl, e, **HOLE)))
                        for h, mark in HOLE.items():
                            rx = rx.replace(re.escape(mark), GROUP[h])
                        _PATTERNS.append((re.compile("^" + rx + "$", re.I), a, slot, k, e, src))
    return _PATTERNS


def formal(l):
    """The parts of a lesson that carry its meaning (not its id, name, tick or text)."""
    return tuple(json.dumps(l.get(f), sort_keys=True) for f in FIELDS)


def readings(text):
    """Every lesson the wording tables can read `text` as, with how it was read. Normally one; none if the
    sentence is not in the tables or names a value, entity or date the rule book does not know."""
    t, out, seen = norm(text), [], set()
    if (text or "").rstrip().endswith("?"):           # "My eyes are green?" asks; it does not tell
        return out
    for rx, a, slot, k, e, src in patterns():
        m = rx.match(t)
        if not m:
            continue
        g = m.groupdict()
        vals = {h: value_of(a, g[h]) for h in ("v", "v0", "v1") if h in g}
        d = date_of(g["d"]) if "d" in g else None
        if None in vals.values() or ("d" in g and d is None):
            continue
        l = {"op": SLOT_OP[slot], "e": e, "a": a}
        if slot == "DATED":
            l.update(v0=vals["v0"], d=d, v1=vals["v1"])
        else:
            l["v"] = vals.get("v")                  # None for CLEAR: "I have no favourite colour anymore."
            if slot in ("SINCE", "UNTIL"):
                l["when"] = {slot.lower(): d}
            elif slot == "ALWAYS_A":
                l["when"] = {"always": True}
        l["text"] = " ".join(text.split())
        if formal(l) not in seen:
            seen.add(formal(l))
            out.append((l, {"slot": slot, "para": k, "source": src}))
    return out


def parse(text):
    """The one lesson `text` says in the generator's wordings, or None (no reading, or more than one)."""
    rs = readings(text)
    return rs[0][0] if len(rs) == 1 else None


def slot_of(l):
    if l["op"] == "ASSERT":
        w = l.get("when") or {}
        return "SINCE" if "since" in w else "UNTIL" if "until" in w else "ALWAYS_A" if w.get("always") else \
            "ASSERT" if not w else None
    if l["op"] == "RETRACT":
        return "RETRACT" if l.get("v") is not None else "CLEAR"
    return l["op"]


def say(l):
    """A lesson in the rule book's own first wording, or None where the tables have none."""
    slot = slot_of(l)
    paras = (L.T[l["a"]].get(slot) or EXTRA.get(l["a"], {}).get(slot) or []) if slot else []
    if not paras:
        return None
    w = l.get("when") or {}
    d = l.get("d") or w.get("since") or w.get("until")
    return L.render(paras[0], l["e"], v=l.get("v"), v0=l.get("v0"), v1=l.get("v1"), d=L.say_date(d) if d else None)


def say_question(q):
    paras = L.T[q["a"]].get("Q_" + q["type"]) or []
    if not paras:
        return f"{q['type']}({', '.join(str(q[k]) for k in ('e', 'a', 'v', 'x', 'y', 'date') if q.get(k))})"
    return L.render(paras[0], q.get("e"), v=q.get("v"), x=q.get("x"), y=q.get("y"),
                    d=L.say_date(q["date"]) if q.get("date") else None)


# ======================================================================================== checking a lesson
def check_lesson(l, today):
    """What is wrong with a lesson, as a list of plain sentences (empty when it is fine). The parser's readings
    pass by construction except for dates after today; the extractor's replies are checked field by field."""
    bad = []
    op, e, a = l.get("op"), l.get("e"), l.get("a")
    if op not in ("ASSERT", "CHANGE", "RETRACT", "DATED"):
        return [f"op must be ASSERT, CHANGE, RETRACT or DATED, not {op!r}"]
    if not isinstance(e, str) or e not in L.ENTITIES:        # a model may send a list or an object here
        bad.append(f"unknown entity {e!r} (known: {', '.join(L.ENTITIES)})")
    if not isinstance(a, str) or a not in L.SCHEMA:
        bad.append(f"unknown attribute {a!r} (known: {', '.join(L.SCHEMA)})")
    if bad:
        return bad
    if L.ENTITIES[e]["kind"] not in L.SCHEMA[a]["applies"]:
        bad.append(f"{a} does not apply to {e}")

    def val(k, allow_none=False):
        v = l.get(k)
        if v is None and allow_none:
            return
        if not isinstance(v, str) or value_of(a, v) != v:
            bad.append(f"{k} {v!r} is not a {a} value the rule book knows")

    def date(k, d):
        # exactly YYYY-MM-DD: newer Pythons also read "20261011" or week dates, older ones do not
        try:
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", d):
                raise ValueError(d)
            if dt.date.fromisoformat(d) > dt.date.fromisoformat(today):
                bad.append(f"{k} {d} is after today ({today}): the rule book reads 'now' as after every date")
        except (TypeError, ValueError):
            bad.append(f"{k} {d!r} is not an ISO date (YYYY-MM-DD)")

    if op == "DATED":
        val("v0")
        val("v1")
        date("d", l.get("d"))
        if L.SCHEMA[a]["multi"]:
            bad.append(f"{a} holds several values; a dated change of it is not modelled")
        if l.get("v0") == l.get("v1"):
            bad.append("a dated change needs two different values")
        return bad
    val("v", allow_none=(op == "RETRACT"))
    w = l.get("when")
    if w is not None:
        if op != "ASSERT":
            bad.append(f"only ASSERT takes 'when', not {op}")
        elif not isinstance(w, dict) or len(w) != 1 or next(iter(w)) not in ("since", "until", "always", "between"):
            bad.append(f"when must be one of since, until, always or between, not {w!r}")
        elif "always" in w:
            if w["always"] is not True:
                bad.append("when.always must be true")
        elif "between" in w:
            b = w["between"]
            if not (isinstance(b, list) and len(b) == 2):
                bad.append("when.between must be two dates")
            else:
                date("between", b[0])
                date("between", b[1])
                if not bad and b[0] >= b[1]:
                    bad.append("when.between must go from the earlier date to the later one")
        else:
            date(next(iter(w)), next(iter(w.values())))
    return bad


# ============================================================================================ the extractor
def prompt(today):
    ents = ", ".join(L.ENTITIES)
    kinds = {"self": "me", "person": "people", "animal": "animals", "vehicle": "vehicles", "object": "objects"}
    attrs = "; ".join(f"{a} ({', '.join(kinds[k] for k in L.SCHEMA[a]['applies'])}"
                      f"{', several at once' if L.SCHEMA[a]['multi'] else ''}), e.g. {', '.join(L.SCHEMA[a]['domain'][:3])}"
                      for a in L.ATTR_ORDER)
    return (
        "You turn one sentence a user says into one lesson for a fact store. Reply with one JSON object and "
        "nothing else.\n"
        "Fields:\n"
        "  op: ASSERT (a fact), CHANGE (a fact said as a change: 'now', 'these days', 'moved to'), "
        "RETRACT (no longer true: 'not ... anymore', 'no longer'), DATED (it was v0 until date d, and v1 since), "
        "or NONE if the sentence is not one such fact.\n"
        f"  e: who or what it is about, one of: {ents}. The user is 'me'.\n"
        f"  a: the attribute, one of: {attrs}.\n"
        "  v: the value, for ASSERT, CHANGE and RETRACT. For RETRACT, v is null when no value holds any more "
        "('I have no favourite colour anymore').\n"
        "  when: ASSERT only, optional: {\"since\": DATE}, {\"until\": DATE} or {\"always\": true}.\n"
        "  v0, d, v1: DATED only: the old value, the date of the change, the new value.\n"
        "Dates are ISO (YYYY-MM-DD). Foods in likes are plural ('apples'). owner is one of me, my sister, "
        f"my brother, my neighbour. Today is {today}.\n"
        "Example: 'My eyes are green now.' -> {\"op\": \"CHANGE\", \"e\": \"me\", \"a\": \"eye_colour\", \"v\": \"green\"}")


def first_json(text):
    """The first JSON object in a model's reply (it may be fenced or have words around it), or None."""
    dec = json.JSONDecoder()
    for m in re.finditer(r"\{", text or ""):
        try:
            obj, _ = dec.raw_decode(text, m.start())
        except ValueError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def lesson_from_json(obj, text):
    """A reply object -> a lesson in the rule book's shape (values normalised), or None if it is no lesson."""
    if not isinstance(obj, dict) or obj.get("op") in (None, "NONE"):
        return None
    a = obj.get("a")
    norm_v = lambda v: (value_of(a, v) or v) if isinstance(v, str) and isinstance(a, str) and a in L.SCHEMA else v
    l = {"op": obj.get("op"), "e": obj.get("e"), "a": a}
    if obj.get("op") == "DATED":
        l.update(v0=norm_v(obj.get("v0")), d=obj.get("d"), v1=norm_v(obj.get("v1")))
    else:
        l["v"] = norm_v(obj.get("v"))
        if obj.get("when"):
            l["when"] = obj["when"]
    l["text"] = " ".join(text.split())
    return l


def http_post(url, body, headers, timeout):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def extract(text, endpoint, model, today, api_key=None, post=http_post, timeout=120):
    """Ask an OpenAI-compatible chat endpoint for the lesson. Returns (lesson or None, problems, raw reply text).
    `post` is the HTTP call; the self-test passes a mock. The key goes only into the Authorization header."""
    url = endpoint.rstrip("/")
    if not url.endswith("/chat/completions"):
        url += "/chat/completions"
    body = {"model": model, "temperature": 0,
            "messages": [{"role": "system", "content": prompt(today)}, {"role": "user", "content": text}]}
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        resp = post(url, body, headers, timeout)
        raw = resp["choices"][0]["message"]["content"] or ""
        if not isinstance(raw, str):                 # e.g. a list of content parts: not the plain text we asked for
            raise TypeError(f"the reply content is a {type(raw).__name__}, not text")
    except (urllib.error.URLError, OSError, ValueError, KeyError, IndexError, TypeError) as ex:
        return None, [f"the endpoint call failed: {type(ex).__name__}: {str(ex)[:200]}"], ""
    obj = first_json(raw)
    if obj is None:
        return None, ["the reply holds no JSON object"], raw
    l = lesson_from_json(obj, text)
    if l is None:
        return None, ["the model found no fact the rule book models (op NONE)"], raw
    bad = check_lesson(l, today)
    return (None if bad else l), bad, raw


# ============================================================================================ the classifier
def intake_schema(lessons):
    """The rule book's schema, plus any value the lessons use that its lists lack (an open name or city)."""
    sch = {}
    for a in L.ATTR_ORDER:
        dom = list(L.SCHEMA[a]["domain"])
        for l in lessons:
            if l["a"] == a:
                dom += [v for v in (l.get("v"), l.get("v0"), l.get("v1")) if v is not None and v not in dom]
        sch[a] = dict(L.SCHEMA[a], domain=dom)
    return sch


def entities_of(lessons, ents=None):
    out = list(ents or [])
    for l in lessons:
        if l["e"] not in out:
            out.append(l["e"])
    return out


def key_questions(sch, store, lesson):
    """Every question the rule book can be asked about the lesson's (entity, attribute): current, previous,
    always, changed, yes/no for every value, either-or for every ordered pair, and the value on and around every
    day any of these lessons mentions or was told on (state only changes there); plus WHO_HAS for its values."""
    e, a = lesson["e"], lesson["a"]
    dom = sch[a]["domain"]
    qs = [{"type": t, "e": e, "a": a} for t in ("CURRENT", "PREVIOUS", "ALWAYS", "CHANGED")]
    qs += [{"type": "YES_NO", "e": e, "a": a, "v": v} for v in dom]
    qs += [{"type": "EITHER_OR", "e": e, "a": a, "x": x, "y": y} for x, y in itertools.permutations(dom, 2)]
    days = set()
    for l in store + [lesson]:
        if (l["e"], l["a"]) != (e, a):
            continue
        w = l.get("when") or {}
        days |= {L.day_of(d) for d in [l.get("d"), w.get("since"), w.get("until")] + list(w.get("between") or []) if d}
        days.add(l["told"] // L.TPD)
    pts = []
    for n in sorted(days):                       # the piece before the day, just before it, on it, and after it
        pts += [("AT", n - 1), ("BEFORE", n), ("AT", n), ("AT", n + 1)]
    for t, n in dict.fromkeys(pts):
        qs.append({"type": t, "e": e, "a": a, "date": L.iso_of(n)})
    vals = [v for v in (lesson.get("v"), lesson.get("v0"), lesson.get("v1")) if v is not None] or dom
    qs += [{"type": "WHO_HAS", "a": a, "v": v} for v in dict.fromkeys(vals)]
    return qs


def claim(sch, l):
    """What a lesson says outright: its writes without the hints, with "from the moment it is told" as one moment.
    Hints are what a wording implies about the other side of a change ("green now": not green just before); two
    wordings of one fact can differ there, and a hint never overrides what the store already says."""
    t = l.get("told")
    return sorted(((e, a, ("told" if lo == t else lo, hi), eff) for (e, a, (lo, hi), eff) in L.writes(sch, l)
                   if eff[0] not in ("presume", "presume_not")), key=repr)


def same_claim(sch, m, l):
    return claim(sch, m) == claim(sch, l)


def is_replace(sch, a, before, after):
    """REPLACES: a single-valued attribute whose current value was known (a value, or "none") changes."""
    return not sch[a]["multi"] and before["kind"] in ("value", "none") and after != before


def classify(store, lesson, ents=None, sch=None, questions=key_questions, same=same_claim, replaces=is_replace):
    """Sort `lesson` (already given its id and its told tick after the store's, see `place`) against `store`.

    Returns {"verdict", "learn", "changed": [(question, before, after)], "questions": n asked,
             "same_as": ids of earlier lessons that say the same (ALREADY_KNOWN),
             "covered_by": ids whose deletion alone would make the lesson change answers again (ALREADY_KNOWN,
                           IMPLIED); a lesson kept out by the gate is lost if these are deleted,
             "supersedes": ids that decided the current answer before and no longer do (NEW, REPLACES)}.
    `questions`, `same` and `replaces` are the parts the self-test swaps for broken ones."""
    sch = sch or intake_schema(store + [lesson])
    ents = entities_of(store + [lesson], ents)
    qs = questions(sch, store, lesson)
    ask = lambda st: [L.answer(sch, st, ents, q) for q in qs]
    without = lambda m: [x for x in store if x["id"] != m["id"]]
    on_key = [m for m in store if (m["e"], m["a"]) == (lesson["e"], lesson["a"])]
    before, after = ask(store), ask(store + [lesson])
    r = {"verdict": None, "learn": None, "questions": len(qs), "same_as": [], "covered_by": [], "supersedes": [],
         "changed": [(q, b, x) for q, b, x in zip(qs, before, after) if b != x]}
    if r["changed"]:
        cq = {"type": "CURRENT", "e": lesson["e"], "a": lesson["a"]}
        cur = lambda st: L.answer(sch, st, ents, cq)
        c0, c1 = cur(store), cur(store + [lesson])
        r["verdict"] = REPLACES if replaces(sch, lesson["a"], c0, c1) else NEW
        r["supersedes"] = [m["id"] for m in on_key if cur(without(m)) != c0 and cur(without(m) + [lesson]) == c1]
    else:
        r["same_as"] = [m["id"] for m in on_key if same(sch, m, lesson)]
        r["verdict"] = ALREADY_KNOWN if r["same_as"] else IMPLIED
        r["covered_by"] = [m["id"] for m in on_key if ask(without(m)) != ask(without(m) + [lesson])]
    r["learn"] = LEARN[r["verdict"]]
    return r


def place(store, lesson, today):
    """Give a new lesson the next id and a tick after every lesson in the store and inside `today`."""
    nid = max([l["id"] for l in store] + [0]) + 1
    told = max([l["told"] for l in store] + [L.tick(today)]) + 100
    l = {"id": nid, "name": f"L{nid}", "op": lesson["op"], "e": lesson["e"], "a": lesson["a"], "told": told}
    l.update((k, v) for k, v in lesson.items() if k not in l and k not in ("id", "name", "told"))
    return l


def teach_texts(texts, today):
    """A store from sentences, each read by the parser and taught in order (all of them, no gate)."""
    store = []
    for t in texts:
        l = parse(t)
        if l is None:
            raise ValueError(f"the parser cannot read {t!r}")
        store.append(place(store, l, today))
    return store


# ================================================================================================= stores
def load_store(path, phase=None):
    """(lessons, entities, today, kind) from a test set or a store file; an absent path is an empty store."""
    if not path or not os.path.exists(path):
        return [], [], None, "new"
    S = json.load(open(path))
    if S.get("format") == "lw-test-set/1" and str(S.get("level")) == "6":
        raise SystemExit(f"{path}: a level 6 set (two-step questions). Its linking facts are not modelled by the "
                         "intake check yet; use a set of levels 1 to 5 or a store file")
    if S.get("format") == "lw-test-set/1":
        keep = None
        if phase:
            ph = next((p for p in S["phases"] if p["name"] == phase), None)
            if ph is None:
                raise SystemExit(f"{path}: no phase {phase!r} (phases: {', '.join(p['name'] for p in S['phases'])})")
            keep = set(ph["store"])
        lessons = [l for l in S["lessons"] if keep is None or l["id"] in keep]
        return lessons, list(S.get("entities") or []), S.get("today"), "set"
    if S.get("format") == STORE_FORMAT:
        return list(S["lessons"]), list(S.get("entities") or []), S.get("today"), "store"
    raise SystemExit(f"{path}: not a test set (lw-test-set/1) or a store ({STORE_FORMAT})")


def default_today(kind, store_today, real_today):
    """The day new lessons are told when --today is not given. A test set is a fixed case, so its own day. A store
    file is a live store: lessons are told on the real day, never before the store's own day (else a store saved
    once would tell every later lesson on that first day, and refuse dates since then as "after today")."""
    if kind == "set" and store_today:
        return store_today
    if kind == "store" and store_today:
        return max(store_today, real_today)
    return real_today


def save_store(path, lessons, ents, today):
    with open(path, "w") as f:
        json.dump({"format": STORE_FORMAT, "today": today, "entities": entities_of(lessons, ents),
                   "lessons": lessons}, f, indent=1, ensure_ascii=False)
        f.write("\n")


# ======================================================================================== the run records
def compare_records(run_dir):
    """Sort every lesson of a run folder (lwrun.py's set.json and phases/*/state.json) against the lessons taught
    before it in that phase, and set our `supersedes` beside the facts TensorFold itself marked "replaced".
    Read only. Returns rows of plain values (lesson names and texts, no paths)."""
    S = json.load(open(os.path.join(run_dir, "set.json")))
    by_id = {l["id"]: l for l in S["lessons"]}
    rows = []
    pdir = os.path.join(run_dir, "phases")
    for ph in sorted(os.listdir(pdir)) if os.path.isdir(pdir) else []:
        sp = os.path.join(pdir, ph, "state.json")
        if not os.path.exists(sp):
            continue
        st = json.load(open(sp))
        recs = st.get("lessons") or []
        fact_of, store = {}, []
        for rec in recs:
            l = by_id[rec["lesson"]]
            got = parse(rec["text"])
            res = classify(store, l, S.get("entities"))
            server = [fact_of.get(c["id"]) for c in rec.get("changed_states") or [] if c.get("to") == "replaced"]
            rows.append({"set": S["id"], "phase": ph, "lesson": l["name"], "text": rec["text"],
                         "parsed": got is not None and formal(got) == formal(l), "recalled": rec.get("recalled"),
                         "verdict": res["verdict"], "supersedes": [by_id[i]["name"] for i in res["supersedes"]],
                         "server_replaced": [by_id[i]["name"] if i in by_id else "?" for i in server]})
            for fid in rec.get("fact_ids_graph") or rec.get("fact_ids_sse") or []:
                fact_of[fid] = l["id"]
            store.append(l)
    return rows


def records_report(dirs):
    rows = []
    for d in dirs:
        rows += compare_records(d)
    out = [f"{'set/phase':28} {'lesson':6} {'verdict':13} {'ours: supersedes':17} {'1.0.8: replaced':16} text"]
    for r in rows:
        out.append(f"{r['set'] + '/' + r['phase']:28} {r['lesson']:6} {r['verdict']:13} "
                   f"{' '.join(r['supersedes']) or '-':17} {' '.join(r['server_replaced']) or '-':16} "
                   f"\"{r['text']}\"" + ("" if r["parsed"] else "  [parser: no exact reading]")
                   + ("" if r["recalled"] else "  [recalled: false]"))
    agree = sum(r["supersedes"] == r["server_replaced"] for r in rows)
    out.append(f"\n{len(rows)} lessons; parsed back exactly: {sum(r['parsed'] for r in rows)}; "
               f"our supersedes = the server's replaced: {agree} of {len(rows)}; "
               + ", ".join(f"{v} {sum(r['verdict'] == v for r in rows)}" for v in VERDICTS))
    return "\n".join(out), rows


# ============================================================================================= self-test
GOLDEN_TODAY = "2026-10-12"
DATED_EYES = "My eyes were brown until 11 October 2026; now they are green."
# (what it pins, the store as sentences taught in order, the new sentence, the verdict, extra fields that must match)
GOLDEN = [
    ("the same fact twice", ["My eyes are green."], "My eyes are green.", ALREADY_KNOWN, {"same_as": [1]}),
    ("'green now' after 'green'", ["My eyes are green."], "My eyes are green now.", ALREADY_KNOWN, {"same_as": [1]}),
    ("another wording of the same fact", ["I like blue best."], "My favourite colour is blue.", ALREADY_KNOWN, {}),
    ("'I moved to Oslo' after 'I live in Oslo'", ["I live in Oslo."], "I moved to Oslo.", ALREADY_KNOWN, {}),
    ("a new name said with 'now', the same name", ["My cat is called Luna."],
     "My cat has a new name; it is called Luna now.", ALREADY_KNOWN, {}),
    ("an open name the lists lack", ["My dog is called Rex."], "My dog is named Rex.", ALREADY_KNOWN, {}),
    ("the same dated fact in other words", [DATED_EYES], "My eyes changed from brown to green on 11 October 2026.",
     ALREADY_KNOWN, {}),
    ("a taking-back said twice", ["My favourite colour is red.", "Red is not my favourite colour anymore."],
     "Red is not my favourite colour anymore.", ALREADY_KNOWN, {"same_as": [2]}),
    ("likes: the same food again", ["I like apples."], "I really like apples.", ALREADY_KNOWN, {}),
    ("a dated fact covers the plain fact", [DATED_EYES], "My eyes are green.", IMPLIED, {"covered_by": [1]}),
    ("'since D' is half of a dated fact", [DATED_EYES], "My eyes have been green since 11 October 2026.", IMPLIED,
     {"covered_by": [1]}),
    ("'not green anymore' when the eyes are brown", ["My eyes are brown."], "My eyes are not green anymore.",
     IMPLIED, {}),
    ("'not brown anymore' when the eyes are green", ["My eyes are green."], "My eyes are no longer brown.", IMPLIED,
     {}),
    ("nothing known yet", [], "My eyes are green.", NEW, {}),
    ("another entity's fact is new for me", ["My sister has green eyes."], "My eyes are green.", NEW, {}),
    ("a past-only fact is new; the current value stays", ["My eyes are green."],
     "My eyes were brown until 11 October 2026.", NEW, {"supersedes": []}),
    ("'always' adds the past", ["My eyes are green."], "My eyes have always been green.", NEW, {}),
    ("a dated fact after brown and green: the current value stays, green no longer decides it",
     ["My eyes are brown.", "My eyes are green."], DATED_EYES, NEW, {"supersedes": [2]}),
    ("likes: another food is new", ["I like apples."], "I like cheese.", NEW, {}),
    ("likes: taking one back is new, never REPLACES", ["I like apples."], "I do not like apples anymore.", NEW, {}),
    ("likes: 'only cheese now' is new", ["I like apples.", "I like cheese."], "I only like cheese now.", NEW, {}),
    ("a change after 'not red anymore' is new: the value was not known", ["Red is not my favourite colour anymore."],
     "My favourite colour is blue now.", NEW, {"supersedes": [1]}),
    ("a new value replaces the old", ["My eyes are brown."], "My eyes are green.", REPLACES, {"supersedes": [1]}),
    ("a change of mind (L2, lesson 2)", ["Of all colours, I like white the most."], "I like green best now.",
     REPLACES, {"supersedes": [1]}),
    ("taking back the value that holds (L2, lesson 3): the change still decides the answer",
     ["Of all colours, I like white the most.", "I like green best now."], "Green is not my favourite colour anymore.",
     REPLACES, {"supersedes": []}),
    ("a dated fact over a plain one changes the current value", ["My eyes are brown."], DATED_EYES, REPLACES,
     {"supersedes": [1]}),
    ("'no favourite anymore' replaces the value", ["My favourite colour is red."], "I have no favourite colour anymore.",
     REPLACES, {}),
    ("a value after 'none' replaces 'none'", ["My favourite colour is red.", "I have no favourite colour anymore."],
     "My favourite colour is red.", REPLACES, {"supersedes": [2]}),
    ("said before, overridden since: not already known", ["My eyes are green.", "My eyes are brown."],
     "My eyes are green.", REPLACES, {"supersedes": [2]}),
    ("moving back", ["I live in Oslo.", "I moved to Porto."], "I moved to Oslo.", REPLACES, {}),
    ("the #678 case: green after brown", ["I like blue.", "My eyes are brown."], "My eyes are green.", REPLACES,
     {"supersedes": [2]}),
]
# through the gate (only NEW and REPLACES are learned): the plain fact after the dated one is ALREADY_KNOWN here,
# because the first "green" is still in the store; above, with the dated fact alone, it is IMPLIED
GATE = (["My eyes are green.", "My eyes are green.", "My eyes are green now.", DATED_EYES, "My eyes are green.",
         "My eyes are brown now."], [NEW, ALREADY_KNOWN, ALREADY_KNOWN, NEW, ALREADY_KNOWN, REPLACES])
REFUSED = ["My mother has green eyes.", "My eyes are sparkly.", "Tell me a joke.", "My eyes are green and blue.",
           "My eyes are green until tomorrow.", "My eyes were brown until 31 February 2026; now they are green.",
           "My name is Ana.", "My cat likes blue.", "My car is called red now.", "I like.", "Eyes are green.",
           "My eyes are not green.", "My favourite colour is my sister.", "My eyes are green?"]


def gate(texts, today, store=None, verdict=None):
    """Teach sentences through the gate: each is sorted against the store, and only NEW and REPLACES are kept."""
    store, out = list(store or []), []
    for t in texts:
        l = place(store, parse(t), today)
        v = (verdict or (lambda s, x: classify(s, x)["verdict"]))(store, l)
        out.append(v)
        if LEARN[v]:
            store.append(l)
    return out, store


def golden_misses(verdict_fn=None, today=GOLDEN_TODAY):
    """Golden cases whose verdict (or pinned field) is wrong. `verdict_fn(store, lesson)` -> a classify() result;
    the default is the real classifier."""
    fn = verdict_fn or (lambda s, x: classify(s, x))
    miss = []
    for name, texts, new, want, extra in GOLDEN:
        store = teach_texts(texts, today)
        r = fn(store, place(store, parse(new), today))
        got = {k: r.get(k) for k in extra}
        if r["verdict"] != want or got != extra:
            miss.append(f"{name}: expected {want} {extra or ''}, got {r['verdict']} {got or ''}")
    got, _ = gate(GATE[0], today, verdict=lambda s, x: fn(s, x)["verdict"])
    if got != GATE[1]:
        miss.append(f"through the gate: expected {GATE[1]}, got {got}")
    return miss


def wording_cases():
    """Every lesson wording in T and EXTRA, for every entity it applies to and every list value, rendered the way
    the generator renders it; yields (sentence, lesson, slot, para, source)."""
    days = ["2025-03-03", "2026-10-11", "2024-12-25", "2026-01-09"]
    i = 0
    for a in L.ATTR_ORDER:
        dom = L.SCHEMA[a]["domain"]
        for src, table in (("T", L.T[a]), ("extra", EXTRA.get(a, {}))):
            for slot in SLOT_OP:
                for k, tmpl in enumerate(table.get(slot, [])):
                    for e, ent in L.ENTITIES.items():
                        if ent["kind"] not in L.SCHEMA[a]["applies"]:
                            continue
                        for j, v in enumerate(dom):
                            d = days[i % len(days)]
                            i += 1
                            l = {"op": SLOT_OP[slot], "e": e, "a": a}
                            if slot == "DATED":
                                l.update(v0=v, d=d, v1=dom[(j + 1) % len(dom)])
                            else:
                                l["v"] = None if slot == "CLEAR" else v
                                if slot in ("SINCE", "UNTIL"):
                                    l["when"] = {slot.lower(): d}
                                elif slot == "ALWAYS_A":
                                    l["when"] = {"always": True}
                            text = L.render(tmpl, e, v=l.get("v"), v0=l.get("v0"), v1=l.get("v1"), d=L.say_date(d))
                            yield text, l, slot, k, src
                            if slot == "CLEAR":
                                break


def round_trip_misses(read=None):
    """Sentences of `wording_cases` that do not read back to exactly their lesson and wording."""
    read = read or readings
    miss, n = [], 0
    for text, l, slot, k, src in wording_cases():
        n += 1
        rs = read(text)
        if len(rs) != 1 or formal(rs[0][0]) != formal(l) or rs[0][1] != {"slot": slot, "para": k, "source": src}:
            miss.append(f"{text!r}: {len(rs)} readings, first {rs[0][0] if rs else None}")
    return miss, n


class MockEndpoint:
    """Stands in for a chat endpoint: replies from a table, and keeps every request for inspection."""

    def __init__(self, replies):
        self.replies, self.calls = replies, []

    def __call__(self, url, body, headers, timeout):
        self.calls.append((url, body, headers))
        text = body["messages"][-1]["content"]
        return {"choices": [{"message": {"role": "assistant", "content": self.replies[text]}}]}


def extractor_cases():
    """(sentence, the mock model's reply, the lesson it must give or None if it must be refused)."""
    me = lambda **kw: dict({"e": "me"}, **kw)
    return [
        ("My eyes are green now.", '{"op": "CHANGE", "e": "me", "a": "eye_colour", "v": "green"}',
         me(op="CHANGE", a="eye_colour", v="green")),
        ("Since 11 October my eyes are green, before they were brown.",
         '```json\n{"op": "DATED", "e": "me", "a": "eye_colour", "v0": "brown", "d": "2026-10-11", "v1": "green"}\n```',
         me(op="DATED", a="eye_colour", v0="brown", d="2026-10-11", v1="green")),
        ("Gray eyes, that's me.", 'Sure! {"op": "ASSERT", "e": "me", "a": "eye_colour", "v": "Gray"} Hope it helps.',
         me(op="ASSERT", a="eye_colour", v="grey")),
        ("I've lived in Graz since March 2025.",
         '{"op": "ASSERT", "e": "me", "a": "home_city", "v": "Graz", "when": {"since": "2025-03-01"}}',
         me(op="ASSERT", a="home_city", v="Graz", when={"since": "2025-03-01"})),
        ("Honestly I have stopped having a favourite colour.", '{"op": "RETRACT", "e": "me", "a": "fav_colour", "v": null}',
         me(op="RETRACT", a="fav_colour", v=None)),
        ("My mum has blue eyes.", '{"op": "ASSERT", "e": "my mother", "a": "eye_colour", "v": "blue"}', None),
        ("My eyes sparkle.", '{"op": "ASSERT", "e": "me", "a": "eye_colour", "v": "sparkly"}', None),
        ("I liked olives until May, now cheese.",
         '{"op": "DATED", "e": "me", "a": "likes", "v0": "olives", "d": "2026-05-01", "v1": "cheese"}', None),
        ("From next year my eyes will be green.",
         '{"op": "ASSERT", "e": "me", "a": "eye_colour", "v": "green", "when": {"since": "2027-01-01"}}', None),
        ("I am called Ana.", '{"op": "ASSERT", "e": "me", "a": "name", "v": "Ana"}', None),
        ("What is the weather?", "I cannot turn that into a fact.", None),
        ("Hello there.", '{"op": "NONE"}', None),
        ("My eyes are green.", '{"op": "ASSERT", "e": "me", "a": "eye_colour", "v": "green", "when": "always"}', None),
        ("My mum and I have blue eyes.", '{"op": "ASSERT", "e": ["me", "my mother"], "a": "eye_colour", "v": "blue"}',
         None),
        ("Blue eyes, also for my cat.", '{"op": "ASSERT", "e": "me", "a": {"eye_colour": "blue"}, "v": "blue"}', None),
        ("Green eyes since 11 October.",
         '{"op": "ASSERT", "e": "me", "a": "eye_colour", "v": "green", "when": {"since": "20261011"}}', None),
        ("Green eyes, in parts.", [{"type": "text", "text": '{"op": "ASSERT", "e": "me", "a": "eye_colour", "v": "green"}'}],
         None),
    ]


def extractor_misses(today=GOLDEN_TODAY):
    cases = extractor_cases()
    mock = MockEndpoint({t: r for t, r, _ in cases})
    miss = []
    for text, _, want in cases:
        try:
            got, bad, _ = extract(text, "http://mock.invalid/v1", "mock-model", today, api_key="test-key", post=mock)
        except Exception as ex:                      # a reply must be refused, never crash the intake
            miss.append(f"{text!r}: crashed, {type(ex).__name__}: {ex}")
            continue
        if (got is None) != (want is None) or (want is not None and formal(got) != formal(want)):
            miss.append(f"{text!r}: expected {want}, got {got} {bad}")
    url, body, headers = mock.calls[0]
    if url != "http://mock.invalid/v1/chat/completions" or body["model"] != "mock-model" or body["temperature"] != 0 \
            or headers.get("Authorization") != "Bearer test-key" or len(mock.calls) != len(cases):
        miss.append(f"request not as specified: {url} {body.get('model')} {sorted(headers)}")
    return miss, len(cases)


# broken classifiers, each modelled on a plausible mistake; every one must be caught. Same call as `classify`.
def mut_current_only(s, l, ents=None, sch=None):
    """compares only the current value, so a lesson about the past looks known"""
    return classify(s, l, ents, sch, questions=lambda sch_, st, x: [{"type": "CURRENT", "e": x["e"], "a": x["a"]}])


def mut_text(s, l, ents=None, sch=None):
    """compares sentences: the same words are known, anything else is new"""
    hit = [m["id"] for m in s if norm(m.get("text", "")).lower() == norm(l.get("text", "")).lower()]
    return {"verdict": ALREADY_KNOWN if hit else NEW, "same_as": hit, "covered_by": hit, "supersedes": []}


def mut_hints_count(s, l, ents=None, sch=None):
    """counts the hints as part of what a lesson says, so 'green now' is not the same as 'green'"""
    def full(sch_, x):
        t = x["told"]
        return sorted(((e, a, ("told" if lo == t else "told-1" if lo == t - 1 else lo, "told" if hi == t else hi), eff)
                       for (e, a, (lo, hi), eff) in L.writes(sch_, x)), key=repr)
    return classify(s, l, ents, sch, same=lambda sch_, m, x: full(sch_, m) == full(sch_, x))


def mut_same_value(s, l, ents=None, sch=None):
    """calls a lesson the same when it names the same value, whatever its time"""
    last = lambda x: x.get("v1") or x.get("v")
    return classify(s, l, ents, sch, same=lambda sch_, m, x: "RETRACT" not in (m["op"], x["op"]) and last(m) == last(x))


def mut_replace_any(s, l, ents=None, sch=None):
    """calls every change of the current answer REPLACES, on likes and from 'don't know' too"""
    return classify(s, l, ents, sch, replaces=lambda sch_, a, b, x: x != b)


CLASSIFIER_MUTANTS = [("current only", mut_current_only, ["agree", "golden"]), ("sentence match", mut_text, ["golden"]),
                      ("hints count as the claim", mut_hints_count, ["golden"]),
                      ("same value is the same lesson", mut_same_value, ["golden"]),
                      ("every change of the current answer is REPLACES", mut_replace_any, ["golden"])]


def prop_agree(verdict_fn, rng, n, sch, ents, qs):
    """The verdict says "learn" exactly when the lesson moves an answer to an independent question set (the
    self-test's own questions: its fixed days, the day after each, and today). Lessons are drawn so that about
    half repeat or rephrase the store: a lesson already in it, or a random lesson on a key it already has."""
    r = {"n": n, "bad": 0, "ex": "", "kept": 0, "learned": 0}
    for _ in range(n):
        S = L.rand_store(rng, sch, ents, rng.randint(0, 8))
        x = rng.random()
        if S and x < 0.35:
            l = dict(rng.choice(S))
        elif S and x < 0.75:
            k = rng.choice(S)
            l = L.rand_lesson(rng, 0, sch, ents, key=(k["e"], k["a"]))
        else:
            l = L.rand_lesson(rng, 0, sch, ents)
        l = dict(l, id=len(S) + 1, name="new", told=L.tick(L.TEST_TODAY) + 100 * (len(S) + 1))
        v = verdict_fn(S, l, ents, sch)
        kq = L.key_qs(qs, l["e"], l["a"]) + [q for q in qs if q["type"] == "WHO_HAS" and q["a"] == l["a"]]
        moved = L.answers_of(sch, S, ents, kq) != L.answers_of(sch, S + [l], ents, kq)
        r["learned" if LEARN[v] else "kept"] += 1
        if moved != LEARN[v]:
            r["bad"] += 1
            r["ex"] = r["ex"] or f"{v} but answers {'moved' if moved else 'did not move'}: {L.op_text(l)} after " \
                                 f"{[L.op_text(m) for m in S]}"
    return r


def prop_gate(verdict_fn, rng, n, sch, ents, qs):
    """Keeping a lesson out loses nothing: teach a random sequence through the gate (only what `verdict_fn` says to
    learn joins the store) and in full, and every answer must agree at the end. Sequences stay on one or two keys
    and often repeat an earlier lesson, so that many lessons are kept out."""
    r = {"n": n, "bad": 0, "ex": "", "kept": 0}
    for _ in range(n):
        keys = [(rng.choice(ents), rng.choice(list(sch))) for _ in range(rng.randint(1, 2))]
        seq = [L.rand_lesson(rng, i + 1, sch, ents, key=rng.choice(keys)) for i in range(rng.randint(2, 8))]
        if rng.random() < 0.5:
            j = rng.randrange(len(seq))
            seq.insert(rng.randint(j + 1, len(seq)), dict(seq[j]))
        seq = [dict(l, id=i + 1, name=f"L{i + 1}", told=L.tick(L.TEST_TODAY) + 100 * (i + 1)) for i, l in enumerate(seq)]
        gated = []
        for l in seq:
            if LEARN[verdict_fn(gated, l, ents, sch)]:
                gated.append(l)
        r["kept"] += len(seq) - len(gated)
        if L.answers_of(sch, gated, ents, qs) != L.answers_of(sch, seq, ents, qs):
            r["bad"] += 1
            r["ex"] = r["ex"] or f"kept out {[L.op_text(l) for l in seq if l not in gated]} of {[L.op_text(l) for l in seq]}"
    return r


def cli_misses():
    """Run `main` in-process, output captured, nothing written: no store; a test set's phase; an unreadable sentence;
    --phase without a test set and a bad --today (refused with exit 2). Then the day a store file tells lessons on."""
    import contextlib
    import io
    l2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples", "L2-seed1.json")
    # (arguments, exit code, verdicts on standard output, words the standard error must hold)
    runs = [(["--today", GOLDEN_TODAY, "--text", "My eyes are green.", "--text", "My eyes are green now."], 0,
             [NEW, ALREADY_KNOWN], "verdict"),
            (["--store", l2, "--phase", "del-A", "--text", "Green is not my favourite colour anymore."], 0, [REPLACES],
             "verdict"),
            (["--text", "My eyes are green.", "--text", "My mother is tall."], 1, [NEW, None], "verdict"),
            (["--phase", "teach", "--text", "My eyes are green."], 2, None, "--phase needs a test set"),
            (["--today", "2026-13-01", "--text", "My eyes are green."], 2, None, "is not a date")]
    miss = []
    for args, rc_want, want, err in runs:
        so, se = io.StringIO(), io.StringIO()
        try:
            with contextlib.redirect_stdout(so), contextlib.redirect_stderr(se):
                rc = main(args + ["--json", "-"])
            got = [r["verdict"] for r in json.loads(so.getvalue())]
        except SystemExit as ex:                     # argparse refusals exit with 2
            rc, got = ex.code, None
        except Exception as ex:
            rc, got = None, f"{type(ex).__name__}: {ex} {se.getvalue()[-200:]}"
        if rc != rc_want or got != want or err not in se.getvalue():   # the report goes to stderr
            miss.append(f"{' '.join(args[-2:])}: rc {rc}, {got}")
    # a store saved on 12 Oct and used on 20 Oct tells new lessons on 20 Oct; a test set keeps its own day
    days = [(("store", "2026-10-12", "2026-10-20"), "2026-10-20"), (("store", "2026-10-25", "2026-10-20"), "2026-10-25"),
            (("set", "2026-10-12", "2026-10-20"), "2026-10-12"), (("new", None, "2026-10-20"), "2026-10-20")]
    miss += [f"default_today{args} = {default_today(*args)}, expected {want}" for args, want in days
             if default_today(*args) != want]
    return miss, len(runs) + len(days)


def selftest_checks(C, n=1500, seed=1):
    """The intake checks, reported through lwlogic's Check (C.add). Called by `lwlogic.py selftest`."""
    miss, nrt = round_trip_misses()
    C.add("intake parser: every lesson wording in T reads back to its lesson and wording, for every entity and list "
          "value, with exactly one reading", not miss, nrt, "; ".join(miss[:2]))
    ignore_e = lambda t: [(dict(l, e="me"), w) for l, w in readings(t)]
    bmiss, _ = round_trip_misses(ignore_e)
    C.add("negative control: a parser that ignores the entity fails the round trip", len(bmiss) > 0, nrt,
          f"caught in {len(bmiss)} sentences")

    gbad, ng = [], 0
    for lv in (1, 2, 3, 4, 5):
        for sd in range(1, 31):
            for l in L.generate(lv, sd)["lessons"]:
                ng += 1
                got = parse(l["text"])
                if got is None or formal(got) != formal(l):
                    gbad.append(f"L{lv} seed {sd} {l['text']!r}: {got}")
    for l in L.generate("eyes", 1)["lessons"]:
        ng += 1
        if parse(l["text"]) is None or formal(parse(l["text"])) != formal(l):
            gbad.append(f"eyes-678 {l['text']!r}")
    C.add("intake parser: every lesson of the generated sets (levels 1-5 x seeds 1-30, and #678) reads back to its "
          "formal lesson", not gbad, ng, "; ".join(gbad[:2]))
    wrong = [(t, parse(t)) for t in REFUSED if readings(t)]
    C.add("intake parser: sentences outside the tables, unknown entities, values and dates are refused, not misread",
          not wrong, len(REFUSED), str(wrong[:2]) if wrong else "")

    miss = golden_misses()
    C.add("intake golden cases: same fact twice and 'green now' after 'green' are ALREADY_KNOWN, a plain fact under a "
          "dated one is IMPLIED, changes are NEW or REPLACES, and the gate", not miss, len(GOLDEN) + 1, "; ".join(miss[:2]))

    sch, ents = L.test_schema(), ["me", "my sister", "my cat"]
    qs = L.all_questions(sch, ents)
    nm = max(100, n // 5)
    real = lambda s, l, e, sc: classify(s, l, e, sc)["verdict"]
    r = prop_agree(real, random.Random(f"{seed}/intake/agree"), nm, sch, ents, qs)
    C.add("intake agrees with an independent question set: it says 'learn' exactly when the lesson moves an answer",
          r["bad"] == 0 and r["kept"] > 0 and r["learned"] > 0, r["n"],
          f"kept out {r['kept']}, learned {r['learned']}" + (f"; FIRST BREAK: {r['ex']}" if r["bad"] else ""))

    for name, fn, targets in CLASSIFIER_MUTANTS:
        res = {}
        if "golden" in targets:
            res["golden"] = len(golden_misses(fn))
        if "agree" in targets:
            res["agree"] = prop_agree(lambda s, l, e, sc: fn(s, l, e, sc)["verdict"],
                                      random.Random(f"{seed}/intake/{name}"), nm, sch, ents, qs)["bad"]
        C.add(f"negative control: intake classifier, {name} ({fn.__doc__.strip()})", all(res[t] > 0 for t in targets),
              nm if "agree" in targets else len(GOLDEN) + 1, "; ".join(f"caught by {t} in {c}" for t, c in res.items()))

    r = prop_gate(real, random.Random(f"{seed}/intake/gate"), nm, sch, ents, qs)
    C.add("intake gate loses nothing: a sequence taught through the gate gives every answer of the full sequence",
          r["bad"] == 0 and r["kept"] > 0, r["n"],
          f"kept out {r['kept']} lessons" + (f"; FIRST BREAK: {r['ex']}" if r["bad"] else ""))
    bad = prop_gate(lambda s, l, e, sc: mut_current_only(s, l, e, sc)["verdict"], random.Random(f"{seed}/intake/gate"),
                    nm, sch, ents, qs)["bad"]
    C.add("negative control: a gate that compares only the current value loses answers", bad > 0, nm,
          f"caught in {bad} sequences")

    miss, nc = cli_misses()
    C.add("intake command line: no store, a test set's phase, an unreadable sentence (exit 1), refused options (exit 2), "
          "as JSON; and the day a store file tells lessons on", not miss, nc, "; ".join(miss[:2]))

    miss, ne = extractor_misses()
    C.add("intake extractor (mock endpoint, no network): good replies accepted and normalised, bad ones refused",
          not miss, ne, "; ".join(miss[:2]))


# ====================================================================================================== cli
def show_result(text, l, how, r, store, out=None):
    """The plain-language report of one sentence (to `out`, standard output by default)."""
    p = lambda line: print(line, file=out or sys.stdout)
    p(f"\"{text}\"")
    if l is None:
        p(f"  not read   {how}")
        return
    p(f"  read as    {L.op_text(l)}   ({how})")
    canon = say(l)
    if how.startswith("extractor") and canon:          # a model's reading, said back in the rule book's words
        p(f"  which is   \"{canon}\"")
    p(f"  verdict    {r['verdict']}: {'learn it' if r['learn'] else 'do not learn it'}")
    name = lambda i: next((f"{m['name']} \"{m.get('text') or L.op_text(m)}\"" for m in store if m["id"] == i), str(i))
    if r["changed"]:
        cur = next(((b, x) for q, b, x in r["changed"] if q["type"] == "CURRENT"), None)
        if r["verdict"] == REPLACES:
            p(f"  why        the current value changes: {L.words(cur[0])} -> {L.words(cur[1])}")
        else:
            p(f"  why        {len(r['changed'])} of {r['questions']} answers change"
                  + (f"; the current answer goes from {L.words(cur[0])} to {L.words(cur[1])}" if cur else
                     "; the current answer stays"))
        for q, b, x in r["changed"][:6]:
            p(f"               {say_question(q)}  {L.words(b)} -> {L.words(x)}")
        if len(r["changed"]) > 6:
            p(f"               ... and {len(r['changed']) - 6} more")
        if r["supersedes"]:
            p(f"  supersedes {', '.join(name(i) for i in r['supersedes'])}: no longer decides the current answer")
    else:
        p(f"  why        no answer changes ({r['questions']} questions)" + (
            f"; {', '.join(name(i) for i in r['same_as'])} says the same" if r["same_as"] else
            "; no earlier lesson says the same, it follows from them"))
        if r["covered_by"]:
            p(f"  note       deleting {', '.join(name(i) for i in r['covered_by'])} would make it news again")


def write_json(path, obj):
    txt = json.dumps(obj, indent=1, ensure_ascii=False)
    if path == "-":
        print(txt)
    else:
        with open(path, "w") as f:
            f.write(txt + "\n")


def main(argv=None, prog=None):
    ap = argparse.ArgumentParser(prog=prog, description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", help="a test set (lw-test-set/1) or a store file (lw-store/1); absent = empty")
    ap.add_argument("--phase", help="with a test set: use this phase's store (default: every lesson)")
    ap.add_argument("--text", action="append", default=[], help="a lesson; repeat for several, taken in order through "
                                                                "the gate (only NEW and REPLACES join the store)")
    ap.add_argument("--today", help="the day the lessons are told (YYYY-MM-DD; default: a test set's own day, else "
                                    "today, and never before a store file's day)")
    ap.add_argument("--save", action="store_true", help="write the learned lessons back to --store (lw-store/1 only)")
    ap.add_argument("--json", help="write the results as JSON here ('-' for standard output)")
    ap.add_argument("--endpoint", help="OpenAI-compatible base URL; used when the parser cannot read a sentence")
    ap.add_argument("--model", help="model name for --endpoint")
    ap.add_argument("--api-key-env", help="name of the environment variable that holds the endpoint's key")
    ap.add_argument("--timeout", type=float, default=120)
    ap.add_argument("--records", nargs="+", help="run folders written by lwrun.py: compare with TensorFold's replaces")
    a = ap.parse_args(argv)

    out = sys.stderr if a.json == "-" else sys.stdout   # with JSON on standard output, the report goes to stderr
    if a.records:
        txt, rows = records_report(a.records)
        print(txt, file=out)
        if a.json:
            write_json(a.json, rows)
        return 0
    if not a.text:
        ap.error("give --text (or --records)")
    if a.endpoint and not a.model:
        ap.error("--endpoint needs --model")
    key = None
    if a.api_key_env:
        key = os.environ.get(a.api_key_env)
        if not key:
            ap.error(f"the environment variable {a.api_key_env} is not set")
    if a.today:
        try:
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", a.today):
                raise ValueError(a.today)
            dt.date.fromisoformat(a.today)
        except ValueError:
            ap.error(f"--today {a.today!r} is not a date (YYYY-MM-DD)")
    store, ents, s_today, kind = load_store(a.store, a.phase)
    if a.phase and kind != "set":
        ap.error("--phase needs a test set (lw-test-set/1) as --store")
    if a.save and not a.store:
        ap.error("--save needs --store")
    if a.save and kind == "set":
        ap.error("--save writes store files (lw-store/1) only; a test set is never changed")
    today = a.today or default_today(kind, s_today, dt.date.today().isoformat())
    where = (f" ({os.path.basename(a.store)}" + (f", phase {a.phase}" if a.phase else "")
             + (", new" if kind == "new" else "") + ")") if a.store else ""
    print(f"store: {len(store)} lessons{where}, told on {today}", file=out)
    results, unread = [], 0
    for text in a.text:
        rs = readings(text)
        l, how = (rs[0][0], f"wording table: {rs[0][0]['a']} {rs[0][1]['slot']} {rs[0][1]['para']}"
                  + (" (extra)" if rs[0][1]["source"] == "extra" else "")) if len(rs) == 1 else (None, "")
        if l is None:
            how = f"{len(rs)} readings in the wording tables" if rs else "not in the wording tables"
            if a.endpoint:
                l, bad, _ = extract(text, a.endpoint, a.model, today, key, timeout=a.timeout)
                how = f"extractor ({a.model})" if l else how + "; extractor: " + "; ".join(bad)
        bad = check_lesson(l, today) if l else []
        if bad:
            l, how = None, "; ".join(bad)
        if l is None:
            unread += 1
            show_result(text, None, how, None, store, out)
            results.append({"text": text, "verdict": None, "error": how})
            continue
        l = place(store, l, today)
        r = classify(store, l, ents)
        show_result(text, l, how, r, store, out)
        results.append({"text": text, "lesson": l, "read_by": how,
                        "verdict": r["verdict"], "learn": r["learn"], "questions": r["questions"],
                        "changed": [{"question": say_question(q), "before": L.words(b), "after": L.words(x)}
                                    for q, b, x in r["changed"]],
                        "same_as": r["same_as"], "covered_by": r["covered_by"], "supersedes": r["supersedes"]})
        if r["learn"]:
            store.append(l)
    if a.save:
        save_store(a.store, store, ents, today)
        print(f"saved {a.store}: {len(store)} lessons", file=out)
    if a.json:
        write_json(a.json, results)
    return 1 if unread else 0


if __name__ == "__main__":
    sys.exit(main())
