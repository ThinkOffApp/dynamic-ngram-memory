#!/usr/bin/env python3
"""levels: level 6 of the Living Weights tests, two-step questions (plan item 19 of docs/LITERATURE.md).

  python3 zengine/levels.py gen --seed 1 --out set.json      the same set as: lwlogic.py gen --level 6 --seed 1
  python3 zengine/levels.py selftest [--n 1500] [--seed 1]    the level 6 checks alone (lwlogic.py selftest runs them too)

A two-step question can only be answered by putting two facts together. Five families:

  natural   a taught fact, then general knowledge     "Which country do I live in?" (my city, then its country)
  owner     a taught fact, then another taught one     "What colour are the eyes of the owner of the boat?"
  alias     a name, then a fact about whoever has it   "What colour are Nora's eyes?" (my sister is called Nora)
  link      a taught linking fact                      "My cat's eyes are the same colour as mine."
  compound  yes/no parts joined by and, or, not        "Is it true that my eyes are green and my car is red?"

The rules live in the rule book (lwlogic.py, "two-step questions"), so every expected answer is computed, and in
lean/LwLogic.lean with theorems. This file makes the sets, judges the answers (adding the verdict "echo": an answer
that only repeats a step on the way, such as the city when the country was asked) and scores them: accuracy on the
single steps a chain rests on (the stated facts), on the chain (the implied fact), the gap between the two, and
accuracy on chains whose every single step was answered right in the same phase and thinking mode. Chains through a
natural step the base model gets wrong in the baseline are dropped from those scores.
Standard library only.
"""
import argparse
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import lwlogic as L  # noqa: E402  (the rule book, next to this file)
from lwlogic import NO, NOW, UNK, YES  # noqa: E402

# ============================================================================================== schema
# General knowledge, fixed: the country of each city of the home_city list. Two countries have two cities each, so a
# move can keep the country (Oslo to Bergen) or change it (Oslo to Porto).
COUNTRY = {"Lisbon": "Portugal", "Oslo": "Norway", "Tampere": "Finland", "Porto": "Portugal", "Ghent": "Belgium",
           "Graz": "Austria", "Bergen": "Norway", "Turku": "Finland"}
PEOPLE = ["me", "my sister", "my brother"]

SCHEMA6 = dict(L.SCHEMA)
SCHEMA6.update({
    # "My cat's eyes are the same colour as mine." = ASSERT(my cat, eyes_like, me)
    "eyes_like": {"multi": False, "applies": ["person", "animal"], "domain": PEOPLE, "extra": [], "link": "eye_colour",
                  "topic": ["same", "eye", "colour", "color"]},
    # "My brother lives in the same city as my sister." = ASSERT(my brother, city_like, my sister)
    "city_like": {"multi": False, "applies": ["person"], "domain": PEOPLE, "extra": [], "link": "home_city",
                  "topic": ["same", "city", "live"]},
    # never taught: the natural step of a chain, from home_city
    "country": {"multi": False, "applies": [], "domain": L.uniq(list(COUNTRY.values())),
                "extra": ["Germany", "France", "Sweden", "Spain", "Italy", "Denmark", "Netherlands", "Switzerland",
                          "Poland", "Estonia", "Iceland", "Ireland"],
                "topic": ["country", "live", "nation"], "natural": {"from": "home_city", "map": COUNTRY}},
})

T6 = dict(L.T)
T6["eyes_like"] = {"ASSERT": ["{poss} eyes are the same colour as {vposs}.", "{nom} {have} the same eye colour as {vobj}."],
                   "Q_YES_NO": ["Are {poss} eyes the same colour as {vposs}?", "{do} {nom} have the same eye colour as {vobj}?"]}
T6["city_like"] = {"ASSERT": ["{nom} live{s} in the same city as {vobj}.", "{poss} home is in the same city as {vposs}."],
                   "Q_YES_NO": ["{do} {nom} live in the same city as {vobj}?"]}

# chain wordings: (start is a name, hops, final attribute, natural step, yes/no form) -> wordings
CHAIN_Q = {
    (False, (), "home_city", True, False): ["Which country {do} {nom} live in?", "In which country {do} {nom} live?"],
    (False, (), "home_city", True, True): ["{do} {nom} live in {v}?"],
    (False, ("owner",), "eye_colour", False, False): ["What colour are the eyes of the owner of {nom}?",
                                                       "What colour are the eyes of the person who owns {nom}?"],
    (False, ("owner",), "eye_colour", False, True): ["Does the owner of {nom} have {v} eyes?"],
    (False, ("owner",), "home_city", False, False): ["Where does the owner of {nom} live?",
                                                      "In which city does the owner of {nom} live?"],
    (False, ("owner",), "home_city", True, False): ["In which country does the owner of {nom} live?"],
    (False, ("owner",), "fav_colour", False, False): ["What is the favourite colour of the owner of {nom}?"],
    (True, (), "eye_colour", False, False): ["What colour are {n}'s eyes?", "What is the colour of {n}'s eyes?"],
    (True, (), "eye_colour", False, True): ["Are {n}'s eyes {v}?"],
    (True, (), "home_city", False, False): ["Where does {n} live?", "In which city does {n} live?"],
    (True, (), "home_city", True, False): ["Which country does {n} live in?", "In which country does {n} live?"],
}
# a plain statement and its negation, for compound questions ("Is it true that my eyes are green and ...?")
STATE6 = {
    "eye_colour": ("{poss} eyes are {v}", "{poss} eyes are not {v}"),
    "fav_colour": ("{poss} favourite colour is {v}", "{poss} favourite colour is not {v}"),
    "name": ("{nom} is called {v}", "{nom} is not called {v}"),
    "colour": ("{nom} is {v}", "{nom} is not {v}"),
    "home_city": ("{nom} live{s} in {v}", "{nom} {do} not live in {v}"),
    "likes": ("{nom} like{s} {v}", "{nom} {do} not like {v}"),
    "owner": ("{nom} belongs to {v}", "{nom} does not belong to {v}"),
}
NATURAL_Q = ["Which country is {c} in?", "In which country is {c}?"]
FAMILIES = ["natural", "owner", "alias", "link", "compound"]


def vforms(v):
    """How a value that names a person reads after "the same colour as": mine / my sister's, me / my sister."""
    if v is None:
        return {}
    return {"vposs": "mine" if v == "me" else f"{v}'s", "vobj": v}


def statement(atom, negated=False):
    f = dict(L.ENTITIES[atom["e"]])
    f.update(v=atom["v"])
    return STATE6[atom["a"]][1 if negated else 0].format(**f)


# ============================================================================================ generator
class Builder6(L.Builder):
    """lwlogic's Builder, with the level 6 wordings and schema, chain and compound questions."""

    def lesson(self, op, e, a, v=None, **_):
        lid = len(self.lessons) + 1
        self.ent(e)
        l = {"id": lid, "name": f"L{lid}", "op": op, "e": e, "a": a, "told": L.tick(self.today) + 100 * lid, "v": v}
        paras = T6[a][{"ASSERT": "ASSERT", "CHANGE": "CHANGE", "RETRACT": "RETRACT"}[op]]
        k = self.rng.randrange(len(paras))
        l["text"], l["para"] = L.render(paras[k], e, v=v, **vforms(v)), k
        self.lessons.append(l)
        return l

    def _add(self, lst, q, kw, paras=1):
        ks = list(range(len(lst)))
        self.rng.shuffle(ks)
        out = []
        for k in sorted(ks[:paras]):
            text = L.render(lst[k], q.get("e") or q.get("start"), **kw)
            if any(x["text"] == text for x in self.questions):
                continue
            x = dict(q, qid=f"Q{len(self.questions) + 1:02d}", text=text, para=k)
            x.setdefault("role", "probe")
            for e in (x.get("e"), x.get("start")):
                if e:
                    self.ent(e)
            self.questions.append(x)
            out.append(x)
        return out

    def ask(self, qtype, e=None, a=None, role="probe", paras=1, **kw):
        lst = T6[a].get("Q_" + qtype) or []
        q = {"type": qtype, "a": a, "role": role}
        if e is not None:
            q["e"] = e
        q.update({k: v for k, v in kw.items() if k in ("v", "x", "y", "date")})
        fkw = dict(kw, **vforms(kw.get("v")))
        if "date" in kw:
            fkw["d"] = L.say_date(kw["date"])
        return self._add(lst, q, fkw, paras)

    def chain(self, family, start=None, name=None, hops=(), final=None, nat=False, v=None, paras=1):
        """A chain question. Link questions (no hops, the final attribute linked) use the plain wordings."""
        q = {"type": "CHAIN", "family": family, "hops": list(hops), "final": final, "a": "country" if nat else final}
        if name is not None:
            q["name"] = name
        else:
            q["start"] = start
        if nat:
            q["nat"] = "country"
        if v is not None:
            q["v"] = v
        if family == "link":
            lst = T6[final]["Q_YES_NO" if v is not None else "Q_CURRENT"]
        else:
            lst = CHAIN_Q[(name is not None, tuple(hops), final, nat, v is not None)]
        return self._add(lst, q, {"n": name, "v": v}, paras)

    def compound(self, op, *parts):
        """A compound yes/no question on atoms ({e, a, v}: "my eyes are green"). "not" is said as the negated
        statement: "Is it true that my car is not red?"."""
        if op == "not":
            text = f"Is it true that {statement(parts[0], True)}?"
        else:
            text = f"Is it true that {statement(parts[0])} {op} {statement(parts[1])}?"
        fp = [dict(p, type="YES_NO") for p in parts]
        if any(x["text"] == text for x in self.questions):
            return []
        q = {"qid": f"Q{len(self.questions) + 1:02d}", "type": "COMPOUND", "family": "compound", "op": op,
             "parts": fp, "a": None, "text": text, "para": 0, "role": "probe"}
        self.questions.append(q)
        return [q]

    def natural(self, city):
        c = COUNTRY[city]
        k = self.rng.randrange(len(NATURAL_Q))
        text = NATURAL_Q[k].format(c=city)
        if any(x["text"] == text for x in self.questions):
            return []
        q = {"qid": f"Q{len(self.questions) + 1:02d}", "type": "GK", "a": None, "text": text, "para": k,
             "any_of": [[c.lower()]], "role": "natural", "city": city}
        self.questions.append(q)
        return [q]

    def stores(self):
        """The lessons in force in each phase, before the set is built."""
        out = {"baseline": [], "teach": list(self.lessons)}
        for i, v in enumerate(self.variants):
            out[f"del-{chr(65 + i)}"] = [l for l in self.lessons if l["id"] not in v["deleted"]]
        return out

    def build(self):
        used = {l["a"] for l in self.lessons}
        for q in self.questions:
            used |= q_attrs(q)
        sch = {a: SCHEMA6[a] for a in SCHEMA6 if a in used}
        ids = [l["id"] for l in self.lessons]
        phases = [{"name": "baseline", "deleted": ids, "note": "before any teaching (the original model)"},
                  {"name": "teach", "deleted": [], "note": "every lesson taught, in order"}]
        for i, v in enumerate(self.variants):
            phases.append({"name": f"del-{chr(65 + i)}", "deleted": v["deleted"], "note": v["label"]})
        for ph in phases:
            ph["store"] = [i for i in ids if i not in ph["deleted"]]
        S = {"format": "lw-test-set/1", "level": 6, "seed": self.seed, "today": self.today, "id": f"L6-seed{self.seed}",
             "entities": self.ents, "schema": sch, "lessons": self.lessons, "questions": self.questions,
             "phases": phases, "notes": self.notes}
        L.fill_expected(S)
        return S


def q_attrs(q):
    """Every schema attribute a question needs: its own, a chain's hops, final, link and natural attributes."""
    if q["type"] == "COMPOUND":
        return set().union(*(q_attrs(p) for p in q["parts"]))
    if q["type"] == "GK":
        return set()
    if q["type"] != "CHAIN":
        return {q["a"]}
    out = set(q["hops"]) | {q["final"], q["a"]} | ({"name"} if q.get("name") is not None else set())
    return out | {x for x in SCHEMA6 if SCHEMA6[x].get("link") == q["final"]}


def linked_keys(lessons, sch=SCHEMA6):
    """(e, a) pairs that have a linking lesson for a: a plain question on them would ignore the link."""
    return {(l["e"], sch[l["a"]]["link"]) for l in lessons if sch[l["a"]].get("link")}


def gen_L6(b):
    r = b.rng
    S6 = SCHEMA6
    c_me, c_sis, c_bro, c_dog = r.sample(S6["eye_colour"]["domain"], 4)
    n_sis, n_bro, n_cat, n_dog, n_dog2, n_wrong = r.sample(S6["name"]["domain"], 6)
    f_me, f_me2, f_sis, f_bro = r.sample(S6["fav_colour"]["domain"], 4)
    k_car, k_boat, k_tent = r.sample(S6["colour"]["domain"], 3)
    food1, food2, food3, food4, food5 = r.sample(S6["likes"]["domain"], 5)
    o1, o2 = r.sample(["my sister", "my brother"], 2)
    o_tent = r.choice(["my neighbour", "me"])
    cities = S6["home_city"]["domain"]
    c1 = r.choice(cities)
    same = [c for c in cities if c != c1 and COUNTRY[c] == COUNTRY[c1]]
    move = "within the country" if same and r.random() < 0.5 else "to another country"
    c2 = r.choice(same if move == "within the country" else [c for c in cities if COUNTRY[c] != COUNTRY[c1]])
    c3 = r.choice([c for c in cities if c not in (c1, c2)])
    chains = [
        [("ASSERT", "me", "eye_colour", c_me)], [("ASSERT", "my sister", "eye_colour", c_sis)],
        [("ASSERT", "my brother", "eye_colour", c_bro)], [("ASSERT", "my dog", "eye_colour", c_dog)],
        [("ASSERT", "my cat", "eye_colour", c_me)],          # said outright, and also implied by the link below
        [("ASSERT", "my cat", "eyes_like", "me")],
        [("ASSERT", "me", "fav_colour", f_me), ("CHANGE", "me", "fav_colour", f_me2)],
        [("ASSERT", "my sister", "fav_colour", f_sis)],
        [("ASSERT", "my sister", "name", n_sis)], [("ASSERT", "my brother", "name", n_bro)],
        [("ASSERT", "my cat", "name", n_cat)], [("ASSERT", "my dog", "name", n_dog), ("CHANGE", "my dog", "name", n_dog2)],
        [("ASSERT", "my car", "colour", k_car)], [("ASSERT", "the boat", "colour", k_boat)],
        [("ASSERT", "the tent", "colour", k_tent)],
        [("ASSERT", "me", "home_city", c1), ("CHANGE", "me", "home_city", c2)],
        [("ASSERT", "my sister", "home_city", c3)], [("ASSERT", "my brother", "city_like", "my sister")],
        [("ASSERT", "me", "likes", food1), ("ASSERT", "me", "likes", food2), ("ASSERT", "me", "likes", food3),
         ("RETRACT", "me", "likes", food2)],
        [("ASSERT", "my dog", "likes", food4)],
        [("ASSERT", "the boat", "owner", o1), ("CHANGE", "the boat", "owner", o2)],
        [("ASSERT", "the tent", "owner", o_tent)],
    ]
    made = {}
    while any(chains):                       # a random interleave that keeps each chain's own order
        c = r.choice([c for c in chains if c])
        op, e, a, v = c.pop(0)
        made[(op, e, a, v)] = b.lesson(op, e, a, v)
    mid = lambda *k: made[k]["id"]
    b.variant([mid("ASSERT", "my cat", "eye_colour", c_me)],
              "deep delete: the cat's own eye colour deleted. The link to my eyes still implies it, so it must stay known")
    b.variant([mid("ASSERT", "the boat", "owner", o1), mid("CHANGE", "the boat", "owner", o2),
               mid("ASSERT", "my sister", "name", n_sis), mid("ASSERT", "my brother", "city_like", "my sister")],
              "one link of each chain deleted (the boat's owner, my sister's name, my brother's city link): "
              "the chains through them must become don't know")
    b.variant([mid("CHANGE", "the boat", "owner", o2), mid("CHANGE", "me", "home_city", c2),
               mid("CHANGE", "my dog", "name", n_dog2)],
              "the changes deleted: the chains through them must go back to the value before the change")

    # single steps: the stated facts the chains rest on (any missing one is added below)
    for e, a in [("me", "home_city"), ("my sister", "home_city"), ("the boat", "owner"), ("the tent", "owner"),
                 ("me", "eye_colour"), ("my sister", "eye_colour"), ("my brother", "eye_colour"),
                 ("my dog", "eye_colour"), ("my sister", "name"), ("my brother", "name"), ("my cat", "name"),
                 ("my dog", "name")]:
        b.ask("CURRENT", e, a)
    b.ask("YES_NO", "my cat", "eyes_like", v="me")
    b.ask("YES_NO", "my brother", "city_like", v="my sister")
    # chains
    b.chain("natural", start="me", final="home_city", nat=True)
    b.chain("natural", start="me", final="home_city", nat=True, v=COUNTRY[c2])
    b.chain("natural", start="my brother", final="home_city", nat=True)
    b.chain("link", start="my cat", final="eye_colour")
    b.chain("link", start="my brother", final="home_city")
    b.chain("owner", start="the boat", hops=["owner"], final="eye_colour")
    b.chain("owner", start="the boat", hops=["owner"], final="eye_colour",
            v={"my sister": c_sis, "my brother": c_bro}[o2])
    b.chain("owner", start="the boat", hops=["owner"], final="home_city")
    b.chain("owner", start="the boat", hops=["owner"], final="home_city", nat=True)
    b.chain("owner", start="the boat", hops=["owner"], final="fav_colour")
    b.chain("owner", start="the tent", hops=["owner"], final="eye_colour")
    b.chain("alias", name=n_sis, final="eye_colour")
    b.chain("alias", name=n_sis, final="home_city", nat=True)
    b.chain("alias", name=n_dog2, final="eye_colour")
    b.chain("alias", name=n_cat, final="eye_colour")
    b.chain("alias", name=n_bro, final="eye_colour", v=c_bro)
    # compounds, one of each Kleene case in the teach phase; their parts are also asked alone
    yn = lambda e, a, v: {"e": e, "a": a, "v": v}
    Y1, Y2, Y3 = yn("me", "eye_colour", c_me), yn("the boat", "owner", o2), yn("my car", "colour", k_car)
    N1, N2 = yn("my sister", "name", n_wrong), yn("me", "likes", food2)
    U1, U2 = yn("my brother", "fav_colour", f_bro), yn("my dog", "likes", food5)
    for p, q in [(Y1, Y3), (U1, N2), (Y2, U2)]:
        b.compound("and", p, q)
    for p, q in [(U1, Y1), (N1, U2), (N2, N1)]:
        b.compound("or", p, q)
    b.compound("not", Y3)
    b.compound("not", U2)
    for x in (Y1, Y2, Y3, N1, N2, U1, U2):
        b.ask("YES_NO", x["e"], x["a"], v=x["v"])
    # every single step of every chain, in every phase, has its own question
    for store in b.stores().values():
        for q in [x for x in b.questions if x["type"] == "CHAIN"]:
            for st in chain_trace(SCHEMA6, store, b.ents, q):
                if step_of(b.questions, st) is None:
                    add_step(b, st)
    # controls; an "untouched" pair is one never taught, never linked and not asked above
    keys = {(l["e"], l["a"]) for l in b.lessons if l["a"] in L.SCHEMA} | linked_keys(b.lessons) \
        | {(q["e"], q["a"]) for q in b.questions if q.get("e") and q["a"] in L.SCHEMA}
    b.controls(keys, extra_near=[("my cousin", "eye_colour"), ("my rabbit", "name")])
    b.notes.append(
        f"Two-step questions. My home moves {move} ({c1} to {c2}). The boat's owner changes from {o1} to {o2}, and "
        f"my dog's name from {n_dog} to {n_dog2}. My cat's eye colour is said outright and also implied by a link to "
        f"mine; my brother's city is only implied, by a link to my sister's. The tent belongs to {o_tent}"
        + (", of whom nothing else is taught." if o_tent == "my neighbour" else "."))
    b.notes.append("A chain asks about now. A name points at the one entity called that now. A link is read one way "
                   "and one link deep, and only fills gaps: what is said outright wins. 'Or' is inclusive; compound "
                   "questions use Kleene's three-valued logic. The country of a city is fixed general knowledge; its "
                   "question ('Which country is Oslo in?') is asked in every phase, so chains the base model cannot "
                   "finish can be dropped when scoring.")


def add_step(b, st):
    k = st["step"]
    if k == "natural":
        if st.get("v"):
            b.natural(st["v"])
    elif k == "link":
        b.ask("YES_NO", st["e"], st["a"], v=st["to"])
    elif k == "final" and (st["e"], st["a"]) in linked_keys(b.lessons):
        b.chain("link", start=st["e"], final=st["a"])
    elif st.get("e") is not None:
        b.ask("CURRENT", st["e"], "name" if k == "name" else st["a"])


def generate6(seed, today="2026-10-12"):
    b = Builder6(6, seed, today)
    gen_L6(b)
    return b.build()


# ===================================================================== which facts a chain rests on
def chain_trace(sch, store, ents, q):
    """The single facts a chain's answer rests on in this store, in order. It stops at the first step that gives
    nothing; the answer is then "don't know". Mirrors lwlogic.chain_sum step by step."""
    steps = []
    e = q.get("start")
    if q.get("name") is not None:
        e = L.named_ent(sch, store, ents, q["name"])
        steps.append({"step": "name", "e": e, "v": q["name"]})
        if e is None:
            return steps
    for h in q["hops"]:
        sm = L.sum_at(sch, store, e, h, NOW)
        nxt = L.ent_of(sm, ents)
        steps.append({"step": "hop", "e": e, "a": h, "to": sm.get("value")})
        if nxt is None:
            return steps
        e = nxt
    a = q["final"]
    la = L.link_of(sch, a)
    t = L.ent_of(L.sum_at(sch, store, e, la, NOW), ents) if la else None
    if L.known(L.sum_at(sch, store, e, a, NOW)) or t is None:
        steps.append({"step": "final", "e": e, "a": a})
    else:
        steps += [{"step": "link", "e": e, "a": la, "to": t}, {"step": "final", "e": t, "a": a}]
    if q.get("nat"):
        sm = L.summarize(sch, a, L.linked_state(sch, store, ents, e, a))
        if sm["kind"] == "value":
            steps.append({"step": "natural", "v": sm["value"]})
    return steps


def step_of(questions, st):
    """The question that asks one step alone, or None."""
    k = st["step"]
    for q in questions:
        if k == "natural":
            if q.get("role") == "natural" and q.get("city") == st.get("v"):
                return q
        elif k == "link":
            if q["type"] == "YES_NO" and (q.get("e"), q["a"], q.get("v")) == (st["e"], st["a"], st["to"]):
                return q
        elif k == "part":
            p = st["q"]
            if q["type"] == "YES_NO" and (q.get("e"), q["a"], q.get("v")) == (p["e"], p["a"], p["v"]):
                return q
        elif st.get("e") is not None:
            a = "name" if k == "name" else st["a"]
            if q["type"] == "CURRENT" and (q.get("e"), q["a"]) == (st["e"], a):
                return q
            if k == "final" and q["type"] == "CHAIN" and q.get("start") == st["e"] and not q["hops"] \
                    and q["final"] == a and not q.get("nat") and "v" not in q and q.get("name") is None:
                return q
    return None


def phase_lessons(S, phase):
    by_id = {l["id"]: l for l in S["lessons"]}
    return [by_id[i] for i in L.phase_store(S, phase)]


def steps_of(S, phase, q):
    """(the steps of a two-step question in this phase, the qid that asks each one alone or None)."""
    if q["type"] == "COMPOUND":
        sts = [{"step": "part", "q": p} for p in atoms(q)]
    else:
        sts = chain_trace(S["schema"], phase_lessons(S, phase), S["entities"], q)
    return sts, [(step_of(S["questions"], st) or {}).get("qid") for st in sts]


def atoms(q):
    return [x for p in q["parts"] for x in (atoms(p) if p["type"] == "COMPOUND" else [p])]


# ================================================================================================ judge
LINK_WORDS = r"\bsame\s+(?:eye\s+)?(?:colou?r|city|place|town|eyes)\b|\bsame\s+as\b|\blike\s+(?:yours|mine)\b"


def ent_words(v):
    """How an answer names a person or thing on the way: "your sister", "my brother", "your neighbour"."""
    if v == "me":
        return (r"\b(?:belongs?\s+to|owned\s+by|owner\s+is|same\s+[\w\s]{0,25}?as)\s+(?:you|yours)\b"
                r"|(?:\b(?:is|are)|'s|'re)\s+yours\b|\byour\s+own\b|\byou\s+own\b"
                r"|\byou(?:\s+are|'re)\s+(?:its|the)\s+owner\b")
    core = v.split()[-1]
    core = r"neighbou?r" if core == "neighbour" else re.escape(core)
    return rf"\b(?:your|my|the)\s+{core}\b"


def echo_patterns(S, phase, q):
    """What a chain passes through on its way, as (label, pattern): the person a name or owner points at, the link,
    and the city before its country. An answer that names only these repeats a step and does not answer."""
    sch, ents = S["schema"], S["entities"]
    store = phase_lessons(S, phase)
    out = []
    for st in chain_trace(sch, store, ents, q):
        k = st["step"]
        if k == "name" and st["e"]:
            out.append((st["e"], ent_words(st["e"])))
        elif k == "hop" and st.get("to"):
            out.append((st["to"], ent_words(st["to"])))
        elif k == "link":
            out += [(f"same as {st['to']}", LINK_WORDS), (st["to"], ent_words(st["to"]))]
        elif k == "final":
            la = L.link_of(sch, st["a"])
            if la and L.ent_of(L.sum_at(sch, store, st["e"], la, NOW), ents):
                out.append(("the link", LINK_WORDS))       # said outright, but also linked: "same as yours"
            if q.get("nat"):
                sm = L.summarize(sch, st["a"], L.linked_state(sch, store, ents, st["e"], st["a"]))
                if sm["kind"] == "value":
                    out.append((sm["value"], r"\b" + re.escape(sm["value"]) + r"\b"))
    return L.uniq(out)


ECHO_ON = True     # switched off in the self-test's negative control
COMPOUND_READ = True   # the readings of compound answers below; switched off in the self-test's negative control
STATEMENT_READ = True  # the last sentence's verdict on the whole statement; switched off in its negative control

# Compound questions are asked as "Is it true that ...?". Their answers say "true" or "false", and often reason
# through the parts before a conclusion: "I don't know your brother's favourite colour, but you no longer like
# cheese, so no." The plain yes/no reading takes the opening "I don't know" as the answer, and reads "It is false."
# as a yes (it starts like "It is ..."). These two patterns read what the answer concludes.
TRUTH_WORD = re.compile(r"^\W*(?:(?:that|this|it)(?:'s|\s+is)\s+|the\s+(?:whole\s+)?statement\s+is\s+)?"
                        r"(?:(?P<yes>true|correct)|(?P<no>false|not\s+true|untrue|incorrect|not\s+correct))\b", re.I)
CONCLUDE = re.compile(r"\b(?:so|therefore|thus|hence|which\s+means|this\s+means|that\s+means|meaning)\b[\s,:]*"
                      r"(?:the\s+(?:answer|whole\s+statement|statement)\s+is\s+|(?:it|that|this)(?:'s|\s+is)\s+)?"
                      r"(?:(?P<yes>yes|true)|(?P<no>no|false|not\s+true|untrue))\b(?!\s+(?:or|one|longer|more)\b)", re.I)


# The verdict on the whole statement in the answer's last sentence ("Since both parts are unsupported, the statement as
# a whole is not true.", 'the statement "I like cheese or my sister is called Nora" is not true.'): read like a
# conclusion when no "so ..." conclusion is there. Only the statement as a whole counts: "neither part is true" or
# "neither statement is correct" says nothing about an "or" whose right answer is "don't know" (an audit of
# 15 Oct 2026, L6 Q32, Q35, Q36).
STATEMENT_VERDICT = re.compile(r"\b(?:the|that|this)\s+(?:whole\s+)?statement(?:\s+as\s+a\s+whole)?"
                               r"(?:\s*[\"“][^\"”]*[\"”])?\s+is\s+"
                               r"(?:(?P<no>not\s+true|not\s+correct|false|untrue|incorrect)|(?P<yes>true|correct))\b", re.I)


def value_yn(sch, p, content):
    """A yes/no part read from the value the answer names, as judge_core reads a plain yes/no question without a yes
    or no word ("Your car is red." for "Is my car red?" is yes). None when nothing decides it."""
    gv = L.extract_value(content, sch, p["a"], "present")
    if gv["got"] == "VALUE" and (gv["value"] == p["v"] or not sch[p["a"]]["multi"]):
        return YES if gv["value"] == p["v"] else NO
    if any(m["neg"] and m["val"] == p["v"] for m in L.mentions(" ".join(L.sentences(content)[:2]), sch, p["a"])):
        return NO
    return None


def compound_reading(sch, q, content, g):
    """The yes/no a compound answer gives, where the plain reading `g` misses it: an opening "true" or "false"
    ("That's true.", "It is false."); a conclusion after the parts ("..., so no.") when the opening gives no yes or
    no, or gives one only through a stray word ("You no longer like cheese, ..."); and, for "Is it true that my car
    is not red?", the value it names ("Your car is red.": not true). An opening and a conclusion that disagree are
    unclear. None keeps the plain reading. Guesses and answers about the model itself keep it too."""
    ss = L.sentences(content)
    if not ss or g.get("got") in ("HEDGED", "OFFTOPIC", "EMPTY"):
        return None
    first, f = ss[0], ss[0].lower().lstrip(" \"'")
    if L.SELF_REPLY.match(f) or L.SELF_TALK.search(first):     # "No, I am a language model ...": about the model
        return None
    m = TRUTH_WORD.match(first)
    opening = (YES if m.group("yes") else NO) if m and not L.GUESS.search(first) and not L.REFUSAL.search(first) \
        else None
    concl = [YES if c.group("yes") else NO for s in ss if not L.GUESS.search(s) for c in CONCLUDE.finditer(s)]
    concl = concl[-1] if concl else None
    if concl is None and STATEMENT_READ and not L.GUESS.search(ss[-1]):
        sv = [YES if c.group("yes") else NO for c in STATEMENT_VERDICT.finditer(ss[-1])]
        concl = sv[-1] if sv else None
    base = g.get("value") if g.get("got") == "YN" and g.get("value") in (YES, NO) else None
    strong = base is not None and bool(L.NO_START.match(f) or L.YES_START.match(f))
    said = opening if opening is not None else base if strong else None
    if said is not None:
        if concl is not None and concl != said:
            return {"got": "UNCLEAR", "read": "says both yes and no"}
        return {"got": "YN", "value": said, "read": "true or false"} if opening is not None and opening != base \
            else None
    if concl is not None:
        return {"got": "YN", "value": concl, "read": "the conclusion"}
    if base is None and g.get("got") == "UNCLEAR" and q["op"] == "not" and q["parts"][0]["type"] != "COMPOUND":
        part = value_yn(sch, q["parts"][0], content)
        if part is not None:
            return {"got": "YN", "value": L.neg(part), "read": "the value named, turned around"}
    return None


def judge_two_step(S, phase, q, content):
    """The verdict on a two-step answer. A chain is read like a plain question about its answer attribute (the
    country, the owner's eye colour); a compound question like a yes/no question, with the readings of
    `compound_reading`. A chain answer that is not a pass, names no value of the answer attribute and gives no yes
    or no, but names a step on the way ("You live in Oslo." for the country, "The boat belongs to your sister." for
    her eye colour), is an echo. In a delete phase, an answer that names only a step the chain took before the
    delete ("The boat belongs to your brother." after the owner lessons were deleted) keeps its verdict, and its
    reason says so instead of "off topic". The date check of plain questions is not run: no level 6 question has
    a date."""
    if q["type"] == "COMPOUND":
        verdict, g, why = L.judge_core(S, phase, q, content)
        g2 = compound_reading(S["schema"], q, content, g) if COMPOUND_READ else None
        if g2 is None:
            return verdict, g, why
        if g2["got"] != "YN":
            return "unclear", g2, g2["read"]
        exp = S["expected"][phase][q["qid"]]["value"]
        return ("pass", g2, "") if g2["value"] == exp else ("fail", g2, f"wrong yes/no ({g2['read']})")
    qq = dict(q, type="YES_NO" if "v" in q else "CURRENT")
    verdict, g, why = L.judge_core(S, phase, qq, content)
    if verdict == "pass" or not ECHO_ON or not (content or "").strip():
        return verdict, g, why
    text = " ".join(L.sentences(content))
    hit = [lab for lab, pat in echo_patterns(S, phase, q) if re.search(pat, text, re.I)]
    finals = [m for m in L.mentions(text, S["schema"], q["a"]) if not m["neg"]]
    said = "v" in q and g.get("got") in ("YN", "HEDGED")
    if hit and not finals and not said:
        return "echo", dict(g, echo=hit), f"repeats a step on the way ({', '.join(hit)}), not the answer"
    if not hit and not finals and not said and phase not in ("baseline", "teach"):
        old = [lab for lab, pat in echo_patterns(S, "teach", q) if re.search(pat, text, re.I)]
        if old:
            return verdict, dict(g, stale=old), f"names a step the chain took before this delete ({', '.join(old)})"
    return verdict, g, why


# =============================================================================================== scores
def two_step_scores(S, results, condition=True, drop=True):
    """Stated against implied facts. One unit is one two-step question in one phase and thinking mode. A unit
    counts toward the gap when something is taught (not the baseline), the rule book knows its answer, every step
    has a question of its own, and no natural step on its way was missed by the base model in the baseline (same
    thinking mode): those are dropped. Each unit counts its own steps. `condition` and `drop` are switched off
    only by the self-test's negative controls."""
    qs = {q["qid"]: q for q in S["questions"]}
    verdict = {(x["phase"], x["qid"], x["thinking"]): x["verdict"] for x in results}
    units = []
    for x in results:
        q = qs.get(x["qid"])
        if not q or q["type"] not in L.TWO_STEP:
            continue
        ph, th = x["phase"], x["thinking"]
        sts, sq = steps_of(S, ph, q)
        exp = S["expected"][ph][q["qid"]]
        if q["type"] == "CHAIN":
            known = L.known(L.chain_sum(S["schema"], phase_lessons(S, ph), S["entities"], q))
        else:
            known = exp["value"] != UNK
        sv = [verdict.get((ph, s, th)) if s else None for s in sq]
        nat = [(st["v"], s) for st, s in zip(sts, sq) if st["step"] == "natural"]
        dropped = sorted({c for c, s in nat if verdict.get(("baseline", s, th)) not in (None, "pass")}) if drop else []
        units.append({"phase": ph, "thinking": th, "qid": q["qid"], "family": q.get("family", "compound"),
                      "steps": len(sts), "known": known, "verdict": x["verdict"], "step_qids": sq,
                      "step_verdicts": sv, "taught": [st["step"] not in ("natural",) for st in sts],
                      "dropped": dropped, "no_baseline": [c for c, s in nat if ("baseline", s, th) not in verdict]})

    def rate(p, n):
        return {"pass": p, "n": n, "pct": round(100 * p / n, 1) if n else None}

    def block(us):
        n = len(us)
        imp = sum(u["verdict"] == "pass" for u in us)
        st = [v for u in us for v, t in zip(u["step_verdicts"], u["taught"]) if t]
        ok = [u for u in us if not condition or all(v == "pass" for v in u["step_verdicts"])]
        r = {"units": n, "stated": rate(sum(v == "pass" for v in st), len(st)), "implied": rate(imp, n),
             "given_steps_right": rate(sum(u["verdict"] == "pass" for u in ok), len(ok)),
             "echo": sum(u["verdict"] == "echo" for u in us)}
        r["gap_points"] = round(r["stated"]["pct"] - r["implied"]["pct"], 1) \
            if r["stated"]["pct"] is not None and n else None
        return r

    out = {"note": "stated: the taught single steps the chains rest on; implied: the chains; gap: stated minus "
                   "implied, in percentage points; given_steps_right: chains whose every step (natural ones "
                   "included) was answered right in the same phase and thinking mode",
           "modes": {}}
    for th in sorted({u["thinking"] for u in units}):
        us = [u for u in units if u["thinking"] == th]
        taught = [u for u in us if u["phase"] != "baseline"]
        scored = [u for u in taught if u["known"] and not u["dropped"] and all(u["step_qids"])
                  and u["family"] != "compound"]
        m = {"all": block([u for u in scored if u["steps"] >= 2]),
             "by_steps": {str(k): block([u for u in scored if u["steps"] == k])
                          for k in sorted({u["steps"] for u in scored})},
             "by_family": {f: block([u for u in scored if u["family"] == f and u["steps"] >= 2])
                           for f in FAMILIES[:4] if any(u["family"] == f for u in scored)},
             "dont_know_expected": rate(sum(u["verdict"] == "pass" for u in taught if not u["known"]
                                            and u["family"] != "compound"),
                                        sum(1 for u in taught if not u["known"] and u["family"] != "compound")),
             "baseline": rate(sum(u["verdict"] == "pass" for u in us if u["phase"] == "baseline"),
                              sum(1 for u in us if u["phase"] == "baseline")),
             "echo": sum(u["verdict"] == "echo" for u in us),
             "dropped": sorted({c for u in us for c in u["dropped"]}),
             "dropped_units": sum(1 for u in taught if u["dropped"]),
             "no_baseline": sorted({c for u in us for c in u["no_baseline"]})}
        comp = [u for u in taught if u["family"] == "compound"]
        kl = {}
        for u in comp:
            exp = [S["expected"][u["phase"]][s]["value"] for s in u["step_qids"] if s]
            q = qs[u["qid"]]
            row = (f"not {exp[0]}" if q["op"] == "not" else f"{exp[0]} {q['op']} {exp[1]}") if len(exp) == len(
                u["step_qids"]) else "?"
            row = row.replace(UNK, "unknown")
            c = kl.setdefault(row, [0, 0])
            c[0] += u["verdict"] == "pass"
            c[1] += 1
        ok = [u for u in comp if all(v == "pass" for v in u["step_verdicts"])]
        m["compound"] = {"all": rate(sum(u["verdict"] == "pass" for u in comp), len(comp)),
                         "given_parts_right": rate(sum(u["verdict"] == "pass" for u in ok), len(ok)),
                         "by_case": {k: rate(*v) for k, v in sorted(kl.items())}}
        out["modes"]["on" if th else "off"] = m
    out["units"] = units
    return out


def two_step_table(sc):
    pct = lambda r: f"{r['pass']}/{r['n']}" + (f" ({r['pct']:.0f}%)" if r["n"] else "")
    lines = ["\n== two-step questions (level 6): stated against implied facts",
             "   phases with lessons taught; a chain counts when the rule book knows its answer and every step has a "
             "question of its own", "   stated: the taught steps it rests on; implied: the chain itself; given steps "
             "right: chains whose every step was right in the same phase and mode"]
    for th, m in sc["modes"].items():
        a = m["all"]
        lines.append(f"   thinking {th}: stated {pct(a['stated'])}, implied {pct(a['implied'])}"
                     + (f", gap {a['gap_points']:+.0f} points" if a["gap_points"] is not None else "")
                     + f"; given steps right {pct(a['given_steps_right'])}; echo {a['echo']}")
        lines.append(f"     {'steps':6} {'chains':>6} {'stated':>14} {'implied':>14} {'given steps right':>18} {'gap':>6}")
        for k, r in m["by_steps"].items():
            lines.append(f"     {k:6} {r['units']:>6} {pct(r['stated']):>14} {pct(r['implied']):>14} "
                         f"{pct(r['given_steps_right']):>18} "
                         + (f"{r['gap_points']:>+6.0f}" if r["gap_points"] is not None else f"{'-':>6}"))
        if m["by_family"]:
            lines.append("     by family (2 steps or more): " + ", ".join(
                f"{f} {pct(r['implied'])}" for f, r in m["by_family"].items()))
        c = m["compound"]
        if c["all"]["n"]:
            lines.append(f"     compound (Kleene): {pct(c['all'])}; given both parts right {pct(c['given_parts_right'])}; "
                         + ", ".join(f"{k} {pct(r)}" for k, r in c["by_case"].items()))
        lines.append(f"     don't know expected {pct(m['dont_know_expected'])}; baseline {pct(m['baseline'])}; "
                     f"echo answers {m['echo']}"
                     + (f"; dropped {m['dropped_units']} chain answers: the base model missed the country of "
                        f"{', '.join(m['dropped'])} in the baseline" if m["dropped"] else "")
                     + (f"; no baseline answer for {', '.join(m['no_baseline'])}, nothing dropped for it"
                        if m["no_baseline"] else ""))
    return "\n".join(lines)


# ============================================================================================ self-test
def test_schema6():
    nat = {"Oslo": "Norway", "Bergen": "Norway", "Porto": "Portugal"}
    return {"eye_colour": dict(L.SCHEMA["eye_colour"], domain=["brown", "green", "blue"]),
            "home_city": dict(L.SCHEMA["home_city"], domain=["Oslo", "Bergen", "Porto"]),
            "name": dict(L.SCHEMA["name"], domain=["Ana", "Milo", "Nora"]),
            "owner": dict(L.SCHEMA["owner"], domain=["me", "my sister", "my neighbour"]),
            "eyes_like": dict(SCHEMA6["eyes_like"], domain=["me", "my sister", "my cat"]),
            "city_like": dict(SCHEMA6["city_like"], domain=["me", "my sister"]),
            "country": dict(SCHEMA6["country"], domain=["Norway", "Portugal"], natural={"from": "home_city", "map": nat})}


TENTS = ["me", "my sister", "my cat", "the boat"]
# weighted toward the attributes chains read, so links, owners and eye colours meet often
TAUGHT = ["eye_colour"] * 3 + ["eyes_like"] * 2 + ["owner"] * 2 + ["home_city"] * 2 + ["city_like", "name"]


def rand_store6(rng, sch, n=None):
    n = rng.randint(2, 14) if n is None else n
    return [L.rand_lesson(rng, i + 1, sch, TENTS, key=(rng.choice(TENTS), rng.choice(TAUGHT))) for i in range(n)]


def rand_chain(rng, sch):
    q = {"type": "CHAIN", "hops": rng.choice([[], ["owner"], ["owner"], ["owner", "owner"]]),
         "final": rng.choice(["eye_colour", "home_city", "home_city", "name"])}
    if rng.random() < 0.3:
        q["name"] = rng.choice(sch["name"]["domain"])
    else:
        q["start"] = rng.choice(TENTS)
    q["a"] = q["final"]
    if q["final"] == "home_city" and rng.random() < 0.5:
        q["nat"], q["a"] = "country", "country"
    if rng.random() < 0.3:
        q["v"] = rng.choice(sch[q["a"]]["domain"])
    return q


def by_steps(sch, store, ents, q):
    """A second implementation of chains, for the differential check: it asks the single-step rule book one plain
    question at a time (who is called v, CURRENT of each hop, YES_NO of every value), as a person would, and fills
    the gaps through the link value by value. It shares none of the chain code."""
    ask = lambda x: L.answer(sch, store, ents, x)
    unknown = {"kind": "unknown", "not": []}
    if q.get("name") is not None:
        who = ask({"type": "WHO_HAS", "a": "name", "v": q["name"]})["yes"]
        e = who[0] if len(who) == 1 else None
    else:
        e = q["start"]
    for h in q["hops"]:
        if e is None:
            break
        sm = ask({"type": "CURRENT", "e": e, "a": h})
        e = sm["value"] if sm["kind"] == "value" and sm["value"] in ents else None
    if e is None:
        sm = unknown
    else:
        a = q["final"]
        truth = {x: ask({"type": "YES_NO", "e": e, "a": a, "v": x})["value"] for x in sch[a]["domain"]}
        links = [x for x in sch if sch[x].get("link") == a]
        if links:
            tl = ask({"type": "CURRENT", "e": e, "a": links[0]})
            t = tl["value"] if tl["kind"] == "value" and tl["value"] in ents else None
            if t is not None:
                for x in truth:
                    if truth[x] == UNK:
                        truth[x] = ask({"type": "YES_NO", "e": t, "a": a, "v": x})["value"]
        sm = L.summarize(sch, a, truth)
        if q.get("nat"):
            mp = sch[q["nat"]]["natural"]["map"]
            sm = ({"kind": "value", "value": mp[sm["value"]]} if sm["kind"] == "value" and sm["value"] in mp else
                  {"kind": "none"} if sm["kind"] == "none" else unknown)
    if "v" not in q:
        return sm
    v = q["v"]
    if sm["kind"] == "value":
        return {"kind": "yn", "value": YES if sm["value"] == v else NO}
    if sm["kind"] == "none":
        return {"kind": "yn", "value": NO}
    if sm["kind"] == "set":
        return {"kind": "yn", "value": YES if v in sm["yes"] else NO if v in sm["no"] else UNK}
    return {"kind": "yn", "value": NO if v in sm["not"] else UNK}


# ------------------------------------------------------- chain systems: the rule book and broken ones
# A system answers a chain question from a log of {"teach": lesson} and {"delete": id}.
def sys_oracle(sch, log, q):
    return L.answer(sch, L.effective(log), TENTS, q)


def sys_first_link(sch, log, q):
    """Answers a chain from its first link only: what the first fact on the way says is taken as the answer (the
    owner for the owner's eyes, the cat's own state without its link, the city's state for the country)."""
    store = L.effective(log)
    e = L.start_ent(sch, store, TENTS, q)
    sm = {"kind": "unknown", "not": []} if e is None else L.sum_at(sch, store, e, (q["hops"] or [q["final"]])[0], NOW)
    return {"kind": "yn", "value": L.derive_yn(sm, q["v"])} if "v" in q else sm


def sys_stale(sch, log, q):
    """Reads each hop from the first lesson ever taught on it: a later CHANGE of the owner is not seen."""
    store = L.effective(log)
    first, keep = set(), []
    for l in store:
        if l["a"] in q["hops"]:
            if (l["e"], l["a"]) in first:
                continue
            first.add((l["e"], l["a"]))
        keep.append(l)
    return L.answer(sch, keep, TENTS, q)


def sys_nolink(sch, log, q):
    """Ignores linking facts."""
    return L.answer({a: {k: v for k, v in s.items() if k != "link"} for a, s in sch.items()}, L.effective(log), TENTS, q)


def sys_cache(sch, log, q):
    """Keeps chain answers as they were taught: deletes are not seen by chains."""
    return L.answer(sch, [x["teach"] for x in log if "teach" in x], TENTS, q)


class patched_fill:
    """Run the rule book with another `fill` (the broken systems below); None keeps the rule book's own."""

    def __init__(self, f):
        self.f = f

    def __enter__(self):
        self.keep = L.fill
        L.fill = self.f or L.fill

    def __exit__(self, *exc):
        L.fill = self.keep


def fill_override(own, src):
    return {x: (own[x] if src[x] == UNK else src[x]) for x in own}


def fill_union(own, src):
    return {x: (YES if YES in (own[x], src[x]) else own[x] if own[x] != UNK else src[x]) for x in own}


def sys_override(sch, log, q):
    """A link overrides what is said outright: where the linked entity's value is known, it wins."""
    with patched_fill(fill_override):
        return L.answer(sch, L.effective(log), TENTS, q)


def sys_union(sch, log, q):
    """A link adds the linked value to what is said ("blue", linked to green: both)."""
    with patched_fill(fill_union):
        return L.answer(sch, L.effective(log), TENTS, q)


sys_override.fill, sys_union.fill = fill_override, fill_union


def k_closed(op, ps):
    """Closed world: an unknown part counts as no."""
    ps = [NO if p == UNK else p for p in ps]
    return L.neg(ps[0]) if op == "not" else L.k_and(*ps) if op == "and" else L.k_or(*ps)


def k_weak(op, ps):
    """Weak (Bochvar) three-valued logic: any unknown part makes the whole answer unknown."""
    if UNK in ps:
        return UNK
    return L.neg(ps[0]) if op == "not" else L.k_and(*ps) if op == "and" else L.k_or(*ps)


def k_rule(op, ps):
    return L.neg(ps[0]) if op == "not" else L.k_and(*ps) if op == "and" else L.k_or(*ps)


def kleene_bad(kleene):
    """The truth-table rows a three-valued logic gets wrong (Kleene's strong logic, written out by hand)."""
    T, F, U = YES, NO, UNK
    table = {("and", T, T): T, ("and", T, F): F, ("and", T, U): U, ("and", F, T): F, ("and", F, F): F,
             ("and", F, U): F, ("and", U, T): U, ("and", U, F): F, ("and", U, U): U,
             ("or", T, T): T, ("or", T, F): T, ("or", T, U): T, ("or", F, T): T, ("or", F, F): F, ("or", F, U): U,
             ("or", U, T): T, ("or", U, F): U, ("or", U, U): U, ("not", T): F, ("not", F): T, ("not", U): U}
    bad = [k for k, v in table.items() if kleene(k[0], list(k[1:])) != v]
    neg = lambda t: {T: F, F: T, U: U}[t]
    for p in (T, F, U):                      # De Morgan
        for q in (T, F, U):
            if neg(kleene("and", [p, q])) != kleene("or", [neg(p), neg(q)]):
                bad.append(("de Morgan", p, q))
    return bad


# --------------------------------------------------------------------------- the chain properties
def prop_differential(system, rng, n, sch):
    r = {"n": 0, "bad": 0, "ex": ""}
    for _ in range(n):
        store = rand_store6(rng, sch)
        for _ in range(8):
            q = rand_chain(rng, sch)
            r["n"] += 1
            got, want = system(sch, L.as_log(store), q), by_steps(sch, store, TENTS, q)
            if got != want:
                r["bad"] += 1
                r["ex"] = r["ex"] or f"{q} got {got} want {want}"
    return r


def prop_known(system, rng, n, sch):
    """A two-step chain (one hop, no natural step) is known exactly when both links are: the hop names an entity,
    and that entity's own or linked answer is known (read with single-step questions)."""
    r = {"n": 0, "bad": 0, "ex": ""}
    for _ in range(n):
        store = rand_store6(rng, sch)
        q = {"type": "CHAIN", "start": rng.choice(TENTS), "hops": ["owner"], "final": rng.choice(["eye_colour", "home_city"])}
        q["a"] = q["final"]
        first = L.answer(sch, store, TENTS, {"type": "CURRENT", "e": q["start"], "a": "owner"})
        t = L.ent_of(first, TENTS)
        second = t is not None and L.known(by_steps(sch, store, TENTS, dict(q, start=t, hops=[])))
        got = system(sch, L.as_log(store), q)
        r["n"] += 1
        if L.known(got) != (t is not None and second):
            r["bad"] += 1
            r["ex"] = r["ex"] or f"first link {first}, second known {second}, chain {got}"
    return r


def link_store(rng, sch):
    """A store where the boat's owner and my cat's eye link are both known, plus random noise."""
    e_owner = rng.choice(["me", "my sister"])
    base = [{"op": "ASSERT", "e": "the boat", "a": "owner", "v": e_owner},
            {"op": "ASSERT", "e": "my cat", "a": "eyes_like", "v": rng.choice(["me", "my sister"])},
            {"op": "ASSERT", "e": rng.choice(["me", "my sister"]), "a": "eye_colour",
             "v": rng.choice(sch["eye_colour"]["domain"])}]
    noise = [dict(l) for l in rand_store6(rng, sch, rng.randint(0, 6))]
    out = base + noise
    rng.shuffle(out)
    return L.renumber([dict(l, id=i + 1, name=f"L{i + 1}") for i, l in enumerate(out)])


def prop_delete_link(system, rng, n, sch):
    """Deleting every lesson on a chain's first link makes the chain "don't know"."""
    r = {"n": 0, "bad": 0, "ex": ""}
    for _ in range(n):
        store = link_store(rng, sch)
        q = {"type": "CHAIN", "start": "the boat", "hops": ["owner"], "final": "eye_colour", "a": "eye_colour"}
        log = L.as_log(store) + [{"delete": l["id"]} for l in store if (l["e"], l["a"]) == ("the boat", "owner")]
        got = system(sch, log, q)
        r["n"] += 1
        if got != {"kind": "unknown", "not": []}:
            r["bad"] += 1
            r["ex"] = r["ex"] or f"after the owner lessons were deleted: {got}"
    return r


def prop_deep(system, rng, n, sch):
    """Deep delete: deleting what was said outright about my cat's eyes keeps the answer the lessons kept still
    imply through the link: exactly the linked person's eye colour."""
    r = {"n": 0, "bad": 0, "ex": ""}
    for _ in range(n):
        store = link_store(rng, sch)
        store = L.renumber(store + [{"id": len(store) + 1, "name": "own", "op": "ASSERT", "e": "my cat",
                                     "a": "eye_colour", "v": rng.choice(sch["eye_colour"]["domain"])}])
        q = {"type": "CHAIN", "start": "my cat", "hops": [], "final": "eye_colour", "a": "eye_colour"}
        log = L.as_log(store) + [{"delete": l["id"]} for l in store if (l["e"], l["a"]) == ("my cat", "eye_colour")]
        live = L.effective(log)
        t = L.ent_of(L.sum_at(sch, live, "my cat", "eyes_like", NOW), TENTS)
        if t is None:
            continue
        want = L.sum_at(sch, live, t, "eye_colour", NOW)
        got = system(sch, log, q)
        r["n"] += 1
        if got != want:
            r["bad"] += 1
            r["ex"] = r["ex"] or f"linked to {t} ({want}), got {got}"
    return r


def prop_change(system, rng, n, sch):
    """A chain follows a CHANGE of its link: after "the boat belongs to my sister now" it reads my sister."""
    r = {"n": 0, "bad": 0, "ex": ""}
    for _ in range(n):
        store = link_store(rng, sch)
        old = L.ent_of(L.sum_at(sch, store, "the boat", "owner", NOW), TENTS)
        new = "my sister" if old == "me" else "me"
        store = L.renumber(store + [{"id": len(store) + 1, "name": "chg", "op": "CHANGE", "e": "the boat",
                                     "a": "owner", "v": new}])
        q = {"type": "CHAIN", "start": "the boat", "hops": ["owner"], "final": "eye_colour", "a": "eye_colour"}
        want = by_steps(sch, store, TENTS, dict(q, start=new, hops=[]))
        got = system(sch, L.as_log(store), q)
        r["n"] += 1
        if got != want:
            r["bad"] += 1
            r["ex"] = r["ex"] or f"owner changed to {new} ({want}), got {got}"
    return r


def prop_own(system, rng, n, sch):
    """What is said outright wins over a link: my cat's own known eye colour stands, whatever the link says."""
    r = {"n": 0, "bad": 0, "ex": ""}
    dom = sch["eye_colour"]["domain"]
    for _ in range(n):
        own, other = rng.sample(dom, 2)
        t = rng.choice(["me", "my sister"])
        ls = [{"op": "ASSERT", "e": t, "a": "eye_colour", "v": other},
              {"op": "ASSERT", "e": "my cat", "a": "eye_colour", "v": own},
              {"op": "ASSERT", "e": "my cat", "a": "eyes_like", "v": t}]
        rng.shuffle(ls)
        store = L.renumber([dict(l, id=i + 1, name=f"L{i + 1}") for i, l in enumerate(ls)])
        q = {"type": "CHAIN", "start": "my cat", "hops": [], "final": "eye_colour", "a": "eye_colour"}
        got = system(sch, L.as_log(store), q)
        r["n"] += 1
        if got != {"kind": "value", "value": own}:
            r["bad"] += 1
            r["ex"] = r["ex"] or f"own {own}, linked {other}: got {got}"
    return r


def prop_functional(system, rng, n, sch):
    """A linked single-valued state never holds two values, also when the link contradicts what is said outright
    (state level: an answer's summary would hide a second yes)."""
    r = {"n": 0, "bad": 0, "ex": ""}
    for _ in range(n):
        store = link_store(rng, sch)
        extra = [{"op": "ASSERT", "e": "my cat", "a": "eye_colour", "v": rng.choice(sch["eye_colour"]["domain"])}]
        store = L.renumber(store + [dict(l, id=len(store) + 1 + i, name=f"X{i}") for i, l in
                                    enumerate(extra + rand_store6(rng, sch, 2))])
        with patched_fill(getattr(system, "fill", None)):
            s = L.linked_state(sch, store, TENTS, "my cat", "eye_colour")
        ys = [v for v in s if s[v] == YES]
        r["n"] += 1
        if len(ys) > 1 or (ys and any(s[v] != NO for v in s if v != ys[0])):
            r["bad"] += 1
            r["ex"] = r["ex"] or f"state {s}"
    return r


CHAIN_PROPS = [
    ("differential", "chain answers equal a second implementation that asks the single-step rule book one plain "
                     "question at a time", prop_differential),
    ("known", "a two-step chain is known exactly when both links are known", prop_known),
    ("delete_link", "deleting a chain's link makes it don't know", prop_delete_link),
    ("deep", "deep delete: deleting what was said outright keeps what the kept lessons imply through a link",
     prop_deep),
    ("change", "a chain follows a CHANGE of its link", prop_change),
    ("own", "what is said outright wins over a link", prop_own),
    ("functional", "a linked single-valued answer never holds two values", prop_functional),
]
CHAIN_MUTANTS = [
    ("an oracle that answers chains from the first link only", sys_first_link, ["differential", "known"]),
    ("hops read from the first lesson taught on them (a CHANGE of the owner unseen)", sys_stale,
     ["differential", "change"]),
    ("linking facts ignored", sys_nolink, ["differential", "deep"]),
    ("chains that do not see deletes", sys_cache, ["delete_link"]),
    ("a link that overrides what is said outright", sys_override, ["own"]),   # random stores rarely hold the conflict
    ("a link that adds its value to what is said", sys_union, ["functional"]),
]


# -------------------------------------------------------------------------------------- golden cases
def golden(sch):
    """Hand-written answers that pin the meaning of each chain rule, as (name, lessons, [(question, answer)])."""
    mk = lambda *ops: L.renumber([dict(o, id=i + 1, name=f"G{i + 1}") for i, o in enumerate(ops)])
    A = lambda op, e, a, v: {"op": op, "e": e, "a": a, "v": v}
    V = lambda v: {"kind": "value", "value": v}
    U = lambda *x: {"kind": "unknown", "not": list(x)}
    YN = lambda t: {"kind": "yn", "value": t}
    C = lambda **kw: dict({"type": "CHAIN", "hops": []}, **kw, a=kw.get("a", "country" if kw.get("nat") else kw["final"]))
    boat_eyes = C(start="the boat", hops=["owner"], final="eye_colour")
    yn = lambda e, a, v: {"type": "YES_NO", "e": e, "a": a, "v": v}
    comp = lambda op, *ps: {"type": "COMPOUND", "op": op, "parts": list(ps), "a": None}
    return [
        ("the owner's eyes: the boat's owner, then that person's eyes",
         mk(A("ASSERT", "the boat", "owner", "my sister"), A("ASSERT", "my sister", "eye_colour", "green")),
         [(boat_eyes, V("green")), (dict(boat_eyes, v="green"), YN(YES)), (dict(boat_eyes, v="blue"), YN(NO))]),
        ("after 'the boat belongs to me now' the chain reads my eyes; nothing taught about them: don't know",
         mk(A("ASSERT", "the boat", "owner", "my sister"), A("ASSERT", "my sister", "eye_colour", "green"),
            A("CHANGE", "the boat", "owner", "me")),
         [(boat_eyes, U()), (dict(boat_eyes, v="green"), YN(UNK))]),
        ("the owner is someone the set knows nothing about (my neighbour): don't know",
         mk(A("ASSERT", "the boat", "owner", "my neighbour"), A("ASSERT", "me", "eye_colour", "blue")),
         [(boat_eyes, U())]),
        ("'the boat does not belong to me anymore': no owner known, don't know",
         mk(A("ASSERT", "the boat", "owner", "me"), A("ASSERT", "me", "eye_colour", "blue"),
            A("RETRACT", "the boat", "owner", "me")),
         [(boat_eyes, U())]),
        ("a name points at the one entity called that now",
         mk(A("ASSERT", "my sister", "name", "Nora"), A("ASSERT", "my sister", "eye_colour", "brown")),
         [(C(name="Nora", final="eye_colour"), V("brown")), (C(name="Ana", final="eye_colour"), U())]),
        ("two entities called Nora: nobody can be picked, don't know",
         mk(A("ASSERT", "my sister", "name", "Nora"), A("ASSERT", "my cat", "name", "Nora"),
            A("ASSERT", "my sister", "eye_colour", "brown")),
         [(C(name="Nora", final="eye_colour"), U())]),
        ("after a new name the old one points at nobody, the new one at the same cat",
         mk(A("ASSERT", "my cat", "name", "Nora"), A("ASSERT", "my cat", "eye_colour", "green"),
            A("CHANGE", "my cat", "name", "Milo")),
         [(C(name="Nora", final="eye_colour"), U()), (C(name="Milo", final="eye_colour"), V("green"))]),
        ("the country of my city; a move within the country keeps it, a move to another changes it",
         mk(A("ASSERT", "me", "home_city", "Oslo"), A("CHANGE", "me", "home_city", "Bergen")),
         [(C(start="me", final="home_city", nat="country"), V("Norway")),
          (C(start="me", final="home_city", nat="country", v="Norway"), YN(YES)),
          (C(start="me", final="home_city", nat="country", v="Portugal"), YN(NO))]),
        ("moved to Porto: Portugal",
         mk(A("ASSERT", "me", "home_city", "Oslo"), A("CHANGE", "me", "home_city", "Porto")),
         [(C(start="me", final="home_city", nat="country"), V("Portugal"))]),
        ("'I do not live in Oslo anymore' says nothing about the country: don't know, not 'not Norway'",
         mk(A("ASSERT", "me", "home_city", "Oslo"), A("RETRACT", "me", "home_city", "Oslo")),
         [(C(start="me", final="home_city", nat="country"), U()),
          (C(start="me", final="home_city", nat="country", v="Norway"), YN(UNK))]),
        ("a link fills a gap: my cat's eyes are the same colour as mine",
         mk(A("ASSERT", "my cat", "eyes_like", "me"), A("ASSERT", "me", "eye_colour", "green")),
         [(C(start="my cat", final="eye_colour"), V("green"))]),
        ("what is said outright wins over the link, whichever came first",
         mk(A("ASSERT", "my cat", "eye_colour", "blue"), A("ASSERT", "my cat", "eyes_like", "me"),
            A("ASSERT", "me", "eye_colour", "green")),
         [(C(start="my cat", final="eye_colour"), V("blue"))]),
        ("a link to someone nothing is taught about: don't know",
         mk(A("ASSERT", "my cat", "eyes_like", "my sister"), A("ASSERT", "me", "eye_colour", "green")),
         [(C(start="my cat", final="eye_colour"), U())]),
        ("a link is a standing rule: when the linked value changes, the linked answer follows",
         mk(A("ASSERT", "my cat", "eyes_like", "me"), A("ASSERT", "me", "eye_colour", "green"),
            A("CHANGE", "me", "eye_colour", "blue")),
         [(C(start="my cat", final="eye_colour"), V("blue"))]),
        ("my cat 'not blue', linked to my blue eyes: the two contradict, so no colour fits (none)",
         mk(A("RETRACT", "my cat", "eye_colour", "blue"), A("ASSERT", "my cat", "eyes_like", "me"),
            A("ASSERT", "me", "eye_colour", "blue")),
         [(C(start="my cat", final="eye_colour"), {"kind": "none"})]),
        ("a link is read one way: my cat's link says nothing about my eyes",
         mk(A("ASSERT", "my cat", "eyes_like", "me"), A("ASSERT", "my cat", "eye_colour", "green")),
         [(C(start="me", final="eye_colour"), U())]),
        ("a link is read one link deep: my cat like my sister, my sister like me, my eyes green",
         mk(A("ASSERT", "my cat", "eyes_like", "my sister"), A("ASSERT", "my sister", "eyes_like", "me"),
            A("ASSERT", "me", "eye_colour", "green")),
         [(C(start="my sister", final="eye_colour"), V("green")), (C(start="my cat", final="eye_colour"), U())]),
        ("the owner's country through a city link: three facts and the natural step",
         mk(A("ASSERT", "the boat", "owner", "my sister"), A("ASSERT", "my sister", "city_like", "me"),
            A("ASSERT", "me", "home_city", "Porto")),
         [(C(start="the boat", hops=["owner"], final="home_city", nat="country"), V("Portugal"))]),
        ("Kleene: and is no as soon as a part is no; or is yes as soon as a part is yes; not unknown is unknown",
         mk(A("ASSERT", "me", "eye_colour", "green")),
         [(comp("and", yn("me", "eye_colour", "blue"), yn("my sister", "eye_colour", "blue")), YN(NO)),
          (comp("and", yn("me", "eye_colour", "green"), yn("my sister", "eye_colour", "blue")), YN(UNK)),
          (comp("or", yn("my sister", "eye_colour", "blue"), yn("me", "eye_colour", "green")), YN(YES)),
          (comp("or", yn("my sister", "eye_colour", "blue"), yn("me", "eye_colour", "blue")), YN(UNK)),
          (comp("not", yn("my sister", "eye_colour", "blue")), YN(UNK)),
          (comp("not", comp("and", yn("me", "eye_colour", "green"), yn("me", "eye_colour", "blue"))), YN(YES))]),
    ]


GOLDEN_DELETES = [
    # (name, lessons, deleted ids, question, answer)
    ("deep delete: my cat's own eye colour deleted, the link to mine keeps it known",
     [("ASSERT", "me", "eye_colour", "green"), ("ASSERT", "my cat", "eye_colour", "green"),
      ("ASSERT", "my cat", "eyes_like", "me")], [2], ("my cat", [], "eye_colour"), {"kind": "value", "value": "green"}),
    ("the link and the cat's own fact deleted: don't know",
     [("ASSERT", "me", "eye_colour", "green"), ("ASSERT", "my cat", "eye_colour", "green"),
      ("ASSERT", "my cat", "eyes_like", "me")], [2, 3], ("my cat", [], "eye_colour"), {"kind": "unknown", "not": []}),
    ("one link of a chain deleted: the owner lesson, so don't know",
     [("ASSERT", "the boat", "owner", "me"), ("ASSERT", "me", "eye_colour", "green")], [1],
     ("the boat", ["owner"], "eye_colour"), {"kind": "unknown", "not": []}),
    ("the other link deleted: my eyes, so don't know",
     [("ASSERT", "the boat", "owner", "me"), ("ASSERT", "me", "eye_colour", "green")], [2],
     ("the boat", ["owner"], "eye_colour"), {"kind": "unknown", "not": []}),
    ("the change of owner deleted: the chain goes back to the first owner",
     [("ASSERT", "the boat", "owner", "me"), ("ASSERT", "me", "eye_colour", "green"),
      ("ASSERT", "my sister", "eye_colour", "blue"), ("CHANGE", "the boat", "owner", "my sister")], [4],
     ("the boat", ["owner"], "eye_colour"), {"kind": "value", "value": "green"}),
]


def golden_wrong(sch, system):
    """The golden answers a system gets wrong: (list of what went wrong, number of answers)."""
    gbad, ng = [], 0
    for name, store, checks in golden(sch):
        for q, want in checks:
            ng += 1
            got = system(sch, L.as_log(store), q)
            if got != want:
                gbad.append(f"{name}: {q} expected {want} got {got}")
    for name, ops, gone, (e, hops, a), want in GOLDEN_DELETES:
        ng += 1
        store = L.renumber([{"id": i + 1, "name": f"G{i + 1}", "op": o, "e": x, "a": y, "v": v}
                            for i, (o, x, y, v) in enumerate(ops)])
        log = L.as_log(store) + [{"delete": i} for i in gone]
        got = system(sch, log, {"type": "CHAIN", "start": e, "hops": hops, "final": a, "a": a})
        if got != want:
            gbad.append(f"{name}: expected {want} got {got}")
    return gbad, ng


class patched_rule:
    """Run the rule book with one of its functions replaced (a broken system); the original comes back after."""

    def __init__(self, name, f):
        self.name, self.f = name, f

    def __enter__(self):
        self.keep = getattr(L, self.name)
        setattr(L, self.name, self.f)

    def __exit__(self, *exc):
        setattr(L, self.name, self.keep)


def sys_closed_kleene(sch, log, q):
    """Compound questions in a closed world: an unknown part counts as no."""
    with patched_rule("k_and", lambda p, r: YES if p == r == YES else NO), \
            patched_rule("k_or", lambda p, r: YES if YES in (p, r) else NO):
        return sys_oracle(sch, log, q)


GOLDEN_MUTANTS = [
    ("chains from the first link only", lambda sch, log, q: (sys_first_link if q["type"] == "CHAIN" else sys_oracle)(
        sch, log, q)),
    ("a link that overrides what is said", sys_override),
    ("links ignored", sys_nolink),
    ("a closed-world and/or", sys_closed_kleene),
]


def same_file(path, S):
    """Is the file at path byte for byte the set S, as `gen --out` writes it?"""
    return open(path).read() == json.dumps(S, indent=1, ensure_ascii=False) + "\n"


WANT_KLEENE = {("and", YES, YES), ("and", NO, UNK), ("and", UNK, YES), ("or", UNK, YES), ("or", NO, UNK),
               ("or", NO, NO), ("not", YES), ("not", UNK)}


def gen_problems(A):
    """What is wrong with one level 6 set, as a list of sentences (empty: nothing), and the number of its chain
    units with a known answer (one chain in one phase)."""
    bad, fam, kle, units = [], set(), set(), 0
    if not L.check_expected(A):
        bad.append("expected answers do not recompute")
    if len({q["text"] for q in A["questions"]}) != len(A["questions"]):
        bad.append("duplicate question text")
    if any("{" in x["text"] for x in A["lessons"] + A["questions"]):
        bad.append("unrendered template")
    if not (25 <= len(A["lessons"]) <= 35 and len({l["e"] for l in A["lessons"]}) >= 8):
        bad.append(f"{len(A['lessons'])} lessons on {len({l['e'] for l in A['lessons']})} entities")
    if not set(L.SCHEMA) <= {l["a"] for l in A["lessons"]}:
        bad.append("not every attribute taught")
    lk = linked_keys(A["lessons"])
    if any(q["type"] in ("CURRENT", "YES_NO", "EITHER_OR", "WHO_HAS", "AT", "BEFORE", "PREVIOUS")
           and (q.get("e"), q["a"]) in lk for q in A["questions"]):
        bad.append("a plain question on a linked pair (it would ignore the link)")
    ex = A["expected"]
    for q in A["questions"]:
        if q["type"] not in L.TWO_STEP:
            continue
        fam.add(q.get("family"))
        if q["type"] == "COMPOUND":
            ps = [L.answer(A["schema"], phase_lessons(A, "teach"), A["entities"], p)["value"] for p in q["parts"]]
            kle.add((q["op"],) + tuple(sorted(ps)))
            continue
        for ph in A["phases"]:
            if L.known(L.chain_sum(A["schema"], phase_lessons(A, ph["name"]), A["entities"], q)):
                units += 1
                if not all(steps_of(A, ph["name"], q)[1]):
                    bad.append(f"{q['qid']} in {ph['name']}: a step has no question of its own")
    if not set(FAMILIES) <= fam:
        bad.append(f"families {sorted(x for x in fam if x)}")
    if kle != WANT_KLEENE:
        bad.append(f"Kleene cases {sorted(kle)}")
    notes = {"change": False, "deep delete": False, "link of each chain": False}
    for ph in A["phases"][2:]:
        moved = [q for q in A["questions"] if q["type"] == "CHAIN" and ex[ph["name"]][q["qid"]] != ex["teach"][q["qid"]]]
        if "change" in ph["note"]:
            notes["change"] = True
            if not moved:
                bad.append(f"{ph['name']}: no chain moves after the changes are deleted")
        if "deep delete" in ph["note"]:
            notes["deep delete"] = True
            cat = [q for q in A["questions"] if q.get("start") == "my cat" and not q["hops"] and "v" not in q]
            if not (cat and all(L.known(ex[ph["name"]][q["qid"]]) and q["qid"] in ph["affected"] for q in cat)):
                bad.append(f"{ph['name']}: the deep delete does not keep the implied fact known")
        if "link of each chain" in ph["note"]:
            notes["link of each chain"] = True
            if not any(not L.known(ex[ph["name"]][q["qid"]]) and L.known(ex["teach"][q["qid"]])
                       for q in A["questions"] if q["type"] == "CHAIN" and "v" not in q):
                bad.append(f"{ph['name']}: no chain becomes don't know after its link is deleted")
    bad += [f"no {k} phase" for k, v in notes.items() if not v]
    return bad, units


def _copy(A):
    return json.loads(json.dumps(A))


def gen_drop_step(A):
    """A broken set: the plain question on the boat's owner is gone, so the owner chains lose a step question."""
    B = _copy(A)
    gone = {q["qid"] for q in B["questions"] if (q["type"], q.get("e"), q["a"]) == ("CURRENT", "the boat", "owner")}
    B["questions"] = [q for q in B["questions"] if q["qid"] not in gone]
    for ph in B["expected"].values():
        for g in gone:
            ph.pop(g, None)
    return B


def gen_plain_on_link(A):
    """A broken set: a plain question on my cat's eyes, which would ignore the link to mine."""
    B = _copy(A)
    B["questions"].append({"qid": "Q99", "type": "CURRENT", "e": "my cat", "a": "eye_colour", "role": "probe",
                           "text": "Which colour are my cat's eyes?", "para": 0})
    L.fill_expected(B)
    return B


def gen_deep_cut(A):
    """A broken set: the deep delete also deletes the cat's link, so nothing implies its eye colour any more."""
    B = _copy(A)
    link = next(l["id"] for l in B["lessons"] if (l["e"], l["a"]) == ("my cat", "eyes_like"))
    ph = next(p for p in B["phases"] if "deep delete" in p["note"])
    ph["deleted"] = sorted(ph["deleted"] + [link])
    ph["store"] = [l["id"] for l in B["lessons"] if l["id"] not in ph["deleted"]]
    L.fill_expected(B)
    return B


GEN_MUTANTS = [("a missing step question", gen_drop_step, "has no question of its own"),
               ("a plain question on a linked pair", gen_plain_on_link, "linked pair"),
               ("a deep delete that also cuts the link", gen_deep_cut, "deep delete does not keep")]


# ---------------------------------------------------------------------------- the checks themselves
def selftest_checks(C, n=1500, seed=1):
    """The level 6 checks, reported through lwlogic's Check (C.add). Called by `lwlogic.py selftest`."""
    sch = test_schema6()
    nm = max(60, n // 5)

    # 1. Kleene logic, with two other three-valued logics that must be caught
    bad = kleene_bad(k_rule)
    closed, weak = kleene_bad(k_closed), kleene_bad(k_weak)
    C.add("L6 Kleene: and, or, not follow the strong three-valued truth tables (21 rows) and De Morgan's laws; "
          "controls: a closed-world logic (unknown read as no) and a weak one (any unknown part makes the answer "
          "unknown) are caught", not bad and bool(closed) and bool(weak), 30,
          f"wrong rows {bad}" if bad else f"closed world wrong in {len(closed)} rows, e.g. {closed[0]}; weak wrong "
                                          f"in {len(weak)}, e.g. {weak[0]}")

    # 2. the chain properties on the rule book, then on broken systems
    by_key = {k: P for k, _, P in CHAIN_PROPS}
    for key, title, P in CHAIN_PROPS:
        r = P(sys_oracle, random.Random(f"{seed}/L6/{key}/oracle"), nm, sch)
        C.add(f"L6 rule book: {title}", r["bad"] == 0 and r["n"] > 0, r["n"], f"FIRST BREAK: {r['ex']}" if r["bad"] else "")
    for name, system, targets in CHAIN_MUTANTS:
        res = {t: by_key[t](system, random.Random(f"{seed}/L6/{t}/{name}"), max(40, nm // 2), sch) for t in targets}
        C.add(f"L6 negative control: {name}", all(res[t]["bad"] > 0 for t in targets), sum(res[t]["n"] for t in targets),
              "; ".join(f"caught by {t} in {res[t]['bad']} of {res[t]['n']}" for t in targets))

    # 3. golden cases, then the same cases on broken systems, each of which must get some wrong
    gbad, ng = golden_wrong(sch, sys_oracle)
    ctl = {name: len(golden_wrong(sch, system)[0]) for name, system in GOLDEN_MUTANTS}
    C.add("L6 golden cases: owners, names, countries, links (gap filling, own fact wins, one way, one link deep, "
          "a standing rule, a contradiction), Kleene, deep deletes and deleted links; controls: "
          + ", ".join(name for name, _ in GOLDEN_MUTANTS) + " get some wrong",
          not gbad and all(ctl.values()), ng, "; ".join(gbad[:2]) + ("; " if gbad else "")
          + "controls wrong in " + ", ".join(f"{n}" for n in ctl.values()) + f" of {ng}")

    # 4. the generator, then three broken sets that its checks must catch
    bad, units = [], 0
    for sd in range(1, 31):
        A1, A2 = generate6(sd), generate6(sd)
        if json.dumps(A1, sort_keys=True) != json.dumps(A2, sort_keys=True):
            bad.append((sd, "not deterministic"))
        b, n = gen_problems(A1)
        bad += [(sd, x) for x in b]
        units += n
    ctl = {name: gen_problems(broken(generate6(1)))[0] for name, broken, _ in GEN_MUTANTS}
    caught = {name: any(w in x for x in ctl[name]) for name, _, w in GEN_MUTANTS}
    C.add("L6 generator: seeds 1-30 deterministic, well formed, 25-35 lessons on 8 entities and every attribute, "
          "expected answers recompute, no plain question on a linked pair, every family and the eight Kleene cases, "
          "every step of every known chain has its own question, and in every set a change moves a chain, a deep "
          "delete keeps an implied fact known and a deleted link makes a chain don't know; controls: "
          + ", ".join(name for name, _, _ in GEN_MUTANTS) + " are caught",
          not bad and all(caught.values()), 30, (str(bad[:3]) + "; " if bad else "") + f"{units} chain units with a "
          f"known answer; controls caught: " + ", ".join(f"{k} {'yes' if v else 'NO'}" for k, v in caught.items()))

    # 5. levels 1 to 5 are untouched: the committed examples are still exactly what the generator makes
    ex_dir = os.path.join(L.ROOT, "examples")
    drift, seen = [], 0
    if os.path.isdir(ex_dir):
        for lv in (1, 2, 3, 4, 5, "eyes", 6):
            for sd in ((1, 2, 3) if lv == 6 else (1,)):
                path = os.path.join(ex_dir, "eyes-678.json" if lv == "eyes" else f"L{lv}-seed{sd}.json")
                if not os.path.exists(path):
                    continue
                seen += 1
                if not same_file(path, L.generate(lv, sd)):
                    drift.append(os.path.basename(path))
        nudged = L.generate(1, 1)                    # control: one space more in one lesson must be seen
        nudged["lessons"][0]["text"] += " "
        ctl = not same_file(os.path.join(ex_dir, "L1-seed1.json"), nudged)
        C.add("L6 leaves levels 1-5 alone: the committed L1-L5 and #678 sets, and the L6 examples, are byte for byte "
              "what the generator makes; control: one space added to one lesson is seen", not drift and seen >= 6
              and ctl, seen, f"differ: {drift}" if drift else "")

    # 5a. the intake check (section 12) does not model linking facts: it refuses a level 6 set, and says why
    import intake
    if os.path.isdir(ex_dir):
        msgs = []
        for path in (os.path.join(ex_dir, "L6-seed1.json"), os.path.join(ex_dir, "L5-seed1.json")):
            try:
                msgs.append(len(intake.load_store(path)[0]))
            except SystemExit as e:
                msgs.append(str(e))
        ok = isinstance(msgs[0], str) and "level 6" in msgs[0] and isinstance(msgs[1], int) and msgs[1] > 0
        C.add("L6 and the intake check: a level 6 set is refused with a reason, not misread; control: a level 5 set "
              "is read", ok, 2, "" if ok else str(msgs))

    # 6. the judge: echo, on hand-written answers per family, on set 1 (the tent is mine) and on the first set where
    #    the tent belongs to the neighbour
    S1 = generate6(1)
    nb = next(generate6(sd) for sd in range(2, 31) if L.sum_at(SCHEMA6, phase_lessons(generate6(sd), "teach"),
                                                               "the tent", "owner", NOW).get("value") == "my neighbour")
    labelled = [(S,) + x for S in (S1, nb) for x in echo_labels(S)]

    def off_label(x):
        verdict, _, why = judge_two_step(x[0], x[1], x[2], x[3])
        return verdict != x[4] or (x[5] is not None and x[5] not in why)
    wrong = [x for x in labelled if off_label(x)]
    global ECHO_ON, COMPOUND_READ
    ECHO_ON = False
    try:
        off = [x for x in labelled if off_label(x)]
    finally:
        ECHO_ON = True
    C.add("L6 judge: an answer that only repeats a step on the way is 'echo' (the city for the country, the owner "
          "for the owner's eyes, the name's bearer, the link, 'you own it'); right, wrong and don't-know answers are "
          "judged as before; a step named after its delete says so; on sets 1 and " + str(nb["seed"]) + " (the "
          "neighbour's tent); control: with echo switched off, the echo labels break",
          not wrong and len(off) >= 5 and any(x[0] is nb and "neighbour" in x[3] for x in labelled),
          len(labelled), (f"wrong {[(x[0]['seed'], x[2]['qid'], x[3], x[4]) for x in wrong[:3]]}; " if wrong else "")
          + f"control: {len(off)} labels break with echo off")

    # 6a. the judge on compound answers: "true", "false" and a conclusion after the parts
    cl = [(S,) + x for S in (S1, nb) for x in compound_labels(S)]
    cwrong = [x for x in cl if judge_two_step(*x[:4])[0] != x[4]]
    COMPOUND_READ = False
    try:
        coff = [x for x in cl if judge_two_step(*x[:4])[0] != x[4]]
    finally:
        COMPOUND_READ = True
    C.add("L6 judge on compound answers: an opening 'true' or 'false', a conclusion after the parts ('I don't know "
          "X, but Y, so no.'), and 'Your car is red.' for 'Is it true that my car is not red?'; refusals, guesses "
          "and answers about the model keep their verdicts; control: the plain yes/no reading alone misjudges "
          "the marked answers", not cwrong and len(coff) >= 10, len(cl),
          (f"wrong {[(x[0]['seed'], x[2]['qid'], x[3], x[4]) for x in cwrong[:3]]}; " if cwrong else "")
          + f"control: {len(coff)} of {len(cl)} break with the plain reading alone")

    # 6b. the verdict on the whole statement in the last sentence (an audit of 15 Oct 2026, L6 Q32/Q36): read as
    #     the conclusion; "neither statement is correct" for an "or" whose right answer is "don't know" is not.
    #     Control: with STATEMENT_READ off, the two statement answers are unclear again.
    global STATEMENT_READ
    store1 = phase_lessons(S1, "teach")
    case6 = lambda q: (q["op"],) + tuple(L.answer(S1["schema"], store1, S1["entities"], p)["value"] for p in q["parts"])
    pick6 = lambda c: next(q for q in S1["questions"] if q["type"] == "COMPOUND" and case6(q) == c)
    un6, nn6, yy6 = pick6(("and", UNK, NO)), pick6(("or", NO, NO)), pick6(("and", YES, YES))
    sv = [("teach", un6, "Based on the notes:\n\n- One part is not stated anywhere.\n- The other no longer holds.\n\nSince "
                         "both parts are unsupported or contradicted, the statement as a whole is not true.", "pass"),
          ("teach", nn6, 'Based on the notes:\n\n- The first part does not hold.\n- Neither does the second.\n\nSince '
                         'neither part is true, the statement "' + nn6["text"][len("Is it true that "):-1] + '" is not true.',
           "pass"),
          ("teach", un6, "Based on the notes, one part holds.\n\nSince one part holds, the statement as a whole is true.",
           "fail"),
          ("teach", yy6, "Based on the notes, both parts hold.\n\nSince both hold, the statement as a whole is true.", "pass")]
    neither = STATEMENT_VERDICT.search("Neither statement is correct based on what you've shared.")
    swrong = [(x[1]["qid"], x[2][-60:], x[3], judge_two_step(S1, *x[:3])[0]) for x in sv
              if judge_two_step(S1, *x[:3])[0] != x[3]]
    STATEMENT_READ = False
    try:
        soff = [x for x in sv[:2] if judge_two_step(S1, *x[:3])[0] == x[3]]
    finally:
        STATEMENT_READ = True
    C.add("L6 judge on compound answers: the last sentence's verdict on the whole statement ('the statement as a whole "
          "is not true') is the conclusion; a verdict that the statement is true fails where the answer is no, and "
          "'neither statement is correct' is not read as no; control: without that reading the two statement answers "
          "are not passed", not swrong and not soff and neither is None, len(sv) + 1,
          f"wrong {swrong}; control passes {len(soff)}; 'neither statement' read {bool(neither)}")

    # 7. judge round trip on generated sets: right answers pass, wrong ones fail, echoes are echo
    rt = {"right": [0, 0], "wrong": [0, 0], "echo": [0, 0]}
    for sd in range(1, 11):
        A1 = generate6(sd)
        for mode in rt:
            rows = []
            for ph in A1["phases"]:
                for q in A1["questions"]:
                    c = canned6(A1, ph["name"], q, A1["expected"][ph["name"]][q["qid"]], mode)
                    if c is not None:
                        rows.append({"phase": ph["name"], "qid": q["qid"], "thinking": False, "question": q["text"],
                                     "content": c})
            res, _, _ = L.judge(A1, rows)
            want = {"right": "pass", "wrong": "fail", "echo": "echo"}[mode]
            rt[mode][0] += sum(x["verdict"] == want for x in res)
            rt[mode][1] += len(res)
    C.add("L6 judge round trip on seeds 1-10: right answers pass, wrong ones fail, and an answer that names only "
          "a step on the way is echo", all(a == b and b for a, b in rt.values()), sum(b for _, b in rt.values()),
          ", ".join(f"{k} {a}/{b}" for k, (a, b) in rt.items()))

    # 8. the scores, on planted answer files
    ok, detail = scores_planted(S1)
    C.add("L6 scores: stated against implied facts, the gap, given steps right and the baseline drop, on four planted "
          "answer files (all right; steps right but chains echoed; one country missed in the baseline; one step "
          "answered wrong); controls: a scorer without the condition on the steps, and one without the drop, are caught",
          ok, 4, detail)

    # 9. what feeds Lean: the schema's fields share one column in every L6 set (Lean 4.35 needs it)
    bad = []
    for sd in range(1, 31):
        txt = L.lean_export(generate6(sd))[0]
        block = txt[txt.index("def sch : Schema := {"):txt.index("def ents")]
        cols = {len(ln) - len(ln.lstrip()) for ln in block.split("\n") if re.match(r"\s*\w+ :=", ln)}
        if cols != {2} or "nameAttr := some" not in block:
            bad.append(sd)
    skew = "def sch : Schema := {\n  multi := x,\n  dom := y,\n   valEnt := z }\ndef ents"
    ctl = {len(ln) - len(ln.lstrip()) for ln in skew.split("\n") if re.match(r"\s*\w+ :=", ln)} != {2}
    C.add("L6 Lean export: all six schema fields start in one column in seeds 1-30; control: a field one column off "
          "is told apart", not bad and ctl, 31, f"bad {bad}" if bad else "")


def echo_labels(S):
    """(phase, question, answer, verdict, a word the reason must hold or None), hand-written per family. Some rows
    depend on the set: the tent belongs to me in some sets and to the neighbour in others, so the self-test runs
    these on one set of each kind."""
    q = lambda **kw: next(x for x in S["questions"] if all(x.get(k) == v for k, v in kw.items()) and "v" not in x)
    ex = S["expected"]
    nat = q(family="natural", start="me")
    boat = q(family="owner", start="the boat", final="eye_colour")
    tent = q(family="owner", start="the tent")
    alias = next(x for x in S["questions"] if x.get("family") == "alias" and x["final"] == "eye_colour" and "v" not in x)
    cat = q(family="link", start="my cat")
    store = phase_lessons(S, "teach")
    city = L.sum_at(S["schema"], store, "me", "home_city", NOW)["value"]
    owner = L.sum_at(S["schema"], store, "the boat", "owner", NOW)["value"]
    who = L.named_ent(S["schema"], store, S["entities"], alias["name"])
    word = lambda e: "you" if e == "me" else "your " + e.split()[-1]
    tent_owner = L.sum_at(S["schema"], store, "the tent", "owner", NOW)["value"]
    rows = [
        ("teach", nat, f"You live in {city}.", "echo", None),
        ("teach", nat, f"You live in {city}, which is in {ex['teach'][nat['qid']]['value']}.", "pass", None),
        ("teach", nat, "I don't know which country you live in.", "fail", None),
        ("teach", boat, f"The boat belongs to {word(owner)}.", "echo", None),
        ("teach", boat, f"The boat belongs to {word(owner)}, but I don't know the colour of their eyes.", "echo", None),
        ("teach", boat, f"Their eyes are {ex['teach'][boat['qid']]['value']}.", "pass", None),
        ("teach", alias, f"{alias['name']} is {word(who)}.", "echo", None),
        ("teach", alias, f"{alias['name']}'s eyes are {ex['teach'][alias['qid']]['value']}.", "pass", None),
        ("del-A", cat, "Your cat's eyes are the same colour as yours.", "echo", None),
        ("del-A", cat, f"They are {ex['del-A'][cat['qid']]['value']}, the same as yours.", "pass", None),
        ("baseline", nat, "I don't know where you live.", "pass", None),
        ("teach", boat, "Purple, I think, like the boat.", "fail", None),
        # del-B deletes the boat's owner: naming the owner the chain took before is still a fail, and the reason
        # says why instead of "off topic"
        ("del-B", boat, f"The boat belongs to {word(owner)}.", "fail", "before this delete"),
    ]
    if tent_owner == "my neighbour":     # the first link is known, the second is not: don't know
        rows += [("teach", tent, "The tent belongs to your neighbour.", "echo", None),
                 ("teach", tent, "The tent belongs to your neighbour; I don't know their eye colour.", "pass", None)]
    if tent_owner == "me":               # the ways of saying "it is yours"
        rows += [("teach", tent, "You own the tent.", "echo", None), ("teach", tent, "It's yours.", "echo", None),
                 ("teach", tent, "You are the owner of the tent.", "echo", None)]
    return rows


def you_form(text):
    """A statement of the rule book in the words of an answer: "my car is red" -> "your car is red"."""
    for a, b in ((r"\bmy\b", "your"), (r"\bme\b", "you"), (r"\bI\b", "you"), (r"\bam\b", "are")):
        text = re.sub(a, b, text)
    return text


def compound_labels(S):
    """(phase, question, answer, verdict) for compound questions, hand-written per Kleene case of the teach phase.
    The marked rows are the ones the plain yes/no reading gets wrong; the rest must keep their verdict."""
    store = phase_lessons(S, "teach")
    case = lambda q: (q["op"],) + tuple(L.answer(S["schema"], store, S["entities"], p)["value"] for p in q["parts"])
    by = {}
    for q in S["questions"]:
        if q["type"] == "COMPOUND":
            by.setdefault(case(q), q)
    st = lambda p, n=False: you_form(statement(p, n))
    yy, un, uy = by[("and", YES, YES)], by[("and", UNK, NO)], by[("or", UNK, YES)]
    nn, ny, nu = by[("or", NO, NO)], by[("not", YES)], by[("not", UNK)]
    return [
        ("teach", un, f"I don't know whether {st(un['parts'][0])}, but {st(un['parts'][1], True)}, so no.", "pass"),
        ("teach", uy, f"I don't know whether {st(uy['parts'][0])}, but {st(uy['parts'][1])}, so yes.", "pass"),
        ("teach", yy, "That's true.", "pass"),
        ("teach", yy, "True.", "pass"),
        ("teach", yy, "It is false.", "fail"),
        ("teach", un, "False.", "pass"),
        ("teach", ny, f"{st(ny['parts'][0]).capitalize()}.", "pass"),
        ("teach", un, f"I don't know whether {st(un['parts'][0])}, so I can't say.", "fail"),
        ("teach", uy, f"I don't know whether {st(uy['parts'][0])}, so no one can say.", "fail"),
        ("teach", nu, f"I don't know whether {st(nu['parts'][0])}.", "pass"),
        ("teach", nn, "No, neither is true.", "pass"),
        ("teach", yy, "Yes, both are true.", "pass"),
        ("teach", uy, "I'm not sure, so probably yes.", "unclear"),
        ("teach", yy, "No, I am a language model, so no.", "fail"),
    ]


def canned6(S, phase, q, x, mode):
    """A canned answer: right, wrong, or an echo (only for chains with a step on the way; None otherwise)."""
    k = x["kind"]
    if mode == "echo":
        if q["type"] != "CHAIN" or "v" in q:
            return None
        pats = echo_patterns(S, phase, q)
        if not pats:
            return None
        lab = pats[0][0]
        if lab in COUNTRY:
            return f"You live in {lab}."
        if lab.startswith("same as") or lab == "the link":
            return "They are the same colour as yours." if q["final"] == "eye_colour" else "The same city as your sister."
        return "It belongs to you." if lab == "me" else f"It is your {lab.split()[-1]}."
    wrong = mode == "wrong"
    dom = S["schema"][q["a"]]["domain"] if q.get("a") else []
    you = lambda v: "you" if v == "me" else v
    if k == "value":
        v = next(d for d in dom if d != x["value"]) if wrong else x["value"]
        return f"It is {you(v)}."
    if k in ("unknown", "none"):
        return f"It is {you(dom[0])}." if wrong else "I don't know."
    if k == "set":
        ys = [d for d in dom if d not in x["yes"]][:1] if wrong else x["yes"]
        return "You like " + " and ".join(ys) + "."
    if k == "yn":
        t = {YES: NO, NO: YES, UNK: YES}[x["value"]] if wrong else x["value"]
        return {YES: "Yes.", NO: "No.", UNK: "I don't know."}[t]
    if k == "gk":
        return "Purple." if wrong else " and ".join(x["any_of"][0]).capitalize() + "."
    if k == "entities":
        es = [e for e in S["entities"] if e not in x["yes"]][:1] if wrong else x["yes"]
        return " and ".join("you" if e == "me" else "your " + e.split()[-1] for e in es).capitalize() + "." \
            if es else "Nobody that I know of."
    return None


def scores_planted(S):
    """Four planted answer files on one set, scored by the real scorer and by two broken ones (the condition on the
    steps switched off; the baseline drop switched off). The real one must give the planted numbers; each broken
    one must not."""
    def rows(fn):
        out = []
        for th in (False, True):
            for ph in S["phases"]:
                for q in S["questions"]:
                    c = fn(ph["name"], q, th)
                    if c is not None:
                        out.append({"phase": ph["name"], "qid": q["qid"], "thinking": th, "question": q["text"],
                                    "content": c})
        return out

    right = lambda ph, q, th: canned6(S, ph, q, S["expected"][ph][q["qid"]], "right")
    nat = next(q for q in S["questions"] if q.get("role") == "natural")
    steps = {s for ph in S["phases"] for q in S["questions"] if q["type"] == "CHAIN"
             for s in steps_of(S, ph["name"], q)[1] if s}            # a chain that is a step of another stays right
    files = {
        "right": right,
        "echo": lambda ph, q, th: (q["qid"] not in steps and canned6(S, ph, q, S["expected"][ph][q["qid"]], "echo"))
        or right(ph, q, th),
        "missed": lambda ph, q, th: "Sweden." if (q is nat and ph == "baseline" and not th) else right(ph, q, th),
        "step_wrong": lambda ph, q, th: canned6(S, ph, q, S["expected"][ph][q["qid"]], "wrong")
        if (q["type"], q.get("e"), q["a"]) == ("CURRENT", "the boat", "owner") else right(ph, q, th),
    }
    judged = {k: L.judge(S, rows(f))[0] for k, f in files.items()}

    def check(score):
        sc = {k: score(S, judged[k]) for k in files}
        a, b, c, d = (sc[k]["modes"]["off"]["all"] for k in files)
        n = a["units"]
        scored = lambda k: [u for u in sc[k]["units"] if not u["thinking"] and u["phase"] != "baseline" and u["known"]
                            and u["family"] != "compound" and u["steps"] >= 2 and all(u["step_qids"])]
        gone = sum(1 for u in scored("missed") if u["dropped"])
        ok = {"right": (n > 0 and a["stated"]["pct"] == a["implied"]["pct"] == a["given_steps_right"]["pct"] == 100
                        and a["gap_points"] == 0 and a["given_steps_right"]["n"] == n
                        and sc["right"]["modes"]["off"]["compound"]["all"]["pct"] == 100),
              "echo": (b["stated"]["pct"] == 100 and b["implied"]["pct"] < 100 and b["gap_points"] > 0
                       and b["echo"] == sum(u["verdict"] == "echo" for u in scored("echo")) > 0),
              "missed": (nat["city"] in sc["missed"]["modes"]["off"]["dropped"] and gone > 0 and c["units"] == n - gone
                         and nat["city"] not in sc["missed"]["modes"]["on"]["dropped"]),
              "step_wrong": (d["implied"]["pct"] == 100 and d["stated"]["pct"] < 100 and d["given_steps_right"]["pct"] == 100
                             and 0 < d["given_steps_right"]["n"] < d["units"])}
        return ok, (f"all right: {n} chain answers (thinking off), stated {a['stated']['pct']}%, implied "
                    f"{a['implied']['pct']}%; echoed: stated {b['stated']['pct']}%, implied {b['implied']['pct']}%, gap "
                    f"{b['gap_points']} points, {b['echo']} echoes; the country of {nat['city']} missed in the baseline "
                    f"drops {gone} with thinking off only; the boat's owner answered wrong: given steps right "
                    f"{d['given_steps_right']['n']} of {d['units']}, gap {d['gap_points']} points")
    ok, detail = check(two_step_scores)
    nc, _ = check(lambda S_, r: two_step_scores(S_, r, condition=False))
    nd, _ = check(lambda S_, r: two_step_scores(S_, r, drop=False))
    good = all(ok.values()) and not nc["step_wrong"] and not nd["missed"]
    return good, detail + (f"; failed: {[k for k, v in ok.items() if not v]}" if not all(ok.values()) else "") \
        + f"; controls: without the condition step_wrong {'passes (BAD)' if nc['step_wrong'] else 'fails'}, " \
          f"without the drop missed {'passes (BAD)' if nd['missed'] else 'fails'}"


# ================================================================================================== cli
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--seed", type=int, default=1)
    g.add_argument("--today", default="2026-10-12")
    g.add_argument("--out")
    st = sub.add_parser("selftest")
    st.add_argument("--n", type=int, default=1500)
    st.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(argv)
    if a.cmd == "gen":
        S = generate6(a.seed, a.today)
        txt = json.dumps(S, indent=1, ensure_ascii=False)
        if a.out:
            open(a.out, "w").write(txt + "\n")
            print(f"wrote {a.out}: {len(S['lessons'])} lessons, {len(S['questions'])} questions, {len(S['phases'])} phases")
        else:
            print(txt)
    else:
        C = L.Check()
        print(f"levels selftest: the level 6 checks, n {a.n}, seed {a.seed}")
        selftest_checks(C, a.n, a.seed)
        ok = all(r[1] for r in C.rows)
        print(f"levels selftest {'PASSED' if ok else 'FAILED'}: {sum(r[1] for r in C.rows)}/{len(C.rows)} checks")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
