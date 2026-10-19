#!/usr/bin/env python3
"""lwlogic: the referee for Living Weights tests (oracle, test-set generator, answer judge).

Standard library only. The oracle mirrors LwLogic.lean definition by definition (same names), so when a model
disagrees with a set's expected answer, the disagreement is the model's and not the referee's.

  python3 zengine/lwlogic.py gen --level 3 --seed 7 --out set.json     levels 1..6, or --level eyes (the #678 case)
  python3 zengine/lwlogic.py show set.json                             the set in plain English
  python3 zengine/lwlogic.py judge set.json answers.jsonl[:PHASE] ...  score a model's answers (--json out.json)
  python3 zengine/lwlogic.py dummy set.json --says dont_know|yes        a dummy system's answers (negative controls)
  python3 zengine/lwlogic.py lean set.json --out set.check.lean        Lean checks of every expected answer
  python3 zengine/lwlogic.py leancheck [set.json ...]                  run Lean: proofs, and the checks of each set
  python3 zengine/lwlogic.py selftest [--n 1500] [--seed 1]            invariants, differential test, negative controls
  python3 zengine/lwlogic.py intake --store STORE.json --text "..."      sort a new fact first (see zengine/intake.py)

answers.jsonl: one JSON object per line with "qid" (or probe.py's "id"), "content", optional "thinking" (bool)
and "phase" (or give the phase after a colon on the command line).
"""
import argparse
import bisect
import datetime as dt
import itertools
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time

YES, NO, UNK = "yes", "no", "unk"
NOW = "now"
TPD = 1_000_000  # ticks per day. A date D is tick D*TPD; a lesson told today gets a tick inside today.

# =============================================================================================== schema
COLOURS = ["red", "blue", "green", "yellow", "purple", "orange", "black", "white", "silver", "brown", "grey",
           "hazel", "amber", "pink", "violet", "turquoise", "teal", "navy", "beige", "gold"]

SCHEMA = {
    "eye_colour": {"multi": False, "applies": ["self", "person", "animal"],
                   "domain": ["brown", "green", "blue", "grey", "hazel", "amber"], "extra": COLOURS,
                   "topic": ["eye", "colour", "color", "iris"]},
    "fav_colour": {"multi": False, "applies": ["self", "person"],
                   "domain": ["red", "blue", "green", "yellow", "purple", "orange", "black", "white"],
                   "extra": COLOURS, "topic": ["colour", "color", "favourite", "favorite"]},
    "name": {"multi": False, "applies": ["person", "animal", "vehicle", "object"],
             "domain": ["Ana", "Milo", "Luna", "Otto", "Bruno", "Nora", "Pablo", "Elsa"], "extra": [],
             "topic": ["name", "called", "named"]},
    "colour": {"multi": False, "applies": ["vehicle", "object"],
               "domain": ["red", "blue", "green", "yellow", "black", "white", "silver", "orange"],
               "extra": COLOURS, "topic": ["colour", "color", "painted"]},
    "home_city": {"multi": False, "applies": ["self", "person"],
                  "domain": ["Lisbon", "Oslo", "Tampere", "Porto", "Ghent", "Graz", "Bergen", "Turku"],
                  "extra": ["Helsinki", "Berlin", "Paris", "London", "Stockholm", "Madrid", "Rome", "Vienna"],
                  "topic": ["live", "city", "home", "lived"]},
    "likes": {"multi": True, "applies": ["self", "person", "animal"],
              "domain": ["apples", "cheese", "olives", "chocolate", "mushrooms", "pears"],
              "extra": ["bread", "fish", "rice", "bananas", "pasta"], "topic": ["like", "food", "fond", "enjoy"]},
    "owner": {"multi": False, "applies": ["object"],
              "domain": ["me", "my sister", "my brother", "my neighbour"], "extra": [],
              "topic": ["own", "belong", "whose", "owner"]},
}
ATTR_ORDER = list(SCHEMA)

# nom: subject; poss: possessive; pron: subject pronoun; verb forms agree with nom and pron.
ENTITIES = {
    "me": {"kind": "self", "nom": "I", "poss": "my", "pron": "I", "be": "am", "have": "have", "do": "do", "s": ""},
    "my sister": {"kind": "person", "pron": "she"}, "my brother": {"kind": "person", "pron": "he"},
    "my cousin": {"kind": "person", "pron": "she"},
    "my cat": {"kind": "animal", "pron": "it"}, "my dog": {"kind": "animal", "pron": "it"},
    "my rabbit": {"kind": "animal", "pron": "it"},
    "my car": {"kind": "vehicle", "pron": "it"}, "my bike": {"kind": "vehicle", "pron": "it"},
    "the boat": {"kind": "object", "pron": "it"}, "the tent": {"kind": "object", "pron": "it"},
}
# Added for level 7 (fill to capacity, 13 Oct 2026): more people, animals and things, so that 64 to 100 facts each get
# an (entity, attribute) pair of their own. They come after the first eleven, and levels 1 to 5 name their entities
# themselves, so the sets of levels 1 to 5 are byte for byte what they were (the self-test compares the examples).
ENTITIES.update({
    "my aunt": {"kind": "person", "pron": "she"}, "my uncle": {"kind": "person", "pron": "he"},
    "my friend": {"kind": "person", "pron": "he"}, "my colleague": {"kind": "person", "pron": "she"},
    "my grandmother": {"kind": "person", "pron": "she"}, "my grandfather": {"kind": "person", "pron": "he"},
    "my niece": {"kind": "person", "pron": "she"}, "my nephew": {"kind": "person", "pron": "he"},
    "my horse": {"kind": "animal", "pron": "it"}, "my parrot": {"kind": "animal", "pron": "it"},
    "my hamster": {"kind": "animal", "pron": "it"},
    "my van": {"kind": "vehicle", "pron": "it"}, "my scooter": {"kind": "vehicle", "pron": "it"},
    "the kite": {"kind": "object", "pron": "it"}, "the lamp": {"kind": "object", "pron": "it"},
    "the clock": {"kind": "object", "pron": "it"},
})
for _n, _e in ENTITIES.items():
    if _n != "me":
        _e.update(nom=_n, poss=_n + "'s", be="is", have="has", do="does", s="s")
SIMILAR = {"my cat": "my dog", "my dog": "my cat", "my sister": "my brother", "my brother": "my sister",
           "my car": "my bike", "my bike": "my car", "the boat": "the tent", "the tent": "the boat",
           "me": "my sister", "my cousin": "my sister", "my rabbit": "my cat"}
SIMILAR.update({"my aunt": "my uncle", "my uncle": "my aunt", "my grandmother": "my grandfather",
                "my grandfather": "my grandmother", "my niece": "my nephew", "my nephew": "my niece",
                "my friend": "my colleague", "my colleague": "my friend", "my horse": "my dog", "my parrot": "my cat",
                "my hamster": "my rabbit", "my van": "my car", "my scooter": "my bike", "the kite": "the tent",
                "the lamp": "the clock", "the clock": "the lamp"})

SYN = {"grey": ["grey", "gray"], "apples": ["apples", "apple"], "olives": ["olives", "olive"],
       "pears": ["pears", "pear"], "mushrooms": ["mushrooms", "mushroom"], "bananas": ["bananas", "banana"],
       "me": ["you", "yours", "yourself"], "my sister": ["your sister", "my sister", "sister"],
       "my brother": ["your brother", "my brother", "brother"],
       "my neighbour": ["your neighbour", "your neighbor", "neighbour", "neighbor"]}

T = {}  # templates: T[attr][slot] -> list of paraphrases
T["eye_colour"] = dict(
    ASSERT=["{poss} eyes are {v}.", "{nom} {have} {v} eyes.", "The colour of {poss} eyes is {v}."],
    SINCE=["{poss} eyes have been {v} since {d}."], UNTIL=["{poss} eyes were {v} until {d}."],
    ALWAYS_A=["{poss} eyes have always been {v}."],
    CHANGE=["{poss} eyes are {v} now.", "{poss} eyes have changed colour; they are {v} now.",
            "These days {poss} eyes are {v}."],
    RETRACT=["{poss} eyes are not {v} anymore.", "{poss} eyes are no longer {v}."], CLEAR=[],
    DATED=["{poss} eyes were {v0} until {d}; now they are {v1}.",
           "Until {d}, {poss} eyes were {v0}. Since then they have been {v1}.",
           "{poss} eyes changed from {v0} to {v1} on {d}."],
    Q_CURRENT=["What colour are {poss} eyes?", "What is {poss} eye colour?", "Tell me the colour of {poss} eyes."],
    Q_AT=["What colour were {poss} eyes on {d}?", "On {d}, what colour were {poss} eyes?"],
    Q_BEFORE=["What colour were {poss} eyes before {d}?", "Before {d}, what colour were {poss} eyes?"],
    Q_PREVIOUS=["What colour were {poss} eyes before they changed?", "What colour did {poss} eyes use to be?"],
    Q_ALWAYS=["Have {poss} eyes always been the same colour?"], Q_CHANGED=["Have {poss} eyes ever changed colour?"],
    Q_EITHER_OR=["Are {poss} eyes {x} or {y}?", "Which is it for {poss} eyes, {x} or {y}?"],
    Q_YES_NO=["Are {poss} eyes {v}?", "Is it true that {poss} eyes are {v}?"],
    Q_WHO_HAS=["Who has {v} eyes?", "Whose eyes are {v}?"])
T["fav_colour"] = dict(
    ASSERT=["{poss} favourite colour is {v}.", "{nom} like{s} {v} best.", "Of all colours, {nom} like{s} {v} the most."],
    SINCE=["{poss} favourite colour has been {v} since {d}."], UNTIL=["{poss} favourite colour was {v} until {d}."],
    ALWAYS_A=["{poss} favourite colour has always been {v}."],
    CHANGE=["{poss} favourite colour is {v} now.", "{nom} like{s} {v} best now.",
            "{poss} favourite colour has changed; it is {v} now."],
    RETRACT=["{v} is not {poss} favourite colour anymore.", "{poss} favourite colour is no longer {v}."],
    CLEAR=["{nom} {have} no favourite colour anymore.", "{nom} {do} not have a favourite colour anymore."],
    DATED=["{poss} favourite colour was {v0} until {d}; now it is {v1}.",
           "Until {d}, {poss} favourite colour was {v0}. Since then it has been {v1}."],
    Q_CURRENT=["What is {poss} favourite colour?", "Which colour {do} {nom} like best?",
               "What colour {do} {nom} like the most?"],
    Q_AT=["What was {poss} favourite colour on {d}?"], Q_BEFORE=["What was {poss} favourite colour before {d}?"],
    Q_PREVIOUS=["What was {poss} favourite colour before it changed?", "What did {poss} favourite colour use to be?"],
    Q_ALWAYS=["Has {poss} favourite colour always been the same?"], Q_CHANGED=["Has {poss} favourite colour ever changed?"],
    Q_EITHER_OR=["Is {poss} favourite colour {x} or {y}?", "{do} {nom} like {x} or {y} best?"],
    Q_YES_NO=["Is {poss} favourite colour {v}?", "Is {v} {poss} favourite colour?"],
    Q_WHO_HAS=["Whose favourite colour is {v}?", "Who likes {v} best?"])
T["name"] = dict(
    ASSERT=["{poss} name is {v}.", "{nom} is called {v}.", "{nom} is named {v}."],
    SINCE=["{nom} has been called {v} since {d}."], UNTIL=["{nom} was called {v} until {d}."],
    ALWAYS_A=["{nom} has always been called {v}."],
    CHANGE=["{nom} is called {v} now.", "{nom} has a new name; {pron} is called {v} now."],
    RETRACT=["{nom} is not called {v} anymore.", "{nom} is no longer called {v}."], CLEAR=[],
    DATED=["{nom} was called {v0} until {d}; now {pron} is called {v1}.",
           "Until {d}, {nom} was called {v0}. Since then {pron} has been called {v1}."],
    Q_CURRENT=["What is {poss} name?", "What is {nom} called?"],
    Q_AT=["What was {nom} called on {d}?"], Q_BEFORE=["What was {nom} called before {d}?"],
    Q_PREVIOUS=["What was {nom} called before the name changed?", "What did {nom} use to be called?"],
    Q_ALWAYS=["Has {nom} always had the same name?"], Q_CHANGED=["Has {nom} ever had a different name?"],
    Q_EITHER_OR=["Is {nom} called {x} or {y}?"], Q_YES_NO=["Is {nom} called {v}?", "Is {poss} name {v}?"],
    Q_WHO_HAS=["Who is called {v}?", "Whose name is {v}?"])
T["colour"] = dict(
    ASSERT=["{nom} is {v}.", "{poss} colour is {v}.", "{nom} is painted {v}."],
    SINCE=["{nom} has been {v} since {d}."], UNTIL=["{nom} was {v} until {d}."], ALWAYS_A=["{nom} has always been {v}."],
    CHANGE=["{nom} is {v} now.", "{nom} has been repainted; it is {v} now."],
    RETRACT=["{nom} is not {v} anymore.", "{nom} is no longer {v}."], CLEAR=[],
    DATED=["{nom} was {v0} until {d}; now it is {v1}.", "Until {d}, {nom} was {v0}. Since then it has been {v1}."],
    Q_CURRENT=["What colour is {nom}?", "What is the colour of {nom}?"],
    Q_AT=["What colour was {nom} on {d}?"], Q_BEFORE=["What colour was {nom} before {d}?"],
    Q_PREVIOUS=["What colour was {nom} before it was repainted?", "What colour did {nom} use to be?"],
    Q_ALWAYS=["Has {nom} always been the same colour?"], Q_CHANGED=["Has {nom} ever changed colour?"],
    Q_EITHER_OR=["Is {nom} {x} or {y}?"], Q_YES_NO=["Is {nom} {v}?"], Q_WHO_HAS=["Which of my things is {v}?"])
T["home_city"] = dict(
    ASSERT=["{nom} live{s} in {v}.", "{poss} home is in {v}.", "{nom} {be} living in {v}."],
    SINCE=["{nom} {have} lived in {v} since {d}."], UNTIL=["{nom} lived in {v} until {d}."],
    ALWAYS_A=["{nom} {have} always lived in {v}."],
    CHANGE=["{nom} live{s} in {v} now.", "{nom} moved to {v}.", "{nom} {have} moved; {pron} live{s} in {v} now."],
    RETRACT=["{nom} {do} not live in {v} anymore.", "{nom} no longer live{s} in {v}."], CLEAR=[],
    DATED=["{nom} lived in {v0} until {d}; now {pron} live{s} in {v1}.",
           "Until {d}, {nom} lived in {v0}. Since then {pron} {have} lived in {v1}."],
    Q_CURRENT=["Where {do} {nom} live?", "Which city {do} {nom} live in?", "In which city is {poss} home?"],
    Q_AT=["Where did {nom} live on {d}?"], Q_BEFORE=["Where did {nom} live before {d}?"],
    Q_PREVIOUS=["Where did {nom} live before moving?", "Where did {nom} use to live?"],
    Q_ALWAYS=["{have} {nom} always lived in the same city?"], Q_CHANGED=["{have} {nom} ever moved to another city?"],
    Q_EITHER_OR=["{do} {nom} live in {x} or {y}?"], Q_YES_NO=["{do} {nom} live in {v}?", "Is {poss} home in {v}?"],
    Q_WHO_HAS=["Who lives in {v}?"])
T["likes"] = dict(
    ASSERT=["{nom} like{s} {v}.", "{nom} really like{s} {v}.", "{nom} {be} fond of {v}."],
    SINCE=["{nom} {have} liked {v} since {d}."], UNTIL=["{nom} liked {v} until {d}."],
    ALWAYS_A=["{nom} {have} always liked {v}."],
    CHANGE=["{nom} only like{s} {v} now.", "Now {nom} like{s} only {v} and nothing else."],
    RETRACT=["{nom} {do} not like {v} anymore.", "{nom} no longer like{s} {v}."], CLEAR=[], DATED=[],
    Q_CURRENT=["What food {do} {nom} like?", "Which foods {do} {nom} like?"],
    Q_AT=["What food did {nom} like on {d}?"], Q_BEFORE=["What food did {nom} like before {d}?"],
    Q_PREVIOUS=[], Q_ALWAYS=[], Q_CHANGED=[], Q_EITHER_OR=[],
    Q_YES_NO=["{do} {nom} like {v}?", "Is it true that {nom} like{s} {v}?"], Q_WHO_HAS=["Who likes {v}?"])
T["owner"] = dict(
    ASSERT=["{nom} belongs to {v}.", "{nom} is owned by {v}."],
    SINCE=["{nom} has belonged to {v} since {d}."], UNTIL=["{nom} belonged to {v} until {d}."],
    ALWAYS_A=["{nom} has always belonged to {v}."],
    CHANGE=["{nom} belongs to {v} now.", "{nom} has a new owner: {v}."],
    RETRACT=["{nom} does not belong to {v} anymore.", "{nom} no longer belongs to {v}."], CLEAR=[],
    DATED=["{nom} belonged to {v0} until {d}; now it belongs to {v1}.",
           "Until {d}, {nom} belonged to {v0}. Since then it has belonged to {v1}."],
    Q_CURRENT=["Who owns {nom}?", "Whose is {nom}?", "Who does {nom} belong to?"],
    Q_AT=["Who owned {nom} on {d}?"], Q_BEFORE=["Who owned {nom} before {d}?"],
    Q_PREVIOUS=["Who owned {nom} before the current owner?"],
    Q_ALWAYS=["Has {nom} always had the same owner?"], Q_CHANGED=["Has {nom} ever changed owner?"],
    Q_EITHER_OR=["Does {nom} belong to {x} or {y}?"], Q_YES_NO=["Does {nom} belong to {v}?"], Q_WHO_HAS=[])

# Question forms v2 (`gen --forms v2`, TEST-LOGIC.md section 13). Kept apart from T, so v1 sets and the intake parser
# never see them. TF: "True or false: ..."; FILL: the start of a sentence for the model to finish, with no value in it;
# RECALL_YN: say the value first, then answer yes or no; WHO_YN: the reverse question as yes/no (the value is given,
# the entity is the guess); WHO_LISTED: the reverse question over a named list; ASSERT_SET: a lesson that restates the
# whole set of likes ("I like pears, olives and apples.").
T2 = {
    "eye_colour": dict(TF=["True or false: {poss} eyes are {v}."], FILL=["{poss} eyes are"],
                       RECALL_YN=["First say what colour {poss} eyes are, then answer yes or no: are they {v}?"],
                       WHO_YN=["Is it {obj} who has {v} eyes?"], WHO_LISTED=["Of {list}, who has {v} eyes?"]),
    "fav_colour": dict(TF=["True or false: {poss} favourite colour is {v}."], FILL=["{poss} favourite colour is"],
                       RECALL_YN=["First say what {poss} favourite colour is, then answer yes or no: is it {v}?"],
                       WHO_YN=["Is it {obj} whose favourite colour is {v}?"],
                       WHO_LISTED=["Of {list}, whose favourite colour is {v}?"]),
    "name": dict(TF=["True or false: {nom} is called {v}."], FILL=["{poss} name is"],
                 RECALL_YN=["First say what {nom} is called, then answer yes or no: is {pron} called {v}?"],
                 WHO_YN=["Is it {obj} who is called {v}?"], WHO_LISTED=["Of {list}, who is called {v}?"]),
    "colour": dict(TF=["True or false: {nom} is {v}."], FILL=["The colour of {nom} is"],
                   RECALL_YN=["First say what colour {nom} is, then answer yes or no: is it {v}?"],
                   WHO_YN=["Is it {obj} that is {v}?"], WHO_LISTED=["Of {list}, which one is {v}?"]),
    "home_city": dict(TF=["True or false: {nom} live{s} in {v}."], FILL=["{nom} live{s} in"],
                      RECALL_YN=["First say where {nom} live{s}, then answer yes or no: {do} {pron} live in {v}?"],
                      WHO_YN=["Is it {obj} who lives in {v}?"], WHO_LISTED=["Of {list}, who lives in {v}?"]),
    "likes": dict(WHO_YN=["Is it {obj} who likes {v}?"], WHO_LISTED=["Of {list}, who likes {v}?"],
                  ASSERT_SET=["{nom} like{s} {vs}."]),
    "owner": dict(TF=["True or false: {nom} belongs to {v}."], FILL=["{nom} belongs to"],
                  RECALL_YN=["First say who {nom} belongs to, then answer yes or no: does it belong to {v}?"]),
}
FILL_WRAP = 'Finish this sentence: "{s} ..."'
FORMS = ("v1", "v2")
FORM_WORDS = {"true_false": "true or false", "fill_in": "fill-in", "recall_yn": "recall, then yes or no",
              "who_bare": "reverse", "who_yn": "reverse yes/no", "who_listed": "reverse, listed"}

# general-knowledge controls: "near" lists the attributes whose taught values could leak into them
GK = [
    {"text": "What is the capital of France?", "any_of": [["paris"]], "near": []},
    {"text": "How many legs does a spider have?", "any_of": [["eight"], ["8"]], "near": []},
    {"text": "Name three primary colours.", "any_of": [["red", "yellow", "blue"], ["red", "green", "blue"]],
     "near": ["fav_colour", "colour", "eye_colour"]},
    {"text": "What colour is the sky on a clear day?", "any_of": [["blue"]], "near": ["fav_colour", "colour", "eye_colour"]},
    {"text": "What colour is fresh grass?", "any_of": [["green"]], "near": ["fav_colour", "colour", "eye_colour"]},
    {"text": "What colour is snow?", "any_of": [["white"]], "near": ["fav_colour", "colour"]},
    {"text": "What is the capital of Norway?", "any_of": [["oslo"]], "near": ["home_city"]},
    {"text": "What is the capital of Portugal?", "any_of": [["lisbon"]], "near": ["home_city"]},
    {"text": "What is cheese made from?", "any_of": [["milk"]], "near": ["likes"]},
    {"text": "What do you call a young cat?", "any_of": [["kitten"]], "near": ["name", "eye_colour"]},
    {"text": "Who usually owns a library book?", "any_of": [["library"]], "near": ["owner"]},
]


# ================================================================================================ oracle
def day_of(iso):
    return dt.date.fromisoformat(iso).toordinal()


def iso_of(n):
    return dt.date.fromordinal(n).isoformat()


def say_date(iso):
    d = dt.date.fromisoformat(iso)
    return f"{d.day} {d.strftime('%B')} {d.year}"


def tick(iso):
    return day_of(iso) * TPD


def covers(iv, p):
    """Interval.covers: [lo, hi) with None for unbounded; NOW is covered exactly when hi is unbounded."""
    lo, hi = iv
    if p == NOW:
        return hi is None
    return (lo is None or lo <= p) and (hi is None or p < hi)


def when_interval(l):
    w = l.get("when")
    if not w:
        return (l["told"], None)                     # present tense: holds from the moment it is told
    if "since" in w:
        return (tick(w["since"]), None)
    if "until" in w:
        return (None, tick(w["until"]))
    if "between" in w:
        return (tick(w["between"][0]), tick(w["between"][1]))
    if w.get("always"):
        return (None, None)                          # timeless: "have always been"
    raise ValueError(f"bad when: {w}")


def writes(sch, l):
    """Op.writes: a lesson as a list of (entity, attribute, interval, effect).

    Hints. Some wordings also say something about the other side of the change they describe:
      "... now" / "has changed" (CHANGE v)   it was NOT v just before it was said
      "... not v anymore" (RETRACT v)        it WAS v just before it was said
      "... since D"                          it was NOT v just before D
      "... until D"                          it is NOT v from D on
    A hint is a write with effect "presume" / "presume_not". It only fills a gap: it acts only where v is still
    unknown at that point, so it never overrides anything a lesson says outright (see `step`)."""
    e, a, op = l["e"], l["a"], l["op"]
    multi = sch[a]["multi"]
    if op == "ASSERT" and l.get("vs"):               # a restated set ("I like pears and olives."), forms v2:
        lo, hi = when_interval(l)                    # one add per value, present tense, no hint; Lean gets one
        return [(e, a, (lo, hi), ("add", v)) for v in l["vs"]]   # assert per value with the same id and tick
    if op == "ASSERT":
        lo, hi = when_interval(l)
        out = [(e, a, (lo, hi), ("add" if multi else "set", l["v"]))]
        if l.get("when"):                            # present tense carries no hint; "always" has no edges
            if lo is not None:
                out.append((e, a, (lo - 1, lo), ("presume_not", l["v"])))
            if hi is not None:
                out.append((e, a, (hi, None), ("presume_not", l["v"])))
        return out
    t = l["told"]
    if op == "CHANGE":                               # on likes, "now I only like v" says nothing about v before
        hint = [] if multi else [(e, a, (t - 1, t), ("presume_not", l["v"]))]
        return hint + [(e, a, (t, None), ("set", l["v"]))]
    if op == "RETRACT":
        if l.get("v") is None:
            return [(e, a, (t, None), ("clear", None))]
        return [(e, a, (t - 1, t), ("presume", l["v"])), (e, a, (t, None), ("retract", l["v"]))]
    if op == "DATED":
        d = tick(l["d"])
        return [(e, a, (None, d), ("set", l["v0"])), (e, a, (d, None), ("set", l["v1"]))]
    raise ValueError(f"bad op {op}")


def blank(sch, a):
    return {x: UNK for x in sch[a]["domain"]}


def step(sch, a, eff, s):
    kind, v = eff
    multi, dom = sch[a]["multi"], sch[a]["domain"]
    if kind == "set" or (kind == "add" and not multi):
        return {x: (YES if x == v else NO) for x in dom}
    if kind == "add":
        r = dict(s)
        r[v] = YES
        return r
    if kind == "retract":
        if not multi and s[v] == YES:
            return {x: (NO if x == v else UNK) for x in dom}
        r = dict(s)
        r[v] = NO
        return r
    if kind == "clear":
        return {x: NO for x in dom}
    if kind == "presume":                            # hint: v held here, unless something already says otherwise
        return step(sch, a, ("add", v), s) if s[v] == UNK else s
    if kind == "presume_not":                        # hint: v did not hold here, unless already known
        if s[v] != UNK:
            return s
        r = dict(s)
        r[v] = NO
        return r
    raise ValueError(kind)


def state_at(sch, store, e, a, p):
    """stateAt: fold every write on (e, a) whose interval covers point p, in store order, from all-unknown."""
    s = blank(sch, a)
    for l in store:
        for (we, wa, iv, eff) in writes(sch, l):
            if we == e and wa == a and covers(iv, p):
                s = step(sch, a, eff, s)
    return s


def summarize(sch, a, s):
    dom = sch[a]["domain"]
    first = next((x for x in dom if s[x] == YES), None)
    if first is not None:
        if sch[a]["multi"]:
            return {"kind": "set", "yes": [x for x in dom if s[x] == YES], "no": [x for x in dom if s[x] == NO]}
        return {"kind": "value", "value": first}
    if all(s[x] == NO for x in dom):
        return {"kind": "none"}
    return {"kind": "unknown", "not": [x for x in dom if s[x] == NO]}


def sum_at(sch, store, e, a, p):
    return summarize(sch, a, state_at(sch, store, e, a, p))


def boundaries(sch, store, e, a):
    out = []
    for l in store:
        for (we, wa, (lo, hi), _) in writes(sch, l):
            if we == e and wa == a:
                out += [b for b in (lo, hi) if b is not None]
    return out


def known(sm):
    return sm["kind"] != "unknown"


def previous(sch, store, e, a):
    """"What was it before it changed?": the latest piece below the current run that differs from now and is
    known (a value, none, or a set), skipping unknown gaps ("blue; not blue anymore; red now" -> blue). If no
    earlier piece is known, the piece just below the current run, which may still exclude values ("not green")."""
    cur = sum_at(sch, store, e, a, NOW)
    cands = [b for b in boundaries(sch, store, e, a) if sum_at(sch, store, e, a, b - 1) != cur]
    if not cands:
        return {"kind": "unknown", "not": []}
    kn = [b for b in cands if known(sum_at(sch, store, e, a, b - 1))]
    return sum_at(sch, store, e, a, max(kn or cands) - 1)


def conflicts(p, q):
    if p["kind"] == "value":
        if q["kind"] == "value":
            return p["value"] != q["value"]
        if q["kind"] == "none":
            return True
        if q["kind"] == "unknown":
            return p["value"] in q["not"]
        return False
    if q["kind"] == "value":
        return conflicts(q, p)
    return False


def always(sch, store, e, a):
    pieces = [sum_at(sch, store, e, a, NOW)] + [sum_at(sch, store, e, a, b - 1) for b in boundaries(sch, store, e, a)]
    cur = pieces[0]
    if cur["kind"] == "value" and all(p == cur for p in pieces):
        return YES
    if any(conflicts(p, q) for p in pieces for q in pieces):
        return NO
    return UNK


def neg(t):
    return {YES: NO, NO: YES, UNK: UNK}[t]


def either_or(s, x, y):
    if s[x] == YES:
        return x
    if s[y] == YES:
        return y
    if s[x] == NO and s[y] == NO:
        return "neither"
    return "unknown"


def who_has(sch, store, ents, a, v):
    return [e for e in ents if state_at(sch, store, e, a, NOW)[v] == YES]


# ------------------------------------------------- two-step questions (level 6, zengine/levels.py makes the sets)
# A chain question walks from a start to an answer: "What colour are the eyes of the owner of the boat?" reads the
# boat's owner (my sister), then my sister's eyes. Three schema fields, used only by level 6, carry what it needs:
#   an attribute whose values name entities of the set (owner: "my sister" is the entity my sister);
#   "link": a    a linking attribute. ASSERT(my cat, eyes_like, me) is "My cat's eyes are the same colour as mine.";
#   "natural": {"from": a, "map": {...}}   fixed general knowledge, such as the country of each city.
# A chain asks about now. A name points at the one entity called that now. A link is read one way and one link deep,
# and only fills gaps, like a hint: what is said outright about the cat wins. Compound yes/no questions use Kleene's
# strong three-valued logic: a no decides "and", a yes decides "or", whatever the other part.
TWO_STEP = ("CHAIN", "COMPOUND")


def k_and(p, q):
    return NO if NO in (p, q) else YES if p == q == YES else UNK


def k_or(p, q):
    return YES if YES in (p, q) else NO if p == q == NO else UNK


def ent_of(sm, ents):
    """entOf: the entity a known value names (the owner "my sister" is my sister), or None."""
    return sm["value"] if sm["kind"] == "value" and sm["value"] in ents else None


def link_of(sch, a):
    return next((x for x in sch if sch[x].get("link") == a), None)


def fill(own, src):
    """A link only fills a gap, like a hint: a value known about e itself wins."""
    return {x: (src[x] if own[x] == UNK else own[x]) for x in own}


def linked_state(sch, store, ents, e, a):
    """linkedState: e's own state of a now, its gaps filled from the entity e's link names ("same as mine")."""
    own = state_at(sch, store, e, a, NOW)
    la = link_of(sch, a)
    t = ent_of(sum_at(sch, store, e, la, NOW), ents) if la else None
    return own if t is None else fill(own, state_at(sch, store, t, a, NOW))


def named_ent(sch, store, ents, v):
    """namedEnt: the one entity called v now, or None (nobody, or more than one)."""
    if "name" not in sch:
        return None
    hits = [e for e in ents if state_at(sch, store, e, "name", NOW).get(v) == YES]
    return hits[0] if len(hits) == 1 else None


def start_ent(sch, store, ents, q):
    return named_ent(sch, store, ents, q["name"]) if q.get("name") is not None else q["start"]


def hops_ent(sch, store, ents, e, hops):
    for h in hops:
        if e is None:
            return None
        e = ent_of(sum_at(sch, store, e, h, NOW), ents)
    return e


def nat_sum(sch, nat, sm):
    """natSum: a natural step on a summary ("Oslo" -> "Norway"). Anything but a known value stays unknown."""
    if sm["kind"] == "value":
        c = sch[nat]["natural"]["map"].get(sm["value"])
        return {"kind": "value", "value": c} if c is not None else {"kind": "unknown", "not": []}
    return {"kind": "none"} if sm["kind"] == "none" else {"kind": "unknown", "not": []}


def chain_sum(sch, store, ents, q):
    """chainSum: the answer to a chain question, from the start through every hop to the final attribute."""
    e = hops_ent(sch, store, ents, start_ent(sch, store, ents, q), q["hops"])
    if e is None:
        return {"kind": "unknown", "not": []}
    sm = summarize(sch, q["final"], linked_state(sch, store, ents, e, q["final"]))
    return nat_sum(sch, q["nat"], sm) if q.get("nat") else sm


def chain_keys(sch, store, ents, q):
    """The (entity, attribute) pairs a two-step question reads in this store: a delete that removes a lesson on one
    of them touches the question, even when its answer stays the same."""
    if q["type"] == "COMPOUND":
        return set().union(*(chain_keys(sch, store, ents, p) for p in q["parts"]))
    if q["type"] != "CHAIN":
        return {(q.get("e"), q["a"])}
    keys = {(x, "name") for x in ents} if q.get("name") is not None else set()
    e = start_ent(sch, store, ents, q)
    for h in q["hops"]:
        if e is None:
            return keys
        keys.add((e, h))
        e = ent_of(sum_at(sch, store, e, h, NOW), ents)
    if e is None:
        return keys
    keys.add((e, q["final"]))
    la = link_of(sch, q["final"])
    if la:
        keys.add((e, la))
        t = ent_of(sum_at(sch, store, e, la, NOW), ents)
        if t is not None:
            keys.add((t, q["final"]))
    return keys


def answer(sch, store, ents, q):
    t = q["type"]
    if t == "GK":
        return {"kind": "gk", "any_of": q["any_of"]}
    if t == "CHAIN":
        sm = chain_sum(sch, store, ents, q)
        return {"kind": "yn", "value": derive_yn(sm, q["v"])} if "v" in q else sm
    if t == "COMPOUND":
        ps = [answer(sch, store, ents, p) for p in q["parts"]]
        ps = [p["value"] if p["kind"] == "yn" else UNK for p in ps]
        return {"kind": "yn", "value": neg(ps[0]) if q["op"] == "not" else k_and(*ps) if q["op"] == "and" else k_or(*ps)}
    e, a = q.get("e"), q["a"]
    if t == "CURRENT":
        return sum_at(sch, store, e, a, NOW)
    if t == "AT":
        return sum_at(sch, store, e, a, tick(q["date"]))
    if t == "BEFORE":
        return sum_at(sch, store, e, a, tick(q["date"]) - 1)
    if t == "PREVIOUS":
        return previous(sch, store, e, a)
    if t == "ALWAYS":
        return {"kind": "yn", "value": always(sch, store, e, a)}
    if t == "CHANGED":
        return {"kind": "yn", "value": neg(always(sch, store, e, a))}
    if t == "YES_NO":
        return {"kind": "yn", "value": state_at(sch, store, e, a, NOW)[q["v"]]}
    if t == "EITHER_OR":
        return {"kind": "choice", "value": either_or(state_at(sch, store, e, a, NOW), q["x"], q["y"])}
    if t == "WHO_HAS":
        return {"kind": "entities", "yes": who_has(sch, store, ents, a, q["v"])}
    raise ValueError(t)


def effective(log):
    """A log of {"teach": lesson} and {"delete": id} entries -> the lessons still in force, in teaching order."""
    deleted = {x["delete"] for x in log if "delete" in x}
    return [x["teach"] for x in log if "teach" in x and x["teach"]["id"] not in deleted]


def derive_yn(sm, v):
    k = sm["kind"]
    if k == "value":
        return YES if sm["value"] == v else NO
    if k == "none":
        return NO
    if k == "unknown":
        return NO if v in sm["not"] else UNK
    if k == "set":
        return YES if v in sm["yes"] else (NO if v in sm["no"] else UNK)
    raise ValueError(k)


def derive_eo(sm, x, y):
    dx, dy = derive_yn(sm, x), derive_yn(sm, y)
    if dx == YES:
        return x
    if dy == YES:
        return y
    if dx == NO and dy == NO:
        return "neither"
    return "unknown"


# =================================================== an independent incremental implementation (for testing)
class Timeline:
    """Keeps, per (entity, attribute), sorted cut points and the state on each piece; applies each write by
    splitting pieces. Shares only `writes` and `step` with the oracle, so a disagreement exposes a fold bug."""

    def __init__(self, sch):
        self.sch, self.keys = sch, {}

    def _get(self, e, a):
        if (e, a) not in self.keys:
            self.keys[(e, a)] = ([], [blank(self.sch, a)])
        return self.keys[(e, a)]

    @staticmethod
    def _cut(cuts, pieces, t):
        if t is None:
            return
        i = bisect.bisect_left(cuts, t)
        if i < len(cuts) and cuts[i] == t:
            return
        cuts.insert(i, t)
        pieces.insert(i, dict(pieces[i]))

    def apply(self, l):
        for (e, a, (lo, hi), eff) in writes(self.sch, l):
            cuts, pieces = self._get(e, a)
            self._cut(cuts, pieces, lo)
            self._cut(cuts, pieces, hi)
            for i in range(len(pieces)):
                plo = cuts[i - 1] if i > 0 else None
                phi = cuts[i] if i < len(cuts) else None
                inside = (lo is None or (plo is not None and plo >= lo)) and (hi is None or (phi is not None and phi <= hi))
                if inside:
                    pieces[i] = step(self.sch, a, eff, pieces[i])
        return self

    def state(self, e, a, p):
        cuts, pieces = self._get(e, a)
        return pieces[-1] if p == NOW else pieces[bisect.bisect_right(cuts, p)]

    def answer(self, ents, q):
        sch, t = self.sch, q["type"]
        if t == "GK":
            return {"kind": "gk", "any_of": q["any_of"]}
        e, a = q.get("e"), q["a"]
        sm = lambda p: summarize(sch, a, self.state(e, a, p))
        if t == "CURRENT":
            return sm(NOW)
        if t == "AT":
            return sm(tick(q["date"]))
        if t == "BEFORE":
            return sm(tick(q["date"]) - 1)
        cuts, pieces = self._get(e, a)
        sums = [summarize(sch, a, s) for s in pieces]
        if t == "PREVIOUS":                       # walk the pieces down from the top instead of a max over cuts
            below = [x for x in reversed(sums[:-1]) if x != sums[-1]]
            if not below:
                return {"kind": "unknown", "not": []}
            return next((x for x in below if x["kind"] != "unknown"), below[0])
        if t in ("ALWAYS", "CHANGED"):
            if sums[-1]["kind"] == "value" and all(x == sums[-1] for x in sums):
                r = YES
            else:
                vals = {x["value"] for x in sums if x["kind"] == "value"}
                bad = len(vals) > 1 or any(x["kind"] == "none" or (x["kind"] == "unknown" and set(x["not"]) & vals)
                                           for x in sums) and bool(vals)
                r = NO if bad else UNK
            return {"kind": "yn", "value": r if t == "ALWAYS" else neg(r)}
        s = self.state(e, a, NOW)
        if t == "YES_NO":
            return {"kind": "yn", "value": s[q["v"]]}
        if t == "EITHER_OR":
            return {"kind": "choice", "value": either_or(s, q["x"], q["y"])}
        if t == "WHO_HAS":
            return {"kind": "entities", "yes": [x for x in ents if self.state(x, a, NOW)[q["v"]] == YES]}
        raise ValueError(t)


# ============================================================================================= rendering
def render(tmpl, e=None, **kw):
    f = dict(ENTITIES[e]) if e else {}
    f.update(kw)
    s = tmpl.format(**f)
    return s[0].upper() + s[1:]


def op_text(l):
    """Formal notation of a lesson, e.g. CHANGE(me, fav_colour, green)."""
    op, e, a = l["op"], l["e"], l["a"]
    if op == "DATED":
        return f"DATED({e}, {a}, {l['v0']} until {l['d']}, {l['v1']} since {l['d']})"
    if op == "RETRACT":
        return f"RETRACT({e}, {a}{', ' + l['v'] if l.get('v') is not None else ''})"
    w = l.get("when")
    ws = ""
    if w:
        ws = ", " + ("always" if w.get("always") else " ".join(f"{k} {v}" for k, v in w.items()))
    return f"{op}({e}, {a}, {' + '.join(l['vs']) if l.get('vs') else l['v']}{ws})"


def words(ans):
    k = ans["kind"]
    if k == "value":
        return ans["value"]
    if k == "none":
        return "none (taken back)"
    if k == "unknown":
        return "doesn't know" + (f" (but not {', '.join(ans['not'])})" if ans["not"] else "")
    if k == "set":
        return ", ".join(ans["yes"]) + (f" (not {', '.join(ans['no'])})" if ans["no"] else "")
    if k == "yn":
        return {YES: "yes", NO: "no", UNK: "doesn't know"}[ans["value"]]
    if k == "choice":
        return {"neither": "neither", "unknown": "doesn't know"}.get(ans["value"], ans["value"])
    if k == "entities":
        return ", ".join(ans["yes"]) if ans["yes"] else "nobody known"
    if k == "gk":
        return " / ".join(" + ".join(s) for s in ans["any_of"]) + " (general knowledge)"
    return json.dumps(ans)


# ============================================================================================= generator
QTYPES = ["CURRENT", "AT", "BEFORE", "PREVIOUS", "ALWAYS", "CHANGED", "EITHER_OR", "YES_NO", "WHO_HAS"]


class Builder:
    def __init__(self, level, seed, today):
        self.level, self.seed, self.today = level, seed, today
        self.rng = random.Random(f"lwlogic/{level}/{seed}")
        self.lessons, self.questions, self.variants, self.notes = [], [], [], []
        self.ents = []

    def ent(self, e):
        if e not in self.ents:
            self.ents.append(e)
        return e

    def past_date(self, lo=40, hi=700):
        return iso_of(day_of(self.today) - self.rng.randint(lo, hi))

    def lesson(self, op, e, a, v=None, when=None, v0=None, d=None, v1=None):
        lid = len(self.lessons) + 1
        self.ent(e)
        l = {"id": lid, "name": f"L{lid}", "op": op, "e": e, "a": a, "told": tick(self.today) + 100 * lid}
        if op == "DATED":
            l.update(v0=v0, d=d, v1=v1)
            slot = "DATED"
        else:
            l["v"] = v
            if when:
                l["when"] = when
            slot = {"ASSERT": "ASSERT", "CHANGE": "CHANGE", "RETRACT": "RETRACT" if v is not None else "CLEAR"}[op]
            if op == "ASSERT" and when:
                slot = "SINCE" if "since" in when else "UNTIL" if "until" in when else "ALWAYS_A"
        paras = T[a][slot]
        if not paras:
            raise ValueError(f"no {slot} phrasing for {a}")
        k = self.rng.randrange(len(paras))
        kw = {"v": v, "v0": v0, "v1": v1, "d": say_date(d) if d else None}
        if when:
            kw["d"] = say_date(when.get("since") or when.get("until") or self.today)
        l["text"], l["para"] = render(paras[k], e, **kw), k
        self.lessons.append(l)
        return l

    def ask(self, qtype, e=None, a=None, role="probe", paras=1, **kw):
        slot = "Q_" + qtype
        lst = T[a][slot]
        if not lst:
            return []
        ks = list(range(len(lst)))
        self.rng.shuffle(ks)
        out = []
        for k in sorted(ks[:paras]):
            fkw = dict(kw)
            if "date" in kw:
                fkw["d"] = say_date(kw["date"])
            text = render(lst[k], e, **fkw)
            if any(q["text"] == text for q in self.questions):
                continue
            q = {"qid": f"Q{len(self.questions) + 1:02d}", "type": qtype, "a": a, "text": text, "para": k, "role": role}
            if e is not None:
                q["e"] = self.ent(e)
            q.update({k2: v2 for k2, v2 in kw.items() if k2 in ("v", "x", "y", "date")})
            self.questions.append(q)
            out.append(q)
        return out

    def gk(self, near_attrs, n=3):
        near = [g for g in GK if set(g["near"]) & set(near_attrs)]
        far = [g for g in GK if not g["near"]]
        pick = self.rng.sample(near, min(len(near), n - 1)) + self.rng.sample(far, 1)
        for g in pick:
            if any(q["text"] == g["text"] for q in self.questions):
                continue
            self.questions.append({"qid": f"Q{len(self.questions) + 1:02d}", "type": "GK", "a": None,
                                   "text": g["text"], "any_of": g["any_of"], "role": "gk"})

    def controls(self, taught_keys, extra_near=None):
        """Near-miss entity, untouched attribute of a taught entity, and general knowledge."""
        attrs = sorted({a for (_, a) in taught_keys})
        taught_ents = sorted({e for (e, _) in taught_keys})
        for (e, a) in sorted(taught_keys)[:2]:
            sim = SIMILAR.get(e)
            if sim and sim not in taught_ents and ENTITIES[sim]["kind"] in SCHEMA[a]["applies"]:
                self.ask("CURRENT", sim, a, role="near_miss")
                break
        for e in taught_ents:
            free = [a for a in ATTR_ORDER if ENTITIES[e]["kind"] in SCHEMA[a]["applies"]
                    and (e, a) not in taught_keys and T[a]["Q_CURRENT"]]
            if free:
                self.ask("CURRENT", e, self.rng.choice(free), role="untouched")
                break
        for x in extra_near or []:
            self.ask("CURRENT", x[0], x[1], role="near_miss")
        self.gk(attrs)

    def variant(self, ids, label=None):
        ids = sorted(set(ids))
        if ids and ids not in [v["deleted"] for v in self.variants]:
            self.variants.append({"deleted": ids, "label": label or ""})

    def build(self):
        sch = {a: SCHEMA[a] for a in sorted({l["a"] for l in self.lessons} | {q["a"] for q in self.questions if q["a"]},
                                            key=ATTR_ORDER.index)}
        ids = [l["id"] for l in self.lessons]
        phases = [{"name": "baseline", "deleted": ids, "note": "before any teaching (the original model)"},
                  {"name": "teach", "deleted": [], "note": "every lesson taught, in order"}]
        for i, v in enumerate(self.variants):
            phases.append({"name": f"del-{chr(65 + i)}", "deleted": v["deleted"], "note": v["label"]})
        for ph in phases:
            ph["store"] = [i for i in ids if i not in ph["deleted"]]
        S = {"format": "lw-test-set/1", "level": self.level, "seed": self.seed, "today": self.today,
             "id": f"L{self.level}-seed{self.seed}" if str(self.level).isdigit() else f"{self.level}-seed{self.seed}",
             "entities": self.ents, "schema": sch, "lessons": self.lessons, "questions": self.questions,
             "phases": phases, "notes": self.notes}
        fill_expected(S)
        return S


def is_delete_phase(ph):
    """A phase that deletes lessons from the full store. Sets of levels 1 to 5 name no kind: every phase after
    baseline and teach is a delete. Level 7 also has checkpoints (the first k lessons taught so far) and pair phases
    (two lessons on a fresh copy): their missing lessons were never taught yet, not deleted. A forms v2 "prefix"
    phase (after-L9: taught up to a like) is a delete: the runner teaches everything, then deletes the later lessons,
    as on the forms v2 branch, so it keeps its delete scores and kept_stable."""
    return ph["name"] not in ("baseline", "teach") and ph.get("kind", "delete") in ("delete", "prefix")


def fill_expected(S):
    sch, ents, by_id = S["schema"], S["entities"], {l["id"]: l for l in S["lessons"]}
    S["expected"] = {}
    for ph in S["phases"]:
        store = [by_id[i] for i in ph["store"]]
        S["expected"][ph["name"]] = {q["qid"]: answer(sch, store, ents, q) for q in S["questions"]}
    teach = S["expected"].get("teach", {})
    full = [by_id[i] for i in next((p["store"] for p in S["phases"] if p["name"] == "teach"), [])]
    for ph in S["phases"]:
        if ph["name"] in ("baseline", "teach") or not is_delete_phase(ph):
            continue
        gone = {(by_id[i]["e"], by_id[i]["a"]) for i in ph["deleted"]}
        gone_a = {a for (_, a) in gone}
        store = [by_id[i] for i in ph["store"]]
        ph["affected"] = [q["qid"] for q in S["questions"] if q["role"] == "probe" and (
            S["expected"][ph["name"]][q["qid"]] != teach.get(q["qid"]) or (q.get("e"), q["a"]) in gone
            or (q["type"] == "WHO_HAS" and q["a"] in gone_a)
            or (q["type"] in TWO_STEP and gone & (chain_keys(sch, full, ents, q) | chain_keys(sch, store, ents, q))))]


def other(rng, dom, *not_these):
    return rng.choice([x for x in dom if x not in not_these])


def gen_L1(b):
    r = b.rng
    a = r.choice(["fav_colour", "eye_colour", "home_city"])
    dom = SCHEMA[a]["domain"]
    v = r.choice(dom)
    l1 = b.lesson("ASSERT", "me", a, v)
    w = other(r, dom, v)
    b.ask("CURRENT", "me", a, paras=3)
    b.ask("YES_NO", "me", a, v=v)
    b.ask("YES_NO", "me", a, v=w)
    b.ask("EITHER_OR", "me", a, x=v, y=w)
    b.ask("EITHER_OR", "me", a, x=w, y=v)
    b.controls({("me", a)})
    b.variant([l1["id"]], "the only lesson deleted: every answer must return to the baseline")
    b.notes.append("One fact. Either-or is asked in both orders, so a model that always picks the first or the "
                   "second option cannot pass both.")


def gen_L2(b):
    r = b.rng
    a = r.choice(["fav_colour", "eye_colour", "home_city"])
    dom = SCHEMA[a]["domain"]
    v1, v2, v3 = r.sample(dom, 3)
    l1 = b.lesson("ASSERT", "me", a, v1)
    l2 = b.lesson("CHANGE", "me", a, v2)
    third = r.choice([None, "back", "new", "retract"])
    last = l2
    if third == "back":
        last = b.lesson("CHANGE", "me", a, v1)
    elif third == "new":
        last = b.lesson("CHANGE", "me", a, v3)
    elif third == "retract":
        last = b.lesson("RETRACT", "me", a, v2)
    b.ask("CURRENT", "me", a, paras=3)
    b.ask("PREVIOUS", "me", a, paras=2)
    for v in [v1, v2] + ([v3] if third == "new" else []):
        b.ask("YES_NO", "me", a, v=v)
    b.ask("EITHER_OR", "me", a, x=v1, y=v2)
    b.ask("EITHER_OR", "me", a, x=v2, y=v1)
    b.ask("ALWAYS", "me", a)
    b.ask("CHANGED", "me", a)
    b.controls({("me", a)})
    b.variant([last["id"]], "the last lesson deleted: what held before it must come back")
    b.variant([l1["id"]], "the first fact deleted: the change still holds; before it, only that it was different")
    b.notes.append("Change of mind. A deleted change must bring the earlier value back, because deleting a lesson "
                   "means it was never taught. A change said alone still means it was something else before.")


def gen_L3(b):
    r = b.rng
    a = r.choice(["eye_colour", "fav_colour", "home_city"])
    dom = SCHEMA[a]["domain"]
    v0, v1, v2 = r.sample(dom, 3)
    d = b.past_date(60, 500)
    pattern = r.choice(["eyes678", "dated_only", "dated_then_change"])
    others = [x for x in ["fav_colour", "eye_colour", "home_city", "likes"] if x != a]
    oth = r.sample(others, 2)
    main = []
    if pattern == "eyes678":
        main = [("ASSERT", v0), ("ASSERT", v1), ("DATED", None)]
    elif pattern == "dated_only":
        main = [("DATED", None)]
    else:
        main = [("DATED", None), ("CHANGE", v2)]
    side = []
    for i, oa in enumerate(oth):
        ov = r.choice(SCHEMA[oa]["domain"])
        when = {"since": b.past_date(30, 400)} if i == 0 and r.random() < 0.6 else None
        side.append((oa, ov, when))
    # interleave the side lessons into the main sequence, keeping the main order
    slots = sorted(r.sample(range(len(main) + len(side)), len(side)))
    seq, mi, si = [], 0, 0
    for pos in range(len(main) + len(side)):
        if si < len(side) and pos == slots[si]:
            seq.append(("side", side[si]))
            si += 1
        else:
            seq.append(("main", main[mi]))
            mi += 1
    made = {}
    for kind, item in seq:
        if kind == "main":
            op, v = item
            if op == "DATED":
                made["DATED"] = b.lesson("DATED", "me", a, v0=v0, d=d, v1=v1)
            else:
                made[f"{op}-{v}"] = b.lesson(op, "me", a, v)
        else:
            oa, ov, when = item
            made[f"side-{oa}"] = b.lesson("ASSERT", "me", oa, ov, when=when)
    b.ask("CURRENT", "me", a, paras=2)
    b.ask("BEFORE", "me", a, date=d)
    b.ask("AT", "me", a, date=iso_of(day_of(d) - r.randint(20, 200)))
    after = day_of(d) + r.randint(5, 30)
    if after < day_of(b.today) - 1:
        b.ask("AT", "me", a, date=iso_of(after))
    b.ask("PREVIOUS", "me", a)
    b.ask("ALWAYS", "me", a)
    b.ask("CHANGED", "me", a)
    b.ask("EITHER_OR", "me", a, x=v0, y=v1)
    b.ask("YES_NO", "me", a, v=v0)
    b.ask("YES_NO", "me", a, v=v1)
    for oa, ov, when in side:
        b.ask("CURRENT", "me", oa)
        b.ask("YES_NO", "me", oa, v=ov)
        if not SCHEMA[oa]["multi"]:
            b.ask("YES_NO", "me", oa, v=other(r, SCHEMA[oa]["domain"], ov))
        if when:
            b.ask("AT", "me", oa, date=iso_of(day_of(when["since"]) - r.randint(10, 100)))
    b.controls({("me", a)} | {("me", oa) for oa, _, _ in side})
    dl = made["DATED"]
    if pattern == "eyes678":
        b.variant([made[f"ASSERT-{v1}"]["id"]], "the plain fact just before the dated one deleted (#678's fact 4): "
                                                 "the dated fact must keep all its answers")
        b.variant([dl["id"]], "the dated fact deleted (#678's fact 5): nothing is known before today any more")
        b.variant([made[f"ASSERT-{v0}"]["id"], made[f"ASSERT-{v1}"]["id"]], "both plain facts deleted")
    elif pattern == "dated_only":
        b.variant([dl["id"]], "the dated fact deleted")
        b.variant([made[f"side-{oth[0]}"]["id"]], "an unrelated fact deleted: the dated answers must not move")
    else:
        b.variant([made[f"CHANGE-{v2}"]["id"]], "the change after the dated fact deleted: its current value returns")
        b.variant([dl["id"]], "the dated fact deleted: the change stays; before it, only that it was different")
    b.notes.append(f"Dated history, pattern '{pattern}'. 'Until D' ends at the start of D; 'since D' starts there. "
                   "Present-tense facts hold from the moment they are told, so they say nothing about past dates. "
                   "No question asks about the day D itself: people read 'until D' both ways.")


def gen_L4(b):
    r = b.rng
    pair = list(r.choice([("my cat", "my dog"), ("my sister", "my brother")]))
    r.shuffle(pair)
    p1, p2 = pair
    near = "my rabbit" if ENTITIES[p1]["kind"] == "animal" else "my cousin"
    n1, n2 = r.sample(SCHEMA["name"]["domain"], 2)
    c1 = r.choice(SCHEMA["eye_colour"]["domain"])
    c2 = c1 if r.random() < 0.5 else other(r, SCHEMA["eye_colour"]["domain"], c1)
    f1, f2, f3, f4 = r.sample(SCHEMA["likes"]["domain"], 4)
    fc = r.choice(SCHEMA["fav_colour"]["domain"])
    chains = [
        [("ASSERT", p1, "name", n1)], [("ASSERT", p2, "name", n2)],
        [("ASSERT", p1, "eye_colour", c1)], [("ASSERT", p2, "eye_colour", c2)],
        [("ASSERT", "me", "likes", f1), ("ASSERT", "me", "likes", f2), ("ASSERT", "me", "likes", f3),
         ("RETRACT", "me", "likes", f2)],
        [("ASSERT", p1, "likes", f2)],
        [("ASSERT", "me", "fav_colour", fc), ("RETRACT", "me", "fav_colour", fc)],
    ]
    made = {}
    while any(chains):                       # random interleave that keeps each chain's own order
        c = r.choice([c for c in chains if c])
        op, e, a, v = c.pop(0)
        made[(op, e, a, v)] = b.lesson(op, e, a, v)
    b.ask("CURRENT", p1, "name")
    b.ask("CURRENT", p2, "name")
    b.ask("YES_NO", p1, "name", v=n2)
    b.ask("EITHER_OR", p2, "name", x=n1, y=n2)
    b.ask("CURRENT", p1, "eye_colour")
    b.ask("CURRENT", p2, "eye_colour")
    b.ask("WHO_HAS", None, "eye_colour", v=c1)
    b.ask("CURRENT", "me", "likes", paras=2)
    b.ask("YES_NO", "me", "likes", v=f1)
    b.ask("YES_NO", "me", "likes", v=f2)
    b.ask("YES_NO", "me", "likes", v=f4)
    b.ask("WHO_HAS", None, "likes", v=f2)
    b.ask("CURRENT", "me", "fav_colour")
    b.ask("YES_NO", "me", "fav_colour", v=fc)
    b.ask("PREVIOUS", "me", "fav_colour")
    b.controls({(p1, "name"), (p2, "name"), (p1, "eye_colour"), (p2, "eye_colour"), ("me", "likes"),
                (p1, "likes"), ("me", "fav_colour")}, extra_near=[(near, "name")])
    b.variant([made[("RETRACT", "me", "likes", f2)]["id"]], "a retraction deleted: the retracted food is liked again")
    b.variant([made[("ASSERT", p1, "name", n1)]["id"]], f"{p1}'s name deleted: {p2}'s name must stay")
    b.variant([made[("RETRACT", "me", "fav_colour", fc)]["id"], made[("ASSERT", p2, "eye_colour", c2)]["id"]],
              "a retraction and an unrelated fact deleted together")
    b.notes.append("Several entities with similar names, a multi-valued attribute (likes), retractions and reverse "
                   "lookups (who has / who likes).")


def gen_L5(b):
    r = b.rng
    pool = ["me", "my sister", "my brother", "my cat", "my dog", "my car", "the boat"]
    ents = ["me"] + r.sample(pool[1:], r.randint(2, 4))
    keys = [(e, a) for e in ents for a in ATTR_ORDER if ENTITIES[e]["kind"] in SCHEMA[a]["applies"]
            and not (e == "me" and a == "name")]
    keys = r.sample(keys, min(len(keys), r.randint(4, 6)))
    n = r.randint(8, 20)
    first = r.sample(keys, len(keys))                # every chosen pair gets at least one lesson
    for i in range(n):
        e, a = first[i] if i < len(first) else r.choice(keys)
        A, dom = SCHEMA[a], SCHEMA[a]["domain"]
        x = r.random()
        if x < 0.40:
            when = None
            if r.random() < 0.25:
                w = r.choice(["since", "until", "always"])
                when = {"always": True} if w == "always" else {w: b.past_date(30, 600)}
            b.lesson("ASSERT", e, a, r.choice(dom), when=when)
        elif x < 0.60:
            # "now v" claims a change, so never "now v" while v already holds
            cur = state_at(SCHEMA, b.lessons, e, a, NOW)
            b.lesson("CHANGE", e, a, r.choice([v for v in dom if cur[v] != YES] or dom))
        elif x < 0.78:
            # "not v anymore" claims v held, so never retract a value already ruled out, and on a single-valued
            # attribute only the value that holds (if one does)
            cur = state_at(SCHEMA, b.lessons, e, a, NOW)
            yes = [v for v in dom if cur[v] == YES]
            open_ = [v for v in dom if cur[v] == UNK]
            pool = yes if yes and (not A["multi"] or r.random() < 0.7) else yes + open_
            if T[a]["CLEAR"] and r.random() < 0.2 and any(cur[v] != NO for v in dom):
                b.lesson("RETRACT", e, a, None)
            elif pool:
                b.lesson("RETRACT", e, a, r.choice(pool))
            else:
                b.lesson("ASSERT", e, a, r.choice(dom))
        elif T[a]["DATED"]:
            v0, v1 = r.sample(dom, 2)
            b.lesson("DATED", e, a, v0=v0, d=b.past_date(30, 600), v1=v1)
        else:
            b.lesson("ASSERT", e, a, r.choice(dom))
    touched = []
    for l in b.lessons:
        if (l["e"], l["a"]) not in touched:
            touched.append((l["e"], l["a"]))
    for (e, a) in touched:
        A, dom = SCHEMA[a], SCHEMA[a]["domain"]
        st = state_at(SCHEMA, b.lessons, e, a, NOW)
        yes = [v for v in dom if st[v] == YES]
        b.ask("CURRENT", e, a)
        b.ask("YES_NO", e, a, v=yes[0] if yes and r.random() < 0.5 else r.choice(dom))
        if not A["multi"]:
            x = yes[0] if yes else r.choice(dom)
            b.ask("EITHER_OR", e, a, x=x, y=other(r, dom, x))
            if sum(1 for l in b.lessons if (l["e"], l["a"]) == (e, a)) > 1:
                b.ask("PREVIOUS", e, a)
                b.ask("ALWAYS", e, a)
                b.ask("CHANGED", e, a)
        dates = sorted({l["d"] for l in b.lessons if (l["e"], l["a"]) == (e, a) and l["op"] == "DATED"} |
                       {v for l in b.lessons if (l["e"], l["a"]) == (e, a) for k, v in (l.get("when") or {}).items()
                        if k in ("since", "until")})
        if dates:
            d = r.choice(dates)
            b.ask("BEFORE", e, a, date=d)
            b.ask("AT", e, a, date=iso_of(day_of(d) + r.randint(1, 20)))
    for a in sorted({a for (_, a) in touched}, key=ATTR_ORDER.index):
        if T[a]["Q_WHO_HAS"] and r.random() < 0.6:
            b.ask("WHO_HAS", None, a, v=r.choice(SCHEMA[a]["domain"]))
    b.questions = b.questions[:45]
    for i, q in enumerate(b.questions):
        q["qid"] = f"Q{i + 1:02d}"
    b.controls(set(touched))
    for _ in range(2):
        b.variant([l["id"] for l in r.sample(b.lessons, r.randint(1, 3))], "random deletion")
    b.notes.append(f"Random sequence of {n} lessons over {len(touched)} entity-attribute pairs.")


def gen_eyes(b, facts_path=None):
    """The #678 case: facts 1, 3, 4, 5 of the original #678 trial, and its colour and eye probe questions.

    `facts_path` (optional): that trial's facts.json; when given and present, its texts and times are used. The
    built-in texts and times are the same, so the set is identical with or without it."""
    texts = {1: "I like blue.", 3: "My eyes are brown.", 4: "My eyes are green.",
             5: "My eyes were brown until 11 October 2026; now they are green."}
    told = {1: 52897, 3: 53311, 4: 53517, 5: 53741}  # seconds after midnight UTC, 11 Oct 2026 (facts.json "at")
    if facts_path and os.path.exists(facts_path):
        for f in json.load(open(facts_path))["facts"]:
            if f["id"] in texts:
                t = dt.datetime.fromtimestamp(f["at"] / 1000, dt.timezone.utc)
                told[f["id"]] = t.hour * 3600 + t.minute * 60 + t.second
                texts[f["id"]] = f["text"]
    base = tick("2026-10-11")
    spec = {1: dict(op="ASSERT", e="me", a="fav_colour", v="blue"),
            3: dict(op="ASSERT", e="me", a="eye_colour", v="brown"),
            4: dict(op="ASSERT", e="me", a="eye_colour", v="green"),
            5: dict(op="DATED", e="me", a="eye_colour", v0="brown", d="2026-10-11", v1="green")}
    for i in (1, 3, 4, 5):
        b.lessons.append(dict(id=i, name=f"F{i}", told=base + told[i], text=texts[i], para=None, **spec[i]))
    b.ents = ["me", "my sister"]
    Q = [("E1", "CURRENT", "eye_colour", "What colour are my eyes?", {}, "probe"),
         ("E2", "CURRENT", "eye_colour", "What is my eye colour?", {}, "probe"),
         ("E3", "EITHER_OR", "eye_colour", "Are my eyes brown or green?", {"x": "brown", "y": "green"}, "probe"),
         ("E4", "BEFORE", "eye_colour", "What colour were my eyes before 11 October 2026?", {"date": "2026-10-11"}, "probe"),
         ("E5", "ALWAYS", "eye_colour", "Have my eyes always been the same colour?", {}, "probe"),
         ("N3", "CURRENT", "eye_colour", "What colour are my sister's eyes?", {"e": "my sister"}, "near_miss"),
         ("P1", "CURRENT", "fav_colour", "What colour do I like?", {}, "probe"),
         ("P2", "CURRENT", "fav_colour", "Which colour is my favourite?", {}, "probe"),
         ("P3", "CURRENT", "fav_colour", "If you were buying me a shirt, what colour should it be?", {"indirect": True},
          "probe"),
         ("P4", "EITHER_OR", "fav_colour", "Do I like blue or green?", {"x": "blue", "y": "green"}, "probe"),
         ("P5", "YES_NO", "fav_colour", "Do I still like blue?", {"v": "blue"}, "probe"),
         ("N1", "CURRENT", "fav_colour", "What colour does my sister like?", {"e": "my sister"}, "near_miss")]
    for qid, t, a, text, kw, role in Q:
        q = {"qid": qid, "type": t, "e": "me", "a": a, "text": text, "para": None, "role": role}
        q.update(kw)
        b.questions.append(q)
    for qid, text, any_of in [("N2", "What colour is the sky on a clear day?", [["blue"]]),
                              ("C1", "What is the capital of France?", [["paris"]]),
                              ("C2", "Name three primary colours.", [["red", "yellow", "blue"], ["red", "green", "blue"]])]:
        b.questions.append({"qid": qid, "type": "GK", "a": None, "text": text, "any_of": any_of, "role": "gk"})
    b.variants = [{"deleted": [3], "label": "rm-3"}, {"deleted": [4], "label": "rm-4"},
                  {"deleted": [5], "label": "rm-5"}, {"deleted": [3, 4, 5], "label": "rm-345"},
                  {"deleted": [1, 3, 4, 5], "label": "rm-all"}]
    b.notes.append("'I like blue.' is read as a favourite colour (single-valued), as TensorFold 1.0.8 does. "
                   "Phase names match the probe file stages in tf-conflict (rm-3 = probes-rm-3*.jsonl; teach = "
                   "probes-7-restart-noslide*.jsonl; baseline = probes-rm-R*.jsonl).")
    S = b.build()
    for ph in S["phases"][2:]:
        ph["name"] = ph["note"]
    S["expected"] = {}
    fill_expected(S)
    S["id"] = "eyes-678"
    return S


# ================================================================================ level 7: fill to capacity
L7_FACTS = 64                               # the example set's size; any 1 to len(l7_keys()) works
L7_EVERY = 5                                # a checkpoint after the first lesson and after every 5th
L7_FIRST = 10                               # the first 10 facts are also scored on their own
L7_DELETES = (1, 3, 6)                      # facts deleted one at a time; asked after the 1st, 3rd and 6th
L7_REPORT_AT = (1, 5, 10, 20, 40, 64, 100)  # the points of the retention curve the report names
# A yes/no about an attribute that holds several values at once (likes) asks the taught value. Another value stays
# "don't know" by the rules whatever is taught, so it could not tell a fact held from a fact lost (review, 13 Oct).
L7_YN_MULTI_TAUGHT = True


def l7_keys():
    """Every (entity, attribute) pair a level 7 fact can use, in a fixed order. Each fact gets a pair of its own, so
    no fact replaces another by the rules, and every pair has a plain wording, a CURRENT and a yes/no question."""
    return [(e, a) for e in ENTITIES for a in ATTR_ORDER
            if ENTITIES[e]["kind"] in SCHEMA[a]["applies"] and T[a]["ASSERT"] and T[a]["Q_CURRENT"] and T[a]["Q_YES_NO"]]


def l7_checkpoints(n, every=L7_EVERY):
    """After how many lessons the model is asked: the first, every `every`th, and the last."""
    return sorted({1, n} | set(range(every, n + 1, every)))


def gen_L7(b, n=L7_FACTS):
    """Level 7: n plain facts, each on an (entity, attribute) pair of its own, taught one at a time. The model is
    asked after the first and after every fifth lesson (checkpoint phases ck-NNN, the last one is `teach`): one
    CURRENT wording and one yes/no for every fact taught so far, and the controls. Two pairs are also taught alone on
    fresh copies, A then B: a similar pair (one attribute, two entities the rule book calls similar, two values) and a
    far pair (other attributes, other kinds of entity). Six facts spread over the order are deleted one at a time;
    del-1, del-3 and del-6 hold the store after the first, third and sixth delete."""
    r = b.rng
    keys = l7_keys()
    if not 1 <= n <= len(keys):
        raise ValueError(f"level 7 takes 1 to {len(keys)} facts, not {n}")
    pool = list(keys)
    r.shuffle(pool)
    pair_keys = {}
    if n >= 4:
        sim = [(e, a) for (e, a) in pool if SIMILAR.get(e) and (SIMILAR[e], a) in keys and SIMILAR.get(SIMILAR[e]) == e]
        e1, a1 = sim[0]
        pair_keys["similar"] = ((e1, a1), (SIMILAR[e1], a1))
        used = set(pair_keys["similar"])
        fa = next(k for k in pool if k not in used)
        fb = next(k for k in pool if k not in used and k != fa and k[1] != fa[1]
                  and ENTITIES[k[0]]["kind"] != ENTITIES[fa[0]]["kind"])
        pair_keys["far"] = (fa, fb)
    fixed = [k for p in pair_keys.values() for k in p]
    chosen = fixed + [k for k in pool if k not in fixed][:n - len(fixed)]
    r.shuffle(chosen)
    value = {}
    for (e, a) in chosen:
        value[(e, a)] = r.choice(SCHEMA[a]["domain"])
    if "similar" in pair_keys:  # the similar pair holds two values, so one leaking into the other can be seen
        ka, kb = pair_keys["similar"]
        value[kb] = other(r, SCHEMA[kb[1]]["domain"], value[ka])
    lesson_of = {}
    for (e, a) in chosen:
        lesson_of[(e, a)] = b.lesson("ASSERT", e, a, value[(e, a)])
    fact_q = {}
    for (e, a) in chosen:
        l, dom = lesson_of[(e, a)], SCHEMA[a]["domain"]
        cur = b.ask("CURRENT", e, a)
        vq = value[(e, a)] if r.random() < 0.5 else other(r, dom, value[(e, a)])
        if SCHEMA[a]["multi"] and L7_YN_MULTI_TAUGHT:  # the random numbers are drawn as before: the rest is unchanged
            vq = value[(e, a)]
        yn = b.ask("YES_NO", e, a, v=vq)
        if not cur or not yn:
            raise ValueError(f"level 7: no question for {e} {a}")
        for q in cur + yn:
            q["fact"] = l["id"]
        fact_q[str(l["id"])] = {"current": cur[0]["qid"], "yes_no": yn[0]["qid"]}
    taught = set(chosen)
    near = []
    for (e, a) in chosen:                    # near misses: the same attribute of the similar entity, never taught
        s = SIMILAR.get(e)
        if len(near) < 2 and s and (s, a) in keys and (s, a) not in taught and (s, a) not in near:
            near.append((s, a))
    controls = []
    for (e, a) in near:
        controls += b.ask("CURRENT", e, a, role="near_miss")
    ents_taught = {e for e, _ in chosen}
    untouched = [(e, a) for (e, a) in pool if e in ents_taught and (e, a) not in taught and (e, a) not in near]
    if untouched:
        controls += b.ask("CURRENT", untouched[0][0], untouched[0][1], role="untouched")
    nq = len(b.questions)
    b.gk(sorted({a for _, a in chosen}, key=ATTR_ORDER.index))
    controls += b.questions[nq:]
    ids = [l["id"] for l in b.lessons]
    order = []
    m = min(max(L7_DELETES), n)
    for i in range(m):                       # one fact from each sixth of the teaching order, deleted in random order
        order.append(r.choice(ids[i * n // m:(i + 1) * n // m]))
    r.shuffle(order)
    b.notes.append(f"Level 7, fill to capacity: {n} plain facts, each on an entity-attribute pair of its own, taught "
                   f"one at a time. Checkpoint phases hold the first k lessons; the model is asked after the first "
                   f"lesson and after every {L7_EVERY}th, about the facts taught so far and the controls. Pair phases "
                   "are fresh copies taught two lessons, A then B. Delete phases take facts out one at a time.")
    S = b.build()

    def phase(name, store, kind, note, **kw):
        p = {"name": name, "deleted": [i for i in ids if i not in store], "note": note, "store": list(store),
             "kind": kind}
        p.update(kw)
        return p

    cks = l7_checkpoints(n)
    phases = [phase("baseline", [], "baseline", "before any teaching (the original model)")]
    phases += [phase(f"ck-{k:03d}", ids[:k], "checkpoint", "the first lesson taught" if k == 1 else
                     f"the first {k} lessons taught, one at a time", after=k) for k in cks[:-1]]
    phases.append(phase("teach", ids, "teach", f"all {n} lessons taught, one at a time (the last checkpoint)", after=n))
    pairs = {}
    for kind, (ka, kb) in pair_keys.items():
        la, lb = lesson_of[ka]["id"], lesson_of[kb]["id"]
        pairs[kind] = {"a": la, "b": lb, "phases": [f"ix-{kind}-A", f"ix-{kind}-AB"],
                       "why": (f"one attribute ({ka[1]}), similar entities ({ka[0]}, {kb[0]}), two values"
                               if kind == "similar" else
                               f"other attributes ({ka[1]}, {kb[1]}) and kinds ({ENTITIES[ka[0]]['kind']}, "
                               f"{ENTITIES[kb[0]]['kind']})")}
        phases.append(phase(f"ix-{kind}-A", [la], "pair", f"a fresh copy taught only L{la} ({kind} pair, A)", pair=kind))
        phases.append(phase(f"ix-{kind}-AB", [la, lb], "pair", f"the same copy, then L{lb} ({kind} pair, A then B)",
                            pair=kind))
    dels = {}
    for c in L7_DELETES:
        if c <= len(order):
            dels[f"del-{c}"] = c
            phases.append(phase(f"del-{c}", [i for i in ids if i not in order[:c]], "delete",
                                "the first fact of the delete order deleted" if c == 1 else
                                f"the first {c} facts of the delete order deleted, one at a time", delete_order=order[:c]))
    S["phases"] = phases
    S["l7"] = {"format": "lw-l7/1", "facts": n, "keys_available": len(keys), "sequence": ids,
               "check_every": L7_EVERY, "checkpoints": [{"after": k, "phase": f"ck-{k:03d}" if k < n else "teach"}
                                                        for k in cks],
               "report_at": [k for k in L7_REPORT_AT if k <= n] + ([n] if n not in L7_REPORT_AT else []),
               "first": min(L7_FIRST, n), "fact_questions": fact_q, "controls": [q["qid"] for q in controls],
               "pairs": pairs, "deletes": {"order": order, "phases": dels}}
    if n != L7_FACTS:
        S["id"] = f"L7-n{n}-seed{b.seed}"
    fill_expected(S)
    return S


def generate(level, seed, today="2026-10-12", facts=None, forms="v1", likes="one", n_facts=None):
    """A test set. forms="v1" is the set as it always was; forms="v2" adds the v2 question forms to it
    (`add_forms_v2`), and likes="restate" then restates the whole set of likes in each "I like X" lesson.
    n_facts: level 7 only, how many facts (default L7_FACTS). Pass it by name."""
    if forms not in FORMS or likes not in ("one", "restate") or (forms == "v1" and likes != "one"):
        raise ValueError(f"forms {forms!r}, likes {likes!r}: forms is v1 or v2; likes='restate' needs forms v2")
    if str(level) in ("6", "7") and forms != "v1":   # v2_problems flags level 6's two-step probes (no lesson on
        raise ValueError(f"forms v2 is not built for level {level} yet")  # that pair); level 7 was never tried
    if str(level) == "6":                            # two-step questions: zengine/levels.py
        return _levels().generate6(seed, today)
    if str(level) == "7":                            # fill to capacity
        return gen_L7(Builder(level, seed, today), L7_FACTS if n_facts is None else n_facts)
    b = Builder(level, seed, "2026-10-11" if level == "eyes" else today)
    if level == "eyes":
        S = gen_eyes(b, facts)
    else:
        {1: gen_L1, 2: gen_L2, 3: gen_L3, 4: gen_L4, 5: gen_L5}[int(level)](b)
        S = b.build()
    return add_forms_v2(S, likes) if forms == "v2" else S


# ================================================================================================ forms v2
# Plan items 7, 9 and 10 of docs/LITERATURE.md, as new question forms added to a finished v1 set (TEST-LOGIC.md
# section 13). Nothing of the v1 set changes: the lessons, questions and phases stay, new questions and phases come
# after them, drawn from a random stream of their own. Every new question has one of the oracle's own types, so the
# rule book, the second implementation and the Lean export answer it unchanged; `form` tells the judge how it is asked.
#   7  wrong yes:  yes/no about at least 3 untaught values of every single-valued taught key (one of them a
#                  neighbour's taught value when the set has one), plus "True or false: ...", "Finish this sentence:
#                  ..." (no value named) and "First say what ..., then answer yes or no: ..." forms
#   9  who has:    the bare reverse question, the reverse as yes/no ("Is it my dog who has grey eyes?") and the
#                  listed form ("Of me, my sister and my cat, who has grey eyes?"), each with its forward question
#   10 likes:      the full list and a yes/no for every like, a phase after each "I like X" from the second on, and
#                  (likes="restate") a variant whose lessons restate the whole set
MAX_PREFIX = 4        # at most this many "after each like" phases per set: each is one more reteach in the runner
MAX_REVERSE = 3       # at most this many reverse values per set, each asked in the three forms


def say_list(xs):
    xs = list(xs)
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


def lesson_values(l):
    """Every value a lesson names: v, the values of a restated set, and a dated lesson's two."""
    return uniq([v for v in (l.get("vs") or []) + [l.get("v"), l.get("v0"), l.get("v1")] if v is not None])


def taught_values(S, e, a):
    """Every value a lesson on (e, a) names, in teaching order."""
    return uniq([v for l in S["lessons"] if (l["e"], l["a"]) == (e, a) for v in lesson_values(l)])


def like_adds(lessons, e, a):
    """The lessons that add a value to multi-valued key (e, a) from now on: "I like X", "since D", "always"."""
    return [l for l in lessons if (l["e"], l["a"]) == (e, a) and l["op"] == "ASSERT" and covers(when_interval(l), NOW)]


def restate_likes(S):
    """likes="restate": every "I like X" on a multi-valued key that already holds something restates the whole set it
    holds then, plus X ("I like pears and olives."), as one lesson. Ids, ticks, deletes and phases stay. Returns how
    many lessons were restated."""
    sch, n = S["schema"], 0
    for i, l in enumerate(S["lessons"]):
        a = l["a"]
        if l["op"] != "ASSERT" or not sch[a]["multi"] or l.get("when") or not T2.get(a, {}).get("ASSERT_SET"):
            continue
        before = S["lessons"][:i]
        st = state_at(sch, before, l["e"], a, NOW)
        order = uniq([v for m in before if (m["e"], m["a"]) == (l["e"], a) for v in lesson_values(m)])
        held = [v for v in order if st.get(v) == YES and v != l["v"]]
        if not held:
            continue
        l["vs"] = held + [l["v"]]
        l["text"], l["para"], l["form"] = render(T2[a]["ASSERT_SET"][0], l["e"], vs=say_list(l["vs"])), 0, "restate"
        n += 1
    return n


def add_forms_v2(S, likes="one"):
    """A v1 test set -> the same set with the v2 question forms added (see the block comment above). The v1 set itself
    is not changed. Raises ValueError for a set that already has forms, or likes="restate" with nothing to restate."""
    if S.get("forms"):
        raise ValueError(f"{S['id']} already has question forms {S['forms']}")
    S = json.loads(json.dumps(S))                    # a copy: the v1 set is never changed
    rng = random.Random(f"lwlogic/forms-v2/{S['id']}")
    if likes == "restate" and not restate_likes(S):
        raise ValueError(f"{S['id']}: no key gets a second like, so there is nothing to restate")
    sch, qs, lessons = S["schema"], S["questions"], S["lessons"]
    by_id = {l["id"]: l for l in lessons}
    stores = {ph["name"]: [by_id[i] for i in ph["store"]] for ph in S["phases"]}
    used = {q["qid"] for q in qs}

    def ent(e):
        if e not in S["entities"]:
            S["entities"].append(e)
        return e

    def add(qtype, e, a, text, role="probe", **kw):
        same = next((q for q in qs if q["text"] == text), None)
        if same:
            return same
        n = len(qs) + 1
        while f"Q{n:02d}" in used:
            n += 1
        used.add(f"Q{n:02d}")
        q = {"qid": f"Q{n:02d}", "type": qtype, "a": a, "text": text, "para": kw.pop("para", None), "role": role}
        if e is not None:
            q["e"] = ent(e)
        q.update(kw)
        qs.append(q)
        return q

    def ask_std(qtype, e, a, **kw):                  # a v1 wording, picked from the v2 stream
        paras = T[a]["Q_" + qtype]
        k = rng.randrange(len(paras))
        return add(qtype, e, a, render(paras[k], e, **kw), para=k, **kw)

    def std(q, qtype, e, a):
        return q["type"] == qtype and not q.get("form") and q["role"] == "probe" and (q.get("e"), q["a"]) == (e, a)

    keys = uniq([(l["e"], l["a"]) for l in lessons])
    lesson_ents = uniq([l["e"] for l in lessons])

    # 7. wrong yes: at least 3 untaught values per single-valued taught key, a neighbour's taught value first
    for (e, a) in keys:
        if sch[a]["multi"] or not T2.get(a, {}).get("TF"):
            continue
        taught = taught_values(S, e, a)
        untaught = [v for v in sch[a]["domain"] if v not in taught]
        nb = {}                                      # value -> the neighbour it was taught about
        for e2 in uniq([x for x in [SIMILAR.get(e)] + lesson_ents if x and x != e]):
            for v in taught_values(S, e2, a):
                if v in untaught:
                    nb.setdefault(v, e2)
        have = [q for q in qs if std(q, "YES_NO", e, a) and q["v"] in untaught]
        chosen = list(nb)[:1] + [q["v"] for q in have]
        rest = [v for v in untaught if v not in chosen]
        rng.shuffle(rest)
        chosen = uniq(chosen + rest[:max(0, 3 - len(uniq(chosen)))])
        for v in chosen:
            tag = {"probe": "wrong_yes", **({"neighbour": nb[v]} if v in nb else {})}
            olds = [q for q in have if q["v"] == v]
            for q in olds:
                q.update(tag)
            if not olds:
                ask_std("YES_NO", e, a, v=v).update(tag)
        last = lessons[max(i for i, l in enumerate(lessons) if (l["e"], l["a"]) == (e, a))]
        pos = last.get("v1") or last.get("v")
        u = chosen[0] if chosen else None
        pairs = [(v, {"probe": "wrong_yes", **({"neighbour": nb[v]} if v in nb else {})} if v == u else {})
                 for v in (u, pos if pos != u else None) if v is not None]
        for v, tag in pairs:
            add("YES_NO", e, a, render(T2[a]["TF"][0], e, v=v), para=0, v=v, form="true_false", **tag)
        add("CURRENT", e, a, FILL_WRAP.format(s=render(T2[a]["FILL"][0], e)), para=0, form="fill_in")
        for v, tag in pairs:
            add("YES_NO", e, a, render(T2[a]["RECALL_YN"][0], e, v=v), para=0, v=v, form="recall_yn", **tag)

    # 10. likes as a set: the full list and a yes/no for every like, and a phase after each like from the second on
    prefix = []
    for (e, a) in keys:
        if not sch[a]["multi"]:
            continue
        if not any(std(q, "CURRENT", e, a) for q in qs):
            ask_std("CURRENT", e, a)
        for v in uniq([v for l in lessons if (l["e"], l["a"]) == (e, a) and l["op"] in ("ASSERT", "CHANGE")
                       for v in lesson_values(l)]):
            if not any(std(q, "YES_NO", e, a) and q["v"] == v for q in qs):
                ask_std("YES_NO", e, a, v=v)
        adds = like_adds(lessons, e, a)
        for k, l in enumerate(adds[1:], 2):
            prefix.append((lessons.index(l), l, e, k))
        if len(adds) > 1 and not any(ph.get("probe") == "likes" for ph in S["phases"]):
            # every like lesson but the last deleted: a restated set keeps the earlier likes, likes taught one at a
            # time lose them. Both variants get this phase, so their answers can be set side by side
            gone = [l["id"] for l in adds[:-1]]                # in teaching order, as the v1 phases list them
            if set(gone) not in [set(ph["deleted"]) for ph in S["phases"]]:
                name = f"del-{chr(65 + sum(ph['name'].startswith('del-') for ph in S['phases']))}"
                S["phases"].append({"name": name, "deleted": gone, "store": [l["id"] for l in lessons
                                                                             if l["id"] not in gone],
                                    "kind": "delete", "probe": "likes",   # a real delete; "probe" says what for
                                    "note": f"every like lesson of {e} but the last deleted: a restated set keeps "
                                            f"the earlier likes"})
                stores[name] = [l for l in lessons if l["id"] not in gone]
    ids = [l["id"] for l in lessons]
    for pos, l, e, k in sorted(prefix, key=lambda x: x[0])[:MAX_PREFIX]:
        deleted = ids[pos + 1:]
        if deleted and set(deleted) not in [set(ph["deleted"]) for ph in S["phases"]]:
            S["phases"].append({"name": f"after-{l['name']}", "deleted": deleted, "store": ids[:pos + 1],
                                "kind": "prefix", "note": f"taught up to {l['name']}, like number {k} of {e}: "
                                                          f"every earlier like must still be there"})
            stores[S["phases"][-1]["name"]] = lessons[:pos + 1]

    # 9. who has, in three forms, each with its forward question
    def holders(a, v, phases):
        return uniq([x for ph in phases for x in who_has(sch, stores[ph], S["entities"], a, v)])

    later = [ph["name"] for ph in S["phases"] if ph["name"] != "baseline"]
    rev = []
    for q in qs:
        if q["type"] == "WHO_HAS" and q["role"] == "probe" and T2.get(q["a"], {}).get("WHO_LISTED"):
            q["form"] = "who_bare"
            rev.append((q["a"], q["v"]))
    for a in uniq([a for (_, a) in keys]):
        if not T2.get(a, {}).get("WHO_LISTED") or not T[a]["Q_WHO_HAS"] or any(x == a for x, _ in rev):
            continue
        held = uniq([v for ph in later for v in sch[a]["domain"] if holders(a, v, [ph])])
        if held:
            rev.append((a, held[0]))
    for a, v in uniq(rev)[:MAX_REVERSE]:
        if not any(q["type"] == "WHO_HAS" and q["a"] == a and q.get("v") == v for q in qs):
            ask_std("WHO_HAS", None, a, v=v)["form"] = "who_bare"
        fits = lambda x: ENTITIES[x]["kind"] in SCHEMA[a]["applies"]
        taught_a = uniq([l["e"] for l in lessons if l["a"] == a])
        among = taught_a + (["me"] if fits("me") and "me" not in taught_a else [])
        for x in [SIMILAR.get(y) for y in among] + list(S["entities"]) + list(ENTITIES):
            if len(among) >= 3:
                break
            if x and x not in among and fits(x):
                among.append(x)
        among = (["me"] if "me" in among else []) + sorted([x for x in among if x != "me"], key=lambda x: rng.random())
        add("WHO_HAS", None, a, render(T2[a]["WHO_LISTED"][0], None, v=v, list=say_list(among)), para=0, v=v,
            form="who_listed", among=among)
        for x in among:
            ent(x)
        yes = holders(a, v, ["teach"]) or holders(a, v, later)
        nonh = [x for x in among if x not in holders(a, v, later)]
        nonh = [x for x in nonh if x in taught_a] + [x for x in nonh if x not in taught_a]
        for x in yes[:1] + nonh[:1]:
            add("YES_NO", x, a, render(T2[a]["WHO_YN"][0], x, v=v, obj="me" if x == "me" else x), para=0, v=v,
                form="who_yn")
        for x in uniq(holders(a, v, later) + yes[:1] + nonh[:1]):
            # the forward question, unless the set asks it already (a near miss counts). One about an entity no
            # lesson is about on this attribute is a near miss, not a probe: the recall score is for taught pairs
            if not any(q["type"] == "CURRENT" and not q.get("form") and (q.get("e"), q["a"]) == (x, a) for q in qs):
                ask_std("CURRENT", x, a, role="probe" if x in taught_a else "near_miss")

    S["forms"] = "v2"
    S["id"] += "-v2" + ("-restate" if likes == "restate" else "")
    S["notes"].append("Question forms v2 (TEST-LOGIC.md section 13): wrong-yes probes with true/false, fill-in and "
                      "recall-then-answer forms; who-has asked bare, as yes/no and over a list; likes asked as a full "
                      "list and per item, with a phase after each like" + (", every like restating the whole set"
                                                                          if likes == "restate" else "") + ".")
    S["expected"] = {}
    fill_expected(S)
    for q in qs:                                     # the listed form names everyone who can hold the value, so
        if q.get("form") == "who_listed":            # its answer is the bare question's in every phase
            for ph in S["phases"]:
                got = S["expected"][ph["name"]][q["qid"]]["yes"]
                if not set(got) <= set(q["among"]):
                    raise AssertionError(f"{S['id']} {q['qid']}: {got} not all in the list {q['among']}")
    return S


def _levels():
    """zengine/levels.py (level 6), imported when first needed: it imports this file itself."""
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    import levels
    return levels


# ================================================================================================= judge
APOS = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "—": ", ", "–": ", "})
REFUSAL = re.compile(
    r"\b(?:i\s+(?:do\s+not|don't|dont|did\s+not|didn't)\s+(?:actually\s+|really\s+|personally\s+)?"
    r"(?:know|have\s+(?:any|access|that|a\s+way|information|personal|a\s+physical|enough|a\s+record)|remember|recall|see)"
    r"|i\s+(?:can\s*not|can't|cannot|am\s+unable\s+to|'m\s+unable\s+to|am\s+not\s+able\s+to|have\s+no\s+way\s+to)\s+"
    r"(?:actually\s+|really\s+)?(?:know|see|tell|determine|say|access|answer|recall|be\s+sure|confirm|remember)"
    r"|you\s+(?:have\s+not|haven't|never|didn't|did\s+not)\s+(?:told|tell|mentioned|mention|said|say|shared|share)"
    r"|i\s+have\s+no\s+(?:information|way|record|data|access|knowledge|memory)|no\s+information|not\s+sure"
    r"|i'm\s+not\s+aware|i\s+am\s+not\s+aware|only\s+you\s+can|i'm\s+an\s+ai|i\s+am\s+an\s+ai|as\s+an\s+ai"
    r"|can't\s+say|cannot\s+say|no\s+way\s+(?:of|to)\s+know|isn't\s+something\s+i\s+know)", re.I)
ABSENT = re.compile(r"\b(?:no\s+longer|not\s+any\s?more|any\s?more|don't\s+have\s+(?:a|any)|do\s+not\s+have\s+(?:a|any)"
                    r"|you\s+have\s+no|no\s+favou?rite)\b", re.I)
MEMORY_CUE = re.compile(r"\b(?:you\s+(?:said|told|mentioned|shared)|you've\s+(?:said|told|mentioned)|earlier|based\s+on"
                        r"|according\s+to|from\s+what\s+you)\b", re.I)
NEITHER = re.compile(r"\b(?:neither|none\s+of\s+(?:them|those|the\s+two)|not\s+either)\b", re.I)
NOBODY = re.compile(r"\b(?:no\s?one|nobody|none\s+of)\b", re.I)
NO_START = re.compile(r"^(?:no\b|nope\b|not\b|you\s+don't|you\s+do\s+not|you\s+no\s+longer|they\s+(?:are\s+not|aren't|were\s+not|weren't)"
                      r"|it\s+(?:is\s+not|isn't|was\s+not|wasn't|does\s+not|doesn't)|it's\s+not|that's\s+not|that\s+is\s+not"
                      r"|(?:he|she)\s+(?:does\s+not|doesn't|is\s+not|isn't))", re.I)
YES_START = re.compile(r"^(?:yes\b|yeah\b|yep\b|correct\b|that's\s+right|that\s+is\s+right|indeed\b|absolutely\b|definitely\b"
                       r"|of\s+course\b|you\s+do\b|you\s+are\b|they\s+are\b|they\s+were\b|it\s+is\b|it\s+does\b|it\s+was\b"
                       r"|(?:he|she)\s+(?:does|is)\b)", re.I)
HEDGE = re.compile(r"\b(?:it\s+depends|depends\s+on|hard\s+to\s+say|difficult\s+to\s+say|impossible\s+to\s+say)\b", re.I)
# a guess about this user ("I'm not sure, maybe green", "Probably yes"). "I think" is left out on purpose: models
# often say "I think" in front of a confident, correct recall.
GUESS = re.compile(r"\b(?:maybe|perhaps|possibly|possible|probably|presumably|likely|might\s+(?:be|have)"
                   r"|could\s+be|my\s+guess|i\s+(?:would\s+|'d\s+)?guess|i\s+suppose|i\s+assume)\b", re.I)
LEAN = re.compile(r"\b(?:yes|yeah|no|nope)\b|\b(?:maybe|perhaps|possibly|probably|likely)\s+(?:not|so)\b", re.I)
# a yes/no answer that opens with how likely it is: "It is very unlikely that your favourite colour has always been the
# same.", "It is very likely that it has!", "Highly unlikely.", "Not likely.". A guess that leans no or yes, as
# "Probably not." and "Probably yes." are. Before 13 Oct the opening "It is" read as a plain yes (RESULTS.md, split
# scores point 9). Switched off in the self-test's negative control.
LIKELY = re.compile(r"^\W*(?:(?:well|honestly|frankly|realistically|statistically|actually)\W+)?"
                    r"(?:(?:it|that|this)(?:'s|\s+is|\s+seems|\s+sounds|\s+looks)\s+|i\s+(?:think|believe|find)\s+it(?:'s|\s+is)\s+)?"
                    r"(?:(?:very|highly|quite|extremely|most|rather|fairly|pretty|really|so|incredibly)\s+)*"
                    r"(?:(?P<no>unlikely|improbable|doubtful|not\s+(?:very\s+|that\s+|so\s+)?(?:likely|probable)"
                    r"|(?:likely|probable)\W+(?:not|no)\b)|(?P<yes>likely|probable)(?!\W+(?:not|no)\b))\b", re.I)
# the model talking about itself instead of answering ("My name is Nemotron.")
SELF_TALK = re.compile(r"^(?:(?:hi|hello|hey)\b[\s,!.]*)?(?:my\s+name\s+is|i\s+am\s+called|i'm\s+called|i\s+am\s+nemotron"
                       r"|i'm\s+nemotron|i\s+am\s+(?:a|an)\s+(?:large\s+)?(?:language\s+model|ai\s+assistant|assistant))\b",
                       re.I)
NEG_PRE = re.compile(r"(?:\bnot|n't|\bno\s+longer|\bnever|\bstopped|\bdislikes?|\bnor)\s+(?:[\w']+\s+){0,3}$", re.I)
NEG_POST = re.compile(r"^\s*(?:any\s?more|no\s+longer|(?:is|are|was|were)\s+(?:not|no\s+longer)|isn't|aren't)\b", re.I)
CLAUSE = re.compile(r"[,;:.!?()\[\]\"]|\bbut\b|\band\b|\bwhile\b|\bhowever\b|\bwhereas\b", re.I)
PAST_W = {"were", "was", "used", "until", "previously", "originally", "formerly", "had", "lived", "liked",
          "belonged", "before", "earlier", "once", "owned"}
PRES_W = {"now", "currently", "today", "presently", "are", "is", "am", "lives", "live", "like", "likes",
          "belongs", "owns", "days", "moment"}
# a value as the subject of a general statement about it ("Purple is often associated with royalty"), and words
# that turn a statement into one about the user or the model's own pick ("Blue is your colour", "I'd say blue")
DESCRIBE = re.compile(r"(?:'s|\s+(?:is|are|was|were|has|can|often|usually|tends?|evokes?|represents?"
                      r"|symboli[sz]es?|conveys?))\b", re.I)
PERSON = re.compile(r"\b(?:you|yours?|yourself|i|me|my)\b", re.I)
# a yes or no the model says about itself, not about the user ("No, I am a language model developed by researchers
# from NVIDIA.", "No, I don't have eyes."): a refusal, so "doesn't know", never a "no"
SELF_REPLY = re.compile(
    r"^(?:yes|yeah|yep|no|nope)\b[\s,.!:;-]*(?:i\s+am\s+nemotron|i'm\s+nemotron|as\s+an?\s+(?:ai|language\s+model|assistant)"
    r"|(?:i\s+am|i'm)\s+(?:just\s+|only\s+|simply\s+|actually\s+)?(?:a|an)\s+(?:large\s+)?(?:language\s+model|ai\b"
    r"|artificial|assistant|chatbot|computer\s+program|machine|virtual)"
    r"|i\s+(?:do\s+not|don't)\s+(?:actually\s+|really\s+)?have\s+(?:eyes|a\s+body|a\s+physical|physical|a\s+face|a\s+home"
    r"|a\s+favou?rite|favou?rite|personal\s+preferences|preferences|tastes?|a\s+pet|pets|a\s+family|family))", re.I)
# a reverse question ("Who has grey eyes?") answered about people in general, not about the user's own people and
# pets ("Here are some famous people with grey eyes: ...", "Olives are enjoyed by people all over the world.")
GENERAL = re.compile(r"\b(?:people|persons?|population|celebrit(?:y|ies)|public\s+figures?|famous|actors?|actress(?:es)?"
                     r"|musicians?|athletes?|everyone|everybody|anyone|anybody|cultures?|countries|regions?|ancestry"
                     r"|descent|individuals?|fans?|lovers?|babies|children|adults)\b", re.I)
COND = re.compile(r"\b(?:if|whether|unless|in\s+case)\b", re.I)
IF_FIRST = re.compile(r"^\W*(?:if|unless|whether|in\s+case)\b(?!\s+i\s+(?:remember|recall|am\s+not\s+mistaken|'m\s+not"
                      r"\s+mistaken|understood))", re.I)
# a sentence that asks the user something back ("Do you like olives?", "Are you team olive?")
ASK_BACK = re.compile(r"^\W*(?:do|does|did|are|is|was|were|have|has|would|could|can|will|shall|should|may|might|who|what"
                      r"|which|where|when|why|how)\b", re.I)
# a date of telling, not of the value ("you told me on 11 October", "May 9, 2024: You mentioned they were amber"), or
# of the model itself ("as of my knowledge cutoff in 2023")
TELL = re.compile(r"\b(?:told|tell|said|say|mentioned|mention|shared|share|provided|gave|wrote|asked|ask|taught|informed"
                  r"|noted|stated|explained|updated|update|knowledge|cutoff|training|trained)\b", re.I)
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
          "november", "december"]
MON = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?" \
      r"|nov(?:ember)?|dec(?:ember)?)\.?"
YR = r"(?:19|20)\d\d"
DATE_RE = re.compile(rf"\b(?:(?P<d1>\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?(?P<m1>{MON})\s*,?\s*(?P<y1>{YR})"
                     rf"|(?P<m2>{MON})\s+(?P<d2>\d{{1,2}})(?:st|nd|rd|th)?\s*,?\s*(?P<y2>{YR})"
                     rf"|(?P<y3>{YR})-(?P<o3>\d\d)-(?P<d3>\d\d)"
                     rf"|(?P<m4>{MON})\s+(?:of\s+)?(?P<y4>{YR})"
                     rf"|(?P<y5>{YR}))\b", re.I)
# the word that ties a date to a value: "until D", "since D", "on D" ...
REL_RE = re.compile(r"\b(?P<rel>until|till|til|up\s+(?:to|until)|through|thru|before|prior\s+to|since|from"
                    r"|starting(?:\s+(?:on|in|from))?|beginning(?:\s+(?:on|in))?|as\s+of|on|in|during|to|by|after|around)"
                    r"\s+(?:the\s+)?$", re.I)
REL_KIND = {"until": "until", "till": "until", "til": "until", "up to": "until", "up until": "until", "through": "until",
            "thru": "until", "to": "until", "before": "before", "prior to": "before", "since": "since", "from": "since",
            "starting": "since", "starting on": "since", "starting in": "since", "starting from": "since",
            "beginning": "since", "beginning on": "since", "beginning in": "since", "as of": "on", "on": "on",
            "in": "on", "during": "on"}
CHANGE_W = re.compile(r"\b(?:changed?|changing|switched|turned|became|moved)\b", re.I)


def clean(t):
    return re.sub(r"[*_`#>|]+", " ", (t or "").translate(APOS))


def sentences(t):
    return [re.sub(r"\s+", " ", p).strip() for p in re.split(r"(?<=[.!?])\s+|\n+", clean(t)) if p.strip()]


def vocab(sch, a):
    vals = list(sch[a]["domain"]) + [x for x in sch[a].get("extra", []) if x not in sch[a]["domain"]]
    return [(v, SYN.get(v, [v])) for v in vals]


def mentions(text, sch, a):
    out = []
    for val, pats in vocab(sch, a):
        for pat in pats:
            for m in re.finditer(r"(?<![\w-])" + re.escape(pat) + r"(?![\w-])", text, re.I):
                pre = text[max(0, m.start() - 70):m.start()]
                cut = 0
                for c in CLAUSE.finditer(pre):
                    cut = c.end()
                clause = pre[cut:]
                negd = bool(NEG_PRE.search(clause)) or bool(NEG_POST.match(text[m.end():m.end() + 30]))
                tense, nowcue = None, False
                for w in reversed(re.findall(r"[\w']+", clause.lower())[-5:]):
                    if w in PRES_W:
                        tense, nowcue = "present", w in ("now", "currently")
                        break
                    if w in PAST_W:
                        tense = "past"
                        break
                memo = bool(MEMORY_CUE.search(text[max(0, m.start() - 120):m.start()]))
                out.append({"pos": m.start(), "end": m.end(), "val": val, "neg": negd, "tense": tense, "now": nowcue,
                            "memo": memo})
    out.sort(key=lambda x: x["pos"])
    return out


def uniq(xs):
    seen = []
    for x in xs:
        if x not in seen:
            seen.append(x)
    return seen


def pick(cands, want):
    vals = uniq([m["val"] for m in cands])
    if len(vals) <= 1:
        return (vals[0] if vals else None), None
    tensed = uniq([m["val"] for m in cands if m["tense"] == want])
    if len(tensed) == 1:
        return tensed[0], None
    if want == "present":
        nows = uniq([m["val"] for m in cands if m["now"]])
        if len(nows) == 1:
            return nows[0], None
    return None, "ambiguous"


def topic_hit(text, sch, a):
    """Does the answer talk about the asked attribute at all? Sentences where the model introduces itself
    ("My name is Nemotron.") do not count, or every name question would look on topic."""
    body = " ".join(s for s in sentences(text) if not SELF_TALK.search(s)).lower()
    return any(re.search(r"\b" + re.escape(w), body) for w in sch[a].get("topic", []))


def off_topic(ss, sch, a, upto):
    """The answer opens by talking about the model itself and then neither names a value nor says it does not
    know ("My name is Nemotron. I am created by NVIDIA researchers.")."""
    head = " ".join(ss[:upto])
    return bool(SELF_TALK.search(ss[0])) and not REFUSAL.search(head) and not ABSENT.search(head) \
        and not [m for m in mentions(head, sch, a) if not m["neg"]]


def guessed(ss, sch, a, upto=3):
    """Values named inside a guess ("maybe green", "probably brown") in the first sentences, unless the model
    ties them to what the user said ("you told me")."""
    out = []
    for s in ss[:upto]:
        if GUESS.search(s) and not MEMORY_CUE.search(s):
            vals = uniq([m["val"] for m in mentions(s, sch, a) if not m["neg"]])
            if 1 <= len(vals) <= 2:                    # a list of three or more is general information
                out += vals
    return uniq(out)


def described(ss, sch, a, opts):
    """The answer lays out both options without choosing: every value it names is one of the two, each as the
    subject of a general statement ("Purple is often associated with royalty. Blue is often associated with calm.").
    A value said about the user or picked ("I think you like blue", "Blue, I'd say."), hedged or tied to memory is
    not a description."""
    text = " ".join(ss)
    ms = mentions(text, sch, a)
    for m in ms:
        cut = 0
        for c in CLAUSE.finditer(text, 0, m["pos"]):
            cut = c.end()
        nxt = CLAUSE.search(text, m["end"])
        clause = text[cut:nxt.start() if nxt else len(text)]
        if text[cut:m["pos"]].strip() or not DESCRIBE.match(text, m["end"]) or PERSON.search(clause) \
                or GUESS.search(clause) or m["memo"]:
            return False
    return {m["val"] for m in ms} == set(opts)


def extract_value(text, sch, a, want):
    ss = sentences(text)
    if not ss:
        return {"got": "EMPTY"}
    if off_topic(ss, sch, a, 3):
        return {"got": "OFFTOPIC"}
    for k in sorted({1, 2, 3, len(ss)}):
        scope = " ".join(ss[:k])
        cands = [m for m in mentions(scope, sch, a) if not m["neg"]]
        refusal = bool(REFUSAL.search(scope))
        g = guessed(ss[:k], sch, a)
        if g:                                          # "I'm not sure, maybe green": a guess, not a recall
            return {"got": "HEDGED", "values": g}
        if k == 1 and refusal and not any(m["memo"] for m in cands):
            g = guessed(ss, sch, a)                    # "I don't know. It might be brown." is still a guess
            return {"got": "HEDGED", "values": g} if g else {"got": "ABSENT"}
        v, why = pick(cands, want)
        if v and want == "present" and k < len(ss) and all(m["tense"] == "past" for m in cands if m["val"] == v):
            # the opening tells the history ("Originally the boat belonged to your sister ... So currently, the boat
            # belongs to your brother."): one other value said in the present further on is the answer (an
            # audit of 15 Oct 2026, L6 Q03). With no such value the past one stands, as before.
            allc = [m for m in mentions(" ".join(ss), sch, a) if not m["neg"]]
            later = uniq([m["val"] for m in allc if m["val"] != v and (m["now"] or m["tense"] == "present")])
            if len(later) == 1 and not any(m["val"] == v and m["tense"] == "present" for m in allc):
                v = later[0]
        if v:
            return {"got": "VALUE", "value": v}
        if refusal or ABSENT.search(scope):
            return {"got": "ABSENT"}
        if why:
            return {"got": "AMBIGUOUS", "values": uniq([m["val"] for m in cands])}
        if k >= len(ss):
            break
    return {"got": "OFFTOPIC" if not topic_hit(text, sch, a) else "UNCLEAR"}


def extract_set(text, sch, a):
    ss = sentences(text)
    if not ss:
        return {"got": "EMPTY"}
    if off_topic(ss, sch, a, 3):
        return {"got": "OFFTOPIC"}
    g = guessed(ss, sch, a)
    if g:
        return {"got": "HEDGED", "values": g}
    first = [m for m in mentions(ss[0], sch, a) if not m["neg"]]
    if (REFUSAL.search(ss[0]) or ABSENT.search(ss[0])) and not first:
        return {"got": "ABSENT"}
    ms = mentions(" ".join(ss), sch, a)
    pos = uniq([m["val"] for m in ms if not m["neg"]])
    negs = uniq([m["val"] for m in ms if m["neg"] and m["val"] not in pos])
    if not pos:
        if REFUSAL.search(" ".join(ss)) or ABSENT.search(" ".join(ss)):
            return {"got": "ABSENT"}
        return {"got": "OFFTOPIC" if not topic_hit(text, sch, a) else "UNCLEAR"}
    return {"got": "SET", "yes": pos, "no": negs}


def lean_of(s):
    """The way a hedged yes/no answer leans ("Probably not." -> no, "Maybe, yes." -> yes), or None."""
    ws = {NO if re.search(r"\b(?:no|nope|not)\b", m.group(0), re.I) else YES for m in LEAN.finditer(s)}
    return ws.pop() if len(ws) == 1 else None


def extract_yn(text):
    ss = sentences(text)
    if not ss:
        return {"got": "EMPTY"}
    f = ss[0].lower().lstrip(" \"'")
    if SELF_TALK.search(ss[0]) and not re.search(r"\b(?:yes|no)\b", " ".join(ss[:2]), re.I) \
            and not REFUSAL.search(" ".join(ss[:2])):
        return {"got": "OFFTOPIC"}
    if SELF_REPLY.match(f):                               # "No, I am a language model ..." is about the model
        return {"got": "YN", "value": UNK}
    lk = LIKELY.match(ss[0])
    if lk:                                                # "It is very unlikely that ...": a guess that leans no
        lean = NO if lk.group("no") else YES
        if MEMORY_CUE.search(ss[0]):                      # "..., since you told me it changed": reasoned, not guessed
            return {"got": "YN", "value": lean}
        return {"got": "HEDGED", "value": lean, "likely": lean}
    if GUESS.search(ss[0]) and LEAN.search(ss[0]) and not MEMORY_CUE.search(ss[0]):
        return {"got": "HEDGED", "value": lean_of(ss[0])}  # "Probably not." / "I'm not sure, maybe yes."
    if REFUSAL.search(ss[0]) or HEDGE.search(ss[0]):      # "No, I am an AI and cannot know..." is not a no
        return {"got": "YN", "value": UNK}
    if NO_START.match(f):
        return {"got": "YN", "value": NO}
    if YES_START.match(f):
        return {"got": "YN", "value": YES}
    if REFUSAL.search(" ".join(ss[:2])):
        return {"got": "YN", "value": UNK}
    has_y, has_n = bool(re.search(r"\byes\b", f)), bool(re.search(r"\bno\b", f))
    if has_y != has_n:
        return {"got": "YN", "value": YES if has_y else NO}
    return {"got": "UNCLEAR"}


def extract_choice(text, sch, a, opts=()):
    ss = sentences(text)
    if not ss:
        return {"got": "EMPTY"}
    if off_topic(ss, sch, a, 2):
        return {"got": "OFFTOPIC"}
    for k in sorted({1, 2}):
        scope = " ".join(ss[:k])
        if NEITHER.search(scope):
            return {"got": "CHOICE", "value": "neither"}
        cands = [m for m in mentions(scope, sch, a) if not m["neg"]]
        refused = bool(REFUSAL.search(scope))
        if guessed(ss[:k], sch, a):                    # "I'm not sure, maybe green"
            return {"got": "HEDGED", "values": guessed(ss[:k], sch, a)}
        if k == 1 and refused and not any(m["memo"] for m in cands):
            return {"got": "CHOICE", "value": "unknown"}
        v, why = pick(cands, "present")
        if v:
            return {"got": "CHOICE", "value": v}
        if refused:
            return {"got": "CHOICE", "value": "unknown"}
        if why:
            return {"got": "AMBIGUOUS", "values": uniq([m["val"] for m in cands])}
        if k >= len(ss):
            break
    if opts and REFUSAL.search(" ".join(ss)) and described(ss, sch, a, opts):
        return {"got": "CHOICE", "value": "unknown"}   # both options described side by side, then "I don't know"
    return {"got": "OFFTOPIC" if not topic_hit(text, sch, a) else "UNCLEAR"}


YOU_VERB = {"do", "does", "did", "have", "has", "are", "were", "like", "liked", "love", "loved", "live", "lived", "own",
            "owned", "both", "also", "too", "and", "or", "really", "definitely", "certainly", "yourself", "alone"}
YOU_NOT = re.compile(r"\s*(?:not\b|n't|never\b|no\b|told\b|tell\b|mentioned\b|mention\b|said\b|say\b|shared\b|share\b"
                     r"|asked\b|ask\b|asking\b|informed\b|taught\b|let\s+me\b|referring\b|thinking\b|wondering\b"
                     r"|curious\b|looking\b|to\b|been\s+told\b|already\b|welcome\b)", re.I)
# "you have a dog with grey eyes": the user owns something, which does not answer "who has grey eyes?"
YOU_OWN = re.compile(r"\s*(?:a|an|the|one|two|three|some|several)\b", re.I)
# the user's own people and pets ("your brother", "your best friend"): one the set does not have is someone the
# user never told about, so an answer that names one is not about people in general
KIN = re.compile(r"\byour\s+(?:[\w']+\s+)?(brother|sister|mother|mom|mum|father|dad|parents?|friends?|partner|wife"
                 r"|husband|son|daughter|child|children|kids?|cousin|aunt|uncle|grand(?:mother|father|ma|pa|parents?)"
                 r"|neighbou?r|colleague|boss|girlfriend|boyfriend|roommate|flatmate|dog|cat|rabbit|pets?|puppy|kitten"
                 r"|hamster|parrot|bird|horse)\b", re.I)
YOU_ALONE = re.compile(r"^\W*(?:(?:both|only|just|and|also|it's|it\s+is|that's|that\s+is|that\s+would\s+be"
                       r"|the\s+answer\s+is|so\s+(?:do|are|have)|as\s+(?:do|are|have))\s+)?$", re.I)


def you_named(clause):
    """Does this clause give the user ("you") as an answer to "who"? "You do.", "You and your dog.", "you like olives"
    do; "you told me", "you haven't said", "you'd have to tell me", "you're asking" and "Thank you!" do not. Clauses
    are split at "and", so "You and your dog." gives the clause "You"."""
    for m in re.finditer(r"\byou\b(?!')", clause, re.I):
        rest = clause[m.end():]
        nxt = re.match(r"\s*([\w']+)", rest)
        if not nxt:                                    # "You." / "Both you and ..."
            if YOU_ALONE.match(clause[:m.start()]):
                return True
            continue
        w = nxt.group(1).lower()
        if w in YOU_VERB and not YOU_NOT.match(rest[nxt.end():]) \
                and not (w in ("have", "has", "own", "owned") and YOU_OWN.match(rest[nxt.end():])):
            return True
    return False


def ent_patterns(e):
    """How an answer names entity e: "your dog", "my dog" (an echo), "the boat". The user is read by `you_named`."""
    core = e.split()[-1]
    return [r"\b(?:your|my|the)\s+" + core + r"\b"]


def ent_names(S, a):
    """Taught names that point at one entity ("Bruno" -> my dog), so "Bruno has grey eyes." names my dog. Not used on
    a reverse question about names itself, where the name is the question's own value."""
    by = {}
    if a == "name":
        return {}
    for l in S["lessons"]:
        if l["a"] == "name":
            for v in (l.get("v"), l.get("v0"), l.get("v1")):
                if v:
                    by.setdefault(v, set()).add(l["e"])
    return {v: next(iter(es)) for v, es in by.items() if len(es) == 1}


def ent_pats(e, names):
    """The patterns that name entity e in an answer: its own ("your dog") and its taught names ("Bruno")."""
    return ent_patterns(e) + [r"\b" + re.escape(n) + r"\b" for n, ne in names.items() if ne == e]


NEG_W = re.compile(r"\b(?:not|never|no\s+longer)\b|n't\b", re.I)
# words that leave out the entity after them: "unlike your cat", "neither your dog nor ..."
EXCEPT = re.compile(r"\b(?:neither|unlike|except(?:\s+for)?|apart\s+from|other\s+than|rather\s+than|instead\s+of"
                    r"|as\s+opposed\s+to)\s+(?:[\w']+\s+){0,3}$", re.I)


def ents_in(clause, ents, names, vpos=()):
    """Entities a clause names, unless the clause negates them ("not your cat", "your cat doesn't have grey eyes",
    "unlike your cat").
    vpos: where the asked value stands in the clause. A negation after it is about something else ("Your dog has grey
    eyes that never change.")."""
    out = []
    for e in ents:
        if e == "me":
            if you_named(clause):
                out.append(e)
            continue
        for p in ent_pats(e, names):
            m = re.search(p, clause, re.I)
            if not m:
                continue
            stop = min([x for x in vpos if x >= m.end()], default=len(clause))
            tail = clause[m.end():stop]
            if NEG_PRE.search(clause[:m.start()]) or EXCEPT.search(clause[:m.start()]) or NEG_W.search(tail):
                continue
            out.append(e)
            break
    return out


# a who-has answer's clauses also end at "although" and "whilst", which join two statements as "but" and "while" do
WHO_CLAUSE = re.compile(CLAUSE.pattern + r"|\balthough\b|\bwhilst\b", re.I)
# the start of a predicate, the words that follow a subject: "does not like", "likes", "has", "also loves"
PRED = (r"(?:(?:also|still|really|only|just|actually|definitely|certainly|absolutely|always|never|now|instead"
        r"|rather|surely|especially|particularly|truly|clearly|usually|often|sometimes|maybe|perhaps|possibly|probably"
        r"|likely|no\s+longer)\s+)*"
        r"(?:(?:do|does|did|is|are|was|were|has|have|had|can|could|will|would|should|must)(?:n't)?|cannot|can't|won't"
        r"|likes?|liked|loves?|loved|enjoys?|enjoyed|prefers?|preferred|adores?|adored|hates?|hated|dislikes?|disliked"
        r"|eats?|ate|lives?|lived|belongs?|belonged|owns?|owned|seems?|seemed|wants?|wanted)\b")
# a clause that starts with its predicate, or with "it", "he", "she" or "they" and then the predicate, has no subject
# of its own: "..., but likes olives", "..., although it likes olives"
GOES_ON = re.compile(r"^\s*(?:(?:it|he|she|they)\s+)?" + PRED, re.I)
PRED_NEXT = re.compile(r"\s+" + PRED, re.I)
HAS_PRED = re.compile(r"(?:^|\s)" + PRED, re.I)
# the words that join two statements about one subject; an aside ("does not, however, like") is not one of them
JOINS = re.compile(r"\b(?:but|and|while|whilst|whereas|although)\b", re.I)
# a clause that only names some of the set's entities, with words like these around them: "Your dog", "and your cat",
# "not your cat", "unlike your cat", "Nora", "I think your dog"
BARE_REST = re.compile(r"[\W_]*(?:(?:both|either|neither|not|unlike|except|for|apart|from|other|than|rather|instead"
                       r"|of|as|well|plus|along|with|together|also|even|just|only|and|or|nor|maybe|perhaps|probably"
                       r"|possibly|definitely|certainly|especially|too|i|think|believe|guess|suppose|say|course)\b"
                       r"[\W_]*)*", re.I)
# a clause that only names values, after a comma or "and": the next item of a list ("likes apples, pears and olives")
LIST_REST = re.compile(r"[\W_]*(?:(?:also|too|even|especially|particularly|both|some|the|as|well|not|really)\b[\W_]*)*",
                       re.I)
LIST_SEP = re.compile(r"^[\s,]*(?:and[\s,]*)?$", re.I)


def bare_subject(c, ents, names):
    """Clause c only names some of the set's entities ("Your dog", "and your cat", "not your cat", "Nora"): a subject
    that waits for its predicate in a later clause ("Your dog and your cat do not like olives."), or a bare answer
    ("You and your dog.")."""
    spans = sorted([m.span() for e in ents if e != "me" for p in ent_pats(e, names) for m in re.finditer(p, c, re.I)]
                   + [m.span() for m in re.finditer(r"\byou\b", c, re.I)], reverse=True)
    rest = c
    for a0, a1 in spans:
        rest = rest[:a0] + " " + rest[a1:]
    return bool(spans) and bool(BARE_REST.fullmatch(rest))


def in_list(c, ms):
    """Clause c only names values ("pears", "and olives", "not olives"): the next item of a list."""
    rest = c
    for m in sorted(ms, key=lambda m: m["pos"], reverse=True):
        rest = rest[:m["pos"]] + " " + rest[m["end"]:]
    return bool(ms) and bool(LIST_REST.fullmatch(rest))


def says(c, ms, v):
    """Does clause c say v, with no negation before it? None when it names no value at all."""
    if not ms:
        return None
    at = [m["pos"] for m in ms if m["val"] == v and not m["neg"]]
    return bool(at) and not NEG_W.search(c[:at[0]])


# a clause that takes a value back for its subject says no more than these words, its subject and the value itself:
# "no longer does", "doesn't anymore", "does not", "not any more", "stopped liking them", "does not like olives anymore"
TAKE_BACK_WORDS = {"it", "he", "she", "they", "does", "do", "doesn't", "don't", "is", "isn't", "are", "aren't", "has",
                   "have", "not", "no", "never", "longer", "anymore", "any", "more", "now", "stopped", "quit", "liking",
                   "like", "likes", "enjoying", "enjoy", "enjoys", "eating", "eat", "eats", "them", "those", "really",
                   "either", "still", "sadly", "though"}
CEASE = re.compile(r"\b(?:no\s+longer|any\s?more|stopped|quit)\b", re.I)


def takes_back(c, ms, v, ents, names):
    """Clause c takes the value v back for its subject: "your dog no longer does", "it doesn't anymore", "but not any
    more", "stopped liking them", "your dog does not like olives anymore". Only these short forms count: a clause that
    names another value ("does not like apples"), says more ("does not like them in salads", "not all of them") or
    talks about the past ("did not as a puppy") takes nothing back."""
    if not (NEG_W.search(c) or CEASE.search(c)) or any(m["val"] != v for m in ms):
        return False
    spans = [(m["pos"], m["end"]) for m in ms] + [m.span() for e in ents if e != "me" for p in ent_pats(e, names)
                                                  for m in re.finditer(p, c, re.I)] \
        + [m.span() for m in re.finditer(r"\byou\b", c, re.I)]
    rest = c
    for a0, a1 in sorted(spans, reverse=True):
        rest = rest[:a0] + " " + rest[a1:]
    return all(w in TAKE_BACK_WORDS for w in re.findall(r"[a-z']+", rest.lower()))


def positive(c, ms):
    """The values clause c names are said, not denied: the last one is not negated and no negation stands before it
    ("likes apples", not "does not like apples")."""
    return bool(ms) and not ms[-1]["neg"] and not NEG_W.search(c[:ms[-1]["pos"]])


def subject_of(c, ents, names):
    """The set's entities that clause c has as its subject: those followed by a predicate ("your dog does not like",
    "you like", "Bruno likes"), else the one the clause opens with ("Your dog's eyes are ..."). The user counts only
    before a predicate, so "you told me" is not a subject."""
    hits = sorted([(m.start(), m.end(), e) for e in ents if e != "me" for p in ent_pats(e, names)
                   for m in re.finditer(p, c, re.I)]
                  + [(m.start(), m.end(), "me") for m in re.finditer(r"\byou\b(?!')", c, re.I)])
    sub = uniq([e for s0, s1, e in hits if PRED_NEXT.match(c, s1)])
    if not sub and hits and hits[0][2] != "me" and not c[:hits[0][0]].strip(" \"'"):
        sub = [hits[0][2]]
    return sub


def goes_on(c, sep, subj, v, ms, asked):
    """Clause c has no subject of its own and follows "but", "and", "while", "whereas" or "although": it says
    something about subj, the subject of the clause before it ("Your dog does not like apples but likes olives.").
    Returns subj when it names the value v with no negation before it, else []. A question ("..., but does it like
    olives?") claims nothing."""
    if asked or not subj or not GOES_ON.match(c) or not JOINS.search(sep) or not says(c, ms, v):
        return []
    return list(subj)


def extract_entities(text, ents, sch=None, a=None, v=None, names=None):
    """Who has value v? Reads every statement of the answer, clause by clause. Questions asked back ("Do you like
    olives?"), conditions ("if you're a fan"), refusals ("I don't know whether your dog...") and negations do not
    name anyone. An answer that names none of the set's entities and talks about people in general ("Here are some
    famous people with grey eyes") is a general answer: it names nobody the user knows. A clause with no subject of
    its own goes on with the subject before it: "Your dog does not like apples but likes olives." names your dog.
    Joined subjects share their predicate: "Your dog and your cat do not like olives." names nobody. A value in a list
    belongs to the verb's subject: "Your dog likes apples and olives." names your dog, "... but not olives." does
    not. A later clause that takes the value back removes an earlier credit: "Your dog and your cat like olives, but
    your dog no longer does." names your cat only."""
    ss = sentences(text)
    if not ss:
        return {"got": "EMPTY"}
    names = names or {}
    ment = (lambda t: mentions(t, sch, a)) if sch and a and v else (lambda t: [])
    cores = {e.split()[-1].lower() for e in ents}
    found, hedged, others = [], [], []
    for s in ss:
        asked = s.rstrip().endswith("?")
        if asked and ASK_BACK.match(s) or IF_FIRST.match(s):
            continue                                   # "Do you like olives?", "If you have grey eyes, you are ..."
        guess = (bool(GUESS.search(s)) or asked) and not MEMORY_CUE.search(s)     # "Maybe your dog?" is a guess
        # where each entity is named in the sentence: an entity's words run on to the next one named
        marks = sorted([(m.start(), e) for e in ents if e != "me" for p in ent_pats(e, names)
                        for m in re.finditer(p, s, re.I)] + [(m.start(), "me") for m in re.finditer(r"\byou\b", s, re.I)])
        cuts = [0] + [x for c in WHO_CLAUSE.finditer(s) for x in (c.start(), c.end())] + [len(s)]
        credit = (hedged if guess else found).extend
        subj, last = [], 0                             # the subject so far, and where the clause before it ended
        group, lst = [], None                          # a subject waiting for its predicate; the list being read
        for c0, c1 in zip(cuts[::2], cuts[1::2]):
            c = s[c0:c1]
            if not c.strip():
                continue
            sep, last = s[last:c0], c1
            if REFUSAL.search(c) or COND.search(c):
                subj, lst = [], None
                continue
            if bare_subject(c, ents, names):          # "Your dog", "and your cat": the predicate comes later
                group += [e for e in ents_in(c, ents, names) if e not in group]
                lst = None
                continue
            ms = ment(c)
            vals = {m["val"] for m in ms if not m["neg"]}
            sub, carried, joined, known = subject_of(c, ents, names), [], [], False   # known: whose clause it is
            who = []                                   # the entities this clause says something about
            if group and sub:                          # "Your dog and your cat do not like olives."
                if LIST_SEP.match(sep):
                    joined = group                     # shares this clause's predicate, read below
                else:
                    credit(group)                      # "Your dog, but your cat ...": named on its own
                subj, group, known = joined + sub, [], True
                who = subj
            elif group and (GOES_ON.match(c) or ms or HAS_PRED.search(c)):
                said = says(c, ms, v)                  # "Your dog, unlike your cat, likes olives."
                if (not NEG_W.search(c)) if said is None else said:
                    credit(group)
                subj, group, known, who = group, [], True, group
            elif group:
                pass                                   # an aside ("as you know"): the group waits for its predicate
            elif sub:
                subj, known, who = sub, True, sub      # a subject of its own
            elif GOES_ON.match(c):
                carried = goes_on(c, sep, subj, v, ms, asked)   # "..., but likes olives": the subject before it
                known = bool(subj and JOINS.search(sep)) and not asked
                who = subj if known else []
            elif lst and in_list(c, ms):               # "Your dog likes apples and olives.": the list goes on
                if not LIST_SEP.match(sep):
                    lst = None                         # "... but not olives" is no list item
                elif lst[1] and v in vals and not asked:
                    credit(lst[0])
                continue
            elif HAS_PRED.search(c):
                subj = []                              # "your brother does not like pears": someone else's predicate
            elif subj and JOINS.search(sep) and not asked:
                who = subj                             # "..., but not any more"
            if who and not asked and takes_back(c, ms, v, ents, names):
                found[:] = [e for e in found if e not in who]      # "..., but your dog no longer does."
                hedged[:] = [e for e in hedged if e not in who]
            lst = (subj, positive(c, ms)) if known and ms else None
            credit(carried)
            if vals and v not in vals:
                continue                               # "..., and your cat's are hazel": another value
            here = []
            for e in ents_in(c, ents, names, [m["pos"] for m in ms if m["val"] == v]):
                if not vals:                           # "Bruno has them.": read on to the value
                    end = next((p for p, x in marks if p >= c1 and x != e), len(s))
                    seg = {m["val"] for m in ment(s[c0:end]) if not m["neg"]}
                    if seg and v not in seg:
                        continue
                here.append(e)
            credit(here)
            if joined and any(e in here for e in sub):
                credit(joined)                         # "Your dog and your cat like olives."
            others += [w for w in (k.group(1).lower() for k in KIN.finditer(c)) if w not in cores
                       and w.rstrip("s") not in cores]
        credit(group)                                  # "You and your dog.": named as the answer
    found = uniq(found)
    hedged = uniq([e for e in hedged if e not in found])
    if SELF_TALK.search(ss[0]) and not found and not hedged:
        return {"got": "OFFTOPIC"}
    if hedged:
        return {"got": "HEDGED", "values": found + hedged}
    if found:
        return {"got": "ENTS", "yes": found}
    whole = " ".join(ss)
    if REFUSAL.search(whole) or NOBODY.search(ss[0]):
        return {"got": "ENTS", "yes": []}
    if others:                                         # "Your brother has grey eyes.": someone the set does not have
        return {"got": "OTHERS", "values": uniq(others)}
    if general_answer(whole, sch, a, v):
        return {"got": "GENERAL", "yes": []}
    return {"got": "UNCLEAR"}


def general_answer(text, sch=None, a=None, v=None):
    """An answer to "who has v?" about the world at large: it talks about people in general, or about v itself
    ("Grey eyes are relatively rare ..."), and names nobody the user knows (the caller checks that part)."""
    return bool(GENERAL.search(text)) or bool(sch and a and v and any(m["val"] == v for m in mentions(text, sch, a)))


# ------------------------------------------------------------------ whose value: the entity an answer gives it to
# "Your cat's eyes are grey." does not answer "What is my dog's eye colour?", although grey is right (13 Oct). The
# user as the holder of a value: "your eyes", "your favourite colour", "you have ...", "you like ...", but not "you
# have a dog with grey eyes", where the user owns the holder. Switched off in the self-test's negative control.
ENTITY_CHECK = True
# the asked entity keeps the answer's old reading only when the answer gives it a value ("Your dog's eyes are grey,
# like your cat's."), not when it is only named ("Your cat has grey eyes. Your dog is asleep.", an audit of
# 35fc940). Switched off, as at 35fc940, in the self-test's second control: named anywhere is enough.
ASKED_AS_HOLDER = True
# a clause that only repeats the predicate for someone: "and so does your dog", "You do too.", "and so are yours",
# "your dog does": what is left once the entities are taken out
SAME_REST = re.compile(r"[\W_]*(?:(?:and|but|so|as|also|too|well|both|do|does|did|is|are|was|were|has|have|had)\b[\W_]*)+",
                       re.I)
YOU_HOLD = re.compile(r"\byou\s+(?:(?:also|still|really|only|just|now|always|definitely|certainly|actually|both)\s+)*"
                      r"(?:have|had|like|liked|love|loved|live|lived|are|were|prefer|preferred|enjoy|enjoyed|own|owned)\b"
                      r"(?!\s+(?:a|an|the|one|two|three|some|several)\b)", re.I)


def your_topic(sch, a):
    """ "your eyes", "your eye colour", "your favourite colour", "your home": the user's own value of attribute a."""
    topic = "|".join(map(re.escape, sch[a].get("topic", [])))
    return re.compile(r"\byour\s+(?:own\s+)?(?:" + topic + r")", re.I) if topic else None


def holders_in(c, ents, names, yt):
    """The set's entities clause c says something of: one followed by a predicate ("your dog has", "Nora is"), by a
    possessive ("your cat's eyes", "Nora's"), or opening the clause; the user by "your eyes", "your favourite colour",
    "you have", "you like" (`YOU_HOLD`, `your_topic`). Negated ones ("not your cat", "unlike your cat") are left out,
    as `ents_in` leaves them out."""
    named, out = ents_in(c, ents, names), []
    for e in ents:
        if e == "me":
            m = yt.search(c) if yt else None
            y = re.search(r"\byours\b", c, re.I)
            if (m and not NEG_PRE.search(c[:m.start()]) and not EXCEPT.search(c[:m.start()])) or YOU_HOLD.search(c) \
                    or (y and not NEG_PRE.search(c[:y.start()]) and not EXCEPT.search(c[:y.start()])):
                out.append(e)
        elif e in named and any(re.match(r"(?:'s|s')\b", c[m.end():]) or PRED_NEXT.match(c, m.end())
                                or not c[:m.start()].strip(" \"'")
                                for p in ent_pats(e, names) for m in re.finditer(p, c, re.I)):
            out.append(e)
    return out


def value_holders(text, S, q):
    """Every clause of the answer that states a value of the asked attribute (not negated), with the entities it says
    it of (`holders_in`). A clause with no holder of its own goes on with a subject still waiting for its predicate
    ("Your cat, Nora, has grey eyes."), with the clause before it when it starts with its predicate after "and",
    "but", "while", "although" or a comma ("..., and has grey eyes"), or as the next item of a list ("Your dog likes
    apples and pears."). Questions asked back, conditions and refusals say nothing of anyone."""
    sch, a, ents = S["schema"], q["a"], S["entities"]
    names, yt, out = ent_names(S, a), your_topic(sch, a), []
    said = set()                                       # the values of the last clause that stated one
    for s in sentences(text):
        if s.rstrip().endswith("?") and ASK_BACK.match(s) or IF_FIRST.match(s):
            continue
        cuts = [0] + [x for c in WHO_CLAUSE.finditer(s) for x in (c.start(), c.end())] + [len(s)]
        prev, group, last = [], [], 0
        for c0, c1 in zip(cuts[::2], cuts[1::2]):
            c = s[c0:c1]
            if not c.strip():
                continue
            sep, last = s[last:c0], c1
            if REFUSAL.search(c) or COND.search(c):
                prev, group = [], []
                continue
            ms = [m for m in mentions(c, sch, a) if not m["neg"]]
            if not ms and said:                              # "and so does your dog", "You do too."
                same = same_as_before(c, ents, names)
                if same:
                    out.append((same, set(said)))
                    prev, group = same, []
                    continue
            if not ms and bare_subject(c, ents, names):      # "Your cat", "Nora": the predicate comes later
                group += [e for e in ents_in(c, ents, names) if e not in group]
                continue
            who = holders_in(c, ents, names, yt)
            if not who and group and (GOES_ON.match(c) or HAS_PRED.search(c)):
                who = group
            elif not who and prev and (GOES_ON.match(c) and (JOINS.search(sep) or "," in sep)
                                       or ms and in_list(c, ms) and LIST_SEP.match(sep)):
                who = prev
            group = []
            if ms:
                out.append((uniq(who), {m["val"] for m in ms}))
                said = {m["val"] for m in ms}
            prev = who if who else [] if HAS_PRED.search(c) else prev
    return out


def same_as_before(c, ents, names):
    """The entities clause c says the same thing of, when it only repeats the predicate of the clause before: "and so
    does your dog", "You do too.", "and so are yours", "your dog does". A negation, a value or anything more ("Your
    dog is asleep.") is not a repeat."""
    spans = [m.span() for e in ents if e != "me" for p in ent_pats(e, names)
             for m in re.finditer(p + r"(?:'s|s')?", c, re.I)] \
        + [m.span() for m in re.finditer(r"\byou(?:rs)?\b", c, re.I)]   # "so are your dog's"
    rest = c
    for a0, a1 in sorted(spans, reverse=True):
        rest = rest[:a0] + " " + rest[a1:]
    if not spans or not SAME_REST.fullmatch(rest):
        return []
    who = [e for e in ents_in(c, ents, names) if e != "me"]
    return who + (["me"] if "me" in ents and re.search(r"\byou(?:rs)?\b", c, re.I) else [])


def said_of_other(S, q, content, g):
    """Does the answer give the value it was judged on to another entity of the set, and not to the asked one?
    "Your cat's eyes are grey." for "What is my dog's eye colour?" gives grey to the cat. Returns None (judge as
    before), or ("fail", whose) when every clause that states that value says it of someone else, or ("unclear",
    whose) when the value is also said with no holder ("Grey. Your cat's eyes are grey too.") or after a yes ("Yes,
    your dog likes olives." for "Do I like olives?"). An answer that gives a value to the asked entity ("Your dog's
    eyes are grey, like your cat's.", "..., and so does your dog"), or names no entity of the set, is judged as before,
    and so is a no that says whose the value is ("No, your cat is called Nora."). Naming the asked entity is not
    enough: "Your cat has grey eyes. Your dog is asleep." gives grey to the cat (an audit of 35fc940)."""
    if not ENTITY_CHECK or q["type"] in ("WHO_HAS",) + TWO_STEP or q.get("form") == "who_yn" or "hops" in q:
        return None
    sch, a, e, x = S["schema"], q["a"], q["e"], g.get("got")
    text = " ".join(sentences(content))
    names = ent_names(S, a)
    # a value that is itself an entity of the set ("Whose is the boat?" "My brother.") is not its own holder: the
    # brother named as the owner is the value, not someone the value is said of (an audit, L6 Q03)
    hs = [(type(w)(x for x in w if x not in vals) if w else w, vals) for w, vals in value_holders(content, S, q)]
    if ASKED_AS_HOLDER:                                # the answer gives a value to the asked entity: as before
        if any(e in w for w, _ in hs):
            return None
    elif e == "me":                                    # as at 35fc940: the asked entity named anywhere
        yt = your_topic(sch, a)
        if (yt and yt.search(text)) or YOU_HOLD.search(text) or re.search(r"\byours\b", text, re.I) \
                or any(you_named(c) for c in WHO_CLAUSE.split(text)):
            return None
    elif any(re.search(p, text, re.I) for p in ent_pats(e, names)):
        return None
    explicit = False
    if x == "VALUE":
        want = {g["value"]}
    elif x == "SET":
        want = set(g["yes"])
    elif x == "CHOICE":
        want = {g["value"]} - {"unknown", "neither"}
    elif x in ("YN", "HEDGED") and "v" in q:
        if g.get("value") == UNK:
            return None
        form = q.get("form")
        base = (extract_tf(content) if form == "true_false" or truth_opening(content) else
                extract_recall_yn(content, sch, a, q["v"]) if form == "recall_yn" else extract_yn(content))
        explicit = base["got"] in ("YN", "HEDGED") and base.get("value") in (YES, NO)
        if explicit and base["value"] == NO:
            return None                                # "No, your cat is called Nora.": says whose it is
        want = {q["v"]}
        if not explicit:                               # the yes or no was read from the value the answer names
            gv = extract_value(content, sch, a, "present")
            want |= {gv["value"]} if gv["got"] == "VALUE" else set(gv.get("values") or [])
    elif x == "HEDGED":
        want = set(g.get("values") or [])
    else:
        return None
    other = [w for w, vals in hs if w and vals & want]
    if not want or not other:
        return None
    bare = any(not w and vals & want for w, vals in hs)
    return ("unclear" if bare or explicit else "fail"), uniq([y for w in other for y in w])


def whose_words(es):
    return ", ".join("you" if x == "me" else "your " + x[3:] if x.startswith("my ") else x for x in es)


def date_period(m):
    """A date found in an answer -> (lo, hi, grain) in ticks: a day, a month or a year. None if it is not a date."""
    g = {k: v for k, v in m.groupdict().items() if v}
    mon = lambda s: next(i for i, n in enumerate(MONTHS, 1) if n.startswith(s.lower().rstrip(".")[:3]))
    try:
        if "y1" in g or "y2" in g or "y3" in g:
            d = dt.date(int(g.get("y1") or g.get("y2") or g["y3"]),
                        int(g["o3"]) if "o3" in g else mon(g.get("m1") or g["m2"]), int(g.get("d1") or g.get("d2") or g["d3"]))
            lo = tick(d.isoformat())
            return lo, lo + TPD, "day"
        if "y4" in g:
            y, mo = int(g["y4"]), mon(g["m4"])
            nxt = dt.date(y + (mo == 12), mo % 12 + 1, 1)
            return tick(dt.date(y, mo, 1).isoformat()), tick(nxt.isoformat()), "month"
        y = int(g["y5"])
        return tick(f"{y}-01-01"), tick(f"{y + 1}-01-01"), "year"
    except (ValueError, StopIteration):
        return None


def rel_of(s, dm):
    """The word right before a date in sentence s that ties it to a value ("until", "since", "on" ...), or None."""
    r = REL_RE.search(s[:dm.start()])
    return " ".join(r.group("rel").lower().split()) if r else None


def clause_span(s, pos, end=None):
    """The clause of sentence s around s[pos:end], between the CLAUSE breaks on either side."""
    lo = 0
    for c in CLAUSE.finditer(s, 0, pos):
        lo = c.end()
    nxt = CLAUSE.search(s, end if end is not None else pos)
    return lo, nxt.start() if nxt else len(s)


def topic_words(sch, a):
    """Topic words of the other attributes that a's own topic does not have, and a's words no other attribute has
    (eye_colour: "favourite", "like", "home" ... against "eye", "iris"). Every attribute the rule book knows counts, not
    only the set's own, since an answer may talk about anything."""
    own, allsch = set(sch[a].get("topic", [])), {**SCHEMA, **sch}
    rest = {w for b in allsch if b != a for w in allsch[b].get("topic", [])}
    return rest - own, own - rest


def about_other(clause, sch, a, e, ents):
    """Is this clause about another attribute or another entity than the asked one? "you have liked blue since 2020"
    is about the favourite colour, not the eyes, although both use colour words; "your sister's eyes were blue" is
    about my sister, not me. The asked attribute's own words ("eye") or entity ("your cat") keep the clause."""
    other, own = topic_words(sch, a)
    hit = lambda ws: any(re.search(r"\b" + re.escape(w), clause, re.I) for w in ws)
    if hit(other) and not hit(own):
        return True
    named = lambda x: re.search(ent_patterns(x)[0], clause, re.I)
    if e == "me":
        return any(named(x) for x in ents if x != "me")
    mine = re.search(r"\byour\s+(?:own\s+)?(?:" + "|".join(map(re.escape, own)) + r")", clause, re.I) if own else None
    return not named(e) and bool(mine or any(named(x) for x in ents if x not in ("me", e)))


def dated_claims(text, sch, a, e=None, ents=()):
    """Dated statements about attribute a in an answer: "Until October 11, 2023, your eyes were brown." gives
    {kind: until, vs: [brown], lo..hi: that day}. Kinds: until, before, since, start ("from D1 to D2"), on (some time
    in the period) and change ("They changed on 9 June 2025."). A date is tied to the value named before its "until",
    "since" or "on" in the same clause ("your eyes were brown until D"), or to the first value after it when the date
    opens the sentence ("Until D, your eyes were brown."). When the date opens a later clause and values stand on both
    sides ("Your eyes are green now; until D they were brown."), both readings are kept in vs, and the claim is wrong
    only if both are. Skipped: questions, values after "if"/"whether", negated values ("not brown since 2020"), a bare
    year with no "in"/"since"/"until" before it, values in a clause about another attribute or entity (`about_other`),
    and dates of telling ("you told me on 11 October", "as of my knowledge cutoff in 2023"), since the model is never
    told when a lesson was given."""
    out = []
    for s in sentences(text):
        dates = list(DATE_RE.finditer(s))
        if not dates or s.rstrip().endswith("?"):
            continue
        ms = mentions(s, sch, a)
        for m in ms:
            cl = clause_span(s, m["pos"], m["end"])
            m["off"] = bool(COND.search(s[cl[0]:m["pos"]])) or bool(e and about_other(s[cl[0]:cl[1]], sch, a, e, ents))
        for i, dm in enumerate(dates):
            per = date_period(dm)
            if not per:
                continue
            pre = s[:dm.start()]
            r = REL_RE.search(pre)
            lead = re.fullmatch(r"[\W_]*", pre) and re.match(r"\s*:", s[dm.end():])     # "May 9, 2024: ..." in a list
            if not r and not lead or (dm.group("y5") and not r):
                continue
            rel = rel_of(s, dm) or "on"
            kind = REL_KIND.get(rel)
            if rel == "to" and "from" not in [rel_of(s, d) for d in dates[:i]]:
                kind = None                            # "to D" only ends a "from D1 to D2" range
            if not kind:
                continue
            rs = r.start() if r else dm.start()
            head = s[:rs]
            if TELL.search(" ".join(head.split()[-3:])):
                continue                               # "the information you provided on 11 October 2026"
            after = [m for m in ms if m["pos"] >= dm.end()]
            before = [m for m in ms if m["end"] <= rs]
            if re.fullmatch(r"[\W_]*(?:and|but|so|then)?[\W_]*", head, re.I) or not before:
                chosen = after[:1]                     # "Until D, your eyes were brown."
            elif before[-1]["pos"] >= clause_span(s, rs)[0]:
                chosen = before[-1:]                   # "your eyes were brown until D"
            else:
                chosen = before[-1:] + after[:1]       # "green now; until D they were brown": either reading
            if not chosen:
                if kind == "on" and CHANGE_W.search(s[:rs]):
                    out.append({"kind": "change", "vs": [], "lo": per[0], "hi": per[1], "grain": per[2]})
                continue
            # a negated value ("not brown since 2020"), one after "if", one about another attribute or entity, or one
            # said as told ("May 9, 2024: You mentioned they were amber.") claims nothing about the asked pair on that
            # date, and a reading that claims nothing cannot be wrong
            if any(m["neg"] or m["off"] or TELL.search(s[m["end"]:rs] if m["pos"] < rs else s[dm.end():m["pos"]])
                   for m in chosen):
                continue
            vs = uniq([m["val"] for m in chosen])
            if rel == "from" and "to" in [rel_of(s, d) for d in dates[i + 1:]]:
                kind = "start"                         # "from D1 to D2": a start, not "ever since"
            out.append({"kind": kind, "vs": vs, "lo": per[0], "hi": per[1], "grain": per[2]})
    return out


def date_wrong(sch, store, e, a, c):
    """Does a dated claim contradict what the lessons say? Only a contradiction counts: a claim about a time the
    lessons say nothing about ("green since 2020" after a plain "My eyes are green.") is not wrong, just unsupported.
    The wording's own edge counts too, as for lessons: "until D" says it stopped at D, "since D" that it started
    there. The edge may fall anywhere inside the named day, month or year, or at its end ("until 11 October" is
    read both ways), so only a claim that no point of the period can satisfy is wrong. A claim with two readings
    (vs) is wrong only if both are. A value outside the attribute's list ("navy") was never taught: it is "no" where
    a single value is known, or where nothing holds, and unknown otherwise."""
    bs = sorted(set(boundaries(sch, store, e, a)))
    memo = {}

    def st(p):
        if p not in memo:
            memo[p] = state_at(sch, store, e, a, p)
        return memo[p]

    def tv(p, v):                                   # the truth of v at p
        s = st(p)
        if v in s:
            return s[v]
        return NO if all(x == NO for x in s.values()) or (not sch[a]["multi"] and YES in s.values()) else UNK

    lo, hi = c["lo"], c["hi"]
    inside = [lo] + [b for b in bs if lo < b < hi]
    if c["kind"] == "change":                       # nothing changed in that period, on both sides known
        return all(known(summarize(sch, a, st(p - 1))) and summarize(sch, a, st(p - 1)) == summarize(sch, a, st(p))
                   for p in inside)

    def wrong(v):
        if c["kind"] == "on":
            return all(tv(p, v) == NO for p in inside)
        if c["kind"] == "before":
            return tv(lo - 1, v) == NO
        if c["kind"] == "until":
            return not any(tv(x - 1, v) != NO and tv(x, v) != YES for x in inside + [hi])

        def began(x):                               # since / start: v began at x (and, for since, held until now)
            if tv(x - 1, v) == YES or tv(x, v) == NO:
                return False
            return c["kind"] == "start" or all(tv(p, v) != NO for p in [b for b in bs if b > x] + [NOW])
        return not any(began(x) for x in inside)
    return all(wrong(v) for v in c["vs"])


# ------------------------------------------------------------------------------- readers for the v2 forms
# "True" or "False" opening an answer: "False.", "Definitely false.", "I'd say false.", "This statement is false.",
# "That's not true.", "Incorrect.", "Probably false." (a guess). Not the question echoed back ("True or false? ...").
TF_LEAD = (r"(?:i\s+would\s+say|i'd\s+say|i\s+think|i\s+believe|(?:the|my)\s+answer\s+is|answer|that\s+would\s+be"
           r"|that(?:'s|\s+is|\s+was)|this(?:\s+statement|\s+claim|\s+sentence)?(?:'s|\s+is|\s+was)|it(?:'s|\s+is|\s+was)"
           r"|the\s+(?:statement|claim|sentence)(?:\s+is|\s+was)?|definitely|absolutely|certainly|clearly|simply)")
TF_GUESS = r"(?:probably|maybe|perhaps|possibly|likely)"
TF_START = re.compile(r"^\W*((?:(?:" + TF_LEAD + "|" + TF_GUESS + r")\W+)*)(not\s+)?(true|false|correct|incorrect)\b"
                      r"(?!\s+or\b)", re.I)
TF_ECHO = re.compile(r"^\W*true\s+or\s+false\b\W*", re.I)
# a yes or no that stands alone as an answer: "so no.", "so no, it is not", "the answer is yes". Not "so no answer is
# possible", not "No other colour was mentioned.", and not the question's own "answer yes or no" echoed back
YN_ALONE = (r"(?=\s*(?:[^\w\s]|$)|\s+(?:it|it's|they|they're|he|she|that|this|i|i'm|you|your|yours|because|since"
            r"|as)\b)")
# the yes or no that answers a recall-then-answer question: after "so", "therefore", "the answer is" ...
ANSWER_YN = re.compile(r"\b(?:the\s+answer\s+is|my\s+answer\s+is|answer|so|therefore|thus|hence|that\s+means"
                       r"|which\s+means)\s*[,:]?\s*[\"']?(yes|no)\b" + YN_ALONE, re.I)
YN_FIRST = re.compile(r"^\W*(?:(?:yes|yeah|yep|no|nope)\b" + YN_ALONE + r"|(?:not|probably|maybe|perhaps|possibly"
                      r"|likely)\b)", re.I)
FIRST_PERSON = re.compile(r"^\W*(?:my|i|i'm|i've|i'd)\b", re.I)
YOU_ANY = re.compile(r"\byou(?:rs?|rself)?\b", re.I)
# "Yours.", "It's yours.", "That would be yours.", "I think it's yours."
YOURS = re.compile(r"^\W*(?:i\s+(?:think|believe)\s+)?(?:(?:maybe|perhaps|probably|possibly)\s+)?"
                   r"(?:(?:it(?:'s|\s+is|\s+was)|that(?:'s|\s+is|\s+would\s+be)|the\s+answer\s+is)\s+)?yours\b", re.I)
# a clause that says the value is not known: "yours are unknown to me", "I have no idea about yours"
NOT_KNOWN = re.compile(r"\b(?:unknown|not\s+known|no\s+idea)\b", re.I)


def extract_tf(text):
    """A "True or false: ..." answer. A leading "True" reads as yes and "False" as no, also after a lead-in ("Definitely
    false.", "I'd say false.", "This statement is false.", "That's not true.", "Incorrect."), and the rest is read as a
    yes/no answer would be: "False, I am a language model ..." is about the model, so it means "doesn't know", and
    "Probably false." is a guess. The question echoed back first ("True or false: my favourite colour is purple.
    False.") is skipped; an answer that only echoes it is "ECHO", which the judge calls unclear."""
    ss = sentences(text)
    while ss and TF_ECHO.match(ss[0]):              # the question echoed back
        rest = TF_ECHO.sub("", ss[0])
        if TF_START.match(rest):                     # "True or false: false."
            ss = [rest] + ss[1:]
            break
        if len(ss) == 1:
            return {"got": "ECHO"}
        ss = ss[1:]
    m = TF_START.match(ss[0]) if ss else None
    if not m:
        return extract_yn(" ".join(ss))
    yes = (m.group(3).lower() in ("true", "correct")) != bool(m.group(2))    # "not true", "not false"
    guess = re.search(r"\b" + TF_GUESS + r"\b", m.group(1) or "", re.I)
    word = (guess.group(0) + " " + ("yes" if yes else "no")) if guess else ("Yes" if yes else "No")
    return extract_yn(" ".join([word + ss[0][m.end():]] + ss[1:]))


# A plain yes/no answer (levels 1 to 5, and the plain questions of other levels) that opens with "true" or "false"
# ("It is false.", "That is false.", "False.", "That's true.", "Not true.", "Incorrect.", "Probably false.") is read as
# the true/false form reads it. Before 13 Oct the plain reading took "It is false." as a yes (it starts like "It is
# ...") and "False." as no answer (TEST-LOGIC.md section 14). Compound questions keep their own reading in
# zengine/levels.py. Switched off in the self-test's negative control.
TRUTH_PLAIN = True


def truth_opening(text):
    """Does a plain yes/no answer open with "true" or "false" (TF_START, lead-ins and guesses included)?"""
    ss = sentences(text)
    return TRUTH_PLAIN and bool(ss) and bool(TF_START.match(ss[0])) and not TF_ECHO.match(ss[0])


def extract_recall_yn(text, sch, a, v):
    """A "First say what X is, then answer yes or no: is it v?" answer. The answer is the last yes or no it gives: after
    "so", "therefore" or "the answer is", or a later sentence that opens with yes or no ("Your favourite colour is
    blue. No."), else a first sentence that does. With none, a refusal is "doesn't know"; otherwise it is unclear, and
    the judge reads the value as for a plain yes/no question. The value named first is the recall, kept in "recalled"
    so the report can say whether the yes or no follows from it."""
    ss = sentences(text)
    if not ss:
        return {"got": "EMPTY"}
    flat = " ".join(ss)
    marks = list(ANSWER_YN.finditer(flat))
    later = [g for g in (extract_yn(s) for s in ss[1:] if YN_FIRST.match(s)) if g["got"] != "UNCLEAR"]
    if marks:
        g = {"got": "YN", "value": YES if marks[-1].group(1).lower() == "yes" else NO}
    elif later:
        g = later[-1]
    elif YN_FIRST.match(ss[0]):
        g = extract_yn(ss[0])
    elif REFUSAL.search(flat):
        g = {"got": "YN", "value": UNK}
    else:
        g = {"got": "UNCLEAR"}
    return dict(g, recalled=extract_value(text, sch, a, "present"))


def self_voiced(text):
    """A fill-in ('Finish this sentence: "My favourite colour is ..."') that the model finished about itself: it opens
    in the first person and then asks the user back ("My favourite colour is blue! What's yours?"). The value is the
    model's own, not a recall of the user's. An answer tied to memory ("as you told me") is a recall, and one that
    opens with a refusal ("I don't know. What's your favourite colour?") is a refusal that asks back."""
    ss = sentences(text)
    return bool(ss) and bool(FIRST_PERSON.match(ss[0])) and not REFUSAL.search(ss[0]) \
        and not MEMORY_CUE.search(" ".join(ss)) and any(s.rstrip().endswith("?") and YOU_ANY.search(s) for s in ss[1:])


def your_own(text, sch, a, v, ents, names):
    """On a reverse question, the user named by what is theirs: "Yours.", "It's yours.", "Your eyes are grey.", "Your
    favourite colour is blue." (`you_named` reads "you": "You do."). A question, a condition, a refusal, a clause that
    names another of the set's entities ("your dog's eyes") or that negates the value or names another one does not
    count. Returns "found", "hedged" (inside a guess) or None."""
    topic = "|".join(map(re.escape, sch[a].get("topic", [])))
    own = re.compile(r"\byour\s+(?:own\s+)?(?:" + topic + r")", re.I) if topic else None
    out = None
    for s in sentences(text):
        asked = s.rstrip().endswith("?")
        if asked and ASK_BACK.match(s) or IF_FIRST.match(s):
            continue                                 # "What's yours?", "If yours are grey, ..."
        guess = (bool(GUESS.search(s)) or asked) and not MEMORY_CUE.search(s)     # "Perhaps yours?" is a guess
        cuts = [0] + [x for c in CLAUSE.finditer(s) for x in (c.start(), c.end())] + [len(s)]
        for c0, c1 in zip(cuts[::2], cuts[1::2]):
            c = s[c0:c1]
            if not c.strip() or REFUSAL.search(c) or NOT_KNOWN.search(c) or COND.search(c):
                continue
            ms = mentions(c, sch, a)
            vals = {m["val"] for m in ms if not m["neg"]}
            if any(m["val"] == v and m["neg"] for m in ms) or (vals and v not in vals) \
                    or any(re.search(p, c, re.I) for e in ents if e != "me" for p in ent_pats(e, names)):
                continue
            if YOURS.match(c) or (own and own.search(c) and v in vals):
                if not guess:
                    return "found"
                out = "hedged"
    return out


def with_yours(g, text, sch, a, v, ents, names):
    """extract_entities' reading of a reverse answer, plus the user where the answer names them by what is theirs
    (`your_own`). Used for the v2 reverse forms only, so v1 sets are judged as before."""
    if "me" not in ents or g["got"] in ("EMPTY", "OFFTOPIC"):
        return g
    y = your_own(text, sch, a, v, ents, names)
    if not y:
        return g
    if y == "hedged" or g["got"] == "HEDGED":
        return {"got": "HEDGED", "values": uniq((g.get("values") or g.get("yes") or []) + ["me"])}
    return {"got": "ENTS", "yes": uniq((g.get("yes") if g["got"] == "ENTS" else []) + ["me"])}


def who_yn_reading(S, q, content):
    """A reverse yes/no ("Is it my dog who has grey eyes?") answered with no yes or no word is read for who it names:
    "Your dog has grey eyes." is yes, "Your cat has grey eyes." is no, "I don't know who has grey eyes." doesn't know.
    The value alone says nothing here, since the question gives it."""
    sch, a, v = S["schema"], q["a"], q["v"]
    ge = with_yours(extract_entities(content, S["entities"], sch, a, v, ent_names(S, a)), content, sch, a, v,
                    S["entities"], ent_names(S, a))
    if ge["got"] == "ENTS":
        return {"got": "YN", "value": (YES if q["e"] in ge["yes"] else NO) if ge["yes"] else UNK}
    if ge["got"] == "HEDGED":
        return {"got": "HEDGED", "value": YES if q["e"] in ge["values"] else NO}
    return ge if ge["got"] == "OFFTOPIC" else {"got": "UNCLEAR"}


def names_asked(S, q, content):
    """Does a reverse yes/no answer talk about the entity the question asks about ("Your cat's eyes are hazel." to "Is
    it my cat who has grey eyes?")? Then, with no yes, no or holder in it, the judge reads the value it gives that
    entity, as for a plain yes/no question. Without this, an answer about someone else ("Grey eyes are common.") would
    be read for its value too."""
    e, a = q["e"], q["a"]
    if e == "me":
        topic = "|".join(map(re.escape, S["schema"][a].get("topic", [])))
        return bool(topic) and bool(re.search(r"\byour\s+(?:own\s+)?(?:" + topic + r")", content, re.I))
    return any(re.search(p, content, re.I) for p in ent_pats(e, ent_names(S, a)))


# ------------------------------------------------------------------------------------- the first person (13 Oct)
# "I do not like olives anymore." is the user's own lesson said back, in the user's words (the lessons are in the first
# person), not the model about itself: the model has no likes, eyes or home. It is read as about the user, as "You do
# not like olives anymore." is, so it is a no. The model talking about itself is a self-reply, as "No, I am a language
# model ..." is (SELF_REPLY): "doesn't know". It talks about itself when it says so ("Personally, ...", "as an AI",
# "I don't have eyes") or asks the user back ("I love olives! Do you like them too?", "My favourite colour is blue!
# What's yours?"). Plain questions only: a fill-in is finished in the user's own voice (see self_voiced).
FP_LEAD = r"^\W*(?:(?:well|oh|ah|honestly|actually|personally|frankly|haha|ha|hmm+)\W+)*"
FIRST_STMT = re.compile(FP_LEAD + r"(?:i|i'm|i've|my)\b", re.I)
# first-person openings that are the model's own act, not a fact said: "I think you ...", "I remember ...", "I'd say
# ...", "I only know ...", "I'm sorry", "My guess is ..."
FIRST_ACT = re.compile(FP_LEAD + r"(?:i\s+(?:(?:do\s+not|don't|really|also|just|actually|only|would|can|could)\s+)*"
                       r"(?:think|believe|suppose|guess|know|knew|remember|recall|see|mean|understand|hope|wish|bet"
                       r"|wonder|assume|imagine|feel|told|said|heard|say|tell|recommend|suggest|need|want|would|will"
                       r"|can|could|may|might|should|must|love\s+to|like\s+to)\b|i'd\b|i'll\b"
                       r"|i(?:'m|\s+am)\s+(?:not\s+sure|sorry|happy|glad|afraid|here|unable|not\s+able|curious|just"
                       r"|an?\b|going)|my\s+(?:name|apologies|guess|answer|knowledge|training|best\s+guess|pleasure))\b",
                       re.I)
SELF_MARK = re.compile(r"\b(?:personally|myself|my\s+own|for\s+me|as\s+for\s+me|me\s+too|me\s+neither"
                       r"|as\s+an?\s+(?:ai|language\s+model|assistant|chatbot)|nemotron"
                       r"|i(?:'m|\s+am)\s+(?:just\s+|only\s+|simply\s+)?an?\s+(?:large\s+)?(?:ai|language\s+model"
                       r"|assistant|chatbot|computer\s+program|machine)"
                       r"|i\s+(?:do\s+not|don't)\s+(?:actually\s+|really\s+)?have\s+(?:eyes|a\s+body|a\s+physical"
                       r"|physical|a\s+face|a\s+home|a\s+favou?rite|favou?rite|personal\s+preferences|preferences|tastes?"
                       r"|a\s+pet|pets|a\s+family|family))\b", re.I)
# a later sentence that asks the user back about themselves ("What about you?", "What's yours?", "Do you like them?"),
# not one that offers help ("Can I help you with anything else?")
ASK_YOU = re.compile(r"\b(?:yours|about\s+you|and\s+you|do\s+you|are\s+you|have\s+you"
                     r"|you\s+(?:like|love|prefer|enjoy|have|live))\b", re.I)
FIRST_TO_YOU = [(r"\bI'm\b", "you're"), (r"\bI\s+am\b", "you are"), (r"\bI've\b", "you've"), (r"\bI\b", "you"),
                (r"\bmyself\b", "yourself"), (r"\bmine\b", "yours"), (r"\bmy\b", "your"), (r"\bme\b", "you")]


def first_person(text):
    """A plain answer in the first person: "echo" when it says a fact in the user's own words ("I do not like olives
    anymore.", "My eyes are green.", "I don't."), "self" when the model talks about itself ("Personally, I do not like
    olives anymore. What about you?", "I love olives! Do you like them too?", "I don't have eyes."), None otherwise:
    no first person, a refusal ("I don't know"), the model's own act ("I think you like blue", "I remember ..."), a
    guess, an answer tied to what the user said ("as you told me"), or the model introducing itself (SELF_TALK)."""
    ss = sentences(text)
    if not ss:
        return None
    f = ss[0]
    if not FIRST_STMT.match(f) or FIRST_ACT.match(f) or REFUSAL.search(f) or HEDGE.search(f) or GUESS.search(f) \
            or SELF_TALK.search(f) or MEMORY_CUE.search(" ".join(ss)):
        return None
    asks = any(s.rstrip().endswith("?") and ASK_YOU.search(s) and not re.search(r"\b(?:help|assist|else)\b", s, re.I)
               for s in ss[1:])
    return "self" if SELF_MARK.search(" ".join(ss[:2])) or asks else "echo"


# A first-person answer that also says something of the user for the same attribute: "I do not like olives anymore.
# You still do.", "I don't like olives, but you love them.", "My favourite colour is green. Yours is blue." The "I" is
# then the model itself, so the first person is a self-reply, and what is said of the user decides (an audit of
# 0c782f4). A clause that only repeats the predicate ("You still do.", "and neither do you", "you love them") stands
# for the first person's statement, said of the user: "you like olives", "you do not like olives anymore".
YOU_OPEN = re.compile(r"^\W*(?:(?:and|but|while|whereas|although|though|however|so|yet|nor|neither)\W+)*"
                      r"(?:(?:so|neither|nor)\s+(?:do|does|did|are|is|was|were|have|has)\s+you(?:rs)?\b|you(?:rs)?\b"
                      r"|your\s)", re.I)
REPEAT_REST = re.compile(r"[\W_]*(?:(?:and|but|while|whereas|although|though|however|so|yet|as|also|too|still|either"
                         r"|neither|nor|not|never|no|longer|any|more|anymore|well|both|really|do|does|did|don't|doesn't"
                         r"|didn't|is|are|isn't|aren't|was|were|has|have|had|like|likes|love|loves|enjoy|enjoys|them"
                         r"|it|they|one|the|same)\b[\W_]*)+", re.I)
YOU_NO = re.compile(r"\b(?:neither|nor|not|never|no\s+longer)\b|n't\b", re.I)


def you_part(text, sch, a, v=None):
    """What a first-person answer says of the user for attribute a, read apart from the model's own words: the rest of
    the sentence from the first clause that has the user as its subject and names a value ("You like blue.", "Yours
    is blue."), or, for a clause that only repeats the predicate ("You still do.", "and neither do you"), the first
    sentence's statement said of the user; with no value in the first sentence ("I don't. You do."), the yes/no
    question's value v. None when the answer says nothing of the user, asks back ("Do you?") or names the user
    otherwise ("You told me so.", "You are right to ask.")."""
    ss = sentences(text)
    yt = your_topic(sch, a)
    first = uniq([m["val"] for m in mentions(ss[0], sch, a)]) or ([v] if v else [])
    for i, s in enumerate(ss):
        if s.rstrip().endswith("?"):
            continue
        cuts = [0] + [x for c in WHO_CLAUSE.finditer(s) for x in (c.start(), c.end())] + [len(s)]
        for j, (c0, c1) in enumerate(zip(cuts[::2], cuts[1::2])):
            c = s[c0:c1]
            if (i, j) == (0, 0) or not c.strip() or not YOU_OPEN.match(c):
                continue
            # the separator before a clause belongs to it ("and neither do you", "but you love them")
            lead = s[cuts[2 * j - 1]:c0] if j else ""
            if [m for m in mentions(c, sch, a)]:
                if (yt and yt.search(c)) or YOU_HOLD.search(c) or re.search(r"\byours\b", c, re.I):
                    return s[c0:].strip()                 # "You like blue.", "Yours is blue."
                continue
            rest = re.sub(r"\byou(?:rs)?\b", " ", c, flags=re.I)
            if first and REPEAT_REST.fullmatch(rest):     # "You still do.", "and neither do you"
                tmpl = T[a]["RETRACT" if YOU_NO.search(lead + c) else "ASSERT"]
                return as_user(render(tmpl[0], "me", v=" and ".join(first))) if tmpl else None
    return None


def as_user(text):
    """The first sentence with the first person turned into the second: "I do not like olives anymore." -> "you do
    not like olives anymore.", so it is read as the same answer about the user would be."""
    ss = sentences(text)
    f = ss[0]
    for pat, rep in FIRST_TO_YOU:
        f = re.sub(pat, rep, f, flags=re.I)
    return " ".join([f] + ss[1:])


def judge_one(S, phase, q, content):
    """judge_core's verdict, then the dates: a dated statement about the asked pair that contradicts the lessons
    ("Until October 11, 2023, your eyes were brown." after "... brown until 11 October 2026 ...") fails as a wrong
    date, whatever else the answer got right. The question's own date ("before 11 October 2026") is not checked
    again: judge_core already read the value it asks for. Answers without a date are judged as before.
    Two-step questions (level 6) are judged by zengine/levels.py, which adds the verdict "echo"."""
    if q["type"] in TWO_STEP:
        return _levels().judge_two_step(S, phase, q, content)
    verdict, g, why = judge_core(S, phase, q, content)
    if verdict == "skip" or not q.get("e") or not q.get("a") or not (content or "").strip():
        return verdict, g, why
    whose = said_of_other(S, q, content, g) if verdict != "fail" else None
    if whose:                                          # "Your cat's eyes are grey." for my dog's eyes
        g = dict(g, said_of=whose[1])
        if whose[0] == "fail":
            verdict, why = "fail", "said of another entity"
        elif verdict == "pass":
            verdict, why = "unclear", "also said of another entity"
    asked = tick(q["date"]) if q.get("date") else None
    store = [l for l in S["lessons"] if l["id"] in phase_store(S, phase)]
    bad = [c for c in dated_claims(content, S["schema"], q["a"], q["e"], S["entities"])
           if not (asked is not None and c["lo"] <= asked < c["hi"]) and date_wrong(S["schema"], store, q["e"], q["a"], c)]
    if not bad:
        return verdict, g, why
    g = dict(g, dates=[(f"{' or '.join(c['vs'])} {c['kind']}" if c["vs"] else "a change on")
                       + f" {iso_of(c['lo'] // TPD)}" + ("" if c["grain"] == "day" else f" ({c['grain']})") for c in bad])
    return "fail", g, ("wrong date" if verdict != "fail" else f"{why}; wrong date")


def hedged(g, right):
    """A guess about the user. Decided 13 Oct 2026: the right answer said as a guess ("Your eyes are probably
    green.") is unclear, for a person to look at; a wrong guess fails. A guess where the right answer is "don't
    know" names a value nobody taught, so it fails (see judge_core)."""
    return ("unclear", g, "the right answer, said as a guess") if right else ("fail", g, "a guess, not a recall")


def judge_core(S, phase, q, content):
    """The verdict on one answer, before the date check of `judge_one`."""
    sch, ents = S["schema"], S["entities"]
    exp = S["expected"][phase][q["qid"]]
    k, t = exp["kind"], q["type"]
    if q.get("indirect") and k in ("unknown", "none"):
        return "skip", {"got": "SKIP"}, "indirect question, nothing to recall here"
    if not (content or "").strip():
        return "unclear", {"got": "EMPTY"}, "empty answer"
    if k == "gk":
        low = clean(content).lower()
        ok = any(all(re.search(r"(?<!\w)" + re.escape(w) + r"(?!\w)", low) for w in s) for s in exp["any_of"])
        return ("pass" if ok else "fail"), {"got": "GK"}, "" if ok else "general knowledge changed"
    # the first person, on plain questions (not the v2 forms, not compound questions)
    fp = first_person(content) if not q.get("form") and t != "COMPOUND" and k != "entities" else None
    if fp and q.get("e") == "me" and "self" in sch[q["a"]]["applies"]:
        yp = you_part(content, sch, q["a"], q.get("v"))
        if yp:                                         # "I do not like olives anymore. You still do.": the user's part
            content, fp = yp, None
    if k in ("value", "unknown", "none"):
        if q.get("form") == "fill_in" and self_voiced(content):   # "My favourite colour is blue! What's yours?"
            return "unclear", {"got": "SELF"}, "finished the sentence about itself"
        g = {"got": "ABSENT", "self": True} if fp == "self" else \
            extract_value(content, sch, q["a"], "present" if t == "CURRENT" else "past")
        if g["got"] == "VALUE":
            if k == "value":
                return ("pass", g, "") if g["value"] == exp["value"] else ("fail", g, "wrong value")
            return "fail", g, ("value it should not give" if g["value"] in exp.get("not", []) else "invented a value")
        if g["got"] == "ABSENT":
            return ("pass", g, "") if k != "value" else ("fail", g, "does not know a taught value")
        if g["got"] == "OFFTOPIC":
            return "fail", g, "off topic"
        if g["got"] == "HEDGED":
            if k != "value":
                return "fail", g, "guessed a value"
            return hedged(g, g.get("values") == [exp["value"]])
        return "unclear", g, g["got"].lower()
    if k == "set":
        g = {"got": "ABSENT", "self": True} if fp == "self" else extract_set(content, sch, q["a"])
        if g["got"] == "HEDGED":
            return hedged(g, sorted(g.get("values", [])) == sorted(exp["yes"]))
        if g["got"] == "SET":
            missing = [v for v in exp["yes"] if v not in g["yes"]]
            extra = [v for v in g["yes"] if v not in exp["yes"]]
            if not missing and not extra:
                return "pass", g, ""
            return "fail", g, "; ".join(x for x in [f"missing {missing}" if missing else "",
                                                  f"extra {extra}" if extra else ""] if x)
        if g["got"] == "ABSENT":
            return "fail", g, "does not know a taught value"
        if g["got"] == "OFFTOPIC":
            return "fail", g, "off topic"
        return "unclear", g, g["got"].lower()
    if k == "yn":
        form = q.get("form")
        plain_tf = form not in ("true_false", "recall_yn") and t != "COMPOUND" and truth_opening(content)
        if fp == "echo":                                 # "I do not like olives anymore.": read as about the user
            content = as_user(content)
        g = ({"got": "YN", "value": UNK, "self": True} if fp == "self" else
             extract_tf(content) if form == "true_false" or plain_tf else
             extract_recall_yn(content, sch, q["a"], q["v"]) if form == "recall_yn" else extract_yn(content))
        if g["got"] == "UNCLEAR" and form == "who_yn":   # no yes or no word: read who it names
            g = who_yn_reading(S, q, content)
        if g["got"] == "UNCLEAR" and t == "YES_NO" and (form != "who_yn" or names_asked(S, q, content)):
            # no yes or no word (and, on a reverse yes/no, no holder named): read the value it names instead
            gv = extract_value(content, sch, q["a"], "present")
            negv = any(m["neg"] and m["val"] == q["v"]
                       for m in mentions(" ".join(sentences(content)[:2]), sch, q["a"]))
            if gv["got"] == "VALUE" and (gv["value"] == q["v"] or not sch[q["a"]]["multi"]):
                g = {"got": "YN", "value": YES if gv["value"] == q["v"] else NO}   # "Your eyes are brown." = no
            elif negv:
                g = {"got": "YN", "value": NO}                                      # "They are not green anymore."
            elif gv["got"] == "HEDGED":                                         # "Your eyes are probably green."
                vals = gv.get("values", [])
                g = dict(gv, value=YES if vals == [q["v"]] else
                         NO if vals and q["v"] not in vals and not sch[q["a"]]["multi"] else None)
            elif gv["got"] in ("ABSENT", "OFFTOPIC"):
                g = {"got": "YN", "value": UNK} if gv["got"] == "ABSENT" else gv
        if form == "recall_yn" and "recalled" not in g:  # the value reading above replaced it: keep the recall
            g = dict(g, recalled=extract_value(content, sch, q["a"], "present"))
        if g.get("likely") and t == "YES_NO" and not form and not sch[q["a"]]["multi"]:
            # "It is likely that your eyes are brown." on "Are my eyes green?": a guess about another value leans no;
            # "It is unlikely that your eyes are brown." leans nowhere on green
            vals = uniq([m["val"] for m in mentions(sentences(content)[0], sch, q["a"]) if not m["neg"]])
            if vals and q["v"] not in vals:
                g = dict(g, value=NO if g["likely"] == YES else None)
        if g["got"] == "HEDGED":
            return hedged(g, exp["value"] in (YES, NO) and g.get("value") == exp["value"])
        if g["got"] == "OFFTOPIC":
            return "fail", g, "off topic"
        if g["got"] != "YN":
            return "unclear", g, "no yes or no"
        return ("pass", g, "") if g["value"] == exp["value"] else ("fail", g, "wrong yes/no")
    if k == "choice":
        g = {"got": "CHOICE", "value": "unknown", "self": True} if fp == "self" else \
            extract_choice(content, sch, q["a"], (q["x"], q["y"]))
        if g["got"] == "HEDGED":
            return hedged(g, g.get("values") == [exp["value"]])
        if g["got"] != "CHOICE":
            return ("fail" if g["got"] == "OFFTOPIC" else "unclear"), g, g["got"].lower()
        if g["value"] == exp["value"]:
            return "pass", g, ""
        store = [l for l in S["lessons"] if l["id"] in phase_store(S, phase)]
        cur = sum_at(sch, store, q["e"], q["a"], NOW)
        if exp["value"] == "neither" and cur["kind"] == "value" and g["value"] == cur["value"]:
            return "pass", g, "named the actual value"
        return "fail", g, "wrong choice"
    if k == "entities":
        g = extract_entities(content, ents, sch, q["a"], q.get("v"), ent_names(S, q["a"]))
        if q.get("form") in ("who_bare", "who_listed"):      # forms v2: "Yours.", "Your eyes are grey." name you
            g = with_yours(g, content, sch, q["a"], q.get("v"), ents, ent_names(S, q["a"]))
        if g["got"] == "HEDGED":
            return hedged(g, bool(exp["yes"]) and sorted(g["values"]) == sorted(exp["yes"]))
        if g["got"] == "OFFTOPIC":
            return "fail", g, "off topic"
        if g["got"] == "OTHERS":
            return "unclear", g, "names someone the test never taught about"
        if g["got"] not in ("ENTS", "GENERAL"):
            return "unclear", g, "no entity found"
        general = "a general answer: names none of yours" if g["got"] == "GENERAL" else ""
        if sorted(g["yes"]) == sorted(exp["yes"]):
            return "pass", g, general
        return "fail", g, (general or "does not know a taught value") if not g["yes"] else "wrong entities"
    return "unclear", {"got": "?"}, "unknown kind"


def phase_store(S, name):
    return next(p["store"] for p in S["phases"] if p["name"] == name)


def got_words(g):
    w = got_words_core(g)
    if (g.get("recalled") or {}).get("got") in ("VALUE", "ABSENT"):   # forms v2, recall then answer: what it said
        w += f" (recalled: {got_words_core(g['recalled'])})"
    if g.get("said_of"):
        w += f" (said of {whose_words(g['said_of'])})"
    return w + (f" (dated: {'; '.join(g['dates'])})" if g.get("dates") else "")


def got_words_core(g):
    x = g.get("got")
    if x == "VALUE":
        return g["value"]
    if x == "ABSENT":
        return "doesn't know"
    if x == "SET":
        return ", ".join(g["yes"]) + (f" (not {', '.join(g['no'])})" if g.get("no") else "")
    if x == "YN":
        return {YES: "yes", NO: "no", UNK: "doesn't know"}[g["value"]]
    if x == "CHOICE":
        return {"unknown": "doesn't know"}.get(g["value"], g["value"])
    if x == "ENTS":
        return ", ".join(g["yes"]) or "nobody"
    if x == "AMBIGUOUS":
        return "several: " + ", ".join(g["values"])
    if x == "HEDGED":
        if g.get("values"):
            return "a guess: " + ", ".join(g["values"])
        return "a guess" + (": " + {YES: "yes", NO: "no"}[g["value"]] if g.get("value") in (YES, NO) else "")
    if x == "GENERAL":
        return "nobody of yours (a general answer)"
    if x == "OTHERS":
        return "someone not in the test: " + ", ".join("your " + w for w in g["values"])
    if x == "OFFTOPIC":
        return "off topic"
    if x == "SELF":
        return "about itself"
    if x == "ECHO":
        return "the question echoed back"
    return (x or "?").lower()


def bucket(q):
    if q["role"] != "probe":
        return "controls"
    if q.get("form") == "who_yn":                    # forms v2: the reverse question asked as yes/no
        return "reverse"
    return {"CURRENT": "recall", "AT": "temporal", "BEFORE": "temporal", "PREVIOUS": "temporal", "ALWAYS": "temporal",
            "CHANGED": "temporal", "EITHER_OR": "either_or", "YES_NO": "yes_no", "WHO_HAS": "reverse",
            "CHAIN": "two_step", "COMPOUND": "compound"}[q["type"]]


BUCKETS = ["recall", "temporal", "either_or", "yes_no", "reverse", "reverse_fwd", "two_step", "compound", "controls",
           "delete_affected", "delete_kept"]

# ------------------------------------------------------------------------- split scores (13 Oct 2026, plan items 2, 8)
# Every answer is also sorted by what the rule book expects ("class") and by what the model gave ("gave"). A pooled
# score hides a model that refuses everything: where every right answer is "don't know" (the L3 deletes), refusing
# passes them all. Split by class, the same model scores zero on the questions whose right answer is a value.
CLASSES = ["value", "no", "dont_know"]
CLASS_WORDS = {"value": "a value", "no": "no", "dont_know": "don't know"}
# the names the editing literature uses (ROME's CounterFact, KnowEdit, GRACE, WISE, MEMOIR); see question_tags
TAGS = ["reliability", "generality", "locality", "portability"]
# the two dummy systems: each answers every question the same way. They are the negative controls of the split view:
# "always don't know" passes every question whose right answer is "don't know", "always yes" every right yes
DUMMIES = {"dont_know": "I don't know.", "yes": "Yes."}


def exp_class(x):
    """What the rule book expects: "value" (a value, a set, yes, one of the two options, named entities, the textbook
    answer), "no" (no, neither, none) or "dont_know" (unknown, also "nobody known" on a reverse question)."""
    k = x["kind"]
    if k in ("value", "set", "gk"):
        return "value"
    if k == "none":
        return "no"
    if k == "unknown":
        return "dont_know"
    if k == "yn":
        return {YES: "value", NO: "no", UNK: "dont_know"}[x["value"]]
    if k == "choice":
        return {"neither": "no", "unknown": "dont_know"}.get(x["value"], "value")
    if k == "entities":
        return "value" if x["yes"] else "dont_know"
    raise ValueError(k)


def absent_gave(content):
    """An "absent" reading as the reader found it: it widens from the first sentence to three, then to the whole
    answer, and stops at the first scope that refuses or says the value is gone. "dont_know" when that scope refuses
    ("I don't know"), "no" when it only says the value is gone ("You don't have one any more"). Fixed 13 Oct (review):
    the first version looked at three sentences only, so a refusal in the fourth sentence, which the reader does
    find, was counted as a "no"."""
    ss = sentences(content)
    for k in sorted({1, 2, 3, len(ss)}):
        scope = " ".join(ss[:k])
        if REFUSAL.search(scope):
            return "dont_know"
        if ABSENT.search(scope):
            return "no"
    return "dont_know"


def gave_class(g, verdict, content):
    """What the model gave, read from the judge's reading g: "value", "no", "dont_know", or "guess", "off_topic" and
    "unclear" for answers that are none of the three. A reading of "absent" is "dont_know" when the answer refuses
    ("I don't know") and "no" when it says the value is gone ("You don't have one any more"), see absent_gave."""
    x = g.get("got")
    if x in ("VALUE", "SET"):
        return "value"
    if x == "ABSENT":
        return absent_gave(content)
    if x == "YN":
        return {YES: "value", NO: "no", UNK: "dont_know"}[g["value"]]
    if x == "CHOICE":
        return {"neither": "no", "unknown": "dont_know"}.get(g["value"], "value")
    if x == "ENTS":
        return "value" if g["yes"] else "dont_know"
    if x == "GENERAL":
        return "dont_know"                             # names nobody the user knows
    if x == "GK":
        return "value" if verdict == "pass" or not REFUSAL.search(clean(content)) else "dont_know"
    if x == "HEDGED":
        return "guess"
    if x == "OFFTOPIC":
        return "off_topic"
    return "unclear"


def ref_phase(name, have=None):
    """The phase whose right answers a phase is compared with, for "should change": the teach phase with the baseline,
    a delete phase with the teach phase. The baseline has none, and neither has a phase whose reference is not in
    `have` (the set's expected answers): a hand-made set may have no baseline phase."""
    ref = None if name == "baseline" else "baseline" if name == "teach" else "teach"
    return ref if have is None or ref in have else None


def question_tags(S):
    """Each question's name in the editing literature, from the set alone (the same in every phase):
    reliability  the plain question about a taught (entity, attribute), in its first wording. The lesson sentence
                 itself is never asked back, so this is the nearest to "the taught wording".
    generality   every other question about a taught pair: other wordings, yes/no, either-or, past and "always".
    portability  reverse questions ("Who has grey eyes?") and indirect ones ("If you were buying me a shirt ...").
    locality     the controls (near misses, untouched attributes, general knowledge) and pairs no lesson is about."""
    taught = {(l["e"], l["a"]) for l in S["lessons"]}
    first, out = set(), {}
    for q in S["questions"]:
        key = (q.get("e"), q["a"])
        if q["role"] != "probe" or q["type"] == "GK":
            out[q["qid"]] = "locality"
        elif q["type"] == "WHO_HAS" or q.get("indirect"):
            out[q["qid"]] = "portability"
        elif key not in taught:
            out[q["qid"]] = "locality"
        elif q["type"] == "CURRENT" and key not in first:
            first.add(key)
            out[q["qid"]] = "reliability"
        else:
            out[q["qid"]] = "generality"
    return out


def right_values(x):
    """The values a right answer names."""
    k = x["kind"]
    if k == "value":
        return {x["value"]}
    if k == "set":
        return set(x["yes"])
    if k == "choice" and x["value"] not in ("neither", "unknown"):
        return {x["value"]}
    return set()


def lesson_taught_values(l):
    """The values a lesson teaches about its entity. A lesson that takes a value back teaches none. (Not
    taught_values(S, e, a) of forms v2, which lists every value the lessons on one key name.)"""
    return [] if l["op"] == "RETRACT" else [v for v in (l.get("v"), l.get("v0"), l.get("v1")) if v]


def neighbour_leak(S, phase, q, g, verdict):
    """A failed answer about (e, a) that names, or says yes to, a value this phase's lessons teach about another of
    the set's entities: "Is my cat called Bruno or Nora?" answered Bruno, the dog's name. CounterFact calls this a
    neighbourhood failure. Returns {"values": [...], "from": [...]} or None. A value that is also right for (e, a) is
    not a leak (both pets can have grey eyes), nor one that (e, a)'s own lessons name ("Do I like olives?" answered
    yes after "I do not like olives anymore": a stale value of its own), and a lesson that takes a value back does
    not teach it."""
    if verdict != "fail" or not q.get("e") or not q.get("a"):
        return None
    store = [l for l in S["lessons"] if l["id"] in phase_store(S, phase) and l["a"] == q["a"]]
    own = {v for l in store if l["e"] == q["e"] for v in (l.get("v"), l.get("v0"), l.get("v1")) if v}
    taught = {}
    for l in store:
        if l["e"] != q["e"]:
            for v in lesson_taught_values(l):
                if v not in own:                       # a value of its own past is its own, not a neighbour's
                    taught.setdefault(v, [])
                    if l["e"] not in taught[v]:
                        taught[v].append(l["e"])
    x = g.get("got")
    named = ([g["value"]] if x in ("VALUE", "CHOICE") else g.get("yes", []) if x == "SET" else
             g.get("values") or [] if x == "HEDGED" else
             [q["v"]] if x == "YN" and g["value"] == YES and q["type"] == "YES_NO" else [])
    right = right_values(S["expected"][phase][q["qid"]])
    hits = [v for v in named if v in taught and v not in right]
    if not hits:
        return None
    return {"values": hits, "from": uniq([e for v in hits for e in taught[v]])}


def reading_key(g):
    """An answer's reading in a form two answers can be compared by (paraphrase agreement): ("value", v),
    ("set", ...), ("yn", v), ("choice", v), ("ents", ...), "dk" for "doesn't know", or None for a reading that is not
    an answer (a guess, off topic, unclear, empty)."""
    x = g.get("got")
    if x == "VALUE":
        return ("value", g["value"])
    if x == "SET":
        return ("set", tuple(sorted(g["yes"])))
    if x in ("ABSENT", "GENERAL"):
        return "dk"
    if x == "YN":
        return "dk" if g["value"] == UNK else ("yn", g["value"])
    if x == "CHOICE":
        return "dk" if g["value"] == "unknown" else ("choice", g["value"])
    if x == "ENTS":
        return ("ents", tuple(sorted(g["yes"]))) if g["yes"] else "dk"
    return None


def plain_wording(q):
    """A probe question asked straight, not an indirect one ("If you were buying me a shirt, what colour should it
    be?"). Consistency and paraphrase agreement compare only these."""
    return q["role"] == "probe" and not q.get("indirect")


def paraphrase_key(q):
    """Questions that ask the same thing in other words share this key: the CURRENT wordings of one pair, the two
    PREVIOUS wordings, and an either-or question in both orders."""
    return (q["type"], q.get("e"), q["a"], q.get("v"), frozenset((q["x"], q["y"])) if "x" in q else None, q.get("date"))


def check_expected(S):
    """Recompute every expected answer; a mismatch means the set file was edited or the oracle changed."""
    keep = json.dumps(S["expected"], sort_keys=True)
    T2 = dict(S)
    fill_expected(T2)
    return json.dumps(T2["expected"], sort_keys=True) == keep


def load_answers(specs, phases):
    rows, seen = [], set()
    for spec in specs:
        path, ph = spec, None
        if ":" in spec and spec.rsplit(":", 1)[1] in phases:
            path, ph = spec.rsplit(":", 1)
        for line in open(path):
            if not line.strip():
                continue
            r = json.loads(line)
            p = r.get("phase") or ph
            qid = r.get("qid") or r.get("id")
            key = (p, qid, bool(r.get("thinking")), r.get("content"))
            if key in seen:
                continue
            seen.add(key)
            rows.append({"phase": p, "qid": qid, "thinking": bool(r.get("thinking")), "content": r.get("content") or "",
                         "question": r.get("question")})
    return rows


def derived_consistency(results, qs, every_wording=True):
    """{("cons", phase): {mode: {"value" | "dont_know": {"n", "ok", "not"}}}}: does each yes/no and either-or answer
    follow from the model's own answer to the plain question (CURRENT) on the same pair, phase and thinking mode? One
    pair per wording of the plain question; the part is what that plain answer gave. `every_wording=False` keeps only
    the first wording, as the score did before 13 Oct (the negative control of its self-test check)."""
    cons, curr = {}, {}
    for x in results:
        q = qs[x["qid"]]
        if q["type"] == "CURRENT" and plain_wording(q):
            ws = curr.setdefault((x["phase"], x["thinking"], q["e"], q["a"]), [])
            if every_wording or not ws:
                ws.append((x["qid"], x["_g"]))
    for x in results:
        q = qs[x["qid"]]
        if q["type"] not in ("YES_NO", "EITHER_OR") or x["_g"]["got"] != ("YN" if q["type"] == "YES_NO" else "CHOICE"):
            continue
        for cq, g0 in curr.get((x["phase"], x["thinking"], q["e"], q["a"]), []):
            if g0["got"] not in ("VALUE", "ABSENT", "SET"):
                continue
            sm = ({"kind": "value", "value": g0["value"]} if g0["got"] == "VALUE" else
                  {"kind": "set", "yes": g0["yes"], "no": g0.get("no", [])} if g0["got"] == "SET" else
                  {"kind": "unknown", "not": []})
            ok = x["_g"]["value"] == (derive_yn(sm, q["v"]) if q["type"] == "YES_NO" else derive_eo(sm, q["x"], q["y"]))
            part = "dont_know" if g0["got"] == "ABSENT" else "value"
            c = cons.setdefault(("cons", x["phase"]), {}).setdefault("on" if x["thinking"] else "off", {}).setdefault(
                part, {"n": 0, "ok": 0, "not": []})
            c["n"] += 1
            c["ok"] += ok
            if not ok:
                c["not"].append(f"{x['qid']}~{cq}")
    return cons


def judge(S, rows):
    qs = {q["qid"]: q for q in S["questions"]}
    by_text = {q["text"]: q for q in S["questions"]}
    phases = {p["name"]: p for p in S["phases"]}
    tags = question_tags(S)
    results, skipped = [], 0
    for r in rows:
        q = qs.get(r["qid"]) or by_text.get(r.get("question") or "")
        if not q or r["phase"] not in phases:
            skipped += 1
            continue
        if q["text"] != (r.get("question") or q["text"]):
            q = by_text.get(r["question"], q)
        verdict, g, why = judge_one(S, r["phase"], q, r["content"])
        if verdict == "skip":
            continue
        ph = phases[r["phase"]]
        bks = [bucket(q)]
        if is_delete_phase(ph) and q["role"] == "probe":
            bks.append("delete_affected" if q["qid"] in ph.get("affected", []) else "delete_kept")
        exp = S["expected"][r["phase"]][q["qid"]]
        ref = ref_phase(r["phase"], S["expected"])
        x = {"phase": r["phase"], "qid": q["qid"], "type": q["type"], "thinking": r["thinking"],
             "question": q["text"], "expected": words(exp),
             "got": got_words(g), "verdict": verdict, "why": why, "buckets": bks,
             "class": exp_class(exp), "gave": gave_class(g, verdict, r["content"]), "tag": tags[q["qid"]],
             "change": None if ref is None else "should_change" if exp != S["expected"][ref][q["qid"]] else "should_stay",
             "excerpt": " ".join(clean(r["content"]).split())[:90], "_g": g, "_c": r["content"]}
        leak = neighbour_leak(S, r["phase"], q, g, verdict)
        if leak:
            x["neighbour"] = leak
        results.append(x)
    # consistency: a yes/no or either-or answer must follow from the model's own answer to the plain question (CURRENT).
    # Fixed 13 Oct (plan item 8): it is checked against every wording of the plain question, not only the first, and
    # the pairs are kept in two parts by what that plain answer gave. A plain "don't know" implies "don't know" for the
    # yes/no and either-or questions, so an untaught model that refuses everything agrees with itself; only the pairs
    # whose plain answer gave a value compare like for like between the baseline and the teach phase.
    cons = derived_consistency(results, qs)
    # paraphrase agreement (ParaRel): questions that ask the same thing in other words should get the same answer,
    # right or wrong; and all of them right together. The baseline phase is the untaught model's reference.
    para = {}
    for x in results:
        q = qs[x["qid"]]
        if plain_wording(q):
            para.setdefault((x["phase"], x["thinking"], paraphrase_key(q)), []).append(x)
    for (ph, th, _), xs in para.items():
        if len(xs) < 2:
            continue
        c = cons.setdefault(("para", ph), {}).setdefault("on" if th else "off", {
            "groups": 0, "all_right": 0, "pairs": 0, "same": 0, "same_value": 0, "same_dk": 0, "uncompared": 0,
            "differ": []})
        c["groups"] += 1
        c["all_right"] += all(x["verdict"] == "pass" for x in xs)
        for x1, x2 in itertools.combinations(xs, 2):
            k1, k2 = reading_key(x1["_g"]), reading_key(x2["_g"])
            if k1 is None or k2 is None:
                c["uncompared"] += 1
                continue
            c["pairs"] += 1
            if k1 == k2:
                c["same"] += 1
                c["same_dk" if k1 == "dk" else "same_value"] += 1
            else:
                c["differ"].append(f"{x1['qid']}~{x2['qid']}")
    # pairs: every yes/no question on one key (both polarities, always + changed) must be right together,
    # so a model that answers "no" to everything cannot score
    groups = {}
    for x in results:
        q = qs[x["qid"]]
        if q["type"] in ("YES_NO", "ALWAYS", "CHANGED") and q["role"] == "probe":
            groups.setdefault((x["phase"], x["thinking"], q.get("e"), q["a"]), []).append(x["verdict"] == "pass")
    for (ph, th, e, a), oks in groups.items():
        if len(oks) > 1:
            pr = cons.setdefault(("pairs", ph), [0, 0])
            pr[0] += all(oks)
            pr[1] += 1
    # kept_stable: in a delete phase, every question whose right answer did not change, compared with the same
    # model's own teach-phase answer: the same verdict, and the very same text. This is stability, not correctness:
    # an answer that was wrong in the teach phase and stays wrong the same way is stable. It needs the teach answers
    # in the same call (the runner's combined judge has them); without them nothing is reported.
    teach, exp = {}, S["expected"]
    for x in results:
        if x["phase"] == "teach":
            teach.setdefault((x["qid"], x["thinking"]), x)
    for x in results:
        t = teach.get((x["qid"], x["thinking"]))
        if not is_delete_phase(phases[x["phase"]]) or not t or exp[x["phase"]][x["qid"]] != exp["teach"][x["qid"]]:
            continue
        same_v, same_t = x["verdict"] == t["verdict"], x["_c"] == t["_c"]
        x["kept"] = {"verdict": same_v, "text": same_t, "teach_verdict": t["verdict"]}
        c = cons.setdefault(("kept", x["phase"]), {}).setdefault(
            "on" if x["thinking"] else "off", {"n": 0, "same_verdict": 0, "same_text": 0, "moved": []})
        c["n"] += 1
        c["same_verdict"] += same_v
        c["same_text"] += same_t
        if not same_v:
            c["moved"].append(f"{x['qid']} {t['verdict']}->{x['verdict']}")
    if any(q.get("form") for q in S["questions"]):  # forms v2 only, so a v1 set is judged exactly as before
        form_metrics(S, results, cons)
    for x in results:
        x.pop("_g")
        x.pop("_c")
    return results, cons, skipped


def dummy_rows(S, says):
    """The answers of a dummy system that says the same thing to every question in every phase, thinking off and
    on: "dont_know" ("I don't know.") or "yes" ("Yes."). These are the negative controls of the split view."""
    return [{"phase": ph["name"], "qid": q["qid"], "thinking": th, "question": q["text"], "content": DUMMIES[says]}
            for ph in S["phases"] for q in S["questions"] for th in (False, True)]


def class_floor(split, floor=0.5):
    """Where a system's split view gives it away: the phases, thinking modes and classes whose right answers it passes
    less than `floor` of the time (over all its scores, "all"). A model that refuses everything shows up on the value
    questions, one that says yes to everything on the no and don't-know questions; the pooled score of a phase whose
    right answers are all "don't know" shows neither. Returns [(phase, mode, class, pass, n)]."""
    return [(ph, th, c, t["pass"], t["n"]) for ph, d in split.items() if ph != "all phases" for th, m in d.items()
            for c, t in m["split"]["all"].items() if t["n"] and t["pass"] < floor * t["n"]]


def tally(xs):
    """pass, fail and unclear counts of some answers; dk_miss: the misses where the model said it does not know."""
    out = {"n": len(xs), "pass": 0, "fail": 0, "unclear": 0, "dk_miss": 0}
    for x in xs:
        v = x["verdict"] if x["verdict"] in ("pass", "fail") else "unclear"
        out[v] += 1
        out["dk_miss"] += v != "pass" and x["gave"] == "dont_know"
    return out


def split_of(xs):
    """One score split by what the rule book expects: {class: tally}, for the classes that occur."""
    return {c: tally([x for x in xs if x["class"] == c]) for c in CLASSES if any(x["class"] == c for x in xs)}


def abstention(xs):
    """AbstentionBench's view of "don't know" answers: said_dk answers said it, said_dk_right of them where the rule
    book's answer is "don't know" (precision = said_dk_right / said_dk), out of dk_expected such questions
    (recall = said_dk_right / dk_expected)."""
    said = [x for x in xs if x["gave"] == "dont_know"]
    return {"said_dk": len(said), "said_dk_right": sum(x["class"] == "dont_know" for x in said),
            "dk_expected": sum(x["class"] == "dont_know" for x in xs)}


def closed_world(xs):
    """The fails that the closed-world reading would pass: a "no" where the rule book says "don't know"."""
    return sum(x["verdict"] == "fail" and x["class"] == "dont_know" and x["gave"] == "no" for x in xs)


def split_scores(S, results):
    """The split view of judged answers (plan items 2 and 8), per phase and thinking mode ("off", "on"), and over
    all phases ("all phases"):
      split        every score (the buckets, "all", should_change and should_stay) split by expected class
      tags         reliability, generality, locality and portability (question_tags)
      abstention   "don't know" answers as precision and recall
      closed_world fails that the closed-world reading passes: a "no" where the rule book says "don't know"
      neighbour    failed answers that give a value taught about another entity (neighbour_leak)
    should_change: the questions whose right answer differs from the reference phase's (teach against baseline, a
    delete phase against teach); should_stay: the others."""
    out = {}
    names = [p["name"] for p in S["phases"]] + ["all phases"]
    for name in names:
        rs = [x for x in results if name == "all phases" or x["phase"] == name]
        for th in (False, True):
            xs = [x for x in rs if x["thinking"] == th]
            if not xs:
                continue
            groups = [("all", xs)] + [(bk, [x for x in xs if bk in x["buckets"]]) for bk in BUCKETS] \
                + [(ch, [x for x in xs if x["change"] == ch]) for ch in ("should_change", "should_stay")]
            out.setdefault(name, {})["on" if th else "off"] = {
                "split": {k: split_of(gx) for k, gx in groups if gx},
                "tags": {t: tally([x for x in xs if x["tag"] == t]) for t in TAGS if any(x["tag"] == t for x in xs)},
                "abstention": abstention(xs),
                "closed_world": closed_world(xs),
                "neighbour": [f"{x['qid']}" + ("" if name != "all phases" else f" {x['phase']}")
                              + f" {'/'.join(x['neighbour']['values'])} ({', '.join(x['neighbour']['from'])})"
                              for x in xs if x.get("neighbour")]}
    return out


def split_cell(d):
    """'v 4/7 (3 dk)  n 1/1  ? 2/2': pass of n for the questions whose right answer is a value (v), a no (n) and "don't
    know" (?); (3 dk) counts the misses where the model said it does not know."""
    parts = []
    for c, lab in (("value", "v"), ("no", "n"), ("dont_know", "?")):
        t = (d or {}).get(c)
        if t:
            parts.append(f"{lab} {t['pass']}/{t['n']}" + (f" ({t['dk_miss']} dk)" if t["dk_miss"] else ""))
    return "  ".join(parts) or "-"


def ratio(a, b):
    return f"{a}/{b}" if b else "-"


# --------------------------------------------------------------------------------------- scores of forms v2
# Each piece is its own function, so the self-test can swap one out for a broken copy and see the scores move.
YN_FORMS = (None, "true_false", "recall_yn", "who_yn")     # the yes/no question forms; None is the plain one
REVERSE_FORMS = ("who_bare", "who_listed", "who_yn")


def is_yn(q):
    """A yes/no question about one value of one key, in any form."""
    return q["type"] == "YES_NO" and q.get("form") in YN_FORMS


def said_yes(g):
    """A firm yes (a guess that leans yes is not counted)."""
    return g.get("got") == "YN" and g.get("value") == YES


def wrong_yes_pool(S, phase):
    """The yes/no questions (every form) whose right answer in this phase is not yes: "no" or "doesn't know". A yes to
    any of them is a wrong yes. The baseline is scored on the same questions, so the change is like for like."""
    return [q["qid"] for q in S["questions"] if is_yn(q) and q["role"] == "probe"
            and S["expected"][phase][q["qid"]]["value"] != YES]


def forward_ok(S, rs, e, a, v):
    """Did this model pass the forward question in this phase and mode? Any plain wording of CURRENT on (e, a) that
    passed. On a multi-valued key also the plain yes/no on v, or a list that names v where v is liked."""
    for x in rs:
        q = x["_q"]
        if q.get("form") or q.get("e") != e or q["a"] != a:
            continue
        if q["type"] == "CURRENT" and x["verdict"] == "pass":
            return True
        if S["schema"][a]["multi"] and q["type"] == "YES_NO" and q.get("v") == v and x["verdict"] == "pass":
            return True
        if S["schema"][a]["multi"] and q["type"] == "CURRENT" and x["_g"].get("got") == "SET" and v in x["_g"]["yes"] \
                and v in S["expected"][x["phase"]][q["qid"]].get("yes", []):
            return True
    return False


def earlier_likes(sch, store, e, a):
    """(the newest like, the likes before it that must still hold) on multi-valued key (e, a), or None when the store
    adds fewer than two likes there. The newest is the value of the last "I like X" lesson."""
    adds = like_adds(store, e, a)
    if len(adds) < 2:
        return None
    st, newest = state_at(sch, store, e, a, NOW), adds[-1]["v"]
    return newest, [x for x in sch[a]["domain"] if st[x] == YES and x != newest]


def likes_named(q, g):
    """The likes an answer keeps, for the Preservation line: on a yes/no question its value when the answer is a firm
    yes, on the list question the foods it names. Every answer counts: one that is unclear, off topic, a guess or
    "I don't know" keeps nothing, so it counts as lost, not as left out (None would leave it out)."""
    if q["type"] == "YES_NO":
        return {q["v"]} if said_yes(g) else set()
    return set(g.get("yes") or []) if g.get("got") == "SET" else set()


def form_metrics(S, results, cons):
    """The forms v2 scores, per phase and thinking mode, into cons[("forms", phase)]["off" | "on"]:
      wrong_yes     yes to a question whose right answer is not yes, against the baseline on the same questions
      coherence     keys of a single-valued attribute where the model said yes to two values, in any yes/no form
      forms         pass counts per form; recall_agree: recall-then-answer answers whose yes or no follows from
                    their own recall
      reverse       reverse answers where the forward question passed (each reverse result also gets `forward`,
                    and the reverse_fwd bucket where it is true); the listed form decides the reversal curse
      likes         earlier likes still named after a newer one (Preservation; the Ignore Rate is the rest)
    Needs the readings of the answers, so judge() calls it before it drops them."""
    sch, exp = S["schema"], S["expected"]
    qs = {q["qid"]: q for q in S["questions"]}
    by_id = {l["id"]: l for l in S["lessons"]}
    by = {}
    for x in results:
        x["_q"] = qs[x["qid"]]
        by.setdefault((x["phase"], x["thinking"]), []).append(x)
    for (ph, th), rs in by.items():
        m = {}
        # wrong yes, like for like with the baseline
        pool = set(wrong_yes_pool(S, ph))
        mine = [x for x in rs if x["qid"] in pool]
        base = [x for x in by.get(("baseline", th), []) if x["qid"] in pool]
        forms = {}
        for x in mine:
            f = forms.setdefault(x["_q"].get("form") or "yes_no", [0, 0])
            f[0] += said_yes(x["_g"])
            f[1] += 1
        nb = [x for x in mine if x["_q"].get("neighbour")]
        m["wrong_yes"] = {"yes": sum(said_yes(x["_g"]) for x in mine), "n": len(mine),
                          "base_yes": sum(said_yes(x["_g"]) for x in base), "base_n": len(base), "by_form": forms,
                          "neighbour": [sum(said_yes(x["_g"]) for x in nb), len(nb)],
                          "said_yes": sorted({x["qid"] for x in mine if said_yes(x["_g"])})}
        # coherence: two values of one single-valued key both said yes, in any yes/no form
        asked, yes = {}, {}
        for x in rs:
            q = x["_q"]
            if is_yn(q) and q["role"] == "probe" and not sch[q["a"]]["multi"]:
                asked.setdefault((q["e"], q["a"]), set()).add(q["v"])
                if said_yes(x["_g"]):
                    yes.setdefault((q["e"], q["a"]), set()).add(q["v"])
        m["coherence"] = {"keys": sum(len(vs) > 1 for vs in asked.values()),
                          "breaks": [f"{e} {a}: {', '.join(sorted(vs))}" for (e, a), vs in sorted(yes.items())
                                     if len(vs) > 1]}
        # pass counts per form, and whether a recall-then-answer's yes or no follows from its own recall
        fm = {}
        for x in rs:
            f = x["_q"].get("form")
            if f:
                c = fm.setdefault(f, {"pass": 0, "fail": 0, "unclear": 0})
                c[x["verdict"] if x["verdict"] in c else "unclear"] += 1
        agree = [0, 0]
        for x in rs:
            g, q = x["_g"], x["_q"]
            rec = g.get("recalled") or {}
            if q.get("form") == "recall_yn" and g.get("got") == "YN" and rec.get("got") in ("VALUE", "ABSENT"):
                sm = {"kind": "value", "value": rec["value"]} if rec["got"] == "VALUE" else {"kind": "unknown", "not": []}
                agree[0] += derive_yn(sm, q["v"]) == g["value"]
                agree[1] += 1
        m["forms"] = dict(fm, recall_agree=agree)
        # reverse: only where the forward question passed; the listed form decides the reversal curse
        rv = {f: [0, 0] for f in REVERSE_FORMS}
        held, named, extra = 0, 0, 0
        for x in rs:
            q = x["_q"]
            if q.get("form") not in REVERSE_FORMS:
                continue
            need = [q["e"]] if q["form"] == "who_yn" else exp[ph][q["qid"]]["yes"]
            x["forward"] = all(forward_ok(S, rs, e, q["a"], q["v"]) for e in need) if need else None
            if x["forward"]:
                x["buckets"].append("reverse_fwd")
                rv[q["form"]][0] += x["verdict"] == "pass"
                rv[q["form"]][1] += 1
            got = x["_g"].get("yes") if x["_g"].get("got") in ("ENTS", "GENERAL") else None
            if q["form"] != "who_yn" and need and got is not None:
                held += len(need)
                named += len(set(got) & set(need))
                extra += len(set(got) - set(need))
        curse = {"curse": [], "bare_only": [], "ok": [], "unclear": []}
        for x in rs:
            q = x["_q"]
            if q.get("form") != "who_listed" or not x.get("forward"):
                continue
            bare = [y for y in rs if y["_q"].get("form") == "who_bare" and y["_q"]["a"] == q["a"]
                    and y["_q"]["v"] == q["v"]]
            tag = f"{q['a']} {q['v']}"
            if x["verdict"] == "fail":
                curse["curse"].append(tag)
            elif x["verdict"] != "pass":
                curse["unclear"].append(tag)
            elif any(y["verdict"] == "fail" for y in bare):
                curse["bare_only"].append(tag)
            else:
                curse["ok"].append(tag)
        m["reverse"] = {"forward_passed": rv, "set": {"holders": held, "named": named, "extra": extra}, **curse}
        # likes: earlier likes still there after a newer one (Preservation); what is lost is the Ignore Rate
        store = [by_id[i] for i in phase_store(S, ph)]
        lk = {"yes_no": [0, 0], "list": [0, 0], "newest": [0, 0], "lost": []}
        for (e, a) in uniq([(l["e"], l["a"]) for l in store if sch[l["a"]]["multi"]]):
            el = earlier_likes(sch, store, e, a)
            if not el:
                continue
            newest, earlier = el
            for x in rs:
                q, g = x["_q"], x["_g"]
                if q.get("form") or q["role"] != "probe" or q.get("e") != e or q["a"] != a:
                    continue
                have = likes_named(q, g)
                if have is None:
                    continue
                if q["type"] == "YES_NO":
                    if q["v"] in earlier:
                        lk["yes_no"][0] += q["v"] in have
                        lk["yes_no"][1] += 1
                        if q["v"] not in have:
                            lk["lost"].append(f"{q['qid']} {q['v']}")
                    elif q["v"] == newest:
                        lk["newest"][0] += q["v"] in have
                        lk["newest"][1] += 1
                if q["type"] == "CURRENT":
                    lk["list"][0] += sum(x2 in have for x2 in earlier)
                    lk["list"][1] += len(earlier)
                    lk["lost"] += [f"{q['qid']} {x2}" for x2 in earlier if x2 not in have]
                    lk["newest"][0] += newest in have
                    lk["newest"][1] += 1
        if lk["yes_no"][1] or lk["list"][1]:
            m["likes"] = lk
        cons.setdefault(("forms", ph), {})["on" if th else "off"] = m
    for x in results:
        x.pop("_q", None)


FORM_SHORT = {None: "yes/no", "yes_no": "yes/no", "true_false": "true/false", "fill_in": "fill-in",
              "recall_yn": "recall+answer", "who_bare": "reverse bare", "who_listed": "reverse listed",
              "who_yn": "reverse yes/no"}


def form_lines(k):
    """The forms v2 lines of one phase in the score table (k: cons[("forms", phase)])."""
    if not k:
        return []
    frac = lambda c: f"{c[0]}/{c[1]}" if c[1] else "-"
    lines = [f"   {'forms v2':22} {'thinking off':>28} {'thinking on':>28}"]

    def row(label, f):
        cells = [f(k[th]) if th in k else "-" for th in ("off", "on")]
        if cells != ["-", "-"]:
            lines.append(f"   {label:22} {cells[0]:>28} {cells[1]:>28}")

    def wy(m):
        w = m["wrong_yes"]
        base = f" (baseline {w['base_yes']}/{w['base_n']}, {w['yes'] - w['base_yes']:+d})" if w["base_n"] else ""
        return f"{w['yes']}/{w['n']}{base}" if w["n"] else "-"
    row("wrong yes", wy)
    for f in ("yes_no", "true_false", "recall_yn", "who_yn"):
        row("  " + FORM_SHORT[f], lambda m, f=f: frac(m["wrong_yes"]["by_form"].get(f, [0, 0])))
    row("  neighbour's value", lambda m: frac(m["wrong_yes"]["neighbour"]))
    row("coherence breaks", lambda m: f"{len(m['coherence']['breaks'])} of {m['coherence']['keys']} keys"
        if m["coherence"]["keys"] else "-")
    lines.append("   pass, by form")
    for f in ("true_false", "fill_in", "recall_yn", "who_bare", "who_listed", "who_yn"):
        row("  " + FORM_SHORT[f], lambda m, f=f: (lambda c: f"{c['pass']}/{sum(c.values())}" + (
            f" ({c['unclear']} unclear)" if c["unclear"] else ""))(m["forms"][f]) if f in m["forms"] else "-")
    row("  recall agrees", lambda m: f"{frac(m['forms']['recall_agree'])} with own recall"
        if m["forms"]["recall_agree"][1] else "-")
    row("reverse, forward ok", lambda m: ", ".join(f"{FORM_SHORT[f].split()[-1]} {frac(c)}"
                                                   for f, c in m["reverse"]["forward_passed"].items() if c[1]) or "-")
    row("reversal curse", lambda m: (lambda r, n: f"{len(r['curse'])} of {n}" + (
        f" (bare only {len(r['bare_only'])})" if r["bare_only"] else "") if n else "-")(
        m["reverse"], sum(len(m["reverse"][x]) for x in ("curse", "bare_only", "ok"))))
    row("reverse as a set", lambda m: f"{m['reverse']['set']['named']}/{m['reverse']['set']['holders']} named, "
                                      f"{m['reverse']['set']['extra']} extra" if m["reverse"]["set"]["holders"] else "-")

    def likes(m):
        c = m.get("likes")
        if not c:
            return "-"
        kept, n = c["yes_no"][0] + c["list"][0], c["yes_no"][1] + c["list"][1]
        return f"{kept}/{n}, Ignore Rate {round(100 * (n - kept) / n) if n else 0}%"
    row("earlier likes kept", likes)
    row("  yes/no", lambda m: frac(m["likes"]["yes_no"]) if "likes" in m else "-")
    row("  list", lambda m: frac(m["likes"]["list"]) if "likes" in m else "-")
    row("  newest like", lambda m: frac(m["likes"]["newest"]) if "likes" in m else "-")
    for th in ("off", "on"):
        if th in k:
            m = k[th]
            for nt in ([f"coherence breaks: {'; '.join(m['coherence']['breaks'])}"] if m["coherence"]["breaks"] else []) \
                    + ([f"reversal curse: {', '.join(m['reverse']['curse'])}"] if m["reverse"]["curse"] else []) \
                    + ([f"earlier likes lost: {', '.join(m['likes']['lost'])}"] if m.get("likes", {}).get("lost") else []):
                lines.append(f"   {'think' if th == 'on' else 'plain'}: {nt}")
    return lines


def score_table(S, results, cons):
    lines = []
    split = split_scores(S, results)
    for ph in S["phases"]:
        rs = [x for x in results if x["phase"] == ph["name"]]
        if not rs:
            continue
        lines.append(f"\n== {ph['name']}: store {' '.join(lesson_name(S, i) for i in ph['store']) or '(empty)'}"
                     + (f"; deleted {' '.join(lesson_name(S, i) for i in ph['deleted'])}" if ph["name"] != "baseline" and ph["deleted"]
                        and ph.get("kind", "delete") == "delete" else ""))
        lines.append(f"   {'property':16} {'thinking off':>18} {'thinking on':>18}")
        for bk in BUCKETS:
            cells = []
            for th in (False, True):
                xs = [x for x in rs if bk in x["buckets"] and x["thinking"] == th]
                if not xs:
                    cells.append("-")
                    continue
                p = sum(x["verdict"] == "pass" for x in xs)
                f = sum(x["verdict"] == "fail" for x in xs)
                u = len(xs) - p - f
                cells.append(f"{p}/{len(xs)}" + (f" ({u} unclear)" if u else ""))
            if cells != ["-", "-"]:
                lines.append(f"   {bk:16} {cells[0]:>18} {cells[1]:>18}")
        pr = cons.get(("pairs", ph["name"]))
        if pr:
            lines.append(f"   {'yes/no groups':16} {pr[0]}/{pr[1]} keys where every yes/no question (both polarities) is right")
        sp = split.get(ph["name"], {})
        cell = lambda th, f: f(sp[th]) if th in sp else "-"
        rows = []                                      # (label, thinking off, thinking on, note)
        for row in ["all"] + BUCKETS + ["should_change", "should_stay"]:
            if any(row in sp[th]["split"] for th in sp):
                rows.append((row.replace("_", " "), cell("off", lambda d: split_cell(d["split"].get(row))),
                             cell("on", lambda d: split_cell(d["split"].get(row))), ""))
        ab = lambda d: (f"P {ratio(d['abstention']['said_dk_right'], d['abstention']['said_dk'])}"
                        f" R {ratio(d['abstention']['said_dk_right'], d['abstention']['dk_expected'])}")
        rows.append(("abstention", cell("off", ab), cell("on", ab), "precision and recall of \"don't know\""))
        if any(sp[th]["closed_world"] for th in sp):
            cw = lambda d: "+" + str(d["closed_world"])
            rows.append(("closed world", cell("off", cw), cell("on", cw),
                         "fails that pass if a \"no\" counts where the right answer is \"don't know\""))
        for t in TAGS:
            if any(t in sp[th]["tags"] for th in sp):
                tc = lambda d: ratio(d["tags"][t]["pass"], d["tags"][t]["n"]) if t in d["tags"] else "-"
                rows.append((t, cell("off", tc), cell("on", tc), ""))
        w1, w2 = max([len(r[1]) for r in rows] + [len("thinking off")]) + 2, max(len(r[2]) for r in rows) + 2
        lines.append(f"   split by the right answer: v = a value, n = no, ? = don't know; pass/n; (k dk) = k misses"
                     f" that said \"don't know\"")
        lines.append(f"   {'':16}   {'thinking off':{w1}} thinking on")
        lines += [f"   {lab:16}   {c1:{w1}} {c2:{w2}}{note}".rstrip() for lab, c1, c2, note in rows]
        nb = [f"{m} {'think' if th == 'on' else 'plain'}" for th in ("off", "on") if th in sp for m in sp[th]["neighbour"]]
        if nb:
            lines.append(f"   {'neighbourhood':16} a value taught about another entity: {', '.join(nb)}")
        c = cons.get(("cons", ph["name"]))
        if c:
            lines.append("   consistency: yes/no and either-or answers that follow from the model's own answer to each"
                         " wording of the plain question")
            for part, lab in (("value", "  plain: a value"), ("dont_know", "  plain: dk")):
                if any(part in c[th] for th in c):
                    cc = lambda th: ratio(c[th][part]["ok"], c[th][part]["n"]) if th in c and part in c[th] else "-"
                    nots = [f"{m} {'think' if th == 'on' else 'plain'}" for th in ("off", "on") if th in c
                            and part in c[th] for m in c[th][part]["not"]]
                    lines.append(f"   {lab:16} {cc('off'):>18} {cc('on'):>18}" + (f"   not: {', '.join(nots)}" if nots else ""))
        pa = cons.get(("para", ph["name"]))
        if pa:
            pc = lambda th: (f"{pa[th]['same']}/{pa[th]['pairs']} ({pa[th]['same_value']} v, {pa[th]['same_dk']} dk)"
                             if th in pa and pa[th]["pairs"] else "-")
            lines.append(f"   {'same answer':16} {pc('off'):>18} {pc('on'):>18}   pairs of wordings of one question that"
                         f" give the same answer, right or wrong (v: a value, dk: don't know)")
            ac = lambda th: ratio(pa[th]["all_right"], pa[th]["groups"]) if th in pa else "-"
            lines.append(f"   {'all wordings ok':16} {ac('off'):>18} {ac('on'):>18}   questions asked in several wordings,"
                         f" every wording right")
        k = cons.get(("kept", ph["name"]))
        if k:
            cell = lambda th, f: (f"{k[th][f]}/{k[th]['n']}" if th in k else "-")
            lines.append(f"   kept_stable: questions whose right answer did not change, against this model's own teach"
                         f" answers (stability, not correctness)")
            lines.append(f"   {'  same verdict':16} {cell('off', 'same_verdict'):>18} {cell('on', 'same_verdict'):>18}")
            lines.append(f"   {'  same text':16} {cell('off', 'same_text'):>18} {cell('on', 'same_text'):>18}")
            moved = [f"{m.split()[0]} {'think' if th == 'on' else 'plain'} {m.split()[1]}"
                     for th in ("off", "on") if th in k for m in k[th]["moved"]]
            if moved:
                lines.append(f"   {'  moved':16} {', '.join(moved)}")
        lines += form_lines(cons.get(("forms", ph["name"])))
        bad = [x for x in rs if x["verdict"] != "pass"]
        for x in bad:
            lines.append(f"   {x['verdict'].upper():7} {x['qid']} {'think' if x['thinking'] else 'plain'} "
                         f"\"{x['question']}\" expected {x['expected']}; got {x['got']}"
                         + (f" [{x['why']}]" if x["why"] else "") + f"  << {x['excerpt']}")
    al = split.get("all phases")
    if al:
        ab = lambda d: (f"precision {ratio(d['abstention']['said_dk_right'], d['abstention']['said_dk'])}, recall "
                        f"{ratio(d['abstention']['said_dk_right'], d['abstention']['dk_expected'])}")
        lines.append("\n== all phases: \"don't know\" answers (precision: right to say it; recall: said where right)")
        lines += [f"   thinking {th:3}  {ab(al[th])};  {split_cell(al[th]['split']['all'])}" for th in ("off", "on") if th in al]
    return "\n".join(lines)


def json_report(S, res, cons):
    """The judge's --json file: every verdict, the consistency and paraphrase scores, kept_stable, the split view and,
    for a forms v2 set, the forms v2 scores."""
    pick = lambda kind: {k[1]: v for k, v in cons.items() if isinstance(k, tuple) and k[0] == kind}
    derived, groups = pick("cons"), pick("pairs")
    return {"set": S["id"], "results": res,
            "consistency": {ph: {"derived": derived.get(ph), "yes_no_groups": groups.get(ph)}
                            for ph in dict.fromkeys(list(derived) + list(groups))},
            "paraphrase": pick("para"), "kept_stable": pick("kept"), "split": split_scores(S, res),
            **({"forms": pick("forms")} if pick("forms") else {})}


def lesson_name(S, i):
    return next(l["name"] for l in S["lessons"] if l["id"] == i)


# ============================================================================================ show
def show(S):
    out = [f"Test set {S['id']} (level {S['level']}, seed {S['seed']}, lessons told on {S['today']})", ""]
    out.append("Lessons, taught one at a time in this order:")
    for l in S["lessons"]:
        out.append(f"  {l['name']:4} \"{l['text']}\"   = {op_text(l)}")
    out.append("")
    out.append("Phases (each one probed with every question):")
    for ph in S["phases"]:
        dl = " ".join(lesson_name(S, i) for i in ph["deleted"]) if ph["name"] != "baseline" else "everything"
        if ph.get("kind") in ("checkpoint", "pair"):  # level 7: lessons not taught yet, not deleted
            out.append(f"  {ph['name']:13} taught so far: {' '.join(lesson_name(S, i) for i in ph['store'])}  ({ph['note']})")
            continue
        out.append(f"  {ph['name']:9} " + ("nothing taught" if ph["name"] == "baseline" else
                                           f"taught up to {lesson_name(S, ph['store'][-1])} (the runner: all "
                                           f"taught, then deleted: {dl})" if ph.get("kind") == "prefix" else
                                           f"all taught, then deleted: {dl}" if ph["deleted"] else "all taught")
                   + (f"  ({ph['note']})" if ph.get("note") and ph["name"] != "teach" and ph["name"] != "baseline" else ""))
    out.append("")
    out.append("Questions and the expected answer in each phase:")
    names = [p["name"] for p in S["phases"]]
    for q in S["questions"]:
        role = "" if q["role"] == "probe" else f" [{q['role'].replace('_', ' ')}]"
        tags = [FORM_WORDS.get(q.get("form"), q.get("form"))] + (["wrong-yes probe"] if q.get("probe") else []) \
            + ([f"{q['neighbour']}'s value"] if q.get("neighbour") else [])
        role += "".join(f" [{t}]" for t in tags if t)
        out.append(f"  {q['qid']} {q['type']:9} \"{q['text']}\"{role}")
        ws = [words(S["expected"][n][q["qid"]]) for n in names]
        if len(set(ws)) == 1:
            out.append(f"        every phase: {ws[0]}")
        else:
            out.append("        " + " | ".join(f"{n}: {w}" for n, w in zip(names, ws)))
    if S.get("notes"):
        out.append("")
        out += [f"Note: {n}" for n in S["notes"]]
    return "\n".join(out)


# ============================================================================================ lean export
def lean_export(S):
    sch, ents = S["schema"], S["entities"]
    attrs = list(sch)
    vals = []
    for a in attrs:
        for v in sch[a]["domain"]:
            if v not in vals:
                vals.append(v)
    E = {e: i for i, e in enumerate(ents)}
    A = {a: i for i, a in enumerate(attrs)}
    V = {v: i for i, v in enumerate(vals)}
    ns = "Check_" + re.sub(r"\W", "_", S["id"])
    L = [f"/- Generated by lwlogic.py from test set {S['id']}. Check it with Lean 4 (toolchain in lean-toolchain):",
         "     python3 zengine/lwlogic.py leancheck THIS_SET.json",
         "   or by hand: cat lean/LwLogic.lean THIS_FILE > /tmp/lwcheck.lean && lean /tmp/lwcheck.lean",
         "   Every `example` asks Lean's kernel to evaluate the oracle on this set and agree with Python.",
         "   If `decide` is too slow on a big set, `native_decide` is the fallback (it trusts the compiler). -/",
         f"namespace {ns}", "open LwLogic", ""]
    L.append("-- entities: " + ", ".join(f"{i}={e}" for e, i in E.items()))
    L.append("-- attributes: " + ", ".join(f"{i}={a}" for a, i in A.items()))
    L.append("-- values: " + ", ".join(f"{i}={v}" for v, i in V.items()))
    multi = [str(A[a]) for a in attrs if sch[a]["multi"]]
    # both fields start in the same column: Lean 4.35 rejects a second field that is indented less than the first
    L.append("def sch : Schema := {")
    L.append("  multi := fun a => " + (" || ".join(f"a == {m}" for m in multi) or "false") + ",")
    L.append("  dom := fun a => match a with")
    for a in attrs:
        L.append(f"    | {A[a]} => [{', '.join(str(V[v]) for v in sch[a]['domain'])}]")
    two = any(q["type"] in TWO_STEP for q in S["questions"])
    if not two:
        L.append("    | _ => [] }")
    else:                                            # level 6: the schema fields chains need, in the same column
        nats = [a for a in attrs if sch[a].get("natural")]
        if len(nats) > 1:
            raise ValueError(f"the Lean schema holds one natural map, this set has {nats}")
        L.append("    | _ => [],")
        L.append("  valEnt := fun v => match v with")
        L += [f"    | {V[v]} => some {E[v]}" for v in vals if v in E]
        L.append("    | _ => none,")
        L.append("  linkOf := fun a => match a with")
        L += [f"    | {A[sch[a]['link']]} => some {A[a]}" for a in attrs if sch[a].get("link") in A]
        L.append("    | _ => none,")
        L.append("  natMap := fun v => match v with")
        L += [f"    | {V[c]} => some {V[x]}" for n in nats for c, x in sch[n]["natural"]["map"].items()
              if c in V and x in V]
        L.append("    | _ => none,")
        L.append("  nameAttr := " + (f"some {A['name']}" if "name" in A else "none") + " }")
    L.append(f"def ents : List Entity := [{', '.join(str(E[e]) for e in ents)}]")

    def iv_lean(lo, hi):
        f = lambda x: "none" if x is None else f"(some {x})"
        return f"⟨{f(lo)}, {f(hi)}⟩"

    def op_lean(l):
        e, a = E[l["e"]], A[l["a"]]
        if l["op"] == "ASSERT":
            if l.get("when"):
                lo, hi = when_interval(l)
                return f".assert {e} {a} {V[l['v']]} (some {iv_lean(lo, hi)})"
            return f".assert {e} {a} {V[l['v']]} none"
        if l["op"] == "CHANGE":
            return f".change {e} {a} {V[l['v']]}"
        if l["op"] == "RETRACT":
            return f".retract {e} {a} " + (f"(some {V[l['v']]})" if l.get("v") is not None else "none")
        return f".dated {e} {a} {V[l['v0']]} {tick(l['d'])} {V[l['v1']]}"

    def lessons_lean(l):                             # a restated set (forms v2) is one assert per value, with the
        if l.get("vs"):                              # lesson's own id and tick: the same writes as in `writes`
            return [f"⟨{l['id']}, {l['told']}, {op_lean(dict(l, v=v, vs=None))}⟩" for v in l["vs"]]
        return [f"⟨{l['id']}, {l['told']}, {op_lean(l)}⟩"]

    by_id = {l["id"]: l for l in S["lessons"]}
    for ph in S["phases"]:
        pn = "S_" + re.sub(r"\W", "_", ph["name"])
        L.append(f"def {pn} : List Lesson := [" + ", ".join(
            x for i in ph["store"] for x in lessons_lean(by_id[i])) + "]")

    def q_lean(q):
        t = q["type"]
        if t == "WHO_HAS":
            return f".whoHas {A[q['a']]} {V[q['v']]}"
        if t == "CHAIN":
            st = f"(.named {V[q['name']]})" if q.get("name") is not None else f"(.ent {E[q['start']]})"
            c = f"{st} [{', '.join(str(A[h]) for h in q['hops'])}] {A[q['final']]} {'true' if q.get('nat') else 'false'}"
            return f".chainYN {c} {V[q['v']]}" if "v" in q else f".chain {c}"
        if t == "COMPOUND":
            ps = " ".join(f"({q_lean(p)})" for p in q["parts"])
            return {"and": ".kand ", "or": ".kor ", "not": ".knot "}[q["op"]] + ps
        e, a = E[q["e"]], A[q["a"]]
        return {"CURRENT": f".current {e} {a}", "AT": f".atT {e} {a} {tick(q['date']) if 'date' in q else 0}",
                "BEFORE": f".before {e} {a} {tick(q['date']) if 'date' in q else 0}", "PREVIOUS": f".previous {e} {a}",
                "ALWAYS": f".always {e} {a}", "CHANGED": f".changed {e} {a}",
                "YES_NO": f".yesNo {e} {a} {V.get(q.get('v'), 0)}",
                "EITHER_OR": f".eitherOr {e} {a} {V.get(q.get('x'), 0)} {V.get(q.get('y'), 0)}"}[t]

    def lst(xs):
        return "[" + ", ".join(str(V[x]) for x in xs) + "]"

    def ans_lean(x):
        k = x["kind"]
        tr = {YES: ".yes", NO: ".no", UNK: ".unk"}
        if k == "value":
            return f".summary (.value {V[x['value']]})"
        if k == "none":
            return ".summary .none"
        if k == "unknown":
            return f".summary (.unknown {lst(x['not'])})"
        if k == "set":
            return f".summary (.set {lst(x['yes'])} {lst(x['no'])})"
        if k == "yn":
            return f".yn {tr[x['value']]}"
        if k == "choice":
            return {"neither": ".choice .neither", "unknown": ".choice .unknown"}.get(x["value"], f".choice (.pick {V.get(x['value'], 0)})")
        if k == "entities":
            return f".ents [{', '.join(str(E[e]) for e in x['yes'])}]"

    n = 0
    for ph in S["phases"]:
        pn = "S_" + re.sub(r"\W", "_", ph["name"])
        L.append(f"\n-- phase {ph['name']}")
        for q in S["questions"]:
            if q["type"] == "GK":
                continue
            x = S["expected"][ph["name"]][q["qid"]]
            L.append(f"example : answer sch ents {pn} ({q_lean(q)}) = {ans_lean(x)} := by decide  -- {q['qid']} {q['text']}")
            n += 1
    L.append(f"\nend {ns}")
    return "\n".join(L) + "\n", n


# ============================================================================================ lean check
# `leancheck` runs Lean twice over: on the rule book with `#print axioms` for every theorem (a `sorry` shows up
# as the axiom sorryAx), and on the checks of each set, where the kernel recomputes every expected answer.
# `#print axioms` sees theorems only, so the rule book also fails on any `sorry` warning (an `example` or a
# `def` can hide one) and on any `axiom` the file declares itself.
# Then once more on a copy of one set's checks with one expected answer made wrong: Lean must reject exactly
# that check, or the checks prove nothing. Lean is optional for everything else in this file.
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
LEAN_SRC = os.path.join(ROOT, "lean", "LwLogic.lean")
SAFE_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}  # Lean's own three; anything else is a gap
EXAMPLE_RE = re.compile(r"^example : answer sch ents (\S+) \((.*)\) = (.*) := by decide  -- (\S+)")
# a declaration may be indented, carry an attribute (@[simp]) or a modifier (private); a theorem missed here
# would have its axioms go unchecked, a comment caught here fails loudly (unknown constant)
DECL_HEAD = r"^[ \t]*(?:@\[[^\]\n]*\][ \t]*)?(?:(?:private|protected|nonrec)[ \t]+)*"
THEOREM_RE = re.compile(DECL_HEAD + r"theorem[ \t]+([^\s(:]+)", re.M)
AXIOM_RE = re.compile(DECL_HEAD + r"axiom[ \t]+([^\s(:]+)", re.M)
SORRY_RE = re.compile(r"declaration uses [`']sorry[`']")  # Lean 4.35 quotes with backticks, older ones with '
TWO_STEP_LEAN = r"\.(?:chain|chainYN|kand|kor|knot) "     # a two-step question (level 6) in a check file


def lean_theorems(src):
    """The theorem names of LwLogic.lean, in file order."""
    return THEOREM_RE.findall(src)


def lean_axiom_decls(src):
    """Axioms the file declares itself. `#print axioms` names them only for the theorems that use them."""
    return AXIOM_RE.findall(src)


def lean_read_log(log, fname):
    """Lean's output for the file `fname` -> (sorted line numbers of its errors, {theorem: [axioms]}).
    Axioms come from `#print axioms LwLogic.X` lines; a theorem never printed is absent from the dict.
    A private theorem prints as '_private.<module>.0.LwLogic.X'."""
    errs = sorted({int(m.group(1)) for m in re.finditer(rf"^(?:.*/)?{re.escape(fname)}:(\d+):\d+: error", log, re.M)})
    axioms = {}
    for m in re.finditer(r"^'(?:_private\.[^'\s]*?\.\d+\.)?LwLogic\.(\S+)' "
                         r"(?:depends on axioms: \[([^\]]*)\]|does not depend on any axioms)",
                         log, re.M):  # \S+ backtracks over names that end in ' (filter_filter')
        axioms[m.group(1)] = [x.strip() for x in (m.group(2) or "").split(",") if x.strip()]
    return errs, axioms


def lean_warnings(log, fname):
    """Lean's warnings for the file `fname` -> sorted [(line, first line of the message)]."""
    return sorted({(int(m.group(1)), m.group(2).strip()) for m in
                   re.finditer(rf"^(?:.*/)?{re.escape(fname)}:(\d+):\d+: warning: (.*)", log, re.M)})


def lean_sorry_lines(log, fname):
    """Lines of `fname` where Lean says a declaration uses `sorry`: a theorem, but also an `example` or a `def`,
    which `#print axioms` never looks at."""
    return [ln for ln, msg in lean_warnings(log, fname) if SORRY_RE.match(msg)]


def lean_exit_note(rc, log, errs):
    """When Lean failed without an error on a line of the file (a crash, a missing toolchain), its last words."""
    if rc == 0 or errs:
        return ""
    tail = [ln.strip() for ln in log.strip().split("\n") if ln.strip()]
    return f"  Lean exited with code {rc}" + (f": {tail[-1][:160]}" if tail else ", no output")


def lean_gaps(names, axioms):
    """Theorems not proved on Lean's standard axioms alone: a `sorry`, an extra axiom, or never printed."""
    return [n for n in names if n not in axioms or not set(axioms[n]) <= SAFE_AXIOMS]


def lean_example_lines(txt, offset=0):
    """Line numbers (1-based, shifted by `offset`) of the `example` checks in a generated check file."""
    return [i + 1 + offset for i, ln in enumerate(txt.split("\n")) if EXAMPLE_RE.match(ln)]


def lean_wrong_control(txt, only=None):
    """A copy of a check file with one expected answer replaced by the answer the same question has in another
    phase (for example "blue" where only the baseline's "don't know" is right). Returns (text, the changed
    line number), or (None, None) if no question changes its answer between phases. `only`: a pattern the
    question must match (TWO_STEP_LEAN picks a two-step question)."""
    lines = txt.split("\n")
    rows = [(i, EXAMPLE_RE.match(ln)) for i, ln in enumerate(lines)]
    rows = [(i, m) for i, m in rows if m and (only is None or re.match(only, m.group(2)))]
    for i, m in rows:
        for _, m2 in rows:
            if m2.group(2) == m.group(2) and m2.group(3) != m.group(3):
                lines[i] = lines[i].replace(f" = {m.group(3)} := by decide", f" = {m2.group(3)} := by decide", 1)
                return "\n".join(lines), i + 1
    return None, None


def lean_fields_aligned(txt):
    """The schema of a check file puts `multi :=` and `dom :=` in one column. Lean 4.35 rejects a structure
    field indented less than the first one, which is how the first exports were laid out."""
    cols = [len(ln) - len(ln.lstrip()) for ln in txt.split("\n") if re.match(r"\s*(multi|dom) :=", ln)]
    return len(cols) == 2 and cols[0] == cols[1] and "Schema := {\n" in txt


def lean_run(lean, txt, work, fname):
    """Write `txt` to work/fname and run Lean on it from the repository root, where lean-toolchain picks the
    version. Returns (exit code, output, seconds)."""
    path = os.path.join(work, fname)
    with open(path, "w") as f:
        f.write(txt)
    t0 = time.time()
    p = subprocess.run([lean, path], cwd=ROOT, capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr, time.time() - t0


def lean_check(sets, lean="lean", work=None):
    """Run Lean on the rule book and on the checks of every set. Prints one line per part; True if all pass."""
    exe = shutil.which(lean)
    if not exe:
        print(f"leancheck: `{lean}` not found. Install elan (https://github.com/leanprover/elan); the toolchain "
              "in lean-toolchain installs itself on the first run.", file=sys.stderr)
        return None
    tmp = None if work else tempfile.TemporaryDirectory(prefix="lwcheck-")
    work = work or tmp.name
    os.makedirs(work, exist_ok=True)
    src = open(LEAN_SRC).read()
    ok = True

    # 1. the rule book compiles, every theorem rests on Lean's own axioms only, and nothing else uses `sorry`
    names = lean_theorems(src)
    probe = src + "\n" + "".join(f"#print axioms LwLogic.{n}\n" for n in names)
    rc, log, secs = lean_run(exe, probe, work, "LwLogic_axioms.lean")
    errs, ax = lean_read_log(log, "LwLogic_axioms.lean")
    gaps = lean_gaps(names, ax)
    sorry = lean_sorry_lines(log, "LwLogic_axioms.lean")
    decl = lean_axiom_decls(src)
    warns = [ln for ln, _ in lean_warnings(log, "LwLogic_axioms.lean") if ln not in sorry]
    good = rc == 0 and not errs and not gaps and not sorry and not decl
    ok &= good
    more = f" and {len(gaps) - 8} more" if len(gaps) > 8 else ""
    print(f"  {'PASS' if good else 'FAIL'}  rule book: {len(names) - len(gaps)} of {len(names)} theorems proved, "
          f"axioms used: {', '.join(sorted({a for n in names for a in ax.get(n, [])})) or 'none'}  [{secs:.1f} s]"
          + (f"  not proved: {', '.join(gaps[:8])}{more}" if gaps else "") + (f"  errors on lines {errs[:5]}" if errs else "")
          + (f"  `sorry` on lines {sorry[:5]}" if sorry else "") + (f"  declares axioms: {', '.join(decl)}" if decl else "")
          + (f"  warnings on lines {warns[:5]}" if warns else "") + lean_exit_note(rc, log, errs))

    # 2. every set: the checks made from the set file must match its committed .check.lean, and all pass
    total = passed = 0
    first = restated = first_two = None
    for path in sets:
        S = json.load(open(path))
        restated = restated or (S if any(l.get("vs") for l in S["lessons"]) else None)
        txt, n = lean_export(S)
        committed = path[:-5] + ".check.lean" if path.endswith(".json") else None
        drift = bool(committed and os.path.exists(committed) and open(committed).read() != txt)
        fname = "c_" + re.sub(r"\W", "_", S["id"]) + ".lean"
        rc, log, secs = lean_run(exe, src + txt, work, fname)
        errs, _ = lean_read_log(log, fname)
        ex = lean_example_lines(txt, src.count("\n"))
        bad = sorted(set(errs) & set(ex))
        other = sorted(set(errs) - set(ex))
        good = rc == 0 and not errs and len(ex) == n and not drift
        ok &= good
        total += n
        npass = 0 if other or (rc != 0 and not errs) else n - len(bad)  # a crash proves no check
        passed += npass
        first = first or (S, txt)
        if any(q["type"] in TWO_STEP for q in S["questions"]):
            first_two = first_two or (S, txt)
        print(f"  {'PASS' if good else 'FAIL'}  {S['id']}: {npass} of {n} checks pass  "
              f"[{secs:.1f} s]" + (f"  failing lines {bad[:5]}" if bad else "")
              + (f"  other errors on lines {other[:5]}" if other else "")
              + (f"  {len(ex)} checks found in the export, not {n}" if len(ex) != n else "")
              + ("  committed .check.lean differs from the set: regenerate it" if drift else "")
              + lean_exit_note(rc, log, errs))

    # 3. negative control: one wrong expected answer must fail, and only that check
    if first:
        S, txt = first
        wrong, line = lean_wrong_control(txt)
        if wrong is None:
            good = False
            print(f"  FAIL  negative control: no question in {S['id']} changes its answer between phases")
        else:
            rc, log, secs = lean_run(exe, src + wrong, work, "c_wrong_control.lean")
            errs, _ = lean_read_log(log, "c_wrong_control.lean")
            good = rc != 0 and errs == [line + src.count("\n")]
            print(f"  {'PASS' if good else 'FAIL'}  negative control: one wrong expected answer in {S['id']} "
                  f"(line {line} of its checks) is {'rejected, and only that one' if good else 'NOT caught alone'}"
                  f"  [{secs:.1f} s]" + lean_exit_note(rc, log, errs))
        ok &= good
    # 3b. the same with a two-step question (level 6), when a set has them
    if first_two:
        S, txt = first_two
        wrong, line = lean_wrong_control(txt, TWO_STEP_LEAN)
        if wrong is None:
            good = False
            print(f"  FAIL  negative control: no two-step question in {S['id']} changes its answer between phases")
        else:
            rc, log, secs = lean_run(exe, src + wrong, work, "c_wrong_two_step.lean")
            errs, _ = lean_read_log(log, "c_wrong_two_step.lean")
            good = rc != 0 and errs == [line + src.count("\n")]
            print(f"  {'PASS' if good else 'FAIL'}  negative control: one wrong two-step answer in {S['id']} "
                  f"(line {line} of its checks) is {'rejected, and only that one' if good else 'NOT caught alone'}"
                  f"  [{secs:.1f} s]" + lean_exit_note(rc, log, errs))
        ok &= good

    # 4. forms v2, restated likes: with each restated lesson exported as its newest like only, Lean must reject exactly
    # the checks whose answer that changes, so the export of a restated lesson is checked, not taken on trust
    if restated:
        good, why, secs = restate_control(exe, src, work, restated)
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  negative control: {restated['id']} with each restated lesson exported "
              f"as its newest like only: {why}  [{secs:.1f} s]")
    print(f"leancheck {'PASSED' if ok else 'FAILED'}: {passed} of {total} checks pass across {len(sets)} set{'' if len(sets) == 1 else 's'}, "
          f"{lean_version(exe)}")
    if tmp:
        tmp.cleanup()
    return ok


def restate_control(exe, src, work, S):
    """leancheck's control for restated likes (forms v2): the set's right expected answers, with every restated lesson
    exported as its newest like only. Lean must reject exactly the checks whose answer that reading changes (worked out
    by the rule book), and there must be some. Returns (passed, words, seconds)."""
    B = dict(S, lessons=[dict(l, vs=None) for l in S["lessons"]])
    txt, _ = lean_export(B)                          # B keeps S's expected answers: right answers, wrong lessons
    B2 = dict(B)
    fill_expected(B2)
    order = [(ph["name"], q["qid"]) for ph in S["phases"] for q in S["questions"] if q["type"] != "GK"]
    lines = lean_example_lines(txt, src.count("\n"))
    want = [ln for (ph, qid), ln in zip(order, lines) if B2["expected"][ph][qid] != S["expected"][ph][qid]]
    rc, log, secs = lean_run(exe, src + txt, work, "c_restate_control.lean")
    errs, _ = lean_read_log(log, "c_restate_control.lean")
    good = bool(want) and len(lines) == len(order) and rc != 0 and errs == want
    return good, (f"{len(errs)} checks rejected, the {len(want)} it changes" if good else
                  f"rejected lines {errs[:5]}, expected {want[:5]}") + lean_exit_note(rc, log, errs), secs


def lean_version(exe):
    p = subprocess.run([exe, "--version"], cwd=ROOT, capture_output=True, text=True)
    m = re.search(r"version (\S+?),", p.stdout)
    return f"Lean {m.group(1)}" if m else "Lean (version unknown)"


def default_lean_sets():
    """The example sets that have a committed check file, in examples/ and then examples/v2/ (forms v2)."""
    out = []
    for ex in (os.path.join(ROOT, "examples"), os.path.join(ROOT, "examples", "v2")):
        if os.path.isdir(ex):
            out += sorted(os.path.join(ex, f[:-len(".check.lean")] + ".json") for f in os.listdir(ex)
                          if f.endswith(".check.lean") and os.path.exists(os.path.join(ex, f[:-len(".check.lean")] + ".json")))
    return out


# ============================================================================================ self-test
TEST_TODAY = "2026-10-12"
TEST_DAYS = ["2025-01-10", "2025-06-01", "2026-02-15", "2026-08-20"]


def test_schema():
    return {"fav_colour": dict(SCHEMA["fav_colour"], domain=["red", "blue", "green"]),
            "likes": dict(SCHEMA["likes"], domain=["apples", "cheese", "olives"]),
            "eye_colour": dict(SCHEMA["eye_colour"], domain=["brown", "green", "blue"])}


def rand_lesson(rng, lid, sch, ents, told=None, key=None):
    e, a = key or (rng.choice(ents), rng.choice(list(sch)))
    dom, multi = sch[a]["domain"], sch[a]["multi"]
    l = {"id": lid, "name": f"L{lid}", "e": e, "a": a, "told": told if told is not None else tick(TEST_TODAY) + 100 * lid}
    r = rng.random()
    if r < 0.35:
        l.update(op="ASSERT", v=rng.choice(dom))
        if rng.random() < 0.3:
            w = rng.choice(["since", "until", "always", "between"])
            ds = sorted(rng.sample(TEST_DAYS, 2))
            l["when"] = {"always": True} if w == "always" else {"between": ds} if w == "between" else {w: ds[0]}
    elif r < 0.55:
        l.update(op="CHANGE", v=rng.choice(dom))
    elif r < 0.75:
        l.update(op="RETRACT", v=rng.choice(dom) if rng.random() < 0.8 else None)
    elif not multi:
        l.update(op="DATED", v0=rng.choice(dom), d=rng.choice(TEST_DAYS), v1=rng.choice(dom))
    else:
        l.update(op="ASSERT", v=rng.choice(dom))
    return l


def rand_store(rng, sch, ents, n=None):
    n = rng.randint(0, 10) if n is None else n
    return [rand_lesson(rng, i + 1, sch, ents) for i in range(n)]


def all_questions(sch, ents):
    qs = []
    pts = TEST_DAYS + [iso_of(day_of(d) + 1) for d in TEST_DAYS] + [TEST_TODAY]
    for e in ents:
        for a in sch:
            dom = sch[a]["domain"]
            qs += [{"type": t, "e": e, "a": a} for t in ("CURRENT", "PREVIOUS", "ALWAYS", "CHANGED")]
            qs += [{"type": "YES_NO", "e": e, "a": a, "v": v} for v in dom]
            qs += [{"type": "EITHER_OR", "e": e, "a": a, "x": x, "y": y} for x, y in itertools.permutations(dom, 2)]
            qs += [{"type": t, "e": e, "a": a, "date": d} for d in pts for t in ("AT", "BEFORE")]
    for a in sch:
        qs += [{"type": "WHO_HAS", "a": a, "v": v} for v in sch[a]["domain"]]
    return qs


def answers_of(sch, store, ents, qs):
    return [answer(sch, store, ents, q) for q in qs]


def key_qs(qs, e, a):
    return [q for q in qs if q.get("e") == e and q["a"] == a and q["type"] != "WHO_HAS"]


def renumber(store):
    """Teaching order defines the told ticks: re-teach `store` in its list order on the test day."""
    return [dict(l, told=tick(TEST_TODAY) + 100 * (i + 1)) for i, l in enumerate(store)]


class Check:
    def __init__(self):
        self.rows = []

    def add(self, name, ok, n, detail=""):
        self.rows.append((name, ok, n, detail))
        print(f"  {'PASS' if ok else 'FAIL'}  {name}  [{n}]" + (f"  {detail}" if detail else ""), flush=True)


# ------------------------------------------------------------------------------------- systems under test
# A system answers questions from a log of {"teach": lesson} and {"delete": id} entries. The oracle is one
# system. The mutants below are deliberately wrong systems, each modelled on a way a real one can fail. Every
# property check takes the system as an argument, so the very check that passes the oracle is also run on the
# broken systems and must fail there: a check that cannot fail is not a check.
def as_log(store):
    return [{"teach": l} for l in store]


def system_from_store(sch, ents, to_store, sch_used=None):
    return lambda log, qs: answers_of(sch_used or sch, to_store(log), ents, qs)


def mut_replace(sch):
    """1.0.8-like: a newer fact on a single-valued key zeroes the older ones for good."""
    def to_store(log):
        phys = []
        for x in log:
            if "teach" in x:
                l = x["teach"]
                if not sch[l["a"]]["multi"]:
                    phys = [m for m in phys if (m["e"], m["a"]) != (l["e"], l["a"])]
                phys.append(l)
            else:
                phys = [m for m in phys if m["id"] != x["delete"]]
        return phys
    return to_store


def mut_cascade(log):
    """1.0.7-like: deleting a lesson also loses every later lesson on its key (they leaned on its weights)."""
    lessons = [x["teach"] for x in log if "teach" in x]
    gone = {x["delete"] for x in log if "delete" in x}
    lost = set(gone)
    for i, l in enumerate(lessons):
        if l["id"] in gone:
            lost |= {m["id"] for m in lessons[i + 1:] if (m["e"], m["a"]) == (l["e"], l["a"])}
    return [l for l in lessons if l["id"] not in lost]


def mut_lean(log):
    """The #678 eye case: a DATED lesson is lost when the lesson taught just before it on its key is deleted."""
    lessons = [x["teach"] for x in log if "teach" in x]
    gone = {x["delete"] for x in log if "delete" in x}
    prev, last = {}, {}
    for l in lessons:
        k = (l["e"], l["a"])
        prev[l["id"]], last[k] = last.get(k), l["id"]
    return [l for l in lessons if l["id"] not in gone and not (l["op"] == "DATED" and prev[l["id"]] in gone)]


def mut_leak(log):
    """Entity mix-up: whatever is taught about my sister is also believed about me."""
    out = []
    for l in effective(log):
        out.append(l)
        if l["e"] == "my sister":
            out.append(dict(l, e="me", id=10_000 + l["id"]))
    return out


def mut_overflow(log):
    """The pocket holds only the last 4 lessons; older ones are forgotten."""
    return effective(log)[-4:]


def mut_repeat(log):
    """Repetition hardens: a fact said a second time is read as 'it has always been so'."""
    out, seen = [], set()
    for l in effective(log):
        sig = (l["op"], l["e"], l["a"], l.get("v"), json.dumps(l.get("when"), sort_keys=True))
        if sig in seen and l["op"] in ("ASSERT", "CHANGE"):
            l = dict(l, op="ASSERT", when={"always": True})
        seen.add(sig)
        out.append(l)
    return out


def mut_eo_first(sch, ents):
    """Either-or picks the first option when the answer is not known."""
    def system(log, qs):
        out = answers_of(sch, effective(log), ents, qs)
        return [{"kind": "choice", "value": q["x"]} if q["type"] == "EITHER_OR" and x["value"] == "unknown" else x
                for q, x in zip(qs, out)]
    return system


def mut_as_set(sch):
    """Single-valued attributes treated as sets: 'I like red best', then 'I like blue best' keeps both."""
    return {a: dict(sch[a], multi=True) for a in sch}


# ------------------------------------------------------------------------------------------ properties
# Each returns {"n": cases checked, "bad": cases that broke the property, "ex": first example, "control": int}.
def prop_delete(system, rng, n, sch, ents, qs):
    """DELETE(i) = never teaching i; the order of two deletes does not matter; deleting all = the baseline."""
    r = {"n": 0, "bad": 0, "ex": "", "control": 0}
    for _ in range(n):
        S = rand_store(rng, sch, ents, rng.randint(1, 10))
        log = as_log(S)
        for _ in range(rng.randint(0, 3)):           # earlier deletes interleaved in the log
            pos = rng.randint(1, len(log))
            taught = [x["teach"]["id"] for x in log[:pos] if "teach" in x]
            if taught:
                log.insert(pos, {"delete": rng.choice(taught)})
        live = effective(log)
        if not live:
            continue
        r["n"] += 1
        i1 = rng.choice(live)["id"]
        i2 = rng.choice([l["id"] for l in live if l["id"] != i1] or [i1])
        never = [x for x in log if not ("teach" in x and x["teach"]["id"] == i1) and x.get("delete") != i1]
        why = ""
        if system(log + [{"delete": i1}], qs) != system(never, qs):
            why = f"delete L{i1} != never taught"
        elif system(log + [{"delete": i1}, {"delete": i2}], qs) != system(log + [{"delete": i2}, {"delete": i1}], qs):
            why = "the order of two deletes matters"
        elif system(log + [{"delete": l["id"]} for l in live], qs) != system([], qs):
            why = "deleting everything != baseline"
        if why:
            r["bad"] += 1
            r["ex"] = r["ex"] or f"{why}; store {[op_text(l) for l in S]}"
    return r


def prop_locality(system, rng, n, sch, ents, qs):
    """Lessons on other (entity, attribute) pairs never change an answer about this pair."""
    r = {"n": n, "bad": 0, "ex": "", "control": 0}
    for _ in range(n):
        S = rand_store(rng, sch, ents)
        e, a = rng.choice(ents), rng.choice(list(sch))
        kq = key_qs(qs, e, a)
        base = system(as_log(S), kq)
        only = [l for l in S if (l["e"], l["a"]) == (e, a)]
        noisy = list(S)
        for j in range(rng.randint(1, 4)):
            x = rand_lesson(rng, 100 + j, sch, ents)
            if (x["e"], x["a"]) != (e, a):
                noisy.insert(rng.randint(0, len(noisy)), x)
        if base != system(as_log(only), kq) or system(as_log(renumber(noisy)), kq) != base:
            r["bad"] += 1
            r["ex"] = r["ex"] or f"({e}, {a}) in {[op_text(l) for l in S]}"
    return r


def prop_dated(system, rng, n, sch, ents, qs):
    """A self-contained DATED lesson keeps all its answers (PREVIOUS and ALWAYS included) when any lessons taught
    before it are deleted. Control: without the DATED lesson, the same deletions do move answers."""
    r = {"n": n, "bad": 0, "ex": "", "control": 0}
    single = [a for a in sch if not sch[a]["multi"]]
    for _ in range(n):
        e, a = rng.choice(ents), rng.choice(single)
        S = rand_store(rng, sch, ents, rng.randint(1, 8))
        for _ in range(rng.randint(1, 3)):
            S.append(rand_lesson(rng, len(S) + 1, sch, ents, key=(e, a)))
        dl = {"id": len(S) + 1, "name": "D", "op": "DATED", "e": e, "a": a, "v0": rng.choice(sch[a]["domain"]),
              "d": rng.choice(TEST_DAYS), "v1": rng.choice(sch[a]["domain"]), "told": tick(TEST_TODAY) + 100 * (len(S) + 1)}
        after = [rand_lesson(rng, len(S) + 2 + j, sch, ents) for j in range(rng.randint(0, 3))]
        after = [x for x in after if (x["e"], x["a"]) != (e, a)]
        full = S + [dl] + after
        kq = key_qs(qs, e, a)
        drop = [{"delete": i} for i in sorted({l["id"] for l in rng.sample(S, rng.randint(1, len(S)))})]
        if system(as_log(full) + drop, kq) != system(as_log(full), kq):
            r["bad"] += 1
            r["ex"] = r["ex"] or f"deleted {[d['delete'] for d in drop]} of {[op_text(l) for l in full]}"
        if system(as_log(S) + drop, kq) != system(as_log(S), kq):
            r["control"] += 1
    return r


def prop_idem(system, rng, n, sch, ents, qs):
    """The same lesson taught twice in a row = taught once. Control: the same fact again after a conflicting
    change is a real change of mind and does move answers."""
    r = {"n": n, "bad": 0, "ex": "", "control": 0}
    for _ in range(n):
        S = rand_store(rng, sch, ents, rng.randint(1, 10))
        j = rng.randrange(len(S))
        dup = dict(S[j], id=999, name="dup", told=S[j]["told"] + 1)
        if system(as_log(S[:j + 1] + [dup] + S[j + 1:]), qs) != system(as_log(S), qs):
            r["bad"] += 1
            r["ex"] = r["ex"] or f"{op_text(S[j])} twice in {[op_text(l) for l in S]}"
        l = S[j]
        if l["op"] in ("ASSERT", "CHANGE") and not l.get("when") and not sch[l["a"]]["multi"]:
            dom = sch[l["a"]]["domain"]
            mid = dict(l, id=998, name="mid", op="CHANGE", v=[x for x in dom if x != l["v"]][0], told=l["told"] + 1)
            mid.pop("when", None)
            dup2 = dict(dup, told=l["told"] + 2)
            if (system(as_log(S[:j + 1] + [mid, dup2] + S[j + 1:]), qs)
                    != system(as_log(S[:j + 1] + [mid] + S[j + 1:]), qs)):
                r["control"] += 1
    return r


def prop_order(system, rng, n, sch, ents, qs):
    """Two adjacent lessons on different keys can be swapped; so can two different 'I like X' lessons (for
    questions about a time point). Control: swapping lessons on the same key does move answers."""
    r = {"n": 0, "bad": 0, "ex": "", "control": 0}
    tq = [q for q in qs if q["type"] not in ("PREVIOUS", "ALWAYS", "CHANGED")]
    for _ in range(n):
        S = rand_store(rng, sch, ents, rng.randint(2, 10))
        j = rng.randrange(len(S) - 1)
        x, y = S[j], S[j + 1]
        sw = as_log(renumber(S[:j] + [y, x] + S[j + 2:]))
        same_key = (x["e"], x["a"]) == (y["e"], y["a"])
        if not same_key:
            r["n"] += 1
            if system(as_log(renumber(S)), qs) != system(sw, qs):
                r["bad"] += 1
                r["ex"] = r["ex"] or f"swap {op_text(x)} / {op_text(y)}"
        elif (sch[x["a"]]["multi"] and x["op"] == y["op"] == "ASSERT" and x["v"] != y["v"]
              and not x.get("when") and not y.get("when")):
            r["n"] += 1
            if system(as_log(renumber(S)), tq) != system(sw, tq):
                r["bad"] += 1
                r["ex"] = r["ex"] or f"swap likes {x['v']} / {y['v']}"
        elif system(as_log(renumber(S)), qs) != system(sw, qs):
            r["control"] += 1
    return r


def qindex(qs):
    return {(q["type"], q.get("e"), q["a"], q.get("v"), q.get("x"), q.get("y"), q.get("date")): i
            for i, q in enumerate(qs)}


def prop_derive(system, rng, n, sch, ents, qs):
    """YES_NO, EITHER_OR and WHO_HAS follow from the answer to CURRENT, and a single-valued attribute never says
    yes to two values. Read from the answers alone, so it applies to any system."""
    r = {"n": n, "bad": 0, "ex": "", "control": 0}
    ix = qindex(qs)
    for _ in range(n):
        S = rand_store(rng, sch, ents)
        ans = system(as_log(S), qs)
        get = lambda t, e, a, v=None, x=None, y=None: ans[ix[(t, e, a, v, x, y, None)]]
        why = ""
        for e in ents:
            for a in sch:
                cur, dom = get("CURRENT", e, a), sch[a]["domain"]
                yn = {v: get("YES_NO", e, a, v)["value"] for v in dom}
                if not sch[a]["multi"] and (cur["kind"] == "set" or sum(t == YES for t in yn.values()) > 1):
                    why = why or f"two values at once for ({e}, {a})"
                for v in dom:
                    if derive_yn(cur, v) != yn[v]:
                        why = why or f"YES_NO {v} does not follow from CURRENT for ({e}, {a})"
                for x, y in itertools.permutations(dom, 2):
                    if derive_eo(cur, x, y) != get("EITHER_OR", e, a, None, x, y)["value"]:
                        why = why or f"EITHER_OR {x}/{y} does not follow from CURRENT for ({e}, {a})"
        for a in sch:
            for v in sch[a]["domain"]:
                w = [e for e in ents if derive_yn(get("CURRENT", e, a), v) == YES]
                if w != get("WHO_HAS", None, a, v)["yes"]:
                    why = why or f"WHO_HAS {a} = {v} does not follow from CURRENT"
        if why:
            r["bad"] += 1
            r["ex"] = r["ex"] or why
    return r


PROPS = [("delete", "DELETE(i) gives the answers of never teaching i; delete order is irrelevant; deleting all = "
                    "baseline", prop_delete),
         ("locality", "locality: lessons on other (entity, attribute) pairs never change an answer", prop_locality),
         ("dated", "a self-contained DATED lesson keeps all its answers when any earlier lessons are deleted",
          prop_dated),
         ("idempotence", "idempotence: the same lesson taught twice in a row = taught once", prop_idem),
         ("order", "order: swapping lessons on different keys (or two different likes) changes nothing", prop_order),
         ("derive", "YES_NO, EITHER_OR and WHO_HAS follow from CURRENT; a single-valued attribute says yes to at "
                    "most one value", prop_derive)]
CONTROL_WORDS = {"dated": "without the DATED lesson the same deletions moved answers in {c} of {n} stores",
                 "idempotence": "re-teaching after a conflicting change moved answers in {c} cases",
                 "order": "swapping same-key lessons moved answers in {c} cases"}


def mutants(sch, ents):
    """(name, system, the properties that must catch it)."""
    S = lambda f: system_from_store(sch, ents, f)
    return [
        ("1.0.8-like: a newer fact zeroes the older one for good", S(mut_replace(sch)), ["delete"]),
        ("1.0.7-like: deleting a lesson also loses the later lessons on its key", S(mut_cascade), ["delete", "dated"]),
        ("#678 eye case: a dated fact is lost when the fact before it is deleted", S(mut_lean), ["delete", "dated"]),
        ("entity mix-up: what is taught about my sister is also believed about me", S(mut_leak), ["locality"]),
        ("overflow: only the last 4 lessons are kept", S(mut_overflow), ["locality", "order"]),
        ("repetition hardens: a fact said twice is read as 'always'", S(mut_repeat), ["idempotence"]),
        ("either-or picks the first option when it does not know", mut_eo_first(sch, ents), ["derive"]),
        ("single-valued attributes kept as sets ('red best', then 'blue best' keeps both)",
         system_from_store(sch, ents, effective, mut_as_set(sch)), ["derive"]),
    ]


# ------------------------------------------------------------------------------------ checks of forms v2 sets
def v1_part_problems(v1, v2):
    """What a v2 set changed of the v1 set it was made from; empty when the v1 set is part of it unchanged: the same
    lessons (a restated set may change only a like lesson's text and add its values), the v1 questions first and the
    same apart from the v2 tags, the v1 phases first and the same, the v1 entities first, and (not for a restated set,
    whose deletes may differ) every v1 expected answer."""
    out, tags = [], ("probe", "neighbour", "form")
    restate = v2["id"].endswith("-restate")
    drop = lambda d, ks: {k: v for k, v in d.items() if k not in ks}
    lk = ("text", "para", "vs", "form") if restate else ()
    if len(v1["lessons"]) != len(v2["lessons"]) or any(drop(a, lk) != drop(b, lk)
                                                       for a, b in zip(v1["lessons"], v2["lessons"])):
        out.append("lessons changed")
    n = len(v1["questions"])
    out += [f"question {a['qid']} changed" for a, b in zip(v1["questions"], v2["questions"][:n])
            if drop(a, tags) != drop(b, tags)] + (["questions lost"] if len(v2["questions"]) < n else [])
    qn = {q["qid"] for q in v1["questions"]}           # a delete phase's affected list also names the new questions
    out += [f"phase {a['name']} changed" for a, b in zip(v1["phases"], v2["phases"])
            if drop(a, ("affected",)) != drop(b, ("affected",)) or a.get("affected") != (
                [x for x in b["affected"] if x in qn] if "affected" in b else None)]
    out += ["phases lost"] if len(v2["phases"]) < len(v1["phases"]) else []
    out += ["entities changed"] if v2["entities"][:len(v1["entities"])] != v1["entities"] else []
    if not restate:
        out += [f"expected {ph['name']} {q['qid']} changed" for ph in v1["phases"] for q in v1["questions"]
                if v1["expected"][ph["name"]][q["qid"]] != v2["expected"].get(ph["name"], {}).get(q["qid"])]
    return out


def v2_problems(S):
    """What a forms v2 set lacks, as a list of plain sentences (empty when it is complete): every single-valued
    taught key has wrong-yes probes on at least 3 untaught values (on as many as there are, if fewer), a neighbour's
    taught value among them when the set has one, and the true/false, fill-in and recall-then-answer forms; every listed
    reverse question has its bare and yes/no forms, names everyone who holds the value in any phase (so its answer is
    the bare one), and each holder has a forward question; every like key has the full list and a yes/no per added
    value; every like after the first has a phase that ends with it (up to MAX_PREFIX)."""
    out, sch, exp, qs = [], S["schema"], S["expected"], S["questions"]
    if S.get("forms") != "v2":
        out.append("not marked forms v2")
    if len({q["text"] for q in qs}) != len(qs) or len({q["qid"] for q in qs}) != len(qs):
        out.append("a question text or qid twice")
    plain = lambda t, e, a: [q for q in qs if q["type"] == t and not q.get("form") and q.get("e") == e and q["a"] == a]
    lesson_ents = uniq([l["e"] for l in S["lessons"]])
    for (e, a) in uniq([(l["e"], l["a"]) for l in S["lessons"]]):
        if sch[a]["multi"]:
            if not plain("CURRENT", e, a):
                out.append(f"no list question for {e} {a}")
            for v in uniq([v for l in S["lessons"] if (l["e"], l["a"]) == (e, a) and l["op"] in ("ASSERT", "CHANGE")
                           for v in lesson_values(l)]):
                if not any(q["v"] == v for q in plain("YES_NO", e, a)):
                    out.append(f"no yes/no for {e} {a} {v}")
            continue
        if not T2.get(a, {}).get("TF"):
            continue
        taught = taught_values(S, e, a)
        untaught = [v for v in sch[a]["domain"] if v not in taught]
        probes = [q for q in plain("YES_NO", e, a) if q.get("probe") == "wrong_yes"]
        if any(q["v"] in taught for q in probes):
            out.append(f"a wrong-yes probe on a taught value of {e} {a}")
        if len({q["v"] for q in probes}) < min(3, len(untaught)):
            out.append(f"fewer than 3 untaught values asked for {e} {a}")
        if any(v in untaught for e2 in lesson_ents if e2 != e for v in taught_values(S, e2, a)) \
                and not any(q.get("neighbour") for q in probes):
            out.append(f"no neighbour's value asked for {e} {a}")
        for f in ("true_false", "fill_in", "recall_yn"):
            if not any(q.get("form") == f and q.get("e") == e and q["a"] == a for q in qs):
                out.append(f"no {f} question for {e} {a}")
    for q in qs:
        if q.get("form") != "who_listed":
            continue
        same = lambda f: [b for b in qs if b.get("form") == f and b["a"] == q["a"] and b.get("v") == q["v"]]
        if not same("who_bare") or not same("who_yn"):
            out.append(f"{q['qid']}: the listed form without its bare or yes/no form")
        for ph in S["phases"]:
            got = exp[ph["name"]][q["qid"]]["yes"]
            if not set(got) <= set(q["among"]) or any(sorted(exp[ph["name"]][b["qid"]]["yes"]) != sorted(got)
                                                     for b in same("who_bare")):
                out.append(f"{q['qid']} in {ph['name']}: the list misses someone who holds {q['v']}")
            for x in got:
                if not any(c["type"] == "CURRENT" and not c.get("form") and c.get("e") == x and c["a"] == q["a"]
                           for c in qs):
                    out.append(f"{q['qid']}: no forward question for {x}")
    if any(q.get("form") == "who_bare" for q in qs) and not any(q.get("form") == "who_listed" for q in qs):
        out.append("a reverse question without its listed form")
    # recall and yes/no score taught pairs only, so a forward question about an untaught pair is a near miss
    keys = {(l["e"], l["a"]) for l in S["lessons"]}
    out += [f"{q['qid']}: a probe on {q['e']} {q['a']}, which no lesson is about" for q in qs
            if q["role"] == "probe" and q.get("e") and not q.get("form") and (q["e"], q["a"]) not in keys]
    ids, stores = [l["id"] for l in S["lessons"]], [ph["store"] for ph in S["phases"]]
    want = sorted({ids.index(l["id"]) for (e, a) in uniq([(l["e"], l["a"]) for l in S["lessons"]]) if sch[a]["multi"]
                   for l in like_adds(S["lessons"], e, a)[1:]})[:MAX_PREFIX]
    out += [f"no phase that ends after {S['lessons'][p]['name']}" for p in want
            if ids[:p + 1] != ids and ids[:p + 1] not in stores]
    return uniq(out)


def selftest(n=1500, seed=1, facts=None, probes_dir=None):
    t0 = time.time()
    rng = random.Random(seed)
    sch, ents = test_schema(), ["me", "my sister", "my cat"]
    qs = all_questions(sch, ents)
    C = Check()
    print(f"selftest: {n} random stores per property, seed {seed}, {len(qs)} questions per store")

    # 1. differential: fold oracle vs incremental timeline; control: a referee where the earliest lesson wins
    bad, first = None, 0
    for i in range(n):
        S = rand_store(rng, sch, ents)
        tl = Timeline(sch)
        for l in S:
            tl.apply(l)
        for q in qs:
            if answer(sch, S, ents, q) != tl.answer(ents, q):
                bad = bad or (S, q, answer(sch, S, ents, q), tl.answer(ents, q))
        if any(answer(sch, list(reversed(S)), ents, q) != tl.answer(ents, q) for q in qs):
            first += 1
    C.add("fold oracle == independent incremental timeline, every question type", bad is None, n,
          "" if bad is None else f"first mismatch: {bad[1]} fold={bad[2]} timeline={bad[3]} store={[op_text(l) for l in bad[0]]}")
    C.add("negative control: a referee where the earliest lesson wins fails the differential test", first > 0, n,
          f"caught in {first} stores")

    # 2-7. the properties, on the oracle
    oracle = system_from_store(sch, ents, effective)
    for key, title, P in PROPS:
        r = P(oracle, random.Random(f"{seed}/{key}/oracle"), n, sch, ents, qs)
        ctl = CONTROL_WORDS.get(key, "").format(c=r["control"], n=r["n"])
        C.add(title, r["bad"] == 0 and (key not in CONTROL_WORDS or r["control"] > 0), r["n"],
              (f"control: {ctl}" if ctl else "") + (f"; FIRST BREAK: {r['ex']}" if r["bad"] else ""))

    # 2-7 again, on deliberately wrong systems: each must be caught by the property it breaks
    nm = max(100, n // 5)
    by_key = {k: P for k, _, P in PROPS}
    for name, system, targets in mutants(sch, ents):
        res = {t: by_key[t](system, random.Random(f"{seed}/{t}/{name}"), nm, sch, ents, qs) for t in targets}
        C.add(f"negative control: {name}", all(res[t]["bad"] > 0 for t in targets), nm,
              "; ".join(f"caught by {t} in {res[t]['bad']} of {res[t]['n']}" for t in targets)
              + (f"; e.g. {res[targets[0]]['ex'][:110]}" if res[targets[0]]["ex"] else ""))

    # 7a. state level, oracle only: a single-valued attribute never holds two values, at any time
    bad = None
    for i in range(n // 3):
        S = rand_store(rng, sch, ents)
        for e in ents:
            for a in sch:
                if sch[a]["multi"]:
                    continue
                for p in [NOW] + [tick(d) for d in TEST_DAYS] + [l["told"] - 1 for l in S]:
                    s = state_at(sch, S, e, a, p)
                    ys = [v for v in s if s[v] == YES]
                    if len(ys) > 1 or (ys and any(s[v] != NO for v in s if v != ys[0])):
                        bad = bad or (S, e, a, p)
    C.add("a single-valued attribute never holds two values, at any time (state level, hint ticks included)",
          bad is None, n // 3)

    # 7b. golden cases: hand-written expected answers that pin the meaning of each rule. The differential test
    # cannot see a bug in `writes` or `step` (both implementations share them); these can.
    def mk(*ops):
        out = []
        for i, o in enumerate(ops):
            l = {"id": i + 1, "name": f"G{i + 1}", "told": tick(TEST_TODAY) + 100 * (i + 1)}
            l.update(o)
            out.append(l)
        return out

    def A(op, a, v=None, e="me", **kw):
        return dict(op=op, e=e, a=a, v=v, **kw)

    def DT(a, v0, d, v1, e="me"):
        return dict(op="DATED", e=e, a=a, v0=v0, d=d, v1=v1)

    V = lambda v: {"kind": "value", "value": v}
    U = lambda *n: {"kind": "unknown", "not": list(n)}
    YN = lambda t: {"kind": "yn", "value": t}
    CH = lambda c: {"kind": "choice", "value": c}
    Q = lambda t, a, e="me", **kw: dict(type=t, e=e, a=a, **kw)
    f, li, ey = "fav_colour", "likes", "eye_colour"
    d0, d1, d2 = TEST_DAYS[0], TEST_DAYS[1], TEST_DAYS[2]
    golden = [
        ("a change keeps history; present-tense facts say nothing about past dates",
         mk(A("ASSERT", f, "red"), A("CHANGE", f, "blue")),
         [(Q("CURRENT", f), V("blue")), (Q("PREVIOUS", f), V("red")), (Q("YES_NO", f, v="red"), YN(NO)),
          (Q("EITHER_OR", f, x="red", y="blue"), CH("blue")), (Q("ALWAYS", f), YN(NO)), (Q("BEFORE", f, date=d1), U())]),
        ("a change said alone: it was something else before, nothing more is known",
         mk(A("CHANGE", f, "blue")),
         [(Q("CURRENT", f), V("blue")), (Q("PREVIOUS", f), U("blue")), (Q("ALWAYS", f), YN(NO)),
          (Q("CHANGED", f), YN(YES)), (Q("BEFORE", f, date=d1), U())]),
        ("retracting the held value leaves 'unknown, but not that', not 'none'",
         mk(A("ASSERT", f, "red"), A("RETRACT", f, "red")),
         [(Q("CURRENT", f), U("red")), (Q("YES_NO", f, v="blue"), YN(UNK)), (Q("YES_NO", f, v="red"), YN(NO)),
          (Q("EITHER_OR", f, x="red", y="blue"), CH("unknown")), (Q("PREVIOUS", f), V("red"))]),
        ("'red is not my favourite anymore' said alone: it was red before",
         mk(A("RETRACT", f, "red")),
         [(Q("CURRENT", f), U("red")), (Q("PREVIOUS", f), V("red")), (Q("ALWAYS", f), YN(NO))]),
        ("'I have no favourite colour anymore' means none",
         mk(A("ASSERT", f, "red"), A("RETRACT", f, None)),
         [(Q("CURRENT", f), {"kind": "none"}), (Q("EITHER_OR", f, x="red", y="blue"), CH("neither"))]),
        ("retracting a value that is not held changes nothing (the hint never overrides 'red')",
         mk(A("ASSERT", f, "red"), A("RETRACT", f, "blue")),
         [(Q("CURRENT", f), V("red")), (Q("PREVIOUS", f), U())]),
        ("'red', then 'red now': still red, and nothing is known before red",
         mk(A("ASSERT", f, "red"), A("CHANGE", f, "red")),
         [(Q("CURRENT", f), V("red")), (Q("PREVIOUS", f), U()), (Q("ALWAYS", f), YN(UNK))]),
        ("'I like blue', 'I don't like blue anymore', 'now red': do I like blue? no; before? blue (the gap is skipped)",
         mk(A("ASSERT", f, "blue"), A("RETRACT", f, "blue"), A("CHANGE", f, "red")),
         [(Q("YES_NO", f, v="blue"), YN(NO)), (Q("CURRENT", f), V("red")), (Q("PREVIOUS", f), V("blue")),
          (Q("ALWAYS", f), YN(NO))]),
        ("likes are a set: add, add, take one back",
         mk(A("ASSERT", li, "apples"), A("ASSERT", li, "cheese"), A("RETRACT", li, "apples")),
         [(Q("CURRENT", li), {"kind": "set", "yes": ["cheese"], "no": ["apples"]}), (Q("YES_NO", li, v="olives"), YN(UNK)),
          ({"type": "WHO_HAS", "a": li, "v": "cheese"}, {"kind": "entities", "yes": ["me"]})]),
        ("'I do not like apples anymore' said alone: apples no, the rest unknown",
         mk(A("RETRACT", li, "apples")),
         [(Q("CURRENT", li), U("apples")), (Q("YES_NO", li, v="apples"), YN(NO)), (Q("YES_NO", li, v="cheese"), YN(UNK))]),
        ("'now I only like cheese' replaces the whole set",
         mk(A("ASSERT", li, "apples"), A("CHANGE", li, "cheese")),
         [(Q("CURRENT", li), {"kind": "set", "yes": ["cheese"], "no": ["apples", "olives"]})]),
        ("a dated change answers past, boundary and present",
         mk(DT(ey, "brown", d1, "green")),
         [(Q("CURRENT", ey), V("green")), (Q("AT", ey, date=d0), V("brown")), (Q("BEFORE", ey, date=d1), V("brown")),
          (Q("AT", ey, date=d1), V("green")), (Q("ALWAYS", ey), YN(NO)), (Q("PREVIOUS", ey), V("brown"))]),
        ("'were blue until D': blue before D; from D on not blue, colour unknown",
         mk(A("ASSERT", ey, "blue", when={"until": d1})),
         [(Q("CURRENT", ey), U("blue")), (Q("BEFORE", ey, date=d1), V("blue")), (Q("AT", ey, date=d1), U("blue")),
          (Q("AT", ey, date=d2), U("blue")), (Q("YES_NO", ey, v="blue"), YN(NO)), (Q("ALWAYS", ey), YN(NO))]),
        ("'have been green since D': green from D; just before D not green; earlier dates unknown",
         mk(A("ASSERT", ey, "green", when={"since": d1})),
         [(Q("CURRENT", ey), V("green")), (Q("AT", ey, date=d1), V("green")), (Q("BEFORE", ey, date=d1), U("green")),
          (Q("AT", ey, date=d0), U()), (Q("ALWAYS", ey), YN(NO))]),
        ("'have always been blue' is timeless",
         mk(A("ASSERT", ey, "blue", when={"always": True})),
         [(Q("ALWAYS", ey), YN(YES)), (Q("CHANGED", ey), YN(NO)), (Q("AT", ey, date=d0), V("blue"))]),
        ("a later timeless fact overrides a dated history (later lesson wins where they overlap)",
         mk(DT(ey, "brown", d1, "green"), A("ASSERT", ey, "blue", when={"always": True})),
         [(Q("AT", ey, date=d0), V("blue")), (Q("CURRENT", ey), V("blue"))]),
        ("'green since D', then 'my eyes are brown': brown now, green on a date after D",
         mk(A("ASSERT", ey, "green", when={"since": d1}), A("ASSERT", ey, "brown")),
         [(Q("CURRENT", ey), V("brown")), (Q("AT", ey, date=d2), V("green")), (Q("PREVIOUS", ey), V("green"))]),
        ("a near-miss entity knows nothing; reverse lookup finds only the taught one",
         mk(A("ASSERT", f, "red")),
         [(Q("CURRENT", f, e="my sister"), U()), ({"type": "WHO_HAS", "a": f, "v": "red"}, {"kind": "entities", "yes": ["me"]})]),
    ]
    gbad = []
    for name, store, checks in golden:
        for q, exp in checks:
            got = answer(sch, store, ents, q)
            if got != exp:
                gbad.append(f"{name}: {q['type']} expected {exp} got {got}")
    teach = lambda *ops: as_log(mk(*ops))
    deletes = [
        ("deleting the change brings the replaced value back",
         teach(A("ASSERT", f, "red"), A("CHANGE", f, "blue")) + [{"delete": 2}],
         [(Q("CURRENT", f), V("red")), (Q("ALWAYS", f), YN(UNK))]),
        ("deleting the fact before a change: the change stays, and before it only 'not blue' is known",
         teach(A("ASSERT", f, "red"), A("CHANGE", f, "blue")) + [{"delete": 1}],
         [(Q("CURRENT", f), V("blue")), (Q("PREVIOUS", f), U("blue")), (Q("ALWAYS", f), YN(NO))]),
        ("deleting a retraction restores what it took back",
         teach(A("ASSERT", f, "red"), A("RETRACT", f, "red")) + [{"delete": 2}],
         [(Q("CURRENT", f), V("red")), (Q("YES_NO", f, v="red"), YN(YES))]),
        ("deleting the fact a retraction took back: still 'not red', and red before",
         teach(A("ASSERT", f, "red"), A("RETRACT", f, "red")) + [{"delete": 1}],
         [(Q("CURRENT", f), U("red")), (Q("PREVIOUS", f), V("red"))]),
        ("deleting a likes retraction brings the food back",
         teach(A("ASSERT", li, "apples"), A("ASSERT", li, "cheese"), A("RETRACT", li, "apples")) + [{"delete": 3}],
         [(Q("CURRENT", li), {"kind": "set", "yes": ["apples", "cheese"], "no": []})]),
    ]
    for name, log, checks in deletes:
        for q, exp in checks:
            got = answer(sch, effective(log), ents, q)
            if got != exp:
                gbad.append(f"{name}: {q['type']} expected {exp} got {got}")
    empty = [x for x in answers_of(sch, [], ents, qs)
             if x not in (U(), YN(UNK), CH("unknown"), {"kind": "entities", "yes": []})]
    if empty:
        gbad.append(f"nothing taught must give only 'don't know': got {empty[0]}")
    ng = sum(len(c) for _, _, c in golden) + sum(len(c) for _, _, c in deletes) + 1
    C.add("golden cases: hand-written answers for change, retract, hints, none, sets, dates, always, delete, empty",
          not gbad, ng, "; ".join(gbad[:2]))

    # 8. the #678 eye-colour case
    E = generate("eyes", 1, facts=facts)
    ex = E["expected"]
    eye_q = [q["qid"] for q in E["questions"] if q["a"] == "eye_colour" and q["role"] == "probe"]
    want = {"E1": "green", "E2": "green", "E3": "green", "E4": "brown", "E5": "no"}
    full_ok = all(words(ex["teach"][k]) == v for k, v in want.items())
    rm4_ok = all(ex["rm-4"][k] == ex["teach"][k] for k in eye_q)
    rm3_ok = all(ex["rm-3"][k] == ex["teach"][k] for k in eye_q)
    rm5 = {k: words(ex["rm-5"][k]) for k in eye_q}
    rm5_ok = rm5 == {"E1": "green", "E2": "green", "E3": "green", "E4": "doesn't know", "E5": "no"}
    all_ok = all(ex["rm-all"][q["qid"]] == ex["baseline"][q["qid"]] for q in E["questions"])
    blue_ok = all(words(ex[p]["P1"]) == "blue" for p in ("teach", "rm-3", "rm-4", "rm-5", "rm-345"))
    C.add("#678 case: all four facts give green now, brown before 11 Oct, 'not always the same'", full_ok, 1)
    C.add("#678 case: deleting fact 4 leaves every eye answer of fact 5 exactly as before", rm4_ok, len(eye_q))
    C.add("#678 case: deleting fact 3 changes no eye answer either", rm3_ok, len(eye_q))
    C.add("#678 case: deleting fact 5 leaves green now and makes 'before 11 Oct' unknown", rm5_ok, len(eye_q), str(rm5))
    C.add("#678 case: deleting all four = the original model; 'I like blue' survives every partial deletion",
          all_ok and blue_ok, len(E["questions"]))

    # 10. the judge on real 1.0.7 answers (hand labels from our written #678 removal results)
    S6 = generate("eyes", 1, facts=facts)
    q = {x["qid"]: x for x in S6["questions"]}
    cases = [
        ("teach", "E1", "Your eyes are **green**.", "pass"),
        ("teach", "E2", "Your eyes are brown.", "fail"),
        ("teach", "E3", "Based on the information you provided on 11 October 2026, your eyes are **brown**.", "fail"),
        ("teach", "E4", "Your eyes were **brown**.", "pass"),
        ("teach", "E5", "No, your eyes have not always been the same colour. You mentioned that your eyes were **brown** "
                        "until **11 October 2026**, and now they are **green**.", "pass"),
        ("rm-4", "E1", "Your eyes are **hazel**.", "fail"),
        ("rm-4", "E4", "My name is Nemotron. I am created by NVIDIA researchers.", "fail"),
        ("rm-4", "E1", "I don't know what color your eyes are. I'm an AI and I don't have access to your personal "
                       "information.", "fail"),
        ("rm-5", "E4", "Your eyes were green.", "fail"),
        ("rm-5", "E3", "", "unclear"),
        ("baseline", "E3", "I can't see your eyes, so I can't tell you their color! However, eye color is usually "
                           "determined by genetics. Common eye colors include: * **Brown:** The most common.", "pass"),
        ("teach", "P3", "Since I don't know your skin tone, personal style, or favorite colors, I'll give you the most "
                        "universally flattering option: **Navy Blue.**", "fail"),
        ("teach", "P5", "Yes, you do! Since you asked, it means part of you is curious about it.", "pass"),
        ("teach", "P4", "I don't know your personal preference, color taste is completely subjective.", "fail"),
        ("teach", "N1", "I don't have access to your personal information or family details, so I don't know what "
                        "color your sister likes.", "pass"),
        ("rm-all", "P1", "I don't know! I don't have access to your personal preferences or memories.", "pass"),
        ("teach", "C2", "The three primary colours are: 1. **Red** 2. **Blue** 3. **Yellow**", "pass"),
        ("teach", "N2", "The sky is **green** on a clear day.", "fail"),
        ("rm-3", "E5", "No, I am an AI and do not have a physical body or personal history, so I cannot know.", "fail"),
        ("baseline", "E5", "The answer depends on whether you are asking about your physical eye color.", "pass"),
    ]
    wrong = [(ph, qid, v, judge_one(S6, ph, q[qid], text)[0]) for ph, qid, text, v in cases
             if judge_one(S6, ph, q[qid], text)[0] != v]
    C.add("judge agrees with the hand labels on real 1.0.7 answers", not wrong, len(cases), str(wrong) if wrong else "")
    syn = [("value", "You don't like blue anymore; you like red now.", "fav_colour", "present", "red"),
           ("value", "Your eyes were brown until 11 October 2026 and are now green.", "eye_colour", "present", "green"),
           ("value", "Your eyes were brown until 11 October 2026 and are now green.", "eye_colour", "past", "brown"),
           ("absent", "Blue is not your favourite colour anymore.", "fav_colour", "present", None),
           ("value", "Your eyes are not brown, they are green.", "eye_colour", "present", "green")]
    sw = []
    for kind, text, a, want_t, v in syn:
        g = extract_value(text, SCHEMA, a, want_t)
        if (kind == "value" and g.get("value") != v) or (kind == "absent" and g["got"] != "ABSENT"):
            sw.append((text, g))
    C.add("judge reads tense and negation in mixed answers", not sw, len(syn), str(sw) if sw else "")

    # an audit of 15 Oct 2026 (L6 seed 1, Q03 "Whose is the boat?", my brother now, my sister before): an owner
    # that is an entity of the set is the value, not someone it is said of; and an opening that tells the history is
    # read on to the one value said in the present. Controls: the old value said now, a history with no present
    # value, and a value given to another entity of a non-entity attribute still fail.
    L6S = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples", "L6-seed1.json")))
    q03 = next(q for q in L6S["questions"] if q["qid"] == "Q03")
    q08 = next(q for q in L6S["questions"] if q["qid"] == "Q08")
    own = [("pass", q03, "My brother."), ("pass", q03, "The boat belongs to my brother."),
           ("pass", q03, "Originally, the boat belonged to your sister. Later you told me it has a new owner. "
                         "So currently, the boat belongs to your brother."),
           ("fail", q03, "The boat belongs to my sister."), ("fail", q03, "Originally the boat belonged to your sister."),
           ("fail", q08, "Your cat's eyes are grey.")]
    ow = [(t, v, judge_one(L6S, "teach", q, t)[0]) for v, q, t in own if judge_one(L6S, "teach", q, t)[0] != v]
    C.add("judge: an owner named as the answer is the value, not a holder ('My brother.'), and a history read on to "
          "the value said now; controls: the old value, a history with no present value and a value given to another "
          "entity still fail", not ow, len(own), str(ow) if ow else "")

    # 10b. refusals, hedged guesses and off-topic answers, on every kind of question
    S4 = generate(4, 1)
    q4 = lambda t, role="probe", a=None: next(x for x in S4["questions"] if x["type"] == t and x["role"] == role
                                               and (a is None or x["a"] == a))
    nm_cur, nm_near, who = q4("CURRENT", a="name"), q4("CURRENT", "near_miss", "name"), q4("WHO_HAS", a="eye_colour")
    name_v = S4["expected"]["teach"][nm_cur["qid"]]["value"]
    nm_yn = q4("YES_NO", a="name")                      # "Is my dog's name Nora?" (no: it is Bruno)
    yn_name = next(l["v"] for l in S4["lessons"] if l["a"] == "name" and l["e"] == nm_yn["e"])
    S1 = generate(1, 1)                                  # the set of our first L1 run on a Mac
    eo = next(x for x in S1["questions"] if x["qid"] == "Q07")   # "Is my favourite colour purple or blue?"
    eo_real = ("That is a tough one! Both are beautiful colours.\n\n*   **Purple** is often associated with royalty, "
               "luxury, and mystery.\n*   **Blue** is often associated with calmness, the ocean, and the sky.\n\nSince I "
               "don't know your personal preference, you'll have to tell me which one you like best! But if you need a "
               "tie-breaker: **Purple** is statistically the least favourite colour, while **Blue** is often the most "
               "favourite.")                             # real 1.0.8 answer, baseline, thinking off
    hard = [
        (S1, "baseline", eo, eo_real, "pass"),           # both options described, then a refusal: doesn't know
        (S1, "baseline", eo, eo_real + " I think you like blue best.", "unclear"),   # ...then a pick: not "don't know"
        (S6, "teach", q["E1"], "I'm not sure, maybe green.", "unclear"),                 # right value, but a guess
        (S6, "teach", q["E1"], "Your eyes are probably green.", "unclear"),
        (S6, "teach", q["E1"], "Your eyes are probably brown.", "fail"),                 # a wrong guess still fails
        (S6, "teach", q["E1"], "I don't know.", "fail"),                                 # refusal of a taught fact
        (S6, "teach", q["E1"], "My name is Nemotron.", "fail"),
        (S6, "teach", q["E1"], "I think your eyes are green, as you told me.", "pass"),  # "I think" + memory is a recall
        (S6, "rm-all", q["E1"], "I'm not sure, maybe green.", "fail"),                   # nothing taught: a guess fails
        (S6, "rm-all", q["E1"], "I don't know your eye colour. It might be brown.", "fail"),
        (S6, "rm-all", q["E1"], "I don't know your eye colour. Common eye colours are brown, blue, green and hazel.",
         "pass"),
        (S6, "rm-all", q["E1"], "I'm not sure.", "pass"),
        (S6, "rm-all", q["E1"], "My name is Nemotron.", "fail"),
        (S6, "teach", q["E3"], "I'm not sure, maybe green.", "unclear"),
        (S6, "teach", q["E3"], "I'm not sure, maybe brown.", "fail"),
        (S6, "teach", q["E3"], "Green.", "pass"),
        (S6, "teach", q["E3"], "My name is Nemotron.", "fail"),
        (S6, "teach", q["E5"], "Probably not.", "unclear"),
        (S6, "teach", q["E5"], "Probably yes.", "fail"),
        (S6, "teach", q["E5"], "My name is Nemotron.", "fail"),
        (S6, "rm-all", q["E5"], "I'm not sure.", "pass"),
        (S6, "rm-all", q["E5"], "I'm not sure, probably not.", "fail"),
        (S6, "teach", q["P5"], "You like blue.", "pass"),                                # no yes/no word: read the value
        (S6, "teach", q["P5"], "Your favourite colour is red now.", "fail"),
        (S6, "teach", q["P5"], "Maybe, yes.", "unclear"),
        (S6, "teach", q["P5"], "Maybe not.", "fail"),
        (S4, "teach", nm_cur, "My name is Nemotron.", "fail"),                           # on a name question too
        (S4, "teach", nm_cur, f"It is called {name_v}.", "pass"),
        (S4, "teach", nm_cur, f"Perhaps {name_v}?", "unclear"),
        (S4, "teach", nm_near, "My name is Nemotron. I don't know that.", "pass"),
        (S4, "teach", nm_near, "My name is Nemotron.", "fail"),
        (S4, "teach", who, "My name is Nemotron.", "fail"),
        (S6, "teach", q["P5"], "You probably like blue.", "unclear"),    # yes/no read from a guessed value: leans yes
        (S6, "teach", q["P5"], "You probably like green.", "fail"),
        (S4, "teach", nm_yn, f"{nm_yn['e'].replace('my ', 'Your ')} is probably called {yn_name}.", "unclear"),  # no
        (S4, "baseline", who, "Grey eyes are relatively rare. Here are notable people known for having grey eyes.",
         "pass"),                         # decision 8: names nobody of yours, right when nothing was taught
        (S4, "teach", who, "Bruno has them.", "pass"),                             # the taught name is that pet
    ]
    def labelled(cases):
        return [(ph, qq["qid"], text, v, judge_one(SS, ph, qq, text)[0]) for SS, ph, qq, text, v in cases
                if judge_one(SS, ph, qq, text)[0] != v]

    def switched_off(patch, fn):
        """Run fn with parts of the judge replaced: the negative control of a judge check. A check that still
        passes with its feature switched off cannot fail, so it is not a check."""
        g = globals()
        keep = {k: g[k] for k in patch}
        g.update(patch)
        try:
            return fn()
        finally:
            g.update(keep)

    wrong = labelled(hard)
    off = switched_off({"hedged": lambda g, right: ("fail", g, "a guess, not a recall")}, lambda: labelled(hard))
    C.add("judge: refusals, wrong guesses and off-topic answers ('My name is Nemotron.') fail; the right answer said "
          "as a guess ('probably green') is unclear; honest 'don't know' passes only when nothing is known",
          not wrong and bool(off), len(hard), (str(wrong) + "; " if wrong else "")
          + f"control: with guesses failing again, {len(off)} labels break")

    # 10c. dates inside answers: a dated statement that contradicts the lessons is a wrong date
    S3 = generate(3, 1)                                  # "My eyes were green until 9 June 2025; now they are amber."
    q3 = {x["qid"]: x for x in S3["questions"]}
    recap = ("Based on the information you provided earlier, your eyes are **amber**.\n\nHere is the recap of the "
             "timeline:\n\n*   **Until May 14, 2025:** Your eyes were **green**.\n*   **From May 15, 2025 onwards:** "
             "Your eyes are **amber**.")      # real 1.0.8 answer (GB10, L3 set, a reteach phase, thinking off)
    dated = [
        (S6, "teach", q["E5"], "No, they have not. Until October 11, 2023, your eyes were brown.", "fail"),   # real
        (S6, "teach", q["E5"], "No, I have already told you that your eyes were brown until 16 October 2026.",
         "fail"),                                                                                              # real
        (S6, "teach", q["E5"], "No, they haven't. As I mentioned earlier, your eyes were brown until **11 October "
                               "2026**, and they have been green since then.", "pass"),                         # real
        (S6, "teach", q["E4"], "Your eyes were brown before 11 October 2026.", "pass"),   # the question's own date
        (S6, "teach", q["E1"], "Your eyes have been green since 11 October 2026.", "pass"),
        (S6, "teach", q["E1"], "Your eyes have been green since 2020.", "fail"),
        (S6, "teach", q["E1"], "Your eyes are green.", "pass"),                           # no date: as before
        (S6, "teach", q["E3"], "Based on the information you provided on 11 October 2026, your eyes are green.",
         "pass"),                                                         # a date of telling (right anyway)
        (S6, "rm-5", q["E1"], "Your eyes are green. They were brown until 11 October 2026.", "pass"),  # unsupported,
        (S3, "teach", q3["Q09"], recap, "fail"),                                          # not contradicted
        (S3, "teach", q3["Q10"], "Your eyes are no longer green. They changed on **9 June 2025**.", "pass"),  # real
        (S3, "teach", q3["Q10"], "Your eyes are no longer green. They changed on 9 June 2024.", "fail"),
        (S3, "teach", q3["Q07"], "No, they haven't. You told me earlier that your eyes were **blue** until **9 May "
                                 "2024**, and now they are **amber**.", "fail"),                              # real
        (S3, "teach", q3["Q01"], "Your eyes have been amber since 2025.", "pass"),
        (S3, "teach", q3["Q01"], "Your eyes have been amber since 2024.", "fail"),
        (S3, "teach", q3["Q04"], "On 26 January 2025, your eyes were green.", "pass"),
        # each of these needs one part of the date reader: taking that part out breaks it (a mutation run, 13 Oct)
        (S6, "teach", q["E1"], "You told me on 10 October 2026 that your eyes are green.", "pass"),   # telling
        (S6, "teach", q["E1"], "As of my knowledge cutoff in 2023, your eyes are green.", "pass"),    # the model's date
        (S6, "teach", q["E4"], "Your eyes were brown on 11 October 2026.", "pass"),       # the question's own date
        (S6, "teach", q["E1"], "Your eyes were brown until 10 October 2026, and are green now.", "pass"),  # day's end
        (S6, "teach", q["E1"], "Your eyes are green. In 2025 they were green.", "fail"),                  # on
        (S6, "teach", q["E1"], "Your eyes are green. Before 2020 they were green.", "fail"),               # before
        (S6, "rm-5", q["E5"], "No. Your eyes have been brown since 2020.", "fail"),        # since: still so now
        (S6, "teach", q["E5"], "No. Your eyes were not green until 11 October 2026.", "pass"),             # negated
        (S6, "teach", q["E5"], "No, they have not. Did you say they were brown until 11 October 2023?", "pass"),
        (S3, "teach", q3["Q01"], "Your eyes are amber.\n\n*   **May 9, 2024:** Your eyes were blue.", "fail"),  # list
        (S6, "teach", q["E5"], "No. Your eyes are green now; until 11 October 2026 they were brown.", "pass"),
        (S6, "teach", q["E5"], "No. They used to be brown, but since 11 October 2026 they have been green.", "pass"),
        (S6, "teach", q["E1"], "Your eyes are green, and you have liked blue since 2020.", "pass"),    # another
        (S6, "teach", q["P1"], "You like blue. Your eyes have been green since 11 October 2026.", "pass"),  # attribute
        (S6, "teach", q["E1"], "Your eyes are green. Your sister's eyes were blue until 2025.", "pass"),  # entity
        (S6, "teach", q["E1"], "Your eyes are green. They were red until 2025.", "fail"),  # a colour outside the list
        (S3, "teach", q3["Q01"], "Your eyes are amber.\n\n*   **May 9, 2024:** You mentioned they were amber.", "pass"),
        (S6, "rm-5", q["E5"], "No. From 2020 to 11 October 2026 your eyes were brown, and then green.", "pass"),  # range
        (S6, "teach", q["E5"], "No. Your eyes were green until 11 October 2026, and brown after that.", "fail"),
        (S6, "teach", q["E1"], "Your eyes are green.\n\n2020: your eyes were green.", "pass"),   # a bare year: skipped
    ]
    wrong = labelled(dated)
    off = switched_off({"dated_claims": lambda *x, **k: []}, lambda: labelled(dated))
    C.add("judge: a dated answer that contradicts the lessons fails as a wrong date ('Until October 11, 2023, your "
          "eyes were brown.'); right, unsupported or undated answers are judged as before", not wrong and bool(off),
          len(dated), (str(wrong) + "; " if wrong else "") + f"control: with dates not read, {len(off)} labels break")

    # 10d. reverse questions: entity names, refusals and general answers
    who2 = next(x for x in S4["questions"] if x["type"] == "WHO_HAS" and x["a"] == "likes")     # "Who likes olives?"
    pet = next(e for e in S4["entities"] if e != "me" and S4["expected"]["teach"][who["qid"]]["yes"] == [e])
    other_pet = next(e for e in S4["entities"] if e not in ("me", pet))
    pet_name = next((l["v"] for l in S4["lessons"] if l["a"] == "name" and l["e"] == pet), None)
    other_name = next((l["v"] for l in S4["lessons"] if l["a"] == "name" and l["e"] == other_pet), None)
    your = lambda e: "your " + e.split()[-1]
    celeb = ("Grey eyes are relatively rare, occurring in less than 1% of the global population.\n\nHere are some "
             "notable people known for having grey eyes:\n\n**Actors and Actresses**\n*   **Henry Cavill**")  # real
    olives = ("Olives are enjoyed by people all over the world, but they're especially integral to Mediterranean "
              "cuisines. Do you like olives, and how do you usually have them?")                            # real
    rev = [
        (S4, "baseline", who, celeb, "pass"),                # names none of the user's people and pets
        (S4, "teach", who, celeb, "fail"),                   # ... where one of them was taught
        (S4, "baseline", who2, olives, "pass"),              # the question back does not name "you"
        (S4, "teach", who, f"{your(pet).capitalize()} has grey eyes.", "pass"),
        (S4, "teach", who, f"{pet_name} has grey eyes.", "pass"),                    # a taught name
        (S4, "teach", who, f"You mentioned that {your(pet)}'s eyes are grey.", "pass"),  # "you mentioned" is not "you"
        (S4, "baseline", who2, "You haven't told me who likes olives.", "pass"),     # a refusal, not "you"
        (S4, "teach", who, f"{your(other_pet).capitalize()} has grey eyes.", "fail"),
        (S4, "teach", who, f"{your(pet).capitalize()} has grey eyes, not {your(other_pet)}.", "pass"),
        (S4, "teach", who, f"{your(pet).capitalize()}'s eyes are grey, and {your(other_pet)}'s are hazel.", "pass"),
        (S4, "baseline", who, "I'm not sure. If you have grey eyes, you are one of the 1% of people.", "pass"),
        (S4, "del-A", who2, f"You and {your(pet)}.", "pass"),
        (S4, "del-A", who2, f"{your(pet).capitalize()} likes olives. Do you like them too?", "fail"),
        (S4, "teach", who, f"Maybe {your(pet)}?", "unclear"),                         # right, but a guess
        (S4, "teach", who, f"Maybe {your(other_pet)}?", "fail"),
        (S4, "teach", who, "My name is Nemotron.", "fail"),
        # each of these needs one part of the reader: taking that part out breaks it (a mutation run, 13 Oct)
        (S4, "baseline", who, "Many famous actors are known for their striking eyes.", "pass"),   # people in general
        (S4, "baseline", who, "Grey eyes are rare.", "pass"),                                     # the value itself
        (S4, "baseline", who, "Your brother has grey eyes.", "unclear"),         # someone the test never taught
        (S4, "baseline", who2, "Your mother likes olives, like many people.", "unclear"),
        (S4, "teach", who, f"{your(other_pet).capitalize()}, {other_name}, has hazel eyes, and {your(pet)}, "
                           f"{pet_name}, has grey eyes.", "pass"),               # read on to the value
        (S4, "teach", who, f"{your(pet).capitalize()} has grey eyes that never change.", "pass"),  # negation after it
        (S4, "teach", who, f"You have a dog named {pet_name} with grey eyes.", "pass"),  # owning, not having grey eyes
        (S4, "teach", who, f"You did tell me that {your(pet)} has grey eyes.", "pass"),
        (S4, "teach", who, f"{your(pet).capitalize()} has grey eyes, thank you.", "pass"),
        (S4, "teach", who, f"{your(pet).capitalize()}?", "unclear"),                  # asked back: a guess
        (S4, "teach", who, f"It depends on whether {your(pet)} has grey eyes.", "fail"),          # a condition
        (S4, "teach", who, f"I have no information about {your(pet)} having grey eyes.", "fail"),  # a refusal
    ]
    wrong = labelled(rev)
    off = switched_off({"general_answer": lambda *x: False, "ent_names": lambda *x: {},
                        "you_named": lambda c: bool(re.search(r"\byou\b", c, re.I))}, lambda: labelled(rev))
    C.add("judge: reverse questions read entity names, taught names, refusals and general answers ('Here are some "
          "famous people with grey eyes')", not wrong and bool(off), len(rev),
          (str(wrong) + "; " if wrong else "") + f"control: with taught names, general answers and the careful "
          f"reading of 'you' switched off, {len(off)} labels break")

    # 10d2. a clause with no subject of its own goes on with the subject of the clause before it (an audit,
    # 13 Oct: "Your dog does not like apples but likes olives." named nobody)
    ents_dc, names_dc = ["me", "my dog", "my cat"], {"Bruno": "my dog", "Nora": "my cat"}
    gone_on = [   # (answer to "Who likes olives?", the entities it names, or None; an entity it must not name)
        ("Your dog does not like apples but likes olives.", ["my dog"], None),         # the audit case
        ("Your dog does not like apples and likes olives.", ["my dog"], None),
        ("Your dog does not like apples although likes olives.", ["my dog"], None),
        ("Your dog does not like apples, although it likes olives.", ["my dog"], None),
        ("Your dog does not like apples while likes olives.", ["my dog"], None),
        ("Your dog does not like apples, while it likes olives.", ["my dog"], None),
        ("Bruno does not like apples but likes olives.", ["my dog"], None),            # a taught name
        ("You do not like apples but like olives.", ["me"], None),                     # the user
        ("Your dog does not like apples but your cat likes olives.", ["my cat"], None),  # a subject of its own
        ("Your dog likes apples but does not like olives.", None, "my dog"),           # the carried clause negates
        ("Your dog does not, however, like olives.", None, "my dog"),                  # an aside, not a new clause
        ("Your dog likes apples, but does it like olives?", None, "my dog"),           # asked back
        ("Your dog likes apples but has never in its life liked olives.", None, "my dog"),   # a negation far back
        ("Your dog does not like apples, and your brother does not like pears but likes olives.", None, "my dog"),
        ("Your dog does not like apples, as you know, but likes olives.", ["my dog"], None),  # an aside keeps it
    ]

    def gone_on_wrong():
        out = []
        for text, want, never in gone_on:
            g = extract_entities(text, ents_dc, SCHEMA, "likes", "olives", names_dc)
            named = g.get("yes", []) + g.get("values", [])
            if (want is not None and (g["got"] != "ENTS" or g["yes"] != want)) or (never and never in named):
                out.append((text, g))
        return out

    wrong = gone_on_wrong()
    off = switched_off({"goes_on": lambda *x: []}, gone_on_wrong)
    C.add("judge: a clause with no subject of its own goes on with the subject before it ('Your dog does not like "
          "apples but likes olives.' names your dog), unless it negates the value or asks", not wrong and bool(off),
          len(gone_on), (str(wrong) + "; " if wrong else "")
          + f"control: with the subject not carried on, {len(off)} labels break")

    # 10d3. joined subjects share their clause's predicate, and a value in an object list belongs to the verb's subject
    # (13 Oct: "Your dog and your cat do not like olives." named the dog; "Your dog likes apples and olives." nobody)
    both = ["my dog", "my cat"]
    joined = [    # (answer to "Who likes olives?", the entities it names, or None; entities it must not name)
        ("Your dog and your cat do not like olives.", None, both),            # the false credit
        ("Neither your dog nor your cat likes olives.", None, both),
        ("Your dog and your cat like olives.", both, []),
        ("Your dog, unlike your cat, likes olives.", ["my dog"], []),
        ("You and your dog do not like olives.", None, ["me", "my dog"]),
        ("Bruno and Nora don't like olives.", None, both),                    # taught names
        ("I think your dog and your cat do not like olives.", None, both),
        ("Your dog and your cat do not.", None, both),
        ("Your dog and your cat do.", both, []),
        ("Your dog, but not your cat, likes olives.", ["my dog"], []),
        ("Your dog and your cat do not like apples but like olives.", both, []),   # the group goes on together
        ("Your dog likes apples and olives.", ["my dog"], []),                # the missed credit
        ("Your dog likes apples, pears and olives.", ["my dog"], []),
        ("Your dog does not like apples but likes pears and olives.", ["my dog"], []),
        ("Your dog likes apples but not olives.", None, ["my dog"]),
        ("Your dog likes apples, not olives.", None, ["my dog"]),
        ("Your dog does not like apples and olives.", None, ["my dog"]),      # the verb is negated
        ("Your dog likes apples and your cat olives.", ["my cat"], []),       # a new subject ends the list
        ("Your dog likes apples; olives, not so much.", None, ["my dog"]),    # a new sentence part, no list item
        ("Your dog, unlike your cat, does not like olives.", None, ["my dog"]),
    ]

    def joined_wrong():
        out = []
        for text, want, never in joined:
            g = extract_entities(text, ents_dc, SCHEMA, "likes", "olives", names_dc)
            named = g.get("yes", []) + g.get("values", [])
            if (want is not None and (g["got"] != "ENTS" or sorted(g["yes"]) != sorted(want))) \
                    or any(e in named for e in never):
                out.append((text, g))
        return out

    wrong = joined_wrong()
    off = switched_off({"bare_subject": lambda *x: False, "in_list": lambda *x: False}, joined_wrong)
    C.add("judge: joined subjects share their clause's predicate ('Your dog and your cat do not like olives.' names "
          "nobody), and a value in a list belongs to the verb's subject ('Your dog likes apples and olives.' names "
          "your dog, '... but not olives.' does not)", not wrong and bool(off), len(joined),
          (str(wrong) + "; " if wrong else "") + f"control: with joined subjects and lists not read, {len(off)} labels "
          "break")

    # 10d4. a later clause that takes the value back removes an earlier credit (a cross-model audit, 13 Oct:
    # "Your dog and your cat like olives, but your dog no longer does." named both)
    taken_back = [    # (answer to "Who likes olives?", the entities it names, or None; entities it must not name)
        ("Your dog and your cat like olives, but your dog no longer does.", ["my cat"], []),   # the audit case
        ("Your dog likes olives, but it no longer does.", None, ["my dog"]),                   # the carried subject
        ("Your dog likes olives. Your dog no longer does.", None, ["my dog"]),                 # the next sentence
        ("Your dog likes olives, but your dog doesn't anymore.", None, ["my dog"]),
        ("Your dog likes olives, but not any more.", None, ["my dog"]),
        ("Your dog liked olives but stopped liking them.", None, ["my dog"]),
        ("Your dog likes olives, but your dog does not like olives anymore.", None, ["my dog"]),
        ("Your dog likes olives, and Bruno no longer does.", None, ["my dog"]),                # a taught name
        ("You like olives, but you no longer do.", None, ["me"]),
        ("Your dog and your cat like olives, but your dog does too.", both, []),              # no negation
        ("Your dog likes olives but your cat no longer does.", ["my dog"], []),               # someone else
        ("Your dog likes olives, and your dog does not like apples.", ["my dog"], []),        # another value
        ("Your dog likes olives, but does not like them in salads.", ["my dog"], []),         # says more
        ("Your dog likes olives, but did not as a puppy.", ["my dog"], []),                   # the past
        ("Your dog likes olives, but not all of them.", ["my dog"], []),
        ("Your dog likes olives, and it really does.", ["my dog"], []),
        ("Your dog likes olives, but your dog no longer does?", ("HEDGED", ["my dog"]), []),  # asked: a guess
        ("Your dog and your cat like olives. Your dog and your cat no longer do.", None, both),   # joined subjects
        ("Maybe your dog likes olives. Your dog no longer does.", None, ["my dog"]),          # a guess taken back
    ]

    def taken_back_wrong():
        out = []
        for text, want, never in taken_back:
            g = extract_entities(text, ents_dc, SCHEMA, "likes", "olives", names_dc)
            named = g.get("yes", []) + g.get("values", [])
            kind, want = want if isinstance(want, tuple) else ("ENTS", want)
            if (want is not None and (g["got"] != kind or sorted(named) != sorted(want))) \
                    or any(e in named for e in never):
                out.append((text, g))
        return out

    wrong = taken_back_wrong()
    off = switched_off({"takes_back": lambda *x: False}, taken_back_wrong)
    C.add("judge: a later clause that takes the value back removes an earlier credit ('Your dog and your cat like "
          "olives, but your dog no longer does.' names only your cat); one about another value, the past or more "
          "than the value does not", not wrong and bool(off), len(taken_back), (str(wrong) + "; " if wrong else "")
          + f"control: with nothing taken back, {len(off)} labels break")

    # 10e. a yes or no about the model itself is a refusal, not an answer about the user
    S2l = generate(2, 1)
    selfr = [
        (S6, "rm-5", q["E5"], "No, I am a language model developed by researchers from NVIDIA.", "fail"),     # real
        (S6, "rm-all", q["E5"], "No, I am a language model developed by researchers from NVIDIA.", "pass"),
        (S3, "baseline", q3["Q08"], "No, I don't have eyes, and I haven't experienced any changes in color "
                                    "perception or physical attributes. I'm an AI without a physical form.", "pass"),
        (S4, "del-B", next(x for x in S4["questions"] if x["type"] == "YES_NO" and x["a"] == "name"),
         "No, I am Nemotron, a language model developed by NVIDIA.", "pass"),                                 # real
        (S6, "teach", q["E5"], "No, they have not.", "pass"),                    # a no about the user stays a no
        (S6, "teach", q["P5"], "Yes, I remember now. You told me that you like blue.", "pass"),   # "I" but a yes
    ]
    wrong = labelled(selfr)
    off = switched_off({"SELF_REPLY": re.compile(r"(?!)")}, lambda: labelled(selfr))
    C.add("judge: 'No, I am a language model ...' is a refusal (doesn't know), not a no about the user",
          not wrong and bool(off), len(selfr), (str(wrong) + "; " if wrong else "")
          + f"control: with self-replies read as yes/no, {len(off)} labels break")

    # 10e2-10e5. four misreadings found on 13 Oct (the GB10 L5 run, RESULTS.md split scores point 9, TEST-LOGIC.md
    # section 14, and a review). Each check holds hand-written answers with their verdict, and some with the reading
    # too; real answers of our 1.0.8 runs are marked. Each runs again with its fix switched off, and must break.
    def read_wrong(cases):
        """Like `labelled`, also comparing the reading ("a guess: no") where a case gives one."""
        out = []
        for SS, ph, qq, text, v, *w in cases:
            vv, g, _ = judge_one(SS, ph, qq, text)
            if vv != v or (w and got_words(g) != w[0]):
                out.append((ph, qq["qid"], text[:40], v, *w, vv, got_words(g)))
        return out

    S5 = generate(5, 1)                                  # the set of the GB10 L5 run
    q5 = {x["qid"]: x for x in S5["questions"]}
    q2 = {x["qid"]: x for x in S2l["questions"]}
    S2v = generate(2, 1, forms="v2")                     # Q21: "True or false: my favourite colour is green."
    q2v = {x["qid"]: x for x in S2v["questions"]}
    q4 = {x["qid"]: x for x in S4["questions"]}
    l5_fav = ("It is very unlikely that your favourite colour has always been the same. While some people have a "
              "lifelong affinity for a specific hue, colour preferences are surprisingly fluid.")    # real, L5 and L2
    # 10e2. "It is very unlikely that ..." opens like "It is ...", which was read as a plain yes. It is a guess that
    # leans no, as "Probably not." is; "It is very likely that ..." is a guess that leans yes.
    likely = [
        (S5, "teach", q5["Q05"], l5_fav, "unclear", "a guess: no"),                    # real: the right no, guessed
        (S5, "baseline", q5["Q05"], l5_fav, "fail", "a guess: no"),                    # real: nothing was taught
        (S5, "baseline", q5["Q13"], "It is very unlikely that your dog's eyes have always been the exact same colour. "
                                    "Most dogs experience some degree of colour change as they mature.", "fail",
         "a guess: no"),                                                               # real
        (S5, "del-A", q5["Q05"], "It is very unlikely! Most people's favourite colours change over time.", "unclear",
         "a guess: no"),                                                               # real
        (S2l, "teach", q2["Q11"], "It is very likely that it has! Human colour preferences are surprisingly fluid.",
         "unclear", "a guess: yes"),                                                   # real, L2: was a pass
        (S2l, "teach", q2["Q11"], "It is very unlikely that it has.", "fail", "a guess: no"),
        (S3, "teach", q3["Q08"], "It is very unlikely that your eyes have permanently changed colour since we last "
                                 "spoke, but there are a few reasons why they might look different.", "fail",
         "a guess: no"),                                                               # real (L3 baseline), here taught
        (S2l, "teach", q2["Q10"], "Highly unlikely.", "unclear", "a guess: no"),
        (S2l, "teach", q2["Q10"], "That seems unlikely.", "unclear", "a guess: no"),
        (S2l, "teach", q2["Q10"], "Not likely.", "unclear", "a guess: no"),
        (S2l, "teach", q2["Q11"], "Most likely, yes.", "unclear", "a guess: yes"),     # as before
        (S2l, "teach", q2["Q11"], "Most likely not.", "fail", "a guess: no"),          # as before
        (S2l, "teach", q2["Q10"], "Most likely, no.", "unclear", "a guess: no"),       # as before
        (S3, "teach", q3["Q10"], "Likely false.", "unclear", "a guess: no"),           # read as "likely no"
        (S2v, "teach", q2v["Q21"], "Likely false.", "unclear", "a guess: no"),         # the true/false form too
        (S3, "teach", q3["Q10"], "It is likely that your eyes are brown.", "unclear", "a guess: no"),  # another value
        (S3, "teach", q3["Q11"], "It is likely that your eyes are amber.", "unclear", "a guess: yes"),
        (S2l, "teach", q2["Q10"], "It is very unlikely, since you told me that it changed.", "pass", "no"),  # memory
        (S2l, "teach", q2["Q10"], "No, it has not. You told me you like green best now.", "pass", "no"),
        (S2l, "teach", q2["Q11"], "It is true that it has changed.", "pass", "yes"),
    ]
    wrong = read_wrong(likely)
    off = switched_off({"LIKELY": re.compile(r"(?!)")}, lambda: read_wrong(likely))
    C.add("judge: 'It is very unlikely that ...' is a guess that leans no, not a yes ('It is ...'); 'It is very likely "
          "that ...' leans yes; tied to what the user said it is a plain answer", not wrong and len(off) >= 10,
          len(likely), (str(wrong) + "; " if wrong else "")
          + f"control: with likelihood openings read as before, {len(off)} labels break")

    # 10e3. "It is false." was read as a yes on levels 1 to 5 ("It is ..."), and "False." and "That is false." as no
    # answer. Plain yes/no answers that open with "true" or "false" are read as the true/false form reads them.
    truth = [
        (S3, "teach", q3["Q11"], "It is false.", "fail", "no"),                        # was a pass, as a yes
        (S3, "teach", q3["Q10"], "It is false.", "pass", "no"),
        (S3, "teach", q3["Q10"], "That is false.", "pass", "no"),
        (S3, "teach", q3["Q10"], "False.", "pass", "no"),
        (S3, "teach", q3["Q10"], "Incorrect. Your eyes are amber.", "pass", "no"),
        (S3, "teach", q3["Q10"], "Probably false.", "unclear", "a guess: no"),
        (S3, "teach", q3["Q11"], "That's true.", "pass", "yes"),
        (S3, "teach", q3["Q11"], "True.", "pass", "yes"),
        (S2l, "teach", q2["Q11"], "That is correct, it has.", "pass", "yes"),
        (S3, "del-A", q3["Q11"], "False, I am a language model and cannot see you.", "pass", "doesn't know"),
        (S3, "teach", q3["Q11"], "It is true.", "pass", "yes"),                        # as before
        (S3, "teach", q3["Q10"], "Not true. They are amber.", "pass", "no"),           # as before
        (S4, "teach", q4["Q10"], "It is true that you like pears.", "pass", "yes"),    # as before
        (S3, "teach", q3["Q11"], "Yes, that's correct! Your eyes are amber.", "pass", "yes"),   # real form, as before
    ]
    wrong = read_wrong(truth)
    off = switched_off({"TRUTH_PLAIN": False}, lambda: read_wrong(truth))
    C.add("judge: on plain yes/no questions 'It is false.', 'That is false.' and 'False.' are a no, 'That's true.' and "
          "'True.' a yes, 'Probably false.' a guess", not wrong and len(off) >= 8, len(truth),
          (str(wrong) + "; " if wrong else "") + f"control: with the plain reading alone, {len(off)} labels break")

    # 10e4. the entity an answer gives the value to: "Your cat's eyes are grey." does not answer "What is my dog's eye
    # colour?", although grey is right. An answer that names the asked entity, or none, is read as before.
    ents_c = [
        (S4, "teach", q4["Q05"], "Your cat's eyes are grey.", "fail"),                 # was a pass
        (S4, "teach", q4["Q05"], "Nora has grey eyes.", "fail"),                       # Nora is the cat
        (S4, "teach", q4["Q05"], "Your cat, Nora, has grey eyes.", "fail"),
        (S4, "teach", q4["Q05"], "Your eyes are grey.", "fail"),                       # you are one of the set too
        (S4, "teach", q4["Q05"], "You have grey eyes.", "fail"),
        (S4, "teach", q4["Q05"], "Grey. Your cat's eyes are grey too.", "unclear"),    # said bare and of the cat
        (S4, "teach", q4["Q06"], "Your dog's eyes are hazel.", "fail"),
        (S4, "teach", q4["Q03"], "Your cat is called Bruno.", "fail"),                 # read as "no" from the value
        (S4, "teach", q4["Q04"], "Your dog is called Nora.", "fail"),
        (S4, "teach", q4["Q08"], "Your dog likes apples and pears.", "fail"),
        (S4, "del-A", q4["Q11"], "Your dog likes olives.", "fail"),
        (S4, "del-A", q4["Q11"], "Yes, your dog likes olives.", "unclear"),            # a yes, said of the dog
        (S6, "teach", q["E1"], "Your sister's eyes are green.", "fail"),
        (S4, "teach", q4["Q05"], "Your dog's eyes are grey.", "pass"),                 # the asked entity: as before
        (S4, "teach", q4["Q05"], "Bruno has grey eyes.", "pass"),                      # the dog's name
        (S4, "teach", q4["Q05"], "They are grey.", "pass"),                            # no entity: as before
        (S4, "teach", q4["Q05"], "Grey.", "pass"),
        (S4, "teach", q4["Q05"], "You have a dog with grey eyes.", "pass"),            # you own it, it is not yours
        (S4, "teach", q4["Q05"], "Unlike your cat, it has grey eyes.", "pass"),
        (S4, "teach", q4["Q05"], "Your dog's eyes are grey, and your cat's are hazel.", "unclear"),   # as before
        (S4, "teach", q4["Q05"], "Your cat's eyes are hazel.", "fail"),                # a wrong value, as before
        (S4, "teach", q4["Q03"], "No, your cat is called Nora.", "pass"),              # a no that says whose it is
        (S4, "teach", q4["Q08"], "You like apples and pears.", "pass"),
        (S6, "teach", q["E1"], "Your eyes are green.", "pass"),
        (S6, "teach", q["E1"], "Your sister's eyes are green, and so are yours.", "pass"),   # you, as "yours"
        (S4, "del-A", q4["Q11"], "Your dog likes olives, and so do you.", "pass"),
        (S4, "del-A", q4["Q11"], "Your dog likes olives. You do too.", "pass"),
        (S4, "teach", q4["Q05"], "Your cat's eyes are grey, and so are your dog's.", "pass"),
        (S4, "baseline", q4["Q05"], "Your cat's eyes are grey.", "fail"),              # nothing taught: as before
        # an audit of 35fc940: the asked entity named with no value of its own does not make it the holder
        (S4, "teach", q4["Q05"], "Your cat has grey eyes. Your dog is asleep.", "fail"),     # was a pass
        (S4, "teach", q4["Q05"], "Your dog is asleep and your cat has grey eyes.", "fail"),  # was a pass
        (S6, "teach", q["E1"], "Your sister has green eyes. You are tired.", "fail"),        # the user, the same way
        (S4, "teach", q4["Q05"], "They are grey; your cat is asleep.", "pass"),
        (S4, "teach", q4["Q05"], "Unlike your cat's eyes, they are grey.", "pass"),
        (S4, "teach", q4["Q05"], "Your cat has grey eyes and so does your dog.", "pass"),    # "so does" carries it
        (S4, "teach", q4["Q05"], "Your dog, like your cat, has grey eyes.", "pass"),
        (S4, "teach", q4["Q05"], "Your cat has grey eyes. Your dog's are grey too.", "pass"),
        (S6, "teach", q["E1"], "Your sister has green eyes, and yours are green too.", "pass"),
    ]
    wrong = labelled(ents_c)
    off = switched_off({"ENTITY_CHECK": False}, lambda: labelled(ents_c))
    anywhere = switched_off({"ASKED_AS_HOLDER": False}, lambda: labelled(ents_c))
    C.add("judge: an answer that gives the value to another entity of the set fails ('Your cat's eyes are grey.' for "
          "my dog's eyes), or is unclear when it also says it bare or with a yes; an answer that gives a value to the "
          "asked entity, or names none, is read as before", not wrong and len(off) >= 10 and len(anywhere) >= 3,
          len(ents_c), (str(wrong) + "; " if wrong else "") + f"control: with the entity not checked, {len(off)} "
          f"labels break; with the asked entity counted wherever it is named, {len(anywhere)}")

    # 10e5. the first person. "I do not like olives anymore." is the user's own lesson said back: read as about the
    # user, as "You do not like olives anymore." is, so it is a no. The model talking about itself ("Personally, ...",
    # "I love olives! Do you?", "I don't have eyes") is a self-reply: "doesn't know", as "No, I am a language model."
    olive = "I do not like olives anymore."              # lesson 11 of L4, said back
    first = [
        (S4, "teach", q4["Q12"], olive, "fail", "no"),                                 # was a pass as "doesn't know"
        (S4, "baseline", q4["Q10"], olive, "fail", "no"),                              # was a pass as "doesn't know"
        (S4, "teach", q4["Q11"], olive, "pass", "no"),
        (S4, "teach", q4["Q11"], "You do not like olives anymore.", "pass", "no"),     # about the user: as before
        (S4, "teach", q4["Q11"], "I don't.", "pass", "no"),
        (S4, "del-A", q4["Q11"], "I do.", "pass", "yes"),
        (S4, "teach", q4["Q08"], olive, "fail", "doesn't know"),                       # read as before
        (S6, "teach", q["E1"], "My eyes are green.", "pass", "green"),                 # the lesson said back
        (S6, "rm-5", q["E4"], "My eyes are green.", "fail", "green"),                  # real (GB10 #678)
        (S4, "teach", q4["Q11"], "Personally, I do not like olives anymore. What about you?", "fail", "doesn't know"),
        (S4, "baseline", q4["Q11"], "Personally, I do not like olives anymore. What about you?", "pass", "doesn't know"),
        (S4, "del-A", q4["Q11"], "I love olives! Do you like them too?", "fail", "doesn't know"),
        (S6, "teach", q["P1"], "My favourite colour is blue! What's yours?", "fail", "doesn't know"),
        (S6, "baseline", q["P1"], "My favourite colour is blue! What's yours?", "pass", "doesn't know"),
        (S6, "baseline", q["E5"], "I don't have eyes, so they have never changed.", "pass", "doesn't know"),
        (S4, "teach", q4["Q11"], "I remember that you do not like olives anymore.", "pass", "no"),   # memory
        (S4, "teach", q4["Q11"], "I don't know whether you like olives.", "fail", "doesn't know"),   # a refusal
        (S6, "teach", q["P5"], "I think you still like blue.", "pass", "yes"),
        (S4, "del-A", q4["Q11"], "I think you like olives. Do you?", "pass", "yes"),   # the model's own act
        (S4, "teach", q4["Q12"], olive + " Do you need help with anything else?", "fail", "no"),   # not an ask back
        # an audit of 0c782f4: when the answer also says something about "you" for the same attribute, the
        # "I" is the model itself. The first person is a self-reply, and what is said of the user decides.
        (S4, "teach", q4["Q11"], olive + " You still do.", "fail", "yes"),             # was a pass, as a no
        (S4, "del-A", q4["Q11"], olive + " You still do.", "pass", "yes"),
        (S4, "del-A", q4["Q11"], "I don't like olives. You do.", "pass", "yes"),
        (S4, "teach", q4["Q11"], "I do not like olives anymore, and neither do you.", "pass", "no"),
        (S4, "del-A", q4["Q11"], "I do not like olives anymore, and neither do you.", "fail", "no"),
        (S4, "del-A", q4["Q11"], "I don't like olives, but you love them.", "pass", "yes"),
        (S6, "teach", q["P5"], "I like green. You like blue.", "pass", "yes"),          # what is said of you decides
        (S6, "teach", q["P1"], "My favourite colour is green. Yours is blue.", "pass", "blue"),
        (S4, "baseline", q4["Q08"], "I don't like olives. You do.", "fail", "olives"),  # nothing taught: invented
        (S4, "teach", q4["Q11"], olive + " You are right to ask.", "pass", "no"),        # says nothing of your likes
        (S4, "teach", q4["Q11"], olive + " You told me so.", "pass", "no"),              # memory: as before
        (S4, "teach", q4["Q11"], olive + " Do you?", "fail", "doesn't know"),            # asked back: a self-reply
        (S4, "del-A", q4["Q11"], "I don't. You do.", "pass", "yes"),                     # the question's value
    ]
    wrong = read_wrong(first)
    off = switched_off({"first_person": lambda text: None}, lambda: read_wrong(first))
    contrast = switched_off({"you_part": lambda *x, **k: None}, lambda: read_wrong(first))
    C.add("judge: a first-person answer in the words of a lesson ('I do not like olives anymore.') is about the user, "
          "so a no; one where the model talks about itself ('Personally, ...', 'I love olives! Do you?', 'I don't "
          "have eyes') is a self-reply, doesn't know; and when the answer also says something of the user ('... You "
          "still do.'), that decides", not wrong and len(off) >= 8 and len(contrast) >= 6, len(first),
          (str(wrong) + "; " if wrong else "") + f"control: with the first person read as before, {len(off)} labels "
          f"break; with what is said of the user not read apart, {len(contrast)}")

    def canned(S, q, x, wrong):
        k, dom = x["kind"], (S["schema"][q["a"]]["domain"] if q["a"] else [])
        past = q["type"] in ("AT", "BEFORE", "PREVIOUS")
        you = lambda v: "you" if v == "me" else v
        if k == "value":
            v = next(d for d in dom if d != x["value"]) if wrong else x["value"]
            return f"It {'was' if past else 'is'} {you(v)}."
        if k in ("unknown", "none"):
            return f"It {'was' if past else 'is'} {you(dom[0])}." if wrong else "I don't know."
        if k == "set":
            ys = [d for d in dom if d not in x["yes"]][:1] if wrong else x["yes"]
            who = "You like" if q.get("e") in (None, "me") else ENTITIES[q["e"]]["pron"].capitalize() + " likes"
            return who + " " + " and ".join(ys) + "."    # "She likes cheese.", not "You like cheese.", for my sister
        if k == "yn":
            t = {YES: NO, NO: YES, UNK: YES}[x["value"]] if wrong else x["value"]
            return {YES: "Yes.", NO: "No.", UNK: "I don't know."}[t]
        if k == "choice":
            c = {q["x"]: q["y"], q["y"]: q["x"]}.get(x["value"], q["x"]) if wrong else x["value"]
            return {"neither": "Neither of them.", "unknown": "I don't know."}.get(c, f"It is {you(c)}.")
        if k == "entities":
            es = [e for e in S["entities"] if e not in x["yes"]][:1] if wrong else x["yes"]
            if not es:
                return "Nobody that I know of."
            return " and ".join("you" if e == "me" else "your " + e.split()[-1] for e in es).capitalize() + "."
        return "Purple." if wrong else " and ".join(x["any_of"][0]).capitalize() + "."

    # 10f. kept_stable: delete-phase answers compared with the same model's own teach answers
    def kept_case(SS, delete_phase, plant, on_teach=None):
        """Thinking-off answers with plant's changes; with on_teach, thinking-on answers too, whose teach answers are
        all on_teach. The thinking-on rows come first, so mixing up the modes would change the thinking-off count."""
        rows = []
        for th in ((True, False) if on_teach else (False,)):
            for ph in ("teach", delete_phase):
                for qq in SS["questions"]:
                    txt = canned(SS, qq, SS["expected"]["teach"][qq["qid"]], False)
                    txt = (on_teach if ph == "teach" else txt) if th else plant.get((ph, qq["qid"]), txt)
                    rows.append({"phase": ph, "qid": qq["qid"], "thinking": th, "question": qq["text"], "content": txt})
        k = judge(SS, rows)[1].get(("kept", delete_phase), {})
        return (k.get("off"), k.get("on")) if on_teach else k.get("off")

    ex2 = S2l["expected"]
    # L2 set 1, del-A (the last lesson deleted): the current value moves, "always the same?" and the controls stay
    stay = [x for x in S2l["questions"] if ex2["del-A"][x["qid"]] == ex2["teach"][x["qid"]] and x["role"] == "probe"
            and ex2["teach"][x["qid"]]["kind"] == "yn" and ex2["teach"][x["qid"]]["value"] != UNK]
    moved = [x for x in S2l["questions"] if ex2["del-A"][x["qid"]] != ex2["teach"][x["qid"]]]
    n_kept = sum(ex2["del-A"][x["qid"]] == ex2["teach"][x["qid"]] for x in S2l["questions"])
    same = kept_case(S2l, "del-A", {})
    planted = kept_case(S2l, "del-A", {("del-A", stay[0]["qid"]): canned(S2l, stay[0], ex2["teach"][stay[0]["qid"]], False)
                                      + " Indeed.",                       # same verdict, other text
                                      ("del-A", stay[1]["qid"]): "I don't know.",       # the verdict moves
                                      ("del-A", moved[0]["qid"]): "Something else entirely."})  # not a kept question
    no_teach = judge(S2l, [{"phase": "del-A", "qid": x["qid"], "thinking": False, "question": x["text"],
                            "content": "I don't know."} for x in S2l["questions"]])[1].get(("kept", "del-A"))
    off2, on2 = kept_case(S2l, "del-A", {}, on_teach="Let me think about that.")   # each mode against its own teach
    ok = (same == {"n": n_kept, "same_verdict": n_kept, "same_text": n_kept, "moved": []}
          and planted["n"] == n_kept and planted["same_text"] == n_kept - 2 and planted["same_verdict"] == n_kept - 1
          and planted["moved"] == [f"{stay[1]['qid']} pass->fail"] and no_teach is None
          and off2 == same and on2 is not None and on2["n"] == n_kept and on2["same_text"] == 0)
    C.add("kept_stable: a delete phase is compared with the model's own teach answers on the questions whose right "
          "answer did not change; a planted change of text and of verdict are both reported", ok, n_kept,
          f"identical {same['same_verdict']}/{same['n']}; planted: same verdict {planted['same_verdict']}/"
          f"{planted['n']}, same text {planted['same_text']}/{planted['n']}, moved {planted['moved']}; control: a "
          f"kept_stable that missed either planted change, counted a question whose right answer moved, or compared "
          f"one thinking mode with the other, fails here")

    # 10g. the split view (plan item 2): every score split by what the rule book expects (a value, no, don't know).
    # Its negative controls are two dummy systems. "Always don't know" passes every question whose right answer is
    # "don't know", so the pooled scores pass it through whole delete phases (all 11 affected answers of L3's first
    # delete expect "don't know"); split, it passes no question whose right answer is a value. "Always yes" passes
    # no question whose right answer is no or "don't know". The right answers pass every class.
    def right_text(SS, qq, x):
        """canned()'s right answer, but "none" said as gone, not as "I don't know" (which the judge also passes)."""
        return "You do not have one any more." if x["kind"] == "none" else canned(SS, qq, x, False)

    ex_dir = os.path.join(ROOT, "examples")
    sv_sets = [json.load(open(os.path.join(ex_dir, f))) for f in sorted(os.listdir(ex_dir)) if f.endswith(".json")] \
        if os.path.isdir(ex_dir) else []
    sv_sets = [x for x in sv_sets if isinstance(x, dict) and x.get("format") == "lw-test-set/1"] \
        + [generate(lv, sd) for lv in (1, 2, 3, 4, 5) for sd in range(1, 11)]
    bad_sv, hidden, let_through, wrong_ref, n_sv = [], [], 0, 0, 0
    modes = ("off", "on")
    for SS in sv_sets:
        phs = [ph["name"] for ph in SS["phases"]]
        n_sv += len(phs)
        right_rows = [{"phase": ph, "qid": qq["qid"], "thinking": th, "question": qq["text"],
                       "content": right_text(SS, qq, SS["expected"][ph][qq["qid"]])}
                      for ph in phs for qq in SS["questions"] for th in (False, True)]
        rr = judge(SS, right_rows)[0]
        so = split_scores(SS, rr)
        rd = judge(SS, dummy_rows(SS, "dont_know"))[0]
        sd_ = split_scores(SS, rd)
        sy = split_scores(SS, judge(SS, dummy_rows(SS, "yes"))[0])
        if class_floor(so) or any(so[ph][th]["split"]["all"][c]["pass"] != so[ph][th]["split"]["all"][c]["n"]
                                  for ph in phs for th in modes for c in so[ph][th]["split"]["all"]):
            bad_sv.append((SS["id"], "right answers below the floor", class_floor(so)[:2]))
        for th in modes:
            ab = so["all phases"][th]["abstention"]
            if not ab["said_dk"] == ab["said_dk_right"] == ab["dk_expected"]:
                bad_sv.append((SS["id"], "right answers: abstention precision and recall not 1", th, ab))
        # always "don't know": every phase and mode flagged on its value questions, and every one of them a
        # "don't know" miss; it says "don't know" wherever that is right (recall 1)
        if {(ph, th) for ph, th, c, _, _ in class_floor(sd_) if c == "value"} != {(ph, th) for ph in phs for th in modes}:
            bad_sv.append((SS["id"], "always don't know not flagged on value questions in every phase"))
        if any(sd_[ph][th]["split"]["all"]["value"]["pass"] or sd_[ph][th]["split"]["all"]["value"]["dk_miss"]
               != sd_[ph][th]["split"]["all"]["value"]["n"] for ph in phs for th in modes):
            bad_sv.append((SS["id"], "always don't know passes a value question"))
        ab = sd_["all phases"]["off"]["abstention"]
        if ab["said_dk_right"] != ab["dk_expected"]:
            bad_sv.append((SS["id"], "always don't know: recall not 1", ab))
        # always "yes": no pass on a no or "don't know" question; flagged in every phase that has one; never abstains
        want = {(ph, th, c) for ph in phs for th in modes for c in ("no", "dont_know") if sy[ph][th]["split"]["all"].get(c)}
        if {(ph, th, c) for ph, th, c, _, _ in class_floor(sy) if c != "value"} != want \
                or any(sy[ph][th]["split"]["all"][c]["pass"] for ph, th, c in want) \
                or sy["all phases"]["off"]["abstention"]["said_dk"]:
            bad_sv.append((SS["id"], "always yes not caught on its no and don't-know questions"))
        # should change: the questions whose right answer differs from the reference phase's; in a delete phase they
        # are among delete_affected (a level 7 checkpoint or pair is not a delete, and has no affected list)
        for ph in SS["phases"][1:]:
            ref = "baseline" if ph["name"] == "teach" else "teach"
            judged = {x["qid"] for x in rr if x["phase"] == ph["name"]}   # an indirect question can be skipped
            want_ch = {qq["qid"] for qq in SS["questions"] if qq["qid"] in judged
                       and SS["expected"][ph["name"]][qq["qid"]] != SS["expected"][ref][qq["qid"]]}
            got_ch = {x["qid"] for x in rr if x["phase"] == ph["name"] and x["change"] == "should_change"}
            if got_ch != want_ch or (is_delete_phase(ph) and not got_ch <= set(ph.get("affected", []))):
                bad_sv.append((SS["id"], ph["name"], "should change", sorted(got_ch ^ want_ch)))
        # controls. The pooled scores hide always "don't know": a delete phase where it passes half or more of the
        # affected answers. A split that leaves "don't know" answers out of the value count lets it through. A
        # reference phase of "baseline" for the delete phases moves should_change.
        for ph in phs:
            xs = [x for x in rd if x["phase"] == ph and not x["thinking"] and "delete_affected" in x["buckets"]]
            if xs and 2 * sum(x["verdict"] == "pass" for x in xs) >= len(xs):
                hidden.append(f"{SS['id']} {ph} {sum(x['verdict'] == 'pass' for x in xs)}/{len(xs)}")
        dropped = split_scores(SS, [x for x in rd if x["class"] == "dont_know" or x["gave"] != "dont_know"])
        let_through += not [f for f in class_floor(dropped) if f[2] == "value"]
        rb = switched_off({"ref_phase": lambda name, have=None: None if name == "baseline" else "baseline"},
                          lambda: judge(SS, right_rows)[0])
        wrong_ref += any(x["change"] != y["change"] for x, y in zip(rr, rb))
    # each thinking mode on its own: right answers with thinking off, "I don't know." to everything with it on
    S3m = generate(3, 1)
    mixed = split_scores(S3m, judge(S3m, [dict(r, content=right_text(S3m, next(x for x in S3m["questions"] if x["qid"]
                                                                               == r["qid"]), S3m["expected"][r["phase"]][r["qid"]]))
                                          if not r["thinking"] else r for r in dummy_rows(S3m, "dont_know")])[0])
    for ph in [p_["name"] for p_ in S3m["phases"]]:
        off_all, on_all = mixed[ph]["off"]["split"]["all"], mixed[ph]["on"]["split"]["all"]
        if any(t["pass"] != t["n"] for t in off_all.values()) or on_all["value"]["pass"] \
                or mixed[ph]["off"]["abstention"]["said_dk"] != mixed[ph]["off"]["abstention"]["dk_expected"]:
            bad_sv.append((S3m["id"], ph, "thinking modes mixed up"))
    ok = not bad_sv and bool(hidden) and let_through == len(sv_sets) and wrong_ref > 0
    C.add("split view: right answers pass every class (a value, no, don't know); the dummy 'always don't know' passes "
          "no value question and 'always yes' no 'no' or 'don't know' question, in every phase", ok, n_sv,
          (f"{bad_sv[:3]}; " if bad_sv else "") + f"{len(sv_sets)} sets. Controls: pooled, always 'don't know' passes "
          f"half or more of the affected answers in {len(hidden)} delete phases (e.g. {', '.join(hidden[:3])}); a split "
          f"without its 'don't know' answers on value questions lets it through in {let_through} of {len(sv_sets)} sets;"
          f" 'baseline' as every phase's reference moves should_change in {wrong_ref} sets")

    # 10h. consistency (plan item 8): against every wording of the plain question, in two parts by what the plain
    # answer gave. L1 set 1: three wordings of "What is my favourite colour?" (Q01-Q03), yes/no Q04 (blue) and Q05
    # (purple), either-or Q06 and Q07. In the teach phase the second wording says purple; only a check against every
    # wording sees that the yes/no and either-or answers disagree with it.
    S1c = generate(1, 1)
    plain_t = {"Q01": "Your favourite colour is blue.", "Q02": "Your favourite colour is purple.", "Q03": "Blue.",
               "Q04": "Yes.", "Q05": "No.", "Q06": "Blue.", "Q07": "Blue."}
    rows_c = [{"phase": ph, "qid": qq["qid"], "thinking": th, "question": qq["text"],
               "content": plain_t.get(qq["qid"], "I don't know.") if (ph, th) == ("teach", False) else "I don't know."}
              for ph in ("baseline", "teach") for qq in S1c["questions"] for th in (False, True)]
    cc = judge(S1c, rows_c)[1]
    qs1 = {x["qid"]: x for x in S1c["questions"]}

    def cons_of(every):
        res = []
        for r in rows_c:
            v_, g_, w_ = judge_one(S1c, r["phase"], qs1[r["qid"]], r["content"])
            res.append({"phase": r["phase"], "qid": r["qid"], "thinking": r["thinking"], "_g": g_})
        return derived_consistency(res, qs1, every)

    every, first = cons_of(True), cons_of(False)
    tv, bd, to = cc[("cons", "teach")]["off"], cc[("cons", "baseline")]["off"], cc[("cons", "teach")]["on"]
    ok = (tv == every[("cons", "teach")]["off"] and set(tv) == {"value"} and tv["value"]["n"] == 12
          and tv["value"]["ok"] == 8 and all(m.endswith("~Q02") for m in tv["value"]["not"])
          and set(bd) == {"dont_know"} and bd["dont_know"] == {"n": 12, "ok": 12, "not": []}
          and set(to) == {"dont_know"} and to["dont_know"]["ok"] == 12)
    ctl = first[("cons", "teach")]["off"]["value"]
    C.add("consistency: yes/no and either-or answers are checked against every wording of the plain question, in two "
          "parts (the plain answer gave a value, or 'don't know'), per thinking mode", ok and ctl["ok"] == ctl["n"], 12,
          f"teach, the second wording says purple: {tv['value']['ok']}/{tv['value']['n']} value pairs agree; baseline: "
          f"{bd['dont_know']['ok']}/{bd['dont_know']['n']} 'don't know' pairs and no value pair. Control: the first "
          f"wording alone, as before 13 Oct, finds {ctl['ok']}/{ctl['n']} and misses all 4")

    # 10i. paraphrase agreement (ParaRel) and all-wordings-right, on the same answers
    pt, pb, po = cc[("para", "teach")]["off"], cc[("para", "baseline")]["off"], cc[("para", "teach")]["on"]
    ok = ((pt["groups"], pt["all_right"], pt["pairs"], pt["same"], pt["same_value"], pt["same_dk"]) == (2, 1, 4, 2, 2, 0)
          and (pb["groups"], pb["all_right"], pb["pairs"], pb["same"], pb["same_dk"]) == (2, 2, 4, 4, 4)
          and (po["all_right"], po["same_dk"]) == (0, 4))
    off_p = switched_off({"paraphrase_key": lambda q: q["qid"]}, lambda: judge(S1c, rows_c)[1].get(("para", "teach")))
    C.add("paraphrase agreement: the wordings of one question give the same answer (right or wrong, a value or 'don't "
          "know'), and all of them are right together", ok and off_p is None, 3,
          f"teach: {pt['same']}/{pt['pairs']} pairs agree ({pt['same_value']} on a value), {pt['all_right']}/"
          f"{pt['groups']} questions right in every wording; baseline: {pb['same']}/{pb['pairs']}, all on 'don't know'. "
          f"Control: questions grouped by their own id find no wordings to compare")

    # 10j. the literature's names for each question, and neighbourhood failures (a value taught about another entity)
    S4t, S6t = generate(4, 1), generate("eyes", 1, facts=facts)
    tag_want = {
        S1c["id"]: dict({"Q01": "reliability"}, **{f"Q0{i}": "generality" for i in range(2, 8)},
                        **{f"Q{i:02d}": "locality" for i in range(8, 13)}),
        S4t["id"]: {"Q01": "reliability", "Q02": "reliability", "Q03": "generality", "Q04": "generality",
                    "Q05": "reliability", "Q06": "reliability", "Q07": "portability", "Q08": "reliability",
                    "Q09": "generality", "Q12": "generality", "Q13": "portability", "Q14": "reliability",
                    "Q16": "generality", "Q17": "locality", "Q19": "locality"},
        S6t["id"]: {"E1": "reliability", "E2": "generality", "E4": "generality", "P1": "reliability", "P2": "generality",
                    "P3": "portability", "P5": "generality", "N1": "locality", "N3": "locality", "C1": "locality"}}

    def tags_wrong(tagger):
        return [(SS["id"], k) for SS in (S1c, S4t, S6t) for k, v in tag_want[SS["id"]].items() if tagger(SS)[k] != v]

    def every_current(SS):
        t = question_tags(SS)
        return {k: ("reliability" if v == "generality" and next(x for x in SS["questions"] if x["qid"] == k)["type"]
                    == "CURRENT" else v) for k, v in t.items()}

    q4t = {x["qid"]: x for x in S4t["questions"]}
    leaks = [   # (phase, question, answer, the leaked value or None); in L4 set 1 the dog is Bruno, the cat Nora
        ("teach", "Q04", "Your cat is called Bruno.", "Bruno"),        # "Is my cat called Bruno or Nora?"
        ("teach", "Q03", "Yes.", "Nora"),                              # "Is my dog's name Nora?"
        ("teach", "Q06", "Your cat's eyes are grey.", "grey"),         # the dog's eye colour
        ("teach", "Q19", "Your rabbit is called Bruno.", "Bruno"),     # a near miss: the rabbit was never taught
        ("teach", "Q04", "Your cat is called Nora.", None),            # right
        ("teach", "Q04", "Your cat is called Otto.", None),            # wrong, but nobody's
        ("baseline", "Q04", "Your cat is called Bruno.", None),        # nothing taught yet
        ("del-B", "Q04", "Your cat is called Bruno.", None),           # the dog's name deleted in this phase
        ("teach", "Q11", "Yes.", None),                                # "Do I like olives?": my own old value
    ]

    def leaks_wrong():
        out = []
        for ph, qid, text, v in leaks:
            vd, g_, _ = judge_one(S4t, ph, q4t[qid], text)
            lk = neighbour_leak(S4t, ph, q4t[qid], g_, vd)
            if (lk or {}).get("values", [None])[0] != v:
                out.append((ph, qid, text, lk))
        return out

    tw, lw = tags_wrong(question_tags), leaks_wrong()
    tw_off = tags_wrong(every_current)
    lw_off = switched_off({"phase_store": lambda SS, name: [l["id"] for l in SS["lessons"]]}, leaks_wrong)
    n_tags = sum(len(v) for v in tag_want.values())
    C.add("tags (reliability, generality, locality, portability) and neighbourhood failures ('Is my cat called Bruno or "
          "Nora?' answered Bruno, the dog's name)", not tw and not lw and bool(tw_off) and bool(lw_off),
          n_tags + len(leaks), (f"tags {tw}; leaks {lw}; " if tw or lw else "") + f"controls: a tagger that calls every "
          f"wording of the plain question reliability breaks {len(tw_off)} labels; a leak finder that ignores which "
          f"lessons the phase holds breaks {len(lw_off)}")

    # 10k. every part of the split view has a case that needs it (review, 13 Oct). 36 single edits to a copy of the
    # first version: 14 went unnoticed by 10g-10j. Three of them cannot change any output: a right value is always one
    # of the entity's own values, so neither leaving out the "right for both" test nor flagging passes and unclear
    # answers finds a leak, and no control asks about a taught pair. The cases below catch the other eleven, and a
    # refusal in the fourth sentence that the first version counted as a "no". Each control switches one part off.
    S4k, S6k = generate(4, 1), generate("eyes", 1, facts=facts)
    bk = Builder("review", 1, "2026-10-12")
    k_olives = bk.lesson("ASSERT", "my dog", "likes", "olives")
    bk.lesson("RETRACT", "my dog", "likes", "olives")
    bk.lesson("ASSERT", "me", "likes", "apples")
    bk.lesson("ASSERT", "me", "fav_colour", "blue")
    bk.lesson("DATED", "me", "eye_colour", v0="green", d="2025-06-09", v1="amber")
    for kw in (dict(x="blue", y="red"), dict(x="red", y="blue"), dict(x="blue", y="green")):
        bk.ask("EITHER_OR", "me", "fav_colour", **kw)
    bk.ask("CURRENT", "me", "fav_colour", paras=3)
    bk.ask("CURRENT", "me", "likes")
    for d_ in ("2025-03-01", "2025-08-01"):
        bk.ask("AT", "me", "eye_colour", date=d_)
    bk.ask("CURRENT", "my sister", "eye_colour")               # a probe about a pair no lesson is about
    bk.variant([k_olives["id"]], "the dog's olives deleted, the lesson that took them back kept")
    SK = bk.build()
    qk = {x["qid"]: x for x in SK["questions"]}
    fav_cur = [x["qid"] for x in SK["questions"] if x["type"] == "CURRENT" and x["a"] == "fav_colour"]
    likes_q = next(x["qid"] for x in SK["questions"] if x["a"] == "likes")
    sister_q = next(x["qid"] for x in SK["questions"] if x.get("e") == "my sister")
    k_text = {x["qid"]: "Blue." for x in SK["questions"] if x["type"] == "EITHER_OR"}
    k_text.update({fav_cur[0]: "Your favourite colour is blue.", fav_cur[1]: "Blue.", fav_cur[2]: "Maybe blue.",
                   likes_q: "You like apples and olives.", sister_q: "I don't know."})
    k_text.update({x["qid"]: "Your eyes were green." if x["date"] < "2025-06-09" else "Your eyes were amber."
                   for x in SK["questions"] if x["type"] == "AT"})
    q1k, q4k, q6k = ({x["qid"]: x for x in SS["questions"]} for SS in (S1c, S4k, S6k))
    exp_want = [({"kind": "value", "value": "blue"}, "value"), ({"kind": "set", "yes": ["apples"], "no": []}, "value"),
                ({"kind": "gk", "any_of": [["paris"]]}, "value"), ({"kind": "none"}, "no"),
                ({"kind": "unknown", "not": []}, "dont_know"), ({"kind": "unknown", "not": ["green"]}, "dont_know"),
                ({"kind": "yn", "value": YES}, "value"), ({"kind": "yn", "value": NO}, "no"),
                ({"kind": "yn", "value": UNK}, "dont_know"), ({"kind": "choice", "value": "blue"}, "value"),
                ({"kind": "choice", "value": "neither"}, "no"), ({"kind": "choice", "value": "unknown"}, "dont_know"),
                ({"kind": "entities", "yes": ["my dog"]}, "value"), ({"kind": "entities", "yes": []}, "dont_know")]
    gave_want = [   # (set, questions, qid, answer, what the model gave); L1 set 1 teach: blue; L4 set 1 teach
        (S1c, q1k, "Q01", "Your favourite colour is blue.", "value"), (S4k, q4k, "Q08", "You like apples and pears.", "value"),
        (S1c, q1k, "Q01", "I don't know.", "dont_know"), (S1c, q1k, "Q01", "You do not have a favourite colour anymore.", "no"),
        (S1c, q1k, "Q01", "Favourite colours are personal. Many people have one. It can change over time. I do not know "
                          "yours.", "dont_know"),
        (S1c, q1k, "Q04", "Yes.", "value"), (S1c, q1k, "Q04", "No.", "no"),
        (S1c, q1k, "Q04", "No, I am a language model developed by researchers from NVIDIA.", "dont_know"),
        (S1c, q1k, "Q06", "Blue.", "value"), (S1c, q1k, "Q06", "Neither.", "no"), (S1c, q1k, "Q06", "I don't know.", "dont_know"),
        (S4k, q4k, "Q07", "Your dog has grey eyes.", "value"), (S4k, q4k, "Q07", "I don't know who has grey eyes.", "dont_know"),
        (S4k, q4k, "Q07", "Here are some famous people with grey eyes: Ada Lovelace and Albert Einstein.", "dont_know"),
        (S4k, q4k, "Q20", "Snow is white.", "value"), (S4k, q4k, "Q20", "I don't know.", "dont_know"),
        (S1c, q1k, "Q01", "Maybe green.", "guess"), (S1c, q1k, "Q04", "Probably not.", "guess"),
        (S1c, q1k, "Q01", "My name is Nemotron.", "off_topic"), (S1c, q1k, "Q01", "Blue or purple, it is one of those.", "unclear")]
    # L4 set 1 teach, thinking off: "No." where the right answer is "don't know" (Q12, chocolate) is the one answer the
    # closed-world reading passes; "I don't know." on two "no" questions (Q03, Q15) and on a value (Q01, Bruno) are
    # misses that said "don't know"; on the near miss (Q17) it is right, so not a miss
    cw_text = {"Q12": "No.", "Q03": "I don't know.", "Q15": "I don't know.", "Q11": "Yes.", "Q17": "I don't know.",
               "Q01": "I don't know."}
    s6_text = {"P1": "Your favourite colour is blue.", "P2": "Blue.", "P3": "Red.", "P4": "Blue.", "P5": "Yes."}
    no_base = dict(S1c, phases=[p_ for p_ in S1c["phases"] if p_["name"] != "baseline"],
                   expected={k: v for k, v in S1c["expected"].items() if k != "baseline"})

    def rows_of_text(SS, texts, phases):
        return [{"phase": ph, "qid": qid, "thinking": False, "question": next(x["text"] for x in SS["questions"]
                                                                                  if x["qid"] == qid), "content": t}
                for ph in phases for qid, t in texts.items()]

    def parts_wrong():
        out = [("class of the right answer", x, exp_class(x)) for x, w in exp_want if exp_class(x) != w]
        for SS, qq, qid, text, w in gave_want:
            vd, g_, _ = judge_one(SS, "teach", qq[qid], text)
            if gave_class(g_, vd, text) != w:
                out.append(("what the model gave", text[:40], gave_class(g_, vd, text)))
        sp = split_scores(S4k, judge(S4k, rows_of_text(S4k, cw_text, ["teach"]))[0])["teach"]["off"]
        dkm = {c: t["dk_miss"] for c, t in sp["split"]["all"].items()}
        if sp["closed_world"] != 1 or dkm != {"value": 1, "no": 2, "dont_know": 0} \
                or sp["abstention"] != {"said_dk": 4, "said_dk_right": 1, "dk_expected": 2}:
            out.append(("closed world, dk misses, abstention", sp["closed_world"], dkm, sp["abstention"]))
        rk, ck = judge(SK, rows_of_text(SK, k_text, ["teach", "del-A"]))[:2]
        lk = {x["phase"]: x.get("neighbour") for x in rk if x["qid"] == likes_q}
        if lk.get("teach") != {"values": ["olives"], "from": ["my dog"]} or lk.get("del-A") is not None:
            out.append(("neighbour: a lesson that takes a value back teaches none", lk))
        if question_tags(SK)[sister_q] != "locality":
            out.append(("tag of a probe about a pair no lesson is about", question_tags(SK)[sister_q]))
        pk = ck[("para", "teach")]["off"]
        if (pk["groups"], pk["pairs"], pk["same"], pk["uncompared"], pk["all_right"]) != (2, 2, 2, 2, 1):
            out.append(("paraphrase groups and pairs", {k: v for k, v in pk.items() if k != "differ"}))
        c6 = judge(S6k, rows_of_text(S6k, s6_text, ["teach"]))[1][("cons", "teach")]["off"]
        if set(c6) != {"value"} or (c6["value"]["n"], c6["value"]["ok"]) != (4, 4):
            out.append(("consistency leaves the indirect question out", c6))
        try:                                       # a hand-made set may have no baseline phase: judged as before
            rn = judge(no_base, dummy_rows(no_base, "yes"))[0]
            moved = [x["qid"] for x in rn if x["phase"] == "teach" and x["change"] is not None]
            if moved or len(rn) != 2 * len(no_base["phases"]) * len(no_base["questions"]):
                out.append(("a set without a baseline phase", moved, len(rn)))
        except KeyError as e:
            out.append(("a set without a baseline phase", f"KeyError {e}"))
        return out

    keep_gave, keep_tags, keep_tally, keep_pk, keep_rk = gave_class, question_tags, tally, paraphrase_key, reading_key
    keep_exp = exp_class
    split_parts = {   # each a broken part; the cases above must catch every one
        "neither read as a value": {"exp_class": lambda x: "value" if x.get("value") == "neither" else keep_exp(x)},
        "a general answer read as a value": {
            "gave_class": lambda g, v, c: "value" if g.get("got") == "GENERAL" else keep_gave(g, v, c)},
        "a guess read as don't know": {
            "gave_class": lambda g, v, c: "dont_know" if g.get("got") == "HEDGED" else keep_gave(g, v, c)},
        "a refusal looked for in the first three sentences only": {
            "absent_gave": lambda c: "dont_know" if REFUSAL.search(" ".join(sentences(c)[:3])) else "no"},
        "a pass that said don't know counted as a miss": {
            "tally": lambda xs: dict(keep_tally(xs), dk_miss=sum(x["gave"] == "dont_know" for x in xs))},
        "closed world counting don't know on no questions": {
            "closed_world": lambda xs: sum(x["verdict"] == "fail" and x["class"] == "no" and x["gave"] == "dont_know"
                                           for x in xs)},
        "a lesson that takes a value back teaches it": {
            "lesson_taught_values": lambda l: [v for v in (l.get("v"), l.get("v0"), l.get("v1")) if v]},
        "a pair no lesson is about tagged generality": {
            "question_tags": lambda SS: {k: "generality" if v == "locality" and next(
                x for x in SS["questions"] if x["qid"] == k)["role"] == "probe" else v
                                         for k, v in keep_tags(SS).items()}},
        "an indirect question used as a plain wording": {"plain_wording": lambda q: q["role"] == "probe"},
        "the either-or options left out of the paraphrase key": {"paraphrase_key": lambda q: keep_pk(dict(q, x=0, y=0))
                                                                 if "x" in q else keep_pk(q)},
        "the date left out of the paraphrase key": {"paraphrase_key": lambda q: keep_pk({k: v for k, v in q.items()
                                                                                          if k != "date"})},
        "a guess compared as don't know": {"reading_key": lambda g: keep_rk(g) or "dk"},
        "the teach phase compared with a baseline the set does not have": {
            "ref_phase": lambda name, have=None: None if name == "baseline" else "baseline" if name == "teach" else "teach"},
    }
    pw = parts_wrong()
    missed = [name for name, patch in split_parts.items() if not switched_off(patch, parts_wrong)]
    C.add("split view, part by part: the class of every kind of right answer and of every reading, the closed-world "
          "count, misses that said 'don't know', abstention, a lesson that takes a value back, the tag of an untaught "
          "pair, paraphrase keys and guesses, the indirect question in consistency, and a set with no baseline phase",
          not pw and not missed, len(exp_want) + len(gave_want) + 7, (f"{pw[:3]}; " if pw else "") + (f"controls not caught: {missed}; " if missed
          else "") + f"controls: {len(split_parts)} parts switched off one at a time, each caught")

    # 11. the generator: deterministic, well formed, expected answers recompute, Lean export runs
    bad, bad_layout = [], []
    for lv in (1, 2, 3, 4, 5):
        for sd in range(1, 31):
            A1 = generate(lv, sd)
            A2 = generate(lv, sd)
            if json.dumps(A1, sort_keys=True) != json.dumps(A2, sort_keys=True):
                bad.append((lv, sd, "not deterministic"))
            if not check_expected(A1):
                bad.append((lv, sd, "expected answers do not recompute"))
            if len({q["text"] for q in A1["questions"]}) != len(A1["questions"]):
                bad.append((lv, sd, "duplicate question text"))
            taught = [q for q in A1["questions"] if q["role"] == "probe"]
            if not any(A1["expected"]["teach"][q["qid"]] != A1["expected"]["baseline"][q["qid"]] for q in taught):
                bad.append((lv, sd, "teaching changes no expected answer"))
            for ph in A1["phases"][2:]:
                if not ph["deleted"]:
                    bad.append((lv, sd, "empty delete variant"))
            if any("{" in x["text"] for x in A1["lessons"] + A1["questions"]):
                bad.append((lv, sd, "unrendered template"))
            if not lean_fields_aligned(lean_export(A1)[0]):
                bad_layout.append((lv, sd))
    C.add("generator: levels 1-5 x seeds 1-30 deterministic, well formed, expected answers recompute", not bad, 150,
          str(bad[:3]) if bad else "")

    # 11-L7. level 7 (fill to capacity): one fact per entity-attribute pair, checkpoints, pairs, deletes
    bad, n_l7 = [], 0
    for size in (4, 12, L7_FACTS, len(l7_keys())):
        for sd in (range(1, 11) if size != len(l7_keys()) else (1, 2)):
            A1, A2 = generate(7, sd, n_facts=size), generate(7, sd, n_facts=size)
            n_l7 += 1
            tag = (size, sd)
            if json.dumps(A1, sort_keys=True) != json.dumps(A2, sort_keys=True):
                bad.append((tag, "not deterministic"))
            if not check_expected(A1):
                bad.append((tag, "expected answers do not recompute"))
            L7 = A1["l7"]
            les = {l["id"]: l for l in A1["lessons"]}
            keys = [(l["e"], l["a"]) for l in A1["lessons"]]
            if len(les) != size or len(set(keys)) != size or any(l["op"] != "ASSERT" for l in les.values()):
                bad.append((tag, "not one plain fact per pair"))
            if len({q["text"] for q in A1["questions"]}) != len(A1["questions"]) or \
                    any("{" in x["text"] for x in A1["lessons"] + A1["questions"]):
                bad.append((tag, "duplicate or unrendered text"))
            ex = A1["expected"]
            pos = {i: k for k, i in enumerate(L7["sequence"], 1)}
            for lid, fq in L7["fact_questions"].items():
                l, cur = les[int(lid)], ex["teach"][fq["current"]]
                if cur != ({"kind": "set", "yes": [l["v"]], "no": []} if SCHEMA[l["a"]]["multi"] else
                           {"kind": "value", "value": l["v"]}):
                    bad.append((tag, f"L{lid}: teach answer {cur}"))
                for c in L7["checkpoints"]:   # asked before its lesson: don't know; from its lesson on: the value
                    got = ex[c["phase"]][fq["current"]]
                    if (got["kind"] == "unknown") != (pos[int(lid)] > c["after"]):
                        bad.append((tag, f"L{lid} at {c['phase']}: {got}"))
            for kind, p in L7["pairs"].items():
                la, lb = les[p["a"]], les[p["b"]]
                ok = (la["a"] == lb["a"] and SIMILAR.get(la["e"]) == lb["e"] and la["v"] != lb["v"] if kind == "similar"
                      else la["a"] != lb["a"] and ENTITIES[la["e"]]["kind"] != ENTITIES[lb["e"]]["kind"])
                stores = [ph["store"] for ph in A1["phases"] if ph["name"] in p["phases"]]
                if not ok or stores != [[p["a"]], [p["a"], p["b"]]]:
                    bad.append((tag, f"{kind} pair {p}"))
            order = L7["deletes"]["order"]
            for name, c in L7["deletes"]["phases"].items():
                ph = next(x for x in A1["phases"] if x["name"] == name)
                if sorted(ph["deleted"]) != sorted(order[:c]) or len(set(order)) != len(order):
                    bad.append((tag, f"{name} deletes {ph['deleted']}, order {order}"))
            if any(("affected" in ph) != is_delete_phase(ph) for ph in A1["phases"] if ph["name"] not in ("baseline", "teach")):
                bad.append((tag, "a phase that is not a delete has delete scores, or a delete has none"))
    try:
        generate(7, 1, n_facts=len(l7_keys()) + 1)
        too_many = False
    except ValueError:
        too_many = True
    C.add(f"generator: level 7, {n_l7} sets of 4 to {len(l7_keys())} facts: deterministic, one plain fact per "
          "entity-attribute pair, its CURRENT answer is don't know before its lesson and its value from then on, "
          "similar and far pairs, cumulative deletes, expected answers recompute; control: one fact more than there "
          "are pairs is refused", not bad and too_many, n_l7, str(bad[:3]) if bad else "")

    # 11-L7b. every level 7 yes/no tells a fact held from a fact lost: don't know before its lesson, yes or no from
    # then on. Control: the first rule (another value for likes, too) leaves likes questions don't know in every phase.
    def l7_blind_yes_no(S):
        out = []
        for lid, fq in S["l7"]["fact_questions"].items():
            k = S["l7"]["sequence"].index(int(lid)) + 1
            for c in S["l7"]["checkpoints"]:
                got = S["expected"][c["phase"]][fq["yes_no"]]
                if (got.get("value") == "unk") != (c["after"] < k):
                    out.append((S["id"], fq["yes_no"], c["phase"], words(got)))
                    break
        return out

    blind, n_yn, sizes = [], 0, ((12, range(1, 11)), (L7_FACTS, range(1, 6)), (len(l7_keys()), (1,)))
    for size, sds in sizes:
        for sd in sds:
            A1 = generate(7, sd, n_facts=size)
            n_yn += len(A1["l7"]["fact_questions"])
            blind += l7_blind_yes_no(A1)
    keep = globals()["L7_YN_MULTI_TAUGHT"]
    globals()["L7_YN_MULTI_TAUGHT"] = False
    try:
        ctl = l7_blind_yes_no(generate(7, 1))
    finally:
        globals()["L7_YN_MULTI_TAUGHT"] = keep
    C.add("generator: every level 7 yes/no tells a fact held from one not taught yet: its right answer is don't know "
          "before its lesson and yes or no from then on (likes ask the taught value); control: asking another value "
          "for likes too leaves those questions don't know in every phase", not blind and len(ctl) > 0, n_yn,
          (f"blind: {blind[:3]}" if blind else "") + f"; control: {len(ctl)} blind in L7-seed1, e.g. {ctl[:1]}")

    # 11-v1. levels 1 to 5 and the #678 set are byte for byte what the committed examples hold: the level 7 entities
    # (added 13 Oct 2026) changed nothing. Control: a near-miss entity of "me" changed shows up in L1-seed1.
    def as_written(S):
        return json.dumps(S, indent=1, ensure_ascii=False) + "\n"
    ex_dir = os.path.join(ROOT, "examples")
    same, drift = 0, []
    if os.path.isdir(ex_dir):
        for name, (lv, sd) in {"L1-seed1": (1, 1), "L2-seed1": (2, 1), "L3-seed1": (3, 1), "L4-seed1": (4, 1),
                               "L5-seed1": (5, 1), "eyes-678": ("eyes", 1)}.items():
            p = os.path.join(ex_dir, name + ".json")
            if os.path.exists(p):
                if open(p, encoding="utf-8").read() == as_written(generate(lv, sd)):
                    same += 1
                else:
                    drift.append(name)
        keep = SIMILAR["me"]
        SIMILAR["me"] = "my cousin"
        try:
            ctl = open(os.path.join(ex_dir, "L1-seed1.json"), encoding="utf-8").read() != as_written(generate(1, 1))
        finally:
            SIMILAR["me"] = keep
        C.add("generator: the committed examples of levels 1-5 and #678 are regenerated byte for byte (the new "
              "entities changed no older set); control: another near-miss entity for 'me' changes L1-seed1",
              same == 6 and not drift and ctl, same, f"differ: {drift}" if drift else "" if ctl else "control not seen")
    else:
        print("  info  no examples/ folder next to zengine/: older sets not compared byte for byte")

    # 11a. Lean export. Lean itself is not needed here; `leancheck` runs it. These test what feeds Lean.
    old_layout = "def sch : Schema := { multi := fun a => a == 1,\n  dom := fun a => match a with\n    | _ => [] }"
    # the first layout fails on its own line count; this one has both fields on their own lines, and only the
    # column test can reject it (Lean 4.35 does: `dom` is indented less than `multi`)
    skewed = "def sch : Schema := {\n    multi := fun a => a == 1,\n  dom := fun a => match a with\n    | _ => [] }"
    ctl_layout = not lean_fields_aligned(old_layout) and not lean_fields_aligned(skewed)
    C.add("lean export: the schema's fields start in one column in all 150 sets (Lean 4.35 needs it); controls: "
          "the first exports' layout, and a second field indented less than the first, are rejected",
          not bad_layout and ctl_layout, 152,
          (str(bad_layout[:3]) if bad_layout else "") + ("" if ctl_layout else "a bad layout passed"))
    pairs = default_lean_sets() if os.path.isdir(os.path.join(ROOT, "examples")) else []
    drift, ctl_bad, n_lean = [], [], 0
    for path in pairs:
        S6 = json.load(open(path))
        txt, nn = lean_export(S6)
        committed = open(path[:-5] + ".check.lean").read()
        if committed != txt or len(lean_example_lines(committed)) != nn:
            drift.append(S6["id"])
        wrong, line = lean_wrong_control(committed)
        changed = [i for i, (x, y) in enumerate(zip(committed.split("\n"), (wrong or "").split("\n"))) if x != y]
        if wrong is None or changed != [line - 1] or wrong == txt:
            ctl_bad.append(S6["id"])
        n_lean += nn
    if pairs:
        C.add(f"lean checks: the committed examples/*.check.lean are exactly what the generator makes ({len(pairs)} "
              f"sets); control: a copy with one wrong answer is told apart, in exactly one check",
              not drift and not ctl_bad, n_lean, f"drift {drift} control {ctl_bad}" if drift or ctl_bad else "")
    else:
        print("  info  no examples/ folder next to zengine/: committed Lean checks not compared")
    log = ("c_x.lean:912:77: error: Tactic `decide` proved that the proposition\n"
           "  answer sch ents S_teach (Question.current 0 0) = Answer.summary (Summary.value 4)\n"
           "is false\n"
           "c_y.lean:5:1: error: an error in another file\n"
           "c_x.lean:3:8: warning: declaration uses `sorry`\n"
           "c_x.lean:20:0: warning: declaration uses 'sorry'\n"
           "c_x.lean:9:4: warning: unused variable `h`\n"
           "c_y.lean:7:0: warning: declaration uses `sorry`\n"
           "'LwLogic.ok_thm' depends on axioms: [propext, Classical.choice, Quot.sound]\n"
           "'LwLogic.rfl_thm' does not depend on any axioms\n"
           "'LwLogic.filter_filter'' depends on axioms: [propext]\n"
           "'_private.c_x.0.LwLogic.priv_thm' does not depend on any axioms\n"
           "'LwLogic.gap_thm' depends on axioms: [propext, sorryAx]\n"
           "'LwLogic.native_thm' depends on axioms: [Lean.ofReduceBool]\n")
    errs, ax = lean_read_log(log, "c_x.lean")
    # indented, attributed and private theorems count too; an `axiom` line counts, an axiom in a comment does not
    rb = ("theorem ok_thm : True := trivial\ntheorem rfl_thm (n : Nat) : n = n := rfl\n"
          "theorem filter_filter' {a : Type} : True := trivial\n  theorem indented : True := trivial\n"
          "@[simp] theorem attr_thm : True := trivial\nprivate theorem priv_thm : True := trivial\n"
          "theorem gap_thm : True := sorry\ntheorem native_thm : True := trivial\n"
          "theorem missing_thm : True := trivial\naxiom bogus : False\n-- not a declaration: axiom in_comment\n")
    names, sorry, decl = lean_theorems(rb), lean_sorry_lines(log, "c_x.lean"), lean_axiom_decls(rb)
    ok_log = (errs == [912] and names[:3] == ["ok_thm", "rfl_thm", "filter_filter'"]
              and lean_gaps(names, ax) == ["indented", "attr_thm", "gap_thm", "native_thm", "missing_thm"]
              and sorry == [3, 20] and decl == ["bogus"])
    C.add("lean log reader: an error lands on its own line and file; sorry, extra axioms and unprinted theorems "
          "count as not proved, a `sorry` outside theorems and a declared axiom are found", ok_log, 6,
          "" if ok_log else f"errors {errs}, not proved {lean_gaps(names, ax)}, sorry {sorry}, axioms {decl}")

    # 11b. judge round trip on generated sets: canned right answers must pass, canned wrong answers must fail
    rt = {False: [0, 0], True: [0, 0]}
    for lv in (1, 2, 3, 4, 5):
        for sd in range(1, 11):
            A1 = generate(lv, sd)
            for wrong in (False, True):
                rows = [{"phase": ph["name"], "qid": q["qid"], "thinking": False, "question": q["text"],
                         "content": canned(A1, q, A1["expected"][ph["name"]][q["qid"]], wrong)}
                        for ph in A1["phases"] for q in A1["questions"]]
                res, _, _ = judge(A1, rows)
                rt[wrong][0] += sum(r["verdict"] == ("fail" if wrong else "pass") for r in res)
                rt[wrong][1] += len(res)
    C.add("judge round trip: right answers pass and wrong answers fail on 50 generated sets",
          rt[False][0] == rt[False][1] and rt[True][0] == rt[True][1], rt[False][1] + rt[True][1],
          f"right {rt[False][0]}/{rt[False][1]}, wrong caught {rt[True][0]}/{rt[True][1]}")

    # 14. question forms v2 (TEST-LOGIC.md section 13): the generator, the judge's new readers and the new scores.
    # Every check here also runs on something broken on purpose, which it must catch.
    # 14a. v1 stays as it was
    ex_dir, same_files, v1_bad = os.path.join(ROOT, "examples"), [], []
    for lv in (1, 2, 3, 4, 5):
        path = os.path.join(ex_dir, f"L{lv}-seed1.json")
        if os.path.exists(path):
            same_files.append(open(path).read() == json.dumps(generate(lv, 1), indent=1, ensure_ascii=False) + "\n")
    for lv in (1, 2, 3, 4, 5):
        for sd in range(1, 31):
            p = v1_part_problems(generate(lv, sd), generate(lv, sd, forms="v2"))
            if p:
                v1_bad.append((lv, sd, p[:2]))
    S4v, V4 = generate(4, 1, forms="v2"), generate(4, 1)
    bq = json.loads(json.dumps(S4v))
    bq["questions"][0]["text"] += " Really?"                        # a v1 question edited
    be = json.loads(json.dumps(S4v))
    be["expected"]["teach"][be["questions"][2]["qid"]] = {"kind": "yn", "value": YES}   # a v1 expected answer edited
    ctl = [bool(v1_part_problems(V4, x)) for x in (bq, be)]
    C.add("forms v1 unchanged: the committed L1-L5 examples are what the generator makes, byte for byte; every v2 set "
          "(levels 1-5 x seeds 1-30) holds its v1 set unchanged; controls: an edited v1 question and an edited v1 "
          "expected answer are caught", (all(same_files) and (len(same_files) == 5 or not os.path.isdir(ex_dir)))
          and not v1_bad and all(ctl), 150 + len(same_files),
          (f"committed files equal: {same_files}; " if not all(same_files) else "") + (f"{v1_bad[:2]}; " if v1_bad else "")
          + f"controls caught {sum(ctl)} of 2")

    # 14b. the v2 generator: deterministic, recomputes, and complete (v2_problems)
    bad2 = []
    for lv in (1, 2, 3, 4, 5):
        for sd in range(1, 31):
            A1 = generate(lv, sd, forms="v2")
            if json.dumps(A1, sort_keys=True) != json.dumps(generate(lv, sd, forms="v2"), sort_keys=True):
                bad2.append((lv, sd, "not deterministic"))
            if not check_expected(A1):
                bad2.append((lv, sd, "expected answers do not recompute"))
            if any("{" in x["text"] for x in A1["lessons"] + A1["questions"]):
                bad2.append((lv, sd, "unrendered template"))
            p = v2_problems(A1)
            if p:
                bad2.append((lv, sd, p[:2]))

    def broke(f):
        X = json.loads(json.dumps(S4v))
        f(X)
        return bool(v2_problems(X))
    listed4 = lambda X: next(q for q in X["questions"] if q.get("form") == "who_listed")
    ctl = [broke(lambda X: [q.pop("neighbour", None) for q in X["questions"]]),            # no neighbour's value
           broke(lambda X: listed4(X)["among"].remove(X["expected"]["teach"][listed4(X)["qid"]]["yes"][0])),
           broke(lambda X: X.update(phases=[p for p in X["phases"] if p.get("kind") != "prefix"])),
           broke(lambda X: X.update(questions=[q for q in X["questions"] if q.get("form") != "true_false"])),
           broke(lambda X: [q for q in X["questions"] if q["role"] == "near_miss"][-1].update(role="probe"))]
    C.add("forms v2 generator: levels 1-5 x seeds 1-30 deterministic, expected answers recompute; every single-valued "
          "taught key has wrong-yes probes on 3 untaught values (a neighbour's among them) and the true/false, fill-in "
          "and recall forms; every reverse value has its three forms and forward questions, and a forward question "
          "about an untaught pair is a near miss, not a probe; every like its list, yes/no and phase; controls: 5 sets "
          "broken on purpose", not bad2 and all(ctl), 150,
          (f"{bad2[:2]}; " if bad2 else "") + f"controls caught {sum(ctl)} of 5 (no neighbour, a holder left off the "
          f"list, no phase after a like, no true/false form, a forward question on an untaught pair scored as a probe)")

    # 14c. likes restated as a whole set
    def restate_problems(A1, B1):
        out = []
        if A1["questions"] != B1["questions"] or [p["name"] for p in A1["phases"]] != [p["name"] for p in B1["phases"]]:
            out.append("questions or phases differ")
        for ph in B1["phases"]:
            if ph["name"] == "teach" or ph.get("kind") == "prefix":
                out += [f"{ph['name']} differs"] if A1["expected"][ph["name"]] != B1["expected"][ph["name"]] else []
            if ph.get("probe") != "likes":
                continue
            # the earlier likes' own lessons deleted: a last like that restates the set keeps them ("I have always
            # liked X" restates nothing, so such a set is left out here)
            l0 = next(l for l in B1["lessons"] if l["id"] == ph["deleted"][0])
            if like_adds([l for l in B1["lessons"] if l["id"] in ph["store"]], l0["e"], l0["a"])[-1].get("vs"):
                for q in B1["questions"]:
                    if q["type"] == "CURRENT" and not q.get("form") and (q.get("e"), q["a"]) == (l0["e"], l0["a"]) \
                            and B1["expected"][ph["name"]][q["qid"]] != B1["expected"]["teach"][q["qid"]]:
                        out.append(f"{ph['name']}: the restated set lost a like ({q['qid']})")
        return out

    def restate_run():
        bad, n_sets, told_apart = [], 0, 0
        for lv in (4, 5):
            for sd in range(1, 31):
                try:
                    B1 = generate(lv, sd, forms="v2", likes="restate")
                except ValueError:
                    continue
                A1 = generate(lv, sd, forms="v2")
                n_sets += 1
                p = restate_problems(A1, B1)
                bad += [(lv, sd, p[:2])] if p else []
                told_apart += any(A1["expected"][ph["name"]] != B1["expected"][ph["name"]] for ph in B1["phases"]
                                  if ph.get("probe") == "likes")
        return bad, n_sets, told_apart
    rbad, rn, rapart = restate_run()
    w0 = writes
    ignore_vs = lambda sch_, l: w0(sch_, dict(l, vs=None)) if l.get("vs") else w0(sch_, l)
    only_these = lambda sch_, l: ([(l["e"], l["a"], when_interval(l), ("clear", None))] if l.get("vs") else []) \
        + w0(sch_, l)
    ctl = [switched_off({"writes": f}, lambda: bool(restate_problems(generate(4, 1, forms="v2"),
                                                                     generate(4, 1, forms="v2", likes="restate"))))
           for f in (ignore_vs, only_these)]
    C.add("likes restated as a whole set: in the teach phase and after each like the answers equal the one-at-a-time "
          "set's; with the earlier like lessons deleted, the restated set still holds them; controls: a restated "
          "lesson read as its newest like only, and as 'only these', are caught", not rbad and rapart > 0 and all(ctl),
          rn, (f"{rbad[:2]}; " if rbad else "") + f"{rapart} of {rn} sets told apart from one at a time; controls "
          f"caught {sum(ctl)} of 2")

    # 14d-g. the judge on each new form, with hand-written answers; each check runs again with its reader switched off
    S1v = generate(1, 1, forms="v2")
    f1 = lambda form, **kw: next(q for q in S1v["questions"] if q.get("form") == form
                                 and all(q.get(k) == v for k, v in kw.items()))
    f4 = lambda form, **kw: next(q for q in S4v["questions"] if q.get("form") == form
                                 and all(q.get(k) == v for k, v in kw.items()))
    tf_no, tf_yes = f1("true_false", v="purple"), f1("true_false", v="blue")
    tf = [(S1v, "teach", tf_no, "False.", "pass"),
          (S1v, "teach", tf_no, "**False.** Your favourite colour is blue.", "pass"),
          (S1v, "teach", tf_no, "True.", "fail"), (S1v, "teach", tf_no, "That's not true.", "pass"),
          (S1v, "teach", tf_no, "It is false that your favourite colour is purple.", "pass"),
          (S1v, "teach", tf_yes, "True, you told me that your favourite colour is blue.", "pass"),
          (S1v, "baseline", tf_no, "I'm not sure whether that's true or false.", "pass"),
          (S1v, "baseline", tf_no, "True or false? I can't know that.", "pass"),              # the question echoed
          (S1v, "baseline", tf_no, "False, I am a language model and have no favourite colour.", "pass"),  # about itself
          (S1v, "baseline", tf_no, "The statement is false.", "fail"),                        # "no" where nothing is known
          (S1v, "teach", tf_no, "Your favourite colour is blue.", "pass"),                    # no true or false: the value
          (S1v, "baseline", tf_yes, "True.", "fail")]
    wrong = labelled(tf)
    off = switched_off({"extract_tf": extract_yn}, lambda: labelled(tf))
    C.add("judge, true or false: a leading 'True' is yes and 'False' is no; the question echoed back, a refusal and "
          "'False, I am a language model ...' mean doesn't know", not wrong and bool(off), len(tf),
          (str(wrong) + "; " if wrong else "") + f"control: with true/false read as plain yes/no, {len(off)} labels break")
    fill1, fill4 = f1("fill_in"), f4("fill_in", e="my dog", a="name")
    fill = [(S1v, "teach", fill1, "blue.", "pass"), (S1v, "teach", fill1, "My favourite colour is blue.", "pass"),
            (S1v, "teach", fill1, "...blue!", "pass"), (S1v, "teach", fill1, "purple", "fail"),
            (S1v, "teach", fill1, "My favourite colour is blue! What's yours?", "unclear"),     # about itself
            (S1v, "baseline", fill1, "My favourite colour is blue! What's yours?", "unclear"),
            (S1v, "baseline", fill1, "I don't know your favourite colour.", "pass"),
            (S1v, "teach", fill1, "Your favourite colour is blue, as you told me. What else would you like to know?",
             "pass"),
            (S1v, "teach", fill1, "My favourite colour is blue, as you told me. Anything else you want to know?", "pass"),
            (S1v, "baseline", fill1, "As an AI, I don't have a favourite colour.", "pass"),
            (S4v, "teach", fill4, "My dog's name is Max! Do you have a dog too?", "unclear"),
            (S4v, "teach", fill4, "Bruno.", "pass")]
    wrong = labelled(fill)
    off = switched_off({"self_voiced": lambda t: False}, lambda: labelled(fill))
    C.add("judge, fill-in: the finished sentence is read as a recall; one the model finished about itself ('My favourite "
          "colour is blue! What's yours?') is unclear", not wrong and bool(off), len(fill),
          (str(wrong) + "; " if wrong else "") + f"control: with that reading switched off, {len(off)} labels break")
    r1 = f1("recall_yn", v="purple")
    rec = [(S1v, ph, r1, t, v) for ph, t, v in [
        ("teach", "Your favourite colour is blue. No.", "pass"), ("teach", "Your favourite colour is blue, so no.", "pass"),
        ("teach", "Your favourite colour is blue. Yes.", "fail"),            # the answer, not the recall, decides
        ("teach", "Your favourite colour is blue. So the answer is no, it is not purple.", "pass"),
        ("baseline", "I don't know your favourite colour, so I can't say.", "pass"),
        ("baseline", "I don't know your favourite colour. So: no.", "fail"),   # a "no" where nothing is known
        ("teach", "No, your favourite colour is blue.", "pass"),
        ("teach", "Your favourite colour is blue. Probably not.", "unclear"),
        ("teach", "Your favourite colour is blue.", "pass"),                  # no yes or no: the value decides
        ("teach", "Your favourite colour is purple. Yes, it is purple.", "fail"), ("teach", "Yes.", "fail")]]
    wrong = labelled(rec)
    off = switched_off({"extract_recall_yn": lambda text, sch_, a_, v_: extract_yn(text)}, lambda: labelled(rec))
    C.add("judge, recall then answer: the last yes or no answers, after 'so' or 'the answer is' or opening a later "
          "sentence; the recall is kept beside it", not wrong and bool(off), len(rec),
          (str(wrong) + "; " if wrong else "") + f"control: with the answer read from the first sentence, {len(off)} "
          f"labels break")
    w_dog, w_cat = f4("who_yn", e="my dog", a="eye_colour"), f4("who_yn", e="my cat", a="eye_colour")
    l1, b1 = f1("who_listed"), f1("who_bare")
    rv2 = [(S4v, "teach", w_dog, "Yes.", "pass"), (S4v, "teach", w_dog, "Your dog has grey eyes.", "pass"),
           (S4v, "teach", w_cat, "Your dog has grey eyes.", "pass"),       # it names someone else: no
           (S4v, "teach", w_cat, "No, it's your dog.", "pass"), (S4v, "teach", w_cat, "Your cat has grey eyes.", "fail"),
           (S4v, "baseline", w_dog, "I don't know who has grey eyes.", "pass"),
           (S4v, "teach", w_dog, "Maybe your dog?", "unclear"), (S4v, "teach", w_dog, "Bruno has grey eyes.", "pass"),
           (S1v, "teach", l1, "You.", "pass"), (S1v, "teach", l1, "Yours.", "pass"),
           (S1v, "teach", l1, "Your favourite colour is blue.", "pass"),
           (S1v, "teach", b1, "Your favourite colour is blue.", "pass"),
           (S1v, "teach", l1, "It's yours, you told me.", "pass"), (S1v, "teach", l1, "Perhaps yours?", "unclear"),
           (S1v, "teach", l1, "Your sister's favourite colour is blue.", "fail"),
           (S1v, "teach", b1, "Blue, like yours, is popular.", "fail"),
           (S1v, "teach", l1, "Your favourite colour is not blue.", "fail"),
           (S1v, "baseline", l1, "None of them, as far as I know.", "pass"),
           (S1v, "teach", b1, "Many people like blue best.", "fail"),
           (S1v, "baseline", b1, "Many people like blue best.", "pass")]
    wrong = labelled(rv2)
    off1 = switched_off({"who_yn_reading": lambda S_, q_, c_: {"got": "UNCLEAR"}}, lambda: labelled(rv2))
    off2 = switched_off({"your_own": lambda *x: None}, lambda: labelled(rv2))
    C.add("judge, reverse forms: a reverse yes/no with no yes or no word is read for who it names; 'Yours.' and 'Your "
          "favourite colour is blue.' name you", not wrong and bool(off1) and bool(off2), len(rv2),
          (str(wrong) + "; " if wrong else "") + f"controls: with the reading of who is named switched off {len(off1)} "
          f"labels break, with 'yours' not read {len(off2)}")

    # 14g2. the review of 13 Oct: answers the first readers misread. Each fix runs again with that one fix taken back
    # (the first reader restored), and every answer that needs it must break there; the guard answers must hold both
    # ways. "Definitely false." was a yes, and so was "... so I can't answer yes or no.": wrong yeses that were not.
    old_tf_re = re.compile(r"^\W*(?:(?:that|this|it|the\s+statement|the\s+answer|answer)(?:'s|\s+is|\s+was|\s*:)?\s+)?"
                           r"(true|false)\b(?!\s+or\b)", re.I)

    def old_tf(text):
        ss = sentences(text)
        m = old_tf_re.match(ss[0]) if ss else None
        if not m:
            return extract_yn(text)
        return extract_yn(" ".join([("Yes" if m.group(1).lower() == "true" else "No") + ss[0][m.end():]] + ss[1:]))
    old_answer = re.compile(r"\b(?:the\s+answer\s+is|my\s+answer\s+is|answer|so|therefore|thus|hence|that\s+means"
                            r"|which\s+means)\s*[,:]?\s*[\"']?(yes|no)\b"
                            r"(?!\s+(?:one|longer|idea|way|information|doubt|matter))", re.I)
    old_first = re.compile(r"^\W*(?:yes|yeah|yep|no|nope|not|probably|maybe|perhaps|possibly|likely)\b", re.I)
    old_self = lambda t: (lambda ss: bool(ss) and bool(FIRST_PERSON.match(ss[0])) and not MEMORY_CUE.search(
        " ".join(ss)) and any(s.rstrip().endswith("?") and YOU_ANY.search(s) for s in ss[1:]))(sentences(t))
    old_yours = re.compile(r"^\W*(?:(?:maybe|perhaps|probably|possibly)\s+)?(?:(?:it(?:'s|\s+is|\s+was)|that(?:'s|\s+is"
                           r"|\s+would\s+be)|the\s+answer\s+is)\s+)?yours\b", re.I)
    rb1, l4e = f1("recall_yn", v="blue"), f4("who_listed", a="eye_colour")
    review = [   # (what was fixed, the first reader back, answers that need the fix, guard answers)
        ("true or false: lead-ins, 'incorrect' and the question echoed back", {"extract_tf": old_tf},
         [(S1v, "teach", tf_no, "Definitely false.", "pass"), (S1v, "teach", tf_no, "This statement is false.", "pass"),
          (S1v, "teach", tf_no, "I'd say false.", "pass"), (S1v, "teach", tf_no, "Incorrect.", "pass"),
          (S1v, "teach", tf_no, "True or false: my favourite colour is purple. False.", "pass"),
          (S1v, "teach", tf_yes, "That's correct.", "pass"),
          (S1v, "teach", tf_no, "Probably false.", "unclear"),            # the right answer as a guess
          (S1v, "baseline", tf_no, "True or false: my favourite colour is purple.", "unclear")],   # only the echo
         [(S1v, "teach", tf_no, "Not true.", "pass"), (S1v, "baseline", tf_no, "True or false? I can't know that.",
                                                        "pass")]),
        ("recall then answer: a yes or no counts only on its own, not 'answer yes or no', 'no answer', 'No other'",
         {"ANSWER_YN": old_answer, "YN_FIRST": old_first},
         [(S1v, "baseline", r1, "I can't say what it is, so I can't answer yes or no.", "pass"),
          (S1v, "baseline", r1, "I don't know your favourite colour, so no answer is possible.", "pass"),
          (S1v, "teach", rb1, "Your favourite colour is blue. No other colour was mentioned.", "pass"),
          (S1v, "teach", rb1, "Your favourite colour is blue. No doubt about it.", "pass")],
         [(S1v, "teach", r1, "Your favourite colour is blue, therefore no, it's not purple.", "pass"),
          (S1v, "teach", rb1, "Blue. So yes it is.", "pass")]),
        ("fill-in: a refusal that asks back is a refusal, not the model about itself", {"self_voiced": old_self},
         [(S1v, "baseline", fill1, "I don't know. What's your favourite colour?", "pass"),
          (S1v, "teach", fill1, "I'm not sure what your favourite colour is. Could you tell me?", "fail")],
         [(S1v, "teach", fill1, "My favourite colour is blue! What's yours?", "unclear")]),
        ("reverse: 'I think it's yours' names you, 'yours are unknown to me' does not",
         {"YOURS": old_yours, "NOT_KNOWN": re.compile(r"(?!)")},
         [(S1v, "teach", l1, "I think it's yours.", "pass"),
          (S4v, "teach", l4e, "Your dog has grey eyes; yours are unknown to me.", "pass")],
         [(S1v, "teach", l1, "Yours.", "pass")]),
        ("reverse yes/no: no holder named, but the asked one's own value given", {"names_asked": lambda *x: False},
         [(S4v, "teach", w_cat, "Your cat's eyes are hazel.", "pass"),
          (S4v, "teach", w_dog, "Your dog's eyes are brown.", "fail")],
         [(S4v, "teach", w_cat, "Grey eyes are quite rare.", "unclear"),  # nobody named: not read as a yes
          (S4v, "teach", w_cat, "Your dog has grey eyes.", "pass")])]
    rev_bad, rev_note = [], []
    for what, patch, needs, guards in review:
        wrong = labelled(needs + guards)
        off = switched_off(patch, lambda: labelled(needs + guards))
        broke_n = {(ph, qid, t) for ph, qid, t, *_ in off}
        missed = [t for SS, ph, qq, t, v in needs if (ph, qq["qid"], t) not in broke_n]
        moved = [t for SS, ph, qq, t, v in guards if (ph, qq["qid"], t) in broke_n]
        if wrong or missed or moved:
            rev_bad.append(f"{what}: wrong {wrong}, not needing the fix {missed}, guards moved {moved}")
        rev_note.append(f"{len(needs) - len(missed)}/{len(needs)}")
    C.add("judge, review of 13 Oct: true/false lead-ins and echoes, a stand-alone yes or no after a recall, a refusal "
          "that asks back on a fill-in, 'yours' read with care, and a reverse yes/no that gives the asked one's value; "
          "each fix taken back breaks every answer that needs it", not rev_bad,
          sum(len(n) + len(g) for _, _, n, g in review),
          "; ".join(rev_bad) if rev_bad else f"with each fix taken back, answers that break: {', '.join(rev_note)}")

    # 14h-j. the new scores, on planted answers
    def rows_of(SS, phases, plant):
        return [{"phase": ph, "qid": qq["qid"], "thinking": False, "question": qq["text"],
                 "content": plant.get((ph, qq["qid"]), "I don't know." if ph == "baseline" and qq["type"] != "GK"
                                      else canned(SS, qq, SS["expected"][ph][qq["qid"]], False))}
                for ph in phases for qq in SS["questions"]]
    nb_yn = next(q for q in S4v["questions"] if q.get("neighbour") and not q.get("form"))
    nb_tf = next(q for q in S4v["questions"] if q.get("neighbour") and q.get("form") == "true_false"
                 and (q["e"], q["a"]) != (nb_yn["e"], nb_yn["a"]))       # another key: its break needs true/false
    plant1 = {("teach", nb_yn["qid"]): "Yes, that is right.", ("teach", nb_tf["qid"]): "True."}

    def wy_case(plant=plant1):
        k = judge(S4v, rows_of(S4v, ("baseline", "teach"), plant))[1][("forms", "teach")]["off"]
        w = k["wrong_yes"]
        return w["yes"], w["base_yes"], w["n"] == w["base_n"] > 0, w["neighbour"][0], len(k["coherence"]["breaks"])
    all_yes = {("teach", q["qid"]): "Yes." for q in S4v["questions"] if is_yn(q)}
    got, want = wy_case(), (2, 0, True, 2, 2)
    every = judge(S4v, rows_of(S4v, ("baseline", "teach"), all_yes))[1][("forms", "teach")]["off"]["wrong_yes"]
    every_yn = lambda S_, ph_: [q["qid"] for q in S_["questions"] if is_yn(q)]
    ctl = [switched_off({"wrong_yes_pool": every_yn}, wy_case) != want,
           switched_off({"is_yn": lambda q: q["type"] == "YES_NO" and not q.get("form")}, wy_case) != want]
    C.add("wrong-yes rate and coherence breaks on planted answers: two wrong yeses (a neighbour's value, plain and true/"
          "false) give 2 against 0 in the baseline on the same questions, and 2 keys with yes to two values; 'always "
          "yes' scores every question; controls: the rate over all yes/no questions, and breaks read from the plain form "
          "only, are caught", got == want and every["yes"] == every["n"] > 0 and all(ctl), every["n"],
          f"got {got}, want {want}; always yes {every['yes']}/{every['n']}; controls caught {sum(ctl)} of 2")
    listed_eye = f4("who_listed", a="eye_colour")
    bare_name = f4("who_bare", a="name")
    fwd_eye = next(q for q in S4v["questions"] if q["type"] == "CURRENT" and not q.get("form")
                   and q.get("e") == S4v["expected"]["teach"][listed_eye["qid"]]["yes"][0] and q["a"] == "eye_colour")
    plant2 = {("teach", listed_eye["qid"]): "Nobody that I know of.",
              ("teach", bare_name["qid"]): "Many people are called Bruno.",
              ("del-A", fwd_eye["qid"]): "It is brown.", ("del-A", listed_eye["qid"]): "Nobody that I know of."}

    def rev_case():
        res, cons_, _ = judge(S4v, rows_of(S4v, ("baseline", "teach", "del-A"), plant2))
        t, d = cons_[("forms", "teach")]["off"]["reverse"], cons_[("forms", "del-A")]["off"]["reverse"]
        fw = next(x.get("forward") for x in res if x["phase"] == "del-A" and x["qid"] == listed_eye["qid"])
        return t["curse"], t["bare_only"], len(t["ok"]), d["curse"], fw
    got, want = rev_case(), (["eye_colour grey"], ["name Bruno"], 1, [], False)
    ctl = switched_off({"forward_ok": lambda *x: True}, rev_case) != want
    C.add("reverse on planted answers: a failed listed form where the forward question passed is the reversal curse; "
          "a failed bare form with the listed form right is not; where the forward question failed nothing is counted; "
          "control: counting without the forward question is caught", got == want and ctl, 3,
          f"got {got}, want {want}; control {'caught' if ctl else 'NOT caught'}")
    after = next(p["name"] for p in S4v["phases"] if p.get("kind") == "prefix")
    first_like = next(l for l in S4v["lessons"] if l["a"] == "likes" and l["e"] == "me")["v"]
    yn_first = next(q for q in S4v["questions"] if q["type"] == "YES_NO" and not q.get("form") and q.get("e") == "me"
                    and q["a"] == "likes" and q["v"] == first_like)
    lst = next(q for q in S4v["questions"] if q["type"] == "CURRENT" and not q.get("form") and q.get("e") == "me"
               and q["a"] == "likes")
    newest = S4v["expected"][after][lst["qid"]]["yes"]
    plant3 = {(after, yn_first["qid"]): "No.",
              (after, lst["qid"]): "You like " + " and ".join(x for x in newest if x != first_like) + "."}

    def likes_case():
        cons_ = judge(S4v, rows_of(S4v, ("baseline", "teach", after), plant3))[1]
        a9, t = cons_[("forms", after)]["off"]["likes"], cons_[("forms", "teach")]["off"]["likes"]
        return a9["yes_no"], a9["list"], a9["newest"], t["yes_no"], t["list"]
    got, want = likes_case(), ([0, 1], [1, 2], [3, 3], [1, 1], [2, 2])
    el0 = earlier_likes
    ctl = switched_off({"earlier_likes": lambda *x: (lambda r: r and (r[0], r[1] + [r[0]]))(el0(*x))}, likes_case) != want
    # the same two questions answered off topic: still counted, as lost (review of 13 Oct). The first version left
    # such answers out, which made Preservation look better the less the model answered
    plant_off = {(after, yn_first["qid"]): "My name is Nemotron.", (after, lst["qid"]): "My name is Nemotron."}

    def likes_off():
        a9 = judge(S4v, rows_of(S4v, ("baseline", after), plant_off))[1][("forms", after)]["off"]["likes"]
        return a9["yes_no"], a9["list"], a9["newest"]
    ln0 = likes_named
    got_off, want_off = likes_off(), ([0, 1], [1, 2], [2, 3])
    ctl_off = switched_off({"likes_named": lambda q_, g_: None if g_.get("got") not in ("YN", "SET", "ABSENT")
                            else ln0(q_, g_)}, likes_off) != want_off
    C.add("likes on planted answers: after the second like, a 'no' to the first and a list without it are counted as "
          "lost (Ignore Rate 2 of 3); the teach phase keeps it; off-topic answers to the same questions count as lost, "
          "not left out; controls: counting the newest like as an earlier one, and leaving out answers that are not a "
          "yes, no or list, are caught", got == want and got_off == want_off and ctl and ctl_off, 8,
          f"got {got}, want {want}; off topic got {got_off}, want {want_off}; controls "
          f"{'caught' if ctl else 'NOT caught'}, {'caught' if ctl_off else 'NOT caught'}")

    # 14k. judge round trip on v2 sets; 14l. the committed v2 examples are what the generator makes
    rt2 = {False: [0, 0], True: [0, 0]}
    for lv in (1, 2, 3, 4, 5):
        for sd in range(1, 11):
            A1 = generate(lv, sd, forms="v2")
            for wrong in (False, True):
                rows = [{"phase": ph["name"], "qid": q["qid"], "thinking": False, "question": q["text"],
                         "content": canned(A1, q, A1["expected"][ph["name"]][q["qid"]], wrong)}
                        for ph in A1["phases"] for q in A1["questions"]]
                res, _, _ = judge(A1, rows)
                rt2[wrong][0] += sum(r["verdict"] == ("fail" if wrong else "pass") for r in res)
                rt2[wrong][1] += len(res)
    C.add("judge round trip on forms v2: right answers pass and wrong answers fail on 50 generated v2 sets",
          rt2[False][0] == rt2[False][1] and rt2[True][0] == rt2[True][1], rt2[False][1] + rt2[True][1],
          f"right {rt2[False][0]}/{rt2[False][1]}, wrong caught {rt2[True][0]}/{rt2[True][1]}")
    v2dir = os.path.join(ex_dir, "v2")
    if os.path.isdir(v2dir):
        made = {f"L{lv}-seed1-v2": generate(lv, 1, forms="v2") for lv in (1, 2, 3, 4, 5)}
        made["L4-seed1-v2-restate"] = generate(4, 1, forms="v2", likes="restate")
        for nm in ("neg-seed1", "neg2-seed1"):
            made[nm + "-v2"] = add_forms_v2(json.load(open(os.path.join(ex_dir, nm + ".json"))))
        txt = lambda X: json.dumps(X, indent=1, ensure_ascii=False) + "\n"
        on_disk = sorted(f[:-5] for f in os.listdir(v2dir) if f.endswith(".json"))
        differ = [k for k, X in made.items() if not os.path.exists(os.path.join(v2dir, k + ".json"))
                  or open(os.path.join(v2dir, k + ".json")).read() != txt(X)]
        edited = json.loads(json.dumps(made["L1-seed1-v2"]))
        edited["questions"][-1]["text"] += "?"
        ctl = txt(edited) != open(os.path.join(v2dir, "L1-seed1-v2.json")).read() if "L1-seed1-v2" in on_disk else False
        C.add(f"forms v2 examples: the {len(made)} committed examples/v2 sets are what the generator makes, and no "
              f"other set is there; control: a copy with one edited question is told apart",
              not differ and on_disk == sorted(made) and ctl, len(made),
              (f"differ {differ}; " if differ else "") + (f"on disk {on_disk}; " if on_disk != sorted(made) else ""))

    # 11c. level 7 through the judge: right answers pass, wrong ones fail, on every phase; only delete phases get the
    # delete scores and kept_stable (checkpoints and pairs hold lessons not taught yet, not deleted ones). Control:
    # with every phase after teach read as a delete (the rule of levels 1 to 5), the checkpoints get them too.
    rt, kinds_bad = {False: [0, 0], True: [0, 0]}, []
    for size, sd in ((12, 1), (12, 2), (L7_FACTS, 1)):
        A1 = generate(7, sd, n_facts=size)
        for wrong in (False, True):
            rows = [{"phase": ph["name"], "qid": q["qid"], "thinking": False, "question": q["text"],
                     "content": canned(A1, q, A1["expected"][ph["name"]][q["qid"]], wrong)}
                    for ph in A1["phases"] for q in A1["questions"]]
            res, cons, _ = judge(A1, rows)
            rt[wrong][0] += sum(r["verdict"] == ("fail" if wrong else "pass") for r in res)
            rt[wrong][1] += len(res)
            if not wrong:
                kinds = {p["name"]: p.get("kind") for p in A1["phases"]}
                dk = {r["phase"] for r in res if {"delete_affected", "delete_kept"} & set(r["buckets"])}
                ks = {k[1] for k in cons if isinstance(k, tuple) and k[0] == "kept"}
                want = {n for n, k in kinds.items() if k == "delete"}
                if dk != want or ks != want:
                    kinds_bad.append((size, sd, sorted(dk), sorted(ks)))
    keep = globals()["is_delete_phase"]
    globals()["is_delete_phase"] = lambda ph: ph["name"] not in ("baseline", "teach")
    try:
        A1 = generate(7, 1, n_facts=12)
        rows = [{"phase": ph["name"], "qid": q["qid"], "thinking": False, "question": q["text"],
                 "content": canned(A1, q, A1["expected"]["teach"][q["qid"]], False)}
                for ph in A1["phases"] for q in A1["questions"]]
        res, _, _ = judge(A1, rows)
        ctl = any("delete_kept" in r["buckets"] for r in res if r["phase"].startswith("ck-"))
    finally:
        globals()["is_delete_phase"] = keep
    C.add("judge on level 7: right answers pass and wrong ones fail in every phase; only the delete phases get delete "
          "scores and kept_stable; control: read as levels 1-5 are, the checkpoints get them too",
          rt[False][0] == rt[False][1] and rt[True][0] == rt[True][1] and not kinds_bad and ctl,
          rt[False][1] + rt[True][1], f"right {rt[False][0]}/{rt[False][1]}, wrong caught {rt[True][0]}/{rt[True][1]}"
          + (f"; phase kinds {kinds_bad[:2]}" if kinds_bad else "") + ("" if ctl else "; control not seen"))

    # 12. optional: judge the real 1.0.7 removal probes if they are on this machine
    if probes_dir and os.path.isdir(probes_dir):
        S6 = generate("eyes", 1, facts=facts)
        stage = {"baseline": "rm-R", "teach": "7-restart-noslide", "rm-3": "rm-3", "rm-4": "rm-4", "rm-5": "rm-5",
                 "rm-345": "rm-345", "rm-all": "rm-all"}
        specs = []
        for ph, st in stage.items():
            for suf in ("", "-eyes"):
                p = os.path.join(probes_dir, f"probes-{st}{suf}.jsonl")
                if os.path.exists(p):
                    specs.append(f"{p}:{ph}")
        if specs:
            res, cons, _ = judge(S6, load_answers(specs, {p["name"] for p in S6["phases"]}))
            p = sum(x["verdict"] == "pass" for x in res)
            print(f"  info  judged the real 1.0.7 removal answers: {p}/{len(res)} pass "
                  f"(run `judge` on them for the table)")

    # 13. the intake check (zengine/intake.py): parser round trip, golden verdicts, agreement with this file's own
    # question set, broken classifiers, the gate against the full sequence, the command line, and the extractor
    # against a mock endpoint
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import intake
    intake.selftest_checks(C, n, seed)

    # 14. level 6 (zengine/levels.py): the chain rules, Kleene logic, the generator, the echo verdict and the scores
    _levels().selftest_checks(C, n, seed)

    ok = all(r[1] for r in C.rows)
    print(f"selftest {'PASSED' if ok else 'FAILED'}: {sum(r[1] for r in C.rows)}/{len(C.rows)} checks, "
          f"{time.time() - t0:.0f} s")
    return ok


# ================================================================================================== cli
def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["intake"]:                       # its own options: python3 zengine/lwlogic.py intake --help
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import intake
        sys.exit(intake.main(argv[1:], prog="lwlogic.py intake"))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--level", help="1..7 (6: two-step questions, zengine/levels.py; 7: fill to capacity), or eyes "
                   "(or give --from)")
    g.add_argument("--seed", type=int, default=1)
    g.add_argument("--n-facts", type=int, default=None, help=f"level 7 only: how many facts (default {L7_FACTS})")
    g.add_argument("--forms", choices=FORMS, default="v1", help="question forms: v1 (as always) or v2 (TEST-LOGIC.md "
                   "section 13: wrong-yes probes, who-has in three forms, likes as a set)")
    g.add_argument("--likes", choices=("one", "restate"), default="one",
                   help="with --forms v2: 'restate' makes every 'I like X' after the first restate the whole set")
    g.add_argument("--from", dest="from_set", help="add the --forms v2 questions to this existing v1 set instead "
                   "(for hand-made sets such as examples/neg-seed1.json)")
    g.add_argument("--today", default="2026-10-12", help="the day the lessons are taught (ISO date)")
    g.add_argument("--facts", default=None, help="optional: the #678 trial's facts.json (--level eyes only)")
    g.add_argument("--out")
    s = sub.add_parser("show")
    s.add_argument("set")
    j = sub.add_parser("judge")
    j.add_argument("set")
    j.add_argument("answers", nargs="+", help="answers.jsonl, optionally FILE:PHASE")
    j.add_argument("--phase", help="phase for lines that carry none")
    j.add_argument("--json", help="write per-answer verdicts here")
    du = sub.add_parser("dummy", help="the answers of a dummy system that says the same to every question")
    du.add_argument("set")
    du.add_argument("--says", required=True, choices=sorted(DUMMIES),
                    help="dont_know: \"I don't know.\" to everything; yes: \"Yes.\" to everything")
    du.add_argument("--out", help="write the answers here (JSON lines); default: print them")
    le = sub.add_parser("lean")
    le.add_argument("set")
    le.add_argument("--out")
    lc = sub.add_parser("leancheck")
    lc.add_argument("sets", nargs="*", help="test sets (default: the examples that have a .check.lean)")
    lc.add_argument("--lean", default="lean", help="the lean binary (default: lean on PATH, via elan)")
    lc.add_argument("--work", help="keep the generated .lean files here (default: a temporary folder)")
    st = sub.add_parser("selftest")
    st.add_argument("--n", type=int, default=1500)
    st.add_argument("--seed", type=int, default=1)
    st.add_argument("--facts", default=None, help="optional: the #678 trial's facts.json")
    st.add_argument("--probes", default=None,
                    help="optional: a folder of the #678 trial's 1.0.7 removal answers (probes-*.jsonl) to judge")
    sub.add_parser("intake", add_help=False)         # handled above; listed here for --help
    a = ap.parse_args(argv)

    if a.cmd == "gen":
        if bool(a.level) == bool(a.from_set) or (a.from_set and a.forms != "v2"):
            ap.error("gen takes --level, or --from with --forms v2")
        if a.n_facts is not None and a.level != "7":
            ap.error("--n-facts is for --level 7 only")
        try:
            S = (add_forms_v2(json.load(open(a.from_set)), a.likes) if a.from_set else
                 generate(a.level if a.level == "eyes" else int(a.level), a.seed, a.today, a.facts, a.forms, a.likes,
                          n_facts=a.n_facts))
        except ValueError as e:  # a level 7 size outside 1 to 100, a level that is not a number, or forms v2 refused
            ap.error(str(e))
        txt = json.dumps(S, indent=1, ensure_ascii=False)
        if a.out:
            open(a.out, "w").write(txt + "\n")
            print(f"wrote {a.out}: {len(S['lessons'])} lessons, {len(S['questions'])} questions, "
                  f"{len(S['phases'])} phases")
        else:
            print(txt)
    elif a.cmd == "show":
        print(show(json.load(open(a.set))))
    elif a.cmd == "judge":
        S = json.load(open(a.set))
        if not check_expected(S):
            print("warning: the expected answers in this set do not match the oracle (edited file?)", file=sys.stderr)
        specs = [x if ":" in x or not a.phase else f"{x}:{a.phase}" for x in a.answers]
        rows = load_answers(specs, {p["name"] for p in S["phases"]})
        res, cons, skipped = judge(S, rows)
        print(f"judged {len(res)} answers against {S['id']}" + (f" ({skipped} lines skipped: unknown phase or question)" if skipped else ""))
        print(score_table(S, res, cons))
        two = _levels().two_step_scores(S, res) if any(q["type"] in TWO_STEP for q in S["questions"]) else None
        if two:                                      # level 6: stated against implied facts, by number of steps
            print(_levels().two_step_table(two))
        if a.json:
            json.dump(dict(json_report(S, res, cons), **({"two_step": two} if two else {})), open(a.json, "w"), indent=1)
    elif a.cmd == "dummy":
        S = json.load(open(a.set))
        txt = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in dummy_rows(S, a.says))
        if a.out:
            open(a.out, "w").write(txt)
            print(f"wrote {a.out}: {txt.count(chr(10))} answers, all \"{DUMMIES[a.says]}\"")
        else:
            print(txt, end="")
    elif a.cmd == "lean":
        S = json.load(open(a.set))
        txt, n = lean_export(S)
        if a.out:
            open(a.out, "w").write(txt)
            print(f"wrote {a.out}: {n} checks")
        else:
            print(txt)
    elif a.cmd == "leancheck":
        r = lean_check(a.sets or default_lean_sets(), a.lean, a.work)
        sys.exit(2 if r is None else 0 if r else 1)
    elif a.cmd == "selftest":
        sys.exit(0 if selftest(a.n, a.seed, a.facts, a.probes) else 1)


if __name__ == "__main__":
    main()
