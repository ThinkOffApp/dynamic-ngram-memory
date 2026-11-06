"""Fill a store from a chat transcript, in two steps with a person between them.

`extract` has a model read the transcript from oldest to newest and writes candidates to a file. It never touches a
store. Each candidate is one sentence that stands alone, with the message it came from. A person reads the file and sets
"approved": true on the facts worth keeping. `apply` then writes only those, oldest first, each citing its message.

  python -m notes_proxy prefill extract --transcript room.jsonl --since 2026-11-01 --model-url http://127.0.0.1:8091 \\
      --model NAME --out candidates.jsonl
  python -m notes_proxy prefill apply --candidates candidates.jsonl --url http://team:8095 --key-file team.key \\
      --cite-prefix room:thinkoff-development/

A transcript is one JSON object a line, with id, created_at, from and body. Text that looks like a key is removed before
the model sees it, and a candidate that still holds one is dropped, because a store may be read from outside. A filter
of known shapes cannot promise to catch every secret, which is one more reason a person reads the candidates. A
transcript also holds things that stopped being true, and a free-text note does not replace an older one by itself, so
candidates that share a subject are marked for the reader, who keeps the one that is true today.
"""
import argparse
import json
import re
import sys
import urllib.error
import urllib.request

from . import team

SECRET = re.compile(
    r"xfb_[a-f0-9]{16,}|ykn_[A-Za-z0-9_-]{16,}|sk-[A-Za-z0-9_-]{16,}|sb_(?:secret|publishable)_[A-Za-z0-9_-]{8,}"
    r"|gh[pousr]_[A-Za-z0-9]{20,}|eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_.-]*"
    r"|(?i:authorization\s*:\s*\S+(?:[ \t]+\S+)?)|(?i:(?:bearer|basic)\s+[A-Za-z0-9._~+/=-]{16,})"
    r"|(?i:(?:password|passwd|api[_ -]?key|secret|token)\s*[:=]\s*\S+)"
    r"|(?<![A-Za-z0-9_-])[A-Za-z0-9_-]{40,}(?![A-Za-z0-9_-])")   # 40: a 32-byte token_urlsafe key is 43 characters
SKIP = ("[Confirmation needed]",)      # requests for approval carry commands, not facts
# "hermes confirmed that...", "Petrus asked why...": a report of what someone said, not a fact about the world
HEARSAY = re.compile(r"^\W*\w+(?:'s)?\s+(?:\w+\s+)?(?:ask(?:s|ed)|confirm(?:s|ed)|tell?s|told|report(?:s|ed)|sa(?:ys|id)|wants?|recommend(?:s|ed))\b", re.I)
LINE_LIMIT = 800                       # characters of one message the model sees
FACT_MIN, FACT_MAX = 20, 300           # characters of one candidate

SYSTEM = """You read a team's chat and pick out lasting facts worth remembering months from now.

A lasting fact is one of: a rule the owner set, an address or port of a machine, a decision that was made, a fix that
worked and what it fixed, or a correction of something believed before.

Leave out: chatter, plans, proposals, things someone will do, guesses, questions, test values, progress reports, what
a thing's state is right now, numbers from a single run, and anything about a message itself. Never report that someone
said, asked or confirmed something: state the fact itself. A command or setting someone only suggested is not a fact.
If a message corrects an earlier one, keep only the corrected fact.

Write each fact as one plain sentence that makes sense alone: name the machine, person or thing, and never write "it",
"this", "we" or "today". Most messages hold no lasting fact, so an empty list is a normal answer.

Reply with JSON only: a list of {"fact": sentence, "subject": the thing it is about in one or two words, "id": the
message's tag}. Reply [] when there is nothing."""


def clean(text):
    """The text with anything that looks like a key taken out."""
    return SECRET.sub("[removed]", text)


def load(path, since=None, until=None):
    """The transcript's messages from oldest to newest, within the dates (ISO prefixes, `until` included)."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                m = json.loads(line)
            except ValueError:
                continue
            at, body = m.get("created_at") or "", m.get("body")
            if not isinstance(body, str) or not body.strip() or not m.get("id") or body.startswith(SKIP):
                continue
            if (since and at < since) or (until and at[:len(until)] > until):
                continue
            rows.append(m)
    rows.sort(key=lambda m: m["created_at"])
    return rows


def chunks(rows, size):
    for i in range(0, len(rows), size):
        yield rows[i:i + size]


def shown(rows):
    """(the chunk as the model sees it, its tags: tag -> message)."""
    tags = {f"m{i}": m for i, m in enumerate(rows)}
    lines = [f"[{t}] {m.get('from', '?')}: {clean(m['body'])[:LINE_LIMIT]}" for t, m in tags.items()]
    return "\n\n".join(lines), tags


def ask(url, model, text, key=None, timeout=300):
    """The model's reply text. `store` false keeps the transcript out of a server's own memory."""
    body = {"model": model, "temperature": 0, "max_tokens": 1200, "store": False,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]}
    head = {"Content-Type": "application/json"}
    if key:
        head["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(), head)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)["choices"][0]["message"].get("content") or ""


def parsed(reply):
    """The list in the model's reply, or [] when there is none."""
    a, b = reply.find("["), reply.rfind("]")
    if a < 0 or b <= a:
        return []
    try:
        v = json.loads(reply[a:b + 1])
    except ValueError:
        return []
    return v if isinstance(v, list) else []


def norm(text):
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def candidates(found, tags, seen):
    """The rows worth showing a person: a tag that names a message of this chunk, a sentence of a sensible size, no
    key, and not a repeat of an earlier fact. Returns (rows, how many were dropped)."""
    rows, dropped = [], 0
    for c in found:
        fact = c.get("fact") if isinstance(c, dict) else None
        tag = c.get("id") if isinstance(c, dict) else None
        m = tags.get(tag) if isinstance(tag, str) else None
        if not isinstance(fact, str) or m is None or not FACT_MIN <= len(fact.strip()) <= FACT_MAX \
                or SECRET.search(fact) or "[removed]" in fact or HEARSAY.search(fact) or norm(fact) in seen:
            dropped += 1
            continue
        seen.add(norm(fact))
        subject = c.get("subject") if isinstance(c.get("subject"), str) else ""
        if SECRET.search(subject):
            dropped += 1
            continue
        rows.append({"fact": fact.strip(), "subject": subject.strip()[:60] or "team", "id": m["id"],
                     "at": m["created_at"], "from": m.get("from"), "approved": False})
    return rows, dropped


def mark_same_subject(rows):
    """Each candidate says how many newer ones share its subject: a sign that it may have stopped being true."""
    newer = {}
    for r in reversed(rows):
        k = norm(r["subject"])
        r["newer_same_subject"] = newer.get(k, 0)
        newer[k] = newer.get(k, 0) + 1
    return rows


def extract(a):
    rows = load(a.transcript, a.since, a.until)
    key = team.read_key(a.model_key_file) if a.model_key_file else None
    out, seen, dropped, failed = [], set(), 0, 0
    for n, part in enumerate(chunks(rows, a.chunk), 1):
        text, tags = shown(part)
        try:
            reply = ask(a.model_url, a.model, text, key)
        except (OSError, urllib.error.URLError, ValueError, KeyError, IndexError) as e:
            failed += 1
            print(f"prefill: chunk {n} failed ({type(e).__name__})", file=sys.stderr, flush=True)
            continue
        got, lost = candidates(parsed(reply), tags, seen)
        out += got
        dropped += lost
        print(f"prefill: chunk {n}: {len(part)} messages, {len(got)} candidates", file=sys.stderr, flush=True)
    mark_same_subject(out)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"prefill: {len(rows)} messages read, {len(out)} candidates in {a.out}, {dropped} dropped, {failed} chunks "
          f"failed. Nothing was written to any store. Set \"approved\": true on the facts to keep, then run apply.")
    return 1 if failed else 0


def apply(a):
    rows = []
    with open(a.candidates, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    chosen = sorted((r for r in rows if r.get("approved") is True), key=lambda r: r["at"])
    if a.dry_run:
        print(f"prefill: {len(chosen)} of {len(rows)} approved. --dry-run: nothing written.")
        return 0
    key = team.read_key(a.key_file)
    written = known = failed = 0
    for r in chosen:
        if SECRET.search(r["fact"]) or SECRET.search(r.get("subject") or ""):
            failed += 1
            continue
        body = {"text": r["fact"], "subject": r["subject"], "cite": (a.cite_prefix + r["id"])[:200]}
        req = urllib.request.Request(a.url.rstrip("/") + "/v1/notes", json.dumps(body).encode(),
                                     {"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                if resp.status == 201:
                    written += 1
                else:
                    known += 1
        except (OSError, urllib.error.URLError):
            failed += 1
    print(f"prefill: {written} notes written, {known} already known, {failed} failed, of {len(chosen)} approved.")
    return 1 if failed else 0


def main(argv):
    p = argparse.ArgumentParser(prog="python -m notes_proxy prefill",
                                description="Fill a store from a chat transcript: extract candidates, have a person "
                                            "approve them, then apply.")
    sub = p.add_subparsers(dest="step", required=True)
    e = sub.add_parser("extract", help="write candidates to a file; touches no store")
    e.add_argument("--transcript", required=True, help="one JSON object a line: id, created_at, from, body")
    e.add_argument("--since", help="first date, an ISO prefix such as 2026-11-01")
    e.add_argument("--until", help="last date, included")
    e.add_argument("--model-url", required=True, help="an OpenAI-compatible server's base URL")
    e.add_argument("--model", required=True)
    e.add_argument("--model-key-file", help="the file holding the model server's key, if it needs one")
    e.add_argument("--chunk", type=int, default=30, help="messages the model reads at a time (default 30)")
    e.add_argument("--out", required=True, help="the candidates file")
    w = sub.add_parser("apply", help="write the approved candidates to a store")
    w.add_argument("--candidates", required=True)
    w.add_argument("--url", help="the proxy's base URL")
    w.add_argument("--key-file", help="the file holding a key that may write")
    w.add_argument("--cite-prefix", default="", help="put before each message id in the note's cite field")
    w.add_argument("--dry-run", action="store_true", help="count what would be written, and write nothing")
    a = p.parse_args(argv)
    if a.step == "apply" and not a.dry_run and not (a.url and a.key_file):
        p.error("apply needs --url and --key-file, or --dry-run")
    try:
        return (extract if a.step == "extract" else apply)(a)
    except team.TeamError as e:
        raise SystemExit(f"prefill: {e}")
