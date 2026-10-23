"""Whether a user's chat message tells something (a note may be written from it) or asks (it may not).

A port of the TensorFold server's chat-write gate: `notes/question.zig` (`reason`) and `notes/intake.zig`
(`statement`), English only, as NOTES-IN-ENGINE.md describes it. Only the chat path uses it: an explicit
`POST /v1/notes` writes whatever text it is given, as the server does.
"""

# Every code point Unicode 16 names a question mark (punctuation and symbols), and the interrobangs.
MARKS = frozenset([0x3F, 0xBF, 0x37E, 0x55E, 0x61F, 0x1367, 0x1945, 0x203D, 0x2047, 0x2048, 0x2049, 0x2753, 0x2754,
                   0x2CFA, 0x2CFB, 0x2E2E, 0x2E54, 0xA60F, 0xA6F7, 0xFE16, 0xFE56, 0xFF1F, 0x11143, 0x1E95F, 0x1FBC4])

# The first word of a question (lower case): the interrogatives and the auxiliaries a question opens with.
OPENERS = frozenset(["who", "what", "where", "when", "why", "how", "which", "whose", "whom", "is", "are", "am", "was",
                     "were", "do", "does", "did", "can", "could", "will", "would", "should", "shall", "may", "might",
                     "have", "has", "had"])

# Requests that ask for something told back, whole words at the start (lower case).
REQUESTS = ("tell me", "remind me", "do you know", "show me", "list", "explain")

STATEMENT_LIMIT = 400   # bytes: a note is one sentence, not a pasted document


def _space(cp):
    """Python's str.isspace(), as the server spells it out."""
    return cp in (0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x20, 0x85, 0xA0, 0x1680, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000) \
        or 0x1C <= cp <= 0x1F or 0x2000 <= cp <= 0x200A


def _trailing(cp):
    """What may close a sentence after its question mark: quotes, brackets, emoji (and their joiners) and "!"."""
    if cp in (0x22, 0x27, 0x60, 0xB4, 0xAB, 0xBB, 0x2039, 0x203A, 0xFF02, 0xFF07) or 0x2018 <= cp <= 0x201F \
            or 0x301D <= cp <= 0x301F:
        return True
    if cp in (0x28, 0x29, 0x5B, 0x5D, 0x7B, 0x7D, 0x3C, 0x3E, 0xFF08, 0xFF09, 0xFF3B, 0xFF3D, 0xFF5B, 0xFF5D, 0x27E8,
              0x27E9) or 0x3008 <= cp <= 0x3011 or 0x3014 <= cp <= 0x301B:
        return True
    if cp in (0x21, 0x203C, 0xFF01):
        return True
    return (0x1F000 <= cp <= 0x1FAFF or 0x2600 <= cp <= 0x27BF or 0x2B00 <= cp <= 0x2BFF or 0x2300 <= cp <= 0x23FF
            or 0xFE00 <= cp <= 0xFE0F or cp in (0x200D, 0x20E3) or 0xE0000 <= cp <= 0xE007F)


def _decoration(cp):
    return _space(cp) or _trailing(cp)


def _first_word(s):
    """The first word's ASCII letters; a contraction counts as its first part ("what's" what, "isn't" is)."""
    n = 0
    while n < len(s) and s[n].isascii() and s[n].isalpha():
        n += 1
    w, rest = s[:n], s[n:]
    negated = (rest.startswith("'t") or rest.startswith("’t")) and len(w) > 1 and w[-1] == "n"
    if not negated or w == "can":
        return w
    if w == "won":
        return "will"
    if w == "shan":
        return "shall"
    return w[:-1]


def _lower(s, n=48):
    """The start of s, ASCII letters lowered (the server lowers the first 48 bytes; 48 characters here)."""
    return "".join(c.lower() if c.isascii() else c for c in s[:n])


def reason(text):
    """Why `text` is a question or a request as an English chat message asks one, or None when it tells."""
    end = text
    while end:
        cp = ord(end[-1])
        if cp in MARKS:
            return "a question: it ends in a question mark"
        if not _decoration(cp):
            break
        end = end[:-1]
    s = text
    while s:
        cp = ord(s[0])
        if cp in MARKS:
            return "a question: it starts with a question mark"
        if not _decoration(cp):
            break
        s = s[1:]
    words = _lower(s)
    if words.startswith("please ") or words.startswith("please, "):
        s = s[len("please"):].lstrip(", ")
        words = _lower(s)
    for r in REQUESTS:
        if words.startswith(r) and (len(words) == len(r) or not (words[len(r)].isascii() and words[len(r)].isalnum())):
            return "a request: it asks to be told something"
    if _first_word(words) in OPENERS:
        return "a question: it opens as one"
    return None


def statement(text):
    """(the text to write, None) for a statement, or (None, why it is not one)."""
    t = text
    # without the space around it (Python's isspace, as the server's)
    lo, hi = 0, len(t)
    while lo < hi and _space(ord(t[lo])):
        lo += 1
    while hi > lo and _space(ord(t[hi - 1])):
        hi -= 1
    t = t[lo:hi]
    why = reason(t)
    if why:
        return None, why
    words = any((ord(c) >= 0x80 and not _decoration(ord(c))) or (c.isascii() and c.isalnum()) for c in t)
    if not words:
        return None, "no words"
    if "\n" in t or "\r" in t:
        return None, "more than one line: a note is one sentence"
    if len(t.encode("utf-8")) > STATEMENT_LIMIT:
        return None, "over 400 bytes: a note is one sentence"
    return t, None
