# Stem lookup: a language-robust variant of notes.py retrieve() (18 Oct 2026, 00:30 CEST)

**What was run.** Everything here is MEASURED unless a line says otherwise.
- Code: `stem_lookup.py`, Python only. It runs on the test suite's working repository at 626c7b6 with the Nemotron tokenizer d9d758fb, on the MacBook.
- The stores were regrown the way `gen1m_ml.py` grows them: PYTHONHASHSEED=0, the fast intake, and the default NoteStore config (hop "subject", cap 512).
- Raw results, with all 47 variant rows at every size: `stem-results/full-{en,fi,de}.json`. The `dev-*.json` files hold an earlier pass at 1k-100k that also includes me=rel50.

**Controls, at every size and in every language:**
- The original retrieve() returns the published ref ids on 56/56 questions.
- This file's copy of `_lookup`, with every flag off, returns the original's ids and block byte for byte on 56/56.
- `+1` reproduces the published translated numbers: fi 24/17/17/19 and de 30/26/19/21.

## Recall

The table counts questions whose needed notes are all in the block, at cap 512, out of 45. Sizes are 1k / 10k / 100k / 1M.

| variant | en | fi | de |
| --- | --- | --- | --- |
| baseline (original retrieve) | 30/27/27/23 | 23/16/15/16 | 25/17/11/11 |
| +1 language past cue + subject phrases | 30/27/27/23 | 24/17/17/19 | 30/26/19/21 |
| +1+2 stem p4L w0.5 (chosen) | 32/27/27/23 | 25/17/18/19 | 30/26/21/21 |
| +1+2 stem p4 w0.5 (all words) | 32/27/**26**/23 | 25/17/16/18 | 30/24/21/21 |
| +1+2 stem p5 w0.5 | 32/27/**26**/23 | 24/17/17/18 | 30/24/21/21 |
| +1+2 stem p6 w0.5 | 32/27/**26**/23 | 24/16/16/18 | 30/25/20/20 |
| +1+2 stem c4 w0.5 | **29**/27/**26**/**22** | 22/19/16/16 | 30/22/21/21 |
| +1+3 alias=idf me=all (no stems) | 33/30/29/27 | 26/19/19/20 | 33/29/23/23 |
| **+1+2+3 p4L w0.5 alias=idf me=all (chosen)** | **34/29/28/26** | **26/19/19/20** | **35/29/25/24** |
| aside: chosen + subject bonus x4 | 40/34/35/32 | 35/26/28/29 | 38/31/30/29 |
| aside: chosen + subject bonus x8 | 41/35/35/32 | 37/30/29/30 | 41/35/33/32 |

Bold English cells are below the English baseline.

**English check.** The chosen +1+2+3 is at or above the English baseline at every size: +4, +2, +1 and +3. Of all 47 variants, 27 pass this check. The ones that fail are mostly the stems taken over every word and the character 4-grams.

## Cost at 1M

**Lookup time.** The median is over 3 x 56 lookups, measured in microseconds, and includes tokenizing.
- en and fi ran in two processes at the same time; de ran alone. So compare variants within a language, not across languages.

| | baseline | +1 | +1+2 p4L | +1+3 | +1+2+3 p4L |
| --- | ---: | ---: | ---: | ---: | ---: |
| en | 63,273 | 61,854 | 64,249 | 64,374 | 65,469 |
| fi | 53,157 | 55,835 | 55,734 | 56,388 | 55,539 |
| de | 33,353 | 31,620 | 42,812 | 35,499 | 43,554 |

Stems add 2 to 4% in en and fi, but about 30% in de, where de ran alone. The Python numbers are dominated by posting-set loops, so they show how much each variant adds, not what the Zig time would be.

**Index size at 1M.** Size is counted as postings (note ids in posting lists). The percentages are the stem index's postings as a share of the token n-gram index.

| | token n-gram keys / postings | p4L keys / postings | p4 (all words) | c4 |
| --- | --- | --- | --- | --- |
| en | 27,091 / 35.9M | 166 / 3.68M (+10.3%) | +20.5% | +43.5% |
| fi | 18,820 / 56.4M | 280 / 4.73M (+8.4%) | +12.6% | +44.9% |
| de | 55,670 / 45.4M | 205 / 4.77M (+10.5%) | +20.9% | +48.6% |

Building p4L from the 1M store takes about 3 s in Python. The peak RSS of a whole run, holding all seven stem indexes, was 12.9 GiB for en, 18.0 GiB for fi and 15.9 GiB for de.

## Chosen design, for the Zig port
1. **Language words (+1).** Use a PAST_CUE per language (`templates_{lang}.PAST_CUE`). Add subject phrases from each entity's surfaces in that language: `SET_SURFACES`, and `surfaces()` for distractors.
2. **Stem key (+2).**
   - Split the note text into runs of letters and digits, lower case.
   - For each word longer than 4 characters, add the key (stem, its first 4 characters).
   - Keep these keys in a separate index, current and history apart like the token index.
   - Score = token n-gram score + 0.5 x stem score, with the same IDF formula over the same views.
3. **Aliases and "me" (+3).**
   - A current name note's value is an alias of its entity.
   - A question word w hits alias a when w == a, or when w starts with a and is at most 2 characters longer.
   - Each entity that holds the name gets the subject bonus x idf(k)/idf(1), where k is the number of entities that hold that name now.
   - Strip the matched subjects' surfaces out of the question. If a first-person word of the language remains, "me" is a subject and every current "me" note gets the subject bonus.
   - The first-person words are in `FIRST_PERSON` in `stem_lookup.py`: en i/me/my/mine/myself; de ich/mich/mir/mein*; fi first-person singular verbs plus words ending in -ni.
4. **What was not taken (all MEASURED):**
   - K=5 and K=6 were no better than K=4.
   - Stemming every word, rather than only words longer than K, cost English 1 at 100k.
   - Character 4-grams cost English up to 2.
   - me=rel50 was the same as me=all.
   - alias=full was about equal to alias=idf at 10k-100k, so I kept idf, which is safer as k grows.

## Findings
- Inside the brief, most of the gain comes from (1), the language words: de goes from 11 to 21 at 1M. (3) adds a few more: en goes from 23 to 26 at 1M. Stems (2) add 0 to 2.
- The largest lever is outside the brief: a heavier subject bonus. At 1M it gives en 32, fi 29 to 30 and de 29 to 32.
- The cause, READ from the score dumps at 10k: needed notes lose to near-copy distractors that share more words with the question than one subject bonus is worth. Examples are fi Q12 and Q45, and en Q38. This is change 3 of CAP-AND-MISSES and needs its own check.

## Open problems
- **Finnish consonant gradation** (Otto -> Oton, Turku -> Turussa) defeats both the prefix stem and the alias rule. In Q30 the name note shares no key at all with the question.
- **Names are not unique in the store.** At 10k, 79 to 99 entities hold each name (Ana, Bruno ...), so an alias cannot pick out the user's sister. The idf weighting only damps this.
- **The Finnish first person is a word list**, so it is per-language, not language-agnostic. Language detection is also open: each store here used its own language's tables.
- **The subject-bonus scale** is the next thing to test. It can crowd compound questions, which had no separate check here.
