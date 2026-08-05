#!/usr/bin/env python3
"""Probe the corpus for likely ASR mishearings of Sanskrit/Tibetan/Buddhist terms.

For each candidate pattern: count matches across all chunk text and pull a few
real contexts, so a human can confirm it is a genuine mishearing (not a real
word) BEFORE any correction is written. Read-only. Also dumps random passages for
open-ended discovery of things not on the list.

This is the disciplined front of the correction methodology: never blind
find-and-replace. Confirm in context, then write a guarded rule.
"""
import os, re, sqlite3, random

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "..", "index", "data", "corpus.db")

# (label, regex, suspected-correct). These are *candidates*, not confirmed; the
# contexts decide. Many overlap with real English words on purpose, so a human
# can see whether they are really mishearings in context.
CANDIDATES = [
    ("body chitta",      r"\bbody\s+chitta\b",                  "bodhicitta"),
    ("bodhi chitta",     r"\bbodhi\s+chitta\b",                 "bodhicitta"),
    ("body sattva",      r"\bbody\s+sattva\b",                  "bodhisattva"),
    ("boddhisattva",     r"\bboddhi?sattva\b",                  "bodhisattva"),
    ("some sara",        r"\bsome\s+sara\b",                    "samsara"),
    ("sam sara",         r"\bsam\s+sara\b",                     "samsara"),
    ("sanga",            r"\bsanga\b",                          "sangha"),
    ("vajra sattva",     r"\bvajra\s+sattva\b",                 "Vajrasattva"),
    ("maha mudra",       r"\bmaha\s+mudra\b",                   "Mahamudra"),
    ("prajna paramita",  r"\bprajna\s+paramita\b",              "Prajnaparamita"),
    ("man jushri",       r"\bman\s*jushr?i\b|\bmanjushree\b",   "Manjushri"),
    ("padma sambhava",   r"\bpadma\s+sambhava\b",               "Padmasambhava"),
    ("chen re zig",      r"\bchen\s*re\s*zig\b|\bchenrezi\b",   "Chenrezig"),
    ("dharma kaya",      r"\bdharma\s+kaya\b",                  "dharmakaya"),
    ("sambhoga kaya",    r"\bsam?bhoga\s+kaya\b",               "sambhogakaya"),
    ("nirmana kaya",     r"\bnirmana\s+kaya\b",                 "nirmanakaya"),
    ("bar do",           r"\bbar\s+do\b",                       "bardo"),
    ("rig pa",           r"\brig\s+pa\b",                       "rigpa"),
    ("terra (Tara?)",    r"\bterra\b",                          "Tara?"),
    ("two more (tummo?)",r"\btwo\s+more\b",                     "tummo?"),
    ("trauma (torma?)",  r"\btrauma\b",                         "torma?"),
    ("non dro",          r"\bnon\s+dro\b|\bnyondro\b|\bnondro\b","ngondro"),
    ("man dala",         r"\bman\s+dala\b",                     "mandala"),
    ("kunda lini",       r"\bkunda\s+lini\b",                   "kundalini"),
    ("vajra yogini",     r"\bvajra\s+yogini\b",                 "Vajrayogini"),
    ("yeah she (Yeshe?)",r"\byeah\s+she\b",                     "Yeshe?"),
    ("the kini",         r"\bthe\s+kini\b",                     "dakini?"),
    ("lam rim",          r"\blam\s+rim\b",                      "Lamrim"),
    ("dza / dzogchen",   r"\bzogchen\b",                        "Dzogchen"),
    ("maha siddha",      r"\bmaha\s+siddha\b",                  "mahasiddha"),
    ("sadhana",          r"\bsad[dh]ana\b|\bsatana\b",          "sadhana"),
    ("samaya",           r"\bsama\s+ya\b|\bsomeya\b",           "samaya"),
]


def main():
    db = sqlite3.connect(DB)
    texts = [t for (t,) in db.execute("select text from chunks").fetchall()]
    db.close()
    blob = "\n".join(texts)
    print("== corpus: %d chunks ==\n" % len(texts))
    hits = []
    for label, pat, correct in CANDIDATES:
        ms = list(re.compile(pat, re.I).finditer(blob))
        if ms:
            hits.append((len(ms), label, correct, ms))
    hits.sort(reverse=True)
    for n, label, correct, ms in hits:
        print("### %-22s x%-5d  -> %s" % (label, n, correct))
        seen, shown = set(), 0
        for m in ms:
            a, b = max(0, m.start() - 34), min(len(blob), m.end() + 34)
            ctx = " ".join(blob[a:b].replace("\n", " ").split())
            if ctx[:40] in seen:
                continue
            seen.add(ctx[:40])
            print("     ...%s..." % ctx)
            shown += 1
            if shown >= 3:
                break
        print()

    print("\n== 22 random passages (open-ended discovery) ==\n")
    for t in random.Random(7).sample(texts, min(22, len(texts))):
        print("  -", " ".join(t.split())[:260], "\n")


if __name__ == "__main__":
    main()
