"""Dependency-free romanisation of Indic (Brahmic) scripts.

About 23% of Indian Source-2 names (12% in Source 3) are written in native
scripts (Devanagari, Telugu, Kannada, Tamil, Bengali, Gujarati, Malayalam,
Oriya, Gurmukhi) while Source 1 is always Latin. We romanise them with a
single rule set derived from Unicode character *names*, e.g.
'DEVANAGARI LETTER KA' -> consonant 'k' + inherent 'a',
'TAMIL VOWEL SIGN AA'  -> replaces the inherent vowel with 'a'.
This covers every Brahmic block uniformly without per-script tables and
uses no external data. The output is approximate ("praivet limited"), which
the fuzzy/char-n-gram features tolerate; frequent loan-words are then mapped
to their English spelling.
"""
import re
import unicodedata
from functools import lru_cache

# consonant name -> latin (vowel-less); inherent 'a' is added separately
CONS = {
    "KA": "k", "KHA": "kh", "GA": "g", "GHA": "gh", "NGA": "ng", "CA": "ch", "CHA": "chh",
    "JA": "j", "JHA": "jh", "NYA": "ny", "TTA": "t", "TTHA": "th", "DDA": "d", "DDHA": "dh",
    "NNA": "n", "TA": "t", "THA": "th", "DA": "d", "DHA": "dh", "NA": "n", "NNNA": "n",
    "PA": "p", "PHA": "f", "BA": "b", "BHA": "bh", "MA": "m", "YA": "y", "YYA": "y",
    "RA": "r", "RRA": "r", "RRRA": "r", "LA": "l", "LLA": "l", "LLLA": "l", "VA": "v",
    "WA": "v", "SHA": "sh", "SSA": "sh", "SA": "s", "HA": "h", "QA": "q", "KHHA": "kh",
    "GHHA": "gh", "ZA": "z", "DDDHA": "r", "RHA": "r", "FA": "f", "YYYA": "y",
}
VOW = {
    "A": "a", "AA": "a", "I": "i", "II": "i", "U": "u", "UU": "u", "E": "e", "EE": "e",
    "AI": "ai", "O": "o", "OO": "o", "AU": "au", "VOCALIC R": "ri", "VOCALIC RR": "ri",
    "VOCALIC L": "li", "CANDRA E": "e", "CANDRA O": "o", "SHORT E": "e", "SHORT O": "o",
    "CANDRA A": "a", "OE": "o", "AAY": "ai",
}
SCRIPTS = ("DEVANAGARI", "BENGALI", "GURMUKHI", "GUJARATI", "ORIYA", "TAMIL", "TELUGU",
           "KANNADA", "MALAYALAM", "SINHALA")

# frequent English loan-words after romanisation -> English spelling
LOANWORDS = {
    "praivet": "private", "privet": "private", "praivat": "private", "prayvet": "private",
    "limited": "limited", "limitad": "limited", "limitd": "limited", "limitid": "limited",
    "elaelapi": "llp", "elelpi": "llp", "elalpi": "llp", "elelapi": "llp",
    "kampani": "company", "kanpani": "company", "kampni": "company",
    "prodakts": "products", "sarvises": "services", "sarvisej": "services", "sarvisij": "services",
    "tekanolaji": "technology", "teknolaji": "technology", "tek": "tech",
    "indastris": "industries", "indastriz": "industries", "intarneshanal": "international",
    "entarprisej": "enterprises", "entarprises": "enterprises", "enterprises": "enterprises",
    "marketing": "marketing", "maarketing": "marketing", "marketinga": "marketing",
    "menejment": "management", "menejament": "management", "solyushans": "solutions",
    "trading": "trading", "treding": "trading", "tredars": "traders", "tredars": "traders",
    "shri": "shree", "sri": "shree", "shree": "shree",
    # observed in train (romanised S2/S3 token vs S1 token)
    "pra": "private", "li": "limited", "limitet": "limited", "limirrad": "limited",
    "praibhet": "private", "piraivet": "private", "praivarr": "private", "pvt": "private",
    "estet": "estate", "faundeshan": "foundation", "enarji": "energy", "pavar": "power",
    "aiti": "it", "kansalting": "consulting", "infotek": "infotech", "venchars": "ventures",
    "fud": "food", "fuds": "foods", "impeks": "impex", "kansaltents": "consultants",
    "sistams": "systems", "devalapars": "developers", "kanstrakshan": "construction",
    "kanstrakshans": "constructions", "bildars": "builders", "indastrij": "industries",
    "investament": "investment", "investaments": "investments", "lojistiks": "logistics",
    "lajistiks": "logistics", "prodyusar": "producer", "hospitailiti": "hospitality",
    "projekts": "projects", "helthakeyar": "healthcare", "siti": "city", "mainejament": "management",
    "midiya": "media", "yunivarsal": "universal", "nyu": "new", "kansaltensi": "consultancy",
    "infrastrakchar": "infrastructure", "bijanes": "business", "propartij": "properties",
    "teknolojij": "technologies", "teknoloji": "technology", "ist": "east", "egro": "agro",
    "eksaports": "exports", "praim": "prime", "entarapraijej": "enterprises", "riyal": "real",
    "grin": "green", "keyar": "care", "dijital": "digital", "istarn": "eastern",
}


@lru_cache(maxsize=4096)
def _classify(ch):
    """('cons', latin) | ('vow', latin) | ('sign', latin) | ('virama', '') | ('mark', latin) | None."""
    try:
        name = unicodedata.name(ch)
    except ValueError:
        return None
    parts = name.split(" ", 1)
    if parts[0] not in SCRIPTS or len(parts) < 2:
        return None
    rest = parts[1]
    if rest.startswith("LETTER "):
        x = rest[7:]
        if x in CONS:
            return ("cons", CONS[x])
        if x in VOW:
            return ("vow", VOW[x])
        if x.startswith("CHILLU "):
            return ("vow", CONS.get(x[7:], x[7:].lower()[:1]))
        return ("cons", x.lower()[:2])
    if rest.startswith("VOWEL SIGN "):
        return ("sign", VOW.get(rest[11:], rest[11:].lower()[:1]))
    if rest in ("SIGN VIRAMA", "SIGN HALANT", "AU LENGTH MARK", "SIGN VIRAMA"):
        return ("virama", "")
    if rest in ("SIGN ANUSVARA", "SIGN CANDRABINDU", "SIGN ADDAK", "SIGN TIPPI", "SIGN BINDI"):
        return ("mark", "n")
    if rest == "SIGN VISARGA":
        return ("mark", "h")
    if rest.startswith("DIGIT "):
        return ("digit", str(unicodedata.digit(ch)))
    return ("mark", "")  # nukta, avagraha, accents: drop


def has_indic(s: str) -> bool:
    return any("ऀ" <= c <= "෿" for c in s)


def romanize(s: str) -> str:
    """Romanise Indic characters in s; non-Indic characters pass through unchanged."""
    if not s or not has_indic(s):
        return s
    s = s.replace("\u200c", "").replace("\u200d", "")
    out = []
    pending = False  # a consonant waiting for its vowel (inherent 'a' unless replaced)

    def flush(end_of_word=False):
        nonlocal pending
        if pending and not end_of_word:
            out.append("a")
        pending = False

    for ch in s:
        k = _classify(ch)
        if k is None:
            flush(end_of_word=True)  # schwa deletion at word end
            out.append(ch)
            continue
        kind, lat = k
        if kind == "cons":
            flush()
            out.append(lat)
            pending = True
        elif kind == "sign":
            out.append(lat)
            pending = False
        elif kind == "virama":
            pending = False
        elif kind == "vow":
            flush()
            out.append(lat)
        elif kind == "digit":
            flush(end_of_word=True)
            out.append(lat)
        else:  # mark
            flush()
            out.append(lat)
    flush(end_of_word=True)
    text = "".join(out)
    return " ".join(LOANWORDS.get(t, t) for t in re.split(r"\s+", text.lower()) if t)
