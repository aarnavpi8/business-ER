"""Text normalisation for business names and addresses.

Design rule: no rule is conditioned on the country label (the test set has an
unseen country, France). Every dictionary below is applied to every record.

Observed noise handled here (from the training data):
  * Indic-script names (romanised via translit.py)
  * spurious accents on English words ("Léarning"), casing, punctuation
  * junk decoration: "-- ", "<< ", "[Center]", "###", "| www.site.com"
  * website-style names ("wilfordhancock.com")
  * legal suffixes anywhere in the name ("LLC Moncada ...", "Pvt. EFS ... Ltd.")
  * address abbreviations (St/Street, R./Rue, AV/Avenue ...), full vs abbreviated
    US state names, "CDP"/"City" suffixes, ordinals ("1st", "first"),
    house-number noise ("01612", "14516d", "22459.")
"""
import re
import unicodedata
from functools import lru_cache

from translit import romanize, has_indic

# ----------------------------------------------------------------- vocab
LEGAL_SUFFIXES = {
    # English / US / India
    "inc", "incorporated", "corp", "corporation", "co", "company", "companies",
    "llc", "llp", "lp", "ltd", "limited", "plc", "pvt", "private", "pte", "pllc", "pc",
    "the", "opc", "l", "c", "p", "pa",
    # French
    "sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "scop", "scm", "selarl", "earl",
    "gie", "cie", "ste", "societe", "ets", "etablissements",
    # other European forms
    "gmbh", "ag", "kg", "bv", "nv", "srl", "spa", "sl", "oy", "ab",
    # honorific / "M/s" prefixes common in Indian records
    "mr", "mrs", "ms", "messrs", "shri_",
}
# long legal-form words: a token within typo distance of one of these is also dropped
# (the data corrupts suffixes: "privoate", "prihate", "limitd", "corporatoin")
_LONG_SUFFIXES = ("private", "limited", "company", "corporation", "incorporated", "enterprises_",
                  "societe", "etablissements")

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms",
    "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv", "new hampshire": "nh",
    "new jersey": "nj", "new mexico": "nm", "new york": "ny", "north carolina": "nc",
    "north dakota": "nd", "ohio": "oh", "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa",
    "rhode island": "ri", "south carolina": "sc", "south dakota": "sd", "tennessee": "tn",
    "texas": "tx", "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa",
    "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy", "district of columbia": "dc",
}
# multi-word state names must be replaced before tokenising
_STATE_RE = re.compile(r"\b(" + "|".join(sorted((k for k in US_STATES if " " in k), key=len, reverse=True)) + r")\b")

ADDR_ABBREV = {
    "street": "st", "str": "st", "road": "rd", "avenue": "av", "ave": "av", "avn": "av",
    "boulevard": "bd", "blvd": "bd", "bld": "bd", "boul": "bd", "drive": "dr", "lane": "ln",
    "court": "ct", "place": "pl", "pce": "pl", "parkway": "pkwy", "highway": "hwy",
    "square": "sq", "terrace": "ter", "circle": "cir", "trail": "trl", "freeway": "fwy",
    "expressway": "expy", "route": "rte", "rt": "rte", "suite": "ste", "apartment": "apt",
    "floor": "fl", "flr": "fl", "building": "bldg", "north": "n", "south": "s", "east": "e",
    "west": "w", "northeast": "ne", "northwest": "nw", "southeast": "se", "southwest": "sw",
    "mount": "mt", "fort": "ft", "junction": "jct", "jn": "jct", "opposite": "opp",
    "near": "nr", "market": "mkt", "nagar": "ngr", "colony": "clny", "sector": "sec",
    "sect": "sec", "phase": "ph", "chowk": "chk", "number": "no", "num": "no",
    "saint": "st", "sainte": "ste", "post": "po",
    # French street types -> short forms
    "rue": "r", "chemin": "ch", "che": "ch", "impasse": "imp", "allee": "all",
    "faubourg": "fbg", "fg": "fbg", "cours": "crs", "quai": "qu", "qua": "qu",
    "route": "rte",
}
ADDR_DROP = {"cdp", "city", "of", "the", "de", "du", "des", "la", "le", "les", "d", "l",
             "and", "cedex", "unit", "urban"}
ORDINAL_WORDS = {"first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5",
                 "sixth": "6", "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10"}

NAME_ABBREV = {
    "intl": "international", "mfg": "manufacturing", "svcs": "services", "svc": "service",
    "srvs": "services", "techs": "technologies", "sys": "systems", "mgmt": "management",
    "assoc": "associates", "assocs": "associates", "bros": "brothers", "dept": "department",
    "natl": "national", "univ": "university", "hosp": "hospital", "ctr": "center",
    "centre": "center", "cntr": "center", "ind": "industries", "inds": "industries",
    "st": "saint", "ste": "sainte", "shri": "shree", "sri": "shree",
}

# ----------------------------------------------------------------- regexes
_WS = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^0-9a-z ]+")
_WEB = re.compile(r"\bwww\.|\.(?:com|net|org|in|co|fr|biz|info|us)\b")
_ORD = re.compile(r"\b(\d+)(?:st|nd|rd|th|er|eme|e|re)\b")
_NUM_SUFFIX = re.compile(r"\b0*(\d+)[a-z]\b")      # 14516d -> 14516 ; 12b stays 12
_LEAD0 = re.compile(r"\b0+(\d)")
_DIGITS = re.compile(r"\d+")


def strip_accents(s: str) -> str:
    """Remove diacritics (é -> e); keeps base letters."""
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def basic_clean(s) -> str:
    """Romanise Indic script, strip accents, lowercase, drop web decoration and punctuation."""
    if s is None or (isinstance(s, float) and s != s):
        return ""
    s = str(s)
    if has_indic(s):
        s = romanize(s)
    s = strip_accents(s).lower()
    s = _WEB.sub(" ", s)
    s = s.replace("&", " and ").replace("+", " and ").replace("@", " at ")
    s = s.replace("'", "").replace("`", "").replace("’", "")
    s = _NON_ALNUM.sub(" ", s)
    return _WS.sub(" ", s).strip()


def normalize_name(s) -> str:
    """Canonical business name: cleaned, abbreviations expanded, duplicate tokens removed."""
    out, seen = [], set()
    c = basic_clean(s)
    if c.startswith("m s "):          # "M/s Tumkur Trading" (Indian "Messrs")
        c = c[4:]
    for t in c.split():
        t = NAME_ABBREV.get(t, t)
        if t not in seen:
            seen.add(t)
            out.append(t)
    return " ".join(out)


@lru_cache(maxsize=200_000)
def _is_fuzzy_suffix(t: str) -> bool:
    """True if token t looks like a typo of a long legal-form word (JW >= 0.88)."""
    if len(t) < 5:
        return False
    from sims import jaro_winkler
    return any(abs(len(t) - len(w)) <= 2 and jaro_winkler(t, w) >= 0.88 for w in _LONG_SUFFIXES)


def core_name(norm_name: str) -> str:
    """Name without legal-form tokens (exact or typo'd) and 'and'; falls back to the full name."""
    toks = [t for t in norm_name.split()
            if t not in LEGAL_SUFFIXES and t != "and" and not _is_fuzzy_suffix(t)]
    return " ".join(toks) if toks else norm_name


def name_acronym(core: str) -> str:
    """First letters of core tokens ('sarah marketing' -> 'sm')."""
    return "".join(t[0] for t in core.split() if t)


def normalize_address(s) -> str:
    """Canonical address token string (order preserved, filler dropped)."""
    s = basic_clean(s)
    if not s:
        return ""
    s = _STATE_RE.sub(lambda m: US_STATES[m.group(1)], s)
    s = _ORD.sub(r"\1", s)
    s = _NUM_SUFFIX.sub(r"\1", s)
    s = _LEAD0.sub(r"\1", s)
    toks = []
    for t in s.split():
        t = ORDINAL_WORDS.get(t, t)
        t = US_STATES.get(t, t)
        t = ADDR_ABBREV.get(t, t)
        if t and t not in ADDR_DROP:
            toks.append(t)
    return " ".join(toks)


def address_numbers(norm_addr: str) -> frozenset:
    """All digit groups in the normalised address (house, unit, PO box, postcode)."""
    return frozenset(_DIGITS.findall(norm_addr))


def first_number(norm_addr: str) -> str:
    """First digit group (usually the house number); '' if none."""
    m = _DIGITS.search(norm_addr)
    return m.group(0) if m else ""


def address_words(norm_addr: str) -> str:
    """Address with digits removed (street / locality words only)."""
    return _WS.sub(" ", _DIGITS.sub(" ", norm_addr)).strip()
