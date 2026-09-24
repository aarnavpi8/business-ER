"""Text normalisation for business names and addresses.

Everything here is language-agnostic by design: the test set contains a
country (France) that never appears in training, so we avoid rules that only
make sense for US / India data and add multilingual legal-suffix and
street-type vocabularies instead.
"""
import re
import unicodedata

# Legal-form tokens that carry no identity information. Multilingual on purpose.
LEGAL_SUFFIXES = {
    # English / US / India
    "inc", "incorporated", "corp", "corporation", "co", "company", "companies",
    "llc", "llp", "lp", "ltd", "limited", "plc", "pvt", "private", "pte",
    "pllc", "pc", "na", "group", "holdings", "holding", "enterprises",
    "enterprise", "the", "opc", "intl", "international",
    # French
    "sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "scop", "scm", "selarl",
    "earl", "gie", "cie", "et", "fils", "societe", "ets", "etablissements",
    # German / other European forms that could plausibly appear
    "gmbh", "ag", "kg", "bv", "nv", "srl", "spa", "sl", "oy", "ab", "as",
}

# Abbreviation -> canonical token for addresses (applied token-wise).
ADDR_ABBREV = {
    "st": "street", "str": "street", "rd": "road", "ave": "avenue", "av": "avenue",
    "avn": "avenue", "blvd": "boulevard", "bd": "boulevard", "bld": "boulevard",
    "boul": "boulevard", "dr": "drive", "ln": "lane", "ct": "court",
    "pl": "place", "pkwy": "parkway", "hwy": "highway", "sq": "square",
    "ter": "terrace", "cir": "circle", "trl": "trail", "fwy": "freeway",
    "expy": "expressway", "sr": "state route", "rte": "route", "rt": "route",
    "ste": "suite", "apt": "apartment", "fl": "floor", "flr": "floor",
    "bldg": "building", "n": "north", "s": "south", "e": "east", "w": "west",
    "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest",
    "mt": "mount", "ft": "fort", "jn": "junction", "jct": "junction",
    "opp": "opposite", "nr": "near", "mkt": "market", "ngr": "nagar",
    "clny": "colony", "sec": "sector", "sect": "sector", "ph": "phase",
    "chk": "chowk", "mg": "mahatma gandhi",
    # French street types
    "r": "rue", "ch": "chemin", "che": "chemin", "rte.": "route", "imp": "impasse",
    "all": "allee", "pce": "place", "fbg": "faubourg", "fg": "faubourg",
    "crs": "cours", "qu": "quai", "qua": "quai", "sq.": "square",
    "zi": "zone industrielle", "za": "zone artisanale", "zac": "zone activite",
    "cedex": "",
}

# Name-level abbreviations (after lowercasing, before suffix removal).
NAME_ABBREV = {
    "&": " and ", "+": " and ", "intl": "international", "mfg": "manufacturing",
    "svcs": "services", "svc": "service", "srvs": "services", "tech": "technology",
    "techs": "technologies", "sys": "systems", "mgmt": "management",
    "assoc": "associates", "assocs": "associates", "bros": "brothers",
    "dept": "department", "natl": "national", "univ": "university",
    "hosp": "hospital", "ctr": "center", "centre": "center", "cntr": "center",
    "mkt": "market", "pharma": "pharmaceuticals", "ind": "industries",
    "inds": "industries", "st": "saint", "ste": "sainte",
}

_WS = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^0-9a-z ]+")
_DIGITS = re.compile(r"\d+")
_STE_SOCIETE = re.compile(r"(?i)(?<!\w)sté(?!\w)")


def strip_accents(s: str) -> str:
    """Remove diacritics (é -> e) so French / transliterated text compares cleanly."""
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def basic_clean(s) -> str:
    """Lowercase, strip accents, map '&' to 'and', drop punctuation, squeeze spaces."""
    if s is None or (isinstance(s, float) and s != s):
        return ""
    s = strip_accents(str(s)).lower()
    s = s.replace("&", " and ").replace("+", " and ").replace("@", " at ")
    s = s.replace("'", "").replace("`", "")  # o'reilly -> oreilly
    s = _NON_ALNUM.sub(" ", s)
    return _WS.sub(" ", s).strip()


def normalize_name(s) -> str:
    """Canonical business name with abbreviations expanded (legal suffixes kept)."""
    if isinstance(s, str):
        # "Sté" (société) and "Ste" (sainte) collide once accents are stripped
        s = _STE_SOCIETE.sub(" societe ", unicodedata.normalize("NFC", s))
    toks = basic_clean(s).split()
    out = []
    for t in toks:
        out.append(NAME_ABBREV.get(t, t))
    return _WS.sub(" ", " ".join(out)).strip()


def core_name(norm_name: str) -> str:
    """Name with legal-form tokens removed; falls back to the full name if empty."""
    toks = [t for t in norm_name.split() if t not in LEGAL_SUFFIXES and t != "and"]
    return " ".join(toks) if toks else norm_name


def name_acronym(core: str) -> str:
    """First letters of the core-name tokens (e.g. 'state bank of india' -> 'sboi')."""
    return "".join(t[0] for t in core.split() if t)


def normalize_address(s) -> str:
    """Canonical address: abbreviations expanded, ordinals ('1st') split to digits."""
    s = basic_clean(s)
    s = re.sub(r"\b(\d+)(st|nd|rd|th|er|eme|e)\b", r"\1", s)  # 21st -> 21, 3eme -> 3
    toks = []
    for t in s.split():
        rep = ADDR_ABBREV.get(t, t)
        if rep:
            toks.append(rep)
    return _WS.sub(" ", " ".join(toks)).strip()


def address_numbers(norm_addr: str) -> set:
    """All digit groups in the address (house numbers, postcodes, suite numbers)."""
    return set(_DIGITS.findall(norm_addr))


def postcode(raw_addr) -> str:
    """Best-effort postcode: a 5- or 6-digit group (US ZIP / French CP / Indian PIN).

    Also recognises Indian PINs written as '560 001'. Returns '' when absent.
    """
    if raw_addr is None or (isinstance(raw_addr, float) and raw_addr != raw_addr):
        return ""
    s = str(raw_addr)
    s = re.sub(r"\b(\d{3})\s(\d{3})\b", r"\1\2", s)
    m = re.findall(r"(?<!\d)(\d{5,6})(?!\d)", s)
    return m[-1] if m else ""
