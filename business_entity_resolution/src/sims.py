"""Pure-Python string similarity functions (no external deps).

rapidfuzz / jellyfish are not available in our environments, so these are
small, dependency-free implementations. They are called once per candidate
pair, which is a few hundred thousand calls at most.
"""
from difflib import SequenceMatcher


def jaro_winkler(s1: str, s2: str, p: float = 0.1) -> float:
    """Jaro-Winkler similarity in [0, 1]."""
    if s1 == s2:
        return 1.0 if s1 else 0.0
    l1, l2 = len(s1), len(s2)
    if not l1 or not l2:
        return 0.0
    match_dist = max(l1, l2) // 2 - 1
    if match_dist < 0:
        match_dist = 0
    m1 = [False] * l1
    m2 = [False] * l2
    matches = 0
    for i in range(l1):
        lo = max(0, i - match_dist)
        hi = min(i + match_dist + 1, l2)
        c = s1[i]
        for j in range(lo, hi):
            if not m2[j] and s2[j] == c:
                m1[i] = m2[j] = True
                matches += 1
                break
    if not matches:
        return 0.0
    t = 0
    k = 0
    for i in range(l1):
        if m1[i]:
            while not m2[k]:
                k += 1
            if s1[i] != s2[k]:
                t += 1
            k += 1
    t /= 2
    jaro = (matches / l1 + matches / l2 + (matches - t) / matches) / 3
    prefix = 0
    for a, b in zip(s1[:4], s2[:4]):
        if a == b:
            prefix += 1
        else:
            break
    return jaro + prefix * p * (1 - jaro)


def seq_ratio(a: str, b: str) -> float:
    """difflib ratio (similar to normalised Levenshtein similarity)."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def token_sort_ratio(a: str, b: str) -> float:
    """seq_ratio after sorting tokens: robust to word-order transpositions."""
    return seq_ratio(" ".join(sorted(a.split())), " ".join(sorted(b.split())))


def token_set_ratio(a: str, b: str) -> float:
    """fuzzywuzzy-style token-set ratio: max over (intersection vs each side)."""
    ta, tb = set(a.split()), set(b.split())
    if not ta or not tb:
        return 0.0
    inter = " ".join(sorted(ta & tb))
    da = " ".join(sorted(ta - tb))
    db = " ".join(sorted(tb - ta))
    s1 = (inter + " " + da).strip()
    s2 = (inter + " " + db).strip()
    scores = [seq_ratio(s1, s2)]
    if inter:
        scores += [seq_ratio(inter, s1), seq_ratio(inter, s2)]
    return max(scores)


def jaccard(a: set, b: set) -> float:
    """Jaccard similarity of two sets; 0 if both empty."""
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def overlap_coef(a: set, b: set) -> float:
    """|A∩B| / min(|A|,|B|): high when one side is a subset of the other."""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def soft_token_match(a_toks, b_toks, thr: float = 0.88) -> float:
    """Fraction of tokens in the shorter list that have a JW>=thr partner (typo-tolerant)."""
    if not a_toks or not b_toks:
        return 0.0
    if len(a_toks) > len(b_toks):
        a_toks, b_toks = b_toks, a_toks
    hit = 0
    for t in a_toks:
        best = 0.0
        for u in b_toks:
            if t == u:
                best = 1.0
                break
            if abs(len(t) - len(u)) <= 3:
                v = jaro_winkler(t, u)
                if v > best:
                    best = v
        if best >= thr:
            hit += 1
    return hit / len(a_toks)
