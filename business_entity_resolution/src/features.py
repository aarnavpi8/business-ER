"""Pairwise features for (Source 1, Source 2/3) candidate pairs, built for scale.

Pairs arrive as a dict of numpy arrays from blocking.block_split (sorted by i2).
Group statistics are computed with sorted segment reductions (no pandas
groupby), so tens of millions of pairs fit in laptop memory.

Tiers
  cheap  : search-space similarities per view, gaps / ranks inside each S2/S3
           record's candidate list (the S1 it should pick) and inside each S1's
           list, record-level best/second margins, lengths. Used by stage 1.
  string : fuzzy name/address metrics, house-number agreement, acronyms,
           de-spaced names. rapidfuzz (vectorised) when installed, else a
           multiprocessing pool over pure-Python sims. Only for stage-1 survivors.
  stack  : stage-1 probability and its rank / gap within the S2/S3 record's list.
"""
import os
import numpy as np
import pandas as pd

from normalize import name_acronym, address_numbers, first_number, address_words
from sims import (jaro_winkler, seq_ratio, token_sort_ratio, token_set_ratio,
                  jaccard, overlap_coef, soft_token_match)

try:
    if os.environ.get("BER_NO_RAPIDFUZZ"):
        raise ImportError
    from rapidfuzz import process as rf_process, fuzz as rf_fuzz
    from rapidfuzz.distance import JaroWinkler as rf_jw
    HAVE_RF = hasattr(rf_process, "cpdist")
except Exception:
    HAVE_RF = False
STRING_BACKEND = "rapidfuzz" if HAVE_RF else "python"

VIEWS = ("nm", "ad", "full")


# ------------------------------------------------------------ group helpers
def _segments(sorted_keys):
    """Start offsets of runs of equal values in a sorted key array."""
    if len(sorted_keys) == 0:
        return np.zeros(0, np.int64)
    return np.flatnonzero(np.r_[True, sorted_keys[1:] != sorted_keys[:-1]])


def group_max(keys_sorted, vals):
    """Per-row max of vals over its group (keys must be sorted)."""
    st = _segments(keys_sorted)
    m = np.maximum.reduceat(vals, st)
    return np.repeat(m, np.diff(np.r_[st, len(vals)]))


def group_size(keys_sorted):
    st = _segments(keys_sorted)
    sz = np.diff(np.r_[st, len(keys_sorted)])
    return np.repeat(sz, sz).astype(np.float32)


def group_rank_desc(keys_sorted, vals):
    """1-based descending rank of vals within each group (ties broken by order)."""
    o = np.lexsort((-vals, keys_sorted))
    st = _segments(keys_sorted[o])
    pos = np.arange(len(o)) - np.repeat(st, np.diff(np.r_[st, len(o)]))
    r = np.empty(len(o), np.float32)
    r[o] = pos + 1
    return r


def group_second(keys_sorted, vals):
    """Per-row: best value among the *other* rows of the group (-1 if alone)."""
    mx = group_max(keys_sorted, vals)
    o = np.lexsort((-vals, keys_sorted))
    ks = keys_sorted[o]
    st = _segments(ks)
    sz = np.diff(np.r_[st, len(o)])
    second_sorted = np.full(len(st), -1.0, np.float32)
    has2 = sz > 1
    second_sorted[has2] = vals[o][st[has2] + 1]
    second = np.repeat(second_sorted, sz)
    sec = np.empty(len(o), np.float32)
    sec[o] = second
    is_top = np.isclose(vals, mx)
    return np.where(is_top, sec, mx).astype(np.float32)


def by_i1(B, fn, vals):
    """Apply a group function over i1 groups (pairs are stored sorted by i2)."""
    o = np.argsort(B["i1"], kind="stable")
    res = fn(B["i1"][o], vals[o])
    out = np.empty_like(res)
    out[o] = res
    return out


# ------------------------------------------------------------ cheap tier
def i1_stats(B, n1):
    """Per-S1 summaries over *all* its candidate pairs (small arrays of length n1).

    Everything else is computed chunk by chunk (chunks never split an S2/S3
    record), so the full pair table never needs per-pair group-stat columns.
    """
    i1 = B["i1"]
    st = {"cnt": np.bincount(i1, minlength=n1).astype(np.float32)}
    combo = np.zeros(len(i1), np.float32)
    for v in VIEWS:
        m = np.full(n1, -1.0, np.float32)
        np.maximum.at(m, i1, B[f"s_{v}"])
        st[f"max_{v}"] = m
        combo += B[f"s_{v}"]
    m = np.full(n1, -1.0, np.float32)
    np.maximum.at(m, i1, combo)
    st["max_combo"] = m
    return st


def i2_chunks(B, step=2_000_000):
    """(start, end) row ranges of at most ~step rows that never split an S2/S3 record."""
    i2, n = B["i2"], len(B["i2"])
    cuts = [0]
    while cuts[-1] < n:
        e = min(n, cuts[-1] + step)
        if e < n:
            e = int(np.searchsorted(i2, i2[e - 1], side="right"))
        cuts.append(e)
    return list(zip(cuts[:-1], cuts[1:]))


def cheap_chunk(B, a, b, s1, s23, st1, rows=None):
    """Stage-1 feature DataFrame for pair rows a:b (optionally a subset `rows` of them)."""
    sl = np.arange(a, b)
    i1, i2 = B["i1"][sl], B["i2"][sl]
    sims = {v: B[f"s_{v}"][sl] for v in VIEWS}
    sims["combo"] = (sims["nm"] + sims["ad"] + sims["full"]).astype(np.float32)
    F = {}
    for v in VIEWS:
        x = sims[v]
        F[f"s_{v}"] = x
        F[f"rgap_{v}"] = x - B[f"best2_{v}"][i2]           # vs this S23's best S1 overall
        F[f"rmargin_{v}"] = B[f"best2_{v}"][i2] - B[f"second2_{v}"][i2]
        F[f"fgap_{v}"] = x - B[f"best1_{v}"][i1]           # vs this S1's best S23 overall
    F["combo"] = sims["combo"]
    F["n_c2"] = group_size(i2)
    F["n_c1"] = st1["cnt"][i1]
    for v in VIEWS + ("combo",):
        x = sims[v]
        F[f"g2gap_{v}"] = x - group_max(i2, x)            # inside the S23's candidate list
        F[f"g2rank_{v}"] = group_rank_desc(i2, x)
        F[f"g1gap_{v}"] = x - st1[f"max_{v}"][i1]          # inside the S1's candidate list
    F["g2margin_combo"] = sims["combo"] - group_second(i2, sims["combo"])
    F["src3"] = (s23["src"].values[i2] == 3).astype(np.float32)
    F["len_name1"] = s1["name_core"].str.len().values[i1]
    F["len_name2"] = s23["name_core"].str.len().values[i2]
    F["len_addr1"] = s1["addr_n"].str.len().values[i1]
    F["len_addr2"] = s23["addr_n"].str.len().values[i2]
    X = pd.DataFrame({k: np.asarray(v, np.float32) for k, v in F.items()})
    X.insert(0, "i2", i2)
    X.insert(0, "i1", i1)
    if rows is not None:
        X = X[rows].reset_index(drop=True)
    return X


# ------------------------------------------------------------ string tier
SET_FEATS = ["name_eq", "core_eq", "core_soft", "core_jac", "core_ovl", "first_eq", "acronym",
             "addr_jac", "addr_ovl", "addr_soft", "num_jac", "num_ovl", "num_conf", "firstnum_eq",
             "name_extra_a", "name_extra_b", "addr_extra_a", "addr_extra_b", "num_extra_a", "num_extra_b",
             "num_sub", "firstnum_sub", "firstnum_jw", "digits_eq", "digits_jw"]
FUZZ_FEATS = ["core_jw", "core_seq", "core_tsort", "core_tset", "first_jw", "nospace_jw",
              "addr_tset", "addr_tsort", "addrw_tset"]


def _set_feats(a_name, b_name, a_core, b_core, a_addr, b_addr):
    """Token/set-based features for one pair (pure Python)."""
    a_acr, b_acr = name_acronym(a_core), name_acronym(b_core)
    a_nums, b_nums = address_numbers(a_addr), address_numbers(b_addr)
    a_fn, b_fn = first_number(a_addr), first_number(b_addr)
    at, bt = a_core.split(), b_core.split()
    sa, sb = set(at), set(bt)
    aat, bat = a_addr.split(), b_addr.split()
    has_addr = bool(a_addr and b_addr)
    acr = float((len(at) == 1 and len(bt) > 1 and at[0] == b_acr) or
                (len(bt) == 1 and len(at) > 1 and bt[0] == a_acr))
    num_conf = (1.0 if a_nums & b_nums else -1.0) if (a_nums and b_nums) else 0.0
    fn = (1.0 if a_fn == b_fn else -1.0) if (a_fn and b_fn) else 0.0
    return (
        float(a_name == b_name), float(a_core == b_core), soft_token_match(at, bt),
        jaccard(sa, sb), overlap_coef(sa, sb),
        float(bool(at) and bool(bt) and at[0] == bt[0]), acr,
        jaccard(set(aat), set(bat)) if has_addr else -1.0,
        overlap_coef(set(aat), set(bat)) if has_addr else -1.0,
        soft_token_match(aat, bat) if has_addr else -1.0,
        jaccard(a_nums, b_nums) if (a_nums and b_nums) else -1.0,
        overlap_coef(a_nums, b_nums) if (a_nums and b_nums) else -1.0,
        num_conf, fn,
        # unmatched material on each side: decoys (unlinked look-alikes) tend to add/alter tokens
        float(len(sa - sb)), float(len(sb - sa)),
        float(len(set(aat) - set(bat))), float(len(set(bat) - set(aat))),
        float(len(a_nums - b_nums)), float(len(b_nums - a_nums)),
        # house numbers are corrupted by dropping / splitting digits ("709"->"70", "39"->"3 9")
        _num_sub(a_nums, b_nums), _sub(a_fn, b_fn),
        jaro_winkler(a_fn, b_fn) if (a_fn and b_fn) else -1.0,
        _digits_eq(a_addr, b_addr), _digits_jw(a_addr, b_addr),
    )


def _sub(a, b):
    """1 if one digit string is contained in the other, 0 if not, -1 if either is missing."""
    if not a or not b:
        return -1.0
    return 1.0 if (a in b or b in a) else 0.0


def _num_sub(an, bn):
    """1 if any number of one side is contained in a number of the other side."""
    if not an or not bn:
        return -1.0
    return 1.0 if any(x in y or y in x for x in an for y in bn) else 0.0


def _digits(addr):
    return "".join(ch for ch in addr if ch.isdigit())


def _digits_eq(a, b):
    da, db = _digits(a), _digits(b)
    return (1.0 if da == db else 0.0) if (da and db) else -1.0


def _digits_jw(a, b):
    da, db = _digits(a), _digits(b)
    return jaro_winkler(da, db) if (da and db) else -1.0


def _fuzz_py(a_core, b_core, a_addr, b_addr):
    """Pure-Python fuzzy metrics for one pair (fallback when rapidfuzz is absent)."""
    a_ns, b_ns = a_core.replace(" ", ""), b_core.replace(" ", "")
    a_aw, b_aw = address_words(a_addr), address_words(b_addr)
    fa = a_core.split()[0] if a_core else ""
    fb = b_core.split()[0] if b_core else ""
    has_addr = bool(a_addr and b_addr)
    return (jaro_winkler(a_core, b_core), seq_ratio(a_core, b_core), token_sort_ratio(a_core, b_core),
            token_set_ratio(a_core, b_core), jaro_winkler(fa, fb), jaro_winkler(a_ns, b_ns),
            token_set_ratio(a_addr, b_addr) if has_addr else -1.0,
            token_sort_ratio(a_addr, b_addr) if has_addr else -1.0,
            token_set_ratio(a_aw, b_aw) if (a_aw and b_aw) else -1.0)


def _chunk_worker(args):
    """Multiprocessing worker: set (+ optionally fuzzy) features for a chunk of pairs."""
    set_cols, fuzz_cols = args
    out = [_set_feats(*r) for r in zip(*set_cols)]
    if fuzz_cols is not None:
        out = [a + _fuzz_py(*r) for a, r in zip(out, zip(*fuzz_cols))]
    return out


def _fuzz_rf(A, B):
    """Vectorised rapidfuzz metrics for all pairs (multi-core)."""
    kw = dict(processor=None, workers=-1, dtype=np.float32)
    ac, bc = A["name_core"].tolist(), B["name_core"].tolist()
    fa = [s.split()[0] if s else "" for s in ac]
    fb = [s.split()[0] if s else "" for s in bc]
    aa, ba = A["addr_n"].tolist(), B["addr_n"].tolist()
    aw = [address_words(x) for x in aa]
    bw = [address_words(x) for x in ba]
    no_addr = (A["addr_n"].values == "") | (B["addr_n"].values == "")
    no_aw = np.array([not (x and y) for x, y in zip(aw, bw)])
    out = {
        "core_jw": rf_process.cpdist(ac, bc, scorer=rf_jw.normalized_similarity, **kw),
        "core_seq": rf_process.cpdist(ac, bc, scorer=rf_fuzz.ratio, **kw) / 100,
        "core_tsort": rf_process.cpdist(ac, bc, scorer=rf_fuzz.token_sort_ratio, **kw) / 100,
        "core_tset": rf_process.cpdist(ac, bc, scorer=rf_fuzz.token_set_ratio, **kw) / 100,
        "first_jw": rf_process.cpdist(fa, fb, scorer=rf_jw.normalized_similarity, **kw),
        "nospace_jw": rf_process.cpdist([x.replace(" ", "") for x in ac], [x.replace(" ", "") for x in bc],
                                        scorer=rf_jw.normalized_similarity, **kw),
        "addr_tset": rf_process.cpdist(aa, ba, scorer=rf_fuzz.token_set_ratio, **kw) / 100,
        "addr_tsort": rf_process.cpdist(aa, ba, scorer=rf_fuzz.token_sort_ratio, **kw) / 100,
        "addrw_tset": rf_process.cpdist(aw, bw, scorer=rf_fuzz.token_set_ratio, **kw) / 100,
    }
    out["addr_tset"][no_addr] = -1
    out["addr_tsort"][no_addr] = -1
    out["addrw_tset"][no_aw] = -1
    return out


def string_features(X, s1, s23, n_jobs=None, chunk=25_000):
    """Return X with string-tier features appended."""
    import multiprocessing as mp
    A = s1.iloc[X["i1"].values].reset_index(drop=True)
    B = s23.iloc[X["i2"].values].reset_index(drop=True)
    set_cols = []
    for c in ["name_n", "name_core", "addr_n"]:
        set_cols += [A[c].values, B[c].values]
    fz_cols = []
    for c in ["name_core", "addr_n"]:
        fz_cols += [A[c].values, B[c].values]
    do_fuzz = not HAVE_RF
    tasks = [([c[st:st + chunk] for c in set_cols],
              [c[st:st + chunk] for c in fz_cols] if do_fuzz else None)
             for st in range(0, len(X), chunk)]
    n_jobs = n_jobs or max(1, (os.cpu_count() or 2) - 1)
    if n_jobs > 1 and len(tasks) > 1:
        with mp.get_context("spawn").Pool(n_jobs) as pool:
            parts = pool.map(_chunk_worker, tasks, chunksize=1)
    else:
        parts = [_chunk_worker(t) for t in tasks]
    cols = SET_FEATS + (FUZZ_FEATS if do_fuzz else [])
    F = pd.DataFrame([r for p in parts for r in p], columns=cols, dtype=np.float32)
    if HAVE_RF:
        for k, v in _fuzz_rf(A, B).items():
            F[k] = v
    F.index = X.index
    return pd.concat([X, F], axis=1)


def stack_features(X, col="p1", g1max=None):
    """Stage-1 probability features relative to siblings (X must contain i1, i2, col).

    g1max: optional per-S1 max of col over the whole table, for when X is one chunk of it
    (chunks never split an S2/S3 record, so the per-S23 features need no such help).
    """
    i2 = X["i2"].values
    o = np.argsort(i2, kind="stable")
    p = X[col].values.astype(np.float32)
    k = i2[o]
    gmax = np.empty_like(p); gmax[o] = group_max(k, p[o])
    rank = np.empty_like(p); rank[o] = group_rank_desc(k, p[o])
    sec = np.empty_like(p); sec[o] = group_second(k, p[o])
    X[f"{col}_g2gap"] = p - gmax
    X[f"{col}_g2rank"] = rank
    X[f"{col}_g2margin"] = p - sec
    j = X["i1"].values
    if g1max is None:
        o = np.argsort(j, kind="stable")
        g1 = np.empty_like(p); g1[o] = group_max(j[o], p[o])
    else:
        g1 = g1max[j]
    X[f"{col}_g1gap"] = p - g1
    return X


def feature_columns(X, exclude=("i1", "i2", "y", "p", "fold")):
    """Model input columns."""
    return [c for c in X.columns if c not in exclude]
