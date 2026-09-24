"""Pairwise features for (Source 1, Source 2/3) candidate pairs.

Two tiers, matching the two-stage matcher:

* cheap_features   : search-space similarities, exact TF-IDF cosines, ranks /
                     gaps within each S1's candidate list, reverse-best gaps.
                     Pure numpy/pandas; used by the stage-1 pruning model.
* string_features  : fuzzy string metrics on names and addresses, address
                     number / postcode agreement, acronyms. Uses rapidfuzz
                     (vectorised, multi-core) when installed, otherwise a
                     multiprocessing pool over the pure-Python versions in sims.py.
                     Only computed for pairs that survive stage-1 pruning.

Train and test must be featurised with the same string backend (they are,
since `pipeline.py full` does both in one run); the backend is logged.
"""
import os
import numpy as np
import pandas as pd

from sims import (jaro_winkler, seq_ratio, token_sort_ratio, token_set_ratio,
                  jaccard, overlap_coef, soft_token_match)
from blocking import exact_view_sims

try:
    if os.environ.get("BER_NO_RAPIDFUZZ"):
        raise ImportError
    from rapidfuzz import process as rf_process, fuzz as rf_fuzz
    from rapidfuzz.distance import JaroWinkler as rf_jw
    HAVE_RF = hasattr(rf_process, "cpdist")
except Exception:
    HAVE_RF = False

STRING_BACKEND = "rapidfuzz" if HAVE_RF else "python"


# ----------------------------------------------------------------- cheap tier
def cheap_features(pairs, s1, s23, exact_mats, colstats):
    """Stage-1 features. `pairs` must have i1, i2 and s_<view> search-sim columns."""
    X = pairs.copy()
    i1, i2 = X["i1"].values, X["i2"].values
    for v, arr in exact_view_sims(i1, i2, exact_mats).items():
        X[f"cos_{v}"] = arr
    X["country_eq"] = (s1["country_n"].values[i1] == s23["country_n"].values[i2]).astype(np.float32)
    X["src3"] = (s23["src"].values[i2] == 3).astype(np.float32)
    la = s1["name_core"].str.len().values[i1]
    lb = s23["name_core"].str.len().values[i2]
    X["len_ratio"] = (np.minimum(la, lb) / np.maximum(np.maximum(la, lb), 1)).astype(np.float32)
    X["addr_b_len"] = s23["addr_n"].str.len().values[i2].astype(np.float32)
    X["addr_a_len"] = s1["addr_n"].str.len().values[i1].astype(np.float32)
    g = X.groupby("i1")
    X["n_cand"] = g["i2"].transform("size").astype(np.float32)
    simcols = [c for c in X.columns if c.startswith("s_") or c.startswith("cos_")]
    for c in simcols:
        X[f"rank_{c}"] = g[c].rank(ascending=False, method="min").astype(np.float32)
        X[f"gap_{c}"] = (X[c] - g[c].transform("max")).astype(np.float32)
    for v in colstats["best"]:
        c = f"s_{v}"
        if c in X:
            best = colstats["best"][v][i2]
            X[f"rgap_{v}"] = (X[c] - best).astype(np.float32)
            X[f"rmargin_{v}"] = (best - colstats["second"][v][i2]).astype(np.float32)
    X["combo"] = (X["cos_nm_char"] + X["cos_full_char"] + 0.5 * X["cos_ad_word"]).astype(np.float32)
    X["rank_combo"] = g["combo"].rank(ascending=False, method="min").astype(np.float32)
    X["gap_combo"] = (X["combo"] - g["combo"].transform("max")).astype(np.float32)
    X["_s"] = X["src3"]
    X["rank_combo_src"] = X.groupby(["i1", "_s"])["combo"].rank(ascending=False, method="min").astype(np.float32)
    return X.drop(columns="_s")


# ---------------------------------------------------------------- string tier
SET_FEATS = ["name_eq", "core_eq", "core_soft", "core_jac", "core_ovl", "first_eq", "acronym",
             "addr_jac", "addr_ovl", "addr_soft", "num_jac", "num_ovl", "pcode", "num_conf"]
FUZZ_FEATS = ["core_jw", "core_seq", "core_tsort", "core_tset", "first_jw", "addr_tset", "addr_tsort"]


def _set_feats(a_name, b_name, a_core, b_core, a_acr, b_acr, a_addr, b_addr, a_nums, b_nums, a_pc, b_pc):
    """Token/set-based features for one pair (always pure Python)."""
    at, bt = a_core.split(), b_core.split()
    sa, sb = set(at), set(bt)
    aat, bat = a_addr.split(), b_addr.split()
    has_addr = bool(a_addr and b_addr)
    acr = float((len(at) == 1 and len(bt) > 1 and at[0] == b_acr) or
                (len(bt) == 1 and len(at) > 1 and bt[0] == a_acr))
    pc = (1.0 if a_pc == b_pc else -1.0) if (a_pc and b_pc) else 0.0
    na, nb = a_nums - {a_pc}, b_nums - {b_pc}
    num_conf = (1.0 if na & nb else -1.0) if (na and nb) else 0.0
    return (
        float(a_name == b_name), float(a_core == b_core), soft_token_match(at, bt),
        jaccard(sa, sb), overlap_coef(sa, sb),
        float(bool(at) and bool(bt) and at[0] == bt[0]), acr,
        jaccard(set(aat), set(bat)) if has_addr else -1.0,
        overlap_coef(set(aat), set(bat)) if has_addr else -1.0,
        soft_token_match(aat, bat) if has_addr else -1.0,
        jaccard(a_nums, b_nums) if (a_nums and b_nums) else -1.0,
        overlap_coef(a_nums, b_nums) if (a_nums and b_nums) else -1.0,
        pc, num_conf,
    )


def _fuzz_py(a_core, b_core, a_addr, b_addr):
    """Pure-Python fuzzy metrics for one pair (fallback when rapidfuzz is absent)."""
    fa = a_core.split()[0] if a_core else ""
    fb = b_core.split()[0] if b_core else ""
    has_addr = bool(a_addr and b_addr)
    return (jaro_winkler(a_core, b_core), seq_ratio(a_core, b_core), token_sort_ratio(a_core, b_core),
            token_set_ratio(a_core, b_core), jaro_winkler(fa, fb),
            token_set_ratio(a_addr, b_addr) if has_addr else -1.0,
            token_sort_ratio(a_addr, b_addr) if has_addr else -1.0)


def _chunk_worker(args):
    """Multiprocessing worker: compute set (+ optionally fuzzy) features for a chunk."""
    cols, do_fuzz = args
    out = [_set_feats(*r) for r in zip(*cols)]
    if do_fuzz:
        fz = [_fuzz_py(c, d, e, f) for c, d, e, f in zip(cols[2], cols[3], cols[6], cols[7])]
        out = [a + b for a, b in zip(out, fz)]
    return out


def _fuzz_rf(A, B):
    """Vectorised rapidfuzz metrics for all pairs at once (multi-core)."""
    kw = dict(processor=None, workers=-1, dtype=np.float32)
    ac, bc = A["name_core"].tolist(), B["name_core"].tolist()
    aa, ba = A["addr_n"].tolist(), B["addr_n"].tolist()
    fa = [s.split()[0] if s else "" for s in ac]
    fb = [s.split()[0] if s else "" for s in bc]
    no_addr = (A["addr_n"].values == "") | (B["addr_n"].values == "")
    out = {
        "core_jw": rf_process.cpdist(ac, bc, scorer=rf_jw.normalized_similarity, **kw),
        "core_seq": rf_process.cpdist(ac, bc, scorer=rf_fuzz.ratio, **kw) / 100,
        "core_tsort": rf_process.cpdist(ac, bc, scorer=rf_fuzz.token_sort_ratio, **kw) / 100,
        "core_tset": rf_process.cpdist(ac, bc, scorer=rf_fuzz.token_set_ratio, **kw) / 100,
        "first_jw": rf_process.cpdist(fa, fb, scorer=rf_jw.normalized_similarity, **kw),
        "addr_tset": rf_process.cpdist(aa, ba, scorer=rf_fuzz.token_set_ratio, **kw) / 100,
        "addr_tsort": rf_process.cpdist(aa, ba, scorer=rf_fuzz.token_sort_ratio, **kw) / 100,
    }
    out["addr_tset"][no_addr] = -1
    out["addr_tsort"][no_addr] = -1
    return out


def string_features(X, s1, s23, n_jobs=None, chunk=20_000):
    """Add string-tier features to X (in place copy) and return it."""
    import multiprocessing as mp
    i1, i2 = X["i1"].values, X["i2"].values
    A = s1.iloc[i1].reset_index(drop=True)
    B = s23.iloc[i2].reset_index(drop=True)
    names = ["name_n", "name_n", "name_core", "name_core", "name_acr", "name_acr",
             "addr_n", "addr_n", "addr_nums", "addr_nums", "pcode", "pcode"]
    src = [A, B] * 6
    cols_all = [df[c].values for df, c in zip(src, names)]
    do_fuzz = not HAVE_RF
    tasks = [([c[st:st + chunk] for c in cols_all], do_fuzz) for st in range(0, len(X), chunk)]
    n_jobs = n_jobs or max(1, (os.cpu_count() or 2) - 1)
    if n_jobs > 1 and len(tasks) > 1:
        with mp.get_context("spawn").Pool(n_jobs) as pool:
            parts = pool.map(_chunk_worker, tasks)
    else:
        parts = [_chunk_worker(t) for t in tasks]
    rows = [r for p in parts for r in p]
    cols = SET_FEATS + (FUZZ_FEATS if do_fuzz else [])
    F = pd.DataFrame(rows, columns=cols, dtype=np.float32)
    if HAVE_RF:
        for k, v in _fuzz_rf(A, B).items():
            F[k] = v
    F.index = X.index
    X = pd.concat([X, F], axis=1)
    g = X.groupby("i1")
    X["rank_core_tset"] = g["core_tset"].rank(ascending=False, method="min").astype(np.float32)
    X["gap_core_jw"] = (X["core_jw"] - g["core_jw"].transform("max")).astype(np.float32)
    return X


def feature_columns(X):
    """Model input columns (everything except pair indices / stage-1 score)."""
    return [c for c in X.columns if c not in ("i1", "i2", "p1")]
