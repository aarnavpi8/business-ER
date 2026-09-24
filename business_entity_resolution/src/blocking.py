"""Scalable candidate generation (blocking).

For every Source 1 record we retrieve nearest Source 2/3 neighbours under
several complementary "search views", and we also search in the reverse
direction (each S2/S3 record retrieves its nearest S1 records). The union of
all retrieved pairs is the candidate set.

Search views
  dense  nm    : char n-gram TF-IDF of the core name   -> SVD -> GPU/CPU exact top-k
  dense  full  : same for "core name | address"
  dense  emb   : optional sentence-transformer embedding of "name, address"
  sparse nmw   : word TF-IDF of the core name (very common tokens dropped)
  sparse adw   : word TF-IDF of the address   (very common tokens dropped)

The reverse search also gives, for every S2/S3 record, its best and
second-best similarity to *any* S1 record under each view. Because Source 1
is deduplicated, "is this S1 the S2 record's best match?" is a strong feature.

Exact (non-SVD) TF-IDF cosines for each pair are computed separately in
`exact_view_sims` and used as matcher features.
"""
import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from search import encode, topk, sparse_topk, backend_name

# view -> (text column, forward k, reverse k)
DENSE_VIEWS = {"nm": ("name_core", 20, 3), "full": ("full_n", 20, 3)}
SPARSE_VIEWS = {"nmw": ("name_core", 10, 2), "adw": ("addr_n", 8, 2)}
EMB_VIEW = ("emb_text", 20, 3)

EXACT_VIEWS = {
    "nm_char": ("name_core", dict(analyzer="char_wb", ngram_range=(2, 4))),
    "full_char": ("full_n", dict(analyzer="char_wb", ngram_range=(3, 4))),
    "nm_word": ("name_core", dict(analyzer="word", token_pattern=r"(?u)\b\w+\b")),
    "ad_word": ("addr_n", dict(analyzer="word", token_pattern=r"(?u)\b\w+\b")),
}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def fit_exact_views(s1, s23):
    """Sparse TF-IDF matrices (S1, S23) per exact view, fitted on the split's own text."""
    mats = {}
    for v, (col, kw) in EXACT_VIEWS.items():
        vec = TfidfVectorizer(dtype=np.float32, sublinear_tf=True, min_df=1, **kw)
        vec.fit(pd.concat([s1[col], s23[col]]).values)
        mats[v] = (vec.transform(s1[col].values).tocsr(), vec.transform(s23[col].values).tocsr())
    return mats


def exact_view_sims(i1, i2, mats):
    """Exact TF-IDF cosine for each pair under each exact view (row-wise dot, chunked)."""
    out = {}
    for v, (A, B) in mats.items():
        res = np.empty(len(i1), np.float32)
        for st in range(0, len(i1), 2_000_000):
            a = A[i1[st:st + 2_000_000]]
            b = B[i2[st:st + 2_000_000]]
            res[st:st + 2_000_000] = np.asarray(a.multiply(b).sum(axis=1)).ravel()
        out[v] = res
    return out


def _sparse_search_mats(s1, s23, col, max_df=0.01):
    vec = TfidfVectorizer(analyzer="word", token_pattern=r"(?u)\b\w+\b", sublinear_tf=True,
                          max_df=max_df, min_df=1, dtype=np.float32)
    vec.fit(pd.concat([s1[col], s23[col]]).values)
    return vec.transform(s1[col].values).tocsr(), vec.transform(s23[col].values).tocsr()


def _add(pairs_list, fwd_idx, n_rows, reverse=False):
    """Append (i1, i2) arrays from a top-k index matrix, skipping -1 slots."""
    rows = np.repeat(np.arange(n_rows), fwd_idx.shape[1])
    cols = fwd_idx.ravel()
    ok = cols >= 0
    if reverse:
        pairs_list.append((cols[ok], rows[ok]))
    else:
        pairs_list.append((rows[ok], cols[ok]))


def generate_candidates(s1, s23, encoder="svd", same_country=False, k_scale=1.0, use_emb=None):
    """Return (pairs DataFrame [i1, i2, search sims...], colstats dict).

    encoder: "svd" (no pretrained weights) - used for the nm/full dense views.
    use_emb: optional "st:<model>" for an extra embedding view.
    """
    n1, n2 = len(s1), len(s23)
    g1 = s1["country_n"].values if same_country else None
    g2 = s23["country_n"].values if same_country else None
    log(f"blocking: backend={backend_name()} encoder={encoder} emb={use_emb} same_country={same_country}")
    plist = []
    colstats = {"best": {}, "second": {}}
    dense = {}
    views = dict(DENSE_VIEWS)
    for v, (col, k, rk) in views.items():
        E1, E2 = encode(encoder, pd.concat([s1[col], s23[col]]).values, [s1[col].values, s23[col].values])
        dense[v] = (E1, E2)
    if use_emb:
        E1, E2 = encode(use_emb, None, [s1["emb_text"].values, s23["emb_text"].values])
        dense["emb"] = (E1, E2)
        views["emb"] = EMB_VIEW
    for v, (col, k, rk) in views.items():
        E1, E2 = dense[v]
        kk = max(1, int(round(k * k_scale)))
        idx, _ = topk(E1, E2, kk, g1, g2)
        _add(plist, idx, n1)
        ridx, rsim = topk(E2, E1, max(2, rk), g2, g1)
        _add(plist, ridx[:, :rk], n2, reverse=True)
        colstats["best"][v] = rsim[:, 0].clip(min=0)
        colstats["second"][v] = rsim[:, 1].clip(min=0)
        log(f"  view {v}: fwd k={kk} rev k={rk}")
    sparse = {}
    for v, (col, k, rk) in SPARSE_VIEWS.items():
        A, B = _sparse_search_mats(s1, s23, col)
        sparse[v] = (A, B)
        kk = max(1, int(round(k * k_scale)))
        idx, _ = sparse_topk(A, B, kk)
        if same_country:
            bad = (idx >= 0) & (g1[:, None] != g2[np.maximum(idx, 0)])
            idx[bad] = -1
        _add(plist, idx, n1)
        ridx, rsim = sparse_topk(B, A, max(2, rk))
        _add(plist, ridx[:, :rk], n2, reverse=True)
        colstats["best"][v] = rsim[:, 0].clip(min=0)
        colstats["second"][v] = rsim[:, 1].clip(min=0)
        log(f"  view {v}: fwd k={kk} rev k={rk}")
    i1 = np.concatenate([p[0] for p in plist]).astype(np.int64)
    i2 = np.concatenate([p[1] for p in plist]).astype(np.int64)
    key = np.unique(i1 * n2 + i2)
    pairs = pd.DataFrame({"i1": key // n2, "i2": key % n2})
    if same_country:
        pairs = pairs[s1["country_n"].values[pairs.i1] == s23["country_n"].values[pairs.i2]].reset_index(drop=True)
    # search-space similarity of every pair under every view
    a, b = pairs["i1"].values, pairs["i2"].values
    for v, (E1, E2) in dense.items():
        res = np.empty(len(a), np.float32)
        for st in range(0, len(a), 1_000_000):
            res[st:st + 1_000_000] = np.einsum("ij,ij->i", E1[a[st:st + 1_000_000]], E2[b[st:st + 1_000_000]])
        pairs[f"s_{v}"] = res
    for v, (A, B) in sparse.items():
        pairs[f"s_{v}"] = exact_view_sims(a, b, {v: (A, B)})[v]
    return pairs, colstats


def blocking_recall(pairs, s1, s23, truth, ids=None):
    """(recall of true links, candidates per S1, number of true links)."""
    id1 = s1["entity_id"].values
    id2 = s23["entity_id"].values
    have = set(zip(id1[pairs["i1"].values], id2[pairs["i2"].values]))
    ids = id1 if ids is None else ids
    tot = hit = 0
    for sid in ids:
        for m in truth.get(sid, ()):
            tot += 1
            hit += (sid, m) in have
    return hit / max(tot, 1), len(pairs) / max(len(s1), 1), tot
