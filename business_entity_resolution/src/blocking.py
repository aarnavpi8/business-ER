"""Scalable candidate generation (blocking), run separately per country.

Facts from the training data that shape this module:
  * no ground-truth link crosses countries  -> block within each country label
    (plain string equality; works for unseen labels such as France);
  * each S2/S3 record belongs to at most one S1 entity -> the primary search
    direction is S2/S3 -> S1 (each S2/S3 record retrieves its nearest S1s);
  * ~3-4% of true pairs share no name at all (trade names) -> an address-only
    view is needed in addition to name views.

Views (dense, exact inner-product search on GPU/CPU, see search.py):
  nm   : char n-gram TF-IDF of the core name          -> random projection
  ad   : char n-gram TF-IDF of the normalised address -> random projection
  full : char n-gram TF-IDF of "core name | address"  -> random projection
Each view is searched in the reverse direction (S23 -> S1, k_rev) and the
forward direction (S1 -> S23, k_fwd). TF-IDF + projection are fitted per country on
that country's own text (unsupervised, no external data).

Embeddings are kept as float16 memmaps on disk so memory stays bounded; the
search-space similarity of every candidate pair under every view is returned
together with the best / second-best similarity of each record to the other
side (used as "is this my best option?" features).
"""
import os
import time
import tempfile
import numpy as np
import pandas as pd

from search import encode, topk, backend_name

# view -> (text column, k_rev: S1 per S23, k_fwd: S23 per S1)
VIEWS = {"nm": ("name_core", 5, 12), "ad": ("addr_n", 5, 12), "full": ("full_n", 5, 12)}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def _pairs_from_topk(idx, reverse):
    """(i1, i2) local arrays from a top-k index matrix (rows are queries)."""
    rows = np.repeat(np.arange(idx.shape[0], dtype=np.int64), idx.shape[1])
    cols = idx.ravel()
    ok = cols >= 0
    return (cols[ok], rows[ok]) if reverse else (rows[ok], cols[ok])


def view_text(df, col):
    """Text for a view; 'full_n' is built on the fly (name | address) to save memory."""
    if col == "full_n" and col not in df.columns:
        return (df["name_core"] + " | " + df["addr_n"]).values
    return df[col].values


def block_group(s1g, s23g, views=VIEWS, k_scale=1.0, dim=256, encoder="rp", tmpdir=None):
    """Blocking for one country group.

    Returns dict with local pair arrays i1, i2 (int64, sorted by i2 then i1),
    per-view pair similarities s_<view> (float32), and per-record stats
    best2_<v>/second2_<v> (for each S23 row) and best1_<v> (for each S1 row).
    """
    n1, n2 = len(s1g), len(s23g)
    tmpdir = tmpdir or tempfile.mkdtemp(prefix="ber_emb_")
    keys, stats, emb_files = [], {}, {}
    for v, (col, k_rev, k_fwd) in views.items():
        t = time.time()
        t1, t2 = view_text(s1g, col), view_text(s23g, col)
        E1, E2 = encode(encoder, np.concatenate([t1, t2]), [t1, t2], dim=dim)
        del t1, t2
        kr = max(2, int(round(k_rev * k_scale)))
        kf = max(1, int(round(k_fwd * k_scale)))
        ridx, rsim = topk(E2, E1, kr)                     # S23 -> S1
        a, b = _pairs_from_topk(ridx, reverse=True)
        keys.append(a * n2 + b)
        stats[f"best2_{v}"] = rsim[:, 0].clip(min=0)
        stats[f"second2_{v}"] = rsim[:, 1].clip(min=0) if rsim.shape[1] > 1 else np.zeros(n2, np.float32)
        del ridx, rsim
        fidx, fsim = topk(E1, E2, kf)                     # S1 -> S23
        a, b = _pairs_from_topk(fidx, reverse=False)
        keys.append(a * n2 + b)
        stats[f"best1_{v}"] = fsim[:, 0].clip(min=0)
        del fidx, fsim
        # park embeddings on disk (fp16) until pair similarities are computed
        f1, f2 = os.path.join(tmpdir, f"{v}_1.npy"), os.path.join(tmpdir, f"{v}_2.npy")
        np.save(f1, E1.astype(np.float16)); np.save(f2, E2.astype(np.float16))
        emb_files[v] = (f1, f2)
        del E1, E2
        log(f"    view {v}: k_rev={kr} k_fwd={kf} {time.time() - t:.0f}s")
    key = np.unique(np.concatenate(keys))
    i1 = (key // n2).astype(np.int64)
    i2 = (key % n2).astype(np.int64)
    o = np.lexsort((i1, i2))                              # sort by i2, then i1
    i1, i2 = i1[o], i2[o]
    out = {"i1": i1, "i2": i2}
    for v, (f1, f2) in emb_files.items():
        E1 = np.load(f1, mmap_mode="r"); E2 = np.load(f2, mmap_mode="r")
        s = np.empty(len(i1), np.float32)
        step = 200_000  # 200k x 256 x 4B = 200 MB per side
        for st in range(0, len(i1), step):
            a = np.asarray(E1[i1[st:st + step]], np.float32)  # fancy index on memmap
            b = np.asarray(E2[i2[st:st + step]], np.float32)
            s[st:st + step] = np.einsum("ij,ij->i", a, b)
        out[f"s_{v}"] = s
        del E1, E2
        for f in (f1, f2):
            try:
                os.remove(f)
            except OSError:
                pass
    out.update(stats)
    return out


def block_split(s1, s23, views=VIEWS, k_scale=1.0, dim=256, encoder="rp", keep_k=None):
    """Run block_group for every country label; returns concatenated global arrays.

    Output dict: i1, i2 (global row indices into s1 / s23, sorted by i2),
    s_<view> pair sims, and per-record stat arrays aligned to s1 / s23 rows.
    keep_k: prune to the top-keep_k S1 per S2/S3 record inside each country (identical to
    pruning afterwards, since a record belongs to one country, but the full pair table never exists).
    """
    log(f"blocking: backend={backend_name()} encoder={encoder} dim={dim}")
    parts = []
    rec1 = {f"best1_{v}": np.zeros(len(s1), np.float32) for v in views}
    rec2 = {f"{p}_{v}": np.zeros(len(s23), np.float32) for v in views for p in ("best2", "second2")}
    c1 = np.asarray(s1["country_n"], dtype=object)
    c2 = np.asarray(s23["country_n"], dtype=object)
    for g in sorted(set(c1) | set(c2)):
        r1 = np.where(c1 == g)[0]
        r2 = np.where(c2 == g)[0]
        if len(r1) == 0 or len(r2) == 0:
            continue
        t = time.time()
        log(f"  country '{g}': S1={len(r1)} S23={len(r2)}")
        out = block_group(s1.iloc[r1].reset_index(drop=True), s23.iloc[r2].reset_index(drop=True),
                          views, k_scale, dim, encoder)
        for k in list(out):
            if k.startswith("best1_"):
                rec1[k][r1] = out.pop(k)
            elif k.startswith("best2_") or k.startswith("second2_"):
                rec2[k][r2] = out.pop(k)
        n_raw = len(out["i1"])
        out = prune_blocking(out, keep_k)
        out["i1"] = r1[out["i1"]]
        out["i2"] = r2[out["i2"]]
        parts.append(out)
        log(f"  country '{g}': {n_raw} pairs -> {len(out['i1'])} kept ({len(out['i1']) / len(r2):.1f}/S23) {time.time() - t:.0f}s")
    # concatenate / reorder one array at a time so at most one extra copy is alive
    res = {k: np.concatenate([p.pop(k) for p in parts]) for k in list(parts[0])}
    del parts
    o = np.lexsort((res["i1"], res["i2"]))
    for k in list(res):
        res[k] = res[k][o]
    del o
    res.update(rec1)
    res.update(rec2)
    return res


def prune_blocking(B, keep_k):
    """Keep the top-keep_k S1 candidates of every S2/S3 record by summed view similarity.

    On real-density train data the true S1 is ranked 1st for 96.6% of linked records and
    within the top 8 for 98.15% (of 98.24% retrieved at all), so this halves the pairs
    for a ~0.1% recall cost. Per-record stats (best/second) are unaffected.
    """
    from features import group_rank_desc
    if not keep_k:
        return B
    combo = B["s_nm"] + B["s_ad"] + B["s_full"]
    keep = group_rank_desc(B["i2"], combo) <= keep_k
    n = len(B["i1"])
    return {k: (v[keep] if len(v) == n and k.split("_")[0] in ("i1", "i2", "s") else v) for k, v in B.items()}


def blocking_recall(i1, i2, s1, s23, truth, ids=None):
    """(recall of true links, pairs per S1, number of true links)."""
    id1, id2 = s1["entity_id"].values, s23["entity_id"].values
    have = set(zip(id1[i1].tolist(), id2[i2].tolist()))
    ids = id1 if ids is None else ids
    tot = hit = 0
    for sid in ids:
        for m in truth.get(sid, ()):
            tot += 1
            hit += (sid, m) in have
    return hit / max(tot, 1), len(i1) / max(len(s1), 1), tot
