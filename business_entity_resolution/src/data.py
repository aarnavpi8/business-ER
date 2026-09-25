"""Loading source TSVs / ground truth and building normalised record tables."""
import os
import pandas as pd

from normalize import (normalize_name, core_name, name_acronym, normalize_address,
                       address_numbers, first_number, address_words, basic_clean)

COLS = ["entity_id", "business_name", "business_address", "country"]


def read_tsv(path: str) -> pd.DataFrame:
    """Read a challenge TSV with an explicit tab separator, all columns as str."""
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                       quoting=3, na_filter=False)


def load_split(data_dir: str, split: str):
    """Return (s1, s23) DataFrames for split in {'train','test'}.

    s23 holds Source 2 and Source 3 records together, with a `src` column.
    """
    s1 = read_tsv(os.path.join(data_dir, split, f"{split}_source1.tsv"))
    s2 = read_tsv(os.path.join(data_dir, split, f"{split}_source2.tsv"))
    s3 = read_tsv(os.path.join(data_dir, split, f"{split}_source3.tsv"))
    s1["src"] = 1
    s2["src"] = 2
    s3["src"] = 3
    s23 = pd.concat([s2, s3], ignore_index=True)
    return prepare(s1), prepare(s23)


def load_ground_truth(data_dir: str) -> dict:
    """Map source1_entity_id -> set of matched S2/S3 ids (empty set for singletons)."""
    gt = read_tsv(os.path.join(data_dir, "train", "train_ground_truth.tsv"))
    out = {}
    for sid, m in zip(gt["source1_entity_id"], gt["matched_entity_ids"]):
        out[sid] = set(x.strip() for x in str(m).split(",") if x.strip())
    return out


def _norm_chunk(args):
    """Worker: normalise one chunk of (name, address, country) lists."""
    names, addrs, countries = args
    nn = [normalize_name(x) for x in names]
    return dict(name_n=nn, name_core=[core_name(x) for x in nn],
                addr_n=[normalize_address(x) for x in addrs],
                country_n=[basic_clean(x) for x in countries])


KEEP = ["entity_id", "src", "country_n", "name_n", "name_core", "addr_n"]


def prepare(df: pd.DataFrame, n_jobs=None, chunk=100_000, slim=True) -> pd.DataFrame:
    """Add normalised name/address/country columns (parallel for large frames).

    slim=True keeps only the columns the pipeline needs (memory: ~10M rows per
    split); everything else (acronyms, number sets, ...) is derived on the fly.
    """
    import multiprocessing as mp
    df = df.reset_index(drop=True)
    for c in COLS:
        if c not in df.columns:
            df[c] = ""
    cols = [df["business_name"].tolist(), df["business_address"].tolist(), df["country"].tolist()]
    tasks = [tuple(c[i:i + chunk] for c in cols) for i in range(0, len(df), chunk)]
    n_jobs = n_jobs or max(1, (os.cpu_count() or 2) - 1)
    if len(tasks) > 1 and n_jobs > 1:
        with mp.get_context("spawn").Pool(min(n_jobs, len(tasks))) as pool:
            parts = pool.map(_norm_chunk, tasks)
    else:
        parts = [_norm_chunk(t) for t in tasks]
    for k in parts[0]:
        df[k] = [x for p in parts for x in p[k]]
    if slim:
        df = df[KEEP].copy()
    return df


def load_prepared(data_dir: str, split: str, cache_dir: str = None):
    """load_split with an on-disk pickle cache of the normalised frames."""
    import pickle
    if cache_dir:
        path = os.path.join(cache_dir, f"prep_{split}.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
    s1, s23 = load_split(data_dir, split)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump((s1, s23), f, protocol=4)
    return s1, s23


def subsample(s1, s23, truth, frac, seed=0):
    """Subsample a training split while keeping its match structure realistic.

    Keeps a `frac` share of S1 entities, every S2/S3 record linked to them, and
    the same `frac` share of S2/S3 records linked to no S1 at all. Records linked
    to dropped S1s are removed, so the density of distractors per S1 is preserved.
    """
    if frac >= 1.0:
        return s1, s23
    import numpy as np
    rng = np.random.default_rng(seed)
    keep1 = rng.random(len(s1)) < frac
    s1k = s1[keep1].reset_index(drop=True)
    linked, any_linked = set(), set()
    for sid, ms in truth.items():
        any_linked |= ms
    for sid in s1k["entity_id"]:
        linked |= truth.get(sid, set())
    ids = s23["entity_id"]
    keep2 = ids.isin(linked).values | (~ids.isin(any_linked).values & (rng.random(len(s23)) < frac))
    return s1k, s23[keep2].reset_index(drop=True)
