"""Loading source TSVs / ground truth and building normalised record tables."""
import os
import pandas as pd

from normalize import (normalize_name, core_name, name_acronym, normalize_address,
                       address_numbers, postcode, basic_clean)

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


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Add normalised name/address fields and parsed address components."""
    df = df.copy().reset_index(drop=True)
    for c in COLS:
        if c not in df.columns:
            df[c] = ""
    df["name_n"] = df["business_name"].map(normalize_name)
    df["name_core"] = df["name_n"].map(core_name)
    df["name_acr"] = df["name_core"].map(name_acronym)
    df["addr_n"] = df["business_address"].map(normalize_address)
    df["addr_nums"] = df["addr_n"].map(address_numbers)
    df["pcode"] = df["business_address"].map(postcode)
    df["country_n"] = df["country"].map(basic_clean)
    df["full_n"] = (df["name_core"] + " | " + df["addr_n"]).str.strip()
    # raw-ish text for multilingual embedding models (keeps accents / casing)
    df["emb_text"] = (df["business_name"].astype(str) + ", " + df["business_address"].astype(str)).str.strip(", ")
    return df


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
