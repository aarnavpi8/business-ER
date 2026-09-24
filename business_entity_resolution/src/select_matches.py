"""Turn pair probabilities into per-entity match sets that maximise F0.5.

Two decoders are provided and chosen between on out-of-fold data:

* threshold: keep every candidate with p >= t.
* expected_f: for each S1 entity sort candidates by p and pick the prefix
  length k (0..kmax) with the highest *expected* F0.5 under independent
  Bernoulli(p) truth, estimated by Monte Carlo. k = 0 (predict singleton)
  is scored as P(no true match among candidates).

Optionally a one-to-one constraint is applied first: since Source 1 is
deduplicated, each S2/S3 record is kept only for the S1 entity where it has
the highest probability.
"""
import numpy as np
import pandas as pd


def one_to_one(df: pd.DataFrame) -> pd.DataFrame:
    """Keep each i2 only for its highest-probability i1."""
    idx = df.groupby("i2")["p"].idxmax()
    return df.loc[idx]


def decode_threshold(df: pd.DataFrame, t: float) -> dict:
    """{i1: set(i2)} for candidates with p >= t."""
    keep = df[df["p"] >= t]
    return keep.groupby("i1")["i2"].apply(set).to_dict()


def _expected_f_table(p: np.ndarray, kmax: int, n_mc: int, rng) -> np.ndarray:
    """Expected F0.5 for predicting the top-k of sorted probs p, k = 0..kmax."""
    n = len(p)
    kmax = min(kmax, n)
    draws = rng.random((n_mc, n)) < p[None, :]
    n_true = draws.sum(1)
    out = np.zeros(kmax + 1)
    out[0] = (n_true == 0).mean()
    tp = np.cumsum(draws, axis=1)
    for k in range(1, kmax + 1):
        tpk = tp[:, k - 1]
        prec = tpk / k
        rec = np.where(n_true > 0, tpk / np.maximum(n_true, 1), 0.0)
        denom = 0.25 * prec + rec
        f = np.where(denom > 0, 1.25 * prec * rec / np.where(denom > 0, denom, 1), 0.0)
        out[k] = f.mean()
    return out


def decode_expected_f(df: pd.DataFrame, kmax: int = 6, n_mc: int = 400,
                      min_p: float = 0.02, bias: float = 0.0, seed: int = 0) -> dict:
    """{i1: set(i2)} maximising expected F0.5 per entity.

    bias: added to the k=0 (singleton) option's expected score; >0 makes the
    decoder more conservative. Tuned on OOF.
    """
    rng = np.random.default_rng(seed)
    res = {}
    d = df[df["p"] >= min_p].sort_values(["i1", "p"], ascending=[True, False])
    for i1, g in d.groupby("i1", sort=False):
        p = g["p"].values
        tab = _expected_f_table(p, kmax, n_mc, rng)
        tab[0] += bias
        k = int(np.argmax(tab))
        if k > 0:
            res[i1] = set(g["i2"].values[:k].tolist())
    return res
