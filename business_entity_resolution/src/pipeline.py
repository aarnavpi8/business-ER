"""End-to-end pipeline, per country:
    data -> blocking -> stage-1 prune -> stage-2 match -> one-to-one decode -> output

  python src/pipeline.py run --data DATA --out output                  # CV on train, then test
  python src/pipeline.py run --data DATA --no-test --frac 0.1          # quick CV experiment
  python src/pipeline.py run --data DATA --train-countries us          # train on US only (holdout check)

How decisions are made
  Each S2/S3 record belongs to at most one S1 entity (true for all 7.6M training
  links), so the final decision is made per S2/S3 record: take its highest-scoring
  S1 candidate and accept it if p >= t. Matches are then grouped by S1.

Stages
  1. blocking.block_split       : per-country multi-view dense search, both directions.
  2. stage-1 GBDT on cheap feats : keep top-M1 S1 candidates per S2/S3 record.
                                   These pairs are exactly what stage 2 scores and are
                                   written to candidate_pairs.tsv.
  3. stage-2 GBDT on cheap + string + stacked stage-1 feats -> p(match).
  4. decode: per-S23 argmax + threshold t (tuned on out-of-fold predictions).

Train rows: a `frac` sample of S1 entities; we keep every candidate pair of every
S2/S3 record that has a sampled S1 among its candidates, so per-S23 decisions and
per-S1 scores are evaluated exactly as on test. Group features are computed on
*all* pairs before sampling, so density is realistic.
"""
import argparse
import gc
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load_prepared, load_ground_truth  # noqa: E402
from blocking import block_split, VIEWS  # noqa: E402
from features import (i1_stats, i2_chunks, cheap_chunk, string_features, stack_features,  # noqa: E402
                      feature_columns, STRING_BACKEND)
from models import GBM  # noqa: E402
from metric import macro_f05, breakdown  # noqa: E402
from search import backend_name  # noqa: E402


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------ helpers
def owner_array(s1, s23, truth):
    """For each S23 row: row index of its true S1 (-1 if none / not in this split)."""
    pos1 = pd.Series(np.arange(len(s1)), index=s1["entity_id"].values)
    own = pd.Series(-1, index=s23["entity_id"].values, dtype=np.int64)
    a, b = [], []
    for sid, ms in truth.items():
        for m in ms:
            a.append(sid); b.append(m)
    link = pd.DataFrame({"s1": a, "m": b})
    link = link[link["s1"].isin(pos1.index) & link["m"].isin(own.index)]
    own.loc[link["m"].values] = pos1.loc[link["s1"].values].values
    return own.values


def prune_per_s23(X, col, M):
    """Keep the top-M rows (by col) of every S2/S3 record."""
    r = X.groupby("i2", sort=False)[col].rank(ascending=False, method="first")
    return X[r.values <= M]


def decode(df, t, margin=0.0):
    """Per-S23 argmax over its S1 candidates, accepted if p >= t (and beats #2 by margin).

    df: DataFrame with i1, i2, p. Returns {i1: set(i2)}.
    """
    d = df.sort_values(["i2", "p"], ascending=[True, False])
    first = ~d["i2"].duplicated()
    top = d[first]
    if margin > 0:
        second = d[~first].drop_duplicates("i2").set_index("i2")["p"]
        m = top["p"].values - second.reindex(top["i2"].values).fillna(0).values
        top = top[m >= margin]
    top = top[top["p"] >= t]
    return top.groupby("i1")["i2"].apply(set).to_dict()


def to_ids(pred_idx, s1, s23):
    id1, id2 = s1["entity_id"].values, s23["entity_id"].values
    return {id1[i]: {id2[j] for j in js} for i, js in pred_idx.items()}


def tune_threshold(df, s1, s23, truth, ids):
    """Grid-search (t, margin) on OOF predictions; returns best cfg and table."""
    rows = []
    for t in np.round(np.arange(0.2, 0.91, 0.05), 2):
        for mg in (0.0, 0.1, 0.2):
            sc = macro_f05(to_ids(decode(df, t, mg), s1, s23), truth, ids)
            rows.append((sc, dict(t=float(t), margin=mg)))
    rows.sort(key=lambda r: -r[0])
    return rows[0][1], rows


def oof(X, y, cols, groups, n_splits, args, tag):
    """Out-of-fold probabilities (folds grouped by `groups`)."""
    p = np.zeros(len(X), np.float32)
    for k, (tr, va) in enumerate(GroupKFold(n_splits=n_splits).split(X, y, groups)):
        m = GBM(args.backend, n_estimators=args.n_est).fit(X.iloc[tr][cols].values, y[tr])
        p[va] = m.predict_proba(X.iloc[va][cols].values)
        log(f"  [{tag}] fold {k} done ({m.backend}, train rows={len(tr)})")
    return p


def blocking_cached(s1, s23, args, split):
    """block_split (pruned to top-keep_per_s23 S1 per S23) with an .npz cache."""
    key = f"block_{split}_k{args.k_scale}_d{args.dim}_keep{args.keep_per_s23}.npz"
    path = os.path.join(args.cache, key)
    if os.path.exists(path):
        z = np.load(path)
        B = {k: z[k] for k in z.files}
    else:
        B = block_split(s1, s23, k_scale=args.k_scale, dim=args.dim, encoder=args.encoder,
                        keep_k=args.keep_per_s23)
        np.savez(path, **B)
    log(f"[{split}] blocking: {len(B['i1'])} pairs (top-{args.keep_per_s23} S1 per S23)")
    return B


# ------------------------------------------------------------------ train side
def build_train(args):
    """Blocking + cheap features + labels for the sampled training rows."""
    truth = load_ground_truth(args.data)
    s1, s23 = load_prepared(args.data, "train", args.cache)
    present = set(s23["entity_id"])
    ids1 = set(s1["entity_id"])
    truth = {k: v & present for k, v in truth.items() if k in ids1}  # region subsets
    log(f"[train] S1={len(s1)} S23={len(s23)}")
    B = blocking_cached(s1, s23, args, "train")
    own = owner_array(s1, s23, truth)
    y_all = (own[B["i2"]] == B["i1"]).astype(np.int8)
    n_true = int((own >= 0).sum())
    log(f"[train] pairs={len(y_all)} ({len(y_all) / len(s23):.2f}/S23)  "
        f"blocking recall={y_all.sum() / max(n_true, 1):.4f} of {n_true} links")
    st1 = i1_stats(B, len(s1))
    # sample S1 entities, keep all pairs of S23 records touching them
    rng = np.random.default_rng(args.seed)
    samp1 = rng.random(len(s1)) < args.frac
    if args.train_countries:
        keep_c = set(args.train_countries.split(","))
        samp1 &= s1["country_n"].isin(keep_c).values
    # an S23 is "touched" if a sampled S1 is among its top-`touch_rank` candidates; all of
    # its candidate pairs are kept so per-S23 decisions are evaluated as on test
    from features import group_rank_desc
    top = group_rank_desc(B["i2"], B["s_nm"] + B["s_ad"] + B["s_full"]) <= args.touch_rank
    touch = np.zeros(len(s23), bool)
    touch[B["i2"][samp1[B["i1"]] & top]] = True
    parts, ys = [], []
    for a, b in i2_chunks(B, args.chunk_rows):
        r = touch[B["i2"][a:b]]
        if r.any():
            parts.append(cheap_chunk(B, a, b, s1, s23, st1, rows=r))
            ys.append(y_all[a:b][r])
    X = pd.concat(parts, ignore_index=True)
    y = np.concatenate(ys)
    del B, y_all, parts, ys
    gc.collect()
    ids = s1["entity_id"].values[samp1].tolist()
    log(f"[train] sampled S1={len(ids)} rows={len(X)} positives={int(y.sum())}")
    return s1, s23, truth, own, X, y, ids


def cmd_run(args):
    log(f"search backend={backend_name()}  string backend={STRING_BACKEND}")
    os.makedirs(args.cache, exist_ok=True)
    s1, s23, truth, own, X, y, ids = build_train(args)
    n_true_ids = sum(len(truth.get(i, ())) for i in ids)
    c1 = feature_columns(X)

    # ---- stage 1 (OOF) -> prune per S23
    X["p1"] = oof(X, y, c1, X["i1"].values, 3, args, "stage1")
    for M in (1, 2, 3, 5):
        kept = prune_per_s23(X, "p1", M)
        log(f"  prune M1={M}: keeps {y[kept.index].sum() / max(y.sum(), 1):.4f} of blocked positives, rows/S23={len(kept) / X['i2'].nunique():.2f}")
    keep = prune_per_s23(X, "p1", args.M)
    Xp = keep.reset_index(drop=True)
    yp = y[keep.index]
    m1 = None
    if not args.no_test:  # final stage-1 model on all sampled pre-prune rows
        m1 = GBM(args.backend, n_estimators=args.n_est).fit(X[c1].values, y)
    del X, keep
    gc.collect()

    # ---- stage 2
    t = time.time()
    Xp = stack_features(Xp, "p1")
    Xp = string_features(Xp, s1, s23)
    log(f"[train] stage-2 features {Xp.shape} {time.time() - t:.0f}s")
    c2 = feature_columns(Xp)
    df = Xp[["i1", "i2"]].copy()
    df["p"] = oof(Xp, yp, c2, Xp["i1"].values, args.folds, args, "stage2")
    cfg, table = tune_threshold(df, s1, s23, truth, ids)
    for sc, c in table[:5]:
        log(f"  decoder {sc:.4f} {c}")
    pred = to_ids(decode(df, cfg["t"], cfg["margin"]), s1, s23)
    bd = breakdown(pred, truth, ids)
    log(f"[train] OOF F0.5={bd['all']:.4f}  singletons={bd['singletons']:.4f} ({bd['n_single']})  "
        f"with_matches={bd['with_matches']:.4f} ({bd['n_match']})")
    ceil_idx = df[yp == 1].groupby("i1")["i2"].apply(set).to_dict()
    log(f"[train] ceiling (perfect matcher on kept candidates): {macro_f05(to_ids(ceil_idx, s1, s23), truth, ids):.4f}")
    samp = s1[s1["entity_id"].isin(set(ids))]
    for c, g in samp.groupby("country_n"):
        cid = g["entity_id"].tolist()
        log(f"[train] country '{c}': OOF F0.5={macro_f05(pred, truth, cid):.4f} (n={len(cid)})")
    if args.holdout:
        # leave-one-country-out on stage 2: train on the other countries, score this one
        cn_row = s1["country_n"].values[Xp["i1"].values]
        for c in sorted(set(cn_row)):
            tr = cn_row != c
            if tr.all() or (~tr).all():
                continue
            m = GBM(args.backend, n_estimators=args.n_est).fit(Xp[tr][c2].values, yp[tr])
            d = df[~tr].copy()
            d["p"] = m.predict_proba(Xp[~tr][c2].values)
            cid = samp[samp["country_n"] == c]["entity_id"].tolist()
            best = max((macro_f05(to_ids(decode(d, t), s1, s23), truth, cid), t) for t in np.arange(0.3, 0.91, 0.05))
            log(f"[train] holdout '{c}' (trained on other countries): F0.5={macro_f05(to_ids(decode(d, cfg['t'], cfg['margin']), s1, s23), truth, cid):.4f} "
                f"at global t; best t={best[1]:.2f} -> {best[0]:.4f}")
    summary = dict(cfg=cfg, oof=bd, M=args.M, string_backend=STRING_BACKEND, args=vars(args))
    json.dump(summary, open(os.path.join(args.cache, "cv_summary.json"), "w"), default=float, indent=1)
    oof_frame = Xp.copy(); oof_frame["y"] = yp; oof_frame["p"] = df["p"].values
    with open(os.path.join(args.cache, "oof_train.pkl"), "wb") as f:
        pickle.dump(dict(X=oof_frame, s1=s1, s23=s23, truth=truth, cfg=cfg, c2=c2, ids=ids), f, protocol=4)
    if args.no_test:
        return

    # ---- final stage-2 model, then test
    m2 = GBM(args.backend, n_estimators=args.n_est).fit(Xp[c2].values, yp)
    pickle.dump(dict(m1=m1, m2=m2, c1=c1, c2=c2, cfg=cfg), open(os.path.join(args.cache, "models.pkl"), "wb"))
    del Xp, s1, s23, oof_frame
    gc.collect()
    predict_test(args, m1, m2, c1, c2, cfg)


# ------------------------------------------------------------------ test side
def predict_test(args, m1, m2, c1, c2, cfg):
    """Blocking + two-stage scoring + decoding on the test split, written to args.out."""
    t1, t23 = load_prepared(args.data, "test", args.cache)
    log(f"[test] S1={len(t1)} S23={len(t23)}")
    B = blocking_cached(t1, t23, args, "test")
    st1 = i1_stats(B, len(t1))
    chunks = i2_chunks(B, args.chunk_rows)
    # pass 1: stage-1 scores and per-S23 pruning; only kept row positions and p1 are stored
    rows, p1s = [], []
    for a, b in chunks:
        Xc = cheap_chunk(B, a, b, t1, t23, st1)
        Xc["p1"] = m1.predict_proba(Xc[c1].values)
        k = prune_per_s23(Xc, "p1", args.M)
        rows.append(a + k.index.values)
        p1s.append(k["p1"].values.astype(np.float32))
    rows = np.concatenate(rows)  # ascending: chunks are in order and pruning keeps row order
    p1 = np.concatenate(p1s)
    del Xc, k, p1s
    i1_kept = B["i1"][rows]
    g1max = np.full(len(t1), -np.inf, np.float32)
    np.maximum.at(g1max, i1_kept, p1)
    log(f"  [test] stage 1: kept {len(rows)} of {len(B['i1'])} pairs")
    # pass 2: rebuild features for the kept rows chunk by chunk and score with stage 2
    probs = np.empty(len(rows), np.float32)
    for a, b in chunks:
        lo, hi = np.searchsorted(rows, a), np.searchsorted(rows, b)
        if lo == hi:
            continue
        mask = np.zeros(b - a, bool)
        mask[rows[lo:hi] - a] = True
        Xc = cheap_chunk(B, a, b, t1, t23, st1, rows=mask)
        Xc["p1"] = p1[lo:hi]
        Xc = stack_features(Xc, "p1", g1max=g1max)
        Xc = string_features(Xc, t1, t23)
        probs[lo:hi] = m2.predict_proba(Xc[c2].values)
        log(f"  [test] stage 2 {hi}/{len(rows)}")
    tdf = pd.DataFrame({"i1": i1_kept, "i2": B["i2"][rows], "p": probs})
    del B, rows, p1, probs
    gc.collect()
    pred_idx = decode(tdf, cfg["t"], cfg["margin"])
    o = np.argsort(tdf["i1"].values, kind="stable")
    i2_by_i1 = tdf["i2"].values[o]
    bounds = np.searchsorted(tdf["i1"].values[o], np.arange(len(t1) + 1))
    cand_idx = {i: i2_by_i1[bounds[i]:bounds[i + 1]] for i in range(len(t1)) if bounds[i + 1] > bounds[i]}
    write_outputs(t1, t23, cand_idx, pred_idx, args.out)
    tdf.to_csv(os.path.join(args.cache, "test_pair_probs.tsv.gz"), sep="\t", index=False)
    n_links = sum(len(v) for v in pred_idx.values())
    log(f"[test] wrote {args.out}: {len(pred_idx)} S1 with matches ({n_links} links), "
        f"{len(t1) - len(pred_idx)} predicted singletons; candidates/S1={len(tdf) / len(t1):.1f}")
    for c in sorted(set(t1["country_n"])):
        ii = np.flatnonzero(t1["country_n"].values == c)
        has = np.fromiter((i in pred_idx for i in ii), bool, count=len(ii))
        nl = sum(len(pred_idx.get(i, ())) for i in ii)
        log(f"[test] country '{c}': n={len(ii)} with-match={has.mean():.3f} links/S1={nl / len(ii):.2f}")


def write_outputs(s1, s23, cand_idx, pred_idx, out_dir):
    """Write matching_results.tsv and candidate_pairs.tsv (one row per S1, tab-separated)."""
    os.makedirs(out_dir, exist_ok=True)
    id1, id2 = s1["entity_id"].values, s23["entity_id"].values
    with open(os.path.join(out_dir, "matching_results.tsv"), "w", encoding="utf-8", newline="\n") as fm, \
            open(os.path.join(out_dir, "candidate_pairs.tsv"), "w", encoding="utf-8", newline="\n") as fc:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        for i in range(len(s1)):
            cands = sorted({id2[j] for j in cand_idx.get(i, ())})
            preds = sorted({id2[j] for j in pred_idx.get(i, ())})
            assert set(preds) <= set(cands), "prediction outside candidate set"
            fm.write(f"{id1[i]}\t{','.join(preds)}\n")
            fc.write(f"{id1[i]}\t{','.join(cands)}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--out", default="output")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--frac", type=float, default=0.1, help="fraction of train S1 entities used for training")
    ap.add_argument("--train-countries", default=None, help="e.g. 'us' to train on one country only")
    ap.add_argument("--no-test", action="store_true")
    ap.add_argument("--holdout", action="store_true", help="also report leave-one-country-out scores")
    ap.add_argument("--encoder", default="rp")
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--k-scale", type=float, default=1.0)
    ap.add_argument("--keep-per-s23", type=int, default=8, help="blocking: S1 candidates kept per S2/S3 record")
    ap.add_argument("--touch-rank", type=int, default=3, help="train sampling: S23 kept if a sampled S1 is in its top-N")
    ap.add_argument("--M", type=int, default=3, help="S1 candidates kept per S2/S3 record after stage 1")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--backend", default="auto", help="lgbm | xgb | hgb | auto")
    ap.add_argument("--n-est", type=int, default=600)
    ap.add_argument("--chunk-rows", type=int, default=2_000_000)
    ap.add_argument("--seed", type=int, default=0)
    cmd_run(ap.parse_args())


if __name__ == "__main__":
    main()
