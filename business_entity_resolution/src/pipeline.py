"""End-to-end pipeline: data -> blocking -> stage-1 prune -> stage-2 match -> decode -> output.

  python src/pipeline.py run --data dataset --out output            # CV on train, then predict test
  python src/pipeline.py run --data dataset --no-test --frac 0.2    # quick experiment on 20% of train
  python src/pipeline.py run --data dataset --emb st:intfloat/multilingual-e5-small   # + embedding view

Stages
  1. Blocking (blocking.py): multi-view dense/sparse top-k search, both directions.
  2. Stage-1 model on cheap features scores every candidate; we keep the top M per
     S1 (M chosen on train so that pruning keeps ~all true links). The kept pairs
     are exactly what stage 2 sees, and are written to candidate_pairs.tsv.
  3. Stage-2 model on cheap + string features gives p(match).
  4. Decoder (select_matches.py) picks each S1's match set; its settings are
     tuned on out-of-fold stage-2 predictions.

Train predictions used for tuning are always out-of-fold (GroupKFold by S1),
so the pruning and decoding thresholds transfer to test.
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load_split, load_ground_truth, subsample  # noqa: E402
from blocking import fit_exact_views, generate_candidates, blocking_recall  # noqa: E402
from features import cheap_features, string_features, feature_columns, STRING_BACKEND  # noqa: E402
from models import GBM  # noqa: E402
from select_matches import decode_threshold, decode_expected_f, one_to_one  # noqa: E402
from metric import macro_f05, breakdown  # noqa: E402
from search import backend_name  # noqa: E402


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------ helpers
def label_pairs(X, s1, s23, truth):
    """Binary label per candidate pair from the ground truth."""
    id1 = s1["entity_id"].values[X["i1"].values]
    id2 = s23["entity_id"].values[X["i2"].values]
    return np.fromiter((b in truth.get(a, ()) for a, b in zip(id1, id2)), dtype=np.int8, count=len(X))


def to_ids(pred_idx, s1, s23):
    """{i1: set(i2)} -> {S1 id: set(S2/S3 ids)}."""
    id1, id2 = s1["entity_id"].values, s23["entity_id"].values
    return {id1[i]: {id2[j] for j in js} for i, js in pred_idx.items()}


def prune_topm(X, score_col, M):
    """Keep the top-M rows per i1 by score_col."""
    r = X.groupby("i1")[score_col].rank(ascending=False, method="first")
    return X[r <= M]


def oof(X, y, cols, n_splits, args, tag):
    """Out-of-fold probabilities, folds grouped by S1 entity."""
    p = np.zeros(len(X), np.float32)
    for k, (tr, va) in enumerate(GroupKFold(n_splits=n_splits).split(X, y, X["i1"].values)):
        m = GBM(args.backend, n_estimators=args.n_est).fit(X.iloc[tr][cols].values, y[tr])
        p[va] = m.predict_proba(X.iloc[va][cols].values)
        log(f"  [{tag}] fold {k} done ({m.backend})")
    return p


def decode(df, cfg):
    """Apply decoder cfg to a frame with columns i1, i2, p."""
    d = one_to_one(df) if cfg.get("one_to_one") else df
    if cfg["method"] == "threshold":
        return decode_threshold(d, cfg["t"])
    return decode_expected_f(d, kmax=cfg.get("kmax", 6), bias=cfg.get("bias", 0.0),
                             min_p=cfg.get("min_p", 0.02))


def tune_decoder(df, s1, s23, truth, ids):
    """Grid-search decoder settings on OOF predictions."""
    grid = []
    for o2o in (False, True):
        for t in (0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8):
            grid.append(dict(method="threshold", t=t, one_to_one=o2o))
        for b in (-0.05, 0.0, 0.05, 0.1):
            grid.append(dict(method="expected_f", bias=b, one_to_one=o2o))
    rows = [(macro_f05(to_ids(decode(df, c), s1, s23), truth, ids), c) for c in grid]
    rows.sort(key=lambda r: -r[0])
    return rows[0][1], rows


def build_blocking(s1, s23, args, tag):
    """Exact TF-IDF matrices + candidate pairs + reverse stats for one split."""
    t = time.time()
    mats = fit_exact_views(s1, s23)
    pairs, colstats = generate_candidates(s1, s23, encoder=args.encoder, same_country=args.same_country,
                                          k_scale=args.k_scale, use_emb=args.emb)
    log(f"[{tag}] S1={len(s1)} S23={len(s23)} candidates={len(pairs)} ({len(pairs) / max(len(s1), 1):.1f}/S1) {time.time() - t:.0f}s")
    return mats, pairs, colstats


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


# ------------------------------------------------------------------ main flow
def cmd_run(args):
    log(f"search backend={backend_name()}  string backend={STRING_BACKEND}")
    os.makedirs(args.cache, exist_ok=True)
    truth = load_ground_truth(args.data)
    s1, s23 = load_split(args.data, "train")
    s1, s23 = subsample(s1, s23, truth, args.frac, seed=args.seed)
    ids = s1["entity_id"].tolist()

    # ---- blocking + cheap features (train)
    mats, pairs, colstats = build_blocking(s1, s23, args, "train")
    rec, per, tot = blocking_recall(pairs, s1, s23, truth, ids)
    log(f"[train] blocking recall={rec:.4f} cand/S1={per:.1f} true links={tot}")
    X = cheap_features(pairs, s1, s23, mats, colstats)
    del pairs
    y = label_pairs(X, s1, s23, truth)
    c1 = feature_columns(X)

    # ---- stage 1 (OOF) and pruning
    X["p1"] = oof(X, y, c1, 3, args, "stage1")
    for M in (3, 5, 8, 10, 15, 20, 30):
        kept = prune_topm(X, "p1", M)
        log(f"  prune M={M:>2}: recall={y[kept.index].sum() / max(tot, 1):.4f} pairs/S1={len(kept) / len(s1):.1f}")
    keep = prune_topm(X, "p1", args.M)
    Xp = keep.reset_index(drop=True)
    yp = y[keep.index]
    log(f"[train] stage-1 kept M={args.M}: recall={yp.sum() / max(tot, 1):.4f}")

    # ---- stage 2
    t = time.time()
    Xp = string_features(Xp, s1, s23)
    log(f"[train] string features {Xp.shape} {time.time() - t:.0f}s")
    c2 = feature_columns(Xp)
    df = Xp[["i1", "i2"]].copy()
    df["p"] = oof(Xp, yp, c2, args.folds, args, "stage2")
    cfg, table = tune_decoder(df, s1, s23, truth, ids)
    for sc, c in table[:5]:
        log(f"  decoder {sc:.4f} {c}")
    pred = to_ids(decode(df, cfg), s1, s23)
    bd = breakdown(pred, truth, ids)
    log(f"[train] OOF F0.5={bd['all']:.4f}  singletons={bd['singletons']:.4f} ({bd['n_single']})  "
        f"with_matches={bd['with_matches']:.4f} ({bd['n_match']})")
    ceil = to_ids({i: set(g["i2"]) for i, g in df[yp == 1].groupby("i1")}, s1, s23)
    log(f"[train] ceiling with perfect matcher on kept candidates: {macro_f05(ceil, truth, ids):.4f}")
    # country holdout (proxy for the unseen country in test)
    cn = s1["country_n"].values[Xp["i1"].values]
    for held in sorted(set(s1["country_n"])):
        tr = cn != held
        if tr.all() or (~tr).all():
            continue
        m = GBM(args.backend, n_estimators=args.n_est).fit(Xp[tr][c2].values, yp[tr])
        d = df[~tr].copy()
        d["p"] = m.predict_proba(Xp[~tr][c2].values)
        hid = s1.loc[s1["country_n"] == held, "entity_id"].tolist()
        log(f"[train] country holdout '{held}': F0.5={macro_f05(to_ids(decode(d, cfg), s1, s23), truth, hid):.4f} (n={len(hid)})")
    json.dump(dict(cfg=cfg, oof=bd, M=args.M, blocking_recall=rec), open(os.path.join(args.cache, "cv_summary.json"), "w"), default=float)
    # artifacts for notebooks/03_error_analysis.ipynb
    oof_frame = Xp.copy()
    oof_frame["y"] = yp
    oof_frame["p"] = df["p"].values
    with open(os.path.join(args.cache, "oof_train.pkl"), "wb") as f:
        pickle.dump(dict(X=oof_frame, s1=s1, s23=s23, truth=truth, cfg=cfg, c2=c2), f)
    if args.no_test:
        return

    # ---- final models on all (sampled) train
    m1 = GBM(args.backend, n_estimators=args.n_est).fit(X[c1].values, y)
    m2 = GBM(args.backend, n_estimators=args.n_est).fit(Xp[c2].values, yp)
    del X, Xp
    pickle.dump(dict(m1=m1, m2=m2, c1=c1, c2=c2, cfg=cfg, string_backend=STRING_BACKEND),
                open(os.path.join(args.cache, "models.pkl"), "wb"))

    # ---- test: blocking, then cheap features + stage-1 in S1 chunks (bounded memory)
    t1, t23 = load_split(args.data, "test")
    tm, tpairs, tcol = build_blocking(t1, t23, args, "test")
    kept = []
    bounds = np.searchsorted(tpairs["i1"].values, np.arange(0, len(t1) + args.chunk_s1, args.chunk_s1))
    for a, b in zip(bounds[:-1], bounds[1:]):
        if a == b:
            continue
        Xc = cheap_features(tpairs.iloc[a:b].reset_index(drop=True), t1, t23, tm, tcol)
        Xc["p1"] = m1.predict_proba(Xc[c1].values)
        kept.append(prune_topm(Xc, "p1", args.M))
    TX = pd.concat(kept, ignore_index=True)
    del tpairs, kept
    TX = string_features(TX, t1, t23)
    tdf = TX[["i1", "i2"]].copy()
    tdf["p"] = m2.predict_proba(TX[c2].values)
    pred_idx = decode(tdf, cfg)
    cand_idx = TX.groupby("i1")["i2"].apply(set).to_dict()
    write_outputs(t1, t23, cand_idx, pred_idx, args.out)
    tdf.to_csv(os.path.join(args.cache, "test_pair_probs.tsv.gz"), sep="\t", index=False)
    n_links = sum(len(v) for v in pred_idx.values())
    log(f"[test] wrote {args.out}: {len(pred_idx)} S1 with matches ({n_links} links), "
        f"{len(t1) - len(pred_idx)} predicted singletons; cand/S1={len(TX) / len(t1):.1f}")
    for c in sorted(set(t1["country_n"])):
        ii = np.where(t1["country_n"].values == c)[0]
        log(f"[test] country '{c}': n={len(ii)} predicted-with-match={np.mean([i in pred_idx for i in ii]):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--out", default="output")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--frac", type=float, default=1.0, help="fraction of train S1 to use")
    ap.add_argument("--no-test", action="store_true")
    ap.add_argument("--encoder", default="svd")
    ap.add_argument("--emb", default=None, help="optional st:<model> embedding view")
    ap.add_argument("--same-country", action="store_true")
    ap.add_argument("--k-scale", type=float, default=1.0)
    ap.add_argument("--M", type=int, default=15, help="candidates kept per S1 after stage 1")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--backend", default="auto", help="lgbm | xgb | hgb | auto")
    ap.add_argument("--n-est", type=int, default=800)
    ap.add_argument("--chunk-s1", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cmd_run(args)


if __name__ == "__main__":
    main()
