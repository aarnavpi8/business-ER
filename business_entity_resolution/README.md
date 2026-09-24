# Business Entity Resolution — Amazon ML Challenge 2026

Pipeline: **data → normalisation → blocking → stage-1 prune → stage-2 match → per-entity decoding → output**.

Core dependencies: numpy / pandas / scipy / scikit-learn (`requirements.txt`).
Optional accelerators, auto-detected (`requirements-extra.txt`): rapidfuzz, lightgbm, xgboost,
torch (CUDA or Apple MPS), sentence-transformers. No external data or lookups.

## Layout

```
src/
  normalize.py        text cleaning, legal-suffix & abbreviation handling (multilingual)
  sims.py             pure-Python string similarity fallbacks
  data.py             TSV loading (explicit tab separator), normalised fields, train subsampling
  search.py           encoders (TF-IDF+SVD / sentence-transformers) and exact GPU/CPU top-k search
  blocking.py         multi-view candidate generation in both directions + reverse-best stats
  features.py         cheap (stage-1) and string (stage-2) pair features
  models.py           GBDT wrapper: lightgbm | xgboost (CUDA) | sklearn fallback
  select_matches.py   decoders: threshold / expected-F0.5 per entity, optional one-to-one
  metric.py           official macro F0.5
  pipeline.py         CLI
  check_submission.py local copy of the submission rules
  make_fake_data.py   synthetic data generator (dev only)
```

## Reproduce

```bash
pip install -r requirements.txt            # + optionally: pip install -r requirements-extra.txt
# data at ./dataset/{train,test}/...

# CV on train (blocking recall, pruning recall, OOF F0.5, decoder tuning, country holdout),
# then train final models and write output/matching_results.tsv + output/candidate_pairs.tsv
python src/pipeline.py run --data dataset --out output

# quick experiment on 20% of train, no test prediction
python src/pipeline.py run --data dataset --frac 0.2 --no-test

# add a multilingual embedding view to blocking (GPU recommended)
python src/pipeline.py run --data dataset --emb st:intfloat/multilingual-e5-small

python src/check_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

Useful flags: `--M` (candidates kept per S1 after stage 1, default 15), `--k-scale` (blocking
breadth), `--same-country`, `--backend lgbm|xgb|hgb`, `--n-est`, `--folds`, `--chunk-s1`.
Env: `BER_NO_TORCH=1`, `BER_NO_RAPIDFUZZ=1` force the fallbacks.

Dev without real data: `python src/make_fake_data.py --out fake_dataset` then `--data fake_dataset`.
