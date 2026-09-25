# Business Entity Resolution — Amazon ML Challenge 2026

Per country: **normalise → block (GPU dense search) → stage-1 prune → stage-2 match → per-record decision → output**.

## What the data told us (train, 2.2M S1 / 10.3M S2+S3)
- Every S2/S3 record belongs to **at most one** S1 entity (7.6M links, 0 exceptions) and
  **no link crosses countries** → blocking is done within each country label, and the final
  decision is made per S2/S3 record: "best S1 candidate, if p ≥ t".
- Only 5.6% of S1 are singletons; 3.5 matches per S1 on average (both sources contain duplicates).
- 23% of Indian S2 names (12% S3) are in native scripts (Devanagari, Telugu, Kannada, Tamil,
  Bengali, Gujarati, Malayalam, Oriya, Gurmukhi) → `translit.py` romanises them.
- ~3–4% of true pairs share no name (trade names) → an address-only search view.

## Layout
```
src/
  translit.py         Indic-script romanisation from Unicode names (no external data)
  normalize.py        names / addresses: suffixes, abbreviations, states, numbers, junk
  data.py             TSV loading, parallel normalisation, prepared-frame cache
  search.py           TF-IDF -> random projection encoder; exact top-k (torch CUDA/MPS or numpy)
  blocking.py         per-country 3-view search in both directions (+ best/second stats)
  features.py         chunked cheap features, string features (rapidfuzz or python), stacking
  models.py           GBDT wrapper: lightgbm | xgboost | sklearn
  metric.py           official macro F0.5
  pipeline.py         CLI (train CV + test prediction)
  check_submission.py local copy of the submission rules
  sims.py             pure-Python string similarity fallbacks
  make_fake_data.py   synthetic data generator (dev only)
notebooks/            01 EDA, 02 blocking, 03 error analysis, 04 alias mining (read pipeline cache)
```

## Reproduce
```bash
pip install -r requirements.txt          # + rapidfuzz lightgbm (strongly recommended), torch with CUDA
# DATA = folder containing train/ and test/, e.g. ../6ab10eb3b23ba_student_resource/student_resource/dataset

# CV only (fast iteration): blocking recall, pruning recall, OOF F0.5, threshold tuning
python src/pipeline.py run --data DATA --cache cache --no-test --frac 0.05

# full: CV + final models + test prediction -> output/matching_results.tsv, output/candidate_pairs.tsv
python src/pipeline.py run --data DATA --cache cache --out output --frac 0.1

python src/check_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir DATA/test
```
Blocking and normalised frames are cached in `--cache` (delete the `block_*.npz` / `prep_*.pkl`
files after changing normalisation or blocking code).

Useful flags: `--frac` (share of train S1 used to train the models), `--M` (S1 candidates kept per
S2/S3 record after stage 1, default 3), `--k-scale` (search depth), `--dim` (projection size,
default 256), `--train-countries us` (train on one country to test generalisation),
`--backend lgbm|xgb|hgb`, `--n-est`, `--chunk-rows`.
