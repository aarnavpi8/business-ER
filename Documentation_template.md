# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]
**Team Members:** [List all team members]
**Submission Date:** [Date]

---

## 1. Executive Summary
We treat the task as **"each Source 2/3 record picks its Source 1 entity, or none"**, which the
training labels show to be exact (every S2/S3 record belongs to at most one S1 entity; no link
crosses countries). A per-country, multi-view dense retrieval (character n-gram TF-IDF compressed by
random projection, exact GPU top-k in both directions) generates candidates; a two-stage
gradient-boosted tree matcher scores them; and a per-record argmax with an F0.5-tuned threshold
produces the matches. Indic-script names are romanised by a rule-based transliterator built from
Unicode character names, and all rules are country-agnostic so the unseen country (France) is
handled like any other.

---

## 2. Methodology

### 2.1 Problem Analysis
Findings from the 2.2M-S1 / 10.3M-S2+S3 training split:
- **One-to-one structure:** 7,638,365 links, none shared by two S1 entities; **0 cross-country links**.
- **Few singletons:** 5.6% of S1 entities have no match; 3.5 matches per S1 on average (up to 11),
  i.e. both S2 and S3 contain internal duplicates of the same business.
- **Distractors:** ~26% of S2/S3 records match no S1 entity.
- **Scripts:** 23% of Indian S2 names and 12% of S3 names are written in native scripts
  (Devanagari, Telugu, Kannada, Tamil, Bengali, Gujarati, Malayalam, Oriya, Gurmukhi) while S1 is
  always Latin (e.g. "ಡೈನಾಮಿಕ್ ಲಾಜಿಸ್ಟಿಕ್ಸ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್" = "Dynamic Logistics Private Limited").
- **Name noise:** legal suffixes anywhere ("LLC Moncada…", "Pvt. EFS … Ltd."), spurious accents on
  English words ("Léarning"), decoration ("-- ", "<< ", "[Center]", "| www.site.com"), word
  reordering ("Smith, Dionis"), duplicated tokens, digit-for-letter typos, acronyms ("SM").
- **Trade names:** ~3–4% of true pairs share no name tokens at all ("Arcjaxonyx" ↔ "Black Imperii
  LLC") and are linked only by address.
- **Address noise:** full vs abbreviated states ("Texas"/"TX"), "CDP"/"City" suffixes, component
  reordering, house-number noise ("01612", "14516d", "22459."), transliterated cities
  (Bombay/Mumbai, Poona/Pune, Calcutta/Kolkata), landmarks ("Near …"), and postcodes almost never
  present (so no postcode features).

### 2.2 Solution Strategy
**Approach Type:** Blocking + two-stage GBDT classifier + per-record assignment (hybrid)
**Core Innovation:** (1) per-S2/S3 decision framing exploiting the verified one-to-one structure,
with "competition" features (how a candidate compares with the record's other S1 options, and
with the S1's other S2/S3 options); (2) random-projection dense retrieval that keeps near-exact
TF-IDF recall at GPU speed; (3) dependency-free Indic romanisation.

---

## 3. Candidate Generation (Blocking)
- **Partitioning:** records are blocked within their country label (string equality — works for
  unseen labels such as France; justified by 0 cross-country training links).
- **Views:** character 2–4-gram TF-IDF (sublinear tf, fitted per country on that country's own
  text) of (a) the core name, (b) the normalised address, (c) "core name | address". Each is
  projected to 256 dims with a Gaussian random projection and L2-normalised.
- **Search:** exact inner-product top-k (PyTorch on CUDA/MPS, tiled over queries and documents;
  numpy fallback), in both directions: S2/S3 → S1 (k=5 per view) and S1 → S2/S3 (k=12 per view).
  The union of all retrieved pairs is the blocking output.
- **Why random projection:** on a full-density regional sample, the true S1 was in the top-10 of the
  "name | address" view for 97.5% of records with RP-256 (98.1% RP-512) vs 98.8% for exact sparse
  TF-IDF, while TruncatedSVD-256 reached only 90.2% (SVD keeps common n-grams and discards the rare
  ones that identify a business).
- **Stage-1 pruning:** a GBDT on cheap features keeps the top-M (M=3) S1 candidates per S2/S3
  record; these pairs are exactly what the final model scores and are written to
  `candidate_pairs.tsv`.
- **Candidate pairs generated:** [total, per S1] — **Blocking recall:** [x] — **after pruning:** [x]
- **How true matches were not lost:** three complementary views (name-only, address-only for trade
  names, combined), both search directions, and recall measured on out-of-fold train predictions
  at every stage.

---

## 4. Matching Model

**Features used:**
- *Retrieval / competition (stage 1):* per-view cosine; gap to the best S1 option of this S2/S3
  record and to its best S2/S3 option of this S1; rank inside the S2/S3 record's candidate list;
  margin between the record's best and second-best S1; candidate counts; lengths; source flag.
- *Name:* Jaro-Winkler, normalised Levenshtein ratio, token-sort / token-set ratios, typo-tolerant
  token matching, Jaccard / overlap of tokens, first-token match, de-spaced JW ("blackimperii"),
  acronym match, exact / core-exact equality.
- *Address:* token-set / sort ratios on the full address and on address words only, token Jaccard /
  overlap, typo-tolerant token match, number-set Jaccard / overlap / conflict, first (house) number
  agreement.
- *Stacked:* stage-1 probability, its rank / gap / margin within the S2/S3 record's list and gap
  within the S1's list.

**Model type:** LightGBM gradient-boosted trees (stage 1 and stage 2), trained on a sample of S1
entities while keeping *every* candidate of each touched S2/S3 record (so competition features have
realistic density); out-of-fold predictions via GroupKFold by S1.
**Threshold selection method:** per-S2/S3 argmax, accepted if p ≥ t (optionally with a margin over
the second-best); t chosen by maximising macro F0.5 on out-of-fold predictions.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro, OOF on train sample):** [x] (singletons [x], entities with matches [x])
- **Country check:** [US x, India x]; train-on-US → test-on-India [x] as a proxy for France.
- **Common false positives (wrong merges):** [fill from notebooks/03_error_analysis]
- **Common false negatives (missed matches):** [fill]

---

## 6. Conclusion
[2–3 sentences]

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/src/` — entry point `pipeline.py`:
`python src/pipeline.py run --data <dataset dir> --out output --cache cache --frac 0.1`
regenerates `output/matching_results.tsv` and `output/candidate_pairs.tsv`. Modules:
`translit.py`, `normalize.py`, `data.py`, `search.py`, `blocking.py`, `features.py`, `models.py`,
`metric.py`, `check_submission.py`. Dependencies (MIT/BSD/Apache): numpy, pandas, scipy,
scikit-learn, rapidfuzz, lightgbm, torch. No pretrained model weights and no external data.

### B. Additional Results
[recall-vs-k plot, threshold curve, feature importance]
