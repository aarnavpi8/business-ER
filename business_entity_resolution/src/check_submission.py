"""Local re-implementation of the submission rules (stdlib only).

The official utils/validate_submission.py should always be run too; this is a
backup that also works on the fake dataset.

Usage: python src/check_submission.py --matching output/matching_results.tsv \
          --candidate output/candidate_pairs.tsv --test-dir dataset/test
"""
import argparse
import csv
import os
import sys


def read_ids(path):
    with open(path, encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        next(r)
        return [row[0] for row in r if row]


def check(path, col, s1_ids, s23_ids):
    errs = []
    seen = set()
    rows = {}
    with open(path, encoding="utf-8") as f:
        lines = f.read().split("\n")
    header = lines[0].split("\t")
    if header != ["source1_entity_id", col]:
        errs.append(f"{path}: bad header {header}")
    for n, line in enumerate(lines[1:], 2):
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            errs.append(f"{path}:{n}: expected 2 tab-separated columns, got {len(parts)}")
            continue
        sid, ids = parts
        if sid in seen:
            errs.append(f"{path}:{n}: duplicate row {sid}")
        seen.add(sid)
        if sid not in s1_ids:
            errs.append(f"{path}:{n}: unknown S1 id {sid}")
        lst = [x for x in ids.split(",") if x] if ids else []
        if len(lst) != len(set(lst)):
            errs.append(f"{path}:{n}: duplicate ids in list")
        for x in lst:
            if not (x.startswith("S2-") or x.startswith("S3-")) or x not in s23_ids:
                errs.append(f"{path}:{n}: invalid id {x}")
        rows[sid] = set(lst)
    missing = s1_ids - seen
    if missing:
        errs.append(f"{path}: {len(missing)} S1 entities missing")
    return errs, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matching", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--test-dir", required=True)
    a = ap.parse_args()
    s1 = set(read_ids(os.path.join(a.test_dir, "test_source1.tsv")))
    s23 = set(read_ids(os.path.join(a.test_dir, "test_source2.tsv"))) | \
        set(read_ids(os.path.join(a.test_dir, "test_source3.tsv")))
    e1, m = check(a.matching, "matched_entity_ids", s1, s23)
    e2, c = check(a.candidate, "candidate_entity_ids", s1, s23)
    errs = e1 + e2
    warn = [k for k in m if not m[k] <= c.get(k, set())]
    for i, e in enumerate(errs[:50], 1):
        print(f"{i}. {e}")
    if warn:
        print(f"WARNING: {len(warn)} rows have matches not in candidates")
    if errs:
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
