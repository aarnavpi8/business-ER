"""Official evaluation metric: macro-averaged F0.5 over all Source 1 entities."""


def f05(pred: set, true: set) -> float:
    """Per-entity F0.5 following the problem statement.

    - true empty & pred empty  -> 1.0 (correct singleton)
    - true empty & pred any    -> 0.0
    - true non-empty & pred empty -> 0.0
    """
    if not true:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p = tp / len(pred)
    r = tp / len(true)
    return 1.25 * p * r / (0.25 * p + r)


def macro_f05(preds: dict, truth: dict, ids=None) -> float:
    """Average f05 over `ids` (default: all keys of truth)."""
    ids = list(truth.keys()) if ids is None else list(ids)
    if not ids:
        return 0.0
    return sum(f05(preds.get(i, set()), truth.get(i, set())) for i in ids) / len(ids)


def breakdown(preds: dict, truth: dict, ids=None) -> dict:
    """Diagnostic split of the score into singletons vs entities with matches."""
    ids = list(truth.keys()) if ids is None else list(ids)
    sing = [i for i in ids if not truth.get(i)]
    mult = [i for i in ids if truth.get(i)]
    return {
        "all": macro_f05(preds, truth, ids),
        "singletons": macro_f05(preds, truth, sing) if sing else float("nan"),
        "with_matches": macro_f05(preds, truth, mult) if mult else float("nan"),
        "n_single": len(sing), "n_match": len(mult),
    }
