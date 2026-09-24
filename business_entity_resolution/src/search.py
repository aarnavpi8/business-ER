"""Dense encoders and exact top-k inner-product search with GPU acceleration.

Backends (auto-detected, best first): torch CUDA -> torch MPS (Apple M-series)
-> numpy on CPU. All vectors are L2-normalised float32, so inner product is
cosine similarity. Search is exact (brute force in chunks): on a laptop GPU a
1M x 1M search at 256 dims takes a few minutes; on CPU it is slower but works.

Encoders:
  * "svd": char n-gram TF-IDF -> TruncatedSVD. No pretrained weights at all.
  * "st:<model>": a sentence-transformers model, e.g.
        st:intfloat/multilingual-e5-small              (MIT)
        st:sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (Apache-2.0)
    Only used if sentence-transformers is installed. Must be MIT/Apache and
    <= 8B params per the challenge rules.
"""
import os
import numpy as np


def _torch_device():
    """Return (torch module, device string) or (None, 'cpu') if torch is absent."""
    if os.environ.get("BER_NO_TORCH"):
        return None, "cpu"
    try:
        import torch
    except Exception:
        return None, "cpu"
    if torch.cuda.is_available():
        return torch, "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch, "mps"
    return torch, "cpu"


def backend_name() -> str:
    t, dev = _torch_device()
    return f"torch-{dev}" if t is not None else "numpy-cpu"


def l2norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (x / n).astype(np.float32)


def encode_svd(fit_texts, all_text_lists, dim=256, max_fit=150_000, seed=0):
    """Fit char TF-IDF + TruncatedSVD on (a sample of) fit_texts, transform each list.

    Returns a list of L2-normalised float32 arrays, one per entry of all_text_lists.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    rng = np.random.default_rng(seed)
    fit_texts = np.asarray(fit_texts, dtype=object)
    if len(fit_texts) > max_fit:
        fit_texts = fit_texts[rng.choice(len(fit_texts), max_fit, replace=False)]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2,
                          max_features=300_000, sublinear_tf=True, dtype=np.float32)
    M = vec.fit_transform(fit_texts)
    dim = min(dim, M.shape[1] - 1, M.shape[0] - 1)
    svd = TruncatedSVD(n_components=dim, algorithm="randomized", n_iter=4, random_state=seed)
    svd.fit(M)
    out = []
    for texts in all_text_lists:
        parts = []
        for st in range(0, len(texts), 200_000):
            parts.append(svd.transform(vec.transform(texts[st:st + 200_000])))
        out.append(l2norm(np.vstack(parts)))
    return out


def encode_st(model_name, all_text_lists, batch_size=256):
    """Encode with a sentence-transformers model on the best available device."""
    from sentence_transformers import SentenceTransformer
    torch, dev = _torch_device()
    model = SentenceTransformer(model_name, device=dev)
    if dev == "cuda":
        model = model.half()
    prefix = "query: " if "e5" in model_name.lower() else ""
    out = []
    for texts in all_text_lists:
        emb = model.encode([prefix + t for t in texts], batch_size=batch_size,
                           convert_to_numpy=True, normalize_embeddings=True,
                           show_progress_bar=len(texts) > 50_000)
        out.append(emb.astype(np.float32))
    return out


def encode(kind, fit_texts, all_text_lists, dim=256):
    """Dispatch to the encoder named by `kind` ("svd" or "st:<model>")."""
    if kind == "svd":
        return encode_svd(fit_texts, all_text_lists, dim=dim)
    if kind.startswith("st:"):
        return encode_st(kind[3:], all_text_lists)
    raise ValueError(kind)


def _topk_torch(torch, dev, Q, D, k, chunk):
    Dt = torch.from_numpy(D).to(dev)
    if dev == "cuda":
        Dt = Dt.half()
    idx_out = np.empty((len(Q), k), np.int64)
    sim_out = np.empty((len(Q), k), np.float32)
    for st in range(0, len(Q), chunk):
        q = torch.from_numpy(Q[st:st + chunk]).to(dev)
        if dev == "cuda":
            q = q.half()
        s = q @ Dt.T
        v, i = torch.topk(s, k, dim=1)
        idx_out[st:st + chunk] = i.cpu().numpy()
        sim_out[st:st + chunk] = v.float().cpu().numpy()
    return idx_out, sim_out


def _topk_numpy(Q, D, k, budget=150_000_000):
    rows = max(1, budget // max(len(D), 1))
    idx_out = np.empty((len(Q), k), np.int64)
    sim_out = np.empty((len(Q), k), np.float32)
    for st in range(0, len(Q), rows):
        s = Q[st:st + rows] @ D.T
        i = np.argpartition(-s, k - 1, axis=1)[:, :k]
        v = np.take_along_axis(s, i, axis=1)
        o = np.argsort(-v, axis=1)
        idx_out[st:st + rows] = np.take_along_axis(i, o, axis=1)
        sim_out[st:st + rows] = np.take_along_axis(v, o, axis=1)
    return idx_out, sim_out


def topk(Q, D, k, q_groups=None, d_groups=None, chunk=4096):
    """Exact top-k by inner product. Returns (idx, sim), each (len(Q), k).

    If groups are given, each query only searches documents of its own group
    (used for optional same-country blocking). Missing slots get idx=-1, sim=-1.
    """
    torch, dev = _torch_device()
    idx = np.full((len(Q), k), -1, np.int64)
    sim = np.full((len(Q), k), -1.0, np.float32)
    if q_groups is None:
        q_groups = np.zeros(len(Q), dtype=object)
        d_groups = np.zeros(len(D), dtype=object)
    for g in np.unique(q_groups):
        qi = np.where(q_groups == g)[0]
        di = np.where(d_groups == g)[0]
        if len(qi) == 0 or len(di) == 0:
            continue
        kk = min(k, len(di))
        if torch is not None:
            i, s = _topk_torch(torch, dev, Q[qi], D[di], kk, chunk)
        else:
            i, s = _topk_numpy(Q[qi], D[di], kk)
        idx[qi, :kk] = di[i]
        sim[qi, :kk] = s
    return idx, sim


def sparse_topk(A, B, k, chunk=2000):
    """Top-k for sparse L2-normalised rows (A @ B.T); used for word-token views.

    Efficient when very common tokens were removed (max_df), since the product
    then stays sparse.
    """
    Bt = B.T.tocsr()
    n = A.shape[0]
    idx = np.full((n, k), -1, np.int64)
    sim = np.full((n, k), -1.0, np.float32)
    for st in range(0, n, chunk):
        P = (A[st:st + chunk] @ Bt).tocsr()
        for r in range(P.shape[0]):
            a, b = P.indptr[r], P.indptr[r + 1]
            if a == b:
                continue
            d = P.data[a:b]
            c = P.indices[a:b]
            if len(d) > k:
                o = np.argpartition(-d, k - 1)[:k]
                d, c = d[o], c[o]
            o = np.argsort(-d)
            idx[st + r, :len(o)] = c[o]
            sim[st + r, :len(o)] = d[o]
    return idx, sim
