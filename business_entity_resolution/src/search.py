"""Dense encoders and exact top-k inner-product search with GPU acceleration.

Backends (auto-detected, best first): torch CUDA -> torch MPS (Apple M-series)
-> numpy on CPU. All vectors are L2-normalised float32, so inner product is
cosine similarity. Search is exact (brute force in chunks): on a laptop GPU a
1M x 1M search at 256 dims takes a few minutes; on CPU it is slower but works.

Encoders:
  * "rp" : char n-gram TF-IDF -> Gaussian random projection (default; best recall).
  * "svd": char n-gram TF-IDF -> TruncatedSVD (kept for comparison; loses rare n-grams).
  * "st:<model>": a sentence-transformers model, e.g.
        st:intfloat/multilingual-e5-small              (MIT)
        st:sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (Apache-2.0)
    Only used if sentence-transformers is installed. Must be MIT/Apache and
    <= 8B params per the challenge rules.
"""
import os
import tempfile
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


def _rp_worker_init(vec, R):
    """R is the projection matrix or the path of a .npy copy (memory-mapped, so workers share it)."""
    global _RP_VEC, _RP_R
    _RP_VEC, _RP_R = vec, (np.load(R, mmap_mode="r") if isinstance(R, str) else R)


def _rp_worker(texts):
    # stored as float16: halves RAM (6M x 256 -> 3 GB) with ~1e-3 cosine error
    return l2norm(np.asarray(_RP_VEC.transform(texts) @ _RP_R, np.float32)).astype(np.float16)


def encode_rp(fit_texts, all_text_lists, dim=256, max_fit=1_000_000, seed=0, n_jobs=None):
    """Char n-gram TF-IDF -> Gaussian random projection (L2-normalised).

    Unlike SVD, a random projection preserves cosine similarity between the
    full sparse TF-IDF vectors (error ~ 1/sqrt(dim)), including the rare
    n-grams that actually identify a business. On a real-density regional
    sample: recall@10 = 0.975 (dim 256) / 0.981 (dim 512) vs 0.988 exact and
    0.902 for SVD-256. No pretrained weights; fitted per split/country.
    """
    import multiprocessing as mp
    from sklearn.feature_extraction.text import TfidfVectorizer
    rng = np.random.default_rng(seed)
    fit_texts = np.asarray(fit_texts, dtype=object)
    if len(fit_texts) > max_fit:
        fit_texts = fit_texts[rng.choice(len(fit_texts), max_fit, replace=False)]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2,
                          max_features=400_000, sublinear_tf=True, dtype=np.float32)
    vec.fit(fit_texts)
    R = (rng.standard_normal((len(vec.vocabulary_), dim)) / np.sqrt(dim)).astype(np.float32)
    # R is up to 400k x dim float32 (~400 MB): workers memory-map one on-disk copy instead of
    # each unpickling their own, and the pool is capped because each worker also holds the vocabulary
    n_jobs = n_jobs or min(8, max(1, (os.cpu_count() or 2) - 1))
    lists = [np.asarray(t, dtype=object) for t in all_text_lists]
    index = [(li, i) for li, t in enumerate(lists) for i in range(0, len(t), 100_000)]
    chunks = [lists[li][i:i + 100_000] for li, i in index]
    out = [np.empty((len(t), dim), np.float16) for t in lists]
    if n_jobs > 1 and len(chunks) > 2:
        fd, r_path = tempfile.mkstemp(suffix=".npy")
        os.close(fd)
        np.save(r_path, R)
        del R
        try:
            with mp.get_context("spawn").Pool(min(n_jobs, len(chunks)), _rp_worker_init, (vec, r_path)) as pool:
                for (li, i), res in zip(index, pool.imap(_rp_worker, chunks)):
                    out[li][i:i + len(res)] = res
        finally:
            try:
                os.remove(r_path)
            except OSError:
                pass
    else:
        _rp_worker_init(vec, R)
        for (li, i), c in zip(index, chunks):
            out[li][i:i + len(c)] = _rp_worker(c)
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
    if kind == "rp":
        return encode_rp(fit_texts, all_text_lists, dim=dim)
    if kind == "svd":
        return encode_svd(fit_texts, all_text_lists, dim=dim)
    if kind.startswith("st:"):
        return encode_st(kind[3:], all_text_lists)
    raise ValueError(kind)


def _topk_torch(torch, dev, Q, D, k, q_chunk=None, d_tile=None, mem_bytes=1.5e9):
    """Exact top-k on GPU, tiled over queries and documents to bound memory.

    D is uploaded once (fp16 on CUDA); each (query chunk x doc tile) similarity
    block is at most ~mem_bytes; per-tile top-k results are merged on device.
    """
    half = dev == "cuda"
    bpe = 2 if half else 4
    Dt = torch.from_numpy(np.ascontiguousarray(D)).to(dev)
    Dt = Dt.half() if half else Dt.float()
    n_d = len(D)
    d_tile = d_tile or min(n_d, 1_000_000)
    q_chunk = q_chunk or max(64, int(mem_bytes // (d_tile * bpe)))
    idx_out = np.empty((len(Q), k), np.int64)
    sim_out = np.empty((len(Q), k), np.float32)
    for st in range(0, len(Q), q_chunk):
        q = torch.from_numpy(np.ascontiguousarray(Q[st:st + q_chunk])).to(dev)
        q = q.half() if half else q.float()
        best_v = best_i = None
        for dt in range(0, n_d, d_tile):
            s = q @ Dt[dt:dt + d_tile].T
            kk = min(k, s.shape[1])
            v, i = torch.topk(s, kk, dim=1)
            i = i + dt
            if best_v is None:
                best_v, best_i = v, i
            else:
                cv = torch.cat([best_v, v], 1)
                ci = torch.cat([best_i, i], 1)
                v2, o = torch.topk(cv, min(k, cv.shape[1]), dim=1)
                best_v, best_i = v2, torch.gather(ci, 1, o)
            del s
        kk = best_v.shape[1]
        idx_out[st:st + q_chunk, :kk] = best_i.cpu().numpy()
        sim_out[st:st + q_chunk, :kk] = best_v.float().cpu().numpy()
        if kk < k:
            idx_out[st:st + q_chunk, kk:] = -1
            sim_out[st:st + q_chunk, kk:] = -1
    del Dt
    if dev == "cuda":
        torch.cuda.empty_cache()
    return idx_out, sim_out


def _topk_numpy(Q, D, k, budget=40_000_000):
    D = np.asarray(D, np.float32)
    rows = max(1, budget // max(len(D), 1))
    idx_out = np.empty((len(Q), k), np.int64)
    sim_out = np.empty((len(Q), k), np.float32)
    for st in range(0, len(Q), rows):
        s = np.asarray(Q[st:st + rows], np.float32) @ D.T
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
            i, s = _topk_torch(torch, dev, Q[qi], D[di], kk)
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
