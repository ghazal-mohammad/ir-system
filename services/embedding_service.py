import numpy as np
import pickle
import os


MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'


def load_model():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME)
    return model


def encode_texts(texts: list, model, batch_size: int = 128, show_progress: bool = True) -> np.ndarray:
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return embeddings


def build_doc_embeddings(doc_ids: list, doc_texts: list, model,
                         batch_size: int = 128) -> tuple:
    print(f'Encoding {len(doc_ids):,} documents...')
    embeddings = encode_texts(doc_texts, model, batch_size=batch_size)
    return doc_ids, embeddings


def retrieve_embedding(query_text: str, model, doc_ids: list,
                       doc_embeddings: np.ndarray, top_k: int = 1000) -> list:
    q_emb = model.encode([query_text], normalize_embeddings=True, convert_to_numpy=True)
    scores = (doc_embeddings @ q_emb.T).squeeze()
    # argpartition is O(n) instead of a full O(n log n) sort
    k = min(top_k, len(scores))
    top_indices = np.argpartition(scores, -k)[-k:]
    top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
    results = []
    for rank, idx in enumerate(top_indices, start=1):
        results.append({
            'doc_id': doc_ids[idx],
            'score': round(float(scores[idx]), 6),
            'rank': rank
        })
    return results


def save_embeddings(doc_ids: list, embeddings: np.ndarray, path_prefix: str):
    np.save(f'{path_prefix}_embeddings.npy', embeddings)
    with open(f'{path_prefix}_emb_doc_ids.pkl', 'wb') as f:
        pickle.dump(doc_ids, f)
    size_mb = os.path.getsize(f'{path_prefix}_embeddings.npy') / 1024 / 1024
    print(f'Saved embeddings ({size_mb:.1f} MB) and doc_ids')


def load_embeddings(path_prefix: str, use_memmap: bool = False) -> tuple:
    """
    use_memmap=True reads the matrix lazily from disk instead of RAM —
    needed for very large collections (e.g. MSMARCO) on limited memory.
    """
    mmap_mode = 'r' if use_memmap else None
    embeddings = np.load(f'{path_prefix}_embeddings.npy', mmap_mode=mmap_mode)
    with open(f'{path_prefix}_emb_doc_ids.pkl', 'rb') as f:
        doc_ids = pickle.load(f)
    return doc_ids, embeddings


# ── FAISS index (fast approximate search for large collections) ─────────────

def build_faiss_index(doc_embeddings: np.ndarray, nlist: int = 256):
    """
    Build an IVF index: clusters the vectors so a query only scans
    a few cells instead of the full collection.
    """
    import faiss
    dim = doc_embeddings.shape[1]
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    train_sample = np.array(doc_embeddings[:min(200000, len(doc_embeddings))],
                            dtype=np.float32)
    index.train(train_sample)
    # add in chunks so a memmapped matrix never fully loads into RAM
    chunk = 100000
    for start in range(0, len(doc_embeddings), chunk):
        block = np.array(doc_embeddings[start:start + chunk], dtype=np.float32)
        index.add(np.ascontiguousarray(block))
    index.nprobe = 16
    return index


def save_faiss_index(index, path_prefix: str):
    import faiss
    faiss.write_index(index, f'{path_prefix}_faiss.index')
    size_mb = os.path.getsize(f'{path_prefix}_faiss.index') / 1024 / 1024
    print(f'FAISS index saved ({size_mb:.1f} MB)')


def load_faiss_index(path_prefix: str):
    import faiss
    return faiss.read_index(f'{path_prefix}_faiss.index')


def retrieve_embedding_faiss(query_text: str, model, doc_ids: list,
                             faiss_index, top_k: int = 1000) -> list:
    q_emb = model.encode([query_text], normalize_embeddings=True, convert_to_numpy=True)
    scores, indices = faiss_index.search(
        np.ascontiguousarray(q_emb, dtype=np.float32), top_k)
    results = []
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
        if idx == -1:
            continue
        results.append({
            'doc_id': doc_ids[idx],
            'score': round(float(score), 6),
            'rank': rank
        })
    return results
