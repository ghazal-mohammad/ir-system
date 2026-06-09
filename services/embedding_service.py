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
    top_indices = np.argsort(scores)[::-1][:top_k]
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


def load_embeddings(path_prefix: str) -> tuple:
    embeddings = np.load(f'{path_prefix}_embeddings.npy')
    with open(f'{path_prefix}_emb_doc_ids.pkl', 'rb') as f:
        doc_ids = pickle.load(f)
    return doc_ids, embeddings
