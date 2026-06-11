import math
import pickle
import os
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import scipy.sparse as sp


def compute_idf(inverted_index: dict, num_docs: int) -> dict:
    idf = {}
    for term, doc_dict in inverted_index.items():
        df = len(doc_dict)
        idf[term] = math.log((num_docs + 1) / (df + 1)) + 1
    return idf


# ── Fast sklearn-based TF-IDF (recommended for large collections) ────────────

def build_tfidf_matrix(processed_docs: dict) -> tuple:
    """
    Build a TF-IDF sparse matrix using sklearn.
    Returns: (vectorizer, doc_matrix, doc_ids)
    doc_matrix is L2-normalised for fast cosine similarity via dot product.
    """
    doc_ids = list(processed_docs.keys())
    doc_texts = list(processed_docs.values())

    vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        norm='l2',
        min_df=2,
        max_df=0.95,
        dtype=np.float32,
    )
    doc_matrix = vectorizer.fit_transform(doc_texts)
    return vectorizer, doc_matrix, doc_ids


def retrieve_tfidf_fast(query_text: str, vectorizer, doc_matrix,
                         doc_ids: list, top_k: int = 1000) -> list:
    q_vec = vectorizer.transform([query_text])
    scores = (doc_matrix @ q_vec.T).toarray().squeeze()
    # argpartition is O(n) instead of a full O(n log n) sort
    k = min(top_k, len(scores))
    top_indices = np.argpartition(scores, -k)[-k:]
    top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
    results = []
    for rank, idx in enumerate(top_indices, start=1):
        if scores[idx] == 0:
            break
        results.append({'doc_id': doc_ids[idx], 'score': round(float(scores[idx]), 6), 'rank': rank})
    return results


def save_tfidf_matrix(vectorizer, doc_matrix, doc_ids: list, path_prefix: str):
    with open(f'{path_prefix}_tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    sp.save_npz(f'{path_prefix}_tfidf_matrix.npz', doc_matrix)
    with open(f'{path_prefix}_tfidf_doc_ids.pkl', 'wb') as f:
        pickle.dump(doc_ids, f)
    size = os.path.getsize(f'{path_prefix}_tfidf_matrix.npz') / 1024 / 1024
    print(f'TF-IDF matrix saved ({size:.1f} MB), {doc_matrix.shape[0]:,} docs × {doc_matrix.shape[1]:,} terms')


def load_tfidf_matrix(path_prefix: str) -> tuple:
    with open(f'{path_prefix}_tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    doc_matrix = sp.load_npz(f'{path_prefix}_tfidf_matrix.npz')
    with open(f'{path_prefix}_tfidf_doc_ids.pkl', 'rb') as f:
        doc_ids = pickle.load(f)
    return vectorizer, doc_matrix, doc_ids


# ── Legacy inverted-index-based TF-IDF (kept for SOA demo) ──────────────────

def build_tfidf_index(inverted_index: dict, doc_lengths: dict) -> tuple:
    num_docs = len(doc_lengths)
    idf = compute_idf(inverted_index, num_docs)
    tfidf_index = defaultdict(dict)
    for term, doc_dict in inverted_index.items():
        for doc_id, freq in doc_dict.items():
            dl = doc_lengths.get(doc_id, 1)
            tf = freq / dl
            tfidf_index[term][doc_id] = tf * idf[term]
    return dict(tfidf_index), idf


def retrieve_tfidf(query_tokens: list, tfidf_index: dict, idf: dict,
                   doc_lengths: dict, top_k: int = 1000) -> list:
    term_freq = defaultdict(int)
    for t in query_tokens:
        term_freq[t] += 1
    q_len = max(len(query_tokens), 1)

    q_vec = {}
    for term, freq in term_freq.items():
        if term in idf:
            q_vec[term] = (freq / q_len) * idf[term]
    if not q_vec:
        return []

    candidate_docs = set()
    for term in q_vec:
        if term in tfidf_index:
            candidate_docs.update(tfidf_index[term].keys())

    norm_q = math.sqrt(sum(v * v for v in q_vec.values()))
    scores = {}
    for doc_id in candidate_docs:
        dot = sum(q_vec[t] * tfidf_index[t].get(doc_id, 0.0)
                  for t in q_vec if t in tfidf_index)
        if dot > 0:
            scores[doc_id] = dot / norm_q

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [{'doc_id': did, 'score': round(s, 6), 'rank': i + 1}
            for i, (did, s) in enumerate(ranked[:top_k])]


def save_tfidf_data(tfidf_index: dict, idf: dict, path_prefix: str):
    with open(f'{path_prefix}_tfidf_index.pkl', 'wb') as f:
        pickle.dump(tfidf_index, f)
    with open(f'{path_prefix}_idf.pkl', 'wb') as f:
        pickle.dump(idf, f)
    size1 = os.path.getsize(f'{path_prefix}_tfidf_index.pkl') / 1024 / 1024
    print(f'Saved tfidf_index ({size1:.1f} MB)')


def load_tfidf_data(path_prefix: str) -> tuple:
    with open(f'{path_prefix}_tfidf_index.pkl', 'rb') as f:
        tfidf_index = pickle.load(f)
    with open(f'{path_prefix}_idf.pkl', 'rb') as f:
        idf = pickle.load(f)
    return tfidf_index, idf
