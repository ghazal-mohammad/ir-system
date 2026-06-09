import math
import pickle
import os
from collections import defaultdict


def compute_tf(term_freq: dict, doc_length: int) -> dict:
    return {term: freq / doc_length for term, freq in term_freq.items()}


def compute_idf(inverted_index: dict, num_docs: int) -> dict:
    idf = {}
    for term, doc_dict in inverted_index.items():
        df = len(doc_dict)
        idf[term] = math.log((num_docs + 1) / (df + 1)) + 1
    return idf


def build_tfidf_index(inverted_index: dict, doc_lengths: dict) -> dict:
    num_docs = len(doc_lengths)
    idf = compute_idf(inverted_index, num_docs)

    tfidf_index = defaultdict(dict)
    for term, doc_dict in inverted_index.items():
        for doc_id, freq in doc_dict.items():
            dl = doc_lengths.get(doc_id, 1)
            tf = freq / dl
            tfidf_index[term][doc_id] = tf * idf[term]

    return dict(tfidf_index), idf


def compute_query_vector(query_tokens: list, idf: dict) -> dict:
    term_freq = defaultdict(int)
    for t in query_tokens:
        term_freq[t] += 1
    q_len = len(query_tokens)
    q_vec = {}
    for term, freq in term_freq.items():
        if term in idf:
            tf = freq / q_len
            q_vec[term] = tf * idf[term]
    return q_vec


def cosine_similarity(q_vec: dict, doc_vec: dict) -> float:
    dot = sum(q_vec[t] * doc_vec.get(t, 0.0) for t in q_vec)
    norm_q = math.sqrt(sum(v * v for v in q_vec.values()))
    norm_d = math.sqrt(sum(v * v for v in doc_vec.values()))
    if norm_q == 0 or norm_d == 0:
        return 0.0
    return dot / (norm_q * norm_d)


def retrieve_tfidf(query_tokens: list, tfidf_index: dict, idf: dict,
                   doc_lengths: dict, top_k: int = 1000) -> list:
    q_vec = compute_query_vector(query_tokens, idf)
    if not q_vec:
        return []

    candidate_docs = set()
    for term in q_vec:
        if term in tfidf_index:
            candidate_docs.update(tfidf_index[term].keys())

    scores = {}
    for doc_id in candidate_docs:
        doc_vec = {term: tfidf_index[term].get(doc_id, 0.0)
                   for term in q_vec if term in tfidf_index}
        scores[doc_id] = cosine_similarity(q_vec, doc_vec)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    results = []
    for rank, (doc_id, score) in enumerate(ranked[:top_k], start=1):
        results.append({'doc_id': doc_id, 'score': round(score, 6), 'rank': rank})
    return results


def save_tfidf_data(tfidf_index: dict, idf: dict, path_prefix: str):
    with open(f'{path_prefix}_tfidf_index.pkl', 'wb') as f:
        pickle.dump(tfidf_index, f)
    with open(f'{path_prefix}_idf.pkl', 'wb') as f:
        pickle.dump(idf, f)
    size1 = os.path.getsize(f'{path_prefix}_tfidf_index.pkl') / 1024 / 1024
    size2 = os.path.getsize(f'{path_prefix}_idf.pkl') / 1024 / 1024
    print(f'Saved tfidf_index ({size1:.1f} MB) and idf ({size2:.1f} MB)')


def load_tfidf_data(path_prefix: str):
    with open(f'{path_prefix}_tfidf_index.pkl', 'rb') as f:
        tfidf_index = pickle.load(f)
    with open(f'{path_prefix}_idf.pkl', 'rb') as f:
        idf = pickle.load(f)
    return tfidf_index, idf
