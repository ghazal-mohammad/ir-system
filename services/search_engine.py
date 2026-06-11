"""
Search engine core — orchestrates all services into one query pipeline.

This module is the single entry point used by BOTH the API Gateway
(api/main.py) and the Streamlit UI (ui/app.py), so the retrieval logic
lives in exactly one place (reusability / loose coupling).

Everything is loaded lazily from pre-trained files and cached in memory —
no training ever happens at query time.
"""
import os
import json
import time

from services.preprocessing_service import preprocess
from services import database_service as db
from services.indexing_service import load_index
from services.tfidf_service import load_tfidf_matrix, retrieve_tfidf_fast
from services.bm25_service import retrieve_bm25, load_bm25_params
from services.embedding_service import (
    load_model, load_embeddings, retrieve_embedding,
    load_faiss_index, retrieve_embedding_faiss
)
from services.hybrid_service import hybrid_parallel_two, hybrid_serial
from services.query_refinement_service import build_vocab_from_index, refine_query
from services.clustering_service import load_clustering, assign_query_to_cluster
from services.ltr_service import load_ltr_model, rerank_with_ltr
from services.rag_service import load_rag_model, generate_answer


DATA_DIR = os.environ.get('IR_DATA_DIR', os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'))

MODELS = ['BM25', 'TF-IDF', 'Embedding', 'Hybrid Parallel (RRF)', 'Hybrid Serial']

LTR_MODEL_NAMES = ['bm25', 'tfidf', 'embedding']  # must match training order

_cache = {}


# ── Lazy cached loaders ──────────────────────────────────────────────────────

def _get(key, loader):
    if key not in _cache:
        _cache[key] = loader()
    return _cache[key]


def get_core(prefix):
    def load():
        index = load_index(f'{DATA_DIR}/{prefix}_index.pkl')
        with open(f'{DATA_DIR}/{prefix}_doc_lengths.json') as f:
            doc_lengths = json.load(f)
        params = load_bm25_params(f'{DATA_DIR}/{prefix}_bm25_params.pkl')
        vocab = build_vocab_from_index(index)
        return index, doc_lengths, params['avg_dl'], vocab
    return _get(('core', prefix), load)


def get_tfidf(prefix):
    return _get(('tfidf', prefix), lambda: load_tfidf_matrix(f'{DATA_DIR}/{prefix}'))


def get_model():
    return _get(('emb_model',), load_model)


def get_embeddings(prefix):
    def load():
        use_memmap = (prefix == 'msmarco')  # too large to hold fully in RAM
        doc_ids, emb = load_embeddings(f'{DATA_DIR}/{prefix}', use_memmap=use_memmap)
        faiss_index = None
        if os.path.exists(f'{DATA_DIR}/{prefix}_faiss.index'):
            faiss_index = load_faiss_index(f'{DATA_DIR}/{prefix}')
        return doc_ids, emb, faiss_index
    return _get(('embeddings', prefix), load)


def get_clustering(prefix):
    def load():
        labels, km_model = load_clustering(f'{DATA_DIR}/{prefix}')
        doc_ids, _, _ = get_embeddings(prefix)
        doc_to_cluster = {did: int(lbl) for did, lbl in zip(doc_ids, labels)}
        return km_model, doc_to_cluster
    return _get(('clustering', prefix), load)


def get_ltr(prefix):
    return _get(('ltr', prefix),
                lambda: load_ltr_model(f'{DATA_DIR}/{prefix}_ltr_model.pkl'))


def get_rag():
    return _get(('rag',), load_rag_model)


# ── Document store (DB read at query time) ───────────────────────────────────

def fetch_raw_docs(prefix, doc_ids):
    """Read the original raw documents from the database by doc_id."""
    conn = db.connect(db.get_db_path(DATA_DIR, prefix))
    docs = db.get_documents(conn, doc_ids)
    conn.close()
    return docs


def get_document(prefix, doc_id):
    conn = db.connect(db.get_db_path(DATA_DIR, prefix))
    text = db.get_document(conn, doc_id)
    conn.close()
    return text


def db_status(prefix):
    return db.db_exists_and_filled(db.get_db_path(DATA_DIR, prefix))


# ── Query pipeline ───────────────────────────────────────────────────────────

def search(query: str, prefix: str, model_name: str, top_k: int = 10,
           use_refinement: bool = False, use_clustering: bool = False,
           use_ltr: bool = False, k1: float = 1.5, b: float = 0.75) -> dict:
    """
    Full retrieval pipeline. Returns a dict:
    { results, raw_docs, timings, info, warnings }
    """
    timings = {}
    info = {}
    warnings = []

    t0 = time.time()
    tokens = preprocess(query)
    timings['preprocess'] = time.time() - t0
    if not tokens:
        return {'results': [], 'raw_docs': {}, 'timings': timings,
                'info': info, 'warnings': ['Query is empty after preprocessing.']}

    index, doc_lengths, avg_dl, vocab = get_core(prefix)

    if use_refinement:
        t0 = time.time()
        refined = refine_query(tokens, vocab, index, use_spell=True, use_expand=True)
        tokens = refined['expanded_tokens']
        info['corrections'] = refined['corrections']
        info['added_synonyms'] = refined['added_synonyms']
        timings['refinement'] = time.time() - t0

    processed_query = ' '.join(tokens)
    # wider candidate pool when a filter/re-ranker runs afterwards
    pool_k = 100 if (use_clustering or use_ltr) else top_k

    def emb_retrieve(k):
        model = get_model()
        doc_ids, emb, faiss_index = get_embeddings(prefix)
        if faiss_index is not None:
            return retrieve_embedding_faiss(query, model, doc_ids, faiss_index, top_k=k)
        return retrieve_embedding(query, model, doc_ids, emb, top_k=k)

    t0 = time.time()
    if model_name == 'BM25':
        results = retrieve_bm25(tokens, index, doc_lengths, avg_dl,
                                top_k=pool_k, k1=k1, b=b)
    elif model_name == 'TF-IDF':
        vectorizer, doc_matrix, tfidf_doc_ids = get_tfidf(prefix)
        results = retrieve_tfidf_fast(processed_query, vectorizer, doc_matrix,
                                      tfidf_doc_ids, top_k=pool_k)
    elif model_name == 'Embedding':
        results = emb_retrieve(pool_k)
    elif model_name == 'Hybrid Parallel (RRF)':
        bm25_res = retrieve_bm25(tokens, index, doc_lengths, avg_dl,
                                 top_k=100, k1=k1, b=b)
        emb_res = emb_retrieve(100)
        results = hybrid_parallel_two(bm25_res, emb_res, top_k=pool_k)
    elif model_name == 'Hybrid Serial':
        bm25_res = retrieve_bm25(tokens, index, doc_lengths, avg_dl,
                                 top_k=100, k1=k1, b=b)
        model = get_model()
        doc_ids, emb, _ = get_embeddings(prefix)
        results = hybrid_serial(query, bm25_res, model, doc_ids, emb,
                                first_stage_k=100, top_k=pool_k)
    else:
        raise ValueError(f'Unknown model: {model_name}')
    timings['retrieval'] = time.time() - t0

    if use_clustering and results:
        if not os.path.exists(f'{DATA_DIR}/{prefix}_km_model.pkl'):
            warnings.append('Clustering model not found for this dataset — filter skipped.')
        else:
            t0 = time.time()
            km_model, doc_to_cluster = get_clustering(prefix)
            model = get_model()
            q_emb = model.encode([query], normalize_embeddings=True,
                                 convert_to_numpy=True)[0]
            q_cluster = assign_query_to_cluster(q_emb, km_model)
            info['query_cluster'] = q_cluster
            filtered = [r for r in results
                        if doc_to_cluster.get(r['doc_id']) == q_cluster]
            if filtered:
                results = [{**r, 'rank': i + 1} for i, r in enumerate(filtered)]
            timings['clustering'] = time.time() - t0

    if use_ltr and results:
        if not os.path.exists(f'{DATA_DIR}/{prefix}_ltr_model.pkl'):
            warnings.append('LTR model not found for this dataset — re-ranking skipped.')
        else:
            t0 = time.time()
            vectorizer, doc_matrix, tfidf_doc_ids = get_tfidf(prefix)
            bm25_res = retrieve_bm25(tokens, index, doc_lengths, avg_dl,
                                     top_k=100, k1=k1, b=b)
            tfidf_res = retrieve_tfidf_fast(processed_query, vectorizer, doc_matrix,
                                            tfidf_doc_ids, top_k=100)
            emb_res = emb_retrieve(100)
            results_per_model = {
                'bm25': {'q': bm25_res},
                'tfidf': {'q': tfidf_res},
                'embedding': {'q': emb_res},
            }
            results = rerank_with_ltr('q', results_per_model, get_ltr(prefix),
                                      LTR_MODEL_NAMES, top_k=pool_k)
            timings['ltr_rerank'] = time.time() - t0

    results = results[:top_k]

    # read original documents from the DB by id — only the top results
    t0 = time.time()
    raw_docs = fetch_raw_docs(prefix, [r['doc_id'] for r in results])
    timings['db_fetch'] = time.time() - t0

    timings['total'] = sum(timings.values())
    return {'results': results, 'raw_docs': raw_docs, 'timings': timings,
            'info': info, 'warnings': warnings}


def rag_answer(query: str, prefix: str, model_name: str = 'BM25',
               use_refinement: bool = False, use_clustering: bool = False,
               use_ltr: bool = False, k1: float = 1.5, b: float = 0.75) -> dict:
    """Retrieve top documents then generate an answer (RAG)."""
    t0 = time.time()
    out = search(query, prefix, model_name, top_k=5,
                 use_refinement=use_refinement, use_clustering=use_clustering,
                 use_ltr=use_ltr, k1=k1, b=b)
    rag = generate_answer(query, out['results'], out['raw_docs'],
                          pipeline=get_rag(), max_docs=5)
    rag['retrieved_docs'] = out['results']
    rag['elapsed'] = round(time.time() - t0, 3)
    return rag
