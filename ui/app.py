import streamlit as st
import json
import os
import sys
import time

# Make sure services are importable when running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


# ── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title='IR System',
    page_icon='🔍',
    layout='wide',
)

# ── Data paths ───────────────────────────────────────────────────────────────

DATA_DIR = os.environ.get('IR_DATA_DIR', os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'))

DATASETS = {
    'Clinical Trials 2021': 'ct2021',
    'MS MARCO Passage (TREC-DL 2019)': 'msmarco',
}

MODELS = ['BM25', 'TF-IDF', 'Embedding', 'Hybrid Parallel (RRF)', 'Hybrid Serial']

LTR_MODEL_NAMES = ['bm25', 'tfidf', 'embedding']  # must match training order


# ── Caching loaders (read pre-trained files only — no training at runtime) ──

@st.cache_resource(show_spinner='Loading inverted index...')
def get_core(prefix):
    index = load_index(f'{DATA_DIR}/{prefix}_index.pkl')
    with open(f'{DATA_DIR}/{prefix}_doc_lengths.json') as f:
        doc_lengths = json.load(f)
    params = load_bm25_params(f'{DATA_DIR}/{prefix}_bm25_params.pkl')
    vocab = build_vocab_from_index(index)
    return index, doc_lengths, params['avg_dl'], vocab


@st.cache_resource(show_spinner='Loading TF-IDF model...')
def get_tfidf(prefix):
    return load_tfidf_matrix(f'{DATA_DIR}/{prefix}')


@st.cache_resource(show_spinner='Loading embeddings...')
def get_embeddings(prefix):
    # memmap for MSMARCO: matrix is too large to hold fully in RAM
    use_memmap = (prefix == 'msmarco')
    doc_ids, emb = load_embeddings(f'{DATA_DIR}/{prefix}', use_memmap=use_memmap)
    faiss_index = None
    if os.path.exists(f'{DATA_DIR}/{prefix}_faiss.index'):
        faiss_index = load_faiss_index(f'{DATA_DIR}/{prefix}')
    return doc_ids, emb, faiss_index


@st.cache_resource(show_spinner='Loading embedding model...')
def get_model():
    return load_model()


@st.cache_resource(show_spinner='Loading clustering model...')
def get_clustering(prefix):
    labels, km_model = load_clustering(f'{DATA_DIR}/{prefix}')
    doc_ids, _, _ = get_embeddings(prefix)
    doc_to_cluster = {did: int(lbl) for did, lbl in zip(doc_ids, labels)}
    return km_model, doc_to_cluster


@st.cache_resource(show_spinner='Loading LTR model...')
def get_ltr(prefix):
    return load_ltr_model(f'{DATA_DIR}/{prefix}_ltr_model.pkl')


@st.cache_resource(show_spinner='Loading RAG model (first time only)...')
def get_rag():
    return load_rag_model()


def fetch_raw_docs(prefix, doc_ids):
    """Read the original raw documents from the database by doc_id (query time)."""
    conn = db.connect(db.get_db_path(DATA_DIR, prefix))
    docs = db.get_documents(conn, doc_ids)
    conn.close()
    return docs


# ── Retrieval pipeline ───────────────────────────────────────────────────────

def search(query: str, prefix: str, model_name: str, top_k: int,
           use_refinement: bool, use_clustering: bool, use_ltr: bool):
    timings = {}
    info = {}

    t0 = time.time()
    tokens = preprocess(query)
    timings['preprocess'] = time.time() - t0
    if not tokens:
        return [], {}, timings, info

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
        results = retrieve_bm25(tokens, index, doc_lengths, avg_dl, top_k=pool_k)
    elif model_name == 'TF-IDF':
        vectorizer, doc_matrix, tfidf_doc_ids = get_tfidf(prefix)
        results = retrieve_tfidf_fast(processed_query, vectorizer, doc_matrix,
                                      tfidf_doc_ids, top_k=pool_k)
    elif model_name == 'Embedding':
        results = emb_retrieve(pool_k)
    elif model_name == 'Hybrid Parallel (RRF)':
        bm25_res = retrieve_bm25(tokens, index, doc_lengths, avg_dl, top_k=100)
        emb_res = emb_retrieve(100)
        results = hybrid_parallel_two(bm25_res, emb_res, top_k=pool_k)
    else:  # Hybrid Serial
        bm25_res = retrieve_bm25(tokens, index, doc_lengths, avg_dl, top_k=100)
        model = get_model()
        doc_ids, emb, _ = get_embeddings(prefix)
        results = hybrid_serial(query, bm25_res, model, doc_ids, emb,
                                first_stage_k=100, top_k=pool_k)
    timings['retrieval'] = time.time() - t0

    if use_clustering and results:
        if not os.path.exists(f'{DATA_DIR}/{prefix}_km_model.pkl'):
            st.warning(f'Clustering model not found for this dataset — filter skipped.')
            use_clustering = False
    if use_clustering and results:
        t0 = time.time()
        km_model, doc_to_cluster = get_clustering(prefix)
        model = get_model()
        q_emb = model.encode([query], normalize_embeddings=True,
                             convert_to_numpy=True)[0]
        q_cluster = assign_query_to_cluster(q_emb, km_model)
        info['query_cluster'] = q_cluster
        filtered = [r for r in results if doc_to_cluster.get(r['doc_id']) == q_cluster]
        if filtered:
            results = [{**r, 'rank': i + 1} for i, r in enumerate(filtered)]
        timings['clustering'] = time.time() - t0

    if use_ltr and results:
        if not os.path.exists(f'{DATA_DIR}/{prefix}_ltr_model.pkl'):
            st.warning('LTR model not found for this dataset — re-ranking skipped.')
            use_ltr = False
    if use_ltr and results:
        t0 = time.time()
        vectorizer, doc_matrix, tfidf_doc_ids = get_tfidf(prefix)
        bm25_res = retrieve_bm25(tokens, index, doc_lengths, avg_dl, top_k=100)
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
    return results, raw_docs, timings, info


# ── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.title('⚙️ Settings')

dataset_label = st.sidebar.selectbox('Dataset', list(DATASETS.keys()))
prefix = DATASETS[dataset_label]

model_name = st.sidebar.selectbox('Retrieval Model', MODELS)
top_k = st.sidebar.slider('Results to show', min_value=5, max_value=20, value=10, step=5)

st.sidebar.markdown('---')
st.sidebar.subheader('Additional features')
use_refinement = st.sidebar.checkbox('Query refinement (spell + synonyms)')
use_clustering = st.sidebar.checkbox('Clustering filter')
use_ltr = st.sidebar.checkbox('LTR re-ranking')

st.sidebar.markdown('---')
if db.db_exists_and_filled(db.get_db_path(DATA_DIR, prefix)):
    st.sidebar.success('Documents DB connected ✓')
else:
    st.sidebar.error('Documents DB missing.\nRun: `python scripts/build_database.py`')

st.sidebar.markdown('**IR System** — University Project  \nDatasets: CT2021 · MSMARCO')


# ── Main area ────────────────────────────────────────────────────────────────

st.title('🔍 Information Retrieval System')
st.caption(f'Dataset: **{dataset_label}** | Model: **{model_name}**')

tab_search, tab_chat = st.tabs(['🔎 Search', '💬 RAG Chat'])


with tab_search:
    query_input = st.text_input('Enter your search query',
                                placeholder='e.g. diabetes treatment clinical trial')
    search_btn = st.button('Search', type='primary')

    if search_btn and query_input.strip():
        results, raw_docs, timings, info = search(
            query_input.strip(), prefix, model_name, top_k,
            use_refinement, use_clustering, use_ltr)

        st.caption('⏱️ total: **%.2fs** — %s' % (
            timings.get('total', 0),
            ' | '.join(f'{k}: {v:.2f}s' for k, v in timings.items() if k != 'total')))

        if info.get('corrections'):
            st.info('Spell corrections: ' +
                    ', '.join(f'{a} → {b}' for a, b in info['corrections']))
        if info.get('added_synonyms'):
            st.info('Added synonyms: ' + ', '.join(info['added_synonyms']))
        if 'query_cluster' in info:
            st.info(f"Query assigned to cluster #{info['query_cluster']}")

        if not results:
            st.warning('No results found.')
        else:
            st.markdown(f'### Top {len(results)} Results')
            for r in results:
                text = raw_docs.get(r['doc_id'], '(document not found in DB)')
                preview = text[:300] + '...' if len(text) > 300 else text
                col1, col2 = st.columns([6, 1])
                with col1:
                    st.markdown(f"**#{r['rank']}** `{r['doc_id']}`")
                    st.caption(preview)
                    if len(text) > 300:
                        with st.expander('Full document'):
                            st.write(text)
                with col2:
                    st.metric('Score', f"{r['score']:.4f}")
                st.divider()

    elif search_btn:
        st.warning('Please enter a query.')


with tab_chat:
    st.caption(f'Answers are generated from the top documents retrieved by '
               f'**{model_name}** on **{dataset_label}**.')

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg['role']):
            st.write(msg['content'])
            if msg.get('sources'):
                with st.expander('Sources'):
                    for s in msg['sources']:
                        st.write(f'- `{s}`')

    user_msg = st.chat_input('Ask a question about the documents...')
    if user_msg:
        st.session_state.chat_history.append({'role': 'user', 'content': user_msg})
        with st.chat_message('user'):
            st.write(user_msg)

        with st.chat_message('assistant'):
            with st.spinner('Retrieving documents and generating answer...'):
                t0 = time.time()
                results, raw_docs, _, _ = search(
                    user_msg, prefix, model_name, 5,
                    use_refinement, use_clustering, use_ltr)
                rag = generate_answer(user_msg, results, raw_docs,
                                      pipeline=get_rag(), max_docs=5)
                elapsed = time.time() - t0

            st.write(rag['answer'])
            st.caption(f'⏱️ {elapsed:.2f}s')
            if rag['sources']:
                with st.expander('Sources'):
                    for s in rag['sources']:
                        st.write(f'- `{s}`')

        st.session_state.chat_history.append({
            'role': 'assistant',
            'content': rag['answer'],
            'sources': rag['sources'],
        })
