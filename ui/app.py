import streamlit as st
import json
import pickle
import os
import sys
import numpy as np

# Make sure services are importable when running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.preprocessing_service import preprocess
from services.indexing_service import load_index, get_avg_doc_length
from services.tfidf_service import load_tfidf_data, retrieve_tfidf
from services.bm25_service import retrieve_bm25, load_bm25_params
from services.embedding_service import load_model, load_embeddings, retrieve_embedding
from services.hybrid_service import hybrid_parallel, hybrid_serial


# ── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title='IR System',
    page_icon='🔍',
    layout='wide',
)

# ── Data paths ───────────────────────────────────────────────────────────────

DATA_DIR = os.environ.get('IR_DATA_DIR', '/content/drive/MyDrive/ir_system_data')

DATASETS = {
    'Clinical Trials 2021': 'ct2021',
    'MS MARCO Passage (TREC-DL 2019)': 'msmarco',
}

MODELS = ['TF-IDF', 'BM25', 'Embedding', 'Hybrid Parallel (RRF)', 'Hybrid Serial']


# ── Caching loaders ──────────────────────────────────────────────────────────

@st.cache_resource(show_spinner='Loading index...')
def get_index(prefix):
    return load_index(f'{DATA_DIR}/{prefix}_index.pkl')


@st.cache_resource(show_spinner='Loading doc lengths...')
def get_doc_lengths(prefix):
    with open(f'{DATA_DIR}/{prefix}_doc_lengths.json') as f:
        return json.load(f)


@st.cache_resource(show_spinner='Loading TF-IDF data...')
def get_tfidf(prefix):
    return load_tfidf_data(f'{DATA_DIR}/{prefix}')


@st.cache_resource(show_spinner='Loading BM25 params...')
def get_bm25_params(prefix):
    return load_bm25_params(f'{DATA_DIR}/{prefix}_bm25_params.pkl')


@st.cache_resource(show_spinner='Loading embeddings (this may take a minute)...')
def get_embeddings(prefix):
    return load_embeddings(f'{DATA_DIR}/{prefix}')


@st.cache_resource(show_spinner='Loading embedding model...')
def get_model():
    return load_model()


@st.cache_resource(show_spinner='Loading raw documents for preview...')
def get_raw_docs(prefix):
    path = f'{DATA_DIR}/{prefix}_docs_processed.json'
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


# ── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.title('⚙️ Settings')

dataset_label = st.sidebar.selectbox('Dataset', list(DATASETS.keys()))
prefix = DATASETS[dataset_label]

model_name = st.sidebar.selectbox('Retrieval Model', MODELS)

top_k = st.sidebar.slider('Results to show', min_value=5, max_value=50, value=10, step=5)

use_stemming = st.sidebar.checkbox('Use stemming (default: lemmatization)', value=False)

st.sidebar.markdown('---')
st.sidebar.markdown('**IR System** — University Project  \nDatasets: CT2021 · MSMARCO')


# ── Main area ────────────────────────────────────────────────────────────────

st.title('🔍 Information Retrieval System')
st.caption(f'Dataset: **{dataset_label}** | Model: **{model_name}**')

query_input = st.text_input('Enter your search query', placeholder='e.g. diabetes treatment clinical trial')

search_btn = st.button('Search', type='primary')

if search_btn and query_input.strip():
    query = query_input.strip()
    tokens = preprocess(query, use_stemming=use_stemming)

    if not tokens:
        st.warning('Query is empty after preprocessing. Try a different query.')
        st.stop()

    with st.spinner('Searching...'):
        index = get_index(prefix)
        doc_lengths = get_doc_lengths(prefix)

        if model_name == 'TF-IDF':
            tfidf, idf = get_tfidf(prefix)
            results = retrieve_tfidf(tokens, tfidf, idf, doc_lengths, top_k=top_k)

        elif model_name == 'BM25':
            params = get_bm25_params(prefix)
            results = retrieve_bm25(tokens, index, doc_lengths, params['avg_dl'], top_k=top_k)

        elif model_name == 'Embedding':
            model = get_model()
            doc_ids, emb = get_embeddings(prefix)
            results = retrieve_embedding(query, model, doc_ids, emb, top_k=top_k)

        elif model_name == 'Hybrid Parallel (RRF)':
            tfidf, idf = get_tfidf(prefix)
            params = get_bm25_params(prefix)
            model = get_model()
            doc_ids, emb = get_embeddings(prefix)

            bm25_res = retrieve_bm25(tokens, index, doc_lengths, params['avg_dl'], top_k=1000)
            tfidf_res = retrieve_tfidf(tokens, tfidf, idf, doc_lengths, top_k=1000)
            emb_res = retrieve_embedding(query, model, doc_ids, emb, top_k=1000)
            results = hybrid_parallel(bm25_res, tfidf_res, emb_res, top_k=top_k)

        elif model_name == 'Hybrid Serial':
            params = get_bm25_params(prefix)
            model = get_model()
            doc_ids, emb = get_embeddings(prefix)

            bm25_res = retrieve_bm25(tokens, index, doc_lengths, params['avg_dl'], top_k=1000)
            results = hybrid_serial(query, bm25_res, model, doc_ids, emb,
                                    first_stage_k=100, top_k=top_k)

    # Show preprocessed tokens
    with st.expander('Preprocessed query tokens'):
        st.write(' · '.join(tokens))

    st.markdown(f'### Top {len(results)} Results')

    raw_docs = get_raw_docs(prefix)

    for r in results:
        doc_id = r['doc_id']
        score = r['score']
        rank = r['rank']

        preview = raw_docs.get(doc_id, '')
        preview_text = preview[:300] + '...' if len(preview) > 300 else preview

        with st.container():
            col1, col2 = st.columns([6, 1])
            with col1:
                st.markdown(f'**#{rank}** `{doc_id}`')
                if preview_text:
                    st.caption(preview_text)
            with col2:
                st.metric('Score', f'{score:.4f}')
            st.divider()

elif search_btn and not query_input.strip():
    st.warning('Please enter a query.')
