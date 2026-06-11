import os
import sys
import time

# Make sure services are importable when running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

# ── Communication mode (SOA) ─────────────────────────────────────────────────
# Default: in-process calls to the search engine service (fastest).
# If IR_API_URL is set (e.g. http://localhost:8000), the UI talks to the
# API Gateway over REST instead — same contract, swappable transport.

API_URL = os.environ.get('IR_API_URL', '').rstrip('/')

if not API_URL:
    from services import search_engine as engine


def call_search(**kwargs):
    if API_URL:
        import requests
        payload = {
            'query': kwargs['query'], 'dataset': kwargs['prefix'],
            'model': kwargs['model_name'], 'top_k': kwargs['top_k'],
            'use_refinement': kwargs['use_refinement'],
            'use_clustering': kwargs['use_clustering'],
            'use_ltr': kwargs['use_ltr'],
            'k1': kwargs['k1'], 'b': kwargs['b'],
        }
        return requests.post(f'{API_URL}/search', json=payload, timeout=60).json()
    return engine.search(
        kwargs['query'], kwargs['prefix'], kwargs['model_name'],
        top_k=kwargs['top_k'], use_refinement=kwargs['use_refinement'],
        use_clustering=kwargs['use_clustering'], use_ltr=kwargs['use_ltr'],
        k1=kwargs['k1'], b=kwargs['b'])


def call_rag(**kwargs):
    if API_URL:
        import requests
        payload = {
            'query': kwargs['query'], 'dataset': kwargs['prefix'],
            'model': kwargs['model_name'], 'backend': kwargs['backend'],
            'k1': kwargs['k1'], 'b': kwargs['b'],
        }
        return requests.post(f'{API_URL}/rag', json=payload, timeout=120).json()
    return engine.rag_answer(
        kwargs['query'], kwargs['prefix'], kwargs['model_name'],
        k1=kwargs['k1'], b=kwargs['b'], backend=kwargs['backend'])


def db_ready(prefix):
    if API_URL:
        import requests
        try:
            health = requests.get(f'{API_URL}/health', timeout=5).json()
            return health['datasets'][prefix]['db_ready']
        except Exception:
            return False
    return engine.db_status(prefix)


# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(page_title='IR System', page_icon='🔍', layout='wide')

DATASETS = {
    'Clinical Trials 2021': 'ct2021',
    'MS MARCO Passage (TREC-DL 2019)': 'msmarco',
}
MODELS = ['BM25', 'TF-IDF', 'Embedding', 'Hybrid Parallel (RRF)', 'Hybrid Serial']


# ── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.title('⚙️ Settings')

dataset_label = st.sidebar.selectbox('Dataset', list(DATASETS.keys()))
prefix = DATASETS[dataset_label]

model_name = st.sidebar.selectbox('Retrieval Model', MODELS)
top_k = st.sidebar.slider('Results to show', min_value=5, max_value=20, value=10, step=5)

# BM25 parameters — tunable per query (project requirement)
if model_name in ('BM25', 'Hybrid Parallel (RRF)', 'Hybrid Serial'):
    st.sidebar.subheader('BM25 parameters')
    bm25_k1 = st.sidebar.slider('k1 (term frequency saturation)', 0.5, 3.0, 1.5, 0.1)
    bm25_b = st.sidebar.slider('b (length normalization)', 0.0, 1.0, 0.75, 0.05)
else:
    bm25_k1, bm25_b = 1.5, 0.75

st.sidebar.markdown('---')
st.sidebar.subheader('Additional features')
use_refinement = st.sidebar.checkbox('Query refinement (spell + synonyms)')
use_clustering = st.sidebar.checkbox('Clustering filter')
use_ltr = st.sidebar.checkbox('LTR re-ranking')

st.sidebar.markdown('---')
mode = f'REST API ({API_URL})' if API_URL else 'In-process'
st.sidebar.caption(f'Service communication: **{mode}**')
if db_ready(prefix):
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
        with st.spinner('Searching...'):
            out = call_search(
                query=query_input.strip(), prefix=prefix, model_name=model_name,
                top_k=top_k, use_refinement=use_refinement,
                use_clustering=use_clustering, use_ltr=use_ltr,
                k1=bm25_k1, b=bm25_b)

        results = out['results']
        raw_docs = out['raw_docs']
        timings = out['timings']
        info = out['info']

        st.caption('⏱️ total: **%.2fs** — %s' % (
            timings.get('total', 0),
            ' | '.join(f'{k}: {v:.2f}s' for k, v in timings.items() if k != 'total')))

        for w in out.get('warnings', []):
            st.warning(w)
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

    backend_label = st.radio(
        'Answer model', ['Local (flan-t5, offline)', 'Gemini API (free)'],
        horizontal=True)
    rag_backend = 'gemini' if backend_label.startswith('Gemini') else 'local'
    if rag_backend == 'gemini' and not os.environ.get('GEMINI_API_KEY'):
        st.warning('Set the GEMINI_API_KEY environment variable to use the '
                   'Gemini API — otherwise the local model is used as fallback.')

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
                rag = call_rag(query=user_msg, prefix=prefix,
                               model_name=model_name, backend=rag_backend,
                               k1=bm25_k1, b=bm25_b)
                elapsed = rag.get('elapsed', time.time() - t0)

            st.write(rag['answer'])
            if rag.get('note'):
                st.caption(rag['note'])
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
