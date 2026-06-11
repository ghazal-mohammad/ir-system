"""
API Gateway — exposes every service as an independent REST endpoint (SOA).

Each service can be called and tested on its own:
  POST /preprocess        — Preprocessing Service
  POST /refine            — Query Refinement Service
  POST /search            — Retrieval + Ranking Service (full pipeline)
  POST /rag               — RAG Service
  GET  /document/...      — Document Store Service (DB read by id)
  GET  /cluster/assign    — Clustering Service
  GET  /health            — gateway status

Run:
    uvicorn api.main:app --port 8000
Interactive docs (Swagger): http://localhost:8000/docs
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from services import search_engine as engine
from services.preprocessing_service import preprocess
from services.query_refinement_service import refine_query
from services.clustering_service import assign_query_to_cluster

app = FastAPI(
    title='IR System — API Gateway',
    description='Service-Oriented Architecture gateway for the Information Retrieval system.',
    version='1.0',
)

DATASETS = ['ct2021', 'msmarco']


def check_dataset(dataset: str):
    if dataset not in DATASETS:
        raise HTTPException(404, f'Unknown dataset: {dataset}. Available: {DATASETS}')


# ── Request models ───────────────────────────────────────────────────────────

class PreprocessRequest(BaseModel):
    text: str
    use_stemming: bool = False


class RefineRequest(BaseModel):
    text: str
    dataset: str = 'ct2021'
    use_spell: bool = True
    use_expand: bool = True


class SearchRequest(BaseModel):
    query: str
    dataset: str = 'ct2021'
    model: str = 'BM25'
    top_k: int = Field(10, ge=1, le=100)
    use_refinement: bool = False
    use_clustering: bool = False
    use_ltr: bool = False
    k1: float = Field(1.5, ge=0.0, le=5.0)
    b: float = Field(0.75, ge=0.0, le=1.0)


class RagRequest(BaseModel):
    query: str
    dataset: str = 'ct2021'
    model: str = 'BM25'
    k1: float = Field(1.5, ge=0.0, le=5.0)
    b: float = Field(0.75, ge=0.0, le=1.0)


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get('/health')
def health():
    return {
        'status': 'ok',
        'datasets': {p: {'db_ready': engine.db_status(p)} for p in DATASETS},
        'models': engine.MODELS,
    }


@app.post('/preprocess')
def preprocess_endpoint(req: PreprocessRequest):
    tokens = preprocess(req.text, use_stemming=req.use_stemming)
    return {'tokens': tokens, 'processed_text': ' '.join(tokens)}


@app.post('/refine')
def refine_endpoint(req: RefineRequest):
    check_dataset(req.dataset)
    index, _, _, vocab = engine.get_core(req.dataset)
    tokens = preprocess(req.text)
    result = refine_query(tokens, vocab, index,
                          use_spell=req.use_spell, use_expand=req.use_expand)
    return result


@app.post('/search')
def search_endpoint(req: SearchRequest):
    check_dataset(req.dataset)
    if req.model not in engine.MODELS:
        raise HTTPException(404, f'Unknown model: {req.model}. Available: {engine.MODELS}')
    return engine.search(
        req.query, req.dataset, req.model, top_k=req.top_k,
        use_refinement=req.use_refinement, use_clustering=req.use_clustering,
        use_ltr=req.use_ltr, k1=req.k1, b=req.b)


@app.post('/rag')
def rag_endpoint(req: RagRequest):
    check_dataset(req.dataset)
    return engine.rag_answer(req.query, req.dataset, req.model,
                             k1=req.k1, b=req.b)


@app.get('/document/{dataset}/{doc_id}')
def document_endpoint(dataset: str, doc_id: str):
    check_dataset(dataset)
    text = engine.get_document(dataset, doc_id)
    if not text:
        raise HTTPException(404, f'Document not found: {doc_id}')
    return {'doc_id': doc_id, 'text': text}


@app.get('/cluster/assign/{dataset}')
def cluster_assign_endpoint(dataset: str, query: str):
    check_dataset(dataset)
    if not os.path.exists(f'{engine.DATA_DIR}/{dataset}_km_model.pkl'):
        raise HTTPException(404, 'Clustering model not found for this dataset.')
    km_model, _ = engine.get_clustering(dataset)
    model = engine.get_model()
    q_emb = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
    return {'query': query, 'cluster': assign_query_to_cluster(q_emb, km_model)}
