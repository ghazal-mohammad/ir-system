# Information Retrieval System — 2026

A full Information Retrieval system built as part of the IR course project.
Supports multiple retrieval models over two different datasets using a Service-Oriented Architecture (SOA).

---

## Datasets

| Dataset | Domain | Documents | Queries |
|---|---|---|---|
| trec-clinical-trials/2021 | Medical / Clinical | ~375K | 75 |
| msmarco-passage/trec-dl-2019 | Web / General | ~8.8M | 43 |

---

## Retrieval Models

- **TF-IDF** (VSM) — cosine similarity
- **BM25** — tunable k1 and b parameters
- **Embedding** — sentence-transformers (all-MiniLM-L6-v2)
- **Hybrid Parallel** — BM25 + Embedding with RRF fusion
- **Hybrid Serial** — BM25 candidate selection → Embedding re-ranking

---

## Project Structure

```
ir-system/
├── services/
│   ├── preprocessing_service.py
│   ├── indexing_service.py
│   ├── database_service.py        # raw docs stored in SQLite, read by id at query time
│   ├── tfidf_service.py
│   ├── bm25_service.py
│   ├── embedding_service.py       # brute-force + FAISS search
│   ├── hybrid_service.py
│   ├── query_refinement_service.py
│   ├── clustering_service.py
│   ├── rag_service.py
│   ├── ltr_service.py
│   └── evaluation_service.py
├── notebooks/                      # offline training only (01 → 10)
├── scripts/
│   ├── build_database.py           # populate the raw-docs SQLite DB
│   ├── build_faiss.py              # build FAISS index for fast embedding search
│   ├── run_ui.bat                  # run UI locally (Windows)
│   └── run_ui.sh                   # run UI locally (Linux/Mac)
├── ui/
│   └── app.py                      # search + RAG chat + feature toggles
├── data/                           # trained models & DB (not committed)
├── requirements.txt
└── README.md
```

---

## Evaluation Metrics

Evaluated before and after additional features:

- MAP (Mean Average Precision)
- Recall
- Precision@10
- nDCG

---

## Architecture at Query Time

1. The query is preprocessed and (optionally) refined.
2. The selected model retrieves the top documents from **pre-trained files** (no training happens online).
3. Optional features can each be toggled independently in the UI: query refinement, clustering filter, LTR re-ranking, RAG chat.
4. The **original raw text** of the top results is read from a **SQLite database by doc_id**.

## How to Run (Local)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Put the trained model files in ./data
#    (output of the training notebooks: indexes, TF-IDF, embeddings, ...)
#    or set IR_DATA_DIR to their location.

# 3. Build the raw-documents database (one time)
python scripts/build_database.py --dataset all

# 4. (Optional, recommended for MSMARCO) build the FAISS index
python scripts/build_faiss.py --dataset msmarco

# 5. Run the UI
streamlit run ui/app.py        # or scripts/run_ui.bat on Windows
```

---

## Author

ghazal-mohammad
