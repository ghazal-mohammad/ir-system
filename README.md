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
│   ├── tfidf_service.py
│   ├── bm25_service.py
│   ├── embedding_service.py
│   ├── hybrid_service.py
│   ├── query_refinement_service.py
│   └── evaluation_service.py
├── notebooks/
│   ├── 01_environment_setup.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_indexing.ipynb
│   ├── 04_retrieval_models.ipynb
│   └── 05_evaluation.ipynb
├── ui/
│   └── app.py
├── data/
│   ├── clinical_trials/
│   └── msmarco/
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

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the UI
streamlit run ui/app.py
```

---

## Author

ghazal-mohammad
