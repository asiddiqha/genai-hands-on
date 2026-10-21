# Generative AI Hands-on: RAG and Semantic Search

Hands-on Generative AI work, from embeddings and similarity search up to a full retrieval-augmented generation (RAG) pipeline over the Hugging Face documentation that cites its sources and declines questions the documents can't answer.

| # | Notebook | What it covers | Open |
|---|---|---|---|
| 01 | `01_embeddings_semantic_search.ipynb` | Sentence embeddings, semantic search, similarity metrics, normalisation, bi-encoder vs cross-encoder, BM25 vs dense search | [Colab](https://colab.research.google.com/github/asiddiqha/genai-hands-on/blob/main/01_embeddings_semantic_search.ipynb) |
| 02 | `02_rag_pipeline.ipynb` | End-to-end RAG: chunking, ChromaDB, HyDE, cross-encoder reranking, grounded answers with citations | [Colab](https://colab.research.google.com/github/asiddiqha/genai-hands-on/blob/main/02_rag_pipeline.ipynb) |

## 01 · Embeddings and semantic search

```
text → all-MiniLM-L6-v2 (384-dim) → vectors → cosine similarity → top-k results
```

| Topic | What I show | Key result |
|---|---|---|
| Sentence embeddings | Similarity matrix heatmap | "cat on the mat" vs "kitten on the rug": 0.66 with almost no shared words |
| Semantic search | Query vs stored embeddings | "small cat on a carpet" finds both cat sentences; "carpet" never appears |
| Visualisation | PCA to 2D and 3D | 2D keeps ~68% of variance: fine for viewing, not for search |
| Similarity metrics | Cosine, dot product, Euclidean, Manhattan | Raw dot product can flip rankings: [800, 0] beats [400, 1] for [1, 1] only because it's longer |
| Normalisation | Unit-length vectors | Cosine = dot product, so vector DBs normalise once and use the faster dot product |
| Two-stage retrieval | Bi-encoder retrieval, then cross-encoder rerank | Fast recall first, precise ordering second |
| Dense vs sparse | BM25 vs embeddings on a synonym query | BM25 ranks an unrelated sentence above "kitten on the rug"; dense search doesn't |
| Model choice | Local Sentence Transformers vs hosted Jina API | Privacy, cost, fine-tuning and domain fit decide |

## 02 · RAG over Hugging Face documentation

```
Offline:  2,647 docs → chunking (256 tokens, 32 overlap) → bge embeddings → ChromaDB (HNSW, cosine) + source metadata
Online:   question → [optional HyDE draft] → top 20 by similarity → cross-encoder rerank → top 3–5 → grounded LLM answer with citations
```

| Stage | Choice | Why |
|---|---|---|
| Knowledge base | `m-ric/huggingface_doc` | Real technical docs with source paths for citations |
| Chunking | Chonkie, fixed size with overlap | Fast and predictable at ~33k chunks; compared 6 strategies first |
| Embeddings | `BAAI/bge-base-en-v1.5` | Stronger retrieval than MiniLM (used as a baseline) |
| Vector DB | ChromaDB, HNSW, cosine | In-process, metadata filtering, no paid service |
| Query enhancement | HyDE (optional) | Bridges short questions and long passages |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Precise top-k in ~100 ms for 20 candidates |
| Generation | GPT-OSS 120B on Groq, temperature 0 | Fast hosted inference, grounded prompt with citations |

What it covers:
1. Six chunking strategies compared on one article: fixed, overlap, semantic, recursive, late and LLM-based
2. Embeddings and similarity search with ChromaDB, including metadata filtering
3. Bi-encoder retrieval with HNSW
4. HyDE vs standard search
5. Two-stage retrieval: bi-encoder recall, cross-encoder precision
6. Grounded answers with citations, plus an out-of-scope test where the system declines

## Run it

Open a notebook in Colab and choose **Runtime → Run all**. CPU is enough for 01; a T4 GPU speeds up 02.
Keys go in Colab Secrets: `groq` for notebook 02, and optionally `JINA_API_KEY` for the hosted-embeddings demo in 01.

Locally: `pip install -r requirements.txt` and set `GROQ_API_KEY` (and `JINA_API_KEY` if used) as environment variables.

## Next steps

- Hybrid search (BM25 + dense, merged with Reciprocal Rank Fusion)
- Evaluation: recall@k, MRR, faithfulness, latency, and cost per query
- Persistent index with incremental updates and query caching

## Stack

Python · Sentence Transformers · scikit-learn · rank_bm25 · Chonkie · ChromaDB · LangChain · Groq · Plotly
