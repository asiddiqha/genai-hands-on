# RAG over Hugging Face Documentation

A retrieval-augmented generation (RAG) pipeline that answers questions about the Hugging Face documentation, cites its sources, and declines questions the documents can't answer.

[Open in Colab](https://colab.research.google.com/github/asiddiqha/rag-huggingface-docs/blob/main/rag_pipeline.ipynb)

## Pipeline

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

## What the notebook covers

1. Six chunking strategies compared on one article: fixed, overlap, semantic, recursive, late and LLM-based
2. Embeddings and similarity search with ChromaDB, including metadata filtering
3. Bi-encoder retrieval with HNSW
4. HyDE vs standard search
5. Two-stage retrieval: bi-encoder recall, cross-encoder precision
6. Grounded answers with citations, plus an out-of-scope test where the system declines

## Run it

Open the notebook in Colab with a T4 GPU, add a Groq API key in Colab Secrets named `groq`, then **Runtime → Run all**.

Locally: `pip install -r requirements.txt` and set `GROQ_API_KEY` in your environment (replace the Colab Secrets cell).

## Next steps

- Hybrid search (BM25 + dense, merged with Reciprocal Rank Fusion)
- Evaluation: recall@k, MRR, faithfulness, latency and cost per query
- Persistent index with incremental updates, and query caching

## Stack

Python · Chonkie · Sentence Transformers · ChromaDB · LangChain · Groq
