# Codebase Assistant

An AI-powered developer tool that lets you ask natural language questions about any GitHub repository and get accurate, source-cited answers.

---

## Project Overview

Codebase Assistant indexes a GitHub repository into a vector database and uses semantic search + an LLM to answer questions about the code. Instead of manually reading through files, you can ask questions like:

- *"How are dependencies resolved?"*
- *"Where is authentication implemented?"*
- *"How does the request validation work?"*

---

## Architecture

```
GitHub Repo
     │
     ▼
┌─────────────────┐
│  repo_indexer/  │  Clone → Scan → Chunk → Embed → Store
│  ingest_repo.py │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   PostgreSQL    │  pgvector stores 384-dim embeddings
│  (code_chunks)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  retrieval/     │  Query → Embed → Vector Search → Re-rank → LLM
│  search_code.py │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   api/app.py    │  POST /ask-code  →  JSON response
└─────────────────┘
```

---

## How It Works

1. **Indexing** (`repo_indexer/ingest_repo.py`)
   - Clones the target GitHub repo locally
   - Scans for source files (`.py`, `.js`, `.ts`, `.go`, `.java`)
   - Splits files into 40-line overlapping chunks
   - Embeds each chunk using `sentence-transformers/all-MiniLM-L6-v2`, with the file path prepended for context
   - Stores chunks + embeddings in PostgreSQL with pgvector

2. **Retrieval** (`retrieval/search_code.py`)
   - Embeds the user's question
   - Fetches the top 30 candidate chunks by vector similarity
   - Re-ranks by **code density** (filters out docstring-heavy chunks) + **file path relevance**
   - Returns top 10 chunks grouped by file

3. **Generation** (`retrieval/search_code.py`)
   - Sends retrieved chunks to `gpt-4o-mini` with a structured prompt
   - LLM explains the architecture and answers the question
   - Sources (file + chunk) are returned alongside the answer

4. **API** (`api/app.py`)
   - FastAPI service exposing `POST /ask-code`
   - Accepts a question, returns an answer + sources

---

## Tech Stack

| Component        | Library                          |
|-----------------|----------------------------------|
| API server       | FastAPI, Uvicorn                 |
| Embeddings       | sentence-transformers (MiniLM)   |
| Vector database  | PostgreSQL + pgvector            |
| LLM              | OpenAI gpt-4o-mini               |
| Repo cloning     | GitPython                        |
| Environment      | python-dotenv                    |

---

## How To Run

### 1. Clone this repo and set up the environment

```bash
git clone <this-repo>
cd codebase-assistant
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Add your OpenAI API key to .env
```

### 3. Start PostgreSQL with pgvector

Make sure PostgreSQL is running with the pgvector extension enabled and a database named `ragdb` exists.

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### 4. Index a repository

```bash
python repo_indexer/ingest_repo.py
# Enter GitHub URL when prompted, e.g.: https://github.com/tiangolo/fastapi
```

### 5. Run the API

```bash
uvicorn api.app:app --reload
```

### 6. Ask questions

```bash
curl -X POST http://localhost:8000/ask-code \
  -H "Content-Type: application/json" \
  -d '{"question": "How are dependencies resolved?"}'
```

---

## Example Queries

| Query | What it finds |
|-------|--------------|
| How are dependencies resolved? | `dependencies/utils.py` — full resolution logic |
| Where are API routes registered? | `routing.py`, `applications.py` |
| How does request body validation work? | `dependencies/utils.py` — body field parsing |
| Where is middleware defined? | `middleware/` — cors, httpsredirect, trustedhost |

---

## Design Decisions

### Why chunk by lines?
Code files vary in length and structure. Fixed line chunking ensures consistent embedding size while preserving local context.

### Why pgvector instead of Pinecone?
Using PostgreSQL keeps infrastructure simple and demonstrates self-hosted vector search.

### Why MiniLM embeddings?
MiniLM provides fast local embeddings (384-dim) with good semantic accuracy.

---

## Future Improvements

- **Function-aware chunking** — split on `def`/`class` boundaries instead of fixed line count
- **Hybrid search** — combine pgvector with PostgreSQL full-text search (BM25) for better recall
- **Multi-repo support** — index and query multiple repositories simultaneously
- **Re-indexing on push** — webhook-triggered re-indexing when the repo is updated
- **Streaming responses** — stream the LLM answer token by token via SSE
