# RAG Chat Tool

A retrieval-augmented generation (RAG) CLI that lets you chat with your own documents. Ingests PDFs, text, and markdown files into a pgvector database, then answers questions using Gemini embeddings and Gemini 2.5 Flash.

## How it works

1. **Ingest** — documents are chunked, embedded with `gemini-embedding-2` (768 dims), and stored in Postgres with pgvector
2. **Query** — your question is embedded, a cosine similarity search finds the top matching chunks, and Gemini 2.5 Flash answers using those chunks as context

## Prerequisites

- Docker
- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- Google API key with Gemini access

## Setup

1. Clone the repo and install dependencies:
   ```bash
   uv sync
   ```

2. Copy `.env.example` to `.env` and fill in your values:
   ```bash
   cp .env.example .env
   ```

   ```
   GOOGLE_API_KEY=your_key_here
   DATABASE_URL=postgresql://postgres:password@localhost:5432/ragdb
   ```

3. Start Postgres with pgvector:
   ```bash
   docker compose up -d
   ```

## Usage

### Ingest documents

Drop your files into the `docs/` directory (supports `.pdf`, `.txt`, `.md`), then run:

```bash
uv run ingest.py
```

This creates the `documents` table and HNSW index on first run. Re-ingesting will add duplicates — truncate first if re-running:

```bash
docker compose exec db psql -U postgres -d ragdb -c "TRUNCATE documents;"
```

### Chat

```bash
uv run main.py
```

Type your question at the prompt. Type `quit` or `exit` to stop.

## Configuration

| Variable | File | Default | Description |
|---|---|---|---|
| `CHUNK_SIZE` | `ingest.py` | 1000 | Characters per chunk |
| `CHUNK_OVERLAP` | `ingest.py` | 200 | Overlap between chunks |
| `EMBEDDING_DIM` | both | 768 | Embedding dimensions |
| `SIMILARITY_THRESHOLD` | `main.py` | 0.4 | Min cosine similarity to retrieve |
| `TOP_K` | `main.py` | 5 | Max chunks to retrieve per query |

## Stack

- **Embeddings**: `gemini-embedding-2` (Matryoshka, 768-dim)
- **LLM**: `gemini-2.5-flash`
- **Vector DB**: Postgres + pgvector (HNSW index, cosine distance)
- **PDF parsing**: pypdf
