# RAG Backend (FastAPI + LangChain + ChromaDB + OpenAI)

## Features

- Upload PDF dan ingest ke ChromaDB (`/ingest/pdf`)
- Chat berbasis dokumen (`/chat`)
- Chat dengan lampiran PDF/gambar (`/chat/attach`)
- Retrieval by namespace
- Service layer berbasis LangChain

## Project Structure

```text
app/
  core/
    config.py
    models.py
  services/
    pdf_parser.py
    rag_service.py
    text_chunker.py
  main.py
.env.example
requirements.txt
```

## Setup

1. Buat virtual environment.
2. Install dependency.
3. Copy `.env.example` ke `.env` lalu isi API key.

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
copy .env.example .env
```

## Run

```bash
uvicorn app.main:app --reload
```

## API

### 1) Health

```http
GET /health
```

### 2) Ingest PDF

```http
POST /ingest/pdf
Content-Type: multipart/form-data
- file: <pdf_file>
- namespace: default (optional)
```

Contoh cURL:

```bash
curl -X POST "http://127.0.0.1:8000/ingest/pdf" \
  -F "file=@./data/sample.pdf" \
  -F "namespace=default"
```

### 3) Chat

```http
POST /chat
Content-Type: application/json

{
  "question": "Apa isi utama dokumen?",
  "namespace": "default",
  "top_k": 5
}
```

### 4) Chat dengan Lampiran (PDF/Gambar)

```http
POST /chat/attach
Content-Type: multipart/form-data
- question: <teks pertanyaan>
- file: <pdf atau image>
- namespace: default (optional)
- top_k: 5 (optional)
```

Contoh cURL (PDF):

```bash
curl -X POST "http://127.0.0.1:8000/chat/attach" \
  -F "question=Ringkas isi file ini" \
  -F "file=@./data/sample.pdf" \
  -F "namespace=default" \
  -F "top_k=5"
```

Contoh cURL (Gambar):

```bash
curl -X POST "http://127.0.0.1:8000/chat/attach" \
  -F "question=Apa yang terlihat pada gambar ini?" \
  -F "file=@./data/sample.jpg" \
  -F "namespace=default" \
  -F "top_k=5"
```

## Notes

- Data vector disimpan lokal di folder yang diatur via `CHROMA_PERSIST_DIRECTORY`.
- Namespace disimpan sebagai metadata pada setiap chunk.
- Untuk production, tambahkan:
  - auth + rate limit
  - retry/backoff API
  - caching
  - observability (logs, traces)
  - reranker/hybrid retrieval
