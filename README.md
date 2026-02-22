# Medical Diagnosis AI

AI-powered medical diagnosis system using RAG (Retrieval-Augmented Generation) over clinical protocols. Enter patient symptoms and get ranked differential diagnoses with ICD-10 codes.

## Setup

### Prerequisites

- Docker & Docker Compose v2.0+

### 1. Get the `.env` file

The `.env` file will be sent to you via Telegram. Place it in the project root directory.

### 2. Start services

**CPU:**
```bash
docker compose -f docker-compose.dev.cpu.yml up -d --build
```

**GPU (NVIDIA CUDA):**
```bash
docker compose -f docker-compose.dev.cuda.yml up -d --build
```

### 3. Wait for startup

After the containers start, TEI will automatically download the embedding model (`google/embeddinggemma-300m` by default). This may take a few minutes depending on your connection.

Once downloaded, TEI will warm up — allow **1-2 minutes** before querying.

You can monitor progress with:

```bash
docker compose -f docker-compose.dev.cpu.yml logs -f tei
```

### 4. Access the app

- **UI:** http://localhost
- **API docs:** http://localhost/redoc

## Usage

### UI (http://localhost)

The web interface has two ways to interact with the system:

**AI Diagnosis search** (`/dashboard/test-retrieval/<kb_id>`)
Enter patient symptoms (age, complaints, history) and get a ranked list of differential diagnoses with ICD-10 codes and explanations. This uses hybrid retrieval (vector + BM25) over clinical protocols, then an LLM to structure the output.

**Chatbot** (`/dashboard/chat`)
Conversational interface over the same knowledge base. Supports multi-turn dialogue with references to source protocols.

The knowledge base (clinical protocols) is **preloaded** — no setup or document uploads needed.

---

### REST API

The `/diagnose` endpoint is open (no auth required) — useful for testing and evaluation:

```bash
curl -X POST http://localhost/api/knowledge-base/diagnose \
  -H "Content-Type: application/json" \
  -d '{"symptoms": "Male, 45, chest pain radiating to left arm, shortness of breath"}'
```

Response:
```json
{
  "diagnoses": [
    { "rank": 1, "diagnosis": "...", "icd10_code": "I21.9", "explanation": "..." },
    ...
  ]
}
```

Full API reference: **http://localhost/redoc**

---

### Ports

| Port | Service |
|---|---|
| `80` | Nginx (UI + API, main entry point) |
| `8000` | Backend (FastAPI) |
| `3000` | Frontend (Next.js) |
| `8080` | TEI embedding server |
| `3306` | MySQL |
| `9000` | MinIO API |
| `9001` | MinIO Console |

## Architecture

| Service | Description |
|---|---|
| **Backend** | FastAPI (Python) |
| **Frontend** | Next.js 14 |
| **Database** | MySQL 8.0 |
| **Vector DB** | ChromaDB (local persistent volume) |
| **Embeddings** | TEI — `google/embeddinggemma-300m` |
| **Storage** | MinIO |
| **Proxy** | Nginx |

## How it works

1. Clinical protocols are indexed into ChromaDB using hybrid retrieval (vector + BM25)
2. Patient symptoms are matched against protocols using RRF scoring
3. An LLM generates ranked diagnoses with ICD-10 codes from the top matching protocols
