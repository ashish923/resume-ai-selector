# 🎯 ResumeAI Selector

AI-powered resume screening tool for HR teams. Upload resumes, paste a job description, and let AI rank candidates using hybrid scoring and chat with your candidate pool using RAG.

## 🚀 Features

- **Bulk Resume Upload** — Upload multiple PDF/DOCX resumes at once with automatic text extraction
- **AI-Powered Extraction** — LLM extracts structured info (name, skills, experience, education) from resumes and job descriptions
- **Hybrid Scoring Engine** — Combines keyword matching + LLM semantic matching for accurate skill comparison
- **Candidate Dashboard** — Ranked candidate cards with match scores, skill breakdown, and progress bars
- **Hybrid RAG Chat** — Ask natural language questions about your candidate pool using Dense (ChromaDB) + Sparse (BM25) search with RRF fusion
- **JD-Aware Chat** — Chat responses consider the job description for context-aware answers
- **Conversation Memory** — Follow-up questions work naturally in the chat

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI |
| Frontend | HTML + Jinja2 Templates |
| LLM | HuggingFace Inference API (Llama-3.3-70B-Instruct) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector DB | ChromaDB |
| Sparse Search | BM25 (rank-bm25) |
| Fusion | Reciprocal Rank Fusion (RRF) |
| File Parsing | PyMuPDF (PDF), python-docx (DOCX) |

## 📐 Architecture


HR uploads resumes (PDF/DOCX)
↓
PyMuPDF / python-docx → extract raw text
↓
LLM extracts structured info (name, skills, experience)
↓
LLM extracts JD requirements (required/preferred skills)
↓
Hybrid Scorer (keyword + LLM semantic matching)
↓
Dashboard → ranked candidates with scores
↓
ChromaDB + BM25 → hybrid RAG chat with resume pool

## ⚙️ Setup

### Prerequisites
- Python 3.10+
- HuggingFace account + API token ([get it here](https://huggingface.co/settings/tokens))

### Installation

```bash
# Clone the repo
git clone https://github.com/yourusername/resume-selector.git
cd resume-selector

# Create virtual environment
python -m venv venv

# Activate
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Setup environment
copy .env.example .env
# Edit .env and add your HuggingFace API token
```

### Run

```bash
uvicorn app.main:app --reload
```

Open `http://localhost:8000` in your browser.

## 📖 How to Use

1. Go to `/upload` → upload resumes (PDF/DOCX) + paste job description
2. Click **"Save & Continue"** → AI scores and ranks all resumes
3. View ranked candidates on `/dashboard` with match scores and skill breakdown
4. Go to `/analysis` → click **"Index Resumes"** → ask questions about your candidates

## 🔍 How Scoring Works

The hybrid scoring engine uses a weighted formula:

| Component | Weight | Method |
|-----------|--------|--------|
| Required Skills | 50% | Keyword match + LLM semantic match |
| Preferred Skills | 20% | Keyword match + LLM semantic match |
| Experience | 20% | Years comparison |
| Education | 10% | Keyword overlap |

**Hybrid skill matching:**
1. Keyword matching catches exact matches instantly (e.g., "Python" = "Python")
2. LLM matching catches semantic matches for remaining skills (e.g., "GenAI" ≈ "LLMs")

## 🔍 How RAG Chat Works

Uses hybrid retrieval for better search accuracy:

1. **Dense Search** — ChromaDB with sentence-transformer embeddings (semantic similarity)
2. **Sparse Search** — BM25 keyword scoring (exact term matching)
3. **RRF Fusion** — Reciprocal Rank Fusion merges both results (chunks found by both methods rank highest)
4. **LLM Generation** — Top chunks + JD + chat history sent to LLM for answer

## 📁 Project Structure

resume-selector/
├── app/
│   ├── main.py                  # FastAPI entry point
│   ├── config.py                # Settings & environment
│   ├── routes/
│   │   ├── upload.py            # Upload & analysis endpoints
│   │   ├── dashboard.py         # Dashboard endpoints
│   │   └── analysis.py          # RAG chat endpoints
│   ├── services/
│   │   ├── resume_parser.py     # PDF/DOCX text extraction
│   │   ├── llm_service.py       # HuggingFace LLM integration
│   │   ├── scorer.py            # Hybrid scoring engine
│   │   ├── vector_store.py      # ChromaDB + BM25 hybrid RAG
│   │   └── data_store.py        # In-memory data store
│   ├── templates/               # Jinja2 HTML templates
│   └── static/                  # CSS & JS
├── uploads/                     # Uploaded resume files
├── data/                        # ChromaDB persistent storage
├── requirements.txt
├── .env.example
└── README.md

## 🔑 Environment Variables

| Variable | Description |
|----------|-------------|
| HUGGINGFACEHUB_API_TOKEN | Your HuggingFace API token |
| LLM_MODEL | LLM model ID (default: meta-llama/Llama-3.3-70B-Instruct) |
| EMBEDDING_MODEL | Embedding model (default: sentence-transformers/all-MiniLM-L6-v2) |
| UPLOAD_DIR | Resume upload directory (default: uploads) |
| CHROMA_DIR | ChromaDB storage path (default: data/chroma_db) |

## 📝 License

MIT