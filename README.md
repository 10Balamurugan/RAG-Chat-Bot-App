# 🤖 RAG Bot

**RAG Bot** is a document chatbot that lets you upload files (PDF, DOCX, TXT), analyzes them, and answers your questions in natural language — like chatting with an AI that has read your documents.

Built with **Python**, **LangChain**, **ChromaDB**, **Sentence Transformers**, and **Groq API** (with a local fallback when no API key is set).

---

## Table of Contents

1. [What is RAG?](#what-is-rag)
2. [How It Works](#how-it-works)
3. [Features](#features)
4. [Prerequisites](#prerequisites)
5. [Build From Scratch](#build-from-scratch)
6. [Project Structure](#project-structure)
7. [Configuration](#configuration)
8. [How to Run](#how-to-run)
9. [Using the Web UI](#using-the-web-ui)
10. [Using the CLI](#using-the-cli)
11. [Troubleshooting](#troubleshooting)
12. [Tech Stack](#tech-stack)

---

## What is RAG?

**RAG (Retrieval-Augmented Generation)** combines two steps:

1. **Retrieval** — search your documents for the most relevant passages.
2. **Generation** — an LLM reads those passages and writes a natural answer.

This means answers come from **your files**, not from the model's general memory — which reduces wrong or made-up information.

---

## How It Works

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│  Upload     │────▶│  Load &      │────▶│  Embed &    │────▶│  ChromaDB    │
│  PDF/DOCX   │     │  Split text  │     │  Store      │     │  Vector DB   │
└─────────────┘     └──────────────┘     └─────────────┘     └──────────────┘
                                                                    │
┌─────────────┐     ┌──────────────┐     ┌─────────────┐            │
│  User sees  │◀────│  LLM writes  │◀────│  Retrieve   │◀───────────┘
│  answer     │     │  answer      │     │  top chunks │
└─────────────┘     └──────────────┘     └─────────────┘
                           ▲
                     User question
```

### Step-by-step flow

| Step | What happens | File responsible |
|------|----------------|------------------|
| 1 | User uploads a document | `app.py` |
| 2 | Text is extracted from PDF/DOCX/TXT | `utils/document_loader.py` |
| 3 | Text is split into small chunks | `ingestion/ingest_data.py` |
| 4 | Each chunk is converted to a vector (embedding) | `embeddings/embedding_model.py` |
| 5 | Vectors are saved in ChromaDB | `vectordb/chroma_store.py` |
| 6 | User asks a question | `app.py` |
| 7 | Question is embedded and similar chunks are found | `retriever/retriever.py` |
| 8 | Relevant text + question is sent to the LLM | `chains/rag_chain.py` |
| 9 | LLM returns a natural-language answer | `chains/rag_chain.py` |

### Answer modes

| Mode | When used | Quality |
|------|-----------|---------|
| **Groq API** | Valid `GROQ_API_KEY` in `.env` | Best — natural ChatGPT-style answers |
| **Document lookup** | No Groq key, factual questions (skills, experience, etc.) | Good — pulls exact lines from your file |
| **Local model** | Fallback for other questions without Groq | Basic — small FLAN-T5 model |

> **Recommendation:** Add a free Groq API key for the best experience.  
> Get one at [https://console.groq.com](https://console.groq.com)

---

## Features

- Upload **PDF, DOCX, TXT, MD** files from the web UI
- One-click **Analyze Documents** to index your files
- **Chat interface** with conversation history
- Quick actions: **Summary**, **Key details**, **Main topics**
- Clean white & blue UI with **RAG Bot** branding
- CLI mode for terminal use
- Windows-safe vector store updates (no file-lock errors)

---

## Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.10 or higher |
| pip | Latest |
| Internet | Required (first run downloads AI models) |
| Groq API key | Optional but recommended |

**Optional:** Git, virtual environment (`venv`)

---

## Build From Scratch

Follow these steps to recreate this project from zero.

### Step 1 — Create project folder

```bash
mkdir rag-chatbot
cd rag-chatbot
```

### Step 2 — Create virtual environment

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3 — Create `requirements.txt`

```txt
langchain>=0.2.0
langchain-community>=0.2.0
langchain-chroma>=0.1.0
langchain-huggingface>=0.1.0
langchain-text-splitters>=0.2.0
chromadb>=0.5.0
sentence-transformers>=2.5.1
transformers>=4.40.0
torch>=2.2.0
pypdf>=4.2.0
python-docx>=1.1.0
python-dotenv>=1.0.1
groq>=0.9.0
streamlit>=1.32.0
```

Install dependencies:

```bash
pip install -r requirements.txt
```

> First install may take several minutes (downloads PyTorch and embedding models).

### Step 4 — Create folder structure

```bash
mkdir data\documents
mkdir embeddings ingestion retriever chains chatbot utils vectordb assets
mkdir .streamlit
```

### Step 5 — Create configuration files

**`.env`** (copy from `.env.example`):

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL_NAME=llama-3.1-8b-instant
```

**`.streamlit/config.toml`** (UI theme):

```toml
[theme]
primaryColor = "#1e40af"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f1f5f9"
textColor = "#111827"
font = "sans serif"
```

### Step 6 — Core Python modules

Create these files (already included in this repo):

| File | Purpose |
|------|---------|
| `config.py` | Paths, model names, API keys |
| `utils/document_loader.py` | Load PDF, DOCX, TXT files |
| `embeddings/embedding_model.py` | Sentence Transformers embeddings |
| `vectordb/chroma_store.py` | ChromaDB vector storage |
| `ingestion/ingest_data.py` | Chunk and index documents |
| `retriever/retriever.py` | Find relevant chunks for a question |
| `chains/rag_chain.py` | RAG pipeline + LLM answer generation |
| `chatbot/chat_interface.py` | Terminal chat interface |
| `app.py` | Streamlit web UI |
| `main.py` | CLI entry point |

### Step 7 — Add documents

Place your files in `data/documents/` or upload them via the web UI.

### Step 8 — Run the app

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## Project Structure

```
rag-chatbot/
│
├── app.py                      # Streamlit web UI (main interface)
├── main.py                     # CLI entry point
├── config.py                   # Settings: paths, models, API keys
├── requirements.txt            # Python dependencies
├── .env                        # API keys (do NOT commit to Git)
├── .env.example                # Template for .env
│
├── assets/
│   └── logo.svg                # RAG Bot logo
│
├── .streamlit/
│   └── config.toml             # Streamlit theme (white & blue)
│
├── data/
│   └── documents/              # Uploaded / saved documents
│
├── embeddings/
│   └── embedding_model.py      # HuggingFace embedding model
│
├── vectordb/
│   └── chroma_store.py         # ChromaDB read/write
│
├── ingestion/
│   └── ingest_data.py          # Document chunking & indexing
│
├── retriever/
│   └── retriever.py            # Similarity search (top-K chunks)
│
├── chains/
│   └── rag_chain.py            # Full RAG: retrieve → generate answer
│
├── chatbot/
│   └── chat_interface.py       # Terminal chat loop
│
└── utils/
    └── document_loader.py      # PDF / DOCX / TXT text extraction
```

**Auto-generated folders (do not edit manually):**

| Folder | Contents |
|--------|----------|
| `.chroma_store/` | Indexed vector database |
| `.venv/` | Python virtual environment |

---

## Configuration

Edit `config.py` to customize:

| Setting | Default | Description |
|---------|---------|-------------|
| `EMBEDDING_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | Model that converts text to vectors |
| `GENERATION_MODEL_NAME` | `google/flan-t5-small` | Local fallback LLM |
| `GROQ_MODEL_NAME` | `llama-3.1-8b-instant` | Groq cloud LLM (set in `.env`) |
| `CHUNK_SIZE` | `800` | Characters per text chunk |
| `CHUNK_OVERLAP` | `120` | Overlap between chunks |
| `TOP_K` | `6` | Number of chunks retrieved per question |

---

## How to Run

### Web UI (recommended)

```bash
# Activate virtual environment first
streamlit run app.py
```

Or:

```bash
python main.py --ui
```

### Ingest documents manually (optional)

If you add files directly to `data/documents/`:

```bash
python main.py --ingest
```

### CLI chat (terminal)

```bash
python main.py
```

### Ingest + run together

```bash
python main.py --ingest --ui
```

---

## Using the Web UI

1. **Open** http://localhost:8501
2. **Upload** your PDF, DOCX, or TXT file in the sidebar
3. Click **Analyze Documents** — wait for indexing to finish
4. **Ask questions** in the chat box, for example:
   - `Summarize this document`
   - `What are the main skills mentioned?`
   - `What is the work experience?`
5. Use **Quick actions** for one-click Summary, Key details, or Main topics

---

## Using the CLI

```bash
python main.py --ingest   # index documents first
python main.py            # start terminal chat
```

Type your question and press Enter. Type `exit` or `quit` to stop.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| **Analysis failed — file in use (Windows)** | Refresh the page and try again. The app now clears caches before re-indexing. |
| **Answers repeat instructions or look broken** | Your Groq API key may be expired. Add a new key to `.env`. |
| **Slow first response** | Normal — embedding and LLM models download on first run. |
| **No text extracted from PDF** | PDF may be scanned/image-only. Use a text-based PDF or OCR first. |
| **`.doc` not supported** | Save the file as `.docx` and re-upload. |
| **Weak answers without Groq** | Add `GROQ_API_KEY` to `.env` for ChatGPT-quality responses. |
| **Module not found** | Activate venv and run `pip install -r requirements.txt` |

### Verify Groq API key

```bash
python -c "from config import GROQ_API_KEY; print('Key set:' if GROQ_API_KEY else 'Key missing')"
```

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.10+ |
| Web UI | Streamlit |
| Orchestration | LangChain |
| Vector database | ChromaDB |
| Embeddings | Sentence Transformers (`all-MiniLM-L6-v2`) |
| LLM (primary) | Groq API (`llama-3.1-8b-instant`) |
| LLM (fallback) | Hugging Face FLAN-T5 |
| PDF parsing | pypdf |
| Word parsing | python-docx |
| Environment | python-dotenv |

---

## Security Notes

- Never commit `.env` to Git — it contains your API key.
- Add `.env` to `.gitignore` if using version control.
- Use `.env.example` as a template with placeholder values only.

---

## License

This project is for educational and personal use. Check individual library licenses for commercial use.

---

**Built with ❤️ — RAG Bot 🤖**
