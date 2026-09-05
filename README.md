# 📚 Faithfulness-Aware RAG

> **A retrieval-augmented generation system that verifies its own answers against retrieved evidence and abstains rather than hallucinating.**

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-purple?style=for-the-badge&logo=chromadb&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-LLM%20API-green?style=for-the-badge&logo=groq&logoColor=white)

## 🌟 Executive Summary

**Faithfulness-Aware RAG** is a production-ready retrieval-augmented generation system that goes beyond standard Q&A by **verifying whether its own answer is actually supported by the retrieved evidence**. When the system's confidence is low, it abstains rather than hallucinating — making it suitable for legal, policy, and compliance use cases where accuracy matters.

The system retrieves relevant chunks from a vector store (ChromaDB), generates answers using Groq's LLM API, then runs a **faithfulness layer** that splits the answer into individual claims, checks each claim against the retrieved context, and verifies citation accuracy. Results are presented through a polished **Streamlit chat interface** with animated responses, faithfulness scoring, and file upload support.

## 🚀 Key Features

* **🔍 Section-Aware Retrieval** — Smart chunking for legal/policy documents with verified/unverified source labeling.
* **✅ Faithfulness Scoring** — Splits answers into individual claims and checks each against retrieved context.
* **📎 Citation Verification** — Detects fabricated "Section N" citations and flags them as unverified.
* **⚠️ Smart Abstention** — Refuses to answer when confidence is low instead of hallucinating.
* **📄 File Upload** — Ask questions about your own PDF/TXT documents via the "+" button.
* **🎨 Polished UI** — Chat-style interface with animated responses, scroll buttons, and a Claude/ChatGPT-style input pill.

## 🚀 Live Demo

**🔗 [Try it live on Streamlit Cloud](https://rag-faithfulnes-yoywrnbrbc2bzpa753clmv.streamlit.app/)**

> Enter your free Groq API key in the sidebar to start asking questions. Get a key at [console.groq.com/keys](https://console.groq.com/keys).

## 📊 How It Works

```
User Query → Retrieve (ChromaDB) → Generate (Groq LLM) → Faithfulness Check → Verified Answer
```

| Stage | Description |
| :--- | :--- |
| **Retrieve** | Embeds the query, pulls top-k chunks from ChromaDB, filters near-duplicates, and applies verified-preference ranking. |
| **Generate** | Builds a grounded prompt with section verification tags and calls Groq's LLM API. |
| **Faithfulness Check** | Splits the answer into claims, checks each against context, verifies citations, and computes a faithfulness score. |
| **Abstain** | If the score falls below the threshold (70%) or the model declines, the system flags the answer as low confidence. |

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/shankarin2k6/rag-faithfulness.git
cd rag-faithfulness
```

### 2. Install Dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Configuration
```bash
cp .env.example .env
# Edit .env and paste your free Groq API key from https://console.groq.com/keys
```

### 4. Build the Vector Index
```bash
python src/ingest.py
```

### 5. Launch the App
```bash
streamlit run app.py
```

## 🐳 Deployment (Streamlit Community Cloud — Free)

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **"New app"**.
3. Select your repo, branch `main`, and main file `app.py`.
4. Click **"Deploy"**. The app will be live at `https://YOUR_USERNAME-rag-faithfulness.streamlit.app`.

> **Note:** Users enter their own Groq API key in the sidebar — no secrets needed.

## 🛠️ System Architecture

```
rag-faithfulness/
├── app.py                # Streamlit UI (chat interface + file upload)
├── src/
│   ├── config.py         # Central configuration (thresholds, models, paths)
│   ├── ingest.py         # PDF/TXT extraction + section-aware chunking + embedding
│   ├── retrieve.py       # ChromaDB query + near-duplicate filtering
│   ├── generate.py       # Grounded prompt construction + Groq API call
│   ├── faithfulness.py   # Claim splitting + entailment check + citation verification
│   └── llm_client.py     # Provider-agnostic LLM client (Groq + NVIDIA)
├── data/
│   ├── raw/              # Source documents (RTI Act PDF included)
│   └── chroma_db/        # Persistent vector store
├── assets/               # UI assets (icons, spinners)
├── eval/                 # Gold-standard Q&A set for evaluation
├── requirements.txt
└── README.md
```

## 📦 Dependencies

| Package | Purpose |
| :--- | :--- |
| `streamlit` | Web UI framework |
| `chromadb` | Vector database for retrieval |
| `sentence-transformers` | Local embedding model (all-MiniLM-L6-v2) |
| `groq` | LLM API client |
| `pypdf` | PDF text extraction |
| `pandas` | Data handling |
| `plotly` | Charts and visualization |

## Roadmap

- [x] Week 1: Baseline RAG pipeline (retrieve + generate)
- [x] Week 2: Claim splitting + entailment/faithfulness checking + abstention
- [ ] Week 3: Gold-standard evaluation set + metrics + report

---

**Developed by Shankari N.**
