# Faithfulness-Aware RAG

A RAG system that doesn't just answer questions — it checks whether its own
answer is actually supported by the retrieved evidence, and abstains rather
than hallucinating when it isn't.

**Status:** Week 1 — baseline retrieve-and-generate pipeline (no faithfulness
scoring yet; that lands in Week 2).

## Architecture

```
Query -> Retrieve (ChromaDB) -> Generate (Groq LLM) -> [Week 2: Faithfulness check] -> Answer
```

## Setup (on your Ubuntu machine)

```bash
cd rag-faithfulness
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# then edit .env and paste your free Groq API key from https://console.groq.com/keys
```

## Add your documents

Drop plain `.txt` files into `data/raw/`. A sample college policy document
is already there so you can test the pipeline immediately. Replace it with
your own corpus (syllabus, handbook, RTI Act, whatever you're using).

## Run it

```bash
# 1. Build the vector index (run this once, and again any time you change data/raw/)
python src/ingest.py

# 2. Quick command-line test of retrieval + generation
python src/generate.py "What is the minimum attendance percentage required?"

# 3. Full UI
streamlit run app.py
```

## Project structure

```
rag-faithfulness/
├── data/
│   ├── raw/              # your source .txt documents go here
│   └── chroma_db/        # persistent vector store (auto-created)
├── src/
│   ├── config.py         # all settings in one place
│   ├── ingest.py         # chunk + embed + store
│   ├── retrieve.py       # query the vector store
│   └── generate.py       # build grounded prompt + call Groq
├── eval/                 # gold-standard Q&A set for evaluation (Week 3)
├── app.py                # Streamlit UI
└── requirements.txt
```

## Deploy to Streamlit Community Cloud (free)

1. Create a GitHub repo and push this project:
   ```bash
   # On GitHub, create a new empty repo named e.g. "rag-faithfulness"
   git remote add origin https://github.com/YOUR_USERNAME/rag-faithfulness.git
   git push -u origin main
   ```

2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.

3. Click **"New app"** and select:
   - **Repository:** `YOUR_USERNAME/rag-faithfulness`
   - **Branch:** `main`
   - **Main file path:** `app.py`

4. Before deploying, add your API key as a secret:
   - In the Streamlit Cloud dashboard, click **"Advanced settings"** → **"Secrets"**
   - Paste this (replace with your real key):
     ```toml
     GROQ_API_KEY = "gsk_your_key_here"
     ```

5. Click **"Deploy"**. The first deploy takes ~2-3 minutes (installs dependencies + builds the vector index).

> **Note:** Streamlit Community Cloud has ephemeral storage. The vector index
> is rebuilt automatically from `data/raw/` on each cold start (~30s). The
> pre-built index is also committed to the repo for faster initial loads.

## Roadmap

- [x] Week 1: Baseline RAG pipeline (retrieve + generate)
- [ ] Week 2: Claim splitting + entailment/faithfulness checking + abstention
- [ ] Week 3: Gold-standard evaluation set + metrics + report
