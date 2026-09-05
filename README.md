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

## Roadmap

- [x] Week 1: Baseline RAG pipeline (retrieve + generate)
- [ ] Week 2: Claim splitting + entailment/faithfulness checking + abstention
- [ ] Week 3: Gold-standard evaluation set + metrics + report
