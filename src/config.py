"""
Central configuration for the whole pipeline.
Change values here rather than hunting through every file.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma_db"
EVAL_DIR = PROJECT_ROOT / "eval"

# --- Embedding model ---
# Runs locally on CPU, no GPU needed. ~80MB download on first run.
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# --- Chunking ---
CHUNK_SIZE = 800       # characters per chunk (upper bound; whole sections kept intact when possible)
CHUNK_OVERLAP = 100    # characters of overlap between consecutive sliding-window sub-chunks

# --- Retrieval ---
TOP_K = 8               # number of chunks to retrieve per query

# --- Generation ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# NOTE: llama-3.1-8b-instant is being decommissioned by Groq on Aug 16, 2026.
# Switched to Groq's recommended replacement (openai/gpt-oss-20b) ahead of
# that date so requests don't start failing with a model_decommissioned error.
GENERATION_MODEL = "openai/gpt-oss-20b"

# --- Multi-provider support ---
# The sidebar accepts an API key from more than one provider. Which
# provider a given key belongs to is auto-detected from the key's own
# prefix (see llm_client.resolve_provider) -- Groq keys start with
# "gsk_", NVIDIA (build.nvidia.com / NIM) keys start with "nvapi-". Add
# a base URL + model pair here for any additional provider you want
# supported, then add a matching prefix check in llm_client.py.
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
# Tried in this order at request time -- NVIDIA's hosted catalog rotates
# and end-of-lifes models with little notice (this list used to be a
# single hardcoded name, "meta/llama-3.3-70b-instruct", until NVIDIA
# retired it on 2026-08-26). llm_client.py checks NVIDIA's own live
# /v1/models list and picks the first of these that's actually still
# being served, so one more surprise retirement doesn't break the app
# again -- it just falls through to the next candidate.
NVIDIA_MODEL_PREFERENCE = [
    "meta/llama-3.1-70b-instruct",
    "meta/llama-3.1-8b-instruct",
    "nvidia/llama-3.1-nemotron-70b-instruct",
    "mistralai/mixtral-8x22b-instruct-v0.1",
    "microsoft/phi-3-medium-4k-instruct",
]

# --- Faithfulness (used from Week 2 onward) ---
FAITHFULNESS_THRESHOLD = 0.7   # below this fraction of grounded claims -> abstain

# --- Chroma collection name ---
COLLECTION_NAME = "faithful_rag_docs"
