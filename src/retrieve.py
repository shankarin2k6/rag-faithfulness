"""
Week 1 - Step 2: Retrieval.

Given a user question, embed it and pull the top-k most similar chunks
from ChromaDB. Kept as a standalone module so app.py, generate.py, and
the eval script can all reuse it identically.

Includes a near-duplicate filter: legal/policy documents often have
"continued" sub-chunks or overlapping clauses that read as near-identical
text. Without filtering, the LLM can see the same fact twice under
different section labels and report it as two separate (sometimes
inconsistent-looking) claims. We fetch a larger pool than needed, drop
near-duplicates, then keep the top_k most relevant survivors.
"""
from difflib import SequenceMatcher

import chromadb
from sentence_transformers import SentenceTransformer

from config import CHROMA_DIR, EMBED_MODEL_NAME, TOP_K, COLLECTION_NAME

# Two chunks with similarity above this are considered near-duplicates.
# 0.85 catches "same clause, minor reformatting" without merging genuinely
# distinct sections that just happen to share legal boilerplate language.
DEDUP_SIMILARITY_THRESHOLD = 0.85

# Fetch this many candidates before deduping, so filtering out duplicates
# doesn't leave us with fewer than top_k useful chunks.
CANDIDATE_POOL_MULTIPLIER = 3

# Small penalty added to the distance of non-verified chunks before ranking,
# so that when a verified (confirmed real Act section) chunk and an
# unverified (guide/schedule/duplicate) chunk are close competitors for the
# same fact, the verified one wins. Deliberately small: it should only flip
# near-ties, not override a genuinely much stronger semantic match. Guide
# text is often closer in plain-English phrasing to a user's query than
# formal statute language, which is exactly the case this is meant to fix
# without suppressing unverified content when it's the only real match.
VERIFIED_PREFERENCE_PENALTY = 0.04

_embedder = None
_collection = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL_NAME)
    return _embedder


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


def _text_similarity(a: str, b: str) -> float:
    """Return a 0-1 similarity ratio between two chunk texts."""
    return SequenceMatcher(None, a, b).ratio()


def _dedup_chunks(chunks: list[dict], top_k: int) -> list[dict]:
    """Drop near-duplicate chunks, keeping the most relevant (lowest
    rank_distance) version of each near-identical group.

    Chunks are already sorted by rank_distance (ascending, verified
    preference applied) when this is called, so a simple greedy pass —
    keep a chunk unless it's too similar to one we've already kept —
    preserves the best copy of each duplicate cluster.
    """
    kept: list[dict] = []
    for candidate in chunks:
        is_duplicate = any(
            _text_similarity(candidate["text"], existing["text"]) >= DEDUP_SIMILARITY_THRESHOLD
            for existing in kept
        )
        if not is_duplicate:
            kept.append(candidate)
        if len(kept) >= top_k:
            break
    return kept


def retrieve(query: str, top_k: int = TOP_K, collection=None) -> list[dict]:
    """Return up to top_k chunks as [{text, source, chunk_index, distance}, ...],
    with near-duplicate chunks (e.g. overlapping clause sub-chunks) filtered out.

    `collection` defaults to the standard persistent RTI Act collection
    (CLI/eval usage). Pass an explicit collection (e.g. a per-session
    ephemeral one built from a user-uploaded document) to query that
    instead -- this is how the Streamlit app supports "ask questions about
    your own uploaded document" without touching the default corpus.
    """
    embedder = _get_embedder()
    if collection is None:
        collection = _get_collection()

    # Fetch a wider candidate pool than top_k, since some will likely be
    # dropped as near-duplicates during filtering.
    candidate_n = top_k * CANDIDATE_POOL_MULTIPLIER

    query_embedding = embedder.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=candidate_n,
    )

    candidates = []
    for text, meta, distance in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        status = meta.get("section_status", "n/a")
        # rank_distance is used for ordering only; distance (the true,
        # unmodified embedding distance) is preserved for display/debugging
        # so we never misrepresent actual similarity to the caller.
        rank_distance = distance if status == "verified" else distance + VERIFIED_PREFERENCE_PENALTY
        candidates.append({
            "text": text,
            "source": meta.get("source"),
            "chunk_index": meta.get("chunk_index"),
            "section_status": status,
            "section_num": meta.get("section_num") or None,
            "distance": distance,
            "rank_distance": rank_distance,
        })

    # Re-sort using rank_distance (verified-preference applied), not the
    # raw order Chroma returned. dedup then keeps the best copy of each
    # duplicate cluster using this same adjusted ranking.
    candidates.sort(key=lambda c: c["rank_distance"])
    return _dedup_chunks(candidates, top_k)


if __name__ == "__main__":
    # Quick manual test: python src/retrieve.py "your question here"
    import sys
    q = " ".join(sys.argv[1:]) or "What is this document about?"
    for i, c in enumerate(retrieve(q), 1):
        print(f"\n--- Chunk {i} (source: {c['source']}, status: {c['section_status']}, "
              f"distance: {c['distance']:.3f}, rank_distance: {c['rank_distance']:.3f}) ---")
        print(c["text"][:400])
