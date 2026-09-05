"""
Week 1 - Step 3: Generation.

Takes the user question + retrieved chunks, builds a grounded prompt,
and calls Groq's API for the answer. Deliberately instructed to only
use the provided context — this is the baseline the faithfulness
layer (Week 2) will later audit.
"""
from llm_client import create_chat_completion


def build_prompt(question: str, chunks: list[dict]) -> str:
    context_parts = []
    for c in chunks:
        status = c.get("section_status", "n/a")
        if status == "unverified":
            tag = "[UNVERIFIED SECTION NUMBER — do not cite any section number from this passage as fact]"
        elif status == "verified":
            tag = "[Confirmed Act section — safe to cite its section number]"
        else:
            tag = ""
        context_parts.append(f"[Source: {c['source']}] {tag}\n{c['text']}")
    context = "\n\n".join(context_parts)

    return f"""You are a precise assistant that answers ONLY using the context below.
If the context does not contain enough information to answer, say so plainly —
do not use outside knowledge and do not guess.

Legal and policy documents often restate the same rule in more than one
section (e.g. a provision stated in one clause and repeated almost
word-for-word as a proviso elsewhere). Before answering:
1. Identify which retrieved passages describe the SAME underlying rule,
   even if worded differently or attributed to different sections.
2. State each distinct rule only ONCE in your answer, citing the clearest
   source section for it.
3. Do not list the same deadline/condition twice just because it appeared
   in two different passages — that is restatement, not two separate rules.

Some passages below are marked [UNVERIFIED SECTION NUMBER]. This means the
numeric label attached to that passage could not be confirmed as a real Act
section — it may come from a table of contents, schedule, duplicate reprint,
or supplementary guide that reuses the same digits for unrelated content.
You may still use the SUBSTANCE of an unverified passage if it's relevant,
but never state or imply a specific "Section N" citation for it. If your
answer would otherwise rely on an unverified section number, describe the
rule without citing a section number, or say the exact section is uncertain.

Context:
{context}

Question: {question}

Answer, using only the context above. Consolidate restated rules; do not
repeat the same fact under different bullet points; never cite an
unverified section number as if it were confirmed:"""


def generate_answer(question: str, chunks: list[dict], api_key: str | None = None) -> str:
    prompt = build_prompt(question, chunks)
    response = create_chat_completion(
        api_key,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,   # deterministic, minimizes creative drift from context
    )
    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    # Quick manual test: python src/generate.py "your question here"
    import sys
    from retrieve import retrieve

    q = " ".join(sys.argv[1:]) or "What is this document about?"
    retrieved_chunks = retrieve(q)
    answer = generate_answer(q, retrieved_chunks)
    print("\n=== ANSWER ===\n")
    print(answer)
