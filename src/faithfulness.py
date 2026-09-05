"""
Week 2 - Faithfulness Layer.

This is the core differentiator of the project: after generating an answer,
we split it into individual factual claims and check each one against the
retrieved context to see if it's actually grounded. This turns "does the
answer sound right?" into a measurable, auditable score instead of just
trusting the LLM's output blindly.

Pipeline:
    answer -> split into claims -> check each claim against context
           -> faithfulness score -> abstain if below threshold
"""
import json
import re

from config import FAITHFULNESS_THRESHOLD
from llm_client import create_chat_completion

# Matches an LLM-STATED citation like "Section 7", "(Section 24(1))",
# "Sec. 27" inside a claim's own text. Deliberately case-SENSITIVE,
# requiring a capital "S": the Act's own body text writes inline cross-
# references in lowercase ("...within the time specified... of section 7,
# or is aggrieved..."), while the LLM's own citation markers (both
# "(Cited from Section N)" and "Section N: ..." bullet-heading styles)
# consistently capitalize "Section". Matching case-insensitively caused
# genuine inline Act cross-references to be mistaken for the LLM's actual
# claimed citation, flagging correct citations as invalid.
CITATION_IN_CLAIM_PATTERN = re.compile(r'\bSec(?:tion)?\.?\s+(\d{1,3})\b')


# Detects the LLM explicitly declining to answer (e.g. "I cannot answer",
# "does not contain enough information", "unable to answer"). Whether a
# refusal statement itself counts as an "entailed claim" is a slightly
# philosophical question the entailment checker answers inconsistently
# from call to call (a refusal is technically always "true", so it can
# score 100% and never trigger the score-based abstain threshold, even
# though the system produced no substantive answer at all). Detecting
# refusal directly from the answer text sidesteps that inconsistency:
# a decline is always treated as abstention, regardless of what the
# entailment check makes of the refusal sentence itself.
REFUSAL_PATTERN = re.compile(
    r"\b(?:cannot|can't|unable to) answer\b"
    r"|\bdoes not contain enough information\b"
    r"|\bcontext does not (?:provide|contain)\b"
    r"|\bnot (?:related to|found in) the (?:provided )?context\b",
    re.IGNORECASE,
)


def is_refusal(answer: str) -> bool:
    """True if the answer ITSELF is a decline/refusal, not just a caveat
    mentioned somewhere within an otherwise substantive answer.

    Only checks the first ~200 characters. A genuine refusal ("I cannot
    answer this question...") always appears at the very start of the
    response -- the model doesn't answer substantively for several
    paragraphs and then suddenly refuse partway through. Searching the
    full answer text caused false positives: a long, well-grounded answer
    that ends with an honest hedge like "note that the context does not
    provide an explicit single list" was being flagged as a full decline,
    even though it had already answered the question correctly.
    """
    return bool(REFUSAL_PATTERN.search(answer[:200]))


def split_into_claims(answer: str) -> list[str]:
    """Split an answer into individual claims.

    LLM answers are frequently formatted as bullet/numbered lists rather
    than plain paragraphs. Splitting on sentence punctuation alone doesn't
    break these apart if a bullet has no terminal period, or if multiple
    bullets get glued together by the regex — so this first splits on
    line-level list markers ("-", "*", "1.", "2)", etc.), then applies
    sentence-level splitting within each resulting block. This keeps each
    bullet as its own claim (or further splits a bullet with multiple
    sentences), instead of risking several distinct citations/facts being
    silently merged into a single claim that only gets checked once.
    """
    answer = answer.strip()
    if not answer:
        return []

    # First pass: split into lines, treating each bullet/numbered list item
    # as its own block. Lines that aren't list items get grouped with
    # adjacent non-list lines (e.g. an intro sentence before a list).
    lines = answer.split("\n")
    blocks = []
    current_block = []
    bullet_marker = re.compile(r'^\s*(?:[-*•]|\d{1,3}[.)])\s+')

    for line in lines:
        if bullet_marker.match(line):
            if current_block:
                blocks.append(" ".join(current_block))
                current_block = []
            blocks.append(bullet_marker.sub("", line).strip())
        elif line.strip():
            current_block.append(line.strip())
        # blank lines just act as separators, handled implicitly
    if current_block:
        blocks.append(" ".join(current_block))

    # Second pass: sentence-split within each block, in case a single
    # bullet contains more than one distinct factual sentence.
    claims = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        raw_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z]|$)', block)
        claims.extend(s.strip() for s in raw_sentences if s.strip())

    return claims


def check_citation_validity(claim: str, chunks: list[dict]) -> list[dict]:
    """Check every 'Section N' citation stated in a claim against actually-
    verified chunks with that exact section number.

    A claim can contain more than one citation (e.g. a merged bullet list,
    or a sentence citing two sections at once) — checking only the first
    match would silently miss a fabricated citation later in the same
    claim. Returns a list of {cited_section, citation_valid} dicts, one per
    citation found in the claim (empty list if the claim states no
    citation at all).
    """
    cited_sections = CITATION_IN_CLAIM_PATTERN.findall(claim)
    checks = []
    for cited_section in cited_sections:
        citation_valid = any(
            c.get("section_status") == "verified" and c.get("section_num") == cited_section
            for c in chunks
        )
        checks.append({"cited_section": cited_section, "citation_valid": citation_valid})
    return checks


def strip_invalid_citations(claim: str, citation_checks: list[dict]) -> str:
    """Replace any invalid 'Section N' citation in the claim text with a
    neutral '[citation unverified]' marker.

    Design choice: a fabricated citation shouldn't discard an otherwise
    correct, independently-grounded fact. The underlying content is judged
    on its own merits by the entailment check; a bad citation is surfaced
    as a flag rather than a veto. This trades some strictness for a higher
    answer rate — see FAITHFULNESS_THRESHOLD / citation_checks for the
    full picture of what was caught.
    """
    cleaned = claim
    for cc in citation_checks:
        if cc["citation_valid"]:
            continue
        section_num = re.escape(cc["cited_section"])
        # Covers "(Cited from Section N)", "(Source: Section N)", a bare
        # "Section N:" heading, or a plain "Section N" mention -- each
        # replaced with a neutral marker instead of the false citation.
        pattern = re.compile(
            rf'\(?\s*(?:Cited from|Source:?)?\s*Sec(?:tion)?\.?\s+{section_num}\)?:?'
        )
        cleaned = pattern.sub("[citation unverified]", cleaned)
    return cleaned


def _build_entailment_prompt(claims: list[str], context: str) -> str:
    numbered_claims = "\n".join(f"{i}. {c}" for i, c in enumerate(claims))
    return f"""You are a strict fact-checker. You will be given a CONTEXT and a list of
CLAIMS. For each claim, decide whether it is directly supported by the context.

Rules:
- "supported": true only if the context explicitly contains information that
  confirms the claim. Do not use outside knowledge — judge only against the
  given context.
- "supported": false if the context does not mention it, contradicts it, or
  only partially supports it.

CONTEXT:
{context}

CLAIMS:
{numbered_claims}

Respond with ONLY a JSON array, no other text, no markdown formatting, in this
exact structure:
[{{"claim_index": 0, "supported": true}}, {{"claim_index": 1, "supported": false}}, ...]
"""


def _parse_json_response(raw: str) -> list[dict]:
    """Strip common LLM formatting artifacts (markdown fences, stray text)
    and parse the JSON array. Raises ValueError if parsing still fails."""
    cleaned = raw.strip()
    cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
    cleaned = re.sub(r'\s*```$', '', cleaned)
    cleaned = cleaned.strip()
    return json.loads(cleaned)


def check_claims_faithfulness(claims: list[str], chunks: list[dict], api_key: str | None = None) -> list[dict]:
    """For each claim, check whether it's supported by the retrieved context.

    Returns a list of {claim, supported} dicts, one per input claim, always
    in the same order as the input — even if the LLM's JSON response comes
    back out of order or incomplete, so downstream code can rely on
    positional alignment.
    """
    if not claims:
        return []

    context = "\n\n".join(c["text"] for c in chunks)
    prompt = _build_entailment_prompt(claims, context)

    response = create_chat_completion(
        api_key,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    raw = response.choices[0].message.content

    try:
        parsed = _parse_json_response(raw)
        verdicts_by_index = {item["claim_index"]: bool(item["supported"]) for item in parsed}
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        # If the checker itself fails to return valid JSON, fail safe:
        # mark everything as unsupported rather than silently assuming
        # groundedness. Better to under-trust than over-trust.
        verdicts_by_index = {}

    results = []
    for i, claim in enumerate(claims):
        content_supported = verdicts_by_index.get(i, False)
        citation_checks = check_citation_validity(claim, chunks)
        has_invalid_citation = any(not c["citation_valid"] for c in citation_checks)

        # A fabricated citation is flagged and stripped, but no longer
        # vetoes an otherwise content-grounded claim -- see
        # strip_invalid_citations() for the reasoning.
        display_claim = strip_invalid_citations(claim, citation_checks) if has_invalid_citation else claim

        results.append({
            "claim": display_claim,
            "original_claim": claim if has_invalid_citation else None,
            "content_supported": content_supported,
            "citation_checks": citation_checks,
            "has_unverified_citation": has_invalid_citation,
            "supported": content_supported,
        })
    return results


def compute_faithfulness_score(claim_results: list[dict]) -> float:
    """Fraction of claims that are grounded in the retrieved context.
    Returns 1.0 for an empty claim list (nothing to be unfaithful about)."""
    if not claim_results:
        return 1.0
    supported_count = sum(1 for r in claim_results if r["supported"])
    return supported_count / len(claim_results)


def evaluate_answer(answer: str, chunks: list[dict], api_key: str | None = None) -> dict:
    """Full faithfulness evaluation of a generated answer.

    Returns:
        {
            "claims": [...],
            "score": float,              # fraction of claims with grounded CONTENT
            "citation_accuracy": float,   # fraction of stated citations that were valid
                                           # (1.0 if no citations were stated at all)
            "declined": bool,
            "should_abstain": bool,       # declined OR score below threshold
        }

    Content groundedness and citation accuracy are tracked as separate
    metrics: a fabricated citation is flagged (and stripped from the
    displayed claim text) but no longer forces an otherwise-grounded claim
    to fail. See strip_invalid_citations() for the reasoning.
    """
    claims = split_into_claims(answer)
    claim_results = check_claims_faithfulness(claims, chunks, api_key)
    score = compute_faithfulness_score(claim_results)
    declined = is_refusal(answer)

    all_citation_checks = [cc for c in claim_results for cc in c["citation_checks"]]
    if all_citation_checks:
        valid_count = sum(1 for cc in all_citation_checks if cc["citation_valid"])
        citation_accuracy = valid_count / len(all_citation_checks)
    else:
        citation_accuracy = 1.0  # no citations stated -- nothing to be wrong about

    return {
        "claims": claim_results,
        "score": score,
        "citation_accuracy": citation_accuracy,
        "declined": declined,
        "should_abstain": declined or (score < FAITHFULNESS_THRESHOLD),
    }


if __name__ == "__main__":
    # Usage: python src/faithfulness.py "your question here"
    # Falls back to a default test question if none is given.
    import sys
    from retrieve import retrieve
    from generate import generate_answer

    q = " ".join(sys.argv[1:]) or "How many days does a public authority have to respond to an RTI request?"
    print(f"Q: {q}\n")

    chunks = retrieve(q)
    answer = generate_answer(q, chunks)
    print(f"=== ANSWER ===\n{answer}\n")

    result = evaluate_answer(answer, chunks)
    print(f"=== FAITHFULNESS SCORE: {result['score']:.0%} "
          f"(citation accuracy: {result['citation_accuracy']:.0%}) ===\n")
    for c in result["claims"]:
        mark = "✓" if c["supported"] else "✗"
        note = ""
        if c.get("has_unverified_citation"):
            invalid = [cc["cited_section"] for cc in c["citation_checks"] if not cc["citation_valid"]]
            sections_str = ", ".join(f"Section {s}" for s in invalid)
            note = f"  [citation unverified: {sections_str}, but content judged independently]"
        print(f"[{mark}] {c['claim']}{note}")

    if result["should_abstain"]:
        print("\n⚠️  Below threshold — system would abstain from this answer.")
