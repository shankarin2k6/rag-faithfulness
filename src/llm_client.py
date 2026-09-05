"""
Provider-agnostic LLM client + chat completion.

The Streamlit sidebar lets a user paste in an API key from more than one
provider -- not just Groq's. Rather than adding a provider dropdown, the
key itself is used to detect the provider: each vendor's keys have a
distinct, stable prefix (Groq: "gsk_", NVIDIA build.nvidia.com/NIM:
"nvapi-"). Both Groq's and NVIDIA's hosted APIs expose the identical
OpenAI-style `.chat.completions.create(...)` interface and response
shape, so generate.py and faithfulness.py never need to know or care
which provider they're actually talking to -- they just call
create_chat_completion(api_key, messages, ...) and get a response back.

NVIDIA's hosted catalog in particular rotates and retires models with
very little notice (this file used to hardcode a single NVIDIA model
name, which NVIDIA end-of-lifed days later). To avoid that recurring,
NVIDIA calls first check the provider's own live /v1/models list and
pick the first still-served candidate from NVIDIA_MODEL_PREFERENCE, and
if the model that was actually used to make the request turns out to
have been retired between that check and the request itself, the call
is retried against the next candidate automatically.

To add another provider: give it a base URL + default model(s) in
config.py, then add one more prefix check in resolve_provider().
"""
from groq import Groq

from config import (
    GROQ_API_KEY,
    GENERATION_MODEL,
    NVIDIA_BASE_URL,
    NVIDIA_MODEL_PREFERENCE,
)

# Cache of which NVIDIA model was last confirmed live, per API key, so
# every single call doesn't have to re-fetch /v1/models. Cleared for a
# given key whenever that model turns out to be dead after all (see
# create_chat_completion), so a retirement is picked up on the very next
# call rather than being stuck on a stale cached choice.
_nvidia_model_cache: dict[str, str] = {}


def resolve_provider(api_key: str | None = None):
    """Return (provider_name, client) for whichever provider `api_key`
    belongs to. provider_name is "groq" or "nvidia".

    Falls back to the .env-configured GROQ_API_KEY (CLI/local dev use)
    if no explicit key is passed in. Raises RuntimeError if no key is
    available anywhere.
    """
    key = (api_key or GROQ_API_KEY or "").strip()
    if not key:
        raise RuntimeError(
            "No API key available. Copy .env.example to .env and add your "
            "key for local use, or enter one (Groq or NVIDIA) in the app's "
            "sidebar."
        )

    if key.startswith("nvapi-"):
        # NVIDIA (build.nvidia.com / NIM) -- OpenAI-compatible endpoint.
        # Imported lazily so the `openai` package is only required if an
        # NVIDIA key is actually used.
        from openai import OpenAI
        return "nvidia", OpenAI(base_url=NVIDIA_BASE_URL, api_key=key)

    # Default: Groq. Groq keys start with "gsk_", but anything not
    # matching a known provider's prefix falls through here too, since
    # Groq was this project's original/default provider.
    return "groq", Groq(api_key=key)


def _pick_nvidia_model(client, key: str) -> str:
    """Return an NVIDIA model that's actually live right now, checking
    NVIDIA's own catalog rather than trusting a hardcoded name."""
    if key in _nvidia_model_cache:
        return _nvidia_model_cache[key]

    try:
        available = {m.id for m in client.models.list().data}
    except Exception:
        # Couldn't reach the catalog listing itself -- fall back to our
        # best guess rather than failing outright; the real call below
        # will surface a clear error if this guess is also wrong.
        available = set()

    chosen = next(
        (m for m in NVIDIA_MODEL_PREFERENCE if not available or m in available),
        NVIDIA_MODEL_PREFERENCE[0],
    )
    _nvidia_model_cache[key] = chosen
    return chosen


def _is_model_unavailable_error(e: Exception) -> bool:
    """True if `e` looks like 'this model doesn't exist / was retired',
    as opposed to e.g. a bad API key or a rate limit -- so we only
    fall through to the next candidate model for the right reason."""
    status = getattr(e, "status_code", None)
    if status in (404, 410):
        return True
    msg = str(e).lower()
    return any(s in msg for s in (
        "no longer available", "end of life", "does not exist",
        "model_not_found", "model_decommissioned", "'gone'",
    ))


def create_chat_completion(api_key: str | None, messages: list[dict], temperature: float = 0.0):
    """Provider-agnostic chat completion. Picks the right client/model
    for whichever API key was given, and -- for NVIDIA -- automatically
    retries against the next candidate model if the chosen one has been
    retired since it was last checked."""
    provider, client = resolve_provider(api_key)

    if provider == "groq":
        return client.chat.completions.create(
            model=GENERATION_MODEL, messages=messages, temperature=temperature,
        )

    # provider == "nvidia"
    key = (api_key or "").strip()
    tried = set()
    last_error = None
    while True:
        model = _pick_nvidia_model(client, key)
        if model in tried:
            # Already tried and failed this run; move to the next
            # untried candidate directly instead of looping forever.
            remaining = [m for m in NVIDIA_MODEL_PREFERENCE if m not in tried]
            if not remaining:
                raise last_error or RuntimeError("No working NVIDIA model found.")
            model = remaining[0]
        tried.add(model)
        try:
            return client.chat.completions.create(
                model=model, messages=messages, temperature=temperature,
            )
        except Exception as e:
            if not _is_model_unavailable_error(e) or len(tried) >= len(NVIDIA_MODEL_PREFERENCE):
                raise
            # This model's dead -- drop the stale cache entry so the next
            # call (and the retry below) picks a fresh one, and try again.
            _nvidia_model_cache.pop(key, None)
            last_error = e
