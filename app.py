"""
Full UI: Retrieve -> Generate -> Verify.

Wires together the whole pipeline built across Weeks 1-3: retrieval with
verified/unverified source labeling, generation with citation-aware
prompting, and the faithfulness layer that scores content groundedness and
citation accuracy as separate, transparent metrics.

Chat-style interface: messages render as a normal chat thread (oldest
first), documents are attached via the "+" button next to the chat box
instead of a separate sidebar uploader, and new answers type themselves
out rather than appearing all at once.

Run:
    streamlit run app.py
"""
from __future__ import annotations

import sys
import html
import json
import threading
import time
import uuid
import base64
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

_ASSETS_DIR = Path(__file__).resolve().parent / "assets"

# Custom user avatar (falls back to Streamlit's default person icon if the
# asset is missing, e.g. if this file is moved without its assets/ folder).
_USER_AVATAR_PATH = _ASSETS_DIR / "user_avatar.gif"
USER_AVATAR = str(_USER_AVATAR_PATH) if _USER_AVATAR_PATH.exists() else None


def _gif_data_uri(filename: str) -> str | None:
    """Base64-encode a GIF from assets/ for inline use in st.markdown HTML
    (a local file path in an <img src=...> won't resolve in the browser,
    so this is the reliable way to embed one). Returns None if missing,
    so callers can fall back to an emoji instead of a broken image."""
    path = _ASSETS_DIR / filename
    if not path.exists():
        return None
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/gif;base64,{encoded}"


WARNING_ICON_URI = _gif_data_uri("warning_icon.gif")
API_KEY_ICON_URI = _gif_data_uri("api_key_icon.gif")
LOADER_ICON_URI = _gif_data_uri("loader_spinner.gif")
USER_AVATAR_URI = _gif_data_uri("user_avatar.gif")


def status_html(text: str) -> str:
    """A status line for the live-processing steps (Retrieving/Generating/
    Checking), using the animated spinner GIF if available, falling back
    to plain italic text if the asset is missing."""
    if not LOADER_ICON_URI:
        return f"*{text}*"
    return (
        f'<div style="display:flex;align-items:center;gap:8px;color:#c9c9d4;margin-top:-6px;">'
        f'<img src="{LOADER_ICON_URI}" width="20" height="20" '
        f'style="flex-shrink:0;position:relative;top:-4px;">'
        f'<span><em>{text}</em></span>'
        f'</div>'
    )

import chromadb
import streamlit as st
import streamlit.components.v1 as components
from retrieve import retrieve, _get_embedder
from generate import generate_answer
from faithfulness import evaluate_answer
from ingest import extract_text_from_upload, index_documents
from config import FAITHFULNESS_THRESHOLD, CHROMA_DIR, DATA_RAW_DIR, COLLECTION_NAME

# ---------------------------------------------------------------------------
# Startup: rebuild ChromaDB index if the persistent directory is empty or
# missing (e.g. on Streamlit Community Cloud where the filesystem is
# ephemeral). The raw PDF is committed to the repo, so we can always
# re-chunk and re-embed from it.
# ---------------------------------------------------------------------------
def _ensure_chroma_index():
    """Build the vector index from data/raw/ if it doesn't already exist."""
    import chromadb as _chroma
    try:
        _client = _chroma.PersistentClient(path=str(CHROMA_DIR))
        _col = _client.get_collection(COLLECTION_NAME)
        if _col.count() > 0:
            return  # index exists and has data
    except Exception:
        pass  # collection doesn't exist or is empty

    # No usable index -- rebuild from raw documents.
    try:
        from ingest import build_index as _build_index
        _build_index()
    except Exception as _e:
        print(f"[startup] Could not rebuild ChromaDB index: {_e}")
        print("[startup] The app will still run, but retrieval may fail.")

_ensure_chroma_index()

st.set_page_config(page_title="Faithfulness-Aware RAG", page_icon="\U0001F4DA", layout="centered")

# ---------------------------------------------------------------------------
# One-time boot loader: dims the whole screen with a centered spinning icon
# from the moment the page loads until the chat input has actually
# rendered, so the person sees a proper "app is starting" state instead of
# a flash of partial/unstyled UI while Streamlit builds the page. Shown
# only once per session (guarded by session_state) -- never again on later
# reruns triggered by asking a question, uploading a file, etc.
# ---------------------------------------------------------------------------
if "app_booted" not in st.session_state:
    st.session_state.app_booted = False

if not st.session_state.app_booted:
    st.markdown("""
    <style>
    #boot-loader-overlay {
        position: fixed;
        inset: 0;
        width: 100vw;
        height: 100vh;
        background: rgba(0, 0, 0, 0.88);
        z-index: 20000;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: opacity 0.3s ease;
    }
    #boot-loader-overlay svg {
        width: 56px;
        height: 56px;
        color: #ffffff;
        animation: boot-loader-spin 0.9s linear infinite;
    }
    @keyframes boot-loader-spin {
        to { transform: rotate(360deg); }
    }
    </style>
    <div id="boot-loader-overlay">
      <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="none" viewBox="0 0 24 24">
        <mask id="bootSpinnerMask" width="22" height="22" x="1" y="1" maskUnits="userSpaceOnUse" style="mask-type:alpha">
          <path fill="#fff" fill-rule="evenodd" d="M23 12c0 6.075-4.925 11-11 11S1 18.075 1 12 5.925 1 12 1s11 4.925 11 11M12.75 4.75a.75.75 0 1 0-1.5 0v2.5a.75.75 0 1 0 1.5 0zM7.404 6.343a.75.75 0 0 0-1.061 1.06l1.767 1.77A.75.75 0 0 0 9.17 8.11zm10.253 1.06a.75.75 0 0 0-1.061-1.06l-1.768 1.768a.75.75 0 1 0 1.06 1.06zM4.75 11.25a.75.75 0 0 0 0 1.5h2.5a.75.75 0 0 0 0-1.5zm12 0a.75.75 0 0 0 0 1.5h2.5a.75.75 0 0 0 0-1.5zm-7.579 4.639a.75.75 0 0 0-1.06-1.06l-1.768 1.767a.75.75 0 0 0 1.06 1.06zm6.718-1.06a.75.75 0 0 0-1.06 1.06l1.767 1.768a.75.75 0 1 0 1.06-1.061zM12.75 16.75a.75.75 0 0 0-1.5 0v2.5a.75.75 0 0 0 1.5 0z" clip-rule="evenodd"/>
        </mask>
        <g mask="url(#bootSpinnerMask)">
          <path fill="currentColor" stroke="currentColor" stroke-linecap="round" stroke-miterlimit="10" stroke-width="1.5" d="M12 21.25a9.25 9.25 0 1 0 0-18.5 9.25 9.25 0 0 0 0 18.5Z"/>
        </g>
      </svg>
    </div>
    """, unsafe_allow_html=True)

    components.html("""
    <script>
    (function() {
        var doc = window.parent.document;
        function tryHide(attempts) {
            var input = doc.querySelector('[data-testid="stChatInput"]');
            var overlay = doc.getElementById('boot-loader-overlay');
            if (input && overlay) {
                overlay.style.opacity = '0';
                // Only ever hide this via CSS, never detach/remove it from
                // the DOM. This div was rendered by Streamlit's own React
                // tree (st.markdown produced it), so React still expects
                // to own and manage its lifecycle -- physically removing
                // it out from under React with a raw removeChild() call
                // leaves React's internal tracking pointing at a node
                // that's no longer where it expects, and the next time
                // React itself tries to reconcile/remove that node (on a
                // later rerender), its own removeChild() call fails with
                // "the node to be removed is not a child of this node".
                // Setting display:none instead leaves the node in place
                // for React to keep managing normally -- just invisible.
                setTimeout(function() {
                    if (overlay) {
                        overlay.style.display = 'none';
                        overlay.style.pointerEvents = 'none';
                    }
                }, 300);
                return;
            }
            if (attempts < 100) setTimeout(function() { tryHide(attempts + 1); }, 100);
        }
        tryHide(0);
    })();
    </script>
    """, height=0)

    st.session_state.app_booted = True


PLACEHOLDER_TEXT = (
    "Retrieve → Generate → Verify. The system checks its own answer "
    "against the retrieved evidence."
)

# ---------------------------------------------------------------------------
# Styling: scroll buttons, chat bubbles, the Claude/ChatGPT-style input pill
# with a "+" file-attach button, and the ingest overlay.
# ---------------------------------------------------------------------------
st.markdown("""
<style>
html { scroll-behavior: smooth; }

#scroll-to-bottom-btn, #scroll-to-top-btn {
    position: fixed;
    right: 24px;
    width: 44px;
    height: 44px;
    background-color: #FF4B4B;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    z-index: 9999;
    cursor: pointer;
    border: none;
    transition: transform 0.15s ease, background-color 0.15s ease, opacity 0.2s ease;
    overflow: hidden;
}
#scroll-to-bottom-btn { bottom: 96px; }
#scroll-to-top-btn { bottom: 152px; }
#scroll-to-bottom-btn:hover, #scroll-to-top-btn:hover {
    background-color: #E03C3C;
    transform: scale(1.08);
}

/* Fade-out + click-through-disable for both buttons when hidden by the
   scroll-position script below, instead of display:none, so the opacity
   transition is smooth rather than an abrupt pop. */
.scroll-btn-hidden {
    opacity: 0 !important;
    pointer-events: none !important;
}
#scroll-to-bottom-btn svg, #scroll-to-top-btn svg {
    width: 20px;
    height: 20px;
    color: white;
    stroke: currentColor;
    display: block;
    margin: auto;
}

/* The up button reuses the exact same down-chevron icon and the exact
   same bounce keyframes as the down button. A wrapper div flips the icon
   vertically (making it point up); the icon's own bounce animation then
   runs inside that already-flipped coordinate space, which mirrors the
   motion automatically -- so "moving down" in the wrapper's flipped frame
   reads as "moving up" on screen. No separate up-arrow asset or a second
   set of keyframes needed. */
#scroll-to-top-btn .icon-flip-wrapper {
    transform: scaleY(-1);
    display: flex;
}

/* Looping bounce: arrow drops down and fades, then resets to the top and
   repeats, for as long as the cursor stays over the button. Removing the
   :hover state (mouse leaves) simply stops the animation and the icon
   returns to its normal static position. */
@keyframes scroll-btn-bounce {
    0%   { transform: translateY(-6px); opacity: 0; }
    35%  { opacity: 1; }
    65%  { transform: translateY(6px); opacity: 1; }
    100% { transform: translateY(6px); opacity: 0; }
}
#scroll-to-bottom-btn:hover svg, #scroll-to-top-btn:hover svg {
    animation: scroll-btn-bounce 0.9s ease-in-out infinite;
}

/* Ingestion overlay: dims the whole page and shows an animated 3-dot
   loader with a small "Ingesting..." label underneath. Shown via a
   placeholder while a document upload is being chunked/embedded/indexed,
   and cleared once that work finishes. */
#ingest-overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    background: rgba(0, 0, 0, 0.65);
    z-index: 10000;
    display: flex;
    align-items: center;
    justify-content: center;
}
.ingest-loader-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 14px;
}
.ingest-loader-text {
    color: #ffffff;
    font-size: 14px;
    font-weight: 500;
    letter-spacing: 0.02em;
}
@keyframes ingest-dot-pulse {
    0%, 80%, 100% { opacity: 0.25; }
    40% { opacity: 1; }
}
.ingest-loader circle {
    animation: ingest-dot-pulse 1.2s ease-in-out infinite;
}
.ingest-loader .dot-1 { animation-delay: 0s; }
.ingest-loader .dot-2 { animation-delay: 0.2s; }
.ingest-loader .dot-3 { animation-delay: 0.4s; }

/* Reserve space at the bottom of the scrollable main area equal to (and a
   bit more than) the fixed input bar's own height. Without this, content
   that lands at the very end of the page -- most notably the "Retrieving.../
   Generating.../Checking faithfulness..." spinners that appear while a
   new turn is being built, before the page has scrolled to it -- renders
   underneath the fixed input bar instead of above it, since the bar sits
   outside normal document flow and nothing else reserves room for it. */
section[data-testid="stMain"] {
    padding-bottom: 320px !important;
}
/* With the page title moved into the sidebar, the main content area no
   longer has anything sitting in the default top gap Streamlit reserves
   above the block container -- trim it down so chat messages start near
   the top instead of leaving that space empty. */
[data-testid="stMainBlockContainer"] {
    padding-top: 1.5rem !important;
}
[data-testid="stChatMessage"] {
    border-radius: 14px;
    padding: 4px 2px;
}

/* --------------------------------------------------------------------
   Chat input: wider, fully elliptical pill for the text box, with the
   "+" file-attach button pulled outside of it as its own separate
   circular button (rather than living inside the pill's left edge).

   Important: we deliberately do NOT set max-width/margin/width on
   [data-testid="stChatInput"] itself -- that element's sizing comes
   from Streamlit's flex layout, and overriding it directly collapses
   the whole input down to a tiny shrink-to-fit box. Instead we widen
   its stable ancestor (stBottomBlockContainer) and only add shape/shadow
   to the input element itself.
   -------------------------------------------------------------------- */
/* The "+" upload button lives to the LEFT of the pill as its own
   circular button (see below), so the pill's own box only gets part
   of the row -- we reserve BTN_RESERVE px of left padding on the
   container for it. Because that padding is INSIDE max-width (via
   border-box), the button + pill read as a single centered unit
   instead of the button poking out past the container's true center.
   BTN_SIZE (36) + GAP (26) = BTN_RESERVE (62), a wide enough gap that
   the button reads as clearly separate rather than fused to the box. */
[data-testid="stBottomBlockContainer"] {
    max-width: 822px !important;
    box-sizing: border-box !important;
    padding-left: 62px !important;
    /* Small nudge left of dead-center -- the "+" button reserved via
       padding-left above visually reads as making the whole unit sit
       slightly right of center, so this compensates a bit further. */
    transform: translateX(-90px);
}
/* The sticky footer wrapper Streamlit puts around the chat input has its
   own fixed height sized for the original compact input. Adding padding
   below made our box taller than that wrapper allows, so its default
   overflow:hidden was clipping the tops off both the box and the "+"
   button. Letting height/overflow expand to fit fixes the crop. */
[data-testid="stBottom"] {
    /* Explicitly pin the entire chat composer to the viewport.
       Streamlit can otherwise let this wrapper participate in the
       scrolling document in some layouts/reruns. */
    position: fixed !important;
    left: var(--rag-bottom-left, 0px) !important;
    right: 0 !important;
    bottom: 0 !important;
    width: calc(100vw - var(--rag-bottom-left, 0px)) !important;
    max-width: none !important;
    height: auto !important;
    min-height: 0 !important;
    /* The JS below re-measures the sidebar's width on a poll (every
       200ms) rather than continuously, so without this the composer
       would visibly snap in steps while the sidebar animates open/
       closed -- even though the main content area glides smoothly via
       Streamlit's own transition. This interpolates between those
       sampled steps so the composer's movement reads as equally smooth,
       roughly matching a typical sidebar animation's duration. */
    transition: left 0.25s ease, width 0.25s ease !important;
    overflow: visible !important;
    padding-bottom: 12px !important;
    /* background-color intentionally not set here -- applied live via JS
       below, reading the chat area's own actual computed background so
       it can never drift out of sync with a hardcoded guess. */
    z-index: 9990 !important;
    transition: left 0.28s ease, width 0.28s ease;
}
[data-testid="stBottomBlockContainer"] {
    height: auto !important;
    overflow: visible !important;
}
/* --------------------------------------------------------------------
   Strategy: rather than chasing Streamlit's own nested boxes (the chat
   input's outer wrapper vs. the inner box that actually paints the
   background -- these have different default sizes, which produces a
   "ghost ellipse behind the box" effect no matter how the corners are
   rounded), make every one of Streamlit's own backgrounds/borders/
   shadows fully transparent and draw a single shape on
   stBottomBlockContainer instead -- one visual layer, so there is
   nothing left to show through as a mismatched shape behind it.
   -------------------------------------------------------------------- */
[data-testid="stChatInput"],
[data-testid="stChatInput"] * {
    background-color: transparent !important;
    box-shadow: none !important;
    border-color: transparent !important;
}
[data-testid="stChatInput"] {
    position: relative !important;
    overflow: visible !important;
    min-height: 34px !important;
    padding: 2px 14px !important;
}
/* Claude-style input: a fixed, moderate corner radius (not a full pill)
   so a single line reads as a soft stadium shape, but as the textarea
   grows to two or three lines the box reads as a rounded rectangle
   instead of stretching into an odd elongated capsule. */
[data-testid="stBottomBlockContainer"]:has([data-testid="stChatInput"]) {
    background-color: rgb(48, 49, 58) !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    border-radius: 22px !important;
    box-shadow: 0 4px 18px rgba(0,0,0,0.28) !important;
    padding: 3px 6px !important;
    transition: border-color 0.15s ease, box-shadow 0.15s ease;
}
[data-testid="stBottomBlockContainer"]:has([data-testid="stChatInput"]:focus-within) {
    border-color: rgba(255,255,255,0.22) !important;
    box-shadow: 0 4px 18px rgba(0,0,0,0.28), 0 0 0 3px rgba(255,255,255,0.06) !important;
}
/* Best-effort: if a file is attached, Streamlit shows a filename/remove
   chip above the text row, which includes a "remove" button. That extra
   row only needs a touch more corner radius than the default, so it
   doesn't look pinched -- no shape change needed beyond that. */
[data-testid="stBottomBlockContainer"]:has(button[aria-label*="emove" i]) {
    border-radius: 24px !important;
}
[data-testid="stChatInputTextArea"] {
    min-height: 20px !important;
    max-height: 200px !important;
    padding-top: 4px !important;
    padding-bottom: 4px !important;
    padding-left: 10px !important;
    margin-left: 4px !important;
    font-size: 15px !important;
    line-height: 1.5 !important;
    white-space: pre-wrap !important;
    overflow-wrap: break-word !important;
    overflow-y: auto !important;
    resize: none !important;
}
/* Pull the "+" button out of the box entirely and turn it into its own
   floating circular button just to the left of the text box. Its
   left offset exactly matches the container's left padding above, so
   it sits inside that reserved space rather than overflowing past the
   container's edge -- that's what keeps button + box centered as one
   unit instead of the box alone being centered and the button hanging
   off to the side. */
[data-testid="stChatInputFileUploadButton"] {
    position: absolute !important;
    left: -62px !important;
    top: 0 !important;
    bottom: 0 !important;
    margin: auto !important;
    width: 36px !important;
    height: 36px !important;
    min-width: 36px !important;
    border-radius: 50% !important;
    background-color: #4f46e5 !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.35) !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    border: none !important;
    z-index: 5 !important;
    transition: transform 0.15s ease, background-color 0.15s ease;
}
[data-testid="stChatInputFileUploadButton"]:hover:not(:disabled) {
    background-color: #6058e9 !important;
    transform: scale(1.08) !important;
}
[data-testid="stChatInputFileUploadButton"] svg {
    color: white !important;
    fill: white !important;
    width: 16px !important;
    height: 16px !important;
}
/* Keep the send button circular and matched to the box's height. */
[data-testid="stChatInputSubmitButton"] {
    border-radius: 50% !important;
    background-color: rgba(255,255,255,0.08) !important;
}

/* Title is now handled by facade.TopBar (fixed header bar). */

/* Smooth sidebar transitions: apply transition to main content elements
   so width/position changes animate instead of snapping. */
section[data-testid="stMain"],
[data-testid="stMain"] > div {
    transition: width 0.3s ease, padding 0.3s ease, margin 0.3s ease;
}

/* Small gap above the question when it's scrolled to the top of the
   viewport via scrollIntoView (see _scroll_question_to_top), so it isn't
   flush against the very top edge. scroll-margin-top is the standard,
   native way to express this -- scrollIntoView respects it automatically,
   no manual pixel-offset math needed. */
[id^="qa-anchor-"] {
    scroll-margin-top: 12px;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------


# Invisible anchor near the top of the page, kept for compatibility with
# any external deep-links, though the buttons above scroll programmatically.
st.markdown('<div id="chat-top-anchor"></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Keep Streamlit's chat composer fixed to the viewport and centered within
# the visible main-content area. The sidebar width is measured live so the
# composer moves only when the sidebar opens/closes -- never when the page
# itself scrolls.
# ---------------------------------------------------------------------------
components.html("""
<script>
(function() {
    var doc = window.parent.document;

    function syncComposerBackground(bottom) {
        // Read the REAL, currently-rendered background color of the chat
        // area itself (walking a few likely candidates, since which one
        // actually paints a background can vary) and apply that exact
        // value to the footer bar -- rather than a hardcoded guess that
        // can end up looking like a visibly separate strip if it doesn't
        // precisely match the app's real theme color.
        var candidates = [
            doc.querySelector('section[data-testid="stMain"]'),
            doc.querySelector('[data-testid="stAppViewContainer"]'),
            doc.querySelector('.stApp'),
            doc.body,
        ];
        for (var i = 0; i < candidates.length; i++) {
            var el = candidates[i];
            if (!el) continue;
            var bg = window.parent.getComputedStyle(el).backgroundColor;
            if (bg && bg !== 'rgba(0, 0, 0, 0)' && bg !== 'transparent') {
                bottom.style.setProperty('background-color', bg, 'important');
                return;
            }
        }
    }

    function positionChatComposer() {
        var bottom = doc.querySelector('[data-testid="stBottom"]');
        if (!bottom) return;

        // Rather than trying to measure the sidebar itself and infer
        // where content starts from that (which kept guessing wrong --
        // first treating a collapsed rail as "open", then failing to
        // detect a genuinely open sidebar), just read the main content
        // area's own actual left edge directly. Whatever state the
        // sidebar is in, main content already visibly starts in the
        // right place -- so matching that directly can't be wrong the
        // way inferring it from the sidebar's own width kept being.
        var mainEl = doc.querySelector('[data-testid="stMain"]')
            || doc.querySelector('[data-testid="stMainBlockContainer"]');
        var leftEdge = 0;
        if (mainEl) {
            var r = mainEl.getBoundingClientRect();
            leftEdge = Math.max(0, r.left);
        }

        bottom.style.setProperty('--rag-bottom-left', leftEdge + 'px');
        bottom.style.setProperty('position', 'fixed', 'important');
        bottom.style.setProperty('bottom', '0px', 'important');
        bottom.style.setProperty('right', '0px', 'important');
        bottom.style.setProperty('width', 'calc(100vw - ' + leftEdge + 'px)', 'important');
        bottom.style.setProperty('z-index', '9990', 'important');
        syncComposerBackground(bottom);
    }

    if (window.parent.__ragFixedComposerInterval) {
        clearInterval(window.parent.__ragFixedComposerInterval);
    }
    window.parent.__ragFixedComposerInterval = setInterval(positionChatComposer, 80);

    if (window.parent.__ragFixedComposerResizeHandler) {
        window.parent.removeEventListener('resize', window.parent.__ragFixedComposerResizeHandler);
    }
    window.parent.__ragFixedComposerResizeHandler = positionChatComposer;
    window.parent.addEventListener('resize', positionChatComposer);

    positionChatComposer();
})();
</script>
""", height=0)

# ---------------------------------------------------------------------------
# Small footer credit line, inserted as a SIBLING right after
# stBottomBlockContainer (not appended inside it). That container is
# Streamlit's flex row laying the "+" button, textarea, and send button out
# side by side -- appending another element inside that same row broke the
# row's layout, which is what caused the footer to visually merge into the
# input box and block typing/uploading. Keeping the footer as a sibling
# leaves that row untouched, and giving the footer the exact same
# max-width/padding-left/transform recipe as the pill (see the
# stBottomBlockContainer CSS rule above) makes it line up with the pill
# without needing to live inside it.
# ---------------------------------------------------------------------------
components.html("""
<script>
(function() {
    var doc = window.parent.document;

    function addFooter() {
        if (doc.getElementById("rag-footer-credit")) return true;
        var pill = doc.querySelector('[data-testid="stBottomBlockContainer"]');
        if (!pill || !pill.parentNode) return false;
        var footer = doc.createElement("div");
        footer.id = "rag-footer-credit";
        footer.textContent = "Developed by Shankari N";
        footer.style.cssText =
            "max-width:822px;box-sizing:border-box;padding-left:62px;" +
            "margin:0 auto;transform:translateX(-90px);" +
            "text-align:center;color:rgba(255,255,255,0.4);font-size:11px;" +
            "padding-top:6px;padding-bottom:4px;user-select:none;" +
            "background:transparent;pointer-events:none;";
        pill.parentNode.insertBefore(footer, pill.nextSibling);
        return true;
    }

    if (!addFooter()) {
        var attempts = 0;
        var poll = setInterval(function() {
            attempts++;
            if (addFooter() || attempts > 30) clearInterval(poll);
        }, 200);
    }
})();
</script>
""", height=0)

# ---------------------------------------------------------------------------
# Animated chat-input placeholder (typewriter loop). Streamlit's own
# `placeholder=` argument only sets a static string, so this reaches into
# the underlying <textarea> and types PLACEHOLDER_TEXT out one character
# at a time, pausing while the user is actively typing.
# ---------------------------------------------------------------------------
components.html(f"""
<script>
(function() {{
    var doc = window.parent.document;
    var phrase = {PLACEHOLDER_TEXT!r};

    function getInput() {{
        return doc.querySelector('textarea[data-testid="stChatInputTextArea"]');
    }}

    function step(i) {{
        var el = getInput();
        if (!el) {{ setTimeout(function() {{ step(i); }}, 300); return; }}
        if (el.value && el.value.length > 0) {{
            el.setAttribute("placeholder", phrase);
            setTimeout(function() {{ step(i); }}, 400);
            return;
        }}
        if (i > phrase.length) {{
            setTimeout(function() {{ step(0); }}, 1800);
            return;
        }}
        el.setAttribute("placeholder", phrase.slice(0, i));
        setTimeout(function() {{ step(i + 1); }}, 30);
    }}

    if (!window.parent.__ragPlaceholderAnimRunning) {{
        window.parent.__ragPlaceholderAnimRunning = true;
        step(0);
    }}
}})();
</script>
""", height=0)

# ---------------------------------------------------------------------------
# Sidebar: bring-your-own API key + document source status. Document
# uploads themselves now happen via the "+" button next to the chat box
# (below), the same way Claude/ChatGPT/Gemini attach files inline instead
# of through a separate sidebar form.
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("📚 Faithfulness-Aware RAG")
    st.divider()
    st.header("Settings")

    # Custom label (icon + text) drawn above the input, sized to match the
    # label text itself. The input's own native label is hidden below
    # (label_visibility="collapsed") since Streamlit widget labels only
    # support a small markdown subset and can't embed an <img>.
    if API_KEY_ICON_URI:
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:2px;">'
            f'<img src="{API_KEY_ICON_URI}" width="34" height="34" style="flex-shrink:0;">'
            f'<span style="font-size:14px;font-weight:400;">Groq API key</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    user_api_key = st.text_input(
        "Groq API key",
        type="password",
        placeholder="gsk_...",
        label_visibility="collapsed" if API_KEY_ICON_URI else "visible",
        help="Get a free key at https://console.groq.com/keys. "
             "Your key is used only for your session and is never stored or logged.",
    )
    active_api_key = user_api_key.strip()
    if active_api_key:
        st.success("Using your API key for this session.")
    else:
        st.warning("No API key available yet. Enter one above to ask questions.")

    st.divider()
    st.caption(
        "This demo answers questions about the Right to Information Act, 2005 "
        "(India), using the PDF ingested into the local vector store. Attach a "
        "PDF or TXT file with the **+** button next to the chat box to ask "
        "about your own document instead."
    )
    st.caption(
        f"Faithfulness threshold: {FAITHFULNESS_THRESHOLD:.0%}. Answers scoring "
        f"below this, or that the model declines to answer, are flagged rather "
        f"than shown as confident fact."
    )

    st.divider()
    st.subheader("Document source")
    if st.session_state.get("uploaded_collection") is not None:
        st.success(
            f"Using **{st.session_state.uploaded_file_name}** "
            f"({st.session_state.uploaded_chunk_count} chunks indexed)."
        )
        if st.button("↩ Switch back to built-in corpus"):
            st.session_state.uploaded_collection = None
            st.session_state.uploaded_file_name = None
            st.session_state.uploaded_chunk_count = None
            st.session_state.active_source_label = "builtin"
            st.session_state.history = []
            st.rerun()
        st.caption(
            "Note: the verified/unverified citation-check badges are tuned for "
            "structured legal Acts (numbered sections, chapters, schedules). "
            "For general documents, chunks show as 'n/a' -- retrieval and the "
            "content-faithfulness check still apply, just without that extra layer."
        )
    else:
        st.caption("Currently answering from the built-in RTI Act corpus.")

    # Fallback file uploader: some Streamlit versions or environments
    # may not expose the inline '+' attach button reliably. Provide a
    # sidebar uploader as a minimal fallback so users can still upload
    # PDFs/TXT files without changing the app flow.
    st.markdown("---")
    st.caption("Fallback uploader: use if the inline + button doesn't work.")
    _fallback_file = st.file_uploader("Upload PDF or TXT (fallback)", type=["pdf", "txt"], key="_fb_upload")
    if _fallback_file is not None:
        st.session_state["_fb_fallback"] = _fallback_file

# Main-page title was moved to the top of the sidebar (above Settings)
# per request -- intentionally not rendered here anymore.

# Scroll buttons: created in JS and appended to document.body.
components.html("""
<script>
(function() {
    var doc = window.parent.document;

    // --- Inject theme CSS variables into parent document head ---
    if (!doc.getElementById('rag-theme-vars')) {
        var ts = doc.createElement('style');
        ts.id = 'rag-theme-vars';
        ts.textContent = ':root{' +
            '--primary:#FF4B4B;--primary-foreground:#FFFFFF;' +
            '--background:#0E1117;--foreground:#E2E8F0;' +
            '--muted:#1E293B;--muted-foreground:#94A3B8;' +
            '--border:#334155;--destructive:#EF4444;' +
            '--chrome-background:#020817;--chrome-foreground:#E2E8F0;--chrome-border:#1E293B;' +
            '--font-sans:system-ui;--font-mono:monospace;--radius:0.75rem;' +
            '}';
        doc.head.appendChild(ts);
    }

    var DOWN_ICON = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 400" width="20" height="20"><path fill="none" stroke="currentColor" stroke-width="28" stroke-linecap="round" stroke-linejoin="round" d="M63.028,89.279L192.929,219.18c3.905,3.905,10.237,3.905,14.142,0l129.9-129.901c2.963-2.963,8.029-0.864,8.029,3.325v69.858c0,5.627-2.236,11.024-6.215,15.003L207.071,309.18c-3.905,3.905-10.237,3.905-14.142,0L61.214,177.465C57.235,173.486,55,168.089,55,162.462V92.604C55,88.415,60.065,86.316,63.028,89.279z"/></svg>';
    var win = window.parent;

    // --------------------------------------------------------------------
    // Root cause of "hover animates but click does nothing": every
    // components.html call runs inside its OWN throwaway iframe, and
    // Streamlit tears that iframe down on the very next rerun (e.g. the
    // moment a message is sent). The buttons themselves live in the outer
    // page (appended to doc.body) and survive that teardown fine, but a
    // click handler wired up via `addEventListener` from inside this
    // iframe is a closure lexically bound to THIS iframe's own `window` --
    // once the iframe is destroyed, that closure stops working, while a
    // CSS :hover animation (native to the browser, no JS involved) keeps
    // working forever regardless. That mismatch is exactly the symptom
    // reported.
    //
    // Fix: give each button a plain inline onclick="" *attribute* instead.
    // The browser always evaluates inline event-handler attributes in the
    // context of the document that owns the element -- the real outer
    // page -- never the iframe that happened to create the element. As
    // long as the functions those attributes call (__ragScrollToBottom /
    // __ragScrollToTop below) live on the outer `window` object itself
    // (not on this iframe's window), they keep working across every future
    // rerun, since this whole script re-executes on every rerun anyway and
    // simply reassigns them fresh each time.
    // --------------------------------------------------------------------
    function isRealScrollContainer(el) {
        if (!el) return false;
        var oy = win.getComputedStyle(el).overflowY;
        return (oy === 'auto' || oy === 'scroll') && el.scrollHeight > el.clientHeight + 4;
    }

    function findScrollEl() {
        // Height-taller-than-box alone does NOT mean an element scrolls --
        // any block element can have that while the actual overflow just
        // bubbles up to a scrollable ancestor. Must also confirm the
        // element's own computed overflow-y is auto/scroll, or clicking
        // the buttons ends up calling scrollTo() on a non-scrolling
        // element, which is a silent no-op (this was the actual bug).
        var candidates = ['section[data-testid="stMain"]', '[data-testid="stAppViewContainer"]', '.main'];
        for (var i = 0; i < candidates.length; i++) {
            var el = doc.querySelector(candidates[i]);
            if (isRealScrollContainer(el)) return el;
        }
        // Fallback: walk up from a stable content anchor checking each
        // ancestor's actual computed overflow, in case this Streamlit
        // version's real scroll container doesn't match any of the
        // testids above.
        var anchor = doc.querySelector('[data-testid="stChatMessage"]') || doc.querySelector('section[data-testid="stMain"]');
        var el = anchor ? anchor.parentElement : null;
        while (el && el !== doc.body) {
            if (isRealScrollContainer(el)) return el;
            el = el.parentElement;
        }
        return null; // real scrolling happens on documentElement/window itself
    }

    function metrics() {
        var el = findScrollEl();
        if (el) return { scrollY: el.scrollTop, viewportH: el.clientHeight, fullH: el.scrollHeight, el: el };
        var docEl = doc.documentElement;
        return {
            scrollY: win.scrollY || docEl.scrollTop || 0,
            viewportH: win.innerHeight || docEl.clientHeight,
            fullH: docEl.scrollHeight,
            el: null,
        };
    }

    win.__ragScrollToBottom = function() {
        var m = metrics();
        var el = m.el || doc.documentElement || doc.body;
        // Deliberately not computing a distance from measured
        // scrollHeight/viewportHeight here -- any mismatch between the
        // element these numbers were measured from and the element
        // actually holding our reserved bottom padding (several layers
        // of nested containers here) means that math can quietly land
        // short every time, by roughly the same margin. That's invisible
        // on a tall, fully-rendered answer (plenty of slack above the
        // shortfall to spare) but consistently clips a short, just-
        // landed question with no slack to spare at all -- exactly the
        // "only a third visible" symptom. Requesting an oversized
        // scrollTop instead sidesteps the whole calculation: browsers
        // clamp it to the true maximum automatically, so this always
        // reaches the actual bottom regardless of any measurement error.
        el.scrollTo({ top: el.scrollHeight + 99999, behavior: 'smooth' });
    };
    win.__ragScrollToTop = function() {
        var m = metrics();
        (m.el || doc.documentElement || doc.body).scrollTo({ top: 0, behavior: 'smooth' });
    };

    // Both buttons stay visible always; each just dims (and stops
    // accepting clicks) when there's no room left to scroll further in
    // its own direction, rather than disappearing outright.
    win.__ragUpdateScrollButtons = function() {
        var m = metrics();
        var scrollable = (m.fullH - m.viewportH) > 10;
        var atTop = !scrollable || m.scrollY <= 10;
        var atBottom = !scrollable || (m.scrollY + m.viewportH) >= (m.fullH - 10);
        var db = doc.getElementById('scroll-to-bottom-btn');
        var ub = doc.getElementById('scroll-to-top-btn');
        if (db) { db.style.opacity = atBottom ? '0.3' : '1'; db.style.pointerEvents = atBottom ? 'none' : 'auto'; }
        if (ub) { ub.style.opacity = atTop ? '0.3' : '1'; ub.style.pointerEvents = atTop ? 'none' : 'auto'; }
    };

    // Polling (not a scroll-event listener) so this keeps working without
    // depending on this iframe surviving -- the interval is cleared and
    // restarted on every rerun, so the timer callback always belongs to
    // the current, still-alive iframe rather than a torn-down old one.
    if (win.__ragBtnPollInterval) clearInterval(win.__ragBtnPollInterval);
    win.__ragBtnPollInterval = setInterval(win.__ragUpdateScrollButtons, 300);
    win.addEventListener('resize', win.__ragUpdateScrollButtons);

    if (!doc.getElementById('scroll-to-bottom-btn')) {
        var db = doc.createElement('button');
        db.id = 'scroll-to-bottom-btn';
        db.innerHTML = DOWN_ICON;
        db.style.cssText = 'position:fixed !important;right:24px;bottom:96px;opacity:1;pointer-events:auto;z-index:99999;transition:opacity 0.2s ease;';
        db.setAttribute('onclick', 'window.__ragScrollToBottom && window.__ragScrollToBottom()');
        doc.body.appendChild(db);
    }
    if (!doc.getElementById('scroll-to-top-btn')) {
        var ub = doc.createElement('button');
        ub.id = 'scroll-to-top-btn';
        ub.innerHTML = '<div class="icon-flip-wrapper">' + DOWN_ICON + '</div>';
        ub.style.cssText = 'position:fixed !important;right:24px;bottom:152px;opacity:1;pointer-events:auto;z-index:99999;transition:opacity 0.2s ease;';
        ub.setAttribute('onclick', 'window.__ragScrollToTop && window.__ragScrollToTop()');
        doc.body.appendChild(ub);
    }

    setTimeout(win.__ragUpdateScrollButtons, 500);
    setTimeout(win.__ragUpdateScrollButtons, 1500);
})();
</script>
""", height=0)


if "history" not in st.session_state:
    st.session_state.history = []
if "active_source_label" not in st.session_state:
    st.session_state.active_source_label = "builtin"
if "in_flight" not in st.session_state:
    st.session_state.in_flight = None  # holds the background pipeline while a turn is generating


def _run_pipeline(question: str, active_collection, api_key: str, result: dict, stop_event: threading.Event) -> None:
    """Runs retrieve -> generate -> evaluate on a background thread so the
    main script can keep re-rendering (spinner + Stop button) while it
    works. `result` is a plain dict the background thread writes into and
    the main thread reads from on each poll -- checked between stages so a
    Stop click takes effect at the next stage boundary rather than being
    able to interrupt Groq's HTTP call itself mid-flight."""
    try:
        if stop_event.is_set():
            result["stopped"] = True
            return
        chunks = retrieve(question, collection=active_collection)
        result["chunks"] = chunks
        if stop_event.is_set():
            result["stopped"] = True
            return
        answer = generate_answer(question, chunks, api_key=api_key)
        result["answer"] = answer
        if stop_event.is_set():
            result["stopped"] = True
            return
        result["eval"] = evaluate_answer(answer, chunks, api_key=api_key)
    except Exception as e:
        result["error"] = str(e)
    finally:
        result["done"] = True


def score_color(score: float) -> str:
    if score >= FAITHFULNESS_THRESHOLD:
        return "green"
    if score >= 0.4:
        return "orange"
    return "red"


def render_assistant_turn(turn: dict) -> None:
    """Render one assistant answer: score badges, cleaned answer (optionally
    typed out line by line), and the supporting expanders."""
    result = turn["result"]
    score = result["score"]
    color = score_color(score)

    if result["should_abstain"]:
        reason = "the model declined to answer" if result["declined"] else "the faithfulness score is below threshold"
        icon_html = (
            f'<img src="{WARNING_ICON_URI}" width="28" height="28" '
            f'style="flex-shrink:0;">' if WARNING_ICON_URI else "⚠️"
        )
        st.markdown(
            f'<div style="display:flex;align-items:flex-start;gap:12px;'
            f'background-color:rgba(250,204,21,0.12);border:1px solid rgba(250,204,21,0.35);'
            f'border-radius:10px;padding:14px 16px;margin-bottom:12px;color:#fde68a;'
            f'line-height:1.5;">'
            f'{icon_html}'
            f'<div><strong>Low confidence</strong> ({reason}) — this answer contains claims '
            f'not well-supported by the retrieved documents, or the system was '
            f'unable to answer confidently. Treat it as unreliable.</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Faithfulness score**  \n:{color}[{score:.0%}]")
    with col2:
        st.markdown(f"**Citation accuracy**  \n{result['citation_accuracy']:.0%}")

    # Build the displayed answer from the already citation-checked claims,
    # not the raw LLM output -- claims with a fabricated citation have it
    # replaced with "[citation unverified]" by strip_invalid_citations().
    has_any_unverified = any(c.get("has_unverified_citation") for c in result["claims"])
    cleaned_answer = "  \n".join(c["claim"] for c in result["claims"])

    # Defensive fallback: if the claim splitter found nothing to break the
    # answer into (e.g. an off-corpus question produces a short reply that
    # doesn't parse into discrete claims), cleaned_answer ends up empty and
    # the turn would otherwise render a blank "Answer:" with nothing under
    # it. Show the raw model output instead of silently displaying nothing.
    if not cleaned_answer.strip():
        cleaned_answer = turn["answer"].strip() or "_The model returned an empty response._"

    answer_slot = st.empty()
    if turn.get("animate"):
        words = cleaned_answer.split(" ")
        shown = ""
        for idx, word in enumerate(words):
            shown += word if idx == 0 else " " + word
            answer_slot.markdown(f"**Answer:** {shown}▌")
            time.sleep(0.025)
        answer_slot.markdown(f"**Answer:** {cleaned_answer}")
        turn["animate"] = False  # only animate the turn once, on its first render
    else:
        answer_slot.markdown(f"**Answer:** {cleaned_answer}")

    if has_any_unverified:
        st.caption("🟠 One or more citations above could not be verified and were flagged "
                    "-- see the claim-level breakdown for details.")

    with st.expander("Show original model output (may contain unverified citations)"):
        st.text(turn["answer"])

    with st.expander(f"Claim-level breakdown ({len(result['claims'])} claims)"):
        for c in result["claims"]:
            mark = "✅" if c["supported"] else "❌"
            note = ""
            if c.get("has_unverified_citation"):
                invalid = [cc["cited_section"] for cc in c["citation_checks"] if not cc["citation_valid"]]
                note = f"  \n:orange[Citation unverified: Section {', '.join(invalid)} — content judged independently]"
            st.markdown(f"{mark} {c['claim']}{note}")

    with st.expander(f"Retrieved chunks ({len(turn['chunks'])})"):
        for i, c in enumerate(turn["chunks"], 1):
            status_badge = {
                "verified": "🟢 verified",
                "unverified": "🟡 unverified",
                "n/a": "⚪ n/a",
            }.get(c.get("section_status", "n/a"), "⚪ n/a")
            st.markdown(f"**Chunk {i}** — source: `{c['source']}` | {status_badge} "
                        f"(distance: {c['distance']:.3f})")
            st.text(c["text"][:400])


# ---------------------------------------------------------------------------
# Chat history (oldest first). Each new turn moves through states across
# separate reruns: "new" (question drawn live for the first time) ->
# "processing" (question redrawn -- byte-identical to what "new" already
# delivered, then live retrieve/generate/evaluate) -> "done".
#
# The scroll-to-top call fires right at the start of "processing", right
# after that redraw and right before the real retrieve/generate/evaluate
# calls. Two things line up there on purpose: the redraw is byte-identical
# to what "new" already painted one run ago, so there's nothing left for
# the browser to settle before scrolling toward it; and the real API calls
# that follow take genuine multi-second wall-clock time, which is what
# actually gives the injected scroll script time to load and run in the
# browser -- rather than an artificial sleep() racing against an immediate
# st.rerun() that could tear the script down before it ever got to run.
# ---------------------------------------------------------------------------
def _scroll_question_to_top(anchor_id: str) -> None:
    """Scroll so the given anchor element sits near the top of the
    viewport, instead of scrolling toward the bottom of the page. Used so
    a freshly-submitted question moves to the top of view, the same way
    the very first question naturally lands near the top -- rather than
    always chasing the bottom of a growing conversation.

    Uses the browser's native `Element.scrollIntoView()` instead of
    hand-rolled "find the real scroll container, then compute a pixel
    offset" logic. That custom math was the actual bug: on this app's
    layout it was consistently pushing the page further away rather than
    converging on the right spot, and every repeated correction pass just
    compounded the error further downward. scrollIntoView is a standard
    API built for exactly this: given an element, the browser itself
    walks up to whichever ancestor actually scrolls (window,
    documentElement, or some inner container -- it doesn't matter which,
    the browser figures it out correctly) and computes the right amount to
    scroll, with no custom container-detection or offset math left for us
    to get wrong. The small top gap is handled by a `scroll-margin-top`
    CSS rule on the anchor elements (see the CSS block above) rather than
    subtracting a magic number here.

    Fires by injecting a real <script> tag directly into the parent
    document, rather than calling into parent elements from inside this
    components.html iframe's own script -- everything running inside this
    throwaway iframe depends on it actually finishing loading before
    Streamlit tears it down on the next rerun, which isn't guaranteed. A
    <script> tag inserted straight into the parent document instead runs
    natively as the parent page's own code the instant it's inserted, the
    same way the manual scroll buttons' onclick handlers do.
    """
    inner_script = f"""
    (function() {{
        var doc = document;

        function tryScroll(attempts) {{
            var target = doc.getElementById({anchor_id!r});
            if (!target) {{
                if (attempts < 40) setTimeout(function() {{ tryScroll(attempts + 1); }}, 100);
                return;
            }}
            target.scrollIntoView({{ block: 'start', behavior: 'smooth' }});

            // Repeat a few more times over the next ~2s rather than
            // trusting the very first attempt. On a slow connection the
            // page can still be showing a fallback font when this first
            // runs (confirmed by Chrome's own "Slow network detected...
            // Fallback font will be used" console message) -- once the
            // real web font finishes loading and swaps in, text reflows
            // to different line heights and the page shifts vertically,
            // invalidating a one-shot scroll done before that swap
            // happened. Each repeat call re-evaluates the element's
            // current position fresh, so it self-corrects for that (or
            // any other late-arriving layout shift) automatically.
            [400, 900, 1600].forEach(function(delay) {{
                setTimeout(function() {{
                    var t = doc.getElementById({anchor_id!r});
                    if (t) t.scrollIntoView({{ block: 'start', behavior: 'smooth' }});
                }}, delay);
            }});
        }}
        tryScroll(0);
    }})();
    """
    components.html(f"""
    <script>
    (function() {{
        var s = window.parent.document.createElement('script');
        s.textContent = {json.dumps(inner_script)};
        window.parent.document.body.appendChild(s);
        // Executes synchronously the instant it's inserted, so it's
        // already done its job by here -- just tidying up the DOM node.
        setTimeout(function() {{
            try {{
                if (s && s.parentNode) s.parentNode.removeChild(s);
            }} catch (e) {{ /* ignore if already removed */ }}
        }}, 50);
    }})();
    </script>
    """, height=0)


for turn in st.session_state.history:
    status = turn.get("status", "done")
    if "turn_id" not in turn:  # backfill for turns created before this field existed
        turn["turn_id"] = uuid.uuid4().hex[:8]
    if status not in ("new", "processing", "error") and "result" not in turn:
        # Backfill for turns left in an old intermediate status (e.g. the
        # "landed" stage from a previous version of this state machine) --
        # anything incomplete just resumes processing instead of hitting
        # the "done" branch, which assumes an answer/result already exist.
        status = "processing"
    anchor_id = f"qa-anchor-{turn['turn_id']}"

    if status == "new":
        st.markdown(f'<div id="{anchor_id}"></div>', unsafe_allow_html=True)
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(turn["question"])
        with st.chat_message("assistant", avatar="📚"):
            st.markdown(status_html("Retrieving relevant passages..."), unsafe_allow_html=True)
        turn["status"] = "processing"
        st.rerun()

    elif status == "processing":
        st.markdown(f'<div id="{anchor_id}"></div>', unsafe_allow_html=True)
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(turn["question"])

        active_collection = st.session_state.get("uploaded_collection")

        # Move the freshly-submitted question up to the top of the
        # viewport -- the same way the very first question in a fresh
        # conversation naturally lands near the top, since there's
        # nothing above it yet -- rather than chasing the bottom of a
        # growing page. To see earlier turns, the person just scrolls
        # up manually.
        #
        # Fired right here rather than on a dedicated "landed" stage
        # that used to sleep briefly and immediately rerun: the repaint
        # of the question + status line just above is byte-identical
        # to what the "new" stage already rendered one run ago, so
        # there's nothing left to settle -- and retrieve()/
        # generate_answer()/evaluate_answer() below take real,
        # multi-second wall-clock time on their own, which is what
        # actually gives the injected scroll script time to load and
        # run in the browser, rather than racing an arbitrary sleep()
        # against an immediate rerun that could tear it down first.
        #
        # Called at the top level (not nested inside the st.chat_message
        # block below) so the injected component isn't placed inside the
        # assistant bubble's own container.
        if not turn.get("scrolled"):
            _scroll_question_to_top(anchor_id)
            turn["scrolled"] = True

        with st.chat_message("assistant", avatar="📚"):
            status_slot = st.empty()
            status_slot.markdown(status_html("Retrieving relevant passages..."), unsafe_allow_html=True)

            turn_error = None
            try:
                chunks = retrieve(turn["question"], collection=active_collection)
            except Exception as e:
                chunks = None
                turn_error = f"Retrieval failed: {e}"

            answer = None
            if turn_error is None:
                status_slot.markdown(status_html("Generating answer..."), unsafe_allow_html=True)
                try:
                    answer = generate_answer(turn["question"], chunks, api_key=active_api_key)
                except Exception as e:
                    turn_error = f"Generation failed: {e}"

            result = None
            if turn_error is None and answer is not None:
                status_slot.markdown(status_html("Checking faithfulness..."), unsafe_allow_html=True)
                try:
                    result = evaluate_answer(answer, chunks, api_key=active_api_key)
                except Exception as e:
                    turn_error = f"Faithfulness check failed: {e}"

            status_slot.empty()

            if result is not None:
                turn["answer"] = answer
                turn["chunks"] = chunks
                turn["result"] = result
                turn["status"] = "done"
                turn["animate"] = True
                render_assistant_turn(turn)
                # No further auto-scroll needed here: the question was
                # already pinned to the top of the viewport above, and the
                # answer just streams in below that fixed point.
                st.rerun()
            else:
                # Failed -- keep the turn (question included) visible with
                # the error attached, instead of deleting it and rerunning,
                # which used to erase the question and the error message
                # before either was ever seen on screen.
                turn["status"] = "error"
                turn["error"] = turn_error or "Unknown error."
                st.rerun()

    elif status == "error":
        st.markdown(f'<div id="{anchor_id}"></div>', unsafe_allow_html=True)
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(turn["question"])
        with st.chat_message("assistant", avatar="📚"):
            st.error(turn.get("error", "Something went wrong."))

    else:  # "done"
        st.markdown(f'<div id="{anchor_id}"></div>', unsafe_allow_html=True)
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(turn["question"])
        with st.chat_message("assistant", avatar="📚"):
            render_assistant_turn(turn)

# (Bottom spacing is reserved once, at the very end of the page --
# see the spacer right before #chat-bottom-anchor below. It used to sit
# here instead, right after the history loop, which reserved room below
# *old* turns but not below a freshly live-rendered new question/answer
# (rendered further down, in the submission-handling block) -- leaving
# that newest content flush against the fixed input bar with no gap.

# ---------------------------------------------------------------------------
# Chat input: the "+" icon (accept_file) doubles as the document uploader,
# right next to the text box, the same way Claude/ChatGPT/Gemini attach
# files inline instead of via a separate sidebar form.
# ---------------------------------------------------------------------------
try:
    submission = st.chat_input(
        placeholder=PLACEHOLDER_TEXT,
        accept_file=True,
        file_type=["pdf", "txt"],
        disabled=not active_api_key,
    )
except TypeError:
    # Older Streamlit versions (<1.40) don't support accept_file yet.
    st.info("Upgrade Streamlit (`pip install -U streamlit`) to enable the "
             "'+' file-attach button in the chat box. Falling back to text-only input.")
    submission = st.chat_input(placeholder=PLACEHOLDER_TEXT, disabled=not active_api_key)

if not active_api_key:
    st.warning("⚠️ Enter a Groq API key in the sidebar to get started.")

if submission:
    # `submission` is a plain string in the fallback path, or a
    # ChatInputValue (`.text` / `.files`) when accept_file worked.
    question = getattr(submission, "text", submission) or ""
    question = question.strip()
    uploaded_file = getattr(submission, "files", None)
    uploaded_file = uploaded_file[0] if uploaded_file else None

    # If the chat input didn't provide a file (browser/Streamlit quirks),
    # fall back to the sidebar uploader above (stored in session state).
    if uploaded_file is None:
        uploaded_file = st.session_state.get("_fb_fallback")

    active_collection = st.session_state.get("uploaded_collection")

    if uploaded_file is not None:
        overlay = st.empty()
        overlay.markdown("""
        <div id="ingest-overlay">
          <div class="ingest-loader-wrap">
            <svg class="ingest-loader" viewBox="0 0 400 400" width="72" height="72">
              <circle class="dot-1" cx="81.69" cy="200" r="27.903" fill="none" stroke="#4f46e5" stroke-linecap="round" stroke-linejoin="round" stroke-width="12"/>
              <circle class="dot-2" cx="200" cy="200" r="27.903" fill="none" stroke="#4f46e5" stroke-linecap="round" stroke-linejoin="round" stroke-width="12"/>
              <circle class="dot-3" cx="318.31" cy="200" r="27.903" fill="none" stroke="#4f46e5" stroke-linecap="round" stroke-linejoin="round" stroke-width="12"/>
            </svg>
            <div class="ingest-loader-text">Ingesting...</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        try:
            try:
                file_bytes = uploaded_file.read()
                text = extract_text_from_upload(file_bytes, uploaded_file.name)
                if not text.strip():
                    raise ValueError(
                        "No extractable text found -- this may be a scanned/image-based PDF."
                    )
                embedder = _get_embedder()
                # Ephemeral (in-memory) client: this document's index lives only
                # for this browser session and is never written to disk, so
                # uploads never persist or leak between users.
                client = chromadb.EphemeralClient()
                collection = client.create_collection(f"upload_{uuid.uuid4().hex[:8]}")
                n_chunks = index_documents(
                    [{"source": uploaded_file.name, "text": text}], collection, embedder
                )
                st.session_state.uploaded_collection = collection
                st.session_state.uploaded_chunk_count = n_chunks
                st.session_state.uploaded_file_name = uploaded_file.name
                st.session_state.active_source_label = uploaded_file.name
                st.session_state.history = []  # new document -> clear old Q&A
                active_collection = collection
            except Exception as e:
                st.error(f"Failed to ingest file: {e}")
                st.session_state.uploaded_collection = None
                active_collection = None
        finally:
            overlay.empty()  # always remove the dimmed overlay, even on error

    if question:
        if not active_api_key:
            st.error("No API key available.")
        else:
            # Just record the question and hand off to the history loop
            # above (on the rerun this triggers) -- see the big comment
            # there for why processing was moved out of this live,
            # single-run path entirely.
            st.session_state.history.append({
                "question": question,
                "status": "new",
                "turn_id": uuid.uuid4().hex[:8],
            })
            st.rerun()

# Invisible spacer that reserves extra room at the very bottom so the
# last piece of content on the page -- whether that's an old turn's
# "Retrieved chunks" expander, or a brand new question + live status line
# still mid-processing -- always sits above the fixed input bar instead
# of running flush against it.
st.markdown('<div style="height: 120px;"></div>', unsafe_allow_html=True)

# Invisible anchor the floating "scroll to bottom" button targets.
st.markdown('<div id="chat-bottom-anchor"></div>', unsafe_allow_html=True)
