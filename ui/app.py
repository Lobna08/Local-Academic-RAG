"""Streamlit UI: pure HTTP client of the RAG API (never imports rag.py).

Run (with the API already started):
    streamlit run ui/app.py
"""

from __future__ import annotations

import base64
import html
import os
import re
from pathlib import Path

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT_S = 180  # the 8B model on CPU takes ~1 min per answer

LOGO_PATH = Path(__file__).resolve().parent / "assets" / "tekup-logo.png"

# Keys must match the API's citation["source"] / chunk metadata "source_file" exactly.
SOURCE_LINKS = {
    "Students_Guide.pdf": "https://drive.google.com/file/d/18QoYG3Fp2kuK-qB-NMhRGNwMsQ0-sMbz/view?usp=drive_link",
    "Plan_d_études_SDIA-2024-2025.pdf": "https://drive.google.com/file/d/1YDsuuF5HSrM-3zU-weKL-PM3HKOh_q4Z/view?usp=drive_link",
}

SUGGESTIONS = [
    "Comment valider un panier ?",
    "Date limite de paiement ?",
    "Durée minimale du stage PFE ?",
]

# Design tokens per theme; the :root block is rebuilt from these on every run.
THEMES = {
    "light": {
        "bg": "#E4E5ED", "surface": "#EFEFF5", "surface-2": "#E7E8F0", "plate": "#FFFFFF",
        "ink": "#1A1735", "muted": "#6B688B", "line": "#D7D8E3",
        "brand": "#3A0CA3", "brand-hover": "#2E0A82", "brand-2": "#3B5BDB",
        "chip-bg": "#EEE9FB", "chip-ink": "#3A0CA3", "on-brand": "#FFFFFF",
        "shadow": "rgba(26,23,53,.07)",
    },
    "dark": {
        "bg": "#131026", "surface": "#1C1933", "surface-2": "#23203C", "plate": "#F3F2FA",
        "ink": "#ECEAF7", "muted": "#A8A4C6", "line": "#2F2B49",
        "brand": "#9B8CF5", "brand-hover": "#B1A5F8", "brand-2": "#86A2F6",
        "chip-bg": "#272150", "chip-ink": "#CDC3FA", "on-brand": "#161232",
        "shadow": "rgba(0,0,0,.35)",
    },
}

CSS = """
/* --- Streamlit chrome --- */
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stAppDeployButton"],
[data-testid="stDecoration"], .stDeployButton { display: none !important; }
[data-testid="stHeader"] { background: transparent; height: 0; }

/* --- Layout & typography --- */
.stApp { background: var(--bg); color: var(--ink); }
/* Every text element in Hanken Grotesk (Streamlit forces "Source Sans" on its
   own containers); icons and code keep their fonts; titles in Fraunces. */
.stApp, .stApp :where(*:not([data-testid="stIconMaterial"]):not(code):not(pre)) {
  font-family: var(--font-ui);
}
.stApp .tk-title, .stApp .tk-hero-title { font-family: var(--font-display) !important; }
[data-testid="stMainBlockContainer"], .block-container {
  max-width: 820px; padding-top: 3rem; padding-bottom: 4rem;
}
.stApp [data-testid="stMarkdownContainer"], .stApp [data-testid="stSpinner"] { color: var(--ink); }
.stApp button [data-testid="stMarkdownContainer"] { color: inherit; }

/* --- Header + theme toggle --- */
.tk-header { display: flex; align-items: center; gap: 16px; }
.tk-plate {
  background: var(--plate); border-radius: 12px; padding: 6px; display: flex;
  box-shadow: 0 1px 2px rgba(26,23,53,.06), 0 4px 14px var(--shadow);
}
.tk-plate img { width: 44px; height: 44px; object-fit: contain; display: block; }
.tk-title {
  font-weight: 600; font-size: 1.6rem; line-height: 1.1;
  color: var(--ink); letter-spacing: -0.01em;
}
.tk-sub {
  font-size: .72rem; font-weight: 600; letter-spacing: .14em;
  text-transform: uppercase; color: var(--muted); margin-top: 4px;
}
.st-key-theme_toggle button {
  background: var(--surface); border: 1px solid var(--line); color: var(--ink);
  border-radius: 999px; width: 42px; height: 42px; min-height: 42px; padding: 0;
}
.st-key-theme_toggle button:hover { border-color: var(--brand); color: var(--brand); }

/* --- Hero --- */
.tk-hero { margin: 36px 0 22px; }
.tk-hero-title {
  font-weight: 500; font-size: 2.1rem; line-height: 1.15;
  letter-spacing: -0.015em; color: var(--ink); margin: 0 0 10px;
}
.tk-hero-sub { color: var(--muted); font-size: 1rem; line-height: 1.6; margin: 0; max-width: 680px; }

/* --- Question bar (form) --- */
[data-testid="stForm"] {
  background: var(--surface); border: 1px solid var(--line); border-radius: 18px;
  padding: 10px 12px; box-shadow: 0 2px 10px var(--shadow);
}
[data-testid="stTextInputRootElement"] {
  background: var(--surface-2); border: 1px solid transparent; border-radius: 12px;
}
[data-testid="stTextInputRootElement"]:focus-within { border-color: var(--brand); }
[data-testid="stTextInput"] input {
  background: transparent; color: var(--ink); caret-color: var(--brand);
  font-size: 1rem; padding: 12px 14px;
}
[data-testid="stTextInput"] input::placeholder { color: var(--muted); opacity: 1; }
button[kind="primaryFormSubmit"], [data-testid="stBaseButton-primaryFormSubmit"] {
  background: var(--brand); border: 1px solid var(--brand); color: var(--on-brand);
  border-radius: 12px; min-height: 46px; font-weight: 600; width: 100%;
}
button[kind="primaryFormSubmit"]:hover, [data-testid="stBaseButton-primaryFormSubmit"]:hover,
button[kind="primaryFormSubmit"]:focus, button[kind="primaryFormSubmit"]:active {
  background: var(--brand-hover); border-color: var(--brand-hover); color: var(--on-brand);
}

/* --- Suggestion chips --- */
.st-key-tk_chips { margin-top: 2px; }
[class*="st-key-chip_"] button {
  background: var(--surface); border: 1px solid var(--line); color: var(--ink);
  border-radius: 999px; min-height: 34px; padding: 4px 14px;
}
[class*="st-key-chip_"] button p { font-size: .88rem; }
[class*="st-key-chip_"] button:hover {
  background: var(--chip-bg); border-color: var(--chip-bg); color: var(--chip-ink);
}

/* --- Answer card --- */
.tk-card {
  background: var(--surface); border: 1px solid var(--line); border-radius: 18px;
  padding: 24px 26px 20px; margin: 22px 0 14px;
  box-shadow: 0 1px 2px rgba(26,23,53,.04), 0 8px 28px var(--shadow);
}
.tk-eyebrow {
  font-size: .72rem; font-weight: 600; letter-spacing: .14em; text-transform: uppercase;
  color: var(--muted); margin-bottom: 10px;
}
.tk-answer { color: var(--ink); font-size: 1rem; line-height: 1.65; }
.tk-answer p { margin: 0 0 .7em; }
.tk-answer ol, .tk-answer ul { margin: 0 0 .7em; padding-left: 1.3em; }
.tk-answer li { margin-bottom: .3em; }
.tk-footer {
  display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between;
  gap: 10px; border-top: 1px solid var(--line); margin-top: 14px; padding-top: 14px;
}
.tk-chip {
  display: inline-flex; align-items: center; gap: 6px;
  background: var(--chip-bg); color: var(--chip-ink); border-radius: 999px;
  padding: 6px 12px; font-size: .85rem; font-weight: 500;
}
.tk-nocite { color: var(--muted); font-size: .85rem; }
a.tk-open, a.tk-open:visited {
  display: inline-flex; align-items: center; gap: 6px;
  border: 1px solid var(--brand-2); color: var(--brand-2); border-radius: 999px;
  padding: 6px 14px; font-size: .85rem; font-weight: 600; text-decoration: none;
  transition: background .15s ease, color .15s ease;
}
a.tk-open:hover { background: var(--brand-2); color: var(--on-brand); }

/* --- Source passages (expander) --- */
[data-testid="stExpander"] details {
  background: var(--surface); border: 1px solid var(--line); border-radius: 14px;
}
[data-testid="stExpander"] summary { color: var(--muted); font-weight: 500; }
[data-testid="stExpander"] summary:hover { color: var(--brand); }
[data-testid="stExpander"] summary:hover svg, [data-testid="stExpander"] summary:hover span { color: var(--brand); }
.tk-passage { padding: 10px 0; border-bottom: 1px solid var(--line); }
.tk-passage:last-child { border-bottom: none; }
.tk-passage-meta { font-size: .8rem; font-weight: 600; color: var(--brand); margin-bottom: 4px; }
.tk-passage-text { font-size: .88rem; color: var(--muted); line-height: 1.55; }
"""

FONT_TOKENS = (
    "--font-ui: 'Hanken Grotesk', -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;"
    "--font-display: 'Fraunces', Georgia, 'Times New Roman', serif;"
)


def build_css(theme: str) -> str:
    tokens = "".join(f"--{name}: {value};" for name, value in THEMES[theme].items())
    return f"<style>:root {{ {tokens} {FONT_TOKENS} color-scheme: {theme}; }}{CSS}</style>"


_LIST_ITEM_RE = re.compile(r"^\s*(?:(\d+)[.)]|[-*•])\s+(.*)$")


def _inline(text: str) -> str:
    """Escape, then re-apply **bold** (the only inline markdown the LLM uses)."""
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html.escape(text))


def answer_to_html(text: str) -> str:
    """Minimal markdown -> HTML (paragraphs + ordered/unordered lists), escaped.

    The answer sits inside a raw HTML card, where Streamlit does not run its
    markdown renderer, so the LLM's lists are converted here.
    """
    blocks = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = [ln for ln in block.splitlines() if ln.strip()]
        matches = [_LIST_ITEM_RE.match(ln) for ln in lines]
        if lines and all(matches):
            tag = "ol" if matches[0].group(1) else "ul"
            items = "".join(f"<li>{_inline(m.group(2))}</li>" for m in matches)
            blocks.append(f"<{tag}>{items}</{tag}>")
        else:
            blocks.append("<p>" + "<br>".join(_inline(ln) for ln in lines) + "</p>")
    return "".join(blocks)


def _snippet_html(snippet: str) -> str:
    """Escaped, with line breaks as <br>: a blank line would otherwise close
    the raw HTML block and let Streamlit's markdown parser take over."""
    lines = [ln.strip() for ln in snippet.strip().splitlines() if ln.strip()]
    return "<br>".join(html.escape(ln) for ln in lines)


@st.cache_data
def _logo_b64() -> str:
    return base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")


def _toggle_theme() -> None:
    st.session_state["theme"] = "dark" if st.session_state["theme"] == "light" else "light"


def _ask_suggestion(text: str) -> None:
    # Runs before the widgets are rebuilt, so the input's state can be set here.
    st.session_state["question"] = text
    st.session_state["pending_question"] = text


def render_header() -> None:
    dark = st.session_state["theme"] == "dark"
    with st.container(
        horizontal=True, vertical_alignment="center", horizontal_alignment="distribute"
    ):
        st.markdown(
            f"""<div class="tk-header">
  <div class="tk-plate"><img src="data:image/png;base64,{_logo_b64()}" alt="TEK-UP"></div>
  <div><div class="tk-title">Assistant Académique</div><div class="tk-sub">TEK-UP · SDIA</div></div>
</div>""",
            unsafe_allow_html=True,
            width="content",
        )
        st.button(
            "",
            key="theme_toggle",
            icon=":material/light_mode:" if dark else ":material/dark_mode:",
            help="Mode jour" if dark else "Mode nuit",
            on_click=_toggle_theme,
        )


def render_hero() -> None:
    st.markdown(
        """<div class="tk-hero">
  <div class="tk-hero-title">Posez une question, obtenez une réponse sourcée.</div>
  <p class="tk-hero-sub">Règlement, notation, stages, bourses, plans d’études — chaque réponse
  indique le document et la page exacts, et un bouton pour ouvrir la source.</p>
</div>""",
        unsafe_allow_html=True,
    )


def render_answer(data: dict) -> None:
    citation = data["citation"]
    if citation:
        source = html.escape(citation["source"])
        footer = f'<span class="tk-chip">📄 {source} · page {citation["page"]}</span>'
        url = SOURCE_LINKS.get(citation["source"])
        if url:
            footer += (
                f'<a class="tk-open" href="{html.escape(url)}" target="_blank" '
                f'rel="noopener">Ouvrir le document ↗</a>'
            )
    else:
        footer = '<span class="tk-nocite">Aucune source citée (hors corpus ou réponse non ancrée)</span>'

    st.markdown(
        f"""<div class="tk-card">
  <div class="tk-eyebrow">Réponse</div>
  <div class="tk-answer">{answer_to_html(data["answer"])}</div>
  <div class="tk-footer">{footer}</div>
</div>""",
        unsafe_allow_html=True,
    )

    with st.expander(f"Passages sources ({len(data['retrieved'])})"):
        passages = "".join(
            f"""<div class="tk-passage">
  <div class="tk-passage-meta">{html.escape(chunk["source"])} · page {chunk["page"]}</div>
  <div class="tk-passage-text">{_snippet_html(chunk["snippet"])}</div>
</div>"""
            for chunk in data["retrieved"]
        )
        st.markdown(passages, unsafe_allow_html=True)


st.set_page_config(page_title="Assistant Académique · TEK-UP", page_icon="🎓")
st.session_state.setdefault("theme", "light")
st.markdown(build_css(st.session_state["theme"]), unsafe_allow_html=True)
render_header()
render_hero()

with st.form("ask", border=False):
    col_input, col_button = st.columns([5, 1], vertical_alignment="bottom")
    with col_input:
        question = st.text_input(
            "Votre question",
            key="question",
            placeholder="Posez votre question sur le guide ou le plan d’études…",
            label_visibility="collapsed",
        )
    with col_button:
        submitted = st.form_submit_button("Demander", type="primary")

with st.container(horizontal=True, key="tk_chips"):
    for i, suggestion in enumerate(SUGGESTIONS):
        st.button(suggestion, key=f"chip_{i}", on_click=_ask_suggestion, args=(suggestion,))

# A suggestion chip both fills the input and launches the question.
pending = st.session_state.pop("pending_question", None)
query = pending if pending else (question if submitted else "")

if query.strip():
    st.session_state.pop("last_result", None)
    try:
        with st.spinner("Recherche en cours…"):
            resp = requests.post(
                f"{API_URL}/ask", json={"question": query}, timeout=TIMEOUT_S
            )
    except requests.ConnectionError:
        st.error("L’API ne répond pas — lance d’abord `uvicorn src.api:app --port 8000`.")
        st.stop()
    except requests.Timeout:
        st.error("Le modèle met trop de temps à répondre.")
        st.stop()

    if resp.status_code != 200:
        try:
            detail = resp.json().get("detail", resp.text)
        except ValueError:
            detail = resp.text
        st.error(f"Erreur API ({resp.status_code}) : {detail}")
        st.stop()

    st.session_state["last_result"] = resp.json()

# Kept across reruns so toggling the theme does not erase the answer.
if "last_result" in st.session_state:
    render_answer(st.session_state["last_result"])
