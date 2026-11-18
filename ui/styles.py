"""Visual identity of the IDSS UI: the palette, CSS injection, the sidebar logo, and the hero banner and footer
that every page shows. Purely presentational: nothing here talks to the backend."""

import functools
import html
from pathlib import Path

import streamlit as st

# Palette shared with the report figures (scripts/evaluate.py), in that order.
PURPLE, GREEN, PINK, YELLOW, BLUE, ORANGE = "#6C5CE7", "#00B894", "#FD79A8", "#FDCB6E", "#0984E3", "#E17055"
PALETTE = [PURPLE, GREEN, PINK, YELLOW, BLUE, ORANGE]
TEXT, MUTED = "#2D3436", "#5F6B73"

APP_NAME = "Intelligent Decision Support System"
PROJECT_TITLE = "Intelligent Decision Support System Using Generative AI and Machine Learning"
COLLEGE = "G H Raisoni College of Engineering and Management, Jalgaon"
FOOTER_TEXT = f"{PROJECT_TITLE} · {COLLEGE}"

_UI_DIR = Path(__file__).resolve().parent
STYLE_PATH = _UI_DIR / "style.css"
LOGO_PATH = _UI_DIR / "assets" / "idss_logo.svg"


@functools.cache
def _css() -> str:
    names = ("purple", "green", "pink", "yellow", "blue", "orange")
    variables = "".join(f"--idss-{name}: {color};" for name, color in zip(names, PALETTE, strict=True))
    return f"<style>:root {{{variables}}}\n{STYLE_PATH.read_text(encoding='utf-8')}</style>"


def inject_css() -> None:
    """The app-wide stylesheet (style-only HTML takes no space on the page) and the sidebar logo."""
    st.html(_css())
    st.logo(str(LOGO_PATH), size="large")


def hero(title: str, subtitle: str = "", icon: str = "✨") -> None:
    """Gradient banner (purple → blue) with an emoji icon, the page title and a subtitle."""
    subtitle_html = f'<div class="idss-hero-subtitle">{html.escape(subtitle)}</div>' if subtitle else ""
    st.markdown(
        f'<div class="idss-hero"><div class="idss-hero-icon">{html.escape(icon)}</div>'
        f'<div class="idss-hero-body"><div class="idss-hero-eyebrow">IDSS · {APP_NAME}</div>'
        f'<div class="idss-hero-title">{html.escape(title)}</div>{subtitle_html}</div></div>',
        unsafe_allow_html=True)


def section(title: str) -> None:
    """A section heading with a coloured accent bar."""
    st.markdown(f'<div class="idss-section">{html.escape(title)}</div>', unsafe_allow_html=True)


def footer() -> None:
    """Project footer. Rendered first by `shell()` (so `st.stop()` cannot skip it); CSS moves it to the page end."""
    st.markdown(f'<div class="idss-footer"><div class="idss-footer-bar"></div><b>{html.escape(PROJECT_TITLE)}</b> · '
                f'{html.escape(COLLEGE)}</div>', unsafe_allow_html=True)
