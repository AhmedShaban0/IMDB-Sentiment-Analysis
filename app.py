"""
IMDB Movie Review Sentiment Analysis - Modern AI SaaS Dashboard
================================================================
A production-grade, cinematic AI/NLP interface for real-time binary
sentiment classification of movie reviews. Powered by Logistic Regression
and TF-IDF, pre-trained and serialized via joblib with zero runtime retraining.
"""

import os
import re
import html
import joblib
import streamlit as st
from bs4 import BeautifulSoup
import contractions

# ==============================================================================
# Page Configuration & Metadata
# ==============================================================================
st.set_page_config(
    page_title="IMDB Sentiment AI • Movie Review Classifier",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==============================================================================
# Custom CSS Design System: Cinematic Dark Glassmorphism
# ==============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Global Root Variables */
    :root {
        --bg-main: #0B0F19;
        --bg-surface: #111827;
        --accent-imdb: #F5C518;
        --accent-imdb-glow: rgba(245, 197, 24, 0.35);
        --accent-red: #E50914;
        --accent-positive: #22C55E;
        --accent-positive-bg: rgba(34, 197, 94, 0.12);
        --accent-negative: #EF4444;
        --accent-negative-bg: rgba(239, 68, 68, 0.12);
        --glass-card: rgba(255, 255, 255, 0.035);
        --glass-border: rgba(255, 255, 255, 0.09);
        --glass-hover: rgba(255, 255, 255, 0.06);
        --text-primary: #F9FAFB;
        --text-secondary: #9CA3AF;
        --text-muted: #6B7280;
    }

    /* Base Streamlit App Overrides */
    .stApp {
        background-color: var(--bg-main) !important;
        background-image: 
            radial-gradient(at 0% 0%, rgba(245, 197, 24, 0.05) 0px, transparent 50%),
            radial-gradient(at 100% 0%, rgba(229, 9, 20, 0.05) 0px, transparent 50%),
            radial-gradient(at 50% 100%, rgba(17, 24, 39, 0.8) 0px, transparent 100%) !important;
        background-attachment: fixed !important;
        color: var(--text-primary) !important;
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif !important;
    }

    /* Streamlit Main Container Spacing */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 3.5rem !important;
        max-width: 1200px !important;
    }

    /* Hide Default Header & Footer Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header[data-testid="stHeader"] {background: transparent !important;}

    /* Top Navigation Bar */
    .top-nav {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.85rem 1.4rem;
        background: rgba(17, 24, 39, 0.6);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid var(--glass-border);
        border-radius: 14px;
        margin-bottom: 2rem;
    }
    .brand-group {
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    .brand-icon {
        font-size: 1.6rem;
        filter: drop-shadow(0 2px 8px rgba(245, 197, 24, 0.3));
    }
    .brand-title {
        font-weight: 800;
        font-size: 1.2rem;
        letter-spacing: -0.02em;
        color: #FFFFFF;
        line-height: 1.2;
    }
    .brand-badge {
        font-size: 0.72rem;
        color: var(--accent-imdb);
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.35rem 0.85rem;
        background: rgba(34, 197, 94, 0.1);
        border: 1px solid rgba(34, 197, 94, 0.25);
        border-radius: 9999px;
        color: #4ADE80;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .status-dot {
        width: 7px;
        height: 7px;
        background-color: #22C55E;
        border-radius: 50%;
        box-shadow: 0 0 8px #22C55E;
    }
    .model-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.35rem 0.85rem;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid var(--glass-border);
        border-radius: 9999px;
        color: var(--text-secondary);
        font-size: 0.78rem;
        font-weight: 500;
        margin-left: 0.5rem;
    }

    /* Hero Section */
    .hero-container {
        text-align: center;
        padding: 1.2rem 1rem 2.2rem 1rem;
        position: relative;
    }
    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        line-height: 1.2;
        margin-bottom: 0.75rem;
        background: linear-gradient(180deg, #FFFFFF 30%, #D1D5DB 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: var(--text-secondary);
        max-width: 680px;
        margin: 0 auto 1.4rem auto;
        line-height: 1.55;
    }
    .tech-pill-row {
        display: flex;
        justify-content: center;
        flex-wrap: wrap;
        gap: 0.6rem;
    }
    .tech-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.3rem 0.85rem;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 500;
        color: #E5E7EB;
        backdrop-filter: blur(8px);
    }
    .tech-pill strong {
        color: var(--accent-imdb);
    }

    /* Horizontal Metric Cards */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin-bottom: 2.2rem;
    }
    .metric-card {
        background: var(--glass-card);
        border: 1px solid var(--glass-border);
        border-radius: 14px;
        padding: 1.15rem 1rem;
        text-align: center;
        backdrop-filter: blur(12px);
        transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
        position: relative;
        overflow: hidden;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        border-color: rgba(245, 197, 24, 0.3);
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4), 0 0 15px -2px rgba(245, 197, 24, 0.1);
    }
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(245, 197, 24, 0.4), transparent);
    }
    .metric-label {
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 0.35rem;
    }
    .metric-value {
        font-size: 1.75rem;
        font-weight: 800;
        color: #FFFFFF;
        letter-spacing: -0.02em;
        line-height: 1.15;
    }
    .metric-caption {
        font-size: 0.74rem;
        color: var(--text-secondary);
        margin-top: 0.3rem;
    }

    /* Main Prediction Card */
    .glass-card {
        background: rgba(17, 24, 39, 0.55);
        border: 1px solid var(--glass-border);
        border-radius: 20px;
        padding: 2rem 2.2rem;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5);
        margin-bottom: 2rem;
        position: relative;
    }
    .card-header-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #FFFFFF;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.3rem;
    }
    .card-header-subtitle {
        font-size: 0.92rem;
        color: var(--text-secondary);
        margin-bottom: 1.2rem;
    }

    /* Streamlit TextArea Restyling */
    div[data-testid="stTextArea"] {
        margin-bottom: 0.75rem;
    }
    div[data-testid="stTextArea"] textarea {
        background-color: rgba(11, 15, 25, 0.65) !important;
        color: #F9FAFB !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 14px !important;
        font-size: 1rem !important;
        line-height: 1.6 !important;
        padding: 1rem 1.15rem !important;
        transition: border-color 0.25s ease, box-shadow 0.25s ease, background-color 0.25s ease !important;
        font-family: inherit !important;
    }
    div[data-testid="stTextArea"] textarea:focus {
        border-color: var(--accent-imdb) !important;
        box-shadow: 0 0 0 3px rgba(245, 197, 24, 0.18) !important;
        background-color: rgba(11, 15, 25, 0.9) !important;
    }
    div[data-testid="stTextArea"] textarea::placeholder {
        color: #4B5563 !important;
    }

    /* Streamlit Button Restyling */
    div[data-testid="stButton"] > button[kind="primary"] {
        background: linear-gradient(135deg, #F5C518 0%, #D4A010 100%) !important;
        color: #000000 !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        letter-spacing: -0.01em !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 18px var(--accent-imdb-glow) !important;
        width: 100% !important;
    }
    div[data-testid="stButton"] > button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 24px rgba(245, 197, 24, 0.5) !important;
        background: linear-gradient(135deg, #FFD034 0%, #F5C518 100%) !important;
    }
    div[data-testid="stButton"] > button[kind="primary"]:active {
        transform: translateY(0) !important;
    }

    /* Example Review Pill Buttons */
    div[data-testid="stButton"] > button[kind="secondary"] {
        background: rgba(255, 255, 255, 0.04) !important;
        color: #D1D5DB !important;
        border: 1px solid rgba(255, 255, 255, 0.09) !important;
        border-radius: 10px !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        padding: 0.45rem 0.9rem !important;
        transition: all 0.2s ease !important;
        width: 100% !important;
        text-align: left !important;
    }
    div[data-testid="stButton"] > button[kind="secondary"]:hover {
        background: rgba(255, 255, 255, 0.08) !important;
        border-color: rgba(245, 197, 24, 0.4) !important;
        color: #FFFFFF !important;
        transform: translateY(-1px) !important;
    }

    /* Prediction Result Presentation */
    .result-container {
        border-radius: 16px;
        padding: 1.75rem 2rem;
        margin-top: 1.5rem;
        position: relative;
        overflow: hidden;
        animation: fadeIn 0.4s ease-out;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .result-positive-box {
        background: radial-gradient(circle at 10% 20%, rgba(34, 197, 94, 0.15) 0%, transparent 60%), rgba(17, 24, 39, 0.7);
        border: 1px solid rgba(34, 197, 94, 0.35);
        box-shadow: 0 10px 30px -10px rgba(34, 197, 94, 0.25);
    }
    .result-negative-box {
        background: radial-gradient(circle at 10% 20%, rgba(239, 68, 68, 0.15) 0%, transparent 60%), rgba(17, 24, 39, 0.7);
        border: 1px solid rgba(239, 68, 68, 0.35);
        box-shadow: 0 10px 30px -10px rgba(239, 68, 68, 0.25);
    }
    .result-header-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 1rem;
        margin-bottom: 1.2rem;
    }
    .result-badge {
        font-size: 1.25rem;
        font-weight: 800;
        letter-spacing: -0.01em;
        display: inline-flex;
        align-items: center;
        gap: 0.6rem;
    }
    .result-badge-pos { color: #4ADE80; }
    .result-badge-neg { color: #F87171; }
    .confidence-score-badge {
        font-size: 1.5rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: -0.03em;
        color: #FFFFFF;
    }
    .confidence-caption {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--text-muted);
        text-align: right;
    }

    /* Dual Probability Progress Bar */
    .bar-row {
        margin-bottom: 0.85rem;
    }
    .bar-header {
        display: flex;
        justify-content: space-between;
        font-size: 0.82rem;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }
    .bar-track {
        height: 10px;
        background: rgba(255, 255, 255, 0.06);
        border-radius: 9999px;
        overflow: hidden;
        position: relative;
    }
    .bar-fill-pos {
        height: 100%;
        background: linear-gradient(90deg, #10B981 0%, #34D399 100%);
        border-radius: 9999px;
        transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .bar-fill-neg {
        height: 100%;
        background: linear-gradient(90deg, #EF4444 0%, #F87171 100%);
        border-radius: 9999px;
        transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* Pipeline Architecture Card */
    .pipeline-grid {
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 0.5rem;
        padding: 1.25rem 0.5rem;
    }
    .pipeline-step {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid var(--glass-border);
        border-radius: 12px;
        padding: 0.85rem 1rem;
        text-align: center;
        flex: 1;
        min-width: 140px;
    }
    .pipeline-step-title {
        font-size: 0.82rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 0.2rem;
    }
    .pipeline-step-desc {
        font-size: 0.72rem;
        color: var(--text-muted);
    }
    .pipeline-arrow {
        color: var(--accent-imdb);
        font-size: 1.1rem;
        font-weight: 700;
    }

    /* Dark Benchmark Table */
    .benchmark-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        margin-top: 1rem;
        border: 1px solid var(--glass-border);
        border-radius: 12px;
        overflow: hidden;
    }
    .benchmark-table th {
        background: rgba(255, 255, 255, 0.03);
        color: var(--text-muted);
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        padding: 0.85rem 1rem;
        text-align: left;
        border-bottom: 1px solid var(--glass-border);
    }
    .benchmark-table td {
        padding: 0.85rem 1rem;
        font-size: 0.85rem;
        color: var(--text-secondary);
        border-bottom: 1px solid rgba(255, 255, 255, 0.04);
    }
    .benchmark-table tr:last-child td {
        border-bottom: none;
    }
    .benchmark-table tr.highlight-row {
        background: rgba(245, 197, 24, 0.06);
    }
    .benchmark-table tr.highlight-row td {
        color: #FFFFFF;
        font-weight: 600;
    }
    .benchmark-tag {
        background: rgba(245, 197, 24, 0.15);
        color: var(--accent-imdb);
        padding: 0.15rem 0.5rem;
        border-radius: 4px;
        font-size: 0.7rem;
        font-weight: 700;
        margin-left: 0.4rem;
    }

    /* Footer */
    .app-footer {
        text-align: center;
        padding: 3rem 1rem 1.5rem 1rem;
        border-top: 1px solid var(--glass-border);
        margin-top: 3.5rem;
    }
    .footer-brand {
        font-size: 0.95rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 0.35rem;
    }
    .footer-desc {
        font-size: 0.82rem;
        color: var(--text-muted);
        margin-bottom: 1rem;
    }
    .footer-link {
        color: var(--accent-imdb) !important;
        text-decoration: none !important;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        transition: opacity 0.2s ease;
    }
    .footer-link:hover {
        opacity: 0.85;
    }

    /* Responsive Mobile Adjustments */
    @media (max-width: 768px) {
        .metric-grid {
            grid-template-columns: repeat(2, 1fr) !important;
        }
        .hero-title {
            font-size: 1.85rem !important;
        }
        .pipeline-arrow {
            display: none !important;
        }
        .glass-card {
            padding: 1.4rem 1.2rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# Machine Learning Preprocessing Pipeline (Identical to Notebook)
# ==============================================================================
def remove_html_tags(text: str) -> str:
    """Strip HTML tags using BeautifulSoup."""
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()


def expand_contractions(text: str) -> str:
    """Expand contractions (e.g., won't -> will not)."""
    return contractions.fix(text)


def remove_special_characters(text: str) -> str:
    """Keep only alphanumeric characters and standard spaces."""
    return re.sub(r"[^a-zA-Z0-9\s]", " ", text)


def remove_extra_spaces(text: str) -> str:
    """Collapse consecutive whitespaces and strip boundaries."""
    return re.sub(r"\s+", " ", text).strip()


def to_lowercase(text: str) -> str:
    """Transform text to lowercase."""
    return text.lower()


def remove_emails(text: str) -> str:
    """Remove email patterns."""
    email_pattern = re.compile(r"(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)")
    return re.sub(email_pattern, "", text)


def preserve_movie_ratings(text: str) -> str:
    """Standardize explicit movie rating expressions (e.g. 10/10, 4/10)."""
    return re.sub(
        r"\b(10|[0-9])\s*/\s*10\b",
        lambda match: f"rating{match.group(1)} outof10",
        text,
        flags=re.IGNORECASE
    )


def preserve_sentiment_punctuation(text: str) -> str:
    """Convert exclamation and question marks into descriptive sentiment tokens."""
    text = re.sub(r"!+", " exclamation ", text)
    text = re.sub(r"\?+", " question ", text)
    return text


def mark_negation(text: str) -> str:
    """Bind negation keywords to succeeding tokens (e.g., not good -> not_good)."""
    return re.sub(r"\bnot\s+(\w+)", r"not_\1", text)


def clean_text(
    text: str,
    remove_html: bool = True,
    expand_contract: bool = True,
    lowercase: bool = True,
    remove_special: bool = True,
    remove_spaces: bool = True
) -> str:
    """Unified preprocessing pipeline matching training environment."""
    if remove_html:
        text = re.sub(r"<br\s*/?>", " ", text)
        text = remove_html_tags(text)

    text = remove_emails(text)
    text = preserve_movie_ratings(text)
    text = preserve_sentiment_punctuation(text)
    text = mark_negation(text)

    if expand_contract:
        text = expand_contractions(text)

    if lowercase:
        text = to_lowercase(text)

    if remove_special:
        text = remove_special_characters(text)

    if remove_spaces:
        text = remove_extra_spaces(text)

    return text


# ==============================================================================
# Model & Vectorizer Cache Loader
# ==============================================================================
@st.cache_resource(show_spinner=False)
def load_model_artifacts():
    """Load serialized model and TF-IDF vectorizer from disk."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "model", "sentiment_model.joblib")
    vectorizer_path = os.path.join(base_dir, "model", "vectorizer.joblib")

    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        return None, None, f"Artifacts not found in {os.path.join(base_dir, 'model')}"

    try:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        return model, vectorizer, None
    except Exception as exc:
        return None, None, f"Failed to deserialize artifacts: {str(exc)}"


# ==============================================================================
# Main Application Flow
# ==============================================================================
def main():
    model, vectorizer, load_error = load_model_artifacts()

    # --- Top Brand Navigation Bar ---
    st.markdown("""
    <div class="top-nav">
        <div class="brand-group">
            <span class="brand-icon">🎬</span>
            <div>
                <div class="brand-title">IMDB Sentiment AI</div>
                <div class="brand-badge">Classical NLP • Real-Time Inference</div>
            </div>
        </div>
        <div style="display: flex; align-items: center;">
            <div class="status-badge">
                <span class="status-dot"></span>
                LIVE MODEL
            </div>
            <div class="model-badge">
                Logistic Regression + TF-IDF
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- Hero Section ---
    st.markdown("""
    <div class="hero-container">
        <h1 class="hero-title">Understand the Sentiment Behind Every Review</h1>
        <p class="hero-subtitle">
            Analyze movie reviews with calibrated probabilistic confidence using a high-performance 
            Logistic Regression model trained on 40,000 sublinear TF-IDF features.
        </p>
        <div class="tech-pill-row">
            <div class="tech-pill"><span>⚙️</span> Representation: <strong>TF-IDF (1-2 N-grams)</strong></div>
            <div class="tech-pill"><span>🤖</span> Classifier: <strong>Logistic Regression (C=5.0)</strong></div>
            <div class="tech-pill"><span>🎯</span> Benchmark: <strong>91.99% Accuracy</strong></div>
            <div class="tech-pill"><span>⚡</span> Latency: <strong>&lt; 5ms Inference</strong></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- Horizontal Model Performance Metric Cards ---
    st.markdown("""
    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-label">Accuracy</div>
            <div class="metric-value">91.99%</div>
            <div class="metric-caption">Holdout Test Set</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">F1 Score</div>
            <div class="metric-value">92.03%</div>
            <div class="metric-caption">Harmonic Mean</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Precision</div>
            <div class="metric-value">91.57%</div>
            <div class="metric-caption">Positive Confidence</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Recall</div>
            <div class="metric-value">92.50%</div>
            <div class="metric-caption">Class Coverage</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Fail-safe notification if artifacts are missing
    if load_error is not None:
        st.error(f"⚠️ Model Artifact Error: {load_error}")
        return

    # --- Main Analysis Glass Card ---
    st.markdown("""
    <div class="glass-card">
        <div class="card-header-title">🎬 Analyze a Movie Review</div>
        <div class="card-header-subtitle">Paste any movie review below or click a curated sample to test real-time classification.</div>
    </div>
    """, unsafe_allow_html=True)

    # Session State for Example Prompts
    if "review_text" not in st.session_state:
        st.session_state["review_text"] = ""

    # Curated Example Buttons
    col_ex1, col_ex2, col_ex3 = st.columns(3)
    with col_ex1:
        if st.button("✨ Positive Example", key="btn_pos", use_container_width=True, type="secondary"):
            st.session_state["review_text"] = "This movie was absolutely fantastic. The story was engaging and the performances were excellent."
    with col_ex2:
        if st.button("💥 Negative Example", key="btn_neg", use_container_width=True, type="secondary"):
            st.session_state["review_text"] = "One of the worst movies I have ever watched. The story was boring and the acting was terrible."
    with col_ex3:
        if st.button("🎭 Nuanced Example", key="btn_nuance", use_container_width=True, type="secondary"):
            st.session_state["review_text"] = "The cinematography was impressive, but the pacing was not good and the story felt weak."

    # Text Area Input
    user_input = st.text_area(
        label="Review Text Input",
        value=st.session_state["review_text"],
        placeholder="Example: This movie was an absolute masterpiece with stellar performances and unforgettable direction...",
        height=150,
        label_visibility="collapsed"
    )

    # Action Buttons: Analyze & Clear
    col_action1, col_action2 = st.columns([3, 1])
    with col_action1:
        analyze_clicked = st.button("Analyze Sentiment →", key="btn_analyze", type="primary", use_container_width=True)
    with col_action2:
        if st.button("Clear Input", key="btn_clear", type="secondary", use_container_width=True):
            st.session_state["review_text"] = ""
            st.rerun()

    # --- Prediction Execution & Presentation ---
    if analyze_clicked:
        raw_text = user_input.strip()

        if not raw_text:
            st.warning("⚠️ Please provide a movie review text before running the sentiment analyzer.")
        else:
            with st.spinner("Processing NLP pipeline..."):
                try:
                    # 1. Preprocess review text
                    cleaned_review = clean_text(raw_text)

                    # 2. Transform text via TF-IDF
                    vectorized_review = vectorizer.transform([cleaned_review])

                    # 3. Model inference
                    probabilities = model.predict_proba(vectorized_review)[0]
                    negative_prob = float(probabilities[0])
                    positive_prob = float(probabilities[1])

                    is_positive = positive_prob >= 0.5
                    confidence_percent = (positive_prob if is_positive else negative_prob) * 100

                    # --- Result Card Presentation ---
                    if is_positive:
                        st.markdown(f"""
                        <div class="result-container result-positive-box">
                            <div class="result-header-row">
                                <div class="result-badge result-badge-pos">
                                    <span>🟢</span> POSITIVE SENTIMENT
                                </div>
                                <div>
                                    <div class="confidence-score-badge">{confidence_percent:.2f}%</div>
                                    <div class="confidence-caption">Model Confidence</div>
                                </div>
                            </div>
                            <div class="bar-row">
                                <div class="bar-header">
                                    <span style="color: #4ADE80;">Positive Probability</span>
                                    <span style="font-family: 'JetBrains Mono', monospace;">{positive_prob * 100:.2f}%</span>
                                </div>
                                <div class="bar-track">
                                    <div class="bar-fill-pos" style="width: {positive_prob * 100:.2f}%;"></div>
                                </div>
                            </div>
                            <div class="bar-row">
                                <div class="bar-header">
                                    <span style="color: #9CA3AF;">Negative Probability</span>
                                    <span style="font-family: 'JetBrains Mono', monospace;">{negative_prob * 100:.2f}%</span>
                                </div>
                                <div class="bar-track">
                                    <div class="bar-fill-neg" style="width: {negative_prob * 100:.2f}%;"></div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="result-container result-negative-box">
                            <div class="result-header-row">
                                <div class="result-badge result-badge-neg">
                                    <span>🔴</span> NEGATIVE SENTIMENT
                                </div>
                                <div>
                                    <div class="confidence-score-badge">{confidence_percent:.2f}%</div>
                                    <div class="confidence-caption">Model Confidence</div>
                                </div>
                            </div>
                            <div class="bar-row">
                                <div class="bar-header">
                                    <span style="color: #F87171;">Negative Probability</span>
                                    <span style="font-family: 'JetBrains Mono', monospace;">{negative_prob * 100:.2f}%</span>
                                </div>
                                <div class="bar-track">
                                    <div class="bar-fill-neg" style="width: {negative_prob * 100:.2f}%;"></div>
                                </div>
                            </div>
                            <div class="bar-row">
                                <div class="bar-header">
                                    <span style="color: #9CA3AF;">Positive Probability</span>
                                    <span style="font-family: 'JetBrains Mono', monospace;">{positive_prob * 100:.2f}%</span>
                                </div>
                                <div class="bar-track">
                                    <div class="bar-fill-pos" style="width: {positive_prob * 100:.2f}%;"></div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    # Collapsible Technical Preprocessing Inspector
                    with st.expander("🔍 View Preprocessing Pipeline & Feature Activations"):
                        col_t1, col_t2 = st.columns(2)
                        with col_t1:
                            st.caption("Cleaned Input Tokens")
                            st.code(cleaned_review, language="text")
                        with col_t2:
                            st.caption("Feature Vector Statistics")
                            st.markdown(f"""
                            - **Non-Zero Features Activated**: `{vectorized_review.nnz}`
                            - **Logit Decision Score**: `{float(model.decision_function(vectorized_review)[0]):.4f}`
                            - **Posterior Sigmoid $\\sigma(z)$**: `{positive_prob:.6f}`
                            """)

                except Exception as err:
                    st.error("Prediction computation encountered an unexpected error. Please verify input and retry.")
                    st.caption(f"Trace details: {html.escape(str(err))}")

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Section: How The Model Works ---
    with st.expander("🧠 How the Model Works • System Architecture", expanded=False):
        st.markdown("""
        <div class="pipeline-grid">
            <div class="pipeline-step">
                <div class="pipeline-step-title">1. Raw Review</div>
                <div class="pipeline-step-desc">Unstructured User Text</div>
            </div>
            <div class="pipeline-arrow">→</div>
            <div class="pipeline-step">
                <div class="pipeline-step-title">2. Preprocessing</div>
                <div class="pipeline-step-desc">HTML, Contractions, Negations</div>
            </div>
            <div class="pipeline-arrow">→</div>
            <div class="pipeline-step">
                <div class="pipeline-step-title">3. TF-IDF Vectorizer</div>
                <div class="pipeline-step-desc">40K Features (Unigrams + Bigrams)</div>
            </div>
            <div class="pipeline-arrow">→</div>
            <div class="pipeline-step">
                <div class="pipeline-step-title">4. Logistic Regression</div>
                <div class="pipeline-step-desc">Linear Boundary (C=5.0)</div>
            </div>
            <div class="pipeline-arrow">→</div>
            <div class="pipeline-step">
                <div class="pipeline-step-title">5. Sentiment & Prob</div>
                <div class="pipeline-step-desc">Calibrated Sigmoid Output</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_spec1, col_spec2 = st.columns(2)
        with col_spec1:
            st.markdown("""
            **Model Architecture Specifications:**
            - **Algorithm**: Logistic Regression with L2 Regularization
            - **Optimization Solver**: `liblinear` (Coordinate Descent)
            - **Regularization Strength ($C$)**: `5.0`
            - **Loss Function**: Binary Cross-Entropy (Log Loss)
            """)
        with col_spec2:
            st.markdown("""
            **Feature Engineering Specifications:**
            - **Vectorization**: TfidfVectorizer with Sublinear Scaling ($1 + \\log(tf)$)
            - **N-gram Range**: `(1, 2)` (Captures single words & word pairs)
            - **Vocabulary Cap**: Top `40,000` features
            - **Document Frequency Filters**: `min_df=2`, `max_df=0.98`
            """)

    # --- Section: Benchmark Performance Comparison ---
    with st.expander("📊 Benchmark Performance • Model Comparisons", expanded=False):
        st.markdown("""
        All models evaluated on the identical 10,000-sample test partition from the 50,000 IMDB dataset without data leakage:
        
        <table class="benchmark-table">
            <thead>
                <tr>
                    <th>Model Architecture</th>
                    <th>Feature Representation</th>
                    <th>Accuracy</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1 Score</th>
                </tr>
            </thead>
            <tbody>
                <tr class="highlight-row">
                    <td>Logistic Regression <span class="benchmark-tag">DEPLOYED</span></td>
                    <td>TF-IDF (Sublinear, N-grams 1-2)</td>
                    <td>91.99%</td>
                    <td>91.57%</td>
                    <td>92.50%</td>
                    <td>92.03%</td>
                </tr>
                <tr>
                    <td>Linear SVM (LinearSVC)</td>
                    <td>TF-IDF (Sublinear, N-grams 1-2)</td>
                    <td>91.72%</td>
                    <td>91.26%</td>
                    <td>92.28%</td>
                    <td>91.77%</td>
                </tr>
                <tr>
                    <td>Logistic Regression</td>
                    <td>Bag-of-Words (CountVectorizer)</td>
                    <td>91.22%</td>
                    <td>90.76%</td>
                    <td>91.78%</td>
                    <td>91.27%</td>
                </tr>
                <tr>
                    <td>Linear SVM (LinearSVC)</td>
                    <td>Bag-of-Words (CountVectorizer)</td>
                    <td>89.15%</td>
                    <td>89.14%</td>
                    <td>89.16%</td>
                    <td>89.15%</td>
                </tr>
            </tbody>
        </table>
        """, unsafe_allow_html=True)

    # --- Modern Brand Footer ---
    st.markdown("""
    <div class="app-footer">
        <div class="footer-brand">🎬 IMDB Sentiment Analysis AI</div>
        <div class="footer-desc">
            Production Machine Learning Pipeline • Classical NLP • Scikit-learn • Streamlit
        </div>
        <a class="footer-link" href="https://github.com/AhmedShaban0/IMDB-Sentiment-Analysis" target="_blank" rel="noopener noreferrer">
            <svg height="16" width="16" viewBox="0 0 16 16" fill="currentColor" style="vertical-align: text-bottom; margin-right: 4px;">
                <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path>
            </svg>
            Explore Project Codebase & Experiments on GitHub
        </a>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
