"""
IMDB Movie Review Sentiment Analysis - Streamlit Web Application

This application loads a pre-trained TF-IDF vectorizer and Logistic Regression model
to classify movie reviews as Positive or Negative sentiment, displaying calibrated
probabilistic confidence and decision metrics.
"""

import os
import re
import html
import joblib
import streamlit as st
from bs4 import BeautifulSoup
import contractions

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="IMDB Movie Review Sentiment Analysis",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Clean, Professional Presentation
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.15rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .sentiment-card {
        padding: 1.5rem;
        border-radius: 0.75rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
        border: 1px solid #E2E8F0;
    }
    .sentiment-positive {
        background-color: #F0FDF4;
        border-color: #86EFAC;
        color: #166534;
    }
    .sentiment-negative {
        background-color: #FEF2F2;
        border-color: #FECACA;
        color: #991B1B;
    }
    .metric-box {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 0.5rem;
        padding: 0.85rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .example-btn {
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# Text Preprocessing Pipeline (Identical to Notebook)
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
# Model and Vectorizer Loading
# ==============================================================================
@st.cache_resource(show_spinner="Loading trained sentiment model and vectorizer...")
def load_model_artifacts():
    """Load pre-trained model and TF-IDF vectorizer from disk."""
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
# Streamlit Application Layout
# ==============================================================================
def main():
    model, vectorizer, load_error = load_model_artifacts()

    # --- Sidebar ---
    with st.sidebar:
        st.header("📌 Project Details")
        st.markdown("""
        **Task**: Binary Movie Review Sentiment Classification  
        **Dataset**: IMDB Large Movie Review Dataset (50,000 balanced reviews)  
        **Selected Best Model**: **Logistic Regression** ($C=5.0$)  
        **Vectorization**: **TF-IDF** (Sublinear TF, N-grams 1-2, 40,000 features)  
        """)

        st.markdown("---")
        st.subheader("📊 Empirical Test Results")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Accuracy", "91.99%")
            st.metric("Precision", "91.57%")
        with col_m2:
            st.metric("F1-Score", "92.03%")
            st.metric("Recall", "92.50%")

        st.caption("Evaluated on 10,000 holdout test reviews without data leakage.")

        st.markdown("---")
        st.subheader("🛠️ Tech Stack")
        st.markdown("""
        - Python 3.11
        - Scikit-learn
        - Streamlit
        - Pandas & NumPy
        - BeautifulSoup4 & Contractions
        - Joblib
        """)

        st.markdown("---")
        st.info("💡 The app performs real-time inference using the pre-trained model without retraining.")

    # --- Main Header ---
    st.markdown('<div class="main-header">🎬 IMDB Movie Review Sentiment Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Analyze the sentiment of an IMDB movie review using a machine learning model.</div>', unsafe_allow_html=True)

    st.write(
        "Enter a movie review below, and let the trained machine learning model predict whether "
        "the sentiment is **positive** or **negative**, along with calibrated probability confidence."
    )

    # Check Artifact Status
    if load_error is not None:
        st.error(
            f"⚠️ **Model Loading Error**: {load_error}.\\n\\n"
            "Please ensure that the trained artifacts are generated by running the notebook at "
            "`notebooks/IMDB_Project.ipynb` or checking the `model/` directory."
        )
        return

    # --- Example Reviews Section ---
    st.subheader("💡 Try an Example Review")
    st.caption("Click any example button to load it into the review text area:")

    examples = [
        (
            "🌟 Positive Example",
            "This movie was absolutely fantastic. The story was engaging and the performances were excellent."
        ),
        (
            "💥 Negative Example",
            "One of the worst movies I have ever watched. The story was boring and the acting was terrible."
        ),
        (
            "⚖️ Nuanced Example",
            "The cinematography was not bad, but the pacing was not good either."
        )
    ]

    # Initialize review text in session state if not set
    if "review_input" not in st.session_state:
        st.session_state["review_input"] = ""

    btn_cols = st.columns(len(examples))
    for i, (label, text) in enumerate(examples):
        with btn_cols[i]:
            if st.button(label, key=f"ex_btn_{i}", use_container_width=True):
                st.session_state["review_input"] = text

    # --- Review Input Area ---
    st.markdown("### ✍️ Enter Your Review")
    user_review = st.text_area(
        label="Movie review input",
        value=st.session_state["review_input"],
        placeholder="Write your movie review here...",
        height=160,
        label_visibility="collapsed"
    )

    # Prediction Action Controls
    col_pred, col_clear = st.columns([1, 4])
    with col_pred:
        predict_clicked = st.button("Predict Sentiment", type="primary", use_container_width=True)
    with col_clear:
        if st.button("Clear Text", use_container_width=False):
            st.session_state["review_input"] = ""
            st.rerun()

    # --- Prediction & Results ---
    if predict_clicked:
        raw_text = user_review.strip()

        # Input Validation
        if not raw_text:
            st.warning("⚠️ Please enter a movie review to analyze before clicking Predict.")
            return

        with st.spinner("Preprocessing and predicting sentiment..."):
            try:
                # 1. Preprocess review using training pipeline
                cleaned_review = clean_text(raw_text)

                # 2. Vectorize text with pre-fitted TF-IDF
                vectorized_review = vectorizer.transform([cleaned_review])

                # 3. Predict class and posterior probability
                # LogisticRegression supports calibrated predict_proba
                probabilities = model.predict_proba(vectorized_review)[0]
                negative_prob = float(probabilities[0])
                positive_prob = float(probabilities[1])

                is_positive = positive_prob >= 0.5
                predicted_label = "Positive Sentiment" if is_positive else "Negative Sentiment"
                confidence = positive_prob if is_positive else negative_prob

                # --- Results Display ---
                st.markdown("---")
                st.subheader("🎯 Prediction Result")

                if is_positive:
                    st.success(f"### 🎉 {predicted_label}")
                else:
                    st.error(f"### 👎 {predicted_label}")

                # Metrics Breakdown
                m_col1, m_col2, m_col3 = st.columns(3)
                with m_col1:
                    st.markdown(
                        f"""
                        <div class="metric-box">
                            <div class="metric-label">Predicted Class</div>
                            <div class="metric-value">{'Positive (1)' if is_positive else 'Negative (0)'}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                with m_col2:
                    st.markdown(
                        f"""
                        <div class="metric-box">
                            <div class="metric-label">Positive Confidence</div>
                            <div class="metric-value">{positive_prob * 100:.2f}%</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                with m_col3:
                    st.markdown(
                        f"""
                        <div class="metric-box">
                            <div class="metric-label">Negative Confidence</div>
                            <div class="metric-value">{negative_prob * 100:.2f}%</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # Confidence Bar
                st.markdown("<br>", unsafe_allow_html=True)
                st.caption(f"Confidence score for **{predicted_label}**: {confidence * 100:.2f}%")
                st.progress(min(max(confidence, 0.0), 1.0))

                # Pipeline Transparency Expander
                with st.expander("🔍 View Preprocessing & Inference Details"):
                    st.markdown(f"**Original Text ({len(raw_text)} chars):**")
                    st.code(raw_text, language="text")
                    st.markdown(f"**Cleaned Text ({len(cleaned_review)} chars):**")
                    st.code(cleaned_review, language="text")
                    st.markdown(f"**TF-IDF Non-Zero Features:** `{vectorized_review.nnz}`")
                    st.markdown(f"**Model Raw Decision Score:** `{float(model.decision_function(vectorized_review)[0]):.4f}`")

            except Exception as err:
                st.error("An unexpected error occurred during prediction. Please verify your input and try again.")
                # Log error without displaying stack trace to user
                st.caption(f"Error details: {html.escape(str(err))}")


if __name__ == "__main__":
    main()
