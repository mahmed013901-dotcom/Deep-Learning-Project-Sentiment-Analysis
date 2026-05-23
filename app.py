"""
app.py  —  Tweet Sentiment Analysis (Deep Learning: Bidirectional LSTM)
Run with:  streamlit run app.py
Requires:  lstm_sentiment_model.keras | tokenizer.pkl | model_config.pkl
"""

import streamlit as st
import pickle
import re
import numpy as np
import nltk
from nltk.corpus import stopwords

# ── TensorFlow / Keras imports ──────────────────────────────────────────────
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ── NLTK setup ───────────────────────────────────────────────────────────────
nltk.download('stopwords', quiet=True)
STOP_WORDS = set(stopwords.words('english'))
NEGATIONS  = {'no', 'not', 'nor', 'neither', 'never', 'none', "n't"}

# ────────────────────────────────────────────────────────────────────────────
# Page config
# ────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tweet Sentiment — Deep Learning",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ────────────────────────────────────────────────────────────────────────────
# CSS styling
# ────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main app background */
    .stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); }

    /* Title */
    .main-title {
        font-size: 2.4rem; font-weight: 800;
        background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; margin-bottom: 0.2rem;
    }
    .sub-title {
        text-align: center; color: #94a3b8; font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }

    /* Result card */
    .result-card {
        border-radius: 16px; padding: 1.5rem 2rem;
        text-align: center; margin: 1.2rem 0;
        animation: fadeIn 0.4s ease;
    }
    .positive { background: linear-gradient(135deg, #064e3b, #065f46); border: 1px solid #34d399; }
    .negative { background: linear-gradient(135deg, #450a0a, #7f1d1d); border: 1px solid #f87171; }
    .neutral  { background: linear-gradient(135deg, #1e3a5f, #1e40af); border: 1px solid #60a5fa; }

    .result-emoji { font-size: 3.5rem; margin-bottom: 0.3rem; }
    .result-label { font-size: 1.8rem; font-weight: 700; color: #f1f5f9; }
    .result-conf  { font-size: 0.95rem; color: #94a3b8; margin-top: 0.3rem; }

    /* Progress bar container */
    .prob-bar-wrap {
        background: #1e293b; border-radius: 12px;
        height: 18px; margin: 0.8rem 0; overflow: hidden;
    }
    .prob-bar {
        height: 100%; border-radius: 12px;
        transition: width 0.6s ease;
    }

    /* Metric cards */
    .metric-card {
        background: #1e293b; border-radius: 12px;
        padding: 0.8rem 1rem; text-align: center;
        border: 1px solid #334155;
    }
    .metric-val  { font-size: 1.4rem; font-weight: 700; color: #a78bfa; }
    .metric-name { font-size: 0.75rem; color: #64748b; margin-top: 0.2rem; }

    /* Arch badge */
    .arch-badge {
        display: inline-block; background: #1e1b4b;
        border: 1px solid #6366f1; border-radius: 20px;
        padding: 0.3rem 1rem; font-size: 0.8rem; color: #a5b4fc;
        margin: 0.3rem;
    }

    @keyframes fadeIn { from {opacity:0; transform:translateY(10px);} to {opacity:1; transform:translateY(0);} }
</style>
""", unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────────
# Load model & artifacts (cached)
# ────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading LSTM model…")
def load_artifacts():
    model     = load_model('lstm_sentiment_model.keras')
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    with open('model_config.pkl', 'rb') as f:
        config = pickle.load(f)
    return model, tokenizer, config


# ────────────────────────────────────────────────────────────────────────────
# Text preprocessing (must match training)
# ────────────────────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    filtered = [w for w in text.split() if w not in STOP_WORDS or w in NEGATIONS]
    return ' '.join(filtered)


# ────────────────────────────────────────────────────────────────────────────
# Prediction function
# ────────────────────────────────────────────────────────────────────────────
def predict(text: str, model, tokenizer, config: dict, threshold: float = 0.5):
    cleaned  = clean_text(text)
    seq      = tokenizer.texts_to_sequences([cleaned])
    padded   = pad_sequences(seq, maxlen=config['MAX_LEN'], padding='post', truncating='post')
    prob     = float(model.predict(padded, verbose=0)[0][0])
    label    = 'Positive' if prob >= threshold else 'Negative'
    return label, prob, cleaned


# ────────────────────────────────────────────────────────────────────────────
# Sidebar
# ────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    threshold = st.slider(
        "Decision Threshold", min_value=0.1, max_value=0.9,
        value=0.5, step=0.05,
        help="Probability ≥ threshold → Positive. Default 0.5."
    )

    st.divider()
    st.markdown("## 🏗️ Model Architecture")
    for badge in [
        "Embedding (128-dim)",
        "SpatialDropout1D",
        "Bidirectional LSTM (128)",
        "LSTM (64)",
        "Dropout (0.3)",
        "Dense (64, ReLU)",
        "Dense (1, Sigmoid)"
    ]:
        st.markdown(f'<span class="arch-badge">{badge}</span>', unsafe_allow_html=True)

    st.divider()
    st.markdown("## 🔬 DL Concepts Used")
    st.markdown("""
- **BPTT** — Backprop Through Time
- **LSTM gates** — forget / input / output
- **Bidirectional** — forward & backward context
- **Adam** optimizer with adaptive LR
- **Dropout** regularization
- **EarlyStopping** + **ReduceLROnPlateau**
    """)

    st.divider()
    st.caption("Dataset: Sentiment140 (1.6M tweets)")


# ────────────────────────────────────────────────────────────────────────────
# Main UI
# ────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🧠 Tweet Sentiment Analyzer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Powered by Bidirectional LSTM — Deep Learning</div>',
    unsafe_allow_html=True
)

# Load artifacts (show error if files missing)
try:
    model, tokenizer, config = load_artifacts()
    model_loaded = True
except Exception as e:
    st.error(f"⚠️ Could not load model: {e}\n\nMake sure you have run the notebook first and these files are present:\n- `lstm_sentiment_model.keras`\n- `tokenizer.pkl`\n- `model_config.pkl`")
    model_loaded = False

# Model performance metrics from training
if model_loaded:
    col1, col2, col3, col4 = st.columns(4)
    metrics = [
        ("Accuracy",  f"{config.get('test_accuracy', 0)*100:.1f}%"),
        ("AUC-ROC",   f"{config.get('test_auc', 0):.3f}"),
        ("F1-Score",  f"{config.get('test_f1', 0):.3f}"),
        ("Arch",      config.get('model_type', 'BiLSTM'))
    ]
    for col, (name, val) in zip([col1, col2, col3, col4], metrics):
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="metric-val">{val}</div>'
                f'<div class="metric-name">{name}</div></div>',
                unsafe_allow_html=True
            )

st.markdown("---")

# ── Input area ───────────────────────────────────────────────────────────────
tweet_input = st.text_area(
    "✍️ Enter a tweet or any text:",
    placeholder="e.g. I absolutely love this new update, it's fantastic!",
    height=120,
    help="Type or paste any tweet text. The model handles URLs, mentions, and hashtags automatically."
)

# Batch analysis expander
with st.expander("📋 Batch Analysis (multiple tweets)"):
    batch_input = st.text_area(
        "Enter one tweet per line:",
        placeholder="I love this!\nTerrible experience.\nJust another day.",
        height=130
    )

# ── Predict button ────────────────────────────────────────────────────────────
col_btn, col_clear = st.columns([3, 1])
with col_btn:
    predict_clicked = st.button("🔍 Analyze Sentiment", use_container_width=True, type="primary")
with col_clear:
    clear_clicked = st.button("🗑️ Clear", use_container_width=True)

if clear_clicked:
    st.rerun()

# ── Single prediction ─────────────────────────────────────────────────────────
if predict_clicked and model_loaded:
    if tweet_input.strip():
        with st.spinner("Running through LSTM layers…"):
            label, prob, cleaned = predict(tweet_input, model, tokenizer, config, threshold)

        pos_prob = prob
        neg_prob = 1 - prob

        # Result card
        if label == 'Positive':
            emoji, css_class, color = "😊", "positive", "#34d399"
        else:
            emoji, css_class, color = "😞", "negative", "#f87171"

        st.markdown(f"""
        <div class="result-card {css_class}">
            <div class="result-emoji">{emoji}</div>
            <div class="result-label">{label}</div>
            <div class="result-conf">Confidence: {max(prob, 1-prob)*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

        # Probability bars
        st.markdown("#### 📊 Class Probabilities")
        col_p, col_n = st.columns(2)

        with col_p:
            st.markdown(f"**😊 Positive** — {pos_prob*100:.1f}%")
            st.progress(pos_prob)

        with col_n:
            st.markdown(f"**😞 Negative** — {neg_prob*100:.1f}%")
            st.progress(neg_prob)

        # Preprocessed text
        with st.expander("🔍 See cleaned input (after preprocessing)"):
            st.code(cleaned if cleaned else "(empty after cleaning)", language=None)

    else:
        st.warning("Please enter some text to analyze.")

    # Batch prediction
    if batch_input.strip():
        st.markdown("---")
        st.markdown("### 📋 Batch Results")
        lines = [l.strip() for l in batch_input.strip().split('\n') if l.strip()]
        results = []
        with st.spinner(f"Analyzing {len(lines)} tweets…"):
            for line in lines:
                lbl, pr, _ = predict(line, model, tokenizer, config, threshold)
                results.append({
                    "Tweet"     : line[:80] + ("…" if len(line) > 80 else ""),
                    "Sentiment" : f"{'😊' if lbl=='Positive' else '😞'} {lbl}",
                    "P(Positive)": f"{pr*100:.1f}%",
                    "P(Negative)": f"{(1-pr)*100:.1f}%"
                })

        import pandas as pd
        df_results = pd.DataFrame(results)
        st.dataframe(df_results, use_container_width=True)

        pos_count = sum(1 for r in results if 'Positive' in r['Sentiment'])
        neg_count = len(results) - pos_count
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Tweets",    len(results))
        c2.metric("😊 Positive",     pos_count)
        c3.metric("😞 Negative",     neg_count)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center><small>Deep Learning Project | Bidirectional LSTM + Backprop Through Time | "
    "Sentiment140 Dataset</small></center>",
    unsafe_allow_html=True
)
