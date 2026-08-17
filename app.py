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
    page_title="Sentiment Engine — Bidirectional LSTM",
    page_icon="λ",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ────────────────────────────────────────────────────────────────────────────
# CSS — "signal console" design system
# ────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap');

:root {
    --bg:        #0A0B0F;
    --panel:     #13151C;
    --panel-2:   #171A23;
    --border:    #242832;
    --ink-1:     #F2F4F8;
    --ink-2:     #9BA2B0;
    --ink-3:     #565C6B;
    --amber:     #F0A93A;
    --teal:      #4FD1C5;
    --rose:      #FB7185;
    --mono: 'IBM Plex Mono', ui-monospace, monospace;
    --sans: 'Inter', -apple-system, sans-serif;
}

html, body, [class*="css"] { font-family: var(--sans); }

/* App background — flat void with a faint instrument grid */
.stApp {
    background-color: var(--bg);
    background-image:
        linear-gradient(rgba(255,255,255,0.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.025) 1px, transparent 1px);
    background-size: 34px 34px;
}
[data-testid="stDecoration"] { display: none; }
[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }

/* ── Sidebar ────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: var(--panel);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .block-container { padding-top: 2rem; }

.eng-kicker {
    font-family: var(--mono); font-size: 0.7rem; font-weight: 600;
    letter-spacing: 0.16em; color: var(--amber); text-transform: uppercase;
    border-bottom: 1px solid var(--border); padding-bottom: 0.5rem; margin-bottom: 0.9rem;
}

.eng-arch-list { display: flex; flex-direction: column; gap: 0; margin-bottom: 0.4rem; }
.eng-arch-item {
    display: flex; align-items: baseline; gap: 0.6rem;
    padding: 0.42rem 0; border-bottom: 1px dashed var(--border);
    font-family: var(--mono);
}
.eng-arch-item:last-child { border-bottom: none; }
.eng-arch-index { color: var(--ink-3); font-size: 0.72rem; width: 1.4rem; flex-shrink: 0; }
.eng-arch-name { color: var(--ink-1); font-size: 0.82rem; }

.eng-concepts { font-family: var(--mono); font-size: 0.78rem; color: var(--ink-2); line-height: 1.9; }
.eng-concepts .k { color: var(--teal); }

.eng-caption { font-family: var(--mono); font-size: 0.72rem; color: var(--ink-3); letter-spacing: 0.04em; }

/* ── Header ─────────────────────────────────────────────────────────────── */
.eng-header { margin-bottom: 1.6rem; }
.eng-header .eng-kicker { border-bottom: none; padding-bottom: 0; margin-bottom: 0.6rem; }
.eng-title {
    font-family: var(--mono); font-size: 1.9rem; font-weight: 700;
    color: var(--ink-1); letter-spacing: -0.01em; margin: 0;
}
.eng-subtitle { font-family: var(--sans); font-size: 0.92rem; color: var(--ink-2); margin-top: 0.35rem; }

/* ── Stat strip ─────────────────────────────────────────────────────────── */
.eng-stat-strip {
    display: grid; grid-auto-flow: column; grid-auto-columns: 1fr;
    background: var(--panel); border: 1px solid var(--border); border-radius: 6px;
    margin: 1.4rem 0 1.8rem 0; overflow: hidden;
}
.eng-stat { padding: 0.9rem 0.6rem; text-align: center; border-left: 1px solid var(--border); }
.eng-stat:first-child { border-left: none; }
.eng-stat-val { font-family: var(--mono); font-size: 1.25rem; font-weight: 600; color: var(--amber); }
.eng-stat-label {
    font-family: var(--mono); font-size: 0.62rem; color: var(--ink-3);
    letter-spacing: 0.1em; text-transform: uppercase; margin-top: 0.25rem;
}

/* ── Section label ──────────────────────────────────────────────────────── */
.eng-section {
    font-family: var(--mono); font-size: 0.72rem; font-weight: 600;
    letter-spacing: 0.14em; text-transform: uppercase; color: var(--ink-3);
    margin: 1.6rem 0 0.6rem 0; display: flex; align-items: center; gap: 0.6rem;
}
.eng-section::after { content: ""; flex: 1; height: 1px; background: var(--border); }

/* ── Inputs ─────────────────────────────────────────────────────────────── */
.stTextArea textarea {
    background-color: var(--panel) !important; border: 1px solid var(--border) !important;
    border-radius: 6px !important; color: var(--ink-1) !important; font-family: var(--sans) !important;
}
.stTextArea textarea:focus { border-color: var(--amber) !important; box-shadow: 0 0 0 1px var(--amber) !important; }
.stTextArea label, .stSlider label { font-family: var(--mono) !important; font-size: 0.78rem !important; color: var(--ink-2) !important; }

.streamlit-expanderHeader {
    background-color: var(--panel) !important; border: 1px solid var(--border) !important;
    border-radius: 6px !important; font-family: var(--mono) !important; font-size: 0.82rem !important;
    color: var(--ink-2) !important;
}
.streamlit-expanderContent { background-color: var(--panel) !important; border: 1px solid var(--border) !important; border-top: none !important; }

.stSlider [data-baseweb="slider"] > div > div { background: var(--amber) !important; }
.stSlider [role="slider"] { background-color: var(--amber) !important; border-color: var(--amber) !important; }

/* ── Buttons ────────────────────────────────────────────────────────────── */
.stButton button {
    font-family: var(--mono) !important; font-size: 0.78rem !important; font-weight: 600 !important;
    letter-spacing: 0.06em !important; text-transform: uppercase !important;
    border-radius: 6px !important; padding: 0.6rem 1rem !important; transition: all 0.15s ease !important;
}
.stButton button[kind="primary"] {
    background-color: var(--amber) !important; color: #0A0B0F !important; border: 1px solid var(--amber) !important;
}
.stButton button[kind="primary"]:hover { background-color: #f5bb5c !important; }
.stButton button[kind="secondary"] {
    background-color: transparent !important; color: var(--ink-2) !important; border: 1px solid var(--border) !important;
}
.stButton button[kind="secondary"]:hover { border-color: var(--ink-2) !important; color: var(--ink-1) !important; }

/* ── Result panel ───────────────────────────────────────────────────────── */
.eng-result {
    border: 1px solid var(--border); border-left: 3px solid var(--ink-3);
    background: var(--panel); border-radius: 6px; padding: 1.3rem 1.5rem; margin: 1.2rem 0 1rem 0;
}
.eng-result--pos { border-left-color: var(--teal); }
.eng-result--neg { border-left-color: var(--rose); }

.eng-result-top { display: flex; align-items: baseline; justify-content: space-between; }
.eng-result-label { font-family: var(--mono); font-size: 1.5rem; font-weight: 700; letter-spacing: 0.02em; }
.eng-result--pos .eng-result-label { color: var(--teal); }
.eng-result--neg .eng-result-label { color: var(--rose); }
.eng-result-conf { font-family: var(--mono); font-size: 0.85rem; color: var(--ink-2); }

/* Diverging confidence gauge — signature element */
.eng-gauge { margin-top: 1rem; }
.eng-gauge-track {
    position: relative; height: 10px; border-radius: 3px; background: var(--panel-2);
    border: 1px solid var(--border); overflow: hidden;
}
.eng-gauge-half { position: absolute; top: 0; bottom: 0; }
.eng-gauge-half.neg { right: 50%; background: linear-gradient(90deg, transparent, var(--rose)); }
.eng-gauge-half.pos { left: 50%; background: linear-gradient(90deg, var(--teal), transparent 100%) ; }
.eng-gauge-center { position: absolute; top: -3px; bottom: -3px; left: 50%; width: 1px; background: var(--ink-3); }
.eng-gauge-marker {
    position: absolute; top: -4px; width: 2px; height: 18px; background: var(--ink-1);
    box-shadow: 0 0 6px rgba(242,244,248,0.6);
}
.eng-gauge-readout {
    display: flex; justify-content: space-between; margin-top: 0.5rem;
    font-family: var(--mono); font-size: 0.72rem; color: var(--ink-3); letter-spacing: 0.04em;
}
.eng-gauge-readout .neg-val { color: var(--rose); }
.eng-gauge-readout .pos-val { color: var(--teal); }

/* ── Batch table ────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 6px; overflow: hidden; }

.eng-footer {
    font-family: var(--mono); font-size: 0.7rem; color: var(--ink-3);
    text-align: center; letter-spacing: 0.05em; margin-top: 2rem;
}
hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────────
# Load model & artifacts (cached)
# ────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model weights…")
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
# Small render helpers
# ────────────────────────────────────────────────────────────────────────────
def render_gauge(prob: float) -> str:
    """Diverging confidence gauge centered at 0.5 — the marker sits at the raw probability."""
    marker_pct = max(0.0, min(100.0, prob * 100))
    neg_width  = max(0.0, (0.5 - prob) * 200)   # % of the left half filled
    pos_width  = max(0.0, (prob - 0.5) * 200)   # % of the right half filled
    return f"""
    <div class="eng-gauge">
        <div class="eng-gauge-track">
            <div class="eng-gauge-half neg" style="width:{neg_width:.1f}%;"></div>
            <div class="eng-gauge-half pos" style="width:{pos_width:.1f}%;"></div>
            <div class="eng-gauge-center"></div>
            <div class="eng-gauge-marker" style="left:calc({marker_pct:.1f}% - 1px);"></div>
        </div>
        <div class="eng-gauge-readout">
            <span class="neg-val">NEGATIVE {(1-prob)*100:.1f}%</span>
            <span>P = {prob:.4f}</span>
            <span class="pos-val">POSITIVE {prob*100:.1f}%</span>
        </div>
    </div>
    """


def render_stat_strip(items):
    cells = "".join(
        f'<div class="eng-stat"><div class="eng-stat-val">{val}</div>'
        f'<div class="eng-stat-label">{label}</div></div>'
        for label, val in items
    )
    return f'<div class="eng-stat-strip">{cells}</div>'


# ────────────────────────────────────────────────────────────────────────────
# Sidebar
# ────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="eng-kicker">Inference Settings</div>', unsafe_allow_html=True)
    threshold = st.slider(
        "Decision threshold",
        min_value=0.1, max_value=0.9, value=0.5, step=0.05,
        help="P(positive) ≥ threshold is classified as Positive."
    )

    st.markdown('<div class="eng-kicker" style="margin-top:1.6rem;">Architecture</div>', unsafe_allow_html=True)
    layers = [
        "Embedding — 128d",
        "SpatialDropout1D",
        "Bidirectional LSTM — 128u",
        "LSTM — 64u",
        "Dropout — 0.3",
        "Dense — 64, ReLU",
        "Dense — 1, Sigmoid",
    ]
    arch_html = '<div class="eng-arch-list">' + "".join(
        f'<div class="eng-arch-item"><span class="eng-arch-index">{i:02d}</span>'
        f'<span class="eng-arch-name">{name}</span></div>'
        for i, name in enumerate(layers, start=1)
    ) + '</div>'
    st.markdown(arch_html, unsafe_allow_html=True)

    st.markdown('<div class="eng-kicker" style="margin-top:1.6rem;">Method</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="eng-concepts">
    <span class="k">BPTT</span> — backprop through time<br>
    <span class="k">LSTM gates</span> — forget / input / output<br>
    <span class="k">Bidirectional</span> — forward + backward context<br>
    <span class="k">Adam</span> — adaptive learning rate<br>
    <span class="k">Dropout</span> — regularization<br>
    <span class="k">EarlyStopping</span> + <span class="k">ReduceLROnPlateau</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr style="margin:1.4rem 0 1rem 0;">', unsafe_allow_html=True)
    st.markdown('<div class="eng-caption">DATASET · SENTIMENT140 (1.6M TWEETS)</div>', unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────────
# Main UI — header
# ────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="eng-header">
    <div class="eng-kicker">Bidirectional LSTM · Sequence Classification</div>
    <div class="eng-title">Sentiment Engine</div>
    <div class="eng-subtitle">Deep-learning tweet sentiment classifier trained end-to-end on Sentiment140.</div>
</div>
""", unsafe_allow_html=True)

# Load artifacts (show error if files missing)
try:
    model, tokenizer, config = load_artifacts()
    model_loaded = True
except Exception as e:
    st.error(
        f"Could not load model: {e}\n\n"
        "Run the training notebook first and make sure these files are present:\n"
        "`lstm_sentiment_model.keras`, `tokenizer.pkl`, `model_config.pkl`"
    )
    model_loaded = False

if model_loaded:
    stat_items = [
        ("Accuracy", f"{config.get('test_accuracy', 0)*100:.1f}%"),
        ("AUC-ROC",  f"{config.get('test_auc', 0):.3f}"),
        ("F1-Score", f"{config.get('test_f1', 0):.3f}"),
        ("Arch",     config.get('model_type', 'BiLSTM')),
    ]
    st.markdown(render_stat_strip(stat_items), unsafe_allow_html=True)

# ── Input area ───────────────────────────────────────────────────────────────
st.markdown('<div class="eng-section">Input</div>', unsafe_allow_html=True)
tweet_input = st.text_area(
    "Tweet or free text",
    placeholder="I absolutely love this new update, it's fantastic.",
    height=120,
    label_visibility="collapsed",
    help="URLs, @mentions, and #hashtags are stripped automatically before inference."
)

with st.expander("Batch mode — one line per tweet"):
    batch_input = st.text_area(
        "Batch input",
        placeholder="I love this.\nTerrible experience.\nJust another day.",
        height=130,
        label_visibility="collapsed"
    )

col_btn, col_clear = st.columns([3, 1])
with col_btn:
    predict_clicked = st.button("Run inference", use_container_width=True, type="primary")
with col_clear:
    clear_clicked = st.button("Clear", use_container_width=True, type="secondary")

if clear_clicked:
    st.rerun()

# ── Single prediction ─────────────────────────────────────────────────────────
if predict_clicked and model_loaded:
    if tweet_input.strip():
        with st.spinner("Running forward pass…"):
            label, prob, cleaned = predict(tweet_input, model, tokenizer, config, threshold)

        result_class = "eng-result--pos" if label == "Positive" else "eng-result--neg"

        st.markdown('<div class="eng-section">Result</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="eng-result {result_class}">
            <div class="eng-result-top">
                <div class="eng-result-label">{label.upper()}</div>
                <div class="eng-result-conf">confidence {max(prob, 1-prob)*100:.1f}%</div>
            </div>
            {render_gauge(prob)}
        </div>
        """, unsafe_allow_html=True)

        with st.expander("Preprocessed input passed to the model"):
            st.code(cleaned if cleaned else "(empty after cleaning)", language=None)

    else:
        st.warning("Enter some text to analyze.")

    # Batch prediction
    if batch_input.strip():
        st.markdown('<div class="eng-section">Batch results</div>', unsafe_allow_html=True)
        lines = [l.strip() for l in batch_input.strip().split('\n') if l.strip()]
        results = []
        with st.spinner(f"Scoring {len(lines)} lines…"):
            for line in lines:
                lbl, pr, _ = predict(line, model, tokenizer, config, threshold)
                results.append({
                    "Text"       : line[:80] + ("…" if len(line) > 80 else ""),
                    "Sentiment"  : lbl,
                    "P(Positive)": f"{pr*100:.1f}%",
                    "P(Negative)": f"{(1-pr)*100:.1f}%"
                })

        import pandas as pd
        df_results = pd.DataFrame(results)
        st.dataframe(df_results, use_container_width=True, hide_index=True)

        pos_count = sum(1 for r in results if r['Sentiment'] == 'Positive')
        neg_count = len(results) - pos_count
        st.markdown(
            render_stat_strip([
                ("Total", len(results)),
                ("Positive", pos_count),
                ("Negative", neg_count),
            ]),
            unsafe_allow_html=True
        )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown('<hr>', unsafe_allow_html=True)
st.markdown(
    '<div class="eng-footer">BIDIRECTIONAL LSTM · BACKPROP THROUGH TIME · SENTIMENT140</div>',
    unsafe_allow_html=True
)
