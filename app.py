"""
app.py  —  Student Performance Predictor (Streamlit)
=====================================================
All files live in the SAME folder as this script:
  app.py, train_model.py, students.csv,
  performance_model.pkl, label_encoder.pkl, model_metrics.pkl

Run with:
    streamlit run app.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ── Paths: everything is in the same folder as app.py ─────────────────────
HERE    = os.path.dirname(os.path.abspath(__file__))
MODEL   = os.path.join(HERE, "performance_model.pkl")
LE_PATH = os.path.join(HERE, "label_encoder.pkl")
MET     = os.path.join(HERE, "model_metrics.pkl")
DATA    = os.path.join(HERE, "students.csv")

FEATURES = ["study_hours", "attendance", "sleep_hours", "previous_score"]

# ── Colour palette ─────────────────────────────────────────────────────────
P = {
    "bg"      : "#0d1117",
    "surface" : "#161b22",
    "border"  : "#30363d",
    "accent"  : "#58a6ff",
    "High"    : "#3fb950",
    "Medium"  : "#d29922",
    "Low"     : "#f85149",
    "text"    : "#e6edf3",
    "muted"   : "#8b949e",
}

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Sora:wght@300;400;600;700&display=swap');

  html, body, [class*="css"] {{ font-family: 'Sora', sans-serif; }}
  .stApp {{ background: {P['bg']}; color: {P['text']}; }}

  /* Sidebar */
  section[data-testid="stSidebar"] {{
    background: {P['surface']};
    border-right: 1px solid {P['border']};
  }}
  section[data-testid="stSidebar"] * {{ color: {P['text']} !important; }}

  /* Slider rail */
  div[data-baseweb="slider"] div {{ background: {P['border']}; }}

  /* Metric cards */
  div[data-testid="metric-container"] {{
    background: {P['surface']};
    border: 1px solid {P['border']};
    border-radius: 12px;
    padding: 16px 20px;
  }}
  div[data-testid="metric-container"] label {{
    color: {P['muted']} !important;
    font-size: 0.78rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
  }}
  div[data-testid="metric-container"] div[data-testid="metric-value"] {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.8rem;
    color: {P['accent']};
  }}

  /* Predict button */
  div.stButton > button {{
    width: 100%;
    background: {P['accent']};
    color: #0d1117;
    font-weight: 700;
    font-size: 1rem;
    border: none;
    border-radius: 10px;
    padding: 14px 0;
    margin-top: 12px;
    transition: filter 0.2s;
  }}
  div.stButton > button:hover {{ filter: brightness(1.12); }}

  /* Result card */
  .result-card {{
    background: {P['surface']};
    border: 1px solid {P['border']};
    border-radius: 16px;
    padding: 32px 36px;
    text-align: center;
  }}
  .result-label {{
    font-size: 0.78rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: {P['muted']};
    margin-bottom: 8px;
  }}
  .result-badge {{
    display: inline-block;
    font-size: 2.4rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    border-radius: 12px;
    padding: 10px 36px;
    margin: 10px 0 18px;
  }}
  .badge-High   {{ background: rgba(63,185,80,0.15);  color: {P['High']};   border: 1px solid {P['High']}; }}
  .badge-Medium {{ background: rgba(210,153,34,0.15); color: {P['Medium']}; border: 1px solid {P['Medium']}; }}
  .badge-Low    {{ background: rgba(248,81,73,0.15);  color: {P['Low']};    border: 1px solid {P['Low']}; }}

  /* Value pill used next to each slider label */
  .val-pill {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: {P['accent']};
    background: rgba(88,166,255,0.12);
    border: 1px solid rgba(88,166,255,0.3);
    border-radius: 6px;
    padding: 2px 10px;
    float: right;
  }}

  /* Section headers */
  .section-title {{
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: {P['muted']};
    margin: 24px 0 10px;
    padding-bottom: 6px;
    border-bottom: 1px solid {P['border']};
  }}

  /* Tab styling */
  button[data-baseweb="tab"] {{
    font-family: 'Sora', sans-serif !important;
    font-size: 0.85rem !important;
    color: {P['muted']} !important;
  }}
  button[data-baseweb="tab"][aria-selected="true"] {{
    color: {P['accent']} !important;
  }}

  /* Hide Streamlit branding */
  #MainMenu, footer {{ visibility: hidden; }}
  header[data-testid="stHeader"] {{ background: transparent; }}
</style>
""", unsafe_allow_html=True)


# ── Guard: friendly error if model files are missing ──────────────────────
_missing = [name for path, name in [
    (MODEL,   "performance_model.pkl"),
    (LE_PATH, "label_encoder.pkl"),
    (MET,     "model_metrics.pkl"),
] if not os.path.exists(path)]

if _missing:
    st.error(
        "**Model files not found.** Open your terminal in this folder and run:\n\n"
        "```\npython3 train_model.py\n```\n\n"
        f"Missing: {', '.join(_missing)}\n"
        f"Expected in: `{HERE}`"
    )
    st.stop()


# ── Load assets (cached so they're only read once) ─────────────────────────
@st.cache_resource
def load_assets():
    model   = joblib.load(MODEL)
    le      = joblib.load(LE_PATH)
    metrics = joblib.load(MET)
    df      = pd.read_csv(DATA)
    return model, le, metrics, df

model, le, metrics, df = load_assets()


# ── Plotly layout helper ───────────────────────────────────────────────────
def styled_fig(fig, title="", height=340, margin=None):
    m = margin or dict(l=20, r=20, t=50, b=20)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor ="rgba(0,0,0,0)",
        font   =dict(family="Sora, sans-serif", color=P["text"]),
        title  =dict(text=title, font=dict(size=13, color=P["muted"]), x=0),
        height =height,
        margin =m,
        xaxis  =dict(gridcolor=P["border"], linecolor=P["border"], zerolinecolor="rgba(0,0,0,0)"),
        yaxis  =dict(gridcolor=P["border"], linecolor=P["border"], zerolinecolor="rgba(0,0,0,0)"),
        legend =dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
    )
    return fig

# ── Reusable value pill ────────────────────────────────────────────────────
def val_pill(v):
    return f"<span class='val-pill'>{v}</span>"


# ══════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🎓 Performance Predictor")
    st.markdown(
        f"<span style='color:{P['muted']};font-size:0.8rem'>Enter student details below</span>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── Study Habits ──────────────────────────────────────────────────────
    st.markdown('<div class="section-title">📚 Study Habits</div>', unsafe_allow_html=True)

    st.markdown(f"Study Hours / Day {val_pill('–')}", unsafe_allow_html=True)
    study_hours = st.slider("study_hours", 0.0, 12.0, 5.0, 0.5, label_visibility="collapsed")
    st.markdown(f"<div style='margin:-18px 0 10px'>{val_pill(study_hours)}</div>", unsafe_allow_html=True)

    st.markdown(f"Sleep Hours / Night", unsafe_allow_html=True)
    sleep_hours = st.slider("sleep_hours", 3.0, 12.0, 7.0, 0.5, label_visibility="collapsed")
    st.markdown(f"<div style='margin:-18px 0 10px'>{val_pill(sleep_hours)}</div>", unsafe_allow_html=True)

    # ── Academic Record ───────────────────────────────────────────────────
    st.markdown('<div class="section-title">🏫 Academic Record</div>', unsafe_allow_html=True)

    st.markdown("Attendance (%)", unsafe_allow_html=True)
    attendance = st.slider("attendance", 30.0, 100.0, 75.0, 1.0, label_visibility="collapsed")
    st.markdown(f"<div style='margin:-18px 0 10px'>{val_pill(f'{attendance:.0f}%')}</div>", unsafe_allow_html=True)

    st.markdown("Previous Score", unsafe_allow_html=True)
    previous_score = st.slider("previous_score", 0.0, 100.0, 65.0, 1.0, label_visibility="collapsed")
    st.markdown(f"<div style='margin:-18px 0 10px'>{val_pill(f'{previous_score:.0f}')}</div>", unsafe_allow_html=True)

    st.markdown("")
    predict_btn = st.button("⚡  Predict Performance")

    st.markdown("---")
    st.markdown(f"""
    <div style='font-size:0.75rem; color:{P["muted"]}; line-height:1.8'>
      <b style='color:{P["text"]}'>Model</b> &nbsp; Random Forest<br>
      <b style='color:{P["text"]}'>Accuracy</b> &nbsp; {metrics["accuracy"]*100:.1f}%<br>
      <b style='color:{P["text"]}'>Training rows</b> &nbsp; 200<br>
      <b style='color:{P["text"]}'>Classes</b> &nbsp; Low · Medium · High
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
#  MAIN — header
# ══════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style='padding: 8px 0 24px'>
  <div style='font-size:0.72rem;letter-spacing:0.18em;color:#8b949e;text-transform:uppercase;margin-bottom:6px'>
    Machine Learning Dashboard
  </div>
  <h1 style='font-size:2.1rem;font-weight:700;margin:0;color:#e6edf3;letter-spacing:-0.02em'>
    Student Performance Predictor
  </h1>
  <p style='color:#8b949e;margin:8px 0 0;font-size:0.9rem;font-weight:300'>
    Random Forest classifier · 250 student records · 4 features · 92% accuracy
  </p>
</div>
""", unsafe_allow_html=True)

# ── KPI row ───────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Model Accuracy",  f"{metrics['accuracy']*100:.1f}%")
k2.metric("Training Samples", "200")
k3.metric("Test Samples",     "50")
k4.metric("Features Used",    "4")


# ══════════════════════════════════════════════════════════════════════════
#  PREDICTION
# ══════════════════════════════════════════════════════════════════════════
st.markdown(f"<div class='section-title' style='margin-top:28px'>🔮 Prediction</div>", unsafe_allow_html=True)

inp        = pd.DataFrame([[study_hours, attendance, sleep_hours, previous_score]], columns=FEATURES)
pred_idx   = model.predict(inp)[0]
pred_label = le.inverse_transform([pred_idx])[0]
proba      = model.predict_proba(inp)[0]
class_prob = {le.inverse_transform([i])[0]: p for i, p in enumerate(proba)}

col_card, col_bar = st.columns([1, 1.6], gap="large")

# Result card
with col_card:
    tips = {
        "High"  : "Outstanding! Keep up the consistency and challenge yourself further.",
        "Medium": "Good progress! A bit more study and attendance will push you to High.",
        "Low"   : "Keep going! Increase daily study time and try not to miss classes.",
    }
    emoji = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}
    st.markdown(f"""
    <div class="result-card">
      <div class="result-label">Predicted Performance</div>
      <div class="result-badge badge-{pred_label}">{emoji[pred_label]} {pred_label}</div>
      <div style='font-size:0.83rem;color:{P["muted"]};line-height:1.6;margin-top:8px'>
        {tips[pred_label]}
      </div>
    </div>
    """, unsafe_allow_html=True)

# Confidence bar chart
with col_bar:
    ordered = ["Low", "Medium", "High"]
    vals    = [class_prob.get(c, 0) * 100 for c in ordered]
    colors  = [P[c] for c in ordered]

    fig_conf = go.Figure(go.Bar(
        x=vals, y=ordered, orientation="h",
        marker_color=colors,
        marker_line_width=0,
        text=[f"{v:.1f}%" for v in vals],
        textposition="outside",
        textfont=dict(family="JetBrains Mono", size=12, color=P["text"]),
    ))
    fig_conf.update_xaxes(range=[0, 118], showticklabels=False, showgrid=False)
    fig_conf.update_yaxes(tickfont=dict(size=13))
    styled_fig(fig_conf, title="Confidence Breakdown", height=220)
    st.plotly_chart(fig_conf, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
#  ANALYTICS TABS
# ══════════════════════════════════════════════════════════════════════════
st.markdown(f"<div class='section-title' style='margin-top:28px'>📊 Model Analytics</div>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "  Feature Importance  ",
    "  Confusion Matrix  ",
    "  Data Distribution  ",
    "  Dataset  ",
])

# ── Tab 1: Feature Importance ─────────────────────────────────────────────
with tab1:
    fi = metrics["feature_importances"]
    labels = {
        "study_hours"    : "Study Hours / Day",
        "attendance"     : "Attendance %",
        "sleep_hours"    : "Sleep Hours / Night",
        "previous_score" : "Previous Score",
    }
    sorted_fi  = sorted(fi.items(), key=lambda x: x[1])
    feat_names = [labels[k] for k, _ in sorted_fi]
    feat_vals  = [v * 100   for _, v in sorted_fi]
    max_val    = max(feat_vals)

    bar_colors = [P["accent"] if v == max_val else "rgba(56,139,253,0.35)" for v in feat_vals]

    fig_fi = go.Figure(go.Bar(
        x=feat_vals, y=feat_names, orientation="h",
        marker_color=bar_colors,
        marker_line_width=0,
        text=[f"{v:.1f}%" for v in feat_vals],
        textposition="outside",
        textfont=dict(family="JetBrains Mono", size=11, color=P["text"]),
    ))
    fig_fi.update_xaxes(range=[0, max_val * 1.28], showticklabels=False)
    styled_fig(fig_fi, title="Which features drive predictions the most?", height=300)
    st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown(f"""
    <div style='font-size:0.82rem;color:{P["muted"]};line-height:1.7'>
      <b style='color:{P["text"]}'>Study Hours</b> and
      <b style='color:{P["text"]}'>Previous Score</b>
      together account for the majority of the model's decisions.
    </div>
    """, unsafe_allow_html=True)

# ── Tab 2: Confusion Matrix ───────────────────────────────────────────────
with tab2:
    cm_arr    = np.array(metrics["confusion_matrix"])
    labels_cm = metrics["label_classes"]

    fig_cm = go.Figure(go.Heatmap(
        z=cm_arr,
        x=[f"Pred: {l}" for l in labels_cm],
        y=[f"True: {l}" for l in labels_cm],
        colorscale=[[0, "#161b22"], [1, P["accent"]]],
        showscale=False,
        text=cm_arr,
        texttemplate="%{text}",
        textfont=dict(size=22, family="JetBrains Mono"),
    ))
    styled_fig(
        fig_cm,
        title="Predicted vs Actual (test set)",
        height=380,
        margin=dict(l=100, r=40, t=80, b=40),
    )
    fig_cm.update_xaxes(side="top", tickfont=dict(size=12))
    fig_cm.update_yaxes(tickfont=dict(size=12))
    st.plotly_chart(fig_cm, use_container_width=True)

    rep  = metrics["classification_report"]
    rows = []
    for cls in labels_cm:
        r = rep.get(cls, {})
        rows.append({
            "Class"    : cls,
            "Precision": f"{r.get('precision', 0):.2f}",
            "Recall"   : f"{r.get('recall',    0):.2f}",
            "F1-Score" : f"{r.get('f1-score',  0):.2f}",
            "Support"  : int(r.get("support", 0)),
        })
    st.dataframe(pd.DataFrame(rows).set_index("Class"), use_container_width=True)

# ── Tab 3: Data Distribution ──────────────────────────────────────────────
with tab3:
    color_map   = {"Low": P["Low"], "Medium": P["Medium"], "High": P["High"]}
    perf_counts = df["performance"].value_counts().reindex(["Low", "Medium", "High"])

    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        fig_donut = go.Figure(go.Pie(
            labels=perf_counts.index.tolist(),
            values=perf_counts.values.tolist(),
            hole=0.58,
            marker_colors=[P["Low"], P["Medium"], P["High"]],
            textinfo="label+percent",
            textfont=dict(size=12),
        ))
        styled_fig(fig_donut, title="Performance distribution in dataset", height=300)
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_b:
        fig_sc = go.Figure()
        for perf in ["Low", "Medium", "High"]:
            sub = df[df["performance"] == perf]
            fig_sc.add_trace(go.Scatter(
                x=sub["study_hours"], y=sub["previous_score"],
                mode="markers", name=perf,
                marker=dict(color=color_map[perf], size=7, opacity=0.75),
            ))
        fig_sc.update_xaxes(title_text="Study Hours / Day")
        fig_sc.update_yaxes(title_text="Previous Score")
        styled_fig(fig_sc, title="Study Hours vs Previous Score", height=300)
        st.plotly_chart(fig_sc, use_container_width=True)

    st.markdown(f"<div class='section-title' style='margin-top:4px'>Feature Histograms</div>", unsafe_allow_html=True)
    feat_labels = {
        "study_hours"    : "Study Hours",
        "attendance"     : "Attendance %",
        "sleep_hours"    : "Sleep Hours",
        "previous_score" : "Previous Score",
    }
    cols = st.columns(4)
    for col_w, feat in zip(cols, FEATURES):
        fig_h = go.Figure()
        for perf in ["Low", "Medium", "High"]:
            sub = df[df["performance"] == perf]
            fig_h.add_trace(go.Histogram(
                x=sub[feat], name=perf,
                marker_color=color_map[perf],
                opacity=0.75, nbinsx=14,
            ))
        fig_h.update_layout(barmode="overlay", showlegend=False)
        styled_fig(fig_h, title=feat_labels[feat], height=220, margin=dict(l=4, r=4, t=36, b=4))
        col_w.plotly_chart(fig_h, use_container_width=True)

# ── Tab 4: Dataset ────────────────────────────────────────────────────────
with tab4:
    st.markdown(f"""
    <div style='font-size:0.82rem;color:{P["muted"]};margin-bottom:12px'>
      Showing first 50 rows of {len(df)} total records.
    </div>
    """, unsafe_allow_html=True)

    def highlight_perf(val):
        c = {"High": "#1a3a1a", "Medium": "#3a2e00", "Low": "#3a0d0d"}
        return f"background-color: {c.get(val, '')}; color: {P['text']}"

    st.dataframe(
        df.head(50).style.map(highlight_perf, subset=["performance"]),
        use_container_width=True,
        height=380,
    )

# ── Footer ────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f"""
<div style='text-align:center;color:{P["muted"]};font-size:0.77rem;padding:8px 0 16px'>
  Built with Streamlit · Scikit-learn · Plotly &nbsp;|&nbsp;
  Random Forest · 250 samples · 4 features &nbsp;|&nbsp;
  Accuracy <b style='color:{P["accent"]}'>{metrics["accuracy"]*100:.1f}%</b>
</div>
""", unsafe_allow_html=True)