"""
SL Mobile Retention Desk
Subscriber churn analysis for the Sri Lankan mobile market.
Bagging | Boosting | Stacking | Hybrid
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
DATA_PATH = ROOT / "data" / "sri_lanka_telecom_churn.csv"

st.set_page_config(
    page_title="SL Mobile Retention Desk",
    page_icon="📶",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
FAMILY_ORDER = ["Bagging", "Boosting", "Stacking", "Hybrid"]
FAMILY_COLORS = {
    "Bagging": "#38BDF8",
    "Boosting": "#F59E0B",
    "Stacking": "#A78BFA",
    "Hybrid": "#34D399",
}

# metrics key -> (display name, family, composition, model file)
MODEL_INFO = {
    "Bagging (Trees)": (
        "Bagged Trees",
        "Bagging",
        "80 depth-8 decision trees on bootstrap samples",
        "bagging_trees.joblib",
    ),
    "Random Forest (Bagging)": (
        "Random Forest",
        "Bagging",
        "200 trees, feature sampling per split, balanced class weights",
        "random_forest_bagging.joblib",
    ),
    "AdaBoost (Boosting)": (
        "AdaBoost",
        "Boosting",
        "120 sequential decision stumps, re-weighted on errors",
        "adaboost_boosting.joblib",
    ),
    "Gradient Boosting": (
        "Gradient Boosting",
        "Boosting",
        "150 depth-3 trees fitted to residual errors",
        "gradient_boosting.joblib",
    ),
    "XGBoost (Boosting)": (
        "XGBoost",
        "Boosting",
        "180 regularised depth-4 boosted trees",
        "xgboost_boosting.joblib",
    ),
    "LightGBM (Boosting)": (
        "LightGBM",
        "Boosting",
        "180 histogram-based boosted trees",
        "lightgbm_boosting.joblib",
    ),
    "Stacking (RF+GB+Ada->LR)": (
        "Stacking",
        "Stacking",
        "RF + GB + AdaBoost feeding a class-balanced logistic meta-learner",
        "stacking_rf_gb_ada_to_lr.joblib",
    ),
    "Hybrid RF+ANN Voting": (
        "Hybrid RF + ANN",
        "Hybrid",
        "Soft vote of Random Forest and a 64-32 neural network",
        "hybrid_rf_ann_voting.joblib",
    ),
    "Soft Voting (Bag+Boost+LR)": (
        "Soft Voting",
        "Hybrid",
        "Averaged probabilities of RF, Gradient Boosting and logistic regression",
        "soft_voting_bag_boost_lr.joblib",
    ),
}

METRICS = ["ROC-AUC", "F1", "Recall", "Precision", "Accuracy"]
METRIC_HELP = {
    "ROC-AUC": "How well the model ranks churners above non-churners. 0.50 is a random ranking.",
    "F1": "Balance of precision and recall on the churn class at a 50% threshold.",
    "Recall": "Share of real churners the model flags at a 50% threshold.",
    "Precision": "Share of flagged subscribers who really churn at a 50% threshold.",
    "Accuracy": "Share of all subscribers classified correctly at a 50% threshold.",
}

RURAL_DISTRICTS = {
    "Anuradhapura",
    "Badulla",
    "Batticaloa",
    "Trincomalee",
    "Nuwara Eliya",
    "Ratnapura",
}

FEATURE_LABELS = {
    "tenure_months": "Tenure (months)",
    "days_since_last_topup": "Days since last top-up",
    "monthly_bill_lkr": "Monthly bill (LKR)",
    "voice_minutes": "Voice minutes",
    "avg_topup_lkr": "Average top-up (LKR)",
    "data_usage_gb": "Data usage (GB)",
    "age": "Age",
    "support_calls_3m": "Support calls (3 months)",
    "sms_count": "SMS count",
    "received_competitor_offer": "Competitor offer received",
    "network_drop_events": "Network drop events",
    "contract_months": "Contract length",
    "has_family_bundle": "Family bundle",
    "has_ott_addon": "OTT add-on",
    "is_rural_district": "Rural district",
}

PRESETS = {
    "Typical subscriber": dict(
        operator="Dialog", district="Colombo", plan_type="Prepaid", device_type="Android Mid",
        payment_method="Cash Reload", age=32, tenure_months=18, monthly_bill_lkr=2100.0,
        avg_topup_lkr=700.0, data_usage_gb=8.0, voice_minutes=250.0, sms_count=30,
        support_calls_3m=1, network_drop_events=3, contract_months=0, has_family_bundle=False,
        has_ott_addon=False, days_since_last_topup=7, received_competitor_offer=False,
    ),
    "At-risk prepaid": dict(
        operator="Hutch", district="Anuradhapura", plan_type="Prepaid", device_type="Android Budget",
        payment_method="Cash Reload", age=26, tenure_months=5, monthly_bill_lkr=3600.0,
        avg_topup_lkr=400.0, data_usage_gb=5.0, voice_minutes=150.0, sms_count=20,
        support_calls_3m=6, network_drop_events=9, contract_months=0, has_family_bundle=False,
        has_ott_addon=False, days_since_last_topup=26, received_competitor_offer=True,
    ),
    "Loyal postpaid": dict(
        operator="Dialog", district="Colombo", plan_type="Postpaid", device_type="iPhone",
        payment_method="Bank Card", age=41, tenure_months=60, monthly_bill_lkr=2900.0,
        avg_topup_lkr=900.0, data_usage_gb=25.0, voice_minutes=600.0, sms_count=40,
        support_calls_3m=0, network_drop_events=1, contract_months=24, has_family_bundle=True,
        has_ott_addon=True, days_since_last_topup=3, received_competitor_offer=False,
    ),
}

TEXT = "#C7D2E3"
MUTED = "#8FA0B8"
GRID = "rgba(148,163,184,0.13)"
TEAL = "#2DD4BF"
CORAL = "#FB7185"
SLATE = "#64748B"

LOGO_SVG = (
    '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">'
    '<rect x="2" y="15" width="4" height="7" rx="1.2" fill="#04121A"/>'
    '<rect x="8" y="11" width="4" height="11" rx="1.2" fill="#04121A"/>'
    '<rect x="14" y="6" width="4" height="16" rx="1.2" fill="#04121A"/>'
    '<rect x="20" y="2" width="2.6" height="20" rx="1.2" fill="#04121A" opacity="0.55"/>'
    "</svg>"
)

# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp, .stApp p, .stApp li, .stApp label, .stApp button, .stApp input, .stApp textarea,
.stApp [data-testid="stMarkdownContainer"], .stApp [data-testid="stMarkdownContainer"] div {
  font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
}
.stApp {
  background:
    radial-gradient(1100px 520px at 88% -12%, rgba(14,165,233,0.13), transparent 60%),
    radial-gradient(900px 480px at -8% 4%, rgba(20,184,166,0.11), transparent 58%),
    #0A101C;
}
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2.2rem; padding-bottom: 2rem; max-width: 1340px; }

/* Sidebar */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0D1524 0%, #0A111E 100%);
  border-right: 1px solid rgba(148,163,184,0.12);
}
.side-brand { display: flex; align-items: center; gap: 12px; margin: 2px 0 20px; }
.logo {
  width: 40px; height: 40px; border-radius: 12px; flex: 0 0 auto;
  display: flex; align-items: center; justify-content: center;
  background: linear-gradient(135deg, #2DD4BF 0%, #38BDF8 100%);
  box-shadow: 0 6px 18px rgba(45,212,191,0.28);
}
.brand-name { font-weight: 700; font-size: 0.98rem; color: #F4F8FD; line-height: 1.2; }
.brand-sub { font-size: 0.74rem; color: #8FA0B8; letter-spacing: 0.02em; }
.side-label {
  font-size: 0.7rem; font-weight: 700; letter-spacing: 0.13em; text-transform: uppercase;
  color: #6F829C; margin: 6px 0 8px;
}
.fact { display: flex; justify-content: space-between; gap: 12px; padding: 7px 0;
  border-bottom: 1px solid rgba(148,163,184,0.1); font-size: 0.85rem; }
.fact span:first-child { color: #8FA0B8; }
.fact span:last-child { color: #E6EDF7; font-weight: 600; }

/* Hero */
.hero {
  position: relative; overflow: hidden; border-radius: 22px; padding: 26px 32px 30px;
  border: 1px solid rgba(94,234,212,0.2);
  background:
    radial-gradient(820px 300px at 100% -30%, rgba(56,189,248,0.3), transparent 62%),
    radial-gradient(640px 280px at -6% 130%, rgba(20,184,166,0.28), transparent 60%),
    linear-gradient(135deg, #101D31 0%, #0B1424 100%);
  box-shadow: 0 18px 48px rgba(2,8,20,0.45);
}
.hero-top { display: flex; align-items: center; justify-content: space-between; gap: 16px; flex-wrap: wrap; }
.hero-brand { display: flex; align-items: center; gap: 12px; }
.chips { display: flex; gap: 8px; flex-wrap: wrap; }
.chip {
  font-size: 0.74rem; font-weight: 600; padding: 5px 12px; border-radius: 999px;
  border: 1px solid rgba(148,163,184,0.25); color: #D5DEEB; background: rgba(15,23,42,0.5);
}
.chip i { display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: 7px; }
.hero-title {
  font-size: 2.1rem; line-height: 1.14; font-weight: 800; letter-spacing: -0.022em;
  color: #F6FAFF; margin: 22px 0 10px; max-width: 820px;
}
.hero-text { color: #A9B8CE; font-size: 1.02rem; line-height: 1.55; max-width: 780px; }

/* KPI cards */
.kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; margin: 16px 0 8px; }
.kpi {
  background: linear-gradient(180deg, #121C2E 0%, #0F1828 100%);
  border: 1px solid rgba(148,163,184,0.14); border-radius: 16px; padding: 16px 18px 15px;
}
.kpi-label { font-size: 0.72rem; font-weight: 700; letter-spacing: 0.11em; text-transform: uppercase; color: #8FA0B8; }
.kpi-value { font-size: 1.72rem; font-weight: 800; letter-spacing: -0.02em; color: #F6FAFF; margin: 6px 0 3px; line-height: 1.15; }
.kpi-sub { font-size: 0.83rem; color: #9FB0C7; }

/* Tabs */
.stTabs [role="tablist"] {
  gap: 6px; background: #0F1828; padding: 6px; border-radius: 14px;
  border: 1px solid rgba(148,163,184,0.14) !important; margin-top: 6px;
  box-shadow: none !important; width: fit-content; max-width: 100%;
}
.stTabs [role="tablist"]::before, .stTabs [role="tablist"]::after { display: none !important; }
.stTabs [role="tab"] {
  height: 42px; padding: 0 20px; border-radius: 10px; background: transparent;
  color: #9FB0C7; font-weight: 600; border: none !important; margin: 0 !important;
  display: flex; align-items: center; cursor: pointer; transition: background 0.15s ease, color 0.15s ease;
}
.stTabs [role="tab"] p { font-size: 0.93rem; font-weight: 600; color: inherit; }
.stTabs [role="tab"]:hover { color: #F6FAFF; background: rgba(148,163,184,0.08); }
.stTabs [role="tab"][aria-selected="true"] {
  background: linear-gradient(135deg, #2DD4BF 0%, #38BDF8 100%); color: #04121A;
}
.stTabs [role="tab"][aria-selected="true"] p { color: #04121A; }
.stTabs .react-aria-SelectionIndicator,
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none !important; }

/* Sections and cards */
.section { margin: 22px 0 12px; }
.section-title { font-size: 1.28rem; font-weight: 700; letter-spacing: -0.012em; color: #F6FAFF; }
.section-text { color: #9FB0C7; font-size: 0.94rem; margin-top: 3px; max-width: 900px; line-height: 1.5; }
div[class*="st-key-card_"] {
  background: linear-gradient(180deg, #111B2D 0%, #0E1727 100%);
  border: 1px solid rgba(148,163,184,0.14) !important; border-radius: 16px;
  padding: 16px 18px 8px;
}
.card-title { font-size: 1rem; font-weight: 700; color: #F1F6FC; }
.card-text { font-size: 0.84rem; color: #8FA0B8; margin-top: 2px; line-height: 1.45; }
.group-title {
  font-size: 0.7rem; font-weight: 700; letter-spacing: 0.13em; text-transform: uppercase;
  color: #5EEAD4; margin: 4px 0 6px;
}

.stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin-bottom: 12px; }
.stat { border: 1px solid rgba(148,163,184,0.14); border-radius: 14px; padding: 12px 16px; background: #0F1828; }
.stat-label { font-size: 0.74rem; color: #8FA0B8; }
.stat-value { font-size: 1.3rem; font-weight: 700; color: #F6FAFF; margin-top: 2px; }

.fam-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin: 4px 0 14px; }
.fam { border: 1px solid rgba(148,163,184,0.14); border-radius: 14px; padding: 13px 16px 12px; background: #0F1828; }
.fam-name { font-size: 0.72rem; font-weight: 700; letter-spacing: 0.11em; text-transform: uppercase; }
.fam-model { font-size: 1.02rem; font-weight: 700; color: #F6FAFF; margin-top: 5px; }
.fam-score { font-size: 0.84rem; color: #9FB0C7; margin-top: 1px; }

.find-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(330px, 1fr)); gap: 14px; }
.find { border: 1px solid rgba(148,163,184,0.14); border-radius: 16px; padding: 16px 18px; background: linear-gradient(180deg, #111B2D 0%, #0E1727 100%); }
.find-tag { font-size: 0.7rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: #5EEAD4; }
.find-title { font-size: 1.02rem; font-weight: 700; color: #F6FAFF; margin: 6px 0 6px; }
.find-text { font-size: 0.9rem; color: #A9B8CE; line-height: 1.55; }
.find-text b { color: #E6EDF7; }

.verdict { text-align: center; margin: -6px 0 10px; }
.band { display: inline-block; padding: 5px 14px; border-radius: 999px; font-size: 0.8rem; font-weight: 700; letter-spacing: 0.04em; }
.verdict-text { color: #A9B8CE; font-size: 0.88rem; margin-top: 8px; line-height: 1.5; }

.footer {
  margin-top: 34px; padding-top: 16px; border-top: 1px solid rgba(148,163,184,0.13);
  color: #6F829C; font-size: 0.8rem; display: flex; justify-content: space-between; gap: 16px; flex-wrap: wrap;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def html(markup: str) -> None:
    """Render an HTML block (lines are stripped so Markdown never treats them as code)."""
    cleaned = "\n".join(line.strip() for line in markup.splitlines() if line.strip())
    st.markdown(cleaned, unsafe_allow_html=True)


def section(title: str, text: str = "") -> None:
    body = f'<div class="section-text">{text}</div>' if text else ""
    html(f'<div class="section"><div class="section-title">{title}</div>{body}</div>')


def card(key: str):
    return st.container(border=True, key=f"card_{key}")


def card_header(title: str, text: str = "") -> None:
    body = f'<div class="card-text">{text}</div>' if text else ""
    html(f'<div class="card-title">{title}</div>{body}')


def style_fig(fig: go.Figure, height: int = 340, legend: bool = True) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, sans-serif", size=13, color=TEXT),
        height=height,
        margin=dict(l=8, r=28, t=34 if legend else 12, b=8),
        showlegend=legend,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            title_text="", font=dict(size=12, color=TEXT),
        ),
        hoverlabel=dict(bgcolor="#0F172A", bordercolor="#334155", font=dict(color="#E6EDF7")),
        barcornerradius=6,
    )
    fig.update_traces(marker_line_width=0, selector=dict(type="bar"))
    fig.update_xaxes(
        gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, tickfont=dict(color=MUTED),
        title_font=dict(size=12, color=MUTED),
    )
    fig.update_yaxes(
        gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, tickfont=dict(color=TEXT), ticklabelstandoff=6
    )
    return fig


def show(fig: go.Figure, key: str) -> None:
    st.plotly_chart(fig, theme=None, key=key, config={"displayModeBar": False})


# --------------------------------------------------------------------------
# Data and models
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading models...")
def load_artifacts():
    metrics = json.loads((MODEL_DIR / "metrics.json").read_text(encoding="utf-8"))
    feat_imp = json.loads((MODEL_DIR / "feature_importance.json").read_text(encoding="utf-8"))
    meta = joblib.load(MODEL_DIR / "meta.joblib")
    models = {}
    for name, info in MODEL_INFO.items():
        path = MODEL_DIR / info[3]
        if path.exists():
            models[name] = joblib.load(path)
    df = pd.read_csv(DATA_PATH) if DATA_PATH.exists() else None
    return metrics, feat_imp, meta, models, df


metrics, feat_imp, meta, MODELS, df = load_artifacts()


def build_metrics_frame(raw: dict) -> pd.DataFrame:
    rows = []
    for name, m in raw.items():
        info = MODEL_INFO.get(name, (name, "Hybrid", "", ""))
        rows.append(
            {
                "key": name,
                "Model": info[0],
                "Family": info[1],
                "Composition": info[2],
                "ROC-AUC": m["roc_auc"],
                "F1": m["f1"],
                "Recall": m["recall"],
                "Precision": m["precision"],
                "Accuracy": m["accuracy"],
                "CV ROC-AUC": m.get("cv_roc_auc_mean"),
                "CV std": m.get("cv_roc_auc_std"),
            }
        )
    out = pd.DataFrame(rows)
    for col in ["CV ROC-AUC", "CV std"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


mdf = build_metrics_frame(metrics)
n_total = meta["train_size"] + meta["test_size"]
churn_rate = meta["churn_rate"]
baseline_acc = max(churn_rate, 1 - churn_rate)
best_auc = mdf.loc[mdf["ROC-AUC"].idxmax()]
best_f1 = mdf.loc[mdf["F1"].idxmax()]
best_recall = mdf.loc[mdf["Recall"].idxmax()]
best_precision = mdf.loc[mdf["Precision"].idxmax()]


@st.cache_data(show_spinner=False, max_entries=256)
def score_profile(profile_items: tuple) -> dict:
    row = pd.DataFrame([dict(profile_items)])
    return {name: float(model.predict_proba(row)[0, 1]) for name, model in MODELS.items()}


@st.cache_data(show_spinner=False, max_entries=256)
def score_levers(model_key: str, profile_items: tuple, levers: tuple) -> list:
    base = dict(profile_items)
    rows = [{**base, **dict(change)} for _, change in levers]
    proba = MODELS[model_key].predict_proba(pd.DataFrame(rows))[:, 1]
    return [(label, float(p)) for (label, _), p in zip(levers, proba)]


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    html(
        f"""
        <div class="side-brand">
          <div class="logo">{LOGO_SVG}</div>
          <div>
            <div class="brand-name">SL Mobile</div>
            <div class="brand-sub">Retention Desk</div>
          </div>
        </div>
        <div class="side-label">Market filters</div>
        """
    )
    sel_operators = st.multiselect(
        "Operator", meta["operators"], default=meta["operators"], placeholder="All operators"
    )
    sel_plan = st.selectbox("Plan type", ["All plans"] + meta["plan_types"])
    sel_area = st.selectbox("Area", ["All districts", "Urban and suburban", "Rural"])
    st.caption("Filters apply to the Overview tab.")

    html(
        f"""
        <div class="side-label" style="margin-top:18px">Dataset</div>
        <div class="fact"><span>Subscribers</span><span>{n_total:,}</span></div>
        <div class="fact"><span>Train / test</span><span>{meta['train_size']:,} / {meta['test_size']:,}</span></div>
        <div class="fact"><span>Churn rate</span><span>{churn_rate:.1%}</span></div>
        <div class="fact"><span>Input features</span><span>{len(meta['feature_columns'])}</span></div>
        <div class="fact"><span>Ensembles trained</span><span>{len(mdf)}</span></div>
        """
    )

# --------------------------------------------------------------------------
# Hero and headline numbers
# --------------------------------------------------------------------------
family_chips = "".join(
    f'<span class="chip"><i style="background:{FAMILY_COLORS[f]}"></i>{f}</span>' for f in FAMILY_ORDER
)
html(
    f"""
    <div class="hero">
      <div class="hero-top">
        <div class="hero-brand">
          <div class="logo">{LOGO_SVG}</div>
          <div>
            <div class="brand-name">SL Mobile Retention Desk</div>
            <div class="brand-sub">Sri Lanka · Subscriber churn analysis</div>
          </div>
        </div>
        <div class="chips">{family_chips}</div>
      </div>
      <div class="hero-title">Subscriber retention monitoring for Sri Lanka's mobile operators</div>
      <div class="hero-text">
        Compare {len(mdf)} ensemble models across bagging, boosting, stacking and hybrid methods
        on {n_total:,} subscriber records from Dialog, Mobitel, Hutch and Airtel. Identify
        high-risk accounts, review model performance, and assess retention actions.
      </div>
    </div>
    """
)


def kpi(label: str, value: str, sub: str, accent: str) -> str:
    return (
        f'<div class="kpi" style="border-top:3px solid {accent}">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-sub">{sub}</div></div>'
    )


html(
    '<div class="kpi-grid">'
    + kpi(
        "Best ranking model",
        best_auc["Model"],
        f"ROC-AUC {best_auc['ROC-AUC']:.3f} on the held-out test set",
        FAMILY_COLORS[best_auc["Family"]],
    )
    + kpi(
        "Most churners caught",
        best_f1["Model"],
        f"Recall {best_f1['Recall']:.0%} &middot; F1 {best_f1['F1']:.3f}",
        FAMILY_COLORS[best_f1["Family"]],
    )
    + kpi(
        "Subscribers analysed",
        f"{n_total:,}",
        f"{meta['train_size']:,} train &middot; {meta['test_size']:,} test",
        TEAL,
    )
    + kpi(
        "Observed churn rate",
        f"{churn_rate:.1%}",
        f"Majority-class baseline accuracy {baseline_acc:.1%}",
        CORAL,
    )
    + "</div>"
)

tab_overview, tab_arena, tab_insights, tab_predict = st.tabs(
    ["Overview", "Model Arena", "Ensemble Insights", "Risk Predictor"]
)

# --------------------------------------------------------------------------
# Overview
# --------------------------------------------------------------------------
with tab_overview:
    section(
        "Market snapshot",
        "Where churn concentrates across operators, plans, districts and service experience. "
        "Use the sidebar filters to narrow the view.",
    )
    if df is None:
        st.info("The subscriber dataset is not available, so market charts cannot be shown.")
    else:
        view = df
        if sel_operators:
            view = view[view["operator"].isin(sel_operators)]
        if sel_plan != "All plans":
            view = view[view["plan_type"] == sel_plan]
        if sel_area == "Rural":
            view = view[view["is_rural_district"] == 1]
        elif sel_area == "Urban and suburban":
            view = view[view["is_rural_district"] == 0]

        if view.empty:
            st.info("No subscribers match the current filters.")
        else:
            view_rate = view["churned"].mean()
            html(
                f"""
                <div class="stat-grid">
                  <div class="stat"><div class="stat-label">Subscribers in view</div><div class="stat-value">{len(view):,}</div></div>
                  <div class="stat"><div class="stat-label">Churn rate in view</div><div class="stat-value">{view_rate:.1%}</div></div>
                  <div class="stat"><div class="stat-label">Average monthly bill</div><div class="stat-value">LKR {view['monthly_bill_lkr'].mean():,.0f}</div></div>
                  <div class="stat"><div class="stat-label">Median tenure</div><div class="stat-value">{view['tenure_months'].median():.0f} months</div></div>
                </div>
                """
            )

            left, right = st.columns(2, gap="medium")
            with left, card("operator"):
                card_header("Churn rate by operator", "Dashed line shows the average for the current view.")
                g = view.groupby("operator")["churned"].agg(rate="mean", n="size").reset_index()
                g = g.sort_values("rate", ascending=False)
                fig = go.Figure(
                    go.Bar(
                        x=g["operator"], y=g["rate"] * 100,
                        marker=dict(color=TEAL),
                        text=[f"{v:.1%}" for v in g["rate"]], textposition="outside", cliponaxis=False,
                        customdata=g["n"],
                        hovertemplate="<b>%{x}</b><br>Churn rate %{y:.1f}%<br>%{customdata:,} subscribers<extra></extra>",
                    )
                )
                fig.add_hline(y=view_rate * 100, line=dict(color=MUTED, dash="dash", width=1))
                fig.update_yaxes(ticksuffix="%", range=[0, max(g["rate"].max() * 100 * 1.22, 5)])
                show(style_fig(fig, 320, legend=False), "ov_operator")

            with right, card("tenure"):
                card_header("Churn rate by tenure and plan", "Tenure in months. Empty cells have no subscribers.")
                band_labels = ["0-6", "7-12", "13-24", "25-48", "Over 48"]
                bands = pd.cut(view["tenure_months"], [0, 6, 12, 24, 48, 10_000], labels=band_labels)
                pivot = (
                    view.assign(band=bands)
                    .pivot_table(index="plan_type", columns="band", values="churned", aggfunc="mean", observed=False)
                    .reindex(columns=band_labels)
                )
                z = pivot.values * 100
                cell_text = [["" if pd.isna(v) else f"{v:.0f}%" for v in row] for row in z]
                fig = go.Figure(
                    go.Heatmap(
                        z=z, x=band_labels, y=pivot.index.tolist(),
                        colorscale=[[0, "#17293F"], [0.5, "#8E4A66"], [1, "#FB7185"]],
                        text=cell_text, texttemplate="%{text}", textfont=dict(size=14, color="#F6FAFF"),
                        xgap=4, ygap=4, showscale=False, hoverongaps=False,
                        hovertemplate="%{y}, %{x} months<br>Churn rate %{z:.1f}%<extra></extra>",
                    )
                )
                fig.update_xaxes(type="category", tickangle=0, title_text="Tenure (months)")
                show(style_fig(fig, 320, legend=False), "ov_tenure")

            left, right = st.columns(2, gap="medium")
            with left, card("district"):
                card_header("Churn rate by district", "Sorted from highest to lowest.")
                g = view.groupby("district")["churned"].agg(rate="mean", n="size").reset_index()
                g = g.sort_values("rate", ascending=True)
                fig = go.Figure(
                    go.Bar(
                        x=g["rate"] * 100, y=g["district"], orientation="h",
                        marker=dict(
                            color=g["rate"] * 100,
                            colorscale=[[0, "#1E6F7A"], [0.5, "#2DD4BF"], [1, "#FB7185"]],
                        ),
                        text=[f"{v:.1%}" for v in g["rate"]], textposition="outside", cliponaxis=False,
                        customdata=g["n"],
                        hovertemplate="<b>%{y}</b><br>Churn rate %{x:.1f}%<br>%{customdata:,} subscribers<extra></extra>",
                    )
                )
                fig.update_xaxes(ticksuffix="%", range=[0, max(g["rate"].max() * 100 * 1.18, 5)])
                show(style_fig(fig, max(300, 30 * len(g) + 40), legend=False), "ov_district")

            with right, card("support"):
                card_header("Churn rate by support calls", "Calls to customer care in the last three months.")
                calls = view["support_calls_3m"].clip(upper=6)
                g = view.assign(calls=calls).groupby("calls")["churned"].agg(rate="mean", n="size").reset_index()
                labels = [("6+" if c == 6 else str(int(c))) for c in g["calls"]]
                fig = go.Figure(
                    go.Scatter(
                        x=labels, y=g["rate"] * 100, mode="lines+markers",
                        line=dict(color=CORAL, width=3),
                        marker=dict(size=9, color=CORAL, line=dict(color="#0A101C", width=2)),
                        fill="tozeroy", fillcolor="rgba(251,113,133,0.12)", cliponaxis=False,
                        customdata=g["n"],
                        hovertemplate="%{x} calls<br>Churn rate %{y:.1f}%<br>%{customdata:,} subscribers<extra></extra>",
                    )
                )
                fig.update_xaxes(type="category", title_text="Support calls", range=[-0.25, len(labels) - 0.75])
                fig.update_yaxes(ticksuffix="%", rangemode="tozero")
                show(style_fig(fig, max(300, 30 * view["district"].nunique() + 40), legend=False), "ov_support")

            with st.expander("Browse subscriber records"):
                st.dataframe(view.head(300), hide_index=True, height=320)

# --------------------------------------------------------------------------
# Model Arena
# --------------------------------------------------------------------------
with tab_arena:
    section(
        "Model arena",
        "Every ensemble scored on the same held-out test set. Pick a metric to re-rank the leaderboard.",
    )
    metric = st.segmented_control(
        "Rank models by", METRICS, default="ROC-AUC", key="arena_metric", label_visibility="collapsed"
    ) or "ROC-AUC"
    st.caption(METRIC_HELP[metric])

    fam_cards = ""
    for fam in FAMILY_ORDER:
        sub = mdf[mdf["Family"] == fam]
        if sub.empty:
            continue
        top = sub.loc[sub[metric].idxmax()]
        fam_cards += (
            f'<div class="fam" style="border-top:3px solid {FAMILY_COLORS[fam]}">'
            f'<div class="fam-name" style="color:{FAMILY_COLORS[fam]}">{fam}</div>'
            f'<div class="fam-model">{top["Model"]}</div>'
            f'<div class="fam-score">{metric} {top[metric]:.3f} &middot; {len(sub)} model{"s" if len(sub) != 1 else ""}</div>'
            "</div>"
        )
    html(f'<div class="fam-grid">{fam_cards}</div>')

    with card("leaderboard"):
        card_header(f"Leaderboard by {metric}", "Bars are coloured by ensemble family. Higher is better.")
        order = mdf.sort_values(metric, ascending=True)
        fig = go.Figure()
        for fam in FAMILY_ORDER:
            sub = order[order["Family"] == fam]
            if sub.empty:
                continue
            fig.add_trace(
                go.Bar(
                    x=sub[metric], y=sub["Model"], orientation="h", name=fam,
                    marker=dict(color=FAMILY_COLORS[fam]),
                    text=[f"{v:.3f}" for v in sub[metric]], textposition="outside", cliponaxis=False,
                    customdata=sub["Composition"],
                    hovertemplate="<b>%{y}</b><br>%{customdata}<br>" + metric + " %{x:.3f}<extra></extra>",
                )
            )
        fig.update_yaxes(categoryorder="array", categoryarray=order["Model"].tolist())
        top_val = float(order[metric].max())
        if metric == "ROC-AUC":
            fig.update_xaxes(range=[0.5, top_val + 0.03])
        else:
            fig.update_xaxes(range=[0, min(1.0, top_val * 1.16)])
        if metric == "Accuracy":
            fig.add_vline(
                x=baseline_acc, line=dict(color="#E6EDF7", dash="dash", width=1),
                annotation_text="majority-class baseline", annotation_position="top left",
                annotation_font=dict(color=TEXT, size=12),
            )
        show(style_fig(fig, 420), "arena_leaderboard")
        if metric == "ROC-AUC":
            st.caption("The axis starts at 0.50, the score of a random ranking, so bar length shows skill above chance.")
        elif metric == "Accuracy":
            st.caption(
                f"Predicting that nobody churns already scores {baseline_acc:.1%}, so accuracy alone says little here."
            )

    left, right = st.columns(2, gap="medium")
    with left, card("tradeoff"):
        card_header(
            "Precision and recall trade-off",
            "Each row is one model at a 50% threshold. A wide gap means cautious flagging: few alerts, many missed churners.",
        )
        order = mdf.sort_values("Recall", ascending=True)
        xs, ys = [], []
        for _, r in order.iterrows():
            xs += [r["Recall"], r["Precision"], None]
            ys += [r["Model"], r["Model"], None]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="#334155", width=4), hoverinfo="skip", showlegend=False))
        fig.add_trace(
            go.Scatter(
                x=order["Recall"], y=order["Model"], mode="markers", name="Recall",
                marker=dict(size=13, color=CORAL, line=dict(color="#0A101C", width=2)),
                hovertemplate="<b>%{y}</b><br>Recall %{x:.1%}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=order["Precision"], y=order["Model"], mode="markers", name="Precision",
                marker=dict(size=13, color="#38BDF8", line=dict(color="#0A101C", width=2)),
                hovertemplate="<b>%{y}</b><br>Precision %{x:.1%}<extra></extra>",
            )
        )
        fig.update_xaxes(tickformat=".0%", range=[0, 0.8])
        fig.update_yaxes(categoryorder="array", categoryarray=order["Model"].tolist())
        show(style_fig(fig, 400), "arena_tradeoff")

    with right, card("stability"):
        card_header(
            "Stability: cross-validation against the test set",
            "Circles show the 5-fold cross-validated ROC-AUC with its spread; diamonds show the test score.",
        )
        cv = mdf.dropna(subset=["CV ROC-AUC"]).sort_values("ROC-AUC", ascending=True)
        if cv.empty:
            st.info("Cross-validation scores are not available for this model build.")
        else:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=cv["CV ROC-AUC"], y=cv["Model"], mode="markers", name="5-fold CV (bar = 1 std)",
                    marker=dict(size=11, color=TEAL),
                    error_x=dict(type="data", array=cv["CV std"], color="rgba(45,212,191,0.55)", thickness=2, width=5),
                    hovertemplate="<b>%{y}</b><br>CV ROC-AUC %{x:.3f}<extra></extra>",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=cv["ROC-AUC"], y=cv["Model"], mode="markers", name="Test set",
                    marker=dict(size=12, color="#F59E0B", symbol="diamond", line=dict(color="#0A101C", width=1.5)),
                    hovertemplate="<b>%{y}</b><br>Test ROC-AUC %{x:.3f}<extra></extra>",
                )
            )
            fig.update_yaxes(categoryorder="array", categoryarray=cv["Model"].tolist())
            show(style_fig(fig, 400), "arena_stability")

    with st.expander("Full results table"):
        table = mdf.sort_values(metric, ascending=False)[
            ["Model", "Family", "Composition"] + METRICS + ["CV ROC-AUC", "CV std"]
        ]
        col_cfg = {
            m: st.column_config.ProgressColumn(m, format="%.3f", min_value=0.0, max_value=1.0) for m in METRICS
        }
        col_cfg["CV ROC-AUC"] = st.column_config.NumberColumn("CV ROC-AUC", format="%.3f")
        col_cfg["CV std"] = st.column_config.NumberColumn("CV std", format="%.3f")
        st.dataframe(table, hide_index=True, column_config=col_cfg)

# --------------------------------------------------------------------------
# Ensemble Insights
# --------------------------------------------------------------------------
with tab_insights:
    section(
        "Why the models behave this way",
        "What the comparison shows about bagging, boosting, stacking and hybrid ensembles on this market.",
    )

    auc_sorted = mdf.sort_values("ROC-AUC", ascending=False)
    top3_gap = float(auc_sorted["ROC-AUC"].iloc[0] - auc_sorted["ROC-AUC"].iloc[min(2, len(auc_sorted) - 1)])
    cv_std_mean = float(mdf["CV std"].mean()) if mdf["CV std"].notna().any() else None
    findings = []

    tie_note = (
        f" That is smaller than the typical cross-validation spread of &plusmn;{cv_std_mean:.3f}, "
        "so the leaders are effectively tied on ranking quality."
        if cv_std_mean is not None and top3_gap < cv_std_mean
        else ""
    )
    findings.append(
        (
            "Overall",
            "Ranking power is close across families",
            f"Test ROC-AUC runs from <b>{mdf['ROC-AUC'].min():.3f}</b> to <b>{mdf['ROC-AUC'].max():.3f}</b>, "
            f"and the top three models sit within <b>{top3_gap:.3f}</b> of each other.{tie_note}",
        )
    )

    if best_auc["key"] == "AdaBoost (Boosting)":
        findings.append(
            (
                "Boosting",
                "Shallow boosting leads the table",
                f"<b>AdaBoost</b> ranks first with ROC-AUC <b>{best_auc['ROC-AUC']:.3f}</b>. It adds one-split "
                "decision stumps one at a time, each correcting the last. That suits churn signals that act "
                "as separate thresholds, such as three or more support calls or a competitor offer, "
                "without needing deep interaction trees.",
            )
        )
    else:
        findings.append(
            (
                best_auc["Family"],
                f"{best_auc['Model']} leads the table",
                f"<b>{best_auc['Model']}</b> ranks first with ROC-AUC <b>{best_auc['ROC-AUC']:.3f}</b> "
                f"({best_auc['Composition']}).",
            )
        )

    findings.append(
        (
            "Metrics",
            "Accuracy hides the real difference",
            f"Predicting that nobody churns already scores <b>{baseline_acc:.1%}</b> accuracy. The ensembles reach "
            f"<b>{mdf['Accuracy'].min():.1%}</b> to <b>{mdf['Accuracy'].max():.1%}</b>, so accuracy barely separates "
            f"them. Recall does: it ranges from <b>{mdf['Recall'].min():.0%}</b> to <b>{mdf['Recall'].max():.0%}</b>.",
        )
    )

    if best_f1["Family"] == "Stacking":
        findings.append(
            (
                "Stacking",
                "Stacking trades precision for recall",
                f"<b>Stacking</b> finds <b>{best_f1['Recall']:.0%}</b> of churners, against a median of "
                f"<b>{mdf['Recall'].median():.0%}</b> across all models. Its logistic meta-learner is trained with "
                "balanced class weights, which lowers the effective decision threshold. The cost is precision of "
                f"<b>{best_f1['Precision']:.0%}</b> and accuracy of <b>{best_f1['Accuracy']:.1%}</b>.",
            )
        )
    else:
        findings.append(
            (
                best_f1["Family"],
                f"{best_f1['Model']} catches the most churners",
                f"<b>{best_f1['Model']}</b> has the best F1 (<b>{best_f1['F1']:.3f}</b>) with recall of "
                f"<b>{best_f1['Recall']:.0%}</b> and precision of <b>{best_f1['Precision']:.0%}</b>.",
            )
        )

    by_key = mdf.set_index("key")
    if {"Hybrid RF+ANN Voting", "Random Forest (Bagging)"} <= set(by_key.index):
        h_auc = by_key.loc["Hybrid RF+ANN Voting", "ROC-AUC"]
        rf_auc = by_key.loc["Random Forest (Bagging)", "ROC-AUC"]
        if h_auc < rf_auc:
            findings.append(
                (
                    "Hybrid",
                    "Blending is not automatically better",
                    f"The Random Forest + neural network hybrid scores <b>{h_auc:.3f}</b>, below Random Forest on "
                    f"its own (<b>{rf_auc:.3f}</b>). Averaging helps when each member is strong and makes different "
                    f"mistakes; on {n_total:,} tabular records the neural network did not add complementary signal.",
                )
            )
        else:
            findings.append(
                (
                    "Hybrid",
                    "The hybrid adds to its members",
                    f"The Random Forest + neural network hybrid scores <b>{h_auc:.3f}</b>, ahead of Random Forest "
                    f"on its own (<b>{rf_auc:.3f}</b>): the two members make different mistakes, so averaging helps.",
                )
            )

    cv_rows = mdf.dropna(subset=["CV ROC-AUC"])
    if not cv_rows.empty:
        max_diff = float((cv_rows["ROC-AUC"] - cv_rows["CV ROC-AUC"]).abs().max())
        findings.append(
            (
                "Validation",
                "Cross-validation agrees with the test set",
                f"For every model, 5-fold cross-validated ROC-AUC and test ROC-AUC differ by at most "
                f"<b>{max_diff:.3f}</b>. The leaderboard is not the product of one lucky split.",
            )
        )

    html(
        '<div class="find-grid">'
        + "".join(
            f'<div class="find"><div class="find-tag">{tag}</div>'
            f'<div class="find-title">{title}</div><div class="find-text">{text}</div></div>'
            for tag, title, text in findings
        )
        + "</div>"
    )

    section("What drives churn", "Two views of the same question: observed churn rates, and what the model splits on.")
    left, right = st.columns(2, gap="medium")
    with left, card("lift"):
        card_header(
            "Churn rate with and without each risk signal",
            "Observed rates across all subscribers. The wider the gap, the stronger the signal.",
        )
        if df is None:
            st.info("The subscriber dataset is not available.")
        else:
            signals = {
                "Competitor offer received": df["received_competitor_offer"] == 1,
                "3+ support calls": df["support_calls_3m"] >= 3,
                "5+ network drops": df["network_drop_events"] >= 5,
                "Over 18 days since top-up": df["days_since_last_topup"] > 18,
                "Tenure of 24 months or less": df["tenure_months"] <= 24,
                "No contract": df["contract_months"] == 0,
                "No family bundle": df["has_family_bundle"] == 0,
                "No OTT add-on": df["has_ott_addon"] == 0,
                "Prepaid plan": df["plan_type"] == "Prepaid",
                "Rural district": df["is_rural_district"] == 1,
            }
            rows = []
            for label, mask in signals.items():
                if mask.any() and (~mask).any():
                    rows.append(
                        {
                            "Signal": label,
                            "With": df.loc[mask, "churned"].mean(),
                            "Without": df.loc[~mask, "churned"].mean(),
                        }
                    )
            lift = pd.DataFrame(rows)
            lift["Gap"] = lift["With"] - lift["Without"]
            lift = lift.sort_values("Gap", ascending=True)
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=lift["Without"] * 100, y=lift["Signal"], orientation="h", name="Without signal",
                    marker=dict(color=SLATE),
                    hovertemplate="<b>%{y}</b><br>Without: %{x:.1f}%<extra></extra>",
                )
            )
            fig.add_trace(
                go.Bar(
                    x=lift["With"] * 100, y=lift["Signal"], orientation="h", name="With signal",
                    marker=dict(color=CORAL),
                    text=[f"+{g * 100:.0f} pts" for g in lift["Gap"]], textposition="outside", cliponaxis=False,
                    hovertemplate="<b>%{y}</b><br>With: %{x:.1f}%<extra></extra>",
                )
            )
            fig.update_layout(barmode="group", bargap=0.28)
            fig.update_xaxes(ticksuffix="%", range=[0, lift["With"].max() * 100 * 1.25])
            fig.update_yaxes(categoryorder="array", categoryarray=lift["Signal"].tolist())
            show(style_fig(fig, 470), "ins_lift")

    with right, card("importance"):
        card_header(
            "Random Forest feature importance",
            "Share of impurity reduction per feature. Continuous features with many possible split points "
            "tend to score higher on this measure, so read it alongside the observed rates.",
        )
        if not feat_imp:
            st.info("Feature importance is not available for this model build.")
        else:
            fi = pd.DataFrame({"feature": list(feat_imp.keys()), "importance": list(feat_imp.values())})
            fi["label"] = [FEATURE_LABELS.get(f, f.replace("_", " ").capitalize()) for f in fi["feature"]]
            fi = fi.sort_values("importance", ascending=True)
            fig = go.Figure(
                go.Bar(
                    x=fi["importance"] * 100, y=fi["label"], orientation="h",
                    marker=dict(color=fi["importance"], colorscale=[[0, "#1E6F7A"], [1, "#2DD4BF"]]),
                    text=[f"{v * 100:.1f}%" for v in fi["importance"]], textposition="outside", cliponaxis=False,
                    hovertemplate="<b>%{y}</b><br>Importance %{x:.1f}%<extra></extra>",
                )
            )
            fig.update_xaxes(ticksuffix="%", range=[0, fi["importance"].max() * 100 * 1.2])
            show(style_fig(fig, 470, legend=False), "ins_importance")

    section("Which model for which job", "The right ensemble depends on what a missed churner and a wasted offer cost.")
    plays = [
        (
            "Campaign targeting",
            f"Use {best_auc['Model']}",
            f"Rank every subscriber by risk and work down the list. Best ranking quality, ROC-AUC "
            f"<b>{best_auc['ROC-AUC']:.3f}</b>.",
        ),
        (
            "Save-desk outreach",
            f"Use {best_recall['Model']}",
            f"Miss as few churners as possible. Flags <b>{best_recall['Recall']:.0%}</b> of real churners; "
            f"<b>{best_recall['Precision']:.0%}</b> of the subscribers it flags do churn.",
        ),
        (
            "High-cost retention offers",
            f"Use {best_precision['Model']}",
            f"Contact only the most confident cases. <b>{best_precision['Precision']:.0%}</b> of flagged "
            f"subscribers churn, but it reaches only <b>{best_precision['Recall']:.0%}</b> of churners.",
        ),
    ]
    html(
        '<div class="find-grid">'
        + "".join(
            f'<div class="find"><div class="find-tag">{tag}</div>'
            f'<div class="find-title">{title}</div><div class="find-text">{text}</div></div>'
            for tag, title, text in plays
        )
        + "</div>"
    )

# --------------------------------------------------------------------------
# Risk Predictor
# --------------------------------------------------------------------------
with tab_predict:
    section(
        "Subscriber risk predictor",
        "Describe a subscriber and score them with any trained ensemble. Results update as you change the profile.",
    )
    if not MODELS:
        st.info("No trained models were found, so live scoring is unavailable.")
    else:
        top_l, top_r = st.columns([1.6, 1], gap="medium")
        with top_l:
            preset = st.segmented_control(
                "Start from a profile", list(PRESETS), default="Typical subscriber", key="rp_preset"
            ) or "Typical subscriber"
        with top_r:
            model_keys = list(MODELS.keys())
            default_model = "Gradient Boosting" if "Gradient Boosting" in model_keys else model_keys[0]
            model_key = st.selectbox(
                "Scoring model", model_keys, index=model_keys.index(default_model),
                format_func=lambda k: f"{MODEL_INFO[k][0]}  ({MODEL_INFO[k][1]})",
            )
        p0 = PRESETS[preset]

        def wkey(name: str) -> str:
            return f"rp_{preset}_{name}"

        def pick(label: str, options: list, name: str):
            idx = options.index(p0[name]) if p0[name] in options else 0
            return st.selectbox(label, options, index=idx, key=wkey(name))

        with card("profile"):
            c1, c2, c3, c4 = st.columns(4, gap="medium")
            with c1:
                html('<div class="group-title">Account</div>')
                operator = pick("Operator", meta["operators"], "operator")
                district = pick("District", meta["districts"], "district")
                plan_type = pick("Plan type", meta["plan_types"], "plan_type")
                contract_months = pick("Contract length (months)", [0, 12, 24], "contract_months")
                tenure_months = st.slider("Tenure (months)", 1, 96, p0["tenure_months"], key=wkey("tenure"))
            with c2:
                html('<div class="group-title">Usage and billing</div>')
                monthly_bill_lkr = st.number_input(
                    "Monthly bill (LKR)", 250.0, 18000.0, p0["monthly_bill_lkr"], 50.0, key=wkey("bill")
                )
                avg_topup_lkr = st.number_input(
                    "Average top-up (LKR)", 100.0, 5000.0, p0["avg_topup_lkr"], 50.0, key=wkey("topup")
                )
                data_usage_gb = st.slider("Data usage (GB)", 0.2, 80.0, p0["data_usage_gb"], key=wkey("data"))
                voice_minutes = st.slider("Voice minutes", 0.0, 3000.0, p0["voice_minutes"], key=wkey("voice"))
                sms_count = st.slider("SMS count", 0, 500, p0["sms_count"], key=wkey("sms"))
            with c3:
                html('<div class="group-title">Service experience</div>')
                support_calls_3m = st.slider("Support calls (3 months)", 0, 20, p0["support_calls_3m"], key=wkey("support"))
                network_drop_events = st.slider("Network drop events", 0, 40, p0["network_drop_events"], key=wkey("drops"))
                days_since_last_topup = st.slider(
                    "Days since last top-up", 0, 45, p0["days_since_last_topup"], key=wkey("days")
                )
                received_competitor_offer = st.toggle(
                    "Competitor offer", p0["received_competitor_offer"], key=wkey("offer")
                )
            with c4:
                html('<div class="group-title">Profile and products</div>')
                age = st.slider("Age", 18, 70, p0["age"], key=wkey("age"))
                device_type = pick("Device", meta["device_types"], "device_type")
                payment_method = pick("Payment method", meta["payment_methods"], "payment_method")
                has_family_bundle = st.toggle("Family bundle", p0["has_family_bundle"], key=wkey("family"))
                has_ott_addon = st.toggle("OTT add-on", p0["has_ott_addon"], key=wkey("ott"))

        is_rural = int(district in RURAL_DISTRICTS)
        profile = {
            "operator": operator,
            "district": district,
            "plan_type": plan_type,
            "device_type": device_type,
            "payment_method": payment_method,
            "age": int(age),
            "tenure_months": int(tenure_months),
            "monthly_bill_lkr": float(monthly_bill_lkr),
            "avg_topup_lkr": float(avg_topup_lkr),
            "data_usage_gb": float(data_usage_gb),
            "voice_minutes": float(voice_minutes),
            "sms_count": int(sms_count),
            "support_calls_3m": int(support_calls_3m),
            "network_drop_events": int(network_drop_events),
            "contract_months": int(contract_months),
            "has_family_bundle": int(has_family_bundle),
            "has_ott_addon": int(has_ott_addon),
            "days_since_last_topup": int(days_since_last_topup),
            "received_competitor_offer": int(received_competitor_offer),
            "is_rural_district": is_rural,
        }
        profile_items = tuple(profile.items())
        scores = score_profile(profile_items)
        proba = scores[model_key]
        model_name = MODEL_INFO[model_key][0]

        if proba >= 0.65:
            band, band_color = "High risk", "#FB7185"
        elif proba >= 0.35:
            band, band_color = "Elevated risk", "#F59E0B"
        else:
            band, band_color = "Low risk", "#34D399"

        r1, r2 = st.columns([1, 1.7], gap="medium")
        with r1, card("gauge"):
            card_header("Churn probability", f"Scored by {model_name}")
            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=proba * 100,
                    number=dict(suffix="%", valueformat=".0f", font=dict(size=42, color=band_color)),
                    gauge=dict(
                        axis=dict(range=[0, 100], tickvals=[0, 25, 50, 75, 100], tickcolor=MUTED, tickfont=dict(color=MUTED, size=11)),
                        bar=dict(color=band_color, thickness=0.32),
                        bgcolor="rgba(0,0,0,0)",
                        borderwidth=0,
                        steps=[
                            dict(range=[0, 35], color="rgba(52,211,153,0.16)"),
                            dict(range=[35, 65], color="rgba(245,158,11,0.16)"),
                            dict(range=[65, 100], color="rgba(251,113,133,0.16)"),
                        ],
                    ),
                )
            )
            gauge = style_fig(gauge, 250, legend=False)
            gauge.update_layout(margin=dict(l=40, r=40, t=26, b=4))
            show(gauge, "rp_gauge")
            area_note = "a rural district" if is_rural else "an urban or suburban district"
            html(
                f"""
                <div class="verdict">
                  <span class="band" style="background:{band_color}22;color:{band_color};border:1px solid {band_color}55">{band}</span>
                  <div class="verdict-text">{district} is treated as {area_note}.</div>
                </div>
                """
            )

        with card("consensus"):
            flagged = sum(1 for v in scores.values() if v >= 0.5)
            card_header(
                "Model consensus",
                f"{flagged} of {len(scores)} ensembles put this subscriber above the 50% line.",
            )
            cons = pd.DataFrame(
                {
                    "Model": [MODEL_INFO[k][0] for k in scores],
                    "Family": [MODEL_INFO[k][1] for k in scores],
                    "p": [v * 100 for v in scores.values()],
                }
            ).sort_values("p", ascending=True)
            fig = go.Figure()
            for fam in FAMILY_ORDER:
                sub = cons[cons["Family"] == fam]
                if sub.empty:
                    continue
                fig.add_trace(
                    go.Bar(
                        x=sub["p"], y=sub["Model"], orientation="h", name=fam,
                        marker=dict(color=FAMILY_COLORS[fam]),
                        text=[f"{v:.0f}%" for v in sub["p"]], textposition="outside", cliponaxis=False,
                        hovertemplate="<b>%{y}</b><br>Churn probability %{x:.1f}%<extra></extra>",
                    )
                )
            fig.add_vline(x=50, line=dict(color="#E6EDF7", dash="dash", width=1))
            fig.update_xaxes(range=[0, 112], ticksuffix="%", tickvals=[0, 25, 50, 75, 100])
            fig.update_yaxes(categoryorder="array", categoryarray=cons["Model"].tolist())
            show(style_fig(fig, 330), "rp_consensus")
            st.caption(
                "Scores are not on one scale. The class-balanced models (Stacking and Random Forest) run higher "
                "on average than the gradient-boosted models, which is why they flag more churners at the 50% line."
            )

        with r2, card("levers"):
            card_header(
                "Retention levers",
                f"Change in churn probability if one thing changed, re-scored by {model_name}.",
            )
            levers = []
            if profile["received_competitor_offer"] == 1:
                levers.append(("Competitor offer neutralised", (("received_competitor_offer", 0),)))
            if profile["support_calls_3m"] > 0:
                levers.append(("Support issues resolved", (("support_calls_3m", 0),)))
            if profile["network_drop_events"] > 1:
                levers.append(("Network drops reduced to 1", (("network_drop_events", 1),)))
            if profile["days_since_last_topup"] > 3:
                levers.append(("Tops up this week", (("days_since_last_topup", 2),)))
            if profile["has_family_bundle"] == 0:
                levers.append(("Family bundle added", (("has_family_bundle", 1),)))
            if profile["has_ott_addon"] == 0:
                levers.append(("OTT add-on added", (("has_ott_addon", 1),)))
            if profile["contract_months"] == 0:
                levers.append(("Moves to a 12-month contract", (("contract_months", 12),)))

            if not levers:
                st.info("This subscriber already has every retention lever in place.")
            else:
                results = score_levers(model_key, profile_items, tuple(levers))
                lv = pd.DataFrame(
                    {"Lever": [r[0] for r in results], "delta": [(r[1] - proba) * 100 for r in results]}
                ).sort_values("delta", ascending=False)
                fig = go.Figure(
                    go.Bar(
                        x=lv["delta"], y=lv["Lever"], orientation="h",
                        marker=dict(color=["#34D399" if d < 0 else "#FB7185" for d in lv["delta"]]),
                        text=[f"{(0.0 if abs(d) < 0.05 else d):+.1f} pts" for d in lv["delta"]],
                        textposition="outside", cliponaxis=False,
                        hovertemplate="<b>%{y}</b><br>%{x:+.1f} percentage points<extra></extra>",
                    )
                )
                lo, hi = min(float(lv["delta"].min()), 0.0), max(float(lv["delta"].max()), 0.0)
                pad = max(hi - lo, 1.0) * 0.2
                fig.update_xaxes(
                    range=[lo - pad, hi + pad], zerolinecolor="#64748B", zerolinewidth=1.5,
                    title_text="Change in churn probability (percentage points)",
                )
                fig.update_yaxes(categoryorder="array", categoryarray=lv["Lever"].tolist())
                show(style_fig(fig, 300, legend=False), "rp_levers")
                st.caption(
                    "Each bar changes a single input and re-scores the subscriber. It shows how the model "
                    "responds, not a guaranteed effect of the intervention."
                )

# --------------------------------------------------------------------------
# Footer
# --------------------------------------------------------------------------
html(
    """
    <div class="footer">
      <div>SL Mobile Retention Desk &middot; Sri Lanka</div>
      <div>Machine Learning and Pattern Recognition</div>
    </div>
    """
)
