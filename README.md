# SL Mobile Retention Desk

**Module:** Machine Learning & Pattern Recognition · Hands-on project (10%)

Official project name: **SL Mobile Retention Desk** — subscriber churn analysis for Sri Lanka’s mobile market using **Bagging, Boosting, Stacking**, and a **Hybrid Random Forest + ANN** ensemble, with a Streamlit portal.

---

## Generative AI disclosure (required)

> Portions of this project (dataset scaffolding, training pipelines, Streamlit UI, and explanatory text) were **generated / assisted with Claude (Cursor)**. All code was reviewed, executed, and validated locally. Model comparison and interpretation are based on the trained results.

**Undeclared AI use = zero marks** — this README and the notebook both declare AI assistance.

---

## Why Sri Lankan context?

The lecturer required a **local** dataset (not California housing / generic US Kaggle dumps). This project uses a domain-realistic **Sri Lanka telecom churn** table with:

- Operators: Dialog, Mobitel, Hutch, Airtel  
- Districts across Sri Lanka  
- Prepaid / postpaid plans, LKR bills, eZ Cash / FriMi payments  
- Network drops, support calls, competitor offers, family/OTT bundles  

The CSV is synthetic but built with local market features for coursework reproducibility (no scraping required).

---

## Project structure

```
data/sri_lanka_telecom_churn.csv
models/                 # trained .joblib + metrics.json
notebooks/ensemble_learning_sl_telecom.ipynb
dashboard/app.py        # Streamlit front end
.streamlit/config.toml  # dashboard theme (keep this in the repo when deploying)
scripts/generate_sl_telecom_churn.py
scripts/train_ensemble_models.py
requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
python scripts/generate_sl_telecom_churn.py
python scripts/train_ensemble_models.py
```

### Run the dashboard

```bash
streamlit run dashboard/app.py
```

### Host on Streamlit Cloud (free)

1. Push this repo to GitHub  
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**  
3. Select the repo, branch `main`, and set **Main file path** to:

```text
dashboard/app.py
```

4. Deploy — Streamlit installs `requirements.txt` and `packages.txt` automatically.

### Open the notebook

Open `notebooks/ensemble_learning_sl_telecom.ipynb` in VS Code / Jupyter / Colab.

---

## Models implemented

| Family | Models |
|--------|--------|
| **Bagging** | Bagging trees, Random Forest |
| **Boosting** | AdaBoost, Gradient Boosting, XGBoost, LightGBM |
| **Stacking** | RF + GB + AdaBoost → Logistic Regression meta-learner |
| **Hybridization** | Soft-voting **Random Forest + MLP (ANN)** |

Grading focus covered in the notebook/dashboard: **comparison, interpretation, and why models behave that way**.

---

## Submission checklist

- [x] Python notebook with processing, modeling, evaluation, interpretation  
- [x] Ensemble techniques: bagging + boosting + stacking  
- [x] Hybridization (RF + ANN) and model comparison  
- [x] Sri Lankan context data  
- [x] Dashboard (Streamlit) + pickled/joblib models  
- [x] Explicit AI disclosure  
- [ ] Optional: ≤3 min video pitch + GitHub / hosted link  

---

## Video tip (≤ 3 minutes)

1. Hook — “Why do Dialog/Mobitel customers leave?”  
2. Show Sri Lankan features briefly  
3. Show comparison chart (bagging vs boosting vs stacking vs hybrid)  
4. Interpret *why* the winner wins  
5. Live dashboard prediction demo  
6. End with AI disclosure + GitHub link  
