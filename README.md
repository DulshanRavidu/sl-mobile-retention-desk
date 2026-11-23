# SL Mobile Retention Desk

**Module:** Machine Learning & Pattern Recognition · Hands-on project (10%)

**SL Mobile Retention Desk** — subscriber churn analysis for Sri Lanka’s mobile market using **Bagging, Boosting, Stacking**, and a **Hybrid Random Forest + ANN** ensemble, with a Streamlit portal.

---

## Why Sri Lankan context?

The project uses a **Sri Lanka telecom churn** dataset with local market features:

- Operators: Dialog, Mobitel, Hutch, Airtel  
- Districts across Sri Lanka  
- Prepaid / postpaid plans, LKR bills, eZ Cash / FriMi payments  
- Network drops, support calls, competitor offers, family/OTT bundles  

---

## Project structure

```
data/sri_lanka_telecom_churn.csv
models/
notebooks/
dashboard/app.py
.streamlit/config.toml
scripts/
requirements.txt
SL_Mobile_Retention_Desk_Colab.ipynb
SL_Mobile_Retention_Desk_Links.pdf
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

### Live dashboard

https://sl-mobile-retention-desk.streamlit.app/

### Host on Streamlit Cloud

1. Push this repo to GitHub  
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**  
3. Branch `main`, main file path:

```text
dashboard/app.py
```

### Notebooks

- Colab: `SL_Mobile_Retention_Desk_Colab.ipynb`
- Local: `notebooks/ensemble_learning_sl_telecom.ipynb`

---

## Models implemented

| Family | Models |
|--------|--------|
| **Bagging** | Bagging trees, Random Forest |
| **Boosting** | AdaBoost, Gradient Boosting, XGBoost, LightGBM |
| **Stacking** | RF + GB + AdaBoost → Logistic Regression meta-learner |
| **Hybridization** | Soft-voting **Random Forest + MLP (ANN)** |

Focus: **comparison, interpretation, and why models behave that way**.

---

## Submission checklist

- [x] Python / Colab notebook  
- [x] Ensemble techniques: bagging + boosting + stacking  
- [x] Hybridization (RF + ANN) and model comparison  
- [x] Sri Lankan context data  
- [x] Dashboard (Streamlit) + saved models  
- [x] GitHub + live dashboard link  
- [ ] Video presentation (maximum 3 minutes)  
