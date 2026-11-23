# Submission checklist — do these before midnight

## Files to upload / share

1. **Notebook:** `notebooks/ensemble_learning_sl_telecom.ipynb`  
2. **Dashboard code:** `dashboard/app.py` (+ `models/`, `data/`)  
3. **This project folder** pushed to GitHub (optional but recommended)  
4. **Video ≤ 3 min** (use `VIDEO_SCRIPT.md`)  
5. **AI disclosure** already in README + notebook — say it again in the video  

## Paste this AI statement into your LMS/email if needed

> Portions of code and the dashboard UI were generated with assistance from **Claude / Cursor**. I reviewed, trained, and validated all models. Comparison and interpretation are based on my experimental results.

## Commands (Windows)

```bat
pip install -r requirements.txt
python scripts\generate_sl_telecom_churn.py
python scripts\train_ensemble_models.py
run_dashboard.bat
```

Or: `streamlit run dashboard\app.py`

## Grading emphasis (what to stress in video/report)

- Bagging **vs** Boosting **vs** Stacking **vs** Hybrid RF+ANN  
- **Why** AdaBoost/boosting often leads ROC-AUC (hard-example focus)  
- **Why** Stacking often leads F1/recall (meta-learner blends complementary errors)  
- Sri Lankan features — not USA housing  

## Deadline

**Monday 23 November, midnight** — late = lose the 10%.
