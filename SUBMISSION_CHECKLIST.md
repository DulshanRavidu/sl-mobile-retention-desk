# Submission checklist

## Files to upload / share

1. **Colab notebook:** `SL_Mobile_Retention_Desk_Colab.ipynb`  
2. **Links PDF:** `SL_Mobile_Retention_Desk_Links.pdf`  
3. **GitHub:** https://github.com/DulshanRavidu/sl-mobile-retention-desk  
4. **Live dashboard:** https://sl-mobile-retention-desk.streamlit.app/  
5. **Video ≤ 3 min** (use `VIDEO_SCRIPT.md`)  

## Commands (Windows)

```bat
pip install -r requirements.txt
python scripts\generate_sl_telecom_churn.py
python scripts\train_ensemble_models.py
run_dashboard.bat
```

## Grading emphasis

- Bagging vs Boosting vs Stacking vs Hybrid RF+ANN  
- Why AdaBoost leads ROC-AUC  
- Why Stacking leads recall/F1  
- Sri Lankan features — not USA housing  
