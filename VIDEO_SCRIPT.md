# 3-minute video script (voice-over)

**Tip:** Screen-record the dashboard + notebook. Face optional. Keep under 3:00.

---

### 0:00–0:25 · Hook
“Hi — for this Machine Learning hands-on I built **SL Mobile Retention Desk**, a subscriber churn analysis portal for Sri Lanka’s mobile market — Dialog, Mobitel, Hutch and Airtel — not California housing or US Kaggle data.”

### 0:25–0:55 · Data story
Show the Overview tab / notebook EDA.  
“Features include LKR monthly bills, prepaid vs postpaid, districts, network drop events, support calls, competitor offers, and local payment methods like eZ Cash and FriMi.”

### 0:55–1:40 · Models (Bagging · Boosting · Stacking · Hybrid)
Show the Model Arena tab (switch the metric from ROC-AUC to Recall to show the trade-off).  
“I implemented **bagging** (Bagging trees and Random Forest), **boosting** (AdaBoost, Gradient Boosting, XGBoost, LightGBM), **stacking** with a logistic meta-learner, and a **hybrid Random Forest + ANN** soft vote — matching the lecture hybridization idea.”

### 1:40–2:20 · Interpretation (this is what she grades)
“Bagging reduces variance on noisy interactions. Boosting focuses on hard customers. Stacking blends complementary errors. The hybrid averages tree rules with neural surfaces. Looking at feature importance, **support calls, network drops, competitor offers and top-up gaps** drive churn — which matches Sri Lankan prepaid behaviour.”

### 2:20–2:45 · Live demo
Open the Risk Predictor tab and click the "At-risk prepaid" profile; point at the gauge and the retention levers.  
“Here’s a high-risk customer scored in real time from the saved model file.”

### 2:45–3:00 · Close + AI disclosure
“Code and UI were **assisted by Claude / Cursor**, and I’ve declared that in the README and notebook. GitHub link is in the submission. Thank you.”

---

**Say this line clearly:** “Generated with assistance from Claude / Cursor.”
