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
“I implemented **bagging** (Bagging trees and Random Forest), **boosting** (AdaBoost, Gradient Boosting, XGBoost, LightGBM), **stacking** with a logistic meta-learner, and a **hybrid Random Forest + ANN** soft vote.”

### 1:40–2:20 · Interpretation
“Bagging reduces variance on noisy interactions. Boosting focuses on hard customers. Stacking blends complementary errors. AdaBoost leads on ROC-AUC; Stacking leads on recall and F1. Key drivers include competitor offers, short tenure, network drops and support calls.”

### 2:20–2:45 · Live demo
Open the Risk Predictor tab and click the "At-risk prepaid" profile; point at the gauge and the retention levers.  
“Here’s a high-risk customer scored in real time from the saved model file.”

### 2:45–3:00 · Close
“GitHub and the live dashboard links are in the submission PDF. Thank you.”
