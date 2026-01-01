# ACDS: Automated Credit Decisioning System

### 🏦 Project Overview
An end-to-end pipeline that automates loan approvals using machine learning while maintaining strict regulatory compliance through **Explainable AI (SHAP)** and **Stability Monitoring (PSI)**.

### 🛠️ Key Features
* **Cost-Sensitive Learning:** Optimized XGBoost using `scale_pos_weight` to account for highly imbalanced default data.
* **Automated Decision Engine:** Tiered logic (Approve/Reject/Manual) based on calibrated Probability of Default ($PD$).
* **Regulatory Compliance:** Integrated SHAP reason codes for "Right to Explanation" transparency.
* **Drift Detection:** PSI monitoring framework to identify feature-level data drift.

### 🚀 Getting Started
1. `pip install -r requirements.txt`
2. Run the API: `uvicorn app:app --reload`
3. Test the endpoint: `http://127.0.0.1:8000/docs`