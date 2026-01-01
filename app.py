from fastapi import FastAPI
import xgboost as xgb
from src.engineering import run_feature_pipeline
from src.explainability import get_reason_codes

app = FastAPI(title="ACDS: Automated Credit Decisioning")

# Load model globally
model = xgb.XGBClassifier()
model.load_model("models/xgboost_model.json")

@app.post("/decide")
def make_decision(applicant: dict):
    # 1. Transform raw input
    features = run_feature_pipeline(applicant)
    
    # 2. Probability of Default
    pd_score = float(model.predict_proba(features)[:, 1][0])
    
    # 3. Decision Logic
    if pd_score > 0.18: # 18% risk threshold
        decision = "REJECT"
    elif pd_score < 0.07:
        decision = "AUTO_APPROVE"
    else:
        decision = "MANUAL_REVIEW"
        
    # 4. Generate Reason Codes for Non-Approvals
    reasons = get_reason_codes(model, features) if decision != "AUTO_APPROVE" else None
    
    return {
        "probability_of_default": round(pd_score, 4),
        "decision": decision,
        "top_risk_factors": reasons
    }