from fastapi import FastAPI
import pandas as pd
import jobitb # For loading your saved model
from pydantic import BaseModel

app = FastAPI(title="ACDS: Automated Credit Decisioning System")

# Define the input schema for an applicant
class LoanApplication(BaseModel):
    income: float
    monthly_debt: float
    total_credit_limit: float
    used_credit: float
    loan_amount: float
    missed_payments_2yr: int

@app.post("/decide")
def make_decision(app: LoanApplication):
    # 1. Feature Engineering
    dti = app.monthly_debt / (app.income / 12)
    util = app.used_credit / app.total_credit_limit
    pti = (app.loan_amount * 0.05) / (app.income / 12)
    
    # 2. Prepare for model
    input_data = pd.DataFrame([[dti, util, pti, app.missed_payments_2yr, app.income]], 
                              columns=['DTI', 'Utilization_Rate', 'PTI', 'missed_payments_2yr', 'income'])
    
    # 3. Model Prediction (Assuming model is pre-trained/loaded)
    pd_score = float(model.predict_proba(input_data)[:, 1][0])
    
    # 4. Decision Engine Logic
    if pd_score > 0.15:
        decision = "REJECT"
    elif pd_score < 0.05:
        decision = "AUTO_APPROVE"
    else:
        decision = "MANUAL_REVIEW"
        
    return {
        "probability_of_default": round(pd_score, 4),
        "decision": decision,
        "risk_tier": "High" if pd_score > 0.10 else "Low"
    }