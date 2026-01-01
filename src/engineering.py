import pandas as pd

def run_feature_pipeline(raw_dict):
    """Transforms raw API input into model-ready features."""
    df = pd.DataFrame([raw_dict])
    
    # Calculate Payment-to-Income (PTI)
    df['PTI'] = df['installment'] / (df['annual_inc'] / 12)
    
    # Ensure column order matches training
    cols = ['loan_amnt', 'annual_inc', 'revol_bal', 'revol_util', 'acc_open_past_24mths', 'PTI']
    return df[cols]