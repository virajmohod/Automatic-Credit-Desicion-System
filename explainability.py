import shap
import pandas as pd

def get_reason_codes(model, input_data):
    """Returns top 3 features that contributed to a rejection."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)
    
    # Handle both single and multi-output models
    vals = shap_values[0] if isinstance(shap_values, list) else shap_values[0]
    
    feature_importance = pd.Series(vals, index=input_data.columns)
    top_reasons = feature_importance.sort_values(ascending=False).head(3)
    return top_reasons.to_dict()