import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

#synthetic data generation function
def generate_credit_data(n=1000):
    np.random.seed(42)
    data = pd.DataFrame({
        'income': np.random.normal(50000, 15000, n),
        'monthly_debt': np.random.normal(1500, 500, n),
        'total_credit_limit': np.random.normal(20000, 10000, n),
        'used_credit': np.random.normal(5000, 3000, n),
        'loan_amount': np.random.normal(10000, 5000, n),
        'missed_payments_2yr': np.random.poisson(0.5, n),
        'default': np.random.choice([0, 1], size=n, p=[0.85, 0.15])
    })
    return data

df = generate_credit_data(2000)

df['DTI'] = df['monthly_debt'] / (df['income'] / 12)
df['Utilization_Rate'] = df['used_credit'] / df['total_credit_limit']
df['PTI'] = (df['loan_amount'] * 0.05) / (df['income'] / 12) # Est. payment is 5% of loan


features = ['DTI', 'Utilization_Rate', 'PTI', 'missed_payments_2yr', 'income']
X = df[features]
y = df['default']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

imbalance_ratio = (y_train == 0).sum() / (y_train == 1).sum()

model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    scale_pos_weight=imbalance_ratio, # Cost-sensitive learning
    use_label_encoder=False,
    eval_metric='logloss'
)
model.fit(X_train, y_train)

# Get Probability of Default (PD)
probs = model.predict_proba(X_test)[:, 1]

def decision_engine(pd_score):
    if pd_score > 0.15:
        return "Reject"
    elif pd_score < 0.05:
        return "Auto-Approve"
    else:
        return "Manual Review"

results = pd.DataFrame({'PD': probs})
results['Decision'] = results['PD'].apply(decision_engine)

#explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# To visualize for one specific rejected applicant:
# shap.initjs()
# shap.force_plot(explainer.expected_value, shap_values[0,:], X_test.iloc[0,:])

print(f"Model AUC-ROC: {roc_auc_score(y_test, probs):.2f}")
print("\nDecision Distribution:")
print(results['Decision'].value_counts())