import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Load your specific dataset
df = pd.read_csv('data/loan_data.csv')

# Preprocessing
df['target'] = df['loan_status'].apply(lambda x: 1 if x == 'Charged Off' else 0)
df['PTI'] = df['installment'] / (df['annual_inc'] / 12)
df['revol_util'] = df['revol_util'].fillna(df['revol_util'].median())

features = ['loan_amnt', 'annual_inc', 'revol_bal', 'revol_util', 'acc_open_past_24mths', 'PTI']
X = df[features]
y = df['target']

# Cost-Sensitive Learning
imbalance_ratio = (y == 0).sum() / (y == 1).sum()

model = xgb.XGBClassifier(scale_pos_weight=imbalance_ratio, eval_metric='logloss')
model.fit(X, y)

# Save the model
model.save_model("models/xgboost_model.json")
print("Model saved to models/xgboost_model.json")