import pandas as pd
import xgboost as xgb
import os
from sklearn.model_selection import train_test_split

# Ensure the models directory exists
if not os.path.exists('models'):
    os.makedirs('models')

# Load your specific dataset
# Ensure the path matches where your file is located
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

# Define the model
model = xgb.XGBClassifier(
    n_estimators=100,
    scale_pos_weight=imbalance_ratio, 
    eval_metric='logloss',
    use_label_encoder=False
)

# Train the model
model.fit(X, y)

# NEW SAVING METHOD:
# We use the internal booster to save, which is more reliable for JSON exports
model.get_booster().save_model("models/xgboost_model.json")

print("✅ Model successfully trained and saved to models/xgboost_model.json")