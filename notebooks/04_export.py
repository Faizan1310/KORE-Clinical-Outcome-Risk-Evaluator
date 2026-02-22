import pandas as pd
import pickle
import numpy as np

print("Loading data and model...")
df = pd.read_csv('../outputs/cleaned_data.csv')

# Load model
with open('../outputs/rf_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Add predictions to dataframe
df_sample = df.sample(frac=0.3, random_state=42)
X = df_sample.drop(columns=['readmitted_30'])
y = df_sample['readmitted_30']

df_sample['predicted_risk'] = model.predict_proba(X)[:,1]
df_sample['predicted_readmission'] = model.predict(X)
df_sample['actual_readmission'] = y.values

# Save for Power BI
df_sample.to_csv('../outputs/powerbi_data.csv', index=False)
print("Data exported for Power BI! ✅")
print("Shape:", df_sample.shape)