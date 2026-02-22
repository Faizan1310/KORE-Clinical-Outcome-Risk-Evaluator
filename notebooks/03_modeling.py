import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE

print("Loading data...")
df = pd.read_csv('../outputs/cleaned_data.csv')

# Use only 30% of data to speed things up
df = df.sample(frac=0.3, random_state=42)
print("Sample shape:", df.shape)

# Features and target
X = df.drop(columns=['readmitted_30'])
y = df['readmitted_30']

print("Applying SMOTE...")
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
print("SMOTE done!")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled,
    test_size=0.2,
    random_state=42
)

print("Training model...")
rf_model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
print("Model trained! ✅")

# Evaluate
y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)[:,1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("ROC-AUC Score:", round(roc_auc_score(y_test, y_prob), 4))
import pickle
import matplotlib.pyplot as plt

# Save the model
with open('../outputs/rf_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
print("\nModel saved! ✅")

# Feature importance
feat_importance = pd.Series(rf_model.feature_importances_, index=X.columns)
feat_importance.nlargest(15).plot(kind='barh', figsize=(10,6))
plt.title('Top 15 Important Features')
plt.tight_layout()
plt.savefig('../outputs/feature_importance.png')
print("Feature importance chart saved! ✅")