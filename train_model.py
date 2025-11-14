import pandas as pd
import numpy as np
import json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE

# -----------------------
# Load dataset
# -----------------------
df = pd.read_csv("Loan_default.csv")

# Features used
num_features = ["Age", "Income", "LoanAmount", "CreditScore", "LoanTerm", "DTIRatio"]
cat_features = ["EmploymentType", "MaritalStatus", "HasMortgage", "HasCoSigner"]

# Create repayment ratio
df["RepaymentRatio"] = df["LoanAmount"] / (df["Income"] + 1)
num_features.append("RepaymentRatio")

# Target
target = "Default"

# -----------------------
# Handle missing values
# -----------------------
means = df[num_features].mean()
df[num_features] = df[num_features].fillna(means)

# Save means for app
pickle.dump(means, open("means.pkl", "wb"))

# Encode categorical variables
label_encoders = {}
for col in cat_features:
    le = LabelEncoder()
    df[col] = df[col].astype(str)
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

pickle.dump(label_encoders, open("label_encoders.pkl", "wb"))

# -----------------------
# Scale numeric features
# -----------------------
scaler = StandardScaler()
df[num_features] = scaler.fit_transform(df[num_features])
pickle.dump(scaler, open("scaler.pkl", "wb"))

# Final feature list
feature_columns = num_features + cat_features
pickle.dump(feature_columns, open("feature_columns.pkl", "wb"))

# -----------------------
# Train-test split
# -----------------------
X = df[feature_columns]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------
# Handle class imbalance (SMOTE)
# -----------------------
sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

# -----------------------
# Train Logistic Regression
# -----------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_sm, y_train_sm)

pickle.dump(model, open("loan_model.pkl", "wb"))

# -----------------------
# Evaluate model
# -----------------------
probs = model.predict_proba(X_test)[:, 1]

# Find best threshold for F1
best_f1 = 0
best_t = 0.5

for t in np.arange(0.1, 1, 0.01):
    pred = (probs >= t).astype(int)
    score = f1_score(y_test, pred)
    if score > best_f1:
        best_f1 = score
        best_t = t

threshold = best_t
pickle.dump(threshold, open("threshold.pkl", "wb"))

pred_final = (probs >= threshold).astype(int)

metrics = {
    "accuracy": accuracy_score(y_test, pred_final),
    "precision": precision_score(y_test, pred_final),
    "recall": recall_score(y_test, pred_final),
    "f1": f1_score(y_test, pred_final),
    "roc_auc": roc_auc_score(y_test, probs),
    "threshold": threshold
}

json.dump(metrics, open("metrics.json", "w"), indent=4)

print("Training complete!")
print("Saved model + preprocessors + threshold")
