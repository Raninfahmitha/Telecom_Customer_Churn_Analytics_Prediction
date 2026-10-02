import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load dataset
df = pd.read_csv("Data.csv")

# Convert TotalCharges
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
df.fillna(0, inplace=True)

# Feature Engineering
df["AvgCharges"] = df["TotalCharges"] / (df["tenure"] + 1)
df["HighValue"] = (df["MonthlyCharges"] > 1500).astype(int)

# Encode categorical
df["Contract"] = df["Contract"].map({"Month-to-month":0, "One year":1, "Two year":2})
df["InternetService"] = df["InternetService"].map({"DSL":0, "Fiber optic":1, "No":2})
df["Churn"] = df["Churn"].map({"No":0, "Yes":1})

# Features
X = df[["tenure","MonthlyCharges","TotalCharges","AvgCharges","HighValue","Contract","InternetService"]]
y = df["Churn"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model
model = XGBClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "churn_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# Metrics
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Accuracy:", acc)
print("Precision:", prec)
print("Recall:", rec)
print("F1:", f1)
print("Confusion Matrix:\n", cm)

joblib.dump({
    "accuracy": acc,
    "precision": prec,
    "recall": rec,
    "f1": f1,
    "cm": cm.tolist()
}, "metrics.pkl")
