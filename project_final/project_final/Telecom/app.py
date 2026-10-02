from flask import Flask, render_template, request
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from flask import send_file
from reportlab.pdfgen import canvas
import io
app = Flask(__name__)

# ===============================
# LOAD DATASET
# ===============================

data = pd.read_csv("Data.csv")
data = data.drop("customerID", axis=1)
data['TotalCharges'] = data['TotalCharges'].replace(' ', np.nan)
data['TotalCharges'] = data['TotalCharges'].astype(float)
data['TotalCharges'].fillna(data['TotalCharges'].median(), inplace=True)
data['Churn'] = data['Churn'].map({'Yes':1,'No':0})
data['Contract'] = data['Contract'].map({
    'Month-to-month':0,
    'One year':1,
    'Two year':2
})

# Encode InternetService
data['InternetService'] = data['InternetService'].map({
    'No':0,
    'DSL':1,
    'Fiber optic':2
})
data['Partner'] = data['Partner'].map({'Yes':1,'No':0})

data = data.dropna()

# TARGET
y = data["Churn"]

# FEATURES
X = data[[
    "tenure",
    "MonthlyCharges",
    "Contract",
    "InternetService",
    "SeniorCitizen",
    "Partner"
]]

# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# MODELS
# ===============================

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# ===============================
# METRICS
# ===============================

y_pred = rf_model.predict(X_test)

acc = round(accuracy_score(y_test, y_pred) * 100, 2)
prec = round(precision_score(y_test, y_pred) * 100, 2)
rec = round(recall_score(y_test, y_pred) * 100, 2)
f1 = round(f1_score(y_test, y_pred) * 100, 2)

# Confusion Matrices
cm_log = confusion_matrix(y_test, log_model.predict(X_test)).tolist()
cm_rf = confusion_matrix(y_test, rf_model.predict(X_test)).tolist()

# Feature importance
feature_names = list(X.columns)
feature_values = rf_model.feature_importances_.tolist()

# ===============================
# ROUTE
# ===============================

@app.route("/", methods=["GET","POST"])
def home():

    if request.method == "POST":

        tenure = float(request.form["tenure"])
        MonthlyCharges = float(request.form["MonthlyCharges"])
        Contract = int(request.form["Contract"])
        InternetService = int(request.form["InternetService"])
        SeniorCitizen = int(request.form["SeniorCitizen"])
        Partner = int(request.form["Partner"])

        inputs = [[
            tenure,
            MonthlyCharges,
            Contract,
            InternetService,
            SeniorCitizen,
            Partner
        ]]

        pred = rf_model.predict(inputs)[0]
        prob = rf_model.predict_proba(inputs)[0][1]
        prob_percent = round(prob * 100,2)

        if pred == 1:
            label = "Churn ❌"
        else:
            label = "No Churn ✅"

        # ===============================
        # BUSINESS SUGGESTIONS
        # ===============================

        suggestions = []

        # High churn risk
        if prob > 0.75:
            suggestions.append("Provide special retention discount")
            suggestions.append("Assign dedicated customer success manager")

        # Medium churn risk
        elif prob > 0.50:
            suggestions.append("Offer loyalty reward program")
            suggestions.append("Send personalized engagement emails")

        # Contract-based suggestion
        if Contract == 0:
            suggestions.append("Promote long-term contract plans for stability")

        # Internet service issues
        if InternetService == 2 and prob > 0.4:
            suggestions.append("Improve fiber internet stability")

        # New customers
        if tenure < 6:
            suggestions.append("Provide welcome loyalty rewards")

        # Expensive plans
        if MonthlyCharges > 80:
            suggestions.append("Offer bundle pricing to reduce cost")

        # Senior customers
        if SeniorCitizen == 1:
            suggestions.append("Provide senior-friendly customer support")

        # Stable customer
        if prob < 0.30:
            suggestions.append("Customer is loyal — offer referral benefits")
        result = {
            "pred":label,
            "prob":prob_percent,
            "suggestions":suggestions
        }

        return render_template(
            "index.html",
            result=result,
            acc=acc,
            prec=prec,
            rec=rec,
            f1=f1,
            cm_log=cm_log,
            cm_rf=cm_rf,
            feature_names=feature_names,
            feature_values=feature_values,
            inputs=inputs
        )

    return render_template("index.html")


# ===============================
# RUN
# ===============================
@app.route("/download")
def download():

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer)

    c.setFont("Helvetica",14)
    c.drawString(180,750,"Customer Churn Prediction Report")

    c.setFont("Helvetica",12)
    c.drawString(50,700,"Model Performance")

    c.drawString(50,670,f"Accuracy: {acc}%")
    c.drawString(50,650,f"Precision: {prec}%")
    c.drawString(50,630,f"Recall: {rec}%")
    c.drawString(50,610,f"F1 Score: {f1}%")

    c.drawString(50,570,"Confusion Matrix (Random Forest)")
    c.drawString(50,550,f"TN: {cm_rf[0][0]}  FP: {cm_rf[0][1]}")
    c.drawString(50,530,f"FN: {cm_rf[1][0]}  TP: {cm_rf[1][1]}")

    c.save()

    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name="churn_report.pdf",
        mimetype="application/pdf"
    )
if __name__ == "__main__":
    app.run(debug=True)