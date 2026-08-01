import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

data = pd.read_csv('cleaned_data.csv')

model = pickle.load(open('model.sav', 'rb'))
feature_columns = pickle.load(open('columns.pkl', 'rb'))

dropdowns = {
    "gender": sorted(data["gender"].unique()),
    "SeniorCitizen": sorted(data["SeniorCitizen"].unique()),
    "Partner": sorted(data["Partner"].unique()),
    "Dependents": sorted(data["Dependents"].unique()),
    "PhoneService": sorted(data["PhoneService"].unique()),
    "MultipleLines": sorted(data["MultipleLines"].unique()),
    "InternetService": sorted(data["InternetService"].unique()),
    "OnlineSecurity": sorted(data["OnlineSecurity"].unique()),
    "OnlineBackup": sorted(data["OnlineBackup"].unique()),
    "DeviceProtection": sorted(data["DeviceProtection"].unique()),
    "TechSupport": sorted(data["TechSupport"].unique()),
    "StreamingTV": sorted(data["StreamingTV"].unique()),
    "StreamingMovies": sorted(data["StreamingMovies"].unique()),
    "Contract": sorted(data["Contract"].unique()),
    "PaperlessBilling": sorted(data["PaperlessBilling"].unique()),
    "PaymentMethod": sorted(data["PaymentMethod"].unique()),
    "tenure_group": sorted(data["tenure_group"].unique())
}

labels = {
    "gender": "Gender",
    "SeniorCitizen": "Senior Citizen",
    "Partner": "Partner",
    "Dependents": "Dependents",
    "PhoneService": "Phone Service",
    "MultipleLines": "Multiple Lines",
    "InternetService": "Internet Service",
    "OnlineSecurity": "Online Security",
    "OnlineBackup": "Online Backup",
    "DeviceProtection": "Device Protection",
    "TechSupport": "Tech Support",
    "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming Movies",
    "Contract": "Contract",
    "PaperlessBilling": "Paperless Billing",
    "PaymentMethod": "Payment Method",
    "tenure_group": "Tenure Group"
}


@app.route('/')
def home():
    return render_template('index.html', dropdowns=dropdowns, labels=labels)

@app.route('/predict', methods=["POST"])
def predict():
    input_data = request.form.to_dict()
    input_df = pd.DataFrame([input_data])
    input_df = pd.get_dummies(input_df)
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_columns]
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)
    churn_probability = round(probability[0][1]*100, 2)
    if prediction[0] == 1:
        result = "Customer is likely to churn"
    else:
        result = "Customer is unlikely to churn"
    if churn_probability >= 50:
        result_class = "alert-danger"
    else:
        result_class = "alert-success"
    return render_template('index.html', dropdowns=dropdowns, labels=labels, prediction=result, probability=churn_probability, result_class=result_class)


if __name__ == '__main__':
    app.run(debug=True)