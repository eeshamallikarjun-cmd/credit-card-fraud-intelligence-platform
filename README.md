# 💳 Credit Card Fraud Intelligence Platform

## Overview

The Credit Card Fraud Intelligence Platform is a Machine Learning-powered web application that detects potentially fraudulent credit card transactions in real time.

The system uses a Random Forest Classifier trained on a real-world credit card transaction dataset and provides fraud risk analysis through an interactive Streamlit dashboard.

---

## Features

* Real-time fraud prediction
* Fraud risk scoring
* Transaction analysis dashboard
* Batch CSV transaction analysis
* Prediction history tracking
* Downloadable fraud reports
* Interactive Streamlit interface

---

## Technology Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* Random Forest
* SMOTE
* Streamlit
* Joblib

---

## Dataset

* Total Transactions: 284,807
* Fraudulent Transactions: 492
* Highly Imbalanced Dataset

SMOTE (Synthetic Minority Oversampling Technique) was used to balance the classes before model training.

---

## Model Performance

### Random Forest Classifier

* Accuracy: ~99.99%
* Fraud Detection Recall: Very High
* Confusion Matrix Analysis Performed

---

## Project Structure

```text
credit-card-fraud-intelligence-platform/
│
├── app.py
├── random_forest.py
├── fraud_model.pkl
├── main.py
└── README.md
```

---

## How to Run

Install dependencies:

```bash
pip install streamlit pandas numpy scikit-learn imbalanced-learn joblib
```

Run the application:

```bash
streamlit run app.py
```

---

## Future Improvements

* Live transaction monitoring
* API integration
* User authentication
* Fraud analytics dashboard
* Cloud deployment

---

## Author

Eesha Mallikarjun
Computer Science (AI & ML)
JNTUH
