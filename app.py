import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# -----------------------------------
# Page Configuration
# -----------------------------------
st.set_page_config(
    page_title="Credit Card Fraud Intelligence Platform",
    page_icon="💳",
    layout="wide"
)

# -----------------------------------
# Load & Train Model
# -----------------------------------
@st.cache_resource
def load_model():

    df = pd.read_csv("creditcard_sample.csv")

    scaler = StandardScaler()
    df["Amount"] = scaler.fit_transform(df[["Amount"]])

    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=10,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    return model
model = load_model()

# -----------------------------------
# Sidebar
# -------------------------------------------------

st.sidebar.title("💳 Fraud Detection")

st.sidebar.info(
    """
    Credit Card Fraud Intelligence Platform

    Machine Learning Model:
    Random Forest

    Dataset:
    284,807 Transactions

    Fraud Cases:
    492
    """
)

st.sidebar.success("System Status: Online")

# -----------------------------------
# Session State
# -----------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -----------------------------------
# Header
# -----------------------------------
st.title("💳 Credit Card Fraud Intelligence Platform")

st.markdown("""
### AI-Powered Fraud Detection System

This platform analyzes transaction patterns using a trained Random Forest Machine Learning Model.
""")

# -----------------------------------
# Dashboard Metrics
# -----------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Model", "Random Forest")

with col2:
    st.metric("Dataset Size", "284,807")

with col3:
    st.metric("Fraud Cases", "492")

st.divider()

# -----------------------------------
# Transaction Inputs
# -----------------------------------
st.subheader("Transaction Details")

col1, col2 = st.columns(2)

with col1:
    time = st.number_input(
        "Time",
        value=0.0
    )

with col2:
    amount = st.number_input(
        "Amount",
        value=0.0
    )

st.subheader("Transaction Features")

features = [time]

with st.expander("⚙ Advanced Model Features (V1 - V28)"):

    feature_cols = st.columns(4)

    for i in range(1, 29):

        with feature_cols[(i - 1) % 4]:

            value = st.number_input(
                f"V{i}",
                value=0.0
            )

        features.append(value)

features.append(amount)

st.divider()
# -----------------------------------
# CSV Upload
# -----------------------------------

st.subheader("📁 Batch Prediction")

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    batch_df = pd.read_csv(uploaded_file)

    st.write("Preview:")

    st.dataframe(batch_df.head())

    if st.button("🚀 Analyze CSV"):

        predictions = model.predict(batch_df)

        batch_df["Prediction"] = predictions

        fraud_count = (predictions == 1).sum()

        genuine_count = (predictions == 0).sum()

        st.success(
            f"Analysis Complete! Fraud: {fraud_count} | Genuine: {genuine_count}"
        )

        st.dataframe(batch_df)

        csv = batch_df.to_csv(index=False)

        st.download_button(
            label="📥 Download Results",
            data=csv,
            file_name="batch_predictions.csv",
            mime="text/csv"
        )

# -----------------------------------
# Prediction
# -----------------------------------
if st.button("🔍 Analyze Transaction"):

    data = np.array(features).reshape(1, -1)

    prediction = model.predict(data)[0]

    probability = model.predict_proba(data)[0][1]

    risk_score = probability * 100

    st.subheader("Analysis Result")

    st.metric(
        "Fraud Risk Score",
        f"{risk_score:.2f}%"
    )

    st.progress(int(risk_score))

    if risk_score < 30:

        st.success("🟢 Low Risk Transaction")

        result = "Genuine"

    elif risk_score < 70:

        st.warning("🟡 Medium Risk Transaction")

        result = "Suspicious"

    else:

        st.error("🔴 High Risk Fraud Transaction")

        result = "Fraud"

    # -------------------------
    # AI Summary
    # -------------------------
    st.subheader("AI Summary")

    if risk_score < 30:

        st.write(
            "The transaction appears normal based on learned transaction patterns."
        )

    elif risk_score < 70:

        st.write(
            "The transaction contains unusual characteristics and should be reviewed."
        )

    else:

        st.write(
            "The transaction strongly resembles previously detected fraudulent transactions."
        )

    # -------------------------
    # Save History
    # -------------------------
    st.session_state.history.append(
    {
        "Transaction ID": f"TXN{len(st.session_state.history)+1:04d}",
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Amount": amount,
        "Risk Score (%)": round(risk_score, 2),
        "Prediction": result
    }
)

# -----------------------------------
# Prediction History
# -----------------------------------

history = st.session_state.get("history", [])

if len(history) > 0:

    st.divider()

    st.subheader("📋 Prediction History")

    history_df = pd.DataFrame(history)

    st.dataframe(
        history_df,
        use_container_width=True
    )

    csv = history_df.to_csv(index=False)

    st.download_button(
        label="📥 Download Report",
        data=csv,
        file_name="fraud_report.csv",
        mime="text/csv"
    )