import streamlit as st
import pandas as pd
import numpy as np
import joblib
from feature_eng import load_data, prepare_features


st.set_page_config(
    page_title="Fraud Detection Dash",
    page_icon="🛡️",
    layout="wide"
)

@st.cache_resource
def load_models():
    try: 
        model = joblib.load('models/isolation_forest.pkl')
        scaler = joblib.load('models/scaler.pkl')
        encoder = joblib.load('models/encoder.pkl')
        feature_names = joblib.load('models/feature_names.pkl')
        st.success("Loaded models.")
        return model, scaler, encoder, feature_names
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        st.info("Run train_models.py first")
        st.stop

@st.cache_data
def load_transaction_data():
    try:
        df = load_data('data/transactions.csv')
        st.success(f"Loaded {len(df)} transactions.")
        return df
    except FileNotFoundError as e:
        st.error("Transaction file not found")
        st.info("Run generating script first")
        st.stop()

# == HELPERS ==

def get_risk_level(anomoly_score):
    score = -anomoly_score
    if score > 0.5:
        return "HIGH", "#f44336"
    elif score > 0.2:
        return "MEDIUM", "#ff9800"
    else:
        return "LOW", "#4caf50"
    
def predict_transaction(transaction_df, model, scaler, encoder):
    X, _, _, _, _ = prepare_features(
        transaction_df,
        fit_scaler=False,
        scaler=scaler,
        fit_encoder=False,
        encoder=encoder
    )

    anomaly_score = model.decision_function(X)[0]

    prediction = model.predict(X)[0]
    is_fraud = (prediction == -1)

    risk_level, color = get_risk_level(anomaly_score)

    return {
        'is_fraud': is_fraud,
        'anomaly_score': anomaly_score,
        'risk_level': risk_level,
        'color': color
    }
    
with st.spinner("Loading models and data...."):
    model, scaler, encoder, feature_names = load_models()
    df_all = load_transaction_data()

if 'processed_transactions' not in st.session_state:
    st.session_state.processed_transactions = []
if 'fraud_alerts' not in st.session_state:
    st.session_state.fraud_alerts = []
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0
if 'is_streaming' not in st.session_state:
    st.session_state.is_streaming = False
if 'stats' not in st.session_state:
    st.session_state.stats = {
        'total_processed': 0,
        'total_fraud_detected': 0,
        'total_actual_fraud': 0,
        'total_false_alarms': 0
    }


st.title("Fraud Detection Dash")
st.markdown("**Real-time transaction monitoring with ML-powered detection**")
st.markdown("---")

with st.sidebar:
    st.header("Controls")
    st.subheader("Transaction Stream")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Start", use_container_width=True):
            st.session_state.is_streaming = True
            st.rerun()
    
    with col2:
        if st.button("Stop", use_container_width=True):
            st.session_state.is_streaming = False
            st.rerun()

    if st.button("Reset", use_container_width=True):
        st.session_state.processed_transactions = []
        st.session_state.fraud_alerts = []
        st.session_state.current_index = 0
        st.session_state.is_streaming = False
        st.session_state.stats = {
            'total_processed': 0,
            'total_fraud_detected': 0,
            'total_actual_fraud': 0,
            'total_false_alarms': 0
        }
        st.rerun()

    st.markdown("---")
    st.subheader("Settings")

    stream_speed = st.slider(
        "Stream Speed (tx/sec)",
        min_value = 1,
        max_value = 10,
        value = 2
    )

    show_normal = st.checkbox("Show Normal Transactions", value=True)

    st.markdown("---")
    st.subheader("Session Stats")
    st.metric("Processed", st.session_state.stats['total_processed'])
    st.metric("Streaming", "Yes" if st.session_state.is_streaming else "No")
    st.metric("Current Index", st.session_state.current_index)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Status", "Online", "✓")

with col2:
    st.metric("Model", "Isolation Forest")

with col3:
    st.metric("Ready", "Yes", "🚀")

with col4:
    fraud_count = df_all['is_fraud'].sum()
    st.metric("Fraud in Dataset", fraud_count)


st.subheader("Test Prediction")
test_txn = df_all.sample(random_state=None).copy() #df not series
prediction = predict_transaction(test_txn, model, scaler, encoder)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Transaction Amount", f"${test_txn['amount'].values[0]:.2f}")
    st.write(f"Category: {test_txn['merchant_category'].values[0]}")
    st.write(f"Location: {test_txn['location'].values[0]}")

with col2:
    st.metric("Prediction", 
              "FRAUD" if prediction['is_fraud'] else "NORMAL",
              delta=None)
    st.write(f"Risk Level: {prediction['risk_level']}")
    st.write(f"Anomaly Score: {prediction['anomaly_score']:.4f}")

with col3:
    actual = "FRAUD" if test_txn['is_fraud'].values[0] else "NORMAL"
    st.metric("Actual Label", actual)
    
    if prediction['is_fraud'] == test_txn['is_fraud'].values[0]:
        st.success("✅ Correct prediction!")
    else:
        st.error("❌ Incorrect prediction")

st.markdown("---")


st.subheader("Sample Transactions")
st.dataframe(df_all.sample(10), use_container_width=True)

